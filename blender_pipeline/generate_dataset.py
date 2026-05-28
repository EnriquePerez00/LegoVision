import bpy
import os
import sys
import json
import random
import argparse
from mathutils import Vector, Euler

# Añadir la carpeta actual al path de Python para poder importar config y utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for root_path, dirs, files in os.walk(os.path.join(project_root, ".venv", "lib")):
    if "site-packages" in dirs:
        sys.path.insert(0, os.path.join(root_path, "site-packages"))
        break
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.camera_setup import setup_camera
from utils.lighting_setup import setup_lighting
from utils.physics_sim import setup_physics_world, create_belt_collider, apply_rigid_body_to_piece, run_simulation
from utils.yolo_export import get_yolo_bbox, save_yolo_label

def clean_scene():
    """Elimina todos los objetos de la escena actual."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Limpiar colecciones sobrantes
    for col in bpy.data.collections:
        if col.name not in ["Collection"]:
            bpy.data.collections.remove(col)

def load_catalog_index():
    """Carga el índice de piezas indexadas de LDraw."""
    if not os.path.exists(config.CATALOG_INDEX):
        print(f"[LegoVision ERROR] No se encontró el archivo de índice del catálogo en: {config.CATALOG_INDEX}")
        print("Asegúrate de ejecutar scripts/download_ldraw.sh primero.")
        sys.exit(1)
        
    with open(config.CATALOG_INDEX, 'r') as f:
        return json.load(f)

def import_ldraw_part(part_id, color_hex=None):
    """Importa una pieza individual de LDraw usando el addon ldr_tools_blender."""
    # Intentar buscar en parts/ o en p/
    part_filename = f"{part_id}.dat"
    filepath = os.path.join(config.LDRAW_PATH, "parts", part_filename)
    
    if not os.path.exists(filepath):
        filepath = os.path.join(config.LDRAW_PATH, "p", part_filename)
        
    if not os.path.exists(filepath):
        print(f"[LegoVision Warning] Archivo de pieza no encontrado: {part_filename}")
        return None

    # ldr_tools_blender import operator
    # bpy.ops.import_scene.ldr(filepath="...")
    try:
        # Guardar objetos actuales para detectar qué se importó
        old_objs = set(bpy.data.objects)
        
        # Ejecutar importación
        bpy.ops.import_scene.importldr(filepath=filepath)
        
        new_objs = set(bpy.data.objects) - old_objs
        if not new_objs:
            return None
            
        # Encontrar el objeto raíz (el que no tiene padre entre los nuevos objetos)
        imported_piece = None
        for obj in new_objs:
            if obj.parent is None:
                imported_piece = obj
                break
                
        if not imported_piece:
            for obj in new_objs:
                if obj.type == 'MESH':
                    imported_piece = obj
                    break
        if not imported_piece and new_objs:
            imported_piece = list(new_objs)[0]
            
        if imported_piece:
            # Centrar el origen en el centro de la geometría de sus bordes (bound box)
            bpy.context.view_layer.objects.active = imported_piece
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
            
            # Ajustar la escala a metros Blender reales (1 LDU = 0.4mm = 0.0004m)
            imported_piece.scale = (0.0004, 0.0004, 0.0004)
            
            # Seleccionar la pieza y sus hijos y aplicar la escala
            bpy.ops.object.select_all(action='DESELECT')
            imported_piece.select_set(True)
            
            def select_children(parent):
                for child in parent.children:
                    child.select_set(True)
                    select_children(child)
            select_children(imported_piece)
            
            bpy.ops.object.transform_apply(scale=True)
            
            # Aplicar color si se especifica
            if color_hex:
                hex_str = color_hex.lstrip('#')
                rgb = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                
                # Crear material brillante
                mat = bpy.data.materials.new(name=f"LegoColor_{color_hex}")
                mat.use_nodes = True
                principled = mat.node_tree.nodes.get("Principled BSDF")
                if principled:
                    principled.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
                    principled.inputs['Roughness'].default_value = 0.15
                    principled.inputs['Specular IOR Level'].default_value = 0.5
                
                # Aplicar a la pieza raíz y a todos sus hijos mallas
                all_objs = [imported_piece]
                def get_all_children(parent):
                    for child in parent.children:
                        all_objs.append(child)
                        get_all_children(child)
                get_all_children(imported_piece)
                
                for obj in all_objs:
                    if obj.type == 'MESH':
                        obj.data.materials.clear()
                        obj.data.materials.append(mat)
            
            # Deseleccionar al terminar
            bpy.ops.object.select_all(action='DESELECT')
            
        return imported_piece
    except Exception as e:
        print(f"[LegoVision ERROR] Falló la importación de {part_id}: {e}")
        return None

def link_recursive(parent_obj, collection):
    """Mueve un objeto y todos sus descendientes a una colección específica."""
    for col in list(parent_obj.users_collection):
        col.objects.unlink(parent_obj)
    if parent_obj.name not in collection.objects:
        collection.objects.link(parent_obj)
    for child in parent_obj.children:
        link_recursive(child, collection)

def main():
    # Blender procesa los argumentos pasados después de "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Pipeline de generación de dataset LegoVision")
    parser.add_argument("--num_images", type=int, default=config.NUM_IMAGES, help="Número de imágenes a generar")
    parser.add_argument("--pieces_per_image", type=int, default=config.PIECES_PER_IMAGE, help="Piezas por imagen")
    parser.add_argument("--single_class", action="store_true", help="Usa clase única (0) para YOLO detector")
    parser.add_argument("--belt_mode", action="store_true",
                        help="Modo cinta: posiciona piezas a lo largo del eje Y simulando la cinta en movimiento."
                             " Sin simulación de física Blender (~8x más rápido).")
    parser.add_argument("--set_id", type=str, default=None, help="ID del set LEGO para restringir piezas")
    args = parser.parse_args(argv)

    print(f"=========================================================")
    print(f"LegoVision Dataset Generator - Iniciando")
    print(f"Imágenes objetivo: {args.num_images}")
    print(f"Piezas por imagen: {args.pieces_per_image}")
    print(f"Single Class Mode: {args.single_class}")
    if args.set_id:
        print(f"Restringido al set: {args.set_id}")
    print(f"=========================================================")

    # 1. Cargar catálogo
    catalog_raw = load_catalog_index()
    catalog = {p["ldraw_id"]: p for p in catalog_raw["classes"]}
    # Tomar la lista de partes válidas
    valid_parts = list(catalog.keys())
    part_to_color = {}
    
    if args.set_id:
        # Añadir project_root al path de python para importar database
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        try:
            from database.set_catalog import get_set_data
            set_data = get_set_data(args.set_id)
            set_parts = [p["ref"] for p in set_data.get("parts", [])]
            # Solo usar piezas del set que existan en el catálogo
            valid_parts = [p for p in valid_parts if p in set_parts]
            part_to_color = {p["ref"]: p["color_hex"] for p in set_data.get("parts", [])}
            print(f"[LegoVision Dataset] Filtrado a {len(valid_parts)} piezas válidas del set {args.set_id}")
        except Exception as e:
            print(f"[LegoVision Dataset ERROR] No se pudo cargar el set {args.set_id}: {e}")
    
    # 2. Configurar salida
    os.makedirs(os.path.join(config.DATASET_OUTPUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(config.DATASET_OUTPUT, "labels"), exist_ok=True)

    # 3. Preparar Escena Base
    scene = bpy.context.scene
    clean_scene()
    
    # Setup cámara, luces (y física sólo si no estamos en belt_mode)
    camera_obj = setup_camera(scene)
    setup_lighting(scene)
    if not args.belt_mode:
        create_belt_collider(scene)
        setup_physics_world(scene)
    
    # Configurar motor de Render (EEVEE para máxima velocidad)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_percentage = 100
    
    # Bucle principal de generación de imágenes
    for img_idx in range(args.num_images):
        print(f"\n--- Generando Imagen {img_idx + 1} de {args.num_images} ---")
        
        # Colección temporal para las piezas de esta iteración
        pieces_col = bpy.data.collections.new(f"TempPieces_{img_idx}")
        scene.collection.children.link(pieces_col)
        
        # Almacenar datos para exportar YOLO labels
        detections = []
        imported_objs = []

        if args.belt_mode:
            # ─────────────────────────────────────────────────────────────────
            # MODO CINTA: posicionamiento determinista, sin física Blender.
            #
            # Concepto: la cámara mira hacia abajo (nadir). El eje Y de Blender
            # es la dirección de avance de la cinta. Para cada imagen sorteamos:
            #   • velocidad v ∈ [2, 8] m/min → pixels/frame de referencia
            #   • instante t ∈ [0, 1.5] s → cuánto ha avanzado la cinta
            # Cada pieza se coloca en:
            #   X: posición lateral aleatoria dentro del ancho de la cinta
            #   Y: posición inicial + desplazamiento según v*t individualizado
            #   Z: sobre la superficie de la cinta (z_belt + radio_pieza aprox.)
            # La rotación es aleatoria sólo en Z (cara estable hacia arriba).
            # ─────────────────────────────────────────────────────────────────
            import math

            belt_z  = 0.0 * config.BLENDER_SCALE   # superficie de la cinta (z=0)
            piece_z = 20.0 * config.BLENDER_SCALE  # altura aprox. de la pieza sobre la cinta

            # Parámetros del FOV en metros Blender (cinta 200mm × 250mm)
            fov_x_half = (config.BELT_WIDTH_MM / 2.0)     * config.BLENDER_SCALE   # ±100mm
            fov_y_half = (config.FOV_LONGITUDINAL_MM / 2.0) * config.BLENDER_SCALE  # ±125mm

            # Sortear número de piezas en este frame (grupos 3-8)
            n_pieces = random.randint(3, min(8, args.pieces_per_image))

            # Variar temperatura de color de la luz para aumentación fotométrica
            for light in bpy.data.lights:
                if hasattr(light, 'color_temperature'):
                    light.color_temperature = random.uniform(4500, 7000)

            for p_idx in range(n_pieces):
                part_id = random.choice(valid_parts)
                class_idx = 0 if args.single_class else catalog[part_id]["idx"]
                color_hex = part_to_color.get(part_id)

                obj = import_ldraw_part(part_id, color_hex=color_hex)
                if not obj:
                    continue

                link_recursive(obj, pieces_col)

                # Posición lateral (X) aleatoria dentro del ancho de cinta
                x = random.uniform(-fov_x_half * 0.85, fov_x_half * 0.85)

                # Posición longitudinal (Y): simula distintos momentos del avance
                # Distribuimos piezas uniformemente por el FOV con offset aleatorio
                y_base = fov_y_half * (1.0 - 2.0 * p_idx / max(n_pieces - 1, 1))
                y_jitter = random.uniform(-fov_y_half * 0.25, fov_y_half * 0.25)
                y = y_base + y_jitter
                y = max(-fov_y_half * 0.9, min(fov_y_half * 0.9, y))  # clamping

                obj.location = (x, y, belt_z + piece_z)

                # Rotación: cara estable (X=0, Y=0) + rotación aleatoria en Z (360°)
                obj.rotation_euler = Euler(
                    (0.0, 0.0, random.uniform(0, 2 * math.pi)), 'XYZ'
                )

                imported_objs.append((obj, class_idx, part_id))

            # Sin física — actualizar dependencias de la escena
            bpy.context.view_layer.update()

        else:
            # ─────────────────────────────────────────────────────────────────
            # MODO CLÁSICO: spawn aleatorio + simulación de física Blender
            # ─────────────────────────────────────────────────────────────────
            spawn_min_x = -70.0 * config.BLENDER_SCALE
            spawn_max_x =  70.0 * config.BLENDER_SCALE
            spawn_min_y = -90.0 * config.BLENDER_SCALE
            spawn_max_y =  90.0 * config.BLENDER_SCALE
            spawn_min_z = 100.0 * config.BLENDER_SCALE
            spawn_max_z = 250.0 * config.BLENDER_SCALE

            for p_idx in range(args.pieces_per_image):
                part_id = random.choice(valid_parts)
                class_idx = 0 if args.single_class else catalog[part_id]["idx"]
                color_hex = part_to_color.get(part_id)

                obj = import_ldraw_part(part_id, color_hex=color_hex)
                if not obj:
                    continue

                for col in obj.users_collection:
                    col.objects.unlink(obj)
                pieces_col.objects.link(obj)

                x = random.uniform(spawn_min_x, spawn_max_x)
                y = random.uniform(spawn_min_y, spawn_max_y)
                z = random.uniform(spawn_min_z, spawn_max_z)
                obj.location = (x, y, z)

                obj.rotation_euler = Euler((
                    random.uniform(0, 3.1415),
                    random.uniform(0, 3.1415),
                    random.uniform(0, 3.1415)
                ), 'XYZ')

                apply_rigid_body_to_piece(obj)
                imported_objs.append((obj, class_idx, part_id))

            # Ejecutar física
            run_simulation(scene, end_frame=70)
        
        # Calcular Bounding Boxes e indexar
        for obj, class_idx, part_id in imported_objs:
            bbox = get_yolo_bbox(obj, camera_obj, scene)
            if bbox:
                detections.append({
                    "class_idx": class_idx,
                    "part_id": part_id,
                    "bbox": bbox
                })
                
        if not detections:
            print("[LegoVision Warning] No se detectaron piezas en el FOV de la cámara. Saltando render.")
            # Limpiar colección temporal
            for item in imported_objs:
                bpy.data.objects.remove(item[0], do_unlink=True)
            scene.collection.children.unlink(pieces_col)
            bpy.data.collections.remove(pieces_col)
            continue
            
        # Render e Imagen de Salida
        image_name = f"train_{img_idx:05d}.png"
        label_name = f"train_{img_idx:05d}.txt"
        meta_name = f"train_{img_idx:05d}.json"
        
        image_path = os.path.join(config.DATASET_OUTPUT, "images", image_name)
        label_path = os.path.join(config.DATASET_OUTPUT, "labels", label_name)
        meta_path = os.path.join(config.DATASET_OUTPUT, "labels", meta_name)
        
        scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        print(f"[LegoVision Render] Imagen guardada en: {image_path}")
        
        # Guardar YOLO labels
        save_yolo_label(label_path, detections)
        print(f"[LegoVision YOLO] Etiquetas guardadas en: {label_path} (Detecciones: {len(detections)})")

        # Guardar metadatos JSON para evaluación de simulación
        with open(meta_path, 'w') as f:
            json.dump({
                "image": image_name,
                "detections": [{
                    "part_id": det["part_id"],
                    "bbox": det["bbox"]
                } for det in detections]
            }, f, indent=2)
        
        # Limpiar escena para la siguiente iteración
        for item in imported_objs:
            bpy.data.objects.remove(item[0], do_unlink=True)
            
        scene.collection.children.unlink(pieces_col)
        bpy.data.collections.remove(pieces_col)

    print("\n=========================================================")
    print("[LegoVision] Generación de dataset finalizada exitosamente.")
    print("=========================================================")

if __name__ == "__main__":
    main()
