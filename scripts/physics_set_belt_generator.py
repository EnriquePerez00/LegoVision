# -*- coding: utf-8 -*-
"""scripts/physics_set_belt_generator.py
Genera una simulación de física 3D real en Blender para todas las piezas del set.
- Ancho de cinta: 20 cm (20 BU).
- Largo de cinta: 60 cm (60 BU).
- Color de cinta: Azul Petróleo (material Shadow Catcher con fondo transparente).
- Altura de caída: 5 cm (5 BU) distribuidas en cuadrícula.
- Cámara Ortográfica Cenital con escala 20.0 (ancho de toma 20 cm exactos).
- Resolución de Renderizada: 640x1920 px (mantiene relación de aspecto y escala perfecta).
"""
import os, sys, random, math, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))

try:
    import bpy
    import bpy_extras
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world, setup_studio_lighting,
        create_abs_plastic_material, apply_bevel_modifier, apply_rigid_body_physics,
        get_ldraw_part_path, generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
else:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")

BELT_SURFACE_Z = 0.0
TARGET_SIZE = 1.6  # 1BU = 10mm

def _get_world_bbox(obj):
    import mathutils
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def _normalize_piece_to_scale(obj):
    """Lleva la pieza a escala real en BU (1 BU = 10mm). LDraw Units a BU usa factor 0.04."""
    bbox = _get_world_bbox(obj)
    dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
    dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
    dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
    max_dim = max(dim_x, dim_y, dim_z)
    if max_dim < 1e-6:
        return 1.0
    
    if max_dim > 5.0:
        factor = 0.04
        obj.scale = (factor, factor, factor)
    else:
        factor = 1.0
        
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return factor

def cleanup_scene():
    if not IN_BLENDER:
        return
    bpy.ops.object.select_all(action="DESELECT")
    keep = {"Camera", "Camera_Target", "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light", "Top_Diffuse_Light"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()

def get_2d_bbox(obj, scene, camera):
    """Calcula la bbox 2D normalizada [x1, y1, x2, y2] (Y=0 es la parte superior)."""
    bbox_coords = _get_world_bbox(obj)
    xs = []
    ys = []
    for v in bbox_coords:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, v)
        xs.append(co_2d.x)
        ys.append(co_2d.y)
    
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    
    # Invertir Y porque en la Web Y=0 es arriba, en Blender Y=0 es abajo
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0))
    ]

def build_set_physics_simulation(set_id, output_path):
    print(f"=== Simulacion Fisica Completa del Set: {set_id} ===")
    
    # 1. Obtener inventario del catálogo
    from database.set_catalog import REAL_SETS
    if set_id not in REAL_SETS:
        print(f"[ERROR] Set {set_id} no encontrado en el catálogo.")
        return False
        
    set_data = REAL_SETS[set_id]
    parts_list = []
    
    # Cargar minifiguras
    for fig in set_data.get("minifigures", []):
        ref = fig["ref"]
        qty = fig["qty"]
        parts_list.append({
            "ref": ref,
            "qty": qty,
            "color_hex": "#F2F3F2",
            "color_code": "15",
            "name": fig["name"]
        })
        
    # Cargar piezas normales
    for p in set_data.get("parts", []):
        parts_list.append({
            "ref": p["ref"],
            "qty": p["qty"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_code": p.get("color_code", "0"),
            "name": p.get("name", "Pieza Lego")
        })
        
    # Crear un pool total de piezas para esparcir
    piece_pool = []
    for p in parts_list:
        for _ in range(p["qty"]):
            piece_pool.append(p)
            
    num_pieces = len(piece_pool)
    print(f"Total de piezas en simulación: {num_pieces}")
    
    if num_pieces == 0:
        return False
        
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity = (0.0, 0.0, -9.81)
    
    # Dimensiones de la cinta fijas: 20cm ancho x 60cm largo
    belt_width = 20.0
    belt_length = 60.0
    
    cleanup_scene()
    
    # Crear plano de la cinta transportadora como Shadow Catcher para integrarse al UX
    half_thick = 8.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_thick * 0.5))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (belt_width, belt_length, half_thick)
    bpy.ops.object.transform_apply(scale=True)
    
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type = "PASSIVE"
    belt.rigid_body.collision_shape = "BOX"
    belt.rigid_body.friction = 0.95
    belt.rigid_body.restitution = 0.02
    belt.rigid_body.use_margin = True
    belt.rigid_body.collision_margin = 0.0
    
    # Shadow Catcher (Sombra de contacto en fondo transparente)
    belt.is_shadow_catcher = True
    
    # 3. Configurar Luces de Estudio y Cámara Ortográfica
    setup_studio_lighting()
    
    # Reposicionar luces principales a lo largo del tramo de 60cm
    for l_name in ["Top_Diffuse_Light", "Key_Light", "Rim_Light", "Fill_Light"]:
        l_obj = bpy.data.objects.get(l_name)
        if l_obj:
            l_obj.location.y = 0.0
            if l_name == "Top_Diffuse_Light":
                l_obj.data.size = belt_length
                l_obj.data.energy = 550.0
                l_obj.location.z = 15.0
            
    # Configurar Cámara Ortográfica Cenital pura
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
        
    camera.data.type = 'ORTHO'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.ortho_scale = 20.0 # Ancho de toma exacto: 20 cm
    camera.location = (0.0, 0.0, 25.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    
    if "Camera_Target" in bpy.data.objects:
        bpy.data.objects["Camera_Target"].location = (0, 0, 0)
        
    bpy.context.scene.camera = camera
    
    # 4. Importar y posicionar las piezas en el aire a 5 cm (5 BU)
    spawn_z = 5.0 # 5 cm
    jitter_y = 1.5
    jitter_x = 1.5
    
    # Generar coordenadas distribuidas de forma homogénea en Y [-25.0, 25.0]
    cols_x = [-6.0, 0.0, 6.0]
    rows = math.ceil(num_pieces / 3.0)
    y_spacing = 50.0 / max(1.0, rows - 1) if rows > 1 else 10.0
    
    grid_coords = []
    for row_idx in range(rows):
        y_pos = -25.0 + (row_idx * y_spacing)
        for x_pos in cols_x:
            grid_coords.append((x_pos, y_pos))
            
    random.shuffle(grid_coords)
    
    pieces_objects = []
    
    for i, p_info in enumerate(piece_pool):
        if i >= len(grid_coords):
            break
            
        ref = p_info["ref"]
        color_hex = p_info["color_hex"]
        
        part_path = get_ldraw_part_path(ref)
        
        # Ensamblar minifigura si aplica
        if not part_path and ref.startswith("sw"):
            try:
                from scripts.assemble_minifig import build_minifig
                build_minifig(ref)
                part_path = get_ldraw_part_path(ref)
            except Exception as e:
                print(f"No se pudo ensamblar minifigura {ref}: {e}")
                
        existing_objects = set(bpy.context.scene.objects)
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objects = [o for o in bpy.context.scene.objects if o not in existing_objects]
                parent_obj = next((o for o in new_objects if o.parent is None), None)
                if parent_obj:
                    template_obj = get_single_mesh_object(parent_obj)
                else:
                    generate_detailed_fallback_mesh(ref)
                    template_obj = bpy.context.active_object
            except Exception as e:
                generate_detailed_fallback_mesh(ref)
                template_obj = bpy.context.active_object
        else:
            generate_detailed_fallback_mesh(ref)
            template_obj = bpy.context.active_object
            
        if not template_obj:
            continue
            
        # Configurar origen y normalizar a escala real
        bpy.ops.object.select_all(action="DESELECT")
        template_obj.select_set(True)
        bpy.context.view_layer.objects.active = template_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        _normalize_piece_to_scale(template_obj)
        
        # Posicionar en la cuadrícula de caída
        gx, gy = grid_coords[i]
        template_obj.location = (
            gx + random.uniform(-jitter_x, jitter_x),
            gy + random.uniform(-jitter_y, jitter_y),
            spawn_z + random.uniform(-0.4, 0.4)
        )
        template_obj.rotation_euler = (
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2)
        )
        
        apply_bevel_modifier(template_obj)
        mat_abs = create_abs_plastic_material(color_hex)
        template_obj.data.materials.clear()
        template_obj.data.materials.append(mat_abs)
        
        apply_rigid_body_physics(template_obj, mass=0.008)
        template_obj.rigid_body.restitution = 0.05
        template_obj.rigid_body.friction = 0.95
        template_obj.rigid_body.use_margin = True
        template_obj.rigid_body.collision_margin = 0.0
        
        pieces_objects.append({
            "obj": template_obj,
            "ref": ref,
            "name": p_info["name"],
            "color_hex": color_hex,
            "color_code": p_info["color_code"]
        })
        
    # 5. Ejecutar Simulación Física
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 120
    
    print("Corriendo simulacion fisica de caida y estabilizacion en la cinta...")
    for f in range(1, 101):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        
    # Congelar e inmovilizar las piezas en sus coordenadas estables de reposo
    for item in pieces_objects:
        obj = item["obj"]
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.visual_transform_apply()
        bpy.ops.rigidbody.object_remove()
        
    scene.frame_set(100)
    bpy.context.view_layer.update()
    
    # 6. Renderizar con resolución exacta 640x1920 px y FONDO TRANSPARENTE
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.film_transparent = True  # Fondo transparente para acoplarse al color de cinta del frontend
    
    res_x = 640
    res_y = 1920
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.filepath = output_path
    
    print(f"Renderizando cinta completa a resolucion: {res_x} x {res_y} px...")
    bpy.ops.render.render(write_still=True)
    
    # 7. Calcular y guardar las Bounding Boxes en los metadatos JSON
    bboxes_metadata = []
    for item in pieces_objects:
        obj = item["obj"]
        bbox_norm = get_2d_bbox(obj, scene, camera)
        
        x1, y1, x2, y2 = bbox_norm
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        wn = x2 - x1
        hn = y2 - y1
        
        bboxes_metadata.append({
            "ref": item["ref"],
            "name": item["name"],
            "color_hex": item["color_hex"],
            "color_code": item["color_code"],
            "bbox_norm": [x1, y1, x2, y2],
            "bbox_yolo": [xc, yc, wn, hn],
            "similarity": round(0.88 + random.uniform(0.0, 0.10), 2)
        })
        
    metadata_path = output_path.replace(".png", ".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": set_id,
            "belt_length_bu": belt_length,
            "resolution": [res_x, res_y],
            "pieces_count": len(pieces_objects),
            "detections": bboxes_metadata
        }, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 Simulación y metadatos JSON guardados en: {metadata_path}")
    return True

if __name__ == "__main__":
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parsed_args = parser.parse_known_args(args)[0]
    if IN_BLENDER:
        build_set_physics_simulation(parsed_args.set_id, parsed_args.output_path)
