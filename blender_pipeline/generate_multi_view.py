import bpy
import sys
import os
import argparse
from mathutils import Vector, Euler

# Añadir la carpeta actual al path de Python para poder importar config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for root_path, dirs, files in os.walk(os.path.join(project_root, ".venv", "lib")):
    if "site-packages" in dirs:
        sys.path.insert(0, os.path.join(root_path, "site-packages"))
        break
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clean_scene():
    """Limpia todos los objetos de la escena."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)

def setup_render_env(width=224, height=224, ortho_scale=0.05):
    """Configura el entorno de renderizado con cámara cenital (ortográfica para zoom óptimo)."""
    import config
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True  # Fondo transparente
    
    # Cámara Cenital Ortográfica
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = ortho_scale
    camera_data.clip_start = 0.01
    camera_data.clip_end = 2.0
    
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    
    # Posicionar cámara cenital sobre el origen
    camera_obj.location = (0.0, 0.0, 0.355)
    camera_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')

    # 2. Luces
    light_data = bpy.data.lights.new(name="MainLight", type='AREA')
    light_data.energy = 8.0
    light_data.size = 0.2
    light_obj = bpy.data.objects.new("MainLight", light_data)
    light_obj.location = (0.1, -0.1, 0.2)
    scene.collection.objects.link(light_obj)
    
    # Relleno uniforme
    light_fill = bpy.data.lights.new(name="FillLight", type='SUN')
    light_fill.energy = 1.5
    light_fill_obj = bpy.data.objects.new("FillLight", light_fill)
    light_fill_obj.location = (-0.1, 0.1, 0.1)
    light_fill_obj.rotation_euler = Euler((0.5, 0.5, 0.0), 'XYZ')
    scene.collection.objects.link(light_fill_obj)

def import_part(ldraw_path, part_id):
    """Importa el archivo LDraw de la pieza robustamente, escalando toda la jerarquía y aplicando el material gris neutral."""
    part_filename = f"{part_id}.dat"
    filepath = os.path.join(ldraw_path, "parts", part_filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(ldraw_path, "p", part_filename)
    if not os.path.exists(filepath):
        print(f"Error: No se encontró la pieza LDraw en {filepath}")
        return None
        
    try:
        old_objs = set(bpy.data.objects)
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
            imported_piece.location = (0.0, 0.0, 0.0)
            
            # Seleccionar la pieza y sus hijos y aplicar la escala
            bpy.ops.object.select_all(action='DESELECT')
            imported_piece.select_set(True)
            
            def select_children(parent):
                for child in parent.children:
                    child.select_set(True)
                    select_children(child)
            select_children(imported_piece)
            
            bpy.ops.object.transform_apply(scale=True)
            
            # Asignar color gris plástico neutral para embeddings
            mat = bpy.data.materials.new(name="LegoNeutral")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            principled = nodes.get("Principled BSDF")
            if principled:
                principled.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)
                principled.inputs['Roughness'].default_value = 0.15
            
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
        print(f"[LegoVision Multi-view ERROR] Falló la importación de {part_id}: {e}")
        return None


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Renderizador Multi-vista LDraw")
    parser.add_argument("--part_id", type=str, default=None)
    parser.add_argument("--set_id", type=str, default=None)
    parser.add_argument("--ldraw_path", type=str, default="./data/ldraw")
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args(argv)
    
    if not args.part_id and not args.set_id:
        print("[LegoVision Index ERROR] Se requiere --part_id o --set_id")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolver la lista de piezas a procesar
    parts_to_render = []
    if args.part_id:
        parts_to_render.append(args.part_id)
        
    if args.set_id:
        # Añadir project_root al path de python para importar database
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)
        try:
            from database.set_catalog import get_set_data
            set_data = get_set_data(args.set_id)
            for part in set_data.get("parts", []):
                p_ref = part["ref"]
                if p_ref not in parts_to_render:
                    parts_to_render.append(p_ref)
            print(f"[LegoVision Multi-view] Se renderizarán {len(parts_to_render)} piezas para el set {args.set_id}")
        except Exception as e:
            print(f"[LegoVision Multi-view ERROR] No se pudo cargar el set {args.set_id}: {e}")
            sys.exit(1)

    # Renderizar las 36 orientaciones (3 caras × 12 ángulos) para cada pieza
    # Cara 0: Normal, Cara 1: De lado, Cara 2: Invertido
    rotations_face = [
        Euler((0, 0, 0), 'XYZ'),       # Cara 0
        Euler((1.5708, 0, 0), 'XYZ'),  # Cara 1 (90 grados X)
        Euler((3.1415, 0, 0), 'XYZ')   # Cara 2 (180 grados X)
    ]
    
    for part_id in parts_to_render:
        print(f"[LegoVision Multi-view] Renderizando pieza: {part_id}")
        
        # Limpiar escena y preparar entorno una sola vez por pieza
        clean_scene()
        setup_render_env()
        
        # Importar la pieza una sola vez
        obj = import_part(args.ldraw_path, part_id)
        if not obj:
            print(f"[LegoVision Multi-view] Saltando pieza {part_id} por error de importación.")
            continue
            
        # Ajustar la escala ortográfica de la cámara a un valor físico fijo (70mm = 0.07m)
        bpy.context.scene.camera.data.ortho_scale = 0.07
            
        for face_idx, base_rot in enumerate(rotations_face):
            for rot_z in range(0, 360, 30):
                # Aplicar la rotación estable base y la rotación del plano (Z) combinadas
                # Dado que el orden es 'XYZ', Z se aplica al final sobre el eje Z global de la escena
                angle_rad = (rot_z / 180.0) * 3.14159
                obj.rotation_euler = Euler((base_rot.x, base_rot.y, angle_rad), 'XYZ')
                
                # Forzar actualización de escena
                bpy.context.view_layer.update()
                
                # Render
                out_file = os.path.join(args.output_dir, f"part_{part_id}_face{face_idx}_rot{rot_z}.png")
                bpy.context.scene.render.filepath = out_file
                bpy.ops.render.render(write_still=True)
                
    print("[LegoVision Multi-view] Proceso de renderizado multi-vista finalizado.")

if __name__ == "__main__":
    main()
