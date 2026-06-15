# -*- coding: utf-8 -*-
# scripts/render_poses_fotorrealistas.py
# Renderiza imágenes 3D de estudio (Eevee) de las poses de LEGO para el reporte comparativo.
import os, sys, json, math, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root    = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "database"))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "database"))

try:
    import bpy, mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

def setup_studio():
    # 1. Configurar motor Cycles en CPU de forma rápida
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.max_bounces = 4
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 4
    bpy.context.scene.cycles.transmission_bounces = 0
    
    bpy.context.scene.render.resolution_x = 180
    bpy.context.scene.render.resolution_y = 180
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True
    
    # 2. Eliminar todos los objetos previos
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
            
    # 3. Crear luces de estudio
    # Luz principal (Key Light)
    bpy.ops.object.light_add(type='SUN', radius=1.0, location=(5.0, -5.0, 10.0))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = 4.0
    key_light.rotation_euler = (math.radians(35), math.radians(25), math.radians(-45))
    
    # Luz de relleno (Fill Light)
    bpy.ops.object.light_add(type='SUN', radius=1.0, location=(-5.0, 5.0, 5.0))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 2.0
    fill_light.rotation_euler = (math.radians(45), math.radians(-25), math.radians(135))

    # Luz trasera (Rim Light)
    bpy.ops.object.light_add(type='SUN', radius=1.0, location=(0.0, 8.0, 8.0))
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    rim_light.data.energy = 2.5
    rim_light.rotation_euler = (math.radians(65), 0, math.radians(180))

    # Crear plano de suelo semi-transparente a Z = -0.001
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, -0.001))
    floor = bpy.context.active_object
    floor.name = "StudioFloor"
    
    floor_mat = bpy.data.materials.new(name="StudioFloorMat")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs['Alpha'].default_value = 0.4  # Semi-transparent
        bsdf.inputs['Roughness'].default_value = 0.5
    floor.data.materials.append(floor_mat)

    # 4. Crear Cámara Isométrica (elevada a ~45 grados)
    bpy.ops.object.camera_add(location=(10.0, -10.0, 14.14))
    cam = bpy.context.active_object
    cam.name = "StudioCamera"
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 3.5  # Se ajustará dinámicamente
    
    # Apuntar cámara al origen
    constraint = cam.constraints.new(type='TRACK_TO')
    # Crear objeto target vacío si no existe
    if "CamTarget" not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
        target = bpy.context.active_object
        target.name = "CamTarget"
    else:
        target = bpy.data.objects["CamTarget"]
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = cam
    return cam

def create_plastic_material():
    mat_name = "LegoPlastic"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # Obtener el nodo Principled BSDF
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        if 'Base Color' in bsdf.inputs:
            bsdf.inputs['Base Color'].default_value = (0.2, 0.4, 0.7, 1.0)
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = 0.15
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.5
        elif 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.5
    return mat

def load_ldraw_part(part_ref):
    bases = [
        os.path.join(project_root, "data", "ldraw"),
        os.path.expanduser("~/Library/Application Support/LDraw"),
        os.path.expanduser("~/ldraw"),
        "/Applications/Studio 2.0/ldraw",
        os.path.expanduser("~/Library/Application Support/BrickLink Studio 2.0/ldraw"),
    ]
    part_file = None
    for base in bases:
        for sub in ["UnOfficial/parts", "parts", "UnOfficial/p", "p"]:
            c = os.path.join(base, sub, part_ref + ".dat")
            if os.path.exists(c): part_file = c; break
        if part_file: break
    before = set(bpy.data.objects)
    if part_file:
        try: bpy.ops.import_scene.importldr(filepath=part_file)
        except Exception as e: print("    [WARN] " + part_ref + ": " + str(e)); part_file = None
    new_objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new_objs:
        bpy.ops.mesh.primitive_cube_add(size=0.8); new_objs = [bpy.context.active_object]
    bpy.ops.object.select_all(action="DESELECT")
    for o in new_objs: o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object; obj.name = "Piece_" + part_ref
    s = 0.04; obj.scale = (s, s, s)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    
    # Aplicar material plástico limpiando los previos
    mat = create_plastic_material()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    # Reset material indices
    for face in obj.data.polygons:
        face.material_index = 0
        
    return obj

def render_pose(obj, cam, contact_normal, fallback_quat, output_path):
    # Calcular rotación a partir del contact_normal para que apoye sobre Z=0
    if contact_normal:
        n = mathutils.Vector(contact_normal).normalized()
        target_vec = mathutils.Vector((0.0, 0.0, -1.0))
        dot = max(-1.0, min(1.0, n.dot(target_vec)))
        if dot > 0.9999:
            quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        elif dot < -0.9999:
            quat = mathutils.Quaternion((1.0, 0.0, 0.0), math.pi)
        else:
            axis = n.cross(target_vec).normalized()
            quat = mathutils.Quaternion(axis, math.acos(dot))
    else:
        quat = mathutils.Quaternion(fallback_quat) if fallback_quat else mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        
    # Aplicar rotación
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = quat
    bpy.context.view_layer.update()
    
    # Apoyar pieza en Z=0 (usando vértices reales de la pieza)
    vertices_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_z = min(v.z for v in vertices_world)
    obj.location.z -= min_z
    bpy.context.view_layer.update()
    
    # Recalcular centro y dimensiones reales a partir de los vértices (AABB)
    vertices_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_x = min(v.x for v in vertices_world)
    max_x = max(v.x for v in vertices_world)
    min_y = min(v.y for v in vertices_world)
    max_y = max(v.y for v in vertices_world)
    min_z = min(v.z for v in vertices_world)
    max_z = max(v.z for v in vertices_world)
    
    center = mathutils.Vector(((min_x + max_x)/2.0, (min_y + max_y)/2.0, (min_z + max_z)/2.0))
    
    # Centrar la pieza horizontalmente en (0, 0)
    obj.location.x -= center.x
    obj.location.y -= center.y
    bpy.context.view_layer.update()
    
    # Recalcular el centro definitivo para el target de la cámara
    vertices_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_x = min(v.x for v in vertices_world)
    max_x = max(v.x for v in vertices_world)
    min_y = min(v.y for v in vertices_world)
    max_y = max(v.y for v in vertices_world)
    min_z = min(v.z for v in vertices_world)
    max_z = max(v.z for v in vertices_world)
    center = mathutils.Vector(((min_x + max_x)/2.0, (min_y + max_y)/2.0, (min_z + max_z)/2.0))
    
    target = bpy.data.objects["CamTarget"]
    target.location = center
    
    # Ajustar escala de la cámara según tamaño de la pieza (encuadre dinámico)
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    dim_z = max_z - min_z
    max_dim = max(dim_x, dim_y, dim_z)
    cam.data.ortho_scale = max_dim * 1.6
    
    # Renderizar
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def main():
    if not IN_BLENDER:
        print("[ERROR] Este script debe ejecutarse en Blender"); return
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="")
    args = parser.parse_known_args(args_list)[0]
    
    # Configurar escena
    cam = setup_studio()
    
    # Crear carpeta de salida de imágenes
    images_dir = os.path.join(repo_root, "data", "reports", "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Obtener lista de piezas a procesar
    from supabase_client import get_connection
    with get_connection() as conn, conn.cursor() as cur:
        if args.part:
            cur.execute("SELECT DISTINCT part_ref FROM stable_poses WHERE part_ref = %s", (args.part,))
        else:
            cur.execute("SELECT DISTINCT part_ref FROM stable_poses")
        parts = [r["part_ref"] for r in cur.fetchall()]
        
    print(f"[RenderFotorrealista] Procesando {len(parts)} piezas...")
    
    for idx, ref in enumerate(parts):
        print(f"[{idx+1}/{len(parts)}] Renderizando {ref}...")
        
        # Limpiar piezas previas
        for o in list(bpy.data.objects):
            if o.name.startswith("Piece_"):
                bpy.data.objects.remove(o, do_unlink=True)
                
        try:
            obj = load_ldraw_part(ref)
        except Exception as e:
            print(f"  ❌ Error cargando pieza {ref}: {e}")
            continue
            
        # Buscar poses de esta pieza en producción
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT pose_index, orientation_quat, contact_normal FROM stable_poses 
                WHERE part_ref = %s AND is_stable = true
            """, (ref,))
            prod_poses = cur.fetchall()
            
            # Buscar poses de esta pieza en test
            cur.execute("""
                SELECT pose_index, orientation_quat, contact_normal FROM stable_poses_test 
                WHERE part_ref = %s AND is_stable = true
            """, (ref,))
            test_poses = cur.fetchall()
            
        # Renderizar poses de producción
        for p in prod_poses:
            p_idx = p["pose_index"]
            quat = p["orientation_quat"]
            normal = p.get("contact_normal")
            out_path = os.path.join(images_dir, f"{ref}_pose_prod_{p_idx}.png")
            render_pose(obj, cam, normal, quat, out_path)
                
        # Renderizar poses de test
        for p in test_poses:
            p_idx = p["pose_index"]
            quat = p["orientation_quat"]
            normal = p.get("contact_normal")
            out_path = os.path.join(images_dir, f"{ref}_pose_test_{p_idx}.png")
            render_pose(obj, cam, normal, quat, out_path)
                
    print("[RenderFotorrealista] Renderizado completado.")

if __name__ == "__main__":
    main()
