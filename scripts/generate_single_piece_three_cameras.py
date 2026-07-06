# -*- coding: utf-8 -*-
"""scripts/generate_single_piece_three_cameras.py
Renderiza cada pieza única de un set desde 3 cámaras en línea sobre una cinta de 10cm.
"""
import os
import sys
import random
import math
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))
sys.path.append(os.path.join(project_root, 'scripts'))

try:
    import bpy
    import bpy_extras
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import (
    setup_physics_world,
    create_abs_plastic_material, apply_bevel_modifier,
    get_ldraw_part_path, generate_detailed_fallback_mesh,
)
from generate_synthetic_dataset import get_single_mesh_object
from inference import config

def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def _normalize_piece_to_scale(obj, is_ldraw=True):
    if is_ldraw:
        # LDraw to centimeters: 1 LDU = 0.4 mm = 0.04 cm
        factor = 0.04
        obj.scale = (factor, factor, factor)
    else:
        # For fallback meshes, fit them within a 2-4 cm bounding box
        bbox = _get_world_bbox(obj)
        dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
        dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
        dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
        max_dim = max(dim_x, dim_y, dim_z)
        if max_dim > 1e-6:
            factor = 2.0 / max_dim
            obj.scale = (factor, factor, factor)
        else:
            factor = 1.0
            
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return factor

def cleanup_scene():
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()

def get_2d_bbox(obj, scene, camera):
    bbox_coords = _get_world_bbox(obj)
    xs = []
    ys = []
    for v in bbox_coords:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, v)
        xs.append(co_2d.x)
        ys.append(co_2d.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0))
    ]

def get_stable_poses_from_db_subprocess(part_ref):
    import subprocess
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    code = f"""
import sys, json
sys.path.append('{project_root}')
try:
    from core.db import supabase_client
    poses = supabase_client.get_stable_poses('{part_ref}')
    print(json.dumps(poses))
except Exception as e:
    print(json.dumps([]))
"""
    try:
        res = subprocess.run([python_exec, "-c", code], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            return json.loads(res.stdout.strip())
    except Exception as e:
        print(f"[WARN] Error consultando DB en subproceso: {e}")
    return []

def main():
    # 1. Analizar argumentos
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--output_dir", type=str, required=True)
    parsed_args = parser.parse_known_args(args)[0]
    
    set_id = parsed_args.set_id
    output_dir = parsed_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Configurar motor Cycles con aceleración METAL (Apple Silicon GPU)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.film_transparent = False  # Render solid backgrounds (belt + lateral walls)
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 1080
    
    # Limitar hilos de Cycles dinámicamente preservando capacidad CPU
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = max(1, config.AUTO_CPU_THREADS // 2)
    
    try:
        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons['cycles'].preferences
        cycles_preferences.compute_device_type = 'METAL' if config.IS_APPLE_SILICON else 'CUDA'
        cycles_preferences.get_devices()
        for device in cycles_preferences.devices:
            if device.type == 'METAL' or (not config.IS_APPLE_SILICON and device.type == 'CUDA'):
                device.use = True
                print(f"[Blender SinglePiece] Activada aceleración en Cycles GPU: {device.name}")
        scene.cycles.device = 'GPU'
    except Exception as e:
        print(f"[Blender SinglePiece Warning] No se pudo activar GPU, usando CPU: {e}")
        scene.cycles.device = 'CPU'
        
    cleanup_scene()
    
    # 3. Crear cinta de 10cm de ancho
    # 10cm = 10.0 BU. Creamos un plano de 10.0 x 100.0 x 0.1 (1m de longitud)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.05))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_10cm"
    belt.scale = (10.0, 100.0, 0.1)
    bpy.ops.object.transform_apply(scale=True)
    belt.is_shadow_catcher = False
    
    # Crear material Azul Petróleo para la cinta
    belt_mat = bpy.data.materials.new(name="Petrol_Blue_Belt")
    belt_mat.use_nodes = True
    nodes = belt_mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        # Base Color: Petrol Blue (RGB 0.059, 0.165, 0.239)
        bsdf.inputs['Base Color'].default_value = (0.059, 0.165, 0.239, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.6
        bsdf.inputs['Specular IOR Level'].default_value = 0.2
    belt.data.materials.clear()
    belt.data.materials.append(belt_mat)
    
    # Crear paredes laterales (Guards) de aluminio mate en los márgenes de la cinta (X = -5.1 y X = 5.1)
    aluminum_mat = bpy.data.materials.new(name="Matte_Aluminum_Guard")
    aluminum_mat.use_nodes = True
    nodes_al = aluminum_mat.node_tree.nodes
    bsdf_al = nodes_al.get("Principled BSDF")
    if bsdf_al:
        # Base Color: Matte Aluminum Grey (RGB 0.55, 0.55, 0.57)
        bsdf_al.inputs['Base Color'].default_value = (0.55, 0.55, 0.57, 1.0)
        bsdf_al.inputs['Metallic'].default_value = 1.0
        bsdf_al.inputs['Roughness'].default_value = 0.4
        
    # Guard Izquierdo (X = -5.1)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-5.1, 0.0, 5.0))
    guard_l = bpy.context.active_object
    guard_l.name = "Guard_Left"
    guard_l.scale = (0.1, 100.0, 10.0)
    bpy.ops.object.transform_apply(scale=True)
    guard_l.data.materials.clear()
    guard_l.data.materials.append(aluminum_mat)
    
    # Guard Derecho (X = 5.1)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(5.1, 0.0, 5.0))
    guard_r = bpy.context.active_object
    guard_r.name = "Guard_Right"
    guard_r.scale = (0.1, 100.0, 10.0)
    bpy.ops.object.transform_apply(scale=True)
    guard_r.data.materials.clear()
    guard_r.data.materials.append(aluminum_mat)
    
    # 4. Configurar iluminación difusa sin sombras marcadas
    # Luz cenital difusa
    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 20.0))
    top_light = bpy.context.active_object
    top_light.name = "Diffuse_Top_Light"
    top_light.data.size = 30.0
    top_light.data.energy = 250.0
    
    # Configurar iluminación ambiental del mundo
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Strength'].default_value = 0.4
        bg_node.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1.0)
        
    # 5. Crear Target y Cámaras
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    camera_target = bpy.context.active_object
    camera_target.name = "Camera_Target"
    
    # Cámara 1: Cenital (Z=15cm, mirando abajo)
    bpy.ops.object.camera_add(location=(0.0, 0.0, 15.0))
    cam_cenital = bpy.context.active_object
    cam_cenital.name = "Cam_Cenital"
    cam_cenital.rotation_euler = (0.0, 0.0, 0.0)
    cam_cenital.data.type = 'PERSP'
    cam_cenital.data.lens = 52.5  # 50% zoom (35mm -> 52.5mm)
    
    # Cámara 2: Lateral Izquierda (X=-5, Y=0, Z=15, mirando a target)
    bpy.ops.object.camera_add(location=(-5.0, 0.0, 15.0))
    cam_lat_l = bpy.context.active_object
    cam_lat_l.name = "Cam_Lat_L"
    cam_lat_l.data.type = 'PERSP'
    cam_lat_l.data.lens = 52.5  # 50% zoom
    track_l = cam_lat_l.constraints.new(type='TRACK_TO')
    track_l.target = camera_target
    track_l.track_axis = 'TRACK_NEGATIVE_Z'
    track_l.up_axis = 'UP_Y'
    
    # Cámara 3: Lateral Derecha (X=5, Y=0, Z=15, mirando a target)
    bpy.ops.object.camera_add(location=(5.0, 0.0, 15.0))
    cam_lat_r = bpy.context.active_object
    cam_lat_r.name = "Cam_Lat_R"
    cam_lat_r.data.type = 'PERSP'
    cam_lat_r.data.lens = 52.5  # 50% zoom
    track_r = cam_lat_r.constraints.new(type='TRACK_TO')
    track_r.target = camera_target
    track_r.track_axis = 'TRACK_NEGATIVE_Z'
    track_r.up_axis = 'UP_Y'
    
    # 6. Cargar catálogo de piezas del set
    from core.db.set_catalog import REAL_SETS
    if set_id not in REAL_SETS:
        print(f"[ERROR] Set {set_id} no encontrado en catálogo local.")
        sys.exit(1)
        
    set_data = REAL_SETS[set_id]
    
    # Filtrar piezas únicas por ref y color
    unique_parts = {}
    
    # Agregar minifiguras
    for fig in set_data.get("minifigures", []):
        key = (fig["ref"], "15") # minifigs in default white/neutral ldraw color
        unique_parts[key] = {
            "ref": fig["ref"],
            "color_hex": "#F2F3F2",
            "color_code": "15",
            "name": fig["name"]
        }
        
    # Agregar piezas comunes
    for p in set_data.get("parts", []):
        key = (p["ref"], p.get("color_code", "0"))
        unique_parts[key] = {
            "ref": p["ref"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_code": p.get("color_code", "0"),
            "name": p.get("name", "Pieza Lego")
        }
        
    print(f"[Blender SinglePiece] Cargadas {len(unique_parts)} piezas únicas para el set {set_id}.")
    
    # 7. Renderizar cada pieza
    results_meta = []
    
    for idx, (key, p_info) in enumerate(unique_parts.items()):
        ref = p_info["ref"]
        color_hex = p_info["color_hex"]
        color_code = p_info["color_code"]
        name = p_info["name"]
        
        print(f"[{idx+1}/{len(unique_parts)}] Procesando pieza: {ref} (color {color_code})...")
        
        # Cargar malla
        part_path = get_ldraw_part_path(ref)
        if not part_path and ref.startswith("sw"):
            try:
                from scripts.assemble_minifig import build_minifig
                build_minifig(ref)
                part_path = get_ldraw_part_path(ref)
            except Exception as e:
                print(f"Error cargando minifig {ref}: {e}")
                
        existing_objects = set(bpy.context.scene.objects)
        is_ldraw = False
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing_objects]
                par = next((o for o in new_objs if o.parent is None), None)
                if par:
                    part_obj = get_single_mesh_object(par)
                    is_ldraw = True
                else:
                    generate_detailed_fallback_mesh(ref)
                    part_obj = bpy.context.active_object
            except Exception:
                generate_detailed_fallback_mesh(ref)
                part_obj = bpy.context.active_object
        else:
            generate_detailed_fallback_mesh(ref)
            part_obj = bpy.context.active_object
            
        if not part_obj:
            print(f"[WARN] No se pudo cargar malla para {ref}. Omitiendo.")
            continue
            
        # Posicionar en origen de geometría y normalizar
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        _normalize_piece_to_scale(part_obj, is_ldraw=is_ldraw)
        
        # Obtener stable poses y aplicar una aleatoria
        poses = get_stable_poses_from_db_subprocess(ref)
        if poses:
            pose = random.choice(poses)
            quat = pose.get("orientation_quat")
            if quat and len(quat) == 4:
                part_obj.rotation_mode = 'QUATERNION'
                part_obj.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                euler = pose.get("orientation_euler")
                if euler and len(euler) == 3:
                    part_obj.rotation_mode = 'XYZ'
                    part_obj.rotation_euler = mathutils.Euler(euler)
        else:
            # Fallback simple
            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler = (0, 0, 0)
            
        # Yaw aleatorio adicional
        part_obj.rotation_mode = 'XYZ'
        part_obj.rotation_euler.z += random.uniform(0.0, math.pi * 2)
        
        bpy.context.view_layer.update()
        
        # Ajustar altura Z para descansar en la cinta
        bbox = _get_world_bbox(part_obj)
        min_z = min(pt.z for pt in bbox)
        part_obj.location = (0.0, 0.0, -min_z + 0.02)
        
        # Aplicar material plástico de color correspondiente
        apply_bevel_modifier(part_obj)
        mat_abs = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat_abs)
        
        bpy.context.view_layer.update()
        
        # Renderizar desde las 3 cámaras
        cameras = {
            "cenital": cam_cenital,
            "lateral_l": cam_lat_l,
            "lateral_r": cam_lat_r
        }
        
        piece_render_meta = {
            "ref": ref,
            "name": name,
            "color_hex": color_hex,
            "color_code": color_code,
            "cameras": {}
        }
        
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            
            # Obtener 2D bounding box
            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            
            # Formatear ruta de salida
            file_name = f"single_{ref}_{color_code}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            piece_render_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "bbox_norm": bbox_norm,
                "image_url": f"/renders/multicam/{file_name}"
            }
            
        results_meta.append(piece_render_meta)
        
        # Limpieza de la pieza en escena
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.ops.object.delete()
        
    # Guardar metadatos del set renders
    meta_path = os.path.join(output_dir, "multicam_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": set_id,
            "pieces_count": len(results_meta),
            "renders": results_meta
        }, f, indent=2, ensure_ascii=False)
        
    print(f"[Blender SinglePiece DONE] Guardados renders y metadatos JSON en: {output_dir}")

if __name__ == "__main__":
    main()
