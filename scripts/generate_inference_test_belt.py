# -*- coding: utf-8 -*-
"""scripts/generate_inference_test_belt.py
Genera una imagen fotorrealista con las piezas del set sobre la cinta
usando posiciones estables de la BD concentradas y sin solapes (distancia min 10mm).
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

if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world, setup_studio_lighting,
        create_abs_plastic_material, apply_bevel_modifier,
        get_ldraw_part_path, generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
    from scene_config import (
        BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU,
        TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z,
        CORNER_LIGHT_OFFSET_XY, CORNER_LIGHT_Z, CORNER_LIGHT_SIZE, CORNER_LIGHT_ENERGY,
    )
    
    _OFF = CORNER_LIGHT_OFFSET_XY
    _CZ = CORNER_LIGHT_Z
    CORNER_LIGHT_NAMES = ["Corner_Light_PP", "Corner_Light_PN", "Corner_Light_NP", "Corner_Light_NP_2"] # avoid name collision if any
    CORNER_LIGHT_POSITIONS = [
        ( _OFF,  _OFF, _CZ),
        ( _OFF, -_OFF, _CZ),
        (-_OFF,  _OFF, _CZ),
        (-_OFF, -_OFF, _CZ),
    ]
else:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def _normalize_piece_to_scale(obj):
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
    bpy.ops.object.select_all(action="DESELECT")
    keep = {"Camera", "Camera_Target"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()

def setup_lights_fov():
    """Configura las condiciones de iluminación idénticas al FOV (DINOv2)."""
    # Eliminar luces existentes
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    # 4 Luces de esquina uniformes
    for name, pos in zip(CORNER_LIGHT_NAMES, CORNER_LIGHT_POSITIONS):
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
            obj.location = pos
        else:
            bpy.ops.object.light_add(type='AREA', location=pos)
            obj = bpy.context.active_object
            obj.name = name
        obj.data.size = CORNER_LIGHT_SIZE
        obj.data.energy = CORNER_LIGHT_ENERGY
        obj.rotation_euler = (0.0, 0.0, 0.0)

    # Luz cenital difusa
    top_name = "Top_Diffuse_Light"
    top = bpy.data.objects.get(top_name)
    if not top:
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, TOP_LIGHT_Z))
        top = bpy.context.active_object
        top.name = top_name
    top.location = (0.0, 0.0, TOP_LIGHT_Z)
    top.data.size = TOP_LIGHT_SIZE
    top.data.energy = TOP_LIGHT_ENERGY

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

def get_bounding_radius(obj):
    max_dist = 0.0
    def traverse(o):
        nonlocal max_dist
        if o.type == 'MESH' and o.data:
            for v in o.data.vertices:
                dist = math.sqrt(v.co.x**2 + v.co.y**2)
                if dist > max_dist:
                    max_dist = dist
        for child in o.children:
            traverse(child)
    traverse(obj)
    return max(0.6, max_dist)

def get_stable_poses_from_db_subprocess(part_ref):
    import subprocess
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    
    code = f"""
import sys, json
sys.path.append('{project_root}')
try:
    from database import supabase_client
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

def find_compact_position(r, placed_pieces, margin_bu, limit_x, limit_y):
    """Búsqueda compacta en espiral desde el centro (0,0) hacia fuera."""
    theta = 0.0
    d_theta = 0.1
    r_spiral = 0.0
    dr_spiral = 0.05
    
    while r_spiral < max(limit_x, limit_y):
        cx = r_spiral * math.cos(theta)
        cy = r_spiral * math.sin(theta)
        
        # Comprobar colisión con bordes de la cinta
        if (cx - r >= -limit_x) and (cx + r <= limit_x) and (cy - r >= -limit_y) and (cy + r <= limit_y):
            # Comprobar solapamiento con piezas ya colocadas
            overlap = False
            for px, py, pr in placed_pieces:
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < (r + pr + margin_bu):
                    overlap = True
                    break
            if not overlap:
                return cx, cy
                
        theta += d_theta
        r_spiral += dr_spiral
        
    return None

def main():
    enable_metal_gpu_acceleration()
    setup_physics_world()
    
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--pieces_in_field", type=int, default=30)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--is_rolling", type=str, default="true")
    parsed_args = parser.parse_known_args(args)[0]
    
    set_id = parsed_args.set_id
    pieces_in_field = parsed_args.pieces_in_field
    output_path = parsed_args.output_path
    is_rolling = parsed_args.is_rolling.lower() == "true"
    
    # 1. Cargar catálogo
    from database.set_catalog import REAL_SETS
    if set_id not in REAL_SETS:
        print(f"[ERROR] Set {set_id} no encontrado en el catálogo.")
        sys.exit(1)
        
    set_data = REAL_SETS[set_id]
    parts_list = []
    
    for fig in set_data.get("minifigures", []):
        for _ in range(fig["qty"]):
            parts_list.append({
                "ref": fig["ref"],
                "color_hex": "#F2F3F2",
                "color_code": "15",
                "name": fig["name"]
            })
            
    for p in set_data.get("parts", []):
        for _ in range(p["qty"]):
            parts_list.append({
                "ref": p["ref"],
                "color_hex": p.get("color_hex", "#A0A5A9"),
                "color_code": p.get("color_code", "0"),
                "name": p.get("name", "Pieza Lego")
            })
            
    random.seed(42)
    if len(parts_list) > pieces_in_field:
        parts_list = random.sample(parts_list, pieces_in_field)
    else:
        while len(parts_list) < pieces_in_field and len(parts_list) > 0:
            parts_list.append(random.choice(parts_list))
            
    cleanup_scene()
    
    # Crear plano de cinta
    half_thick = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_thick))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    belt.is_shadow_catcher = True
    
    # Configurar iluminación idéntica al FOV
    setup_lights_fov()
    
    # Cámara Ortográfica Cenital
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    camera.data.type = 'ORTHO'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.ortho_scale = 20.0
    camera.location = (0.0, 0.0, 25.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    
    if "Camera_Target" in bpy.data.objects:
        bpy.data.objects["Camera_Target"].location = (0, 0, 0)
    bpy.context.scene.camera = camera
    
    placed_pieces = [] # lista de tuplas (x, y, r) para colisiones
    pieces_objects = []
    
    # Pre-cargar plantillas para medir su radio
    loaded_templates = {}
    for p_info in parts_list:
        ref = p_info["ref"]
        if ref in loaded_templates:
            continue
            
        part_path = get_ldraw_part_path(ref)
        if not part_path and ref.startswith("sw"):
            try:
                from scripts.assemble_minifig import build_minifig
                build_minifig(ref)
                part_path = get_ldraw_part_path(ref)
            except Exception as e:
                print(f"Error ensamblando minifig {ref}: {e}")
                
        existing_objects = set(bpy.context.scene.objects)
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing_objects]
                par = next((o for o in new_objs if o.parent is None), None)
                if par:
                    temp_obj = get_single_mesh_object(par)
                else:
                    generate_detailed_fallback_mesh(ref)
                    temp_obj = bpy.context.active_object
            except Exception:
                generate_detailed_fallback_mesh(ref)
                temp_obj = bpy.context.active_object
        else:
            generate_detailed_fallback_mesh(ref)
            temp_obj = bpy.context.active_object
            
        if temp_obj:
            bpy.ops.object.select_all(action="DESELECT")
            temp_obj.select_set(True)
            bpy.context.view_layer.objects.active = temp_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            _normalize_piece_to_scale(temp_obj)
            radius = get_bounding_radius(temp_obj)
            
            # Mover a la colección temporal de plantillas
            if "Templates" not in bpy.data.collections:
                col = bpy.data.collections.new("Templates")
                bpy.context.scene.collection.children.link(col)
            else:
                col = bpy.data.collections["Templates"]
            col.hide_viewport = True
            col.hide_render = True
            
            for c in list(temp_obj.users_collection):
                c.objects.unlink(temp_obj)
            col.objects.link(temp_obj)
            
            loaded_templates[ref] = (temp_obj, radius)

    # Ordenar las piezas a colocar por tamaño (radio) descendente para empaquetado más óptimo
    parts_to_place = []
    for p_info in parts_list:
        ref = p_info["ref"]
        if ref in loaded_templates:
            temp_obj, radius = loaded_templates[ref]
            parts_to_place.append((radius, p_info))
            
    parts_to_place.sort(key=lambda x: x[0], reverse=True)
    
    margin_bu = 1.0  # 10 mm de distancia mínima
    limit_x = (BELT_WIDTH_BU / 2.0) - margin_bu
    limit_y = (BELT_LENGTH_BU / 2.0) - margin_bu
    
    for radius, p_info in parts_to_place:
        ref = p_info["ref"]
        color_hex = p_info["color_hex"]
        
        # Encontrar posición compacta sin colisión
        pos = find_compact_position(radius, placed_pieces, margin_bu, limit_x, limit_y)
        if not pos:
            print(f"[WARN] No se encontró espacio para colocar la pieza {ref} de radio {radius}.")
            continue
            
        cx, cy = pos
        placed_pieces.append((cx, cy, radius))
        
        # Clonar plantilla
        template_obj = loaded_templates[ref][0]
        oc = template_obj.copy()
        oc.data = template_obj.data.copy()
        oc.name = f"Placed_{ref}_{len(pieces_objects)}"
        
        bpy.context.scene.collection.objects.link(oc)
        oc.hide_viewport = False
        oc.hide_render = False
        
        # Seleccionar orientación de poses estables de la BD
        poses = get_stable_poses_from_db_subprocess(ref)
        if poses:
            if is_rolling:
                # Caída aleatoria uniforme
                pose = random.choice(poses)
            else:
                # Proporcional a stability_ratio
                ratios = [p.get("stability_ratio", 1.0) for p in poses]
                total_ratio = sum(ratios)
                if total_ratio > 0:
                    weights = [r / total_ratio for r in ratios]
                    pose = random.choices(poses, weights=weights, k=1)[0]
                else:
                    pose = random.choice(poses)
                    
            quat = pose.get("orientation_quat")
            if quat and len(quat) == 4:
                oc.rotation_mode = 'QUATERNION'
                oc.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                euler = pose.get("orientation_euler")
                if euler and len(euler) == 3:
                    oc.rotation_mode = 'XYZ'
                    oc.rotation_euler = mathutils.Euler(euler)
        else:
            # Fallback
            oc.rotation_mode = 'XYZ'
            oc.rotation_euler = (
                random.choice([0.0, math.pi]),
                random.choice([0.0, math.pi]),
                0.0
            )
            
        # Yaw Z aleatorio
        oc.rotation_mode = 'XYZ'
        oc.rotation_euler.z += random.uniform(0.0, math.pi * 2)
        
        bpy.context.view_layer.update()
        
        # Ajustar Z para que toque la cinta
        bbox = _get_world_bbox(oc)
        min_z = min(pt.z for pt in bbox)
        oc.location = (cx, cy, -min_z + 0.02)
        
        apply_bevel_modifier(oc)
        mat_abs = create_abs_plastic_material(color_hex)
        oc.data.materials.clear()
        oc.data.materials.append(mat_abs)
        
        pieces_objects.append({
            "obj": oc,
            "ref": ref,
            "name": p_info["name"],
            "color_hex": color_hex,
            "color_code": p_info["color_code"]
        })
        
    bpy.context.view_layer.update()
    
    # Motor de render
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.render.film_transparent = True
    
    res_x = 640
    res_y = 1920
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.filepath = output_path
    
    print(f"Renderizando cinta compacta a resolución: {res_x} x {res_y} px...")
    bpy.ops.render.render(write_still=True)
    
    # Calcular y guardar las Bounding Boxes en los metadatos JSON
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
            "belt_length_bu": BELT_LENGTH_BU,
            "resolution": [res_x, res_y],
            "pieces_placed": len(pieces_objects),
            "pieces_in_field": pieces_in_field,
            "is_rolling": is_rolling,
            "detections": bboxes_metadata
        }, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 Simulación estática compactada y metadatos JSON guardados en: {metadata_path}")

if __name__ == "__main__":
    main()
