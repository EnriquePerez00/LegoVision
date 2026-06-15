# -*- coding: utf-8 -*-
"""scripts/generate_test10_dataset.py
Blender rendering script for rendering 10 random pieces test dataset.
"""
from __future__ import annotations
import os, sys, random, math, json

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

from config_loader import cfg
from generate_synthetic_set import (
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object
from generate_test_set import setup_lab_lightbox
from logger import get_logger

log = get_logger("blender")

# Constants
BELT_WIDTH_BU = 2.0
BELT_LENGTH_BU = 12.0
BELT_THICKNESS_BU = 0.1
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
TAA_SAMPLES_OPT = 8

# Setup cameras
CAM_CEN_LOC = (0.0, 0.0, 1.5)
CAM_LAT_LOC = (1.5, 0.0, 0.25)
LOOK_AT = (0.0, 0.0, 0.0)

def apply_eevee_optimizations(scene):
    try:
        scene.eevee.taa_render_samples = TAA_SAMPLES_OPT
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
    except Exception as e:
        log.warning(f"[opt] EEVEE opt parcial: {e}")

def create_belt_collider():
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (BELT_LENGTH_BU, BELT_WIDTH_BU, BELT_THICKNESS_BU)
    belt.location = (0.0, 0.0, -BELT_THICKNESS_BU / 2.0)
    
    mat = bpy.data.materials.new("Belt_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
        bsdf.inputs["Roughness"].default_value = 0.8
    belt.data.materials.append(mat)

def create_floor():
    bpy.ops.mesh.primitive_plane_add(size=100.0)
    floor = bpy.context.active_object
    floor.name = "Lab_Floor"
    floor.location = (0.0, 0.0, -0.2)

def setup_camera(name, loc):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.name = name
    cam.data.lens = 27.0
    cam.data.sensor_width = 36.0
    
    # Point at origin
    direction = mathutils.Vector(LOOK_AT) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    return cam

def setup_cameras():
    cam_c = setup_camera("Cam_Cenital", CAM_CEN_LOC)
    cam_l = setup_camera("Cam_Lateral", CAM_LAT_LOC)
    return cam_c, cam_l

def cleanup_piece_objects():
    for o in list(bpy.context.scene.objects):
        if o.name not in ["Cam_Cenital", "Cam_Lateral", "Conveyor_Belt_Plane", "Lab_Floor", "Lightbox_Ceiling"]:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

def get_2d_bbox(obj, scene, camera):
    bbox_coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs, ys = [], []
    for v in bbox_coords:
        co_2d = world_to_camera_view(scene, camera, v)
        xs.append(co_2d.x)
        ys.append(co_2d.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0)),
    ]

def sample_valid_position(part_obj, scene, cam_cen, cam_lat, margin_bu=0.2):
    # conveyor belt sampling region: belt_width = 2.0 BU
    sample_range_x = 0.5
    sample_range_y = 0.5
    for _ in range(100):
        rx = random.uniform(-sample_range_x, sample_range_x)
        ry = random.uniform(-sample_range_y, sample_range_y)
        part_obj.location = (rx, ry, 0.0)
        bpy.context.view_layer.update()
        
        # Snap to belt
        bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
        min_z = min(pt.z for pt in bbox_world)
        part_obj.location.z = -min_z + 0.005
        bpy.context.view_layer.update()
        
        # Check inside margins
        b_cen = get_2d_bbox(part_obj, scene, cam_cen)
        b_lat = get_2d_bbox(part_obj, scene, cam_lat)
        
        def ok(b):
            return b[0] > 0.02 and b[1] > 0.02 and b[2] < 0.98 and b[3] < 0.98
            
        if ok(b_cen) and ok(b_lat):
            return True
            
    # Fallback to center
    part_obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
    min_z = min(pt.z for pt in bbox_world)
    part_obj.location.z = -min_z + 0.005
    return True

def main():
    sel_path = os.path.join(project_root, "data", "test10", "test10_selection.json")
    with open(sel_path, "r", encoding="utf-8") as f:
        selection = json.load(f)
        
    output_dir = os.path.join(project_root, "data", "test10")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read stable poses cache
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        poses_cache = json.load(f)

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    cam_c, cam_l = setup_cameras()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    apply_eevee_optimizations(scene)

    results_meta = []
    cameras = {"cenital": cam_c, "lateral": cam_l}

    for i, item in enumerate(selection):
        part_ref = item["part_ref"]
        color_hex = item["color_hex"]
        pose_idx = item["pose_index"]
        color_code = item["color_code"]
        
        # Load mesh
        part_path = get_ldraw_part_path(part_ref)
        existing_objects = set(bpy.context.scene.objects)
        part_obj = None

        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing_objects]
                par = next((o for o in new_objs if o.parent is None), None)
                if par:
                    part_obj = get_single_mesh_object(par)
            except Exception as e:
                log.warning(f"import LDraw {part_ref}: {e}")

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            log.error(f"No se pudo cargar mesh para {part_ref}")
            continue

        bpy.ops.object.select_all(action='DESELECT')
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        from generate_test_set import _normalize_piece
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        mat = create_abs_plastic_material(f"#{color_hex}")
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        
        # Find exact pose dict
        part_poses = poses_cache.get(part_ref, [])
        pose_dict = next((p for p in part_poses if p.get("pose_index") == pose_idx), None)
        if not pose_dict:
            pose_dict = {"pose_index": pose_idx, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}

        # Apply pose orientation + Z rot
        quat = pose_dict.get("orientation_quat")
        if quat and len(quat) == 4:
            part_obj.rotation_mode = 'QUATERNION'
            part_obj.rotation_quaternion = mathutils.Quaternion(quat)
        else:
            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler = mathutils.Euler(pose_dict.get("orientation_euler", [0, 0, 0]))

        part_obj.rotation_mode = 'XYZ'
        part_obj.rotation_euler.z = random.uniform(0, 2.0 * math.pi)

        # Place at valid random position
        sample_valid_position(part_obj, scene, cam_c, cam_l)

        # Build name using the index
        # We need the filenames to be format: sample500_018_2431_156_p2_cenital.png
        # Let's use format: sample500_000_<part_ref>_<color_code>_p<pose_idx>_cenital.png to match run_evaluation.py's regex!
        # Wait, what regex does run_evaluation.py use to parse filenames?
        # format: sample500_(\\d+)_([0-9a-zA-Z]+)_(\\d+)_p(\\d+)_cenital.png
        # Let's use exactly: sample500_{i:03d}_{part_ref}_{color_code}_p{pose_idx}_cenital.png
        
        sample_meta = {
            "index": i,
            "ref": part_ref,
            "pose_index": pose_idx,
            "original_pose_index": pose_idx,
            "color_code": color_code,
            "color_hex": "#" + color_hex,
            "lateral_height_gt": float(pose_dict.get("lateral_height", 8.0)),
            "effective_height_gt": float(pose_dict.get("effective_height", 4.0)),
            "zenith_silhouette_area_gt": float(pose_dict.get("zenith_silhouette_area", 100.0)),
            "zenith_observable_area_gt": float(pose_dict.get("zenith_observable_area", 100.0)),
            "position_bu": list(part_obj.location),
            "z_rotation_rad": float(part_obj.rotation_euler.z),
            "cameras": {},
        }

        # Render from both cameras
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()

            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"sample500_{i:03d}_{part_ref}_{color_code}_p{pose_idx}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path

            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"[WARN] Render fallido {cam_name} muestra {i}: {e}")
                continue

            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "bbox_norm": bbox_norm,
                "image_path": file_path,
            }

        if len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"[OK] Muestra {i}: {part_ref} | pose={pose_idx}")
        else:
            print(f"[WARN] Muestra {i} incompleta, descartada.")

        cleanup_piece_objects()

    # Save metadata matching dataset format
    meta_path = os.path.join(output_dir, "test10_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": "75078-1",
            "render_engine": "BLENDER_EEVEE",
            "resolution": "640x640",
            "samples_count": len(results_meta),
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[TestGen DONE] {len(results_meta)} muestras generadas en {output_dir}")

if __name__ == "__main__":
    main()
