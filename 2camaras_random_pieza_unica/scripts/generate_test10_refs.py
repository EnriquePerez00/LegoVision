# -*- coding: utf-8 -*-
"""scripts/generate_test10_refs.py
Blender rendering script for rendering 10 random pieces reference images.
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

def main():
    sel_path = os.path.join(project_root, "data", "test10", "test10_selection.json")
    with open(sel_path, "r", encoding="utf-8") as f:
        selection = json.load(f)
        
    out_dir = os.path.join(project_root, "data", "test10", "dinov2_refs")
    for c in ["cenital", "lateral"]:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
        
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
    scene.render.film_transparent = True
    scene.render.resolution_x = 384
    scene.render.resolution_y = 384
    apply_eevee_optimizations(scene)

    for item in selection:
        part_ref = item["part_ref"]
        color_hex = item["color_hex"]
        pose_idx = item["pose_index"]
        
        log.info(f"Rendering refs for {part_ref} pose={pose_idx} color={color_hex}")
        
        # Find exact pose dict
        part_poses = poses_cache.get(part_ref, [])
        pose_dict = next((p for p in part_poses if p.get("pose_index") == pose_idx), None)
        if not pose_dict:
            pose_dict = {"pose_index": pose_idx, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}
            
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
        
        # Scale & Normalize
        from generate_test_set import _normalize_piece
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        mat = create_abs_plastic_material(f"#{color_hex}")
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        
        n_rots = 12
        rot_step = 2.0 * math.pi / n_rots

        for rot_i in range(n_rots):
            rot_rad = rot_i * rot_step
            rot_deg = int(round(math.degrees(rot_rad)))

            # Apply pose orientation
            quat = pose_dict.get("orientation_quat")
            if quat and len(quat) == 4:
                part_obj.rotation_mode = 'QUATERNION'
                part_obj.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler = mathutils.Euler(pose_dict.get("orientation_euler", [0, 0, 0]))

            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler.z += rot_rad

            part_obj.location = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
            min_z = min(pt.z for pt in bbox_world)
            part_obj.location.z = -min_z + 0.02
            bpy.context.view_layer.update()

            fname = f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png"

            # Hide conveyor elements for clean alpha
            _hide_targets = ["Lab_Floor", "Conveyor_Belt_Plane", "Side_Rail_L", "Side_Rail_R"]
            _prev_hide = {}
            for _n in _hide_targets:
                _o = bpy.data.objects.get(_n)
                if _o is not None:
                    _prev_hide[_n] = _o.hide_render
                    _o.hide_render = True

            # Cenital
            scene.camera = cam_c
            scene.render.filepath = os.path.join(out_dir, "cenital", fname)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"Cenital render failed: {e}")

            # Lateral
            scene.camera = cam_l
            scene.render.filepath = os.path.join(out_dir, "lateral", fname)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"Lateral render failed: {e}")

            # Restore visibility
            for _n, _prev in _prev_hide.items():
                _o = bpy.data.objects.get(_n)
                if _o is not None:
                    _o.hide_render = _prev

        cleanup_piece_objects()

if __name__ == "__main__":
    main()
