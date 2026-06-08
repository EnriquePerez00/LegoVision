# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_yolo_lateral0_dataset.py
==================================================================
Phase 2: Genera el dataset YOLO lateral con la cámara a Z=0.05 BU
(casi a ras del suelo) para entrenar `yolo_lateral0.pt`.

Mismo proceso que `generate_yolo_training_dataset.py --camera lateral`,
pero con cam_lateral_loc = (15.0, 0.0, 0.05) en lugar de (15.0, 0.0, 2.5).
Genera 1000 imágenes (mismo número que el dataset lateral original).

Output
------
    data/yolo_lateral0/images/train_NNNNN.png
    data/yolo_lateral0/labels/train_NNNNN.txt

Uso
---
    blender -b -P 2camaras_pieza_unica/scripts/generate_yolo_lateral0_dataset.py -- [--num_frames N]
"""
from __future__ import annotations

import os
import sys
import random
import math
import json
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovision_root = os.path.dirname(project_root)
sys.path.insert(0, legovision_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from config_loader import cfg  # noqa: E402
from generate_synthetic_set import (  # noqa: E402
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object  # noqa: E402
from _pose_utils import (  # noqa: E402
    apply_stable_pose,
    get_stable_poses_for_ref,
    select_pose_tarps,
    TARPS_MIN_TIPPING_DEFAULT,
)

# ── Config ────────────────────────────────────────────────────────────────────
SELECTED_PARTS = cfg.pieces.selected_parts
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
RENDER_RES = cfg.render.resolution.width

# ── Posición cámara lateral a ras del suelo ──────────────────────────────────
CAM_LATERAL0_LOC = (15.0, 0.0, 0.05)  # Z=0.05 BU ≈ 1.5 mm

OUTPUT_DIR = os.path.join(project_root, "data", "yolo_lateral0")
NUM_FRAMES_DEFAULT = 1000  # igual que el dataset yolo_lateral original


# ── Helpers de escena ─────────────────────────────────────────────────────────
def load_lego_color_palette():
    path = os.path.join(project_root, "database", "color_catalog.json")
    fallback = ["#A0A5A9", "#1B1B1B", "#C91A09", "#F2F3F2", "#FE8A18",
                "#0A3C9F", "#5A5A5A", "#3B5E28", "#F2CD37", "#FF7E14"]
    if not os.path.exists(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    pal = []
    for _, info in catalog.items():
        hx = info.get("hex", "")
        if hx and info.get("alpha", 1.0) >= 0.6 and \
                info.get("material_type", "solid") in ("solid", "metallic", "rubber"):
            pal.append(hx if hx.startswith("#") else "#" + hx)
    return list(set(pal)) or fallback


def get_stable_poses(part_ref):
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    return get_stable_poses_for_ref(part_ref, cache_path)


def setup_lab_lightbox():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral0", "Lab_Floor"}
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT" and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.3

    neutral = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 12.0))
    m = bpy.context.active_object
    m.name = "Lab_Main_Dome"
    m.data.size = 35.0; m.data.size_y = 35.0
    m.data.shape = "RECTANGLE"; m.data.color = neutral; m.data.energy = 2000.0

    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    for wname, wloc in [
        ("Lab_Wall_N", (0.0, +12.0, 6.0)),
        ("Lab_Wall_S", (0.0, -12.0, 6.0)),
        ("Lab_Wall_E", (+12.0, 0.0, 6.0)),
        ("Lab_Wall_W", (-12.0, 0.0, 6.0)),
    ]:
        bpy.ops.object.light_add(type="AREA", location=wloc)
        wp = bpy.context.active_object
        wp.name = wname
        wp.data.size = 20.0; wp.data.size_y = 12.0
        wp.data.shape = "RECTANGLE"; wp.data.color = neutral; wp.data.energy = 600.0
        tr = wp.constraints.new(type="TRACK_TO")
        tr.target = target; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.5))
    gf = bpy.context.active_object
    gf.name = "Lab_Ground_Fill"
    gf.data.size = 30.0; gf.data.size_y = 30.0
    gf.data.shape = "RECTANGLE"; gf.data.color = neutral; gf.data.energy = 200.0
    gf.rotation_euler = (math.pi, 0.0, 0.0)


def create_floor():
    if "Lab_Floor" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Lab_Floor"].select_set(True)
        bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, -2.0))
    floor = bpy.context.active_object
    floor.name = "Lab_Floor"
    floor.scale = (60.0, 60.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.get("Lab_Floor_Black")
    if not mat:
        mat = bpy.data.materials.new("Lab_Floor_Black")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Roughness"].default_value = 1.0
    floor.data.materials.clear()
    floor.data.materials.append(mat)


def create_belt_collider():
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    ht = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.get("Belt_Material")
    if not mat:
        mat = bpy.data.materials.new("Belt_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
            bsdf.inputs["Roughness"].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)

    for name in ["Side_Rail_L", "Side_Rail_R"]:
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.delete()
    rail_w, rail_h = 0.2, 0.4
    for xoff, rname in [(-BELT_WIDTH_BU / 2.0 + rail_w / 2.0, "Side_Rail_L"),
                         ( BELT_WIDTH_BU / 2.0 - rail_w / 2.0, "Side_Rail_R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(xoff, 0.0, rail_h / 2.0))
        rail = bpy.context.active_object
        rail.name = rname
        rail.scale = (rail_w, BELT_LENGTH_BU, rail_h)
        bpy.ops.object.transform_apply(scale=True)


def setup_cameras():
    """Configura cámara cenital (estándar) + cámara lateral0 (Z=0.05)."""
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    # Cenital (no cambia)
    cam_c_name = "Cam_Cenital"
    if cam_c_name in bpy.data.objects:
        cam_c = bpy.data.objects[cam_c_name]
    else:
        bpy.ops.object.camera_add(location=(0, 0, 15.0))
        cam_c = bpy.context.active_object
        cam_c.name = cam_c_name
    cam_c.location = (0.0, 0.0, 15.0)
    cam_c.constraints.clear()
    tr_c = cam_c.constraints.new(type="TRACK_TO")
    tr_c.target = target; tr_c.track_axis = "TRACK_NEGATIVE_Z"; tr_c.up_axis = "UP_Y"
    cam_c.data.type = "PERSP"; cam_c.data.lens = 27.0
    cam_c.data.clip_start = 0.01; cam_c.data.clip_end = 100.0

    # Lateral0 — Z=0.05 BU (ÚNICO CAMBIO respecto al script original)
    cam_l_name = "Cam_Lateral0"
    if cam_l_name in bpy.data.objects:
        cam_l = bpy.data.objects[cam_l_name]
    else:
        bpy.ops.object.camera_add(location=CAM_LATERAL0_LOC)
        cam_l = bpy.context.active_object
        cam_l.name = cam_l_name
    cam_l.location = CAM_LATERAL0_LOC
    cam_l.constraints.clear()
    tr_l = cam_l.constraints.new(type="TRACK_TO")
    tr_l.target = target; tr_l.track_axis = "TRACK_NEGATIVE_Z"; tr_l.up_axis = "UP_Y"
    cam_l.data.type = "PERSP"; cam_l.data.lens = 27.0
    cam_l.data.clip_start = 0.01; cam_l.data.clip_end = 100.0

    return cam_c, cam_l


def _normalize_piece(obj):
    if not obj.data or not hasattr(obj.data, "vertices"):
        return
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if mx < 1e-6:
        return
    factor = 0.04 if mx > 5.0 else 1.0
    cx = (max(xs) + min(xs)) / 2.0; cy = (max(ys) + min(ys)) / 2.0
    cz = (max(zs) + min(zs)) / 2.0
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) * factor
        v.co.y = (v.co.y - cy) * factor
        v.co.z = (v.co.z - cz) * factor
    obj.data.update()
    obj.scale = (1.0, 1.0, 1.0)
    obj.location = (0.0, 0.0, 0.0)


def cleanup_piece():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral0", "Lab_Floor",
            "Lab_Main_Dome", "Lab_Wall_N", "Lab_Wall_S", "Lab_Wall_E", "Lab_Wall_W",
            "Lab_Ground_Fill"}
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and o.type not in ("CAMERA", "LIGHT", "EMPTY"):
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def compute_bbox_yolo(obj, cam, scene):
    """Devuelve [cx, cy, w, h] YOLO normalizado, o None."""
    world_verts = []
    if obj.type == "MESH" and obj.data:
        m = obj.matrix_world
        world_verts = [m @ v.co for v in obj.data.vertices]
    if not world_verts:
        world_verts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    if not world_verts:
        return None
    xs, ys = [], []
    for v in world_verts:
        c = world_to_camera_view(scene, cam, v)
        xs.append(c.x); ys.append(c.y)
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0
    if w < 0.005 or h < 0.005:
        return None
    return [x0 + w / 2.0, 1.0 - (y0 + h / 2.0), w, h]


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_dataset(output_dir: str, num_frames: int):
    import time as _time
    t0 = _time.perf_counter()

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print("=" * 72)
    print("YOLO LATERAL0 DATASET — cámara lateral Z=0.05 BU")
    print(f"  Output     : {output_dir}")
    print(f"  Cam loc    : {CAM_LATERAL0_LOC}")
    print(f"  Num frames : {num_frames}")
    print("=" * 72)

    color_palette = load_lego_color_palette()

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    _, cam_l = setup_cameras()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    scene.camera = cam_l  # dataset lateral → solo usamos la cámara lateral0

    empty_ratio = getattr(cfg.yolo.dataset, "empty_frame_ratio", 0.05)
    num_empty = max(1, int(num_frames * empty_ratio))
    num_piece = num_frames - num_empty
    frame_types = ["piece"] * num_piece + ["empty"] * num_empty
    random.shuffle(frame_types)

    saved = 0
    for fi, ftype in enumerate(frame_types):
        cleanup_piece()

        if ftype == "empty":
            scene.camera = cam_l
            img_fn = f"train_{fi:05d}.png"
            lbl_fn = f"train_{fi:05d}.txt"
            scene.render.filepath = os.path.join(images_dir, img_fn)
            bpy.ops.render.render(write_still=True)
            open(os.path.join(labels_dir, lbl_fn), "w").close()
            saved += 1
            continue

        # Pieza aleatoria con pose TARPS
        part_ref = random.choice(SELECTED_PARTS)
        poses = get_stable_poses(part_ref)
        pose = select_pose_tarps(poses)
        if pose is None:
            pose = {"orientation_quat": [1.0, 0.0, 0.0, 0.0]}

        part_path = get_ldraw_part_path(part_ref)
        existing = set(bpy.context.scene.objects)
        part_obj = None

        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing]
                par = next((o for o in new_objs if o.parent is None), None)
                if par:
                    part_obj = get_single_mesh_object(par)
            except Exception as e:
                print(f"[WARN] import LDraw {part_ref}: {e}")

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            continue

        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        color_hex = random.choice(color_palette)
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)

        scene.camera = cam_l
        bpy.context.view_layer.update()

        bb = compute_bbox_yolo(part_obj, cam_l, scene)
        if not bb:
            continue

        img_fn = f"train_{fi:05d}.png"
        lbl_fn = f"train_{fi:05d}.txt"
        scene.render.filepath = os.path.join(images_dir, img_fn)

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            print(f"[WARN] Render falló frame {fi}: {e}")
            continue

        with open(os.path.join(labels_dir, lbl_fn), "w") as lf:
            lf.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        saved += 1
        if saved % 100 == 0:
            elapsed = _time.perf_counter() - t0
            print(f"  [{saved}/{num_frames}] guardados  ({elapsed:.0f}s)")

    cleanup_piece()
    elapsed = _time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"DONE — {saved} frames guardados en {output_dir}  ({elapsed:.1f}s)")
    print("=" * 72)


if __name__ == "__main__":
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    pa = parser.parse_known_args(raw)[0]
    generate_dataset(pa.output_dir, pa.num_frames)
