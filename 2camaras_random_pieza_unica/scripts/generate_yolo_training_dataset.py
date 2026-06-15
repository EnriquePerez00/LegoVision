# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_yolo_training_dataset.py
Genera dataset YOLO para el setup `2camaras_random_pieza_unica` con:

  - Todas las refs unicas de TODA la BD (todos los sets en REAL_SETS).
  - Color real de la BD por (ref): cada (ref, color_hex) viene de REAL_SETS (todos los sets).
  - Pose estable aleatoria por frame (TARPS canonico).
  - Posicion XY aleatoria dentro del FOV de AMBAS camaras (cenital + lateral).
  - Rotacion Z aleatoria.
  - Sin hardcoding a ningun set_id especifico.

Uso:
    /opt/homebrew/bin/blender -b -P \\
        2camaras_random_pieza_unica/scripts/generate_yolo_training_dataset.py -- \\
            --camera cenital \\
            --output_dir 2camaras_random_pieza_unica/data/yolo_cenital \\
            --num_frames 2000 \\
            --seed 42

Salida:
    <output_dir>/images/train_NNNNN.png
    <output_dir>/labels/train_NNNNN.txt   (formato YOLO: class cx cy w h)
    <output_dir>/dataset_metadata.json    (mapa frame -> (ref, color, pose))
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

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

from config_loader import cfg
from generate_synthetic_set import (
    apply_bevel_modifier,
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
    generate_detailed_fallback_mesh,
    get_ldraw_part_path,
    setup_physics_world,
)
from generate_synthetic_dataset import get_single_mesh_object
from generate_test_set import (
    _get_world_bbox,
    _normalize_piece,
    cleanup_piece_objects,
    create_belt_collider,
    create_floor,
    setup_camera,
    setup_lab_lightbox,
)
from _pose_utils import (
    apply_stable_pose,
)
from database.set_catalog import REAL_SETS

from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("yolo")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
# Sin hardcoding: se usa toda la BD dinamicamente desde REAL_SETS
RENDER_RES = cfg.render.resolution.width
HALF_FOV_BU = 10.0
MARGIN_BU = 0.5
MARGIN_NORM = MARGIN_BU / (2.0 * HALF_FOV_BU)
MAX_PLACEMENT_ATTEMPTS = 200
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
EMPTY_FRAME_RATIO_DEFAULT = float(cfg.yolo.dataset.empty_frame_ratio)
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")


# ─────────────────────────────────────────────────────────────────
# Geometry / scene helpers (mismo modelo que generate_300_random_set.py)
# ─────────────────────────────────────────────────────────────────
def _project_bbox_norm(obj, scene, camera):
    bbox_world = _get_world_bbox(obj)
    xs, ys, zs = [], [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(co.y)
        zs.append(co.z)
    return min(xs), min(ys), max(xs), max(ys), min(zs)


def _bbox_within_margin(bbox_norm, margin):
    x1, y1, x2, y2, depth_min = bbox_norm
    if depth_min <= 0:
        return False
    return (x1 >= margin and y1 >= margin
            and x2 <= 1.0 - margin and y2 <= 1.0 - margin)


def sample_valid_position(part_obj, scene, cam_cen, cam_lat,
                          margin_norm=MARGIN_NORM,
                          max_attempts=MAX_PLACEMENT_ATTEMPTS):
    sample_range = HALF_FOV_BU - 0.05
    for _ in range(max_attempts):
        rx = random.uniform(-sample_range, sample_range)
        ry = random.uniform(-sample_range, sample_range)
        part_obj.location = (rx, ry, 0.0)
        bpy.context.view_layer.update()
        bbox_world = _get_world_bbox(part_obj)
        min_z = min(pt.z for pt in bbox_world)
        part_obj.location.z = -min_z + 0.02
        bpy.context.view_layer.update()
        bbox_cen = _project_bbox_norm(part_obj, scene, cam_cen)
        if not _bbox_within_margin(bbox_cen, margin_norm):
            continue
        bbox_lat = _project_bbox_norm(part_obj, scene, cam_lat)
        if not _bbox_within_margin(bbox_lat, margin_norm):
            continue
        return (
            float(part_obj.location.x),
            float(part_obj.location.y),
            float(part_obj.location.z),
        )
    return None


def build_scene():
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    enable_metal_gpu_acceleration()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    cam_cenital = setup_camera("Cam_Cenital", (0.0, 0.0, 30.0))
    cam_lateral = setup_camera("Cam_Lateral", (15.0, 0.0, 2.5))
    return cam_cenital, cam_lateral


def import_part(part_ref):
    part_path = get_ldraw_part_path(part_ref)
    existing = set(bpy.context.scene.objects)
    obj = None
    if part_path:
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objs = [o for o in bpy.context.scene.objects if o not in existing]
            par = next((o for o in new_objs if o.parent is None), None)
            if par:
                obj = get_single_mesh_object(par)
        except Exception:
            obj = None
    if not obj:
        generate_detailed_fallback_mesh(part_ref)
        obj = bpy.context.active_object
    return obj


# ─────────────────────────────────────────────────────────────────
# Plan: combos (ref, color, pose) balanceados
# ─────────────────────────────────────────────────────────────────
def filter_stable_poses(poses):
    if not poses:
        return []
    stable = [p for p in poses
              if float(p.get("tipping_energy_ratio", 0.0)) >= TARPS_MIN_TIPPING]
    if stable:
        return stable
    best = max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))
    return [best]


def get_unique_ref_color_combinations_all():
    """Devuelve todas las (ref, color_code, color_hex) unicas de TODA la BD."""
    seen = set()
    combos = []
    for set_id, set_data in REAL_SETS.items():
        for p in set_data.get("parts", []):
            ref = p.get("ref", "")
            if not ref or "stk" in ref.lower() or ref.lower().startswith("sw") or ref.lower().startswith("fig"):
                continue
            key = (ref, str(p.get("color_code", "0")))
            if key in seen:
                continue
            seen.add(key)
            combos.append({
                "ref": ref,
                "color_code": str(p.get("color_code", "0")),
                "color_hex": p.get("color_hex", "#A0A5A9"),
                "color_name": p.get("color_name", "Unknown"),
            })
    return combos


def build_universe(combos, cache):
    """Universo expandido: cada (combo x pose_estable) es un item.
    Cada item es una "configuracion canonica" balanceada."""
    universe = []
    for combo in combos:
        ref = combo["ref"]
        poses = cache.get(ref, [])
        stable_poses = filter_stable_poses(poses)
        if not stable_poses:
            log.warning(f"[plan] {ref} sin poses estables")
            if poses:
                stable_poses = [poses[0]]
            else:
                continue
        for pose in stable_poses:
            universe.append({
                "ref": ref,
                "color_code": combo["color_code"],
                "color_hex": combo["color_hex"],
                "color_name": combo["color_name"],
                "pose": pose,
            })
    return universe


def build_frame_plan(universe, num_piece_frames):
    """Plan de N frames con piezas: garantiza >= 1 muestra por item del
    universo y rellena round-robin barajado hasta completar N."""
    if not universe:
        return []
    plan = []
    if num_piece_frames <= len(universe):
        idxs = list(range(len(universe)))
        random.shuffle(idxs)
        plan = [universe[i] for i in idxs[:num_piece_frames]]
    else:
        plan = list(universe)
        pool = list(universe)
        random.shuffle(pool)
        while len(plan) < num_piece_frames:
            for item in pool:
                if len(plan) >= num_piece_frames:
                    break
                plan.append(item)
    random.shuffle(plan)
    return plan


# ─────────────────────────────────────────────────────────────────
# YOLO bbox computation
# ─────────────────────────────────────────────────────────────────
def compute_bbox_yolo(obj, camera, scene):
    """Devuelve (cx, cy, w, h) normalizados [0,1] formato YOLO o None si bbox invalido."""
    bbox_world = _get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        # Y de Blender (0=abajo, 1=arriba) -> YOLO (0=arriba, 1=abajo)
        xs.append(co.x)
        ys.append(1.0 - co.y)
    x1 = max(0.0, min(xs))
    x2 = min(1.0, max(xs))
    y1 = max(0.0, min(ys))
    y2 = min(1.0, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    if w < 1e-3 or h < 1e-3:
        return None
    return (cx, cy, w, h)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, choices=["cenital", "lateral"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--empty_ratio", type=float, default=EMPTY_FRAME_RATIO_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    pa = parser.parse_known_args(args)[0]

    random.seed(pa.seed)

    images_dir = os.path.join(pa.output_dir, "images")
    labels_dir = os.path.join(pa.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    log_execution_header(log, "generate_yolo_training_dataset.py",
                         camera=pa.camera, output_dir=pa.output_dir,
                         num_frames=pa.num_frames, empty_ratio=pa.empty_ratio)

    if not os.path.isfile(CACHE_PATH):
        log.error(f"No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    combos = get_unique_ref_color_combinations_all()
    log.info(f"[plan] {len(combos)} combinaciones (ref, color) de toda la BD")
    universe = build_universe(combos, cache)
    log.info(f"[plan] Universo (combo x pose): {len(universe)} items")

    num_empty = max(1, int(pa.num_frames * pa.empty_ratio))
    num_piece = pa.num_frames - num_empty
    log.info(f"[plan] {num_piece} frames con pieza, {num_empty} frames vacios")
    plan_pieces = build_frame_plan(universe, num_piece)
    plan_types = ["piece"] * len(plan_pieces) + ["empty"] * num_empty
    random.shuffle(plan_types)

    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene
    active_cam = cam_cenital if pa.camera == "cenital" else cam_lateral

    saved = 0
    skipped = 0
    plan_idx = 0
    metadata = []
    current_ref = None
    part_obj = None

    for fi, ftype in enumerate(plan_types):
        if ftype == "empty":
            cleanup_piece_objects()
            current_ref = None
            scene.camera = active_cam
            img_fn = f"train_{fi:05d}.png"
            lbl_fn = f"train_{fi:05d}.txt"
            scene.render.filepath = os.path.join(images_dir, img_fn)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"[{fi}] empty render fallido: {e}")
                skipped += 1
                continue
            open(os.path.join(labels_dir, lbl_fn), "w").close()
            metadata.append({"frame": fi, "type": "empty", "file": img_fn})
            saved += 1
            if (saved % 50) == 0:
                log.info(f"  [{saved}/{pa.num_frames}] frames guardados")
            continue

        item = plan_pieces[plan_idx]
        plan_idx += 1
        ref = item["ref"]
        color_hex = item["color_hex"]
        color_code = item["color_code"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        # Re-importar mesh solo si cambia ref
        if ref != current_ref:
            cleanup_piece_objects()
            part_obj = import_part(ref)
            if not part_obj:
                log.warning(f"[{fi}] {ref}: no se pudo importar")
                skipped += 1
                current_ref = None
                continue
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            current_ref = ref

        # Pose + Z aleatorio + snap
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Posicion aleatoria valida en CENITAL + LATERAL
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_norm=MARGIN_NORM, max_attempts=MAX_PLACEMENT_ATTEMPTS,
        )
        if valid_pos is None:
            log.warning(f"[{fi}] {ref}: no hay posicion valida")
            skipped += 1
            continue

        # Color real del set
        cf = color_hex if color_hex.startswith("#") else f"#{color_hex}"
        mat = create_abs_plastic_material(cf)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        # Render camara activa
        scene.camera = active_cam
        bpy.context.view_layer.update()
        bb = compute_bbox_yolo(part_obj, active_cam, scene)
        if bb is None:
            log.warning(f"[{fi}] {ref}: bbox invalido")
            skipped += 1
            continue

        img_fn = f"train_{fi:05d}.png"
        lbl_fn = f"train_{fi:05d}.txt"
        scene.render.filepath = os.path.join(images_dir, img_fn)
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log.warning(f"[{fi}] render fallido: {e}")
            skipped += 1
            continue

        # Label YOLO: class=0 (lego_piece) + cx cy w h
        with open(os.path.join(labels_dir, lbl_fn), "w") as f:
            f.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        metadata.append({
            "frame": fi,
            "type": "piece",
            "file": img_fn,
            "ref": ref,
            "color_code": color_code,
            "color_hex": cf,
            "pose_index": pose_index,
            "face_class": pose.get("face_class"),
            "position_bu": [round(valid_pos[0], 4), round(valid_pos[1], 4),
                            round(valid_pos[2], 4)],
            "bbox_yolo": [round(bb[0], 6), round(bb[1], 6),
                          round(bb[2], 6), round(bb[3], 6)],
        })
        saved += 1
        if (saved % 50) == 0:
            log.info(f"  [{saved}/{pa.num_frames}] frames guardados | "
                     f"ultimo: {ref} c={color_code} pose={pose_index}")

    cleanup_piece_objects()

    meta_path = os.path.join(pa.output_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": "ALL",
            "camera": pa.camera,
            "total_frames_planned": pa.num_frames,
            "total_frames_saved": saved,
            "skipped": skipped,
            "num_combos": len(combos),
            "num_universe_items": len(universe),
            "frames": metadata,
        }, f, indent=2, ensure_ascii=False)

    log_execution_footer(log, "generate_yolo_training_dataset.py",
                         saved=saved, skipped=skipped,
                         num_combos=len(combos), universe=len(universe),
                         metadata=meta_path)


if __name__ == "__main__":
    main()
