# -*- coding: utf-8 -*-
"""generate_yolo_training_dataset_parallel.py

Versión PARALELIZABLE de generate_yolo_training_dataset.py.

CAMBIOS:
  - Acepta --start_frame, --end_frame, --total_frames, --worker_id
  - Plan global construido con --master_seed compartido
  - Opción B1: TAA samples 8
  - Opción B2: Pre-cargar todas las meshes una sola vez
  - Opción B3: Desactivar bloom/SSR/AO en EEVEE
  - Cada worker escribe dataset_metadata_workerNN.json
"""
from __future__ import annotations
import argparse
import json
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
    from bpy_extras.object_utils import world_to_camera_view
except ImportError:
    print("[ERROR] Ejecutar dentro de Blender (-b -P)")
    sys.exit(1)

from config_loader import cfg
from generate_synthetic_set import (
    apply_bevel_modifier, create_abs_plastic_material,
    enable_metal_gpu_acceleration, generate_detailed_fallback_mesh,
    get_ldraw_part_path, setup_physics_world,
)
from generate_synthetic_dataset import get_single_mesh_object
from generate_test_set import (
    _get_world_bbox, _normalize_piece, cleanup_piece_objects,
    create_belt_collider, create_floor, setup_camera, setup_lab_lightbox,
)
from _pose_utils import apply_stable_pose
from database.set_catalog import REAL_SETS

from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("yolo_parallel")

SET_ID = "75078-1"
SELECTED_PARTS = cfg.pieces.selected_parts
RENDER_RES = cfg.render.resolution.width
HALF_FOV_BU = 10.0
MARGIN_NORM = 0.5 / (2.0 * HALF_FOV_BU)
MAX_PLACEMENT_ATTEMPTS = 200
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
EMPTY_FRAME_RATIO_DEFAULT = float(cfg.yolo.dataset.empty_frame_ratio)
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")

TAA_SAMPLES_OPT = 8
DISABLE_BLOOM = True
DISABLE_SSR = True
DISABLE_AO = True


def _project_bbox_norm(obj, scene, camera):
    bbox_world = _get_world_bbox(obj)
    xs, ys, zs = [], [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x); ys.append(co.y); zs.append(co.z)
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
        if not _bbox_within_margin(_project_bbox_norm(part_obj, scene, cam_cen), margin_norm):
            continue
        if not _bbox_within_margin(_project_bbox_norm(part_obj, scene, cam_lat), margin_norm):
            continue
        return (float(part_obj.location.x), float(part_obj.location.y), float(part_obj.location.z))
    return None


def apply_eevee_optimizations(scene):
    """Opción B1+B3"""
    try:
        scene.eevee.taa_render_samples = TAA_SAMPLES_OPT
        if DISABLE_BLOOM and hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if DISABLE_SSR and hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if DISABLE_AO and hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
        log.info(f"[opt] EEVEE: TAA={TAA_SAMPLES_OPT}, bloom/SSR/AO=False")
    except Exception as e:
        log.warning(f"[opt] EEVEE parcial: {e}")


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
    apply_eevee_optimizations(scene)
    cam_cenital = setup_camera("Cam_Cenital", (0.0, 0.0, 15.0))
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


# ─── Opción B2: Pre-cargar meshes ───
def preload_all_meshes(refs):
    """Importa todas las refs unicas al inicio."""
    log.info(f"[preload] Importando {len(refs)} meshes únicas (Opción B2)...")
    mesh_cache = {}
    for i, ref in enumerate(refs):
        obj = import_part(ref)
        if not obj:
            log.warning(f"[preload] {ref}: import fallido")
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        _normalize_piece(obj)
        apply_bevel_modifier(obj)
        obj.name = f"PRELOAD_{ref}"
        obj.hide_render = True
        obj.hide_viewport = True
        obj.location = (1000.0, 1000.0, -1000.0)
        mesh_cache[ref] = obj
        if (i + 1) % 10 == 0:
            log.info(f"[preload] {i+1}/{len(refs)}")
    log.info(f"[preload] ✅ {len(mesh_cache)} meshes preloaded")
    return mesh_cache


def activate_preloaded_mesh(mesh_cache, ref):
    target = mesh_cache.get(ref)
    if not target:
        return None
    for r, obj in mesh_cache.items():
        obj.hide_render = True
        obj.hide_viewport = True
        obj.location = (1000.0, 1000.0, -1000.0)
    target.hide_render = False
    target.hide_viewport = False
    target.location = (0.0, 0.0, 0.0)
    return target


# ─── Plan global ───
def filter_stable_poses(poses):
    if not poses:
        return []
    stable = [p for p in poses
              if float(p.get("tipping_energy_ratio", 0.0)) >= TARPS_MIN_TIPPING]
    if stable:
        return stable
    best = max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))
    return [best]


def get_unique_ref_color_combinations(set_id):
    seen = set()
    combos = []
    for p in REAL_SETS[set_id]["parts"]:
        if p["ref"] not in SELECTED_PARTS:
            continue
        key = (p["ref"], p["color_code"])
        if key in seen:
            continue
        seen.add(key)
        combos.append({
            "ref": p["ref"],
            "color_code": p["color_code"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_name": p.get("color_name", "Unknown"),
        })
    return combos


def build_universe(combos, cache):
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


def build_global_frame_plan(universe, num_piece_frames, num_empty_frames, master_seed):
    """Plan GLOBAL determinístico (todos los workers ven lo mismo)."""
    rng = random.Random(master_seed)
    if not universe or num_piece_frames <= 0:
        plan_pieces = []
    else:
        plan_pieces = []
        if num_piece_frames <= len(universe):
            idxs = list(range(len(universe)))
            rng.shuffle(idxs)
            plan_pieces = [universe[i] for i in idxs[:num_piece_frames]]
        else:
            plan_pieces = list(universe)
            pool = list(universe)
            rng.shuffle(pool)
            while len(plan_pieces) < num_piece_frames:
                for item in pool:
                    if len(plan_pieces) >= num_piece_frames:
                        break
                    plan_pieces.append(item)
        rng.shuffle(plan_pieces)
    plan_types = ["piece"] * len(plan_pieces) + ["empty"] * num_empty_frames
    rng.shuffle(plan_types)
    return plan_pieces, plan_types


def compute_bbox_yolo(obj, camera, scene):
    bbox_world = _get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
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


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, choices=["cenital", "lateral"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--end_frame", type=int, required=True)
    parser.add_argument("--total_frames", type=int, required=True)
    parser.add_argument("--empty_ratio", type=float, default=EMPTY_FRAME_RATIO_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--worker_id", type=int, default=0)
    pa = parser.parse_known_args(args)[0]

    random.seed(pa.seed)

    images_dir = os.path.join(pa.output_dir, "images")
    labels_dir = os.path.join(pa.output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    log_execution_header(log, "generate_yolo_training_dataset_parallel.py",
                         worker=pa.worker_id, camera=pa.camera,
                         start=pa.start_frame, end=pa.end_frame,
                         total=pa.total_frames)

    if not os.path.isfile(CACHE_PATH):
        log.error(f"No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Plan global determinístico
    combos = get_unique_ref_color_combinations(SET_ID)
    universe = build_universe(combos, cache)
    num_empty = max(1, int(pa.total_frames * pa.empty_ratio))
    num_piece = pa.total_frames - num_empty
    plan_pieces, plan_types = build_global_frame_plan(
        universe, num_piece, num_empty, master_seed=pa.master_seed)
    log.info(f"[w{pa.worker_id}] plan: {len(plan_pieces)} piece, {num_empty} empty")

    # Mapeo frame_index -> piece_index
    plan_idx_for_frame = {}
    pi = 0
    for fi, ftype in enumerate(plan_types):
        if ftype == "piece":
            plan_idx_for_frame[fi] = pi
            pi += 1

    start = max(0, pa.start_frame)
    end = min(pa.total_frames, pa.end_frame)
    log.info(f"[w{pa.worker_id}] Frames [{start}, {end}) -> {end - start} frames")

    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene
    active_cam = cam_cenital if pa.camera == "cenital" else cam_lateral

    # Opción B2: Pre-cargar meshes
    refs_in_use = sorted({item["ref"] for item in plan_pieces})
    mesh_cache = preload_all_meshes(refs_in_use)

    saved = 0
    skipped = 0
    metadata = []

    for fi in range(start, end):
        ftype = plan_types[fi]
        img_fn = f"train_{fi:05d}.png"
        lbl_fn = f"train_{fi:05d}.txt"

        if ftype == "empty":
            for o in mesh_cache.values():
                o.hide_render = True
                o.hide_viewport = True
            scene.camera = active_cam
            scene.render.filepath = os.path.join(images_dir, img_fn)
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"[w{pa.worker_id}][{fi}] empty fallido: {e}")
                skipped += 1
                continue
            open(os.path.join(labels_dir, lbl_fn), "w").close()
            metadata.append({"frame": fi, "type": "empty", "file": img_fn})
            saved += 1
            if saved % 50 == 0:
                log.info(f"[w{pa.worker_id}] {saved} guardados (frame={fi})")
            continue

        item = plan_pieces[plan_idx_for_frame[fi]]
        ref = item["ref"]
        color_hex = item["color_hex"]
        color_code = item["color_code"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        part_obj = activate_preloaded_mesh(mesh_cache, ref)
        if not part_obj:
            log.warning(f"[w{pa.worker_id}][{fi}] {ref}: mesh no preloaded")
            skipped += 1
            continue

        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        valid_pos = sample_valid_position(part_obj, scene, cam_cenital, cam_lateral)
        if valid_pos is None:
            log.warning(f"[w{pa.worker_id}][{fi}] {ref}: no posición válida")
            skipped += 1
            continue

        cf = color_hex if color_hex.startswith("#") else f"#{color_hex}"
        mat = create_abs_plastic_material(cf)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        scene.camera = active_cam
        bpy.context.view_layer.update()
        bb = compute_bbox_yolo(part_obj, active_cam, scene)
        if bb is None:
            log.warning(f"[w{pa.worker_id}][{fi}] {ref}: bbox inválido")
            skipped += 1
            continue

        scene.render.filepath = os.path.join(images_dir, img_fn)
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log.warning(f"[w{pa.worker_id}][{fi}] render fallido: {e}")
            skipped += 1
            continue

        with open(os.path.join(labels_dir, lbl_fn), "w") as f:
            f.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        metadata.append({
            "frame": fi, "type": "piece", "file": img_fn,
            "ref": ref, "color_code": color_code, "color_hex": cf,
            "pose_index": pose_index, "face_class": pose.get("face_class"),
            "position_bu": [round(valid_pos[0], 4), round(valid_pos[1], 4),
                            round(valid_pos[2], 4)],
            "bbox_yolo": [round(bb[0], 6), round(bb[1], 6),
                          round(bb[2], 6), round(bb[3], 6)],
        })
        saved += 1
        if saved % 50 == 0:
            log.info(f"[w{pa.worker_id}] {saved} guardados (frame={fi}, "
                     f"ref={ref}, pose={pose_index})")

    # Guardar metadata por worker
    meta_path = os.path.join(pa.output_dir,
                              f"dataset_metadata_worker{pa.worker_id:02d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "worker_id": pa.worker_id,
            "camera": pa.camera,
            "start_frame": pa.start_frame,
            "end_frame": pa.end_frame,
            "total_frames_planned": end - start,
            "total_frames_saved": saved,
            "skipped": skipped,
            "frames": metadata,
        }, f, indent=2, ensure_ascii=False)

    log_execution_footer(log, "generate_yolo_training_dataset_parallel.py",
                         worker=pa.worker_id, saved=saved, skipped=skipped,
                         metadata=meta_path)


if __name__ == "__main__":
    main()
