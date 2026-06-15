# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_yolo_pose_training_dataset.py
============================================================================
Genera dataset YOLO-Pose para el setup `2camaras_random_pieza_unica`:
  - Escena CANONICA (`scene_canonical.build_scene_canonical`).
  - 38 refs unicas del set 75078-1, color real del set por (ref).
  - Pose estable aleatoria (TARPS) + posicion XY + rot Z aleatoria.
  - 9 keypoints canonicos por pieza (`canonical_keypoints.json`)
    proyectados al frame 2D con su visibilidad.

Formato de label YOLO-Pose por frame (1 linea por instancia):
  class cx cy w h kp1_x kp1_y kp1_v ... kp9_x kp9_y kp9_v
con coordenadas normalizadas [0,1] y v in {0,1,2}:
  v=0: keypoint fuera del frame (o sin info)
  v=1: keypoint dentro del frame pero ocluido
  v=2: keypoint dentro del frame y visible

Uso:
  /opt/homebrew/bin/blender -b -P \\
      2camaras_random_pieza_unica/scripts/generate_yolo_pose_training_dataset.py -- \\
        --camera cenital \\
        --output_dir 2camaras_random_pieza_unica/data/yolo_pose_cenital \\
        --num_frames 2000 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "scripts"))

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from config_loader import cfg
from generate_synthetic_set import (
    apply_bevel_modifier,
    create_abs_plastic_material,
)
from scene_canonical import (
    build_scene_canonical,
    cleanup_piece_objects,
    import_part,
    normalize_piece,
    sample_valid_position,
    HALF_FOV_BU,
    MARGIN_BU_DEFAULT,
)
from _pose_utils import apply_stable_pose
from database.set_catalog import REAL_SETS

try:
    from logger import get_logger, log_execution_header, log_execution_footer
    log = get_logger("yolo")
    HAS_LOGGER = True
except Exception:
    HAS_LOGGER = False
    class _D:
        def info(self, m): print(m)
        def warning(self, m): print(f"[WARN] {m}")
        def error(self, m): print(f"[ERROR] {m}")
    log = _D()


# Sin hardcoding: se usa toda la BD dinamicamente desde REAL_SETS
RENDER_RES_DEFAULT = cfg.render.resolution.width
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
EMPTY_FRAME_RATIO_DEFAULT = float(cfg.yolo.dataset.empty_frame_ratio)
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")
KEYPOINTS_PATH = os.path.join(project_root, "data", "canonical_keypoints.json")
N_KPS = 9


# ─────────────────────────────────────────────────────────────────
# Plan TARPS (igual que generate_300_canonical_set.py)
# ─────────────────────────────────────────────────────────────────
def get_unique_ref_color_combinations_all():
    """Todas las (ref, color_code, color_hex) unicas de TODA la BD."""
    seen, combos = set(), []
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


def filter_stable_poses(poses):
    if not poses:
        return []
    stable = [p for p in poses if float(p.get("tipping_energy_ratio", 0.0))
              >= TARPS_MIN_TIPPING]
    if stable:
        return stable
    return [max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))]


def build_universe(combos, cache):
    universe = []
    for combo in combos:
        ref = combo["ref"]
        poses = cache.get(ref, [])
        sp = filter_stable_poses(poses)
        if not sp:
            log.warning(f"[plan] {ref} sin poses estables")
            if poses:
                sp = [poses[0]]
            else:
                continue
        for pose in sp:
            universe.append({**combo, "pose": pose})
    return universe


def build_frame_plan(universe, num_piece_frames):
    if not universe:
        return []
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
# Bbox + keypoints projection
# ─────────────────────────────────────────────────────────────────
def _world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def compute_bbox_yolo(obj, camera, scene):
    """Devuelve (cx, cy, w, h) normalizados [0,1] formato YOLO o None."""
    bbox_world = _world_bbox(obj)
    xs, ys = [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(1.0 - co.y)  # YOLO Y down
    x1 = max(0.0, min(xs)); x2 = min(1.0, max(xs))
    y1 = max(0.0, min(ys)); y2 = min(1.0, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = x2 - x1; h = y2 - y1
    if w < 1e-3 or h < 1e-3:
        return None
    return (cx, cy, w, h)


def project_keypoints(obj, kps_local_bu, camera, scene):
    """Proyecta los 9 keypoints (en frame local pieza) a 2D imagen.

    Devuelve lista de (x_norm, y_norm, visibility) con
      v=2: dentro del frame y depth>0 (visible),
      v=1: dentro pero depth<0 o muy cerca del borde (etiqueta como ocluido),
      v=0: fuera del frame.

    No hacemos raycast de oclusion (es caro). Para simplificar, el
    train de YOLO-Pose tolera bien etiquetas con v in {0,2} sin v=1;
    aqui usamos v=2 si depth>0 y dentro del frame, v=0 en caso contrario.
    """
    out = []
    for kp_local in kps_local_bu:
        v_local = mathutils.Vector(kp_local)
        v_world = obj.matrix_world @ v_local
        co = world_to_camera_view(scene, camera, v_world)
        # YOLO: y down
        x_norm = float(co.x)
        y_norm = float(1.0 - co.y)
        depth = float(co.z)
        in_frame = (0.0 <= x_norm <= 1.0) and (0.0 <= y_norm <= 1.0) and (depth > 0)
        v = 2 if in_frame else 0
        out.append((x_norm, y_norm, v))
    return out


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["cenital", "lateral"], required=True)
    # Default: renders/yolo_training/{camera} (separación de dominios)
    parser.add_argument("--output_dir", default=None,
                        help="Directorio de salida YOLO (default: renders/yolo_training/{camera})")
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--empty_ratio", type=float, default=EMPTY_FRAME_RATIO_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_res", type=int, default=RENDER_RES_DEFAULT)
    pa = parser.parse_known_args(args)[0]

    # Resolver output_dir si no se especificó
    if pa.output_dir is None:
        pa.output_dir = os.path.join(project_root, "renders", "yolo_training", pa.camera)
    
    random.seed(pa.seed)
    images_dir = os.path.join(pa.output_dir, "images", "train")
    labels_dir = os.path.join(pa.output_dir, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if HAS_LOGGER:
        log_execution_header(
            log, "generate_yolo_pose_training_dataset.py",
            camera=pa.camera, output_dir=pa.output_dir,
            num_frames=pa.num_frames, empty_ratio=pa.empty_ratio,
            seed=pa.seed,
        )

    if not os.path.isfile(CACHE_PATH):
        log.error(f"No existe cache: {CACHE_PATH}"); sys.exit(1)
    if not os.path.isfile(KEYPOINTS_PATH):
        log.error(f"No existe canonical_keypoints.json. Corre "
                  f"compute_canonical_keypoints.py primero."); sys.exit(1)
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    with open(KEYPOINTS_PATH) as f:
        kps_data = json.load(f)
    kps_by_ref = {ref: data["keypoints_bu"] for ref, data in kps_data["pieces"].items()}

    combos = get_unique_ref_color_combinations_all()
    universe = build_universe(combos, cache)
    log.info(f"[plan] combos={len(combos)} | universe={len(universe)}")

    num_empty = max(1, int(pa.num_frames * pa.empty_ratio))
    num_piece = pa.num_frames - num_empty
    log.info(f"[plan] {num_piece} frames con pieza, {num_empty} frames vacios")
    plan_pieces = build_frame_plan(universe, num_piece)
    plan_types = ["piece"] * len(plan_pieces) + ["empty"] * num_empty
    random.shuffle(plan_types)

    cam_cenital, cam_lateral = build_scene_canonical(
        render_res=pa.render_res, film_transparent=False,
    )
    scene = bpy.context.scene
    try:
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 8
        if hasattr(scene.eevee, "render_samples"):
            scene.eevee.render_samples = 8
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
        print("[opt] EEVEE Optimized: TAA samples=8, bloom/SSR/AO=False")
    except Exception as e:
        print(f"[opt] EEVEE optimization failed: {e}")
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
            if (saved % 100) == 0:
                log.info(f"  [{saved}/{pa.num_frames}] frames saved")
            continue

        item = plan_pieces[plan_idx]; plan_idx += 1
        ref = item["ref"]
        color_hex = item["color_hex"]
        color_code = item["color_code"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        if ref not in kps_by_ref:
            log.warning(f"[{fi}] {ref}: sin keypoints canonicos. Skip.")
            skipped += 1
            continue
        kps_local = kps_by_ref[ref]

        if ref != current_ref:
            cleanup_piece_objects()
            part_obj = import_part(ref)
            if part_obj is None:
                log.warning(f"[{fi}] {ref}: no se pudo importar")
                skipped += 1
                current_ref = None
                continue
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            current_ref = ref

        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_bu=MARGIN_BU_DEFAULT,
        )
        if valid_pos is None:
            log.warning(f"[{fi}] {ref}: no hay posicion valida")
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
            log.warning(f"[{fi}] {ref}: bbox invalido")
            skipped += 1
            continue

        kps_2d = project_keypoints(part_obj, kps_local, active_cam, scene)

        img_fn = f"train_{fi:05d}.png"
        lbl_fn = f"train_{fi:05d}.txt"
        scene.render.filepath = os.path.join(images_dir, img_fn)
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log.warning(f"[{fi}] render fallido: {e}")
            skipped += 1
            continue

        parts_lbl = [f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}"]
        for (kx, ky, kv) in kps_2d:
            kx_c = min(max(kx, 0.0), 1.0)
            ky_c = min(max(ky, 0.0), 1.0)
            parts_lbl.append(f"{kx_c:.6f} {ky_c:.6f} {int(kv)}")
        with open(os.path.join(labels_dir, lbl_fn), "w") as f:
            f.write(" ".join(parts_lbl) + "\n")

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
            "keypoints": [[round(x, 6), round(y, 6), int(v)] for (x, y, v) in kps_2d],
        })
        saved += 1
        if (saved % 100) == 0:
            log.info(f"  [{saved}/{pa.num_frames}] frames saved | "
                     f"last: {ref} c={color_code} pose={pose_index}")

    cleanup_piece_objects()

    meta_path = os.path.join(pa.output_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": "ALL",
            "camera": pa.camera,
            "scene": "canonical",
            "render_res": pa.render_res,
            "kpt_shape": [N_KPS, 3],
            "total_frames_planned": pa.num_frames,
            "total_frames_saved": saved,
            "skipped": skipped,
            "num_combos": len(combos),
            "num_universe_items": len(universe),
            "frames": metadata,
        }, f, indent=2, ensure_ascii=False)

    if HAS_LOGGER:
        log_execution_footer(
            log, "generate_yolo_pose_training_dataset.py",
            saved=saved, skipped=skipped,
            num_combos=len(combos), universe=len(universe),
            metadata=meta_path,
        )
    else:
        log.info(f"DONE. saved={saved} skipped={skipped} | metadata={meta_path}")


if __name__ == "__main__":
    main()
