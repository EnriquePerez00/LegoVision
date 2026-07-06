# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_300_canonical_set.py
====================================================================
Genera N (default 300) muestras (cenital + lateral) del set 75078-1
usando la ESCENA CANONICA (`scene_canonical.build_scene_canonical`).

Plan TARPS (opcion C):
  - Cada (ref, color, pose_estable) aparece >= 1 vez si N lo permite.
  - El resto se reparte round-robin barajado hasta `num_samples`.
  - Posicion XY uniforme aleatoria DENTRO del FOV de AMBAS camaras
    (cenital + lateral; rejection sampling con margen MARGIN_BU_DEFAULT).
  - Rotacion Z aleatoria (apply_stable_pose con random_z=True).

Cada render guarda en metadata:
  - bbox_norm proyectado (cenital y lateral).
  - position_bu, z_rotation_rad, pose_index, face_class, contact_normal.
  - color_code, color_name, color_hex.
  - lateral_height_gt, zenith_silhouette_area_gt (para reporte de errores).

Output (default `data/300/`):
  sample300_NNN_<ref>_<color>_pNN_{cenital,lateral}.png
  random_300_metadata.json   (compatible con run_evaluation.py)

Uso:
  /opt/homebrew/bin/blender -b -P \\
      2camaras_random_pieza_unica/scripts/generate_300_canonical_set.py -- \\
      --num_samples 300 \\
      --output_dir 2camaras_random_pieza_unica/data/300 \\
      --metadata_filename random_300_metadata.json \\
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
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "scripts"))

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from core.utils.config_loader import cfg
from generate_synthetic_set import (
    apply_bevel_modifier,
    create_abs_plastic_material,
)
from scene_canonical import (
    build_scene_canonical,
    cleanup_piece_objects,
    get_2d_bbox,
    import_part,
    normalize_piece,
    sample_valid_position,
    HALF_FOV_BU,
    FOV_FULL_MM,
    MARGIN_BU_DEFAULT,
)
from _pose_utils import apply_stable_pose
from core.db.set_catalog import REAL_SETS

try:
    from core.utils.logger import get_logger, log_execution_header, log_execution_footer
    log = get_logger("blender")
    HAS_LOGGER = True
except Exception:
    HAS_LOGGER = False
    class _DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    log = _DummyLog()


SET_ID = "75078-1"
DEFAULT_NUM_SAMPLES = 300
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "data", "300")
DEFAULT_METADATA_FILENAME = "random_300_metadata.json"
DEFAULT_RENDER_RES = cfg.render.resolution.width
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")


# ─────────────────────────────────────────────────────────────────
# Plan TARPS
# ─────────────────────────────────────────────────────────────────
def get_unique_ref_color_combinations(set_id):
    seen = set()
    combos = []
    for p in REAL_SETS[set_id]["parts"]:
        key = (p["ref"], p["color_code"])
        if key in seen:
            continue
        seen.add(key)
        combos.append({
            "ref": p["ref"],
            "color_code": p["color_code"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_name": p.get("color_name", "Unknown"),
            "qty": p.get("qty", 1),
        })
    return combos


def filter_stable_poses(poses):
    if not poses:
        return []
    stable = [p for p in poses
              if float(p.get("tipping_energy_ratio", 0.0)) >= TARPS_MIN_TIPPING]
    if stable:
        return stable
    best = max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))
    return [best]


def build_sample_plan(combos, cache, num_samples):
    """Garantiza 1 muestra por (ref, color, pose_estable); rellena
    round-robin barajado hasta num_samples."""
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
                "qty": combo["qty"],
                "pose": pose,
            })
    n_universe = len(universe)
    log.info(f"[plan] Universo (combo x pose): {n_universe} items "
             f"(combos={len(combos)})")

    if num_samples <= n_universe:
        idxs = list(range(n_universe))
        random.shuffle(idxs)
        plan = [universe[i] for i in idxs[:num_samples]]
    else:
        plan = list(universe)
        pool = list(universe)
        random.shuffle(pool)
        while len(plan) < num_samples:
            for item in pool:
                if len(plan) >= num_samples:
                    break
                plan.append(item)
    random.shuffle(plan)
    return plan


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata_filename", type=str,
                        default=DEFAULT_METADATA_FILENAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_res", type=int, default=DEFAULT_RENDER_RES)
    pa = parser.parse_known_args(args_raw)[0]

    import time as _time
    t0 = _time.perf_counter()

    if HAS_LOGGER:
        log_execution_header(
            log, "generate_300_canonical_set.py",
            num_samples=pa.num_samples, output_dir=pa.output_dir,
            seed=pa.seed, render_res=pa.render_res,
        )

    random.seed(pa.seed)
    output_dir = pa.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(CACHE_PATH):
        log.error(f"No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    combos = get_unique_ref_color_combinations(SET_ID)
    log.info(f"[300set] Set {SET_ID}: {len(combos)} combinaciones (ref,color)")

    plan = build_sample_plan(combos, cache, pa.num_samples)
    log.info(f"[300set] Plan: {len(plan)} muestras")

    # Build escena canonica (mismo entorno que las refs DINOv2 canonicas)
    cam_cenital, cam_lateral = build_scene_canonical(
        render_res=pa.render_res, film_transparent=False,
    )
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    results_meta = []
    skipped = []

    # Cache de mesh ya importado para no re-importar la misma ref muchas
    # veces; el color y la pose se aplican en cada iteracion.
    current_ref = None
    part_obj = None

    for idx, item in enumerate(plan):
        ref = item["ref"]
        color_code = item["color_code"]
        color_hex = item["color_hex"]
        color_name = item["color_name"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        log.info(
            f"[{idx+1:03d}/{len(plan)}] ref={ref} "
            f"color={color_code}({color_name}) pose={pose_index}"
        )

        # Re-importar mesh si cambia la ref
        if ref != current_ref:
            cleanup_piece_objects()
            part_obj = import_part(ref)
            if part_obj is None:
                log.warning(f"   [SKIP] no se pudo importar mesh de {ref}")
                skipped.append({"idx": idx, "ref": ref, "reason": "import_failed"})
                current_ref = None
                continue
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            current_ref = ref

        # Aplicar pose + rotacion Z aleatoria + snap a cinta
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)
        z_rotation_rad = float(part_obj.rotation_euler.z)

        # Posicion XY valida en cenital + lateral (rejection sampling)
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_bu=MARGIN_BU_DEFAULT,
        )
        if valid_pos is None:
            log.warning(f"   [SKIP] no hay posicion valida en {ref}")
            skipped.append({"idx": idx, "ref": ref, "reason": "no_valid_placement"})
            continue
        rx, ry, rz = valid_pos

        # Material ABS con color real del catalogo
        color_hex_full = color_hex if color_hex.startswith("#") else f"#{color_hex}"
        mat = create_abs_plastic_material(color_hex_full)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "index": idx,
            "ref": ref,
            "pose_index": pose_index,
            "original_pose_index": pose.get("original_pose_index"),
            "face_class": pose.get("face_class"),
            "color_code": color_code,
            "color_name": color_name,
            "color_hex": color_hex_full,
            "lateral_height_gt": pose.get("lateral_height"),
            "effective_height_gt": pose.get("effective_height"),
            "zenith_silhouette_area_gt": pose.get("zenith_silhouette_area"),
            "zenith_observable_area_gt": pose.get("zenith_observable_area"),
            "contact_normal_gt": pose.get("contact_normal"),
            "tipping_energy_ratio_gt": pose.get("tipping_energy_ratio"),
            "stability_ratio_gt": pose.get("stability_ratio"),
            "position_bu": [round(rx, 4), round(ry, 4), round(rz, 4)],
            "z_rotation_rad": round(z_rotation_rad, 6),
            "cameras": {},
        }

        # Render por camara con bbox proyectado en metadata
        ok = True
        prefix = f"sample300_{idx:03d}_{ref}_{color_code}_p{pose_index}"
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"{prefix}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                log.warning(f"   [WARN] render fallido {cam_name}: {e}")
                ok = False
                break
            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "image_path": file_path,
                "bbox_norm": [round(float(v), 6) for v in bbox_norm],
            }

        if ok and len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            log.info(f"   [OK] pos=({rx:.2f},{ry:.2f}) BU")
        else:
            skipped.append({"idx": idx, "ref": ref, "reason": "render_failed"})

    cleanup_piece_objects()

    duration = _time.perf_counter() - t0
    meta_path = os.path.join(output_dir, pa.metadata_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": SET_ID,
            "scene": "canonical (scene_canonical.py)",
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{pa.render_res}x{pa.render_res}",
            "fov_cenital_mm": FOV_FULL_MM,
            "margin_mm": int(MARGIN_BU_DEFAULT * 100),
            "samples_count": len(results_meta),
            "samples_planned": len(plan),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "renders": results_meta,
            "duration_s": round(duration, 1),
            "seed": pa.seed,
        }, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info(f"[300set DONE] {len(results_meta)} renders OK | "
             f"plan={len(plan)} | skipped={len(skipped)} | "
             f"{duration:.1f}s")
    log.info(f"Metadata: {meta_path}")
    log.info("=" * 60)

    if HAS_LOGGER:
        log_execution_footer(
            log, "generate_300_canonical_set.py",
            duration_s=duration,
            samples_count=len(results_meta),
            samples_planned=len(plan),
            skipped_count=len(skipped),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
