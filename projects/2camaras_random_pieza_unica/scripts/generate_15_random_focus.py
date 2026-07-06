# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_15_random_focus.py
====================================================================
Genera N renders cenital + lateral (default 15) de UNA SOLA pieza
elegida al azar entre las del set 75078-1 con poses estables, usando
la ESCENA CANONICA `scene_canonical` (la misma que produjo
`data/inferencia_test_v3_colors`).

Cada muestra:
    - pose estable aleatoria entre las disponibles
    - posicion XY aleatoria dentro del FOV cenital + lateral
    - rotacion Z (yaw) aleatoria

Salida en `data/random_focus_<ref>/`:
    sample_NN_cenital.png, sample_NN_lateral.png
    random_focus_metadata.json   (compatible con run_evaluation.py)

Uso:
    /opt/homebrew/bin/blender -b -P \
        2camaras_random_pieza_unica/scripts/generate_15_random_focus.py
    /opt/homebrew/bin/blender -b -P \
        2camaras_random_pieza_unica/scripts/generate_15_random_focus.py -- \
        --ref 3023 --num_samples 20 --seed 42
"""
from __future__ import annotations

import json
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

SET_ID = "75078-1"
NUM_SAMPLES = 15
DEFAULT_RENDER_RES = cfg.render.resolution.width  # 640 (config global)
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")


def unique_refs_of_set(set_id):
    seen, refs = set(), []
    for p in REAL_SETS[set_id]["parts"]:
        ref = p["ref"]
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def get_color_for_ref(set_id, ref):
    for p in REAL_SETS[set_id]["parts"]:
        if p["ref"] == ref:
            return (
                p.get("color_hex", "#A0A5A9"),
                p.get("color_name", "Unknown"),
                p.get("color_code", "86"),
            )
    return "#A0A5A9", "Light Bluish Gray", "86"


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default=None,
                        help="Forzar el ref a usar (debug).")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render_res", type=int, default=DEFAULT_RENDER_RES)
    parsed_args = parser.parse_known_args(args)[0]

    if parsed_args.seed is not None:
        random.seed(parsed_args.seed)

    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    set_refs = unique_refs_of_set(SET_ID)
    candidate_refs = [r for r in set_refs if r in cache and cache[r]]
    if not candidate_refs:
        print(f"[ERROR] No hay refs del set {SET_ID} con poses en cache")
        sys.exit(1)

    if parsed_args.ref and parsed_args.ref in candidate_refs:
        chosen_ref = parsed_args.ref
        print(f"[Focus] Pieza forzada: {chosen_ref}")
    else:
        chosen_ref = random.choice(candidate_refs)
        print(f"[Focus] Pieza al azar: {chosen_ref} "
              f"(de {len(candidate_refs)} candidatas)")

    output_dir = parsed_args.output_dir or os.path.join(
        project_root, "data", f"random_focus_{chosen_ref}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Focus] Output dir: {output_dir}")

    color_hex, color_name, color_code = get_color_for_ref(SET_ID, chosen_ref)
    if not color_hex.startswith("#"):
        color_hex = "#" + color_hex
    print(f"[Focus] Color: {color_name} ({color_hex})")

    poses = cache[chosen_ref]
    print(f"[Focus] Poses disponibles: {len(poses)}")

    render_res = int(parsed_args.render_res)
    cam_cenital, cam_lateral = build_scene_canonical(
        render_res=render_res, film_transparent=False,
    )
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    results_meta = []
    skipped = []
    num_samples = int(parsed_args.num_samples)

    for i in range(num_samples):
        print(f"\n[{i+1}/{num_samples}] Generando muestra...")
        pose = random.choice(poses)
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        cleanup_piece_objects()
        part_obj = import_part(chosen_ref)
        if not part_obj:
            print(f"   [SKIP] No se pudo importar mesh de {chosen_ref}")
            skipped.append({"index": i, "reason": "import_failed"})
            continue

        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        # Pose + Z aleatorio + snap
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Posicion XY valida en cenital+lateral
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_bu=MARGIN_BU_DEFAULT,
        )
        if valid_pos is None:
            print("   [SKIP] no hay posicion valida en 200 intentos")
            skipped.append({"index": i, "reason": "no_valid_placement"})
            continue
        rx, ry, rz = valid_pos

        # Material
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "index": i,
            "ref": chosen_ref,
            "pose_index": pose_index,
            "original_pose_index": pose.get("original_pose_index"),
            "face_class": pose.get("face_class"),
            "lateral_height_gt": pose.get("lateral_height"),
            "zenith_silhouette_area_gt": pose.get("zenith_silhouette_area"),
            "zenith_observable_area_gt": pose.get("zenith_observable_area"),
            "contact_normal_gt": pose.get("contact_normal"),
            "color_hex": color_hex,
            "color_name": color_name,
            "color_code": color_code,
            "position_bu": [round(rx, 4), round(ry, 4), round(rz, 4)],
            "cameras": {},
        }

        ok = True
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"sample_{i:02d}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"   [WARN] Render fallido {cam_name}: {e}")
                ok = False
                break
            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "image_path": file_path,
                "bbox_norm": bbox_norm,
            }

        if ok and len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"   [OK] sample {i:02d} pose={pose_index} "
                  f"pos=({rx:.2f},{ry:.2f}) BU")
        else:
            skipped.append({"index": i, "reason": "render_failed"})

    cleanup_piece_objects()

    meta_path = os.path.join(output_dir, "random_focus_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": SET_ID,
            "ref": chosen_ref,
            "color_hex": color_hex,
            "color_name": color_name,
            "color_code": color_code,
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{render_res}x{render_res}",
            "fov_cenital_mm": FOV_FULL_MM,
            "margin_mm": int(MARGIN_BU_DEFAULT * 100),
            "samples_count": len(results_meta),
            "skipped": skipped,
            "renders": results_meta,
            "scene": "canonical (scene_canonical.py)",
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[Focus DONE] {len(results_meta)} muestras de {chosen_ref} "
          f"({color_name})")
    if skipped:
        print(f"[Focus]      {len(skipped)} omitidas:")
        for s in skipped:
            print(f"    - sample {s['index']}: {s['reason']}")
    print(f"[Focus] Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
