# -*- coding: utf-8 -*-
"""generate_test50_canonical.py
Genera 50 renders de test con la escena CANONICA (cam cenital 30cm, focal 55mm, 2048x2048).
Piezas, colores y poses aleatorias del stable_poses_cache.

Uso:
    /opt/homebrew/bin/blender -b -P \
        2camaras_random_pieza_unica/scripts/generate_test50_canonical.py -- \
            --num_samples 50 \
            --output_dir 2camaras_random_pieza_unica/data/reports/test_500fullhd \
            --seed 2026
"""
from __future__ import annotations
import json, math, os, random, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import (
    apply_bevel_modifier,
    configure_eevee_for_translucent,
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
    setup_physics_world,
)
from _pose_utils import apply_stable_pose
import scene_canonical

# ─── Config ───────────────────────────────────────────────────────────────────
RENDER_RES = 2048
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")
# COLOR_CATALOG_PATH removed — colors come from REAL_SETS now
OUTPUT_DIR_DEFAULT = os.path.join(project_root, "data", "reports", "test_500fullhd")


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    pa = parser.parse_known_args(args)[0]

    random.seed(pa.seed)
    output_dir = pa.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load stable poses cache
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Build universe using ONLY (ref, color) pairs that exist in REAL_SETS
    # This ensures the test represents production conditions.
    from database.set_catalog import REAL_SETS

    real_pairs = {}  # (ref, color_code) -> {color_hex, color_name}
    for set_id, set_data in REAL_SETS.items():
        for part in set_data.get("parts", []):
            ref = part["ref"]
            cc = str(part.get("color_code", ""))
            ch = part.get("color_hex", "")
            cn = part.get("color_name", "Unknown")
            if ref in cache and ch and ch.startswith("#") and len(ch) == 7:
                real_pairs[(ref, cc)] = {"hex": ch, "name": cn}

    print(f"[test50] {len(real_pairs)} pares (ref,color) reales de REAL_SETS con poses")

    universe = []
    for (ref, color_code), color_info in real_pairs.items():
        poses = cache[ref]
        if not poses:
            continue
        for pose in poses:
            universe.append({
                "ref": ref,
                "pose": pose,
                "color_code": color_code,
                "color_name": color_info["name"],
                "color_hex": color_info["hex"],
            })

    if not universe:
        print("[ERROR] No hay combinaciones validas")
        sys.exit(1)

    # Sample
    samples = random.sample(universe, min(pa.num_samples, len(universe)))
    print(f"[test50] {len(samples)} muestras seleccionadas de {len(universe)} posibles")

    # Build canonical scene at 2048x2048
    cam_cenital, cam_lateral = scene_canonical.build_scene_canonical(
        render_res=RENDER_RES, film_transparent=False
    )
    scene = bpy.context.scene
    # Verify resolution
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    bpy.context.view_layer.update()

    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}
    results_meta = []
    skipped = []
    current_ref = None
    part_obj = None

    for idx, item in enumerate(samples):
        ref = item["ref"]
        color_code = item["color_code"]
        color_hex = item["color_hex"]
        color_name = item["color_name"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        print(f"\n[{idx+1:03d}/{len(samples)}] ref={ref} color={color_code}({color_name}) pose={pose_index}")

        # Import part (re-use if same ref)
        if ref != current_ref:
            scene_canonical.cleanup_piece_objects()
            part_obj = scene_canonical.import_part(ref)
            if not part_obj:
                print(f"   [SKIP] no se pudo importar mesh de {ref}")
                skipped.append({"idx": idx, "ref": ref, "reason": "import_failed"})
                current_ref = None
                continue
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            scene_canonical.normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            current_ref = ref

        # Apply pose with random Z rotation
        part_obj.location = (0.0, 0.0, 0.0)
        part_obj.rotation_euler = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Find valid position
        valid_pos = scene_canonical.sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral
        )
        if valid_pos is None:
            print(f"   [SKIP] no hay posicion valida")
            skipped.append({"idx": idx, "ref": ref, "reason": "no_valid_placement"})
            continue

        # Apply color material
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "index": idx,
            "sample_index": idx,
            "ref": ref,
            "pose_index": pose_index,
            "face_class": pose.get("face_class"),
            "color_code": color_code,
            "color_name": color_name,
            "color_hex": color_hex,
            "lateral_height_gt": pose.get("lateral_height"),
            "effective_height_gt": pose.get("effective_height"),
            "zenith_silhouette_area_gt": pose.get("zenith_silhouette_area"),
            "zenith_observable_area_gt": pose.get("zenith_observable_area"),
            "contact_normal_gt": pose.get("contact_normal"),
            "tipping_energy_ratio_gt": pose.get("tipping_energy_ratio"),
            "position_bu": list(valid_pos),
            "cameras": {},
        }

        ok = True
        prefix = f"sample500_{idx:03d}_{ref}_{color_code}_p{pose_index}"
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            bbox_norm = scene_canonical.get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"{prefix}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"   [WARN] render fallido {cam_name}: {e}")
                ok = False
                break
            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "image_path": file_path,
                "bbox_norm": bbox_norm,
            }

        if ok and len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"   [OK] pos=({valid_pos[0]:.3f},{valid_pos[1]:.3f},{valid_pos[2]:.3f}) BU")
        else:
            skipped.append({"idx": idx, "ref": ref, "reason": "render_failed"})

    scene_canonical.cleanup_piece_objects()

    # Save metadata
    meta_path = os.path.join(output_dir, "random_500_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{RENDER_RES}x{RENDER_RES}",
            "cam_cenital_z_cm": 30,
            "cam_cenital_focal_mm": 55,
            "cam_lateral_pos_cm": [15, 0, 2.5],
            "cam_lateral_focal_mm": 27,
            "samples_count": len(results_meta),
            "samples_planned": len(samples),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[test50 DONE] {len(results_meta)} muestras renderizadas (skipped={len(skipped)})")
    print(f"[test50] Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()