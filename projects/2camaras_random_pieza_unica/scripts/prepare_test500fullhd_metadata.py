#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_test500fullhd_metadata.py
Convierte metadata lista -> {"renders":[...]} para run_evaluation.py
"""
import argparse, json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

POSE_TO_FACE = {
    "flat_top": "Top", "flat_bottom": "Bottom",
    "side_long": "Side", "side_short": "Side",
    "side": "Side", "top": "Top", "bottom": "Bottom",
}

def convert(input_path, output_path, test_dir):
    with open(input_path) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("renders", [raw])

    renders, skipped = [], 0
    for item in items:
        idx = item.get("idx", item.get("index", len(renders)))
        cen_rel = item.get("cenital_render") or item.get("cenital_file") or ""
        lat_rel = item.get("lateral_render") or item.get("lateral_file") or ""

        if not os.path.isfile(os.path.join(test_dir, cen_rel)):
            print(f"  [WARN] cenital no encontrado: {cen_rel} - skip {idx}")
            skipped += 1; continue
        if not os.path.isfile(os.path.join(test_dir, lat_rel)):
            print(f"  [WARN] lateral no encontrado: {lat_rel} - skip {idx}")
            skipped += 1; continue

        ref = item.get("ref") or item.get("piece_id") or "3001"
        color_code = str(item.get("color_code") or item.get("color_id") or "4")
        pose_label = item.get("pose_label", "flat_top")
        face_class = item.get("face_class") or POSE_TO_FACE.get(pose_label, "Top")
        sp_id = str(item.get("stable_pose_id", "sp_0"))
        pose_index = int(sp_id.replace("sp_", "")) if "sp_" in sp_id else int(item.get("pose_index", 0))
        default_bbox = [0.05, 0.05, 0.95, 0.95]

        renders.append({
            "index": idx,
            "ref": ref,
            "piece_id": ref,
            "color_code": color_code,
            "color_name": item.get("color_name", "Red"),
            "color_hex": item.get("color_hex", "#C91A09"),
            "pose_index": pose_index,
            "face_class": face_class,
            "pose_label": pose_label,
            "stable_pose_id": sp_id,
            "rot_z_deg": item.get("rot_z_deg", 0.0),
            "x_offset": item.get("x_offset", 0.0),
            "y_offset": item.get("y_offset", 0.0),
            "lateral_height_gt": item.get("lateral_height_gt"),
            "effective_height_gt": item.get("effective_height_gt"),
            "zenith_silhouette_area_gt": item.get("zenith_silhouette_area_gt"),
            "zenith_observable_area_gt": item.get("zenith_observable_area_gt"),
            "cameras": {
                "cenital": {
                    "file_name": cen_rel,
                    "image_path": os.path.join(test_dir, cen_rel),
                    "bbox_norm": item.get("cenital_bbox_norm", default_bbox),
                },
                "lateral": {
                    "file_name": lat_rel,
                    "image_path": os.path.join(test_dir, lat_rel),
                    "bbox_norm": item.get("lateral_bbox_norm", default_bbox),
                },
            },
        })

    out = {
        "set_id": "75078-1",
        "render_engine": "BLENDER_EEVEE",
        "resolution": "1920x1080",
        "samples_count": len(renders),
        "samples_skipped": skipped,
        "test_dir": test_dir,
        "renders": renders,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default=os.path.join(PROJECT_ROOT, "data", "test_500fullhd", "metadata.json"))
    p.add_argument("--output",   default=os.path.join(PROJECT_ROOT, "data", "test_500fullhd", "metadata_eval.json"))
    p.add_argument("--test_dir", default=os.path.join(PROJECT_ROOT, "data", "test_500fullhd"))
    args = p.parse_args()
    print(f"[prepare_metadata] Input : {args.input}")
    print(f"[prepare_metadata] Output: {args.output}")
    print(f"[prepare_metadata] TestDir: {args.test_dir}")
    meta = convert(args.input, args.output, args.test_dir)
    print(f"[prepare_metadata] OK: {meta['samples_count']} renders (skipped={meta['samples_skipped']})")
    print(f"[prepare_metadata] Guardado: {args.output}")

if __name__ == "__main__":
    main()
