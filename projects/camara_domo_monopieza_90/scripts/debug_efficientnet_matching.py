# -*- coding: utf-8 -*-
"""
camara_domo/scripts/debug_efficientnet_matching.py
=================================================
Diagnostic script to test LegoEfficientNetClassifier on a sample from data10.
"""

import os
import sys
import json
import math
import numpy as np
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from scripts.efficientnet_classifier import LegoEfficientNetClassifier, rgb_to_lab, hex_to_rgb
from run_evaluation import observe_zenithal_surface_mm2, estimate_color_dual
from core.db.set_catalog import REAL_SETS


def main():
    metadata_path = os.path.join(legovic_root, "camara_domo", "data", "data10", "simulation_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata not found at {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Let's inspect the first frame and its first visible piece
    # Load Frame 20 where piece 3020 is centered
    frame_entry = meta_data["frames"][0]
    piece_entry = None
    for p in frame_entry["visible_pieces"]:
        if p["ref"] == "3020":
            piece_entry = p
            break
            
    if piece_entry is None:
        print("Error: piece 3710 not found in Frame 0")
        return
        
    ref_gt = piece_entry["ref"]
    color_code_gt = piece_entry["color_code"]
    color_name_gt = piece_entry["color_name"]
    bbox_cen = piece_entry["bbox_cenital_norm"]
    gt_silhouette_area = piece_entry.get("zenith_silhouette_area_gt")
    
    print("=== GROUND TRUTH ===")
    print(f"Ref: {ref_gt}")
    print(f"Color: {color_name_gt} (code: {color_code_gt})")
    print(f"GT Silhouette Area: {gt_silhouette_area}")
    print(f"Bbox Cenital: {bbox_cen}")

    # Load image
    img_cen_path = os.path.join(legovic_root, "camara_domo", "data", "data10", frame_entry["file_name"])
    if not os.path.exists(img_cen_path):
        print(f"Error: image not found at {img_cen_path}")
        return
        
    img_cen = Image.open(img_cen_path)
    w_c, h_c = img_cen.size

    # Segment (we simulate SAM output by cropping and mask)
    # Let's run MobileSAM or load a mock mask (we can just segment using simple color/threshold or load SAM)
    from ultralytics import SAM
    print("Cargando SAM...")
    sam_model = SAM("mobile_sam.pt")
    
    px1_c, py1_c = int(bbox_cen[0] * w_c), int(bbox_cen[1] * h_c)
    px2_c, py2_c = int(bbox_cen[2] * w_c), int(bbox_cen[3] * h_c)
    crop_cen = img_cen.crop((px1_c, py1_c, px2_c, py2_c))
    
    sam_res_cen = sam_model(np.array(img_cen.convert("RGB")), bboxes=[[px1_c, py1_c, px2_c, py2_c]], verbose=False)
    mask_cen = sam_res_cen[0].masks.data[0].cpu().numpy().astype(np.uint8)

    # Measure apparent area
    zen_obs = observe_zenithal_surface_mm2(mask_cen, bbox_cen, measured_lateral_height_mm=9.6, img_res_px_val=w_c)
    obs_apparent_area_mm2 = zen_obs["apparent_area_mm2"]
    print(f"Measured Apparent Area Cenital: {obs_apparent_area_mm2:.2f} mm2")

    # Initialize Classifier
    clf = LegoEfficientNetClassifier()
    
    # 1. CIELAB Color Median extraction
    color_lab_cen = clf.extract_median_lab(crop_cen, mask_cen)
    print(f"Estimated CIELAB color: {color_lab_cen}")

    # Query DB colors to find top 3
    from core.db import supabase_client
    db_colors = []
    try:
        with supabase_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT color_code, color_hex, color_name FROM lego_set_parts")
                db_colors = [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"DB color query failed: {e}")

    if db_colors:
        dists = []
        for c in db_colors:
            rgb = hex_to_rgb(c["color_hex"])
            lab = rgb_to_lab(rgb)
            dist = np.linalg.norm(color_lab_cen - lab)
            dists.append((dist, c["color_code"], c["color_name"], c["color_hex"]))
        dists.sort()
        print("\nTop 5 closest database colors:")
        for dist, code, name, hex_val in dists[:5]:
            print(f" - Code: {code:4s} | Name: {name:20s} | Hex: {hex_val} | Dist: {dist:.2f}")

    # 2. Deterministic Gating
    print("\nRunning get_deterministic_candidates...")
    candidates = clf.get_deterministic_candidates(obs_apparent_area_mm2, color_lab_cen)
    print(f"Candidates returned: {candidates}")
    print(f"Is GT Ref '{ref_gt}' in candidates? {'YES' if ref_gt in candidates else 'NO'}")

    # Let's inspect why GT ref is/isn't in candidates
    min_area = obs_apparent_area_mm2 * 0.8
    max_area = obs_apparent_area_mm2 * 1.2
    print(f"Area range: [{min_area:.2f}, {max_area:.2f}] mm2")

    # Fetch GT poses in DB
    try:
        with supabase_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pose_index, zenith_silhouette_area, zenith_observable_area, face_class, is_stable
                    FROM stable_poses
                    WHERE part_ref = %s
                """, (ref_gt,))
                rows = cur.fetchall()
                print(f"\nStable Poses in DB for GT {ref_gt}:")
                for r in rows:
                    nom = r["zenith_silhouette_area"] or r["zenith_observable_area"]
                    in_range = "IN RANGE" if (nom and min_area <= nom <= max_area) else "OUT OF RANGE"
                    print(f" - Pose: {r['pose_index']} | Face: {r['face_class']} | Nominal Area: {nom} | Stable: {r['is_stable']} | {in_range}")
    except Exception as e:
        print(f"Failed to query stable poses: {e}")

    # Run full classification
    print("\nRunning full classification...")
    preds = clf.classify(crop_cen, mask_cen, area_cenital=obs_apparent_area_mm2)
    print("Top predictions:")
    for p in preds:
        print(f" - Ref: {p['part_ref']} | Pose: {p['pose_index']} | Score: {p['score']:.4f} | raw_cen: {p['raw_sim_cen']:.4f}")


if __name__ == "__main__":
    main()
