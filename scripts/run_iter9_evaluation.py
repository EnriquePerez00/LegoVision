# -*- coding: utf-8 -*-
"""scripts/run_iter9_evaluation.py
Python script to run evaluation of multicam inference on generated test set
and output accuracy metrics.
"""
import os
import sys
import json
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.knn_classifier import LegoKNNClassifier, get_knn_classifier
from inference.api import PART_HEIGHTS_MM

def main():
    test_dir = os.path.join(project_root, "data", "iter9_test")
    metadata_path = os.path.join(test_dir, "test_metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"[Evaluation Error] Metadata file not found: {metadata_path}")
        sys.exit(1)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
        
    clf = get_knn_classifier()
    # Force loading embeddings
    clf.load_projection_head()
    clf.load_reference_embeddings()
    
    if not clf.is_ready():
        print("[Evaluation Error] Classifier is not ready. DB embeddings not loaded.")
        sys.exit(1)
        
    correct_count = 0
    total_count = 0
    results = []
    
    # Track accuracy per part
    part_stats = {}
    
    for sample_idx, render_entry in enumerate(meta_data.get("renders", [])):
        ref_gt = render_entry["ref"]
        color_code = render_entry["color_code"]
        name_gt = render_entry["name"]
        cameras_data = render_entry["cameras"]
        
        cam_results = {}
        for cam_name in ["cenital", "lateral_l", "lateral_r"]:
            cam_meta = cameras_data[cam_name]
            img_filename = cam_meta["file_name"]
            img_path = os.path.join(test_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"[Warning] Image not found: {img_path}")
                continue
                
            img_full = Image.open(img_path).convert("RGB")
            w, h = img_full.size
            x1, y1, x2, y2 = cam_meta["bbox_norm"]
            
            # Crop
            x1_px = max(0, min(int(x1 * w), w - 1))
            y1_px = max(0, min(int(y1 * h), h - 1))
            x2_px = max(x1_px + 1, min(int(x2 * w), w))
            y2_px = max(y1_px + 1, min(int(y2 * h), h))
            
            crop_img = img_full.crop((x1_px, y1_px, x2_px, y2_px))
            
            # KNN classify (filtering by color = True matches real pipeline)
            top3 = clf.classify(crop_img, set_id="75078-1", filter_by_color=True)
            
            # Height restriction modification for lateral cameras
            if cam_name in ["lateral_l", "lateral_r"] and top3:
                cen_meta = cameras_data.get("cenital")
                if cen_meta:
                    cx1, cy1, cx2, cy2 = cen_meta["bbox_norm"]
                    cen_width_norm = cx2 - cx1
                    lat_height_norm = y2 - y1
                    if cen_width_norm > 0:
                        obs_ratio = lat_height_norm / cen_width_norm
                        
                        for cand in top3:
                            cand_ref = cand["part_ref"]
                            cand_height = PART_HEIGHTS_MM.get(cand_ref, 3.2)
                            
                            from inference.knn_classifier import FALLBACK_FOOTPRINT_MM
                            cand_dims = FALLBACK_FOOTPRINT_MM.get(cand_ref, (8.0, 8.0))
                            cand_width = max(cand_dims)
                            
                            theoretical_ratio = cand_height / cand_width
                            ratio_diff = abs(obs_ratio - theoretical_ratio)
                            
                            if ratio_diff > 0.25:
                                cand["score"] = max(0.01, cand["score"] * 0.4)
                            elif ratio_diff < 0.1:
                                cand["score"] = min(0.99, cand["score"] * 1.15)
                                
                top3.sort(key=lambda x: x["score"], reverse=True)
                
            if top3:
                best = top3[0]
                cam_results[cam_name] = {
                    "predicted_ref": best["part_ref"],
                    "score": best["score"]
                }
            else:
                cam_results[cam_name] = {
                    "predicted_ref": "Desconocido",
                    "score": 0.0
                }
                
        # Aggregate consensus decision
        candidate_votes = {}
        weights = {"cenital": 0.4, "lateral_l": 0.3, "lateral_r": 0.3}
        
        for cam_name, r_cam in cam_results.items():
            ref_pred = r_cam["predicted_ref"]
            score = r_cam["score"]
            if ref_pred != "Desconocido":
                candidate_votes[ref_pred] = candidate_votes.get(ref_pred, 0.0) + score * weights[cam_name]
                
        if candidate_votes:
            consensus_ref = max(candidate_votes, key=candidate_votes.get)
            consensus_score = min(0.9999, candidate_votes[consensus_ref])
        else:
            consensus_ref = "Desconocido"
            consensus_score = 0.0
            
        is_correct = (consensus_ref == ref_gt)
        total_count += 1
        if is_correct:
            correct_count += 1
            
        # Stats per part
        if ref_gt not in part_stats:
            part_stats[ref_gt] = {"correct": 0, "total": 0}
        part_stats[ref_gt]["total"] += 1
        if is_correct:
            part_stats[ref_gt]["correct"] += 1
            
        results.append({
            "index": sample_idx,
            "ref_gt": ref_gt,
            "name_gt": name_gt,
            "consensus_ref": consensus_ref,
            "consensus_score": round(consensus_score, 4),
            "is_correct": is_correct,
            "cameras": cam_results
        })
        
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print(f"\n================ EVALUATION RESULTS ================")
    print(f"Total Samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"====================================================")
    
    print("\nAccuracy per LEGO piece:")
    for part, stats in sorted(part_stats.items()):
        part_acc = (stats["correct"] / stats["total"] * 100)
        print(f"Piece {part}: {part_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
    # Save evaluation report to data/iter9_test/eval_report.json
    report_path = os.path.join(test_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump({
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": round(accuracy, 2),
            "part_stats": part_stats,
            "results": results
        }, rf, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed evaluation report to: {report_path}")

if __name__ == "__main__":
    main()
