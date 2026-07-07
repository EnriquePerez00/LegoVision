# -*- coding: utf-8 -*-
"""evaluate_color_only_1000.py
Evalúa la precisión de inferencia de color aislando el problema (sin matching 3D ni YOLO).
Implementa la Inferencia en Cascada (Fast Path vs Deep Path con Softmax Voting) en 19D.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from ultralytics import SAM

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_ROOT))
for _p in [_HERE, _ROOT, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from color_classifier_v2 import ColorClassifierV2

def _rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    m = arr > 0.04045
    arr[m] = ((arr[m] + 0.055) / 1.055) ** 2.4
    arr[~m] /= 12.92
    x = arr[:, 0]*0.4124 + arr[:, 1]*0.3576 + arr[:, 2]*0.1805
    y = arr[:, 0]*0.2126 + arr[:, 1]*0.7152 + arr[:, 2]*0.0722
    z = arr[:, 0]*0.0193 + arr[:, 1]*0.1192 + arr[:, 2]*0.9505
    x /= 0.95047; z /= 1.08883
    def f(t):
        r = np.zeros_like(t)
        m2 = t > 0.008856
        r[m2] = t[m2] ** (1/3)
        r[~m2] = 7.787*t[~m2] + 16/116
        return r
    fx, fy, fz = f(x), f(y), f(z)
    return np.column_stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)])

def extract_features_fast_path(img_path, bbox_norm, min_pixels=10):
    """Extrae features muy rápido usando un crop central de la bbox (sin SAM)."""
    if not os.path.exists(img_path): return None
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    x1n,y1n,x2n,y2n = bbox_norm
    x1=max(0,int(x1n*W)); y1=max(0,int(y1n*H))
    x2=min(W,int(x2n*W)); y2=min(H,int(y2n*H))
    if x2-x1<4 or y2-y1<4: return None
    
    # Recorte central estricto (margen 25% para evitar bordes)
    margin_x = max(1, int((x2-x1)*0.25))
    margin_y = max(1, int((y2-y1)*0.25))
    crop_rgb = img_rgb[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x].reshape(-1,3)
    
    if len(crop_rgb) < min_pixels: return None
    
    pixels_hsv = cv2.cvtColor(crop_rgb.reshape(-1,1,3), cv2.COLOR_RGB2HSV).reshape(-1,3).astype(np.float32)
    pixels_lab = _rgb_to_lab_batch(crop_rgb)
    
    mean_lab = np.median(pixels_lab, axis=0)
    std_lab = pixels_lab.std(axis=0)
    mean_hsv = np.median(pixels_hsv, axis=0)
    std_hsv = pixels_hsv.std(axis=0)
    
    # Vector de 19D (rellenando textura/brillo/fondo con ceros, será tolerado por el MLP si L es alto)
    return np.array([
        mean_lab[0], std_lab[0], mean_lab[1], std_lab[1], mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0], mean_hsv[1], std_hsv[1], mean_hsv[2], std_hsv[2],
        mean_lab[0], 0.0, 0.0,
        0.0, 0.0, 0.0, mean_lab[0] # L_p95=mean, diffs=0, L_p5=mean
    ], dtype=np.float32)

def extract_features_sam_clean(img_path, bbox_norm, sam_model, min_pixels=10):
    """Deep Path: SAM + Texturas + Fondo 19D."""
    if not os.path.exists(img_path): return None
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    x1n,y1n,x2n,y2n = bbox_norm
    x1=max(0,int(x1n*W)); y1=max(0,int(y1n*H))
    x2=min(W,int(x2n*W)); y2=min(H,int(y2n*H))
    if x2-x1<4 or y2-y1<4: return None
    try:
        results = sam_model(img_rgb, bboxes=[[x1,y1,x2,y2]], verbose=False)
        if not results or results[0].masks is None or len(results[0].masks.data)==0:
            raise ValueError("no mask")
        mask = results[0].masks.data[0].cpu().numpy().astype(bool)
        if mask.shape!=(H,W):
            mask=cv2.resize(mask.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
        pixels_rgb = img_rgb[mask]
    except Exception:
        margin_x=max(2,int((x2-x1)*0.1)); margin_y=max(2,int((y2-y1)*0.1))
        pixels_rgb = img_rgb[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x].reshape(-1,3)
        mask = np.zeros((H,W), dtype=bool)
        mask[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x] = True

    if len(pixels_rgb)<min_pixels: return None
    px_r = pixels_rgb.reshape(-1,1,3)
    pixels_hsv = cv2.cvtColor(px_r,cv2.COLOR_RGB2HSV).reshape(-1,3).astype(np.float32)
    v_ok = pixels_hsv[:,2]>=15
    if v_ok.sum()>=min_pixels: pixels_rgb,pixels_hsv=pixels_rgb[v_ok],pixels_hsv[v_ok]
    sp_ok=(pixels_hsv[:,1]>=20)|(pixels_hsv[:,2]<235)
    if sp_ok.sum()>=min_pixels: pixels_rgb,pixels_hsv=pixels_rgb[sp_ok],pixels_hsv[sp_ok]
    if len(pixels_rgb)<min_pixels: return None
    pixels_lab=_rgb_to_lab_batch(pixels_rgb)
    q25,q75=np.percentile(pixels_lab[:,0],[25,75]); iqr=q75-q25
    ok=(pixels_lab[:,0]>=q25-1.2*iqr)&(pixels_lab[:,0]<=q75+1.2*iqr)
    if ok.sum()>=min_pixels: pixels_lab,pixels_hsv=pixels_lab[ok],pixels_hsv[ok]
    if len(pixels_lab)<min_pixels: return None
    mean_lab=np.median(pixels_lab,axis=0); std_lab=pixels_lab.std(axis=0)
    mean_hsv=np.median(pixels_hsv,axis=0); std_hsv=pixels_hsv.std(axis=0)
    
    # 19D Advanced Features
    L_p95 = float(np.percentile(pixels_lab[:, 0], 95))
    L_p5 = float(np.percentile(pixels_lab[:, 0], 5))
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_masked = lap[mask]
    tex_contrast = float(np.std(lap_masked)) if len(lap_masked) > 0 else 0.0
    
    mean_l = cv2.blur(gray.astype(np.float32), (5, 5))
    mean_sq_l = cv2.blur(gray.astype(np.float32)**2, (5, 5))
    l_var = np.clip(mean_sq_l - mean_l**2, 0, None)
    l_std = np.sqrt(l_var)
    l_std_masked = l_std[mask]
    tex_homogeneity = float(np.mean(1.0 / (1.0 + l_std_masked))) if len(l_std_masked) > 0 else 0.0

    # Local conveyor belt calibration (19D)
    kernel = np.ones((5,5), np.uint8)
    mask_area = mask.sum()
    iters = 5 if mask_area < 200 else 3
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iters).astype(bool)
    belt_ring_mask = dilated_mask & ~mask
    
    belt_pixels = img_rgb[belt_ring_mask]
    if len(belt_pixels) > 10:
        belt_lab = _rgb_to_lab_batch(belt_pixels)
        belt_median_lab = np.median(belt_lab, axis=0)
    else:
        belt_median_lab = np.array([21.0, 9.0, -37.0]) # Fallback blue belt Lab
        
    diff_L = mean_lab[0] - belt_median_lab[0]
    diff_a = mean_lab[1] - belt_median_lab[1]
    diff_b = mean_lab[2] - belt_median_lab[2]

    return np.array([mean_lab[0],std_lab[0],mean_lab[1],std_lab[1],mean_lab[2],std_lab[2],
                     mean_hsv[0],std_hsv[0],mean_hsv[1],std_hsv[1],mean_hsv[2],std_hsv[2],
                     L_p95, tex_contrast, tex_homogeneity,
                     diff_L, diff_a, diff_b, L_p5],
                    dtype=np.float32)

def print_stats(results_v2, count, fast_count, deep_count):
    correct_name_v2 = sum(1 for r in results_v2 if r["correct_name_v2"])
    correct_code_v2 = sum(1 for r in results_v2 if r["correct_code_v2"])
    acc_name_v2 = 100 * correct_name_v2 / count
    acc_code_v2 = 100 * correct_code_v2 / count

    print("\n" + "="*70)
    print(f"ESTADÍSTICAS A LAS {count} PIEZAS")
    print(f"  Fast Path activado en: {fast_count} piezas ({100*fast_count/count:.1f}%)")
    print(f"  Deep Path activado en: {deep_count} piezas ({100*deep_count/count:.1f}%)")
    print("="*70)
    print(f"  Accuracy nombre: {acc_name_v2:.2f}% ({correct_name_v2}/{count})")
    print(f"  Accuracy código: {acc_code_v2:.2f}% ({correct_code_v2}/{count})")
    
    # Error families
    fam_stats = {
        "Trans → Sólido":    0,
        "Sólido → Trans":    0,
        "Metálico/Pearl → Gris": 0,
        "Gris → Gris (wrong)":   0,
        "Cromático → Cromático":  0,
        "Blanco/Claro → Blanco":  0,
        "Negro/Oscuro → Negro":   0,
    }
    for r in results_v2:
        if r["correct_name_v2"]: continue
        if r["pred_name_v2"] == "FAIL": continue
        gt = r["color_name_gt"].lower()
        pred = r["pred_name_v2"].lower()
        gt_trans = "trans-" in gt or "glitter" in gt or "satin" in gt
        pred_trans = "trans-" in pred or "glitter" in pred or "satin" in pred
        gt_metal = any(k in gt for k in ["chrome", "metallic", "pearl", "flat silver", "speckle"])
        pred_gray = any(k in pred for k in ["gray", "grey", "silver"])
        gt_gray = any(k in gt for k in ["gray", "grey", "silver"])

        if gt_trans and not pred_trans:
            fam_stats["Trans → Sólido"] += 1
        elif not gt_trans and pred_trans:
            fam_stats["Sólido → Trans"] += 1
        elif gt_metal and pred_gray:
            fam_stats["Metálico/Pearl → Gris"] += 1
        elif gt_gray and pred_gray:
            fam_stats["Gris → Gris (wrong)"] += 1
        elif "white" in gt or "light" in gt:
            fam_stats["Blanco/Claro → Blanco"] += 1
        elif "black" in gt or "dark" in gt:
            fam_stats["Negro/Oscuro → Negro"] += 1
        else:
            fam_stats["Cromático → Cromático"] += 1

    total_fails = count - correct_name_v2 - sum(1 for r in results_v2 if r["pred_name_v2"] == "FAIL")
    print("\n  FAMILIAS DE ERRORES:")
    for fam, n in sorted(fam_stats.items(), key=lambda x: -x[1]):
        if n > 0:
            pct = 100 * n / total_fails if total_fails > 0 else 0
            print(f"    {fam:35s}  {n:4} fallos ({pct:.1f}% de los errores)")
    print("="*70 + "\n")

def run_evaluation(metadata_path, data_dir):
    with open(metadata_path) as f:
        meta = json.load(f)

    print("Cargando SAM y ColorClassifierV2...")
    _dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sam_model = SAM("mobile_sam.pt").to(_dev)
    clf_v2 = ColorClassifierV2(use_l_bias=False)

    seen_positions = {}
    for frame in meta["frames"]:
        frame_path = os.path.join(data_dir, frame["file_name"])
        offset = frame["belt_offset_mm"]
        for p in frame["visible_pieces"]:
            x_abs = offset - p["x_belt_local_mm"]
            y_abs = p["y_belt_local_mm"]
            key = (round(x_abs, -1), round(y_abs, -1))
            if key not in seen_positions:
                seen_positions[key] = []
            seen_positions[key].append({
                "ref": p["ref"],
                "color_code_gt": str(p["color_code"]),
                "color_name_gt": p.get("color_name", "Unknown"),
                "bbox_norm": p["bbox_cenital_norm"],
                "frame_path": frame_path,
                "x_mm": p["x_belt_local_mm"],
            })

    # Convert to list of observations per physical piece
    samples = []
    for key, obs_list in seen_positions.items():
        # sort by x_mm closest to 0 (center of camera)
        sorted_obs = sorted(obs_list, key=lambda x: abs(x["x_mm"]))
        samples.append(sorted_obs)

    total = len(samples)
    print(f"\nDataset procesado: {total} piezas físicas únicas extraídas.")

    results_v2 = []
    n_feat_fail = 0
    n_fast_path = 0
    n_deep_path = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        best_obs = s[0]
        
        # --- FAST PATH ---
        fast_feat = extract_features_fast_path(best_obs["frame_path"], best_obs["bbox_norm"])
        if fast_feat is not None:
            probs = clf_v2.predict_gated_probs_cielab(fast_feat, None, "cenital", is_simulation=True)
            pred_name, pred_code, conf = clf_v2.last_prediction
            
            # Umbral de Fast Path (0.95)
            if conf >= 0.95:
                n_fast_path += 1
                correct_name = (pred_name.strip().lower() == best_obs["color_name_gt"].strip().lower())
                results_v2.append({
                    "color_code_gt": best_obs["color_code_gt"],
                    "color_name_gt": best_obs["color_name_gt"],
                    "pred_name_v2": pred_name,
                    "pred_code_v2": str(pred_code),
                    "conf_v2": conf,
                    "correct_name_v2": correct_name,
                    "correct_code_v2": str(pred_code) == str(best_obs["color_code_gt"]),
                    "path": "FAST"
                })
                count = i + 1
                if count % 100 == 0: print_stats(results_v2, count, n_fast_path, n_deep_path)
                continue

        # --- DEEP PATH (Multi-Frame Softmax Voting) ---
        n_deep_path += 1
        deep_probs_list = []
        best_deep_feat = None
        
        # Evaluate ONLY the single best centered frame (avoiding peripheral lighting issues)
        for obs in s[:1]:
            feat = extract_features_sam_clean(obs["frame_path"], obs["bbox_norm"], sam_model)
            if feat is not None:
                if best_deep_feat is None: best_deep_feat = feat
                probs = clf_v2.predict_gated_probs_cielab(feat, None, "cenital", is_simulation=True)
                deep_probs_list.append(probs)
                
        if not deep_probs_list:
            n_feat_fail += 1
            results_v2.append({
                "color_code_gt": best_obs["color_code_gt"],
                "color_name_gt": best_obs["color_name_gt"],
                "pred_name_v2": "FAIL", "pred_code_v2": "0",
                "correct_name_v2": False, "correct_code_v2": False, "path": "FAIL"
            })
        else:
            # Voting
            avg_probs = np.mean(deep_probs_list, axis=0)
            idx = int(np.argmax(avg_probs))
            pred_name = clf_v2.classes[idx]
            conf = float(avg_probs[idx])
            
            # Resolve code
            lab_est = np.array([best_deep_feat[0], best_deep_feat[2], best_deep_feat[4]])
            pred_code = clf_v2._stage3(pred_name, lab_est, "cenital")
            
            correct_name = (pred_name.strip().lower() == best_obs["color_name_gt"].strip().lower())
            results_v2.append({
                "color_code_gt": best_obs["color_code_gt"],
                "color_name_gt": best_obs["color_name_gt"],
                "pred_name_v2": pred_name,
                "pred_code_v2": str(pred_code),
                "conf_v2": conf,
                "correct_name_v2": correct_name,
                "correct_code_v2": str(pred_code) == str(best_obs["color_code_gt"]),
                "path": "DEEP"
            })

        count = i + 1
        if count % 100 == 0:
            print_stats(results_v2, count, n_fast_path, n_deep_path)

    elapsed = time.time() - t0
    print(f"\nEvaluación finalizada en {elapsed:.1f}s. Fallos SAM: {n_feat_fail}/{total}")
    if total % 100 != 0:
        print_stats(results_v2, total, n_fast_path, n_deep_path)

    # Save failed samples
    failures = []
    for r, s in zip(results_v2, samples):
        if not r["correct_name_v2"] and r["pred_name_v2"] != "FAIL":
            best_obs = s[0]
            feat = extract_features_sam_clean(best_obs["frame_path"], best_obs["bbox_norm"], sam_model)
            if feat is not None:
                failures.append({
                    "features": feat.tolist(),
                    "color_name_gt": best_obs["color_name_gt"],
                    "pred_name_v2": r["pred_name_v2"],
                    "color_code_gt": best_obs["color_code_gt"]
                })
    
    failures_path = os.path.join(_ROOT, "data", "color_failures.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"Se guardaron {len(failures)} fallos en: {failures_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    run_evaluation(args.metadata, args.data_dir)
