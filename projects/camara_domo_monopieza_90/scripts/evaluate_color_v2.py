# -*- coding: utf-8 -*-
"""evaluate_color_v2.py — Evaluación standalone de color V2 vs baseline.

Extrae features de color directamente de las imágenes del dataset
simulation_x5_1D_all usando las bboxes GT (sin YOLO/SAM para velocidad) y
compara ColorClassifierV2 contra ColorClassifierAllAdapted.

Uso:
    cd projects/camara_domo_monopieza_90
    python scripts/evaluate_color_v2.py [--max-samples N]

Fase 4 del plan de mejora — 2026-06-07
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_ROOT))
for _p in [_HERE, _ROOT, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from color_classifier_v2 import ColorClassifierV2
from _belt_mask import filter_out_belt

def extract_features_sam_clean(img_path, bbox_norm, sam_model, min_pixels=10):
    """Extract 12D color features using SAM mask — no belt background contamination.
    Use_l_bias=False when calling ColorClassifierV2 with these features."""
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
    return np.array([mean_lab[0],std_lab[0],mean_lab[1],std_lab[1],mean_lab[2],std_lab[2],
                     mean_hsv[0],std_hsv[0],mean_hsv[1],std_hsv[1],mean_hsv[2],std_hsv[2]],
                    dtype=np.float32)


try:
    from color_classifier_all_adapted import ColorClassifierAllAdapted
    HAS_BASELINE = True
except Exception as _e:
    HAS_BASELINE = False
    print(f"[WARN] ColorClassifierAllAdapted no disponible: {_e}")

# Mapa LEGO code → Rebrickable code (mismo que run_evaluation_1D_all.py)
LEGO_TO_RB = {str(k): str(k) for k in range(1000)}
LEGO_TO_RB.update({"-1": "-1", "-10": "-10", "-11": "-11", "-12": "-12"})


# ── Extracción de features (sin SAM) ─────────────────────────────────────────
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


def extract_features_from_bbox(
    img_path: str,
    bbox_norm: List[float],
    erosion_px: int = 4,
) -> Optional[np.ndarray]:
    """Extrae feature vector 12D de la región de la bbox en la imagen."""
    if not os.path.exists(img_path):
        return None
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    x1n, y1n, x2n, y2n = bbox_norm
    x1 = max(0, int(x1n * W))
    y1 = max(0, int(y1n * H))
    x2 = min(W, int(x2n * W))
    y2 = min(H, int(y2n * H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = img_rgb[y1:y2, x1:x2]

    # Erosionar bordes
    ksize = min(erosion_px * 2 + 1, min(crop.shape[:2]) - 1)
    if ksize >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_full = np.ones(crop.shape[:2], dtype=np.uint8) * 255
        eroded_mask = cv2.erode(mask_full, kernel, iterations=1)
        valid = eroded_mask > 0
        if not np.any(valid):
            valid = np.ones(crop.shape[:2], dtype=bool)
    else:
        valid = np.ones(crop.shape[:2], dtype=bool)

    pixels_rgb = crop[valid]
    if len(pixels_rgb) < 5:
        pixels_rgb = crop.reshape(-1, 3)

    # Filtro fondo negro (simulación)
    v_filter = pixels_rgb.max(axis=1) >= 15
    if v_filter.sum() >= 5:
        pixels_rgb = pixels_rgb[v_filter]

    # Convertir a HSV y filtrar chromakey de cinta
    px_reshaped = pixels_rgb.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(px_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    pixels_rgb, pixels_hsv = filter_out_belt(pixels_rgb, pixels_hsv)

    if len(pixels_rgb) < 5:
        return None

    # Filtrar especularidades (S<25 Y V>230)
    spec_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    if spec_mask.sum() >= 5:
        pixels_rgb = pixels_rgb[spec_mask]
        pixels_hsv = pixels_hsv[spec_mask]

    # Convertir a Lab
    pixels_lab = _rgb_to_lab_batch(pixels_rgb)

    # Filtro IQR [P25, P75] en canal L
    q25, q75 = np.percentile(pixels_lab[:, 0], [25, 75])
    iqr_mask = (pixels_lab[:, 0] >= q25 - 1.2*(q75-q25)) & (pixels_lab[:, 0] <= q75 + 1.2*(q75-q25))
    if iqr_mask.sum() >= 5:
        pixels_lab = pixels_lab[iqr_mask]
        pixels_hsv = pixels_hsv[iqr_mask]

    # Estadísticas (mediana + std)
    mean_lab = np.median(pixels_lab, axis=0)
    std_lab = pixels_lab.std(axis=0)
    mean_hsv = np.median(pixels_hsv, axis=0)
    std_hsv = pixels_hsv.std(axis=0)

    return np.array([
        mean_lab[0], std_lab[0],
        mean_lab[1], std_lab[1],
        mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0],
        mean_hsv[1], std_hsv[1],
        mean_hsv[2], std_hsv[2],
    ], dtype=np.float32)


# ── Evaluación principal ──────────────────────────────────────────────────────
def run_evaluation(max_samples: int = 0, use_sam: bool = False):
    METADATA_PATH = os.path.join(_ROOT, "data", "simulation_x5_1D_all", "simulation_metadata.json")
    DATA_DIR = os.path.join(_ROOT, "data", "simulation_x5_1D_all")

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    # Cargar clasificadores
    print("Cargando clasificadores...")
    sam_model = None
    if use_sam:
        try:
            from ultralytics import SAM
            import torch
            _dev = "mps" if torch.backends.mps.is_available() else "cpu"
            sam_model = SAM("mobile_sam.pt").to(_dev)
            print(f"  SAM cargado (MobileSAM, device={_dev})")
        except Exception as e:
            print(f"  [WARN] SAM no disponible: {e} — usando bbox mode")
            use_sam = False
    clf_v2 = ColorClassifierV2(use_l_bias=(not use_sam))
    clf_v1 = ColorClassifierAllAdapted() if HAS_BASELINE else None
    if clf_v1:
        print(f"  Baseline: ColorClassifierAllAdapted")
    print(f"  Nuevo:    ColorClassifierV2 (MLP={clf_v2._mat is not None})")

    # Construir lista de muestras únicas (deduplicar por posición X-Y)
    seen_positions = {}
    samples = []
    for frame in meta["frames"]:
        frame_path = os.path.join(DATA_DIR, frame["file_name"])
        offset = frame["belt_offset_mm"]
        for p in frame["visible_pieces"]:
            x_abs = offset - p["x_belt_local_mm"]
            y_abs = p["y_belt_local_mm"]
            key = (round(x_abs, 1), round(y_abs, 1))
            if key not in seen_positions:
                seen_positions[key] = {
                    "ref": p["ref"],
                    "color_code_gt": str(p["color_code"]),
                    "color_name_gt": p.get("color_name", "Unknown"),
                    "bbox_norm": p["bbox_cenital_norm"],
                    "frame_path": frame_path,
                    "x_mm": p["x_belt_local_mm"],
                }
                samples.append(seen_positions[key])

    if max_samples > 0:
        samples = samples[:max_samples]

    total = len(samples)
    print(f"\nDataset: {total} muestras únicas de {len(meta['frames'])} frames")

    # Resultados
    results_v2 = []
    results_v1 = []
    t0 = time.time()
    n_feat_fail = 0

    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Procesando {i+1}/{total} ({elapsed:.1f}s)...")

        if use_sam and sam_model is not None:
            feat = extract_features_sam_clean(s["frame_path"], s["bbox_norm"], sam_model)
        else:
            feat = extract_features_from_bbox(s["frame_path"], s["bbox_norm"])
        if feat is None:
            n_feat_fail += 1
            # Fallback: dar como incorrecto
            results_v2.append({
                "color_code_gt": s["color_code_gt"],
                "color_name_gt": s["color_name_gt"],
                "pred_name_v2": "FAIL",
                "pred_code_v2": "0",
                "correct_name_v2": False,
                "correct_code_v2": False,
            })
            if clf_v1:
                results_v1.append({
                    "color_code_gt": s["color_code_gt"],
                    "color_name_gt": s["color_name_gt"],
                    "pred_name_v1": "FAIL",
                    "pred_code_v1": "0",
                    "correct_name_v1": False,
                    "correct_code_v1": False,
                })
            continue

        # V2
        probs_v2 = clf_v2.predict_gated_probs_cielab(feat, None, "cenital", is_simulation=True)
        pred_name_v2, pred_code_v2, conf_v2 = clf_v2.last_prediction
        correct_name_v2 = pred_name_v2.strip().lower() == s["color_name_gt"].strip().lower()
        correct_code_v2 = str(pred_code_v2) == str(s["color_code_gt"])
        results_v2.append({
            "color_code_gt": s["color_code_gt"],
            "color_name_gt": s["color_name_gt"],
            "pred_name_v2": pred_name_v2,
            "pred_code_v2": str(pred_code_v2),
            "conf_v2": conf_v2,
            "correct_name_v2": correct_name_v2,
            "correct_code_v2": correct_code_v2,
        })

        # Baseline V1
        if clf_v1:
            probs_v1 = clf_v1.predict_gated_probs_cielab(feat, None, "cenital", is_simulation=True)
            idx_v1 = int(np.argmax(probs_v1))
            pred_name_v1 = clf_v1.classes[idx_v1]
            correct_name_v1 = pred_name_v1.strip().lower() == s["color_name_gt"].strip().lower()
            results_v1.append({
                "color_code_gt": s["color_code_gt"],
                "color_name_gt": s["color_name_gt"],
                "pred_name_v1": pred_name_v1,
                "correct_name_v1": correct_name_v1,
            })

    elapsed = time.time() - t0
    print(f"\nExtracción completada en {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/muestra)")
    print(f"Features fallidas: {n_feat_fail}/{total}")

    # ── Métricas globales ──────────────────────────────────────────────────────
    correct_name_v2 = sum(1 for r in results_v2 if r["correct_name_v2"])
    correct_code_v2 = sum(1 for r in results_v2 if r["correct_code_v2"])
    acc_name_v2 = 100 * correct_name_v2 / total
    acc_code_v2 = 100 * correct_code_v2 / total

    print("\n" + "="*70)
    print("RESULTADOS GLOBALES")
    print("="*70)
    print(f"  ColorClassifierV2  — Accuracy nombre: {acc_name_v2:.2f}% ({correct_name_v2}/{total})")
    print(f"  ColorClassifierV2  — Accuracy código: {acc_code_v2:.2f}% ({correct_code_v2}/{total})")
    if clf_v1:
        correct_name_v1 = sum(1 for r in results_v1 if r["correct_name_v1"])
        acc_name_v1 = 100 * correct_name_v1 / total
        print(f"  Baseline (V1)      — Accuracy nombre: {acc_name_v1:.2f}% ({correct_name_v1}/{total})")
        delta = acc_name_v2 - acc_name_v1
        print(f"  MEJORA V2 vs V1:   {delta:+.2f} puntos porcentuales")

    # ── Accuracy por color ─────
    # ── Accuracy por color ─────────────────────────────────────────────────────
    # Agrupar por color_name_gt
    by_color_v2 = defaultdict(lambda: {"total": 0, "correct_name": 0, "correct_code": 0, "preds": []})
    for r in results_v2:
        gt = r["color_name_gt"]
        by_color_v2[gt]["total"] += 1
        if r["correct_name_v2"]:
            by_color_v2[gt]["correct_name"] += 1
        if r["correct_code_v2"]:
            by_color_v2[gt]["correct_code"] += 1
        by_color_v2[gt]["preds"].append(r["pred_name_v2"])

    if clf_v1:
        by_color_v1 = defaultdict(lambda: {"total": 0, "correct_name": 0})
        for r in results_v1:
            gt = r["color_name_gt"]
            by_color_v1[gt]["total"] += 1
            if r["correct_name_v1"]:
                by_color_v1[gt]["correct_name"] += 1

    print("\n" + "="*70)
    print("ACCURACY POR COLOR (min 2 muestras, ordenado por V2 accuracy DESC)")
    print("="*70)
    print(f"{'Color GT':35s} {'N':4} {'V2%':7} {'V1%':7} {'DELTA':7} {'Top pred (si falla)'}")
    print("-"*70)

    color_rows = []
    for cname, d in sorted(by_color_v2.items()):
        if d["total"] < 2:
            continue
        acc_v2 = 100.0 * d["correct_name"] / d["total"]
        acc_v1 = 100.0 * by_color_v1[cname]["correct_name"] / by_color_v1[cname]["total"] if clf_v1 and by_color_v1[cname]["total"] > 0 else -1
        # Top wrong prediction
        wrong_preds = [p for r, p in zip(results_v2, [r["pred_name_v2"] for r in results_v2])
                       if r["color_name_gt"] == cname and not r["correct_name_v2"]]
        # Rebuild from by_color_v2
        wrong = [p for p in by_color_v2[cname]["preds"] if p.lower() != cname.lower()]
        top_wrong = Counter(wrong).most_common(1)
        top_wrong_str = f"{top_wrong[0][0]} ({top_wrong[0][1]}x)" if top_wrong else "—"
        color_rows.append((acc_v2, cname, d["total"], acc_v1, top_wrong_str))

    # Sort by V2 accuracy descending
    for acc_v2, cname, n, acc_v1, top_wrong in sorted(color_rows, key=lambda x: -x[0]):
        delta = f"{acc_v2 - acc_v1:+.0f}" if acc_v1 >= 0 else "  N/A"
        acc_v1_str = f"{acc_v1:.0f}%" if acc_v1 >= 0 else " N/A"
        print(f"  {cname:33s} {n:4} {acc_v2:6.0f}% {acc_v1_str:7} {delta:7}  {top_wrong}")

    # ── Top 15 colores con más fallos ──────────────────────────────────────────
    print("\n" + "="*70)
    print("TOP 15 COLORES CON MÁS FALLOS EN V2")
    print("="*70)
    fail_by_color = [(cname, d["total"] - d["correct_name"], d["total"])
                     for cname, d in by_color_v2.items() if d["total"] >= 2]
    fail_by_color.sort(key=lambda x: -x[1])
    for cname, n_fail, n_total in fail_by_color[:15]:
        pct = 100 * (n_total - n_fail) / n_total
        wrong = [p for p in by_color_v2[cname]["preds"] if p.lower() != cname.lower()]
        top_wrong = Counter(wrong).most_common(2)
        wrongs_str = ", ".join(f"{p}({n}x)" for p, n in top_wrong)
        print(f"  {cname:35s} {n_fail:3}/{n_total:3} ({pct:3.0f}%OK)  → {wrongs_str}")

    # ── Análisis de familias de fallos ─────────────────────────────────────────
    print("\n" + "="*70)
    print("FAMILIAS DE FALLOS V2")
    print("="*70)

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
        if r["correct_name_v2"]:
            continue
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

    total_fails = total - correct_name_v2
    for fam, n in sorted(fam_stats.items(), key=lambda x: -x[1]):
        if n > 0:
            pct = 100 * n / total_fails if total_fails > 0 else 0
            print(f"  {fam:35s}  {n:4} fallos ({pct:.1f}% de errores)")

    # ── Guardar resultados ──────────────────────────────────────────────────────
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "n_feat_fail": n_feat_fail,
        "v2": {
            "acc_name": round(acc_name_v2, 4),
            "acc_code": round(acc_code_v2, 4),
            "correct_name": correct_name_v2,
            "correct_code": correct_code_v2,
        },
        "per_color_v2": {
            cname: {
                "total": d["total"],
                "correct_name": d["correct_name"],
                "acc_pct": round(100.0 * d["correct_name"] / d["total"], 1),
                "top_wrong": Counter([p for p in d["preds"] if p.lower() != cname.lower()]).most_common(3),
            }
            for cname, d in by_color_v2.items()
        },
        "fail_families": fam_stats,
    }
    if clf_v1:
        output["v1"] = {
            "acc_name": round(acc_name_v1, 4),
            "correct_name": correct_name_v1,
        }
        output["delta_name_pct"] = round(acc_name_v2 - acc_name_v1, 4)

    out_path = os.path.join(_ROOT, "data", "eval_color_v2_results.json")
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluación de color V2 vs baseline")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limitar número de muestras (0=todas)")
    parser.add_argument("--use-sam", action="store_true",
                        help="Usar MobileSAM para máscara limpia (sin fondo azul)")
    args = parser.parse_args()
    run_evaluation(max_samples=args.max_samples, use_sam=args.use_sam)


if __name__ == "__main__":
    main()
