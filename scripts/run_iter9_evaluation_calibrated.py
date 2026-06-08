# -*- coding: utf-8 -*-
"""
scripts/run_iter9_evaluation_calibrated.py
==========================================
Evaluación definitiva del pipeline de inferencia multicámara (Iteración 9).
Implementa un Algoritmo de Decisión en Cascada (Cascaded Probabilistic Consensus):
  - Fase 1: Gating de Color (Cenital)
  - Fase 2: Gating de Superficie Cenital (5% de tolerancia)
  - Fase 3: Gating de Altura Frontal (5% de tolerancia)
  - Fase 4: Fusión de Similitud Visual DINOv2 (Cenital 0.7 + Frontal 0.3)
"""
import os
import sys
import json
import math
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.knn_classifier import LegoKNNClassifier, get_knn_classifier, FALLBACK_FOOTPRINT_MM
from inference.api import PART_HEIGHTS_MM
from database.set_catalog import REAL_SETS

# ---------------------------------------------------------------------------
# Parámetros de calibración
# ---------------------------------------------------------------------------
PX_PER_MM_REF   = 6.7
PX_PER_MM_TEST  = 6.7
BG_CORNER_DARK_THRESH = 15
BG_DIST_THRESH   = 20

PART_DIMENSIONS_MM = {
    "3005": (8.0, 8.0, 9.6),
    "3001": (32.0, 16.0, 9.6),
    "3039": (16.0, 16.0, 9.6),
    "3665": (16.0, 8.0, 9.6),
    "3010": (32.0, 8.0, 9.6),
    "3002": (24.0, 16.0, 9.6),
    "3020": (32.0, 16.0, 3.2),
    "4070": (8.0, 8.0, 9.6),
    "4032": (16.0, 16.0, 3.2),
    "3700": (16.0, 8.0, 9.6),
}

def get_part_dimensions(ref: str) -> tuple[float, float, float]:
    if ref in PART_DIMENSIONS_MM:
        return PART_DIMENSIONS_MM[ref]
    footprint = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
    height = PART_HEIGHTS_MM.get(ref, 9.6)
    return (max(footprint), min(footprint), height)

def get_nominal_heights(ref: str) -> list[float]:
    dims = get_part_dimensions(ref)
    L, W, H = sorted(dims, reverse=True)
    h_flat = H * 0.986 + ((L + W) / 2.0) * 0.164 + 0.9
    h_side = W * 0.986 + ((L + H) / 2.0) * 0.164
    h_stand = L * 0.986 + ((W + H) / 2.0) * 0.164
    return [h_flat, h_side, h_stand]

def measure_lateral_height_mm(crop_img: Image.Image) -> float:
    """Mide la altura física a partir de la imagen frontal de perfil."""
    try:
        import cv2
        img_np = np.array(crop_img.convert("RGB"))
        corners = [
            img_np[0, 0].astype(np.float32),
            img_np[0, -1].astype(np.float32),
            img_np[-1, 0].astype(np.float32),
            img_np[-1, -1].astype(np.float32),
        ]
        bg_color = np.mean(corners, axis=0)
        
        if np.mean(bg_color) < BG_CORNER_DARK_THRESH:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        else:
            img_sum = np.sum(img_np, axis=2, keepdims=True) + 1e-5
            chroma = img_np / img_sum
            bg_sum = np.sum(bg_color) + 1e-5
            bg_chroma = bg_color / bg_sum
            dist = np.linalg.norm(chroma - bg_chroma, axis=2)
            mask = (dist > 0.04).astype(np.uint8) * 255
            
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, fg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, fg_mask)
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        
        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            height_px = max(ys) - min(ys)
            return height_px / 6.6
    except Exception as e:
        print(f"[Eval Warning] measure_lateral_height_mm failed: {e}")
    return crop_img.height / 6.6

def detect_scale(crop_img: Image.Image) -> float:
    return PX_PER_MM_TEST

def get_oriented_dims_mm(crop_img: Image.Image) -> tuple[float, float]:
    try:
        import cv2
        img_np = np.array(crop_img.convert("RGB"))
        px_per_mm = detect_scale(crop_img)
        corners = [
            img_np[0, 0].astype(np.float32),
            img_np[0, -1].astype(np.float32),
            img_np[-1, 0].astype(np.float32),
            img_np[-1, -1].astype(np.float32),
        ]
        bg_color = np.mean(corners, axis=0)

        if np.mean(bg_color) < BG_CORNER_DARK_THRESH:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        else:
            img_sum = np.sum(img_np, axis=2, keepdims=True) + 1e-5
            chroma = img_np / img_sum
            bg_sum = np.sum(bg_color) + 1e-5
            bg_chroma = bg_color / bg_sum
            dist = np.linalg.norm(chroma - bg_chroma, axis=2)
            mask = (dist > 0.04).astype(np.uint8) * 255
            
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, fg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, fg_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if valid_contours:
            all_pts = np.vstack(valid_contours)
            rect = cv2.minAreaRect(all_pts)
            (_, _), (w_px, h_px), _ = rect
            return max(w_px, h_px) / px_per_mm, min(w_px, h_px) / px_per_mm
    except Exception as e:
        print(f"[Eval Warning] get_oriented_dims_mm failed: {e}")
    px_per_mm = detect_scale(crop_img)
    w, h = crop_img.size
    return max(w, h) / px_per_mm, min(w, h) / px_per_mm

def size_score(max_query: float, min_query: float, ref: str,
               clf: LegoKNNClassifier, cam_name: str = "cenital") -> float:
    if cam_name == "frontal":
        cand_height = PART_HEIGHTS_MM.get(ref, 3.2)
        ref_dim = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
        max_ref, min_ref = max(ref_dim), min(ref_dim)
        
        diff_height = abs(min_query - cand_height)
        dist_width = 0.0
        if max_query < min_ref:
            dist_width = min_ref - max_query
        elif max_query > max_ref:
            dist_width = max_query - max_ref
            
        score_height = math.exp(-(diff_height**2) / (2 * (1.5**2)))
        score_width = math.exp(-(dist_width**2) / (2 * (4.0**2)))
        return score_height * score_width
    else:
        ref_dim = FALLBACK_FOOTPRINT_MM.get(ref)
        if not ref_dim:
            return 0.5
        max_ref, min_ref = max(ref_dim), min(ref_dim)
        diff_max = abs(max_query - max_ref)
        diff_min = abs(min_query - min_ref)
        dist_size = math.sqrt(diff_max**2 + diff_min**2)
        return math.exp(-(dist_size**2) / (2 * (5.0**2)))

def classify_camera(crop_img: Image.Image, clf: LegoKNNClassifier,
                    valid_part_refs: list[str], cam_name: str = "cenital") -> dict[str, float]:
    if not clf._ref_embeddings:
        return {}

    canvas_size = 224
    scale_factor = 208.0 / 640.0
    w_p, h_p = crop_img.size
    if w_p > 0 and h_p > 0:
        new_w = max(1, int(w_p * scale_factor))
        new_h = max(1, int(h_p * scale_factor))
        resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
        clean_crop = canvas
    else:
        clean_crop = crop_img

    max_query, min_query = get_oriented_dims_mm(crop_img)

    cam_id = 1 if cam_name == "cenital" else 2
    filtered = [
        r for r in clf._ref_embeddings
        if (r["face"] % 10 == cam_id) and (r["part_ref"] in valid_part_refs)
    ]
    if not filtered:
        filtered = [
            r for r in clf._ref_embeddings
            if (r["face"] % 10 == cam_id)
        ]

    query_vec = clf._extract_embedding(clean_crop, size_info=(max_query, min_query))
    ref_matrix = np.stack([r["embedding"] for r in filtered])
    visual_scores = ref_matrix @ query_vec

    sz_scores = []
    for r in filtered:
        ref = r["part_ref"]
        sc = size_score(max_query, min_query, ref, clf, cam_name=cam_name)
        sz_scores.append(sc)

    combined = visual_scores * np.array(sz_scores)

    class_scores = {}
    for idx, r in enumerate(filtered):
        ref = r["part_ref"]
        score = float(combined[idx])
        if ref not in class_scores or score > class_scores[ref]:
            class_scores[ref] = score

    return class_scores

def main():
    test_dir = os.path.join(project_root, "data", "iter9_test")
    metadata_path = os.path.join(test_dir, "test_metadata.json")

    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata no encontrada: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    print("[Evaluación Iter9] Cargando clasificador…")
    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()

    if not clf.is_ready():
        print("[ERROR] Clasificador no está listo.")
        sys.exit(1)

    correct_count = 0
    total_count = 0
    part_stats: dict[str, dict] = {}
    results = []

    for sample_idx, entry in enumerate(meta_data.get("renders", [])):
        ref_gt       = entry["ref"]
        cameras_data = entry["cameras"]

        cam_results: dict[str, dict] = {}

        # 1. Cámara Cenital
        cen_meta = cameras_data.get("cenital")
        if not cen_meta:
            print(f"[Warning] Muestra {sample_idx+1} sin metadatos de cámara cenital")
            continue

        cen_filename = cen_meta["file_name"]
        cen_path = os.path.join(test_dir, cen_filename)
        if not os.path.exists(cen_path):
            print(f"[Warning] Imagen cenital no encontrada: {cen_path}")
            continue

        img_cen_full = Image.open(cen_path).convert("RGB")
        iw, ih = img_cen_full.size
        cx1, cy1, cx2, cy2 = cen_meta["bbox_norm"]
        cx1_px = max(0, min(int(cx1 * iw), iw - 1))
        cy1_px = max(0, min(int(cy1 * ih), ih - 1))
        cx2_px = max(cx1_px + 1, min(int(cx2 * iw), iw))
        cy2_px = max(cy1_px + 1, min(int(cy2 * ih), ih))
        crop_cen = img_cen_full.crop((cx1_px, cy1_px, cx2_px, cy2_px))

        # 2. Cámara Frontal
        front_meta = cameras_data.get("frontal")
        if not front_meta:
            print(f"[Warning] Muestra {sample_idx+1} sin metadatos de cámara frontal")
            continue

        front_filename = front_meta["file_name"]
        front_path = os.path.join(test_dir, front_filename)
        if not os.path.exists(front_path):
            print(f"[Warning] Imagen frontal no encontrada: {front_path}")
            continue

        img_front_full = Image.open(front_path).convert("RGB")
        fiw, fih = img_front_full.size
        fx1, fy1, fx2, fy2 = front_meta["bbox_norm"]
        fx1_px = max(0, min(int(fx1 * fiw), fiw - 1))
        fy1_px = max(0, min(int(fy1 * fih), fih - 1))
        fx2_px = max(fx1_px + 1, min(int(fx2 * fiw), fiw))
        fy2_px = max(fy1_px + 1, min(int(fy2 * fih), fih))
        crop_front = img_front_full.crop((fx1_px, fy1_px, fx2_px, fy2_px))

        # --- APLICACIÓN DEL ALGORITMO EN CASCADA ---

        # 1. Filtro de Color Cenital
        color_code_cen = clf._classify_color(crop_cen)
        parts_in_set = REAL_SETS["75078-1"]["parts"]
        valid_by_color = [p["ref"] for p in parts_in_set if p["color_code"] == color_code_cen]
        if not valid_by_color:
            valid_by_color = [p["ref"] for p in parts_in_set]

        # 2. Filtro de Superficie Cenital (5% Tolerancia)
        max_query, min_query = get_oriented_dims_mm(crop_cen)
        obs_surface = max_query * min_query
        
        valid_by_surface = []
        for ref in valid_by_color:
            dims = get_part_dimensions(ref)
            L, W, H = sorted(dims, reverse=True)
            nominals_surface = [L * W, L * H, W * H]
            passed = False
            for nom in nominals_surface:
                if 0.95 * nom <= obs_surface <= 1.05 * nom:
                    passed = True
                    break
            if passed:
                valid_by_surface.append(ref)
                
        if not valid_by_surface:
            valid_by_surface = valid_by_color

        # 3. Filtro de Altura Frontal (5% Tolerancia)
        measured_height = measure_lateral_height_mm(crop_front)
        
        valid_by_height = []
        for ref in valid_by_surface:
            nominals_height = get_nominal_heights(ref)
            passed = False
            for nom in nominals_height:
                if 0.95 * nom <= measured_height <= 1.05 * nom:
                    passed = True
                    break
            if passed:
                valid_by_height.append(ref)
                
        if not valid_by_height:
            valid_by_height = valid_by_surface

        # 4. Fusión de Similitud Visual DINOv2
        scores_cenital = classify_camera(crop_cen, clf, valid_by_height, cam_name="cenital")
        scores_frontal = classify_camera(crop_front, clf, valid_by_height, cam_name="frontal")
        
        final_scores = {}
        for ref in valid_by_height:
            s_cen = scores_cenital.get(ref, 0.0)
            s_front = scores_frontal.get(ref, 0.0)
            final_scores[ref] = 0.7 * s_cen + 0.3 * s_front

        if final_scores:
            consensus_ref = max(final_scores, key=final_scores.get)
            consensus_score = final_scores[consensus_ref]
        else:
            consensus_ref = "Desconocido"
            consensus_score = 0.0

        is_correct = (consensus_ref == ref_gt)
        total_count += 1
        if is_correct:
            correct_count += 1

        if ref_gt not in part_stats:
            part_stats[ref_gt] = {"correct": 0, "total": 0}
        part_stats[ref_gt]["total"] += 1
        if is_correct:
            part_stats[ref_gt]["correct"] += 1

        status = "✓" if is_correct else "✗"
        print(
            f"[{sample_idx+1:02d}/{len(meta_data['renders'])}] GT={ref_gt:6s} "
            f"-> Pred={consensus_ref:6s}  {status}  (score: {consensus_score:.4f} | col:{color_code_cen})"
        )

        results.append({
            "index":           sample_idx,
            "ref_gt":          ref_gt,
            "consensus_ref":   consensus_ref,
            "consensus_score": round(consensus_score, 4),
            "is_correct":      is_correct,
        })

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    print()
    print("=" * 60)
    print("      RESULTADOS DE EVALUACIÓN — ITERACIÓN 9 (CASCADA)")
    print("=" * 60)
    print(f"  Muestras totales : {total_count}")
    print(f"  Correctas        : {correct_count}")
    print(f"  Precisión global : {accuracy:.2f}%")
    print("-" * 60)
    print("  Precisión por pieza:")
    for part_ref, stats in sorted(part_stats.items()):
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"    {part_ref:8s}  {bar}  {pct:5.1f}%  ({stats['correct']}/{stats['total']})")
    print("=" * 60)

    report_path = os.path.join(test_dir, "eval_report_iter9.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump({
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": round(accuracy, 2),
            "render_engine": "BLENDER_EEVEE",
            "resolution": "640x640",
            "part_stats": part_stats,
            "results": results,
        }, rf, indent=2, ensure_ascii=False)
    print(f"  Reporte guardado en: {report_path}")

if __name__ == "__main__":
    main()
