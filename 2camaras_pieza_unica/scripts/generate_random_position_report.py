# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_random_position_report.py
====================================================================
Ejecuta el pipeline de inferencia (YOLO cenital + lateral, SAM,
DINOv2, cascada de gating en 4 fases) sobre el dataset generado por
`generate_set_random_position.py` y produce dos CSVs en
`2camaras_pieza_unica/data/reports/`:

  1. random_position_areas.csv
       pieza, pose, surface_silhouette_mm2, surface_convex_hull_mm2,
       surface_estimated_inference_mm2, err_silh_pct, err_convex_pct

  2. random_position_heights.csv
       pieza, pose, height_stable_pose_mm,
       height_estimated_inference_mm, err_pct

Adicionalmente imprime por pantalla, por cada pieza renderizada,
la pieza inferida y si la inferencia ha acertado o no.

Uso:
    .venv/bin/python \\
        2camaras_pieza_unica/scripts/generate_random_position_report.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT_SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGOVISION_ROOT = os.path.dirname(PROJECT_ROOT_SUB)
# project_root primero para que su config_loader tenga prioridad
sys.path.insert(0, LEGOVISION_ROOT)
sys.path.insert(0, PROJECT_ROOT_SUB)

from config_loader import cfg  # noqa: E402
from database.set_catalog import REAL_SETS  # noqa: E402

# Reusamos las funciones de inferencia/medida del evaluador oficial
# para garantizar consistencia 1-a-1 con el pipeline real.
from run_evaluation import (  # noqa: E402
    classify_camera,
    estimate_color_predominant_sam,
    estimate_surface_area_sam_corrected,
    find_closest_catalog_color,
    get_nominal_heights,
    get_oriented_dims_mm_sam,
    get_part_dimensions,
    get_sam_model,
    measure_lateral_height_mm_sam,
    segment_crop_sam,
    yolo_detect_bbox,
)
from inference.knn_classifier import get_knn_classifier  # noqa: E402

# ─────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────
SET_ID = "75078-1"
SELECTED_PARTS = list(cfg.pieces.selected_parts)

DATA_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "random_position")
META_PATH = os.path.join(DATA_DIR, "random_position_metadata.json")
CACHE_PATH = os.path.join(PROJECT_ROOT_SUB, "data", "stable_poses_cache.json")

REPORTS_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "reports")
AREAS_CSV = os.path.join(REPORTS_DIR, "random_position_areas.csv")
HEIGHTS_CSV = os.path.join(REPORTS_DIR, "random_position_heights.csv")

YOLO_CEN_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_cenital.pt")
YOLO_LAT_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_lateral.pt")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def lookup_pose(cache: dict, ref: str, pose_index: int) -> Optional[dict]:
    """Localiza la entrada de pose en el cache, tolerando sufijos
    a/b/c en la ref."""
    candidate_refs = [ref]
    if ref not in cache:
        for suf in ("b", "a", "c"):
            if (ref + suf) in cache:
                candidate_refs.append(ref + suf)
                break
    for cref in candidate_refs:
        poses = cache.get(cref, [])
        for p in poses:
            if p.get("original_pose_index") == pose_index \
                    or p.get("pose_index") == pose_index:
                return p
        if 0 <= pose_index < len(poses):
            return poses[pose_index]
    return None


def pct_err(estimated, reference) -> Optional[float]:
    if reference is None or reference == 0:
        return None
    return 100.0 * (estimated - reference) / reference


def _resolve_image_path(meta_entry: dict, fallback_dir: str) -> Optional[str]:
    p = meta_entry.get("image_path")
    if p and os.path.isfile(p):
        return p
    fname = meta_entry.get("file_name")
    if fname:
        candidate = os.path.join(fallback_dir, fname)
        if os.path.isfile(candidate):
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────
# Inference cascade (idéntica a run_evaluation.py, pero adaptada
# para devolver el ref predicho + score y registrar todos los
# campos numéricos que necesitan los dos CSVs).
# ─────────────────────────────────────────────────────────────────
def run_inference_for_sample(
    entry: dict,
    yolo_cenital,
    yolo_lateral,
    clf,
) -> dict:
    ref_gt = entry["ref"]
    cameras_data = entry["cameras"]

    cen_meta = cameras_data.get("cenital", {})
    lat_meta = cameras_data.get("lateral", {})
    cen_path = _resolve_image_path(cen_meta, DATA_DIR)
    lat_path = _resolve_image_path(lat_meta, DATA_DIR)
    if not cen_path or not lat_path:
        return {"error": "missing_image"}

    img_cen_full = Image.open(cen_path).convert("RGB")
    img_lat_full = Image.open(lat_path).convert("RGB")
    iw, ih = img_cen_full.size
    liw, lih = img_lat_full.size

    # ── YOLO ──
    cen_bbox, cen_conf = (None, 0.0)
    if yolo_cenital is not None:
        cen_bbox, cen_conf = yolo_detect_bbox(yolo_cenital, cen_path)
    if cen_bbox is None:
        cen_bbox = cen_meta.get("bbox_norm")

    lat_bbox, lat_conf = (None, 0.0)
    if yolo_lateral is not None:
        lat_bbox, lat_conf = yolo_detect_bbox(yolo_lateral, lat_path)
    if lat_bbox is None:
        lat_bbox = lat_meta.get("bbox_norm")

    if cen_bbox is None or lat_bbox is None:
        return {"error": "no_bbox"}

    cx1, cy1, cx2, cy2 = cen_bbox
    lx1, ly1, lx2, ly2 = lat_bbox

    crop_cen = img_cen_full.crop((
        max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
        min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih)),
    ))
    crop_lat = img_lat_full.crop((
        max(0, int(lx1 * liw)), max(0, int(ly1 * lih)),
        min(liw, int(lx2 * liw)), min(lih, int(ly2 * lih)),
    ))

    # Robustez: si la pieza ha caído fuera del FOV lateral (común con
    # posiciones aleatorias en Y), el crop puede tener alguna dim = 0.
    cen_crop_ok = crop_cen.size[0] > 0 and crop_cen.size[1] > 0
    lat_crop_ok = crop_lat.size[0] > 0 and crop_lat.size[1] > 0
    if not cen_crop_ok:
        return {"error": "cenital_crop_empty"}

    # ── SAM ──
    mask_cen = segment_crop_sam(img_cen_full, [cx1, cy1, cx2, cy2])
    mask_lat = segment_crop_sam(img_lat_full, [lx1, ly1, lx2, ly2])

    # ── Color (cenital, sobre máscara SAM) ──
    cen_rgb = estimate_color_predominant_sam(crop_cen, mask_cen)
    cen_color = find_closest_catalog_color(cen_rgb)
    color_code_cen = cen_color["color_code"]

    # ── Phase 1: color gating ──
    parts_in_set = [p for p in REAL_SETS[SET_ID]["parts"]
                    if p["ref"] in SELECTED_PARTS]
    valid_by_color = [p["ref"] for p in parts_in_set
                      if p["color_code"] == color_code_cen]
    if not valid_by_color:
        valid_by_color = [p["ref"] for p in parts_in_set]

    # ── Phase 2: surface gating (cenital, ±20 %) ──
    valid_by_surface = []
    for ref in valid_by_color:
        dims = get_part_dimensions(ref)
        L, W, H = sorted(dims, reverse=True)
        configs = [(L * W, H), (L * H, W), (W * H, L)]
        for nom_area, rest_h in configs:
            filling = 1.0
            if ref in ("6141", "98138", "4032", "3062", "59900"):
                filling = 0.785
            elif ref == "2420":
                filling = 0.75
            elif ref in ("3039", "3298", "3037", "3665", "85984",
                         "54200", "11477", "15068"):
                filling = 0.85
            elif ref in ("2412", "2877"):
                filling = 0.92
            obs_area = estimate_surface_area_sam_corrected(
                mask_cen, [cx1, cy1, cx2, cy2], rest_h
            )
            target = nom_area * filling
            if 0.80 * target <= obs_area <= 1.20 * target:
                valid_by_surface.append(ref)
                break
    if not valid_by_surface:
        valid_by_surface = valid_by_color

    # ── Phase 3: height gating (lateral, ±15 %) ──
    measured_h = measure_lateral_height_mm_sam(mask_lat)
    valid_by_height = []
    for ref in valid_by_surface:
        for nom in get_nominal_heights(ref):
            if 0.85 * nom <= measured_h <= 1.15 * nom:
                valid_by_height.append(ref)
                break
    if not valid_by_height:
        valid_by_height = valid_by_surface

    # ── Phase 4: DINOv2 fusion (cenital 70 % + lateral 30 %) ──
    max_q_c, min_q_c = get_oriented_dims_mm_sam(mask_cen)
    max_q_l, min_q_l = get_oriented_dims_mm_sam(mask_lat)

    s_cen = classify_camera(crop_cen, clf, valid_by_height,
                            max_q_c, min_q_c, cam_name="cenital")
    if lat_crop_ok:
        s_lat = classify_camera(crop_lat, clf, valid_by_height,
                                max_q_l, min_q_l, cam_name="lateral")
        final = {r: 0.7 * s_cen.get(r, 0.0) + 0.3 * s_lat.get(r, 0.0)
                 for r in valid_by_height}
    else:
        # Sin información lateral fiable → usamos sólo cenital.
        s_lat = {}
        final = {r: s_cen.get(r, 0.0) for r in valid_by_height}
    if final:
        pred_ref = max(final, key=final.get)
        pred_score = final[pred_ref]
    else:
        pred_ref = "Desconocido"
        pred_score = 0.0

    # ── Estimación de área cenital (mejor configuración) ──
    # Para el CSV de áreas usamos rest_h = altura medida lateralmente
    # (es la mejor aproximación a "altura del lado vertical" sin GT).
    rest_h_for_perspective = measured_h if measured_h > 0 else 9.6
    estimated_area_mm2 = estimate_surface_area_sam_corrected(
        mask_cen, [cx1, cy1, cx2, cy2], rest_h=rest_h_for_perspective
    )

    return {
        "ref_gt": ref_gt,
        "pose_index": int(entry.get("pose_index", 0)),
        "pred_ref": pred_ref,
        "pred_score": pred_score,
        "is_correct": pred_ref == ref_gt,
        "yolo_conf_cenital": cen_conf,
        "yolo_conf_lateral": lat_conf,
        "color_code": color_code_cen,
        "color_name": cen_color["color_name"],
        "estimated_area_mm2": estimated_area_mm2,
        "estimated_height_mm": measured_h,
        "valid_color": len(valid_by_color),
        "valid_surface": len(valid_by_surface),
        "valid_height": len(valid_by_height),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 78)
    print("RANDOM-POSITION INFERENCE REPORT — set 75078-1")
    print("=" * 78)

    if not os.path.isfile(META_PATH):
        print(f"[ERROR] No se encuentra {META_PATH}.")
        print("        Ejecuta primero generate_set_random_position.py")
        return 1
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        return 1

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    renders = meta.get("renders", [])
    print(f"Renders en metadata : {len(renders)}")
    print(f"FOV cenital         : {meta.get('fov_cenital_mm', '?')} mm")
    print(f"Margen FOV          : {meta.get('margin_mm', '?')} mm\n")

    # ── Cargar modelos ──
    print("Cargando YOLO + SAM + DINOv2...")
    from ultralytics import YOLO
    yolo_cen = YOLO(YOLO_CEN_PATH) if os.path.isfile(YOLO_CEN_PATH) else None
    yolo_lat = YOLO(YOLO_LAT_PATH) if os.path.isfile(YOLO_LAT_PATH) else None
    if yolo_cen is None:
        print(f"[WARN] No se encuentra {YOLO_CEN_PATH}, fallback bbox metadata.")
    if yolo_lat is None:
        print(f"[WARN] No se encuentra {YOLO_LAT_PATH}, fallback bbox metadata.")

    _ = get_sam_model()  # warm-up

    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()
    if not clf.is_ready():
        print("[ERROR] DINOv2/KNN classifier no está listo.")
        return 1
    print("OK.\n")

    # ── CSV writers ──
    os.makedirs(REPORTS_DIR, exist_ok=True)
    f_areas = open(AREAS_CSV, "w", newline="", encoding="utf-8")
    w_areas = csv.writer(f_areas)
    w_areas.writerow([
        "pieza", "pose",
        "surface_silhouette_mm2", "surface_convex_hull_mm2",
        "surface_estimated_inference_mm2",
        "err_silh_pct", "err_convex_pct",
    ])

    f_heights = open(HEIGHTS_CSV, "w", newline="", encoding="utf-8")
    w_heights = csv.writer(f_heights)
    w_heights.writerow([
        "render_lateral",
        "pieza", "pose",
        "height_stable_pose_mm", "height_estimated_inference_mm",
        "err_pct",
    ])

    # ── Procesado ──
    rows_console = []
    correct = 0
    classifiable = 0
    correct_classifiable = 0
    t0 = time.time()

    print("-" * 78)
    print(f"{'#':>3} {'Pieza':<8} {'Pose':>4}  {'Predicha':<10} "
          f"{'Score':>7}  Resultado")
    print("-" * 78)

    for i, entry in enumerate(renders):
        ref_gt = entry["ref"]
        pose_idx = int(entry.get("pose_index", 0))
        is_in_selected = ref_gt in SELECTED_PARTS

        # Nombre de fichero lateral relativo al proyecto
        lat_meta = entry.get("cameras", {}).get("lateral", {})
        lat_abs = _resolve_image_path(lat_meta, DATA_DIR)
        lat_rel = (
            os.path.relpath(lat_abs, LEGOVISION_ROOT)
            if lat_abs else lat_meta.get("file_name", "")
        )

        result = run_inference_for_sample(entry, yolo_cen, yolo_lat, clf)
        if "error" in result:
            print(f"{i+1:3d} {ref_gt:<8} {pose_idx:>4}  "
                  f"{'(error)':<10} {'---':>7}  ✗ {result['error']}")
            continue

        pose_info = lookup_pose(cache, ref_gt, pose_idx)
        sil_mm2 = pose_info.get("zenith_silhouette_area") if pose_info else None
        cvx_mm2 = pose_info.get("zenith_observable_area") if pose_info else None
        h_cache = pose_info.get("lateral_height") if pose_info else None

        est_area = result["estimated_area_mm2"]
        est_h = result["estimated_height_mm"]
        err_sil = pct_err(est_area, sil_mm2)
        err_cvx = pct_err(est_area, cvx_mm2)
        err_h = pct_err(est_h, h_cache)

        # CSV áreas
        w_areas.writerow([
            ref_gt, pose_idx,
            f"{sil_mm2:.2f}" if sil_mm2 is not None else "",
            f"{cvx_mm2:.2f}" if cvx_mm2 is not None else "",
            f"{est_area:.2f}",
            f"{err_sil:.2f}" if err_sil is not None else "",
            f"{err_cvx:.2f}" if err_cvx is not None else "",
        ])
        f_areas.flush()

        # CSV alturas
        w_heights.writerow([
            lat_rel,
            ref_gt, pose_idx,
            f"{h_cache:.2f}" if h_cache is not None else "",
            f"{est_h:.2f}",
            f"{err_h:.2f}" if err_h is not None else "",
        ])
        f_heights.flush()

        is_correct = result["is_correct"]
        if is_correct:
            correct += 1
        if is_in_selected:
            classifiable += 1
            if is_correct:
                correct_classifiable += 1

        if not is_in_selected:
            status = "— sin referencia DINOv2"
        else:
            status = "✓ ACIERTO" if is_correct else "✗ FALLO"
        print(f"{i+1:3d} {ref_gt:<8} {pose_idx:>4}  "
              f"{result['pred_ref']:<10} {result['pred_score']:7.4f}  {status}")

        rows_console.append({
            "ref_gt": ref_gt,
            "pose_index": pose_idx,
            "pred_ref": result["pred_ref"],
            "pred_score": result["pred_score"],
            "is_correct": is_correct,
            "is_in_selected": is_in_selected,
        })

    f_areas.close()
    f_heights.close()
    dt = time.time() - t0

    # ── Resumen final ──
    total = len(rows_console)
    print("\n" + "=" * 78)
    print("RESUMEN GLOBAL")
    print("=" * 78)
    print(f"Muestras procesadas              : {total}")
    print(f"Aciertos (todas)                 : {correct} / {total}  "
          f"({100.0 * correct / max(total,1):.1f} %)")
    print(f"Aciertos (refs en selected_parts): "
          f"{correct_classifiable} / {classifiable}  "
          f"({100.0 * correct_classifiable / max(classifiable,1):.1f} %)")
    out_of_selected = total - classifiable
    if out_of_selected > 0:
        print(f"Refs fuera de selected_parts     : {out_of_selected}  "
              "(no clasificables, siempre fallarán Phase 1)")
    print(f"Tiempo                           : {dt:.1f} s "
          f"({dt / max(total,1):.2f} s/muestra)")
    print(f"\nCSV áreas   : {AREAS_CSV}")
    print(f"CSV alturas : {HEIGHTS_CSV}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
