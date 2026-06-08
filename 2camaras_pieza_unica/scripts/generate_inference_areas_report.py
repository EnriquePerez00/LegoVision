# -*- coding: utf-8 -*-
"""
2camaras_pieza_unica/scripts/generate_inference_areas_report.py
================================================================
Genera dos reports CSV en 2camaras_pieza_unica/data/reports/ ejecutando
inferencia (YOLO cenital + lateral, SAM segmentation) sobre los renders
del set de test `2camaras_pieza_unica/data/test_dual/`.

Reports producidos
------------------
1. inference_areas_report.csv
     sample, render_cenital, ref, pose,
     silhouette_area_mm2, convex_hull_area_mm2,
     estimated_area_mm2, err_silh_pct, err_convex_pct

2. inference_heights_report.csv
     sample, render_lateral, ref, pose,
     lateral_height_cache_mm, estimated_height_mm, err_pct

El area estimada se calcula a partir de la mascara SAM + correccion de
perspectiva (igual que `estimate_surface_area_sam_corrected` de
`run_evaluation.py`). La altura estimada usa la diferencia vertical de
pixeles de la mascara SAM en la imagen lateral.

Imprime resultados parciales por pantalla cada 5 muestras.

Uso:
    .venv/bin/python 2camaras_pieza_unica/scripts/generate_inference_areas_report.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

PROJECT_ROOT_SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGOVISION_ROOT = os.path.dirname(PROJECT_ROOT_SUB)
sys.path.insert(0, LEGOVISION_ROOT)
sys.path.insert(0, PROJECT_ROOT_SUB)

from config_loader import cfg  # noqa: E402

PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
PX_PER_MM_LATERAL = cfg.inference.calibration.px_per_mm_lateral
CAMERA_DIST_MM = cfg.inference.calibration.camera_dist_mm

TEST_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "test_dual")
META_PATH = os.path.join(TEST_DIR, "test_metadata.json")
CACHE_PATH = os.path.join(PROJECT_ROOT_SUB, "data", "stable_poses_cache.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "reports")
AREAS_CSV = os.path.join(REPORTS_DIR, "inference_areas_reportv2.csv")
HEIGHTS_CSV = os.path.join(REPORTS_DIR, "inference_heights_report.csv")

YOLO_CEN_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_cenital.pt")
YOLO_LAT_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_lateral.pt")
SAM_PATH = os.path.join(PROJECT_ROOT_SUB, "mobile_sam.pt")

PRINT_EVERY = 5  # imprimir resultados parciales cada N muestras


# --------------------------------------------------------------------- IO
def load_cache() -> dict:
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_pose(cache: dict, ref: str, pose_index: int) -> dict | None:
    """Busca la pose por `original_pose_index` (el indice del test_metadata
    es el indice ORIGINAL de la BD). El cache renumera 0..N-1 pero conserva
    `original_pose_index`. Si la ref no está, prueba variantes con sufijo
    de letra ('2412' -> '2412b')."""
    candidate_refs = [ref]
    if ref not in cache:
        # Probar variantes con letra final (b, a, c)
        for suf in ("b", "a", "c"):
            if (ref + suf) in cache:
                candidate_refs.append(ref + suf)
                break
    for cref in candidate_refs:
        poses = cache.get(cref, [])
        for p in poses:
            if p.get("original_pose_index") == pose_index:
                return p
        if 0 <= pose_index < len(poses):
            return poses[pose_index]
    return None


# --------------------------------------------------------------------- Detection
_yolo_cenital = None
_yolo_lateral = None
_sam = None


def get_yolos():
    global _yolo_cenital, _yolo_lateral
    from ultralytics import YOLO
    if _yolo_cenital is None and os.path.exists(YOLO_CEN_PATH):
        _yolo_cenital = YOLO(YOLO_CEN_PATH)
    if _yolo_lateral is None and os.path.exists(YOLO_LAT_PATH):
        _yolo_lateral = YOLO(YOLO_LAT_PATH)
    return _yolo_cenital, _yolo_lateral


def get_sam():
    global _sam
    if _sam is None:
        from ultralytics import SAM
        _sam = SAM(SAM_PATH)
    return _sam


def yolo_detect_bbox(yolo_model, image_path: str):
    """Devuelve bbox normalizada [x1,y1,x2,y2] o None."""
    if yolo_model is None:
        return None, 0.0
    try:
        results = yolo_model.predict(image_path, conf=0.20, iou=0.45,
                                     imgsz=640, max_det=1, verbose=False)
        if not results or len(results) == 0:
            return None, 0.0
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None, 0.0
        box = r.boxes.xyxyn[0].tolist()
        conf = float(r.boxes.conf[0])
        return box, conf
    except Exception:
        return None, 0.0


def segment_crop_sam(img_full: Image.Image, bbox_norm: list) -> np.ndarray:
    """Devuelve mascara binaria del tamaño del CROP definido por bbox_norm."""
    try:
        sam = get_sam()
        iw, ih = img_full.size
        x1, y1, x2, y2 = bbox_norm
        bbox_pix = [int(x1 * iw), int(y1 * ih), int(x2 * iw), int(y2 * ih)]
        results = sam(np.array(img_full), bboxes=[bbox_pix], verbose=False)
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return np.zeros((bbox_pix[3] - bbox_pix[1],
                             bbox_pix[2] - bbox_pix[0]), dtype=np.uint8)
        mask_full = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        # crop
        mask_crop = mask_full[bbox_pix[1]:bbox_pix[3], bbox_pix[0]:bbox_pix[2]]
        return (mask_crop * 255).astype(np.uint8)
    except Exception as e:
        print(f"  [WARN] SAM falló: {e}")
        return np.zeros((1, 1), dtype=np.uint8)


# --------------------------------------------------------------------- Geometric estimation
def estimate_surface_area_sam_corrected(mask_cen: np.ndarray, bbox_norm: list,
                                        rest_h: float = 9.6) -> float:
    """Area en mm^2 de la mascara cenital corrigiendo por perspectiva.

    Replicada de run_evaluation.py para coherencia con el pipeline real.
    """
    try:
        num_pixels = int(np.sum(mask_cen > 0))
        if num_pixels == 0:
            return 0.0
        cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
        cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
        cx_px = cx * 640.0
        cy_px = cy * 640.0

        dx_mm = (cx_px - 320.0) / PX_PER_MM_CENITAL
        dy_mm = (320.0 - cy_px) / PX_PER_MM_CENITAL
        r_mm = math.sqrt(dx_mm ** 2 + dy_mm ** 2)

        d_cam = math.sqrt(r_mm ** 2 + (CAMERA_DIST_MM - rest_h) ** 2)
        # 480 = focal_length_px aproximada usada en run_evaluation
        px_per_mm = 480.0 / d_cam

        area_raw_mm2 = num_pixels / (px_per_mm ** 2)

        w_bbox_mm = (bbox_norm[2] - bbox_norm[0]) * 640.0 / px_per_mm
        h_bbox_mm = (bbox_norm[3] - bbox_norm[1]) * 640.0 / px_per_mm
        perimeter_half = (w_bbox_mm + h_bbox_mm) / 2.0

        side_width_projected = (r_mm * rest_h) / max(CAMERA_DIST_MM - rest_h, 1.0)
        added_side_area_mm2 = perimeter_half * side_width_projected * 0.5

        area_corrected = area_raw_mm2 - added_side_area_mm2
        return max(0.1, area_corrected)
    except Exception:
        return float(np.sum(mask_cen > 0)) / (PX_PER_MM_CENITAL ** 2)


def measure_lateral_height_mm(mask: np.ndarray) -> float:
    """Altura en mm a partir de la mascara lateral (rango vertical de pixeles)."""
    try:
        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            height_px = ys.max() - ys.min()
            return float(height_px) / PX_PER_MM_LATERAL
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------- Main
def main() -> int:
    print("=" * 78)
    print("INFERENCE AREAS + HEIGHTS REPORT — 2camaras_pieza_unica/test_dual")
    print("=" * 78)

    if not os.path.isfile(META_PATH):
        print(f"[ERROR] no se encuentra {META_PATH}")
        return 1
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] no se encuentra {CACHE_PATH}. "
              "Corre primero `sync_stable_poses_cache.py`.")
        return 1

    cache = load_cache()
    meta = load_metadata()
    renders = meta["renders"]
    print(f"Renders en metadata: {len(renders)}")
    print(f"Pieza únicas en cache: {len(cache)}")

    print("\nCargando modelos YOLO + SAM...")
    yolo_cen, yolo_lat = get_yolos()
    if yolo_cen is None or yolo_lat is None:
        print(f"[ERROR] modelos YOLO no encontrados ({YOLO_CEN_PATH} / {YOLO_LAT_PATH})")
        return 1
    _ = get_sam()  # warm-up
    print("OK.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    f_areas = open(AREAS_CSV, "w", newline="", encoding="utf-8")
    w_areas = csv.writer(f_areas)
    w_areas.writerow([
        "sample", "render_cenital", "ref", "pose",
        "silhouette_area_mm2", "convex_hull_area_mm2",
        "estimated_area_mm2", "err_silh_pct", "err_convex_pct",
    ])

    f_heights = open(HEIGHTS_CSV, "w", newline="", encoding="utf-8")
    w_heights = csv.writer(f_heights)
    w_heights.writerow([
        "sample", "render_lateral", "ref", "pose",
        "lateral_height_cache_mm", "estimated_height_mm", "err_pct",
    ])

    print("\n" + "-" * 78)
    print(f"{'#':>3} {'sample':<8} {'ref':<8} {'pose':>4}  "
          f"{'sil_mm2':>9} {'cvx_mm2':>9} {'est_mm2':>9} {'eS%':>7} {'eC%':>7}  "
          f"{'h_cache':>8} {'h_est':>7} {'eH%':>7}")
    print("-" * 78)

    t0 = time.time()
    skipped = 0
    rows_areas: list[list] = []
    rows_heights: list[list] = []

    for i, r in enumerate(renders):
        ref = r["ref"]
        pose_idx = int(r["pose_index"])
        cen = r["cameras"]["cenital"]
        lat = r["cameras"]["lateral"]

        cen_path = os.path.join(LEGOVISION_ROOT, cen["image_path"]) \
            if not os.path.isabs(cen["image_path"]) else cen["image_path"]
        lat_path = os.path.join(LEGOVISION_ROOT, lat["image_path"]) \
            if not os.path.isabs(lat["image_path"]) else lat["image_path"]
        if not os.path.isfile(cen_path):
            cen_path = os.path.join(TEST_DIR, cen["file_name"])
        if not os.path.isfile(lat_path):
            lat_path = os.path.join(TEST_DIR, lat["file_name"])
        if not (os.path.isfile(cen_path) and os.path.isfile(lat_path)):
            print(f"[{i:03d}] {ref} pose#{pose_idx}: faltan imágenes, skip")
            skipped += 1
            continue

        # Lookup silhouette & convex en cache
        pose_info = lookup_pose(cache, ref, pose_idx)
        if pose_info is None:
            sil_mm2 = None
            cvx_mm2 = None
            h_cache_mm = None
            eff_h_mm = None
        else:
            sil_mm2 = pose_info.get("zenith_silhouette_area")
            cvx_mm2 = pose_info.get("zenith_observable_area")
            h_cache_mm = pose_info.get("lateral_height")
            eff_h_mm = pose_info.get("effective_height") or pose_info.get("efective_height")

        # ── Cenital: YOLO -> SAM -> area ──
        bbox_cen, conf_cen = yolo_detect_bbox(yolo_cen, cen_path)
        if bbox_cen is None:
            # fallback al GT del metadata
            bbox_cen = cen.get("bbox_norm")
        img_cen = Image.open(cen_path).convert("RGB")
        mask_cen = segment_crop_sam(img_cen, bbox_cen) if bbox_cen else np.zeros((1, 1), dtype=np.uint8)
        
        # Usar la altura efectiva si existe, si no, fallback a lateral_height
        rest_h_for_perspective = eff_h_mm if eff_h_mm is not None else (h_cache_mm if h_cache_mm is not None else 9.6)
        est_area_mm2 = estimate_surface_area_sam_corrected(
            mask_cen, bbox_cen, rest_h=rest_h_for_perspective
        ) if bbox_cen else 0.0

        # ── Lateral: YOLO -> SAM -> altura ──
        bbox_lat, conf_lat = yolo_detect_bbox(yolo_lat, lat_path)
        if bbox_lat is None:
            bbox_lat = lat.get("bbox_norm")
        img_lat = Image.open(lat_path).convert("RGB")
        mask_lat = segment_crop_sam(img_lat, bbox_lat) if bbox_lat else np.zeros((1, 1), dtype=np.uint8)
        est_height_mm = measure_lateral_height_mm(mask_lat) if bbox_lat is not None else 0.0

        # ── Errores relativos ──
        def pct_err(est, ref_val):
            if ref_val is None or ref_val == 0:
                return None
            return 100.0 * (est - ref_val) / ref_val

        err_sil_pct = pct_err(est_area_mm2, sil_mm2)
        err_cvx_pct = pct_err(est_area_mm2, cvx_mm2)
        err_h_pct = pct_err(est_height_mm, h_cache_mm)

        # ── CSV rows ──
        w_areas.writerow([
            i,
            cen["file_name"],
            ref,
            pose_idx,
            f"{sil_mm2:.2f}" if sil_mm2 is not None else "",
            f"{cvx_mm2:.2f}" if cvx_mm2 is not None else "",
            f"{est_area_mm2:.2f}",
            f"{err_sil_pct:.2f}" if err_sil_pct is not None else "",
            f"{err_cvx_pct:.2f}" if err_cvx_pct is not None else "",
        ])
        f_areas.flush()

        w_heights.writerow([
            i,
            lat["file_name"],
            ref,
            pose_idx,
            f"{h_cache_mm:.2f}" if h_cache_mm is not None else "",
            f"{est_height_mm:.2f}",
            f"{err_h_pct:.2f}" if err_h_pct is not None else "",
        ])
        f_heights.flush()

        rows_areas.append([i, ref, pose_idx, sil_mm2, cvx_mm2, est_area_mm2,
                           err_sil_pct, err_cvx_pct])
        rows_heights.append([i, ref, pose_idx, h_cache_mm, est_height_mm, err_h_pct])

        # Print en una linea
        sil_s = f"{sil_mm2:9.2f}" if sil_mm2 is not None else "      ---"
        cvx_s = f"{cvx_mm2:9.2f}" if cvx_mm2 is not None else "      ---"
        eS_s = f"{err_sil_pct:7.1f}" if err_sil_pct is not None else "    ---"
        eC_s = f"{err_cvx_pct:7.1f}" if err_cvx_pct is not None else "    ---"
        hC_s = f"{h_cache_mm:8.2f}" if h_cache_mm is not None else "     ---"
        eH_s = f"{err_h_pct:7.1f}" if err_h_pct is not None else "    ---"
        print(f"{i:3d} {cen['file_name'][:8]:<8} {ref:<8} {pose_idx:>4}  "
              f"{sil_s} {cvx_s} {est_area_mm2:9.2f} {eS_s} {eC_s}  "
              f"{hC_s} {est_height_mm:7.2f} {eH_s}")

        # Resumen parcial cada PRINT_EVERY
        if (i + 1) % PRINT_EVERY == 0 and rows_areas:
            valid_a = [r for r in rows_areas if r[3] is not None and r[5] is not None]
            valid_h = [r for r in rows_heights if r[3] is not None and r[4] is not None]
            if valid_a:
                ratios_sil = [r[5] / r[3] for r in valid_a if r[3]]
                ratios_cvx = [r[5] / r[4] for r in valid_a if r[4]]
                print(f"   ↳ parcial @{i+1}: ratio est/sil mean={np.mean(ratios_sil):.3f} "
                      f"med={np.median(ratios_sil):.3f}  |  "
                      f"est/cvx mean={np.mean(ratios_cvx):.3f}", end="")
            if valid_h:
                ratios_h = [r[4] / r[3] for r in valid_h if r[3]]
                if ratios_h:
                    print(f"  |  est/h mean={np.mean(ratios_h):.3f}")
                else:
                    print()
            else:
                print()

    f_areas.close()
    f_heights.close()
    dt = time.time() - t0

    # ── Resumen final ──
    print("\n" + "=" * 78)
    print("RESUMEN GLOBAL")
    print("=" * 78)
    valid_a = [r for r in rows_areas if r[3] is not None and r[5] is not None]
    valid_h = [r for r in rows_heights if r[3] is not None and r[4] is not None]
    if valid_a:
        ratios_sil = np.array([r[5] / r[3] for r in valid_a if r[3]])
        ratios_cvx = np.array([r[5] / r[4] for r in valid_a if r[4]])
        print(f"\nÁreas (n={len(valid_a)}):")
        print(f"  est/silhouette : mean={ratios_sil.mean():.3f}  "
              f"median={np.median(ratios_sil):.3f}  "
              f"std={ratios_sil.std():.3f}  "
              f"p10/p90={np.percentile(ratios_sil,10):.3f}/{np.percentile(ratios_sil,90):.3f}")
        print(f"  est/convex_hull: mean={ratios_cvx.mean():.3f}  "
              f"median={np.median(ratios_cvx):.3f}  "
              f"std={ratios_cvx.std():.3f}  "
              f"p10/p90={np.percentile(ratios_cvx,10):.3f}/{np.percentile(ratios_cvx,90):.3f}")
        # MAPE
        ape_sil = np.array([abs(r[6]) for r in valid_a if r[6] is not None])
        ape_cvx = np.array([abs(r[7]) for r in valid_a if r[7] is not None])
        print(f"  MAPE silhouette : {ape_sil.mean():.2f}%  (mediana {np.median(ape_sil):.2f}%)")
        print(f"  MAPE convex_hull: {ape_cvx.mean():.2f}%  (mediana {np.median(ape_cvx):.2f}%)")
    if valid_h:
        ratios_h = np.array([r[4] / r[3] for r in valid_h if r[3]])
        ape_h = np.array([abs(r[5]) for r in valid_h if r[5] is not None])
        print(f"\nAlturas (n={len(valid_h)}):")
        print(f"  est/cache : mean={ratios_h.mean():.3f}  "
              f"median={np.median(ratios_h):.3f}  "
              f"std={ratios_h.std():.3f}  "
              f"p10/p90={np.percentile(ratios_h,10):.3f}/{np.percentile(ratios_h,90):.3f}")
        print(f"  MAPE      : {ape_h.mean():.2f}%  (mediana {np.median(ape_h):.2f}%)")

    print(f"\nProcesadas: {len(rows_areas)} muestras  (omitidas {skipped})  "
          f"en {dt:.1f}s ({dt/max(len(rows_areas),1):.2f}s/sample)")
    print(f"\nCSV Áreas   : {AREAS_CSV}")
    print(f"CSV Alturas : {HEIGHTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
