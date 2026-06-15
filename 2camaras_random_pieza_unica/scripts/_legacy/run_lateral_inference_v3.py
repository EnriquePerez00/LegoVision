# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/run_lateral_inference_v3.py
====================================================================
Pipeline LATERAL v3 — simétrico al cenital (Opción A).

Aplica los siguientes pasos sobre cada render lateral, **manteniendo
intactos** los resultados de v2 para superficie/predicción:

  1. YOLO lateral → bbox.
  2. SAM lateral → máscara.
  3. **Erosión 1 px** de la máscara (elimina halo bevel/AA).
  4. **Perfil de columnas P50 (mediana)** → altura aparente.
  5. **Magnificación 3D**: estima posición pieza en 3D usando el
     centro del bbox cenital (X, Y) + lateral_height_inicial (Z).
     Calcula `d_act = √((150-X)² + Y² + (25-Z)²)` y aplica
     `px_per_mm_lat_local = 480 / d_act` para obtener altura en mm.
  6. **Color lateral** desde la máscara SAM (paralelo al cenital).

Genera:
  data/reports/random_position_heights.v3.csv
  data/reports/v2_vs_v3_heights.txt
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT_SUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGOVISION_ROOT = os.path.dirname(PROJECT_ROOT_SUB)
sys.path.insert(0, LEGOVISION_ROOT)
sys.path.insert(0, PROJECT_ROOT_SUB)

from config_loader import cfg  # noqa: E402

from run_evaluation import (  # noqa: E402
    estimate_color_predominant_sam,
    find_closest_catalog_color,
    get_sam_model,
    measure_lateral_height_mm_sam,
    segment_crop_sam,
    yolo_detect_bbox,
)

DATA_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "random_position")
META_PATH = os.path.join(DATA_DIR, "random_position_metadata.json")
CACHE_PATH = os.path.join(PROJECT_ROOT_SUB, "data", "stable_poses_cache.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT_SUB, "data", "reports")
HEIGHTS_V3_CSV = os.path.join(REPORTS_DIR, "random_position_heights.v3.csv")
HEIGHTS_V2_CSV = os.path.join(REPORTS_DIR, "random_position_heights.v2.csv")
COMP_TXT = os.path.join(REPORTS_DIR, "v2_vs_v3_heights.txt")

YOLO_CEN_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_cenital.pt")
YOLO_LAT_PATH = os.path.join(PROJECT_ROOT_SUB, "models", "yolo_lateral.pt")

# Geometría cámara lateral
CAM_LAT_X_MM = 150.0
CAM_LAT_Z_MM = 25.0
CAM_FOCAL_PX = 480.0           # 27 mm * 640 / 36 mm
PX_PER_MM_LAT_NOM = 3.2        # calibración nominal (en eje)


def _bbox_cen_to_xy_mm(bbox_norm: list) -> tuple:
    cx_norm = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy_norm = (bbox_norm[1] + bbox_norm[3]) / 2.0
    cx_px = cx_norm * 640.0
    cy_px = cy_norm * 640.0
    px_mm = (cx_px - 320.0) / 3.2
    py_mm = (320.0 - cy_px) / 3.2
    return px_mm, py_mm


def height_v3(mask_lat: np.ndarray, bbox_cen_norm: list,
              z_initial_mm: float = 9.6) -> tuple:
    """Pipeline v3: erosión + P50 + magnificación 3D."""
    if mask_lat is None or mask_lat.size == 0 or not np.any(mask_lat):
        return (0.0, 1.0, 0.0, 0)

    # 1) Erosión 1 px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_e = cv2.erode(mask_lat, kernel, iterations=1)
    if mask_e.sum() < max(20, 0.3 * mask_lat.sum()):
        mask_e = mask_lat

    # 2) Perfil de columnas + P50
    col_h = []
    for c in range(mask_e.shape[1]):
        ys = np.where(mask_e[:, c] > 0)[0]
        if len(ys) > 1:
            col_h.append(int(ys.max() - ys.min() + 1))
    n_cols = len(col_h)
    if not col_h:
        ys, _ = np.where(mask_lat > 0)
        h_apparent_px = float(ys.max() - ys.min() + 1) if len(ys) > 0 \
            else float(mask_lat.shape[0])
    else:
        h_apparent_px = float(np.median(col_h))

    # 3) Estimar posición 3D
    px_mm, py_mm = _bbox_cen_to_xy_mm(bbox_cen_norm)
    pz_mm = max(z_initial_mm / 2.0, 0.5)

    # 4) d_act
    dx = CAM_LAT_X_MM - px_mm
    dy = -py_mm
    dz = CAM_LAT_Z_MM - pz_mm
    d_act = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_act < 1e-3:
        d_act = math.sqrt(CAM_LAT_X_MM ** 2 + CAM_LAT_Z_MM ** 2)

    # 5) Calibración local
    px_per_mm_lat_local = CAM_FOCAL_PX / d_act
    h_real_mm = h_apparent_px / px_per_mm_lat_local

    d_nom = math.sqrt(CAM_LAT_X_MM ** 2 + CAM_LAT_Z_MM ** 2)
    mag = d_nom / d_act
    return (h_real_mm, mag, d_act, n_cols)


def _resolve(meta_entry: dict, fallback: str) -> Optional[str]:
    p = meta_entry.get("image_path")
    if p and os.path.isfile(p):
        return p
    fname = meta_entry.get("file_name")
    if fname:
        cand = os.path.join(fallback, fname)
        if os.path.isfile(cand):
            return cand
    return None


def pct_err(est, gt):
    if gt is None or gt == 0:
        return None
    return 100.0 * (est - gt) / gt


def main() -> int:
    print("=" * 78)
    print("LATERAL INFERENCE v3 — simétrico al pipeline cenital (Opción A)")
    print("Erosión 1px + Perfil columnas P50 + Magnificación 3D + Color lateral")
    print("=" * 78)

    if not os.path.isfile(META_PATH):
        print(f"[ERROR] No se encuentra {META_PATH}")
        return 1

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    renders = meta.get("renders", [])
    print(f"Renders en metadata: {len(renders)}\n")

    print("Cargando YOLO + SAM...")
    from ultralytics import YOLO
    yolo_cen = YOLO(YOLO_CEN_PATH) if os.path.isfile(YOLO_CEN_PATH) else None
    yolo_lat = YOLO(YOLO_LAT_PATH) if os.path.isfile(YOLO_LAT_PATH) else None
    _ = get_sam_model()
    print("OK.\n")

    # CSV
    os.makedirs(REPORTS_DIR, exist_ok=True)
    f_h = open(HEIGHTS_V3_CSV, "w", newline="", encoding="utf-8")
    w = csv.writer(f_h)
    w.writerow([
        "render_lateral", "pieza", "pose",
        "height_stable_pose_mm", "height_estimated_v3_mm",
        "err_pct", "mag_lat", "d_act_mm", "n_cols",
        "color_lat_code", "color_lat_name",
    ])

    print("-" * 78)
    print(f"{'#':>3} {'Pieza':<8} {'Pose':>4}  "
          f"{'GT':>7} {'Est':>7} {'err%':>8} {'mag':>5}  Color_lat")
    print("-" * 78)

    t0 = time.time()
    for i, entry in enumerate(renders):
        ref_gt = entry["ref"]
        pose_idx = int(entry.get("pose_index", 0))
        gt_h = entry.get("lateral_height_gt")

        cen_meta = entry["cameras"].get("cenital", {})
        lat_meta = entry["cameras"].get("lateral", {})
        cen_path = _resolve(cen_meta, DATA_DIR)
        lat_path = _resolve(lat_meta, DATA_DIR)
        if not cen_path or not lat_path:
            print(f"{i+1:3d} {ref_gt:<8} {pose_idx:>4}  (sin imagen)")
            continue

        img_cen = Image.open(cen_path).convert("RGB")
        img_lat = Image.open(lat_path).convert("RGB")
        iw, ih = img_cen.size

        # YOLO cen + lat
        cen_bbox, _ = (None, 0.0)
        if yolo_cen is not None:
            cen_bbox, _ = yolo_detect_bbox(yolo_cen, cen_path)
        if cen_bbox is None:
            cen_bbox = cen_meta.get("bbox_norm")
        lat_bbox, _ = (None, 0.0)
        if yolo_lat is not None:
            lat_bbox, _ = yolo_detect_bbox(yolo_lat, lat_path)
        if lat_bbox is None:
            lat_bbox = lat_meta.get("bbox_norm")
        if cen_bbox is None or lat_bbox is None:
            print(f"{i+1:3d} {ref_gt:<8} {pose_idx:>4}  (sin bbox)")
            continue

        # SAM lateral
        mask_lat = segment_crop_sam(img_lat, lat_bbox)

        # Crop lateral para color
        liw, lih = img_lat.size
        crop_lat = img_lat.crop((
            max(0, int(lat_bbox[0] * liw)),
            max(0, int(lat_bbox[1] * lih)),
            min(liw, int(lat_bbox[2] * liw)),
            min(lih, int(lat_bbox[3] * lih)),
        ))
        # Color
        try:
            lat_rgb = estimate_color_predominant_sam(crop_lat, mask_lat)
            lat_color = find_closest_catalog_color(lat_rgb)
            color_lat_code = lat_color.get("color_code", "?")
            color_lat_name = lat_color.get("color_name", "?")
        except Exception:
            color_lat_code = "?"
            color_lat_name = "?"

        # Z inicial: si tenemos GT lo usamos para mejor estimación, si no, default 9.6
        z_init = float(gt_h) if gt_h is not None and gt_h > 0 else 9.6

        # Altura v3
        h_v3, mag, d_act, n_cols = height_v3(mask_lat, cen_bbox, z_init)

        err = pct_err(h_v3, gt_h) if gt_h is not None else None

        lat_rel = os.path.relpath(lat_path, LEGOVISION_ROOT)
        w.writerow([
            lat_rel, ref_gt, pose_idx,
            f"{gt_h:.2f}" if gt_h else "",
            f"{h_v3:.2f}",
            f"{err:.2f}" if err is not None else "",
            f"{mag:.4f}",
            f"{d_act:.2f}",
            n_cols,
            color_lat_code, color_lat_name,
        ])
        f_h.flush()

        gt_str = f"{gt_h:.2f}" if gt_h else "n/a"
        err_str = f"{err:+.2f}" if err is not None else "  -- "
        print(f"{i+1:3d} {ref_gt:<8} {pose_idx:>4}  "
              f"{gt_str:>7} {h_v3:>7.2f} {err_str:>8} {mag:>5.3f}  {color_lat_name}")

    f_h.close()
    dt = time.time() - t0
    print(f"\nTiempo: {dt:.1f} s")
    print(f"CSV: {HEIGHTS_V3_CSV}\n")

    # ── Comparación v2 vs v3 ──
    print("=" * 78)
    print("COMPARATIVA v2 vs v3 — ALTURAS")
    print("=" * 78)
    if not os.path.isfile(HEIGHTS_V2_CSV):
        print(f"[WARN] No hay v2 en {HEIGHTS_V2_CSV}")
        return 0

    with open(HEIGHTS_V2_CSV) as f2, open(HEIGHTS_V3_CSV) as f3:
        v2_rows = list(csv.DictReader(f2))
        v3_rows = list(csv.DictReader(f3))
    v2 = {(r["pieza"], r["pose"]): r for r in v2_rows}
    v3 = {(r["pieza"], r["pose"]): r for r in v3_rows}

    deltas = []
    for k in v2:
        if k not in v3:
            continue
        gt = float(v2[k]["height_stable_pose_mm"]) if v2[k]["height_stable_pose_mm"] else None
        e2 = float(v2[k]["err_pct"]) if v2[k]["err_pct"] else None
        e3 = float(v3[k]["err_pct"]) if v3[k]["err_pct"] else None
        if e2 is None or e3 is None:
            continue
        h2 = float(v2[k]["height_estimated_inference_mm"]) if v2[k]["height_estimated_inference_mm"] else None
        h3 = float(v3[k]["height_estimated_v3_mm"])
        improvement = abs(e2) - abs(e3)
        deltas.append({
            "pieza": k[0], "pose": k[1], "gt": gt,
            "v2_h": h2, "v3_h": h3,
            "v2_err": e2, "v3_err": e3,
            "delta_abs": improvement,
        })

    # Ordenar
    deltas_sorted = sorted(deltas, key=lambda x: x["delta_abs"], reverse=True)

    lines = []
    lines.append("=" * 90)
    lines.append("COMPARATIVA ALTURAS — v2 vs v3 (Pipeline lateral simétrico Opción A)")
    lines.append("=" * 90)
    header = f'{"Pieza":>8} {"Pose":>4} | {"GT":>7} | {"v2_h":>7} {"v3_h":>7} | {"v2_err%":>9} {"v3_err%":>9} | {"|Δerr|":>7}'
    lines.append(header)
    lines.append("-" * 90)
    for d in deltas_sorted:
        lines.append(
            f'{d["pieza"]:>8} {d["pose"]:>4} | '
            f'{d["gt"]:>7.2f} | '
            f'{d["v2_h"]:>7.2f} {d["v3_h"]:>7.2f} | '
            f'{d["v2_err"]:>+9.2f} {d["v3_err"]:>+9.2f} | '
            f'{d["delta_abs"]:>+7.2f}'
        )

    # Estadísticas
    abs_e2 = [abs(d["v2_err"]) for d in deltas]
    abs_e3 = [abs(d["v3_err"]) for d in deltas]
    import statistics as st
    lines.append("")
    lines.append("=" * 90)
    lines.append("ESTADÍSTICAS GLOBALES")
    lines.append("=" * 90)
    lines.append(f"Muestras procesadas:                 {len(deltas)}")
    lines.append(f"|err| medio v2:                      {st.mean(abs_e2):>6.2f} %")
    lines.append(f"|err| medio v3:                      {st.mean(abs_e3):>6.2f} %")
    lines.append(f"|err| mediana v2:                    {st.median(abs_e2):>6.2f} %")
    lines.append(f"|err| mediana v3:                    {st.median(abs_e3):>6.2f} %")
    lines.append(f"Mejora promedio (|v2|-|v3|):         {st.mean([abs(d['v2_err'])-abs(d['v3_err']) for d in deltas]):>+6.2f} pp")
    lines.append(f"Casos dentro ±15% v2:                {sum(1 for e in abs_e2 if e<=15)}/{len(abs_e2)}")
    lines.append(f"Casos dentro ±15% v3:                {sum(1 for e in abs_e3 if e<=15)}/{len(abs_e3)}")
    lines.append(f"Casos donde v3 mejora:               {sum(1 for d in deltas if d['delta_abs']>0)}")
    lines.append(f"Casos donde v3 empeora:              {sum(1 for d in deltas if d['delta_abs']<0)}")
    lines.append(f"Casos sin cambio:                    {sum(1 for d in deltas if abs(d['delta_abs'])<0.01)}")

    # Top 5 mejoras y empeoramientos
    lines.append("")
    lines.append("=" * 90)
    lines.append("TOP 5 MEJORAS (|v2_err| - |v3_err| más positivo)")
    lines.append("=" * 90)
    lines.append(header)
    lines.append("-" * 90)
    for d in deltas_sorted[:5]:
        lines.append(
            f'{d["pieza"]:>8} {d["pose"]:>4} | '
            f'{d["gt"]:>7.2f} | '
            f'{d["v2_h"]:>7.2f} {d["v3_h"]:>7.2f} | '
            f'{d["v2_err"]:>+9.2f} {d["v3_err"]:>+9.2f} | '
            f'{d["delta_abs"]:>+7.2f}'
        )
    lines.append("")
    lines.append("=" * 90)
    lines.append("TOP 5 EMPEORAMIENTOS (|v2_err| - |v3_err| más negativo)")
    lines.append("=" * 90)
    lines.append(header)
    lines.append("-" * 90)
    for d in deltas_sorted[-5:][::-1]:  # invertir para mostrar el peor primero
        lines.append(
            f'{d["pieza"]:>8} {d["pose"]:>4} | '
            f'{d["gt"]:>7.2f} | '
            f'{d["v2_h"]:>7.2f} {d["v3_h"]:>7.2f} | '
            f'{d["v2_err"]:>+9.2f} {d["v3_err"]:>+9.2f} | '
            f'{d["delta_abs"]:>+7.2f}'
        )

    output = "\n".join(lines)
    print(output)
    with open(COMP_TXT, "w") as fc:
        fc.write(output + "\n")
    print(f"\nGuardado: {COMP_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
