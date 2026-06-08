# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/analyze_bbox_accuracy.py
==================================================================
Evalúa la precisión de los bounding boxes generados por los modelos YOLO
(cenital y lateral) sobre el set de 300 muestras de posición aleatoria.

Para cada sample:
  • Ejecuta YOLO (modelos cenital y lateral) sobre la imagen.
  • Compara el bbox predicho (mayor confianza) con el bbox GT del metadata.
  • Calcula IoU.

Métricas:
  • % éxito con IoU >= 0.5 (criterio estándar COCO).
  • % éxito con IoU >= 0.75 (criterio estricto).
  • % de samples sin detección (false negatives).

Salidas:
  • data/reports/bbox_accuracy_300.csv         (todos los samples)
  • data/reports/bbox_accuracy_300_failures.csv (solo fallos)
  • data/reports/bbox_accuracy_300_summary.txt (resumen).
  • Resumen por consola.

Uso:
  .venv/bin/python 2camaras_random_pieza_unica/scripts/analyze_bbox_accuracy.py
"""
import os
import sys
import json
import csv
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from ultralytics import YOLO


METADATA_PATH = os.path.join(
    project_root, "data", "random_position", "random_position_300_metadata.json"
)
TEST_DIR = os.path.join(project_root, "data", "random_position")
YOLO_CENITAL = os.path.join(project_root, "models", "yolo_cenital.pt")
YOLO_LATERAL = os.path.join(project_root, "models", "yolo_lateral.pt")
OUT_DIR = os.path.join(project_root, "data", "reports")

IOU_GOOD = 0.5
IOU_STRICT = 0.75


def iou_norm(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def yolo_detect_bbox(model, img_path, conf_threshold=0.25):
    try:
        results = model(img_path, verbose=False, conf=conf_threshold)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            bbox_norm = boxes.xyxyn[best_idx].cpu().numpy().tolist()
            conf = float(boxes.conf[best_idx].cpu().numpy())
            return bbox_norm, conf
    except Exception as e:
        print(f"[WARN] YOLO error en {img_path}: {e}")
    return None, 0.0


def main():
    if not os.path.isfile(METADATA_PATH):
        print(f"[ERROR] No se encuentra metadata: {METADATA_PATH}")
        sys.exit(1)
    if not os.path.isfile(YOLO_CENITAL):
        print(f"[ERROR] No se encuentra YOLO cenital: {YOLO_CENITAL}")
        sys.exit(1)
    if not os.path.isfile(YOLO_LATERAL):
        print(f"[ERROR] No se encuentra YOLO lateral: {YOLO_LATERAL}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[Info] Cargando YOLO cenital: {YOLO_CENITAL}")
    yolo_cen = YOLO(YOLO_CENITAL)
    print(f"[Info] Cargando YOLO lateral: {YOLO_LATERAL}")
    yolo_lat = YOLO(YOLO_LATERAL)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    renders = meta.get("renders", [])
    n_total = len(renders)
    print(f"[Info] Total de samples: {n_total}")

    rows = []
    stats = {
        "cenital": {"detected": 0, "good": 0, "strict": 0, "no_det": 0, "low_iou": 0},
        "lateral": {"detected": 0, "good": 0, "strict": 0, "no_det": 0, "low_iou": 0},
    }
    per_piece = defaultdict(lambda: {
        "total": 0, "cen_good": 0, "lat_good": 0,
        "cen_no_det": 0, "lat_no_det": 0,
    })

    for idx, entry in enumerate(renders):
        ref = entry["ref"]
        pose = entry.get("pose_index", 0)
        cams = entry["cameras"]
        cen_meta = cams["cenital"]
        lat_meta = cams["lateral"]
        cen_path = os.path.join(TEST_DIR, cen_meta["file_name"])
        lat_path = os.path.join(TEST_DIR, lat_meta["file_name"])
        cen_gt = cen_meta["bbox_norm"]
        lat_gt = lat_meta["bbox_norm"]

        cen_pred, cen_conf = yolo_detect_bbox(yolo_cen, cen_path)
        if cen_pred is None:
            cen_iou = 0.0
            cen_status = "no_detection"
            stats["cenital"]["no_det"] += 1
        else:
            stats["cenital"]["detected"] += 1
            cen_iou = iou_norm(cen_pred, cen_gt)
            if cen_iou >= IOU_STRICT:
                stats["cenital"]["strict"] += 1
            if cen_iou >= IOU_GOOD:
                stats["cenital"]["good"] += 1
                cen_status = "ok"
            else:
                stats["cenital"]["low_iou"] += 1
                cen_status = "low_iou"

        lat_pred, lat_conf = yolo_detect_bbox(yolo_lat, lat_path)
        if lat_pred is None:
            lat_iou = 0.0
            lat_status = "no_detection"
            stats["lateral"]["no_det"] += 1
        else:
            stats["lateral"]["detected"] += 1
            lat_iou = iou_norm(lat_pred, lat_gt)
            if lat_iou >= IOU_STRICT:
                stats["lateral"]["strict"] += 1
            if lat_iou >= IOU_GOOD:
                stats["lateral"]["good"] += 1
                lat_status = "ok"
            else:
                stats["lateral"]["low_iou"] += 1
                lat_status = "low_iou"

        per_piece[ref]["total"] += 1
        if cen_status == "ok":
            per_piece[ref]["cen_good"] += 1
        if cen_status == "no_detection":
            per_piece[ref]["cen_no_det"] += 1
        if lat_status == "ok":
            per_piece[ref]["lat_good"] += 1
        if lat_status == "no_detection":
            per_piece[ref]["lat_no_det"] += 1

        rows.append({
            "index": idx,
            "ref": ref,
            "pose": pose,
            "cenital_file": cen_meta["file_name"],
            "lateral_file": lat_meta["file_name"],
            "cen_iou": round(cen_iou, 4),
            "cen_conf": round(cen_conf, 4),
            "cen_status": cen_status,
            "cen_pred_bbox": [round(v, 4) for v in cen_pred] if cen_pred is not None else None,
            "cen_gt_bbox": [round(v, 4) for v in cen_gt],
            "lat_iou": round(lat_iou, 4),
            "lat_conf": round(lat_conf, 4),
            "lat_status": lat_status,
            "lat_pred_bbox": [round(v, 4) for v in lat_pred] if lat_pred is not None else None,
            "lat_gt_bbox": [round(v, 4) for v in lat_gt],
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == n_total:
            print(
                f"  [{idx+1:03d}/{n_total}] cen_iou={cen_iou:.2f} ({cen_status}) | "
                f"lat_iou={lat_iou:.2f} ({lat_status})"
            )

    csv_all = os.path.join(OUT_DIR, "bbox_accuracy_300.csv")
    csv_fail = os.path.join(OUT_DIR, "bbox_accuracy_300_failures.csv")
    summary_path = os.path.join(OUT_DIR, "bbox_accuracy_300_summary.txt")

    with open(csv_all, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "ref", "pose", "cenital_file", "lateral_file",
            "cen_iou", "cen_conf", "cen_status", "cen_pred_bbox", "cen_gt_bbox",
            "lat_iou", "lat_conf", "lat_status", "lat_pred_bbox", "lat_gt_bbox",
        ])
        for r in rows:
            w.writerow([
                r["index"], r["ref"], r["pose"], r["cenital_file"], r["lateral_file"],
                r["cen_iou"], r["cen_conf"], r["cen_status"], r["cen_pred_bbox"], r["cen_gt_bbox"],
                r["lat_iou"], r["lat_conf"], r["lat_status"], r["lat_pred_bbox"], r["lat_gt_bbox"],
            ])

    failures = [r for r in rows if r["cen_status"] != "ok" or r["lat_status"] != "ok"]
    with open(csv_fail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ref", "pose", "cenital_file", "lateral_file",
            "cen_iou", "cen_status", "lat_iou", "lat_status",
            "cen_conf", "lat_conf",
        ])
        for r in failures:
            w.writerow([
                r["ref"], r["pose"], r["cenital_file"], r["lateral_file"],
                r["cen_iou"], r["cen_status"], r["lat_iou"], r["lat_status"],
                r["cen_conf"], r["lat_conf"],
            ])

    # ── Resumen ──
    n = n_total
    def pct(x):
        return (x / n * 100.0) if n > 0 else 0.0

    s_cen = stats["cenital"]
    s_lat = stats["lateral"]

    lines = []
    lines.append("=" * 78)
    lines.append("BBOX ACCURACY REPORT — 300 random_position samples")
    lines.append("=" * 78)
    lines.append(f"Total samples: {n}")
    lines.append("")
    lines.append("--- CENITAL ---")
    lines.append(f"  Detecciones (algo detectado): {s_cen['detected']}/{n}  ({pct(s_cen['detected']):.2f}%)")
    lines.append(f"  Sin detección              : {s_cen['no_det']}/{n}  ({pct(s_cen['no_det']):.2f}%)")
    lines.append(f"  Éxito IoU>=0.50            : {s_cen['good']}/{n}  ({pct(s_cen['good']):.2f}%)")
    lines.append(f"  Éxito IoU>=0.75            : {s_cen['strict']}/{n}  ({pct(s_cen['strict']):.2f}%)")
    lines.append(f"  Detectado pero IoU<0.50    : {s_cen['low_iou']}/{n}  ({pct(s_cen['low_iou']):.2f}%)")
    lines.append("")
    lines.append("--- LATERAL ---")
    lines.append(f"  Detecciones (algo detectado): {s_lat['detected']}/{n}  ({pct(s_lat['detected']):.2f}%)")
    lines.append(f"  Sin detección              : {s_lat['no_det']}/{n}  ({pct(s_lat['no_det']):.2f}%)")
    lines.append(f"  Éxito IoU>=0.50            : {s_lat['good']}/{n}  ({pct(s_lat['good']):.2f}%)")
    lines.append(f"  Éxito IoU>=0.75            : {s_lat['strict']}/{n}  ({pct(s_lat['strict']):.2f}%)")
    lines.append(f"  Detectado pero IoU<0.50    : {s_lat['low_iou']}/{n}  ({pct(s_lat['low_iou']):.2f}%)")
    lines.append("")
    lines.append(f"Total fallos (cen o lat): {len(failures)}/{n}  ({(len(failures)/n*100):.2f}%)")
    lines.append("")
    lines.append("--- FALLOS ---")
    lines.append(f"{'ref':<8} {'pose':<5} {'cen_iou':<8} {'cen_st':<14} {'lat_iou':<8} {'lat_st':<14} cen_file")
    for r in failures:
        lines.append(
            f"{r['ref']:<8} {str(r['pose']):<5} "
            f"{r['cen_iou']:<8.3f} {r['cen_status']:<14} "
            f"{r['lat_iou']:<8.3f} {r['lat_status']:<14} {r['cenital_file']}"
        )
    lines.append("")
    lines.append("--- PER-PIECE STATS ---")
    lines.append(f"{'ref':<8} {'total':<6} {'cen_ok':<7} {'lat_ok':<7} {'cen_nd':<7} {'lat_nd':<7}")
    for ref in sorted(per_piece.keys()):
        s = per_piece[ref]
        lines.append(
            f"{ref:<8} {s['total']:<6} {s['cen_good']:<7} {s['lat_good']:<7} "
            f"{s['cen_no_det']:<7} {s['lat_no_det']:<7}"
        )

    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print()
    print(summary)
    print()
    print(f"[OK] CSV completo:    {csv_all}")
    print(f"[OK] CSV fallos:      {csv_fail}")
    print(f"[OK] Resumen:         {summary_path}")


if __name__ == "__main__":
    main()
