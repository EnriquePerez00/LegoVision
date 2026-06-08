# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_inference_300_report.py
Genera un report ad-hoc (CSV + HTML resumen) a partir del eval_report.json
producido por `run_evaluation.py` corriendo sobre el dataset de 300
muestras (random_position_300_metadata.json).

Salida:
  <out_dir>/inference_300_full.csv         - 1 fila por muestra (todos los campos)
  <out_dir>/inference_300_summary.html     - vista web con metricas y tablas
  <out_dir>/inference_300_per_piece.csv    - resumen por (ref, color)
  <out_dir>/inference_300_per_pose.csv     - resumen por (ref, pose_index)

Uso:
  python3 2camaras_random_pieza_unica/scripts/generate_inference_300_report.py \\
      --eval 2camaras_random_pieza_unica/data/reports/inference_300_eval.json \\
      --out  2camaras_random_pieza_unica/data/reports/
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────
# Columnas del CSV (orden estable). Se mapean 1:1 desde results[i] del
# eval_report.json producido por run_evaluation.py.
# ─────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "index", "sample_index", "cenital_file", "lateral_file",
    "ref_gt", "pose_index_gt", "face_class_gt",
    "ref_inferred", "model_match", "consensus_score",
    "color_code_gt", "color_name_gt", "color_hex_gt",
    "color_cenital_rgb_est", "color_cenital_normalized_code", "color_cenital_normalized_name",
    "color_lateral_rgb_est", "color_lateral_normalized_code", "color_lateral_normalized_name",
    "color_match_cenital", "color_match_lateral", "color_decision_used",
    "surface_obs_apparent_mm2", "surface_obs_footprint_mm2", "surface_db_silhouette_mm2",
    "surface_error_rel_pct",
    "lateral_height_meas_mm", "lateral_height_db_mm", "lateral_height_error_rel_pct",
    "effective_height_db_mm",
    "yolo_conf_cenital", "yolo_conf_lateral",
    "valid_by_color_count", "valid_by_surface_count", "valid_by_height_count",
]


def _serialize(v):
    """Convierte valores Python a forma plana para CSV."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, bool):
        return "True" if v else "False"
    return v


def write_full_csv(results, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            flat = {k: _serialize(row.get(k)) for k in CSV_COLUMNS}
            writer.writerow(flat)


def aggregate_per_piece(results):
    """Agrega por (ref_gt, color_code_gt). Devuelve lista de dicts."""
    buckets = defaultdict(list)
    for r in results:
        key = (r.get("ref_gt"), r.get("color_code_gt"))
        buckets[key].append(r)
    out = []
    for (ref, color), rows in sorted(buckets.items()):
        n = len(rows)
        n_correct = sum(1 for x in rows if x.get("model_match"))
        accuracy = (n_correct / n * 100.0) if n else 0.0
        col_match_cen = sum(1 for x in rows if x.get("color_match_cenital"))
        col_match_lat = sum(1 for x in rows if x.get("color_match_lateral"))
        surf_errs = [x.get("surface_error_rel_pct") for x in rows if x.get("surface_error_rel_pct") is not None]
        h_errs = [x.get("lateral_height_error_rel_pct") for x in rows if x.get("lateral_height_error_rel_pct") is not None]
        out.append({
            "ref": ref,
            "color_code": color,
            "n_samples": n,
            "model_accuracy_pct": round(accuracy, 2),
            "color_cenital_match_pct": round(col_match_cen / n * 100.0, 2) if n else 0.0,
            "color_lateral_match_pct": round(col_match_lat / n * 100.0, 2) if n else 0.0,
            "surface_err_mean_pct": round(statistics.mean([abs(e) for e in surf_errs]), 2) if surf_errs else None,
            "surface_err_median_pct": round(statistics.median([abs(e) for e in surf_errs]), 2) if surf_errs else None,
            "lateral_h_err_mean_pct": round(statistics.mean([abs(e) for e in h_errs]), 2) if h_errs else None,
            "lateral_h_err_median_pct": round(statistics.median([abs(e) for e in h_errs]), 2) if h_errs else None,
        })
    return out


def aggregate_per_pose(results):
    """Agrega por (ref_gt, pose_index_gt). Devuelve lista de dicts."""
    buckets = defaultdict(list)
    for r in results:
        key = (r.get("ref_gt"), r.get("pose_index_gt"), r.get("face_class_gt"))
        buckets[key].append(r)
    out = []
    for (ref, pose, face), rows in sorted(buckets.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        n = len(rows)
        n_correct = sum(1 for x in rows if x.get("model_match"))
        accuracy = (n_correct / n * 100.0) if n else 0.0
        surf_errs = [x.get("surface_error_rel_pct") for x in rows if x.get("surface_error_rel_pct") is not None]
        h_errs = [x.get("lateral_height_error_rel_pct") for x in rows if x.get("lateral_height_error_rel_pct") is not None]
        out.append({
            "ref": ref,
            "pose_index": pose,
            "face_class": face,
            "n_samples": n,
            "accuracy_pct": round(accuracy, 2),
            "surface_err_mean_pct": round(statistics.mean([abs(e) for e in surf_errs]), 2) if surf_errs else None,
            "lateral_h_err_mean_pct": round(statistics.mean([abs(e) for e in h_errs]), 2) if h_errs else None,
        })
    return out


def write_per_piece_csv(rows, out_path):
    cols = ["ref", "color_code", "n_samples", "model_accuracy_pct",
            "color_cenital_match_pct", "color_lateral_match_pct",
            "surface_err_mean_pct", "surface_err_median_pct",
            "lateral_h_err_mean_pct", "lateral_h_err_median_pct"]
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _serialize(r.get(k)) for k in cols})


def write_per_pose_csv(rows, out_path):
    cols = ["ref", "pose_index", "face_class", "n_samples", "accuracy_pct",
            "surface_err_mean_pct", "lateral_h_err_mean_pct"]
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _serialize(r.get(k)) for k in cols})


# ─────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────
HTML_HEAD = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Inferencia 300 - Set 75078-1</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f8;color:#222;margin:0;padding:24px}}
  h1{{margin-top:0}}
  .card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:16px;
    box-shadow:0 1px 3px rgba(0,0,0,.08)}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}
  .metric{{background:#fff;border-radius:8px;padding:12px 16px;border-left:4px solid #2563eb}}
  .metric .v{{font-size:24px;font-weight:600;color:#0f172a}}
  .metric .k{{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
  table{{border-collapse:collapse;width:100%;font-size:13px}}
  th,td{{padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}}
  th{{background:#f1f5f9;font-weight:600}}
  tr:hover{{background:#fafafa}}
  .ok{{color:#15803d;font-weight:600}}
  .ko{{color:#b91c1c;font-weight:600}}
  .err-bad{{background:#fee2e2}}
  .err-warn{{background:#fef3c7}}
  .err-good{{background:#dcfce7}}
</style></head><body>
"""

HTML_FOOT = "</body></html>"


def _err_class(err_pct):
    if err_pct is None:
        return ""
    a = abs(err_pct)
    if a > 50:
        return "err-bad"
    if a > 20:
        return "err-warn"
    return "err-good"


def render_html(eval_data, per_piece, per_pose, out_path):
    results = eval_data.get("results", [])
    n = len(results)
    correct = sum(1 for r in results if r.get("model_match"))
    accuracy = correct / n * 100.0 if n else 0.0
    n_with_color = sum(1 for r in results if r.get("color_code_gt") is not None)
    cm_cen = sum(1 for r in results if r.get("color_match_cenital"))
    cm_lat = sum(1 for r in results if r.get("color_match_lateral"))
    cm_cen_pct = cm_cen / n_with_color * 100.0 if n_with_color else 0.0
    cm_lat_pct = cm_lat / n_with_color * 100.0 if n_with_color else 0.0

    # Errores agregados (medias de |error_relativo|)
    surf_errs = [r.get("surface_error_rel_pct") for r in results if r.get("surface_error_rel_pct") is not None]
    h_errs = [r.get("lateral_height_error_rel_pct") for r in results if r.get("lateral_height_error_rel_pct") is not None]
    surf_mean = round(statistics.mean([abs(e) for e in surf_errs]), 2) if surf_errs else None
    surf_med = round(statistics.median([abs(e) for e in surf_errs]), 2) if surf_errs else None
    h_mean = round(statistics.mean([abs(e) for e in h_errs]), 2) if h_errs else None
    h_med = round(statistics.median([abs(e) for e in h_errs]), 2) if h_errs else None

    # Top fallos (mayor error de superficie / altura)
    surf_sorted = sorted(
        [r for r in results if r.get("surface_error_rel_pct") is not None],
        key=lambda r: abs(r["surface_error_rel_pct"]),
        reverse=True,
    )[:10]
    h_sorted = sorted(
        [r for r in results if r.get("lateral_height_error_rel_pct") is not None],
        key=lambda r: abs(r["lateral_height_error_rel_pct"]),
        reverse=True,
    )[:10]
    color_mismatches = [r for r in results
                        if (r.get("color_match_cenital") is False
                            or r.get("color_match_lateral") is False)][:15]
    model_mismatches = [r for r in results if r.get("model_match") is False][:15]

    # Confusion matrix simplificada (top pares ref_gt -> ref_inferred)
    confusion = defaultdict(int)
    for r in results:
        confusion[(r.get("ref_gt"), r.get("ref_inferred"))] += 1
    confusion_top = sorted(
        [(gt, pred, n) for (gt, pred), n in confusion.items() if gt != pred],
        key=lambda x: -x[2],
    )[:20]

    # ── Construir HTML ──
    out = [HTML_HEAD]
    out.append(f"<h1>Inferencia 300 - Set 75078-1</h1>")
    out.append(f"<p>Total muestras: <b>{n}</b> | Correctas: <b>{correct}</b> | "
               f"Accuracy global: <b class='ok'>{accuracy:.2f}%</b></p>")

    # ── Metricas globales ──
    out.append("<div class='grid'>")
    out.append(f"<div class='metric'><div class='k'>Accuracy modelo</div><div class='v'>{accuracy:.2f}%</div></div>")
    out.append(f"<div class='metric'><div class='k'>Color match cenital</div><div class='v'>{cm_cen_pct:.1f}%</div></div>")
    out.append(f"<div class='metric'><div class='k'>Color match lateral</div><div class='v'>{cm_lat_pct:.1f}%</div></div>")
    out.append(f"<div class='metric'><div class='k'>Err. superficie media |%|</div><div class='v'>{surf_mean if surf_mean is not None else '-'}</div></div>")
    out.append(f"<div class='metric'><div class='k'>Err. superficie mediana |%|</div><div class='v'>{surf_med if surf_med is not None else '-'}</div></div>")
    out.append(f"<div class='metric'><div class='k'>Err. altura lat. media |%|</div><div class='v'>{h_mean if h_mean is not None else '-'}</div></div>")
    out.append(f"<div class='metric'><div class='k'>Err. altura lat. mediana |%|</div><div class='v'>{h_med if h_med is not None else '-'}</div></div>")
    out.append("</div>")

    # ── Tabla por pieza (ref, color) ──
    out.append("<div class='card'><h2>Resumen por (ref, color)</h2><table><thead><tr>")
    out.append("<th>ref</th><th>color</th><th>n</th><th>acc%</th>"
               "<th>color cen%</th><th>color lat%</th>"
               "<th>err sup mean</th><th>err sup med</th>"
               "<th>err h mean</th><th>err h med</th></tr></thead><tbody>")
    for r in per_piece:
        out.append(
            f"<tr><td>{r['ref']}</td><td>{r['color_code']}</td><td>{r['n_samples']}</td>"
            f"<td>{r['model_accuracy_pct']}</td>"
            f"<td>{r['color_cenital_match_pct']}</td><td>{r['color_lateral_match_pct']}</td>"
            f"<td class='{_err_class(r['surface_err_mean_pct'])}'>{r['surface_err_mean_pct']}</td>"
            f"<td>{r['surface_err_median_pct']}</td>"
            f"<td class='{_err_class(r['lateral_h_err_mean_pct'])}'>{r['lateral_h_err_mean_pct']}</td>"
            f"<td>{r['lateral_h_err_median_pct']}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Tabla por pose ──
    out.append("<div class='card'><h2>Resumen por (ref, pose)</h2><table><thead><tr>")
    out.append("<th>ref</th><th>pose</th><th>face_class</th><th>n</th><th>acc%</th>"
               "<th>err sup mean</th><th>err h mean</th></tr></thead><tbody>")
    for r in per_pose:
        out.append(
            f"<tr><td>{r['ref']}</td><td>{r['pose_index']}</td><td>{r['face_class']}</td>"
            f"<td>{r['n_samples']}</td><td>{r['accuracy_pct']}</td>"
            f"<td class='{_err_class(r['surface_err_mean_pct'])}'>{r['surface_err_mean_pct']}</td>"
            f"<td class='{_err_class(r['lateral_h_err_mean_pct'])}'>{r['lateral_h_err_mean_pct']}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Top fallos superficie ──
    out.append("<div class='card'><h2>Top 10 fallos de superficie</h2><table><thead><tr>"
               "<th>idx</th><th>ref</th><th>pose</th><th>face</th>"
               "<th>obs (mm²)</th><th>db (mm²)</th><th>err %</th></tr></thead><tbody>")
    for r in surf_sorted:
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}</td><td>{r.get('face_class_gt')}</td>"
            f"<td>{r.get('surface_obs_footprint_mm2')}</td>"
            f"<td>{r.get('surface_db_silhouette_mm2')}</td>"
            f"<td class='{_err_class(r.get('surface_error_rel_pct'))}'>"
            f"{r.get('surface_error_rel_pct')}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Top fallos altura ──
    out.append("<div class='card'><h2>Top 10 fallos de altura lateral</h2><table><thead><tr>"
               "<th>idx</th><th>ref</th><th>pose</th><th>face</th>"
               "<th>meas (mm)</th><th>db (mm)</th><th>err %</th></tr></thead><tbody>")
    for r in h_sorted:
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}</td><td>{r.get('face_class_gt')}</td>"
            f"<td>{r.get('lateral_height_meas_mm')}</td>"
            f"<td>{r.get('lateral_height_db_mm')}</td>"
            f"<td class='{_err_class(r.get('lateral_height_error_rel_pct'))}'>"
            f"{r.get('lateral_height_error_rel_pct')}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Mismatch de color ──
    out.append("<div class='card'><h2>Muestras con mismatch de color (top 15)</h2><table><thead><tr>"
               "<th>idx</th><th>ref</th><th>color GT</th>"
               "<th>cen est (rgb)</th><th>cen norm</th><th>match cen</th>"
               "<th>lat est (rgb)</th><th>lat norm</th><th>match lat</th></tr></thead><tbody>")
    for r in color_mismatches:
        cm_c = "<span class='ok'>OK</span>" if r.get("color_match_cenital") else "<span class='ko'>NO</span>"
        cm_l = "<span class='ok'>OK</span>" if r.get("color_match_lateral") else "<span class='ko'>NO</span>"
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('color_code_gt')} ({r.get('color_name_gt')})</td>"
            f"<td>{r.get('color_cenital_rgb_est')}</td>"
            f"<td>{r.get('color_cenital_normalized_code')} ({r.get('color_cenital_normalized_name')})</td>"
            f"<td>{cm_c}</td>"
            f"<td>{r.get('color_lateral_rgb_est')}</td>"
            f"<td>{r.get('color_lateral_normalized_code')} ({r.get('color_lateral_normalized_name')})</td>"
            f"<td>{cm_l}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Top mismatches del modelo ──
    out.append("<div class='card'><h2>Mismatches del modelo (top 15)</h2><table><thead><tr>"
               "<th>idx</th><th>ref GT</th><th>pose</th><th>ref inferido</th><th>score</th></tr></thead><tbody>")
    for r in model_mismatches:
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}</td>"
            f"<td class='ko'>{r.get('ref_inferred')}</td>"
            f"<td>{r.get('consensus_score')}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Confusion matrix simple ──
    out.append("<div class='card'><h2>Confusion matrix (pares ref_gt -> ref_inferido, top 20)</h2>"
               "<table><thead><tr><th>ref_gt</th><th>ref_inferido</th><th>n</th></tr></thead><tbody>")
    for gt, pred, cnt in confusion_top:
        out.append(f"<tr><td>{gt}</td><td class='ko'>{pred}</td><td>{cnt}</td></tr>")
    out.append("</tbody></table></div>")

    out.append(HTML_FOOT)

    with open(out_path, "w", encoding="utf-8") as fp:
        # Tenemos llaves dobles {{}} en el CSS y placeholders sueltos; como
        # construimos string con f-strings ya, escribimos directamente.
        # Convertimos las {{ }} del CSS a {} reales.
        html = "".join(out).replace("{{", "{").replace("}}", "}")
        fp.write(html)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Genera reporte CSV+HTML del eval 300")
    parser.add_argument("--eval", required=True, help="Ruta al eval_report.json")
    parser.add_argument("--out", required=True, help="Directorio de salida")
    args = parser.parse_args()

    if not os.path.isfile(args.eval):
        print(f"[ERROR] No se encuentra {args.eval}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    with open(args.eval, "r", encoding="utf-8") as fp:
        eval_data = json.load(fp)
    results = eval_data.get("results", [])
    if not results:
        print("[WARN] El eval_report no contiene resultados.")

    # CSV completo
    csv_full = os.path.join(args.out, "inference_300_full.csv")
    write_full_csv(results, csv_full)
    print(f"[report] CSV completo  : {csv_full} ({len(results)} filas)")

    # Agregaciones
    per_piece = aggregate_per_piece(results)
    per_pose = aggregate_per_pose(results)

    csv_piece = os.path.join(args.out, "inference_300_per_piece.csv")
    write_per_piece_csv(per_piece, csv_piece)
    print(f"[report] CSV por pieza : {csv_piece} ({len(per_piece)} filas)")

    csv_pose = os.path.join(args.out, "inference_300_per_pose.csv")
    write_per_pose_csv(per_pose, csv_pose)
    print(f"[report] CSV por pose  : {csv_pose} ({len(per_pose)} filas)")

    # HTML resumen
    html_path = os.path.join(args.out, "inference_300_summary.html")
    render_html(eval_data, per_piece, per_pose, html_path)
    print(f"[report] HTML resumen  : {html_path}")


if __name__ == "__main__":
    main()
