# -*- coding: utf-8 -*-
"""generate_inference_300_errors_report.py

Reporte focalizado en ERRORES de la ultima inferencia sobre el dataset
de 300 muestras (random_position_300_metadata.json).
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict, Counter


HEIGHT_CSV_COLS = [
    "index", "sample_index", "ref_gt", "pose_index_gt", "face_class_gt",
    "ref_inferred", "model_match",
    "lateral_height_meas_mm", "lateral_height_db_mm",
    "lateral_height_error_rel_pct", "effective_height_db_mm",
    "yolo_conf_lateral", "cenital_file", "lateral_file",
]

SURFACE_CSV_COLS = [
    "index", "sample_index", "ref_gt", "pose_index_gt", "face_class_gt",
    "ref_inferred", "model_match",
    "surface_obs_apparent_mm2", "surface_obs_footprint_mm2",
    "surface_db_silhouette_mm2", "surface_error_rel_pct",
    "yolo_conf_cenital", "cenital_file", "lateral_file",
]

COLOR_CSV_COLS = [
    "index", "sample_index", "ref_gt", "pose_index_gt", "face_class_gt",
    "color_code_gt", "color_name_gt", "color_hex_gt",
    "color_cenital_rgb_est", "color_cenital_normalized_code", "color_cenital_normalized_name",
    "color_match_cenital",
    "color_lateral_rgb_est", "color_lateral_normalized_code", "color_lateral_normalized_name",
    "color_match_lateral",
    "color_decision_used", "color_consensus_status", "color_consensus_ok",
    "cenital_file", "lateral_file",
]


def _serialize(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, bool):
        return "True" if v else "False"
    return v


def _err_class(err_pct, warn=20.0, bad=50.0):
    if err_pct is None or not isinstance(err_pct, (int, float)):
        return ""
    a = abs(err_pct)
    if a > bad:
        return "err-bad"
    if a > warn:
        return "err-warn"
    return "err-good"


def _has_color_mismatch(r):
    return (r.get("color_match_cenital") is False
            or r.get("color_match_lateral") is False
            or r.get("color_consensus_ok") is False)


def write_csv(rows, cols, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _serialize(r.get(k)) for k in cols})


def histogram(values, bins):
    counts = [0] * len(bins)
    for v in values:
        if v is None:
            continue
        a = abs(v)
        for i, (lo, hi) in enumerate(bins):
            if (lo <= a < hi) or (hi == float("inf") and a >= lo):
                counts[i] += 1
                break
    labels = []
    for lo, hi in bins:
        labels.append(f">{lo:g}%" if hi == float("inf") else f"{lo:g}-{hi:g}%")
    return list(zip(labels, counts))


def _rgb_to_hex(rgb):
    if not rgb or not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
        return None
    try:
        return "#{:02X}{:02X}{:02X}".format(
            max(0, min(255, int(round(rgb[0])))),
            max(0, min(255, int(round(rgb[1])))),
            max(0, min(255, int(round(rgb[2])))),
        )
    except Exception:
        return None


def _swatch(hex_):
    if not hex_:
        return ""
    return f'<span class="swatch" style="background:{hex_}"></span>'


def _file_cell(r):
    cen = r.get("cenital_file") or ""
    lat = r.get("lateral_file") or ""
    return (f"<div class='filename'>cen: {cen}</div>"
            f"<div class='filename'>lat: {lat}</div>")


def _fmt(v, decimals=2):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _stats_block(values):
    if not values:
        return None
    sv = sorted(values)
    p90 = sv[int(len(sv) * 0.9) - 1] if len(sv) >= 10 else sv[-1]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
        "p90": round(p90, 2),
    }


def _bar_html(count, max_count, width_px=240):
    if max_count <= 0:
        return ""
    w = int(round(count / max_count * width_px))
    return f"<span class='bar' style='width:{w}px'></span>"

HTML_HEAD = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Inferencia 300 - Reporte de errores</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f8;color:#222;margin:0;padding:24px}
  h1{margin:0 0 8px}
  h2{margin-top:0;color:#0f172a}
  .sub{color:#64748b;margin-top:0}
  .card{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
  .metric{background:#fff;border-radius:8px;padding:12px 16px;border-left:4px solid #2563eb;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  .metric.bad{border-left-color:#b91c1c}
  .metric.warn{border-left-color:#d97706}
  .metric.ok{border-left-color:#15803d}
  .metric .v{font-size:24px;font-weight:600;color:#0f172a}
  .metric .k{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
  .metric .extra{color:#475569;font-size:11px;margin-top:4px}
  table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
  th{background:#f1f5f9;font-weight:600}
  tr:hover{background:#fafafa}
  .ok{color:#15803d;font-weight:600}
  .ko{color:#b91c1c;font-weight:600}
  .err-bad{background:#fee2e2}
  .err-warn{background:#fef3c7}
  .err-good{background:#dcfce7}
  .swatch{display:inline-block;width:16px;height:16px;border:1px solid #cbd5e1;vertical-align:middle;margin-right:6px;border-radius:3px}
  .bar{display:inline-block;height:14px;background:#3b82f6;border-radius:2px;vertical-align:middle}
  .filename{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#475569}
  .nav{margin:12px 0 24px}
  .nav a{display:inline-block;background:#1e40af;color:#fff;padding:6px 12px;border-radius:4px;text-decoration:none;margin-right:8px;font-size:13px}
  .nav a:hover{background:#1e3a8a}
</style></head><body>
"""

HTML_FOOT = "</body></html>"


def render_html(eval_data, args, out_path):
    results = eval_data.get("results", [])
    n = len(results)

    h_errs = [r for r in results if isinstance(r.get("lateral_height_error_rel_pct"), (int, float))]
    s_errs = [r for r in results if isinstance(r.get("surface_error_rel_pct"), (int, float))]
    color_universe = [r for r in results if r.get("color_code_gt") is not None]

    n_h_bad = sum(1 for r in h_errs if abs(r["lateral_height_error_rel_pct"]) > args.th_height)
    n_s_bad = sum(1 for r in s_errs if abs(r["surface_error_rel_pct"]) > args.th_surface)
    n_c_cen_bad = sum(1 for r in color_universe if r.get("color_match_cenital") is False)
    n_c_lat_bad = sum(1 for r in color_universe if r.get("color_match_lateral") is False)
    n_c_cons_bad = sum(1 for r in color_universe if r.get("color_consensus_ok") is False)
    n_c_any_bad = sum(1 for r in color_universe if _has_color_mismatch(r))

    pct_h_bad = (n_h_bad / len(h_errs) * 100.0) if h_errs else 0.0
    pct_s_bad = (n_s_bad / len(s_errs) * 100.0) if s_errs else 0.0
    pct_c_cen = (n_c_cen_bad / len(color_universe) * 100.0) if color_universe else 0.0
    pct_c_lat = (n_c_lat_bad / len(color_universe) * 100.0) if color_universe else 0.0
    pct_c_cons = (n_c_cons_bad / len(color_universe) * 100.0) if color_universe else 0.0

    h_abs = [abs(r["lateral_height_error_rel_pct"]) for r in h_errs]
    s_abs = [abs(r["surface_error_rel_pct"]) for r in s_errs]
    h_stats = _stats_block(h_abs)
    s_stats = _stats_block(s_abs)

    h_sorted = sorted(h_errs, key=lambda r: abs(r["lateral_height_error_rel_pct"]), reverse=True)[:args.top_n]
    s_sorted = sorted(s_errs, key=lambda r: abs(r["surface_error_rel_pct"]), reverse=True)[:args.top_n]

    color_cen_bad = [r for r in color_universe if r.get("color_match_cenital") is False]
    color_lat_bad = [r for r in color_universe if r.get("color_match_lateral") is False]
    color_cons_bad = [r for r in color_universe if r.get("color_consensus_ok") is False]

    bins = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 75), (75, 100), (100, float("inf"))]
    h_hist = histogram(h_abs, bins)
    s_hist = histogram(s_abs, bins)

    pose_buckets = defaultdict(list)
    for r in results:
        key = (r.get("ref_gt"), r.get("pose_index_gt"), r.get("face_class_gt"))
        pose_buckets[key].append(r)
    pose_rows = []
    for (ref, pose, face), rows in pose_buckets.items():
        h_vals = [abs(x["lateral_height_error_rel_pct"]) for x in rows
                  if isinstance(x.get("lateral_height_error_rel_pct"), (int, float))]
        s_vals = [abs(x["surface_error_rel_pct"]) for x in rows
                  if isinstance(x.get("surface_error_rel_pct"), (int, float))]
        c_vals = [1 for x in rows if _has_color_mismatch(x)]
        h_mean = statistics.mean(h_vals) if h_vals else 0.0
        s_mean = statistics.mean(s_vals) if s_vals else 0.0
        combined = h_mean + s_mean
        pose_rows.append({
            "ref": ref, "pose": pose, "face": face, "n": len(rows),
            "h_mean": round(h_mean, 2),
            "s_mean": round(s_mean, 2),
            "color_bad": len(c_vals),
            "combined": round(combined, 2),
        })
    pose_rows = sorted(pose_rows, key=lambda x: x["combined"], reverse=True)[:args.top_n]

    out = [HTML_HEAD]
    out.append("<h1>Inferencia 300 - Reporte de errores</h1>")
    out.append(f"<p class='sub'>Total muestras: <b>{n}</b> &middot; "
               f"Umbrales: altura &gt; {args.th_height}% &middot; superficie &gt; {args.th_surface}% &middot; "
               f"Top-N: {args.top_n}</p>")
    out.append("<div class='nav'>")
    out.append("<a href='#height'>Altura</a><a href='#surface'>Superficie</a>"
               "<a href='#color-cen'>Color cenital</a><a href='#color-lat'>Color lateral</a>"
               "<a href='#color-cons'>Color consenso</a><a href='#poses'>Peores poses</a>"
               "<a href='#hist'>Histogramas</a>")
    out.append("</div>")

    out.append("<div class='grid'>")
    cls_h = "bad" if pct_h_bad > 30 else ("warn" if pct_h_bad > 10 else "ok")
    cls_s = "bad" if pct_s_bad > 30 else ("warn" if pct_s_bad > 10 else "ok")
    cls_c = "bad" if pct_c_cons > 15 else ("warn" if pct_c_cons > 5 else "ok")
    out.append(f"<div class='metric {cls_h}'><div class='k'>Errores altura &gt; {args.th_height}%</div>"
               f"<div class='v'>{n_h_bad} / {len(h_errs)}</div>"
               f"<div class='extra'>{pct_h_bad:.1f}% &middot; mean={_fmt(h_stats['mean']) if h_stats else '-'} "
               f"med={_fmt(h_stats['median']) if h_stats else '-'} "
               f"max={_fmt(h_stats['max']) if h_stats else '-'}</div></div>")
    out.append(f"<div class='metric {cls_s}'><div class='k'>Errores superficie &gt; {args.th_surface}%</div>"
               f"<div class='v'>{n_s_bad} / {len(s_errs)}</div>"
               f"<div class='extra'>{pct_s_bad:.1f}% &middot; mean={_fmt(s_stats['mean']) if s_stats else '-'} "
               f"med={_fmt(s_stats['median']) if s_stats else '-'} "
               f"max={_fmt(s_stats['max']) if s_stats else '-'}</div></div>")
    out.append(f"<div class='metric warn'><div class='k'>Mismatch color cenital</div>"
               f"<div class='v'>{n_c_cen_bad} / {len(color_universe)}</div>"
               f"<div class='extra'>{pct_c_cen:.1f}%</div></div>")
    out.append(f"<div class='metric warn'><div class='k'>Mismatch color lateral</div>"
               f"<div class='v'>{n_c_lat_bad} / {len(color_universe)}</div>"
               f"<div class='extra'>{pct_c_lat:.1f}%</div></div>")
    out.append(f"<div class='metric {cls_c}'><div class='k'>Fallo consenso color</div>"
               f"<div class='v'>{n_c_cons_bad} / {len(color_universe)}</div>"
               f"<div class='extra'>{pct_c_cons:.1f}%</div></div>")
    out.append(f"<div class='metric'><div class='k'>Cualquier mismatch color</div>"
               f"<div class='v'>{n_c_any_bad} / {len(color_universe)}</div>"
               f"<div class='extra'>cen | lat | consenso</div></div>")
    out.append("</div>")

    # ── Tabla altura ──
    out.append("<div class='card' id='height'><h2>Top-N peores estimaciones de altura</h2>")
    out.append("<table><thead><tr><th>idx</th><th>ref GT</th><th>pose</th><th>face</th>"
               "<th>h meas (mm)</th><th>h db (mm)</th><th>err %</th>"
               "<th>model match</th><th>conf lat</th><th>archivos</th></tr></thead><tbody>")
    for r in h_sorted:
        e = r.get("lateral_height_error_rel_pct")
        mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='ko'>NO</span>"
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}</td><td>{r.get('face_class_gt')}</td>"
            f"<td>{_fmt(r.get('lateral_height_meas_mm'))}</td>"
            f"<td>{_fmt(r.get('lateral_height_db_mm'))}</td>"
            f"<td class='{_err_class(e)}'>{_fmt(e)}</td>"
            f"<td>{mm}</td><td>{_fmt(r.get('yolo_conf_lateral'))}</td>"
            f"<td>{_file_cell(r)}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Tabla superficie ──
    out.append("<div class='card' id='surface'><h2>Top-N peores estimaciones de superficie</h2>")
    out.append("<table><thead><tr><th>idx</th><th>ref GT</th><th>pose</th><th>face</th>"
               "<th>obs apparent (mm²)</th><th>obs footprint (mm²)</th><th>db silhouette (mm²)</th>"
               "<th>err %</th><th>model match</th><th>conf cen</th><th>archivos</th></tr></thead><tbody>")
    for r in s_sorted:
        e = r.get("surface_error_rel_pct")
        mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='ko'>NO</span>"
        out.append(
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}</td><td>{r.get('face_class_gt')}</td>"
            f"<td>{_fmt(r.get('surface_obs_apparent_mm2'))}</td>"
            f"<td>{_fmt(r.get('surface_obs_footprint_mm2'))}</td>"
            f"<td>{_fmt(r.get('surface_db_silhouette_mm2'))}</td>"
            f"<td class='{_err_class(e)}'>{_fmt(e)}</td>"
            f"<td>{mm}</td><td>{_fmt(r.get('yolo_conf_cenital'))}</td>"
            f"<td>{_file_cell(r)}</td></tr>"
        )
    out.append("</tbody></table></div>")

    def _color_row(r):
        gt_hex = r.get("color_hex_gt")
        cen_hex = _rgb_to_hex(r.get("color_cenital_rgb_est"))
        lat_hex = _rgb_to_hex(r.get("color_lateral_rgb_est"))
        cm_c = "<span class='ok'>OK</span>" if r.get("color_match_cenital") else "<span class='ko'>NO</span>"
        cm_l = "<span class='ok'>OK</span>" if r.get("color_match_lateral") else "<span class='ko'>NO</span>"
        cons = r.get("color_consensus_status") or ""
        cons_cls = "ok" if r.get("color_consensus_ok") else "ko"
        return (
            f"<tr><td>{r.get('index')}</td><td>{r.get('ref_gt')}</td>"
            f"<td>{r.get('pose_index_gt')}/{r.get('face_class_gt')}</td>"
            f"<td>{_swatch(gt_hex)}{r.get('color_code_gt')} ({r.get('color_name_gt')}) "
            f"<span class='filename'>{gt_hex or ''}</span></td>"
            f"<td>{_swatch(cen_hex)}{r.get('color_cenital_normalized_code')} "
            f"({r.get('color_cenital_normalized_name')}) "
            f"<span class='filename'>{cen_hex or ''} rgb={r.get('color_cenital_rgb_est')}</span></td>"
            f"<td>{cm_c}</td>"
            f"<td>{_swatch(lat_hex)}{r.get('color_lateral_normalized_code')} "
            f"({r.get('color_lateral_normalized_name')}) "
            f"<span class='filename'>{lat_hex or ''} rgb={r.get('color_lateral_rgb_est')}</span></td>"
            f"<td>{cm_l}</td>"
            f"<td><span class='{cons_cls}'>{cons}</span></td>"
            f"<td>{_file_cell(r)}</td></tr>"
        )

    color_hdr = ("<table><thead><tr><th>idx</th><th>ref GT</th><th>pose/face</th>"
                 "<th>color GT (BD)</th><th>cen normalizado</th><th>match cen</th>"
                 "<th>lat normalizado</th><th>match lat</th><th>consenso</th>"
                 "<th>archivos</th></tr></thead><tbody>")

    # ── Mismatch color cenital ──
    out.append(f"<div class='card' id='color-cen'><h2>Mismatches color cenital "
               f"({len(color_cen_bad)} muestras)</h2>")
    out.append(color_hdr)
    for r in color_cen_bad[:args.top_n * 3]:
        out.append(_color_row(r))
    out.append("</tbody></table></div>")

    # ── Mismatch color lateral ──
    out.append(f"<div class='card' id='color-lat'><h2>Mismatches color lateral "
               f"({len(color_lat_bad)} muestras)</h2>")
    out.append(color_hdr)
    for r in color_lat_bad[:args.top_n * 3]:
        out.append(_color_row(r))
    out.append("</tbody></table></div>")

    # ── Fallo consenso ──
    out.append(f"<div class='card' id='color-cons'><h2>Fallo de consenso de color "
               f"({len(color_cons_bad)} muestras)</h2>"
               "<p class='sub'>El color final decidido por la fusion cenital+lateral "
               "no coincide con el GT, o las dos vistas discrepan.</p>")
    out.append(color_hdr)
    for r in color_cons_bad[:args.top_n * 3]:
        out.append(_color_row(r))
    out.append("</tbody></table></div>")

    # ── Top peores poses ──
    out.append("<div class='card' id='poses'><h2>Top peores combinaciones (ref, pose, face)</h2>"
               "<p class='sub'>Ordenado por suma de errores medios |altura|+|superficie|.</p>"
               "<table><thead><tr><th>ref</th><th>pose</th><th>face</th><th>n</th>"
               "<th>err alt mean</th><th>err sup mean</th><th>color mismatches</th>"
               "<th>combined</th></tr></thead><tbody>")
    for r in pose_rows:
        out.append(
            f"<tr><td>{r['ref']}</td><td>{r['pose']}</td><td>{r['face']}</td>"
            f"<td>{r['n']}</td>"
            f"<td class='{_err_class(r['h_mean'])}'>{r['h_mean']}</td>"
            f"<td class='{_err_class(r['s_mean'])}'>{r['s_mean']}</td>"
            f"<td>{r['color_bad']}</td>"
            f"<td>{r['combined']}</td></tr>"
        )
    out.append("</tbody></table></div>")

    # ── Histogramas ──
    out.append("<div class='card' id='hist'><h2>Distribucion de errores</h2>")
    out.append("<table><thead><tr><th>bin</th><th>n altura</th><th></th>"
               "<th>n superficie</th><th></th></tr></thead><tbody>")
    max_h = max((c for _, c in h_hist), default=1)
    max_s = max((c for _, c in s_hist), default=1)
    for (lh, ch), (ls, cs) in zip(h_hist, s_hist):
        out.append(
            f"<tr><td class='mono'>{lh}</td>"
            f"<td>{ch}</td><td>{_bar_html(ch, max_h)}</td>"
            f"<td>{cs}</td><td>{_bar_html(cs, max_s)}</td></tr>"
        )
    out.append("</tbody></table></div>")

    out.append(HTML_FOOT)

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("".join(out))


def render_markdown(eval_data, args, out_path,
                    n_h_bad, n_s_bad, n_c_cen_bad, n_c_lat_bad, n_c_cons_bad,
                    h_top, s_top, c_top):
    results = eval_data.get("results", [])
    n = len(results)
    lines = []
    lines.append(f"# Inferencia 300 — Reporte de errores")
    lines.append("")
    lines.append(f"- Total muestras: **{n}**")
    lines.append(f"- Umbrales: altura > **{args.th_height}%**, superficie > **{args.th_surface}%**")
    lines.append("")
    lines.append("## Resumen")
    lines.append(f"- Errores de altura > {args.th_height}%: **{n_h_bad}**")
    lines.append(f"- Errores de superficie > {args.th_surface}%: **{n_s_bad}**")
    lines.append(f"- Mismatches color cenital: **{n_c_cen_bad}**")
    lines.append(f"- Mismatches color lateral: **{n_c_lat_bad}**")
    lines.append(f"- Fallo consenso color: **{n_c_cons_bad}**")
    lines.append("")

    lines.append("## Top 10 peores estimaciones de altura")
    lines.append("| idx | ref | pose | face | h_meas (mm) | h_db (mm) | err % |")
    lines.append("|-----|-----|------|------|-------------|-----------|-------|")
    for r in h_top[:10]:
        lines.append(f"| {r.get('index')} | {r.get('ref_gt')} | {r.get('pose_index_gt')} | "
                     f"{r.get('face_class_gt')} | {_fmt(r.get('lateral_height_meas_mm'))} | "
                     f"{_fmt(r.get('lateral_height_db_mm'))} | "
                     f"{_fmt(r.get('lateral_height_error_rel_pct'))} |")
    lines.append("")

    lines.append("## Top 10 peores estimaciones de superficie")
    lines.append("| idx | ref | pose | face | obs footprint | db silhouette | err % |")
    lines.append("|-----|-----|------|------|---------------|---------------|-------|")
    for r in s_top[:10]:
        lines.append(f"| {r.get('index')} | {r.get('ref_gt')} | {r.get('pose_index_gt')} | "
                     f"{r.get('face_class_gt')} | {_fmt(r.get('surface_obs_footprint_mm2'))} | "
                     f"{_fmt(r.get('surface_db_silhouette_mm2'))} | "
                     f"{_fmt(r.get('surface_error_rel_pct'))} |")
    lines.append("")

    lines.append("## Mismatches de color (top 15, cualquier categoría)")
    lines.append("| idx | ref | color GT | cen norm | match cen | lat norm | match lat | consenso |")
    lines.append("|-----|-----|----------|----------|-----------|----------|-----------|----------|")
    for r in c_top[:15]:
        lines.append(
            f"| {r.get('index')} | {r.get('ref_gt')} | "
            f"{r.get('color_code_gt')} ({r.get('color_name_gt')}) | "
            f"{r.get('color_cenital_normalized_code')} ({r.get('color_cenital_normalized_name')}) | "
            f"{'OK' if r.get('color_match_cenital') else 'NO'} | "
            f"{r.get('color_lateral_normalized_code')} ({r.get('color_lateral_normalized_name')}) | "
            f"{'OK' if r.get('color_match_lateral') else 'NO'} | "
            f"{r.get('color_consensus_status')} |"
        )
    lines.append("")

    lines.append("## Salidas")
    lines.append("- `inference_300_errors.html` — vista web completa")
    lines.append("- `inference_300_errors_height.csv` — todas las muestras con error de altura > umbral")
    lines.append("- `inference_300_errors_surface.csv` — todas las muestras con error de superficie > umbral")
    lines.append("- `inference_300_errors_color.csv` — todas las muestras con cualquier mismatch de color")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Reporte focalizado en errores de inferencia 300")
    parser.add_argument("--eval", required=True, help="Ruta al inference_300_eval.json")
    parser.add_argument("--out", required=True, help="Directorio de salida")
    parser.add_argument("--top-n", type=int, default=30, help="Filas Top-N en cada tabla")
    parser.add_argument("--th-height", type=float, default=20.0,
                        help="Umbral en %% para considerar error de altura significativo")
    parser.add_argument("--th-surface", type=float, default=20.0,
                        help="Umbral en %% para considerar error de superficie significativo")
    args = parser.parse_args()

    if not os.path.isfile(args.eval):
        print(f"[ERROR] No se encuentra {args.eval}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    with open(args.eval, "r", encoding="utf-8") as fp:
        eval_data = json.load(fp)
    results = eval_data.get("results", [])
    print(f"[errors-report] Cargadas {len(results)} muestras")

    # Filtrar y ordenar
    h_bad = sorted(
        [r for r in results
         if isinstance(r.get("lateral_height_error_rel_pct"), (int, float))
         and abs(r["lateral_height_error_rel_pct"]) > args.th_height],
        key=lambda r: abs(r["lateral_height_error_rel_pct"]), reverse=True,
    )
    s_bad = sorted(
        [r for r in results
         if isinstance(r.get("surface_error_rel_pct"), (int, float))
         and abs(r["surface_error_rel_pct"]) > args.th_surface],
        key=lambda r: abs(r["surface_error_rel_pct"]), reverse=True,
    )
    color_universe = [r for r in results if r.get("color_code_gt") is not None]
    c_bad = [r for r in color_universe if _has_color_mismatch(r)]

    csv_h = os.path.join(args.out, "inference_300_errors_height.csv")
    csv_s = os.path.join(args.out, "inference_300_errors_surface.csv")
    csv_c = os.path.join(args.out, "inference_300_errors_color.csv")
    write_csv(h_bad, HEIGHT_CSV_COLS, csv_h)
    write_csv(s_bad, SURFACE_CSV_COLS, csv_s)
    write_csv(c_bad, COLOR_CSV_COLS, csv_c)
    print(f"[errors-report] CSV altura   : {csv_h} ({len(h_bad)} filas)")
    print(f"[errors-report] CSV superficie: {csv_s} ({len(s_bad)} filas)")
    print(f"[errors-report] CSV color    : {csv_c} ({len(c_bad)} filas)")

    html_path = os.path.join(args.out, "inference_300_errors.html")
    render_html(eval_data, args, html_path)
    print(f"[errors-report] HTML        : {html_path}")

    md_path = os.path.join(args.out, "inference_300_errors_summary.md")
    n_c_cen_bad = sum(1 for r in color_universe if r.get("color_match_cenital") is False)
    n_c_lat_bad = sum(1 for r in color_universe if r.get("color_match_lateral") is False)
    n_c_cons_bad = sum(1 for r in color_universe if r.get("color_consensus_ok") is False)
    render_markdown(eval_data, args, md_path,
                    len(h_bad), len(s_bad), n_c_cen_bad, n_c_lat_bad, n_c_cons_bad,
                    h_bad, s_bad, c_bad)
    print(f"[errors-report] Markdown    : {md_path}")


if __name__ == "__main__":
    main()
