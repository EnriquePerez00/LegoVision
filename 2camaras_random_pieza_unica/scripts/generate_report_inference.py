# -*- coding: utf-8 -*-
"""generate_report_inference.py
Report unificado de diagnostico de inferencia LEGO.
Reemplaza: generate_inference_300_report.py  y  generate_inference_300_errors_report.py

Secciones HTML navegables:
  0. Resumen Ejecutivo  | 1. COLOR | 2. SUPERFICIE CENITAL
  3. ALTURA LATERAL     | 4. INFERENCIA AGREGADA | 5. TENDENCIAS Y PATRONES

Salidas por --out:
  inference_report.html, inference_full.csv, inference_per_piece.csv,
  inference_per_pose.csv, inference_errors_color.csv,
  inference_errors_surface.csv, inference_errors_height.csv

Uso:
  python generate_report_inference.py \\
      --eval       data/test_500allhd/eval_report.json \\
      --images_dir data/test_500allhd/ \\
      --out        data/reports/inference_report/ \\
      [--th_surface 10.0] [--th_height 10.0] [--top_n 50]
"""
from __future__ import annotations
import argparse, csv, json, os, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(v):
    if v is None: return ""
    if isinstance(v, (list, tuple)): return ";".join(str(x) for x in v)
    if isinstance(v, bool): return "True" if v else "False"
    return str(v)

def _f(v, d=2):
    if v is None or v == "": return "\u2014"
    try:
        if isinstance(v, float): return f"{v:.{d}f}"
    except Exception: pass
    return str(v)

def _pct(n, d, dec=1): return round(n / d * 100, dec) if d else 0.0

def _ec(v, w=10.0, b=30.0):
    if v is None: return ""
    a = abs(float(v))
    return "bad" if a >= b else ("warn" if a >= w else "ok")

def _rc(v, w=10.0, b=30.0):
    c = _ec(v, w, b); return f"row-{c}" if c else ""

def _sw(h):
    if not h: return ""
    hx = str(h) if str(h).startswith("#") else "#" + str(h)
    return '<span class="sw" style="background:' + hx + '"></span>'

def _rgb2hex(rgb):
    if not rgb: return None
    try:
        if isinstance(rgb, str):
            p = rgb.replace(";", ",").split(",")
            r, g, b = [max(0, min(255, int(round(float(x))))) for x in p[:3]]
        elif isinstance(rgb, (list, tuple)):
            r, g, b = [max(0, min(255, int(round(float(x))))) for x in list(rgb)[:3]]
        else: return None
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception: return None

def _rdsp(rgb):
    h = _rgb2hex(rgb)
    return _sw(h) + '<span class="mono">' + h + "</span>" if h else "\u2014"

def _stats(vals):
    sv = sorted([float(v) for v in vals if v is not None])
    if not sv: return {}
    n = len(sv)
    return dict(n=n, mean=statistics.mean(sv), median=statistics.median(sv),
                std=statistics.stdev(sv) if n > 1 else 0.0, min=sv[0], max=sv[-1],
                p10=sv[max(0, int(n*.10)-1)], p25=sv[max(0, int(n*.25)-1)],
                p75=sv[min(n-1, int(n*.75))], p90=sv[min(n-1, int(n*.90))])

BINS = [(0,5),(5,10),(10,20),(20,35),(35,50),(50,100),(100,float("inf"))]

def _hist(vals):
    counts = [0]*len(BINS)
    for v in vals:
        if v is None: continue
        a = abs(float(v))
        for i,(lo,hi) in enumerate(BINS):
            if lo<=a<hi or(hi==float("inf") and a>=lo): counts[i]+=1; break
    return [(f">{lo:g}%" if hi==float("inf") else f"{lo:g}\u2013{hi:g}%", c)
            for(lo,hi),c in zip(BINS,counts)]

def _thumb(fname, imgdir):
    if not fname or not imgdir: return ""
    full = os.path.join(imgdir, fname) if not os.path.isabs(fname) else fname
    if not os.path.isfile(full): return ""
    esc = full.replace('"', "&quot;")
    return '<img class="thumb" loading="lazy" src="' + esc + '" onerror="this.style.display=\'none\'" title="' + fname + '"/>'

def _kpi(label, val, extra="", cls=""):
    e = "<div class=\'extra\'>" + extra + "</div>" if extra else ""
    return "<div class=\'kpi " + cls + "\'><div class=\'v\'>" + str(val) + "</div><div class=\'k\'>" + label + "</div>" + e + "</div>"

def _badge(t, cls=""): return "<span class=\'badge " + cls + "\'>" + t + "</span>"

def _hist_html(data, maxc):
    if not maxc: maxc = 1
    rows = []
    for label, cnt in data:
        w = max(2, int(cnt/maxc*220))
        cls = "b" if "50" in label or "100" in label else("w" if "20" in label or "35" in label else "")
        rows.append("<div class=\'hr\'><span class=\'hl\'>" + label + "</span>"
                    "<div class=\'hb " + cls + "\' style=\'width:" + str(w) + "px\'></div>"
                    "<span class=\'hv\'>" + str(cnt) + "</span></div>")
    return "\n".join(rows)

def _kpi_cls(pct): return "ok" if pct>=90 else("warn" if pct>=60 else "bad")
def _err_kpi_cls(mp): return "ok" if mp<=10 else("warn" if mp<=25 else "bad")


# ---------------------------------------------------------------------------
# CSS  (single-line string to avoid multiline quoting issues)
# ---------------------------------------------------------------------------
CSS = (
    "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}"
    ":root{--bg:#f0f4f8;--card:#fff;--br:#dde3ec;--ok:#15803d;--wn:#b45309;--bd:#b91c1c;"
    "--okb:#dcfce7;--wnb:#fef9c3;--bdb:#fee2e2;--ac:#1e40af;--tx:#0f172a;--mt:#64748b;}"
    "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);"
    "color:var(--tx);padding:0 0 80px;line-height:1.5;font-size:13px}"
    "#nav{position:sticky;top:0;z-index:100;background:#1e293b;padding:7px 20px;"
    "display:flex;gap:9px;align-items:center;flex-wrap:wrap;box-shadow:0 2px 8px rgba(0,0,0,.4)}"
    ".brand{font-weight:700;font-size:13px;color:#93c5fd;margin-right:10px}"
    "#nav a{color:#94a3b8;text-decoration:none;font-size:11px;padding:3px 8px;"
    "border-radius:4px;transition:background .15s;white-space:nowrap}"
    "#nav a:hover,#nav a.cur{background:#334155;color:#fff}"
    ".page{max-width:1440px;margin:0 auto;padding:0 20px}"
    ".sec{margin-top:24px;scroll-margin-top:44px}"
    "h1{font-size:20px;font-weight:700;margin-bottom:4px}"
    "h2{font-size:15px;font-weight:700;margin:18px 0 8px;color:var(--ac);"
    "border-bottom:2px solid var(--br);padding-bottom:5px;scroll-margin-top:50px}"
    "h3{font-size:13px;font-weight:600;margin:12px 0 5px;color:#334155}"
    ".sub{color:var(--mt);font-size:12px;margin-bottom:10px}"
    ".card{background:var(--card);border-radius:8px;padding:14px 18px;margin-bottom:12px;"
    "box-shadow:0 1px 4px rgba(0,0,0,.08);border:1px solid var(--br)}"
    ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:9px;margin-bottom:14px}"
    ".kpi{background:var(--card);border-radius:8px;padding:11px 14px;"
    "border-left:5px solid var(--ac);box-shadow:0 1px 4px rgba(0,0,0,.08)}"
    ".kpi.ok{border-left-color:var(--ok)}.kpi.warn{border-left-color:var(--wn)}.kpi.bad{border-left-color:var(--bd)}"
    ".kpi .v{font-size:22px;font-weight:700;line-height:1.1}"
    ".kpi .k{color:var(--mt);font-size:10px;text-transform:uppercase;letter-spacing:.06em;margin-top:3px}"
    ".kpi .extra{color:var(--mt);font-size:10px;margin-top:2px}"
    ".tbl{overflow-x:auto;margin:5px 0;max-height:560px;overflow-y:auto}.tbl.tall{max-height:900px}"
    "table{border-collapse:collapse;width:100%;font-size:11px;min-width:460px}"
    "th,td{padding:5px 8px;border-bottom:1px solid var(--br);text-align:left;vertical-align:top}"
    "th{background:#f8fafc;font-weight:600;color:#334155;position:sticky;top:0;z-index:2;white-space:nowrap}"
    "tr:hover td{background:#f8fafc}"
    ".ok{color:var(--ok);font-weight:600}.warn{color:var(--wn);font-weight:600}.bad{color:var(--bd);font-weight:600}"
    "tr.row-ok td{background:#f0fdf4}tr.row-warn td{background:#fffdf0}tr.row-bad td{background:#fff5f5}"
    "tr.row-hard td{background:#fdf4ff}"
    ".mono{font-family:ui-monospace,Menlo,monospace;font-size:10px}"
    ".num{text-align:right;font-family:ui-monospace,monospace}.nw{white-space:nowrap}"
    ".sw{display:inline-block;width:13px;height:13px;border:1px solid #aaa;"
    "vertical-align:middle;border-radius:2px;margin-right:3px}"
    ".hr{display:flex;align-items:center;gap:6px;margin:2px 0}"
    ".hb{height:11px;border-radius:2px;min-width:2px;background:#3b82f6}"
    ".hb.w{background:var(--wn)}.hb.b{background:var(--bd)}"
    ".hl{font-size:11px;color:var(--mt);min-width:82px;white-space:nowrap}"
    ".hv{font-size:11px;font-weight:600;min-width:35px}"
    "img.thumb{display:inline-block;width:90px;height:auto;border:1px solid #ddd;"
    "border-radius:3px;cursor:zoom-in;transition:transform .2s;vertical-align:top}"
    "img.thumb:hover{transform:scale(3);z-index:50;position:relative;"
    "box-shadow:0 4px 20px rgba(0,0,0,.5);border-radius:4px}"
    ".ip{display:flex;gap:3px}"
    ".diag{background:#fff7ed;border-left:4px solid #ea580c;padding:10px 14px;"
    "border-radius:6px;margin-top:9px;font-size:12px}"
    ".diag strong{color:#9a3412}.diag ul{margin:5px 0 0 14px;line-height:1.9}"
    ".diag.g{background:#f0fdf4;border-left-color:var(--ok)}.diag.g strong{color:var(--ok)}"
    ".badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;"
    "font-weight:600;white-space:nowrap}"
    ".badge.ok{background:var(--okb);color:var(--ok)}.badge.bad{background:var(--bdb);color:var(--bd)}"
    ".badge.warn{background:var(--wnb);color:var(--wn)}.badge.hard{background:#f3e8ff;color:#7c3aed}"
    ".badge.info{background:#eff6ff;color:#1d4ed8}"
    "code{background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:10px;font-family:monospace}"
    ".alert{background:#fef9c3;border:1px solid #fde047;border-radius:5px;"
    "padding:8px 12px;font-size:12px;margin-bottom:8px}"
    ".two{display:grid;grid-template-columns:1fr 1fr;gap:14px}"
    "@media(max-width:800px){.two{grid-template-columns:1fr}}"
)

JS = (
    "<script>(function(){"
    "const secs=document.querySelectorAll('.sec[id]');"
    "const links=document.querySelectorAll('#nav a[href]');"
    "function upd(){let cur='';"
    "secs.forEach(s=>{if(s.getBoundingClientRect().top<=80)cur=s.id;});"
    "links.forEach(a=>{const h=a.getAttribute('href');"
    "a.classList.toggle('cur',h==='#'+cur);});}"
    "window.addEventListener('scroll',upd,{passive:true});upd();"
    "})();</script>"
)

# ---------------------------------------------------------------------------
# CSV column lists
# ---------------------------------------------------------------------------
FULL_COLS = [
    "index","sample_index","cenital_file","lateral_file",
    "ref_gt","pose_index_gt","face_class_gt",
    "ref_inferred","model_match","consensus_score",
    "color_code_gt","color_name_gt","color_hex_gt",
    "color_cenital_rgb_est","color_cenital_normalized_code","color_cenital_normalized_name",
    "color_lateral_rgb_est","color_lateral_normalized_code","color_lateral_normalized_name",
    "color_match_cenital","color_match_lateral","color_decision_used",
    "color_consensus_status","color_consensus_ok",
    "surface_obs_apparent_mm2","surface_obs_footprint_mm2",
    "surface_db_silhouette_mm2","surface_error_rel_pct",
    "lateral_height_meas_mm","lateral_height_db_mm","lateral_height_error_rel_pct",
    "effective_height_db_mm","yolo_conf_cenital","yolo_conf_lateral",
    "valid_by_color_count","valid_by_surface_count","valid_by_height_count",
]
COLOR_COLS = [
    "index","sample_index","ref_gt","pose_index_gt","face_class_gt",
    "color_code_gt","color_name_gt","color_hex_gt",
    "color_cenital_rgb_est","color_cenital_normalized_code","color_cenital_normalized_name",
    "color_match_cenital",
    "color_lateral_rgb_est","color_lateral_normalized_code","color_lateral_normalized_name",
    "color_match_lateral","color_decision_used","color_consensus_status","color_consensus_ok",
    "ref_inferred","model_match","cenital_file","lateral_file",
]
SURF_COLS = [
    "index","sample_index","ref_gt","pose_index_gt","face_class_gt",
    "ref_inferred","model_match",
    "surface_obs_apparent_mm2","surface_obs_footprint_mm2",
    "surface_db_silhouette_mm2","surface_error_rel_pct",
    "yolo_conf_cenital","cenital_file","lateral_file",
]
HEIGHT_COLS = [
    "index","sample_index","ref_gt","pose_index_gt","face_class_gt",
    "ref_inferred","model_match",
    "lateral_height_meas_mm","lateral_height_db_mm","lateral_height_error_rel_pct",
    "effective_height_db_mm","yolo_conf_lateral","cenital_file","lateral_file",
]
PIECE_COLS = [
    "ref","color_code","n_samples","accuracy_pct",
    "color_cen_match_pct","color_lat_match_pct",
    "surf_err_mean","surf_err_median","h_err_mean","h_err_median",
]
POSE_COLS = [
    "ref","pose_index","face_class","n_samples","accuracy_pct",
    "surf_err_mean","h_err_mean",
]


def _write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _s(r.get(k)) for k in cols})


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def _aggregate(results, th_s, th_h):
    n = len(results)
    correct = sum(1 for r in results if r.get('model_match'))

    # ---- color ----
    color_u = [r for r in results if r.get('color_code_gt') is not None]
    n_cen_bad  = sum(1 for r in color_u if r.get('color_match_cenital') is False)
    n_lat_bad  = sum(1 for r in color_u if r.get('color_match_lateral') is False)
    n_cons_bad = sum(1 for r in color_u if r.get('color_consensus_ok') is False)
    n_dec_bad  = sum(1 for r in color_u if r.get('color_decision_used') != r.get('color_code_gt'))
    n_any_bad  = sum(1 for r in color_u if (
        r.get('color_match_cenital') is False or
        r.get('color_match_lateral') is False or
        r.get('color_consensus_ok') is False))

    cen_rgbs = defaultdict(list); lat_rgbs = defaultdict(list)
    gt_info = {}; cen_pred = defaultdict(Counter); lat_pred = defaultdict(Counter)
    for r in color_u:
        gt = r['color_code_gt']
        gt_info.setdefault(gt, (r.get('color_name_gt','?'), r.get('color_hex_gt','')))
        cen_pred[gt][r.get('color_cenital_normalized_code')] += 1
        lat_pred[gt][r.get('color_lateral_normalized_code')] += 1
        if r.get('color_cenital_rgb_est'): cen_rgbs[gt].append(r['color_cenital_rgb_est'])
        if r.get('color_lateral_rgb_est'): lat_rgbs[gt].append(r['color_lateral_rgb_est'])
    gt_colors = sorted(gt_info.keys(), key=lambda c:-sum(1 for r in color_u if r.get('color_code_gt')==c))

    # ---- surface ----
    s_vals = [r['surface_error_rel_pct'] for r in results if isinstance(r.get('surface_error_rel_pct'),(int,float))]
    s_abs  = [abs(float(v)) for v in s_vals]
    s_st   = _stats(s_abs)
    s_bad  = [r for r in results if isinstance(r.get('surface_error_rel_pct'),(int,float)) and abs(r['surface_error_rel_pct'])>th_s]

    # ---- height ----
    h_vals = [r['lateral_height_error_rel_pct'] for r in results if isinstance(r.get('lateral_height_error_rel_pct'),(int,float))]
    h_abs  = [abs(float(v)) for v in h_vals]
    h_st   = _stats(h_abs)
    h_bad  = [r for r in results if isinstance(r.get('lateral_height_error_rel_pct'),(int,float)) and abs(r['lateral_height_error_rel_pct'])>th_h]

    # ---- YOLO ----
    yolo_cen_det = sum(1 for r in results if (r.get('yolo_conf_cenital') or 0)>0)
    yolo_lat_det = sum(1 for r in results if (r.get('yolo_conf_lateral') or 0)>0)

    # ---- hard cases (all 3 gating phases maxed out) ----
    hard = [r for r in results
            if r.get('valid_by_color_count') == r.get('valid_by_surface_count') == r.get('valid_by_height_count')]

    # ---- per-piece aggregation ----
    piece_bkt = defaultdict(list)
    for r in results:
        piece_bkt[(r.get('ref_gt'), r.get('color_code_gt'))].append(r)
    per_piece = []
    for (ref,cc), rows in sorted(piece_bkt.items()):
        nn=len(rows); nc=sum(1 for x in rows if x.get('model_match'))
        se=[abs(x['surface_error_rel_pct']) for x in rows if isinstance(x.get('surface_error_rel_pct'),(int,float))]
        he=[abs(x['lateral_height_error_rel_pct']) for x in rows if isinstance(x.get('lateral_height_error_rel_pct'),(int,float))]
        per_piece.append({
            'ref':ref,'color_code':cc,'n_samples':nn,
            'accuracy_pct':_pct(nc,nn),
            'color_cen_match_pct':_pct(sum(1 for x in rows if x.get('color_match_cenital')),nn),
            'color_lat_match_pct':_pct(sum(1 for x in rows if x.get('color_match_lateral')),nn),
            'surf_err_mean':round(statistics.mean(se),2) if se else None,
            'surf_err_median':round(statistics.median(se),2) if se else None,
            'h_err_mean':round(statistics.mean(he),2) if he else None,
            'h_err_median':round(statistics.median(he),2) if he else None,
        })

    # ---- per-pose aggregation ----
    pose_bkt = defaultdict(list)
    for r in results:
        pose_bkt[(r.get('ref_gt'),r.get('pose_index_gt'),r.get('face_class_gt'))].append(r)
    per_pose = []
    for (ref,pi,fc), rows in sorted(pose_bkt.items(), key=lambda x:(str(x[0][0]),str(x[0][1]))):
        nn=len(rows); nc=sum(1 for x in rows if x.get('model_match'))
        se=[abs(x['surface_error_rel_pct']) for x in rows if isinstance(x.get('surface_error_rel_pct'),(int,float))]
        he=[abs(x['lateral_height_error_rel_pct']) for x in rows if isinstance(x.get('lateral_height_error_rel_pct'),(int,float))]
        per_pose.append({
            'ref':ref,'pose_index':pi,'face_class':fc,'n_samples':nn,
            'accuracy_pct':_pct(nc,nn),
            'surf_err_mean':round(statistics.mean(se),2) if se else None,
            'h_err_mean':round(statistics.mean(he),2) if he else None,
        })

    # ---- confusion matrix ----
    confusion = Counter()
    for r in results:
        if r.get('ref_gt') != r.get('ref_inferred'):
            confusion[(r.get('ref_gt'), r.get('ref_inferred'))] += 1
    confusion_top = [(gt,pr,c) for(gt,pr),c in sorted(confusion.items(),key=lambda x:-x[1])][:25]

    # ---- score gap analysis ----
    score_gap = sorted(
        [r for r in results if not r.get('model_match') and r.get('consensus_score') is not None],
        key=lambda r: r.get('consensus_score',0), reverse=True)[:20]

    return dict(
        n=n, correct=correct, accuracy=_pct(correct,n),
        color_u=color_u, n_cen_bad=n_cen_bad, n_lat_bad=n_lat_bad,
        n_cons_bad=n_cons_bad, n_dec_bad=n_dec_bad, n_any_bad=n_any_bad,
        gt_colors=gt_colors, gt_info=gt_info,
        cen_rgbs=cen_rgbs, lat_rgbs=lat_rgbs, cen_pred=cen_pred, lat_pred=lat_pred,
        s_vals=s_vals, s_abs=s_abs, s_st=s_st, s_bad=s_bad,
        h_vals=h_vals, h_abs=h_abs, h_st=h_st, h_bad=h_bad,
        yolo_cen_det=yolo_cen_det, yolo_lat_det=yolo_lat_det,
        hard=hard, per_piece=per_piece, per_pose=per_pose,
        confusion_top=confusion_top, score_gap=score_gap,
    )


# ---------------------------------------------------------------------------
# Section 0: Executive Summary
# ---------------------------------------------------------------------------

def sec_summary(d, th_s, th_h):
    acc = d['accuracy']
    nc = len(d['color_u']); nb = d['n_any_bad']
    s_mean = d['s_st'].get('mean') or 0
    h_mean = d['h_st'].get('mean') or 0
    yolo_r_c = _pct(d['yolo_cen_det'], d['n']); yolo_r_l = _pct(d['yolo_lat_det'], d['n'])

    kpis = [
        _kpi('Accuracy modelo', f"{acc:.1f}%",
             f"{d['correct']}/{d['n']}", _kpi_cls(acc)),
        _kpi('Color match cenital',
             f"{_pct(nc - d['n_cen_bad'], nc):.1f}%" if nc else 'N/A',
             f"{nc - d['n_cen_bad']}/{nc}",
             _kpi_cls(_pct(nc - d['n_cen_bad'], nc) if nc else 100)),
        _kpi('Color match lateral',
             f"{_pct(nc - d['n_lat_bad'], nc):.1f}%" if nc else 'N/A',
             f"{nc - d['n_lat_bad']}/{nc}",
             _kpi_cls(_pct(nc - d['n_lat_bad'], nc) if nc else 100)),
        _kpi('Error sup. medio |%|', _f(s_mean), f"umbral {th_s}%", _err_kpi_cls(s_mean)),
        _kpi('Error altura medio |%|', _f(h_mean), f"umbral {th_h}%", _err_kpi_cls(h_mean)),
        _kpi('YOLO cenital det.', f"{yolo_r_c:.1f}%",
             f"{d['yolo_cen_det']}/{d['n']}", _kpi_cls(yolo_r_c)),
        _kpi('YOLO lateral det.', f"{yolo_r_l:.1f}%",
             f"{d['yolo_lat_det']}/{d['n']}", _kpi_cls(yolo_r_l)),
        _kpi('Hard cases (sin filtrado)', str(len(d['hard'])),
             'color+sup+alt=max', 'warn' if d['hard'] else 'ok'),
    ]

    # Key findings alert
    alerts = []
    if acc < 30: alerts.append(f'Accuracy critica ({acc:.1f}%) — revisar pipeline completo')
    if d['n_any_bad'] > nc * 0.4 and nc: alerts.append(f'{d["n_any_bad"]} muestras ({_pct(d["n_any_bad"],nc):.0f}%) con algun error de color')
    if s_mean > 30: alerts.append(f'Error medio de superficie elevado ({s_mean:.1f}%) — revisar calibracion perspectiva')
    if h_mean > 30: alerts.append(f'Error medio de altura elevado ({h_mean:.1f}%) — revisar segmentacion SAM lateral')
    if d['hard']: alerts.append(f'{len(d["hard"])} muestras donde los 3 gatings no filtraron candidatos — DINOv2 opera sin contexto previo')

    alert_html = ''.join(f'<div class="alert">⚠️ {a}</div>' for a in alerts) if alerts else ''

    return (
        '<div class="sec" id="summary"><h2>0. Resumen Ejecutivo</h2>'
        + alert_html
        + '<div class="grid">' + ''.join(kpis) + '</div></div>'
    )


# ---------------------------------------------------------------------------
# Section 1: COLOR
# ---------------------------------------------------------------------------

def _color_row(r, imgdir):
    gt_hex = r.get("color_hex_gt", "")
    cm_c = "<span class='ok'>OK</span>" if r.get("color_match_cenital") else "<span class='bad'>NO</span>"
    cm_l = "<span class='ok'>OK</span>" if r.get("color_match_lateral") else "<span class='bad'>NO</span>"
    cons_cls = "ok" if r.get("color_consensus_ok") else "bad"
    mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='bad'>NO</span>"
    imgs = "<div class='ip'>" + _thumb(r.get("cenital_file",""), imgdir) + _thumb(r.get("lateral_file",""), imgdir) + "</div>"
    cls = "row-ok" if (r.get("color_match_cenital") and r.get("color_match_lateral")) else "row-bad"
    parts = [
        "<tr class='" + cls + "'>",
        "<td>" + str(r.get("index","")) + "</td>",
        "<td>" + imgs + "</td>",
        "<td class='nw'>" + str(r.get("ref_gt","")) + "</td>",
        "<td>" + str(r.get("pose_index_gt","")) + "</td>",
        "<td>" + str(r.get("face_class_gt","")) + "</td>",
        "<td>" + _sw(gt_hex) + "<span class='mono'>" + str(r.get("color_code_gt","")) + " " + str(r.get("color_name_gt","")) + "</span></td>",
        "<td>" + _rdsp(r.get("color_cenital_rgb_est")) + " &rarr; " + str(r.get("color_cenital_normalized_code","")) + " " + str(r.get("color_cenital_normalized_name","")) + "</td>",
        "<td>" + cm_c + "</td>",
        "<td>" + _rdsp(r.get("color_lateral_rgb_est")) + " &rarr; " + str(r.get("color_lateral_normalized_code","")) + " " + str(r.get("color_lateral_normalized_name","")) + "</td>",
        "<td>" + cm_l + "</td>",
        "<td><span class='" + cons_cls + "'>" + str(r.get("color_consensus_status","")) + "</span></td>",
        "<td>" + mm + " " + str(r.get("ref_inferred","")) + "</td>",
        "</tr>",
    ]
    return "".join(parts)


def sec_color(d, imgdir):
    color_u = d["color_u"]; gt_colors = d["gt_colors"]; gt_info = d["gt_info"]
    nc = len(color_u)
    if nc == 0:
        return '<div class="sec" id="color"><h2>1. COLOR</h2><p>Sin datos de color.</p></div>'

    kpis = [
        _kpi("Mismatch cenital",   f"{_pct(d['n_cen_bad'],nc):.1f}%",  str(d['n_cen_bad'])+"/"+str(nc),  _err_kpi_cls(_pct(d['n_cen_bad'],nc))),
        _kpi("Mismatch lateral",   f"{_pct(d['n_lat_bad'],nc):.1f}%",  str(d['n_lat_bad'])+"/"+str(nc),  _err_kpi_cls(_pct(d['n_lat_bad'],nc))),
        _kpi("Fallo consenso",     f"{_pct(d['n_cons_bad'],nc):.1f}%", str(d['n_cons_bad'])+"/"+str(nc), _err_kpi_cls(_pct(d['n_cons_bad'],nc))),
        _kpi("Decision != GT",     f"{_pct(d['n_dec_bad'],nc):.1f}%",  str(d['n_dec_bad'])+"/"+str(nc),  _err_kpi_cls(_pct(d['n_dec_bad'],nc))),
        _kpi("Cualquier mismatch", f"{_pct(d['n_any_bad'],nc):.1f}%",  str(d['n_any_bad'])+"/"+str(nc),  _err_kpi_cls(_pct(d['n_any_bad'],nc))),
    ]

    gt_rows = []
    for c in gt_colors:
        name, hex_ = gt_info[c]
        sub = [r for r in color_u if r.get("color_code_gt") == c]
        nn = len(sub)
        co = sum(1 for r in sub if r.get("color_match_cenital"))
        lo = sum(1 for r in sub if r.get("color_match_lateral"))
        do = sum(1 for r in sub if r.get("color_decision_used") == c)
        cp=_pct(co,nn); lp=_pct(lo,nn); dp=_pct(do,nn)
        gt_rows.append("<tr><td>"+c+"</td><td>"+_sw(hex_)+name+"</td><td><code>"+hex_+"</code></td><td>"+str(nn)+"</td>"
            "<td class='"+_ec(100-cp)+"'>"+f"{cp:.0f}%"+"</td>"
            "<td class='"+_ec(100-lp)+"'>"+f"{lp:.0f}%"+"</td>"
            "<td class='"+_ec(100-dp)+"'>"+f"{dp:.0f}%"+"</td></tr>")
    gt_table = ("<div class='card'><h3>Aciertos por color GT</h3><div class='tbl'><table><thead>"
        "<tr><th>GT</th><th>Nombre</th><th>Hex</th><th>N</th><th>Cen OK%</th><th>Lat OK%</th><th>Decision OK%</th></tr>"
        "</thead><tbody>"+"".join(gt_rows)+"</tbody></table></div></div>")

    def rgb_table(title, rgbs_dict):
        rrows = []
        for c in gt_colors:
            name, hex_ = gt_info[c]
            rgbs = rgbs_dict.get(c, [])
            if not rgbs: continue
            try:
                def parse(x):
                    if isinstance(x,(list,tuple)): return float(x[0]),float(x[1]),float(x[2])
                    p=str(x).replace(";",",").split(","); return float(p[0]),float(p[1]),float(p[2])
                Rs,Gs,Bs = zip(*[parse(x) for x in rgbs])
                R,G,B = statistics.mean(Rs),statistics.mean(Gs),statistics.mean(Bs)
                mh = "#{:02X}{:02X}{:02X}".format(max(0,min(255,int(round(R)))),max(0,min(255,int(round(G)))),max(0,min(255,int(round(B)))))
                try: dr=int(R)-int(hex_[1:3],16); dg=int(G)-int(hex_[3:5],16); db=int(B)-int(hex_[5:7],16)
                except: dr=dg=db=0
                def dfmt(v):
                    c2="bad" if abs(v)>30 else("warn" if abs(v)>10 else "ok")
                    return "<span class='"+c2+"'>"+f"{v:+d}"+"</span>"
                rrows.append("<tr><td>"+c+"</td><td>"+_sw(hex_)+name+"</td><td><code>"+hex_+"</code></td>"
                    "<td>"+_sw(mh)+"<code>"+mh+"</code></td>"
                    "<td>"+dfmt(dr)+"</td><td>"+dfmt(dg)+"</td><td>"+dfmt(db)+"</td>"
                    "<td>"+str(len(rgbs))+"</td></tr>")
            except: pass
        if not rrows: return ""
        return ("<div class='card'><h3>RGB esperado vs medido &mdash; "+title+"</h3>"
            "<div class='tbl'><table><thead><tr>"
            "<th>GT</th><th>Nombre</th><th>Hex GT</th><th>Medido</th>"
            "<th>&#916;R</th><th>&#916;G</th><th>&#916;B</th><th>N</th>"
            "</tr></thead><tbody>"+"".join(rrows)+"</tbody></table></div></div>")

    rgb_cen = rgb_table("CENITAL", d["cen_rgbs"])
    rgb_lat = rgb_table("LATERAL", d["lat_rgbs"])

    color_errors = [r for r in color_u if (r.get("color_match_cenital") is False
                    or r.get("color_match_lateral") is False or r.get("color_consensus_ok") is False)]
    err_hdr = ("<tr><th>idx</th><th>Imgs</th><th>ref</th><th>pose</th><th>face</th>"
        "<th>color GT</th><th>cen estimado</th><th>cen OK</th>"
        "<th>lat estimado</th><th>lat OK</th><th>consenso</th><th>modelo</th></tr>")
    err_rows = "".join(_color_row(r, imgdir) for r in color_errors)
    err_table = ("<div class='card'><h3>Listado completo de errores de color ("+str(len(color_errors))+" muestras)</h3>"
        "<div class='sub'>Todas las muestras con mismatch cenital, lateral o de consenso.</div>"
        "<div class='tbl tall'><table><thead>"+err_hdr+"</thead><tbody>"+err_rows+"</tbody></table></div></div>")

    bias = []
    for c in gt_colors:
        nm, hx = gt_info[c]
        sub = [r for r in color_u if r.get("color_code_gt")==c]
        wrong = [r for r in sub if r.get("color_match_cenital") is False]
        if wrong and len(wrong) > len(sub)*0.4:
            top = Counter(r.get("color_cenital_normalized_code") for r in wrong).most_common(1)
            pred = top[0][0] if top else "?"
            bias.append("<li>"+_sw(hx)+c+" ("+nm+") confundido con <strong>"+pred+"</strong> ("+str(len(wrong))+"/"+str(len(sub))+" fallos)</li>")

    if bias:
        diag_txt = "<strong>Patrones de confusion:</strong><ul>"+"".join(bias)+"</ul>Posibles causas: sesgo iluminacion o colores proximos en CIELAB."
        diag_cls = ""
    elif d["n_any_bad"] < nc*0.10:
        diag_txt = "<strong>Color OK.</strong> Menos del 10% de mismatches."
        diag_cls = "g"
    else:
        diag_txt = ("<strong>"+str(d["n_any_bad"])+" mismatches ("+f"{_pct(d['n_any_bad'],nc):.0f}%"+").</strong> "
            "Revisar iluminacion Blender, aplicar white-point correction y verificar catalogo de colores.")
        diag_cls = ""

    return ('<div class="sec" id="color"><h2>1. COLOR</h2>'
        +'<div class="grid">'+"".join(kpis)+'</div>'
        +gt_table+rgb_cen+rgb_lat+err_table
        +"<div class='diag "+diag_cls+"'>"+diag_txt+"</div>"
        +'</div>')


# ---------------------------------------------------------------------------
# Section 2: SUPERFICIE CENITAL
# ---------------------------------------------------------------------------

def _surf_row(r, imgdir, th_s):
    e = r.get("surface_error_rel_pct")
    mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='bad'>NO</span>"
    imgs = "<div class='ip'>" + _thumb(r.get("cenital_file",""), imgdir) + "</div>"
    ec = _ec(e, th_s, th_s*3) if e is not None else ""
    return ("".join([
        "<tr class='"+_rc(e, th_s, th_s*3)+"'>",
        "<td>"+str(r.get("index",""))+"</td>",
        "<td>"+imgs+"</td>",
        "<td class='nw'>"+str(r.get("ref_gt",""))+"</td>",
        "<td>"+str(r.get("pose_index_gt",""))+"</td>",
        "<td>"+str(r.get("face_class_gt",""))+"</td>",
        "<td class='num'>"+_f(r.get("surface_db_silhouette_mm2"))+" mm²</td>",
        "<td class='num'>"+_f(r.get("surface_obs_footprint_mm2"))+" mm²</td>",
        "<td class='num'>"+_f(r.get("surface_obs_apparent_mm2"))+" mm²</td>",
        "<td class='num "+ec+"'>"+_f(e)+"%</td>",
        "<td>"+_f(r.get("yolo_conf_cenital"),3)+"</td>",
        "<td>"+str(r.get("valid_by_surface_count",""))+"</td>",
        "<td>"+mm+" "+str(r.get("ref_inferred",""))+"</td>",
        "</tr>",
    ]))


def sec_surface(d, imgdir, th_s):
    st = d["s_st"]
    if not st:
        return '<div class="sec" id="surface"><h2>2. SUPERFICIE CENITAL</h2><p>Sin datos de superficie.</p></div>'
    s_abs = d["s_abs"]; s_bad = d["s_bad"]; ns = len(s_abs)
    pct_bad = _pct(len(s_bad), ns)
    kpis = [
        _kpi("Error medio |%|",    _f(st["mean"]),   "mediana "+_f(st["median"])+"%", _err_kpi_cls(st["mean"])),
        _kpi("std / p90",          _f(st["std"])+" / "+_f(st["p90"]), "p10="+_f(st["p10"]), ""),
        _kpi("Errores > "+str(th_s)+"%",  str(len(s_bad))+"/"+str(ns), f"{pct_bad:.1f}%", _err_kpi_cls(pct_bad)),
        _kpi("Min / Max error",   _f(st["min"])+"% / "+_f(st["max"])+"%", "rango absoluto", ""),
    ]

    # Histogram
    hdata = _hist(d["s_vals"])
    maxc = max(c for _,c in hdata) if hdata else 1
    hist_html = _hist_html(hdata, maxc)

    # Full table sorted by |error| descending
    all_with_surf = [r for r in d.get("all_results",[]) if isinstance(r.get("surface_error_rel_pct"),(int,float))]
    all_with_surf.sort(key=lambda r: abs(r["surface_error_rel_pct"]), reverse=True)
    surf_hdr = ("<tr><th>idx</th><th>Img cen</th><th>ref</th><th>pose</th><th>face</th>"
        "<th>superficie BD (silueta) mm²</th><th>obs footprint mm²</th>"
        "<th>obs apparent mm²</th><th>error %</th>"
        "<th>YOLO conf</th><th>v_surf</th><th>modelo</th></tr>")
    surf_rows = "".join(_surf_row(r, imgdir, th_s) for r in all_with_surf)
    surf_table = ("<div class='card'><h3>Listado completo errores superficie ("+str(len(all_with_surf))+" muestras con dato, ordenadas por |error|)</h3>"
        "<div class='sub'>Verde &lt;"+str(th_s)+"% | Amarillo "+str(th_s)+"–"+str(th_s*3)+"% | Rojo &gt;"+str(th_s*3)+"%. "
        "superficie_db=zenith_silhouette_area (silueta 2D mesh LDraw). "
        "obs_apparent incluye caras laterales visibles por perspectiva.</div>"
        "<div class='tbl tall'><table><thead>"+surf_hdr+"</thead><tbody>"+surf_rows+"</tbody></table></div></div>")

    # per-pose worst
    pose_bkt = defaultdict(list)
    for r in all_with_surf:
        pose_bkt[(r.get("ref_gt"),r.get("pose_index_gt"),r.get("face_class_gt"))].append(abs(r["surface_error_rel_pct"]))
    pose_means = [(ref,pi,fc, statistics.mean(vs), len(vs)) for(ref,pi,fc),vs in pose_bkt.items()]
    pose_means.sort(key=lambda x:-x[3])
    pose_rows = "".join(
        "<tr class='"+_rc(m,th_s,th_s*3)+"'><td>"+str(ref)+"</td><td>"+str(pi)+"</td><td>"+str(fc)+"</td>"
        "<td>"+str(n)+"</td><td class='num "+_ec(m,th_s,th_s*3)+"'>"+_f(m)+"%</td></tr>"
        for ref,pi,fc,m,n in pose_means[:20])
    pose_table = ("<div class='card'><h3>Top 20 peores poses por error medio superficie</h3>"
        "<div class='tbl'><table><thead><tr><th>ref</th><th>pose</th><th>face</th><th>n</th><th>err medio |%|</th></tr></thead>"
        "<tbody>"+pose_rows+"</tbody></table></div></div>")

    # Diagnosis
    sys_bias = st["mean"] > 20
    if sys_bias:
        diag_txt = ("<strong>Error sistematico elevado (media="+_f(st["mean"])+"%)</strong>. "
            "Posibles causas: <ul>"
            "<li>Calibracion px/mm incorrecta (verificar PX_PER_MM_CENITAL)</li>"
            "<li>Correccion de perspectiva insuficiente para piezas en borde del FOV</li>"
            "<li>surface_db_silhouette_mm2 no poblada en BD (ejecutar populate_silhouette_areas.py)</li>"
            "<li>SAM segmenta con ruido en determinadas poses (especialmente Side con caras laterales visibles)</li>"
            "</ul>")
        diag_cls = ""
    else:
        diag_txt = "<strong>Superficie OK.</strong> Error medio "+_f(st["mean"])+"%. Distribucion aceptable."
        diag_cls = "g"

    return ('<div class="sec" id="surface"><h2>2. SUPERFICIE CENITAL</h2>'
        +'<div class="grid">'+"".join(kpis)+'</div>'
        +'<div class="card"><h3>Distribucion de errores de superficie</h3>'+hist_html+'</div>'
        +surf_table+pose_table
        +"<div class='diag "+diag_cls+"'>"+diag_txt+"</div>"
        +'</div>')


# ---------------------------------------------------------------------------
# Section 3: ALTURA LATERAL
# ---------------------------------------------------------------------------

def _height_row(r, imgdir, th_h):
    e = r.get("lateral_height_error_rel_pct")
    mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='bad'>NO</span>"
    imgs = "<div class='ip'>" + _thumb(r.get("lateral_file",""), imgdir) + "</div>"
    ec = _ec(e, th_h, th_h*3) if e is not None else ""
    return ("".join([
        "<tr class='"+_rc(e, th_h, th_h*3)+"'>",
        "<td>"+str(r.get("index",""))+"</td><td>"+imgs+"</td>",
        "<td class='nw'>"+str(r.get("ref_gt",""))+"</td>",
        "<td>"+str(r.get("pose_index_gt",""))+"</td>",
        "<td>"+str(r.get("face_class_gt",""))+"</td>",
        "<td class='num'>"+_f(r.get("lateral_height_db_mm"))+" mm</td>",
        "<td class='num'>"+_f(r.get("effective_height_db_mm"))+" mm</td>",
        "<td class='num'>"+_f(r.get("lateral_height_meas_mm"))+" mm</td>",
        "<td class='num "+ec+"'>"+_f(e)+"%</td>",
        "<td>"+_f(r.get("yolo_conf_lateral"),3)+"</td>",
        "<td>"+str(r.get("valid_by_height_count",""))+"</td>",
        "<td>"+mm+" "+str(r.get("ref_inferred",""))+"</td></tr>",
    ]))


def sec_height(d, imgdir, th_h):
    st = d["h_st"]
    if not st:
        return '<div class="sec" id="height"><h2>3. ALTURA LATERAL</h2><p>Sin datos.</p></div>'
    h_abs=d["h_abs"]; h_bad=d["h_bad"]; nh=len(h_abs)
    pct_bad=_pct(len(h_bad),nh)
    kpis=[
        _kpi("Error medio |%|", _f(st["mean"]), "mediana "+_f(st["median"])+"%", _err_kpi_cls(st["mean"])),
        _kpi("std / p90", _f(st["std"])+" / "+_f(st["p90"]), "p10="+_f(st["p10"]), ""),
        _kpi("Errores > "+str(th_h)+"%", str(len(h_bad))+"/"+str(nh), f"{pct_bad:.1f}%", _err_kpi_cls(pct_bad)),
        _kpi("Min / Max error", _f(st["min"])+"% / "+_f(st["max"])+"%", "rango absoluto", ""),
    ]
    hdata=_hist(d["h_vals"]); maxc=max(c for _,c in hdata) if hdata else 1
    hist_html=_hist_html(hdata,maxc)
    all_with_h=[r for r in d.get("all_results",[]) if isinstance(r.get("lateral_height_error_rel_pct"),(int,float))]
    all_with_h.sort(key=lambda r:abs(r["lateral_height_error_rel_pct"]),reverse=True)
    h_hdr=("<tr><th>idx</th><th>Img lat</th><th>ref</th><th>pose</th><th>face</th>"
        "<th>h DB mm</th><th>h efectiva mm</th><th>h medida mm</th>"
        "<th>error %</th><th>YOLO lat</th><th>v_h</th><th>modelo</th></tr>")
    h_rows="".join(_height_row(r,imgdir,th_h) for r in all_with_h)
    h_table=("<div class='card'><h3>Listado completo errores altura ("+str(len(all_with_h))+" muestras, ordenadas por |error|)</h3>"
        "<div class='sub'>h_DB=lateral_height_db_mm. h_efectiva=effective_height_db_mm. h_medida=SAM imagen lateral.</div>"
        "<div class='tbl tall'><table><thead>"+h_hdr+"</thead><tbody>"+h_rows+"</tbody></table></div></div>")
    face_bkt=defaultdict(list)
    for r in all_with_h: face_bkt[r.get("face_class_gt","?")].append(abs(r["lateral_height_error_rel_pct"]))
    face_rows="".join(
        "<tr><td>"+fc+"</td><td>"+str(len(vs))+"</td>"
        "<td class='num "+_ec(statistics.mean(vs),th_h,th_h*3)+"'>"+_f(statistics.mean(vs))+"%</td>"
        "<td class='num'>"+_f(statistics.median(vs))+"%</td>"
        "<td class='num'>"+_f(max(vs))+"%</td></tr>"
        for fc,vs in sorted(face_bkt.items()))
    face_table=("<div class='card'><h3>Error altura por face_class</h3>"
        "<div class='tbl'><table><thead><tr><th>face_class</th><th>n</th><th>err medio</th><th>mediana</th><th>max</th></tr></thead>"
        "<tbody>"+face_rows+"</tbody></table></div>"
        "<div class='sub'>Error mayor en Side indica problemas de segmentacion SAM en piezas de canto.</div></div>")
    if st["mean"]>20:
        diag_txt=("<strong>Error sistematico (media="+_f(st["mean"])+"%)</strong>. "
            "Causas probables: SAM no segmenta correctamente la silueta lateral "
            "(sombras proyectadas, piezas transparentes o muy oscuras), "
            "calibracion PX_PER_MM_LATERAL incorrecta, o piezas con studs/protuberancias "
            "que aumentan la altura aparente vs la altura nominal de la pose.")
        diag_cls=""
    else:
        diag_txt="<strong>Altura OK.</strong> Error medio "+_f(st["mean"])+"%. Dentro de tolerancia."
        diag_cls="g"
    return ('<div class="sec" id="height"><h2>3. ALTURA LATERAL</h2>'
        +'<div class="grid">'+"".join(kpis)+'</div>'
        +'<div class="card"><h3>Distribucion errores altura</h3>'+hist_html+'</div>'
        +h_table+face_table
        +"<div class='diag "+diag_cls+"'>"+diag_txt+"</div>"
        +'</div>')


# ---------------------------------------------------------------------------
# Section 4: INFERENCIA AGREGADA
# ---------------------------------------------------------------------------

def sec_inference(d, imgdir, top_n):
    acc=d["accuracy"]; n=d["n"]; correct=d["correct"]
    per_piece=d["per_piece"]; confusion_top=d["confusion_top"]
    score_gap=d["score_gap"]; hard=d["hard"]
    kpis=[
        _kpi("Accuracy global", f"{acc:.1f}%", str(correct)+"/"+str(n), _kpi_cls(acc)),
        _kpi("Piezas acc=0%", str(sum(1 for p in per_piece if p["accuracy_pct"]==0.0)),
             "de "+str(len(per_piece))+" refs", "bad"),
        _kpi("Hard cases", str(len(hard)), "sin filtrado efectivo", "warn" if hard else "ok"),
        _kpi("Pares confundidos", str(len(confusion_top)), "top pares gt!=pred", ""),
    ]
    piece_rows="".join(
        "<tr class='"+_rc(100-p["accuracy_pct"],40,70)+"'>"
        "<td>"+str(p["ref"])+"</td><td>"+str(p["color_code"])+"</td><td>"+str(p["n_samples"])+"</td>"
        "<td class='"+_kpi_cls(p["accuracy_pct"])+"'>"+_f(p["accuracy_pct"],1)+"%</td>"
        "<td>"+_f(p["color_cen_match_pct"],1)+"%</td><td>"+_f(p["color_lat_match_pct"],1)+"%</td>"
        "<td class='num "+_ec(p["surf_err_mean"] or 0)+"'>"+_f(p["surf_err_mean"])+"</td>"
        "<td class='num "+_ec(p["h_err_mean"] or 0)+"'>"+_f(p["h_err_mean"])+"</td></tr>"
        for p in sorted(per_piece, key=lambda x:x["accuracy_pct"]))
    piece_table=("<div class='card'><h3>Accuracy y errores por (ref, color) &mdash; ordenado por accuracy ASC</h3>"
        "<div class='tbl'><table><thead><tr>"
        "<th>ref</th><th>color</th><th>n</th><th>acc%</th>"
        "<th>col cen%</th><th>col lat%</th><th>err sup|%|</th><th>err h|%|</th>"
        "</tr></thead><tbody>"+piece_rows+"</tbody></table></div></div>")
    conf_rows="".join(
        "<tr><td>"+str(gt)+"</td><td class='bad'>"+str(pr)+"</td><td>"+str(c)+"</td></tr>"
        for gt,pr,c in confusion_top[:top_n])
    conf_table=("<div class='card'><h3>Confusion matrix (top "+str(min(top_n,len(confusion_top)))+" pares ref_gt &rarr; ref_inferido)</h3>"
        "<div class='sub'>Pares recurrentes: DINOv2 no discrimina bien esas piezas. Generar mas referencias para ellas.</div>"
        "<div class='tbl'><table><thead><tr><th>GT</th><th>Predicho</th><th>N</th></tr></thead>"
        "<tbody>"+conf_rows+"</tbody></table></div></div>")
    score_rows="".join(
        "<tr class='row-bad'>"
        "<td>"+str(r.get("index",""))+"</td>"
        "<td class='nw'>"+str(r.get("ref_gt",""))+"</td>"
        "<td class='bad'>"+str(r.get("ref_inferred",""))+"</td>"
        "<td>"+str(r.get("pose_index_gt",""))+"</td><td>"+str(r.get("face_class_gt",""))+"</td>"
        "<td class='num bad'>"+_f(r.get("consensus_score"),4)+"</td>"
        "<td>"+str(r.get("valid_by_color_count",""))+"</td>"
        "<td>"+str(r.get("valid_by_surface_count",""))+"</td>"
        "<td>"+str(r.get("valid_by_height_count",""))+"</td>"
        "<td><div class='ip'>"+_thumb(r.get("cenital_file",""),imgdir)+_thumb(r.get("lateral_file",""),imgdir)+"</div></td>"
        "</tr>"
        for r in score_gap[:top_n])
    score_table=("<div class='card'><h3>High-confidence errors: fallos con mayor score ("+str(len(score_gap))+" total)</h3>"
        "<div class='sub'>Criticos: el modelo predice con alta confianza pero se equivoca. "
        "Indican solapamiento de embeddings DINOv2 entre piezas visualmente similares.</div>"
        "<div class='tbl'><table><thead><tr>"
        "<th>idx</th><th>GT</th><th>Pred</th><th>pose</th><th>face</th><th>score</th>"
        "<th>v_col</th><th>v_sup</th><th>v_h</th><th>imgs</th>"
        "</tr></thead><tbody>"+score_rows+"</tbody></table></div></div>")
    hard_section=""
    if hard:
        hard_rows="".join(
            "<tr class='row-hard'>"
            "<td>"+str(r.get("index",""))+"</td>"
            "<td class='nw'>"+str(r.get("ref_gt",""))+"</td>"
            "<td class='bad'>"+str(r.get("ref_inferred",""))+"</td>"
            "<td>"+str(r.get("valid_by_color_count",""))+"</td>"
            "<td>"+_f(r.get("consensus_score"),4)+"</td>"
            "</tr>"
            for r in hard[:top_n])
        hard_section=("<div class='card'><h3>Hard cases: "+str(len(hard))+" muestras donde el gating no redujo candidatos</h3>"
            "<div class='sub'>v_color=v_surface=v_height. DINOv2 opera sobre todos los candidatos. Revisar thresholds de gating.</div>"
            "<div class='tbl'><table><thead><tr>"
            "<th>idx</th><th>GT</th><th>Pred</th><th>v_color</th><th>score</th>"
            "</tr></thead><tbody>"+hard_rows+"</tbody></table></div></div>")
    return ('<div class="sec" id="inference"><h2>4. INFERENCIA AGREGADA (DINOv2 + Pipeline)</h2>'
        +'<div class="grid">'+"".join(kpis)+'</div>'
        +piece_table+conf_table+score_table+hard_section+'</div>')


# ---------------------------------------------------------------------------
# Section 5: TENDENCIAS Y PATRONES
# ---------------------------------------------------------------------------

def sec_trends(d, top_n):
    per_piece=d["per_piece"]; per_pose=d["per_pose"]

    def combined_score(p):
        return max(0, 100-(p["accuracy_pct"] or 0)) + (p["surf_err_mean"] or 0)*0.5 + (p["h_err_mean"] or 0)*0.5

    worst = sorted(per_piece, key=combined_score, reverse=True)[:top_n]
    wp_rows="".join(
        "<tr><td>"+str(p["ref"])+"</td><td>"+str(p["color_code"])+"</td><td>"+str(p["n_samples"])+"</td>"
        "<td class='"+_kpi_cls(p["accuracy_pct"])+"'>"+_f(p["accuracy_pct"],1)+"%</td>"
        "<td class='num "+_ec(p["surf_err_mean"] or 0)+"'>"+_f(p["surf_err_mean"])+"</td>"
        "<td class='num "+_ec(p["h_err_mean"] or 0)+"'>"+_f(p["h_err_mean"])+"</td>"
        "<td class='num'>"+_f(combined_score(p),1)+"</td></tr>"
        for p in worst)
    wp_table=("<div class='card'><h3>Top "+str(len(worst))+" piezas mas problematicas (score combinado)</h3>"
        "<div class='sub'>Score = (100-acc%) + 0.5*err_sup + 0.5*err_h. Candidatas prioritarias para la siguiente iteracion.</div>"
        "<div class='tbl'><table><thead><tr><th>ref</th><th>color</th><th>n</th>"
        "<th>acc%</th><th>err sup</th><th>err h</th><th>score</th>"
        "</tr></thead><tbody>"+wp_rows+"</tbody></table></div></div>")

    worst_poses=sorted(per_pose, key=lambda p:(p["surf_err_mean"] or 0)+(p["h_err_mean"] or 0), reverse=True)[:20]
    wp2_rows="".join(
        "<tr><td>"+str(p["ref"])+"</td><td>"+str(p["pose_index"])+"</td><td>"+str(p["face_class"])+"</td>"
        "<td>"+str(p["n_samples"])+"</td>"
        "<td class='"+_kpi_cls(p["accuracy_pct"])+"'>"+_f(p["accuracy_pct"],1)+"%</td>"
        "<td class='num "+_ec(p["surf_err_mean"] or 0)+"'>"+_f(p["surf_err_mean"])+"</td>"
        "<td class='num "+_ec(p["h_err_mean"] or 0)+"'>"+_f(p["h_err_mean"])+"</td></tr>"
        for p in worst_poses)
    wp2_table=("<div class='card'><h3>Top 20 peores poses</h3>"
        "<div class='tbl'><table><thead><tr>"
        "<th>ref</th><th>pose</th><th>face</th><th>n</th><th>acc%</th><th>err sup</th><th>err h</th>"
        "</tr></thead><tbody>"+wp2_rows+"</tbody></table></div></div>")

    face_acc=defaultdict(lambda:{"c":0,"t":0})
    for p in per_pose:
        fc=p["face_class"] or "?"
        nn=p["n_samples"]; nc=int(round((p["accuracy_pct"] or 0)*nn/100))
        face_acc[fc]["c"]+=nc; face_acc[fc]["t"]+=nn
    face_rows="".join(
        "<tr><td>"+fc+"</td><td>"+str(v["t"])+"</td>"
        "<td class='"+_kpi_cls(_pct(v["c"],v["t"]))+"'>"+str(round(_pct(v["c"],v["t"]),1))+"%</td></tr>"
        for fc,v in sorted(face_acc.items()))
    face_table=("<div class='card'><h3>Accuracy global por face_class</h3>"
        "<div class='tbl'><table><thead><tr><th>face_class</th><th>n</th><th>acc%</th></tr></thead>"
        "<tbody>"+face_rows+"</tbody></table></div>"
        "<div class='sub'>Si face_class Side tiene accuracy mucho menor, "
        "generar mas referencias para poses de canto.</div></div>")

    actions=["Generar mas referencias DINOv2 para piezas con accuracy=0%",
             "Revisar calibracion px/mm si error de superficie es sistematicamente alto",
             "Ajustar thresholds de gating (th_surface, th_height) para reducir hard cases",
             "Para pares mas confundidos en confusion matrix: anadir renders con angulos discriminativos",
             "Si face_class Side tiene accuracy baja: mejorar segmentacion SAM en vistas de canto",
             "Verificar que surface_db_silhouette_mm2 esta poblado en BD para todas las piezas"]
    tips=("<div class='card'><h3>Acciones de mejora para la proxima iteracion</h3>"
        "<ul style='font-size:12px;line-height:2.2;padding-left:18px'>"
        +"".join("<li>"+a+"</li>" for a in actions)
        +"</ul></div>")

    return ('<div class="sec" id="trends"><h2>5. TENDENCIAS Y PATRONES</h2>'
        +wp_table+wp2_table+face_table+tips+'</div>')


# ---------------------------------------------------------------------------
# HTML assembly + main
# ---------------------------------------------------------------------------

def build_html(d, imgdir, th_s, th_h, top_n, eval_path, ts):
    sections = [
        sec_summary(d, th_s, th_h),
        sec_color(d, imgdir),
        sec_surface(d, imgdir, th_s),
        sec_height(d, imgdir, th_h),
        sec_inference(d, imgdir, top_n),
        sec_trends(d, top_n),
    ]
    nav = ("".join([
        '<a href="#summary">0.Resumen</a>',
        '<a href="#color">1.Color</a>',
        '<a href="#surface">2.Superficie</a>',
        '<a href="#height">3.Altura</a>',
        '<a href="#inference">4.Inferencia</a>',
        '<a href="#trends">5.Tendencias</a>',
    ]))
    n=d["n"]; acc=d["accuracy"]
    meta=("<span style='margin-left:auto;color:#64748b;font-size:11px'>"
        +str(n)+" muestras | acc="+f"{acc:.1f}%"
        +" | "+ts+" | "+os.path.basename(eval_path)+"</span>")
    return ("<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Inference Report "+str(n)+" muestras</title>"
        "<style>"+CSS+"</style></head><body>"
        "<div id='nav'><span class='brand'>LegoVision InfReport</span>"
        +nav+meta+"</div>"
        "<div class='page'>"
        +"".join(sections)
        +"</div>"
        +JS+"</body></html>")


def main():
    ap = argparse.ArgumentParser(description="Unified inference report generator")
    ap.add_argument("--eval",       required=True, help="Path to eval_report.json")
    ap.add_argument("--images_dir", required=True, help="Directory containing test images")
    ap.add_argument("--out",        required=True, help="Output directory")
    ap.add_argument("--th_surface", type=float, default=10.0, help="Surface error threshold %%")
    ap.add_argument("--th_height",  type=float, default=10.0, help="Height error threshold %%")
    ap.add_argument("--top_n",      type=int,   default=50,   help="Rows in summary tables")
    args = ap.parse_args()

    print(f"[InfReport] Loading {args.eval}...")
    with open(args.eval, "r", encoding="utf-8") as fp:
        eval_data = json.load(fp)
    results = eval_data.get("results", [])
    if not results:
        print("[InfReport] ERROR: No results in eval_report"); sys.exit(1)
    print(f"[InfReport] {len(results)} samples loaded")

    os.makedirs(args.out, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Aggregate
    d = _aggregate(results, args.th_surface, args.th_height)
    d["all_results"] = results  # needed by surf/height sections

    # Write CSVs
    csv_full = os.path.join(args.out, "inference_full.csv")
    _write_csv(csv_full, results, FULL_COLS)
    print(f"[InfReport] CSV full: {csv_full} ({len(results)} rows)")

    csv_piece = os.path.join(args.out, "inference_per_piece.csv")
    _write_csv(csv_piece, d["per_piece"], PIECE_COLS)
    print(f"[InfReport] CSV per piece: {csv_piece} ({len(d['per_piece'])} rows)")

    csv_pose = os.path.join(args.out, "inference_per_pose.csv")
    _write_csv(csv_pose, d["per_pose"], POSE_COLS)
    print(f"[InfReport] CSV per pose: {csv_pose} ({len(d['per_pose'])} rows)")

    color_errors = [r for r in d["color_u"] if (r.get("color_match_cenital") is False
                    or r.get("color_match_lateral") is False or r.get("color_consensus_ok") is False)]
    csv_color = os.path.join(args.out, "inference_errors_color.csv")
    _write_csv(csv_color, color_errors, COLOR_COLS)
    print(f"[InfReport] CSV color errors: {csv_color} ({len(color_errors)} rows)")

    csv_surf = os.path.join(args.out, "inference_errors_surface.csv")
    _write_csv(csv_surf, d["s_bad"], SURF_COLS)
    print(f"[InfReport] CSV surface errors: {csv_surf} ({len(d['s_bad'])} rows)")

    csv_h = os.path.join(args.out, "inference_errors_height.csv")
    _write_csv(csv_h, d["h_bad"], HEIGHT_COLS)
    print(f"[InfReport] CSV height errors: {csv_h} ({len(d['h_bad'])} rows)")

    # Write HTML
    html = build_html(d, args.images_dir, args.th_surface, args.th_height, args.top_n, args.eval, ts)
    html_path = os.path.join(args.out, "inference_report.html")
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    print(f"[InfReport] HTML report: {html_path}")
    print(f"[InfReport] Done. Accuracy={d['accuracy']:.1f}%  N={d['n']}")


if __name__ == "__main__":
    main()
