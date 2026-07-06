# -*- coding: utf-8 -*-
"""generate_color_focus_report.py
Reporte de foco en COLOR. Genera CSV + HTML con thumbnails."""
from __future__ import annotations
import argparse, base64, csv, json, os, statistics, sys
from collections import Counter, defaultdict

CSV_COLS = [
    "index", "cenital_file", "lateral_file",
    "ref_gt", "pose_index_gt",
    "color_code_gt", "color_name_gt", "color_hex_gt",
    "color_cenital_rgb_est", "color_cenital_normalized_code", "color_cenital_normalized_name",
    "color_lateral_rgb_est", "color_lateral_normalized_code", "color_lateral_normalized_name",
    "color_match_cenital", "color_match_lateral",
    "color_consensus_ok", "color_consensus_status", "color_decision_used",
    "model_match", "ref_inferred",
]


def _ser(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, bool):
        return "True" if v else "False"
    return v


def write_csv(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _ser(r.get(k)) for k in CSV_COLS})


def encode_image_b64(path, max_w=180):
    if not path or not os.path.isfile(path):
        return ""
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_w:
            new_h = int(h * (max_w / w))
            img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def is_forced_sample(r):
    fn = r.get("cenital_file") or ""
    return "_forced_" in fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images_dir", required=True)
    args = parser.parse_args()
    if not os.path.isfile(args.eval):
        print(f"[ERROR] no existe {args.eval}"); sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    with open(args.eval, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    results = data.get("results", [])
    print(f"[color_focus] {len(results)} muestras")
    csv_path = os.path.join(args.out, "color_focus_report.csv")
    write_csv(results, csv_path)
    print(f"[color_focus] CSV: {csv_path}")
    html_path = os.path.join(args.out, "color_focus_report.html")
    render_html(results, args.images_dir, html_path)
    print(f"[color_focus] HTML: {html_path}")


HTML_HEAD = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Color Focus Report</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f8;color:#222;margin:0;padding:24px}
  h1{margin-top:0}
  h2{border-bottom:2px solid #cbd5e1;padding-bottom:6px;margin-top:32px}
  .card{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:16px;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
  .metric{background:#fff;border-radius:8px;padding:12px 16px;border-left:4px solid #2563eb}
  .metric .v{font-size:24px;font-weight:600;color:#0f172a}
  .metric .k{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
  table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}
  th,td{padding:6px 8px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
  th{background:#f1f5f9;font-weight:600;position:sticky;top:0;z-index:1}
  tr:hover{background:#fafafa}
  .ok{color:#15803d;font-weight:600}
  .ko{color:#b91c1c;font-weight:600}
  .row-ko{background:#fef2f2}
  .row-ok{background:#f0fdf4}
  .forced td{background:#fffbeb !important}
  .forced td:first-child{border-left:4px solid #f59e0b}
  img.thumb{display:block;max-width:160px;border:1px solid #ddd;border-radius:4px}
  .sw{display:inline-block;vertical-align:middle;width:14px;height:14px;border:1px solid #888;margin-right:4px}
  .mono{font-family:monospace}
</style></head><body>
"""

HTML_FOOT = "</body></html>"


def _rgb_swatch(rgb_val):
    if rgb_val is None:
        return ""
    if isinstance(rgb_val, str):
        try:
            parts = rgb_val.replace(";", ",").split(",")
            r, g, b = [int(round(float(x))) for x in parts[:3]]
        except Exception:
            return rgb_val
    elif isinstance(rgb_val, (list, tuple)) and len(rgb_val) >= 3:
        r, g, b = [int(round(float(x))) for x in rgb_val[:3]]
    else:
        return str(rgb_val)
    return (f"<span class='sw' style='background:rgb({r},{g},{b})'></span>"
            f"<span class='mono'>[{r},{g},{b}]</span>")


def _hex_swatch(hex_str):
    if not hex_str:
        return ""
    h = hex_str if hex_str.startswith("#") else "#" + hex_str
    return f"<span class='sw' style='background:{h}'></span><span class='mono'>{h}</span>"


def _row_html(r, images_dir, force_class=False):
    cen_p = os.path.join(images_dir, r.get("cenital_file") or "")
    lat_p = os.path.join(images_dir, r.get("lateral_file") or "")
    cen_b = encode_image_b64(cen_p)
    lat_b = encode_image_b64(lat_p)
    cen_img = f"<img class='thumb' src='{cen_b}'/>" if cen_b else "<i>(no img)</i>"
    lat_img = f"<img class='thumb' src='{lat_b}'/>" if lat_b else "<i>(no img)</i>"
    cm_c = "<span class='ok'>OK</span>" if r.get("color_match_cenital") else "<span class='ko'>NO</span>"
    cm_l = "<span class='ok'>OK</span>" if r.get("color_match_lateral") else "<span class='ko'>NO</span>"
    cons = ("<span class='ok'>agree</span>" if r.get("color_consensus_ok")
            else "<span class='ko'>error_disagree</span>")
    mm = "<span class='ok'>OK</span>" if r.get("model_match") else "<span class='ko'>NO</span>"
    cls_parts = []
    if force_class:
        cls_parts.append("forced")
    if r.get("color_match_cenital") and r.get("color_match_lateral"):
        cls_parts.append("row-ok")
    else:
        cls_parts.append("row-ko")
    cls = " ".join(cls_parts)
    return (
        f"<tr class='{cls}'>"
        f"<td>{r.get('index')}</td>"
        f"<td>{cen_img}</td>"
        f"<td>{lat_img}</td>"
        f"<td>{r.get('ref_gt')}</td>"
        f"<td>{r.get('pose_index_gt')}</td>"
        f"<td>{_hex_swatch(r.get('color_hex_gt'))}<br>"
        f"<small>{r.get('color_code_gt')} - {r.get('color_name_gt')}</small></td>"
        f"<td>{_rgb_swatch(r.get('color_cenital_rgb_est'))}<br>"
        f"<small>->{r.get('color_cenital_normalized_code')} {r.get('color_cenital_normalized_name')}</small></td>"
        f"<td>{cm_c}</td>"
        f"<td>{_rgb_swatch(r.get('color_lateral_rgb_est'))}<br>"
        f"<small>->{r.get('color_lateral_normalized_code')} {r.get('color_lateral_normalized_name')}</small></td>"
        f"<td>{cm_l}</td>"
        f"<td>{cons}</td>"
        f"<td>{r.get('color_decision_used')}</td>"
        f"<td>{mm}<br><small>pred={r.get('ref_inferred')}</small></td>"
        f"</tr>"
    )


TABLE_HEAD = (
    "<table><thead><tr>"
    "<th>idx</th><th>cenital</th><th>lateral</th>"
    "<th>ref</th><th>pose</th>"
    "<th>color GT</th>"
    "<th>cen est</th><th>match cen</th>"
    "<th>lat est</th><th>match lat</th>"
    "<th>consenso</th><th>decision</th>"
    "<th>model</th>"
    "</tr></thead><tbody>"
)


def render_html(results, images_dir, out_path):
    n = len(results)
    cm_cen = sum(1 for r in results if r.get("color_match_cenital"))
    cm_lat = sum(1 for r in results if r.get("color_match_lateral"))
    cons_ok = sum(1 for r in results if r.get("color_consensus_ok"))
    mm_ok = sum(1 for r in results if r.get("model_match"))
    pct = lambda k: f"{(k / n * 100.0) if n else 0.0:.1f}%"

    # Confusion color: GT -> normalizado cenital
    conf_cen = Counter()
    for r in results:
        gt = r.get("color_code_gt")
        pr = r.get("color_cenital_normalized_code")
        if gt is not None:
            conf_cen[(gt, pr)] += 1
    conf_top = sorted(conf_cen.items(), key=lambda kv: -kv[1])

    # Forzadas
    forced = [r for r in results if is_forced_sample(r)]
    others = [r for r in results if not is_forced_sample(r)]

    out = [HTML_HEAD]
    out.append(f"<h1>Color Focus Report</h1>")
    out.append(f"<p>Total muestras: <b>{n}</b> | Forzadas: <b>{len(forced)}</b> | "
               f"Random: <b>{len(others)}</b></p>")

    # Métricas
    out.append("<div class='grid'>")
    out.append(f"<div class='metric'><div class='k'>Match color cenital</div>"
               f"<div class='v'>{pct(cm_cen)}</div><small>{cm_cen}/{n}</small></div>")
    out.append(f"<div class='metric'><div class='k'>Match color lateral</div>"
               f"<div class='v'>{pct(cm_lat)}</div><small>{cm_lat}/{n}</small></div>")
    out.append(f"<div class='metric'><div class='k'>Consenso cenital==lateral</div>"
               f"<div class='v'>{pct(cons_ok)}</div><small>{cons_ok}/{n}</small></div>")
    out.append(f"<div class='metric'><div class='k'>Model match (ref)</div>"
               f"<div class='v'>{pct(mm_ok)}</div><small>{mm_ok}/{n}</small></div>")
    out.append("</div>")

    # Sección destacada: muestras forzadas
    if forced:
        out.append("<h2>Muestras forzadas (foco color)</h2>")
        out.append("<div class='card'>")
        out.append(TABLE_HEAD)
        for r in forced:
            out.append(_row_html(r, images_dir, force_class=True))
        out.append("</tbody></table></div>")

    # Tabla de todas las muestras
    out.append("<h2>Todas las muestras</h2>")
    out.append("<div class='card'>")
    out.append(TABLE_HEAD)
    for r in results:
        out.append(_row_html(r, images_dir, force_class=is_forced_sample(r)))
    out.append("</tbody></table></div>")

    # Confusion de color top
    out.append("<h2>Confusión de color (GT → cenital normalizado)</h2>")
    out.append("<div class='card'><table><thead><tr>"
               "<th>GT code</th><th>Pred cen code</th><th>n</th><th>match</th>"
               "</tr></thead><tbody>")
    for (gt, pr), cnt in conf_top:
        cls = "ok" if str(gt) == str(pr) else "ko"
        out.append(f"<tr><td>{gt}</td><td class='{cls}'>{pr}</td><td>{cnt}</td>"
                   f"<td>{'OK' if str(gt) == str(pr) else 'NO'}</td></tr>")
    out.append("</tbody></table></div>")

    out.append(HTML_FOOT)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("".join(out))


if __name__ == "__main__":
    main()
