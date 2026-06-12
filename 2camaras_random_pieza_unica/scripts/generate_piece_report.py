# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_piece_report.py
================================================================
Genera un report HTML standalone de diagnostico para una pieza
especifica (filtrada opcionalmente por color), usando los samples
del set 300 (`data/300/`) y el reporte de inferencia
(`inference_300_eval.json`).

El report contiene, para cada sample que coincide con el filtro:
  - Imagen cenital + lateral originales con bbox dibujado.
  - Mascara SAM superpuesta.
  - Crop SAM con fondo cinta (= input que ve DINOv2).
  - Estimaciones de color (cenital + lateral) RGB y CIELAB.
  - Estimacion de superficie y altura, errores vs GT.
  - Top-1 inferido y consenso_score.

Uso:
  .venv/bin/python 2camaras_random_pieza_unica/scripts/generate_piece_report.py \\
      --ref 32054 --color_code 11

  # Multiples piezas en una sola llamada:
  .venv/bin/python 2camaras_random_pieza_unica/scripts/generate_piece_report.py \\
      --pieces 32054:11 61184:86 32000:86 3040:86
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from ultralytics import SAM


CINTA_RGB = (37, 65, 84)
DEFAULT_DATA_DIR = os.path.join(project_root, "data", "300")
DEFAULT_EVAL = os.path.join(DEFAULT_DATA_DIR, "inference_300_eval.json")
DEFAULT_METADATA = os.path.join(DEFAULT_DATA_DIR, "random_300_metadata.json")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_DATA_DIR, "reports")


_sam = None
def get_sam():
    global _sam
    if _sam is None:
        _sam = SAM("mobile_sam.pt")
    return _sam


def sam_mask(img, bbox_norm):
    try:
        # Convert to RGB to avoid alpha-channel mismatch issues
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        x1 = max(0, int(bbox_norm[0] * w))
        y1 = max(0, int(bbox_norm[1] * h))
        x2 = min(w, int(bbox_norm[2] * w))
        y2 = min(h, int(bbox_norm[3] * h))
        
        results = get_sam()(np.array(img_rgb), bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            
            # Dynamic Local Background subtraction inside the bbox to eliminate holes/background projections
            crop_img = img_rgb.crop((x1, y1, x2, y2))
            crop_arr = np.array(crop_img)
            h_c, w_c = crop_arr.shape[:2]
            if h_c > 2 and w_c > 2:
                edges = np.vstack([
                    crop_arr[0, :],         # top edge
                    crop_arr[-1, :],        # bottom edge
                    crop_arr[:, 0],         # left edge
                    crop_arr[:, -1]         # right edge
                ])
                local_bg = edges.mean(axis=0)
                dists = np.linalg.norm(crop_arr.astype(np.float32) - local_bg, axis=-1)
                
                # Extract the crop mask, zero out bg, and write back to full mask
                crop_mask = mask[y1:y2, x1:x2].copy()
                crop_mask[dists < 25.0] = 0
                mask[y1:y2, x1:x2] = crop_mask
                
            return mask
    except Exception:
        pass
    fallback = np.zeros((h, w), dtype=np.uint8)
    fallback[y1:y2, x1:x2] = 255
    return fallback


def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.4):
    arr = np.array(img.convert("RGB"))
    h, w = mask.shape[:2]
    if (h, w) != arr.shape[:2]:
        import cv2
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    fg = mask > 0
    overlay = arr.copy()
    overlay[fg] = (arr[fg] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def draw_bbox(img, bbox_norm, color=(0, 255, 255), width=2):
    out = img.convert("RGB").copy()
    drw = ImageDraw.Draw(out)
    w, h = out.size
    x1 = int(bbox_norm[0] * w); y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w); y2 = int(bbox_norm[3] * h)
    drw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def crop_with_cinta_bg(img, mask, bbox_norm, canvas_size=224, margin_px=8):
    """Replica `_build_clean_canvas` + `apply_sam_mask_to_crop` de run_eval
    (post-fixes 1 y 2): crop por bbox, fondo CINTA fuera de la mascara,
    canvas 224 fit-to-canvas con fondo cinta."""
    w, h = img.size
    x1 = max(0, int(bbox_norm[0] * w))
    y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w))
    y2 = min(h, int(bbox_norm[3] * h))
    arr = np.array(img.convert("RGB"))
    crop = arr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return Image.new("RGB", (canvas_size, canvas_size), CINTA_RGB)
    sub_mask = mask[y1:y2, x1:x2] if mask.shape[:2] == arr.shape[:2] else None
    if sub_mask is not None and sub_mask.shape == crop.shape[:2]:
        fg = sub_mask > 0
        if fg.any():
            crop[~fg] = CINTA_RGB
    crop_pil = Image.fromarray(crop)
    cw, ch = crop_pil.size
    max_dim = canvas_size - 2 * margin_px
    scale = min(max_dim / cw, max_dim / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    resized = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), CINTA_RGB)
    canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
    return canvas


def to_b64(pil, max_side=320, fmt="JPEG", quality=88):
    img = pil.copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_eval_results(eval_path):
    with open(eval_path) as f:
        return json.load(f).get("results", [])


def load_metadata(meta_path):
    if not os.path.isfile(meta_path):
        return {"renders": []}
    with open(meta_path) as f:
        return json.load(f)


def build_meta_lookup(meta):
    """Retorna dict[(sample_index, cam_name)] = bbox_norm."""
    out = {}
    for r in meta.get("renders", []):
        idx = r.get("index")
        for cam_name in ("cenital", "lateral"):
            cd = (r.get("cameras") or {}).get(cam_name)
            if cd:
                out[(idx, cam_name)] = cd.get("bbox_norm")
    return out


def select_samples(eval_results, ref, color_code=None, max_samples=None):
    matches = []
    for r in eval_results:
        if r.get("ref_gt") != ref:
            continue
        if color_code is not None and str(r.get("color_code_gt")) != str(color_code):
            continue
        matches.append(r)
    matches.sort(key=lambda r: r.get("index", 0))
    if max_samples is not None:
        matches = matches[:max_samples]
    return matches


def _err_class(err_pct, scale=(20, 50)):
    if err_pct is None:
        return ""
    a = abs(err_pct)
    if a > scale[1]:
        return "err-bad"
    if a > scale[0]:
        return "err-warn"
    return "err-good"


def render_html(ref, color_code, samples, data_dir, meta_lookup):
    n = len(samples)
    n_correct = sum(1 for s in samples if s.get("model_match"))
    accuracy = (n_correct / n * 100.0) if n else 0.0
    n_color_cen_ok = sum(1 for s in samples if s.get("color_match_cenital") is True)
    n_color_lat_ok = sum(1 for s in samples if s.get("color_match_lateral") is True)
    surf_errs = [abs(s.get("surface_error_rel_pct"))
                 for s in samples if s.get("surface_error_rel_pct") is not None]
    h_errs = [abs(s.get("lateral_height_error_rel_pct"))
              for s in samples if s.get("lateral_height_error_rel_pct") is not None]

    title = f"Piece Report — {ref}" + (f" (color {color_code})" if color_code else "")
    css = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
           "background:#f4f6f8;color:#222;margin:0;padding:24px}"
           "h1{margin-top:0}"
           ".summary{background:#fff;border-radius:8px;padding:16px 20px;"
           "margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}"
           ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));"
           "gap:12px;margin:8px 0}"
           ".metric{background:#f8fafc;border-radius:6px;padding:8px 12px;"
           "border-left:4px solid #2563eb}"
           ".metric .v{font-size:18px;font-weight:600;color:#0f172a}"
           ".metric .k{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.05em}"
           ".sample{background:#fff;border-radius:8px;padding:14px;margin-bottom:14px;"
           "box-shadow:0 1px 3px rgba(0,0,0,.08)}"
           ".sample h3{margin:0 0 8px 0;font-size:15px}"
           ".row{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin-top:8px}"
           ".panel{border:1px solid #ddd;padding:6px;border-radius:4px;background:#fafafa;"
           "text-align:center;font-size:11px}"
           ".panel .lbl{font-weight:600;margin-bottom:4px;color:#444}"
           ".panel img{display:block;max-width:220px;height:auto;border:1px solid #aaa;margin:0 auto}"
           "table{border-collapse:collapse;margin:6px 0;font-size:12px;width:100%}"
           "td,th{padding:4px 8px;border:1px solid #ddd;text-align:left;vertical-align:top}"
           "th{background:#eef}"
           ".ok{color:#15803d;font-weight:600}"
           ".ko{color:#b91c1c;font-weight:600}"
           ".err-bad{background:#fee2e2}"
           ".err-warn{background:#fef3c7}"
           ".err-good{background:#dcfce7}"
           ".swatch{display:inline-block;width:14px;height:14px;border:1px solid #999;"
           "vertical-align:middle;margin-right:4px;border-radius:2px}")
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
             f"<title>{title}</title><style>{css}</style></head><body>"]
    parts.append(f"<h1>{title}</h1>")
    parts.append("<div class='summary'><div class='grid'>")

    def _m(k, v):
        return f"<div class='metric'><div class='k'>{k}</div><div class='v'>{v}</div></div>"

    parts.append(_m("Samples", n))
    parts.append(_m("Modelo accuracy", f"{n_correct}/{n} ({accuracy:.1f}%)"))
    parts.append(_m("Color cenital OK",
                    f"{n_color_cen_ok}/{n} ({100.0*n_color_cen_ok/max(n,1):.1f}%)"))
    parts.append(_m("Color lateral OK",
                    f"{n_color_lat_ok}/{n} ({100.0*n_color_lat_ok/max(n,1):.1f}%)"))
    if surf_errs:
        parts.append(_m("|Err superficie| medio",
                        f"{sum(surf_errs)/len(surf_errs):.1f}%"))
    if h_errs:
        parts.append(_m("|Err altura lat.| medio",
                        f"{sum(h_errs)/len(h_errs):.1f}%"))
    parts.append("</div></div>")

    parts.append("<h2>Samples</h2>")
    for s in samples:
        sample_idx = s.get("sample_index", s.get("index"))
        ref_pred = s.get("ref_inferred")
        ok_model = s.get("model_match")
        score = s.get("consensus_score")
        gt_cls = "ok" if ok_model else "ko"
        gt_status = "✓" if ok_model else "✗"

        cen_file = s.get("cenital_file")
        lat_file = s.get("lateral_file")
        cen_path = os.path.join(data_dir, cen_file) if cen_file else None
        lat_path = os.path.join(data_dir, lat_file) if lat_file else None

        bbox_cen = meta_lookup.get((sample_idx, "cenital"))
        bbox_lat = meta_lookup.get((sample_idx, "lateral"))

        parts.append("<div class='sample'>")
        parts.append(
            f"<h3>Sample {sample_idx} — GT={ref} pose={s.get('pose_index_gt')} "
            f"face={s.get('face_class_gt')} | Pred=<span class='{gt_cls}'>{ref_pred}</span> "
            f"{gt_status} score={score}</h3>"
        )

        cen_img = None; lat_img = None
        if cen_path and os.path.isfile(cen_path):
            try: cen_img = Image.open(cen_path).convert("RGB")
            except Exception: pass
        if lat_path and os.path.isfile(lat_path):
            try: lat_img = Image.open(lat_path).convert("RGB")
            except Exception: pass

        parts.append("<div class='row'>")
        if cen_img and bbox_cen:
            mask_c = sam_mask(cen_img, bbox_cen)
            cen_bb = draw_bbox(cen_img, bbox_cen)
            cen_ov = overlay_mask(cen_img, mask_c, color=(0, 255, 0))
            cen_can = crop_with_cinta_bg(cen_img, mask_c, bbox_cen)
            parts.append(
                f"<div class='panel'><div class='lbl'>CEN orig+bbox</div>"
                f"<img src='{to_b64(cen_bb)}'/></div>"
                f"<div class='panel'><div class='lbl'>CEN SAM mask</div>"
                f"<img src='{to_b64(cen_ov)}'/></div>"
                f"<div class='panel'><div class='lbl'>CEN canvas DINOv2</div>"
                f"<img src='{to_b64(cen_can, max_side=224)}'/></div>"
            )
        if lat_img and bbox_lat:
            mask_l = sam_mask(lat_img, bbox_lat)
            lat_bb = draw_bbox(lat_img, bbox_lat)
            lat_ov = overlay_mask(lat_img, mask_l, color=(0, 255, 0))
            lat_can = crop_with_cinta_bg(lat_img, mask_l, bbox_lat)
            parts.append(
                f"<div class='panel'><div class='lbl'>LAT orig+bbox</div>"
                f"<img src='{to_b64(lat_bb)}'/></div>"
                f"<div class='panel'><div class='lbl'>LAT SAM mask</div>"
                f"<img src='{to_b64(lat_ov)}'/></div>"
                f"<div class='panel'><div class='lbl'>LAT canvas DINOv2</div>"
                f"<img src='{to_b64(lat_can, max_side=224)}'/></div>"
            )
        parts.append("</div>")

        gt_color_hex = (s.get("color_hex_gt") or "")
        cen_rgb = s.get("color_cenital_rgb_est") or ""
        lat_rgb = s.get("color_lateral_rgb_est") or ""
        cen_norm = f"{s.get('color_cenital_normalized_code')} ({s.get('color_cenital_normalized_name')})"
        lat_norm = f"{s.get('color_lateral_normalized_code')} ({s.get('color_lateral_normalized_name')})"
        cm_c = "<span class='ok'>OK</span>" if s.get("color_match_cenital") else "<span class='ko'>NO</span>"
        cm_l = "<span class='ok'>OK</span>" if s.get("color_match_lateral") else "<span class='ko'>NO</span>"

        parts.append("<table><thead><tr>")
        parts.append("<th>Magnitud</th><th>GT</th><th>Cenital obs</th>"
                     "<th>Lateral obs</th><th>Match / Err</th></tr></thead><tbody>")
        parts.append(
            f"<tr><td>color (code)</td>"
            f"<td><span class='swatch' style='background:{gt_color_hex}'></span>"
            f"{s.get('color_code_gt')} ({s.get('color_name_gt')})</td>"
            f"<td>RGB={cen_rgb} → {cen_norm}</td>"
            f"<td>RGB={lat_rgb} → {lat_norm}</td>"
            f"<td>cen={cm_c} | lat={cm_l}</td></tr>"
        )
        surf_err = s.get("surface_error_rel_pct")
        parts.append(
            f"<tr><td>superficie cenital (mm²)</td>"
            f"<td>db={s.get('surface_db_silhouette_mm2')}</td>"
            f"<td>apparent={s.get('surface_obs_apparent_mm2')} | "
            f"footprint={s.get('surface_obs_footprint_mm2')}</td>"
            f"<td>—</td>"
            f"<td class='{_err_class(surf_err)}'>err={surf_err}%</td></tr>"
        )
        h_err = s.get("lateral_height_error_rel_pct")
        parts.append(
            f"<tr><td>altura lateral (mm)</td>"
            f"<td>db={s.get('lateral_height_db_mm')}</td>"
            f"<td>—</td>"
            f"<td>meas={s.get('lateral_height_meas_mm')}</td>"
            f"<td class='{_err_class(h_err)}'>err={h_err}%</td></tr>"
        )
        parts.append(
            f"<tr><td>YOLO conf / gating</td>"
            f"<td>—</td>"
            f"<td>conf={s.get('yolo_conf_cenital')} | "
            f"v_color={s.get('valid_by_color_count')} v_surf={s.get('valid_by_surface_count')} "
            f"v_h={s.get('valid_by_height_count')}</td>"
            f"<td>conf={s.get('yolo_conf_lateral')}</td>"
            f"<td>—</td></tr>"
        )
        parts.append("</tbody></table>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default=None,
                        help="Pieza unica a reportar (ej. 32054).")
    parser.add_argument("--color_code", type=str, default=None,
                        help="Filtro opcional por color_code (ej. 11).")
    parser.add_argument("--pieces", nargs="+", default=None,
                        help="Lista de 'ref:color' (ej. '32054:11 61184:86'). "
                             "Genera un HTML por entrada en --out_dir.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directorio con renders e infer report (default: data/300/).")
    parser.add_argument("--eval", type=str, default=DEFAULT_EVAL)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=str, default=None,
                        help="Archivo HTML de salida (solo --ref).")
    parser.add_argument("--max_samples", type=int, default=None)
    pa = parser.parse_args()

    eval_results = load_eval_results(pa.eval)
    meta = load_metadata(pa.metadata)
    meta_lookup = build_meta_lookup(meta)
    print(f"[piece_report] eval samples = {len(eval_results)} | "
          f"metadata bboxes = {len(meta_lookup)}")

    targets = []  # list of (ref, color_code)
    if pa.pieces:
        for tok in pa.pieces:
            if ":" in tok:
                r, c = tok.split(":", 1)
                targets.append((r, c))
            else:
                targets.append((tok, None))
    elif pa.ref:
        targets.append((pa.ref, pa.color_code))
    else:
        print("[ERROR] indica --ref o --pieces.")
        sys.exit(1)

    os.makedirs(pa.out_dir, exist_ok=True)
    for ref, color_code in targets:
        samples = select_samples(eval_results, ref, color_code, pa.max_samples)
        if not samples:
            print(f"[piece_report] {ref} (color {color_code}): SIN SAMPLES en eval.")
            continue
        if pa.out and pa.ref and not pa.pieces:
            out_path = pa.out
        else:
            cc = color_code or "all"
            out_path = os.path.join(pa.out_dir, f"piece_{ref}_{cc}.html")
        print(f"[piece_report] {ref} (color {color_code}): {len(samples)} samples → {out_path}")
        html = render_html(ref, color_code, samples, pa.data_dir, meta_lookup)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)


if __name__ == "__main__":
    main()
