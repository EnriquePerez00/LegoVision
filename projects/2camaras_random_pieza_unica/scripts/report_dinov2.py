# -*- coding: utf-8 -*-
"""report_dinov2.py - Reporte DINOv2 con TOP-K configurable (default 2).

Para cada muestra generada por `generate_15_random_focus.py`, y para cada
camara (cenital, lateral):
  1. YOLO detecta el bbox.
  2. SAM extrae la mascara dentro del bbox.
  3. Se aplica la mascara al crop -> fondo negro (igual que las refs).
  4. Se calcula el embedding DINOv2 query.
  5. Se compara contra TODAS las refs de la camara correspondiente
     (sin filtrar por color/tamano). El score es coseno.
  6. Se seleccionan los TOP-K embeddings mas similares (default K=2).
  7. Se recupera el render PNG fisico de cada ref desde
     `data/dinov2_refs_v2/{cenital,lateral}/`.

Visual por sample (cenital arriba, lateral abajo), 4 columnas:
    [render+bbox YOLO]  [crop SAM]  [TOP-1 ref render]  [TOP-2 ref render]

Salida (en `<input_dir>`):
  - dinov2_report.json   datos crudos
  - dinov2_report.html   visualizacion
  - composites/composite_<NN>.png   un PNG por sample
  - masks/sample_<NN>_<cam>_mask.png   crops SAM

Uso:
    python 2camaras_random_pieza_unica/scripts/report_dinov2.py \
        --input_dir 2camaras_random_pieza_unica/data/random_focus_<ref>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from inference.knn_classifier import get_knn_classifier  # noqa: E402

REFS_DIR = os.path.join(project_root, "data", "dinov2_refs_v2")
TOP_K = 2


def parse_ref_filename(fname):
    m = re.match(
        r"^ref_([0-9A-Za-z]+)_([0-9A-Fa-f]{6})_pose(\d+)_rot(\d+)\.png$",
        fname,
    )
    if not m:
        return None
    return m.group(1), m.group(2).upper(), int(m.group(3)), int(m.group(4))


def yolo_detect_bbox(model, img_path, conf=0.25):
    try:
        results = model(img_path, verbose=False, conf=conf)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            bbox = boxes.xyxyn[best_idx].cpu().numpy().tolist()
            score = float(boxes.conf[best_idx].cpu().numpy())
            return bbox, score
    except Exception as e:
        print(f"[YOLO] WARN: {e}")
    return None, 0.0


def sam_mask_in_bbox(sam_model, img_full, bbox_norm):
    w, h = img_full.size
    x1 = max(0, int(bbox_norm[0] * w))
    y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w))
    y2 = min(h, int(bbox_norm[3] * h))
    bbox_px = [x1, y1, x2, y2]
    try:
        img_np = np.array(img_full)
        results = sam_model(img_np, bboxes=[bbox_px], verbose=False)
        if results and results[0].masks is not None:
            full = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            return full[y1:y2, x1:x2], (x1, y1, x2, y2)
    except Exception as e:
        print(f"[SAM] WARN: {e}")
    crop_h, crop_w = max(1, y2 - y1), max(1, x2 - x1)
    return np.ones((crop_h, crop_w), dtype=np.uint8) * 255, (x1, y1, x2, y2)


def apply_mask_to_crop(crop_pil, mask):
    arr = np.array(crop_pil.convert("RGB"))
    h, w = arr.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) != (h, w):
        import cv2 as _cv2
        mask = _cv2.resize(mask, (w, h), interpolation=_cv2.INTER_NEAREST)
    arr[mask == 0] = (0, 0, 0)
    return Image.fromarray(arr)


def build_clean_canvas(crop_img, canvas_size=224, scale_factor=208.0 / 640.0):
    w_p, h_p = crop_img.size
    if w_p > 0 and h_p > 0:
        new_w = max(1, int(w_p * scale_factor))
        new_h = max(1, int(h_p * scale_factor))
        resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
        return canvas
    return crop_img


def get_oriented_dims_mm_from_mask(mask, px_per_mm=3.2):
    try:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]
        if valid:
            all_pts = np.vstack(valid)
            rect = cv2.minAreaRect(all_pts)
            (_, _), (w_px, h_px), _ = rect
            return max(w_px, h_px) / px_per_mm, min(w_px, h_px) / px_per_mm
    except Exception:
        pass
    return mask.shape[1] / px_per_mm, mask.shape[0] / px_per_mm


def draw_bbox_on_img(img_pil, bbox_norm, color=(255, 0, 0), width=4, label=None):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        tx, ty = x1 + 4, max(0, y1 - 22)
        draw.rectangle([tx - 2, ty - 2, tx + 200, ty + 18], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)
    return img


def composite_row(thumbs_with_labels, row_height=260, col_width=240,
                  pad=8, header=None):
    n = len(thumbs_with_labels)
    img_h = row_height + (28 if header else 0)
    img_w = n * col_width + pad * (n + 1)
    canvas = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font_h = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_l = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font_h = ImageFont.load_default()
        font_l = ImageFont.load_default()
    y_offset = 0
    if header:
        draw.text((pad, 4), header, fill=(0, 0, 0), font=font_h)
        y_offset = 28
    for i, (thumb, label) in enumerate(thumbs_with_labels):
        x = pad + i * (col_width + pad)
        slot_h = row_height - 32
        draw.rectangle([x, y_offset, x + col_width, y_offset + slot_h],
                       fill=(255, 255, 255), outline=(180, 180, 180))
        if thumb is not None:
            t = thumb.copy()
            t.thumbnail((col_width - 8, slot_h - 8), Image.Resampling.LANCZOS)
            tx = x + (col_width - t.width) // 2
            ty = y_offset + (slot_h - t.height) // 2
            canvas.paste(t, (tx, ty))
        if label:
            ly = y_offset + slot_h + 2
            draw.text((x + 4, ly), label, fill=(20, 20, 20), font=font_l)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--refs_dir", default=REFS_DIR)
    parsed = parser.parse_args()

    input_dir = parsed.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.abspath(input_dir)

    meta_path = os.path.join(input_dir, "random_focus_metadata.json")
    if not os.path.isfile(meta_path):
        print(f"[ERROR] No se encuentra {meta_path}")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    gt_ref = meta.get("ref")
    gt_color = meta.get("color_hex", "")
    gt_color_name = meta.get("color_name", "")
    print(f"[Report] Pieza GT: {gt_ref}  Color: {gt_color_name} ({gt_color})")
    print(f"[Report] Samples: {len(meta.get('renders', []))}  Top-K: {parsed.top_k}")

    from ultralytics import YOLO, SAM
    yolo_cen_path = os.path.join(project_root, "models", "yolo_cenital.pt")
    yolo_lat_path = os.path.join(project_root, "models", "yolo_lateral.pt")
    if not (os.path.isfile(yolo_cen_path) and os.path.isfile(yolo_lat_path)):
        print(f"[ERROR] Faltan modelos YOLO en {project_root}/models/")
        sys.exit(1)
    yolo_cen = YOLO(yolo_cen_path)
    yolo_lat = YOLO(yolo_lat_path)
    sam_model = SAM("mobile_sam.pt")

    print("[Report] Cargando KNN/DINOv2...")
    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()
    if not clf.is_ready():
        print("[ERROR] KNN no listo")
        sys.exit(1)

    refs_cen = [r for r in clf._ref_embeddings if (r["face"] % 10 == 1)]
    refs_lat = [r for r in clf._ref_embeddings if (r["face"] % 10 == 2)]
    print(f"[Report] refs cenital DB: {len(refs_cen)} | refs lateral DB: {len(refs_lat)}")

    refs_dir = parsed.refs_dir
    cen_files_dir = os.path.join(refs_dir, "cenital")
    lat_files_dir = os.path.join(refs_dir, "lateral")

    def find_ref_png(part_ref, pose, angle, camera):
        d = cen_files_dir if camera == "cenital" else lat_files_dir
        if not os.path.isdir(d):
            return None
        prefix = f"ref_{part_ref}_"
        suffix = f"_pose{int(pose):02d}_rot{int(angle):03d}.png"
        for fn in os.listdir(d):
            if fn.startswith(prefix) and fn.endswith(suffix):
                return os.path.join(d, fn)
        return None

    sample_results = []
    out_masks_dir = os.path.join(input_dir, "masks")
    os.makedirs(out_masks_dir, exist_ok=True)
    out_composites_dir = os.path.join(input_dir, "composites")
    os.makedirs(out_composites_dir, exist_ok=True)

    for entry in meta.get("renders", []):
        idx = entry.get("index")
        cd = entry.get("cameras", {})
        cm = cd.get("cenital")
        lm = cd.get("lateral")
        if not (cm and lm):
            continue
        cen_path = os.path.join(input_dir, cm["file_name"])
        lat_path = os.path.join(input_dir, lm["file_name"])
        if not (os.path.isfile(cen_path) and os.path.isfile(lat_path)):
            print(f"[Report] sample {idx}: faltan archivos, saltado")
            continue
        print(f"\n[{idx:02d}] Procesando sample {idx}...")
        img_cen = Image.open(cen_path).convert("RGB")
        img_lat = Image.open(lat_path).convert("RGB")
        bbox_cen, conf_cen = yolo_detect_bbox(yolo_cen, cen_path, conf=0.20)
        bbox_lat, conf_lat = yolo_detect_bbox(yolo_lat, lat_path, conf=0.20)
        if bbox_cen is None:
            bbox_cen = cm.get("bbox_norm"); conf_cen = 0.0
        if bbox_lat is None:
            bbox_lat = lm.get("bbox_norm"); conf_lat = 0.0
        mask_cen, bbox_px_cen = sam_mask_in_bbox(sam_model, img_cen, bbox_cen)
        mask_lat, bbox_px_lat = sam_mask_in_bbox(sam_model, img_lat, bbox_lat)
        cx1, cy1, cx2, cy2 = bbox_px_cen
        lx1, ly1, lx2, ly2 = bbox_px_lat
        crop_cen = img_cen.crop((cx1, cy1, cx2, cy2))
        crop_lat = img_lat.crop((lx1, ly1, lx2, ly2))
        crop_cen_masked = apply_mask_to_crop(crop_cen, mask_cen)
        crop_lat_masked = apply_mask_to_crop(crop_lat, mask_lat)
        mask_cen_png = os.path.join(out_masks_dir, f"sample_{idx:02d}_cenital_mask.png")
        mask_lat_png = os.path.join(out_masks_dir, f"sample_{idx:02d}_lateral_mask.png")
        crop_cen_masked.save(mask_cen_png)
        crop_lat_masked.save(mask_lat_png)
        canvas_cen = build_clean_canvas(crop_cen_masked)
        canvas_lat = build_clean_canvas(crop_lat_masked)
        max_cen, min_cen = get_oriented_dims_mm_from_mask(mask_cen)
        max_lat, min_lat = get_oriented_dims_mm_from_mask(mask_lat)
        q_cen = clf._extract_embedding(canvas_cen, size_info=(max_cen, min_cen))
        q_lat = clf._extract_embedding(canvas_lat, size_info=(max_lat, min_lat))
        if refs_cen:
            mat_cen = np.stack([r["embedding"] for r in refs_cen])
            sims_cen = mat_cen @ q_cen
            top_idx_cen = np.argsort(sims_cen)[::-1][:parsed.top_k]
            top_cen = [{"part_ref": refs_cen[i]["part_ref"], "face": int(refs_cen[i]["face"]),
                        "angle": int(refs_cen[i]["angle"]), "color_hex": refs_cen[i].get("color_hex"),
                        "score": float(sims_cen[i])} for i in top_idx_cen]
        else:
            top_cen = []
        if refs_lat:
            mat_lat = np.stack([r["embedding"] for r in refs_lat])
            sims_lat = mat_lat @ q_lat
            top_idx_lat = np.argsort(sims_lat)[::-1][:parsed.top_k]
            top_lat = [{"part_ref": refs_lat[i]["part_ref"], "face": int(refs_lat[i]["face"]),
                        "angle": int(refs_lat[i]["angle"]), "color_hex": refs_lat[i].get("color_hex"),
                        "score": float(sims_lat[i])} for i in top_idx_lat]
        else:
            top_lat = []
        for t in top_cen:
            pose = t["face"] // 10
            t["pose"] = pose
            t["png"] = find_ref_png(t["part_ref"], pose, t["angle"], "cenital")
        for t in top_lat:
            pose = t["face"] // 10
            t["pose"] = pose
            t["png"] = find_ref_png(t["part_ref"], pose, t["angle"], "lateral")
        gt_marker_cen = " (GT)" if any(t["part_ref"] == gt_ref for t in top_cen) else ""
        gt_marker_lat = " (GT)" if any(t["part_ref"] == gt_ref for t in top_lat) else ""
        cen_thumbs = [
            (draw_bbox_on_img(img_cen, bbox_cen, color=(255, 80, 80), label=f"GT {gt_ref}"),
             f"render cenital (yolo {conf_cen:.2f})"),
            (crop_cen_masked, "SAM crop"),
        ]
        for k, t in enumerate(top_cen):
            label = f"top{k+1}: {t['part_ref']} pose{t['pose']:02d} s={t['score']:.3f}"
            if t["part_ref"] == gt_ref:
                label = "* " + label
            png = t.get("png")
            thumb = Image.open(png).convert("RGB") if png and os.path.isfile(png) else None
            cen_thumbs.append((thumb, label))
        lat_thumbs = [
            (draw_bbox_on_img(img_lat, bbox_lat, color=(80, 80, 255), label=f"GT {gt_ref}"),
             f"render lateral (yolo {conf_lat:.2f})"),
            (crop_lat_masked, "SAM crop"),
        ]
        for k, t in enumerate(top_lat):
            label = f"top{k+1}: {t['part_ref']} pose{t['pose']:02d} s={t['score']:.3f}"
            if t["part_ref"] == gt_ref:
                label = "* " + label
            png = t.get("png")
            thumb = Image.open(png).convert("RGB") if png and os.path.isfile(png) else None
            lat_thumbs.append((thumb, label))
        row_cen = composite_row(cen_thumbs,
            header=f"Sample {idx:02d} . CENITAL . GT={gt_ref} ({gt_color_name}){gt_marker_cen}")
        row_lat = composite_row(lat_thumbs,
            header=f"Sample {idx:02d} . LATERAL . GT={gt_ref} ({gt_color_name}){gt_marker_lat}")
        comp_w = max(row_cen.width, row_lat.width)
        comp_h = row_cen.height + row_lat.height + 10
        comp = Image.new("RGB", (comp_w, comp_h), (230, 230, 230))
        comp.paste(row_cen, (0, 0))
        comp.paste(row_lat, (0, row_cen.height + 10))
        comp_path = os.path.join(out_composites_dir, f"composite_{idx:02d}.png")
        comp.save(comp_path)
        sample_results.append({
            "index": idx, "ref_gt": gt_ref, "color_hex_gt": gt_color,
            "cenital": {"file_name": cm["file_name"],
                        "yolo_bbox_norm": list(bbox_cen) if bbox_cen else None,                        "yolo_conf": float(conf_cen),
                        "bbox_px": list(bbox_px_cen),
                        "mask_png": os.path.relpath(mask_cen_png, input_dir),
                        "top": top_cen,
                        "gt_in_top": any(t["part_ref"] == gt_ref for t in top_cen)},
            "lateral": {"file_name": lm["file_name"],
                        "yolo_bbox_norm": list(bbox_lat) if bbox_lat else None,
                        "yolo_conf": float(conf_lat),
                        "bbox_px": list(bbox_px_lat),
                        "mask_png": os.path.relpath(mask_lat_png, input_dir),
                        "top": top_lat,
                        "gt_in_top": any(t["part_ref"] == gt_ref for t in top_lat)},
            "composite_png": os.path.relpath(comp_path, input_dir),
        })
        print(
            f"   cenital top1={top_cen[0]['part_ref']} s={top_cen[0]['score']:.3f} | "
            f"lateral top1={top_lat[0]['part_ref']} s={top_lat[0]['score']:.3f} | "
            f"GT_in_top: cen={'Y' if sample_results[-1]['cenital']['gt_in_top'] else 'N'} "
            f"lat={'Y' if sample_results[-1]['lateral']['gt_in_top'] else 'N'}"
        )

    n = len(sample_results)
    gt_in_top_cen = sum(1 for s in sample_results if s["cenital"]["gt_in_top"])
    gt_in_top_lat = sum(1 for s in sample_results if s["lateral"]["gt_in_top"])
    gt_top1_cen = sum(1 for s in sample_results
                      if s["cenital"]["top"] and s["cenital"]["top"][0]["part_ref"] == gt_ref)
    gt_top1_lat = sum(1 for s in sample_results
                      if s["lateral"]["top"] and s["lateral"]["top"][0]["part_ref"] == gt_ref)

    summary = {
        "set_id": meta.get("set_id"),
        "ref_gt": gt_ref,
        "color_hex_gt": gt_color,
        "color_name_gt": gt_color_name,
        "samples": n,
        "top_k": parsed.top_k,
        "stats": {
            "cenital_gt_in_top": gt_in_top_cen,
            "cenital_gt_top1": gt_top1_cen,
            "lateral_gt_in_top": gt_in_top_lat,
            "lateral_gt_top1": gt_top1_lat,
        },
        "results": sample_results,
    }
    json_path = os.path.join(input_dir, "dinov2_report.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)
    print(f"\n[Report] JSON: {json_path}")

    rows_html = []
    for s in sample_results:
        idx = s["index"]
        comp_rel = s["composite_png"].replace(os.sep, "/")
        cen_top = s["cenital"]["top"]
        lat_top = s["lateral"]["top"]
        cen_top_html = " | ".join(
            f"<b>{t['part_ref']}</b> pose{t['pose']:02d} (s={t['score']:.3f})"
            for t in cen_top)
        lat_top_html = " | ".join(
            f"<b>{t['part_ref']}</b> pose{t['pose']:02d} (s={t['score']:.3f})"
            for t in lat_top)
        rows_html.append(f"""
        <div class="sample">
          <h3>Sample {idx:02d}
            &nbsp;&middot;&nbsp; cenital top: {cen_top_html}
            &nbsp;&middot;&nbsp; lateral top: {lat_top_html}
          </h3>
          <img src="{comp_rel}" alt="composite {idx}" />
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>DINOv2 Report &mdash; {gt_ref}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #555; margin-bottom: 18px; }}
  .stats {{ background: #f4f4f7; padding: 10px 14px; border-radius: 6px;
            margin-bottom: 18px; display: inline-block; }}
  .sample {{ margin-bottom: 28px; padding: 12px;
             border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }}
  .sample h3 {{ margin: 4px 0 10px 0; font-size: 14px; font-weight: 600; }}
  .sample img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>DINOv2 Report</h1>
<p class="meta">
  Pieza GT: <b>{gt_ref}</b> &nbsp;&middot;&nbsp;
  Color: <b>{gt_color_name}</b> ({gt_color}) &nbsp;&middot;&nbsp;
  Samples: <b>{n}</b> &nbsp;&middot;&nbsp;
  Top-K: <b>{parsed.top_k}</b>
</p>
<div class="stats">
  <b>Stats</b><br/>
  GT en top-{parsed.top_k} cenital: {gt_in_top_cen}/{n}
    &nbsp;|&nbsp; GT top-1 cenital: {gt_top1_cen}/{n}<br/>
  GT en top-{parsed.top_k} lateral: {gt_in_top_lat}/{n}
    &nbsp;|&nbsp; GT top-1 lateral: {gt_top1_lat}/{n}
</div>
{''.join(rows_html)}
</body>
</html>
"""
    html_path = os.path.join(input_dir, "dinov2_report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[Report] HTML: {html_path}")

    print("\n" + "=" * 60)
    print(f"  RESUMEN DINOv2 - {gt_ref} ({gt_color_name})")
    print("=" * 60)
    print(f"  Samples procesados: {n}")
    print(f"  GT in top-{parsed.top_k} cenital: {gt_in_top_cen}/{n}")
    print(f"  GT top-1 cenital   : {gt_top1_cen}/{n}")
    print(f"  GT in top-{parsed.top_k} lateral: {gt_in_top_lat}/{n}")
    print(f"  GT top-1 lateral   : {gt_top1_lat}/{n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
