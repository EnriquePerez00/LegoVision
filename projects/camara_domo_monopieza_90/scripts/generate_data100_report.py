import os
import sys
import json

# Add virtual environment site-packages to sys.path so Blender Python can find installed packages like torch, numpy, etc.
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Add imports for database
project_root_abs = "/Users/I764690/Code_personal/LegoVision"
if project_root_abs not in sys.path:
    sys.path.insert(0, project_root_abs)
if os.path.join(project_root_abs, "core", "db") not in sys.path:
    sys.path.insert(0, os.path.join(project_root_abs, "core", "db"))
from supabase_client import get_connection

_db_conn = None

def get_db_connection():
    global _db_conn
    if _db_conn is None or getattr(_db_conn, "closed", True):
        _db_conn = get_connection()
    return _db_conn

_topological_features_cache = {}

def get_topological_features(ldraw_id):
    if not ldraw_id:
        return {}
    if ldraw_id in _topological_features_cache:
        return _topological_features_cache[ldraw_id]
    
    features = {}
    try:
        conn = get_db_connection()
        if hasattr(conn, "cursor") and conn.__class__.__name__ != "MockConnection":
            with conn.cursor() as cur:
                cur.execute("SELECT topological_features FROM lego_classes WHERE ldraw_id = %s;", (ldraw_id,))
                row = cur.fetchone()
                if row and "topological_features" in row and row["topological_features"]:
                    val = row["topological_features"]
                    if isinstance(val, str):
                        features = json.loads(val)
                    elif isinstance(val, dict):
                        features = val
    except Exception as e:
        print(f"Error fetching features for {ldraw_id}: {e}")
        
    _topological_features_cache[ldraw_id] = features
    return features

def format_features_html(features, is_gt=True):
    FEATURE_LABELS = {
        "stud_solid": "Espiga Sólida",
        "stud_hollow": "Espiga Hueca",
        "technic_hole_round": "Agujero Redondo",
        "technic_hole_cross": "Cruz Technic",
        "clip_jaw": "Clip/Pinza",
        "bar_handle": "Barra/Asa",
        "bottom_tube": "Tubo Inferior",
        "bottom_pin": "Pin Inferior"
    }
    
    html = '<div style="margin-top: 10px; border-top: 1px dashed var(--border-color); padding-top: 8px;">'
    html += '<div style="font-size: 11px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">Topología (8 características):</div>'
    html += '<div style="font-size: 9px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px;">'
    
    for key, label in FEATURE_LABELS.items():
        val = features.get(key, 0) if features else 0
        if val > 0:
            # Present
            bg_color = "rgba(16, 185, 129, 0.15)" if is_gt else "rgba(56, 189, 248, 0.15)"
            border_color = "rgba(16, 185, 129, 0.3)" if is_gt else "rgba(56, 189, 248, 0.3)"
            text_color = "#34d399" if is_gt else "#38bdf8"
            icon = "✓"
            count_str = f": {val}"
            font_weight = "bold"
        else:
            # Absent
            bg_color = "rgba(255, 255, 255, 0.02)"
            border_color = "rgba(255, 255, 255, 0.05)"
            text_color = "#64748b"
            icon = "✗"
            count_str = ""
            font_weight = "normal"
            
        html += f'<div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 4px; padding: 3px 5px; color: {text_color}; display: flex; align-items: center; gap: 3px; font-weight: {font_weight}; white-space: nowrap;" title="{label}">'
        html += f'<span style="font-size: 10px; font-weight: bold;">{icon}</span> {label}{count_str}'
        html += '</div>'
        
    html += '</div></div>'
    return html


def compute_iou(boxA, boxB):
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def draw_overlay(image_path, bbox_gt, bbox_inf, ref_gt, ref_inf, score, iou):
    if not os.path.exists(image_path):
        return None
        
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    try:
        font = ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()

    # 1. Dibujar Bounding Box Ground Truth (Verde)
    if bbox_gt:
        gx1, gy1, gx2, gy2 = bbox_gt
        gx1_px, gy1_px, gx2_px, gy2_px = int(gx1*w), int(gy1*h), int(gx2*w), int(gy2*h)
        draw.rectangle([gx1_px, gy1_px, gx2_px, gy2_px], outline="#10b981", width=4)
        
        # Etiqueta GT flotante en Verde
        draw.rectangle([gx1_px, max(0, gy1_px-20), gx1_px+120, gy1_px], fill="#10b981")
        draw.text((gx1_px+4, max(0, gy1_px-18)), f"GT: {ref_gt}", fill="white", font=font)

    # 2. Dibujar Bounding Box Inferido (Azul si IoU > 70% sino Rojo)
    if bbox_inf:
        ix1, iy1, ix2, iy2 = bbox_inf
        ix1_px, iy1_px, ix2_px, iy2_px = int(ix1*w), int(iy1*h), int(ix2*w), int(iy2*h)
        
        color = "#38bdf8" if iou > 0.70 else "#ef4444"
        draw.rectangle([ix1_px, iy1_px, ix2_px, iy2_px], outline=color, width=4)
        
        # Etiqueta INF flotante
        label_text = f"INF: {ref_inf} ({score:.2f}) IoU: {iou*100:.1f}%"
        draw.rectangle([ix1_px, max(0, iy1_px-20), ix1_px+240, iy1_px], fill=color)
        draw.text((ix1_px+4, max(0, iy1_px-18)), label_text, fill="black" if iou > 0.70 else "white", font=font)
        
    return img

def rgb_to_lab(rgb_val):
    r, g, b = rgb_val[0] / 255.0, rgb_val[1] / 255.0, rgb_val[2] / 255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.array([l_val, a_val, b_val])

def find_closest_color_code(rgb_est, catalog_colors):
    if not catalog_colors:
        return "0", "Various", "#808080"
    lab_est = rgb_to_lab(rgb_est)
    best_dist = float("inf")
    best_color = catalog_colors[0]
    for c in catalog_colors:
        lab_ref = rgb_to_lab(c["rgb"])
        dL = lab_est[0] - lab_ref[0]
        da = lab_est[1] - lab_ref[1]
        db = lab_est[2] - lab_ref[2]
        dist = np.sqrt(0.2 * (dL ** 2) + (da ** 2) + (db ** 2))
        if dist < best_dist:
            best_dist = dist
            best_color = c
    return best_color["color_code"], best_color["color_name"], best_color["color_hex"]

def estimate_color_bbox(img_path, bbox):
    if not bbox or not os.path.exists(img_path):
        return [128.0, 128.0, 128.0]
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [128.0, 128.0, 128.0]
    cropped = img.crop((x1, y1, x2, y2))
    arr = np.array(cropped)
    return list(arr.mean(axis=(0,1)))

def main():
    project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_75078"
    eval_json_path = os.path.join(project_root, "data", "reports", "new_weights_eval.html")
    metadata_path = os.path.join(project_root, "data", "data100", "simulation_metadata.json")
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    template_path = "/Users/I764690/Code_personal/LegoVision/projects/2camaras_pieza_unica/data100/report.html"
    output_dir = os.path.join(project_root, "data", "reports")
    output_html_path = os.path.join(output_dir, "data100_comparative_report.html")
    
    os.makedirs(os.path.join(output_dir, "visual_debug"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crops_debug"), exist_ok=True)

    print("Cargando JSON de evaluación y metadatos...")
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar paleta de color
    catalog_colors = []
    if os.path.exists(palette_path):
        with open(palette_path, "r", encoding="utf-8") as f:
            palette = json.load(f)
            for item in palette:
                catalog_colors.append({
                    "color_code": str(item.get("color_code", "")),
                    "color_name": item.get("color_name", "Unknown"),
                    "color_hex": item.get("color_hex", "#808080"),
                    "rgb": np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float)
                })

    # Cargar modelos YOLO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cargando modelos YOLO en {device}...")
    model_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    model_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)

    # Agrupar piezas por trayectoria global
    frames_list = meta_data.get("frames", [])
    global_piece_tracks = {}
    for frame in frames_list:
        offset = frame["belt_offset_mm"]
        f_name = frame["file_name"]
        for p in frame["visible_pieces"]:
            x_abs = p["x_belt_local_mm"] + offset
            y_abs = p["y_belt_local_mm"]
            
            matched_key = None
            for key in global_piece_tracks.keys():
                kx, ky = key
                if abs(x_abs - kx) < 1.5 and abs(y_abs - ky) < 1.5:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (x_abs, y_abs)
                global_piece_tracks[matched_key] = []
                
            global_piece_tracks[matched_key].append({
                "ref": p["ref"],
                "color_code": p["color_code"],
                "file_name": f_name,
                "bbox_cenital_norm": p["bbox_cenital_norm"],
                "bbox_frontal_norm": p.get("bbox_frontal_norm"),
                "x_belt_local_mm": p["x_belt_local_mm"],
                "y_belt_local_mm": p["y_belt_local_mm"]
            })

    # Cargar estructura de la plantilla HTML
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    header = template_content.split('<tbody>')[0] + '<tbody>\n'
    footer = '</tbody>' + template_content.split('</tbody>')[1]

    # Reemplazar métricas de la cabecera
    acc_pct = eval_data.get("accuracy", 0.0)
    total_s = eval_data.get("total_samples", len(eval_data.get("results", [])))
    correct_s = eval_data.get("correct_samples", 0)

    header = header.replace("1.00%", f"{acc_pct:.2f}%")
    header = header.replace("Total Correctas: 1 / 100", f"Total Correctas: {correct_s} / {total_s}")
    header = header.replace("190.3s", "N/A")
    header = header.replace("Media: 1903 ms / frame", "Dataset: data100 a 1024x1024")
    header = header.replace("Posición Y: 0 mm", "Vista Domo")
    header = header.replace("Correctos: 1/100", f"Correctos: {correct_s}/{total_s}")
    header = header.replace("width: 1.0%", f"width: {acc_pct:.1f}%")
    header = header.replace("1.0%", f"{acc_pct:.1f}%")
    header = header.replace("Reporte de Inferencia de Cinta Transportadora (Centrada)", "Reporte Inferencia Domo Multi-Vista — data100")
    header = header.replace("Evaluación de la posición más cercana al centro (Y = 0.0 mm) para las 100 piezas de la base de datos", 
                            "Análisis con overlays visuales de Bounding Boxes (GT en Verde vs YOLO en Azul/Rojo)")

    rows_html = ""
    print("Dibujando overlays y generando recortes para el UX...")
    
    for idx, r in enumerate(eval_data["results"]):
        ref_gt = r["ref_gt"]
        color_code_gt = r["color_code_gt"]
        ref_inf = r["ref_inferred"]
        score = r["consensus_score"]
        is_ok = r["model_match"]
        
        color_name_gt = "Unknown"
        for c in catalog_colors:
            if c["color_code"] == str(color_code_gt):
                color_name_gt = c["color_name"]
                break

        cenital_file_eval = r["cenital_file"]
        
        matched_track = None
        for key, track_obs in global_piece_tracks.items():
            has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt and str(o["color_code"]) == str(color_code_gt) for o in track_obs)
            if has_match:
                matched_track = track_obs
                break
        
        if not matched_track:
            for key, track_obs in global_piece_tracks.items():
                has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt for o in track_obs)
                if has_match:
                    matched_track = track_obs
                    break
        
        if matched_track:
            best_obs = min(matched_track, key=lambda x: abs(x["x_belt_local_mm"]))
            
            cen_file = best_obs["file_name"]
            lat_file = cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = best_obs["bbox_cenital_norm"]
            bbox_lat_gt = best_obs["bbox_frontal_norm"]
        else:
            cen_file = cenital_file_eval
            lat_file = r.get("lateral_file") or cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = None
            bbox_lat_gt = None

        path_cen_img = os.path.join(project_root, "data", "data100", cen_file)
        path_lat_img = os.path.join(project_root, "data", "data100", lat_file)

        bbox_cen_inf = None
        iou_cen = 0.0
        if os.path.exists(path_cen_img) and bbox_cen_gt:
            res_cen = model_cen(path_cen_img, verbose=False, conf=0.15, imgsz=1024)
            best_iou = 0.0
            best_box = None
            if res_cen and len(res_cen[0].boxes) > 0:
                for box in res_cen[0].boxes:
                    det_box = box.xyxyn[0].cpu().numpy().tolist()
                    iou = compute_iou(bbox_cen_gt, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = det_box
            if best_box and best_iou > 0.1:
                bbox_cen_inf = best_box
                iou_cen = best_iou

        bbox_lat_inf = None
        iou_lat = 0.0
        if os.path.exists(path_lat_img) and bbox_lat_gt:
            res_lat = model_lat(path_lat_img, verbose=False, conf=0.15, imgsz=1024)
            best_iou = 0.0
            best_box = None
            if res_lat and len(res_lat[0].boxes) > 0:
                for box in res_lat[0].boxes:
                    det_box = box.xyxyn[0].cpu().numpy().tolist()
                    iou = compute_iou(bbox_lat_gt, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = det_box
            if best_box and best_iou > 0.1:
                bbox_lat_inf = best_box
                iou_lat = best_iou

        name_cen_inf = r.get("color_name_cen", "Unknown")
        name_lat_inf = r.get("color_name_lat", "Unknown")
        
        if name_lat_inf == "Unknown" or name_lat_inf == "0":
            color_lat_rgb = estimate_color_bbox(path_lat_img, bbox_lat_inf or bbox_lat_gt)
            _, name_lat_inf, _ = find_closest_color_code(color_lat_rgb, catalog_colors)

        # COMPARACION ESTRICTA POR NOMBRE DIRECTO (Sin agrupaciones ni sinonimos)
        cen_color_match = (str(name_cen_inf).strip().lower() == str(color_name_gt).strip().lower())
        color_cen_html_color = "#10b981" if cen_color_match else "#ef4444"

        lat_color_match = (str(name_lat_inf).strip().lower() == str(color_name_gt).strip().lower())
        color_lat_html_color = "#10b981" if lat_color_match else "#ef4444"

        img_cen_overlay = draw_overlay(path_cen_img, bbox_cen_gt, bbox_cen_inf, ref_gt, ref_inf, score, iou_cen)
        img_lat_overlay = draw_overlay(path_lat_img, bbox_lat_gt, bbox_lat_inf, ref_gt, ref_inf, score, iou_lat)

        overlay_cen_out_name = f"visual_debug/sample_{idx:03d}_{cen_file.replace('.png', '_overlay.png')}"
        overlay_lat_out_name = f"visual_debug/sample_{idx:03d}_{lat_file.replace('.png', '_overlay.png')}"
        
        if img_cen_overlay:
            img_cen_overlay.save(os.path.join(output_dir, overlay_cen_out_name))
        if img_lat_overlay:
            img_lat_overlay.save(os.path.join(output_dir, overlay_lat_out_name))

        crop_cen_out_name = f"crops_debug/sample_{idx:03d}_{cen_file.replace('.png', '_crop.png')}"
        crop_lat_out_name = f"crops_debug/sample_{idx:03d}_{lat_file.replace('.png', '_crop.png')}"
        
        if img_cen_overlay and bbox_cen_gt:
            w_img, h_img = img_cen_overlay.size
            cx1, cy1, cx2, cy2 = int(bbox_cen_gt[0]*w_img), int(bbox_cen_gt[1]*h_img), int(bbox_cen_gt[2]*w_img), int(bbox_cen_gt[3]*h_img)
            cx1, cy1 = max(0, cx1-20), max(0, cy1-20)
            cx2, cy2 = min(w_img, cx2+20), min(h_img, cy2+20)
            crop_cen_img = img_cen_overlay.crop((cx1, cy1, cx2, cy2))
            crop_cen_img.save(os.path.join(output_dir, crop_cen_out_name))
            
        if img_lat_overlay and bbox_lat_gt:
            w_img, h_img = img_lat_overlay.size
            lx1, ly1, lx2, ly2 = int(bbox_lat_gt[0]*w_img), int(bbox_lat_gt[1]*h_img), int(bbox_lat_gt[2]*w_img), int(bbox_lat_gt[3]*h_img)
            lx1, ly1 = max(0, lx1-20), max(0, ly1-20)
            lx2, ly2 = min(w_img, lx2+20), min(h_img, ly2+20)
            crop_lat_img = img_lat_overlay.crop((lx1, ly1, lx2, ly2))
            crop_lat_img.save(os.path.join(output_dir, crop_lat_out_name))

        row_class = "row-ok" if is_ok else "row-bad"
        res_badge = '<span class="badge badge-ok">CORRECTO</span>' if is_ok else '<span class="badge badge-bad">FALLO</span>'

        apparent_area = r["surface_obs_apparent_mm2"]
        db_area = r.get("surface_db_silhouette_mm2") or 0.0
        area_err = r.get("surface_error_rel_pct", 0.0)
        
        meas_height = 0.0
        db_height = 0.0
        height_err = 0.0

        badge_area_class = "badge-ok" if abs(area_err) < 15 else ("badge-neutral" if abs(area_err) < 30 else "badge-bad")
        badge_height_class = "badge-neutral"

        badge_cen_iou_class = "badge-ok" if iou_cen > 0.70 else "badge-bad"
        badge_lat_iou_class = "badge-ok" if iou_lat > 0.70 else "badge-bad"

        # Get topological features from DB
        features_gt = get_topological_features(ref_gt)
        features_inf = get_topological_features(ref_inf)
        features_gt_html = format_features_html(features_gt, is_gt=True)
        features_inf_html = format_features_html(features_inf, is_gt=False)

        row_html = f"""
        <tr class="{row_class}">
            <td>#{idx+1:03d}</td>
            <td><strong>Domo</strong></td>
            <td>
                <div style="display: flex; gap: 8px;">
                    <img src="{crop_cen_out_name}" class="clickable-img" style="width: 100px; height: 100px; border-radius: 6px; border: 2px solid var(--border-color);" onclick="openModal('{overlay_cen_out_name}')" title="Cenital (Click para ampliar)">
                    <img src="{crop_lat_out_name}" class="clickable-img" style="width: 100px; height: 100px; border-radius: 6px; border: 2px solid var(--border-color);" onclick="openModal('{overlay_lat_out_name}')" title="Lateral (Click para ampliar)">
                </div>
            </td>
            <td>
                <div>Pieza: <strong>{ref_gt}</strong></div>
                <div style="font-size: 12px; color: var(--text-secondary);">Color: {color_name_gt}</div>
                <div style="font-size: 11px; margin-top: 4px;">
                    <span>Cenital GT: {db_area:.1f} mm²</span><br/>
                    <span>Altura GT: N/A mm</span>
                </div>
                {features_gt_html}
            </td>
            <td>
                <div>Pieza: <strong>{ref_inf}</strong></div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                    <span>Confianza: {score:.3f}</span><br/>
                    <span>Color Cen. Inf: <strong style="color: {color_cen_html_color};">{name_cen_inf}</strong></span><br/>
                    <span>Color Lat. Inf: <strong style="color: {color_lat_html_color};">{name_lat_inf}</strong></span>
                </div>
                {features_inf_html}
            </td>
            <td>
                <div>{apparent_area:.1f} mm²</div>
                <div style="margin-top: 4px;"><span class="badge {badge_area_class}">{area_err:+.1f}%</span></div>
            </td>
            <td>
                <div>N/A</div>
                <div style="margin-top: 4px;"><span class="badge {badge_height_class}">Desactivado</span></div>
            </td>
            <td>
                <div>Cenital IoU: <span class="badge {badge_cen_iou_class}">{iou_cen*100:.1f}%</span></div>
                <div style="margin-top: 4px;">Lateral IoU: <span class="badge {badge_lat_iou_class}">{iou_lat*100:.1f}%</span></div>
            </td>
            <td>{res_badge}</td>
        </tr>
        """
        rows_html += row_html

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(header + rows_html + footer)

    print(f"Reporte final con overlays completado y guardado en: {output_html_path}")

if __name__ == "__main__":
    main()
