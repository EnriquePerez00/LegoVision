# -*- coding: utf-8 -*-
"""projects/camara_domo/scripts/generate_piece_report.py
================================================================
Genera un reporte HTML standalone de diagnóstico premium para una pieza específica,
ejecutando predicciones de Características Topológicas (modelos timm) en la pose
más centrada en el FoV, y mostrando comparativas detalladas de Bounding Boxes (GT vs Inferred).
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import timm
from ultralytics import YOLO

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from core.db.supabase_client import get_connection
import generate_data100_report

CLASSES_FEATURES = [
    "stud_solid", "stud_hollow", "technic_hole_round", "technic_hole_cross",
    "clip_jaw", "bar_handle", "bottom_tube", "bottom_pin"
]


# ── Fuente única de verdad del color de la cinta (para placeholders) ────────
import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.abspath(_os.path.join(_HERE, '..', '..', '..')),
           _os.path.abspath(_os.path.join(_HERE, '..', '..', '..', 'scripts'))):
    if _p not in _sys.path: _sys.path.insert(0, _p)
try:
    from scripts.scene_config import BELT_COLOR_RGB_255 as _BELT_RGB
except ImportError:
    from scene_config import BELT_COLOR_RGB_255 as _BELT_RGB  # type: ignore

def preprocess_crop_grayscale(crop_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    gray_img = crop_img.convert("L")
    rgb_gray = Image.merge("RGB", (gray_img, gray_img, gray_img))
    margin = 8
    max_dim = canvas_size - 2 * margin
    w, h = rgb_gray.size
    if w > 0 and h > 0:
        scale = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = rgb_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
    return Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

def predict_features(crop, model, device):
    if crop is None or model is None:
        return [0.0] * len(CLASSES_FEATURES)
    try:
        processed = preprocess_crop_grayscale(crop, canvas_size=224)
        transform_feat = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform_feat(processed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()
        return probs
    except Exception as e:
        print(f"[Features Error] Error al inferir características: {e}")
        return [0.0] * len(CLASSES_FEATURES)

def get_topological_features_from_db(ldraw_id):
    features = {}
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT topological_features FROM lego_classes WHERE ldraw_id = %s;", (ldraw_id,))
                row = cur.fetchone()
                if row and row["topological_features"]:
                    val = row["topological_features"]
                    if isinstance(val, str):
                        features = json.loads(val)
                    elif isinstance(val, dict):
                        features = val
    except Exception as e:
        print(f"[DB Error] Error al leer características para {ldraw_id}: {e}")
    return features

def to_b64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def draw_bbox_with_tag(img_path, bbox_gt, bbox_inf, ref_gt, ref_inf, tag_color_gt=(16, 185, 129), tag_color_inf=(56, 189, 248)):
    """Dibuja la imagen completa con tags de identificación."""
    if not os.path.exists(img_path):
        return Image.new("RGB", (300, 300), tuple(_BELT_RGB))
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    
    # Intenta cargar una fuente
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    # Dibujar GT (Verde)
    if bbox_gt:
        gx1, gy1, gx2, gy2 = int(bbox_gt[0]*w), int(bbox_gt[1]*h), int(bbox_gt[2]*w), int(bbox_gt[3]*h)
        draw.rectangle([gx1, gy1, gx2, gy2], outline=tag_color_gt, width=3)
        # Dibujar etiqueta de fondo
        draw.rectangle([gx1, max(0, gy1-15), gx1+60, gy1], fill=tag_color_gt)
        draw.text((gx1+2, max(0, gy1-14)), f"GT:{ref_gt}", fill=(255, 255, 255), font=font)
        
    # Dibujar Pred (Celeste)
    if bbox_inf:
        ix1, iy1, ix2, iy2 = int(bbox_inf[0]*w), int(bbox_inf[1]*h), int(bbox_inf[2]*w), int(bbox_inf[3]*h)
        draw.rectangle([ix1, iy1, ix2, iy2], outline=tag_color_inf, width=3)
        # Dibujar etiqueta de fondo
        draw.rectangle([ix1, max(0, iy1-15), ix1+60, iy1], fill=tag_color_inf)
        draw.text((ix1+2, max(0, iy1-14)), f"Pred:{ref_inf}", fill=(255, 255, 255), font=font)
        
    return img

def main():
    parser = argparse.ArgumentParser(description="Genera reporte individual premium para diagnóstico de pieza.")
    parser.add_argument("--ref", type=str, required=True, help="ID LDraw de la pieza (ej. 2412b).")
    parser.add_argument("--color_code", type=str, default=None, help="Código de color opcional.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eval", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Cargar datos
    with open(args.eval, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(args.metadata, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
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

    # Cargar modelos timm para predicción on-the-fly
    features_cen_model = None
    features_lat_model = None
    cen_feat_path = os.path.join(project_root, "models", "features_cenital.pt")
    lat_feat_path = os.path.join(project_root, "models", "features_lateral.pt")

    if os.path.exists(cen_feat_path):
        ckpt = torch.load(cen_feat_path, map_location=device)
        model_name = ckpt.get('model_name', 'resnet18')
        features_cen_model = timm.create_model(model_name, num_classes=len(CLASSES_FEATURES))
        features_cen_model.load_state_dict(ckpt['model_state_dict'])
        features_cen_model.to(device)
        features_cen_model.eval()

    if os.path.exists(lat_feat_path):
        ckpt = torch.load(lat_feat_path, map_location=device)
        model_name = ckpt.get('model_name', 'resnet18')
        features_lat_model = timm.create_model(model_name, num_classes=len(CLASSES_FEATURES))
        features_lat_model.load_state_dict(ckpt['model_state_dict'])
        features_lat_model.to(device)
        features_lat_model.eval()

    # 2. Filtrar muestras pertenecientes a la ref
    matching_results = []
    for r in eval_data["results"]:
        if r["ref_gt"] == args.ref:
            matching_results.append(r)

    if not matching_results:
        print(f"[ERROR] No se encontraron resultados de inferencia para la pieza {args.ref}")
        sys.exit(1)

    # Buscar la trayectoria global en la simulación
    global_piece_tracks = {}
    for frame in meta_data["frames"]:
        f_name = frame["file_name"]
        offset = frame["belt_offset_mm"]
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

    # Encontrar la muestra más alineada con el centro (x_belt_local_mm más cercano a 0)
    best_result = None
    best_obs = None
    min_dist = float("inf")

    for r in matching_results:
        cenital_file_eval = r["cenital_file"]
        # Encontrar en el track
        matched_track = None
        for key, track_obs in global_piece_tracks.items():
            has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == args.ref for o in track_obs)
            if has_match:
                matched_track = track_obs
                break
        if matched_track:
            obs = min(matched_track, key=lambda x: abs(x["x_belt_local_mm"]))
            dist = abs(obs["x_belt_local_mm"])
            if dist < min_dist:
                min_dist = dist
                best_result = r
                best_obs = obs

    if best_result is None:
        best_result = matching_results[0]

    # Rutas de imágenes
    cen_file = best_obs["file_name"] if best_obs else best_result["cenital_file"]
    lat_file = best_obs["file_name"].replace(".png", "_frontal.png") if best_obs else best_result.get("lateral_file") or cen_file.replace(".png", "_frontal.png")

    path_cen_img = os.path.join(args.data_dir, cen_file)
    path_lat_img = os.path.join(args.data_dir, lat_file)

    # Buscar Bounding Boxes de Ground Truth
    bbox_cen_gt = best_obs["bbox_cenital_norm"] if best_obs else None
    bbox_lat_gt = best_obs["bbox_frontal_norm"] if best_obs else None

    # Intentar obtener Bounding Boxes Inferidos corriendo YOLO Cenital y Lateral en las imágenes
    yolo_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    yolo_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)

    bbox_cen_inf = None
    if os.path.exists(path_cen_img):
        res = yolo_cen(path_cen_img, verbose=False)[0]
        if len(res.boxes) > 0:
            # Buscar la caja con mayor IoU con el GT
            best_iou = 0.0
            best_box = None
            for box in res.boxes.xyxyn.cpu().numpy().tolist():
                if bbox_cen_gt:
                    iou = calculate_iou(bbox_cen_gt, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box
                else:
                    best_box = box
            bbox_cen_inf = best_box

    bbox_lat_inf = None
    if os.path.exists(path_lat_img):
        res = yolo_lat(path_lat_img, verbose=False)[0]
        if len(res.boxes) > 0:
            best_iou = 0.0
            best_box = None
            for box in res.boxes.xyxyn.cpu().numpy().tolist():
                if bbox_lat_gt:
                    iou = calculate_iou(bbox_lat_gt, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box
                else:
                    best_box = box
            bbox_lat_inf = best_box

    # Cargar imágenes completas y dibujar overlays con etiquetas
    img_cen_full = draw_bbox_with_tag(path_cen_img, bbox_cen_gt, bbox_cen_inf, args.ref, best_result["ref_inferred"])
    img_lat_full = draw_bbox_with_tag(path_lat_img, bbox_lat_gt, bbox_lat_inf, args.ref, best_result["ref_inferred"])

    # Extraer cultivos (crops) para visualización e inferencia
    crop_cen = None
    crop_lat = None
    if os.path.exists(path_cen_img) and bbox_cen_gt:
        img_cen_raw = Image.open(path_cen_img).convert("RGB")
        w_img, h_img = img_cen_raw.size
        cx1, cy1, cx2, cy2 = int(bbox_cen_gt[0]*w_img), int(bbox_cen_gt[1]*h_img), int(bbox_cen_gt[2]*w_img), int(bbox_cen_gt[3]*h_img)
        crop_cen = img_cen_raw.crop((cx1, cy1, cx2, cy2))
    else:
        crop_cen = Image.new("RGB", (100, 100), tuple(_BELT_RGB))

    if os.path.exists(path_lat_img) and bbox_lat_gt:
        img_lat_raw = Image.open(path_lat_img).convert("RGB")
        w_img, h_img = img_lat_raw.size
        lx1, ly1, lx2, ly2 = int(bbox_lat_gt[0]*w_img), int(bbox_lat_gt[1]*h_img), int(bbox_lat_gt[2]*w_img), int(bbox_lat_gt[3]*h_img)
        crop_lat = img_lat_raw.crop((lx1, ly1, lx2, ly2))
    else:
        crop_lat = Image.new("RGB", (100, 100), tuple(_BELT_RGB))

    # Generar Base64 para el HTML
    b64_cen_full = to_b64(img_cen_full)
    b64_lat_full = to_b64(img_lat_full)
    b64_crop_cen = to_b64(crop_cen)
    b64_crop_lat = to_b64(crop_lat)

    # Predecir Características Topológicas
    gt_feat = get_topological_features_from_db(args.ref)
    pred_cen = predict_features(crop_cen, features_cen_model, device)
    pred_lat = predict_features(crop_lat, features_lat_model, device)

    # Resolver Color GT
    color_name_gt = "Unknown"
    color_hex_gt = "#808080"
    for c in catalog_colors:
        if str(c["color_code"]) == str(best_result["color_code_gt"]):
            color_name_gt = c["color_name"]
            color_hex_gt = c["color_hex"]
            break

    # Resolver Color Inferido
    color_name_inf = best_result.get("color_name_cen", "Unknown")
    color_hex_inf = best_result.get("color_hex_cen", "#808080")
    for c in catalog_colors:
        if c["color_name"].strip().lower() == color_name_inf.strip().lower():
            color_hex_inf = c["color_hex"]
            break

    # Resolver Errores de Métricas
    part_match = best_result["model_match"]
    color_match = (color_name_gt.strip().lower() == color_name_inf.strip().lower())
    
    gt_area = best_result.get("surface_db_silhouette_mm2") or 0.0
    inf_area = best_result.get("apparent_area_mm2") or 0.0
    area_err = best_result.get("surface_error_rel_pct") or 0.0

    gt_height = best_result.get("lateral_height_db_mm") or 9.6
    inf_height = best_result.get("lateral_height_meas_mm") or 0.0
    height_err = best_result.get("lateral_height_error_rel_pct") or 0.0

    # Construir Tabla HTML de Características
    FEATURE_LABELS = {
        "stud_solid": "Espiga Sólida (stud_solid)",
        "stud_hollow": "Espiga Hueca (stud_hollow)",
        "technic_hole_round": "Agujero Redondo (technic_hole_round)",
        "technic_hole_cross": "Cruz Technic (technic_hole_cross)",
        "clip_jaw": "Pinza/Clip (clip_jaw)",
        "bar_handle": "Barra/Asa (bar_handle)",
        "bottom_tube": "Tubo Inferior (bottom_tube)",
        "bottom_pin": "Pin Inferior (bottom_pin)"
    }

    features_rows = ""
    for idx, (key, label) in enumerate(FEATURE_LABELS.items()):
        gt_has = gt_feat.get(key, 0) > 0
        p_cen = pred_cen[idx]
        p_lat = pred_lat[idx]
        
        gt_badge = '<span class="badge badge-ok">✓ Sí</span>' if gt_has else '<span class="badge badge-bad">✗ No</span>'
        cen_badge = f'<span class="badge badge-ok">✓ ({p_cen*100:.0f}%)</span>' if p_cen > 0.5 else f'<span class="badge badge-bad">✗ ({p_cen*100:.0f}%)</span>'
        lat_badge = f'<span class="badge badge-ok">✓ ({p_lat*100:.0f}%)</span>' if p_lat > 0.5 else f'<span class="badge badge-bad">✗ ({p_lat*100:.0f}%)</span>'
        
        match_status = "✓ MATCH" if (gt_has == (p_cen > 0.5 or p_lat > 0.5)) else "✗ DISCREPANCIA"
        match_class = "ok" if "✓" in match_status else "ko"
        
        features_rows += f"""
        <tr>
            <td><strong>{label}</strong></td>
            <td style="text-align: center;">{gt_badge}</td>
            <td style="text-align: center;">{cen_badge}</td>
            <td style="text-align: center;">{lat_badge}</td>
            <td style="text-align: center;" class="{match_class}"><strong>{match_status}</strong></td>
        </tr>
        """

    # Template HTML Premium
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Reporte de Diagnóstico Premium — Pieza {args.ref}</title>
    <style>
        :root {{
            --bg-color: #0b1329;
            --card-bg: #1c2541;
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --theme-color: #38bdf8;
            --ok-color: #10b981;
            --bad-color: #ef4444;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 30px;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
            color: var(--theme-color);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }}
        h2 {{
            font-size: 16px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 16px;
            color: var(--theme-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }}
        th {{
            color: var(--text-secondary);
            font-weight: 500;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-ok {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--ok-color);
        }}
        .badge-bad {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--bad-color);
        }}
        .ok {{ color: var(--ok-color); }}
        .ko {{ color: var(--bad-color); }}
        
        /* Contenedores de imagen con soporte Modal */
        .image-col {{
            text-align: center;
        }}
        .img-crop-btn {{
            cursor: pointer;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            max-width: 160px;
            transition: all 0.2s ease;
        }}
        .img-crop-btn:hover {{
            border-color: var(--theme-color);
            transform: scale(1.05);
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(11, 19, 41, 0.95);
            align-items: center;
            justify-content: center;
        }}
        .modal img {{
            max-width: 80%;
            max-height: 80%;
            border: 2px solid var(--theme-color);
            border-radius: 8px;
        }}
        .modal:target {{
            display: flex;
        }}
        .modal-close {{
            position: absolute;
            top: 20px; right: 20px;
            font-size: 32px;
            color: var(--text-primary);
            text-decoration: none;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Reporte de Diagnóstico Premium — Pieza LDraw {args.ref}</h1>
    
    <div class="grid">
        <!-- Panel Izquierdo: Métricas Comparativas y Características -->
        <div style="display: flex; flex-direction: column; gap: 24px;">
            <div class="card">
                <h2>Métricas Principales (GT vs Inferred)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Métrica</th>
                            <th>Ground Truth (Base de Datos)</th>
                            <th>Inferencia (Consenso)</th>
                            <th>Error / Desviación</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Clase de Pieza (LDraw)</strong></td>
                            <td>{args.ref}</td>
                            <td>{best_result["ref_inferred"]}</td>
                            <td>
                                <span class="badge {'badge-ok' if part_match else 'badge-bad'}">
                                    {'CORRECTO' if part_match else 'INCORRECTO'}
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Color Estimado</strong></td>
                            <td>
                                <span style="display:inline-block; width:12px; height:12px; border-radius:2px; background-color:{color_hex_gt}; vertical-align:middle; margin-right:4px;"></span>
                                {color_name_gt} ({best_result["color_code_gt"]})
                            </td>
                            <td>
                                <span style="display:inline-block; width:12px; height:12px; border-radius:2px; background-color:{color_hex_inf}; vertical-align:middle; margin-right:4px;"></span>
                                {color_name_inf}
                            </td>
                            <td>
                                <span class="badge {'badge-ok' if color_match else 'badge-bad'}">
                                    {'CORRECTO' if color_match else 'INCORRECTO'}
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Superficie Cenital</strong></td>
                            <td>{gt_area:.1f} mm²</td>
                            <td>{inf_area:.1f} mm²</td>
                            <td>
                                <span class="badge {'badge-ok' if abs(area_err) < 15 else 'badge-bad'}">
                                    {area_err:+.1f}%
                                </span>
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Altura Lateral</strong></td>
                            <td>{gt_height:.1f} mm</td>
                            <td>{inf_height:.1f} mm</td>
                            <td>
                                <span class="badge {'badge-ok' if abs(height_err) < 15 else 'badge-bad'}">
                                    {height_err:+.1f}%
                                </span>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Clasificación de Características Topológicas (timm on-the-fly)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Característica</th>
                            <th style="text-align: center;">GT (BD)</th>
                            <th style="text-align: center;">Pred Cenital</th>
                            <th style="text-align: center;">Pred Lateral</th>
                            <th style="text-align: center;">Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        {features_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Panel Derecho: UX de Imágenes y Bounding Boxes con Zoom al hacer Click -->
        <div class="card" style="display: flex; flex-direction: column; gap: 24px;">
            <h2>Visualización de Bounding Boxes (Crops & SAM Overlay)</h2>
            <p style="font-size: 12px; color: var(--text-secondary); margin-top: -10px;">
                Haz clic sobre cualquiera de los crops para abrir la imagen general del FoV con el tag de posición de la pieza.
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; text-align: center;">
                <div class="image-col">
                    <p style="font-weight: 600; margin-bottom: 8px;">Cenital Crop</p>
                    <a href="#modal-cenital">
                        <img class="img-crop-btn" src="data:image/jpeg;base64,{b64_crop_cen}" alt="Cenital Crop">
                    </a>
                </div>
                
                <div class="image-col">
                    <p style="font-weight: 600; margin-bottom: 8px;">Lateral Crop</p>
                    <a href="#modal-lateral">
                        <img class="img-crop-btn" src="data:image/jpeg;base64,{b64_crop_lat}" alt="Lateral Crop">
                    </a>
                </div>
            </div>

            <!-- Modales para ver la imagen completa -->
            <div id="modal-cenital" class="modal">
                <a href="#" class="modal-close">&times;</a>
                <img src="data:image/jpeg;base64,{b64_cen_full}" alt="Cenital Vista Completa">
            </div>

            <div id="modal-lateral" class="modal">
                <a href="#" class="modal-close">&times;</a>
                <img src="data:image/jpeg;base64,{b64_lat_full}" alt="Lateral Vista Completa">
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"[Report] HTML generado en: {args.out}")

if __name__ == "__main__":
    main()
