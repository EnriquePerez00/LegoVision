# -*- coding: utf-8 -*-
"""projects/camara_domo_monopieza_90/scripts/generate_inference_report.py
=========================================================
Script para ejecutar la inferencia y evaluación completa de data200,
validar los clasificadores de características topológicas (timm models),
y generar el reporte comparativo HTML.
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import timm
from ultralytics import YOLO

# Configurar paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)

from core.db.supabase_client import get_connection
import generate_data100_report

CLASSES_FEATURES = [
    "stud_solid", "stud_hollow", "technic_hole_round", "technic_hole_cross",
    "clip_jaw", "bar_handle", "bottom_tube", "bottom_pin"
]

def preprocess_crop_grayscale(crop_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """Preprocesa el crop a escala de grises de 3 canales sobre fondo negro."""
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
    """Predice el vector de probabilidad de las 8 características topológicas."""
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
    """Consulta la topología de la base de datos local para Ground Truth."""
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

def format_features_comparison_html(gt_feat, pred_cen, pred_lat):
    """Genera una tabla HTML comparativa para el UX del reporte."""
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
    html += '<table style="width: 100%; border-collapse: collapse; font-size: 10px; text-align: left;">'
    html += '<thead><tr style="border-bottom: 1px solid var(--border-color); color: var(--text-secondary);">'
    html += '<th style="padding: 2px;">Característica</th>'
    html += '<th style="padding: 2px; text-align: center;">GT</th>'
    html += '<th style="padding: 2px; text-align: center;">Cenital</th>'
    html += '<th style="padding: 2px; text-align: center;">Frontal</th>'
    html += '</tr></thead><tbody>'
    
    for idx, (key, label) in enumerate(FEATURE_LABELS.items()):
        gt_has = gt_feat.get(key, 0) > 0
        p_cen = pred_cen[idx]
        p_lat = pred_lat[idx]
        
        gt_icon = '<span style="color: #10b981; font-weight: bold;">✓</span>' if gt_has else '<span style="color: #64748b;">✗</span>'
        
        cen_active = p_cen > 0.5
        cen_icon = f'<span style="color: #38bdf8; font-weight: bold;">✓ ({p_cen*100:.0f}%)</span>' if cen_active else f'<span style="color: #64748b;">✗ ({p_cen*100:.0f}%)</span>'
        
        lat_active = p_lat > 0.5
        lat_icon = f'<span style="color: #38bdf8; font-weight: bold;">✓ ({p_lat*100:.0f}%)</span>' if lat_active else f'<span style="color: #64748b;">✗ ({p_lat*100:.0f}%)</span>'
        
        html += f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.02);">'
        html += f'<td style="padding: 3px 2px; color: var(--text-primary); font-weight: 500;">{label}</td>'
        html += f'<td style="padding: 3px 2px; text-align: center;">{gt_icon}</td>'
        html += f'<td style="padding: 3px 2px; text-align: center;">{cen_icon}</td>'
        html += f'<td style="padding: 3px 2px; text-align: center;">{lat_icon}</td>'
        html += '</tr>'
        
    html += '</tbody></table></div>'
    return html

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=os.path.join(project_root, "data", "data200", "simulation_metadata.json"))
    parser.add_argument("--report", type=str, default=os.path.join(project_root, "data", "reports", "data200_eval.json"))
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "data", "reports", "data200_comparative_report.html"))
    args = parser.parse_args()

    metadata_path = args.metadata
    report_json_path = args.report
    output_html_path = args.output
    template_path = os.path.join(legovic_root, "projects", "2camaras_pieza_unica", "data100", "report.html")
    if not os.path.exists(template_path):
        template_path = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "data", "data100", "report", "inference_report.html")

    # 1. Ejecutar pipeline de inferencia general
    print(f"\n[Eval] Ejecutando run_evaluation.py sobre {metadata_path}...")
    cmd_eval = [
        sys.executable,
        os.path.join(project_root, "scripts", "run_evaluation.py"),
        "--metadata", metadata_path,
        "--report", report_json_path
    ]
    subprocess.run(cmd_eval, check=True)
    print(f"[Eval] Inferencia completada. Resultados en: {report_json_path}")

    # 2. Cargar datos
    print("Cargando resultados de evaluación y metadatos...")
    with open(report_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cargando modelos YOLO en {device}...")
    model_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    model_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)

    print(f"Cargando modelos de Características Topológicas en {device}...")
    features_cen_model = None
    features_lat_model = None
    cen_feat_path = os.path.join(project_root, "models", "features_cenital.pt")
    lat_feat_path = os.path.join(project_root, "models", "features_lateral.pt")

    if os.path.exists(cen_feat_path):
        try:
            ckpt = torch.load(cen_feat_path, map_location=device)
            model_name = ckpt.get('model_name', 'resnet18')
            features_cen_model = timm.create_model(model_name, num_classes=len(CLASSES_FEATURES))
            features_cen_model.load_state_dict(ckpt['model_state_dict'])
            features_cen_model.to(device)
            features_cen_model.eval()
            print(f"  [Loaded] Cenital features model: {cen_feat_path}")
        except Exception as e:
            print(f"  [Error] Failed to load cenital features model: {e}")

    if os.path.exists(lat_feat_path):
        try:
            ckpt = torch.load(lat_feat_path, map_location=device)
            model_name = ckpt.get('model_name', 'resnet18')
            features_lat_model = timm.create_model(model_name, num_classes=len(CLASSES_FEATURES))
            features_lat_model.load_state_dict(ckpt['model_state_dict'])
            features_lat_model.to(device)
            features_lat_model.eval()
            print(f"  [Loaded] Lateral features model: {lat_feat_path}")
        except Exception as e:
            print(f"  [Error] Failed to load lateral features model: {e}")

    # Agrupar piezas por trayectoria global
    frames_list = meta_data.get("frames", [])
    global_piece_tracks = {}
    for frame in frames_list:
        offset = frame["belt_offset_mm"]
        f_name = frame["file_name"]
        for p in frame["visible_pieces"]:
            x_abs = offset - p["x_belt_local_mm"]
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

    acc_pct = eval_data.get("accuracy", 0.0)
    total_s = eval_data.get("total_samples", len(eval_data.get("results", [])))
    correct_s = eval_data.get("correct_samples", 0)

    header = header.replace("1.00%", f"{acc_pct:.2f}%")
    header = header.replace("Total Correctas: 1 / 100", f"Total Correctas: {correct_s} / {total_s}")
    header = header.replace("190.3s", "N/A")
    header = header.replace("Media: 1903 ms / frame", "Dataset: data200 a 1024x1024")
    header = header.replace("Posición Y: 0 mm", "Vista Domo (data200)")
    header = header.replace("Correctos: 1/100", f"Correctos: {correct_s}/{total_s}")
    header = header.replace("width: 1.0%", f"width: {acc_pct:.1f}%")
    header = header.replace("1.0%", f"{acc_pct:.1f}%")
    header = header.replace("Reporte de Inferencia de Cinta Transportadora (Centrada)", "Reporte Inferencia Domo Multi-Vista — data200")
    header = header.replace("Evaluación de la posición más cercana al centro (Y = 0.0 mm) para las 100 piezas de la base de datos", 
                            "Análisis con overlays visuales y validación de Características Topológicas (GT vs Predicción Redes)")

    import glob
    # Contar frames de simulación reales en el directorio
    sim_dir = os.path.dirname(metadata_path)
    frame_files = [f for f in glob.glob(os.path.join(sim_dir, "frame_*.png")) if not f.endswith("_frontal.png") and not f.endswith("_lateral.png")]
    num_frames = len(frame_files)
    if num_frames == 0:
        num_frames = 202
    
    # Generar slider interactivo coordinado
    slider_html = f"""
        <!-- Coordinated Slider Section -->
        <div class="slider-container" style="background: rgba(20, 28, 47, 0.8); border: 1px solid var(--border-color); border-radius: 16px; padding: 24px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
            <h3 style="margin-top: 0; margin-bottom: 20px; color: var(--accent); display: flex; justify-content: space-between; align-items: center;">
                <span>Visualizador de Renderizado Coordenado (Doble Cámara)</span>
                <span id="frame-counter" style="font-family: monospace; background: rgba(56, 189, 248, 0.15); padding: 4px 10px; border-radius: 8px; font-size: 15px; color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3);">Frame: 000 / {num_frames-1:03d}</span>
            </h3>
            <div style="display: flex; gap: 24px; justify-content: center; margin-bottom: 20px;">
                <div style="flex: 1; max-width: 600px; text-align: center;">
                    <h4 style="color: var(--text-secondary); margin-top: 0; margin-bottom: 10px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px;">Cámara Cenital (Zenithal)</h4>
                    <div style="position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); padding-bottom: 100%; background: #000;">
                        <img id="img-slider-cen" src="../simulation_300/frame_000.png" style="position: absolute; top:0; left:0; width:100%; height:100%; object-fit: contain; cursor: pointer;" onclick="openImageModal(this.src)">
                    </div>
                </div>
                <div style="flex: 1; max-width: 600px; text-align: center;">
                    <h4 style="color: var(--text-secondary); margin-top: 0; margin-bottom: 10px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.5px;">Cámara Frontal/Lateral (Frontal)</h4>
                    <div style="position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); padding-bottom: 100%; background: #000;">
                        <img id="img-slider-front" src="../simulation_300/frame_000_frontal.png" style="position: absolute; top:0; left:0; width:100%; height:100%; object-fit: contain; cursor: pointer;" onclick="openImageModal(this.src)">
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <button onclick="stepFrame(-1)" style="background: #1e293b; border: 1px solid var(--border-color); color: var(--text-primary); width: 44px; height: 44px; border-radius: 10px; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.2s; display: flex; align-items: center; justify-content: center;" onmouseover="this.style.borderColor=\'var(--accent)\';" onmouseout="this.style.borderColor=\'var(--border-color)\';">◀</button>
                <input type="range" min="0" max="{num_frames-1}" value="0" id="frame-slider" oninput="updateSliderFrame(this.value)" style="flex: 1; height: 8px; border-radius: 4px; background: #1e293b; outline: none; cursor: pointer; border: 1px solid var(--border-color); accent-color: var(--accent);">
                <button onclick="stepFrame(1)" style="background: #1e293b; border: 1px solid var(--border-color); color: var(--text-primary); width: 44px; height: 44px; border-radius: 10px; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.2s; display: flex; align-items: center; justify-content: center;" onmouseover="this.style.borderColor=\'var(--accent)\';" onmouseout="this.style.borderColor=\'var(--border-color)\';">▶</button>
            </div>
        </div>

        <script>
        function updateSliderFrame(val) {{
            const frameStr = String(val).padStart(3, '0');
            document.getElementById(\'img-slider-cen\').src = \'../simulation_300/frame_\' + frameStr + \'.png\';
            document.getElementById(\'img-slider-front\').src = \'../simulation_300/frame_\' + frameStr + \'_frontal.png\';
            document.getElementById(\'frame-counter\').innerText = \'Frame: \' + frameStr + \' / {num_frames-1:03d}\';
        }}
        function stepFrame(dir) {{
            const slider = document.getElementById(\'frame-slider\');
            let newVal = parseInt(slider.value) + dir;
            if (newVal >= 0 && newVal <= {num_frames-1}) {{
                slider.value = newVal;
                updateSliderFrame(newVal);
            }}
        }}
        function openImageModal(src) {{
            const modal = document.getElementById(\'imageModal\');
            const modalImg = document.getElementById(\'fullImg\');
            if (modal && modalImg) {{
                modal.style.display = \'block\';
                modalImg.src = src;
            }}
        }}
        </script>
    """

    if "        </div>\n        \n        <div class=\"dashboard-grid\">" in header:
        header = header.replace(
            "        </div>\n        \n        <div class=\"dashboard-grid\">",
            "        </div>\n\n" + slider_html + "\n\n        <div class=\"dashboard-grid\">"
        )
    else:
        header = header.replace(
            "        </div>\r\n        \r\n        <div class=\"dashboard-grid\">",
            "        </div>\r\n\r\n" + slider_html + "\r\n\r\n        <div class=\"dashboard-grid\">"
        )

    rows_html = ""
    os.makedirs(os.path.join(project_root, "data", "reports", "visual_debug"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "data", "reports", "crops_debug"), exist_ok=True)

    print("Procesando piezas, dibujando overlays y ejecutando validación de características topológicas...")
    
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

        path_cen_img = os.path.join(os.path.dirname(metadata_path), cen_file)
        path_lat_img = os.path.join(os.path.dirname(metadata_path), lat_file)

        # Detectar cajas inferidas
        bbox_cen_inf = None
        iou_cen = 0.0
        if os.path.exists(path_cen_img) and bbox_cen_gt:
            res_cen = model_cen(path_cen_img, verbose=False, conf=0.15, imgsz=1024)
            best_iou = 0.0
            best_box = None
            if res_cen and len(res_cen[0].boxes) > 0:
                for box in res_cen[0].boxes:
                    det_box = box.xyxyn[0].cpu().numpy().tolist()
                    iou = generate_data100_report.compute_iou(bbox_cen_gt, det_box)
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
                    iou = generate_data100_report.compute_iou(bbox_lat_gt, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = det_box
            if best_box and best_iou > 0.1:
                bbox_lat_inf = best_box
                iou_lat = best_iou

        # Dibujar y guardar overlays y crops
        img_cen_overlay = generate_data100_report.draw_overlay(path_cen_img, bbox_cen_gt, bbox_cen_inf, ref_gt, ref_inf, score, iou_cen)
        img_lat_overlay = generate_data100_report.draw_overlay(path_lat_img, bbox_lat_gt, bbox_lat_inf, ref_gt, ref_inf, score, iou_lat)

        safe_cen_name = cen_file.replace("/", "_")
        safe_lat_name = lat_file.replace("/", "_")
        
        overlay_cen_out_name = f"visual_debug/sample_{idx:03d}_{safe_cen_name.replace('.png', '_overlay.png')}"
        overlay_lat_out_name = f"visual_debug/sample_{idx:03d}_{safe_lat_name.replace('.png', '_overlay.png')}"
        
        if img_cen_overlay:
            img_cen_overlay.save(os.path.join(project_root, "data", "reports", overlay_cen_out_name))
        if img_lat_overlay:
            img_lat_overlay.save(os.path.join(project_root, "data", "reports", overlay_lat_out_name))

        # Generar crops para evaluar características topológicas
        crop_cen = None
        crop_lat = None
        crop_cen_out_name = f"crops_debug/sample_{idx:03d}_{safe_cen_name.replace('.png', '_crop.png')}"
        crop_lat_out_name = f"crops_debug/sample_{idx:03d}_{safe_lat_name.replace('.png', '_crop.png')}"
        
        if os.path.exists(path_cen_img) and bbox_cen_gt:
            img_cen_raw = Image.open(path_cen_img).convert("RGB")
            w_img, h_img = img_cen_raw.size
            cx1, cy1, cx2, cy2 = int(bbox_cen_gt[0]*w_img), int(bbox_cen_gt[1]*h_img), int(bbox_cen_gt[2]*w_img), int(bbox_cen_gt[3]*h_img)
            cx1, cx2 = max(0, min(w_img, cx1)), max(0, min(w_img, cx2))
            cy1, cy2 = max(0, min(h_img, cy1)), max(0, min(h_img, cy2))
            if cx2 > cx1 and cy2 > cy1:
                crop_cen = img_cen_raw.crop((cx1, cy1, cx2, cy2))
                crop_cen.save(os.path.join(project_root, "data", "reports", crop_cen_out_name))
            
        if os.path.exists(path_lat_img) and bbox_lat_gt:
            img_lat_raw = Image.open(path_lat_img).convert("RGB")
            w_img, h_img = img_lat_raw.size
            lx1, ly1, lx2, ly2 = int(bbox_lat_gt[0]*w_img), int(bbox_lat_gt[1]*h_img), int(bbox_lat_gt[2]*w_img), int(bbox_lat_gt[3]*h_img)
            lx1, lx2 = max(0, min(w_img, lx1)), max(0, min(w_img, lx2))
            ly1, ly2 = max(0, min(h_img, ly1)), max(0, min(h_img, ly2))
            if lx2 > lx1 and ly2 > ly1:
                crop_lat = img_lat_raw.crop((lx1, ly1, lx2, ly2))
                crop_lat.save(os.path.join(project_root, "data", "reports", crop_lat_out_name))

        # Inferir características topológicas con redes timm y consultar DB
        gt_feat = get_topological_features_from_db(ref_gt)
        pred_cen = predict_features(crop_cen, features_cen_model, device)
        pred_lat = predict_features(crop_lat, features_lat_model, device)
        
        features_comp_html = format_features_comparison_html(gt_feat, pred_cen, pred_lat)

        name_cen_inf = r.get("color_name_cen", "Unknown")
        name_lat_inf = r.get("color_name_lat", "Unknown")
        
        if name_lat_inf == "Unknown" or name_lat_inf == "0":
            color_lat_rgb = generate_data100_report.estimate_color_bbox(path_lat_img, bbox_lat_inf or bbox_lat_gt)
            _, name_lat_inf, _ = generate_data100_report.find_closest_color_code(color_lat_rgb, catalog_colors)

        cen_color_match = (str(name_cen_inf).strip().lower() == str(color_name_gt).strip().lower())
        color_cen_html_color = "#10b981" if cen_color_match else "#ef4444"
        lat_color_match = (str(name_lat_inf).strip().lower() == str(color_name_gt).strip().lower())
        color_lat_html_color = "#10b981" if lat_color_match else "#ef4444"

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
            </td>
            <td>
                <div>Pieza: <strong>{ref_inf}</strong></div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">
                    <span>Confianza: {score:.3f}</span><br/>
                    <span>Color Cen. Inf: <strong style="color: {color_cen_html_color};">{name_cen_inf}</strong></span><br/>
                    <span>Color Lat. Inf: <strong style="color: {color_lat_html_color};">{name_lat_inf}</strong></span>
                </div>
            </td>
            <td style="min-width: 200px;">
                {features_comp_html}
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

    print(f"\n[Report] Reporte final comparativo completado en: {output_html_path}")

if __name__ == "__main__":
    main()
