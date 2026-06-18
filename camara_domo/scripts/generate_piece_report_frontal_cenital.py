# -*- coding: utf-8 -*-
"""
camara_domo/scripts/generate_piece_report_frontal_cenital.py
============================================================
Genera un reporte de diagnóstico HTML multi-vista (Cenital + Frontal) para piezas LEGO.
Muestra:
  - Comparativa de base de datos (Real) vs Inferencia Agregada.
  - Errores relativos de área y altura.
  - Tabla de observaciones individuales por frame.
  - Imágenes generales Cenital y Frontal con bboxes y tag identificador.
  - Crops (recortes) de la pieza en ambas vistas.
"""

import os
import sys
import json
import math
import random
import argparse
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

# Configuración de directorios
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

DEFAULT_DATA_DIR = os.path.join(project_root, "data", "data100")
DEFAULT_CONSOLIDATED = os.path.join(legovic_root, "logs", "inferencia_consolidada_final.json")
DEFAULT_METADATA = os.path.join(DEFAULT_DATA_DIR, "simulation_metadata.json")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_DATA_DIR, "reports")

def to_b64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = BytesIO()
    save_format = "PNG" if pil_img.mode in ("RGBA", "LA", "P") else format
    pil_img.save(buffered, format=save_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def find_matching_gt_piece(bbox_cen, visible_pieces):
    best_iou = 0.0
    best_piece = None
    for p in visible_pieces:
        iou = compute_iou(bbox_cen, p["bbox_cenital_norm"])
        if iou > best_iou:
            best_iou = iou
            best_piece = p
    if best_iou > 0.2:
        return best_piece
    return None

def draw_bboxes_and_highlight(
    img: Image.Image,
    visible_pieces: List[Dict[str, Any]],
    target_ref: str,
    target_bbox_key: str, # "bbox_cenital_norm" o "bbox_frontal_norm"
    highlight_bbox: List[float],
    color_hex: str = "#FF3B30",
    label: str = "TARGET"
) -> Image.Image:
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    w, h = img_draw.size
    
    # Intentar cargar una fuente de sistema
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    is_failed_match = (highlight_bbox == [0.0, 0.0, 1.0, 1.0])
    
    # Primero buscar cuál es la pieza de ground truth con mayor IoU
    best_iou = 0.0
    best_p = None
    for p in visible_pieces:
        bbox = p.get(target_bbox_key)
        if not bbox:
            continue
        if is_failed_match:
            if str(p.get("ref", "")) == str(target_ref):
                best_p = p
                break
        else:
            iou = compute_iou(highlight_bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_p = p

    # Dibujar todas las piezas no objetivo en azul
    for p in visible_pieces:
        bbox = p.get(target_bbox_key)
        if not bbox:
            continue
        # Si esta pieza no es la mejor coincidencia
        if p != best_p:
            x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
            x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
            draw.rectangle([x1, y1, x2, y2], outline="#38bdf8", width=2)
            
    if is_failed_match:
        if best_p and best_p.get(target_bbox_key):
            # Target missed by model but present in GT! Draw GT box to show where it should be
            bbox = best_p.get(target_bbox_key)
            hx1, hy1 = int(bbox[0] * w), int(bbox[1] * h)
            hx2, hy2 = int(bbox[2] * w), int(bbox[3] * h)
            draw.rectangle([hx1, hy1, hx2, hy2], outline="#f97316", width=5)
            gt_label = f" (GT: {best_p['ref']})"
            draw.rectangle([hx1, max(0, hy1 - 30), hx1 + 300, hy1], fill="#f97316")
            draw.text((hx1 + 5, max(0, hy1 - 26)), f"ASOCIACIÓN LATERAL FALLIDA{gt_label}", fill="white", font=font)
    else:
        # Dibujar la caja analizada real (highlight_bbox) de forma destacada
        hx1, hy1 = int(highlight_bbox[0] * w), int(highlight_bbox[1] * h)
        hx2, hy2 = int(highlight_bbox[2] * w), int(highlight_bbox[3] * h)
        
        draw.rectangle([hx1, hy1, hx2, hy2], outline=color_hex, width=5)
        
        # Etiqueta descriptiva sobre la caja
        gt_label = f" (GT: {best_p['ref']})" if best_p else ""
        draw.rectangle([hx1, max(0, hy1 - 30), hx1 + 220, hy1], fill=color_hex)
        draw.text((hx1 + 5, max(0, hy1 - 26)), f"{label}{gt_label}", fill="white", font=font)
            
    return img_draw

def generate_html_report(
    track_data: Dict[str, Any],
    metadata: Dict[str, Any],
    data_dir: str,
    out_dir: str,
    poses_db: Any,
    colors_db: Any
):
    tid = track_data["tracking_id"]
    history = track_data["history"]
    if not history:
        print(f"[ERROR] Track {tid} no tiene historial de observaciones.")
        return
        
    # 1. Encontrar el frame donde el centro del bbox cenital está más cerca de la mitad del FoV (0.5, 0.5)
    best_obs_idx = -1
    best_dist = float("inf")
    
    for idx, h in enumerate(history):
        bbox = h["bbox_cen"]
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        dist = math.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
        if dist < best_dist:
            best_dist = dist
            best_obs_idx = idx
            
    best_obs = history[best_obs_idx]
    best_frame_id = best_obs["frame_id"]
    
    # 2. Obtener imágenes y metadatos del frame seleccionado
    # Encontrar metadatos del frame en simulation_metadata
    frames_meta = {f["file_name"]: f for f in metadata["frames"]}
    
    # La clave de frame_id es algo como frame_XXX, buscar en frames_meta con .png
    frame_key = f"{best_frame_id}.png"
    if frame_key not in frames_meta:
        frame_key = f"{best_frame_id}.jpg"
        
    if frame_key not in frames_meta:
        print(f"[ERROR] No se encontraron metadatos para el frame {best_frame_id}")
        return
        
    frame_meta = frames_meta[frame_key]
    
    # Imagen cenital y lateral/frontal
    img_cen_path = os.path.join(data_dir, frame_meta["file_name"])
    img_lat_path = os.path.join(data_dir, frame_meta["file_name_frontal"])
    
    if not os.path.exists(img_cen_path) or not os.path.exists(img_lat_path):
        print(f"[ERROR] Imágenes no encontradas para frame {best_frame_id} en {data_dir}")
        return
        
    img_cen = Image.open(img_cen_path)
    img_lat = Image.open(img_lat_path)
    
    # 3. Emparejar con ground truth (GT)
    gt_piece = find_matching_gt_piece(best_obs["bbox_cen"], frame_meta["visible_pieces"])
    
    gt_ref = "Unknown"
    gt_color = "Unknown"
    gt_area_cen = 0.0
    gt_height = 0.0
    
    if gt_piece:
        gt_ref = gt_piece["ref"]
        gt_color = gt_piece["color_name"]
        gt_area_cen = gt_piece["zenith_silhouette_area_gt"] or 0.0
        gt_height = gt_piece["lateral_height_gt"] or 0.0
        
    # 4. Estadísticas agregadas
    avg_area_cen = track_data["confidence_details"]["average_area_cen"]
    avg_area_lat = track_data["confidence_details"]["average_area_lat"]
    avg_height = track_data["confidence_details"]["average_height"]
    
    # --- Inferencia Paramétrica ---
    from scripts.inferencia_neuronal import match_piece_hypothesis
    color_name = track_data["color"]
    color_model_best = next((c for c in colors_db if c.color_name == color_name), colors_db[0] if colors_db else None)
    
    candidates = match_piece_hypothesis(
        poses_db=poses_db,
        color_inferido=color_model_best,
        area_cen_est=avg_area_cen,
        area_lat_est=avg_area_lat,
        height_est=avg_height,
        studs_est=0,
        height_is_fallback=True
    )
    candidates.sort(key=lambda x: x[2])
    ref_parametrica = candidates[0][0] if candidates else "Unknown"
    
    # Errores relativos
    err_area_pct = ((avg_area_cen - gt_area_cen) / gt_area_cen) * 100.0 if gt_area_cen > 0 else 0.0
    err_height_pct = ((avg_height - gt_height) / gt_height) * 100.0 if gt_height > 0 else 0.0
    
    # 5. Generar overlays y crops
    # overlays cenital (usando la referencia detectada por el modelo)
    ref_label = f"INF: {track_data['referencia_detectada']}"
    img_cen_overlay = draw_bboxes_and_highlight(
        img_cen, frame_meta["visible_pieces"], gt_ref, "bbox_cenital_norm", best_obs["bbox_cen"],
        color_hex="#10b981", label=ref_label
    )
    # overlays frontal
    img_lat_overlay = draw_bboxes_and_highlight(
        img_lat, frame_meta["visible_pieces"], gt_ref, "bbox_frontal_norm", best_obs["bbox_lat"],
        color_hex="#10b981", label=ref_label
    )
    
    # Obtener lista de frames cenitales visibles para mostrar en el menú/cabecera
    visible_frames_str = ", ".join(track_data["frames_visible"])
    
    # Crops cenital
    w_c, h_c = img_cen.size
    bbox_cen = best_obs["bbox_cen"]
    cx1, cy1 = int(bbox_cen[0] * w_c), int(bbox_cen[1] * h_c)
    cx2, cy2 = int(bbox_cen[2] * w_c), int(bbox_cen[3] * h_c)
    # Asegurar márgenes mínimos
    img_crop_cen = img_cen.crop((cx1, cy1, cx2, cy2))
    b64_crop_cen = to_b64(img_crop_cen)
    
    # crops-row in HTML to show SAM
    # Import SAM dynamically inside generator
    from ultralytics import SAM
    import numpy as np
    
    # Crops frontal
    w_l, h_l = img_lat.size
    bbox_lat = best_obs["bbox_lat"]
    bbox_lat_is_fallback = (bbox_lat == [0.0, 0.0, 1.0, 1.0])
    
    if bbox_lat_is_fallback:
        placeholder = Image.new("RGB", (200, 150), color="#1e293b")
        draw_ph = ImageDraw.Draw(placeholder)
        try: font_ph = ImageFont.truetype("Arial.ttf", 16)
        except: font_ph = ImageFont.load_default()
        draw_ph.text((10, 60), "NO LATERAL MATCH", fill="#ef4444", font=font_ph)
        b64_crop_lat = to_b64(placeholder)
    else:
        lx1, ly1 = int(bbox_lat[0] * w_l), int(bbox_lat[1] * h_l)
        lx2, ly2 = int(bbox_lat[2] * w_l), int(bbox_lat[3] * h_l)
        img_crop_lat = img_lat.crop((lx1, ly1, lx2, ly2))
        b64_crop_lat = to_b64(img_crop_lat)
    
    # Base64
    b64_cen_overlay = to_b64(img_cen_overlay)
    b64_lat_overlay = to_b64(img_lat_overlay)
    
    b64_sam_mask = b64_crop_cen
    b64_sam_mask_lat = b64_crop_lat
    
    try:
        # Cargar SAM en MPS si está disponible
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)
        
        # 1. SAM Cenital con YOLO-Masking
        img_cen_rgb = img_cen.convert("RGB")
        w_s, h_s = img_cen_rgb.size
        sx1, sy1 = max(0, int(bbox_cen[0]*w_s)), max(0, int(bbox_cen[1]*h_s))
        sx2, sy2 = min(w_s, int(bbox_cen[2]*w_s)), min(h_s, int(bbox_cen[3]*h_s))
        
        # Enmascarar otras piezas en la imagen cenital
        img_cen_np = np.array(img_cen_rgb)
        best_cen_iou = 0.0
        best_cen_p = None
        for p in frame_meta["visible_pieces"]:
            other_box = p.get("bbox_cenital_norm")
            if not other_box:
                continue
            iou = compute_iou(bbox_cen, other_box)
            if iou > best_cen_iou:
                best_cen_iou = iou
                best_cen_p = p
                
        for p in frame_meta["visible_pieces"]:
            if p == best_cen_p:
                continue
            other_box = p.get("bbox_cenital_norm")
            if not other_box:
                continue
            ox1 = max(0, min(int(other_box[0] * w_s), w_s - 1))
            oy1 = max(0, min(int(other_box[1] * h_s), h_s - 1))
            ox2 = max(0, min(int(other_box[2] * w_s), w_s - 1))
            oy2 = max(0, min(int(other_box[3] * h_s), h_s - 1))
            img_cen_np[oy1:oy2, ox1:ox2] = 0
                
        cx_c = (sx1 + sx2) * 0.5
        cy_c = (sy1 + sy2) * 0.5
        sam_results = sam_model(img_cen_np, bboxes=[[sx1, sy1, sx2, sy2]], points=[[[cx_c, cy_c]]], labels=[[1]], verbose=False)
        if sam_results and sam_results[0].masks is not None:
            mask_data = sam_results[0].masks.data[0].cpu().numpy()
            mask_img_arr = (mask_data * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_img_arr).crop((sx1, sy1, sx2, sy2))
            b64_sam_mask = to_b64(mask_pil, format="PNG")
            
        # 2. SAM Frontal con YOLO-Masking (skip si hubo fallo)
        if not bbox_lat_is_fallback:
            img_lat_rgb = img_lat.convert("RGB")
            w_l_s, h_l_s = img_lat_rgb.size
            lx1_s, ly1_s = max(0, int(bbox_lat[0]*w_l_s)), max(0, int(bbox_lat[1]*h_l_s))
            lx2_s, ly2_s = min(w_l_s, int(bbox_lat[2]*w_l_s)), min(h_l_s, int(bbox_lat[3]*h_l_s))
            
            # Enmascarar otras piezas en la imagen frontal/lateral
            img_lat_np = np.array(img_lat_rgb)
            best_lat_iou = 0.0
            best_lat_p = None
            for p in frame_meta["visible_pieces"]:
                other_box = p.get("bbox_frontal_norm")
                if not other_box:
                    continue
                iou = compute_iou(bbox_lat, other_box)
                if iou > best_lat_iou:
                    best_lat_iou = iou
                    best_lat_p = p
                    
            for p in frame_meta["visible_pieces"]:
                if p == best_lat_p:
                    continue
                other_box = p.get("bbox_frontal_norm")
                if not other_box:
                    continue
                ox1 = max(0, min(int(other_box[0] * w_l_s), w_l_s - 1))
                oy1 = max(0, min(int(other_box[1] * h_l_s), h_l_s - 1))
                ox2 = max(0, min(int(other_box[2] * w_l_s), w_l_s - 1))
                oy2 = max(0, min(int(other_box[3] * h_l_s), h_l_s - 1))
                img_lat_np[oy1:oy2, ox1:ox2] = 0
                    
            cx_l = (lx1_s + lx2_s) * 0.5
            cy_l = (ly1_s + ly2_s) * 0.5
            sam_results_lat = sam_model(img_lat_np, bboxes=[[lx1_s, ly1_s, lx2_s, ly2_s]], points=[[[cx_l, cy_l]]], labels=[[1]], verbose=False)
            if sam_results_lat and sam_results_lat[0].masks is not None:
                mask_data_lat = sam_results_lat[0].masks.data[0].cpu().numpy()
                mask_img_arr_lat = (mask_data_lat * 255).astype(np.uint8)
                mask_pil_lat = Image.fromarray(mask_img_arr_lat).crop((lx1_s, ly1_s, lx2_s, ly2_s))
                b64_sam_mask_lat = to_b64(mask_pil_lat, format="PNG")
            
    except Exception as e:
        print(f"[WARNING] Error generación SAM masks: {e}")
        
    from collections import Counter
    colors_cen_votes = [obs.get("color_cenital", obs.get("color", "Unknown")) for obs in history]
    colors_lat_votes = [obs.get("color_lateral", "N/A") for obs in history if obs.get("color_lateral", "N/A") != "N/A"]
    avg_color_cenital = Counter(colors_cen_votes).most_common(1)[0][0] if colors_cen_votes else "Unknown"
    avg_color_lateral = Counter(colors_lat_votes).most_common(1)[0][0] if colors_lat_votes else "N/A"
    
    # 5.1 Seleccionar 5 observaciones equidistantes en distancia cenital de avance (eje X)
    sorted_history = sorted(history, key=lambda h: (h["bbox_cen"][0] + h["bbox_cen"][2]) * 0.5)
    n_obs = len(sorted_history)
    selected_obs = []
    
    if n_obs <= 5:
        selected_obs = sorted_history
    else:
        x_positions = [(h["bbox_cen"][0] + h["bbox_cen"][2]) * 0.5 for h in sorted_history]
        min_x = x_positions[0]
        max_x = x_positions[-1]
        target_x_values = [min_x + (max_x - min_x) * (i / 4.0) for i in range(5)]
        
        for tx in target_x_values:
            best_h = min(sorted_history, key=lambda h: abs(((h["bbox_cen"][0] + h["bbox_cen"][2]) * 0.5) - tx))
            if best_h not in selected_obs:
                selected_obs.append(best_h)
            else:
                remaining = [h for h in sorted_history if h not in selected_obs]
                if remaining:
                    best_h = min(remaining, key=lambda h: abs(((h["bbox_cen"][0] + h["bbox_cen"][2]) * 0.5) - tx))
                    selected_obs.append(best_h)
        # Volver a ordenar
        selected_obs = sorted(selected_obs, key=lambda h: (h["bbox_cen"][0] + h["bbox_cen"][2]) * 0.5)

    sequence_html = ""
    for idx, obs in enumerate(selected_obs):
        f_id = obs["frame_id"]
        f_key = f"{f_id}.png"
        if f_key not in frames_meta:
            f_key = f"{f_id}.jpg"
        if f_key not in frames_meta:
            continue
        f_meta = frames_meta[f_key]
        
        path_c = os.path.join(data_dir, f_meta["file_name"])
        path_l = os.path.join(data_dir, f_meta["file_name_frontal"])
        
        b64_seq_cen = ""
        b64_seq_lat = ""
        
        if os.path.exists(path_c):
            try:
                img_c_raw = Image.open(path_c)
                w_c, h_c = img_c_raw.size
                box_c = obs["bbox_cen"]
                cx1 = max(0, int(box_c[0] * w_c))
                cy1 = max(0, int(box_c[1] * h_c))
                cx2 = min(w_c, int(box_c[2] * w_c))
                cy2 = min(h_c, int(box_c[3] * h_c))
                if cx2 > cx1 and cy2 > cy1:
                    img_crop_c = img_c_raw.crop((cx1, cy1, cx2, cy2))
                    b64_seq_cen = to_b64(img_crop_c)
            except Exception as e:
                print(f"[WARNING] Error en seq crop cenital: {e}")
            
        if os.path.exists(path_l):
            try:
                img_l_raw = Image.open(path_l)
                w_l, h_l = img_l_raw.size
                box_l = obs["bbox_lat"]
                lx1 = max(0, int(box_l[0] * w_l))
                ly1 = max(0, int(box_l[1] * h_l))
                lx2 = min(w_l, int(box_l[2] * w_l))
                ly2 = min(h_l, int(box_l[3] * h_l))
                if lx2 > lx1 and ly2 > ly1:
                    img_crop_l = img_l_raw.crop((lx1, ly1, lx2, ly2))
                    b64_seq_lat = to_b64(img_crop_l)
            except Exception as e:
                print(f"[WARNING] Error en seq crop lateral: {e}")
            
        cx_val = (obs["bbox_cen"][0] + obs["bbox_cen"][2]) * 0.5
        
        frontal_img_tag = f"""
            <div style="margin-top: 10px;">
                <img src="data:image/jpeg;base64,{b64_seq_lat}" style="max-height: 80px; max-width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Frontal Crop">
                <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">Frontal</div>
            </div>
        """ if b64_seq_lat else ""
        
        sequence_html += f"""
        <div style="flex: 1; min-width: 150px; background-color: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 12px; padding: 12px; text-align: center;">
            <div style="font-weight: 600; color: var(--accent); margin-bottom: 8px;">Paso {idx+1}</div>
            <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 12px;">{f_id}<br>Avance X: {cx_val:.3f}</div>
            
            <div style="margin-bottom: 10px;">
                <img src="data:image/jpeg;base64,{b64_seq_cen}" style="max-height: 80px; max-width: 100%; border-radius: 6px; border: 1px solid var(--border-color);" alt="Cenital Crop">
                <div style="font-size: 10px; color: var(--text-secondary); margin-top: 4px;">Cenital</div>
            </div>
            {frontal_img_tag}
        </div>
        """

    # 6. Generar filas de observaciones individuales
    obs_rows = ""
    for idx, obs in enumerate(history):
        f_key = f"{obs['frame_id']}.png"
        f_meta = frames_meta.get(f_key, {})
        gt_p_f = find_matching_gt_piece(obs["bbox_cen"], f_meta.get("visible_pieces", []))
        
        gt_area_f = gt_p_f["zenith_silhouette_area_gt"] if gt_p_f else 0.0
        gt_h_f = gt_p_f["lateral_height_gt"] if gt_p_f else 0.0
        
        err_area_f = ((obs["area_cen"] - gt_area_f) / gt_area_f) * 100.0 if gt_area_f > 0 else 0.0
        err_h_f = ((obs["height"] - gt_h_f) / gt_h_f) * 100.0 if gt_h_f > 0 else 0.0
        
        badge_area_class = "badge-ok" if abs(err_area_f) < 15.0 else ("badge-warn" if abs(err_area_f) < 30.0 else "badge-bad")
        badge_h_class = "badge-ok" if abs(err_h_f) < 15.0 else ("badge-warn" if abs(err_h_f) < 30.0 else "badge-bad")
        
        is_best_frame = " 🎯" if idx == best_obs_idx else ""
        
        obs_rows += f"""
        <tr>
            <td>{obs['frame_id']}{is_best_frame}</td>
            <td>{obs['color']}</td>
            <td>{obs['area_cen']:.2f} mm² <span class="badge {badge_area_class}">{err_area_f:+.1f}%</span></td>
            <td>{obs['area_lat']:.2f} mm²</td>
            <td>{obs['height']:.2f} mm <span class="badge {badge_h_class}">{err_h_f:+.1f}%</span></td>
            <td>{obs['studs']}</td>
        </tr>
        """
        
    badge_area_agg = "badge-ok" if abs(err_area_pct) < 15.0 else ("badge-warn" if abs(err_area_pct) < 30.0 else "badge-bad")
    badge_h_agg = "badge-ok" if abs(err_height_pct) < 15.0 else ("badge-warn" if abs(err_height_pct) < 30.0 else "badge-bad")
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Diagnóstico Multi-Vista - Track {tid}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {{
            --bg-main: #090d16;
            --bg-card: #131a26;
            --border-color: #222d3d;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --accent-gradient: linear-gradient(135deg, #38bdf8, #818cf8);
            --ok: #10b981;
            --warn: #f59e0b;
            --bad: #ef4444;
        }}
        
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            margin: 0;
            padding: 40px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, #131a26, #1e293b);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 32px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}
        
        .header p {{
            color: var(--text-secondary);
            margin: 10px 0 0 0;
            font-size: 16px;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }}
        
        .card h2 {{
            margin-top: 0;
            font-size: 22px;
            color: var(--accent);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 12px;
            font-weight: 600;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .stats-table th, .stats-table td {{
            text-align: left;
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .stats-table th {{
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            display: inline-block;
        }}
        
        .badge-ok {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--ok);
            border: 1px solid var(--ok);
        }}
        
        .badge-warn {{
            background-color: rgba(245, 158, 11, 0.15);
            color: var(--warn);
            border: 1px solid var(--warn);
        }}
        
        .badge-bad {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--bad);
            border: 1px solid var(--bad);
        }}
        
        .image-showcase {{
            display: flex;
            flex-direction: column;
            gap: 20px;
            align-items: center;
        }}
        
        .image-box {{
            width: 100%;
            text-align: center;
        }}
        
        .image-box img {{
            max-width: 100%;
            max-height: 380px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s ease;
        }}
        
        .image-box img:hover {{
            transform: scale(1.02);
        }}
        
        .image-label {{
            margin-top: 8px;
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .crops-row {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
            width: 100%;
        }}
        
        .crop-container {{
            text-align: center;
            flex: 1;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        
        .crop-container img {{
            max-height: 110px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        
        .explanation-card {{
            background-color: rgba(56, 189, 248, 0.05);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .explanation-card strong {{
            color: var(--accent);
        }}
        
        .history-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
            margin-top: 30px;
        }}
        
        .history-card h2 {{
            margin-top: 0;
            font-size: 22px;
            color: var(--accent);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 12px;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte Diagnóstico Multi-Vista - Track {tid}</h1>
            <p>Inferencia combinada de cámaras Cenital y Frontal a lo largo de {len(history)} observaciones consecutivas</p>
            <p style="margin-top: 8px; font-size: 14px; color: var(--accent);"><strong>Frames visible cenitalmente:</strong> {visible_frames_str}</p>
        </div>
        
        <div class="grid-2">
            <!-- Columna Izquierda: Datos y Reportes -->
            <div class="card">
                <h2>Fusión de Inferencia y Ground Truth</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Métrica</th>
                            <th>Base de Datos (Real)</th>
                            <th>Inferencia (Media)</th>
                            <th>Error Relativo</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Referencia (Paramétrica)</strong></td>
                            <td><span class="badge badge-neutral">{gt_ref}</span></td>
                            <td><span class="badge badge-ok">{ref_parametrica}</span></td>
                            <td>{"" if gt_ref == ref_parametrica else "✗ Mismatch"}</td>
                        </tr>
                        <tr>
                            <td><strong>Referencia (EfficientNet)</strong></td>
                            <td><span class="badge badge-neutral">{gt_ref}</span></td>
                            <td><span class="badge badge-ok">{track_data['referencia_detectada']} (Pose {track_data['pose_identificada']})</span></td>
                            <td>{"" if gt_ref == track_data['referencia_detectada'] else "✗ Mismatch"}</td>
                        </tr>
                        <tr>
                            <td><strong>Superficie Cenital</strong></td>
                            <td>{gt_area_cen:.2f} mm²</td>
                            <td>{avg_area_cen:.2f} mm²</td>
                            <td><span class="badge {badge_area_agg}">{err_area_pct:+.1f}%</span></td>
                        </tr>
                        <tr>
                            <td><strong>Altura de Pieza</strong></td>
                            <td>{gt_height:.2f} mm</td>
                            <td>{avg_height:.2f} mm</td>
                            <td><span class="badge {badge_h_agg}">{err_height_pct:+.1f}%</span></td>
                        </tr>
                        <tr>
                            <td><strong>Color Cenital (Media)</strong></td>
                            <td>{gt_color}</td>
                            <td>{avg_color_cenital}</td>
                            <td>—</td>
                        </tr>
                        <tr>
                            <td><strong>Color Frontal (Media)</strong></td>
                            <td>{gt_color}</td>
                            <td>{avg_color_lateral}</td>
                            <td>—</td>
                        </tr>
                        <tr>
                            <td><strong>Color Final Fusionado</strong></td>
                            <td><strong>{gt_color}</strong></td>
                            <td><strong style="color: var(--accent);">{track_data['color']}</strong></td>
                            <td>—</td>
                        </tr>
                        <tr>
                            <td><strong>Confianza Scoring</strong></td>
                            <td>—</td>
                            <td>{track_data['score']:.3f}</td>
                            <td>—</td>
                        </tr>
                    </tbody>
                </table>
                
                <h2 style="margin-top: 45px;">Recortes de Pieza (Crops) y Segmentación en {best_frame_id}</h2>
                <div class="crops-row">
                    <div class="crop-container">
                        <img src="data:image/jpeg;base64,{b64_crop_cen}" alt="Crop Cenital">
                        <div class="image-label">Cenital</div>
                    </div>
                    <div class="crop-container">
                        <img src="data:image/png;base64,{b64_sam_mask}" alt="SAM Cenital">
                        <div class="image-label">SAM Cenital</div>
                    </div>
                    <div class="crop-container">
                        <img src="data:image/jpeg;base64,{b64_crop_lat}" alt="Crop Frontal">
                        <div class="image-label">Frontal</div>
                    </div>
                    <div class="crop-container">
                        <img src="data:image/png;base64,{b64_sam_mask_lat}" alt="SAM Frontal">
                        <div class="image-label">SAM Frontal</div>
                    </div>
                </div>
            </div>
            
            <!-- Columna Derecha: Imágenes Generales -->
            <div class="card image-showcase">
                <h2>Imágenes Generales del Frame Más Centrado</h2>
                
                <div class="image-box">
                    <img src="data:image/jpeg;base64,{b64_cen_overlay}" alt="Vista General Cenital">
                    <div class="image-label">Vista Cenital General - {best_frame_id}</div>
                </div>
                
                <div class="image-box">
                    <img src="data:image/jpeg;base64,{b64_lat_overlay}" alt="Vista General Frontal">
                    <div class="image-label">Vista Frontal General - {best_frame_id}</div>
                </div>
            </div>
        </div>
        
        <!-- Explicación de la Inferencia Multi-Frame -->
        <div class="explanation-card">
            <h2>💡 ¿Qué significan las mediciones en diferentes frames?</h2>
            <p>
                A medida que la pieza avanza por la cinta transportadora, las cámaras cenital e inclinada la observan de forma consecutiva desde diferentes ángulos.
                Las mediciones por frame muestran cómo fluctúan los cálculos de área y altura debido a la perspectiva, el ruido del sensor y el movimiento:
            </p>
            <ul>
                <li><strong>Entrada y Salida del FoV:</strong> En los primeros frames (ej. <code>frame_128</code>) y los últimos, la pieza entra/sale parcialmente de los límites o se observa con ángulos extremos de perspectiva. Esto causa distorsiones ópticas que alteran la proyección del área y la triangulación.</li>
                <li><strong>Frame Más Centrado (🎯):</strong> Es el instante donde la pieza pasa exactamente por la línea centro del FoV (mínima distorsión radial). Este frame es el prioritario y óptimo para capturar las dimensiones físicas de la base de datos.</li>
                <li><strong>Inferencia Agregada Final:</strong> En lugar de basarse en un único frame ruidoso, el sistema realiza un seguimiento temporal (Tracking ID) acumulando la información de toda la trayectoria y consolidando la predicción mediante filtros de consistencia y promedios ponderados.</li>
            </ul>
        </div>
        
        <!-- Secuencia de Avance (5 Imágenes Equidistantes) -->
        <div class="card" style="margin-top: 30px; margin-bottom: 30px;">
            <h2>Secuencia de Avance (5 Observaciones Equidistantes en Avance Cenital)</h2>
            <p style="color: var(--text-secondary); margin-bottom: 20px;">Muestreo de 5 fotogramas distribuidos de forma regular a lo largo del recorrido en el eje X (independientemente de la velocidad de avance).</p>
            <div style="display: flex; gap: 15px; justify-content: space-between; overflow-x: auto; padding-bottom: 10px;">
                {sequence_html}
            </div>
        </div>
        
        <!-- Tabla Inferior: Historial por Frame -->
        <div class="history-card">
            <h2>Historial de Mediciones Individuales por Frame</h2>
            <table class="stats-table" style="margin-top: 10px;">
                <thead>
                    <tr>
                        <th>ID del Frame</th>
                        <th>Color Estimado</th>
                        <th>Área Cenital (Error %)</th>
                        <th>Área Frontal Proyectada</th>
                        <th>Altura Estimada (Error %)</th>
                        <th>Studs / Huecos</th>
                    </tr>
                </thead>
                <tbody>
                    {obs_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"report_piece_multiview_{tid}_{track_data['referencia_detectada']}_{track_data['color']}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[SUCCESS] Reporte multi-vista generado en: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Genera reportes de diagnóstico multi-vista.")
    parser.add_argument("--consolidated", type=str, default=DEFAULT_CONSOLIDATED, help="Consolidado JSON del pipeline.")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA, help="Metadata de simulación.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directorio de imágenes.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Directorio de reportes HTML.")
    parser.add_argument("--tracking_id", type=str, default=None, help="Generar sólo este tracking_id.")
    parser.add_argument("--random_count", type=int, default=2, help="Número de piezas aleatorias si no se especifica tracking_id.")
    args = parser.parse_args()
    
    if not os.path.exists(args.consolidated):
        print(f"[ERROR] Archivo consolidado no encontrado: {args.consolidated}")
        return
        
    if not os.path.exists(args.metadata):
        print(f"[ERROR] Metadatos de simulación no encontrados: {args.metadata}")
        return
        
    print(f"Cargando consolidado: {args.consolidated}...")
    with open(args.consolidated, "r", encoding="utf-8") as f:
        tracks = json.load(f)
        
    print(f"Cargando metadatos: {args.metadata}...")
    with open(args.metadata, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    tids = list(tracks.keys())
    print(f"Encontradas {len(tids)} piezas trackeadas en el consolidado.")
    
    if not tids:
        print("[WARNING] El consolidado no contiene piezas. Runcorrect the pipeline first.")
        return
        
    print("Cargando DB para inferencia paramétrica...")
    from scripts.inferencia_neuronal import load_db_universe
    poses_db, colors_db = load_db_universe(None)
        
    # Selección de IDs a reportar
    selected_tids = []
    if args.tracking_id:
        if args.tracking_id in tracks:
            selected_tids = [args.tracking_id]
        else:
            print(f"[ERROR] Tracking ID {args.tracking_id} no se encuentra en el consolidado.")
            return
    else:
        # Seleccionar random_count piezas aleatorias
        count = min(len(tids), args.random_count)
        selected_tids = random.sample(tids, count)
        
    print(f"Generando reportes para las siguientes piezas: {selected_tids}")
    
    generated_files = []
    for tid in selected_tids:
        try:
            report_file = generate_html_report(tracks[tid], metadata, args.data_dir, args.out_dir, poses_db, colors_db)
            if report_file:
                generated_files.append(report_file)
        except Exception as e:
            print(f"[ERROR] Falló la generación del reporte para {tid}: {e}")
            
    print("\n--- RESUMEN ---")
    print(f"Se generaron {len(generated_files)} reportes multi-vista:")
    for f in generated_files:
        print(f" - [HTML] (file://{f})")

if __name__ == "__main__":
    main()
