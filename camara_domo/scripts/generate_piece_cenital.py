# -*- coding: utf-8 -*-
"""
camara_domo/scripts/generate_piece_cenital.py
============================================
Genera un reporte de diagnóstico HTML para una pieza LEGO agregada (promedios en movimiento).
Muestra:
  - Estadísticas agregadas de superficie y color (BD vs Inferido).
  - La imagen general del frame donde la pieza está más cerca del centro del FoV.
  - Bounding boxes para todas las piezas en dicho frame, destacando la pieza objetivo con un tag.
  - Imagen recortada de la pieza (crop).
  - Máscara de segmentación SAM preprocesada.
"""

import os
import sys
import json
import math
import argparse
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import SAM

# Configuración de directorios
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

DEFAULT_DATA_DIR = os.path.join(project_root, "data", "data100")
DEFAULT_EVAL = os.path.join(DEFAULT_DATA_DIR, "eval_report.json")
DEFAULT_METADATA = os.path.join(DEFAULT_DATA_DIR, "simulation_metadata.json")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_DATA_DIR, "reports")

_sam = None
def get_sam():
    global _sam
    if _sam is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _sam = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)
    return _sam

def sam_mask(img: Image.Image, bbox_norm: list) -> np.ndarray:
    """Genera la máscara binaria SAM del objeto."""
    try:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        x1 = max(0, int(bbox_norm[0] * w))
        y1 = max(0, int(bbox_norm[1] * h))
        x2 = min(w, int(bbox_norm[2] * w))
        y2 = min(h, int(bbox_norm[3] * h))
        
        results = get_sam()(np.array(img_rgb), bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            
            return mask
    except Exception:
        pass
    w, h = img.size
    fallback = np.zeros((h, w), dtype=np.uint8)
    fallback[int(bbox_norm[1]*h):int(bbox_norm[3]*h), int(bbox_norm[0]*w):int(bbox_norm[2]*w)] = 255
    return fallback

def to_b64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_data(eval_path: str, meta_path: str):
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return eval_data, meta_data

def generate_report(ref: str, color_code: str, eval_data: dict, meta_data: dict, data_dir: str, out_path: str):
    # 1. Agrupar muestras para la pieza dada
    results = eval_data.get("results", [])
    matching_samples = []
    
    for r in results:
        ref_gt = r.get("ref_gt")
        cc_gt = r.get("color_code_gt")
        
        # Filtro por ref y color_code (si se provee)
        if ref_gt == ref and (color_code is None or cc_gt == color_code):
            matching_samples.append(r)
            
    if not matching_samples:
        print(f"[generate_piece_cenital] No se encontraron muestras para la pieza {ref} (color {color_code}).")
        return
        
    print(f"[generate_piece_cenital] Encontradas {len(matching_samples)} imágenes en movimiento para la pieza {ref}.")
    
    # 2. Calcular estadísticas agregadas (medias)
    avg_area_cen_inf = float(np.mean([s["footprint_area_mm2"] for s in matching_samples]))
    avg_error_cen = float(np.mean([s.get("surface_error_rel_pct", 0.0) for s in matching_samples]))
    
    # Votación del color inferido
    colors_inferred = [s.get("color_name_cen", "Unknown") for s in matching_samples]
    final_color_inferred = max(set(colors_inferred), key=colors_inferred.count)
    
    # Datos de base de datos / GT (constante para la pieza)
    first_sample = matching_samples[0]
    gt_area_cen = first_sample.get("surface_db_silhouette_mm2", 0.0)
    gt_color_name = first_sample.get("color_name_gt", "Unknown")
    gt_color_hex = first_sample.get("color_hex_cen", "#808080")
    
    # 3. Encontrar el frame donde el bbox está más cerca de la mitad del FoV (dx, dy cercano a 0)
    best_sample = None
    best_dist = float("inf")
    best_frame_meta = None
    best_piece_meta = None
    
    # Mapeo rápido de simulation_metadata
    frames_meta = {f["file_name"]: f for f in meta_data["frames"]}
    
    for s in matching_samples:
        fname = s["cenital_file"]
        if fname not in frames_meta:
            continue
        frame_info = frames_meta[fname]
        
        # Buscar la pieza específica dentro de las piezas visibles de este frame
        for p in frame_info["visible_pieces"]:
            if p["ref"] == ref and (color_code is None or p["color_code"] == color_code):
                bbox = p["bbox_cenital_norm"] # [xmin, ymin, xmax, ymax]
                cx = (bbox[0] + bbox[2]) * 0.5
                cy = (bbox[1] + bbox[3]) * 0.5
                dist = math.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_sample = s
                    best_frame_meta = frame_info
                    best_piece_meta = p
                    
    if best_sample is None:
        print("[ERROR] No se pudo encontrar metadatos de coordenadas para las piezas coincidentes.")
        return
        
    log_file_cen = os.path.join(data_dir, best_sample["cenital_file"])
    if not os.path.exists(log_file_cen):
        print(f"[ERROR] No existe la imagen cenital {log_file_cen}")
        return
        
    # 4. Generar imágenes con overlays
    img_pil = Image.open(log_file_cen).convert("RGB")
    w_img, h_img = img_pil.size
    
    # Imagen general con bboxes y tag
    img_general = img_pil.copy()
    draw = ImageDraw.Draw(img_general)
    
    # Dibujar bboxes para todas las piezas visibles del frame
    for p in best_frame_meta["visible_pieces"]:
        bbox = p["bbox_cenital_norm"]
        x1, y1 = int(bbox[0]*w_img), int(bbox[1]*h_img)
        x2, y2 = int(bbox[2]*w_img), int(bbox[3]*h_img)
        
        is_target = (p["ref"] == ref and (color_code is None or p["color_code"] == color_code))
        
        if is_target:
            # Color destacado para la pieza seleccionada
            draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=4)
            # Dibujar tag/etiqueta
            draw.rectangle([x1, y1 - 25, x1 + 140, y1], fill="#FF3B30")
            draw.text((x1 + 5, y1 - 22), f"OBJETIVO: {ref}", fill="white")
        else:
            # Bbox común para el resto de piezas
            draw.rectangle([x1, y1, x2, y2], outline="#007AFF", width=2)
            
    # Bbox crop
    bbox_target = best_piece_meta["bbox_cenital_norm"]
    tx1, ty1 = int(bbox_target[0]*w_img), int(bbox_target[1]*h_img)
    tx2, ty2 = int(bbox_target[2]*w_img), int(bbox_target[3]*h_img)
    img_crop = img_pil.crop((tx1, ty1, tx2, ty2))
    
    # Máscara SAM
    mask_arr = sam_mask(img_pil, bbox_target)
    mask_crop_arr = mask_arr[ty1:ty2, tx1:tx2]
    img_mask = Image.fromarray(mask_crop_arr)
    
    # Calcular área física para el frame más centrado dinámicamente (Opción SAM puro sin sustracción de fondo)
    cx_mm = best_piece_meta["x_belt_local_mm"]
    cy_mm = best_piece_meta["y_belt_local_mm"]
    r_mm = math.sqrt(cx_mm**2 + cy_mm**2)
    
    cam_z = 300.0  # mm
    d_floor = math.sqrt(r_mm**2 + cam_z**2)
    from config_loader import cfg
    px_per_mm_cen = float(cfg.cameras.cenital.scale_px_per_mm)
    px_per_mm_local = (px_per_mm_cen * cam_z) / d_floor
    
    num_pixels = float(np.sum(mask_arr > 0))
    area_apparent = num_pixels / (px_per_mm_local ** 2)
    
    # Desmagnificación por la altura real de la pieza
    gt_h = best_piece_meta.get("lateral_height_gt", 4.8)
    demag_linear = (cam_z - gt_h * 0.5) / cam_z
    inferred_area_best = area_apparent * (demag_linear ** 2)
    
    # Error relativo vs GT
    error_best_pct = ((inferred_area_best - gt_area_cen) / gt_area_cen) * 100.0 if gt_area_cen > 0 else 0.0
    
    # Imagen con overlay de máscara sobre el crop
    mask_crop_arr = mask_arr[ty1:ty2, tx1:tx2]
    crop_overlay = np.array(img_crop).copy()
    crop_overlay[mask_crop_arr > 0] = crop_overlay[mask_crop_arr > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    img_crop_overlay = Image.fromarray(crop_overlay.astype(np.uint8))
    
    # Convertir a Base64 para reporte HTML inline
    b64_general = to_b64(img_general)
    b64_crop = to_b64(img_crop)
    b64_mask = to_b64(img_mask, format="PNG")
    b64_crop_overlay = to_b64(img_crop_overlay)
    
    # 5. Generar reporte HTML
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Cenital Agregado - Pieza {ref}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0b131a;
            color: #e2e8f0;
            margin: 0;
            padding: 30px;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            color: #38bdf8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .card {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
        }}
        .card h2 {{
            margin-top: 0;
            color: #38bdf8;
            border-bottom: 2px solid #334155;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #334155;
        }}
        th {{
            color: #94a3b8;
        }}
        .color-preview {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 4px;
            vertical-align: middle;
            margin-right: 8px;
            border: 1px solid #ffffff;
        }}
        .image-container {{
            text-align: center;
            margin-top: 15px;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .image-row {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 15px;
        }}
        .image-col {{
            text-align: center;
        }}
        .image-col img {{
            border-radius: 8px;
            border: 1px solid #334155;
            max-height: 150px;
        }}
        .badge {{
            background-color: #ef4444;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Reporte Cenital Agregado — Pieza {ref}</h1>
        <p>Métricas obtenidas a partir de {len(matching_samples)} imágenes consecutivas en movimiento</p>
    </div>
    
    <div class="grid">
        <!-- Panel Izquierdo: Estadísticas y Datos -->
        <div class="card">
            <h2>Métricas Comparativas (Base de Datos vs. Inferido)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parámetro</th>
                        <th>Base de Datos (Real)</th>
                        <th>Inferencia (Más Centrado)</th>
                        <th>Error Relativo</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Superficie Cenital</strong></td>
                        <td>{gt_area_cen:.2f} mm²</td>
                        <td>{inferred_area_best:.2f} mm²</td>
                        <td><span class="badge" style="background-color: {'#22c55e' if abs(error_best_pct) < 15 else '#ef4444'}">{error_best_pct:.1f}%</span></td>
                    </tr>
                    <tr>
                        <td><strong>Color</strong></td>
                        <td>
                            <span class="color-preview" style="background-color: {gt_color_hex}"></span>
                            {gt_color_name} ({gt_color_hex})
                        </td>
                        <td>
                            {final_color_inferred}
                        </td>
                        <td>—</td>
                    </tr>
                </tbody>
            </table>
            
            <h2 style="margin-top: 30px;">Detalles del Frame Más Centrado</h2>
            <p><strong>Archivo:</strong> {best_sample["cenital_file"]}</p>
            <p><strong>Centroide local:</strong> X = {best_piece_meta["x_belt_local_mm"]:.1f} mm, Y = {best_piece_meta["y_belt_local_mm"]:.1f} mm</p>
            <p><strong>Distancia al centro óptico:</strong> {best_dist * 100:.2f}% de la mitad del FoV</p>
        </div>
        
        <!-- Panel Derecho: Visualización de Imágenes -->
        <div class="card">
            <h2>Vista General (Tag Identificador)</h2>
            <div class="image-container">
                <img src="data:image/jpeg;base64,{b64_general}" alt="Imagen general cenital">
            </div>
            
            <h2 style="margin-top: 30px;">Detalles de la Pieza (Crop & SAM)</h2>
            <div class="image-row">
                <div class="image-col">
                    <p>Crop BBox</p>
                    <img src="data:image/jpeg;base64,{b64_crop}" alt="Crop BBox">
                </div>
                <div class="image-col">
                    <p>Máscara SAM</p>
                    <img src="data:image/png;base64,{b64_mask}" alt="Máscara SAM">
                </div>
                <div class="image-col">
                    <p>Mascara Superpuesta</p>
                    <img src="data:image/jpeg;base64,{b64_crop_overlay}" alt="Overlay SAM">
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    # Guardar reporte HTML
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[ReportCenital] HTML generado en: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Genera reporte cenital agregado para una pieza específica.")
    parser.add_argument("--ref", type=str, required=True, help="ID LDraw de la pieza (ej. 3795).")
    parser.add_argument("--color_code", type=str, default=None, help="Código de color opcional (ej. 11).")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval", type=str, default=DEFAULT_EVAL)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    eval_data, meta_data = load_data(args.eval, args.metadata)
    
    # Determinar ruta del archivo de salida
    if args.out:
        out_path = args.out
    else:
        cc_suffix = args.color_code if args.color_code else "all"
        out_path = os.path.join(DEFAULT_OUT_DIR, f"piece_cenital_{args.ref}_{cc_suffix}.html")
        
    generate_report(args.ref, args.color_code, eval_data, meta_data, args.data_dir, out_path)

if __name__ == "__main__":
    main()
