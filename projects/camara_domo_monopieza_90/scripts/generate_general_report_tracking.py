# -*- coding: utf-8 -*-
"""
camara_domo/scripts/generate_general_report_tracking.py
======================================================
Genera un reporte de diagnóstico comparativo de 3 pipelines de inferencia
(CLASSIC, HYBRID, POSE_ONLY) con tracking secuencial multi-vista.
"""

import os
import sys
import json
import math
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# Configuración de directorios
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

DEFAULT_DATA_DIR = os.path.join(project_root, "data", "data100")
DEFAULT_CONSOLIDATED_CLASSIC = os.path.join(project_root, "logs", "inferencia_consolidada_CLASSIC.json")
DEFAULT_CONSOLIDATED_HYBRID = os.path.join(project_root, "logs", "inferencia_consolidada_HYBRID.json")
DEFAULT_CONSOLIDATED_POSE = os.path.join(project_root, "logs", "inferencia_consolidada_POSE_ONLY.json")
DEFAULT_METADATA = os.path.join(DEFAULT_DATA_DIR, "simulation_metadata.json")
DEFAULT_OUT_HTML = os.path.join(DEFAULT_DATA_DIR, "reports", "general_report_tracking.html")

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
        return best_piece, best_iou
    return None, 0.0

def build_general_report(
    consolidated_classic: str,
    consolidated_hybrid: str,
    consolidated_pose: str,
    metadata_path: str,
    out_html_path: str,
    limit: int = 0
):
    print(f"Cargando consolidado CLASSIC: {consolidated_classic}...")
    with open(consolidated_classic, "r", encoding="utf-8") as f:
        tracks_classic = json.load(f)

    print(f"Cargando consolidado HYBRID: {consolidated_hybrid}...")
    with open(consolidated_hybrid, "r", encoding="utf-8") as f:
        tracks_hybrid = json.load(f)

    print(f"Cargando consolidado POSE_ONLY: {consolidated_pose}...")
    with open(consolidated_pose, "r", encoding="utf-8") as f:
        tracks_pose = json.load(f)

    print(f"Cargando metadatos: {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_tids = list(set(tracks_classic.keys()).intersection(tracks_hybrid.keys()).intersection(tracks_pose.keys()))

    if limit > 0 and len(all_tids) > limit:
        import random
        # Seleccionar limit tracks aleatorios para evaluar
        selected_keys = random.sample(all_tids, limit)
        print(f"[INFO] Limitando evaluación a un subconjunto aleatorio de {limit} piezas trackeadas.")
    else:
        selected_keys = all_tids

    print("Cargando base de datos para inferencia paramétrica...")
    from scripts.inferencia_neuronal import load_db_universe, match_piece_hypothesis
    poses_db, colors_db = load_db_universe(None)

    frames_meta = {f["file_name"]: f for f in metadata["frames"]}
    
    reports_dir = os.path.dirname(out_html_path)
    crops_dir = os.path.join(reports_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    sim_dir = os.path.dirname(metadata_path)
    
    total_tracks = len(selected_keys)
    matched_tracks = 0
    detailed_results = []
    
    for tid in selected_keys:
        track_c = tracks_classic[tid]
        track_h = tracks_hybrid[tid]
        track_p = tracks_pose[tid]
        
        history = track_c["history"]
        if not history:
            continue
            
        # Encontrar frame más centrado usando CLASSIC como referencia
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
        
        frame_cen_name = f"{best_frame_id}.png"
        frame_lat_name = f"{best_frame_id}_frontal.png"
        frame_key = frame_cen_name
        if frame_key not in frames_meta:
            frame_cen_name = f"{best_frame_id}.jpg"
            frame_lat_name = f"{best_frame_id}_frontal.jpg"
            frame_key = frame_cen_name
            
        if frame_key not in frames_meta:
            continue
            
        # We need GT data first for drawing
        f_meta = frames_meta[frame_key]
        gt_p, iou_cen = find_matching_gt_piece(best_obs["bbox_cen"], f_meta.get("visible_pieces", []))
        
        iou_lat = 0.0
        gt_ref = "Unknown"
        gt_color = "Unknown"
        gt_area_cen = 0.0
        gt_height = 0.0
        gt_bbox_cen = None
        gt_bbox_lat = None
        
        if gt_p:
            gt_ref = gt_p["ref"]
            gt_color = gt_p["color_name"]
            gt_area_cen = gt_p["zenith_silhouette_area_gt"] or 0.0
            gt_height = gt_p["lateral_height_gt"] or 0.0
            gt_bbox_cen = gt_p.get("bbox_cenital_norm")
            gt_bbox_lat = gt_p.get("bbox_frontal_norm")
            if "bbox_lat" in best_obs and gt_bbox_lat:
                iou_lat = compute_iou(best_obs["bbox_lat"], gt_bbox_lat)
            matched_tracks += 1

        # Process image crops
        crop_cen_rel = f"crops/{tid}_cen.png"
        crop_lat_rel = f"crops/{tid}_lat.png"
        crop_cen_abs = os.path.join(reports_dir, crop_cen_rel)
        crop_lat_abs = os.path.join(reports_dir, crop_lat_rel)
        
        full_cen_rel = ""
        full_lat_rel = ""
        
        # Siempre sobreescribimos los crops para actualizar los rectángulos si cambian las lógicas
        img_cen_path = os.path.join(sim_dir, frame_cen_name)
        if os.path.exists(img_cen_path):
            img = cv2.imread(img_cen_path)
            if img is not None:
                h, w = img.shape[:2]
                if gt_bbox_cen:
                    gx1, gy1, gx2, gy2 = int(gt_bbox_cen[0]*w), int(gt_bbox_cen[1]*h), int(gt_bbox_cen[2]*w), int(gt_bbox_cen[3]*h)
                    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                ix1, iy1, ix2, iy2 = int(best_obs["bbox_cen"][0]*w), int(best_obs["bbox_cen"][1]*h), int(best_obs["bbox_cen"][2]*w), int(best_obs["bbox_cen"][3]*h)
                color_inf = (0, 255, 0) if iou_cen > 0.5 else (0, 0, 255)
                cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color_inf, 2)
                
                # Guardar imagen completa anotada
                full_cen_rel = f"crops/{tid}_cen_full.jpg"
                full_cen_abs = os.path.join(reports_dir, full_cen_rel)
                cv2.imwrite(full_cen_abs, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                x1, y1 = max(0, ix1-40), max(0, iy1-40)
                x2, y2 = min(w, ix2+40), min(h, iy2+40)
                crop_img = img[y1:y2, x1:x2]
                if crop_img.size > 0:
                    cv2.imwrite(crop_cen_abs, crop_img)
                
        img_lat_path = os.path.join(sim_dir, frame_lat_name)
        if os.path.exists(img_lat_path) and "bbox_lat" in best_obs:
            img = cv2.imread(img_lat_path)
            if img is not None:
                h, w = img.shape[:2]
                if gt_bbox_lat:
                    gx1, gy1, gx2, gy2 = int(gt_bbox_lat[0]*w), int(gt_bbox_lat[1]*h), int(gt_bbox_lat[2]*w), int(gt_bbox_lat[3]*h)
                    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                ix1, iy1, ix2, iy2 = int(best_obs["bbox_lat"][0]*w), int(best_obs["bbox_lat"][1]*h), int(best_obs["bbox_lat"][2]*w), int(best_obs["bbox_lat"][3]*h)
                color_inf = (0, 255, 0) if iou_lat > 0.5 else (0, 0, 255)
                cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color_inf, 2)
                
                # Guardar imagen completa anotada
                full_lat_rel = f"crops/{tid}_lat_full.jpg"
                full_lat_abs = os.path.join(reports_dir, full_lat_rel)
                cv2.imwrite(full_lat_abs, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                x1, y1 = max(0, ix1-40), max(0, iy1-40)
                x2, y2 = min(w, ix2+40), min(h, iy2+40)
                crop_img = img[y1:y2, x1:x2]
                if crop_img.size > 0:
                    cv2.imwrite(crop_lat_abs, crop_img)
            
        res_row = {
            "tid": tid,
            "best_frame": best_frame_id,
            "crop_cen_rel": crop_cen_rel,
            "crop_lat_rel": crop_lat_rel,
            "full_cen_rel": full_cen_rel,
            "full_lat_rel": full_lat_rel,
            "gt_ref": gt_ref,
            "gt_color": gt_color,
            "gt_area_cen": gt_area_cen,
            "gt_height": gt_height,
            "iou_cen": iou_cen,
            "iou_lat": iou_lat,
            "pipelines": {}
        }
        
        for p_name, p_data in [("CLASSIC", track_c), ("HYBRID", track_h), ("POSE", track_p)]:
            avg_area_cen = p_data["confidence_details"]["average_area_cen"]
            avg_height = p_data["confidence_details"]["average_height"]
            
            # --- Inferencia Paramétrica ---
            color_name = p_data["color"]
            color_model_best = next((c for c in colors_db if c.color_name == color_name), colors_db[0] if colors_db else None)
            candidates = match_piece_hypothesis(
                poses_db=poses_db,
                color_inferido=color_model_best,
                area_cen_est=avg_area_cen,
                area_lat_est=p_data["confidence_details"]["average_area_lat"],
                height_est=avg_height,
                studs_est=0,
                height_is_fallback=True
            )
            candidates.sort(key=lambda x: x[2])
            ref_param = candidates[0][0] if candidates else "Unknown"
            
            err_area = abs((avg_area_cen - gt_area_cen) / gt_area_cen) * 100.0 if gt_area_cen > 0 else 0.0
            err_height = 0.0
            
            res_row["pipelines"][p_name] = {
                "color": color_name,
                "area_cen": avg_area_cen,
                "err_area": err_area,
                "height": 0.0,
                "err_height": 0.0,
                "ref_param": ref_param,
                "ref_effnet": p_data["referencia_detectada"]
            }
            
        detailed_results.append(res_row)

    rows_html = ""
    for res in detailed_results:
        gt_ref = res["gt_ref"]
        
        # Sub-filas por pipeline
        pipelines_html = ""
        for p_name in ["CLASSIC", "HYBRID", "POSE"]:
            p_data = res["pipelines"][p_name]
            
            ref_eff_class = "ok" if p_data["ref_effnet"] == gt_ref else "bad"
            ref_param_class = "ok" if p_data["ref_param"] == gt_ref else "bad"
            color_class = "ok" if p_data["color"] == res["gt_color"] else "bad"
            
            pipelines_html += f"""
            <tr class="sub-row">
                <td><strong>{p_name}</strong></td>
                <td><span class="badge badge-{color_class}">{p_data['color']}</span></td>
                <td>{p_data['area_cen']:.1f} mm² <small>({p_data['err_area']:+.1f}%)</small></td>
                <td>N/A <small>(Desactivado)</small></td>
                <td>
                    Param: <span class="badge badge-{ref_param_class}">{p_data['ref_param']}</span><br>
                    EffNet: <span class="badge badge-{ref_eff_class}">{p_data['ref_effnet']}</span>
                </td>
            </tr>
            """
            
        rows_html += f"""
        <tr style="border-top: 2px solid var(--border-color);">
            <td rowspan="4" style="text-align: center; vertical-align: middle; border-right: 1px solid var(--border-color);">
                <strong>{res['tid']}</strong>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <div style="display: flex; flex-direction: column; gap: 8px; align-items: center;">
                    <img src="{res['crop_cen_rel']}" alt="Cenital" style="max-width: 80px; max-height: 80px; border-radius: 4px; border: 1px solid var(--border-color); cursor: pointer;" onclick="openImageModal('{res['full_cen_rel']}', 'Vista Cenital Completa - {res['tid']} ({res['best_frame']})')"><br>
                    <small style="color: var(--text-secondary);">Best Frame: {res['best_frame']}</small>
                </div>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <div style="display: flex; flex-direction: column; gap: 8px; align-items: center;">
                    <img src="{res['crop_lat_rel']}" alt="Lateral" style="max-width: 80px; max-height: 80px; border-radius: 4px; border: 1px solid var(--border-color); cursor: pointer;" onclick="openImageModal('{res['full_lat_rel']}', 'Vista Lateral Completa - {res['tid']} ({res['best_frame']})')"><br>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <strong>{res['tid']}</strong><br>
                <small style="color: var(--text-secondary);">Frame: {res['best_frame']}</small><br>
                <div style="margin-top: 8px;">
                    <span class="badge badge-{'ok' if res['iou_cen'] > 0.5 else 'bad'}" style="cursor: pointer;" onclick="openImageModal('{res['full_cen_rel']}', 'Vista Cenital Completa - {res['tid']} ({res['best_frame']})')">BBox Cen: {res['iou_cen']:.2f}</span>
                </div>
                <div style="margin-top: 4px;">
                    <span class="badge badge-{'ok' if res['iou_lat'] > 0.5 else 'bad'}" style="cursor: pointer;" onclick="openImageModal('{res['full_lat_rel']}', 'Vista Lateral Completa - {res['tid']} ({res['best_frame']})')">BBox Lat: {res['iou_lat']:.2f}</span>
                </div>
            </td>
            <td rowspan="4" style="border-right: 1px solid var(--border-color);">
                <div style="margin-bottom: 4px;"><strong>Ref:</strong> <span class="badge badge-neutral">{gt_ref}</span></div>
                <div style="margin-bottom: 4px;"><strong>Color:</strong> {res['gt_color']}</div>
                <div style="margin-bottom: 4px;"><strong>Área:</strong> {res['gt_area_cen']:.1f} mm²</div>
                <div><strong>Altura:</strong> N/A</div>
            </td>
        </tr>
        {pipelines_html}
        """

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Comparativo de Tracking e Inferencia</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {{
            --bg-main: #090d16;
            --bg-card: #131a26;
            --border-color: #222d3d;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
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
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .header {{
            background: linear-gradient(135deg, #131a26, #1e293b);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 32px;
            color: var(--accent);
        }}
        
        .header p {{
            color: var(--text-secondary);
            margin: 10px 0 0 0;
            font-size: 16px;
        }}
        
        .explanation-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .explanation-card {{
            background-color: rgba(56, 189, 248, 0.03);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 12px;
            padding: 20px;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .explanation-card h3 {{
            margin-top: 0;
            color: var(--accent);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        
        .card-table {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 30px;
            overflow-x: auto;
        }}
        
        .card-table h2 {{
            margin-top: 0;
            font-size: 22px;
            color: var(--accent);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 12px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            color: var(--text-secondary);
            font-weight: 600;
            background-color: #1e293b;
        }}
        
        .sub-row td {{
            background-color: #0f141e;
        }}
        
        .badge {{
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
        }}
        .badge-ok {{ background-color: rgba(16, 185, 129, 0.15); color: var(--ok); border: 1px solid var(--ok); }}
        .badge-bad {{ background-color: rgba(239, 68, 68, 0.15); color: var(--bad); border: 1px solid var(--bad); }}
        .badge-neutral {{ background-color: rgba(148, 163, 184, 0.15); color: var(--text-secondary); border: 1px solid var(--text-secondary); }}
        
        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(9, 13, 22, 0.85);
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: center;
        }}
        .modal-content {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            max-width: 90%;
            max-height: 90%;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            position: relative;
            animation: fadeIn 0.3s ease-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: scale(0.95); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        .modal-close {{
            position: absolute;
            top: 15px;
            right: 20px;
            color: var(--text-secondary);
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.2s;
        }}
        .modal-close:hover {{
            color: var(--accent);
        }}
        .modal-img {{
            max-width: 100%;
            max-height: 75vh;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .modal-title {{
            margin-top: 15px;
            margin-bottom: 5px;
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte Comparativo de Tracking e Inferencia</h1>
            <p>Análisis de desempeño entre arquitecturas CLASSIC, HYBRID y POSE_ONLY</p>
        </div>
        
        <div class="explanation-grid">
            <div class="explanation-card">
                <h3>🛠️ CLASSIC Pipeline</h3>
                <p><strong>Proceso:</strong> Bounding Boxes (YOLO) → Segmentación de Máscara (MobileSAM) → Keypoints (YOLO-Pose).</p>
                <p>Pipeline tradicional, de gran precisión en extracción de máscaras gracias a SAM, pero con mayor costo computacional (bajos FPS) al ejecutar 3 modelos secuenciales.</p>
            </div>
            <div class="explanation-card">
                <h3>⚡ HYBRID Pipeline</h3>
                <p><strong>Proceso:</strong> Keypoints (YOLO-Pose) → Bounding Boxes derivadas → Segmentación de Máscara (MobileSAM).</p>
                <p>Elimina el YOLO estándar reaprovechando la inferencia de Pose para generar Bounding Boxes perimetrales antes de pasarlos a SAM. Mantiene la misma precisión semántica con menor latencia global.</p>
            </div>
            <div class="explanation-card">
                <h3>🚀 POSE_ONLY Pipeline</h3>
                <p><strong>Proceso:</strong> Keypoints (YOLO-Pose) → Cálculo de Área vía Convex Hull.</p>
                <p>Elimina MobileSAM por completo. El área y el color de la pieza se estiman directamente usando la envolvente convexa geométrica sobre los keypoints detectados. Es el método más rápido (altos FPS) pero puede perder precisión si los keypoints son ruidosos o incompletos.</p>
            </div>
        </div>
        
        <div class="card-table">
            <h2>Detalle de Inferencia Multi-Pipeline por Pieza Trackeada ({total_tracks} evaluadas)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Imágenes (Crops)</th>
                        <th>Track ID / GT</th>
                        <th>Ground Truth (Real)</th>
                        <th>Pipeline</th>
                        <th>Color Inferido</th>
                        <th>Área Cenital (Error %)</th>
                        <th>Altura Lateral (Error %)</th>
                        <th>Resultado Inferencia</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Modal HTML Container -->
    <div id="imageModal" class="modal" onclick="closeImageModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="modal-close" onclick="document.getElementById('imageModal').style.display='none'">&times;</span>
            <img id="modalImg" class="modal-img" src="" alt="Full View">
            <div id="modalTitle" class="modal-title">Vista de Referencia</div>
        </div>
    </div>

    <script>
        function openImageModal(imgUrl, title) {{
            if (!imgUrl) return;
            document.getElementById('modalImg').src = imgUrl;
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('imageModal').style.display = 'flex';
        }}
        function closeImageModal(event) {{
            document.getElementById('imageModal').style.display = 'none';
        }}
        // Cerrar con Escape
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                document.getElementById('imageModal').style.display = 'none';
            }}
        }});
    </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[SUCCESS] Reporte comparativo guardado en: {out_html_path}")

def main():
    parser = argparse.ArgumentParser(description="Genera el reporte comparativo de tracking e inferencia.")
    parser.add_argument("--consolidated_classic", type=str, default=DEFAULT_CONSOLIDATED_CLASSIC)
    parser.add_argument("--consolidated_hybrid", type=str, default=DEFAULT_CONSOLIDATED_HYBRID)
    parser.add_argument("--consolidated_pose", type=str, default=DEFAULT_CONSOLIDATED_POSE)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_HTML)
    parser.add_argument("--limit", type=int, default=0, help="Limita la evaluación a un número de tracks aleatorios.")
    args = parser.parse_args()
    
    if not os.path.exists(args.consolidated_classic):
        print(f"[ERROR] Archivo CLASSIC no encontrado: {args.consolidated_classic}")
        return
        
    if not os.path.exists(args.metadata):
        print(f"[ERROR] Metadatos de simulación no encontrados: {args.metadata}")
        return
        
    build_general_report(
        args.consolidated_classic, 
        args.consolidated_hybrid, 
        args.consolidated_pose, 
        args.metadata, 
        args.out, 
        args.limit
    )

if __name__ == "__main__":
    main()
