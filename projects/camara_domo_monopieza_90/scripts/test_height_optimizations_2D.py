# -*- coding: utf-8 -*-
"""test_height_optimizations_2D.py
================================
Calibra la estimación de altura por cámara lateral/frontal usando búsqueda de rejilla 3D
(x_camera, z_camera, px_per_mm) sobre los renders de simulación 2D.
"""
import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import SAM

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
legovic_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.append(project_root)

def compute_iou(boxA, boxB):
    if not boxA or not boxB: return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def measure_lateral_height_px_sam(mask_lat):
    row_counts = np.sum(mask_lat > 0, axis=1)
    max_count = np.max(row_counts) if len(row_counts) > 0 else 0
    if max_count == 0: return 0.0
    threshold = max(3.0, 0.05 * max_count)
    valid_rows = row_counts >= threshold
    if not np.any(valid_rows): return 0.0
    first_row = np.argmax(valid_rows)
    last_row = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
    return float(last_row - first_row + 1)

def main():
    metadata_path = os.path.join(project_root, "data", "simulation_10_2D", "simulation_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: No se encuentra el archivo de metadatos {metadata_path}")
        sys.exit(1)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cargando SAM en {device}...")
    sam_model = SAM("mobile_sam.pt").to(device)

    # Agrupar por track (usando coordenadas absolutas en la cinta)
    from collections import defaultdict
    piece_tracks = defaultdict(list)
    for frame in meta_data["frames"]:
        offset = frame["belt_offset_mm"]
        f_name = frame["file_name"]
        for p in frame["visible_pieces"]:
            x_abs = offset - p["x_belt_local_mm"]
            y_abs = p["y_belt_local_mm"]
            
            matched_key = None
            for key in piece_tracks.keys():
                kx, ky = key
                if abs(x_abs - kx) < 1.0 and abs(y_abs - ky) < 1.0:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (x_abs, y_abs)
                
            piece_tracks[matched_key].append({
                "ref": p["ref"],
                "color_code": p["color_code"],
                "file_name": f_name,
                "x_belt_local_mm": p["x_belt_local_mm"],
                "y_belt_local_mm": p["y_belt_local_mm"],
                "bbox_frontal_norm": p.get("bbox_frontal_norm"),
                "lateral_height_gt": p.get("lateral_height_gt")
            })

    # Cachar las alturas crudas en px de la mejor observación en cinta
    cached_data = []
    print("\n--- Extrayendo alturas en píxeles de renders 2D (caching) ---")
    for key, obs_list in piece_tracks.items():
        valid_obs = [o for o in obs_list if o["x_belt_local_mm"] <= 98.18 and o["bbox_frontal_norm"] is not None]
        if not valid_obs:
            continue
            
        last_ob = max(valid_obs, key=lambda x: x["x_belt_local_mm"])
        gt_height = last_ob["lateral_height_gt"]
        ref = last_ob["ref"]
        
        lat_file = last_ob["file_name"].replace(".png", "_frontal.png")
        path_lat_img = os.path.join(os.path.dirname(metadata_path), lat_file)
        if not os.path.exists(path_lat_img):
            continue
        
        img_lat = Image.open(path_lat_img)
        w_l, h_l = img_lat.size
        gt_bbox_lat = last_ob["bbox_frontal_norm"]
        px1_l, py1_l = int(gt_bbox_lat[0] * w_l), int(gt_bbox_lat[1] * h_l)
        px2_l, py2_l = int(gt_bbox_lat[2] * w_l), int(gt_bbox_lat[3] * h_l)
        
        sam_res_lat = sam_model(np.array(img_lat.convert("RGB")), bboxes=[[px1_l, py1_l, px2_l, py2_l]], verbose=False)
        if not sam_res_lat or sam_res_lat[0].masks is None:
            continue
            
        mask_lat = sam_res_lat[0].masks.data[0].cpu().numpy().astype(np.uint8)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_lat = cv2.morphologyEx(mask_lat, cv2.MORPH_OPEN, kernel_morph)
        mask_lat = cv2.resize(mask_lat, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
        
        raw_height_px = measure_lateral_height_px_sam(mask_lat)
        cached_data.append({
            "ref": ref,
            "gt_height": gt_height,
            "raw_height_px": raw_height_px,
            "x_belt_h": last_ob["x_belt_local_mm"],
            "y_belt_h": last_ob["y_belt_local_mm"],
            "w_l": w_l
        })
        print(f"Ref {ref:5s} | GT={gt_height:5.2f}mm | RawHeightPx={raw_height_px:.1f}px (X={last_ob['x_belt_local_mm']:.1f}, Y={last_ob['y_belt_local_mm']:.1f})")

    # Realizar grid-search de calibración 3D
    print(f"\n--- Ejecutando Grid-Search 3D en {len(cached_data)} muestras ---")
    best_err = float("inf")
    best_x_cam = 0.0
    best_z_cam = 0.0
    best_px_per_mm = 0.0
    
    # Rango de X_camera: 230mm a 320mm (paso 2mm)
    x_cam_vals = np.linspace(230.0, 320.0, 46)
    # Rango de Z_camera: 2.0mm a 30.0mm (paso 1mm)
    z_cam_vals = np.linspace(2.0, 30.0, 29)
    # Rango de px_per_mm_lat: 9.5 a 11.5 (paso 0.05)
    px_mm_vals = np.linspace(9.5, 11.5, 41)
    
    for x_cam in x_cam_vals:
        for z_cam in z_cam_vals:
            for px_mm in px_mm_vals:
                errors = []
                for item in cached_data:
                    px_per_mm_lat_scaled = px_mm * (item["w_l"] / 2048.0)
                    raw_height_mm = item["raw_height_px"] / px_per_mm_lat_scaled
                    
                    # Perspectiva 3D
                    D_center = np.sqrt((98.18 - x_cam)**2 + z_cam**2)
                    D_piece = np.sqrt((item["x_belt_h"] - x_cam)**2 + item["y_belt_h"]**2 + z_cam**2)
                    
                    pred_height = raw_height_mm * (D_piece / D_center)
                    err = abs(pred_height - item["gt_height"]) / item["gt_height"] * 100.0
                    errors.append(err)
                
                mean_err = np.mean(errors)
                if mean_err < best_err:
                    best_err = mean_err
                    best_x_cam = x_cam
                    best_z_cam = z_cam
                    best_px_per_mm = px_mm

    print("\n" + "="*60)
    print("RESULTADOS DE OPTIMIZACIÓN DE CALIBRACIÓN 3D (GRID-SEARCH)")
    print("="*60)
    print(f"Mejor Posición de Cámara X (x_cam): {best_x_cam:.2f} mm")
    print(f"Mejor Altura de Cámara Z (z_cam):   {best_z_cam:.2f} mm")
    print(f"Mejor Escala Lateral (px_per_mm):   {best_px_per_mm:.3f} px/mm")
    print(f"Error Medio Mínimo Logrado:         {best_err:.2f}%")
    print("="*60)
    
    # Mostrar el desglose con los mejores parámetros
    print("\nDesglose de errores con la calibración optimizada:")
    for item in cached_data:
        px_per_mm_lat_scaled = best_px_per_mm * (item["w_l"] / 2048.0)
        raw_height_mm = item["raw_height_px"] / px_per_mm_lat_scaled
        D_center = np.sqrt((98.18 - best_x_cam)**2 + best_z_cam**2)
        D_piece = np.sqrt((item["x_belt_h"] - best_x_cam)**2 + item["y_belt_h"]**2 + best_z_cam**2)
        pred_height = raw_height_mm * (D_piece / D_center)
        err = abs(pred_height - item["gt_height"]) / item["gt_height"] * 100.0
        err_mm = abs(pred_height - item["gt_height"])
        print(f"Ref {item['ref']:5s} | GT={item['gt_height']:5.2f}mm | Pred={pred_height:5.2f}mm | Error={err_mm:.2f}mm ({err:5.2f}%)")

if __name__ == "__main__":
    main()
