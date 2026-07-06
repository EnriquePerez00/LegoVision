# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
sys.path.append(project_root)

# Import the classifier from the project scripts
from color_classifier_75078 import ColorClassifier75078
from run_evaluation import CATALOG_COLORS, rgb_matrix_to_lab, rgb_to_lab
ccm_path = os.path.join(project_root, "data", "ccm_dome_light.json")
with open(ccm_path, "r") as f:
    ccm_params = json.load(f)

def get_flat_non_belt_pixels(pixels_rgb, pixels_hsv):
    # Chroma-keying de la cinta: fuente única de verdad = scripts.scene_config
    from _belt_mask import filter_out_belt as _filter_out_belt_pixels
    return _filter_out_belt_pixels(pixels_rgb, pixels_hsv)

def extract_features_custom(img, bbox_norm, mode="erosion_filter", roi_focus=False, camera_type=None, ccm_params=None):
    w, h = img.size
    x1, y1, x2, y2 = bbox_norm
    
    if roi_focus:
        # ROI Focus: 50% central bounding box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        rx = (x2 - x1) * 0.5
        ry = (y2 - y1) * 0.5
        # Reducir el radio a la raíz de 0.5 para tener 50% de área: sqrt(0.5) ≈ 0.707
        x1 = cx - rx * 0.707
        x2 = cx + rx * 0.707
        y1 = cy - ry * 0.707
        y2 = cy + ry * 0.707

    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    px1 = max(0, px1); py1 = max(0, py1)
    px2 = min(w, px2); py2 = min(h, py2)
    if px2 <= px1 or py2 <= py1:
        return None

    # Crear máscara de la bounding box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[py1:py2, px1:px2] = 1

    # Aplicar erosión morfológica 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask
    
    img_arr = np.array(img.convert("RGB"))
    pixels_rgb = img_arr[mask_to_use > 0]
    
    if len(pixels_rgb) == 0:
        return None

    # Aplicar calibración CCM inversa si está configurada
    if ccm_params and camera_type in ccm_params:
        params = ccm_params[camera_type]
        pixels_rgb_cal = np.zeros_like(pixels_rgb, dtype=np.float32)
        for c in range(3):
            gamma, scale, lift = params[f"channel_{c}"]
            channel_data = pixels_rgb[:, c].astype(np.float32)
            val = (channel_data - lift) / (scale + 1e-8)
            val = np.clip(val, 0.0, 1.0)
            pixels_rgb_cal[:, c] = 255.0 * np.power(val, 1.0 / gamma)
        pixels_rgb = np.clip(pixels_rgb_cal, 0.0, 255.0).astype(np.uint8)

    pixels_rgb_reshaped = pixels_rgb.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(pixels_rgb_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Filtrar fondo de cinta
    pixels_rgb, pixels_hsv = get_flat_non_belt_pixels(pixels_rgb, pixels_hsv)
    if len(pixels_rgb) == 0:
        return None

    # Filtrar píxeles extremos (brillos y sombras) en HSV
    # Brillos: V > 230 (~90%) o S < 25 (brillo metálico/gris)
    # Sombras: V < 64 (~25%)
    hsv_v = pixels_hsv[:, 2]
    hsv_s = pixels_hsv[:, 1]
    # Filtro: brillo >= 50 (sombras quitadas) Y (saturación >= 25 O brillo < 230) (especulares quitados)
    valid_color_mask = (hsv_v >= 50) & ((hsv_s >= 25) | (hsv_v < 230))
    
    if np.any(valid_color_mask):
        pixels_rgb = pixels_rgb[valid_color_mask]
        pixels_hsv = pixels_hsv[valid_color_mask]

    if len(pixels_rgb) == 0:
        return None

    pixels_lab = rgb_matrix_to_lab(pixels_rgb)
    mean_lab = pixels_lab.mean(axis=0)
    std_lab = pixels_lab.std(axis=0)
    mean_hsv = pixels_hsv.mean(axis=0)
    std_hsv = pixels_hsv.std(axis=0)

    return np.array([
        mean_lab[0], std_lab[0],
        mean_lab[1], std_lab[1],
        mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0],
        mean_hsv[1], std_hsv[1],
        mean_hsv[2], std_hsv[2]
    ], dtype=np.float32)

def main():
    metadata_path = os.path.join(project_root, "data", "simulation_10", "simulation_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clf = ColorClassifier75078(device=device)

    # Cargar observaciones
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
                "bbox_cenital_norm": p["bbox_cenital_norm"],
                "x_belt_local_mm": p["x_belt_local_mm"]
            })

    correct_baseline = 0
    correct_erosion_filter = 0
    correct_roi_focus = 0
    total_samples = 0

    print("\n--- Evaluando optimizaciones de color en muestras ---")
    for key, obs_list in piece_tracks.items():
        # Tomar la mejor observación (más cercana al centro cenital)
        best_ob = min(obs_list, key=lambda x: abs(x["x_belt_local_mm"]))
        gt_color_code = best_ob["color_code"]
        
        # Buscar el nombre real del color en el catálogo
        gt_color_name = "Unknown"
        for c in CATALOG_COLORS:
            if c["color_code"] == gt_color_code:
                gt_color_name = c["color_name"]
                break
        
        cen_file = best_ob["file_name"]
        path_cen_img = os.path.join(os.path.dirname(metadata_path), cen_file)
        if not os.path.exists(path_cen_img):
            continue
            
        img_c = Image.open(path_cen_img)
        bbox = best_ob["bbox_cenital_norm"]
        total_samples += 1

        # 1. Baseline (con el extractor de características actual del proyecto)
        from run_evaluation import estimate_color_mlp_features_fast
        feat_base = estimate_color_mlp_features_fast(img_c, bbox, "cenital", ccm_params)
        pred_base = "Unknown"
        if feat_base is not None:
            p_cen = clf.predict_cenital_probs(feat_base)
            if np.sum(p_cen) > 0:
                pred_base = clf.classes[np.argmax(p_cen)]
        
        # 2. Erosión 3x3 + Filtrado de píxeles extremos
        feat_opt = extract_features_custom(img_c, bbox, roi_focus=False, camera_type="cenital", ccm_params=ccm_params)
        pred_opt = "Unknown"
        if feat_opt is not None:
            p_cen = clf.predict_cenital_probs(feat_opt)
            if np.sum(p_cen) > 0:
                pred_opt = clf.classes[np.argmax(p_cen)]

        # 3. Erosión 3x3 + Filtrado + ROI Focus (50% area)
        feat_roi = extract_features_custom(img_c, bbox, roi_focus=True, camera_type="cenital", ccm_params=ccm_params)
        pred_roi = "Unknown"
        if feat_roi is not None:
            p_cen = clf.predict_cenital_probs(feat_roi)
            if np.sum(p_cen) > 0:
                pred_roi = clf.classes[np.argmax(p_cen)]

        is_base_ok = (pred_base.strip().lower() == gt_color_name.strip().lower())
        is_opt_ok = (pred_opt.strip().lower() == gt_color_name.strip().lower())
        is_roi_ok = (pred_roi.strip().lower() == gt_color_name.strip().lower())

        if is_base_ok: correct_baseline += 1
        if is_opt_ok: correct_erosion_filter += 1
        if is_roi_ok: correct_roi_focus += 1

        print(f"Ref {best_ob['ref']:5s} | GT={gt_color_name:12s} | Base={pred_base:12s} {'✓' if is_base_ok else '✗'} | Opt={pred_opt:12s} {'✓' if is_opt_ok else '✗'} | ROI={pred_roi:12s} {'✓' if is_roi_ok else '✗'}")

    print("\n" + "="*60)
    print("COMPARATIVA DE OPTIMIZACIÓN DE COLOR CENITAL")
    print("="*60)
    print(f"Muestras evaluadas: {total_samples}")
    print(f"1. Baseline actual:                     {correct_baseline} / {total_samples} ({correct_baseline/total_samples*100:.1f}%)")
    print(f"2. Erosión 3x3 + Filtro Brillo/Sombra:  {correct_erosion_filter} / {total_samples} ({correct_erosion_filter/total_samples*100:.1f}%)")
    print(f"3. Erosión 3x3 + Filtro + ROI Focus 50%: {correct_roi_focus} / {total_samples} ({correct_roi_focus/total_samples*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
