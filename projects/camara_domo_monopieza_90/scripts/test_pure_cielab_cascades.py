# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import SAM

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

from color_classifier_all import rgb_to_lab, delta_e_ciede2000

# Paleta de colores de referencia del catálogo
palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
with open(palette_path, "r", encoding="utf-8") as f:
    CATALOG_PALETTE = json.load(f)

# Reconstruir catálogo en formato rápido
CATALOG_COLORS = []
for item in CATALOG_PALETTE:
    CATALOG_COLORS.append({
        "color_code": str(item.get("color_code", "")),
        "color_name": item.get("color_name", "Unknown"),
        "color_hex": item.get("color_hex", "#808080"),
        "rgb_cenital": np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float)
    })

# Cargar metadatos del dataset simulación 100
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"
sam_model = SAM(os.path.join(os.path.dirname(project_root), "mobile_sam.pt")).to(device)

# Construir mapa de (ref, color_code) -> color_name
gt_color_map = {}
for frame in meta_data.get("frames", []):
    for piece in frame.get("visible_pieces", []):
        ref = piece.get("ref")
        c_code = str(piece.get("color_code", ""))
        c_name = piece.get("color_name")
        if ref and c_code:
            gt_color_map[(ref, c_code)] = c_name

def get_sam_mask(img_pil, bbox_px, img_hw):
    h_c, w_c = img_hw
    px1, py1, px2, py2 = bbox_px
    mask_bin = np.zeros((h_c, w_c), dtype=np.uint8)
    if px2 > px1 and py2 > py1:
        sam_res = sam_model(np.array(img_pil.convert("RGB")), bboxes=[[px1, py1, px2, py2]], verbose=False)
        if sam_res and sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
            mask_sam = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            if mask_sam.shape != (h_c, w_c):
                mask_sam = cv2.resize(mask_sam, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
            mask_bin = mask_sam
        else:
            mask_bin[py1:py2, px1:px2] = 1
    return mask_bin

def extract_lab_cascade(img_arr, mask_bin, mode="baseline"):
    pixels_rgb = img_arr[mask_bin > 0]
    if len(pixels_rgb) == 0:
        return None

    pixels_rgb_reshaped = pixels_rgb.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(pixels_rgb_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    if mode == "baseline":
        # Chroma-keying de la cinta transportadora.
        # Rango HSV derivado dinámicamente de scripts.scene_config.BELT_COLOR_HEX
        # (fuente única de verdad). Ver projects/.../scripts/_belt_mask.py
        from _belt_mask import filter_out_belt as _filter_out_belt_pixels
        pixels_rgb, pixels_hsv = _filter_out_belt_pixels(pixels_rgb, pixels_hsv)
        
        from run_evaluation import rgb_matrix_to_lab
        pixels_lab = rgb_matrix_to_lab(pixels_rgb)
        l_vals = pixels_lab[:, 0]
        p25, p75 = np.percentile(l_vals, 25), np.percentile(l_vals, 75)
        valid_mask = (l_vals >= p25) & (l_vals <= p75)
        if np.any(valid_mask):
            pixels_lab = pixels_lab[valid_mask]
        return pixels_lab.mean(axis=0)
        
    elif mode == "no_belt_key":
        from run_evaluation import rgb_matrix_to_lab
        pixels_lab = rgb_matrix_to_lab(pixels_rgb)
        l_vals = pixels_lab[:, 0]
        p25, p75 = np.percentile(l_vals, 25), np.percentile(l_vals, 75)
        valid_mask = (l_vals >= p25) & (l_vals <= p75)
        if np.any(valid_mask):
            pixels_lab = pixels_lab[valid_mask]
        return pixels_lab.mean(axis=0)
        
    elif mode == "no_belt_key_median":
        from run_evaluation import rgb_matrix_to_lab
        pixels_lab = rgb_matrix_to_lab(pixels_rgb)
        l_vals = pixels_lab[:, 0]
        p25, p75 = np.percentile(l_vals, 25), np.percentile(l_vals, 75)
        valid_mask = (l_vals >= p25) & (l_vals <= p75)
        if np.any(valid_mask):
            pixels_lab = pixels_lab[valid_mask]
        return np.median(pixels_lab, axis=0)
        
    elif mode == "no_belt_key_outliers_median":
        from run_evaluation import rgb_matrix_to_lab
        pixels_lab = rgb_matrix_to_lab(pixels_rgb)
        valid_mask = np.ones(len(pixels_lab), dtype=bool)
        for i in range(3):
            q25, q75 = np.percentile(pixels_lab[:, i], 25), np.percentile(pixels_lab[:, i], 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.2 * iqr
            upper_bound = q75 + 1.2 * iqr
            valid_mask &= (pixels_lab[:, i] >= lower_bound) & (pixels_lab[:, i] <= upper_bound)
        if np.any(valid_mask):
            pixels_lab = pixels_lab[valid_mask]
        return np.median(pixels_lab, axis=0)
        
    return None

def main():
    renders = []
    for f in meta_data.get("frames", []):
        for piece in f.get("visible_pieces", []):
            renders.append({
                "ref": piece["ref"],
                "color_code": piece["color_code"],
                "image_path": os.path.join(project_root, "data", "simulation_100_all", f["file_name"]),
                "bbox_norm": piece["bbox_cenital_norm"]
            })

    modes = ["baseline", "no_belt_key", "no_belt_key_median", "no_belt_key_outliers_median"]
    correct_counts = {m: 0 for m in modes}
    total_count = 0

    print("Evaluando exactitud CIELAB puro (sin MLP Router)...")
    for idx, entry in enumerate(renders[:97]):
        ref_gt = entry["ref"]
        c_code_gt = str(entry["color_code"])
        c_name_gt = gt_color_map.get((ref_gt, c_code_gt))
        if not c_name_gt:
            continue
            
        img_path = entry["image_path"]
        if not os.path.exists(img_path):
            continue
            
        img_pil = Image.open(img_path)
        img_arr = np.array(img_pil.convert("RGB"))
        h_c, w_c = img_pil.size[1], img_pil.size[0]
        bbox = entry["bbox_norm"]
        
        px1, py1, px2, py2 = int(bbox[0]*w_c), int(bbox[1]*h_c), int(bbox[2]*w_c), int(bbox[3]*h_c)
        mask_bin = get_sam_mask(img_pil, [px1, py1, px2, py2], (h_c, w_c))
        
        total_count += 1
        
        for m in modes:
            lab_est = extract_lab_cascade(img_arr, mask_bin, mode=m)
            if lab_est is not None:
                # Encontrar el color más cercano del catálogo usando Delta E 2000
                best_name = None
                min_dist = float('inf')
                for c in CATALOG_COLORS:
                    lab_ref = rgb_to_lab(c["rgb_cenital"])
                    dist = delta_e_ciede2000(lab_est, lab_ref)
                    if dist < min_dist:
                        min_dist = dist
                        best_name = c["color_name"]
                
                if best_name and best_name.strip().lower() == c_name_gt.strip().lower():
                    correct_counts[m] += 1

    print("\n" + "="*70)
    print("COMPARATIVA DE EXACTITUD - PURE CIELAB (SIN MLP ROUTER)")
    print("="*70)
    print(f"Total muestras: {total_count}")
    for m in modes:
        acc = (correct_counts[m] / total_count * 100) if total_count > 0 else 0
        print(f"Modo: {m:28s} -> {correct_counts[m]:2d} / {total_count} ({acc:.2f}%)")
    print("="*70)

if __name__ == "__main__":
    main()
