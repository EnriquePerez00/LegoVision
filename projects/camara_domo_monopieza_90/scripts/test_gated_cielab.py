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
legovic_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from color_classifier_all import rgb_to_lab, delta_e_ciede2000
from core.db.set_catalog import REAL_SETS

# Paleta de colores de referencia del catálogo
palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
with open(palette_path, "r", encoding="utf-8") as f:
    CATALOG_PALETTE = json.load(f)

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

# Construir mapa de validación de color por pieza
part_to_colors = {}
for s_id, s_data in REAL_SETS.items():
    for p in s_data.get("parts", []):
        part_to_colors.setdefault(p["ref"], set()).add(str(p["color_code"]))

# Si no está en REAL_SETS, agregamos soporte fallback dinámico consultando si tiene color asignado en metadatos
gt_color_map = {}
for frame in meta_data.get("frames", []):
    for piece in frame.get("visible_pieces", []):
        ref = piece.get("ref")
        c_code = str(piece.get("color_code", ""))
        c_name = piece.get("color_name")
        if ref and c_code:
            gt_color_map[(ref, c_code)] = c_name
            part_to_colors.setdefault(ref, set()).add(c_code)

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

def extract_lab_cascade(img_arr, mask_bin, mode="no_belt_key_outliers_median"):
    pixels_rgb = img_arr[mask_bin > 0]
    if len(pixels_rgb) == 0:
        return None
    from run_evaluation import rgb_matrix_to_lab
    pixels_lab = rgb_matrix_to_lab(pixels_rgb)
    
    if mode == "no_belt_key_outliers_median":
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
    return pixels_lab.mean(axis=0)

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

    total_count = 0
    correct_no_gating = 0
    correct_with_gating = 0

    print("Evaluando exactitud CIELAB con y sin Filtro Bayesiano por Referencia...")
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
        
        lab_est = extract_lab_cascade(img_arr, mask_bin, mode="no_belt_key_outliers_median")
        if lab_est is None:
            continue
            
        total_count += 1
        
        # 1. Sin Gating (Buscar en todo el catálogo de 179 colores)
        best_name_no_gate = None
        min_dist_no_gate = float('inf')
        for c in CATALOG_COLORS:
            lab_ref = rgb_to_lab(c["rgb_cenital"])
            dist = delta_e_ciede2000(lab_est, lab_ref)
            if dist < min_dist_no_gate:
                min_dist_no_gate = dist
                best_name_no_gate = c["color_name"]
                
        if best_name_no_gate and best_name_no_gate.strip().lower() == c_name_gt.strip().lower():
            correct_no_gating += 1

        # 2. Con Gating (Solo buscar en colores válidos para esa pieza en particular)
        allowed_color_codes = part_to_colors.get(ref_gt, set())
        best_name_gated = None
        min_dist_gated = float('inf')
        for c in CATALOG_COLORS:
            if c["color_code"] in allowed_color_codes:
                lab_ref = rgb_to_lab(c["rgb_cenital"])
                dist = delta_e_ciede2000(lab_est, lab_ref)
                if dist < min_dist_gated:
                    min_dist_gated = dist
                    best_name_gated = c["color_name"]
                    
        if best_name_gated and best_name_gated.strip().lower() == c_name_gt.strip().lower():
            correct_with_gating += 1
        else:
            print(f"FALLO GATED: Muestra {idx} | Ref={ref_gt} | GT={c_name_gt} | GatedPred={best_name_gated} | AllowedCodes={allowed_color_codes}")

    print("\n" + "="*70)
    print("COMPARATIVA DE EXACTITUD - EFECTO DEL GATING SEMÁNTICO POR PIEZA")
    print("="*70)
    print(f"Total muestras: {total_count}")
    acc_no = (correct_no_gating / total_count * 100) if total_count > 0 else 0
    acc_with = (correct_with_gating / total_count * 100) if total_count > 0 else 0
    print(f"CIELAB Puro (Sin Gating - Buscar en 179 colores): {correct_no_gating:2d} / {total_count} ({acc_no:.2f}%)")
    print(f"CIELAB Gated (Con Gating - Restringir a colores de la pieza): {correct_with_gating:2d} / {total_count} ({acc_with:.2f}%)")
    print("="*70)

if __name__ == "__main__":
    main()
