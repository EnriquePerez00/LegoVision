# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2
import torch
import math
from PIL import Image

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
legovic_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from color_classifier_all import ColorClassifierAll, rgb_to_lab, delta_e_ciede2000
from run_evaluation import rgb_matrix_to_lab

device = "mps" if torch.backends.mps.is_available() else "cpu"
hierarchical_clf = ColorClassifierAll(device=device)

# Load metadata
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

frame = meta_data["frames"][4] # frame_004
piece = frame["visible_pieces"][0]
ref_gt = piece["ref"]
c_code_gt = str(piece["color_code"])
c_name_gt = piece["color_name"]

img_cen_path = os.path.join(project_root, "data", "simulation_100_all", frame["file_name"])
img_pil = Image.open(img_cen_path)
w_c, h_c = img_pil.size
bbox_cen = piece["bbox_cenital_norm"]
px1 = max(0, int(bbox_cen[0] * w_c)); py1 = max(0, int(bbox_cen[1] * h_c))
px2 = min(w_c, int(bbox_cen[2] * w_c)); py2 = min(h_c, int(bbox_cen[3] * h_c))

from ultralytics import SAM
sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

sam_res = sam_model(np.array(img_pil.convert("RGB")), bboxes=[[px1, py1, px2, py2]], verbose=False)
mask_sam = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
mask_sam = cv2.resize(mask_sam, (w_c, h_c), interpolation=cv2.INTER_NEAREST)

# Extract pixels
img_arr = np.array(img_pil.convert("RGB"))
pixels_rgb = img_arr[mask_sam > 0]

# Convert to HSV to filter belt
pixels_rgb_reshaped = pixels_rgb.reshape(-1, 1, 3)
pixels_hsv = cv2.cvtColor(pixels_rgb_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3)

# Chroma-keying de la cinta transportadora.
# Rango HSV derivado dinámicamente de scripts.scene_config.BELT_COLOR_HEX
# (fuente única de verdad). Ver projects/.../scripts/_belt_mask.py
from _belt_mask import compute_belt_mask as _compute_belt_mask
belt_mask = _compute_belt_mask(pixels_hsv)
non_belt_mask = ~belt_mask

print(f"Original pixels: {len(pixels_rgb)}, Kept pixels: {np.sum(non_belt_mask)}")

pixels_rgb_filtered = pixels_rgb[non_belt_mask]
if len(pixels_rgb_filtered) > 0:
    median_rgb = np.median(pixels_rgb_filtered, axis=0)
    print(f"Median RGB of piece (with belt keying): {median_rgb.tolist()}")
    
    pixels_lab = rgb_matrix_to_lab(pixels_rgb_filtered)
    lab_est = np.median(pixels_lab, axis=0)
    print(f"LAB Estimado: {lab_est}")
    
    # Compare with catalog
    gain = 1.56
    for c in hierarchical_clf.catalog_colors:
        if c["color_name"] in ["Dark Bluish Gray", "Dark Azure", "White", "Black", "Light Bluish Gray"]:
            hex_str = c["color_hex"].lstrip("#")
            rgb_ref_nominal = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
            rgb_ref_sim = np.clip(rgb_ref_nominal * gain, 0, 255)
            lab_ref = rgb_to_lab(rgb_ref_sim)
            dist = delta_e_ciede2000(lab_est, lab_ref)
            print(f"Catalog {c['color_name']:20s} | CIELAB Dist: {dist:.4f}")
else:
    print("All pixels were filtered out as belt!")
