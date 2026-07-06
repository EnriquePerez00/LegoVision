# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2
import torch
from PIL import Image

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
legovic_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from color_classifier_all import ColorClassifierAll, rgb_to_lab, delta_e_ciede2000
from run_evaluation_all import estimate_color_mlp_features
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
pixels_lab = rgb_matrix_to_lab(pixels_rgb)

# Sort pixels by L channel
L_vals = pixels_lab[:, 0]
sorted_indices = np.argsort(L_vals)

# Keep only the bottom 60% (remove top 40% brightest specular highlights)
cutoff_idx = int(len(sorted_indices) * 0.6)
kept_indices = sorted_indices[:cutoff_idx]

pixels_lab_filtered = pixels_lab[kept_indices]
lab_est = np.median(pixels_lab_filtered, axis=0)

print(f"GT Color Name: {c_name_gt}")
print(f"LAB Estimado con filtro de brillo (bottom 60%): {lab_est}")

# Distancia a catalog colors
for c in hierarchical_clf.catalog_colors:
    if c["color_name"] in ["Dark Bluish Gray", "Dark Azure", "White", "Black", "Light Bluish Gray"]:
        hex_str = c["color_hex"].lstrip("#")
        rgb_ref = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
        lab_ref = rgb_to_lab(rgb_ref)
        dist = delta_e_ciede2000(lab_est, lab_ref)
        print(f"Catalog {c['color_name']:20s} | Hex: {c['color_hex']} | LAB Ref: {lab_ref} | Dist: {dist:.4f}")
