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

from color_classifier_all import ColorClassifierAll, rgb_to_lab, delta_e_ciede2000
from run_evaluation_all import estimate_color_mlp_features

device = "mps" if torch.backends.mps.is_available() else "cpu"
hierarchical_clf = ColorClassifierAll(device=device)
sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

# Load metadata
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

# Collect all database colors to restrict the search space
from efficientnet_classifier_all import LegoEfficientNetClassifierAll
clf = LegoEfficientNetClassifierAll()
db_color_names = set()
for colors in clf.part_to_colors.values():
    db_color_names.update(colors)

print(f"Total unique color names in database parts: {len(db_color_names)}")

def get_sam_mask(img_pil, bbox_px, img_hw):
    h_c, w_c = img_hw
    px1, py1, px2, py2 = bbox_px
    mask_bin = np.zeros((h_c, w_c), dtype=np.uint8)
    if px2 > px1 and py2 > py1:
        sam_res = sam_model(np.array(img_pil.convert("RGB")), bboxes=[[px1, py1, px2, py2]], verbose=False)
        if sam_res and sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
            mask_sam = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_bin = cv2.morphologyEx(mask_sam, cv2.MORPH_OPEN, kernel)
            mask_bin = cv2.resize(mask_bin, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
    return mask_bin

total = 0
correct_cen = 0
correct_lat = 0
correct_fused = 0

for idx, frame in enumerate(meta_data["frames"][:97]):
    piece = frame["visible_pieces"][0]
    ref_gt = piece["ref"]
    c_code_gt = str(piece["color_code"])
    c_name_gt = piece["color_name"]
    
    img_cen_path = os.path.join(project_root, "data", "simulation_100_all", frame["file_name"])
    img_lat_path = os.path.join(project_root, "data", "simulation_100_all", frame["file_name_frontal"])
    
    if not os.path.exists(img_cen_path) or not os.path.exists(img_lat_path):
        continue
        
    total += 1
    
    # Cenital:
    img_c = Image.open(img_cen_path)
    w_c, h_c = img_c.size
    bbox_cen = piece["bbox_cenital_norm"]
    px1 = max(0, int(bbox_cen[0] * w_c)); py1 = max(0, int(bbox_cen[1] * h_c))
    px2 = min(w_c, int(bbox_cen[2] * w_c)); py2 = min(h_c, int(bbox_cen[3] * h_c))
    mask_c = get_sam_mask(img_c, [px1, py1, px2, py2], (h_c, w_c))
    feat_cen = estimate_color_mlp_features(img_c, mask_c, "cenital", None, is_simulation=True)
    
    # Lateral:
    img_l = Image.open(img_lat_path)
    w_l, h_l = img_l.size
    bbox_lat = piece["bbox_frontal_norm"]
    px1_l = max(0, int(bbox_lat[0] * w_l)); py1_l = max(0, int(bbox_lat[1] * h_l))
    px2_l = min(w_l, int(bbox_lat[2] * w_l)); py2_l = min(h_l, int(bbox_lat[3] * h_l))
    mask_l = get_sam_mask(img_l, [px1_l, py1_l, px2_l, py2_l], (h_l, w_l))
    feat_lat = estimate_color_mlp_features(img_l, mask_l, "lateral", None, is_simulation=True)
    
    # Inferencia de color sin gating (o con gating solo de los colores de la BD)
    p_cen = hierarchical_clf.predict_gated_probs_cielab(feat_cen, db_color_names, "cenital", is_simulation=True)
    p_lat = hierarchical_clf.predict_gated_probs_cielab(feat_lat, db_color_names, "lateral", is_simulation=True)
    
    # Fusionado (con alpha = 0.8)
    p_combined = (p_cen ** 0.8) * (p_lat ** 0.2)
    if np.sum(p_combined) == 0:
        p_combined = p_cen
        
    pred_cen = hierarchical_clf.classes[np.argmax(p_cen)]
    pred_lat = hierarchical_clf.classes[np.argmax(p_lat)]
    pred_fused = hierarchical_clf.classes[np.argmax(p_combined)]
    
    if pred_cen.strip().lower() == c_name_gt.strip().lower():
        correct_cen += 1
    if pred_lat.strip().lower() == c_name_gt.strip().lower():
        correct_lat += 1
    if pred_fused.strip().lower() == c_name_gt.strip().lower():
        correct_fused += 1
        
    print(f"Sample {total:02d} | GT: {c_name_gt:20s} | Cen: {pred_cen:20s} | Lat: {pred_lat:20s} | Fused: {pred_fused:20s}")

print("\n" + "="*50)
print(f"Accuracy Cenital: {correct_cen/total*100:.2f}% ({correct_cen}/{total})")
print(f"Accuracy Lateral: {correct_lat/total*100:.2f}% ({correct_lat}/{total})")
print(f"Accuracy Fused:   {correct_fused/total*100:.2f}% ({correct_fused}/{total})")
print("="*50)
