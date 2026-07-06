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

device = "mps" if torch.backends.mps.is_available() else "cpu"
sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

# Probar la primera pieza visible
frame = meta_data["frames"][4] # frame_004
piece = frame["visible_pieces"][0]
print(f"Pieza de prueba: ref={piece['ref']}, color={piece['color_name']}")

img_lat_name = frame["file_name_frontal"]
img_lat_path = os.path.join(project_root, "data", "simulation_100_all", img_lat_name)
print(f"Cargando imagen lateral: {img_lat_path}")
img_lat = Image.open(img_lat_path)
w_l, h_l = img_lat.size

bbox_lat_n = piece["bbox_frontal_norm"]
x1_l, y1_l, x2_l, y2_l = bbox_lat_n
px1_l, py1_l = int(x1_l * w_l), int(y1_l * h_l)
px2_l, py2_l = int(x2_l * w_l), int(y2_l * h_l)
print(f"Bbox normalizado: {bbox_lat_n}")
print(f"Bbox píxeles: [{px1_l}, {py1_l}, {px2_l}, {py2_l}]")

sam_res_lat = sam_model(np.array(img_lat.convert("RGB")), bboxes=[[px1_l, py1_l, px2_l, py2_l]], verbose=False)
if sam_res_lat and sam_res_lat[0].masks is not None and len(sam_res_lat[0].masks.data) > 0:
    mask_sam_l = sam_res_lat[0].masks.data[0].cpu().numpy().astype(np.uint8)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_lat = cv2.morphologyEx(mask_sam_l, cv2.MORPH_OPEN, kernel_morph)
    mask_lat = cv2.resize(mask_lat, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
    
    y_cutoff = int(505 * h_l / 1024)
    print(f"y_cutoff calculado: {y_cutoff}")
    print(f"Suma de máscara antes del corte: {np.sum(mask_lat)}")
    mask_lat[y_cutoff:, :] = 0
    print(f"Suma de máscara después del corte: {np.sum(mask_lat)}")
    
    from run_evaluation_all import measure_lateral_height_mm_sam
    raw_h = measure_lateral_height_mm_sam(mask_lat, img_res_px_val=w_l)
    print(f"Altura medida raw_h: {raw_h:.2f} mm")
else:
    print("Error: SAM no generó máscara")
