# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

# Instanciar SAM localmente
from ultralytics import SAM
device = "cpu"
sam_model = SAM(os.path.join("/Users/I764690/Code_personal/LegoVision", "mobile_sam.pt")).to(device)

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

print("Muestras:")
for idx, frame in enumerate(meta_data["frames"][:10]):
    piece = frame["visible_pieces"][0]
    ref_gt = piece["ref"]
    c_name_gt = piece["color_name"]
    
    img_cen_path = os.path.join(project_root, "data", "simulation_100_all", frame["file_name"])
    img_pil = Image.open(img_cen_path)
    w_c, h_c = img_pil.size
    bbox_cen = piece["bbox_cenital_norm"]
    px1 = max(0, int(bbox_cen[0] * w_c)); py1 = max(0, int(bbox_cen[1] * h_c))
    px2 = min(w_c, int(bbox_cen[2] * w_c)); py2 = min(h_c, int(bbox_cen[3] * h_c))
    
    mask = get_sam_mask(img_pil, [px1, py1, px2, py2], (h_c, w_c))
    pixels_rgb = np.array(img_pil.convert("RGB"))[mask > 0]
    if len(pixels_rgb) > 0:
        median_rgb = np.median(pixels_rgb, axis=0)
        print(f"Sample {idx+1:02d} | GT Color: {c_name_gt:20s} | Median RGB: {median_rgb.tolist()}")
    else:
        print(f"Sample {idx+1:02d} | GT Color: {c_name_gt:20s} | Empty mask")
