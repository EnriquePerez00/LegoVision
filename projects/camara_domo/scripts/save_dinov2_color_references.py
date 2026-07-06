import os
import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import SAM
import torchvision.transforms as T

# site-packages site path
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo"
legovic_root = "/Users/I764690/Code_personal/LegoVision"

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Indexando referencias de color DINOv2 en: {device}")

    metadata_path = os.path.join(project_root, "data", "data100", "simulation_metadata.json")
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    catalog_colors = []
    if os.path.exists(palette_path):
        with open(palette_path, "r", encoding="utf-8") as f:
            palette = json.load(f)
            for item in palette:
                catalog_colors.append({
                    "color_code": str(item.get("color_code", "")),
                    "color_name": item.get("color_name", "Unknown"),
                    "color_hex": item.get("color_hex", "#808080"),
                    "rgb": np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float)
                })

    print("Cargando DINOv2 y SAM...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2_model.eval()
    sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames_list = meta_data.get("frames", [])
    ref_samples_cen = []
    ref_samples_lat = []
    
    for frame in frames_list:
        f_name = frame["file_name"]
        f_name_lat = frame["file_name_frontal"]
        path_cen = os.path.join(project_root, "data", "data100", f_name)
        path_lat = os.path.join(project_root, "data", "data100", f_name_lat)
        
        if not os.path.exists(path_cen) or not os.path.exists(path_lat):
            continue
            
        for p in frame["visible_pieces"]:
            color_name = p.get("color_name", "Unknown")
            if color_name == "Unknown":
                for c in catalog_colors:
                    if c["color_code"] == str(p["color_code"]):
                        color_name = c["color_name"]
                        break
            if color_name == "Unknown":
                continue
                
            ref_samples_cen.append((path_cen, p["bbox_cenital_norm"], color_name))
            ref_samples_lat.append((path_lat, p["bbox_frontal_norm"], color_name))

    print("Extrayendo embeddings Cenitales...")
    ref_embs_cen = []
    ref_names_cen = []
    for path, bbox, c_name in ref_samples_cen[:120]:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        px1, py1 = int(bbox[0]*w), int(bbox[1]*h)
        px2, py2 = int(bbox[2]*w), int(bbox[3]*h)
        cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        
        sam_res = sam_model(np.array(img), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
        if sam_res and sam_res[0].masks is not None:
            mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            img_np = np.array(img)
            img_np[mask == 0] = [0, 0, 0]
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                crop_np = img_np[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
            else:
                crop_np = img_np[py1:py2, px1:px2]
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = dinov2_model(tensor).cpu().numpy().flatten()
                ref_embs_cen.append(emb)
                ref_names_cen.append(c_name)

    print("Extrayendo embeddings Laterales...")
    ref_embs_lat = []
    ref_names_lat = []
    for path, bbox, c_name in ref_samples_lat[:120]:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        px1, py1 = int(bbox[0]*w), int(bbox[1]*h)
        px2, py2 = int(bbox[2]*w), int(bbox[3]*h)
        cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        
        sam_res = sam_model(np.array(img), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
        if sam_res and sam_res[0].masks is not None:
            mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            img_np = np.array(img)
            img_np[mask == 0] = [0, 0, 0]
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                crop_np = img_np[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
            else:
                crop_np = img_np[py1:py2, px1:px2]
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = dinov2_model(tensor).cpu().numpy().flatten()
                ref_embs_lat.append(emb)
                ref_names_lat.append(c_name)

    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    out_path = os.path.join(project_root, "models", "color_ref_embeddings.npz")
    np.savez_compressed(
        out_path,
        embs_cen=np.array(ref_embs_cen),
        names_cen=np.array(ref_names_cen),
        embs_lat=np.array(ref_embs_lat),
        names_lat=np.array(ref_names_lat)
    )
    print(f"Embeddings de referencia guardados con éxito en {out_path}")

if __name__ == "__main__":
    main()
