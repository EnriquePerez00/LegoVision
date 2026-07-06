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
from efficientnet_classifier_all import LegoEfficientNetClassifierAll

# Cargar metadatos del dataset simulación 100
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

device = "mps" if torch.backends.mps.is_available() else "cpu"
sam_model = SAM(os.path.join(os.path.dirname(project_root), "mobile_sam.pt")).to(device)
hierarchical_clf = ColorClassifierAll(device=device)
efficientnet_clf = LegoEfficientNetClassifierAll()

# Mapear colores reales de la simulación
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

# Extraer y pre-procesar imágenes y características de todos los renders para una evaluación rápida en memoria
print("Pre-procesando renders de la simulación en memoria...")
renders = []
for f in meta_data.get("frames", []):
    for piece in f.get("visible_pieces", []):
        renders.append({
            "ref": piece["ref"],
            "color_code": piece["color_code"],
            "image_path": os.path.join(project_root, "data", "simulation_100_all", f["file_name"]),
            "bbox_norm": piece["bbox_cenital_norm"],
            "image_path_lat": os.path.join(project_root, "data", "simulation_100_all", f["file_name_frontal"]),
            "bbox_norm_lat": piece.get("bbox_frontal_norm")
        })

processed_samples = []
for idx, entry in enumerate(renders[:97]):
    ref_gt = entry["ref"]
    c_code_gt = str(entry["color_code"])
    c_name_gt = gt_color_map.get((ref_gt, c_code_gt))
    if not c_name_gt:
        continue
        
    img_path = entry["image_path"]
    img_path_lat = entry["image_path_lat"]
    if not os.path.exists(img_path) or not os.path.exists(img_path_lat):
        continue
        
    # Cenital:
    img_pil = Image.open(img_path)
    w_c, h_c = img_pil.size
    px1, py1, px2, py2 = int(entry["bbox_norm"][0]*w_c), int(entry["bbox_norm"][1]*h_c), int(entry["bbox_norm"][2]*w_c), int(entry["bbox_norm"][3]*h_c)
    if px2 <= px1 or py2 <= py1:
        continue
    mask_bin = get_sam_mask(img_pil, [px1, py1, px2, py2], (h_c, w_c))
    
    # Extraer crop cenital
    box_cen_px = [px1, py1, px2, py2]
    crop_cen_raw = img_pil.crop(box_cen_px)
    crop_cen_np = np.array(crop_cen_raw)
    if crop_cen_np.size == 0:
        continue
    mask_cen_crop = cv2.resize(mask_bin, (w_c, h_c))[box_cen_px[1]:box_cen_px[3], box_cen_px[0]:box_cen_px[2]]
    if mask_cen_crop.size == 0:
        continue
    mask_cen_crop = cv2.resize(mask_cen_crop, (crop_cen_np.shape[1], crop_cen_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    crop_cen = Image.fromarray(cv2.bitwise_and(crop_cen_np, crop_cen_np, mask=mask_cen_crop))
    
    # Extraer características robustas
    from run_evaluation_all import estimate_color_mlp_features
    feat_cen = estimate_color_mlp_features(img_pil, mask_bin, "cenital", ccm_params=None, is_simulation=True)
    
    # Lateral:
    img_pil_lat = Image.open(img_path_lat)
    w_l, h_l = img_pil_lat.size
    px1_l, py1_l, px2_l, py2_l = int(entry["bbox_norm_lat"][0]*w_l), int(entry["bbox_norm_lat"][1]*h_l), int(entry["bbox_norm_lat"][2]*w_l), int(entry["bbox_norm_lat"][3]*h_l)
    if px2_l <= px1_l or py2_l <= py1_l:
        continue
    mask_bin_lat = get_sam_mask(img_pil_lat, [px1_l, py1_l, px2_l, py2_l], (h_l, w_l))
    # y_cutoff = int(505 * h_l / 1024)
    # mask_bin_lat[y_cutoff:, :] = 0
    
    box_lat_px = [px1_l, py1_l, px2_l, py2_l]
    crop_lat_raw = img_pil_lat.crop(box_lat_px)
    crop_lat_np = np.array(crop_lat_raw)
    if crop_lat_np.size == 0:
        continue
    mask_lat_crop = cv2.resize(mask_bin_lat, (w_l, h_l))[box_lat_px[1]:box_lat_px[3], box_lat_px[0]:box_lat_px[2]]
    if mask_lat_crop.size == 0:
        continue
    mask_lat_crop = cv2.resize(mask_lat_crop, (crop_lat_np.shape[1], crop_lat_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    crop_lat = Image.fromarray(cv2.bitwise_and(crop_lat_np, crop_lat_np, mask=mask_lat_crop))
    
    feat_lat = estimate_color_mlp_features(img_pil_lat, mask_bin_lat, "lateral", ccm_params=None, is_simulation=True)
    
    # Extraer área cenital aproximada
    px_per_mm = float(w_c) / 196.363636
    area_cenital = float(np.sum(mask_bin)) / (px_per_mm ** 2)
    
    processed_samples.append({
        "ref_gt": ref_gt,
        "color_code_gt": c_code_gt,
        "color_name_gt": c_name_gt,
        "crop_cen": crop_cen,
        "mask_cen": mask_bin,
        "crop_lat": crop_lat,
        "mask_lat": mask_bin_lat,
        "area_cenital": area_cenital,
        "feat_cen": feat_cen,
        "feat_lat": feat_lat
    })
    
# Precalentar predicciones de geometría para evitar llamarla repetitivamente en la rejilla
print("Pre-calculando predicciones de geometría (Pass 1)...")
for sample in processed_samples:
    preds = efficientnet_clf.classify(
        crop_cen=sample["crop_cen"],
        mask_cen=sample["mask_cen"],
        crop_lat=sample["crop_lat"],
        mask_lat=sample["mask_lat"],
        area_cenital=sample["area_cenital"],
        detected_color=None
    )
    sample["preds_geometry"] = preds

print(f"Pre-procesamiento completado. {len(processed_samples)} muestras listas para búsqueda hiperparamétrica.")

# Búsqueda hiperparamétrica
candidate_counts = [1, 2, 5, 10, 15, 20, 30]
alpha_weights = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

best_acc = 0.0
best_params = None

results = []

for K in candidate_counts:
    for alpha in alpha_weights:
        correct_count = 0
        total_count = 0
        
        for sample in processed_samples:
            preds = sample["preds_geometry"]
            
            # Obtener el Top-K de candidatos
            candidates_ref = []
            if preds:
                for cand in preds[:K]:
                    ref_cand = cand["part_ref"]
                    if ref_cand not in candidates_ref:
                        candidates_ref.append(ref_cand)
            
            # Obtener unión de colores válidos
            allowed_color_names = set()
            for ref_cand in candidates_ref:
                allowed_color_names.update(efficientnet_clf.part_to_colors.get(ref_cand, set()))
                
            # 2. Ejecutar predict_gated_probs_cielab
            p_cen_gated = None
            if sample["feat_cen"] is not None:
                p_cen_gated = hierarchical_clf.predict_gated_probs_cielab(sample["feat_cen"], allowed_color_names, "cenital", is_simulation=True)
                
            p_lat_gated = None
            if sample["feat_lat"] is not None:
                p_lat_gated = hierarchical_clf.predict_gated_probs_cielab(sample["feat_lat"], allowed_color_names, "lateral", is_simulation=True)
                
            # Combinación ponderada exponencial
            if p_cen_gated is not None and p_lat_gated is not None:
                p_combined = (p_cen_gated ** alpha) * (p_lat_gated ** (1.0 - alpha))
                if np.sum(p_combined) == 0:
                    p_combined = p_cen_gated
            elif p_cen_gated is not None:
                p_combined = p_cen_gated
            elif p_lat_gated is not None:
                p_combined = p_lat_gated
            else:
                p_combined = None
                
            if p_combined is not None and len(p_combined) > 0:
                pred_color = hierarchical_clf.classes[np.argmax(p_combined)]
                if pred_color.strip().lower() == sample["color_name_gt"].strip().lower():
                    correct_count += 1
            total_count += 1
            
        acc = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
        results.append((K, alpha, acc))
        print(f"K={K:2d} | alpha={alpha:.2f} | Exactitud Color Fused: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_params = (K, alpha)

print("\n" + "="*70)
print("BÚSQUEDA HIPERPARAMÉTRICA FINALIZADA")
print("="*70)
print(f"Mejor Exactitud Color Fused: {best_acc:.2f}% con K={best_params[0]} y alpha={best_params[1]}")
print("="*70)
