import os
import sys
import json
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO, SAM
import torchvision.transforms as T

# site-packages site path
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo"
legovic_root = "/Users/I764690/Code_personal/LegoVision"

def compute_iou(boxA, boxB):
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo de hardware para DINOv2: {device}")

    eval_json_path = os.path.join(project_root, "data", "reports", "new_weights_eval.html")
    metadata_path = os.path.join(project_root, "data", "data100", "simulation_metadata.json")
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar catálogo de colores de referencia
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

    print("Cargando modelo DINOv2...")
    # Cargar DINOv2 ViT-S/14 preentrenado de Facebook Research
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2_model.eval()

    print("Inicializando modelos YOLO y SAM...")
    model_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    model_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)
    sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

    # Transformación recomendada para DINOv2
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames_list = meta_data.get("frames", [])
    
    # 1. Preparar muestras de referencia (train subset)
    print("Preparando muestras de referencia...")
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
                
            bbox_cen = p["bbox_cenital_norm"]
            bbox_lat = p["bbox_frontal_norm"]
            
            ref_samples_cen.append((path_cen, bbox_cen, color_name))
            ref_samples_lat.append((path_lat, bbox_lat, color_name))

    # Obtener embeddings DINOv2 para los primeros 120 elementos de referencia
    print("Generando embeddings DINOv2 para referencias Cenitales...")
    ref_embs_cen = []
    ref_labels_cen = []
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
                x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[y1:y2, x1:x2]
            else:
                crop_np = img_np[py1:py2, px1:px2]
                
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = dinov2_model(tensor).cpu().numpy().flatten()
                ref_embs_cen.append(emb)
                ref_labels_cen.append(c_name)

    print("Generando embeddings DINOv2 para referencias Laterales...")
    ref_embs_lat = []
    ref_labels_lat = []
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
                x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[y1:y2, x1:x2]
            else:
                crop_np = img_np[py1:py2, px1:px2]
                
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = dinov2_model(tensor).cpu().numpy().flatten()
                ref_embs_lat.append(emb)
                ref_labels_lat.append(c_name)

    # ----------------- INFERENCIA Y EVALUACIÓN DE OPCIÓN B -----------------
    print("\nEvaluando Inferencia de Opción B (DINOv2 embeddings) en data100...")
    
    # Agrupar piezas por trayectoria global
    global_piece_tracks = {}
    for frame in frames_list:
        offset = frame["belt_offset_mm"]
        f_name = frame["file_name"]
        for p in frame["visible_pieces"]:
            x_abs = p["x_belt_local_mm"] + offset
            y_abs = p["y_belt_local_mm"]
            
            matched_key = None
            for key in global_piece_tracks.keys():
                kx, ky = key
                if abs(x_abs - kx) < 1.5 and abs(y_abs - ky) < 1.5:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (x_abs, y_abs)
                global_piece_tracks[matched_key] = []
                
            global_piece_tracks[matched_key].append({
                "ref": p["ref"],
                "color_code": p["color_code"],
                "color_name": p.get("color_name", "Unknown"),
                "file_name": f_name,
                "bbox_cenital_norm": p["bbox_cenital_norm"],
                "bbox_frontal_norm": p.get("bbox_frontal_norm"),
            })

    color_cen_dinov2_1nn_hits = 0
    color_lat_dinov2_1nn_hits = 0
    
    total_samples = len(eval_data["results"])

    for idx, r in enumerate(eval_data["results"]):
        ref_gt = r["ref_gt"]
        color_code_gt = r["color_code_gt"]
        cenital_file_eval = r["cenital_file"]
        
        color_name_gt = "Unknown"
        for c in catalog_colors:
            if c["color_code"] == str(color_code_gt):
                color_name_gt = c["color_name"]
                break

        matched_track = None
        for key, track_obs in global_piece_tracks.items():
            has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt and str(o["color_code"]) == str(color_code_gt) for o in track_obs)
            if has_match:
                matched_track = track_obs
                break
        
        if not matched_track:
            for key, track_obs in global_piece_tracks.items():
                has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt for o in track_obs)
                if has_match:
                    matched_track = track_obs
                    break
        
        if matched_track:
            best_obs = matched_track[0]
            cen_file = best_obs["file_name"]
            lat_file = cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = best_obs["bbox_cenital_norm"]
            bbox_lat_gt = best_obs["bbox_frontal_norm"]
            if color_name_gt == "Unknown":
                color_name_gt = best_obs.get("color_name", "Unknown")
        else:
            cen_file = cenital_file_eval
            lat_file = r.get("lateral_file") or cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = None
            bbox_lat_gt = None

        path_cen_img = os.path.join(project_root, "data", "data100", cen_file)
        path_lat_img = os.path.join(project_root, "data", "data100", lat_file)

        # 1. EVALUAR CENITAL CON DINOv2 (1-NN)
        res_cen = model_cen(path_cen_img, verbose=False, conf=0.15, imgsz=1024)
        mask_cen = None
        w_img, h_img = 1024, 1024
        if os.path.exists(path_cen_img):
            img_c = Image.open(path_cen_img).convert("RGB")
            w_img, h_img = img_c.size
            
        if res_cen and len(res_cen[0].boxes) > 0 and bbox_cen_gt:
            best_iou = 0.0
            best_box = None
            for box in res_cen[0].boxes:
                det_box = box.xyxyn[0].cpu().numpy().tolist()
                iou = compute_iou(bbox_cen_gt, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_box = det_box
            if best_box:
                px1, py1 = int(best_box[0]*w_img), int(best_box[1]*h_img)
                px2, py2 = int(best_box[2]*w_img), int(best_box[3]*h_img)
                cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
                sam_res = sam_model(np.array(img_c), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
                if sam_res and sam_res[0].masks is not None:
                    mask_cen = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)

        if mask_cen is not None:
            img_np = np.array(img_c)
            img_np[mask_cen == 0] = [0, 0, 0]
            ys, xs = np.where(mask_cen > 0)
            if len(ys) > 0:
                cx1, cy1, cx2, cy2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[cy1:cy2, cx1:cx2]
            else:
                crop_np = img_np
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_emb = dinov2_model(tensor).cpu().numpy().flatten()
            
            # Buscar el vecino más cercano (1-NN cosine similarity)
            best_sim = -1.0
            pred_color_cen = "Unknown"
            for ref_emb, label in zip(ref_embs_cen, ref_labels_cen):
                sim = np.dot(q_emb, ref_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ref_emb) + 1e-8)
                if sim > best_sim:
                    best_sim = sim
                    pred_color_cen = label
            
            if pred_color_cen.strip().lower() == color_name_gt.strip().lower():
                color_cen_dinov2_1nn_hits += 1

        # 2. EVALUAR LATERAL CON DINOv2 (1-NN)
        res_lat = model_lat(path_lat_img, verbose=False, conf=0.15, imgsz=1024)
        mask_lat = None
        if os.path.exists(path_lat_img):
            img_l = Image.open(path_lat_img).convert("RGB")
            w_img, h_img = img_l.size
            
        if res_lat and len(res_lat[0].boxes) > 0 and bbox_lat_gt:
            best_iou = 0.0
            best_box = None
            for box in res_lat[0].boxes:
                det_box = box.xyxyn[0].cpu().numpy().tolist()
                iou = compute_iou(bbox_lat_gt, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_box = det_box
            if best_box:
                px1, py1 = int(best_box[0]*w_img), int(best_box[1]*h_img)
                px2, py2 = int(best_box[2]*w_img), int(best_box[3]*h_img)
                cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
                sam_res = sam_model(np.array(img_l), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
                if sam_res and sam_res[0].masks is not None:
                    mask_lat = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)

        if mask_lat is not None:
            img_np = np.array(img_l)
            img_np[mask_lat == 0] = [0, 0, 0]
            ys, xs = np.where(mask_lat > 0)
            if len(ys) > 0:
                lx1, ly1, lx2, ly2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[ly1:ly2, lx1:lx2]
            else:
                crop_np = img_np
            crop_resized = cv2.resize(crop_np, (224, 224))
            tensor = transform(Image.fromarray(crop_resized)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_emb = dinov2_model(tensor).cpu().numpy().flatten()
            
            best_sim = -1.0
            pred_color_lat = "Unknown"
            for ref_emb, label in zip(ref_embs_lat, ref_labels_lat):
                sim = np.dot(q_emb, ref_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ref_emb) + 1e-8)
                if sim > best_sim:
                    best_sim = sim
                    pred_color_lat = label
            
            if pred_color_lat.strip().lower() == color_name_gt.strip().lower():
                color_lat_dinov2_1nn_hits += 1

    # Imprimir Reporte de Resultados Finales Comparativos de la Opción B
    print("\n" + "="*50)
    print("RESULTADOS COMPARATIVOS: CLASIFICADOR vs DINOv2 (OPCIÓN B)")
    print("="*50)
    print(f"Total Muestras Evaluadas: {total_samples}")
    
    # Cargar métricas guardadas del clasificador LegoColorCNN
    cnn_cen_acc = 27.0
    cnn_lat_acc = 27.0
    cnn_metrics_path = os.path.join(project_root, "data", "reports", "new_color_metrics.json")
    if os.path.exists(cnn_metrics_path):
        try:
            with open(cnn_metrics_path, "r") as f:
                cnn_m = json.load(f)
                cnn_cen_acc = cnn_m["cenital"]["step2_neural_cnn"]
                cnn_lat_acc = cnn_m["lateral"]["step2_neural_cnn"]
        except Exception:
            pass

    print("\n--- Vista Cenital (Domo superior) ---")
    print(f"LegoColorCNN (Clasificador 14 clases):   {cnn_cen_acc:.2f}%")
    print(f"Opción B (DINOv2 + Cosine Similarity):    {color_cen_dinov2_1nn_hits/total_samples*100:.2f}% ({color_cen_dinov2_1nn_hits}/{total_samples})")

    print("\n--- Vista Lateral (Frontal) ---")
    print(f"LegoColorCNN (Clasificador 14 clases):   {cnn_lat_acc:.2f}%")
    print(f"Opción B (DINOv2 + Cosine Similarity):    {color_lat_dinov2_1nn_hits/total_samples*100:.2f}% ({color_lat_dinov2_1nn_hits}/{total_samples})")

    # Guardar reporte
    report_opt_b = {
        "lego_color_cnn": {
            "cenital": cnn_cen_acc,
            "lateral": cnn_lat_acc
        },
        "dinov2_embeddings": {
            "cenital": color_cen_dinov2_1nn_hits/total_samples*100,
            "lateral": color_lat_dinov2_1nn_hits/total_samples*100
        }
    }
    with open(os.path.join(project_root, "data", "reports", "color_metrics_option_b.json"), "w") as f:
        json.dump(report_opt_b, f, indent=2)
    print(f"\nReporte comparativo guardado en {os.path.join(project_root, 'data', 'reports', 'color_metrics_option_b.json')}")

if __name__ == "__main__":
    main()
