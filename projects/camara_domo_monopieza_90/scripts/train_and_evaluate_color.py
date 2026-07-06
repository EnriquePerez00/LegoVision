import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO, SAM

# site-packages site path
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_75078"
legovic_root = "/Users/I764690/Code_personal/LegoVision"

# 1. Definición de la Red Neuronal Ligera para Clasificación de Color
class LegoColorCNN(nn.Module):
    def __init__(self, num_classes):
        super(LegoColorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LegoColorDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, mask_binary, color_name = self.samples[idx]
        
        # Cargar y recortar la pieza enmascarada
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Enmascarar con fondo negro
        img_np = np.array(img)
        img_np[mask_binary == 0] = [0, 0, 0]
        
        # Crop tight usando la máscara
        ys, xs = np.where(mask_binary > 0)
        if len(ys) > 0:
            x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        else:
            x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop_np = img_np[y1:y2, x1:x2]
        if crop_np.size == 0:
            crop_np = np.zeros((64, 64, 3), dtype=np.uint8)
            
        # Redimensionar a 64x64
        crop_resized = cv2.resize(crop_np, (64, 64))
        
        # Convertir a tensor PyTorch [C, H, W] norm a [0, 1]
        tensor = torch.tensor(crop_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Normalizar standard
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        label_idx = self.class_to_idx[color_name]
        return tensor, label_idx

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

def estimate_color_robust(img_path, mask, bbox=None):
    """Paso 1: Extracción estadística robusta con erosión de bordes y filtro de especularidad HSV."""
    if not os.path.exists(img_path):
        return [128.0, 128.0, 128.0]
        
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    w, h = img.size
    
    if mask is None:
        if bbox is not None:
            # Crear máscara simple a partir de bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255
        else:
            return [128.0, 128.0, 128.0]

    # 1. Aplicar erosión morfológica en la máscara para eliminar contaminación de bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Si erosionar borró toda la máscara, revertir a la máscara original
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask
    mask_bool = (mask_to_use > 0)
    
    if not np.any(mask_bool):
        return [128.0, 128.0, 128.0]
        
    pixels_rgb = img_arr[mask_bool]
    
    # Conversión a HSV para filtrar especularidades
    hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv_img[mask_bool]
    
    # 2. Filtrar especularidades (brillos intensos / saturación ultra-baja)
    non_specular_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    
    if np.any(non_specular_mask):
        pixels_rgb_filtered = pixels_rgb[non_specular_mask]
    else:
        pixels_rgb_filtered = pixels_rgb
        
    mean_rgb = pixels_rgb_filtered.mean(axis=0)
    return list(mean_rgb)

def rgb_to_lab(rgb_val):
    r, g, b = rgb_val[0] / 255.0, rgb_val[1] / 255.0, rgb_val[2] / 255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.array([l_val, a_val, b_val])

def find_closest_color_code(rgb_est, catalog_colors):
    if not catalog_colors:
        return "0", "Various", "#808080"
    lab_est = rgb_to_lab(rgb_est)
    best_dist = float("inf")
    best_color = catalog_colors[0]
    for c in catalog_colors:
        lab_ref = rgb_to_lab(c["rgb"])
        dL = lab_est[0] - lab_ref[0]
        da = lab_est[1] - lab_ref[1]
        db = lab_est[2] - lab_ref[2]
        dist = np.sqrt(0.2 * (dL ** 2) + (da ** 2) + (db ** 2))
        if dist < best_dist:
            best_dist = dist
            best_color = c
    return best_color["color_code"], best_color["color_name"], best_color["color_hex"]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo de hardware para entrenamiento y evaluación: {device}")

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

    print("Inicializando modelos YOLO y SAM...")
    model_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    model_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)
    sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

    # Extraer piezas por frame
    frames_list = meta_data.get("frames", [])
    
    # Recolectar muestras para el dataset del clasificador neuronal de color
    # Queremos tener muestras enmascaradas con MobileSAM
    print("Preparando muestras de entrenamiento sintéticas...")
    samples_cen = []
    samples_lat = []
    
    # Extraer colores únicos del dataset
    unique_colors_dataset = set()
    
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
                # Mapear con paleta
                for c in catalog_colors:
                    if c["color_code"] == str(p["color_code"]):
                        color_name = c["color_name"]
                        break
            
            if color_name == "Unknown":
                continue
                
            unique_colors_dataset.add(color_name)
            
            # Generar máscaras SAM basadas en Ground Truth bbox
            bbox_cen = p["bbox_cenital_norm"]
            bbox_lat = p["bbox_frontal_norm"]
            
            # Para agilizar, guardaremos la info básica de la muestra
            # Generaremos la máscara SAM al vuelo al cargar el dataset de PyTorch
            samples_cen.append((path_cen, bbox_cen, color_name))
            samples_lat.append((path_lat, bbox_lat, color_name))

    color_list = sorted(list(unique_colors_dataset))
    class_to_idx = {c: idx for idx, c in enumerate(color_list)}
    idx_to_class = {idx: c for c, idx in class_to_idx.items()}
    print(f"Colores únicos detectados en data100 ({len(color_list)}): {color_list}")

    # Generar máscaras SAM de antemano para las muestras para ahorrar tiempo en entrenamiento
    print("Generando máscaras SAM para dataset de color...")
    processed_samples_cen = []
    processed_samples_lat = []
    
    # Procesar cenital
    for path, bbox, c_name in samples_cen[:120]: # Usar subsets representativos para entrenar rápido en 15s
        img = Image.open(path).convert("RGB")
        w, h = img.size
        px1, py1 = int(bbox[0]*w), int(bbox[1]*h)
        px2, py2 = int(bbox[2]*w), int(bbox[3]*h)
        cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        sam_res = sam_model(np.array(img), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
        if sam_res and sam_res[0].masks is not None:
            mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            processed_samples_cen.append((path, bbox, mask, c_name))
            
    # Procesar lateral
    for path, bbox, c_name in samples_lat[:120]:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        px1, py1 = int(bbox[0]*w), int(bbox[1]*h)
        px2, py2 = int(bbox[2]*w), int(bbox[3]*h)
        cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        sam_res = sam_model(np.array(img), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
        if sam_res and sam_res[0].masks is not None:
            mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
            processed_samples_lat.append((path, bbox, mask, c_name))

    print(f"Muestras cenitales segmentadas con éxito: {len(processed_samples_cen)}")
    print(f"Muestras laterales segmentadas con éxito: {len(processed_samples_lat)}")

    # Crear datasets de PyTorch
    ds_cen = LegoColorDataset(processed_samples_cen, class_to_idx)
    ds_lat = LegoColorDataset(processed_samples_lat, class_to_idx)
    
    loader_cen = DataLoader(ds_cen, batch_size=8, shuffle=True)
    loader_lat = DataLoader(ds_lat, batch_size=8, shuffle=True)

    # Entrenar Clasificador Cenital de Color
    print("\n--- Entrenando LegoColorCNN para Vista Cenital (Paso 2) ---")
    model_color_cen = LegoColorCNN(len(color_list)).to(device)
    optimizer_cen = optim.Adam(model_color_cen.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model_color_cen.train()
    for epoch in range(12):
        running_loss = 0.0
        for inputs, labels in loader_cen:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_cen.zero_grad()
            outputs = model_color_cen(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cen.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/12 | Loss: {running_loss/len(loader_cen):.4f}")

    # Entrenar Clasificador Lateral de Color
    print("\n--- Entrenando LegoColorCNN para Vista Lateral (Paso 2) ---")
    model_color_lat = LegoColorCNN(len(color_list)).to(device)
    optimizer_lat = optim.Adam(model_color_lat.parameters(), lr=0.001)
    
    model_color_lat.train()
    for epoch in range(12):
        running_loss = 0.0
        for inputs, labels in loader_lat:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_lat.zero_grad()
            outputs = model_color_lat(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_lat.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/12 | Loss: {running_loss/len(loader_lat):.4f}")

    # Guardar los pesos de los modelos y las clases
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    torch.save(model_color_cen.state_dict(), os.path.join(project_root, "models", "color_model_cen.pt"))
    torch.save(model_color_lat.state_dict(), os.path.join(project_root, "models", "color_model_lat.pt"))
    with open(os.path.join(project_root, "models", "color_classes.txt"), "w") as f:
        for c in color_list:
            f.write(f"{c}\n")
    print(f"Modelos de color guardados en {os.path.join(project_root, 'models')}")

    # ----------------- NUEVA INFERENCIA DE EVALUACIÓN DE COLOR -----------------
    print("\n" + "="*50)
    print("INICIANDO EVALUACIÓN COMPARATIVA EN DATA100")
    print("="*50)

    # Mapeo de Ground Truth a tracks reales
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

    color_cen_baseline_hits = 0
    color_cen_step1_hits = 0
    color_cen_step2_hits = 0
    
    color_lat_baseline_hits = 0
    color_lat_step1_hits = 0
    color_lat_step2_hits = 0
    
    total_samples = len(eval_data["results"])

    model_color_cen.eval()
    model_color_lat.eval()

    print("Evaluando muestras...")
    for idx, r in enumerate(eval_data["results"]):
        ref_gt = r["ref_gt"]
        color_code_gt = r["color_code_gt"]
        cenital_file_eval = r["cenital_file"]
        
        # Mapear color gt
        color_name_gt = "Unknown"
        for c in catalog_colors:
            if c["color_code"] == str(color_code_gt):
                color_name_gt = c["color_name"]
                break

        # Buscar track
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
            best_obs = matched_track[0] # primera obs del track
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

        # Baseline predictions (desde el json original)
        name_cen_baseline = r.get("color_name_cen", "Unknown")
        if name_cen_baseline.strip().lower() == color_name_gt.strip().lower():
            color_cen_baseline_hits += 1

        # Estimación lateral baseline (bbox mean)
        w_img, h_img = 1024, 1024
        if os.path.exists(path_lat_img):
            img_lat = Image.open(path_lat_img).convert("RGB")
            w_img, h_img = img_lat.size
            
        # Correr YOLO para inferencia
        res_cen = model_cen(path_cen_img, verbose=False, conf=0.15, imgsz=1024)
        res_lat = model_lat(path_lat_img, verbose=False, conf=0.15, imgsz=1024)
        
        # 1. CENITAL COLOR
        mask_cen = None
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
                sam_res = sam_model(np.array(Image.open(path_cen_img).convert("RGB")), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
                if sam_res and sam_res[0].masks is not None:
                    mask_cen = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
        
        # Paso 1 Cenital
        rgb_cen_robust = estimate_color_robust(path_cen_img, mask_cen, bbox_cen_gt)
        _, name_cen_step1, _ = find_closest_color_code(rgb_cen_robust, catalog_colors)
        if name_cen_step1.strip().lower() == color_name_gt.strip().lower():
            color_cen_step1_hits += 1
            
        # Paso 2 Cenital
        if mask_cen is not None:
            # Preparar entrada a la red
            img_c = Image.open(path_cen_img).convert("RGB")
            img_np = np.array(img_c)
            img_np[mask_cen == 0] = [0, 0, 0]
            ys, xs = np.where(mask_cen > 0)
            if len(ys) > 0:
                cx1, cy1, cx2, cy2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[cy1:cy2, cx1:cx2]
            else:
                crop_np = img_np
            crop_resized = cv2.resize(crop_np, (64, 64))
            tensor = torch.tensor(crop_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            tensor = (tensor - mean) / std
            with torch.no_grad():
                pred_out = model_color_cen(tensor)
                pred_class_idx = torch.argmax(pred_out, dim=1).item()
                name_cen_step2 = idx_to_class[pred_class_idx]
        else:
            name_cen_step2 = name_cen_step1
            
        if name_cen_step2.strip().lower() == color_name_gt.strip().lower():
            color_cen_step2_hits += 1

        # 2. LATERAL COLOR
        mask_lat = None
        bbox_lat_inf = None
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
                bbox_lat_inf = best_box
                px1, py1 = int(best_box[0]*w_img), int(best_box[1]*h_img)
                px2, py2 = int(best_box[2]*w_img), int(best_box[3]*h_img)
                cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
                sam_res = sam_model(np.array(Image.open(path_lat_img).convert("RGB")), bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
                if sam_res and sam_res[0].masks is not None:
                    mask_lat = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)

        # Baseline Lateral (crop bbox mean sin SAM)
        if bbox_lat_gt:
            x1, y1, x2, y2 = int(bbox_lat_gt[0]*w_img), int(bbox_lat_gt[1]*h_img), int(bbox_lat_gt[2]*w_img), int(bbox_lat_gt[3]*h_img)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            cropped = Image.open(path_lat_img).convert("RGB").crop((x1, y1, x2, y2))
            rgb_lat_baseline = list(np.array(cropped).mean(axis=(0,1)))
            _, name_lat_baseline, _ = find_closest_color_code(rgb_lat_baseline, catalog_colors)
            if name_lat_baseline.strip().lower() == color_name_gt.strip().lower():
                color_lat_baseline_hits += 1

        # Paso 1 Lateral
        rgb_lat_robust = estimate_color_robust(path_lat_img, mask_lat, bbox_lat_gt)
        _, name_lat_step1, _ = find_closest_color_code(rgb_lat_robust, catalog_colors)
        if name_lat_step1.strip().lower() == color_name_gt.strip().lower():
            color_lat_step1_hits += 1

        # Paso 2 Lateral
        if mask_lat is not None:
            img_l = Image.open(path_lat_img).convert("RGB")
            img_np = np.array(img_l)
            img_np[mask_lat == 0] = [0, 0, 0]
            ys, xs = np.where(mask_lat > 0)
            if len(ys) > 0:
                lx1, ly1, lx2, ly2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
                crop_np = img_np[ly1:ly2, lx1:lx2]
            else:
                crop_np = img_np
            crop_resized = cv2.resize(crop_np, (64, 64))
            tensor = torch.tensor(crop_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            tensor = (tensor - mean) / std
            with torch.no_grad():
                pred_out = model_color_lat(tensor)
                pred_class_idx = torch.argmax(pred_out, dim=1).item()
                name_lat_step2 = idx_to_class[pred_class_idx]
        else:
            name_lat_step2 = name_lat_step1

        if name_lat_step2.strip().lower() == color_name_gt.strip().lower():
            color_lat_step2_hits += 1

    # Imprimir Reporte de Resultados Finales Comparativos
    print("\n" + "="*50)
    print("RESULTADOS COMPARATIVOS DE PRECISIÓN DE COLOR")
    print("="*50)
    print(f"Total Muestras Evaluadas: {total_samples}")
    
    print("\n--- Vista Cenital (Domo superior) ---")
    print(f"Baseline (Mean original):      {color_cen_baseline_hits} / {total_samples} ({color_cen_baseline_hits/total_samples*100:.2f}%)")
    print(f"Paso 1 (SAM Robust + HSV):     {color_cen_step1_hits} / {total_samples} ({color_cen_step1_hits/total_samples*100:.2f}%)")
    print(f"Paso 2 (Neural LegoColorCNN):  {color_cen_step2_hits} / {total_samples} ({color_cen_step2_hits/total_samples*100:.2f}%)")

    print("\n--- Vista Lateral (Frontal) ---")
    print(f"Baseline (Mean original):      {color_lat_baseline_hits} / {total_samples} ({color_lat_baseline_hits/total_samples*100:.2f}%)")
    print(f"Paso 1 (SAM Robust + HSV):     {color_lat_step1_hits} / {total_samples} ({color_lat_step1_hits/total_samples*100:.2f}%)")
    print(f"Paso 2 (Neural LegoColorCNN):  {color_lat_step2_hits} / {total_samples} ({color_lat_step2_hits/total_samples*100:.2f}%)")

    # Guardar en archivo para registrar
    new_metrics = {
        "cenital": {
            "baseline": color_cen_baseline_hits/total_samples*100,
            "step1_sam_robust": color_cen_step1_hits/total_samples*100,
            "step2_neural_cnn": color_cen_step2_hits/total_samples*100
        },
        "lateral": {
            "baseline": color_lat_baseline_hits/total_samples*100,
            "step1_sam_robust": color_lat_step1_hits/total_samples*100,
            "step2_neural_cnn": color_lat_step2_hits/total_samples*100
        }
    }
    with open(os.path.join(project_root, "data", "reports", "new_color_metrics.json"), "w") as f:
        json.dump(new_metrics, f, indent=2)
    print(f"\nReporte final comparativo guardado en {os.path.join(project_root, 'data', 'reports', 'new_color_metrics.json')}")

if __name__ == "__main__":
    main()
