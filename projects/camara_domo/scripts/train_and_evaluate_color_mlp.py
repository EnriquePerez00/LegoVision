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
from ultralytics import SAM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# site-packages site path
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo"
legovic_root = "/Users/I764690/Code_personal/LegoVision"

# Helper color conversion and matching (from original script)
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

# Exact CIELAB conversion for 2D matrices
def rgb_matrix_to_lab(rgb_matrix):
    """Converts a (N, 3) matrix of RGB pixels to CIELAB (N, 3)"""
    arr = rgb_matrix.astype(np.float32) / 255.0
    mask_gt = arr > 0.04045
    arr[mask_gt] = ((arr[mask_gt] + 0.055) / 1.055) ** 2.4
    arr[~mask_gt] = arr[~mask_gt] / 12.92

    x = arr[:, 0] * 0.4124 + arr[:, 1] * 0.3576 + arr[:, 2] * 0.1805
    y = arr[:, 0] * 0.2126 + arr[:, 1] * 0.7152 + arr[:, 2] * 0.0722
    z = arr[:, 0] * 0.0193 + arr[:, 1] * 0.1192 + arr[:, 2] * 0.9505

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    # fx, fy, fz
    fx = np.zeros_like(x)
    fy = np.zeros_like(y)
    fz = np.zeros_like(z)

    mask_x = x > 0.008856
    fx[mask_x] = x[mask_x] ** (1/3)
    fx[~mask_x] = (7.787 * x[~mask_x]) + (16 / 116)

    mask_y = y > 0.008856
    fy[mask_y] = y[mask_y] ** (1/3)
    fy[~mask_y] = (7.787 * y[~mask_y]) + (16 / 116)

    mask_z = z > 0.008856
    fz[mask_z] = z[mask_z] ** (1/3)
    fz[~mask_z] = (7.787 * z[~mask_z]) + (16 / 116)

    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.column_stack([l_val, a_val, b_val])

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

# Extract features from a SAM-masked image crop using erosion and HSV spec filtering
def extract_stats_features(img_path, bbox, sam_model):
    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    img_arr = np.array(img)

    px1, py1 = int(bbox[0]*w), int(bbox[1]*h)
    px2, py2 = int(bbox[2]*w), int(bbox[3]*h)
    cx, cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5

    sam_res = sam_model(img_arr, bboxes=[[px1, py1, px2, py2]], points=[[[cx, cy]]], labels=[[1]], verbose=False)
    if not sam_res or sam_res[0].masks is None or len(sam_res[0].masks.data) == 0:
        return None
    
    mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
    
    # 1. Morphological erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask
    mask_bool = (mask_to_use > 0)

    if not np.any(mask_bool):
        return None

    pixels_rgb = img_arr[mask_bool]
    hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv_img[mask_bool]

    # 2. Specular highlight filtering (same as Paso 1)
    non_specular_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    if np.any(non_specular_mask):
        pixels_rgb_filt = pixels_rgb[non_specular_mask]
        pixels_hsv_filt = pixels_hsv[non_specular_mask]
    else:
        pixels_rgb_filt = pixels_rgb
        pixels_hsv_filt = pixels_hsv

    # Convert RGB to Lab for all pixels
    pixels_lab = rgb_matrix_to_lab(pixels_rgb_filt)

    # Discard pixels outside 25-75th percentile of L channel
    l_vals = pixels_lab[:, 0]
    p25 = np.percentile(l_vals, 25)
    p75 = np.percentile(l_vals, 75)
    valid_mask = (l_vals >= p25) & (l_vals <= p75)
    if np.any(valid_mask):
        pixels_lab = pixels_lab[valid_mask]
        pixels_rgb_filt = pixels_rgb_filt[valid_mask]
        pixels_hsv_filt = pixels_hsv_filt[valid_mask]

    # Compute statistics (Mean & Std Dev) for:
    # L, a, b (from Lab)
    # H, S, V (from HSV)
    mean_lab = pixels_lab.mean(axis=0)
    std_lab = pixels_lab.std(axis=0)
    
    mean_hsv = pixels_hsv_filt.mean(axis=0)
    std_hsv = pixels_hsv_filt.std(axis=0)

    # Output vector of 12 statistics features
    features = np.array([
        mean_lab[0], std_lab[0],
        mean_lab[1], std_lab[1],
        mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0],
        mean_hsv[1], std_hsv[1],
        mean_hsv[2], std_hsv[2]
    ], dtype=np.float32)

    return features, list(pixels_rgb_filt.mean(axis=0))

# Define Neural MLP Model
class ColorMLP(nn.Module):
    def __init__(self, input_dim=12, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class SimpleFeatureDataset(Dataset):
    def __init__(self, X, y, add_noise=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.add_noise = add_noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.add_noise:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        return x, self.y[idx]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo de hardware: {device}")

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

    print("Cargando MobileSAM...")
    sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)

    # 1. Recolectar datos y extraer características
    cache_file = os.path.join(project_root, "data", "mlp_features_cache.npz")
    
    if os.path.exists(cache_file):
        print("Cargando características estadísticas de color del caché...")
        cache_data = np.load(cache_file, allow_pickle=True)
        X_cen = cache_data["X_cen"]
        y_cen = list(cache_data["y_cen"])
        rgb_cen_list = list(cache_data["rgb_cen_list"])
        X_lat = cache_data["X_lat"]
        y_lat = list(cache_data["y_lat"])
        rgb_lat_list = list(cache_data["rgb_lat_list"])
    else:
        print("Extrayendo características estadísticas (Lab/HSV) de las muestras...")
        frames_list = meta_data.get("frames", [])
        X_cen, y_cen, rgb_cen_list = [], [], []
        X_lat, y_lat, rgb_lat_list = [], [], []
        
        for idx, frame in enumerate(frames_list[::8]):
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
                
                # Cenital
                res_cen = extract_stats_features(path_cen, p["bbox_cenital_norm"], sam_model)
                if res_cen is not None:
                    feat, mean_rgb = res_cen
                    X_cen.append(feat)
                    y_cen.append(color_name)
                    rgb_cen_list.append(mean_rgb)
                    
                # Lateral
                res_lat = extract_stats_features(path_lat, p["bbox_frontal_norm"], sam_model)
                if res_lat is not None:
                    feat, mean_rgb = res_lat
                    X_lat.append(feat)
                    y_lat.append(color_name)
                    rgb_lat_list.append(mean_rgb)

        X_cen = np.array(X_cen)
        X_lat = np.array(X_lat)
        
        # Guardar en caché para futuras ejecuciones instantáneas
        np.savez_compressed(
            cache_file,
            X_cen=X_cen,
            y_cen=np.array(y_cen),
            rgb_cen_list=np.array(rgb_cen_list),
            X_lat=X_lat,
            y_lat=np.array(y_lat),
            rgb_lat_list=np.array(rgb_lat_list)
        )
        print(f"Características guardadas en caché: {cache_file}")

    # Mapeo de clases
    unique_colors = sorted(list(set(y_cen + y_lat)))
    class_to_idx = {c: idx for idx, c in enumerate(unique_colors)}
    print(f"Clases a clasificar ({len(unique_colors)}): {class_to_idx}")

    y_cen_idx = np.array([class_to_idx[c] for c in y_cen])
    y_lat_idx = np.array([class_to_idx[c] for c in y_lat])

    print(f"Muestras extraídas Cenitales: {len(X_cen)}, Laterales: {len(X_lat)}")

    # 2. Validación cruzada (5-Fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Listas para almacenar métricas
    mlp_cen_accs, paso1_cen_accs = [], []
    mlp_lat_accs, paso1_lat_accs = [], []

    # Evaluar Vista Cenital
    print("\n--- Evaluando Vista Cenital ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cen, y_cen_idx)):
        X_train, X_val = X_cen[train_idx], X_cen[val_idx]
        y_train, y_val = y_cen_idx[train_idx], y_cen_idx[val_idx]
        
        # Escalar características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_dataset = SimpleFeatureDataset(X_train, y_train, add_noise=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

        model = ColorMLP(input_dim=12, num_classes=len(unique_colors)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Entrenamiento
        model.train()
        for epoch in range(120):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluación MLP
        model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            preds = torch.argmax(model(val_x_t), dim=1).cpu().numpy()
            mlp_acc = np.mean(preds == y_val) * 100.0
            mlp_cen_accs.append(mlp_acc)

        # Evaluación Paso 1
        paso1_preds = []
        for v_i in val_idx:
            rgb_est = rgb_cen_list[v_i]
            _, name_est, _ = find_closest_color_code(rgb_est, catalog_colors)
            paso1_preds.append(class_to_idx.get(name_est, -1))
        paso1_acc = np.mean(np.array(paso1_preds) == y_val) * 100.0
        paso1_cen_accs.append(paso1_acc)

        print(f"Fold {fold+1} Cenital: MLP={mlp_acc:.2f}% | Paso 1={paso1_acc:.2f}%")

    # Evaluar Vista Lateral
    print("\n--- Evaluando Vista Lateral ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lat, y_lat_idx)):
        X_train, X_val = X_lat[train_idx], X_lat[val_idx]
        y_train, y_val = y_lat_idx[train_idx], y_lat_idx[val_idx]
        
        # Escalar
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_dataset = SimpleFeatureDataset(X_train, y_train, add_noise=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

        model = ColorMLP(input_dim=12, num_classes=len(unique_colors)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Entrenamiento
        model.train()
        for epoch in range(120):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluación MLP
        model.eval()
        with torch.no_grad():
            val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            preds = torch.argmax(model(val_x_t), dim=1).cpu().numpy()
            mlp_acc = np.mean(preds == y_val) * 100.0
            mlp_lat_accs.append(mlp_acc)

        # Evaluación Paso 1
        paso1_preds = []
        for v_i in val_idx:
            rgb_est = rgb_lat_list[v_i]
            _, name_est, _ = find_closest_color_code(rgb_est, catalog_colors)
            paso1_preds.append(class_to_idx.get(name_est, -1))
        paso1_acc = np.mean(np.array(paso1_preds) == y_val) * 100.0
        paso1_lat_accs.append(paso1_acc)

        print(f"Fold {fold+1} Lateral: MLP={mlp_acc:.2f}% | Paso 1={paso1_acc:.2f}%")

    print("\n" + "="*50)
    print("PROMEDIO DE RESULTADOS DE VALIDACIÓN CRUZADA (5-FOLD)")
    print("="*50)
    print(f"Cenital MLP Accuracy:   {np.mean(mlp_cen_accs):.2f}% ± {np.std(mlp_cen_accs):.2f}%")
    print(f"Cenital Paso 1 Accuracy: {np.mean(paso1_cen_accs):.2f}% ± {np.std(paso1_cen_accs):.2f}%")
    print("-" * 50)
    print(f"Lateral MLP Accuracy:   {np.mean(mlp_lat_accs):.2f}% ± {np.std(mlp_lat_accs):.2f}%")
    print(f"Lateral Paso 1 Accuracy: {np.mean(paso1_lat_accs):.2f}% ± {np.std(paso1_lat_accs):.2f}%")
    print("="*50)

    # 3. Entrenar Modelo Final con 100% de los Datos Combinados (Cenital + Lateral)
    print("\nEntrenando modelo final unificado con el 100% de los datos...")
    X_all = np.concatenate([X_cen, X_lat], axis=0)
    y_all = np.concatenate([y_cen_idx, y_lat_idx], axis=0)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    train_dataset_all = SimpleFeatureDataset(X_all_scaled, y_all, add_noise=True)
    train_loader_all = DataLoader(train_dataset_all, batch_size=64, shuffle=True, drop_last=True)

    final_model = ColorMLP(input_dim=12, num_classes=len(unique_colors)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.01, weight_decay=1e-4)

    final_model.train()
    for epoch in range(150):
        for batch_x, batch_y in train_loader_all:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = final_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Guardar pesos del modelo
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "color_mlp_model.pt")
    torch.save(final_model.state_dict(), model_path)
    print(f"Modelo final guardado en {model_path}")

    # Guardar metadatos (scaler y clases)
    metadata = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "classes": unique_colors
    }
    metadata_path = os.path.join(models_dir, "color_mlp_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadatos del scaler y clases guardados en {metadata_path}")

if __name__ == "__main__":
    main()
