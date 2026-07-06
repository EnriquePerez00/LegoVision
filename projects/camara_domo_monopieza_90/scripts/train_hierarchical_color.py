# -*- coding: utf-8 -*-
"""
train_hierarchical_color.py
===========================
Trains the hierarchical color classification system (Router + 5 Specialists)
for both Cenital and Lateral cameras.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Setup project paths
project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_75078"
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, project_root)

from hierarchical_router import RouterMLP, get_group_id
from train_and_evaluate_color_mlp import ColorMLP, SimpleFeatureDataset

CACHE_PATH = os.path.join(project_root, "data", "mlp_features_cache.npz")
MODELS_DIR = os.path.join(project_root, "models", "hierarchical")

# Helper color converters
def local_rgb_to_lab(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
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
    return np.array([(116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)])

def local_rgb_to_hsv(rgb):
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0.0
    elif mx == r:
        h = (60.0 * ((g-b)/df) + 360.0) % 360.0
    elif mx == g:
        h = (60.0 * ((b-r)/df) + 120.0) % 360.0
    elif mx == b:
        h = (60.0 * ((r-g)/df) + 240.0) % 360.0
    s = 0.0 if mx == 0.0 else (df/mx)*255.0
    v = mx*255.0
    return np.array([h, s, v])

def train_network(model, dataset, epochs=120, lr=0.01):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=len(dataset) > 2048)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def augment_dataset(X, y_names, camera_name):
    """Generates synthetic training features for all 179 palette colors using feature space shifting and jitter."""
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    with open(palette_path, "r", encoding="utf-8") as f:
        palette = json.load(f)
        
    rgb_key = "rgb_cenital" if camera_name == "cenital" else "rgb_lateral"
    
    # Map colors from palette
    color_db = {}
    for item in palette:
        name = item["color_name"].strip().lower()
        rgb = np.array(item.get(rgb_key, [128, 128, 128]), dtype=np.float32)
        color_db[name] = {
            "name": item["color_name"],
            "group": get_group_id(name),
            "lab": local_rgb_to_lab(rgb),
            "hsv": local_rgb_to_hsv(rgb)
        }
        
    X_aug = []
    y_aug_names = []
    
    # Group X by source color name
    source_colors = list(set(name.strip().lower() for name in y_names))
    source_indices = {c: [i for i, val in enumerate(y_names) if val.strip().lower() == c] for c in source_colors}
    
    print(f"[Augment] Generando muestras sintéticas para los 179 colores del catálogo...")
    
    # Target: 500 samples per color class in the training set
    samples_per_class = 400
    
    for target_name, target_info in color_db.items():
        # Find closest available source color based on group ID first, fallback to any
        src_color = None
        for sc in source_colors:
            if color_db.get(sc, {}).get("group") == target_info["group"]:
                src_color = sc
                break
        if src_color is None:
            src_color = source_colors[0]
            
        src_info = color_db[src_color]
        src_indices = source_indices[src_color]
        
        # Calculate color space differences
        delta_lab = target_info["lab"] - src_info["lab"]
        delta_hsv = target_info["hsv"] - src_info["hsv"]
        
        for _ in range(samples_per_class):
            # Select random base sample from source class
            base_idx = random.choice(src_indices)
            feat = X[base_idx].copy()
            
            # Apply shift
            feat[0] += delta_lab[0]  # mean L
            feat[2] += delta_lab[1]  # mean a
            feat[4] += delta_lab[2]  # mean b
            
            feat[6] = (feat[6] + delta_hsv[0]) % 360.0  # mean H
            feat[8] += delta_hsv[1]  # mean S
            feat[10] += delta_hsv[2] # mean V
            
            # --- Jitter & Shadow Augmentation (Max HW execution) ---
            # 1. Shadow Factor (decrease L and V)
            shadow_factor = random.uniform(0.65, 1.15)
            feat[0] *= shadow_factor
            feat[10] *= shadow_factor
            
            # 2. Color Jitter on HSV channels
            feat[6] = (feat[6] + random.uniform(-10.0, 10.0)) % 360.0
            feat[8] = np.clip(feat[8] + random.uniform(-15.0, 15.0), 0.0, 255.0)
            feat[10] = np.clip(feat[10] + random.uniform(-25.0, 15.0), 0.0, 255.0)
            
            # 3. Add noise to variance/std devs
            feat[1] = np.clip(feat[1] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[3] = np.clip(feat[3] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[5] = np.clip(feat[5] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[7] = np.clip(feat[7] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[9] = np.clip(feat[9] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[11] = np.clip(feat[11] * random.uniform(0.8, 1.3), 1.0, 40.0)
            
            X_aug.append(feat)
            y_aug_names.append(target_info["name"])
            
    return np.array(X_aug, dtype=np.float32), y_aug_names

import random

def train_pipeline_for_camera(camera_name, X, y_names):
    print(f"\n==========================================")
    print(f"ENTRENANDO PIPELINE JERÁRQUICO - CÁMARA: {camera_name.upper()}")
    print(f"==========================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cam_dir = os.path.join(MODELS_DIR, camera_name)
    os.makedirs(cam_dir, exist_ok=True)
    
    # Apply synthetic augmentation to generate X_aug for all 179 colors
    random.seed(2026)
    X_aug, y_aug_names = augment_dataset(X, y_names, camera_name)
    
    y_groups = np.array([get_group_id(name) for name in y_aug_names])
    
    # --- ENTRENAR EL ENRUTADOR (NIVEL 1) ---
    print(f"\n[Nivel 1] Entrenando Enrutador sobre {len(X_aug)} muestras...")
    router_scaler = StandardScaler()
    X_router_scaled = router_scaler.fit_transform(X_aug)
    
    router_dataset = SimpleFeatureDataset(X_router_scaled, y_groups, add_noise=True)
    router_model = RouterMLP(input_dim=12, num_groups=5)
    router_model = train_network(router_model, router_dataset, epochs=60, lr=0.01)
    
    # Guardar enrutador
    router_path = os.path.join(cam_dir, "router.pt")
    torch.save(router_model.state_dict(), router_path)
    print(f"✓ Enrutador guardado en {router_path}")
    
    # --- ENTRENAR LOS ESPECIALISTAS (NIVEL 2) ---
    specialist_meta = {}
    
    for g_id in range(5):
        mask = (y_groups == g_id)
        X_g = X_aug[mask]
        y_names_g = [y_aug_names[i] for i, m in enumerate(mask) if m]
        
        if len(X_g) < 5:
            print(f"\n[Nivel 2] Grupo {g_id} ({len(X_g)} muestras) - Muy pocos datos. Saltando.")
            continue
            
        unique_classes_g = sorted(list(set(y_names_g)))
        class_to_idx = {name: idx for idx, name in enumerate(unique_classes_g)}
        y_g_local = np.array([class_to_idx[name] for name in y_names_g])
        
        print(f"\n[Nivel 2] Entrenando Especialista {g_id} ({len(unique_classes_g)} clases, {len(X_g)} muestras)...")
        spec_scaler = StandardScaler()
        X_g_scaled = spec_scaler.fit_transform(X_g)
        
        spec_dataset = SimpleFeatureDataset(X_g_scaled, y_g_local, add_noise=True)
        spec_model = ColorMLP(input_dim=12, num_classes=len(unique_classes_g))
        spec_model = train_network(spec_model, spec_dataset, epochs=70, lr=0.01)
        
        # Guardar especialista
        spec_path = os.path.join(cam_dir, f"spec{g_id}.pt")
        torch.save(spec_model.state_dict(), spec_path)
        print(f"✓ Especialista {g_id} guardado en {spec_path}")
        
        # Guardar metadatos del especialista
        specialist_meta[str(g_id)] = {
            "classes": unique_classes_g,
            "mean": spec_scaler.mean_.tolist(),
            "scale": spec_scaler.scale_.tolist()
        }
        
    # Guardar metadatos generales
    metadata = {
        "router": {
            "mean": router_scaler.mean_.tolist(),
            "scale": router_scaler.scale_.tolist()
        },
        "specialists": specialist_meta
    }
    
    meta_path = os.path.join(cam_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadatos guardados en {meta_path}")

def main():
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] No existe el caché {CACHE_PATH}. Ejecuta primero el script de extracción de características.")
        sys.exit(1)
        
    print("Cargando características del caché...")
    cache_data = np.load(CACHE_PATH, allow_pickle=True)
    X_cen = cache_data["X_cen"]
    y_cen = list(cache_data["y_cen"])
    X_lat = cache_data["X_lat"]
    y_lat = list(cache_data["y_lat"])
    
    # Entrenar pipelines
    train_pipeline_for_camera("cenital", X_cen, y_cen)
    train_pipeline_for_camera("lateral", X_lat, y_lat)
    
    print("\n✓ Proceso de entrenamiento jerárquico completado con éxito.")

if __name__ == "__main__":
    main()
