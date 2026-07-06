import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

# Setup project paths
project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
CACHE_PATH = os.path.join(project_root, "data", "mlp_features_cache.npz")
MODELS_DIR = os.path.join(project_root, "models")

# Define the 6 groups
GROUP_NAMES = ['Grays', 'Blues', 'Greens', 'Reds', 'Yellows', 'Browns']

def get_group_id(color_name):
    name = color_name.strip().lower()
    if any(w in name for w in ['gray', 'grey', 'white', 'black', 'silver', 'slate', 'stone', 'metallic']):
        return 0
    elif any(w in name for w in ['blue', 'purple', 'violet', 'indigo', 'lavender', 'lilac', 'sky']):
        return 1
    elif any(w in name for w in ['green', 'lime', 'olive', 'turquoise', 'teal']):
        return 2
    elif any(w in name for w in ['red', 'pink', 'magenta', 'coral', 'rose']):
        return 3
    elif any(w in name for w in ['yellow', 'orange', 'gold', 'peach', 'apricot']):
        return 4
    elif any(w in name for w in ['brown', 'nougat', 'flesh', 'tan', 'sand', 'copper', 'salmon']):
        return 5
    return 0

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
    h = h / 2.0  # Scale to [0-180] to match OpenCV
    return np.array([h, s, v])

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

def augment_dataset(X, y_names, camera_name):
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    with open(palette_path, "r", encoding="utf-8") as f:
        palette = json.load(f)
        
    rgb_key = "rgb_cenital" if camera_name == "cenital" else "rgb_lateral"
    
    color_db = {}
    for item in palette:
        name = item["color_name"].strip().lower()
        rgb = np.array(item.get(rgb_key, [128, 128, 128]), dtype=np.float32)
        color_db[name] = {
            "name": item["color_name"],
            "group": get_group_id(item["color_name"]),
            "lab": local_rgb_to_lab(rgb),
            "hsv": local_rgb_to_hsv(rgb)
        }
        
    X_aug = []
    y_aug_groups = []
    
    source_colors = list(set(name.strip().lower() for name in y_names))
    source_indices = {c: [i for i, val in enumerate(y_names) if val.strip().lower() == c] for c in source_colors}
    
    samples_per_class = 500
    
    for target_name, target_info in color_db.items():
        src_color = None
        for sc in source_colors:
            if color_db.get(sc, {}).get("group") == target_info["group"]:
                src_color = sc
                break
        if src_color is None:
            src_color = source_colors[0]
            
        src_info = color_db[src_color]
        src_indices = source_indices[src_color]
        
        delta_lab = target_info["lab"] - src_info["lab"]
        delta_hsv = target_info["hsv"] - src_info["hsv"]
        
        for _ in range(samples_per_class):
            base_idx = random.choice(src_indices)
            feat = X[base_idx].copy()
            
            feat[0] += delta_lab[0]
            feat[2] += delta_lab[1]
            feat[4] += delta_lab[2]
            
            feat[6] = (feat[6] + delta_hsv[0]) % 180.0
            feat[8] += delta_hsv[1]
            feat[10] += delta_hsv[2]
            
            # Jitter
            shadow_factor = random.uniform(0.65, 1.15)
            feat[0] *= shadow_factor
            feat[10] *= shadow_factor
            
            feat[6] = (feat[6] + random.uniform(-5.0, 5.0)) % 180.0
            feat[8] = np.clip(feat[8] + random.uniform(-15.0, 15.0), 0.0, 255.0)
            feat[10] = np.clip(feat[10] + random.uniform(-25.0, 15.0), 0.0, 255.0)
            
            feat[1] = np.clip(feat[1] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[3] = np.clip(feat[3] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[5] = np.clip(feat[5] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[7] = np.clip(feat[7] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[9] = np.clip(feat[9] * random.uniform(0.8, 1.3), 1.0, 40.0)
            feat[11] = np.clip(feat[11] * random.uniform(0.8, 1.3), 1.0, 40.0)
            
            X_aug.append(feat)
            y_aug_groups.append(target_info["group"])
            
    return np.array(X_aug, dtype=np.float32), y_aug_groups

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] Cache file not found at {CACHE_PATH}")
        return
        
    cache_data = np.load(CACHE_PATH, allow_pickle=True)
    X_cen_raw = cache_data["X_cen"]
    y_cen_names = list(cache_data["y_cen"])
    X_lat_raw = cache_data["X_lat"]
    y_lat_names = list(cache_data["y_lat"])
    
    random.seed(2026)

    # Augmentación paralela de ambas cámaras usando multiprocessing
    print("Generating augmented datasets in parallel (2 procesos)...")
    with mp.Pool(processes=2) as pool:
        results = pool.starmap(augment_dataset, [
            (X_cen_raw, y_cen_names, "cenital"),
            (X_lat_raw, y_lat_names, "lateral")
        ])
    X_cen_aug, y_cen_groups = results[0]
    X_lat_aug, y_lat_groups = results[1]

    # Scale Cenital
    scaler_cen = StandardScaler()
    X_cen_scaled = scaler_cen.fit_transform(X_cen_aug)

    # Scale Lateral
    scaler_lat = StandardScaler()
    X_lat_scaled = scaler_lat.fit_transform(X_lat_aug)

    # Combine datasets
    X_all = np.concatenate([X_cen_scaled, X_lat_scaled], axis=0)
    y_all = np.concatenate([y_cen_groups, y_lat_groups], axis=0)

    print(f"Total samples: {len(X_all)}")

    # Batch grande para aprovechar GPU M4 y 48 GB de RAM
    batch_size = 1024
    dataset = SimpleFeatureDataset(X_all, y_all, add_noise=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    model = ColorMLP(input_dim=12, num_classes=len(GROUP_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    print("Training router model...")
    model.train()
    for epoch in range(150):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/150, Loss: {total_loss/len(loader):.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")
            
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "color_router_all_colors.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    metadata = {
        "mean_cenital": scaler_cen.mean_.tolist(),
        "scale_cenital": scaler_cen.scale_.tolist(),
        "mean_lateral": scaler_lat.mean_.tolist(),
        "scale_lateral": scaler_lat.scale_.tolist(),
        "classes": GROUP_NAMES
    }
    metadata_path = os.path.join(MODELS_DIR, "color_router_all_colors_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
