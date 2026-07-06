import os
import json
import torch
import torch.nn as nn
import numpy as np
import math

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

def delta_e_ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h1_prime = math.atan2(b1, a1_prime) % (2.0 * math.pi)
    h2_prime = math.atan2(b2, a2_prime) % (2.0 * math.pi)

    H_bar_prime = (h1_prime + h2_prime) / 2.0
    if abs(h1_prime - h2_prime) > math.pi:
        H_bar_prime = (h1_prime + h2_prime + 2.0 * math.pi) / 2.0

    T = 1.0 - 0.17 * math.cos(H_bar_prime - math.pi / 6.0) + \
        0.24 * math.cos(2.0 * H_bar_prime) + \
        0.32 * math.cos(3.0 * H_bar_prime + math.pi / 30.0) - \
        0.20 * math.cos(4.0 * H_bar_prime - 63.0 * math.pi / 180.0)

    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > math.pi:
        if h2_prime <= h1_prime:
            delta_h_prime += 2.0 * math.pi
        else:
            delta_h_prime -= 2.0 * math.pi

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime / 2.0)

    S_L = 1.0 + (0.015 * (L1 - 50.0)**2) / math.sqrt(20.0 + (L1 - 50.0)**2)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T

    pow7 = C_bar_prime**7
    R_C = 2.0 * math.sqrt(pow7 / (pow7 + 25.0**7))
    d_theta = (30.0 * math.pi / 180.0) * math.exp(-((H_bar_prime - 275.0 * math.pi / 180.0) / (25.0 * math.pi / 180.0))**2)
    R_T = -math.sin(2.0 * d_theta) * R_C

    dist = math.sqrt((delta_L_prime / S_L)**2 + (delta_C_prime / S_C)**2 + (delta_H_prime / S_H)**2 + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))
    return dist

class ColorClassifierAll:
    """
    Hybrid color classifier designed to scale to all database colors.
    Predicts the color group using an MLP router, then determines the exact
    color by evaluating CIEDE2000 distance within the predicted color group.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cen_ready = True
        self.lat_ready = True
        
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(scripts_dir)
        model_path = os.path.join(project_root, "models", "color_router_all.pt")
        metadata_path = os.path.join(project_root, "models", "color_router_all_metadata.json")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.mean_cen = np.array(self.metadata["mean_cenital"], dtype=np.float32)
        self.scale_cen = np.array(self.metadata["scale_cenital"], dtype=np.float32)
        self.mean_lat = np.array(self.metadata["mean_lateral"], dtype=np.float32)
        self.scale_lat = np.array(self.metadata["scale_lateral"], dtype=np.float32)
        
        self.model = ColorMLP(input_dim=12, num_classes=len(self.metadata["classes"]))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load catalog colors
        palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
        self.catalog_colors = []
        if os.path.exists(palette_path):
            with open(palette_path, "r", encoding="utf-8") as f:
                palette = json.load(f)
                for item in palette:
                    self.catalog_colors.append({
                        "color_code": str(item.get("color_code", "")),
                        "color_name": item.get("color_name", "Unknown"),
                        "color_hex": item.get("color_hex", "#808080"),
                        "rgb_cenital": np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float),
                        "rgb_lateral": np.array(item.get("rgb_lateral", [128, 128, 128]), dtype=float),
                        "group_id": get_group_id(item.get("color_name", "Unknown"))
                    })
                    
        self.classes = sorted(list(set(c["color_name"] for c in self.catalog_colors)))
        
    def predict_cenital_probs(self, feature_vector):
        if feature_vector is None:
            return np.zeros(len(self.classes))
        
        # 1. Router probabilities (size 6)
        scaled = (np.array(feature_vector, dtype=np.float32) - self.mean_cen) / (self.scale_cen + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            p_group = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # 2. CIELAB similarities for each class
        lab_est = np.array([feature_vector[0], feature_vector[2], feature_vector[4]], dtype=float)
        
        group_similarities = {i: [] for i in range(6)}
        group_colors = {i: [] for i in range(6)}
        
        for c in self.catalog_colors:
            rgb_ref = c["rgb_cenital"]
            lab_ref = rgb_to_lab(rgb_ref)
            dist = delta_e_ciede2000(lab_est, lab_ref)
            sim = math.exp(-dist / 8.0)
            group_similarities[c["group_id"]].append(sim)
            group_colors[c["group_id"]].append(c["color_name"])
            
        # Normalize similarities within each group
        normalized_probs = {}
        for g_id in range(6):
            sims = group_similarities[g_id]
            sum_sims = sum(sims)
            for c_name, sim in zip(group_colors[g_id], sims):
                p_c_g = (sim / sum_sims) if sum_sims > 0 else (1.0 / len(sims))
                normalized_probs[c_name] = p_group[g_id] * p_c_g
                
        # Build global probability vector
        prob_vector = np.zeros(len(self.classes))
        for idx, c_name in enumerate(self.classes):
            prob_vector[idx] = normalized_probs.get(c_name, 0.0)
            
        return prob_vector

    def predict_lateral_probs(self, feature_vector):
        if feature_vector is None:
            return np.zeros(len(self.classes))
        
        # 1. Router probabilities (size 6)
        scaled = (np.array(feature_vector, dtype=np.float32) - self.mean_lat) / (self.scale_lat + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            p_group = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # 2. CIELAB similarities for each class
        lab_est = np.array([feature_vector[0], feature_vector[2], feature_vector[4]], dtype=float)
        
        group_similarities = {i: [] for i in range(6)}
        group_colors = {i: [] for i in range(6)}
        
        for c in self.catalog_colors:
            rgb_ref = c["rgb_lateral"]
            lab_ref = rgb_to_lab(rgb_ref)
            dist = delta_e_ciede2000(lab_est, lab_ref)
            sim = math.exp(-dist / 8.0)
            group_similarities[c["group_id"]].append(sim)
            group_colors[c["group_id"]].append(c["color_name"])
            
        # Normalize similarities within each group
        normalized_probs = {}
        for g_id in range(6):
            sims = group_similarities[g_id]
            sum_sims = sum(sims)
            for c_name, sim in zip(group_colors[g_id], sims):
                p_c_g = (sim / sum_sims) if sum_sims > 0 else (1.0 / len(sims))
                normalized_probs[c_name] = p_group[g_id] * p_c_g
                
        # Build global probability vector
        prob_vector = np.zeros(len(self.classes))
        for idx, c_name in enumerate(self.classes):
            prob_vector[idx] = normalized_probs.get(c_name, 0.0)
            
        return prob_vector

    def predict_gated_probs_cielab(self, feature_vector, allowed_color_names, camera_type="cenital", is_simulation=False):
        if feature_vector is None:
            return np.zeros(len(self.classes))
            
        lab_est = np.array([feature_vector[0], feature_vector[2], feature_vector[4]], dtype=float)
        use_gating = (allowed_color_names is not None and len(allowed_color_names) > 0)
        
        if is_simulation:
            # Pure CIELAB matching (bypassing OOD MLP router on simulation data)
            sims = []
            c_names = []
            gain = 1.56  # Compensate for Blender dome light intensity = 1.5
            for c in self.catalog_colors:
                c_name_lower = c["color_name"].strip().lower()
                if not use_gating or c_name_lower in allowed_color_names:
                    hex_str = c["color_hex"].lstrip("#")
                    rgb_ref = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
                    rgb_ref = np.clip(rgb_ref * gain, 0.0, 255.0)
                    lab_ref = rgb_to_lab(rgb_ref)
                    dist = delta_e_ciede2000(lab_est, lab_ref)
                    sim = math.exp(-dist / 8.0)
                    sims.append(sim)
                    c_names.append(c["color_name"])
                    
            prob_vector = np.zeros(len(self.classes))
            sum_sims = sum(sims)
            if sum_sims > 0:
                for c_name, sim in zip(c_names, sims):
                    if c_name in self.classes:
                        idx = self.classes.index(c_name)
                        prob_vector[idx] = sim / sum_sims
            elif len(c_names) > 0:
                for c_name in c_names:
                    if c_name in self.classes:
                        idx = self.classes.index(c_name)
                        prob_vector[idx] = 1.0 / len(c_names)
            return prob_vector
            
        # 1. Router probabilities (size 6)
        mean_val = self.mean_cen if camera_type == "cenital" else self.mean_lat
        scale_val = self.scale_cen if camera_type == "cenital" else self.scale_lat
        scaled = (np.array(feature_vector, dtype=np.float32) - mean_val) / (scale_val + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            p_group = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # 2. CIELAB similarities for each class
        group_similarities = {i: [] for i in range(6)}
        group_colors = {i: [] for i in range(6)}
        
        for c in self.catalog_colors:
            c_name_lower = c["color_name"].strip().lower()
            if not use_gating or c_name_lower in allowed_color_names:
                rgb_ref = c["rgb_cenital"] if camera_type == "cenital" else c["rgb_lateral"]
                lab_ref = rgb_to_lab(rgb_ref)
                dist = delta_e_ciede2000(lab_est, lab_ref)
                sim = math.exp(-dist / 8.0)
                group_similarities[c["group_id"]].append(sim)
                group_colors[c["group_id"]].append(c["color_name"])
                
        # Normalize similarities within each group
        normalized_probs = {}
        for g_id in range(6):
            sims = group_similarities[g_id]
            sum_sims = sum(sims)
            if len(sims) > 0:
                for c_name, sim in zip(group_colors[g_id], sims):
                    p_c_g = (sim / sum_sims) if sum_sims > 0 else (1.0 / len(sims))
                    normalized_probs[c_name] = p_group[g_id] * p_c_g
                    
        # Build global probability vector
        prob_vector = np.zeros(len(self.classes))
        sum_probs = sum(normalized_probs.values())
        if sum_probs > 0:
            for idx, c_name in enumerate(self.classes):
                prob_vector[idx] = normalized_probs.get(c_name, 0.0) / sum_probs
        elif len(normalized_probs) > 0:
            for idx, c_name in enumerate(self.classes):
                prob_vector[idx] = normalized_probs.get(c_name, 0.0)
                    
        return prob_vector

    def predict_fused_colors_flexible(self, feat_cen, feat_lat, threshold=0.25):
        p_cen = self.predict_cenital_probs(feat_cen)
        p_lat = self.predict_lateral_probs(feat_lat)
        
        if feat_cen is not None and feat_lat is not None:
            p_combined = p_cen * p_lat
            if np.sum(p_combined) == 0:
                p_combined = p_cen + p_lat
        elif feat_cen is not None:
            p_combined = p_cen
        elif feat_lat is not None:
            p_combined = p_lat
        else:
            return ["Unknown"]
            
        sorted_indices = np.argsort(p_combined)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        
        top1_prob = p_combined[top1_idx]
        top2_prob = p_combined[top2_idx]
        
        sum_prob = np.sum(p_combined)
        if sum_prob > 0:
            top1_prob /= sum_prob
            top2_prob /= sum_prob
            
        colors = [self.classes[top1_idx]]
        if (top1_prob - top2_prob) < threshold:
            colors.append(self.classes[top2_idx])
        return colors

    @property
    def all_classes(self):
        return self.classes
