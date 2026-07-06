# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import numpy as np
import math
import cv2

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

def delta_e_ciede2000(lab1, lab2, wL=1.0, wC=1.0, wH=1.0):
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

    dist = math.sqrt(
        (delta_L_prime / (wL * S_L))**2 + 
        (delta_C_prime / (wC * S_C))**2 + 
        (delta_H_prime / (wH * S_H))**2 + 
        R_T * (delta_C_prime / (wC * S_C)) * (delta_H_prime / (wH * S_H))
    )
    return dist

class ColorClassifierAllAdapted:
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cen_ready = True
        self.lat_ready = True
        
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(scripts_dir)
        model_path = os.path.join(project_root, "models", "color_all.pt")
        metadata_path = os.path.join(project_root, "models", "color_all_metadata.json")
        
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
                    
        # Load empirical colors cache to bypass domain calibration for known simulation colors
        self.empirical_colors = {}
        cache_path = os.path.join(project_root, "data", "mlp_features_cache.npz")
        if os.path.exists(cache_path):
            try:
                cache_data = np.load(cache_path, allow_pickle=True)
                X_cen_raw = cache_data["X_cen"]
                y_cen_raw = cache_data["y_cen"]
                for c_name in set(y_cen_raw):
                    c_name_str = str(c_name)
                    mask = (y_cen_raw == c_name)
                    mean_feats = X_cen_raw[mask].mean(axis=0)
                    self.empirical_colors[c_name_str.strip().lower()] = np.array([
                        mean_feats[0], mean_feats[2], mean_feats[4]
                    ])
            except Exception as e:
                print(f"[Warning] Error loading empirical colors from cache: {e}")

        self.classes = sorted(list(set(c["color_name"] for c in self.catalog_colors)))

    def _make_single_class_prob(self, class_name):
        prob_vector = np.zeros(len(self.classes))
        for idx, c_name in enumerate(self.classes):
            if c_name.strip().lower() == class_name.strip().lower():
                prob_vector[idx] = 1.0
                break
        return prob_vector

    def predict_cenital_probs(self, feature_vector):
        return self.predict_gated_probs_cielab(feature_vector, None, "cenital")

    def predict_lateral_probs(self, feature_vector):
        return self.predict_gated_probs_cielab(feature_vector, None, "lateral")

    def predict_gated_probs_cielab(self, feature_vector, allowed_color_names, camera_type="cenital", is_simulation=False):
        if feature_vector is None:
            return np.zeros(len(self.classes))
            
        lab_est = np.array([feature_vector[0], feature_vector[2], feature_vector[4]], dtype=float)
        hsv_est = np.array([feature_vector[6], feature_vector[8], feature_vector[10]], dtype=float)
        lab_std = np.array([feature_vector[1], feature_vector[3], feature_vector[5]], dtype=float)
        hsv_std = np.array([feature_vector[7], feature_vector[9], feature_vector[11]], dtype=float)
        
        use_gating = (allowed_color_names is not None and len(allowed_color_names) > 0)
        if use_gating:
            allowed_color_names = set(x.strip().lower() for x in allowed_color_names)
        
        # 1. Router probabilities (size 6)
        mean_val = self.mean_cen if camera_type == "cenital" else self.mean_lat
        scale_val = self.scale_cen if camera_type == "cenital" else self.scale_lat
        scaled = (np.array(feature_vector, dtype=np.float32) - mean_val) / (scale_val + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            p_group = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        max_idx = np.argmax(p_group)
        max_prob = p_group[max_idx]
        
        # ==========================================
        # CAPA 1: Fast-Path Determinista (evita domain shift del router MLP)
        # ==========================================
        chroma_est = math.sqrt(lab_est[1]**2 + lab_est[2]**2)
        
        # Caso 1.1: Blanco puro o Negro puro (luminancia extrema y bajo croma)
        if chroma_est < 6.0:
            if lab_est[0] > 82.0:
                return self._make_single_class_prob("White")
            elif lab_est[0] < 15.0:
                return self._make_single_class_prob("Black")
                
        # Caso 1.2: Colores altamente saturados y claros
        if chroma_est > 35.0:
            hue = hsv_est[0]
            g_id = None
            if hue < 15.0 or hue > 340.0:
                g_id = 3  # Red
            elif 15.0 <= hue < 55.0:
                g_id = 4  # Yellow/Orange
            elif 55.0 <= hue < 160.0:
                g_id = 2  # Green
            elif 160.0 <= hue < 250.0:
                g_id = 1  # Blue
                
            if g_id is not None:
                allowed_group_ids = {g_id}
                wL, wC, wH = 1.0, 1.0, 1.0
                
                group_similarities = {i: [] for i in range(6)}
                group_colors = {i: [] for i in range(6)}
                for c in self.catalog_colors:
                    c_name_lower = c["color_name"].strip().lower()
                    if (not use_gating or c_name_lower in allowed_color_names) and (c["group_id"] in allowed_group_ids):
                        if c_name_lower in self.empirical_colors:
                            lab_ref = self.empirical_colors[c_name_lower].copy()
                        else:
                            hex_str = c["color_hex"].lstrip("#")
                            rgb_ref = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
                            rgb_ref = np.clip(rgb_ref * 1.56, 0.0, 255.0)
                            lab_ref = rgb_to_lab(rgb_ref)
                            lab_ref[0] = 0.3928 * lab_ref[0] + 23.2449
                            lab_ref[1] = 0.40 * lab_ref[1]
                            lab_ref[2] = 0.37 * lab_ref[2]
                            
                        dist = delta_e_ciede2000(lab_est, lab_ref, wL, wC, wH)
                        sim = math.exp(-dist / 8.0)
                        group_similarities[c["group_id"]].append(sim)
                        group_colors[c["group_id"]].append(c["color_name"])
                
                sims = group_similarities[g_id]
                if len(sims) > 0:
                    normalized_probs = {}
                    for c_name, sim in zip(group_colors[g_id], sims):
                        normalized_probs[c_name] = sim
                    prob_vector = np.zeros(len(self.classes))
                    sum_probs = sum(normalized_probs.values())
                    if sum_probs > 0:
                        for idx, c_name in enumerate(self.classes):
                            prob_vector[idx] = normalized_probs.get(c_name, 0.0) / sum_probs
                        return prob_vector
                        
        # ==========================================
        # CAPA 2: Rutas Especializadas para Casos Complejos
        # ==========================================
        # Determinar si es un color Translúcido por alta varianza y claridad (Rama 2A)
        is_translucent_est = (hsv_std[1] > 28.0 and hsv_std[2] > 28.0 and lab_est[0] > 60.0)
        
        # Determinar si es Neutro o Metálico por bajo croma (Rama 2B)
        is_neutral = (chroma_est < 15.0)
        
        allowed_group_ids = {0, 1, 2, 3, 4, 5}
        
        group_similarities = {i: [] for i in range(6)}
        group_colors = {i: [] for i in range(6)}
        
        for c in self.catalog_colors:
            c_name_lower = c["color_name"].strip().lower()
            if (not use_gating or c_name_lower in allowed_color_names) and (c["group_id"] in allowed_group_ids):
                if c_name_lower in self.empirical_colors:
                    lab_ref = self.empirical_colors[c_name_lower].copy()
                else:
                    hex_str = c["color_hex"].lstrip("#")
                    rgb_ref = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
                    rgb_ref = np.clip(rgb_ref * 1.56, 0.0, 255.0)
                    lab_ref = rgb_to_lab(rgb_ref)
                    lab_ref[0] = 0.3928 * lab_ref[0] + 23.2449
                    lab_ref[1] = 0.40 * lab_ref[1]
                    lab_ref[2] = 0.37 * lab_ref[2]
                
                is_ref_trans = "trans-" in c_name_lower
                if is_translucent_est:
                    trans_penalty = 1.0 if is_ref_trans else 0.15
                else:
                    trans_penalty = 0.05 if is_ref_trans else 1.0
                
                if is_translucent_est:
                    # Rama 2A: Manejador de Translúcidos (pesos y penalización de Hue)
                    ref_hex = c["color_hex"].lstrip("#")
                    ref_rgb = np.array([int(ref_hex[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)
                    ref_hsv = cv2.cvtColor(ref_rgb.reshape(1,1,3), cv2.COLOR_RGB2HSV).reshape(-1)[0] * 2.0
                    
                    hue_diff = abs(hsv_est[0] - ref_hsv)
                    if hue_diff > 180.0:
                        hue_diff = 360.0 - hue_diff
                    hue_penalty = math.exp(-hue_diff / 30.0)
                    
                    dist = delta_e_ciede2000(lab_est, lab_ref, wL=1.5, wC=0.8, wH=0.8)
                    sim = math.exp(-dist / 8.0) * trans_penalty * hue_penalty
                
                elif is_neutral:
                    # Rama 2B: Manejador de Neutros y Metálicos (pesos en luminosidad y especularidad)
                    dist = delta_e_ciede2000(lab_est, lab_ref, wL=0.5, wC=3.0, wH=3.0)
                    sim = math.exp(-dist / 8.0) * trans_penalty
                    
                    is_ref_metallic = any(w in c_name_lower for w in ["silver", "gold", "metallic"])
                    if is_ref_metallic:
                        if lab_std[0] < 4.0:
                            sim *= 0.1
                    else:
                        if lab_std[0] > 6.0:
                            sim *= 0.3
                
                else:
                    # Rama 2C: Sombras Cromáticas (Red vs Dark Red, etc.)
                    dist = delta_e_ciede2000(lab_est, lab_ref, wL=1.0, wC=1.0, wH=1.0)
                    sim = math.exp(-dist / 8.0) * trans_penalty
                    
                    if "dark red" in c_name_lower and lab_est[0] > 40.0:
                        sim *= 0.2
                    elif c_name_lower == "red" and lab_est[0] < 32.0:
                        sim *= 0.2
                
                group_similarities[c["group_id"]].append(sim)
                group_colors[c["group_id"]].append(c["color_name"])
                
        normalized_probs = {}
        for g_id in allowed_group_ids:
            sims = group_similarities[g_id]
            if len(sims) > 0:
                for c_name, sim in zip(group_colors[g_id], sims):
                    normalized_probs[c_name] = sim
                    
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
        p_combined = p_cen
            
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
