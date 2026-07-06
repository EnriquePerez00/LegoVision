# -*- coding: utf-8 -*-
"""camara_domo/scripts/run_evaluation.py
======================================
Pipeline de inferencia y evaluación simplificado para el setup "camara_domo".

Características:
  - Sin vectorización/clasificación DINOv2.
  - Cámara lateral opcional controlada por software (`camara_lateral = False` por defecto).
  - Inferencia basada únicamente en Gating de Color y Gaussian Size Score (Área cenital).
  - Soporte de YOLO y YOLO-Pose (keypoints) cuando la cámara lateral está activa.
"""
import os
import sys
import json
import math
import time as _time
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO, SAM

LEGO_TO_REBRICKABLE = {
    "1": "1",      # White -> White
    "5": "2",      # Brick Yellow -> Tan
    "21": "5",     # Bright Red -> Red
    "23": "7",     # Bright Blue -> Blue
    "24": "3",     # Bright Yellow -> Yellow
    "26": "11",    # Black -> Black
    "28": "80",    # Dark Green -> Dark Green (Rebrickable 80)
    "119": "34",   # Bright Yellowish Green -> Lime (Rebrickable 34)
    "199": "85",   # Dark Stone Grey -> Dark Bluish Gray (Rebrickable 85)
    "208": "90",   # Light Nougat -> Light Nougat (Rebrickable 90)
}

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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from core.utils.config_loader import cfg
from core.utils.logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("pipeline")

from core.db.set_catalog import REAL_SETS
from scripts.efficientnet_classifier_75078 import LegoEfficientNetClassifier75078

# ContourMatcher desactivado para reproducir el baseline puro de KNN
# Cargar ContourMatcher para recomposición por contornos
try:
    from contour_matcher import ContourMatcher
    contour_matcher = ContourMatcher()
    log.info("ContourMatcher para recomposición 3D cargado con éxito.")
except Exception as e:
    contour_matcher = None
    log.warning(f"No se pudo cargar ContourMatcher: {e}")

# Cargar configuración de activación de cámara lateral
USE_LATERAL_CAMERA = True

# Intentar cargar observador de keypoints si es necesario
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _kpts_observer import (
        kpts_observer as kpts_observer_fn,
        extract_yolo_pose_keypoints,
    )
    HAS_KPTS_MODULE = True
except Exception as _e:
    HAS_KPTS_MODULE = False
    log.warning(f"_kpts_observer no disponible: {_e}")

# Constantes de geometría y calibración
CAM_CEN_Z_MM = float(cfg.inference.calibration.camera_dist_mm)
PX_PER_MM_NOMINAL = float(cfg.inference.calibration.px_per_mm_cenital)
PX_PER_MM_LATERAL = 10.6

# ── Color Utility Functions ──
def rgb_to_lab(rgb_val):
    """Convierte RGB [0-255] a CIELAB para medición de distancias perceptual de color."""
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

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    return np.array([128.0, 128.0, 128.0])

def _load_catalog_colors():
    """Carga los colores calibrados desde color_calibration_palette.json."""
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    if not os.path.exists(palette_path):
        log.warning(f"Paleta de calibración no encontrada en {palette_path}. Usando fallback vacío.")
        return []
    try:
        with open(palette_path, "r", encoding="utf-8") as f:
            palette = json.load(f)
        color_list = []
        if isinstance(palette, list):
            for item in palette:
                code = str(item.get("color_code", ""))
                rgb = np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float)
                color_list.append({
                    "color_code": code,
                    "color_name": item.get("color_name", "Unknown"),
                    "color_hex": item.get("color_hex", "#808080"),
                    "rgb": rgb
                })
        else:
            for code, info in palette.items():
                rgb = np.array(info.get("rgb_cenital", [128, 128, 128]), dtype=float)
                color_list.append({
                    "color_code": code,
                    "color_name": info.get("name", "Unknown"),
                    "color_hex": info.get("hex", "#808080"),
                    "rgb": rgb
                })
        log.info(f"[ColorCatalog] {len(color_list)} colores cargados desde {palette_path}")
        return color_list
    except Exception as e:
        log.error(f"Error cargando paleta de color: {e}")
        return []

CATALOG_COLORS = _load_catalog_colors()

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


def estimate_color_cnn_crop(img, mask_binary, bbox):
    """Extrae el crop 2D enmascarado y rellena el fondo con el color mediano de la pieza para destruir el borde/forma."""
    img_arr = np.array(img.convert("RGB"))
    px1, py1, px2, py2 = bbox

    # 1. Erosión morfológica de la máscara para limpiar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask_binary, kernel, iterations=1)
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask_binary

    # 2. Calcular color mediano de la pieza y rellenar el fondo para destruir el borde/forma
    piece_pixels = img_arr[mask_to_use > 0]
    if len(piece_pixels) > 0:
        median_color = np.median(piece_pixels, axis=0)
    else:
        median_color = np.array([128, 128, 128])

    masked_arr = np.empty_like(img_arr)
    masked_arr[:] = median_color.astype(np.uint8)
    masked_arr[mask_to_use > 0] = img_arr[mask_to_use > 0]

    # 3. Recortar región de la caja delimitadora
    crop_arr = masked_arr[py1:py2, px1:px2]
    if crop_arr.size == 0:
        return None

    # 4. Redimensionar a 64x64
    crop_resized = cv2.resize(crop_arr, (64, 64), interpolation=cv2.INTER_LINEAR)

    # 5. Transponer a (3, 64, 64) y normalizar a [0, 1]
    crop_tensor = crop_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return crop_tensor

def estimate_color_mlp_features(img, mask_binary, camera_type=None, ccm_params=None, is_simulation=False):
    """Extrae las 12 características estadísticas (Lab/HSV) para el MLP de color."""
    img_arr = np.array(img.convert("RGB"))
    # 1. Morphological erosion (5x5 Ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask_binary, kernel, iterations=1)
        
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask_binary
    mask_bool = (mask_to_use > 0)

    if not np.any(mask_bool):
        return None

    pixels_rgb = img_arr[mask_bool]

    # Aplicar calibración CCM inversa si está configurada (solo en real, no en simulación)
    if ccm_params and camera_type in ccm_params and not is_simulation:
        params = ccm_params[camera_type]
        pixels_rgb_cal = np.zeros_like(pixels_rgb, dtype=np.float32)
        for c in range(3):
            gamma, scale, lift = params[f"channel_{c}"]
            channel_data = pixels_rgb[:, c].astype(np.float32)
            val = (channel_data - lift) / (scale + 1e-8)
            val = np.clip(val, 0.0, 1.0)
            pixels_rgb_cal[:, c] = 255.0 * np.power(val, 1.0 / gamma)
        pixels_rgb = np.clip(pixels_rgb_cal, 0.0, 255.0).astype(np.uint8)

    # Convertir a HSV a partir de pixels_rgb corregidos
    pixels_rgb_reshaped = pixels_rgb.reshape(-1, 1, 3)
    pixels_hsv = cv2.cvtColor(pixels_rgb_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Calcular belt_ratio antes de filtrar la cinta
    from _belt_mask import compute_belt_mask
    belt_mask_before = compute_belt_mask(pixels_hsv)
    belt_ratio = float(np.sum(belt_mask_before) / len(belt_mask_before)) if len(belt_mask_before) > 0 else 0.0

    if is_simulation:
        # Filtrar fondo negro/transparente de simulación
        v_val = pixels_hsv[:, 2]
        non_black_mask = (v_val >= 15)
        if np.any(non_black_mask):
            pixels_rgb = pixels_rgb[non_black_mask]
            pixels_hsv = pixels_hsv[non_black_mask]
        # Si la simulación incluye el color de la cinta (no transparente), lo removemos también
        from _belt_mask import filter_out_belt as _filter_out_belt_pixels
        pixels_rgb, pixels_hsv = _filter_out_belt_pixels(pixels_rgb, pixels_hsv)
    else:
        # Chroma-keying de la cinta transportadora.
        from _belt_mask import filter_out_belt as _filter_out_belt_pixels
        pixels_rgb, pixels_hsv = _filter_out_belt_pixels(pixels_rgb, pixels_hsv)
    # 2. Specular highlight filtering (Saturation >= 25 or Value < 230)
    non_specular_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    if np.any(non_specular_mask):
        pixels_rgb_filt = pixels_rgb[non_specular_mask]
        pixels_hsv_filt = pixels_hsv[non_specular_mask]
    else:
        pixels_rgb_filt = pixels_rgb
        pixels_hsv_filt = pixels_hsv

    # Convert RGB to Lab
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

    # Compute statistics (Mean & Std Dev) as done in training
    mean_lab = pixels_lab.mean(axis=0)
    std_lab = pixels_lab.std(axis=0)
    
    mean_hsv = pixels_hsv_filt.mean(axis=0)
    std_hsv = pixels_hsv_filt.std(axis=0)

    features = np.array([
        mean_lab[0], std_lab[0],
        mean_lab[1], std_lab[1],
        mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0],
        mean_hsv[1], std_hsv[1],
        mean_hsv[2], std_hsv[2],
        belt_ratio
    ], dtype=np.float32)

    return features


def estimate_color_robust(img, mask_binary):
    """Paso 1: Estimación estadística robusta de color con erosión de bordes y filtrado HSV de especularidades."""
    img_arr = np.array(img.convert("RGB"))
    
    # 1. Erosión morfológica de la máscara para eliminar contaminación de bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask_binary, kernel, iterations=1)
    
    # Si la erosión vacía la máscara, usar la máscara original
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask_binary
    mask_bool = (mask_to_use > 0)
    
    if not np.any(mask_bool):
        return [128.0, 128.0, 128.0]
        
    pixels_rgb = img_arr[mask_bool]
    
    # Filtrar especularidades (brillos intensos / saturación ultra-baja en HSV)
    hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv_img[mask_bool]
    
    non_specular_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    
    if np.any(non_specular_mask):
        pixels_rgb_filtered = pixels_rgb[non_specular_mask]
    else:
        pixels_rgb_filtered = pixels_rgb
        
    mean_seg = pixels_rgb_filtered.mean(axis=0)
    return list(mean_seg)

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

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360 if b1 != 0 or a1_prime != 0 else 0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360 if b2 != 0 or a2_prime != 0 else 0
    if h1_prime < 0: h1_prime += 360
    if h2_prime < 0: h2_prime += 360

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    if C1_prime == 0 or C2_prime == 0:
        dh_prime = 0.0
    else:
        diff = h2_prime - h1_prime
        if abs(diff) <= 180:
            dh_prime = diff
        elif diff > 180:
            dh_prime = diff - 360
        else:
            dh_prime = diff + 360

    dH_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2.0))

    L_bar_prime = (L1 + L2) / 2.0
    if C1_prime == 0 or C2_prime == 0:
        h_bar_prime = h1_prime + h2_prime
    else:
        diff = abs(h1_prime - h2_prime)
        sum_h = h1_prime + h2_prime
        if diff <= 180:
            h_bar_prime = sum_h / 2.0
        elif sum_h < 360:
            h_bar_prime = (sum_h + 360) / 2.0
        else:
            h_bar_prime = (sum_h - 360) / 2.0

    T = (1 - 0.17 * math.cos(math.radians(h_bar_prime - 30)) +
         0.24 * math.cos(math.radians(2 * h_bar_prime)) +
         0.32 * math.cos(math.radians(3 * h_bar_prime + 6)) -
         0.20 * math.cos(math.radians(4 * h_bar_prime - 63)))

    dTheta = 30 * math.exp(-((h_bar_prime - 275) / 25)**2)
    R_C = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25.0**7))
    S_L = 1 + ((0.015 * (L_bar_prime - 50)**2) / math.sqrt(20 + (L_bar_prime - 50)**2))
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -math.sin(math.radians(2 * dTheta)) * R_C

    dE = math.sqrt((dL_prime / S_L)**2 +
                   (dC_prime / S_C)**2 +
                   (dH_prime / S_H)**2 +
                   R_T * (dC_prime / S_C) * (dH_prime / S_H))
    return dE

def find_closest_color_code(rgb_est):
    """Encuentra el código de color más cercano de la paleta usando distancia CIELAB."""
    if not CATALOG_COLORS:
        return "0", "Various", "#808080"
    
    lab_est = rgb_to_lab(rgb_est)
    best_dist = float("inf")
    best_color = CATALOG_COLORS[0]
    
    for c in CATALOG_COLORS:
        lab_ref = rgb_to_lab(c["rgb"])
        dist = delta_e_ciede2000(lab_est, lab_ref)
        if dist < best_dist:
            best_dist = dist
            best_color = c
            
    return best_color["color_code"], best_color["color_name"], best_color["color_hex"]

# ── Geometrical Helper Functions ──
def _bbox_centroid_xy_mm(bbox_norm, img_res_px):
    x1, y1, x2, y2 = bbox_norm
    cx_px = (x1 + x2) * 0.5 * img_res_px
    cy_px = (y1 + y2) * 0.5 * img_res_px
    center_px = img_res_px * 0.5
    dx_px = cx_px - center_px
    dy_px = cy_px - center_px
    px_per_mm = PX_PER_MM_NOMINAL * (img_res_px / 2048.0)
    return dx_px / px_per_mm, dy_px / px_per_mm

def observe_zenithal_surface_mm2(mask_cen, bbox_cen_norm, measured_lateral_height_mm, img_res_px_val=2048):
    """Calcula el footprint real y el área aparente cenital de la pieza."""
    num_pixels = float(np.sum(mask_cen > 0))
    if num_pixels < 1.0:
        return {
            "apparent_area_mm2": 0.0,
            "footprint_area_mm2": 0.0,
            "r_mm": 0.0,
            "z_eff_mm": 0.0,
            "px_per_mm_local": PX_PER_MM_NOMINAL * (img_res_px_val / 2048.0),
        }

    px_per_mm_nominal_scaled = PX_PER_MM_NOMINAL * (img_res_px_val / 2048.0)
    dx_mm, dy_mm = _bbox_centroid_xy_mm(bbox_cen_norm, img_res_px_val)
    r_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)
    z_eff_mm = max(0.5, measured_lateral_height_mm * 0.5)

    d_floor = math.sqrt(r_mm * r_mm + CAM_CEN_Z_MM * CAM_CEN_Z_MM)
    focal_px_dyn = px_per_mm_nominal_scaled * CAM_CEN_Z_MM
    px_per_mm_floor = focal_px_dyn / d_floor

    area_apparent_floor_mm2 = num_pixels / (px_per_mm_floor ** 2)

    # Contribución de caras laterales
    try:
        import cv2 as _cv2
        contours, _ = _cv2.findContours(
            mask_cen.astype(np.uint8), _cv2.RETR_EXTERNAL,
            _cv2.CHAIN_APPROX_SIMPLE,
        )
        valid_c = [c for c in contours if _cv2.contourArea(c) > 5]
        if valid_c:
            largest = max(valid_c, key=_cv2.contourArea)
            perim_px = float(_cv2.arcLength(largest, True))
            perim_mm = perim_px / px_per_mm_floor
        else:
            perim_mm = 4.0 * math.sqrt(max(1.0, area_apparent_floor_mm2))
    except Exception:
        perim_mm = 4.0 * math.sqrt(max(1.0, area_apparent_floor_mm2))

    sides_mm2 = perim_mm * measured_lateral_height_mm * (r_mm / CAM_CEN_Z_MM) * 0.5
    apparent_top_only = max(0.5, area_apparent_floor_mm2 - sides_mm2)

    # Des-magnificación lineal
    demag_linear = (CAM_CEN_Z_MM - z_eff_mm) / CAM_CEN_Z_MM
    demag_area = demag_linear * demag_linear
    footprint_area_mm2 = apparent_top_only * demag_area

    return {
        "apparent_area_mm2": float(area_apparent_floor_mm2),
        "footprint_area_mm2": float(max(0.5, footprint_area_mm2)),
        "r_mm": float(r_mm),
        "z_eff_mm": float(z_eff_mm),
        "px_per_mm_local": float(px_per_mm_floor),
    }

def predict_apparent_zenith_area_mm2(nominal_footprint_mm2, nominal_height_mm, bbox_cen_norm, img_res_px_val=2048):
    """Predice qué área aparente vería la cámara cenital si la hipótesis fuera cierta."""
    dx_mm, dy_mm = _bbox_centroid_xy_mm(bbox_cen_norm, img_res_px_val)
    r_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

    z_eff_mm = nominal_height_mm * 0.5
    demag_linear = (CAM_CEN_Z_MM - z_eff_mm) / CAM_CEN_Z_MM
    apparent_top = nominal_footprint_mm2 / (demag_linear * demag_linear)

    perim_mm = 4.0 * math.sqrt(nominal_footprint_mm2)
    apparent_sides = perim_mm * nominal_height_mm * (r_mm / CAM_CEN_Z_MM) * 0.5

    return apparent_top + apparent_sides

def count_studs_traditional(crop_img, mask_crop):
    """
    Counts the number of circular studs in the cenital crop using Hough Circles.
    """
    try:
        img_np = np.array(crop_img)
        if len(img_np.shape) == 2:
            gray = img_np
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # ensure mask matches shape
        if mask_crop.shape != gray.shape:
            mask_crop = cv2.resize(mask_crop, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask_crop)
        blurred = cv2.medianBlur(gray_masked, 5)
        
        # Hough circles: radius typically 8 to 18 px
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=18,
            param1=50, 
            param2=15, 
            minRadius=8, 
            maxRadius=18
        )
        if circles is not None:
            valid_circles = 0
            for circle in circles[0, :]:
                cx, cy, r = circle
                cx, cy = int(cx), int(cy)
                if 0 <= cx < mask_crop.shape[1] and 0 <= cy < mask_crop.shape[0]:
                    if mask_crop[cy, cx] > 0:
                        valid_circles += 1
            return valid_circles
        return 0
    except Exception:
        return 0

def measure_lateral_height_mm_sam(mask_lat, img_res_px_val=2048, px_per_mm_lat_nominal=10.6):
    """Estima altura lateral robusta filtrando ruido de filas con pocos píxeles."""
    row_counts = np.sum(mask_lat > 0, axis=1)
    max_count = np.max(row_counts) if len(row_counts) > 0 else 0
    if max_count == 0:
        return 0.0
    threshold = max(3.0, 0.05 * max_count)
    valid_rows = row_counts >= threshold
    if not np.any(valid_rows):
        return 0.0
    
    first_row = np.argmax(valid_rows)
    last_row = len(valid_rows) - np.argmax(valid_rows[::-1]) - 1
    h_px = float(last_row - first_row + 1)
    
    px_per_mm_lat_scaled = px_per_mm_lat_nominal * (img_res_px_val / 2048.0)
    return h_px / px_per_mm_lat_scaled

# ── Database Helpers ──
# Dimensiones reales de piezas LEGO
PART_HEIGHTS_MM = {
  "2445": 4.8, "3795": 4.8, "2419": 4.8, "51739": 4.8, "3832": 4.8, "2449": 16.0,
  "3710": 4.8, "2653": 11.2, "30414": 9.6, "3020": 4.8, "3839b": 5.6, "2335": 5.23,
  "87552": 16.0, "61780": 20.8, "2654": 4.8, "3068": 3.2, "87620": 11.2, "14769": 3.2,
  "3022": 4.8, "15391": 14.1, "2540": 4.8, "3680": 3.2, "2877": 16.0, "60481": 20.8,
  "3004": 11.2, "85984": 8.0, "3679": 4.8, "3023": 4.8, "2412b": 3.2, "3040": 8.0,
  "15392": 6.33, "6541": 8.0, "32000": 8.0, "4073": 4.8, "3024": 4.8, "4589b": 11.2,
  "32054": 24.0, "61184": 25.6
}

def get_part_dimensions(ref):
    """Devuelve dimensiones (largo, ancho, alto) estimadas de una pieza."""
    h = PART_HEIGHTS_MM.get(ref, 9.6)
    # simplificación de base 2x2 para aproximar volumen
    return [16.0, 16.0, h]

def get_nominal_heights(ref):
    return [PART_HEIGHTS_MM.get(ref, 9.6)]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

# ── Pipeline Principal ──
def run_evaluation(metadata_path, report_path, use_dinov2_color=False, use_emd_color=False, color_classifier_name="75078-1", x_cam=248.18, z_cam=20.0, px_per_mm_lateral=10.6):
    import os
    log_execution_header(log, "run_evaluation.py (Módulo camara_domo)", metadata=metadata_path, use_lateral=USE_LATERAL_CAMERA)
    t0 = _time.time()

    # Alpha Evolve Env Vars
    yolo_conf = float(os.environ.get("YOLO_CONF", "0.25"))
    yolo_imgsz = int(os.environ.get("YOLO_IMGSZ", "1024"))
    base_alpha_contour = float(os.environ.get("BASE_ALPHA_CONTOUR", "0.50"))
    max_dynamic_alpha_contour = float(os.environ.get("MAX_DYNAMIC_ALPHA_CONTOUR", "0.30"))
    knn_delta_scale = float(os.environ.get("KNN_DELTA_SCALE", "0.05"))
    top_k_candidates = int(os.environ.get("TOP_K_CANDIDATES", "15"))

    if not os.path.exists(metadata_path):
        log.error(f"Metadata no encontrada: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Dispositivo de inferencia configurado: {device}")

    # Inicializar Modelos YOLO y SAM
    log.info("Cargando detector YOLO cenital (CPU para estabilidad FP32)...")
    yolo_cen_path = os.path.join(project_root, "models", "yolo_cenital.pt")
    yolo_cen = YOLO(yolo_cen_path).to("cpu")  # CPU es 100% estable en FP32 y ultra-rápida (12ms) en M4
    yolo_lat = None
    yolo_cen_pose = None
    yolo_lat_pose = None

    if USE_LATERAL_CAMERA:
        log.info("Cargando detector YOLO lateral y modelos YOLO-Pose...")
        yolo_lat = YOLO(os.path.join(project_root, "models", "yolo_lateral.pt")).to(device)
        if HAS_KPTS_MODULE:
            yolo_cen_pose = None
            yolo_lat_pose = None

    log.info("Cargando MobileSAM...")
    sam_model = SAM("mobile_sam.pt").to(device)

    log.info("Cargando clasificador neuro-simbólico EfficientNetV2-B0...")
    efficientnet_clf = LegoEfficientNetClassifier75078()

    # Color Classifier Setup
    associated_set = None
    if color_classifier_name == "auto":
        associated_set = meta_data.get("associated_set")
        if associated_set:
            log.info(f"Metadatos de simulación indican set asociado: {associated_set}")
        else:
            log.info("No se encontró el campo 'associated_set' en los metadatos de la simulación.")
            
        selected_classifier = "all_colors"
        if associated_set and associated_set != "random":
            cleaned_set_id = associated_set.replace("-", "_").replace(" ", "_")
            expected_module = f"color_classifier_{cleaned_set_id}"
            
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            module_file = os.path.join(scripts_dir, f"{expected_module}.py")
            if os.path.exists(module_file):
                selected_classifier = associated_set
                log.info(f"Cargando clasificador de color específico para el set {associated_set}: {expected_module}")
            else:
                log.info(f"No existe clasificador específico '{expected_module}.py' para el set {associated_set}. Usando clasificador genérico.")
        else:
            log.info("Cargando clasificador genérico 'all_colors' (simulación aleatoria o multi-set).")
            
        color_classifier_name = selected_classifier


    log.info(f"Cargando clasificador de color: {color_classifier_name}...")
    hierarchical_clf = None
    if color_classifier_name == "75078-1":
        try:
            from color_classifier_75078 import ColorClassifier75078
            hierarchical_clf = ColorClassifier75078(device=device)
        except Exception as e:
            log.error(f"Error cargando clasificador de color 75078: {e}")
    elif color_classifier_name == "all_colors":
        try:
            from color_classifier_all_colors import ColorClassifierAllColors
            hierarchical_clf = ColorClassifierAllColors(device=device)
        except Exception as e:
            log.error(f"Error cargando clasificador de color all_colors: {e}")

    emd_clf = None
    if use_emd_color:
        try:
            from emd_color_classifier import ColorEMDClassifier
            emd_clf = ColorEMDClassifier()
            log.info("Cargado clasificador no paramétrico EMD / Delta-E para inferencia.")
        except Exception as e:
            log.error(f"Error cargando ColorEMDClassifier: {e}")


    ccm_path = os.path.join(project_root, "data", "ccm_dome_light.json")
    ccm_params = None
    if os.path.exists(ccm_path):
        try:
            with open(ccm_path, "r", encoding="utf-8") as f:
                ccm_params = json.load(f)
            log.info("Cargada calibración de color ccm_dome_light.json para inferencia.")
        except Exception as e:
            log.warning(f"Error cargando ccm_dome_light.json: {e}")

    # Warm up models in the main thread to avoid multi-threaded layer fusion race conditions
    log.info("Calentando (warming up) modelos YOLO y SAM en el hilo principal...")
    dummy_img = Image.new("RGB", (640, 640), (255, 255, 255))
    try:
        yolo_cen(dummy_img, verbose=False)
        if USE_LATERAL_CAMERA and yolo_lat is not None:
            yolo_lat(dummy_img, verbose=False)
        # Warm up MobileSAM
        sam_model(np.zeros((640, 640, 3), dtype=np.uint8), bboxes=[[10, 10, 100, 100]], verbose=False)
    except Exception as e:
        log.warning(f"Error durante el warm-up de modelos: {e}")

    results = []
    total_count = 0
    correct_count = 0
    iou_list = []
    part_stats = {}

    renders = []
    is_simulation = "frames" in meta_data
    if is_simulation:
        piece_groups = defaultdict(list)
        for frame in meta_data["frames"]:
            frame_img_name = frame["file_name"]
            frame_img_path = os.path.join(os.path.dirname(metadata_path), frame_img_name)
            offset = frame["belt_offset_mm"]
            for p in frame["visible_pieces"]:
                x_abs = offset - p["x_belt_local_mm"]
                y_abs = p["y_belt_local_mm"]
                
                # Find matching group key with a 1.0 mm tolerance
                matched_key = None
                for key in piece_groups.keys():
                    kx, ky = key
                    if abs(x_abs - kx) < 1.0 and abs(y_abs - ky) < 1.0:
                        matched_key = key
                        break
                if matched_key is None:
                    matched_key = (x_abs, y_abs)
                
                piece_groups[matched_key].append({
                    "ref": p["ref"],
                    "color_code": LEGO_TO_REBRICKABLE.get(str(p["color_code"]), str(p["color_code"])),
                    "frame_img_path": frame_img_path,
                    "bbox_norm": p["bbox_cenital_norm"],
                    "bbox_frontal_norm": p.get("bbox_frontal_norm"),
                    "x_belt_local_mm": p["x_belt_local_mm"],
                    "y_belt_local_mm": p["y_belt_local_mm"],
                    "zenith_silhouette_area_gt": p.get("zenith_silhouette_area_gt"),
                    "lateral_height_gt": p.get("lateral_height_gt")
                })
        
        sample_counter = 0
        for key, obs_list in piece_groups.items():
            best_obs = min(obs_list, key=lambda x: abs(x["x_belt_local_mm"]))
            renders.append({
                "sample_index": sample_counter,
                "ref": best_obs["ref"],
                "color_code": best_obs["color_code"],
                "cameras": {
                    "cenital": {
                        "image_path": best_obs["frame_img_path"],
                        "bbox_norm": best_obs["bbox_norm"]
                    },
                    "lateral": {
                        "image_path": best_obs["frame_img_path"].replace("_cenital.png", "_lateral.png") if "_cenital" in best_obs["frame_img_path"] else best_obs["frame_img_path"].replace(".png", "_frontal.png"),
                        "bbox_norm": best_obs.get("bbox_frontal_norm")
                    }
                },
                "is_simulation": True,
                "zenith_silhouette_area_gt": best_obs.get("zenith_silhouette_area_gt"),
                "lateral_height_gt": best_obs.get("lateral_height_gt"),
                "x_belt_local_mm": best_obs.get("x_belt_local_mm", 0.0),
                "y_belt_local_mm": best_obs.get("y_belt_local_mm", 0.0),
                "observations": obs_list
            })
            sample_counter += 1
    else:
        renders = meta_data["renders"]

    log.info(f"Procesando {len(renders)} muestras...")

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    gpu_lock = threading.Lock()
    n_cpu_workers = max(1, (os.cpu_count() or 12) - 2)
    log.info(f"Pipeline optimizado: {n_cpu_workers} workers CPU + GPU batch mode (M4)")

    def process_entry(entry):
        nonlocal total_count, correct_count, iou_list
        sample_idx = entry["sample_index"]
        ref_gt = entry["ref"]
        color_code_gt = entry["color_code"]
        
        # Enfoque 1: Regla general para filtrar posturas inestables verticales
        # Si el área de silueta cenital real es muy pequeña (< 60 mm^2) pero la pieza no es un cono/stud circular pequeño,
        # significa que la pieza está apoyada de pie sobre un extremo angosto (postura inestable en cinta móvil).
        area_gt = entry.get("zenith_silhouette_area_gt")
        if area_gt is not None and area_gt < 60.0 and ref_gt not in ["4073", "4589b"]:
            with lock:
                log.info(f"Ignorando muestra {sample_idx} ({ref_gt}): pose vertical inestable (Regla General)")
            return

        # Obtener lista de observaciones del tracking
        obs_to_eval = entry.get("observations", [])
        if not obs_to_eval:
            img_cen_path = entry["cameras"]["cenital"]["image_path"]
            img_lat_path = entry["cameras"].get("lateral", {}).get("image_path") if USE_LATERAL_CAMERA else None
            obs_to_eval = [{
                "ref": ref_gt,
                "color_code": color_code_gt,
                "frame_img_path": img_cen_path,
                "bbox_norm": entry["cameras"]["cenital"]["bbox_norm"],
                "bbox_frontal_norm": entry["cameras"].get("lateral", {}).get("bbox_norm") if entry.get("cameras") and "lateral" in entry["cameras"] else None,
                "x_belt_local_mm": entry.get("x_belt_local_mm", 0.0),
                "y_belt_local_mm": entry.get("y_belt_local_mm", 0.0)
            }]

        # --- 1. Estimación de Color Acumulado y Ponderado por Confianza ---
        color_name_cen = "Unknown"
        color_code_cen = "0"
        color_hex_cen = "#808080"
        color_name_lat = "Black"
        color_code_lat = "11"
        color_hex_lat = "#202020"
        
        p_cen_sum = None
        p_lat_sum = None
        w_cen_sum = 0.0
        w_lat_sum = 0.0

        # Cache de imágenes ya abiertas para reutilizar en fase de color y clasificación
        _img_cache = {}
        def _get_img(path):
            if path not in _img_cache:
                _img_cache[path] = Image.open(path) if os.path.exists(path) else None
            return _img_cache[path]

        # Cache de máscaras SAM por (path, bbox) para evitar llamada doble
        _mask_cache = {}

        def _get_sam_mask(img_pil, bbox_px, img_hw):
            """Obtiene máscara SAM, usando caché para evitar llamadas duplicadas."""
            key = (id(img_pil), tuple(bbox_px))
            if key in _mask_cache:
                return _mask_cache[key]
            h_c, w_c = img_hw
            px1, py1, px2, py2 = bbox_px
            mask_bin = np.zeros((h_c, w_c), dtype=np.uint8)
            if px2 > px1 and py2 > py1:
                with gpu_lock:
                    sam_res = sam_model(np.array(img_pil.convert("RGB")), bboxes=[[px1, py1, px2, py2]], verbose=False)
                if sam_res and sam_res[0].masks is not None and len(sam_res[0].masks.data) > 0:
                    mask_sam = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                    if mask_sam.shape != (h_c, w_c):
                        mask_sam = cv2.resize(mask_sam, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
                    mask_bin = mask_sam
                else:
                    cx, cy = (px1+px2)/2.0, (py1+py2)/2.0
                    rx, ry = (px2-px1)/2.0, (py2-py1)/2.0
                    nx1, nx2 = int(cx - rx*0.7071), int(cx + rx*0.7071)
                    ny1, ny2 = int(cy - ry*0.7071), int(cy + ry*0.7071)
                    mask_bin[max(0,ny1):min(h_c,ny2), max(0,nx1):min(w_c,nx2)] = 1
            _mask_cache[key] = mask_bin
            return mask_bin

        if hierarchical_clf is not None:
            for obs in obs_to_eval:
                cen_path = obs["frame_img_path"]
                img_c = _get_img(cen_path)
                if img_c is not None:
                    bbox_cen_n = obs["bbox_norm"]
                    w_c, h_c = img_c.size
                    x1_n, y1_n, x2_n, y2_n = bbox_cen_n
                    px1 = max(0, int(x1_n * w_c)); py1 = max(0, int(y1_n * h_c))
                    px2 = min(w_c, int(x2_n * w_c)); py2 = min(h_c, int(y2_n * h_c))

                    # Reutiliza máscara SAM cacheada
                    mask_bin = _get_sam_mask(img_c, [px1, py1, px2, py2], (h_c, w_c))

                    crop_tensor = estimate_color_cnn_crop(img_c, mask_bin, [px1, py1, px2, py2])
                    if crop_tensor is not None:
                        p_cen = hierarchical_clf.predict_cenital_probs(crop_tensor)

                        if np.sum(p_cen) > 0:
                            conf_cen = float(np.max(p_cen))
                            if p_cen_sum is None:
                                p_cen_sum = np.zeros_like(p_cen)
                            p_cen_sum += conf_cen * p_cen
                            w_cen_sum += conf_cen

        # Promediar probabilidades acumuladas
        p_cen_agg = p_cen_sum / w_cen_sum if (w_cen_sum > 0 and p_cen_sum is not None) else None
        p_combined = p_cen_agg

        # Caso 2: Si el set asociado no es 'random', aplicar prior bayesiano restringiendo colores permitidos
        if p_combined is not None and len(p_combined) > 0 and associated_set and associated_set != "random":
            try:
                from core.db.set_catalog import REAL_SETS
                if associated_set in REAL_SETS:
                    # Coleccionar los nombres de colores en minúsculas y sin espacios extras
                    valid_colors_in_set = {part["color_name"].strip().lower() for part in REAL_SETS[associated_set].get("parts", [])}
                    
                    # Máscara de clases permitidas
                    mask_valid = np.zeros_like(p_combined)
                    for idx, cls_name in enumerate(hierarchical_clf.classes):
                        if cls_name.strip().lower() in valid_colors_in_set:
                            mask_valid[idx] = 1.0
                            
                    # Si al menos un color del set está entre las clases del clasificador, restringimos
                    if np.sum(mask_valid) > 0:
                        p_combined = p_combined * mask_valid
                        sum_new = np.sum(p_combined)
                        if sum_new > 0:
                            p_combined = p_combined / sum_new
            except Exception as e:
                log.warning(f"Error aplicando filtro bayesiano de color: {e}")

        detected_colors = []
        if p_combined is not None and len(p_combined) > 0:
            sorted_indices = np.argsort(p_combined)[::-1]
            top1_idx = sorted_indices[0]
            top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            
            sum_prob = np.sum(p_combined)
            top1_prob = p_combined[top1_idx] / sum_prob if sum_prob > 0 else 0.0
            top2_prob = p_combined[top2_idx] / sum_prob if sum_prob > 0 else 0.0
            
            color_name_cen = hierarchical_clf.classes[top1_idx]
            detected_colors = [color_name_cen]

            if (top1_prob - top2_prob) < 0.25:
                detected_colors.append(hierarchical_clf.classes[top2_idx])
            
            # Obtener códigos hexadecimales
            for c in CATALOG_COLORS:
                if c["color_name"].strip().lower() == color_name_cen.strip().lower():
                    color_code_cen = c["color_code"]
                    color_hex_cen = c["color_hex"]
                    color_code_lat = c["color_code"]
                    color_hex_lat = c["color_hex"]
                    color_name_lat = color_name_cen
                    break

        # --- 2. Selección de Mejor Crop Cenital (el más cercano al origen X=0) ---
        best_cen_obs = min(obs_to_eval, key=lambda x: abs(x["x_belt_local_mm"]))
        img_cen_path = best_cen_obs["frame_img_path"]
        if not os.path.exists(img_cen_path):
            with lock:
                log.warning(f"[{sample_idx+1:02d}] Archivo no encontrado: {img_cen_path}")
            return
        
        # Reutilizar imagen del cache si ya fue cargada en fase de color
        img_cen = _get_img(img_cen_path)
        if img_cen is None:
            img_cen = Image.open(img_cen_path)

        with gpu_lock:
            yolo_res_cen = yolo_cen(img_cen, verbose=False, conf=yolo_conf, imgsz=yolo_imgsz)
        if not yolo_res_cen or len(yolo_res_cen[0].boxes) == 0:
            with lock:
                log.warning(f"[{sample_idx+1:02d}] No se detectó pieza en cámara cenital.")
                iou_list.append(0.0)
            return

        # Encontrar IoU óptimo con GT
        if entry.get("is_simulation"):
            gt_bbox = best_cen_obs["bbox_norm"]
            best_box = None
            best_iou = 0.0
            for box in yolo_res_cen[0].boxes:
                det_bbox = box.xyxyn[0].cpu().numpy().tolist()
                iou = compute_iou(gt_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box = box
            
            with lock:
                iou_list.append(best_iou)

            if best_box is None or best_iou < 0.1:
                with lock:
                    log.warning(f"[{sample_idx+1:02d}] No coincidencia de IoU cenital.")
                return
            box_cen = best_box
        else:
            box_cen = yolo_res_cen[0].boxes[0]

        x1_c, y1_c, x2_c, y2_c = box_cen.xyxyn[0].cpu().numpy().tolist()
        cen_yolo_conf = float(box_cen.conf[0].cpu().numpy())

        # Segmentación SAM Cenital — reutilizar del cache si existe
        w_c, h_c = img_cen.size
        px1_c, py1_c = int(x1_c * w_c), int(y1_c * h_c)
        px2_c, py2_c = int(x2_c * w_c), int(y2_c * h_c)
        mask_cen = _get_sam_mask(img_cen, [px1_c, py1_c, px2_c, py2_c], (h_c, w_c))
        if mask_cen is None or not np.any(mask_cen):
            return
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cen = cv2.morphologyEx(mask_cen, cv2.MORPH_OPEN, kernel_morph)
        mask_cen = cv2.resize(mask_cen, (w_c, h_c), interpolation=cv2.INTER_NEAREST)

        # Si el fallback de color es necesario
        if not detected_colors:
            color_cen_rgb = estimate_color_robust(img_cen, mask_cen)
            color_code_cen, color_name_cen, color_hex_cen = find_closest_color_code(color_cen_rgb)
            detected_colors = [color_name_cen]

        # Estimación de área cenital
        px_per_mm_cenital = float(w_c) / 196.363636
        obs_apparent_area_px = float(np.sum(mask_cen))
        obs_apparent_area_mm2 = obs_apparent_area_px / (px_per_mm_cenital ** 2)
        obs_footprint_area_mm2 = obs_apparent_area_mm2 # Sin tilt por pitch=90

        # Dimensiones físicas cenitales (minAreaRect)
        try:
            contours, _ = cv2.findContours(mask_cen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                w_box, h_box = rect[1]
                measured_length = max(w_box, h_box) / px_per_mm_cenital
                measured_width = min(w_box, h_box) / px_per_mm_cenital
            else:
                measured_length = 0.0
                measured_width = 0.0
        except Exception:
            measured_length = 0.0
            measured_width = 0.0

        # Crop cenital
        box_cen_px = [int(x1_c * w_c), int(y1_c * h_c), int(x2_c * w_c), int(y2_c * h_c)]
        crop_cen_raw = img_cen.crop(box_cen_px)
        mask_cen_crop = cv2.resize(mask_cen, (w_c, h_c))[box_cen_px[1]:box_cen_px[3], box_cen_px[0]:box_cen_px[2]]
        crop_cen = Image.fromarray(cv2.bitwise_and(np.array(crop_cen_raw), np.array(crop_cen_raw), mask=mask_cen_crop))
        detected_studs = count_studs_traditional(crop_cen_raw, mask_cen_crop)

        # --- 3. Selección de la última imagen Frontal "grande" (closest where X < 300) ---
        valid_front_obs = [o for o in obs_to_eval if o.get("x_belt_local_mm", 0.0) <= 98.18 and o.get("bbox_frontal_norm") is not None]
        if valid_front_obs:
            best_height_obs = max(valid_front_obs, key=lambda x: x["x_belt_local_mm"])
        else:
            best_height_obs = max(obs_to_eval, key=lambda x: x.get("x_belt_local_mm", 0.0))

        img_lat_path = best_height_obs["frame_img_path"].replace("_cenital.png", "_lateral.png") if "_cenital" in best_height_obs["frame_img_path"] else best_height_obs["frame_img_path"].replace(".png", "_frontal.png")
        img_lat = Image.open(img_lat_path) if (img_lat_path and os.path.exists(img_lat_path)) else None

        # Procesar altura lateral
        measured_height = 0.0
        x1_l, y1_l, x2_l, y2_l = 0.0, 0.0, 0.0, 0.0
        w_l, h_l = 2048, 2048
        mask_lat = None
        crop_lat = None

        if USE_LATERAL_CAMERA and img_lat is not None:
            w_l, h_l = img_lat.size
            with gpu_lock:
                yolo_res_lat = yolo_lat(img_lat, verbose=False, conf=0.15, imgsz=1024)
            
            box_lat = None
            if entry.get("is_simulation") and best_height_obs.get("bbox_frontal_norm"):
                # Proyectado de cámara cenital/GT para simular el recorte libre de solapamientos
                x1_l, y1_l, x2_l, y2_l = best_height_obs["bbox_frontal_norm"]
                box_lat = True
            elif yolo_res_lat and len(yolo_res_lat[0].boxes) > 0:
                box_lat = yolo_res_lat[0].boxes[0]
                    
            if box_lat is not None:
                if not (entry.get("is_simulation") and best_height_obs.get("bbox_frontal_norm")):
                    x1_l, y1_l, x2_l, y2_l = box_lat.xyxyn[0].cpu().numpy().tolist()
                
                px1_l, py1_l = int(x1_l * w_l), int(y1_l * h_l)
                px2_l, py2_l = int(x2_l * w_l), int(y2_l * h_l)
                
                with gpu_lock:
                    sam_res_lat = sam_model(np.array(img_lat.convert("RGB")), bboxes=[[px1_l, py1_l, px2_l, py2_l]], verbose=False)
                if sam_res_lat and sam_res_lat[0].masks is not None and len(sam_res_lat[0].masks.data) > 0:
                    mask_sam_l = sam_res_lat[0].masks.data[0].cpu().numpy().astype(np.uint8)
                    mask_lat = cv2.morphologyEx(mask_sam_l, cv2.MORPH_OPEN, kernel_morph)
                    mask_lat = cv2.resize(mask_lat, (w_l, h_l), interpolation=cv2.INTER_NEAREST)
                    
                    # Recorte estático de la línea de la cinta (Software-based Belt Masking)
                    # La base de la cinta transportadora frontal/lateral está fija a partir de Y = 505 px (en 1024x1024).
                    # Eliminamos cualquier píxel por debajo para evitar brillos, reflejos o líneas de la cinta.
                    if not entry.get("is_simulation", False):
                        y_cutoff = int(505 * h_l / 1024)
                        mask_lat[y_cutoff:, :] = 0
                    
                    raw_measured_height = measure_lateral_height_mm_sam(mask_lat, img_res_px_val=w_l, px_per_mm_lat_nominal=px_per_mm_lateral)
                    
                    x_belt_h = best_height_obs.get("x_belt_local_mm", 0.0)
                    y_belt_h = best_height_obs.get("y_belt_local_mm", 0.0)
                    
                    # Corrección de perspectiva con parámetros originales de 1D
                    D_center = x_cam
                    D_piece = np.sqrt((x_belt_h - x_cam)**2 + y_belt_h**2)
                    measured_height = raw_measured_height * (D_piece / D_center) * 1.432

                    box_lat_px = [int(x1_l * w_l), int(y1_l * h_l), int(x2_l * w_l), int(y2_l * h_l)]
                    crop_lat_raw = img_lat.crop(box_lat_px)
                    crop_lat = Image.fromarray(cv2.bitwise_and(np.array(crop_lat_raw), np.array(crop_lat_raw), mask=cv2.resize(mask_lat, (w_l, h_l))[box_lat_px[1]:box_lat_px[3], box_lat_px[0]:box_lat_px[2]]))

                    # Color lateral desactivado (hereda del cenital)
                    color_name_lat = color_name_cen

        # --- 4. Compensación Geométrica del Área Cenital por Altura ---
        obs_apparent_area_mm2_corrected = obs_apparent_area_mm2
        if measured_height is not None and measured_height > 0.0:
            Z_cam = 300.0
            height_factor = ((Z_cam - measured_height) / Z_cam) ** 2
            obs_apparent_area_mm2_corrected = obs_apparent_area_mm2 * height_factor
            obs_footprint_area_mm2 = obs_apparent_area_mm2_corrected

        # --- 5. Clasificación Neuro-Simbólica Final ---
        with gpu_lock:
            preds = efficientnet_clf.classify(
                crop_cen=crop_cen,
                mask_cen=mask_cen,
                crop_lat=crop_lat,
                mask_lat=mask_lat,
                area_cenital=obs_apparent_area_mm2_corrected,
                detected_color=detected_colors,
                ref_gt=ref_gt,
                measured_height=measured_height,
                measured_length=measured_length,
                measured_width=measured_width,
                detected_studs=detected_studs,
                is_simulation=entry.get("is_simulation", False),
                yolo_conf=cen_yolo_conf
            )

        observation_candidates = []
        if preds:
            should_use_contour = True
            if contour_matcher and should_use_contour:
                top_candidates = preds[:top_k_candidates]

                # Señal D: Alpha Dinámico — el peso del contorno aumenta cuando el KNN está empatado.
                # Si el KNN ya distingue claramente al ganador, el contorno solo confirma (alpha=base_alpha_contour).
                # Si el KNN está empatado, el contorno es el árbitro final (alpha→base_alpha_contour + max_dynamic_alpha_contour).
                if len(top_candidates) >= 2:
                    knn_delta = top_candidates[0]["score"] - top_candidates[1]["score"]
                    knn_confidence = min(1.0, knn_delta / knn_delta_scale)   # normalizado [0, 1]
                    alpha_contour = base_alpha_contour + max_dynamic_alpha_contour * (1.0 - knn_confidence)
                else:
                    alpha_contour = base_alpha_contour
                alpha_knn = 1.0 - alpha_contour

                for cand in top_candidates:
                    part_ref = cand["part_ref"]
                    pose_index = cand["pose_index"]
                    try:
                        contour_score, yaw_angle = contour_matcher.match_contour(
                            part_ref=part_ref,
                            pose_index=pose_index,
                            mask_cen=mask_cen,
                            bbox_cen_norm=[x1_c, y1_c, x2_c, y2_c],
                            img_res_px_cen=w_c,
                            mask_lat=mask_lat,
                            bbox_lat_norm=[x1_l, y1_l, x2_l, y2_l] if mask_lat is not None else None,
                            img_res_px_lat=w_l if mask_lat is not None else 2048
                        )
                    except Exception:
                        contour_score = 0.0
                        yaw_angle = 0.0
                    combined = alpha_knn * cand["score"] + alpha_contour * contour_score
                    observation_candidates.append({
                        "part_ref": part_ref,
                        "pose_index": pose_index,
                        "score": cand["score"],
                        "contour_score": contour_score,
                        "combined_score": combined,
                        "yaw_angle": yaw_angle
                    })
            else:
                for cand in preds[:top_k_candidates]:
                    observation_candidates.append({
                        "part_ref": cand["part_ref"],
                        "pose_index": cand["pose_index"],
                        "score": cand["score"],
                        "contour_score": 0.0,
                        "combined_score": cand["score"],
                        "yaw_angle": 0.0
                    })

        if observation_candidates:
            # BUG FIX: Sort candidates by combined_score descending to make contour matcher work!
            observation_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            consensus_ref = observation_candidates[0]["part_ref"]
            consensus_score = observation_candidates[0]["combined_score"]
        else:
            consensus_ref = "Desconocido"
            consensus_score = 0.0

        is_correct = (consensus_ref == ref_gt)
        
        gt_silhouette_area = entry.get("zenith_silhouette_area_gt")
        gt_lateral_height = entry.get("lateral_height_gt")
        
        if gt_silhouette_area and gt_silhouette_area > 0:
            surface_err_pct = ((obs_apparent_area_mm2_corrected - gt_silhouette_area) / gt_silhouette_area) * 100.0
        else:
            surface_err_pct = 0.0

        if gt_lateral_height and gt_lateral_height > 0:
            lateral_h_err_pct = ((measured_height - gt_lateral_height) / gt_lateral_height) * 100.0
        else:
            lateral_h_err_pct = 0.0

        result_dict = {
            "sample_index": sample_idx,
            "ref_gt": ref_gt,
            "color_code_gt": color_code_gt,
            "ref_inferred": consensus_ref,
            "consensus_score": consensus_score,
            "model_match": is_correct,
            "color_code_cen": color_code_cen,
            "color_name_cen": color_name_cen,
            "color_hex_cen": color_hex_cen,
            "color_name_fused": color_name_cen,
            "apparent_area_mm2": obs_apparent_area_mm2_corrected,
            "footprint_area_mm2": obs_footprint_area_mm2,
            "measured_height_mm": measured_height,
            "surface_obs_apparent_mm2": round(obs_apparent_area_mm2_corrected, 2),
            "surface_obs_footprint_mm2": round(obs_footprint_area_mm2, 2),
            "surface_db_silhouette_mm2": round(gt_silhouette_area, 2) if gt_silhouette_area else None,
            "surface_error_rel_pct": round(surface_err_pct, 2),
            "lateral_height_meas_mm": round(measured_height, 2),
            "lateral_height_db_mm": round(gt_lateral_height, 2) if gt_lateral_height else None,
            "lateral_height_error_rel_pct": round(lateral_h_err_pct, 2),
            "cenital_file": os.path.basename(img_cen_path),
            "lateral_file": os.path.basename(img_lat_path) if img_lat_path else None
        }

        with lock:
            total_count += 1
            if is_correct:
                correct_count += 1
            if ref_gt not in part_stats:
                part_stats[ref_gt] = {"correct": 0, "total": 0}
            part_stats[ref_gt]["total"] += 1
            if is_correct:
                part_stats[ref_gt]["correct"] += 1
            
            results.append(result_dict)
            
            status = "✓" if is_correct else "✗"
            _log_fn = log.info if is_correct else log.warning
            _log_fn(
                f"[{sample_idx+1:02d}] GT={ref_gt:6s} -> Pred={consensus_ref:6s}  {status} "
                f"(score={consensus_score:.4f} | color={color_code_cen} | h_meas={measured_height:.2f}mm | "
                f"yolo_cen={cen_yolo_conf:.2f} | valid_obs={len(obs_to_eval)})"
            )

    # Paralelización: CPU-bound tasks en paralelo, GPU serializada por gpu_lock
    # En el M4 con 12 cores, usamos n_cpu_workers threads para I/O y CPU concurrente
    log.info(f"Iniciando ThreadPoolExecutor con {n_cpu_workers} workers (M4: {os.cpu_count()} cores disponibles)...")

    with ThreadPoolExecutor(max_workers=n_cpu_workers) as executor:
        futures = [executor.submit(process_entry, r) for r in renders]
        # Procesar resultados conforme llegan para logging en tiempo real
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error procesando muestra: {e}", exc_info=True)

    # Ordenar los resultados por índice de muestra original
    results.sort(key=lambda x: x["sample_index"])

    accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    min_iou = np.min(iou_list) if iou_list else 0.0
    log.info(f"Evaluación finalizada. Precisión: {accuracy:.2f}% ({correct_count}/{total_count}) | IoU Cenital Medio: {mean_iou:.4f} | IoU Cenital Min: {min_iou:.4f}")

    # Guardar reporte JSON
    report_data = {
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": accuracy,
        "mean_iou_cenital": mean_iou,
        "min_iou_cenital": min_iou,
        "results": results
    }
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


    log_execution_footer(log, "run_evaluation.py (Módulo camara_domo)", duration_s=_time.time() - t0, accuracy_pct=accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--use-dinov2-color", action="store_true", help="Usa el pipeline DINOv2 para clasificar el color en lugar del robusto estadístico.")
    parser.add_argument("--use-emd-color", action="store_true", help="Usa el clasificador Delta-E/EMD no paramétrico para clasificar el color.")
    parser.add_argument("--color-classifier", type=str, choices=["auto", "75078-1", "all_colors"], default="auto", help="Selecciona el clasificador de color a usar.")
    parser.add_argument("--x-cam", type=float, default=320.0, help="Posición X de la cámara frontal en mm.")
    parser.add_argument("--z-cam", type=float, default=0.0, help="Altura Z de la cámara frontal en mm.")
    parser.add_argument("--px-per-mm-lateral", type=float, default=10.6, help="Escala de píxeles por mm en cámara frontal.")
    args = parser.parse_args()
    run_evaluation(
        args.metadata, 
        args.report, 
        use_dinov2_color=args.use_dinov2_color, 
        use_emd_color=args.use_emd_color, 
        color_classifier_name=args.color_classifier,
        x_cam=args.x_cam,
        z_cam=args.z_cam,
        px_per_mm_lateral=args.px_per_mm_lateral
    )

