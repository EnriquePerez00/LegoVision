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
from scripts.efficientnet_classifier import LegoEfficientNetClassifier

# Cargar ContourMatcher para recomposición por contornos
try:
    from contour_matcher import ContourMatcher
    contour_matcher = ContourMatcher()
    log.info("ContourMatcher para recomposición 3D cargado con éxito.")
except Exception as e:
    contour_matcher = None
    log.warning(f"No se pudo cargar ContourMatcher: {e}")

# Cargar configuración de activación de cámara lateral
USE_LATERAL_CAMERA = getattr(cfg.inference, "camara_lateral", False)

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
PX_PER_MM_LATERAL = float(cfg.inference.calibration.px_per_mm_lateral)

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

def estimate_color_mlp_features(img, mask_binary):
    """Extrae las 12 características estadísticas (Lab/HSV) para el MLP de color."""
    img_arr = np.array(img.convert("RGB"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(mask_binary, kernel, iterations=1)
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask_binary
    mask_bool = (mask_to_use > 0)

    if not np.any(mask_bool):
        return None

    pixels_rgb = img_arr[mask_bool]
    hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv_img[mask_bool]

    non_specular_mask = (pixels_hsv[:, 1] >= 25) | (pixels_hsv[:, 2] < 230)
    if np.any(non_specular_mask):
        pixels_rgb_filt = pixels_rgb[non_specular_mask]
        pixels_hsv_filt = pixels_hsv[non_specular_mask]
    else:
        pixels_rgb_filt = pixels_rgb
        pixels_hsv_filt = pixels_hsv

    # Convertir RGB a Lab
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

    mean_lab = pixels_lab.mean(axis=0)
    std_lab = pixels_lab.std(axis=0)
    
    mean_hsv = pixels_hsv_filt.mean(axis=0)
    std_hsv = pixels_hsv_filt.std(axis=0)

    return np.array([
        mean_lab[0], std_lab[0],
        mean_lab[1], std_lab[1],
        mean_lab[2], std_lab[2],
        mean_hsv[0], std_hsv[0],
        mean_hsv[1], std_hsv[1],
        mean_hsv[2], std_hsv[2]
    ], dtype=np.float32)

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

def find_closest_color_code(rgb_est):
    """Encuentra el código de color más cercano de la paleta usando distancia CIELAB."""
    if not CATALOG_COLORS:
        return "0", "Various", "#808080"
    
    lab_est = rgb_to_lab(rgb_est)
    best_dist = float("inf")
    best_color = CATALOG_COLORS[0]
    
    for c in CATALOG_COLORS:
        lab_ref = rgb_to_lab(c["rgb"])
        dL = lab_est[0] - lab_ref[0]
        da = lab_est[1] - lab_ref[1]
        db = lab_est[2] - lab_ref[2]
        dist = np.sqrt(0.2 * (dL ** 2) + (da ** 2) + (db ** 2))
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

def measure_lateral_height_mm_sam(mask_lat, img_res_px_val=2048):
    """Estima altura lateral básica usando la caja delimitadora de SAM."""
    rows = np.any(mask_lat > 0, axis=1)
    if not np.any(rows):
        return 0.0
    h_px = float(len(rows) - np.argmax(rows[::-1]) - np.argmax(rows))
    px_per_mm_lat_scaled = PX_PER_MM_LATERAL * (img_res_px_val / 2048.0)
    return h_px / px_per_mm_lat_scaled

# ── Database Helpers ──
# Dimensiones reales de piezas LEGO
PART_HEIGHTS_MM = {
    "3001": 9.6, "3002": 9.6, "3003": 9.6, "3004": 9.6, "3005": 9.6, "3010": 9.6, "3622": 9.6,
    "3020": 3.2, "3021": 3.2, "3022": 3.2, "3023": 3.2, "3024": 3.2, "3710": 3.2, "3666": 3.2,
    "3795": 3.2, "2420": 3.2, "2431": 3.2, "3068": 3.2, "3069": 3.2, "6636": 3.2, "4162": 3.2,
    "4032": 3.2, "6141": 3.2, "98138": 3.2, "2412b": 3.2, "15573": 3.2, "15068": 9.6, "15391": 6.4,
    "15392": 6.4, "32054": 9.6, "85984": 3.2, "32000": 9.6, "3037": 11.2, "3039": 11.2,
    "2445": 3.2, "51739": 3.2, "60481": 9.6, "6541": 9.6, "87620": 9.6, "3832": 3.2, "3839b": 3.2,
    "4070": 9.6, "2449": 9.6, "2540": 3.2
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
def run_evaluation(metadata_path, report_path, use_dinov2_color=False):
    log_execution_header(log, "run_evaluation.py (Módulo camara_domo)", metadata=metadata_path, use_lateral=USE_LATERAL_CAMERA)
    t0 = _time.time()

    if not os.path.exists(metadata_path):
        log.error(f"Metadata no encontrada: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Dispositivo de inferencia configurado: {device}")

    # Inicializar Modelos YOLO y SAM
    log.info("Cargando detector YOLO cenital...")
    yolo_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital.pt")).to(device)
    yolo_lat = None
    yolo_cen_pose = None
    yolo_lat_pose = None

    if USE_LATERAL_CAMERA:
        log.info("Cargando detector YOLO lateral y modelos YOLO-Pose...")
        yolo_lat = YOLO(os.path.join(project_root, "models", "yolo_lateral.pt")).to(device)
        if HAS_KPTS_MODULE:
            yolo_cen_pose = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
            lat_pose_path = os.path.join(project_root, "models", "yolo_lateral_pose.pt")
            if not os.path.exists(lat_pose_path):
                lat_pose_path = os.path.join(project_root, "models", "yolo_frontal_pose.pt")
            yolo_lat_pose = YOLO(lat_pose_path).to(device)

    log.info("Cargando MobileSAM...")
    sam_model = SAM("mobile_sam.pt").to(device)

    log.info("Cargando clasificador neuro-simbólico EfficientNetV2-B0...")
    efficientnet_clf = LegoEfficientNetClassifier()

    # Color Hierarchical Classifier Setup
    log.info("Cargando clasificador jerárquico de color...")
    try:
        from hierarchical_color_classifier import HierarchicalColorClassifier
        hierarchical_clf = HierarchicalColorClassifier(device=device)
    except Exception as e:
        log.error(f"Error cargando clasificador jerárquico de color: {e}")
        hierarchical_clf = None




    results = []
    total_count = 0
    correct_count = 0
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
                x_abs = p["x_belt_local_mm"] + offset
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
                    "color_code": p["color_code"],
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
                "lateral_height_gt": best_obs.get("lateral_height_gt")
            })
            sample_counter += 1
    else:
        renders = meta_data["renders"]

    log.info(f"Procesando {len(renders)} muestras...")

    for idx, entry in enumerate(renders[58:80]):
        sample_idx = entry["sample_index"]
        ref_gt = entry["ref"]
        color_code_gt = entry["color_code"]
        
        # Cargar imágenes cenital y lateral
        img_cen_path = entry["cameras"]["cenital"]["image_path"]
        img_lat_path = entry["cameras"].get("lateral", {}).get("image_path") if USE_LATERAL_CAMERA else None
        
        if not os.path.exists(img_cen_path):
            log.warning(f"Ignorando sample {sample_idx}, archivo no encontrado: {img_cen_path}")
            continue

        img_cen = Image.open(img_cen_path)
        img_lat = Image.open(img_lat_path) if (img_lat_path and os.path.exists(img_lat_path)) else None

        # --- 1. Detección YOLO Cenital ---
        yolo_res_cen = yolo_cen(img_cen, verbose=False, conf=0.25)
        if not yolo_res_cen or len(yolo_res_cen[0].boxes) == 0:
            log.warning(f"[{sample_idx+1:02d}] No se detectó pieza en cámara cenital.")
            continue
        
        # BBox cenital
        if entry.get("is_simulation"):
            gt_bbox = entry["cameras"]["cenital"]["bbox_norm"]
            best_box = None
            best_iou = 0.0
            for box in yolo_res_cen[0].boxes:
                det_bbox = box.xyxyn[0].cpu().numpy().tolist()
                iou = compute_iou(gt_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_box = box
            
            if best_box is None or best_iou < 0.1:
                log.warning(f"[{sample_idx+1:02d}] No se encontró detección YOLO coincidente para GT ref={ref_gt} (Best IoU={best_iou:.2f})")
                continue
            box_cen = best_box
        else:
            box_cen = yolo_res_cen[0].boxes[0]

        x1_c, y1_c, x2_c, y2_c = box_cen.xyxyn[0].cpu().numpy().tolist()
        cen_yolo_conf = float(box_cen.conf[0].cpu().numpy())

        # --- 2. Segmentación SAM Cenital ---
        w_c, h_c = img_cen.size
        px1_c, py1_c = int(x1_c * w_c), int(y1_c * h_c)
        px2_c, py2_c = int(x2_c * w_c), int(y2_c * h_c)
        
        sam_res_cen = sam_model(np.array(img_cen.convert("RGB")), bboxes=[[px1_c, py1_c, px2_c, py2_c]], verbose=False)
        if not sam_res_cen or sam_res_cen[0].masks is None:
            log.warning(f"[{sample_idx+1:02d}] Falló segmentación SAM cenital.")
            continue
        mask_cen = sam_res_cen[0].masks.data[0].cpu().numpy().astype(np.uint8)

        # --- 3. Estimación de Color Cenital ---
        color_name_cen = "Unknown"
        use_hierarchical = (hierarchical_clf is not None and hierarchical_clf.cen_ready)
        if use_hierarchical:
            try:
                feat = estimate_color_mlp_features(img_cen, mask_cen)
                if feat is not None:
                    p_cen = hierarchical_clf.predict_cenital_probs(feat)
                    if np.sum(p_cen) > 0:
                        pred_idx = np.argmax(p_cen)
                        color_name_cen = hierarchical_clf.all_classes[pred_idx]
            except Exception as e:
                log.warning(f"Error en inferencia jerárquica cenital de color: {e}")
                use_hierarchical = False

        if not use_hierarchical or color_name_cen == "Unknown":
            color_cen_rgb = estimate_color_robust(img_cen, mask_cen)
            color_code_cen, color_name_cen, color_hex_cen = find_closest_color_code(color_cen_rgb)
        else:
            color_code_cen = "0"
            color_hex_cen = "#808080"
            for c in CATALOG_COLORS:
                if c["color_name"].strip().lower() == color_name_cen.strip().lower():
                    color_code_cen = c["color_code"]
                    color_hex_cen = c["color_hex"]
                    break

        # --- 4. Medición / Inferencia de Altura Lateral (Si está activa) ---
        measured_height = 9.6  # Fallback por defecto si no hay cámara lateral
        lat_yolo_conf = 0.0
        mask_lat = None
        color_code_lat, color_name_lat, color_hex_lat = "0", "Unknown", "#808080"

        if USE_LATERAL_CAMERA and img_lat is not None:
            # 1. Project horizontal range of the cenital box (box_cen)
            c_x = (x1_c + x2_c) / 2.0
            c_y = (y1_c + y2_c) / 2.0
            x1_proj = min(-0.958605 * y2_c + 0.979126, -0.958605 * y1_c + 0.979126)
            x2_proj = max(-0.958605 * y2_c + 0.979126, -0.958605 * y1_c + 0.979126)
            
            # Predict vertical range using height
            h_gt = entry.get("lateral_height_gt") or 9.6 # catalog height fallback
            y2_proj = 0.751817 * c_x - 0.00282 * c_y + 0.18565
            y1_proj = 0.753156 * c_x + 0.002287 * c_y - 0.002981 * h_gt + 0.127686
            
            # YOLO lateral
            yolo_res_lat = yolo_lat(img_lat, verbose=False, conf=0.15) # lower conf to allow matching
            best_yolo_box = None
            best_yolo_iou = 0.0
            lat_yolo_conf = 0.0
            
            if yolo_res_lat and len(yolo_res_lat[0].boxes) > 0:
                for box in yolo_res_lat[0].boxes:
                    det_bbox = box.xyxyn[0].cpu().numpy().tolist()
                    det_x1, det_y1, det_x2, det_y2 = det_bbox
                    
                    # Horizontal overlap match (Improvement 3 & 4)
                    horiz_overlap = max(0.0, min(x2_proj, det_x2) - max(x1_proj, det_x1))
                    horiz_union = max(1e-8, (x2_proj - x1_proj) + (det_x2 - det_x1) - horiz_overlap)
                    horiz_iou = horiz_overlap / horiz_union
                    
                    if horiz_iou >= 0.25:
                        # Compute overlap with target region
                        iou = compute_iou([x1_proj, y1_proj, x2_proj, y2_proj], det_bbox)
                        if iou > best_yolo_iou:
                            best_yolo_iou = iou
                            best_yolo_box = box
                            
            if best_yolo_box is not None:
                # Use YOLO's precise boundaries
                x1_l, y1_l, x2_l, y2_l = best_yolo_box.xyxyn[0].cpu().numpy().tolist()
                lat_yolo_conf = float(best_yolo_box.conf[0].cpu().numpy())
                # Refine/clip horizontal boundaries to projected coordinates to reduce edge noise
                x1_l = max(x1_l, x1_proj - 0.03)
                x2_l = min(x2_l, x2_proj + 0.03)
            else:
                # Fallback to pure geometric projection (Improvement 2)
                x1_l, y1_l, x2_l, y2_l = x1_proj, y1_proj, x2_proj, y2_proj
                lat_yolo_conf = 0.0
                
            # SAM lateral
            w_l, h_l = img_lat.size
            px1_l, py1_l = int(max(0.0, x1_l) * w_l), int(max(0.0, y1_l) * h_l)
            px2_l, py2_l = int(min(1.0, x2_l) * w_l), int(min(1.0, y2_l) * h_l)
            sam_res_lat = sam_model(np.array(img_lat.convert("RGB")), bboxes=[[px1_l, py1_l, px2_l, py2_l]], verbose=False)
            
            if sam_res_lat and sam_res_lat[0].masks is not None:
                mask_lat = sam_res_lat[0].masks.data[0].cpu().numpy().astype(np.uint8)
                measured_height = measure_lateral_height_mm_sam(mask_lat, img_res_px_val=w_l)
                
                # Intentar Triangulación Keypoints 3D
                if HAS_KPTS_MODULE and yolo_cen_pose is not None and yolo_lat_pose is not None:
                    try:
                        kp_cen = extract_yolo_pose_keypoints(yolo_cen_pose, img_cen, conf=0.20)
                        kp_lat = extract_yolo_pose_keypoints(yolo_lat_pose, img_lat, conf=0.20)
                        if kp_cen is not None and kp_lat is not None:
                            kpts_obs = kpts_observer_fn(kp_cen, kp_lat, conf_min=0.20)
                            if kpts_obs.get("n_valid", 0) >= 4:
                                h_kpts = float(kpts_obs.get("lateral_height_mm", 0.0))
                                if h_kpts > 0:
                                    measured_height = h_kpts
                    except Exception:
                        pass
                
                # Estimación de Color Lateral (con Fusión Jerárquica Multivista)
                color_name_lat = "Unknown"
                use_hierarchical_lat = (hierarchical_clf is not None and hierarchical_clf.lat_ready)
                if use_hierarchical_lat:
                    try:
                        feat_lat = estimate_color_mlp_features(img_lat, mask_lat)
                        feat_cen = estimate_color_mlp_features(img_cen, mask_cen)
                        if feat_lat is not None:
                            fused_color = hierarchical_clf.predict_fused_color(feat_cen, feat_lat)
                            color_name_lat = fused_color
                            color_name_cen = fused_color
                            
                            # Sincronizar códigos y colores cenitales
                            for c in CATALOG_COLORS:
                                if c["color_name"].strip().lower() == fused_color.strip().lower():
                                    color_code_cen = c["color_code"]
                                    color_hex_cen = c["color_hex"]
                                    break
                    except Exception as e:
                        log.warning(f"Error en inferencia jerárquica lateral/fusión de color: {e}")
                        use_hierarchical_lat = False

                if not use_hierarchical_lat or color_name_lat == "Unknown":
                    color_lat_rgb = estimate_color_robust(img_lat, mask_lat)
                    color_code_lat, color_name_lat, color_hex_lat = find_closest_color_code(color_lat_rgb)
                else:
                    color_code_lat = "0"
                    color_hex_lat = "#808080"
                    for c in CATALOG_COLORS:
                        if c["color_name"].strip().lower() == color_name_lat.strip().lower():
                            color_code_lat = c["color_code"]
                            color_hex_lat = c["color_hex"]
                            break

        # --- 5. Cálculo de Área Cenital Aparente Observada ---
        zen_obs = observe_zenithal_surface_mm2(
            mask_cen, [x1_c, y1_c, x2_c, y2_c],
            measured_lateral_height_mm=measured_height,
            img_res_px_val=w_c
        )
        obs_apparent_area_mm2 = zen_obs["apparent_area_mm2"]
        obs_footprint_area_mm2 = zen_obs["footprint_area_mm2"]

        # --- 6. Inferencia Neuro-Simbólica con EfficientNetV2-B0 ---
        # Tight crop cenital using SAM mask and black background masking
        img_cen_np = np.array(img_cen.convert("RGB"))
        img_cen_np[mask_cen == 0] = [0, 0, 0]
        img_cen_masked = Image.fromarray(img_cen_np)
        
        ys_c, xs_c = np.where(mask_cen > 0)
        if len(ys_c) > 0:
            px1_c_tight, py1_c_tight, px2_c_tight, py2_c_tight = int(np.min(xs_c)), int(np.min(ys_c)), int(np.max(xs_c)), int(np.max(ys_c))
            crop_cen = img_cen_masked.crop((px1_c_tight, py1_c_tight, px2_c_tight, py2_c_tight))
        else:
            px1_c, py1_c = int(x1_c * w_c), int(y1_c * h_c)
            px2_c, py2_c = int(x2_c * w_c), int(y2_c * h_c)
            crop_cen = img_cen_masked.crop((px1_c, py1_c, px2_c, py2_c))
        
        # Crop lateral (si la cámara lateral está activa y la imagen existe)
        crop_lat = None
        if USE_LATERAL_CAMERA and img_lat is not None and mask_lat is not None:
            w_l, h_l = img_lat.size
            img_lat_np = np.array(img_lat.convert("RGB"))
            img_lat_np[mask_lat == 0] = [0, 0, 0]
            img_lat_masked = Image.fromarray(img_lat_np)
            
            ys_l, xs_l = np.where(mask_lat > 0)
            if len(ys_l) > 0:
                px1_l_tight, py1_l_tight, px2_l_tight, py2_l_tight = int(np.min(xs_l)), int(np.min(ys_l)), int(np.max(xs_l)), int(np.max(ys_l))
                crop_lat = img_lat_masked.crop((px1_l_tight, py1_l_tight, px2_l_tight, py2_l_tight))
            else:
                px1_l, py1_l = int(x1_l * w_l), int(y1_l * h_l)
                px2_l, py2_l = int(x2_l * w_l), int(y2_l * h_l)
                crop_lat = img_lat_masked.crop((px1_l, py1_l, px2_l, py2_l))
            
        preds = efficientnet_clf.classify(
            crop_cen=crop_cen,
            mask_cen=mask_cen,
            crop_lat=crop_lat,
            mask_lat=mask_lat,
            area_cenital=obs_apparent_area_mm2
        )
        
        # Mocks para variables de log remanentes
        valid_by_color = []
        valid_by_surface = []
        
        if preds:
            # Evolución Técnica: Desempate Geométrico por Contornos
            if contour_matcher:
                # Evaluar match_contour para el top-3 de candidatos
                top_candidates = preds[:3]
                candidates_with_contour = []
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
                    except Exception as e:
                        log.warning(f"Error en match_contour para {part_ref} pose {pose_index}: {e}")
                        contour_score = 0.0
                        yaw_angle = 0.0
                        
                    # Fusión: 40% EfficientNet score + 60% Contour score
                    # Le damos mayor peso al contorno porque es una validación dimensional física directa
                    combined = 0.4 * cand["score"] + 0.6 * contour_score
                    candidates_with_contour.append({
                        "part_ref": part_ref,
                        "pose_index": pose_index,
                        "score": cand["score"],
                        "contour_score": contour_score,
                        "combined_score": combined,
                        "yaw_angle": yaw_angle
                    })
                
                # Ordenar por el score combinado de mayor a menor
                candidates_with_contour.sort(key=lambda x: x["combined_score"], reverse=True)
                
                # Reportar si cambió el primer candidato (desempate exitoso)
                orig_top = preds[0]["part_ref"]
                new_top = candidates_with_contour[0]["part_ref"]
                if orig_top != new_top:
                    log.info(f"  [Desempate Contorno] Cambió predicción: {orig_top} -> {new_top} "
                             f"(Contour IoU: {candidates_with_contour[0]['contour_score']:.4f} vs "
                             f"{next((c['contour_score'] for c in candidates_with_contour if c['part_ref'] == orig_top), 0.0):.4f})")
                
                consensus_ref = new_top
                consensus_score = candidates_with_contour[0]["combined_score"]
            else:
                consensus_ref = preds[0]["part_ref"]
                consensus_score = preds[0]["score"]
        else:
            consensus_ref = "Desconocido"
            consensus_score = 0.0

        is_correct = (consensus_ref == ref_gt)
        total_count += 1
        if is_correct:
            correct_count += 1

        if ref_gt not in part_stats:
            part_stats[ref_gt] = {"correct": 0, "total": 0}
        part_stats[ref_gt]["total"] += 1
        if is_correct:
            part_stats[ref_gt]["correct"] += 1

        status = "✓" if is_correct else "✗"
        _log_fn = log.info if is_correct else log.warning
        _log_fn(
            f"[{sample_idx+1:02d}] GT={ref_gt:6s} -> Pred={consensus_ref:6s}  {status} "
            f"(score={consensus_score:.4f} | color={color_code_cen} | h_meas={measured_height:.2f}mm | "
            f"yolo_cen={cen_yolo_conf:.2f} | valid_color={len(valid_by_color)} | valid_surf={len(valid_by_surface)})"
        )

        gt_silhouette_area = entry.get("zenith_silhouette_area_gt") or entry.get("zenith_observable_area_gt")
        gt_lateral_height = entry.get("lateral_height_gt") or entry.get("effective_height_gt")
        
        if gt_silhouette_area and gt_silhouette_area > 0:
            surface_err_pct = ((obs_apparent_area_mm2 - gt_silhouette_area) / gt_silhouette_area) * 100.0
        else:
            surface_err_pct = 0.0

        if gt_lateral_height and gt_lateral_height > 0:
            lateral_h_err_pct = ((measured_height - gt_lateral_height) / gt_lateral_height) * 100.0
        else:
            lateral_h_err_pct = 0.0

        results.append({
            "sample_index": sample_idx,
            "ref_gt": ref_gt,
            "color_code_gt": color_code_gt,
            "ref_inferred": consensus_ref,
            "consensus_score": consensus_score,
            "model_match": is_correct,
            "color_code_cen": color_code_cen,
            "color_name_cen": color_name_cen,
            "color_hex_cen": color_hex_cen,
            "color_code_lat": color_code_lat,
            "color_name_lat": color_name_lat,
            "color_hex_lat": color_hex_lat,
            "apparent_area_mm2": obs_apparent_area_mm2,
            "footprint_area_mm2": obs_footprint_area_mm2,
            "measured_height_mm": measured_height,
            "surface_obs_apparent_mm2": round(obs_apparent_area_mm2, 2),
            "surface_obs_footprint_mm2": round(obs_footprint_area_mm2, 2),
            "surface_db_silhouette_mm2": round(gt_silhouette_area, 2) if gt_silhouette_area else None,
            "surface_error_rel_pct": round(surface_err_pct, 2),
            "lateral_height_meas_mm": round(measured_height, 2),
            "lateral_height_db_mm": round(gt_lateral_height, 2) if gt_lateral_height else None,
            "lateral_height_error_rel_pct": round(lateral_h_err_pct, 2),
            "cenital_file": os.path.basename(img_cen_path),
            "lateral_file": os.path.basename(img_lat_path) if img_lat_path else None
        })

    accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    log.info(f"Evaluación finalizada. Precisión: {accuracy:.2f}% ({correct_count}/{total_count})")

    # Guardar reporte JSON
    report_data = {
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": accuracy,
        "results": results
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    log_execution_footer(log, "run_evaluation.py (Módulo camara_domo)", duration_s=_time.time() - t0, accuracy_pct=accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    parser.add_argument("--use-dinov2-color", action="store_true", help="Usa el pipeline DINOv2 para clasificar el color en lugar del robusto estadístico.")
    args = parser.parse_args()
    run_evaluation(args.metadata, args.report, use_dinov2_color=args.use_dinov2_color)
