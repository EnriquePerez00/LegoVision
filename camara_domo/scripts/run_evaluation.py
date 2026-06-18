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
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO, SAM

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from config_loader import cfg
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("pipeline")

from database.set_catalog import REAL_SETS
from scripts.efficientnet_classifier import LegoEfficientNetClassifier

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

def estimate_color_dual(img, mask_binary):
    """Estima el color promedio del objeto (completo + segmentado)."""
    img_arr = np.array(img.convert("RGB"))
    mask_bool = (mask_binary > 0)
    if not np.any(mask_bool):
        return [128.0, 128.0, 128.0]
    
    # 1. Promedio segmentado
    pixels = img_arr[mask_bool]
    mean_seg = pixels.mean(axis=0)
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
        dist = np.linalg.norm(lab_est - lab_ref)
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
def run_evaluation(metadata_path, report_path):
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
            yolo_lat_pose = YOLO(os.path.join(project_root, "models", "yolo_lateral_pose.pt")).to(device)

    log.info("Cargando MobileSAM...")
    sam_model = SAM("mobile_sam.pt").to(device)

    log.info("Cargando clasificador neuro-simbólico EfficientNetV2-B0...")
    efficientnet_clf = LegoEfficientNetClassifier()

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
                        "image_path": best_obs["frame_img_path"].replace(".png", "_frontal.png"),
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

    for idx, entry in enumerate(renders):
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
        color_cen_rgb = estimate_color_dual(img_cen, mask_cen)
        color_code_cen, color_name_cen, color_hex_cen = find_closest_color_code(color_cen_rgb)

        # --- 4. Medición / Inferencia de Altura Lateral (Si está activa) ---
        measured_height = 9.6  # Fallback por defecto si no hay cámara lateral
        lat_yolo_conf = 0.0
        mask_lat = None

        if USE_LATERAL_CAMERA and img_lat is not None:
            # YOLO lateral
            yolo_res_lat = yolo_lat(img_lat, verbose=False, conf=0.25)
            if yolo_res_lat and len(yolo_res_lat[0].boxes) > 0:
                if entry.get("is_simulation") and "lateral" in entry["cameras"] and entry["cameras"]["lateral"].get("bbox_norm"):
                    gt_bbox_lat = entry["cameras"]["lateral"]["bbox_norm"]
                    best_box_lat = None
                    best_iou_lat = 0.0
                    for box in yolo_res_lat[0].boxes:
                        det_bbox = box.xyxyn[0].cpu().numpy().tolist()
                        iou = compute_iou(gt_bbox_lat, det_bbox)
                        if iou > best_iou_lat:
                            best_iou_lat = iou
                            best_box_lat = box
                    if best_box_lat is not None and best_iou_lat >= 0.1:
                        box_lat = best_box_lat
                    else:
                        box_lat = yolo_res_lat[0].boxes[0]
                else:
                    box_lat = yolo_res_lat[0].boxes[0]

                x1_l, y1_l, x2_l, y2_l = box_lat.xyxyn[0].cpu().numpy().tolist()
                lat_yolo_conf = float(box_lat.conf[0].cpu().numpy())

                # SAM lateral
                w_l, h_l = img_lat.size
                px1_l, py1_l = int(x1_l * w_l), int(y1_l * h_l)
                px2_l, py2_l = int(x2_l * w_l), int(y2_l * h_l)
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
    args = parser.parse_args()
    run_evaluation(args.metadata, args.report)
