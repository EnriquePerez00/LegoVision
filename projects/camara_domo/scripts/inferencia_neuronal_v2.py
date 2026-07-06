# -*- coding: utf-8 -*-
"""
camara_domo/scripts/inferencia_neuronal.py
=========================================
Pipeline de inferencia secuencial (Batch Processing) para Brickshare.
Implementa:
  - Carga de BD (Supabase/PostgreSQL) con fallback robusto a JSON locales.
  - Aceleración por GPU nativa (mps) para chips Apple Silicon (M4).
  - Detección con YOLO y segmentación con MobileSAM.
  - Corrección geométrica con warp de perspectiva (OpenCV) para la vista inclinada.
  - Tracking espacial (compensando velocidad de cinta) y asignación de tracking_ids.
  - Votación temporal consolidada y salida estructurada en JSON por tracking_id.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
import threading
from PIL import Image
import torch
from ultralytics import YOLO, SAM
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

# Configurar directorios del proyecto en el sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)
repo_root = os.path.dirname(legovic_root)

sys.path.insert(0, repo_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from core.utils.config_loader import cfg
from core.utils.logger import get_logger
from _kpts_observer import _P_CEN, _P_LAT, kpts_observer
from scripts.efficientnet_classifier import LegoEfficientNetClassifier

log = get_logger("inferencia_neuronal")

# Cargar ContourMatcher para recomposición por contornos
try:
    from contour_matcher import ContourMatcher
    contour_matcher = ContourMatcher()
    log.info("ContourMatcher para recomposición 3D cargado con éxito.")
except Exception as e:
    contour_matcher = None
    log.warning(f"No se pudo cargar ContourMatcher: {e}")

# ─────────────────────────────────────────────────────────────────
# 1. Modelos de Validación (Pydantic V2)
# ─────────────────────────────────────────────────────────────────

class StablePoseModel(BaseModel):
    part_ref: str
    pose_index: int
    zenith_observable_area: Optional[float] = None
    zenith_silhouette_area: Optional[float] = None
    lateral_height: Optional[float] = None
    effective_height: Optional[float] = None
    contact_stable_length: Optional[float] = None
    contact_stable_width: Optional[float] = None
    face_class: str
    is_stable: bool

class ColorModel(BaseModel):
    color_code: str
    color_name: str
    color_hex: str
    rgb_cenital: Optional[List[float]] = None
    rgb_lateral: Optional[List[float]] = None

class PiecePrediction(BaseModel):
    tracking_id: str
    referencia_detectada: str
    color: str
    pose_identificada: int
    score: float
    confidence_details: Dict[str, Any]
    frames_visible: List[str]

# ─────────────────────────────────────────────────────────────────
# 2. Utilidades de Carga de Datos (Base de Datos + Cache Local)
# ─────────────────────────────────────────────────────────────────

def load_db_universe(data_dir: Optional[str] = None) -> Tuple[Dict[str, List[StablePoseModel]], List[ColorModel]]:
    """Carga el catálogo de poses estables y la paleta de colores con fallback local."""
    poses_by_ref: Dict[str, List[StablePoseModel]] = {}
    colors_list: List[ColorModel] = []
    
    # 0. Cargar mapa de referencias permitidas si existe simulation_metadata.json en data_dir
    allowed_refs = None
    if data_dir and os.path.isdir(data_dir):
        meta_path = os.path.join(data_dir, "simulation_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                allowed_refs = set()
                for frame in meta.get("frames", []):
                    for piece in frame.get("visible_pieces", []):
                        allowed_refs.add(piece["ref"])
                log.info(f"[DB] Modo Catálogo Activo: Limitando búsqueda a las {len(allowed_refs)} referencias del dataset.")
            except Exception as e:
                log.warning(f"No se pudo cargar simulation_metadata.json para filtrado: {e}")
    
    # 0b. Cargar mapa de calibración empírica de color
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    if not os.path.exists(palette_path):
        palette_path = os.path.join(project_root, "data", "ccm_dome_light.json")
        
    calibrated_rgbs_cen = {}
    calibrated_rgbs_lat = {}
    palette_names = {}
    palette_hexes = {}
    if os.path.exists(palette_path):
        try:
            with open(palette_path, "r", encoding="utf-8") as f:
                palette = json.load(f)
                items = palette if isinstance(palette, list) else palette.values()
                for item in items:
                    c_code = str(item.get("color_code", item.get("code", "")))
                    rgb_val_cen = item.get("rgb_cenital", item.get("rgb"))
                    rgb_val_lat = item.get("rgb_lateral", item.get("rgb_cenital", item.get("rgb")))
                    c_name = item.get("color_name", item.get("name"))
                    c_hex = item.get("color_hex", item.get("hex"))
                    if c_code:
                        if rgb_val_cen:
                            calibrated_rgbs_cen[c_code] = rgb_val_cen
                        if rgb_val_lat:
                            calibrated_rgbs_lat[c_code] = rgb_val_lat
                        if c_name:
                            palette_names[c_code] = c_name
                        if c_hex:
                            palette_hexes[c_code] = c_hex
        except Exception as e:
            log.warning(f"Error cargando mapa de calibración de color: {e}")
            
    # Intentar conexión a Supabase
    try:
        from core.db.supabase_client import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Cargar poses estables
                cur.execute("""
                    SELECT part_ref, pose_index, zenith_observable_area, zenith_silhouette_area, 
                           lateral_height, effective_height, contact_stable_length, contact_stable_width,
                           face_class, is_stable
                    FROM stable_poses
                    WHERE is_stable = TRUE
                """)
                for row in cur.fetchall():
                    # Validación con Pydantic
                    pose = StablePoseModel(**row)
                    if allowed_refs is None or pose.part_ref in allowed_refs:
                        poses_by_ref.setdefault(pose.part_ref, []).append(pose)
                
                # Cargar colores
                cur.execute("""
                    SELECT DISTINCT color_code, color_name, color_hex
                    FROM lego_set_parts
                    WHERE color_code IS NOT NULL AND color_hex IS NOT NULL
                """)
                # Cargar colores deduplicados y corregidos con la paleta de calibración
                unique_colors = {}
                for row in cur.fetchall():
                    c_code = str(row["color_code"])
                    c_name = row["color_name"] or "Unknown"
                    c_hex = row["color_hex"]
                    
                    # Sobreescribir con paleta limpia si es necesario
                    if (c_name == "Unknown Color" or c_name == "Unknown") and c_code in palette_names:
                        c_name = palette_names[c_code]
                    if (not c_hex or c_hex == "#808080") and c_code in palette_hexes:
                        c_hex = palette_hexes[c_code]
                    
                    if c_code in unique_colors:
                        # Si ya existe, preferir el nombre válido sobre "Unknown Color"
                        if unique_colors[c_code].color_name == "Unknown Color" and c_name != "Unknown Color":
                            unique_colors[c_code].color_name = c_name
                            unique_colors[c_code].color_hex = c_hex
                    else:
                        rgb_cal_cen = calibrated_rgbs_cen.get(c_code)
                        rgb_cal_lat = calibrated_rgbs_lat.get(c_code)
                        unique_colors[c_code] = ColorModel(
                            color_code=c_code,
                            color_name=c_name,
                            color_hex=c_hex,
                            rgb_cenital=rgb_cal_cen,
                            rgb_lateral=rgb_cal_lat
                        )
                colors_list = list(unique_colors.values())
                
                if poses_by_ref:
                    log.info(f"[DB] Cargadas {sum(len(v) for v in poses_by_ref.values())} poses y {len(colors_list)} colores desde Supabase con calibración aplicada.")
                    return poses_by_ref, colors_list
    except Exception as e:
        log.warning(f"[DB] No se pudo conectar a la base de datos ({e}). Usando fallbacks de cache local.")

    # Fallback 1: stable_poses_cache.json
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    if not os.path.exists(cache_path):
        # Intentar en subproyecto
        cache_path = os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
        
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for ref, poses in data.items():
                    if allowed_refs is not None and ref not in allowed_refs:
                        continue
                    poses_by_ref[ref] = []
                    for p in poses:
                        # Asegurar compatibilidad de campos
                        p_mapped = {
                            "part_ref": ref,
                            "pose_index": p.get("pose_index", 0),
                            "zenith_observable_area": p.get("zenith_observable_area"),
                            "zenith_silhouette_area": p.get("zenith_silhouette_area"),
                            "lateral_height": p.get("lateral_height"),
                            "effective_height": p.get("effective_height") or p.get("efective_height"),
                            "contact_stable_length": p.get("contact_stable_length"),
                            "contact_stable_width": p.get("contact_stable_width"),
                            "face_class": p.get("face_class", "Side"),
                            "is_stable": p.get("is_stable", True)
                        }
                        poses_by_ref[ref].append(StablePoseModel(**p_mapped))
            log.info(f"[Cache] Cargadas poses para {len(poses_by_ref)} piezas desde {cache_path}.")
        except Exception as e:
            log.error(f"[Cache] Error leyendo cache de poses: {e}")

    # Fallback 2: color_calibration_palette.json
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    if not os.path.exists(palette_path):
        palette_path = os.path.join(project_root, "data", "ccm_dome_light.json")
        
    if os.path.exists(palette_path):
        try:
            with open(palette_path, "r", encoding="utf-8") as f:
                palette = json.load(f)
                items = palette if isinstance(palette, list) else palette.values()
                for item in items:
                    colors_list.append(ColorModel(
                        color_code=str(item.get("color_code", item.get("code", ""))),
                        color_name=item.get("color_name", item.get("name", "Unknown")),
                        color_hex=item.get("color_hex", item.get("hex", "#808080")),
                        rgb_cenital=item.get("rgb_cenital", item.get("rgb", [128, 128, 128])),
                        rgb_lateral=item.get("rgb_lateral", item.get("rgb_cenital", [128, 128, 128]))
                    ))
            log.info(f"[Palette] Cargados {len(colors_list)} colores de paleta de calibración.")
        except Exception as e:
            log.error(f"[Palette] Error leyendo paleta de colores: {e}")
            
    # Fallback extremo de colores mínimos si está vacío
    if not colors_list:
        colors_list = [
            ColorModel(color_code="11", color_name="Black", color_hex="#202020", rgb_cenital=[32,32,32]),
            ColorModel(color_code="7", color_name="Blue", color_hex="#0032B1", rgb_cenital=[0,50,177]),
            ColorModel(color_code="6", color_name="Green", color_hex="#24793D", rgb_cenital=[36,121,61]),
            ColorModel(color_code="5", color_name="Red", color_hex="#C30025", rgb_cenital=[195,0,37]),
            ColorModel(color_code="1", color_name="White", color_hex="#F2F3F2", rgb_cenital=[242,243,242]),
            ColorModel(color_code="14", color_name="Yellow", color_hex="#F9EF36", rgb_cenital=[249,239,54])
        ]
        
    return poses_by_ref, colors_list

# ─────────────────────────────────────────────────────────────────
# 3. Conversión de Color e Inferencia Perceptual
# ─────────────────────────────────────────────────────────────────

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convierte RGB [0-255] a CIELAB para comparación perceptual Delta E."""
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

    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.array([l_val, a_val, b_val], dtype=np.float32)

class ColorMLP(torch.nn.Module):
    def __init__(self, input_dim=12, num_classes=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def rgb_matrix_to_lab(rgb_matrix):
    arr = rgb_matrix.astype(np.float32) / 255.0
    mask_gt = arr > 0.04045
    arr[mask_gt] = ((arr[mask_gt] + 0.055) / 1.055) ** 2.4
    arr[~mask_gt] = arr[~mask_gt] / 12.92

    x = arr[:, 0] * 0.4124 + arr[:, 1] * 0.3576 + arr[:, 2] * 0.1805
    y = arr[:, 0] * 0.2126 + arr[:, 1] * 0.7152 + arr[:, 2] * 0.0722
    z = arr[:, 0] * 0.0193 + arr[:, 1] * 0.1192 + arr[:, 2] * 0.9505

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

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

def estimate_color_mlp_features(img, mask_binary):
    if isinstance(img, np.ndarray):
        img_arr = img
    else:
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

    pixels_lab = rgb_matrix_to_lab(pixels_rgb_filt)

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

def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    return np.array([128.0, 128.0, 128.0], dtype=np.float32)

def estimate_color_hsv(img: Any, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Estima el color medio en RGB y HSV aplicando erosión morfológica adaptativa y filtrado de brillos especulares (percentil 92)."""
    if isinstance(img, np.ndarray):
        img_arr = img
    else:
        img_arr = np.array(img.convert("RGB"))
    
    # 1. Aplicar erosión morfológica adaptativa según el tamaño de la máscara
    mask_area = np.sum(mask > 0)
    if mask_area > 5000:
        kernel_size = (7, 7)
    elif mask_area > 1500:
        kernel_size = (5, 5)
    else:
        kernel_size = (3, 3)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask
    mask_bool = (mask_to_use > 0)
    
    if not np.any(mask_bool):
        return np.array([128.0, 128.0, 128.0]), (0.0, 0.0, 0.0)
        
    # Extraer el núcleo central del 40% (para evitar brillos y bordes de fondo)
    ys, xs = np.where(mask_bool)
    ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = ymax - ymin, xmax - xmin
    cy, cx = ymin + h//2, xmin + w//2
    h_crop, w_crop = int(max(1, h * 0.4)), int(max(1, w * 0.4))
    
    central_mask = np.zeros_like(mask_bool)
    central_mask[max(0, cy - h_crop//2):cy + h_crop//2, max(0, cx - w_crop//2):cx + w_crop//2] = True
    core_mask = mask_bool & central_mask
    
    if np.any(core_mask):
        mask_bool = core_mask
        
    pixels_rgb = img_arr[mask_bool]
    
    hsv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    pixels_hsv = hsv_img[mask_bool]
    
    # 2. Filtrar especularidades usando percentiles: descartar el 8% de píxeles más brillantes
    # Y filtrar sombras absolutas (V < 15)
    values = pixels_hsv[:, 2]
    non_shadow_mask = values > 15
    
    if np.any(non_shadow_mask):
        pixels_rgb = pixels_rgb[non_shadow_mask]
        pixels_hsv = pixels_hsv[non_shadow_mask]
        values = values[non_shadow_mask]
        
    # Discard top 8% brightest pixels to eliminate specular highlights
    if len(values) > 10:
        val_threshold = np.percentile(values, 92)
        valid_specular_mask = values <= val_threshold
        if np.any(valid_specular_mask):
            pixels_rgb_filtered = pixels_rgb[valid_specular_mask]
            pixels_hsv_filtered = pixels_hsv[valid_specular_mask]
        else:
            pixels_rgb_filtered = pixels_rgb
            pixels_hsv_filtered = pixels_hsv
    else:
        pixels_rgb_filtered = pixels_rgb
        pixels_hsv_filtered = pixels_hsv
        
    mean_rgb = pixels_rgb_filtered.mean(axis=0)
    mean_hsv = pixels_hsv_filtered.mean(axis=0)
    
    return mean_rgb, (float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2]))

def find_nearest_color_fused(
    rgb_cen: np.ndarray, 
    rgb_lat: Optional[np.ndarray], 
    area_cen_px: float, 
    area_lat_px: float, 
    colors: List[ColorModel]
) -> ColorModel:
    """Busca el color más cercano calculando la distancia CIELAB ponderada por el área de máscara de cada cámara."""
    lab_cen = rgb_to_lab(rgb_cen)
    lab_lat = rgb_to_lab(rgb_lat) if rgb_lat is not None else None
    
    total_area = area_cen_px + area_lat_px
    if total_area > 0:
        w_cen = area_cen_px / total_area
        w_lat = area_lat_px / total_area
    else:
        w_cen, w_lat = 1.0, 0.0
        
    best_dist = float("inf")
    best_color = colors[0]
    
    for c in colors:
        ref_cen = c.rgb_cenital if c.rgb_cenital else hex_to_rgb(c.color_hex)
        lab_ref_cen = rgb_to_lab(ref_cen)
        dist_cen = np.linalg.norm(lab_cen - lab_ref_cen)
        
        if lab_lat is not None:
            ref_lat = c.rgb_lateral if c.rgb_lateral else (c.rgb_cenital if c.rgb_cenital else hex_to_rgb(c.color_hex))
            lab_ref_lat = rgb_to_lab(ref_lat)
            dist_lat = np.linalg.norm(lab_lat - lab_ref_lat)
            dist = w_cen * dist_cen + w_lat * dist_lat
        else:
            dist = dist_cen
            
        if dist < best_dist:
            best_dist = dist
            best_color = c
            
    return best_color

# ─────────────────────────────────────────────────────────────────
# 4. Procesamiento Geométrico, Epipolar y Keypoints
# ─────────────────────────────────────────────────────────────────

def backproject_cen_to_world(u, v, z, P_cen, w_img=1024, h_img=1024):
    # Scale from actual pixel space to 640x640 space expected by P_cen
    u_640 = u * (640.0 / w_img)
    v_640 = v * (640.0 / h_img)
    A = np.array([
        [P_cen[0, 0] - u_640 * P_cen[2, 0], P_cen[0, 1] - u_640 * P_cen[2, 1]],
        [P_cen[1, 0] - v_640 * P_cen[2, 0], P_cen[1, 1] - v_640 * P_cen[2, 1]]
    ])
    B = np.array([
        u_640 * (P_cen[2, 2] * z + P_cen[2, 3]) - (P_cen[0, 2] * z + P_cen[0, 3]),
        v_640 * (P_cen[2, 2] * z + P_cen[2, 3]) - (P_cen[1, 2] * z + P_cen[1, 3])
    ])
    xy = np.linalg.solve(A, B)
    return xy[0], xy[1], z

def project_world_to_lat(X, Y, Z, P_lat, w_img=1024, h_img=1024):
    pt = np.array([X, Y, Z, 1.0])
    proj = P_lat @ pt
    u_640 = proj[0] / proj[2]
    v_640 = proj[1] / proj[2]
    # Scale from 640x640 space to actual pixel space
    u = u_640 * (w_img / 640.0)
    v = v_640 * (h_img / 640.0)
    return u, v

def dist_point_to_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    tx = x1 + t * dx
    ty = y1 + t * dy
    return math.sqrt((px - tx)**2 + (py - ty)**2)

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

def extract_keypoints_for_bbox(pose_results, target_bbox_norm, conf_thresh=0.25):
    if not pose_results or pose_results[0].boxes is None or pose_results[0].keypoints is None:
        return None
    boxes = pose_results[0].boxes
    keypoints = pose_results[0].keypoints
    if len(boxes) == 0:
        return None
    best_iou = 0.0
    best_idx = -1
    for idx, box in enumerate(boxes):
        det_bbox = box.xyxyn[0].cpu().numpy().tolist()
        iou = compute_iou(target_bbox_norm, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    if best_idx != -1 and best_iou > 0.1:
        xyn = keypoints.xyn[best_idx].cpu().numpy()
        kconf = (keypoints.conf[best_idx].cpu().numpy() 
                 if keypoints.conf is not None 
                 else np.ones(len(xyn)) * float(boxes.conf[best_idx].cpu().numpy()))
        return np.hstack([xyn, kconf.reshape(-1, 1)])
    return None

# ─────────────────────────────────────────────────────────────────
# 4b. Procesamiento de Perspectiva (Original 4)
# ─────────────────────────────────────────────────────────────────

def warp_inclined_to_cenital(
    mask_lat: np.ndarray, 
    bbox_lat_norm: List[float], 
    bbox_cen_norm: List[float], 
    img_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Aplica cv2.getPerspectiveTransform y cv2.warpPerspective para proyectar
    el plano de la pieza en la vista inclinada al espacio cenital.
    """
    h_img, w_img = img_shape
    
    # Bounding box en píxeles
    lx1, ly1, lx2, ly2 = [int(bbox_lat_norm[0]*w_img), int(bbox_lat_norm[1]*h_img), 
                          int(bbox_lat_norm[2]*w_img), int(bbox_lat_norm[3]*h_img)]
    cx1, cy1, cx2, cy2 = [int(bbox_cen_norm[0]*w_img), int(bbox_cen_norm[1]*h_img), 
                          int(bbox_cen_norm[2]*w_img), int(bbox_cen_norm[3]*h_img)]
    
    # Puntos de origen (inclinada) y destino (cenital)
    src_pts = np.float32([[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]])
    dst_pts = np.float32([[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]])
    
    # Obtener matriz de homografía
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Aplicar transformación
    warped_mask = cv2.warpPerspective(mask_lat, M, (w_img, h_img), flags=cv2.INTER_NEAREST)
    return warped_mask

# ─────────────────────────────────────────────────────────────────
# 5. Extracción de Studs / Huecos
# ─────────────────────────────────────────────────────────────────

def detect_studs_and_holes(img_cen: Image.Image, mask_cen: np.ndarray) -> int:
    """
    Detección del número de studs o huecos en la superficie superior.
    Usa contornos circulares locales bajo la máscara de segmentación.
    """
    cv_img = cv2.cvtColor(np.array(img_cen.convert("RGB")), cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.bitwise_and(cv_img, cv_img, mask=mask_cen)
    
    # Filtro bilateral y detección de bordes
    blurred = cv2.bilateralFilter(masked_gray, 9, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_contours_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        # Buscar contornos circulares pequeños de studs (rango de 5 a 150 píxeles de área)
        if 5.0 < area < 150.0 and circularity > 0.65:
            circle_contours_count += 1
            
    return min(8, circle_contours_count)

# ─────────────────────────────────────────────────────────────────
# 6. Cálculo Físico de Superficies
# ─────────────────────────────────────────────────────────────────

def get_physical_centroid_mm(bbox_cen_norm: List[float], px_per_mm: float) -> Tuple[float, float]:
    nominal_scale = float(cfg.cameras.cenital.scale_px_per_mm)
    cx_px = (bbox_cen_norm[0] + bbox_cen_norm[2]) * 0.5 * 2048.0
    cy_px = (bbox_cen_norm[1] + bbox_cen_norm[3]) * 0.5 * 2048.0
    dx_px = cx_px - 1024.0
    dy_px = cy_px - 1024.0
    return dx_px / nominal_scale, dy_px / nominal_scale

def compute_physical_areas(
    mask_cen: np.ndarray, 
    warped_mask_lat: np.ndarray, 
    bbox_cen_norm: List[float], 
    px_per_mm: float
) -> Tuple[float, float]:
    """Calcula las superficies reales cenital y vertical corregidas en mm2."""
    dx_mm, dy_mm = get_physical_centroid_mm(bbox_cen_norm, px_per_mm)
    r_mm = math.sqrt(dx_mm**2 + dy_mm**2)
    
    # Escala local basada en distancia radial y altura real de la cámara cenital
    cam_z = float(cfg.cameras.cenital.position[2]) * 10.0  # 150.0 mm
    d_floor = math.sqrt(r_mm**2 + cam_z**2)
    px_per_mm_local = (px_per_mm * cam_z) / d_floor
    
    # Área cenital corregida
    px_count_cen = float(np.sum(mask_cen > 0))
    area_cen_mm2 = px_count_cen / (px_per_mm_local ** 2)
    
    # Área vertical corregida a partir de la máscara inclinada warped
    px_count_lat = float(np.sum(warped_mask_lat > 0))
    area_lat_mm2 = px_count_lat / (px_per_mm_local ** 2)
    
    return area_cen_mm2, area_lat_mm2

# ─────────────────────────────────────────────────────────────────
# 7. Algoritmo de Matching de Base de Datos
# ─────────────────────────────────────────────────────────────────

def match_piece_hypothesis(
    poses_db: Dict[str, List[StablePoseModel]],
    color_inferido: ColorModel,
    area_cen_est: float,
    area_lat_est: float,
    height_est: float,
    studs_est: int,
    epsilon: float = 0.40,
    epsilon_vertical: float = 0.40,
    epsilon_height: float = 0.40,
    height_is_fallback: bool = False
) -> List[Tuple[str, int, float]]:
    """
    Cruza características reales con las hipótesis de la base de datos aplicando tolerancias.
    """
    candidates = []
    
    for ref, poses in poses_db.items():
        for pose in poses:
            # Criterio 2 (Cenital): Comparar área horizontal estimada con la teórica
            nominal_cen = pose.zenith_silhouette_area or pose.zenith_observable_area
            if not nominal_cen:
                continue
            
            diff_cen = abs(area_cen_est - nominal_cen) / nominal_cen
            if diff_cen > epsilon:
                continue  # No cumple la tolerancia cenital
                
            # Criterio 3 (Inclinada): Deshabilitado debido a proyección diagonal de 45°
            diff_lat_best = 0.0

            # Criterio 3b (Altura): Comparar la altura física medida/triangulada
            nominal_h = pose.lateral_height or pose.effective_height or 9.6
            if height_is_fallback:
                diff_h = 0.0
            else:
                diff_h = abs(height_est - nominal_h) / nominal_h
                if diff_h > epsilon_height:
                    continue
                
            # Criterio 4 (Condicionador - Studs): Evaluar coherencia de studs
            expected_studs_up = pose.face_class == "Bottom"
            
            # Score de similitud (menor es mejor)
            if height_is_fallback:
                score = diff_cen * 1.0
            else:
                score = (diff_cen * 0.7) + (diff_h * 0.3)
                
            if (expected_studs_up and studs_est == 0) or (not expected_studs_up and studs_est > 0):
                score += 0.2  # Penalización por studs incongruentes
                
            candidates.append((ref, pose.pose_index, float(1.0 / (1.0 + score))))
            
    # Ordenar por el score de confianza (de mayor a menor)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates

# ─────────────────────────────────────────────────────────────────
# 8. Pipeline Principal y Procesamiento Secuencial
# ─────────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str, output_path: str, belt_speed: float, fps: float, max_frames: Optional[int] = None, mode: str = "CLASSIC"):
    # Detección del dispositivo GPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"Dispositivo PyTorch configurado: {device}")
    
    # Cargar modelos neuronales
    log.info("Cargando modelos YOLO y SAM...")
    
    def load_yolo_model(default_filename, fallback_filename=None):
        # 1. Intentar cargar versión CoreML (.mlpackage) en macOS
        coreml_filename = default_filename.replace(".pt", ".mlpackage")
        coreml_path = os.path.join(project_root, "models", coreml_filename)
        if os.path.exists(coreml_path):
            log.info(f"Cargando modelo CoreML optimizado para Neural Engine: {coreml_filename}")
            return YOLO(coreml_path)
            
        if fallback_filename:
            fb_coreml_filename = fallback_filename.replace(".pt", ".mlpackage")
            fb_coreml_path = os.path.join(project_root, "models", fb_coreml_filename)
            if os.path.exists(fb_coreml_path):
                log.info(f"Cargando modelo CoreML optimizado para Neural Engine: {fb_coreml_filename}")
                return YOLO(fb_coreml_path)
        
        # 2. Fallback al modelo de PyTorch (.pt)
        pt_path = os.path.join(project_root, "models", default_filename)
        if os.path.exists(pt_path):
            log.info(f"Cargando modelo PyTorch (.pt): {default_filename}")
            return YOLO(pt_path).to(device)
            
        if fallback_filename:
            fb_pt_path = os.path.join(project_root, "models", fallback_filename)
            if os.path.exists(fb_pt_path):
                log.info(f"Cargando modelo PyTorch (.pt) de respaldo: {fallback_filename}")
                return YOLO(fb_pt_path).to(device)
                
        log.warning(f"No se encontró el modelo en {pt_path}, intentando carga directa...")
        return YOLO(pt_path).to(device)

    yolo_cen = load_yolo_model("yolo_cenital.pt")
    yolo_lat = load_yolo_model("yolo_lateral.pt")
    yolo_cen_pose = load_yolo_model("yolo_cenital_pose.pt")
    yolo_lat_pose = load_yolo_model("yolo_lateral_pose.pt", "yolo_frontal_pose.pt")
    sam_model = SAM(os.path.join(legovic_root, "mobile_sam.pt")).to(device)
    sam_lock = threading.Lock()

    log.info("Cargando clasificador neuro-simbólico EfficientNetV2-B0...")
    efficientnet_clf = LegoEfficientNetClassifier()

    color_mlp_model = None
    color_mlp_meta = None
    mlp_model_path = os.path.join(project_root, "models", "color_mlp_model.pt")
    mlp_meta_path = os.path.join(project_root, "models", "color_mlp_metadata.json")
    if os.path.exists(mlp_model_path) and os.path.exists(mlp_meta_path):
        log.info("Cargando clasificador MLP de color...")
        try:
            with open(mlp_meta_path, "r", encoding="utf-8") as f:
                color_mlp_meta = json.load(f)
            num_classes = len(color_mlp_meta["classes"])
            color_mlp_model = ColorMLP(input_dim=12, num_classes=num_classes).to(device)
            color_mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
            color_mlp_model.eval()
            log.info(f"MLP de color cargado correctamente con {num_classes} clases.")
        except Exception as e:
            log.error(f"Error cargando MLP de color: {e}")
    
    # Cargar universo DB
    poses_db, colors_db = load_db_universe(data_dir)
    
    # Agrupar archivos por ID de frame
    frames_found = {}
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if "_cenital" in f and (f.endswith(".png") or f.endswith(".jpg")):
                fid = f.split("_cenital")[0]
                inc_candidate_1 = os.path.join(data_dir, f.replace("_cenital", "_inclinada_45"))
                inc_candidate_2 = os.path.join(data_dir, f.replace("_cenital", "_frontal"))
                inc_path = inc_candidate_1 if os.path.exists(inc_candidate_1) else (inc_candidate_2 if os.path.exists(inc_candidate_2) else inc_candidate_1)
                frames_found[fid] = {
                    "cenital": os.path.join(data_dir, f),
                    "inclinada": inc_path
                }
            elif (f.endswith(".png") or f.endswith(".jpg")) and not any(x in f for x in ["_frontal", "_inclinada_45", "_cenital"]):
                fid = os.path.splitext(f)[0]
                ext = os.path.splitext(f)[1]
                inc_candidate_1 = os.path.join(data_dir, f"{fid}_inclinada_45{ext}")
                inc_candidate_2 = os.path.join(data_dir, f"{fid}_frontal{ext}")
                inc_path = inc_candidate_1 if os.path.exists(inc_candidate_1) else (inc_candidate_2 if os.path.exists(inc_candidate_2) else None)
                if inc_path:
                    frames_found[fid] = {
                        "cenital": os.path.join(data_dir, f),
                        "inclinada": inc_path
                    }
    
    sorted_fids = sorted(frames_found.keys())
    if max_frames is not None:
        sorted_fids = sorted_fids[:max_frames]
    log.info(f"Encontrados {len(sorted_fids)} frames secuenciales para procesar.")
    
    # Intentar cargar offsets reales de la simulación para sincronizar el tracking
    offsets = {}
    metadata_path = os.path.join(data_dir, "simulation_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            for frame in meta_data.get("frames", []):
                fname = frame.get("file_name")
                if fname:
                    fid = os.path.splitext(fname)[0]
                    val = frame.get("belt_offset_mm")
                    if val is not None:
                        offsets[fid] = val
            if offsets:
                log.info(f"[Sync] Sincronización activa: Cargados {len(offsets)} offsets de simulación.")
            else:
                log.info("[Sync] No se encontraron offsets de cinta en simulación (se usará avance teórico).")
        except Exception as e:
            log.warning(f"No se pudo cargar simulation_metadata.json para sincronizar tracking: {e}")

    # === Pre-Inferencia YOLO en Lotes ===
    log.info("===> Iniciando Pre-Inferencia YOLO en lotes (Batch Inference)...")
    batch_size = 16
    
    cen_paths = []
    lat_paths = []
    for fid in sorted_fids:
        img_cen_path = frames_found[fid]["cenital"]
        img_lat_path = frames_found[fid]["inclinada"]
        if not os.path.exists(img_lat_path):
            img_lat_path = img_lat_path.replace(".png", ".jpg") if img_lat_path.endswith(".png") else img_lat_path.replace(".jpg", ".png")
        cen_paths.append(img_cen_path)
        lat_paths.append(img_lat_path)
    
    if mode == "CLASSIC":
        res_cen_all = yolo_cen(cen_paths, batch=batch_size, conf=0.25, verbose=False)
        res_lat_all = yolo_lat(lat_paths, batch=batch_size, conf=0.25, verbose=False)
        res_cen_pose_all = yolo_cen_pose(cen_paths, batch=batch_size, conf=0.25, verbose=False)
        res_lat_pose_all = yolo_lat_pose(lat_paths, batch=batch_size, conf=0.25, verbose=False)
    else:
        res_cen_all = yolo_cen_pose(cen_paths, batch=batch_size, conf=0.25, verbose=False)
        res_lat_all = yolo_lat_pose(lat_paths, batch=batch_size, conf=0.25, verbose=False)
        res_cen_pose_all = res_cen_all
        res_lat_pose_all = res_lat_all
        
    log.info("===> Pre-Inferencia YOLO finalizada. Iniciando loop de tracking...")

    delta_t = 1.0 / fps
    active_tracks: List[Dict[str, Any]] = []
    finished_tracks: List[Dict[str, Any]] = []
    track_counter = 0
    prev_offset = None
    
    for idx, fid in enumerate(sorted_fids):
        log.info(f"--- Procesando Frame ID: {fid} ---")
        
        # Calcular el desplazamiento real de la cinta para este frame
        current_offset = offsets.get(fid)
        if current_offset is not None and prev_offset is not None:
            dx_belt = current_offset - prev_offset
            log.info(f"[{fid}] Desplazamiento sincronizado de cinta: dx_belt = {dx_belt:.2f} mm")
        else:
            dx_belt = belt_speed * delta_t
            log.info(f"[{fid}] Desplazamiento teórico de cinta: dx_belt = {dx_belt:.2f} mm")
            
        if current_offset is not None:
            prev_offset = current_offset
        img_cen_path = cen_paths[idx]
        img_lat_path = lat_paths[idx]
        
        img_cen = Image.open(img_cen_path)
        img_lat = Image.open(img_lat_path)
        
        w_c, h_c = img_cen.size
        w_l, h_l = img_lat.size
        
        # Escalas de cámara calibradas dinámicas según la resolución del render
        px_per_mm_cen = float(cfg.cameras.cenital.scale_px_per_mm) * (w_c / 2048.0)
        px_per_mm_lat = float(cfg.cameras.lateral.scale_px_per_mm) * (w_l / 2048.0)
        
        # 1. Recuperar Detecciones YOLO Pre-calculadas
        yolo_res_cen = [res_cen_all[idx]]
        yolo_res_lat = [res_lat_all[idx]]
        
        n_cen = len(yolo_res_cen[0].boxes) if yolo_res_cen and yolo_res_cen[0].boxes is not None else 0
        n_lat = len(yolo_res_lat[0].boxes) if yolo_res_lat and yolo_res_lat[0].boxes is not None else 0
        if n_cen > 0 or n_lat > 0:
            log.info(f"[{fid}] Detecciones YOLO: Cenital={n_cen}, Lateral/Inclinada={n_lat}")
            
        # Correr YOLO-Pose una vez por frame si hay detecciones
        pose_res_cen = [res_cen_pose_all[idx]] if n_cen > 0 else None
        pose_res_lat = [res_lat_pose_all[idx]] if n_lat > 0 else None
        
        # Convertir a numpy de forma segura en el hilo principal para evitar colisiones de PIL
        img_cen_np = np.array(img_cen.convert("RGB"))
        img_lat_np = np.array(img_lat.convert("RGB"))
        
        detections_frame = []
        
        # 1.1 Preprocesar y asociar epipolarmente de forma global y única
        cen_to_lat_match = {} 
        assigned_lat_indices = set()
        
        # Helpers para gating de color y escala en el matching epipolar
        def get_crop_mean_rgb(img_np, bbox_norm):
            h, w = img_np.shape[:2]
            x1, y1, x2, y2 = bbox_norm
            px1, py1, px2, py2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            crop = img_np[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
            if crop.size == 0:
                return np.array([128.0, 128.0, 128.0], dtype=np.float32)
            return np.mean(crop, axis=(0,1))

        def delta_e_lab(rgb1, rgb2):
            lab1 = rgb_to_lab(rgb1)
            lab2 = rgb_to_lab(rgb2)
            return float(np.linalg.norm(lab1 - lab2))
        
        if yolo_res_cen and len(yolo_res_cen[0].boxes) > 0 and yolo_res_lat and len(yolo_res_lat[0].boxes) > 0:
            epipolar_candidates = []
            for i, box in enumerate(yolo_res_cen[0].boxes):
                bbox_cen = box.xyxyn[0].cpu().numpy().tolist()
                cx_cen = (bbox_cen[0] + bbox_cen[2]) * 0.5 * w_c
                cy_cen = (bbox_cen[1] + bbox_cen[3]) * 0.5 * h_c
                try:
                    X0, Y0, _ = backproject_cen_to_world(cx_cen, cy_cen, 0.0, _P_CEN, w_c, h_c)
                    X35, Y35, _ = backproject_cen_to_world(cx_cen, cy_cen, 35.0, _P_CEN, w_c, h_c)
                    u0_lat, v0_lat = project_world_to_lat(X0, Y0, 0.0, _P_LAT, w_l, h_l)
                    u35_lat, v35_lat = project_world_to_lat(X35, Y35, 35.0, _P_LAT, w_l, h_l)
                except Exception:
                    u0_lat, v0_lat = cx_cen, cy_cen
                    u35_lat, v35_lat = cx_cen, cy_cen
                
                for l_idx, l_box in enumerate(yolo_res_lat[0].boxes):
                    bbox_l = l_box.xyxyn[0].cpu().numpy().tolist()
                    
                    # 1. Filtro rápido de escala (Descartar si hay más de 4x diferencia de escala en área de bbox)
                    area_cen_bbox = (bbox_cen[2] - bbox_cen[0]) * (bbox_cen[3] - bbox_cen[1])
                    area_lat_bbox = (bbox_l[2] - bbox_l[0]) * (bbox_l[3] - bbox_l[1])
                    ratio = max(area_cen_bbox, area_lat_bbox) / (min(area_cen_bbox, area_lat_bbox) + 1e-6)
                    if ratio > 4.0:
                        continue
                        
                    # 2. Gating de color perceptual (Descartar si la distancia de color LAB es muy grande)
                    rgb_cen = get_crop_mean_rgb(img_cen_np, bbox_cen)
                    rgb_lat = get_crop_mean_rgb(img_lat_np, bbox_l)
                    if delta_e_lab(rgb_cen, rgb_lat) > 30.0:
                        continue

                    cx_l = (bbox_l[0] + bbox_l[2]) * 0.5 * w_l
                    cy_l = (bbox_l[1] + bbox_l[3]) * 0.5 * h_l
                    dist = dist_point_to_segment(cx_l, cy_l, u0_lat, v0_lat, u35_lat, v35_lat)
                    if dist < 100.0:
                        epipolar_candidates.append((dist, i, l_idx))
            
            epipolar_candidates.sort(key=lambda x: x[0])
            for dist, cen_idx, lat_idx in epipolar_candidates:
                if cen_idx not in cen_to_lat_match and lat_idx not in assigned_lat_indices:
                    cen_to_lat_match[cen_idx] = (lat_idx, dist)
                    assigned_lat_indices.add(lat_idx)
        
        if yolo_res_cen and len(yolo_res_cen[0].boxes) > 0:
            from concurrent.futures import ThreadPoolExecutor
            
            def process_detection(i):
                try:
                    box = yolo_res_cen[0].boxes[i]
                    bbox_cen = box.xyxyn[0].cpu().numpy().tolist()
                    
                    px1_c, py1_c = int(bbox_cen[0] * w_c), int(bbox_cen[1] * h_c)
                    px2_c, py2_c = int(bbox_cen[2] * w_c), int(bbox_cen[3] * h_c)
                    cx_c = (px1_c + px2_c) * 0.5
                    cy_c = (py1_c + py2_c) * 0.5
                    
                    if mode == "POSE_ONLY":
                        mask_cen = np.zeros((h_c, w_c), dtype=np.uint8)
                        if pose_res_cen and pose_res_cen[0].keypoints is not None:
                            kpts = pose_res_cen[0].keypoints.xy[i].cpu().numpy()
                            valid_kpts = [pt for pt in kpts if pt[0] > 0 and pt[1] > 0]
                            if len(valid_kpts) >= 3:
                                pts = np.array(valid_kpts, np.int32)
                                hull = cv2.convexHull(pts)
                                cv2.fillConvexPoly(mask_cen, hull, 1)
                            else:
                                cv2.rectangle(mask_cen, (px1_c, py1_c), (px2_c, py2_c), 1, -1)
                        else:
                            cv2.rectangle(mask_cen, (px1_c, py1_c), (px2_c, py2_c), 1, -1)
                    else:
                        with sam_lock:
                            sam_res_cen = sam_model(img_cen_np, bboxes=[[px1_c, py1_c, px2_c, py2_c]], points=[[[cx_c, cy_c]]], labels=[[1]], verbose=False)
                        if not sam_res_cen or sam_res_cen[0].masks is None:
                            return None
                        mask_cen = sam_res_cen[0].masks.data[0].cpu().numpy().astype(np.uint8)
                        
                        # Fallback a Convex Hull de keypoints si SAM devuelve una máscara vacía o demasiado pequeña (e.g., bajo contraste)
                        bbox_area_px = (px2_c - px1_c) * (py2_c - py1_c)
                        sam_area_px = np.sum(mask_cen > 0)
                        if sam_area_px < 0.20 * bbox_area_px:
                            if pose_res_cen and pose_res_cen[0].keypoints is not None:
                                kpts = pose_res_cen[0].keypoints.xy[i].cpu().numpy()
                                valid_kpts = [pt for pt in kpts if pt[0] > 0 and pt[1] > 0]
                                if len(valid_kpts) >= 3:
                                    mask_cen.fill(0)
                                    pts = np.array(valid_kpts, np.int32)
                                    hull = cv2.convexHull(pts)
                                    cv2.fillConvexPoly(mask_cen, hull, 1)
                    
                    # Tight crop cenital usando segmentación e indexado NumPy (hilo seguro)
                    img_cen_proc_np = img_cen_np.copy()
                    img_cen_proc_np[mask_cen == 0] = [0, 0, 0]
                    
                    ys_c, xs_c = np.where(mask_cen > 0)
                    if len(ys_c) > 0:
                        px1_c_tight, py1_c_tight, px2_c_tight, py2_c_tight = int(np.min(xs_c)), int(np.min(ys_c)), int(np.max(xs_c)), int(np.max(ys_c))
                        crop_cen_np = img_cen_proc_np[py1_c_tight:py2_c_tight+1, px1_c_tight:px2_c_tight+1]
                    else:
                        crop_cen_np = img_cen_proc_np[py1_c:py2_c+1, px1_c:px2_c+1]
                    crop_cen = Image.fromarray(crop_cen_np)
                    
                    # Estimación de Color Cenital
                    mean_rgb, mean_hsv = estimate_color_hsv(img_cen_np, mask_cen)
                    
                    best_lat_idx, best_lat_dist = cen_to_lat_match.get(i, (-1, float("inf")))
                    
                    bbox_lat = [0.0, 0.0, 1.0, 1.0]
                    mask_lat = np.zeros(mask_cen.shape, dtype=np.uint8)
                    crop_lat = None
                    height_est = 9.6
                    height_is_fallback = True
                    mean_rgb_lat = None
                    
                    if best_lat_idx != -1 and best_lat_dist < 100.0:
                        bbox_lat = yolo_res_lat[0].boxes[best_lat_idx].xyxyn[0].cpu().numpy().tolist()
                        px1_l, py1_l = int(bbox_lat[0] * w_l), int(bbox_lat[1] * h_l)
                        px2_l, py2_l = int(bbox_lat[2] * w_l), int(bbox_lat[3] * h_l)
                        cx_l = (px1_l + px2_l) * 0.5
                        cy_l = (py1_l + py2_l) * 0.5
                        
                        if mode == "POSE_ONLY":
                            mask_lat = np.zeros((h_l, w_l), dtype=np.uint8)
                            if pose_res_lat and pose_res_lat[0].keypoints is not None:
                                kpts_l = pose_res_lat[0].keypoints.xy[best_lat_idx].cpu().numpy()
                                valid_kpts_l = [pt for pt in kpts_l if pt[0] > 0 and pt[1] > 0]
                                if len(valid_kpts_l) >= 3:
                                    pts_l = np.array(valid_kpts_l, np.int32)
                                    hull_l = cv2.convexHull(pts_l)
                                    cv2.fillConvexPoly(mask_lat, hull_l, 1)
                                else:
                                    cv2.rectangle(mask_lat, (px1_l, py1_l), (px2_l, py2_l), 1, -1)
                            else:
                                cv2.rectangle(mask_lat, (px1_l, py1_l), (px2_l, py2_l), 1, -1)
                        else:
                            with sam_lock:
                                sam_res_lat = sam_model(img_lat_np, bboxes=[[px1_l, py1_l, px2_l, py2_l]], points=[[[cx_l, cy_l]]], labels=[[1]], verbose=False)
                            if sam_res_lat and sam_res_lat[0].masks is not None:
                                mask_lat = sam_res_lat[0].masks.data[0].cpu().numpy().astype(np.uint8)
                                
                                # Fallback a Convex Hull de keypoints si SAM devuelve una máscara vacía o demasiado pequeña (e.g., bajo contraste)
                                bbox_area_lat_px = (px2_l - px1_l) * (py2_l - py1_l)
                                sam_area_lat_px = np.sum(mask_lat > 0)
                                if sam_area_lat_px < 0.20 * bbox_area_lat_px:
                                    if pose_res_lat and pose_res_lat[0].keypoints is not None:
                                        kpts_l = pose_res_lat[0].keypoints.xy[best_lat_idx].cpu().numpy()
                                        valid_kpts_l = [pt for pt in kpts_l if pt[0] > 0 and pt[1] > 0]
                                        if len(valid_kpts_l) >= 3:
                                            mask_lat.fill(0)
                                            pts_l = np.array(valid_kpts_l, np.int32)
                                            hull_l = cv2.convexHull(pts_l)
                                            cv2.fillConvexPoly(mask_lat, hull_l, 1)
                            
                            # Tight crop lateral usando segmentación e indexado NumPy (hilo seguro)
                            img_lat_proc_np = img_lat_np.copy()
                            img_lat_proc_np[mask_lat == 0] = [0, 0, 0]
                            
                            ys_l, xs_l = np.where(mask_lat > 0)
                            if len(ys_l) > 0:
                                px1_l_tight, py1_l_tight, px2_l_tight, py2_l_tight = int(np.min(xs_l)), int(np.min(ys_l)), int(np.max(xs_l)), int(np.max(ys_l))
                                crop_lat_np = img_lat_proc_np[py1_l_tight:py2_l_tight+1, px1_l_tight:px2_l_tight+1]
                            else:
                                crop_lat_np = img_lat_proc_np[py1_l:py2_l+1, px1_l:px2_l+1]
                            crop_lat = Image.fromarray(crop_lat_np)
                            
                            # Estimación de Color Lateral
                            mean_rgb_lat, _ = estimate_color_hsv(img_lat_np, mask_lat)
                        
                        kps_cen = extract_keypoints_for_bbox(pose_res_cen, bbox_cen)
                        kps_lat = extract_keypoints_for_bbox(pose_res_lat, bbox_lat)
                        if kps_cen is not None and kps_lat is not None:
                            obs_3d = kpts_observer(kps_cen, kps_lat, conf_min=0.20)
                            h_triang = obs_3d.get("lateral_height_mm")
                            if h_triang and 2.0 < h_triang < 40.0:
                                height_est = h_triang
                                height_is_fallback = False
                            else:
                                height_est = (bbox_lat[3] - bbox_lat[1]) * h_l / px_per_mm_lat
                        else:
                            height_est = (bbox_lat[3] - bbox_lat[1]) * h_l / px_per_mm_lat
                    
                    # --- Predicción de Color mediante ColorMLP con Fusion Multi-Vista ---
                    color_name_cen = None
                    color_name_lat = None
                    color_best_name = None
                    
                    if color_mlp_model is not None and color_mlp_meta is not None:
                        try:
                            # 1. Obtener probabilidades cenitales
                            prob_cen = None
                            feat_cen = estimate_color_mlp_features(img_cen_np, mask_cen)
                            if feat_cen is not None:
                                mean_arr = np.array(color_mlp_meta["mean"], dtype=np.float32)
                                scale_arr = np.array(color_mlp_meta["scale"], dtype=np.float32)
                                feat_scaled = (feat_cen - mean_arr) / (scale_arr + 1e-8)
                                feat_t = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    logits = color_mlp_model(feat_t)
                                    prob_cen = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            
                            # 2. Obtener probabilidades laterales
                            prob_lat = None
                            if mean_rgb_lat is not None:
                                feat_lat = estimate_color_mlp_features(img_lat_np, mask_lat)
                                if feat_lat is not None:
                                    mean_arr = np.array(color_mlp_meta["mean"], dtype=np.float32)
                                    scale_arr = np.array(color_mlp_meta["scale"], dtype=np.float32)
                                    feat_scaled = (feat_lat - mean_arr) / (scale_arr + 1e-8)
                                    feat_t = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        logits = color_mlp_model(feat_t)
                                        prob_lat = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            
                            # 3. Fusión Multi-Vista o predicción individual
                            if prob_cen is not None and prob_lat is not None:
                                w_cen = float(np.sum(mask_cen))
                                w_lat = float(np.sum(mask_lat))
                                fused_prob = (prob_cen * w_cen + prob_lat * w_lat) / (w_cen + w_lat + 1e-8)
                                
                                color_best_name = color_mlp_meta["classes"][np.argmax(fused_prob)]
                                color_name_cen = color_mlp_meta["classes"][np.argmax(prob_cen)]
                                color_name_lat = color_mlp_meta["classes"][np.argmax(prob_lat)]
                            elif prob_cen is not None:
                                color_best_name = color_mlp_meta["classes"][np.argmax(prob_cen)]
                                color_name_cen = color_best_name
                                color_name_lat = "N/A"
                            elif prob_lat is not None:
                                color_best_name = color_mlp_meta["classes"][np.argmax(prob_lat)]
                                color_name_cen = "Unknown"
                                color_name_lat = color_best_name
                        except Exception as e:
                            log.warning(f"Error procesando ColorMLP: {e}")
                    
                    # Fallback a CIELAB si el MLP no pudo ejecutarse
                    if color_best_name is None:
                        color_best = find_nearest_color_fused(mean_rgb, mean_rgb_lat, float(np.sum(mask_cen)), float(np.sum(mask_lat)), colors_db)
                        color_best_name = color_best.color_name
                    if color_name_cen is None:
                        color_name_cen = find_nearest_color_fused(mean_rgb, None, 1.0, 0.0, colors_db).color_name
                    if color_name_lat is None:
                        color_name_lat = find_nearest_color_fused(mean_rgb_lat, None, 1.0, 0.0, colors_db).color_name if mean_rgb_lat is not None else "N/A"
                    
                    n_studs = detect_studs_and_holes(img_cen, mask_cen)
                    warped_mask = warp_inclined_to_cenital(mask_lat, bbox_lat, bbox_cen, mask_cen.shape)
                    area_cen, area_lat = compute_physical_areas(mask_cen, warped_mask, bbox_cen, px_per_mm_cen)
                    cx_mm, cy_mm = get_physical_centroid_mm(bbox_cen, px_per_mm_cen)
                    
                    return {
                        "centroid_mm": (cx_mm, cy_mm),
                        "bbox_cen": bbox_cen,
                        "bbox_lat": bbox_lat,
                        "crop_cen": crop_cen,
                        "mask_cen": mask_cen,
                        "crop_lat": crop_lat,
                        "mask_lat": mask_lat,
                        "color": color_best_name,
                        "color_cenital": color_name_cen,
                        "color_lateral": color_name_lat,
                        "area_cen": area_cen,
                        "area_lat": area_lat,
                        "height": height_est,
                        "height_valid": not height_is_fallback,
                        "studs": n_studs,
                        "candidates": []
                    }
                except Exception as ex:
                    log.error(f"Error procesando detección {i} en frame {fid}: {ex}")
                    return None
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                res_futures = list(executor.map(process_detection, range(len(yolo_res_cen[0].boxes))))
                for r in res_futures:
                    if r is not None:
                        detections_frame.append(r)
        
        # 2. Asignación y tracking
        matched_detections = set()
        # Compensación del avance de la cinta (desplazamiento en X)
        # (Usando dx_belt sincronizado calculado al inicio de la iteración)
        
        for track in active_tracks:
            pred_x = track["last_centroid"][0] - dx_belt
            pred_y = track["last_centroid"][1]
            
            best_det_idx = -1
            best_dist = float("inf")
            
            for d_idx, det in enumerate(detections_frame):
                if d_idx in matched_detections:
                    continue
                dist = math.sqrt((det["centroid_mm"][0] - pred_x)**2 + (det["centroid_mm"][1] - pred_y)**2)
                if dist < best_dist and dist < 20.0:
                    best_dist = dist
                    best_det_idx = d_idx
            
            if best_det_idx != -1:
                det = detections_frame[best_det_idx]
                track["history"].append(det)
                track["last_centroid"] = det["centroid_mm"]
                track["frames"].append(fid)
                track["consecutive_misses"] = 0
                matched_detections.add(best_det_idx)
            else:
                track["consecutive_misses"] += 1
                
        # Inicializar nuevos tracks para detecciones no emparejadas
        for d_idx, det in enumerate(detections_frame):
            if d_idx in matched_detections:
                continue
            track_counter += 1
            active_tracks.append({
                "tracking_id": f"T{track_counter:03d}",
                "history": [det],
                "last_centroid": det["centroid_mm"],
                "frames": [fid],
                "consecutive_misses": 0
            })
            
        # Filtrar y archivar tracks que salieron de vista (X < -110 mm o demasiados fallos)
        still_active = []
        for track in active_tracks:
            pred_x_exit = track["last_centroid"][0] - dx_belt
            if pred_x_exit < -110.0 or track["consecutive_misses"] >= 5:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

    # Archivar tracks restantes al finalizar
    finished_tracks.extend(active_tracks)
    
    # 3. Consolidación de Votación Temporal
    consolidated_output = {}
    print(f"[DEBUG] Entrando a la fase de consolidación. Tracks totales: {len(finished_tracks)}", flush=True)
    for track_idx, track in enumerate(finished_tracks):
        tid = track["tracking_id"]
        print(f"[DEBUG] Consolidando Track {track_idx+1}/{len(finished_tracks)} (ID: {tid})", flush=True)
        history = track["history"]
        if not history:
            continue
            
        # Calcular pesos Gaussianos basados en distancia al centro óptico para ponderación
        weights = []
        for h in history:
            cx_mm, cy_mm = h["centroid_mm"]
            dist_center = math.sqrt(cx_mm**2 + cy_mm**2)
            # Desviación típica sigma = 45.0 mm (peso decae rápidamente fuera de la zona de +/- 45mm del centro)
            w = math.exp(-(dist_center**2) / (2.0 * (45.0**2)))
            # Evitar peso cero puro
            weights.append(max(0.01, w))
            
        sum_weights = sum(weights)
        
        # Subconjunto de los últimos N frames para inferencia frontal (cinta con punto de corte, sin oclusiones)
        N_FRONTAL = 5
        sub_history_frontal = history[-N_FRONTAL:]
        frontal_weights = weights[-len(sub_history_frontal):]
        
        # Votación Ponderada del Color con Zona de Exclusión Cromática sobre todo el historial (priorizando el centro de la cámara)
        color_weights = {}
        for idx, h in enumerate(history):
            bbox = h["bbox_cen"]
            cx_norm = (bbox[0] + bbox[2]) * 0.5
            cy_norm = (bbox[1] + bbox[3]) * 0.5
            
            # Si el centroide está dentro de la zona central de confianza (excluyendo márgenes de 12%)
            if 0.12 <= cx_norm <= 0.88 and 0.12 <= cy_norm <= 0.88:
                c_name = h["color"]
                color_weights[c_name] = color_weights.get(c_name, 0.0) + weights[idx]
                
        # Fallback si todas las observaciones cayeron en la zona de exclusión
        if not color_weights:
            for idx, h in enumerate(history):
                c_name = h["color"]
                color_weights[c_name] = color_weights.get(c_name, 0.0) + weights[idx]
                
        final_color = max(color_weights.keys(), key=lambda k: color_weights[k])

        
        # Promedio Ponderado por Confianza de la Referencia y Pose usando el clasificador neuro-simbólico EfficientNetV2-B0
        best_cen_obs = min(history, key=lambda h: math.sqrt(h["centroid_mm"][0]**2 + h["centroid_mm"][1]**2))
        best_lat_obs = max(sub_history_frontal, key=lambda h: h["area_lat"])
        
        crop_cen = best_cen_obs["crop_cen"]
        mask_cen = best_cen_obs["mask_cen"]
        crop_lat = best_lat_obs.get("crop_lat")
        mask_lat = best_lat_obs.get("mask_lat")
        area_cen = best_cen_obs["area_cen"]
        
        preds = efficientnet_clf.classify(
            crop_cen=crop_cen,
            mask_cen=mask_cen,
            crop_lat=crop_lat,
            mask_lat=mask_lat,
            area_cenital=area_cen
        )
        
        if preds:
            # Evolución Técnica: Desempate Geométrico por Contornos
            if contour_matcher:
                # Evaluar match_contour para el top-3 de candidatos
                top_candidates = preds[:3]
                candidates_with_contour = []
                for cand in top_candidates:
                    part_ref = cand["part_ref"]
                    pose_index = cand["pose_index"]
                    
                    # Normalizar bboxes para match_contour
                    bbox_cen_abs = best_cen_obs["bbox_cen"] # [x1, y1, x2, y2] en px
                    h_img, w_img = mask_cen.shape[:2]
                    bbox_cen_norm = [
                        bbox_cen_abs[0] / w_img,
                        bbox_cen_abs[1] / h_img,
                        bbox_cen_abs[2] / w_img,
                        bbox_cen_abs[3] / h_img
                    ]
                    
                    bbox_lat_norm = None
                    if mask_lat is not None and "bbox_lat" in best_lat_obs:
                        bbox_lat_abs = best_lat_obs["bbox_lat"]
                        h_lat_img, w_lat_img = mask_lat.shape[:2]
                        bbox_lat_norm = [
                            bbox_lat_abs[0] / w_lat_img,
                            bbox_lat_abs[1] / h_lat_img,
                            bbox_lat_abs[2] / w_lat_img,
                            bbox_lat_abs[3] / h_lat_img
                        ]
                    
                    try:
                        contour_score, yaw_angle = contour_matcher.match_contour(
                            part_ref=part_ref,
                            pose_index=pose_index,
                            mask_cen=mask_cen,
                            bbox_cen_norm=bbox_cen_norm,
                            img_res_px_cen=w_img,
                            mask_lat=mask_lat,
                            bbox_lat_norm=bbox_lat_norm,
                            img_res_px_lat=mask_lat.shape[1] if mask_lat is not None else 2048
                        )
                    except Exception as e:
                        log.warning(f"Error en match_contour para {part_ref} pose {pose_index}: {e}")
                        contour_score = 0.0
                        yaw_angle = 0.0
                        
                    # Fusión: 40% Classifier score + 60% Contour score
                    combined = 0.4 * cand["score"] + 0.6 * contour_score
                    candidates_with_contour.append({
                        "part_ref": part_ref,
                        "pose_index": pose_index,
                        "score": cand["score"],
                        "contour_score": contour_score,
                        "combined_score": combined,
                        "yaw_angle": yaw_angle
                    })
                
                candidates_with_contour.sort(key=lambda x: x["combined_score"], reverse=True)
                
                # Reportar desempate si cambia el top
                orig_top = preds[0]["part_ref"]
                new_top = candidates_with_contour[0]["part_ref"]
                if orig_top != new_top:
                    log.info(f"  [{tid}] [Desempate Contorno] Cambió predicción: {orig_top} -> {new_top} "
                             f"(Contour IoU: {candidates_with_contour[0]['contour_score']:.4f} vs "
                             f"{next((c['contour_score'] for c in candidates_with_contour if c['part_ref'] == orig_top), 0.0):.4f})")
                
                final_ref = new_top
                final_pose = candidates_with_contour[0]["pose_index"]
                final_score = candidates_with_contour[0]["combined_score"]
            else:
                best_pred = preds[0]
                final_ref = best_pred["part_ref"]
                final_pose = best_pred["pose_index"]
                final_score = best_pred["score"]
        else:
            final_ref, final_pose, final_score = "Unknown", -1, 0.0
            
        # Ponderación Gaussiana para Altura (basado únicamente en los últimos N frames)
        valid_heights_data = [(h["height"], frontal_weights[idx]) for idx, h in enumerate(sub_history_frontal) if h.get("height_valid", True)]
        if valid_heights_data:
            w_sum_h = sum(vh[0]*vh[1] for vh in valid_heights_data)
            w_den_h = sum(vh[1] for vh in valid_heights_data)
            final_height = round(float(w_sum_h / w_den_h), 2)
        else:
            final_height = 9.6
            if final_ref != "Unknown" and final_ref in poses_db:
                for p in poses_db[final_ref]:
                    if p.pose_index == final_pose:
                        final_height = p.lateral_height or p.effective_height or 9.6
                        break
            
        # Ponderación Gaussiana para Áreas (Cenital usa todo, Lateral usa los últimos N frames)
        weighted_area_cen = round(float(sum(h["area_cen"]*weights[idx] for idx, h in enumerate(history)) / sum_weights), 2)
        weighted_area_lat = round(float(sum(h["area_lat"]*frontal_weights[idx] for idx, h in enumerate(sub_history_frontal)) / sum(frontal_weights)), 2)
            
        consolidated_output[tid] = {
            "tracking_id": tid,
            "referencia_detectada": final_ref,
            "color": final_color,
            "pose_identificada": final_pose,
            "score": round(final_score, 3),
            "frames_visible": track["frames"],
            "confidence_details": {
                "num_observations": len(history),
                "average_area_cen": weighted_area_cen,
                "average_area_lat": weighted_area_lat,
                "average_height": final_height
            },
            "history": [
                {
                    "frame_id": track["frames"][obs_idx],
                    "centroid_mm": h["centroid_mm"],
                    "bbox_cen": h["bbox_cen"],
                    "bbox_lat": h["bbox_lat"],
                    "color": h["color"],
                    "color_cenital": h.get("color_cenital", h["color"]),
                    "color_lateral": h.get("color_lateral", "N/A"),
                    "area_cen": round(h["area_cen"], 2),
                    "area_lat": round(h["area_lat"], 2),
                    "height": round(h["height"], 2),
                    "height_valid": h.get("height_valid", True),
                    "studs": h["studs"]
                }
                for obs_idx, h in enumerate(history)
            ]
        }
        
    # Guardar en archivo JSON final
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_output, f, indent=4, ensure_ascii=False)
        
    log.info(f"Pipeline completado. Resultados consolidados guardados en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Inferencia Secuencial con Tracking.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(project_root, "data", "frames"))
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "logs", "inferencia_consolidada.json"))
    parser.add_argument("--belt_speed", type=float, default=83.3, help="Velocidad de cinta en mm/s (por defecto 83.3).")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames por segundo de captura (por defecto 5.0).")
    parser.add_argument("--max_frames", type=int, default=None, help="Número máximo de frames a procesar.")
    parser.add_argument("--mode", type=str, default="CLASSIC", choices=["CLASSIC", "HYBRID", "POSE_ONLY"], help="Modo de inferencia")
    args = parser.parse_args()
    
    # Comprobación de directorio de frames absoluto de reserva
    data_dir_path = args.data_dir
    if not os.path.exists(data_dir_path):
        data_dir_path = "/data/frames/"
        
    run_pipeline(data_dir_path, args.output, args.belt_speed, args.fps, args.max_frames, args.mode)
