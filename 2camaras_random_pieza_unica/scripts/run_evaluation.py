# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/run_evaluation.py
==================================================
Pipeline de inferencia y evaluación para el setup de pieza única con 2 cámaras.

Algoritmo de Decisión en Cascada (rediseño v4 — observador/comparador separados):
  - Fase 1: Gating de Color (Cenital) — Estimación Dual CIELAB.
  - Observación A: altura lateral medida (cámara lateral + setup).
  - Observación B: superficie cenital aparente (UNA por imagen, sin
    presuponer conocimiento del candidato; usa la altura lateral medida).
  - Fase 2: Surface gating gaussiano. El comparador conoce el candidato y
    PREDICE qué área aparente se vería si la hipótesis fuera cierta. Score
    = exp(-residual²/(2σ²)) con σ adaptativo a la altura hipotética.
  - Fase 3: Gating de Altura Lateral (±15%).
  - Fase 4: Fusión DINOv2 (Cenital 70% + Lateral 30%).

CONSIGNA: el observador NO presupone conocimiento de la pieza a estimar;
solo usa información de la cámara y del setup conocido (cinta + cámaras).
La información del candidato entra únicamente en el comparador (Fases 2-4).

Detección de piezas mediante modelos YOLO específicos por cámara.
Estimación de color mediante algoritmo dual (recorte completo + segmentado).
"""
import os, sys, json, math
import time as _time
import numpy as np
from PIL import Image
from ultralytics import YOLO, SAM

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
# IMPORTANT: project_root must be FIRST so our config_loader.py takes priority
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from config_loader import cfg
from inference.knn_classifier import LegoKNNClassifier, get_knn_classifier, FALLBACK_FOOTPRINT_MM
from inference.api import PART_HEIGHTS_MM
from database.set_catalog import REAL_SETS

from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("pipeline")

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


# 7 colores reales del inventario del set 75078-1 (BrickLink codes).
# El comparador CIELAB elige el mas cercano al RGB observado.
SET_CATALOG_COLORS = [
    {"color_code": "1",  "color_name": "White",             "color_hex": "#FFFFFF", "rgb": [242, 243, 242]},
    {"color_code": "5",  "color_name": "Red",               "color_hex": "#C91A09", "rgb": [201,  26,   9]},
    {"color_code": "11", "color_name": "Black",             "color_hex": "#1B1B1B", "rgb": [ 27,  42,  52]},
    {"color_code": "13", "color_name": "Trans-Brown",       "color_hex": "#583927", "rgb": [101,  77,  47]},
    {"color_code": "17", "color_name": "Trans-Red",         "color_hex": "#C91A09", "rgb": [200,  85,  61]},
    {"color_code": "85", "color_name": "Dark Bluish Gray",  "color_hex": "#646464", "rgb": [ 99,  95,  97]},
    {"color_code": "86", "color_name": "Light Bluish Gray", "color_hex": "#A0A5A9", "rgb": [160, 165, 169]},
]


def find_closest_catalog_color(avg_rgb):
    """Busca el color del set 75078-1 mas similar a avg_rgb en CIELAB."""
    avg_lab = rgb_to_lab(avg_rgb)
    best_match = SET_CATALOG_COLORS[0]
    min_dist = float("inf")
    for sc in SET_CATALOG_COLORS:
        sc_lab = rgb_to_lab(sc["rgb"])
        dist = np.linalg.norm(avg_lab - sc_lab)
        if dist < min_dist:
            min_dist = dist
            best_match = sc
    return best_match


def estimate_color_predominant(crop_img, use_segmentation=False):
    """Estima el color de la pieza.
    Si use_segmentation es True, usa sólo píxeles de la máscara segmentada.
    Si es False, usa todos los píxeles descartando el fondo azul petróleo de la cinta."""
    try:
        import cv2
        img_rgb = np.array(crop_img.convert("RGB"))

        if use_segmentation:
            mask = segment_crop(crop_img)
            mask_fg = mask > 0
        else:
            # Usar todos los píxeles pero descartar fondo azul petróleo
            bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
            dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_color, axis=-1)
            mask_fg = dist > 18.0

        if not np.any(mask_fg):
            mask_fg = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s_channel = img_hsv[mask_fg, 1]
        v_channel = img_hsv[mask_fg, 2]

        # Filtrar sombras (V < 40) y reflejos de luz (V > 235 o saturación muy baja)
        valid_mask = (v_channel >= 40) & (v_channel <= 235) & (s_channel >= 20)

        if np.sum(valid_mask) > 10:
            avg_rgb = img_rgb[mask_fg][valid_mask].mean(axis=0)
        else:
            avg_rgb = img_rgb[mask_fg].mean(axis=0)

        return avg_rgb
    except Exception as e:
        log.error(f"Error en estimate_color_predominant: {e}")
        return np.array([160.0, 165.0, 169.0])


# ── YOLO Inference Helper ──
def yolo_detect_bbox(model, img_path, conf_threshold=0.25):
    """Ejecuta inferencia YOLO y devuelve la bbox con mayor confianza como [x1, y1, x2, y2] normalizada.
    Retorna (None, 0.0) si no hay detección."""
    try:
        results = model(img_path, verbose=False, conf=conf_threshold)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Tomar la detección con mayor confianza
            best_idx = boxes.conf.argmax().item()
            # xyxyn = coordenadas normalizadas [0,1]
            bbox_norm = boxes.xyxyn[best_idx].cpu().numpy().tolist()
            conf = float(boxes.conf[best_idx].cpu().numpy())
            return bbox_norm, conf
    except Exception as e:
        log.warning(f"Error en YOLO inference: {e}")
    return None, 0.0


# ── Config ──
SELECTED_PARTS = cfg.pieces.selected_parts
PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
PX_PER_MM_LATERAL = cfg.inference.calibration.px_per_mm_lateral
CAMERA_DIST_MM = cfg.inference.calibration.camera_dist_mm


def get_part_dimensions(ref: str) -> tuple:
    dims_cfg = cfg.pieces.dimensions_mm
    if hasattr(dims_cfg, ref):
        d = getattr(dims_cfg, ref)
        return tuple(d)
    footprint = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
    height = PART_HEIGHTS_MM.get(ref, 9.6)
    return (max(footprint), min(footprint), height)


def get_nominal_heights(ref: str) -> list:
    dims = get_part_dimensions(ref)
    L, W, H = sorted(dims, reverse=True)
    return [H + 0.9, H, W, L]


sam_model = None

def get_sam_model():
    global sam_model
    if sam_model is None:
        sam_model = SAM("mobile_sam.pt")
    return sam_model


def segment_crop_sam(img_full: Image.Image, bbox_norm: list) -> np.ndarray:
    try:
        model = get_sam_model()
        w, h = img_full.size
        x1 = max(0, int(bbox_norm[0] * w))
        y1 = max(0, int(bbox_norm[1] * h))
        x2 = min(w, int(bbox_norm[2] * w))
        y2 = min(h, int(bbox_norm[3] * h))
        bbox_px = [x1, y1, x2, y2]
        
        img_np = np.array(img_full)
        results = model(img_np, bboxes=[bbox_px], verbose=False)
        if results and results[0].masks is not None:
            full_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            crop_mask = full_mask[y1:y2, x1:x2]
            return crop_mask
    except Exception:
        pass
    # Fallback to simple color distance
    try:
        import cv2
        w, h = img_full.size
        x1 = max(0, int(bbox_norm[0] * w))
        y1 = max(0, int(bbox_norm[1] * h))
        x2 = min(w, int(bbox_norm[2] * w))
        y2 = min(h, int(bbox_norm[3] * h))
        crop_img = img_full.crop((x1, y1, x2, y2))
        img_np = np.array(crop_img.convert("RGB"))
        bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
        dist = np.linalg.norm(img_np.astype(np.float32) - bg_color, axis=2)
        mask = (dist > 18.0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    except Exception:
        h_crop = max(1, int(bbox_norm[3] * 640) - int(bbox_norm[1] * 640))
        w_crop = max(1, int(bbox_norm[2] * 640) - int(bbox_norm[0] * 640))
        return np.ones((h_crop, w_crop), dtype=np.uint8) * 255


def estimate_color_predominant_sam(crop_img: Image.Image, mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
        img_rgb = np.array(crop_img.convert("RGB"))
        mask_fg = mask > 0
        if not np.any(mask_fg):
            mask_fg = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s_channel = img_hsv[mask_fg, 1]
        v_channel = img_hsv[mask_fg, 2]

        valid_mask = (v_channel >= 40) & (v_channel <= 235) & (s_channel >= 20)
        if np.sum(valid_mask) > 10:
            avg_rgb = img_rgb[mask_fg][valid_mask].mean(axis=0)
        else:
            avg_rgb = img_rgb[mask_fg].mean(axis=0)
        return avg_rgb
    except Exception:
        return np.array([160.0, 165.0, 169.0])


# ─────────────────────────────────────────────────────────────────
# Constantes geométricas del setup cenital (idénticas para todas
# las muestras; NO dependen de la pieza observada).
# Cám cenital en mm (0, 0, 150) → centro óptico.
# Focal sensor: 27 mm @ 36 mm sensor → focal_px = 27 * 640 / 36 = 480 px.
CAM_CEN_Z_MM = 150.0
CAM_CEN_FOCAL_PX = 480.0
IMG_RES_PX = 640.0
IMG_CENTER_PX = 320.0
PX_PER_MM_NOMINAL = 3.2  # @ Z = 0 (plano de la cinta)


def _bbox_centroid_xy_mm(bbox_norm: list) -> tuple:
    """Posición XY (mm) del centro del bbox cenital sobre el plano de la cinta,
    asumiendo proyección ortográfica nominal. Es una estimación de offset
    radial, NO requiere conocer la pieza."""
    cx_norm = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy_norm = (bbox_norm[1] + bbox_norm[3]) / 2.0
    cx_px = cx_norm * IMG_RES_PX
    cy_px = cy_norm * IMG_RES_PX
    dx_mm = (cx_px - IMG_CENTER_PX) / PX_PER_MM_NOMINAL
    dy_mm = (IMG_CENTER_PX - cy_px) / PX_PER_MM_NOMINAL
    return (dx_mm, dy_mm)


def observe_zenithal_surface_mm2(
    mask_cen: np.ndarray,
    bbox_cen_norm: list,
    measured_lateral_height_mm: float,
) -> dict:
    """OBSERVADOR PURO: estima la superficie aparente en mm² que el sistema
    está viendo en la cámara cenital, **sin presuponer conocimiento de la pieza**.

    Entradas:
      - mask_cen: máscara SAM cenital (observación de la cámara).
      - bbox_cen_norm: bbox normalizada cenital (observación de la cámara).
      - measured_lateral_height_mm: altura medida por la cámara lateral
        (observación, NO altura del candidato).

    Salida (dict):
      - "apparent_area_mm2"   : área tal cual la ve la cámara (sin des-magnificar).
      - "footprint_area_mm2"  : área proyectada al plano del suelo (Z=0),
                                des-magnificando por la altura medida.
      - "r_mm"                : offset radial del centroide al eje óptico.
      - "z_eff_mm"            : altura efectiva usada (mitad de la lateral).
      - "px_per_mm_local"     : calibración local en el plano del suelo.

    Observa que devuelve **un único valor por imagen**: NO itera sobre candidatos.
    """
    try:
        num_pixels = float(np.sum(mask_cen > 0))
        if num_pixels < 1.0:
            return {
                "apparent_area_mm2": 0.0,
                "footprint_area_mm2": 0.0,
                "r_mm": 0.0,
                "z_eff_mm": 0.0,
                "px_per_mm_local": PX_PER_MM_NOMINAL,
            }

        # 1) Posición radial del centroide en el plano de la cinta.
        dx_mm, dy_mm = _bbox_centroid_xy_mm(bbox_cen_norm)
        r_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

        # 2) Altura efectiva: mitad de la altura lateral medida (centro de masa
        #    vertical del cuerpo). Esto NO usa info del candidato; usa la
        #    medida directa de la cámara lateral.
        z_eff_mm = max(0.5, measured_lateral_height_mm * 0.5)

        # 3) Calibración local en el plano del SUELO (Z=0). Distancia 3D del
        #    centro óptico al punto (X=dx, Y=dy, Z=0).
        d_floor = math.sqrt(r_mm * r_mm + CAM_CEN_Z_MM * CAM_CEN_Z_MM)
        px_per_mm_floor = CAM_CEN_FOCAL_PX / d_floor

        # 4) Área aparente "bruta": qué área en mm² ocuparía cada píxel si
        #    estuviera apoyado a Z=0 (calibración del plano del suelo).
        area_apparent_floor_mm2 = num_pixels / (px_per_mm_floor ** 2)

        # 5) Des-magnificación por altura: la silueta vista corresponde a un
        #    contenido elevado a Z=z_eff, que se ve más grande por
        #    factor (Zcam / (Zcam - Z)). Para volver al plano del suelo:
        #    factor lineal = (Zcam - Z) / Zcam, cuadrado en área.
        demag_linear = (CAM_CEN_Z_MM - z_eff_mm) / CAM_CEN_Z_MM
        demag_area = demag_linear * demag_linear
        footprint_area_mm2 = area_apparent_floor_mm2 * demag_area

        return {
            "apparent_area_mm2": float(area_apparent_floor_mm2),
            "footprint_area_mm2": float(max(0.5, footprint_area_mm2)),
            "r_mm": float(r_mm),
            "z_eff_mm": float(z_eff_mm),
            "px_per_mm_local": float(px_per_mm_floor),
        }
    except Exception as e:
        # Fallback paraxial sencillo
        n = float(np.sum(mask_cen > 0))
        return {
            "apparent_area_mm2": n / (PX_PER_MM_NOMINAL ** 2),
            "footprint_area_mm2": n / (PX_PER_MM_NOMINAL ** 2),
            "r_mm": 0.0,
            "z_eff_mm": 0.0,
            "px_per_mm_local": PX_PER_MM_NOMINAL,
        }


def predict_apparent_zenith_area_mm2(
    nominal_footprint_mm2: float,
    nominal_height_mm: float,
    bbox_cen_norm: list,
) -> float:
    """COMPARADOR: dado un candidato (footprint nominal + altura nominal de
    una configuración hipotética), predice qué superficie aparente vería la
    cámara cenital si esa hipótesis fuera cierta.

    Esta función SÍ usa información del candidato (es su trabajo). Se aplica
    en el bucle de gating, no en el observador.

    Modelo:
      apparent ≈ footprint_top_magnificada + side_faces_proyectadas
      footprint_top_magnificada = footprint_real * (Zcam / (Zcam - z_eff))²
      side_faces ≈ perímetro * altura * (r / Zcam) * 0.5
    """
    try:
        z_eff = max(0.5, nominal_height_mm * 0.5)
        dx_mm, dy_mm = _bbox_centroid_xy_mm(bbox_cen_norm)
        r_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

        mag_linear = CAM_CEN_Z_MM / max(1.0, CAM_CEN_Z_MM - z_eff)
        mag_area = mag_linear * mag_linear
        apparent_top = nominal_footprint_mm2 * mag_area

        # Perímetro aproximado (usando rectángulo equivalente del footprint)
        # Como no conocemos L, W exactos del candidato (solo footprint), usamos
        # un perímetro asumiendo cuadrado equivalente: perim ≈ 4 * sqrt(area)
        perim_approx = 4.0 * math.sqrt(max(1.0, nominal_footprint_mm2))
        # Solo las dos caras laterales más cercanas al eje óptico contribuyen
        # al área aparente; modelo paraxial:
        apparent_sides = perim_approx * nominal_height_mm * (r_mm / CAM_CEN_Z_MM) * 0.5

        return float(apparent_top + apparent_sides)
    except Exception:
        return float(nominal_footprint_mm2)


def measure_lateral_height_mm_sam(mask: np.ndarray) -> float:
    try:
        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            height_px = max(ys) - min(ys)
            return height_px / PX_PER_MM_LATERAL
    except Exception:
        pass
    return mask.shape[0] / PX_PER_MM_LATERAL


# ─────────────────────────────────────────────────────────────────
# v3 (Opción A): pipeline lateral simétrico al cenital
#   - Erosión 1px de la máscara (elimina halo bevel/AA).
#   - P50 del perfil de columnas (mediana, robusto a sombras).
#   - Magnificación 3D usando posición XY estimada del bbox cenital
#     y Z inicial (lateral_height GT si disponible, o 9.6 mm).
# Constantes geométricas de la cámara lateral.
# Cám lateral en BU (15, 0, 2.5) → mm (150, 0, 25); mira a (0,0,0).
# Cám cenital en mm (0, 0, 150) → centro óptico.
# Focal sensor: 27 mm @ 36 mm sensor → focal_px = 27 * 640 / 36 = 480 px.
CAM_LAT_X_MM_V3 = 150.0
CAM_LAT_Z_MM_V3 = 25.0
CAM_FOCAL_PX_V3 = 480.0


def _bbox_cen_xy_mm_v3(bbox_norm: list) -> tuple:
    """Centro del bbox cenital en coords [0,1] → posición XY (mm) en
    el plano de la cinta respecto al centro óptico."""
    cx_norm = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy_norm = (bbox_norm[1] + bbox_norm[3]) / 2.0
    cx_px = cx_norm * 640.0
    cy_px = cy_norm * 640.0
    return ((cx_px - 320.0) / 3.2, (320.0 - cy_px) / 3.2)


def estimate_lateral_height_mm_corrected_v3(
    mask_lat: np.ndarray,
    bbox_cen_norm: list,
    estimated_height_mm_initial: float = 9.6,
) -> tuple:
    """Versión v3 (Opción A): simétrica al pipeline cenital.

    Pasos:
      1. Erosión 1 px del mask SAM (elimina halo bevel/AA, ~6.7 % px).
      2. Perfil de columnas: mediana P50 del `(max_y - min_y + 1)` por
         columna (más robusta que max-min global o que P75/P90).
      3. Posición 3D estimada de la pieza: XY desde bbox cenital,
         Z = `estimated_height_mm_initial / 2` (centro de masa vertical).
      4. Distancia 3D real: `d_act = √((150-X)² + Y² + (25-Z)²)`.
         Calibración local: `px_per_mm_lat_local = 480 / d_act`.

    Devuelve `(altura_real_mm, magnificación, d_act_mm)`.
    """
    import cv2 as _cv2
    if mask_lat is None or mask_lat.size == 0 or not np.any(mask_lat):
        return (0.0, 1.0, 0.0)

    # 1) Erosión 1 px
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    mask_e = _cv2.erode(mask_lat, kernel, iterations=1)
    if mask_e.sum() < max(20, 0.3 * mask_lat.sum()):
        mask_e = mask_lat

    # 2) Perfil de columnas P50
    col_h = []
    for c in range(mask_e.shape[1]):
        ys = np.where(mask_e[:, c] > 0)[0]
        if len(ys) > 1:
            col_h.append(int(ys.max() - ys.min() + 1))
    if col_h:
        h_apparent_px = float(np.median(col_h))
    else:
        ys, _ = np.where(mask_lat > 0)
        h_apparent_px = (
            float(ys.max() - ys.min() + 1) if len(ys) > 0 else float(mask_lat.shape[0])
        )

    # 3) Posición 3D estimada
    px_mm, py_mm = _bbox_cen_xy_mm_v3(bbox_cen_norm)
    pz_mm = max(estimated_height_mm_initial / 2.0, 0.5)

    # 4) Distancia 3D y calibración local
    dx = CAM_LAT_X_MM_V3 - px_mm
    dy = -py_mm
    dz = CAM_LAT_Z_MM_V3 - pz_mm
    d_act = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_act < 1e-3:
        d_act = math.sqrt(CAM_LAT_X_MM_V3 ** 2 + CAM_LAT_Z_MM_V3 ** 2)
    px_per_mm_lat_local = CAM_FOCAL_PX_V3 / d_act
    h_real_mm = h_apparent_px / px_per_mm_lat_local
    d_nom = math.sqrt(CAM_LAT_X_MM_V3 ** 2 + CAM_LAT_Z_MM_V3 ** 2)
    mag = d_nom / d_act
    return (h_real_mm, mag, d_act)


def get_oriented_dims_mm_sam(mask: np.ndarray) -> tuple:
    try:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if valid_contours:
            all_pts = np.vstack(valid_contours)
            rect = cv2.minAreaRect(all_pts)
            (_, _), (w_px, h_px), _ = rect
            return max(w_px, h_px) / PX_PER_MM_CENITAL, min(w_px, h_px) / PX_PER_MM_CENITAL
    except Exception:
        pass
    return mask.shape[1] / PX_PER_MM_CENITAL, mask.shape[0] / PX_PER_MM_CENITAL


def size_score(max_query, min_query, ref, clf, cam_name="cenital"):
    if cam_name == "lateral":
        cand_height = PART_HEIGHTS_MM.get(ref, 3.2)
        ref_dim = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
        max_ref, min_ref = max(ref_dim), min(ref_dim)
        diff_height = abs(min_query - cand_height)
        dist_width = 0.0
        if max_query < min_ref:
            dist_width = min_ref - max_query
        elif max_query > max_ref:
            dist_width = max_query - max_ref
        score_height = math.exp(-(diff_height**2) / (2 * (1.5**2)))
        score_width = math.exp(-(dist_width**2) / (2 * (4.0**2)))
        return score_height * score_width
    else:
        ref_dim = FALLBACK_FOOTPRINT_MM.get(ref)
        if not ref_dim:
            return 0.5
        max_ref, min_ref = max(ref_dim), min(ref_dim)
        diff_max = abs(max_query - max_ref)
        diff_min = abs(min_query - min_ref)
        dist_size = math.sqrt(diff_max**2 + diff_min**2)
        return math.exp(-(dist_size**2) / (2 * (5.0**2)))


def classify_camera(crop_img, clf, valid_part_refs, max_query, min_query, cam_name="cenital"):
    if not clf._ref_embeddings:
        return {}

    canvas_size = 224
    scale_factor = 208.0 / 640.0
    w_p, h_p = crop_img.size
    if w_p > 0 and h_p > 0:
        new_w = max(1, int(w_p * scale_factor))
        new_h = max(1, int(h_p * scale_factor))
        resized = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
        clean_crop = canvas
    else:
        clean_crop = crop_img

    cam_id = 1 if cam_name == "cenital" else 2

    filtered = [
        r for r in clf._ref_embeddings
        if (r["face"] % 10 == cam_id) and (r["part_ref"] in valid_part_refs)
    ]
    if not filtered:
        filtered = [r for r in clf._ref_embeddings if (r["face"] % 10 == cam_id)]

    query_vec = clf._extract_embedding(clean_crop, size_info=(max_query, min_query))
    ref_matrix = np.stack([r["embedding"] for r in filtered])
    visual_scores = ref_matrix @ query_vec

    sz_scores = []
    for r in filtered:
        sc = size_score(max_query, min_query, r["part_ref"], clf, cam_name=cam_name)
        sz_scores.append(sc)

    combined = visual_scores * np.array(sz_scores)

    class_scores = {}
    for idx, r in enumerate(filtered):
        ref = r["part_ref"]
        score = float(combined[idx])
        if ref not in class_scores or score > class_scores[ref]:
            class_scores[ref] = score

    return class_scores


def main():
    # ── Argumentos CLI ──
    import argparse
    parser = argparse.ArgumentParser(description="Inferencia y evaluacion 2-cam pieza unica")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Ruta absoluta o relativa al JSON de metadata. "
                             "Por defecto: data/test_dual/test_metadata.json")
    parser.add_argument("--report", type=str, default=None,
                        help="Ruta del JSON de salida con resultados. "
                             "Por defecto: data/eval_report.json")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directorio raiz de las imagenes (si != dir(metadata)).")
    parsed_args, _ = parser.parse_known_args()

    if parsed_args.metadata:
        metadata_path = parsed_args.metadata
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(project_root, metadata_path)
        test_dir = (parsed_args.test_dir
                    if parsed_args.test_dir
                    else os.path.dirname(metadata_path))
    else:
        test_dir = os.path.join(project_root, "data", "test_dual")
        metadata_path = os.path.join(test_dir, "test_metadata.json")

    if parsed_args.report:
        report_path_arg = parsed_args.report
        if not os.path.isabs(report_path_arg):
            report_path_arg = os.path.join(project_root, report_path_arg)
    else:
        report_path_arg = os.path.join(project_root, "data", "eval_report.json")

    if not os.path.exists(metadata_path):
        log.error(f"Metadata no encontrada: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar stable_poses_cache para enriquecer con campos GT (silueta, h_lat).
    stable_cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    stable_cache = {}
    if os.path.isfile(stable_cache_path):
        try:
            with open(stable_cache_path, "r", encoding="utf-8") as fc:
                stable_cache = json.load(fc)
        except Exception:
            stable_cache = {}

    _t_eval_start = _time.perf_counter()
    log_execution_header(log, "run_evaluation.py",
                         test_dir=test_dir,
                         metadata=metadata_path,
                         report=report_path_arg,
                         num_samples=len(meta_data.get("renders", [])))

    # ── Cargar modelos YOLO específicos por cámara ──
    yolo_cenital_path = os.path.join(project_root, "models", "yolo_cenital.pt")
    yolo_lateral_path = os.path.join(project_root, "models", "yolo_lateral.pt")

    if os.path.exists(yolo_cenital_path):
        log.info(f"Cargando modelo YOLO cenital: {yolo_cenital_path}")
        yolo_cenital = YOLO(yolo_cenital_path)
    else:
        log.warning(f"Modelo YOLO cenital no encontrado en {yolo_cenital_path}. Se usará fallback de metadata.")
        yolo_cenital = None

    if os.path.exists(yolo_lateral_path):
        log.info(f"Cargando modelo YOLO lateral: {yolo_lateral_path}")
        yolo_lateral = YOLO(yolo_lateral_path)
    else:
        log.warning(f"Modelo YOLO lateral no encontrado en {yolo_lateral_path}. Se usará fallback de metadata.")
        yolo_lateral = None

    log.info("Cargando clasificador KNN + DINOv2...")
    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()

    if not clf.is_ready():
        log.error("Clasificador no está listo.")
        sys.exit(1)

    correct_count = 0
    total_count = 0
    part_stats = {}
    results = []
    yolo_detections_cenital = 0
    yolo_detections_lateral = 0

    for sample_idx, entry in enumerate(meta_data.get("renders", [])):
        ref_gt = entry["ref"]
        cameras_data = entry["cameras"]

        # 1. Cámara Cenital — Detección YOLO
        cen_meta = cameras_data.get("cenital")
        if not cen_meta:
            continue
        cen_path = os.path.join(test_dir, cen_meta["file_name"])
        if not os.path.exists(cen_path):
            continue
        img_cen_full = Image.open(cen_path).convert("RGB")
        iw, ih = img_cen_full.size

        # Inferencia YOLO cenital (con fallback a metadata)
        cen_bbox = None
        cen_yolo_conf = 0.0
        if yolo_cenital is not None:
            cen_bbox, cen_yolo_conf = yolo_detect_bbox(yolo_cenital, cen_path)

        if cen_bbox is not None:
            cx1, cy1, cx2, cy2 = cen_bbox
            yolo_detections_cenital += 1
        else:
            # Fallback: usar bbox del metadata ground truth
            cx1, cy1, cx2, cy2 = cen_meta["bbox_norm"]

        crop_cen = img_cen_full.crop((
            max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
            min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih))
        ))

        # 2. Cámara Lateral — Detección YOLO
        lat_meta = cameras_data.get("lateral")
        if not lat_meta:
            continue
        lat_path = os.path.join(test_dir, lat_meta["file_name"])
        if not os.path.exists(lat_path):
            continue
        img_lat_full = Image.open(lat_path).convert("RGB")
        liw, lih = img_lat_full.size

        # Inferencia YOLO lateral (con fallback a metadata)
        lat_bbox = None
        lat_yolo_conf = 0.0
        if yolo_lateral is not None:
            lat_bbox, lat_yolo_conf = yolo_detect_bbox(yolo_lateral, lat_path)

        if lat_bbox is not None:
            lx1, ly1, lx2, ly2 = lat_bbox
            yolo_detections_lateral += 1
        else:
            # Fallback: usar bbox del metadata ground truth
            lx1, ly1, lx2, ly2 = lat_meta["bbox_norm"]

        crop_lat = img_lat_full.crop((
            max(0, int(lx1 * liw)), max(0, int(ly1 * lih)),
            min(liw, int(lx2 * liw)), min(lih, int(ly2 * lih))
        ))

        # Segmentación SAM
        mask_cen = segment_crop_sam(img_cen_full, [cx1, cy1, cx2, cy2])
        mask_lat = segment_crop_sam(img_lat_full, [lx1, ly1, lx2, ly2])

        # ── ESTIMACIÓN DE COLOR DENTRO DEL CONTORNO SAM ──
        cen_est2_rgb = estimate_color_predominant_sam(crop_cen, mask_cen)
        cen_est2_catalog = find_closest_catalog_color(cen_est2_rgb)
        
        lat_est2_rgb = estimate_color_predominant_sam(crop_lat, mask_lat)
        lat_est2_catalog = find_closest_catalog_color(lat_est2_rgb)

        # Color de decisión para Phase 1: segmentación cenital
        color_code_cen = cen_est2_catalog["color_code"]
        color_code_lat = lat_est2_catalog["color_code"]

        # ── Regla de consenso entre cámaras ──
        # Si los códigos de color normalizados de cenital y lateral coinciden,
        # se da por bueno y se usa para el filtrado de embeddings.
        # Si NO coinciden, se marca internamente como ERROR de color, pero
        # por consigna se sigue usando el color CENITAL como decisión de
        # filtrado (se mantiene Phase 1 activa).
        colors_consensus_ok = (color_code_cen == color_code_lat)
        if colors_consensus_ok:
            color_consensus_status = "agree"
        else:
            color_consensus_status = "error_disagree"

        # Log de estimaciones de color
        log.info(
            f"  [Color] Cenital: est2_seg=[{cen_est2_rgb[0]:.0f},{cen_est2_rgb[1]:.0f},{cen_est2_rgb[2]:.0f}]"
            f"->{cen_est2_catalog['color_name']} (decisión: {color_code_cen})"
        )
        log.info(
            f"  [Color] Lateral: est2_seg=[{lat_est2_rgb[0]:.0f},{lat_est2_rgb[1]:.0f},{lat_est2_rgb[2]:.0f}]"
            f"->{lat_est2_catalog['color_name']}"
        )
        if colors_consensus_ok:
            log.info(
                f"  [Color] Consenso ✓ (cen={color_code_cen} == lat={color_code_lat}) "
                f"→ filtrado por color activo"
            )
        else:
            log.warning(
                f"  [Color] CONFLICTO ✗ ERROR (cen={color_code_cen} ≠ lat={color_code_lat}) "
                f"→ se mantiene cenital ({color_code_cen}) como decisión de filtrado"
            )

        # ── ALGORITMO EN CASCADA (rediseñado) ──
        # Principio: el observador NO presupone conocimiento de la pieza.
        # Produce UNA observación por imagen. Los candidatos solo entran en
        # el comparador (gating).

        # Phase 1: Color gating (usando estimación dual cenital)
        parts_in_set = [p for p in REAL_SETS["75078-1"]["parts"] if p["ref"] in SELECTED_PARTS]
        valid_by_color = [p["ref"] for p in parts_in_set if p["color_code"] == color_code_cen]
        if not valid_by_color:
            valid_by_color = [p["ref"] for p in parts_in_set]

        # ── OBSERVACIONES PURAS (independientes del candidato) ──
        # Obs A: altura lateral medida (cámara lateral + setup de cámaras)
        try:
            measured_height, _mag_lat, _d_act_lat = (
                estimate_lateral_height_mm_corrected_v3(
                    mask_lat, [cx1, cy1, cx2, cy2],
                    estimated_height_mm_initial=9.6,  # prior genérico, NO de candidato
                )
            )
        except Exception:
            measured_height = measure_lateral_height_mm_sam(mask_lat)
        if measured_height <= 0:
            measured_height = measure_lateral_height_mm_sam(mask_lat)

        # Obs B: superficie cenital observada (única para esta imagen)
        # Usa la altura lateral MEDIDA (no la del candidato).
        zen_obs = observe_zenithal_surface_mm2(
            mask_cen,
            [cx1, cy1, cx2, cy2],
            measured_lateral_height_mm=measured_height,
        )
        obs_apparent_area_mm2 = zen_obs["apparent_area_mm2"]
        obs_footprint_area_mm2 = zen_obs["footprint_area_mm2"]

        if sample_idx < 3:
            log.info(
                f"    [OBS] mask_pixels={int(np.sum(mask_cen > 0))} | "
                f"h_lat_meas={measured_height:.2f}mm | "
                f"r_centroid={zen_obs['r_mm']:.1f}mm | "
                f"z_eff={zen_obs['z_eff_mm']:.2f}mm | "
                f"area_apparent={obs_apparent_area_mm2:.1f}mm² | "
                f"area_footprint={obs_footprint_area_mm2:.1f}mm²"
            )

        # Phase 2: Surface gating (gaussian, comparador con candidatos)
        # El comparador SÍ conoce el candidato → predice qué área aparente
        # vería la cámara si esa hipótesis fuera cierta, y compara contra la
        # observación única.
        valid_by_surface = []
        surface_scores = {}
        for ref in valid_by_color:
            dims = get_part_dimensions(ref)
            L, W, H = sorted(dims, reverse=True)
            configs = [(L * W, H), (L * H, W), (W * H, L)]

            # Filling factor por familia (atributo del candidato, OK aquí).
            filling_factor = 1.0
            if ref in ["6141", "98138", "4032", "3062", "59900"]:
                filling_factor = 0.785
            elif ref == "2420":
                filling_factor = 0.75
            elif ref in ["3039", "3298", "3037", "3665", "85984", "54200", "11477", "15068"]:
                filling_factor = 0.85
            elif ref in ["2412", "2877"]:
                filling_factor = 0.92

            best_score = 0.0
            best_residual = float("inf")
            for nom_area, nom_h in configs:
                nom_footprint = nom_area * filling_factor
                # Predicción de área aparente para esta configuración candidata.
                target_apparent = predict_apparent_zenith_area_mm2(
                    nom_footprint, nom_h, [cx1, cy1, cx2, cy2]
                )
                # Score gaussiano blando (σ relativo, más laxo cuanto más
                # alta la pose hipotética → reconoce que poses verticales
                # tienen más incertidumbre intrínseca).
                sigma_rel = 0.20 + 0.012 * nom_h  # 20%@H=0; ~58%@H=32mm
                sigma = max(2.0, sigma_rel * target_apparent)
                residual = abs(obs_apparent_area_mm2 - target_apparent)
                score = math.exp(-(residual ** 2) / (2.0 * sigma * sigma))
                if score > best_score:
                    best_score = score
                    best_residual = residual

            surface_scores[ref] = best_score
            # Aceptamos en el gating si el score gaussiano supera 0.05
            # (~2.5σ); en lugar de descartar duro, queda como score blando.
            if best_score >= 0.05:
                valid_by_surface.append(ref)

            if sample_idx < 3 and ref == ref_gt:
                log.info(
                    f"    [DEBUG-SURF] Ref={ref}(GT) | obs_apparent={obs_apparent_area_mm2:.1f} | "
                    f"best_score={best_score:.3f} | residual={best_residual:.1f}mm²"
                )

        if not valid_by_surface:
            valid_by_surface = valid_by_color

        # Phase 3: Height gating (lateral, +/-15%)
        # La medida de altura ya está calculada arriba (observación única).
        valid_by_height = []
        for ref in valid_by_surface:
            nominals = get_nominal_heights(ref)
            for nom in nominals:
                if 0.85 * nom <= measured_height <= 1.15 * nom:
                    valid_by_height.append(ref)
                    break
        if not valid_by_height:
            valid_by_height = valid_by_surface

        # Phase 4: DINOv2 fusion (cenital 70% + lateral 30%)
        max_query_cen, min_query_cen = get_oriented_dims_mm_sam(mask_cen)
        max_query_lat, min_query_lat = get_oriented_dims_mm_sam(mask_lat)

        scores_cenital = classify_camera(crop_cen, clf, valid_by_height, max_query_cen, min_query_cen, cam_name="cenital")
        scores_lateral = classify_camera(crop_lat, clf, valid_by_height, max_query_lat, min_query_lat, cam_name="lateral")

        final_scores = {}
        for ref in valid_by_height:
            s_cen = scores_cenital.get(ref, 0.0)
            s_lat = scores_lateral.get(ref, 0.0)
            final_scores[ref] = 0.7 * s_cen + 0.3 * s_lat

        if final_scores:
            consensus_ref = max(final_scores, key=final_scores.get)
            consensus_score = final_scores[consensus_ref]
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
            f"[{sample_idx+1:02d}/{len(meta_data['renders'])}] GT={ref_gt:6s} "
            f"-> Pred={consensus_ref:6s}  {status}  "
            f"(score={consensus_score:.4f} | color={color_code_cen} | "
            f"h_meas={measured_height:.2f}mm | "
            f"yolo_cen={cen_yolo_conf:.2f} yolo_lat={lat_yolo_conf:.2f} | "
            f"valid_color={len(valid_by_color)} | valid_surf={len(valid_by_surface)} | valid_h={len(valid_by_height)})"
        )

        # ── Reporte ad-hoc completo: GT desde metadata + cache + obs ──
        # Datos GT del propio sample (rellenados en metadata)
        gt_pose_index = entry.get("pose_index")
        gt_face_class = entry.get("face_class")
        gt_color_code = entry.get("color_code")
        gt_color_name = entry.get("color_name")
        gt_color_hex = entry.get("color_hex")
        gt_silhouette_area = entry.get("zenith_silhouette_area_gt")
        gt_lateral_height = entry.get("lateral_height_gt")
        gt_effective_height = entry.get("effective_height_gt")

        # Si la metadata no trae los GTs (formato legacy), buscarlos en cache
        if (gt_silhouette_area is None or gt_lateral_height is None) and ref_gt in stable_cache:
            poses = stable_cache.get(ref_gt, [])
            # Buscar pose por pose_index si esta disponible
            target_pose = None
            if gt_pose_index is not None:
                for p in poses:
                    if p.get("pose_index") == gt_pose_index or p.get("original_pose_index") == gt_pose_index:
                        target_pose = p
                        break
            if target_pose is None and poses:
                target_pose = poses[0]
            if target_pose:
                gt_silhouette_area = gt_silhouette_area or target_pose.get("zenith_silhouette_area")
                gt_lateral_height = gt_lateral_height or target_pose.get("lateral_height")
                gt_effective_height = gt_effective_height or target_pose.get("effective_height")
                gt_face_class = gt_face_class or target_pose.get("face_class")

        # Errores relativos (vs DB ground truth). Usamos footprint para superficie.
        def _rel_err(meas, gt):
            if gt is None or gt == 0:
                return None
            try:
                return round((float(meas) - float(gt)) / float(gt) * 100.0, 2)
            except Exception:
                return None

        surface_err_pct = _rel_err(obs_footprint_area_mm2, gt_silhouette_area)
        lateral_h_err_pct = _rel_err(measured_height, gt_lateral_height)

        # Match de color (cenital, lateral) vs GT
        color_match_cen = (str(cen_est2_catalog["color_code"]) == str(gt_color_code)) if gt_color_code is not None else None
        color_match_lat = (str(lat_est2_catalog["color_code"]) == str(gt_color_code)) if gt_color_code is not None else None

        results.append({
            "index": sample_idx,
            "sample_index": entry.get("index", sample_idx),
            "cenital_file": cen_meta.get("file_name"),
            "lateral_file": lat_meta.get("file_name"),
            # --- Pieza GT ---
            "ref_gt": ref_gt,
            "pose_index_gt": gt_pose_index,
            "face_class_gt": gt_face_class,
            # --- Pieza inferida ---
            "ref_inferred": consensus_ref,
            "model_match": is_correct,
            "consensus_score": round(consensus_score, 4),
            # --- Color ---
            "color_code_gt": gt_color_code,
            "color_name_gt": gt_color_name,
            "color_hex_gt": gt_color_hex,
            "color_cenital_rgb_est": [round(float(v), 1) for v in cen_est2_rgb.tolist()],
            "color_cenital_normalized_code": cen_est2_catalog["color_code"],
            "color_cenital_normalized_name": cen_est2_catalog["color_name"],
            "color_lateral_rgb_est": [round(float(v), 1) for v in lat_est2_rgb.tolist()],
            "color_lateral_normalized_code": lat_est2_catalog["color_code"],
            "color_lateral_normalized_name": lat_est2_catalog["color_name"],
            "color_match_cenital": color_match_cen,
            "color_match_lateral": color_match_lat,
            "color_decision_used": color_code_cen,
            "color_consensus_status": color_consensus_status,
            "color_consensus_ok": colors_consensus_ok,
            # --- Superficie cenital ---
            "surface_obs_apparent_mm2": round(obs_apparent_area_mm2, 2),
            "surface_obs_footprint_mm2": round(obs_footprint_area_mm2, 2),
            "surface_db_silhouette_mm2": (
                round(float(gt_silhouette_area), 2) if gt_silhouette_area is not None else None
            ),
            "surface_error_rel_pct": surface_err_pct,
            # --- Altura lateral ---
            "lateral_height_meas_mm": round(measured_height, 2),
            "lateral_height_db_mm": (
                round(float(gt_lateral_height), 2) if gt_lateral_height is not None else None
            ),
            "lateral_height_error_rel_pct": lateral_h_err_pct,
            "effective_height_db_mm": (
                round(float(gt_effective_height), 2) if gt_effective_height is not None else None
            ),
            # --- Senales auxiliares ---
            "yolo_conf_cenital": round(cen_yolo_conf, 3),
            "yolo_conf_lateral": round(lat_yolo_conf, 3),
            "valid_by_color_count": len(valid_by_color),
            "valid_by_surface_count": len(valid_by_surface),
            "valid_by_height_count": len(valid_by_height),
        })

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    _t_eval_end = _time.perf_counter()
    _duration = _t_eval_end - _t_eval_start

    log.info("")
    log.info("=" * 60)
    log.info("  RESULTADOS DE EVALUACIÓN — 2 CÁMARAS PIEZA ÚNICA")
    log.info("=" * 60)
    log.info(f"  Muestras totales : {total_count}")
    log.info(f"  Correctas        : {correct_count}")
    log.info(f"  Precisión global : {accuracy:.2f}%")
    log.info(f"  YOLO cenital     : {yolo_detections_cenital}/{total_count} detecciones")
    log.info(f"  YOLO lateral     : {yolo_detections_lateral}/{total_count} detecciones")
    log.info("-" * 60)
    log.info("  Precisión por pieza:")
    for part_ref, stats in sorted(part_stats.items()):
        pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        log.info(f"    {part_ref:8s}  {bar}  {pct:5.1f}%  ({stats['correct']}/{stats['total']})")
    log.info("=" * 60)

    report_path = report_path_arg
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump({
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": round(accuracy, 2),
            "render_engine": "BLENDER_EEVEE",
            "resolution": "640x640",
            "metadata_path": metadata_path,
            "test_dir": test_dir,
            "yolo_detections_cenital": yolo_detections_cenital,
            "yolo_detections_lateral": yolo_detections_lateral,
            "part_stats": part_stats,
            "results": results,
        }, rf, indent=2, ensure_ascii=False)

    log_execution_footer(log, "run_evaluation.py",
                         duration_s=_duration,
                         accuracy_pct=f"{accuracy:.2f}%",
                         correct=correct_count,
                         total=total_count,
                         report=report_path)


if __name__ == "__main__":
    main()
