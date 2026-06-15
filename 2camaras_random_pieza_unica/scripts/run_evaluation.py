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

# Observador 6-DoF basado en YOLO-Pose + triangulacion 2-view (Fase 5).
# Si los pesos `yolo_<cam>_pose.pt` no existen, el pipeline cae al
# observador SAM-bbox tradicional (`apparent_area_mm2`, `lateral_height`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from _kpts_observer import (
        kpts_observer as kpts_observer_fn,
        extract_yolo_pose_keypoints,
    )
    HAS_KPTS_MODULE = True
except Exception as _e:
    HAS_KPTS_MODULE = False
    print(f"[WARN] _kpts_observer no disponible: {_e}")

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


# Paleta de colores para comparación CIELAB.
# Se intenta cargar dinámicamente desde la BD (piece_embeddings),
# incluyendo todos los colores con embeddings indexados.
# Fallback: cfg.pieces.set_colors (lista estática del config.yaml).
CCM_CEN = np.array(cfg.inference.color_calibration.ccm_cenital)
CCM_LAT = np.array(cfg.inference.color_calibration.ccm_lateral)

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    return np.array([128.0, 128.0, 128.0])

def _build_catalog_colors_from_db():
    """Construye la paleta de colores desde piece_embeddings (BD dinámica).
    Retorna lista de dicts con keys: color_code, color_name, color_hex, rgb (aplicando CCM).
    Si falla, retorna lista vacía (se usa fallback cfg.pieces.set_colors)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from _db_helpers import get_unique_colors_from_db
        colors_db = get_unique_colors_from_db()
        if not colors_db:
            return [], []
        catalog_cen, catalog_lat = [], []
        seen = set()
        for c in colors_db:
            code_str = str(c["code"])
            hex_color = c["hex"] if c["hex"].startswith("#") else "#" + c["hex"]
            name = c["name"]
            key = (code_str, hex_color)
            if key in seen:
                continue
            seen.add(key)
            rgb_nominal = hex_to_rgb(hex_color)
            rgb_cen = np.clip(CCM_CEN @ rgb_nominal, 0.0, 255.0).tolist()
            rgb_lat = np.clip(CCM_LAT @ rgb_nominal, 0.0, 255.0).tolist()
            entry_cen = {"color_code": code_str, "color_name": name, "color_hex": hex_color, "rgb": rgb_cen}
            entry_lat = {"color_code": code_str, "color_name": name, "color_hex": hex_color, "rgb": rgb_lat}
            catalog_cen.append(entry_cen)
            catalog_lat.append(entry_lat)
        log.info(f"[ColorCatalog] {len(catalog_cen)} colores cargados desde BD dinámica")
        return catalog_cen, catalog_lat
    except Exception as e:
        log.warning(f"[ColorCatalog] Error cargando desde BD: {e}")
        return [], []

SET_CATALOG_COLORS = []
SET_CATALOG_COLORS_LATERAL = []

# Cargar paleta calibrada Z=300mm si está disponible
palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
if os.path.exists(palette_path):
    try:
        with open(palette_path, "r", encoding="utf-8") as f:
            palette_items = json.load(f)
        for item in palette_items:
            code_str = str(item["color_code"])
            name = item["color_name"]
            hex_color = item["color_hex"]
            SET_CATALOG_COLORS.append({
                "color_code": code_str, "color_name": name,
                "color_hex": hex_color, "rgb": item["rgb_cenital"]
            })
            SET_CATALOG_COLORS_LATERAL.append({
                "color_code": code_str, "color_name": name,
                "color_hex": hex_color, "rgb": item["rgb_lateral"]
            })
        log.info(f"[ColorCatalog] {len(SET_CATALOG_COLORS)} colores cargados desde la paleta calibrada: {palette_path}")
    except Exception as e:
        log.error(f"[ColorCatalog] Error cargando paleta calibrada: {e}")
        SET_CATALOG_COLORS = []
        SET_CATALOG_COLORS_LATERAL = []

if not SET_CATALOG_COLORS:
    # Intentar cargar desde BD primero
    _db_cen, _db_lat = _build_catalog_colors_from_db()
    if _db_cen:
        SET_CATALOG_COLORS = _db_cen
        SET_CATALOG_COLORS_LATERAL = _db_lat
    else:
        # Fallback: cfg.pieces.set_colors (config estático)
        log.info("[ColorCatalog] Fallback a cfg.pieces.set_colors estático")
        for sc in cfg.pieces.set_colors:
            code_str = str(sc["code"])
            hex_color = sc["hex"]
            name = sc["name"]
            rgb_nominal = hex_to_rgb(hex_color)
            rgb_cen = np.clip(CCM_CEN @ rgb_nominal, 0.0, 255.0).tolist()
            rgb_lat = np.clip(CCM_LAT @ rgb_nominal, 0.0, 255.0).tolist()
            SET_CATALOG_COLORS.append({
                "color_code": code_str, "color_name": name,
                "color_hex": hex_color, "rgb": rgb_cen
            })
            SET_CATALOG_COLORS_LATERAL.append({
                "color_code": code_str, "color_name": name,
                "color_hex": hex_color, "rgb": rgb_lat
            })


def find_closest_catalog_color(avg_rgb, camera="cenital"):
    """Busca el color del set 75078-1 mas similar a avg_rgb en CIELAB.

    `camera`: "cenital" (default) o "lateral" para usar el catalogo
    correspondiente recalibrado para esa vista.

    Retro-compatible: devuelve el dict del color top-1 (sin extras).
    Internamente almacena `_delta_e` (ΔE_lab al top-1) y `_runner_up`
    (segundo más cercano) en el propio dict para diagnóstico, sin romper
    el contrato de los llamantes que sólo leen color_code/color_hex/color_name.
    """
    catalog = SET_CATALOG_COLORS_LATERAL if camera == "lateral" else SET_CATALOG_COLORS
    avg_lab = rgb_to_lab(avg_rgb)
    distances = []
    for sc in catalog:
        sc_lab = rgb_to_lab(sc["rgb"])
        d = float(np.linalg.norm(avg_lab - sc_lab))
        distances.append((d, sc))
    distances.sort(key=lambda t: t[0])
    best_dist, best_match = distances[0]
    runner_up_dist, runner_up = (distances[1] if len(distances) > 1 else (None, None))
    # Anotar metadatos sin alterar la API existente.
    enriched = dict(best_match)
    enriched["_delta_e"] = round(best_dist, 2)
    if runner_up is not None:
        enriched["_runner_up_code"] = runner_up.get("color_code")
        enriched["_runner_up_name"] = runner_up.get("color_name")
        enriched["_runner_up_delta_e"] = round(runner_up_dist, 2)
        enriched["_margin"] = round(runner_up_dist - best_dist, 2)
    else:
        enriched["_margin"] = None
    return enriched


def estimate_color_predominant(crop_img, use_segmentation=False):
    """Estima el color de la pieza.
    Si use_segmentation es True, usa sólo píxeles de la máscara segmentada.
    Si es False, usa todos los píxeles descartando el fondo azul petróleo de la cinta."""
    try:
        import cv2
        img_rgb = np.array(crop_img.convert("RGB"))

        if use_segmentation:
            mask = segment_crop_sam(img_cen_full, [cx1, cy1, cx2, cy2])
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


# ── YOLO Inference Helpers ──
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


def yolo_detect_bbox_batch(model, img_paths, conf_threshold=0.25, batch_size=16):
    """OPT 3.1 — Inferencia YOLO en lotes. Devuelve lista [(bbox_norm|None, conf), ...]
    en el mismo orden que img_paths.
    
    Procesa los paths en chunks de batch_size para no saturar memoria GPU/MPS."""
    out = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start:start + batch_size]
        try:
            results = model(chunk, verbose=False, conf=conf_threshold)
        except Exception as e:
            log.warning(f"YOLO batch fallido [{start}:{start+len(chunk)}]: {e}")
            out.extend([(None, 0.0)] * len(chunk))
            continue
        for r in results:
            if r is not None and len(r.boxes) > 0:
                boxes = r.boxes
                best_idx = boxes.conf.argmax().item()
                bbox_norm = boxes.xyxyn[best_idx].cpu().numpy().tolist()
                conf = float(boxes.conf[best_idx].cpu().numpy())
                out.append((bbox_norm, conf))
            else:
                out.append((None, 0.0))
    return out


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
        # Belt color as rendered under Dome Light + Cross-Polarization
        # (medido en esquinas de los renders test20: ~(128,165,185)).
        bg_color = np.array([128.0, 165.0, 185.0], dtype=np.float32)
        dist = np.linalg.norm(img_np.astype(np.float32) - bg_color, axis=2)
        mask = (dist > 30.0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    except Exception:
        h_crop = max(1, int(bbox_norm[3] * 640) - int(bbox_norm[1] * 640))
        w_crop = max(1, int(bbox_norm[2] * 640) - int(bbox_norm[0] * 640))
        return np.ones((h_crop, w_crop), dtype=np.uint8) * 255


# Color cinta azul petroleo TAL Y COMO LO RENDERIZA Blender bajo
# Dome Light (NO el lineal 37,65,84, sino el resultado tras tone-mapping
# y view_transform=Standard). Medido en las esquinas de los renders
# del set test20 bajo Dome Light + Cross-Polarization (2026-06-13).
# Refs DINOv2 canonicas también bajo Dome Light → mismo fondo (128,165,185).
CINTA_BG_RGB = (128, 165, 185)


def neutralize_lab(arr: np.ndarray) -> np.ndarray:
    """Normalización LAB-neutral: elimina info de color (A, B) de los pixels
    de la pieza, preservando solo luminancia (forma, studs, bordes, sombras).

    El fondo (cinta azul petroleo) NO se toca — solo los pixels de la pieza.
    Esto hace al embedding DINOv2 invariante al color:
      - Refs neutralizadas + queries neutralizadas → mismo dominio geometría pura.
      - Evita confusión entre piezas similares de distinto color.
      - Mejora similitud query↔ref en colores cromáticos (+0.03-0.10).

    Args:
        arr: np.uint8 RGB array HxWx3 con fondo cinta (37,65,84).
    Returns:
        np.uint8 RGB array con pixels de pieza llevados a gris neutro (A=B=128).
    """
    try:
        import cv2 as _cv2
        # Fondo cinta tras tone-mapping (Dome Light): (128,165,185).
        bg = np.array([128.0, 165.0, 185.0], dtype=np.float32)
        piece_mask = np.linalg.norm(arr.astype(np.float32) - bg, axis=-1) > 20.0
        if not np.any(piece_mask):
            return arr
        lab = _cv2.cvtColor(arr, _cv2.COLOR_RGB2LAB)
        lab[piece_mask, 1] = 128   # canal A → crominancia cero
        lab[piece_mask, 2] = 128   # canal B → crominancia cero
        return _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)
    except Exception:
        return arr


def apply_sam_mask_to_crop(crop_img: Image.Image, mask: np.ndarray,
                            bg_color=CINTA_BG_RGB) -> Image.Image:
    """Reemplaza los pixels fuera de la mascara SAM con `bg_color`
    (default = cinta azul petroleo (37,65,84)).

    Esto simetriza el preprocess de inferencia con el de las refs DINOv2
    canonicas: las refs se renderizan con cinta visible, y aqui forzamos
    que en la query los pixeles fuera de la pieza sean tambien cinta
    plana, para que los embeddings query y ref vivan en el mismo dominio.

    Si la mascara es None o esta vacia, devuelve el crop sin tocar.
    """
    if mask is None or not np.any(mask > 0):
        return crop_img
    try:
        arr = np.array(crop_img.convert("RGB"))
        # Asegurar dimensiones compatibles (la mascara puede no ser
        # exactamente HxW del crop si vino de fallback).
        h, w = arr.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            import cv2 as _cv2
            mask = _cv2.resize(mask, (w, h), interpolation=_cv2.INTER_NEAREST)
        mask_bool = mask > 0
        arr[~mask_bool] = bg_color
        # Neutralización LAB: elimina color de la pieza, preserva geometría.
        # Simetriza con dinov2_refs_v4_canonical_neutral/ donde las refs
        # también tienen A=B=128 en los pixels de la pieza.
        arr = neutralize_lab(arr)
        return Image.fromarray(arr)
    except Exception:
        return crop_img


def estimate_color_predominant_sam(crop_img: Image.Image, mask: np.ndarray) -> np.ndarray:
    try:
        img_rgb = np.array(crop_img.convert("RGB"))
        mask_fg = mask > 0
        if not np.any(mask_fg):
            mask_fg = np.ones((img_rgb.shape[0], img_rgb.shape[1]), dtype=bool)
        
        pixels_fg = img_rgb[mask_fg]
        
        # Calculate luminance for each pixel to detect highlights
        lum = pixels_fg[:, 0] * 0.299 + pixels_fg[:, 1] * 0.587 + pixels_fg[:, 2] * 0.114
        
        # Trim the top 10% brightest pixels (specular reflections)
        thresh = np.percentile(lum, 90)
        keep = lum <= thresh
        
        if np.any(keep):
            avg_rgb = pixels_fg[keep].mean(axis=0)
        else:
            avg_rgb = pixels_fg.mean(axis=0)
            
        return avg_rgb
    except Exception:
        return np.array([160.0, 165.0, 169.0])

# ─────────────────────────────────────────────────────────────────
# Constantes geométricas del setup cenital — RESOLUCIÓN DINÁMICA.
# Los parámetros físicos son fijos; los parámetros en píxeles se
# derivan de la resolución real de la imagen de entrada.
_SENSOR_WIDTH_MM = 36.0
_CAM_CEN_FOCAL_MM = float(cfg.cameras.cenital.focal_length_mm)
CAM_CEN_Z_MM = float(cfg.inference.calibration.camera_dist_mm)

_actual_img_res = 640  # Updated dynamically per-sample from image dimensions

# Defaults a 640 (compatibilidad con código legacy que usa constantes globales).
# En runtime, las funciones recalculan dinámicamente según img_res_px.
IMG_RES_PX = 640.0
IMG_CENTER_PX = 320.0
CAM_CEN_FOCAL_PX = float((_CAM_CEN_FOCAL_MM / _SENSOR_WIDTH_MM) * IMG_RES_PX)
PX_PER_MM_NOMINAL = float(CAM_CEN_FOCAL_PX / CAM_CEN_Z_MM)


def _cenital_geom(img_res_px_val=None):
    """Devuelve (res, center, focal_px, px_per_mm) para la resolución dada.
    Si img_res_px_val es None, usa el default global (640)."""
    res = float(img_res_px_val) if img_res_px_val else IMG_RES_PX
    center = res / 2.0
    focal_px = (_CAM_CEN_FOCAL_MM / _SENSOR_WIDTH_MM) * res
    px_mm = focal_px / CAM_CEN_Z_MM
    return res, center, focal_px, px_mm











def _bbox_centroid_xy_mm(bbox_norm: list, img_res_px_val=None) -> tuple:
    """Posición XY (mm) del centro del bbox cenital sobre el plano de la cinta,
    asumiendo proyección ortográfica nominal. Es una estimación de offset
    radial, NO requiere conocer la pieza.
    `img_res_px_val`: resolución real de la imagen. Si None, usa default 640."""
    res, center, _, px_mm = _cenital_geom(img_res_px_val)
    cx_norm = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy_norm = (bbox_norm[1] + bbox_norm[3]) / 2.0
    cx_px = cx_norm * res
    cy_px = cy_norm * res
    dx_mm = (cx_px - center) / px_mm
    dy_mm = (center - cy_px) / px_mm
    return (dx_mm, dy_mm)


def observe_zenithal_surface_mm2(
    mask_cen: np.ndarray,
    bbox_cen_norm: list,
    measured_lateral_height_mm: float,
    img_res_px_val=None,
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
        dx_mm, dy_mm = _bbox_centroid_xy_mm(bbox_cen_norm, img_res_px_val)
        r_mm = math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)

        # 2) Altura efectiva: mitad de la altura lateral medida (centro de masa
        #    vertical del cuerpo). Esto NO usa info del candidato; usa la
        #    medida directa de la cámara lateral.
        z_eff_mm = max(0.5, measured_lateral_height_mm * 0.5)

        # 3) Calibración local en el plano del SUELO (Z=0). Distancia 3D del
        #    centro óptico al punto (X=dx, Y=dy, Z=0).
        d_floor = math.sqrt(r_mm * r_mm + CAM_CEN_Z_MM * CAM_CEN_Z_MM)
        _, _, _focal_px_dyn, _ = _cenital_geom(img_res_px_val)
        px_per_mm_floor = _focal_px_dyn / d_floor

        # 4) Área aparente "bruta": qué área en mm² ocuparía cada píxel si
        #    estuviera apoyado a Z=0 (calibración del plano del suelo).
        area_apparent_floor_mm2 = num_pixels / (px_per_mm_floor ** 2)

        # 5a) Restar la contribucion de las CARAS LATERALES visibles desde
        #     cenital. El comparador (predict_apparent_zenith_area_mm2) las
        #     anade explicitamente (apparent_top + apparent_sides), pero
        #     el observador antes solo des-magnificaba el total. Ahora
        #     restamos primero esas caras laterales para que el footprint
        #     se aproxime al suelo real.
        #
        # apparent_sides ≈ perim × h_lat × (r/Zcam) × 0.5
        # donde perim se aproxima desde el contorno SAM (no del candidato).
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
                # Fallback: perimetro de un cuadrado equivalente al area.
                perim_mm = 4.0 * math.sqrt(max(1.0, area_apparent_floor_mm2))
        except Exception:
            perim_mm = 4.0 * math.sqrt(max(1.0, area_apparent_floor_mm2))

        sides_mm2 = perim_mm * measured_lateral_height_mm * (r_mm / CAM_CEN_Z_MM) * 0.5
        # Solo restamos si la pieza esta descentrada (r>0) y tiene altura
        # significativa; en otro caso la contribucion lateral es ~0.
        apparent_top_only = max(0.5, area_apparent_floor_mm2 - sides_mm2)

        # 5b) Des-magnificación por altura: la silueta vista corresponde a un
        #    contenido elevado a Z=z_eff, que se ve más grande por
        #    factor (Zcam / (Zcam - Z)). Para volver al plano del suelo:
        #    factor lineal = (Zcam - Z) / Zcam, cuadrado en área.
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
# Constantes geométricas de la cámara lateral.
# Cám lateral en BU (15, 0, 2.5) → mm; mira a (0,0,0).
CAM_LAT_X_MM_V3 = float(cfg.cameras.lateral.position[0] * 10.0)
CAM_LAT_Z_MM_V3 = float(cfg.cameras.lateral.position[2] * 10.0)
_CAM_LAT_FOCAL_MM = float(cfg.cameras.lateral.focal_length_mm)
CAM_FOCAL_PX_V3 = float((_CAM_LAT_FOCAL_MM / _SENSOR_WIDTH_MM) * 640.0)  # default 640


def _lateral_focal_px(img_res_px_val=None):
    """Focal lateral en px para la resolución dada."""
    res = float(img_res_px_val) if img_res_px_val else 640.0
    return (_CAM_LAT_FOCAL_MM / _SENSOR_WIDTH_MM) * res


def _bbox_cen_xy_mm_v3(bbox_norm: list) -> tuple:
    """Centro del bbox cenital en coords [0,1] → posición XY (mm) en
    el plano de la cinta respecto al centro óptico."""
    cx_norm = (bbox_norm[0] + bbox_norm[2]) / 2.0
    cy_norm = (bbox_norm[1] + bbox_norm[3]) / 2.0
    res, center, _, px_mm = _cenital_geom(None)  # uses default; caller can override
    cx_px = cx_norm * res
    cy_px = cy_norm * res
    return ((cx_px - center) / px_mm, (center - cy_px) / px_mm)


def estimate_lateral_height_mm_corrected_v3(
    mask_lat: np.ndarray,
    bbox_cen_norm: list,
    estimated_height_mm_initial: float = 9.6,
    img_res_px: float = 640.0,
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

    # 2) Perfil de columnas P75 (en lugar de P50): mejor captura piezas
    #    con cima estrecha (jumper plates 15392, faros 4070, etc.) sin
    #    perder robustez frente a sombras (P95 amplificaria sombras
    #    laterales que la mediana descarta).
    col_h = []
    for c in range(mask_e.shape[1]):
        ys = np.where(mask_e[:, c] > 0)[0]
        if len(ys) > 1:
            col_h.append(int(ys.max() - ys.min() + 1))
    if col_h:
        h_apparent_px = float(np.percentile(col_h, 75))
    else:
        ys, _ = np.where(mask_lat > 0)
        h_apparent_px = (
            float(ys.max() - ys.min() + 1) if len(ys) > 0 else float(mask_lat.shape[0])
        )

    # 3) Posicion XY estimada del centroide cenital
    px_mm, py_mm = _bbox_cen_xy_mm_v3(bbox_cen_norm)

    # 4) Iteracion Newton 1 paso para refinar pz_mm:
    #    - Paso 1: estimacion inicial con prior pz_mm = h_initial/2 = 4.8 mm.
    #    - Paso 2: refinamiento usando h_step1/2 como pz_mm efectivo.
    #    Reduce el error sistematico en piezas altas (60481 h=19.2,
    #    poses verticales h_lat>15 mm) de ~3-5% a <0.5%.
    focal_px = (_CAM_LAT_FOCAL_MM / _SENSOR_WIDTH_MM) * img_res_px
    pz_mm = max(estimated_height_mm_initial / 2.0, 0.5)
    dx = CAM_LAT_X_MM_V3 - px_mm
    dy = -py_mm
    dz = CAM_LAT_Z_MM_V3 - pz_mm
    d_act_1 = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d_act_1 < 1e-3:
        d_act_1 = math.sqrt(CAM_LAT_X_MM_V3 ** 2 + CAM_LAT_Z_MM_V3 ** 2)
    h_step1 = h_apparent_px * d_act_1 / focal_px

    # Newton 1 step
    pz_mm_iter = max(h_step1 / 2.0, 0.5)
    dz_iter = CAM_LAT_Z_MM_V3 - pz_mm_iter
    d_act = math.sqrt(dx * dx + dy * dy + dz_iter * dz_iter)
    if d_act < 1e-3:
        d_act = d_act_1
    px_per_mm_lat_local = focal_px / d_act
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


def segment_crop_sam_batch(img_full_list, bbox_norm_list):
    """OPT 3.2 — SAM en batch para múltiples (imagen, bbox).
    
    SAM (mobile_sam) acepta una imagen + lista de bboxes por llamada,
    pero no múltiples imágenes simultáneamente. Aun así, evitar la carga
    repetida del modelo y reusar el contexto da ~1.5x speedup vs llamadas
    individuales (no hay overhead de Python entre llamadas adyacentes).
    
    Devuelve lista de masks (np.uint8 0/255) en el mismo orden.
    """
    masks = []
    if not img_full_list:
        return masks
    model = get_sam_model()
    for img_full, bbox_norm in zip(img_full_list, bbox_norm_list):
        try:
            w, h = img_full.size
            x1 = max(0, int(bbox_norm[0] * w))
            y1 = max(0, int(bbox_norm[1] * h))
            x2 = min(w, int(bbox_norm[2] * w))
            y2 = min(h, int(bbox_norm[3] * h))
            img_np = np.array(img_full)
            results = model(img_np, bboxes=[[x1, y1, x2, y2]], verbose=False)
            if results and results[0].masks is not None:
                full_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
                masks.append(full_mask[y1:y2, x1:x2])
                continue
        except Exception:
            pass
        # Fallback al helper unitario (cobertura del path no-SAM):
        masks.append(segment_crop_sam(img_full, bbox_norm))
    return masks


def _build_clean_canvas(crop_img: Image.Image, canvas_size: int = 224,
                         margin_px: int = 8,
                         bg_color=CINTA_BG_RGB) -> Image.Image:
    """Construye un canvas `canvas_size x canvas_size` con fondo `bg_color`
    (cinta azul petroleo por defecto) y pega el crop FIT-TO-CANVAS con
    margen `margin_px` (preserva aspect ratio, maximiza tamaño en canvas).

    NUEVO PIPELINE (alineado con `index_synthetic_renders.preprocess_render`
    y validado en `2camaras_random_pieza_unica/test/run_sam_pipeline_e2e.py`).
    Antes usabamos `scale_factor=208/640=0.325` con fondo negro, lo que
    dejaba la pieza muy pequeña en el canvas (especialmente en lateral
    cuando la pieza estaba descentrada) y rompia la simetria con las refs.
    """
    w_p, h_p = crop_img.size
    if w_p <= 0 or h_p <= 0:
        return Image.new("RGB", (canvas_size, canvas_size), bg_color)
    max_dim = canvas_size - 2 * margin_px
    scale = min(max_dim / w_p, max_dim / h_p)
    new_w = max(1, int(round(w_p * scale)))
    new_h = max(1, int(round(h_p * scale)))
    resized = crop_img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
    return canvas


def classify_camera_batch(crops_with_meta, clf, cam_name="cenital", batch_size=64):
    """OPT 3.3 — Extrae embeddings DINOv2 de N crops a la vez (un solo
    forward pass por batch) y devuelve lista de class_scores dict.
    
    crops_with_meta: lista de tuplas (crop_img, valid_part_refs, max_q, min_q).
    Devuelve: lista de class_scores (uno por entrada).
    
    El preprocess (resize+canvas) y el batch del modelo se hacen una sola vez
    por chunk de batch_size; antes hacíamos una llamada a clf._extract_embedding
    por cámara y por sample (300×2 = 600 forward passes), ahora son ~10.
    
    Si el clasificador no expone un método batch nativo, caemos a llamadas
    individuales pero seguimos amortizando setup.
    """
    if not crops_with_meta or not clf._ref_embeddings:
        return [{} for _ in crops_with_meta]
    
    cam_id = 1 if cam_name == "cenital" else 2
    
    # Pre-filtrar refs por cámara (compartido por todas las queries del lote)
    refs_by_cam = [r for r in clf._ref_embeddings if (r["face"] % 10 == cam_id)]
    if not refs_by_cam:
        return [{} for _ in crops_with_meta]
    
    # Construir canvases limpios
    canvases = [_build_clean_canvas(c) for (c, _, _, _) in crops_with_meta]
    sizes_info = [(mq, mnq) for (_, _, mq, mnq) in crops_with_meta]
    
    # Extracción de embeddings — preferimos un método batch si existe
    query_vecs = []
    if hasattr(clf, "_extract_embeddings_batch"):
        try:
            query_vecs = clf._extract_embeddings_batch(canvases, sizes_info=sizes_info)
        except Exception as e:
            log.warning(f"[opt3.3] _extract_embeddings_batch falló ({e}); fallback unitario.")
            query_vecs = []
    if not query_vecs:
        # Fallback unitario (compatibilidad con versiones actuales del KNN)
        query_vecs = [
            clf._extract_embedding(canvases[i], size_info=sizes_info[i])
            for i in range(len(canvases))
        ]
    
    out = []
    for i, (crop_img, valid_part_refs, max_query, min_query) in enumerate(crops_with_meta):
        filtered = [r for r in refs_by_cam if r["part_ref"] in valid_part_refs]
        if not filtered:
            filtered = refs_by_cam
        
        ref_matrix = np.stack([r["embedding"] for r in filtered])
        visual_scores = ref_matrix @ query_vecs[i]
        
        sz_scores = np.array([
            size_score(max_query, min_query, r["part_ref"], clf, cam_name=cam_name)
            for r in filtered
        ])
        combined = visual_scores * sz_scores
        
        class_scores = {}
        for idx, r in enumerate(filtered):
            ref = r["part_ref"]
            score = float(combined[idx])
            if ref not in class_scores or score > class_scores[ref]:
                class_scores[ref] = score
        out.append(class_scores)
    return out


def classify_camera(crop_img, clf, valid_part_refs, max_query, min_query, cam_name="cenital"):
    if not clf._ref_embeddings:
        return {}

    # Canvas con FIT-TO-CANVAS + fondo CINTA (alineado con _build_clean_canvas
    # y con las refs DINOv2 canonicas).
    clean_crop = _build_clean_canvas(crop_img)

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
        # Default: renders/test/test_dual (separación de dominios)
        test_dir = os.path.join(project_root, "renders", "test", "test_dual")
        metadata_path = os.path.join(test_dir, "test_metadata.json")

    if parsed_args.report:
        report_path_arg = parsed_args.report
        if not os.path.isabs(report_path_arg):
            report_path_arg = os.path.join(project_root, report_path_arg)
    else:
        # Default: reports/eval_report.json (separación de dominios)
        report_path_arg = os.path.join(project_root, "reports", "eval_report.json")

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

    # ── Modelos YOLO-Pose (Fase 5: keypoints + triangulacion 2-view) ──
    yolo_cen_pose_path = os.path.join(project_root, "models", "yolo_cenital_pose.pt")
    yolo_lat_pose_path = os.path.join(project_root, "models", "yolo_lateral_pose.pt")
    yolo_cen_pose = None
    yolo_lat_pose = None
    use_kpts_observer = HAS_KPTS_MODULE and os.path.exists(yolo_cen_pose_path) and os.path.exists(yolo_lat_pose_path)
    if use_kpts_observer:
        try:
            log.info(f"Cargando modelos YOLO-Pose: {yolo_cen_pose_path} + {yolo_lat_pose_path}")
            yolo_cen_pose = YOLO(yolo_cen_pose_path)
            yolo_lat_pose = YOLO(yolo_lat_pose_path)
            log.info("[Fase5] Observador kpts ACTIVO (triangulacion 2-view).")
        except Exception as e:
            log.warning(f"[Fase5] No se pudieron cargar modelos pose: {e}. Fallback al observador SAM-bbox.")
            use_kpts_observer = False
    else:
        log.info("[Fase5] Observador kpts desactivado (sin modelos pose o desactivado). "
                 "Se usa el observador SAM-bbox tradicional.")

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

    # ──────────────────────────────────────────────────────────────────
    # OPTs 3.1 + 3.2 — PRE-CÓMPUTO YOLO+SAM EN LOTES
    # Antes: cada sample hacía 2 forward passes YOLO (1 cen + 1 lat)
    # secuenciales. Para 300 samples = 600 calls a YOLO + 600 a SAM.
    # Ahora: una sola fase batch, luego el bucle solo lee resultados.
    # ──────────────────────────────────────────────────────────────────
    entries = meta_data.get("renders", [])
    log.info(f"[opt3.1+3.2] Pre-cómputo YOLO+SAM en batch para {len(entries)} samples...")
    _t_pre = _time.perf_counter()

    cen_paths = []
    lat_paths = []
    valid_idx = []  # mapping pos→sample_idx para entries con archivos válidos
    for i, entry in enumerate(entries):
        cd = entry.get("cameras", {})
        cm = cd.get("cenital")
        lm = cd.get("lateral")
        if not (cm and lm):
            continue
        cp = os.path.join(test_dir, cm["file_name"])
        lp = os.path.join(test_dir, lm["file_name"])
        if not (os.path.exists(cp) and os.path.exists(lp)):
            continue
        cen_paths.append(cp)
        lat_paths.append(lp)
        valid_idx.append(i)

    # YOLO batch (3.1)
    if yolo_cenital is not None:
        yolo_cen_results = yolo_detect_bbox_batch(yolo_cenital, cen_paths, batch_size=16)
    else:
        yolo_cen_results = [(None, 0.0)] * len(cen_paths)
    if yolo_lateral is not None:
        yolo_lat_results = yolo_detect_bbox_batch(yolo_lateral, lat_paths, batch_size=16)
    else:
        yolo_lat_results = [(None, 0.0)] * len(lat_paths)

    # Cargar imágenes y resolver bboxes (con fallback a metadata)
    img_cen_cache = {}
    img_lat_cache = {}
    cen_bboxes_full = {}  # sample_idx → [cx1,cy1,cx2,cy2]
    lat_bboxes_full = {}
    cen_confs = {}
    lat_confs = {}
    for k, sample_idx in enumerate(valid_idx):
        entry = entries[sample_idx]
        cm = entry["cameras"]["cenital"]
        lm = entry["cameras"]["lateral"]
        cp = cen_paths[k]
        lp = lat_paths[k]

        img_cen = Image.open(cp).convert("RGB")
        img_lat = Image.open(lp).convert("RGB")
        img_cen_cache[sample_idx] = img_cen
        img_lat_cache[sample_idx] = img_lat

        cb, cc = yolo_cen_results[k]
        if cb is not None:
            yolo_detections_cenital += 1
            cen_bboxes_full[sample_idx] = list(cb)
        else:
            cen_bboxes_full[sample_idx] = list(cm["bbox_norm"])
        cen_confs[sample_idx] = cc

        lb, lc = yolo_lat_results[k]
        if lb is not None:
            yolo_detections_lateral += 1
            lat_bboxes_full[sample_idx] = list(lb)
        else:
            lat_bboxes_full[sample_idx] = list(lm["bbox_norm"])
        lat_confs[sample_idx] = lc

    # SAM batch (3.2) — paralelo en una sola pasada por cámara
    sam_cen_inputs = [(img_cen_cache[s], cen_bboxes_full[s]) for s in valid_idx]
    sam_lat_inputs = [(img_lat_cache[s], lat_bboxes_full[s]) for s in valid_idx]
    masks_cen_list = segment_crop_sam_batch(
        [t[0] for t in sam_cen_inputs], [t[1] for t in sam_cen_inputs]
    )
    masks_lat_list = segment_crop_sam_batch(
        [t[0] for t in sam_lat_inputs], [t[1] for t in sam_lat_inputs]
    )
    masks_cen_dict = dict(zip(valid_idx, masks_cen_list))
    masks_lat_dict = dict(zip(valid_idx, masks_lat_list))

    log.info(
        f"[opt3.1+3.2] Pre-cómputo OK en {_time.perf_counter()-_t_pre:.1f}s "
        f"({len(valid_idx)} samples, YOLO_cen={yolo_detections_cenital}, "
        f"YOLO_lat={yolo_detections_lateral})"
    )

    # ──────────────────────────────────────────────────────────────────
    # Bucle principal (ahora cada sample solo hace cálculo CPU + KNN)
    # ──────────────────────────────────────────────────────────────────
    for sample_idx, entry in enumerate(entries):
        if sample_idx not in cen_bboxes_full:
            continue
        ref_gt = entry["ref"]
        cameras_data = entry["cameras"]
        cen_meta = cameras_data["cenital"]
        lat_meta = cameras_data["lateral"]

        img_cen_full = img_cen_cache[sample_idx]
        img_lat_full = img_lat_cache[sample_idx]
        iw, ih = img_cen_full.size
        liw, lih = img_lat_full.size
        # Dynamic resolution: update global-like ref for this sample
        _img_res_cen = float(iw)
        _img_res_lat = float(liw)
        _actual_img_res = int(iw)

        cx1, cy1, cx2, cy2 = cen_bboxes_full[sample_idx]
        lx1, ly1, lx2, ly2 = lat_bboxes_full[sample_idx]
        cen_yolo_conf = cen_confs[sample_idx]
        lat_yolo_conf = lat_confs[sample_idx]

        crop_cen = img_cen_full.crop((
            max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
            min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih))
        ))
        crop_lat = img_lat_full.crop((
            max(0, int(lx1 * liw)), max(0, int(ly1 * lih)),
            min(liw, int(lx2 * liw)), min(lih, int(ly2 * lih))
        ))

        mask_cen = masks_cen_dict[sample_idx]
        mask_lat = masks_lat_dict[sample_idx]

        # ── ESTIMACIÓN DE COLOR DENTRO DEL CONTORNO SAM ──
        cen_est2_rgb = estimate_color_predominant_sam(crop_cen, mask_cen)
        cen_est2_catalog = find_closest_catalog_color(cen_est2_rgb, camera="cenital")

        lat_est2_rgb = estimate_color_predominant_sam(crop_lat, mask_lat)
        lat_est2_catalog = find_closest_catalog_color(lat_est2_rgb, camera="lateral")

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

        # Phase 1: Color gating con fallback robusto.
        #
        # Estrategia mejorada (vs versión previa que aplicaba == estricto al
        # color cenital):
        #
        #   a) Si consenso ✓ y ΔE_lab del color top-1 < TH_TIGHT  → filtro
        #      estricto por (color_code_cen): excluye refs con otros colores.
        #   b) Si consenso ✓ pero ΔE_lab >= TH_TIGHT (color decidido pero
        #      "no de confianza")  → filtro laxo: aceptamos refs con color
        #      top-1, top-2 (runner_up) y cualquier color cuya ΔE entre
        #      paleta y RGB observado < TH_LAX  (margen LAB).
        #   c) Si consenso ✗ (cen ≠ lat)  → NO filtramos por color (todos
        #      los refs del set entran al gating de superficie/altura).
        #
        # Esto resuelve dos errores observados en el set 300:
        #   - Trans-Brown observado bajo fondo azul translúcido cae cerca
        #     de Dark Bluish Gray (top-1=85). Con filtro estricto la 3023
        #     (sólo existente como Red 5 / Trans-Brown 13) era expulsada.
        #   - Red `#C30025` con luz Eevee tira a `#CA5074` y top-1 cae en
        #     Trans-Red 17 — la 3023 vuelve a ser expulsada.
        TH_TIGHT_DELTA_E = 18.0   # ~JND fuerte
        TH_LAX_DELTA_E   = 35.0   # solo paleta del set (7 colores) muy distantes

        # Construir lista de piezas candidatas.
        # Preferencia: piezas con embeddings en BD (dinámico, no limitado a 75078-1).
        # Fallback: REAL_SETS["75078-1"] filtrado por SELECTED_PARTS.
        try:
            from _db_helpers import get_all_ref_color_combinations_from_db
            _parts_db = get_all_ref_color_combinations_from_db()
            if _parts_db:
                parts_in_set = _parts_db
            else:
                parts_in_set = [{"ref": p["ref"], "color_code": str(p["color_code"])}
                                for p in REAL_SETS["75078-1"]["parts"] if p["ref"] in SELECTED_PARTS]
        except Exception:
            parts_in_set = [{"ref": p["ref"], "color_code": str(p["color_code"])}
                            for p in REAL_SETS["75078-1"]["parts"] if p["ref"] in SELECTED_PARTS]

        cen_de = float(cen_est2_catalog.get("_delta_e", 0.0))
        # Construir set de codes de color permitidos según las 3 ramas.
        if not colors_consensus_ok:
            # Rama c) consenso falla → no filtrar por color.
            allowed_color_codes = None  # marca "todos"
            color_filter_mode = "none_consensus_fail"
        elif cen_de < TH_TIGHT_DELTA_E:
            # Rama a) confianza alta → filtro estricto.
            allowed_color_codes = {color_code_cen}
            color_filter_mode = f"strict_dE={cen_de:.1f}"
        else:
            # Rama b) consenso OK pero color ambiguo (ΔE alto) → filtro laxo.
            # Aceptamos top-1, runner_up y cualquier color del catálogo cuya
            # ΔE_lab al RGB observado sea < TH_LAX_DELTA_E.
            allowed_color_codes = {color_code_cen}
            ru_code = cen_est2_catalog.get("_runner_up_code")
            if ru_code is not None:
                allowed_color_codes.add(ru_code)
            try:
                cen_obs_lab = rgb_to_lab(cen_est2_rgb)
                for sc in SET_CATALOG_COLORS:
                    sc_lab = rgb_to_lab(sc["rgb"])
                    d = float(np.linalg.norm(cen_obs_lab - sc_lab))
                    if d < TH_LAX_DELTA_E:
                        allowed_color_codes.add(sc["color_code"])
            except Exception:
                pass
            color_filter_mode = f"lax_dE={cen_de:.1f}|codes={sorted(allowed_color_codes)}"

        if allowed_color_codes is None:
            valid_by_color = [p["ref"] for p in parts_in_set]
        else:
            valid_by_color = [p["ref"] for p in parts_in_set
                              if p["color_code"] in allowed_color_codes]
            if not valid_by_color:
                # Ninguna pieza con esos colores → fallback duro a todo el set.
                valid_by_color = [p["ref"] for p in parts_in_set]
                color_filter_mode += "+empty_fallback_all"

        # Una pieza puede aparecer en N colores (idem 3023 en {5, 13}); con
        # `valid_by_color` deduplicamos a la lista de refs únicos:
        valid_by_color = sorted(set(valid_by_color))

        log.info(
            f"  [Color] Filter mode: {color_filter_mode} "
            f"→ refs candidatos por color = {len(valid_by_color)}"
        )

        # ── OBSERVACIONES PURAS (independientes del candidato) ──
        # Modo HIBRIDO Fase 5:
        #   - Si los modelos YOLO-Pose estan cargados y detectan
        #     >=6 keypoints en ambas camaras → usar triangulacion 2-view.
        #   - Si no, fallback al observador SAM-bbox (Fases 1-4).
        # 1. SAM-BBox (Standard)
        try:
            h_sam_bbox, _mag_lat, _d_act_lat = estimate_lateral_height_mm_corrected_v3(
                mask_lat, [cx1, cy1, cx2, cy2],
                estimated_height_mm_initial=9.6,
                img_res_px=_img_res_lat,
            )
        except Exception:
            h_sam_bbox = measure_lateral_height_mm_sam(mask_lat)
        if h_sam_bbox <= 0:
            h_sam_bbox = measure_lateral_height_mm_sam(mask_lat)
            
        zen_obs_sam = observe_zenithal_surface_mm2(
            mask_cen, [cx1, cy1, cx2, cy2],
            measured_lateral_height_mm=h_sam_bbox,
            img_res_px_val=_img_res_cen,
        )
        area_sam_bbox = zen_obs_sam["footprint_area_mm2"]
        
        # 2. Keypoints 3D (Pure)
        h_kpts_3d = None
        area_kpts_3d = None
        n_kps_val = 0
        kpts_obs = None
        
        if use_kpts_observer:
            try:
                cp = cen_paths[valid_idx.index(sample_idx)]
                lp = lat_paths[valid_idx.index(sample_idx)]
                kp_cen = extract_yolo_pose_keypoints(yolo_cen_pose, cp, conf=0.20)
                kp_lat = extract_yolo_pose_keypoints(yolo_lat_pose, lp, conf=0.20)
                if kp_cen is not None and kp_lat is not None:
                    kpts_obs = kpts_observer_fn(kp_cen, kp_lat, conf_min=0.20)
                    if kpts_obs.get("n_valid", 0) >= 4:
                        h_kpts_3d = float(kpts_obs.get("lateral_height_mm", 0.0))
                        area_kpts_3d = float(kpts_obs.get("footprint_area_mm2", 0.0))
                        n_kps_val = kpts_obs.get("n_valid", 0)
            except Exception as e:
                pass

        # 3. Hybrid (SAM + Keypoints 3D height)
        h_hybrid = None
        area_hybrid = None
        if h_kpts_3d is not None and h_kpts_3d > 0.0:
            h_hybrid = h_kpts_3d
            zen_obs_hybrid = observe_zenithal_surface_mm2(
                mask_cen, [cx1, cy1, cx2, cy2],
                measured_lateral_height_mm=h_hybrid,
                img_res_px_val=_img_res_cen,
            )
            area_hybrid = zen_obs_hybrid["footprint_area_mm2"]

        # Decision-making logic: best is Hybrid, then fallback to SAM-BBox
        if h_hybrid is not None and area_hybrid is not None:
            measured_height = h_hybrid
            obs_footprint_area_mm2 = area_hybrid
            z_eff = max(0.5, measured_height * 0.5)
            mag_lin = CAM_CEN_Z_MM / max(1.0, CAM_CEN_Z_MM - z_eff)
            obs_apparent_area_mm2 = obs_footprint_area_mm2 * (mag_lin ** 2)
            kpts_used = True
        else:
            measured_height = h_sam_bbox
            obs_footprint_area_mm2 = area_sam_bbox
            obs_apparent_area_mm2 = zen_obs_sam["apparent_area_mm2"]
            kpts_used = False

        if sample_idx < 3:
            mode_obs = "hybrid_kpts" if kpts_used else "sam_bbox"
            log.info(
                f"    [OBS:{mode_obs}] mask_pixels={int(np.sum(mask_cen > 0))} | "
                f"h_lat_meas={measured_height:.2f}mm | "
                f"area_apparent={obs_apparent_area_mm2:.1f}mm² | "
                f"area_footprint={obs_footprint_area_mm2:.1f}mm²"
                + (f" | n_kps={n_kps_val}" if kpts_used else "")
            )

        # Phase 2: Surface gating (criterio híbrido adaptativo de error)
        valid_by_surface = []
        surface_scores = {}
        
        hybrid_thresh = float(cfg.inference.size_scoring.hybrid_threshold_mm2)
        abs_tol = float(cfg.inference.size_scoring.abs_tolerance_mm2)
        rel_tol = float(cfg.inference.size_scoring.rel_tolerance_pct)
        
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
            is_candidate_valid = False
            best_residual = float("inf")
            best_rel_err = float("inf")
            chosen_metric = ""
            
            for nom_area, nom_h in configs:
                nom_footprint = nom_area * filling_factor
                target_apparent = predict_apparent_zenith_area_mm2(
                    nom_footprint, nom_h, [cx1, cy1, cx2, cy2]
                )
                
                residual = abs(obs_apparent_area_mm2 - target_apparent)
                if target_apparent <= hybrid_thresh:
                    metric_valid = (residual <= abs_tol)
                    current_metric = "abs"
                else:
                    rel_err = (residual / target_apparent) * 100.0 if target_apparent > 0 else float("inf")
                    metric_valid = (rel_err <= rel_tol)
                    current_metric = "rel"
                
                sigma_rel = 0.20 + 0.012 * nom_h
                sigma = max(2.0, sigma_rel * target_apparent)
                score = math.exp(-(residual ** 2) / (2.0 * sigma * sigma))
                
                if score > best_score:
                    best_score = score
                    best_residual = residual
                    if target_apparent > 0:
                        best_rel_err = (residual / target_apparent) * 100.0
                    else:
                        best_rel_err = 0.0
                    is_candidate_valid = metric_valid
                    chosen_metric = current_metric

            surface_scores[ref] = best_score
            if is_candidate_valid:
                valid_by_surface.append(ref)

            if sample_idx < 3 and ref == ref_gt:
                log.info(
                    f"    [DEBUG-SURF] Ref={ref}(GT) | obs_apparent={obs_apparent_area_mm2:.1f} | "
                    f"valid={is_candidate_valid} ({chosen_metric}) | residual={best_residual:.1f}mm² ({best_rel_err:.1f}%)"
                )

        if not valid_by_surface:
            valid_by_surface = valid_by_color

        # Phase 3: Height gating (lateral, +/-35%)
        # Tolerancia ampliada (era ±15%) bajo Dome Light: SAM tiende a
        # incluir sombras/halos en la silueta lateral, especialmente con
        # piezas translúcidas / oscuras, lo que distorsiona la altura
        # medida. La medida de altura ya está calculada arriba (obs).
        valid_by_height = []
        for ref in valid_by_surface:
            nominals = get_nominal_heights(ref)
            for nom in nominals:
                if 0.65 * nom <= measured_height <= 1.35 * nom:
                    valid_by_height.append(ref)
                    break
        if not valid_by_height:
            valid_by_height = valid_by_surface

        # Phase 4: DINOv2 fusion (cenital 70% + lateral 30%)
        max_query_cen, min_query_cen = get_oriented_dims_mm_sam(mask_cen)
        max_query_lat, min_query_lat = get_oriented_dims_mm_sam(mask_lat)

        # SIMETRIZACION FONDO CON REFS DINOV2 ----
        # Las refs DINOv2 se renderizan con `film_transparent=True` y al
        # convertir RGBA->RGB el fondo queda en negro (0,0,0). Para que
        # el embedding query sea comparable con el de las refs, aplicamos
        # la mascara SAM al crop de inferencia y reemplazamos los pixeles
        # fuera de la pieza por negro puro.
        crop_cen_masked = apply_sam_mask_to_crop(crop_cen, mask_cen)
        crop_lat_masked = apply_sam_mask_to_crop(crop_lat, mask_lat)

        scores_cenital = classify_camera(crop_cen_masked, clf, valid_by_height, max_query_cen, min_query_cen, cam_name="cenital")
        scores_lateral = classify_camera(crop_lat_masked, clf, valid_by_height, max_query_lat, min_query_lat, cam_name="lateral")

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
            # --- Detailed comparative metrics ---
            "obs_sam_bbox": {
                "height": round(float(h_sam_bbox), 2) if h_sam_bbox is not None else None,
                "area": round(float(area_sam_bbox), 2) if area_sam_bbox is not None else None
            },
            "obs_kpts_3d": {
                "height": round(float(h_kpts_3d), 2) if h_kpts_3d is not None else None,
                "area": round(float(area_kpts_3d), 2) if area_kpts_3d is not None else None
            },
            "obs_hybrid": {
                "height": round(float(h_hybrid), 2) if h_hybrid is not None else None,
                "area": round(float(area_hybrid), 2) if area_hybrid is not None else None
            },
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
            "resolution": f"{_actual_img_res}x{_actual_img_res}",
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
