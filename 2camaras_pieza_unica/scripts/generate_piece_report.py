# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_piece_report.py
=========================================================
Genera un report HTML standalone de diagnóstico para una pieza específica.
Usa renders existentes del test set (data/test_dual/).

Uso:
    .venv/bin/python 2camaras_pieza_unica/scripts/generate_piece_report.py --ref 3001
    .venv/bin/python 2camaras_pieza_unica/scripts/generate_piece_report.py --ref 3020 --pose 1
"""
import os, sys, json, math, argparse, base64
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from config_loader import cfg
from inference.knn_classifier import get_knn_classifier, FALLBACK_FOOTPRINT_MM
from inference.api import PART_HEIGHTS_MM
from database.set_catalog import REAL_SETS

SELECTED_PARTS = cfg.pieces.selected_parts
PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
PX_PER_MM_LATERAL = cfg.inference.calibration.px_per_mm_lateral


from ultralytics import SAM

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

def get_sam_segmented_crop(crop_img: Image.Image, mask: np.ndarray) -> Image.Image:
    try:
        img_np = np.array(crop_img.convert("RGB"))
        segmented_np = np.zeros_like(img_np)
        mask_fg = mask > 0
        segmented_np[mask_fg] = img_np[mask_fg]
        return Image.fromarray(segmented_np)
    except Exception:
        return crop_img


# ─────────────────────────────────────────────────────────────────────────────
# Constantes geométricas LEGO usadas para descontar studs en cenital
# (radio típico stud LEGO ≈ 2.4 mm; altura stud ≈ 1.7-1.8 mm pero el offset
# añadido por la geometría completa de la cara superior con studs sobre una
# pieza es ~0.9 mm para piezas brick/plate cuando se mira lateralmente).
# ─────────────────────────────────────────────────────────────────────────────
STUD_RADIUS_MM = 2.4
STUD_AREA_MM2 = math.pi * (STUD_RADIUS_MM ** 2)
STUD_HEIGHT_MM = 0.9

# Geometría de cámara (BU → mm). bu_per_mm = 0.1 ⇒ 1 BU = 10 mm.
CAM_CENITAL_HEIGHT_MM = 150.0  # cámara cenital en (0, 0, 15 BU) = 150 mm
CAM_LATERAL_X_MM = 150.0       # cámara lateral en (15, 0, 2.5) BU = (150, 0, 25) mm
CAM_LATERAL_Z_MM = 25.0
CAM_FOCAL_PX = 480.0           # f * width / sensor = 27 mm * 640 / 36 mm


def estimate_studs_count(part_ref: str) -> int:
    """Estima el número de studs en la cara superior de una pieza brick/plate.

    Se usa para descontar el área efectiva de studs cuando la pose tiene
    studs hacia arriba (cara `Bottom` apoyada en la cinta).
    Los valores se aproximan a partir del nombre de la pieza (1×n, 2×n, …)
    y de un mapeo manual para piezas comunes del set 75078-1.
    """
    studs_map = {
        # bricks 1x?
        "3005": 1, "3004": 2, "3622": 3, "3010": 4, "3009": 6, "3008": 8,
        # bricks 2x?
        "3003": 4, "3002": 6, "3001": 8, "2456": 12, "3007": 16,
        # plates 1x?
        "3024": 1, "3023": 2, "3623": 3, "3710": 4, "3666": 6, "3460": 8,
        # plates 2x?
        "3022": 4, "3021": 6, "3020": 8, "3795": 12, "3034": 16,
        # plates espesor 1×… que aparecen en este set
        "3069": 2,   # plate 1x2 lisa (tile sin studs!) — caso especial,
                     # ver más abajo en el código que lo trata como 0 studs.
        "3068": 4,   # tile 2x2 (sin studs)
        "63864": 4,  # tile 1x3
        "3070b": 1,  # tile 1x1
        # cilíndricos / round
        "4032": 4, "4073": 1, "6141": 1, "98138": 1, "59900": 1,
        # otros
        "3700": 2,   # technic brick 1x2 (con stud)
        "4070": 1,   # brick 1x1 with headlight
        "2877": 2,   # brick 1x2 with grille (sin studs en cara grille)
        "3037": 16,  # slope 4x6
        "3038": 24,  # slope 2x6 (grande)
    }
    if part_ref in studs_map:
        return studs_map[part_ref]
    # Heurística por dimensiones (sin información explícita)
    fp = FALLBACK_FOOTPRINT_MM.get(part_ref, (8.0, 8.0))
    n_long = max(1, int(round(max(fp) / 8.0)))
    n_short = max(1, int(round(min(fp) / 8.0)))
    return n_long * n_short


# Piezas del catálogo conocidas como "tile" (sin studs en su cara superior)
# o piezas tipo grille/textured: la silueta cenital ya incluye su cara
# superior plana; no hay studs que descontar en cenital.
TILES_NO_STUDS = {"3069", "3068", "63864", "3070b", "2877", "3023"}


def detect_studs_top_from_pose(pose_info: dict, part_ref: str) -> tuple:
    """Determina si los studs apuntan hacia arriba en esta pose.

    Devuelve (has_studs_top, n_studs, reason). Usa preferentemente:
      1) `face_class` del cache: "Bottom" → la base apoya, studs arriba.
                                  "Top"    → cara superior apoya, studs abajo.
                                  "Side"   → studs apuntan al lado.
      2) Si la pieza es de tipo `tile` (sin studs en cara superior) se devuelve
         siempre n_studs=0 independientemente de la pose.
    """
    face = (pose_info or {}).get("face_class", "")
    n_studs = estimate_studs_count(part_ref)
    if part_ref in TILES_NO_STUDS:
        return (False, 0, f"tile sin studs (face={face})")
    if face == "Bottom":
        return (True, n_studs, f"face={face} → base apoyada, studs ↑")
    if face == "Top":
        return (False, 0, f"face={face} → studs ↓ (no visibles cenital)")
    return (False, 0, f"face={face} → studs laterales")


def filter_shadows_in_mask(crop_img: Image.Image, mask: np.ndarray,
                            v_min: int = 45, s_min: int = 25) -> np.ndarray:
    """Refina la máscara descartando píxeles con luminosidad/saturación
    bajas (típicas de sombra proyectada sobre la cinta o el suelo).
    """
    try:
        if mask is None or mask.size == 0:
            return mask
        rgb = np.array(crop_img.convert("RGB"))
        if rgb.shape[:2] != mask.shape[:2]:
            rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        # Sombra típica = baja V y baja S → eliminar.
        keep = (v >= v_min) | (s >= s_min)
        refined = mask.copy()
        refined[~keep] = 0
        # Limpieza morfológica final
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
        return refined
    except Exception as e:
        print(f"[WARN] filter_shadows_in_mask: {e}")
        return mask


def _heights_from_mask(mask: np.ndarray) -> np.ndarray:
    """Devuelve un array con la altura (max_y - min_y) de cada columna no
    vacía de la máscara binaria. Filtra columnas con sólo 1 píxel para
    eliminar ruido de los bordes laterales del bbox."""
    if mask is None or mask.size == 0:
        return np.array([])
    cols = []
    for c in range(mask.shape[1]):
        ys = np.where(mask[:, c] > 0)[0]
        if len(ys) > 0:
            cols.append(int(ys.max() - ys.min()))
    arr = np.array(cols)
    return arr[arr > 1]


def measure_lateral_height_robust(img_lateral: Image.Image,
                                   crop_lat: Image.Image,
                                   bbox_lat_norm: list) -> tuple:
    """Estima la altura del crop lateral combinando SAM + filtrado de sombras
    + estadísticas robustas del perfil de columnas.

    Estrategia:
      • Tomamos la silueta de la pieza desde SAM (con filtro HSV de sombras).
      • Medimos altura por columna y agregamos con la **mediana** (robusta
        a sombras alargadas en una mitad de la imagen y a estallidos en
        bordes con bevel/halo).
      • También calculamos P90 como cota superior y un "modo" más fino
        (la altura más frecuente con tolerancia ±1 px). Si la mediana y
        el modo coinciden, devolvemos la mediana como valor canónico.
      • Si SAM no está disponible o la máscara es casi vacía tras filtrar,
        caemos a la segmentación HSV del crop.

    Devuelve (height_px, n_cols_validas, source_label).
    """
    def _summarize(arr: np.ndarray) -> tuple:
        """Devuelve (h_px, label_extra) escogiendo la métrica más robusta."""
        if arr.size == 0:
            return (None, "empty")
        median_v = float(np.median(arr))
        p90_v = float(np.percentile(arr, 90))
        # Modo aproximado: altura más frecuente con tolerancia ±1 px.
        from collections import Counter
        rounded = np.rint(arr).astype(int)
        counter = Counter(rounded.tolist())
        mode_v = counter.most_common(1)[0][0]
        # La mediana es la estadística más robusta cuando hay sombras;
        # P90 puede sobreestimar si el bbox incluye sombras alargadas
        # en columnas concretas.
        return (median_v, f"P50={median_v:.1f}, P90={p90_v:.1f}, mode={mode_v}")

    # 1) SAM sobre la imagen completa.
    try:
        mask_sam = segment_crop_sam(img_lateral, bbox_lat_norm)
        if mask_sam is not None and mask_sam.size > 0 and np.any(mask_sam):
            mask_sam_clean = filter_shadows_in_mask(crop_lat, mask_sam)
            if np.sum(mask_sam_clean > 0) < max(20, 0.05 * mask_sam_clean.size):
                # El filtro deja poco; nos quedamos con la SAM cruda.
                arr = _heights_from_mask(mask_sam)
                src_prefix = "SAM (sin filtro: pieza muy oscura)"
            else:
                arr = _heights_from_mask(mask_sam_clean)
                src_prefix = "SAM+shadow_filter"
            if arr.size >= 3:
                h_px, extra = _summarize(arr)
                if h_px is not None:
                    return (h_px, int(arr.size), f"{src_prefix}+P50 [{extra}]")
    except Exception as e:
        print(f"[WARN] measure_lateral_height_robust SAM: {e}")

    # 2) Fallback: segmentación HSV del crop con filtro de sombras.
    try:
        mask_hsv = segment_crop(crop_lat)
        mask_hsv_clean = filter_shadows_in_mask(crop_lat, mask_hsv)
        arr = _heights_from_mask(mask_hsv_clean)
        if arr.size >= 3:
            h_px, extra = _summarize(arr)
            if h_px is not None:
                return (h_px, int(arr.size), f"HSV+shadow_filter+P50 [{extra}]")
    except Exception as e:
        print(f"[WARN] measure_lateral_height_robust HSV: {e}")

    # 3) Último fallback: altura del bbox.
    h_bbox_px = (bbox_lat_norm[3] - bbox_lat_norm[1]) * img_lateral.height
    return (h_bbox_px, 0, "bbox_height_fallback (sin segmentación)")


def img_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_bbox_on_image(img: Image.Image, bbox_norm: list, color="lime", width=3) -> Image.Image:
    """Draw bounding box on image. bbox_norm = [x1_norm, y1_norm, x2_norm, y2_norm]."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    x1 = int(bbox_norm[0] * w)
    y1 = int(bbox_norm[1] * h)
    x2 = int(bbox_norm[2] * w)
    y2 = int(bbox_norm[3] * h)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    # Label
    draw.text((x1 + 2, y1 + 2), f"bbox", fill=color)
    return img_copy


def find_nearest_set_color(crop_img: Image.Image) -> dict:
    """Finds the closest color among the pieces of set 75078-1 based on Lab distance."""
    try:
        import cv2
        img_rgb = np.array(crop_img.convert("RGB"))
        bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
        dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_color, axis=-1)
        mask_fg = dist > 18.0
        if not np.any(mask_fg):
            return {"color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray"}
        
        # Filtrar brillos y sombras
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        v_channel = img_hsv[mask_fg, 2]
        valid_brightness = (v_channel >= 40) & (v_channel <= 235)
        
        if np.sum(valid_brightness) > 10:
            avg_rgb = img_rgb[mask_fg][valid_brightness].mean(axis=0)
        else:
            avg_rgb = img_rgb[mask_fg].mean(axis=0)
            
        # Conversión RGB a CIELAB para comparación perceptiva uniforme
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
            
        avg_lab = rgb_to_lab(avg_rgb)
        
        # Colores reales en el catálogo del set 75078-1
        set_colors = [
            {"color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "rgb": [160, 165, 169]},
            {"color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "rgb": [27, 27, 27]},
            {"color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "rgb": [201, 26, 9]},
        ]
        
        best_match = set_colors[0]
        min_dist = float("inf")
        for sc in set_colors:
            sc_lab = rgb_to_lab(sc["rgb"])
            dist = np.linalg.norm(avg_lab - sc_lab)
            if dist < min_dist:
                min_dist = dist
                best_match = sc
        return best_match
    except Exception as e:
        print(f"[ERROR] find_nearest_set_color: {e}")
        return {"color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray"}


def draw_bbox_and_mask(img: Image.Image, bbox_norm: list, mask: np.ndarray) -> Image.Image:
    try:
        import cv2
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        w, h = img.size
        x1 = int(bbox_norm[0] * w)
        y1 = int(bbox_norm[1] * h)
        x2 = int(bbox_norm[2] * w)
        y2 = int(bbox_norm[3] * h)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > 0 and crop_h > 0:
            mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            overlay = img_cv.copy()
            # Overlay de color cian semi-transparente sobre la máscara
            overlay[y1:y2, x1:x2][mask_resized > 0] = [255, 255, 0] # BGR para cian
            cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0, img_cv)
            
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, "BBox + Mask", (x1 + 4, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[ERROR] draw_bbox_and_mask: {e}")
        return img


def draw_binary_mask(img: Image.Image, bbox_norm: list, mask: np.ndarray) -> Image.Image:
    try:
        w, h = img.size
        mask_img = np.zeros((h, w), dtype=np.uint8)
        x1 = int(bbox_norm[0] * w)
        y1 = int(bbox_norm[1] * h)
        x2 = int(bbox_norm[2] * w)
        y2 = int(bbox_norm[3] * h)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > 0 and crop_h > 0:
            mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            mask_img[y1:y2, x1:x2] = mask_resized
            
        rgb_mask = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_mask)
    except Exception as e:
        print(f"[ERROR] draw_binary_mask: {e}")
        return img


def draw_crop_contour(crop_img: Image.Image, mask: np.ndarray) -> Image.Image:
    try:
        # Crear una imagen completamente negra
        img_cv = np.zeros((crop_img.height, crop_img.width, 3), dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dibujar únicamente las líneas del contorno identificado en amarillo
        cv2.drawContours(img_cv, contours, -1, (0, 255, 255), 1)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[ERROR] draw_crop_contour: {e}")
        return crop_img


def segment_crop(crop_img: Image.Image) -> np.ndarray:
    try:
        # Convertir a BGR para uso con OpenCV
        crop_bgr = cv2.cvtColor(np.array(crop_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # 1. Convertir a espacio de color HSV
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        # Color del fondo en BGR: (84, 65, 37) -> Correspondiente a RGB: (37, 65, 84)
        bg_bgr = np.uint8([[[84, 65, 37]]])
        bg_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        # 2. Definir rango de color para el fondo en HSV
        lower_bg = np.array([max(0, int(bg_hsv[0]) - 15), max(0, int(bg_hsv[1]) - 60), max(0, int(bg_hsv[2]) - 60)])
        upper_bg = np.array([min(180, int(bg_hsv[0]) + 15), min(255, int(bg_hsv[1]) + 60), min(255, int(bg_hsv[2]) + 60)])
        
        bg_mask = cv2.inRange(crop_hsv, lower_bg, upper_bg)
        
        # 3. Invertir la máscara: lo que no es fondo es la pieza (foreground)
        fg_mask = cv2.bitwise_not(bg_mask)
        
        # 4. Operaciones morfológicas para limpiar y conectar huecos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Encontrar contornos externos para rellenar huecos interiores de studs y sombras
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros((crop_img.height, crop_img.width), dtype=np.uint8)
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
            if valid_contours:
                # Rellenar y solidificar el contorno
                cv2.drawContours(mask, valid_contours, -1, 255, -1)
                
        return mask
    except Exception as e:
        print(f"[ERROR] segment_crop: {e}")
        return np.ones((crop_img.height, crop_img.width), dtype=np.uint8) * 255



def measure_surface_mm2(crop_img: Image.Image) -> float:
    """Estimate surface area in mm² from cenital crop using mask pixel count."""
    try:
        mask = segment_crop(crop_img)
        num_pixels = np.sum(mask > 0)
        return num_pixels / (PX_PER_MM_CENITAL ** 2)
    except Exception:
        pass
    return (crop_img.width / PX_PER_MM_CENITAL) * (crop_img.height / PX_PER_MM_CENITAL)


def measure_bbox_surface_mm2(crop_img: Image.Image) -> float:
    """Estimate surface area in mm² from cenital crop using minimum bounding box (minAreaRect)."""
    try:
        import cv2
        mask = segment_crop(crop_img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]
        if valid:
            all_pts = np.vstack(valid)
            rect = cv2.minAreaRect(all_pts)
            (_, _), (w_px, h_px), _ = rect
            w_mm = w_px / PX_PER_MM_CENITAL
            h_mm = h_px / PX_PER_MM_CENITAL
            return w_mm * h_mm
    except Exception:
        pass
    return (crop_img.width / PX_PER_MM_CENITAL) * (crop_img.height / PX_PER_MM_CENITAL)


def measure_height_mm(crop_img: Image.Image) -> float:
    """Estimate height in mm from lateral crop."""
    try:
        import cv2
        mask = segment_crop(crop_img)
        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            return (max(ys) - min(ys)) / PX_PER_MM_LATERAL
    except Exception:
        pass
    return crop_img.height / PX_PER_MM_LATERAL


def get_oriented_dims_mm(crop_img: Image.Image) -> tuple:
    try:
        import cv2
        mask = segment_crop(crop_img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]
        if valid:
            all_pts = np.vstack(valid)
            rect = cv2.minAreaRect(all_pts)
            (_, _), (w_px, h_px), _ = rect
            return max(w_px, h_px) / PX_PER_MM_CENITAL, min(w_px, h_px) / PX_PER_MM_CENITAL
    except Exception:
        pass
    return crop_img.width / PX_PER_MM_CENITAL, crop_img.height / PX_PER_MM_CENITAL


def get_part_dimensions(ref: str) -> tuple:
    dims_cfg = cfg.pieces.dimensions_mm
    if hasattr(dims_cfg, ref):
        return tuple(getattr(dims_cfg, ref))
    fp = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
    h = PART_HEIGHTS_MM.get(ref, 9.6)
    return (max(fp), min(fp), h)


def get_catalog_color(part_ref: str) -> dict:
    """Get real color from set catalog."""
    for p in REAL_SETS["75078-1"]["parts"]:
        if p["ref"] == part_ref:
            return {
                "color_hex": p.get("color_hex", "#A0A5A9"),
                "color_code": p.get("color_code", "85"),
                "color_name": p.get("color_name", "Light Bluish Gray"),
            }
    return {"color_hex": "#A0A5A9", "color_code": "85", "color_name": "Unknown"}


def get_stable_pose_info(part_ref: str, pose_index: int = 0) -> dict:
    """Get stable pose info from cache.

    El cache (sync_stable_poses_cache.py) renumera `pose_index` 0..N-1
    tras filtrado pero conserva el `original_pose_index` con la enumeración
    original de la BD. Buscamos primero por `original_pose_index` (lo que
    guarda test_metadata.json viene de la BD) y caemos a `pose_index`
    nuevo si no está disponible.
    """
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        cache = json.load(f)
    poses = cache.get(part_ref, [])
    # Match por original_pose_index (canónico)
    for p in poses:
        if p.get("original_pose_index") == pose_index:
            return p
    # Fallback: pose_index nuevo
    for p in poses:
        if p.get("pose_index", -1) == pose_index:
            return p
    return poses[0] if poses else {}


def classify_with_topk(crop_img, clf, cam_name, valid_refs, k=3):
    """Classify and return top-k results."""
    if not clf._ref_embeddings:
        return []

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

    max_q, min_q = get_oriented_dims_mm(crop_img)
    cam_id = 1 if cam_name == "cenital" else 2

    filtered = [r for r in clf._ref_embeddings if (r["face"] % 10 == cam_id) and (r["part_ref"] in valid_refs)]
    if not filtered:
        filtered = [r for r in clf._ref_embeddings if (r["face"] % 10 == cam_id)]
    if not filtered:
        return []

    query_vec = clf._extract_embedding(clean_crop, size_info=(max_q, min_q))
    ref_matrix = np.stack([r["embedding"] for r in filtered])
    scores = ref_matrix @ query_vec

    # Aggregate by part_ref (max score)
    class_scores = {}
    for idx, r in enumerate(filtered):
        ref = r["part_ref"]
        s = float(scores[idx])
        if ref not in class_scores or s > class_scores[ref]:
            class_scores[ref] = s

    sorted_scores = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]


# ═══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN HTML
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Report Pieza {part_ref} — Pose {pose_index}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
h2 {{ color: #283593; margin-top: 30px; }}
.header-info {{ background: #e8eaf6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
.images-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }}
.images-grid img {{ width: 100%; border: 2px solid #ccc; border-radius: 4px; }}
.images-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }}
.images-grid-3 img {{ width: 100%; border: 2px solid #ccc; border-radius: 4px; }}
.images-grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }}
.images-grid-4 img {{ width: 100%; border: 2px solid #ccc; border-radius: 4px; }}
.img-label {{ text-align: center; font-weight: bold; margin-top: 5px; color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #e8eaf6; font-weight: 600; }}
.match {{ color: #2e7d32; font-weight: bold; }}
.mismatch {{ color: #c62828; font-weight: bold; }}
.score {{ font-family: monospace; }}
.color-swatch {{ display: inline-block; width: 20px; height: 20px; border: 1px solid #999; vertical-align: middle; margin-right: 5px; border-radius: 3px; }}
.section {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.verdict {{ font-size: 1.3em; padding: 10px; border-radius: 5px; text-align: center; margin: 20px 0; }}
.verdict.correct {{ background: #c8e6c9; color: #1b5e20; }}
.verdict.incorrect {{ background: #ffcdd2; color: #b71c1c; }}
</style>
</head>
<body>

<h1>🧱 Report de Diagnóstico — Pieza {part_ref}</h1>

<div class="header-info">
<strong>Referencia:</strong> {part_ref} &nbsp;|&nbsp;
<strong>Pose:</strong> {pose_index} &nbsp;|&nbsp;
<strong>Set:</strong> 75078-1 &nbsp;|&nbsp;
<strong>Generado:</strong> {timestamp}
</div>

<div class="section">
<h2>📷 1. Renders de la Pieza</h2>
<div class="images-grid">
<div>
<img src="data:image/png;base64,{img_cenital_b64}" alt="Cenital">
<div class="img-label">Cámara Cenital (0, 0, 15)</div>
</div>
<div>
<img src="data:image/png;base64,{img_lateral_b64}" alt="Lateral">
<div class="img-label">Cámara Lateral (15, 0, 2.5)</div>
</div>
</div>
</div>

<div class="section">
<h2>🔲 2. Bounding Boxes y Máscaras Detectadas</h2>
<div class="images-grid-3">
<div>
<img src="data:image/png;base64,{img_crop_cenital_b64}" alt="BBox Cenital Crop">
<div class="img-label">Zoom BBox Cenital (Original)</div>
</div>
<div>
<img src="data:image/png;base64,{img_crop_contour_cenital_b64}" alt="Contorno Cenital Crop">
<div class="img-label">segmentacion SAM</div>
</div>
<div>
<img src="data:image/png;base64,{img_crop_lateral_b64}" alt="BBox Lateral Crop">
<div class="img-label">Zoom BBox Lateral</div>
</div>
</div>
</div>

<div class="section">
<h2>🎨 3. Información de Color</h2>
<table>
<tr><th>Cámara / Origen</th><th>Color Detectado (Pixels / Máscara)</th><th>Color Catálogo LEGO Más Cercano (Inferido)</th></tr>
<tr>
  <td><strong>Cámara Cenital</strong></td>
  <td><span class="color-swatch" style="background:{color_detected_cenital_hex}"></span> {color_detected_cenital_code} ({color_detected_cenital_hex})</td>
  <td><span class="color-swatch" style="background:{color_nearest_cenital_hex}"></span> {color_nearest_cenital_code} — {color_nearest_cenital_name} ({color_nearest_cenital_hex})</td>
</tr>
<tr>
  <td><strong>Cámara Lateral</strong></td>
  <td><span class="color-swatch" style="background:{color_detected_lateral_hex}"></span> {color_detected_lateral_code} ({color_detected_lateral_hex})</td>
  <td><span class="color-swatch" style="background:{color_nearest_lateral_hex}"></span> {color_nearest_lateral_code} — {color_nearest_lateral_name} ({color_nearest_lateral_hex})</td>
</tr>
<tr>
  <td><strong>Catálogo (Ground Truth)</strong></td>
  <td colspan="2"><span class="color-swatch" style="background:{color_catalog_hex}"></span> {color_catalog_code} — {color_catalog_name} ({color_catalog_hex})</td>
</tr>
<tr>
  <td><strong>Match Color Cenital (Inferido vs Catálogo)</strong></td>
  <td colspan="2" class="{color_match_class}">{color_match_text}</td>
</tr>
</table>
</div>

<div class="section">
<h2>📐 4. Superficie Cenital</h2>
<p style="color:#555; font-size:0.9em;">
"Superficie estimada" es el área en mm² medida sobre la máscara segmentada
del crop cenital ({sam_label_text}) <strong>después de aplicar las correcciones
de perspectiva y distancia</strong> (división por el factor de magnificación
<code>(150/d_3D)²</code> para des-escorzar piezas excéntricas o con altura).
"BD, superficie silueta" es el valor de referencia
<code>zenith_silhouette_area</code> de <code>stable_poses_cache.json</code>:
silueta 2D estricta proyectada a −Z, sin caras laterales.
</p>
<table>
<tr><th>Campo</th><th>Valor</th></tr>
<tr>
  <td><strong>Superficie estimada</strong></td>
  <td><strong>{surface_estimated_corrected:.1f} mm²</strong></td>
</tr>
<tr>
  <td>BD, superficie silueta</td>
  <td>{surface_silhouette_db:.1f} mm²</td>
</tr>
<tr>
  <td><strong>Error relativo</strong></td>
  <td class="{surface_err_silh_class}">{surface_error_silhouette_pct:+.1f}%</td>
</tr>
</table>
</div>

<div class="section">
<h2>📏 5. Altura Lateral</h2>
<p style="color:#555; font-size:0.9em;">
La altura se mide a partir de la silueta lateral del crop usando segmentación
HSV del fondo y la <strong>mediana</strong> del perfil de columnas (mecanismo
original del script). <em>No</em> se le suma <code>stud_offset_mm</code> a la
altura nominal: la silueta lateral ya contiene los studs si la pose los tiene
visibles.
</p>
<table>
<tr><th>Campo</th><th>Valor</th></tr>
<tr><td>Altura binding box</td><td>{height_bbox_lat:.2f} mm</td></tr>
<tr><td>Altura estimada (corregida)</td><td><strong>{height_estimated:.2f} mm</strong></td></tr>
<tr><td>Altura real pose estable (BD, sin stud_offset añadido)</td><td>{height_nominal:.2f} mm</td></tr>
<tr><td>Factor magnificación perspectiva lateral</td><td>{mag_lat:.4f} — (d_nom / d_act)</td></tr>
<tr><td>Error relativo</td><td class="{height_error_class}">{height_error:.1f}%</td></tr>
</table>
</div>

<div class="section">
<h2>🧠 6. Clasificación DINOv2</h2>
<table>
<tr><th>Fuente</th><th>Top-1</th><th>Top-2</th><th>Top-3</th></tr>
<tr><td>DINOv2 Cenital</td>{dinov2_cenital_topk}</tr>
<tr><td>DINOv2 Lateral</td>{dinov2_lateral_topk}</tr>
<tr><td><strong>Fusión (70/30)</strong></td>{dinov2_fusion_topk}</tr>
</table>

<div class="verdict {verdict_class}">
<strong>Predicción final:</strong> {prediction} &nbsp;|&nbsp;
<strong>Ground Truth:</strong> {part_ref} &nbsp;|&nbsp;
<strong>Resultado:</strong> {verdict_emoji}
</div>
</div>

<div class="section" style="font-size: 0.85em; color: #666;">
<h2>ℹ️ Metadatos</h2>
<p><strong>Test file cenital:</strong> {test_file_cenital}</p>
<p><strong>Test file lateral:</strong> {test_file_lateral}</p>
<p><strong>Embeddings en BD:</strong> {embeddings_count}</p>
<p><strong>Pose info:</strong> {pose_info_json}</p>
</div>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def find_test_sample(part_ref: str, pose_index: int = None) -> dict:
    """Find a test sample for the given part_ref in test_dual metadata."""
    meta_path = os.path.join(project_root, "data", "test_dual", "test_metadata.json")
    if not os.path.exists(meta_path):
        print(f"[ERROR] No test metadata found at {meta_path}")
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for entry in meta.get("renders", []):
        if entry["ref"] == part_ref:
            if pose_index is not None and entry.get("pose_index") != pose_index:
                continue
            return entry
    # Fallback: any sample with this ref
    for entry in meta.get("renders", []):
        if entry["ref"] == part_ref:
            return entry
    return None


def generate_report(part_ref: str, pose_index: int = None):
    """Generate the full HTML report for a piece."""
    print(f"[Report] Generando report para pieza {part_ref}...")

    # 1. Find test sample
    sample = find_test_sample(part_ref, pose_index)
    if not sample:
        print(f"[ERROR] No se encontró muestra de test para {part_ref}. Ejecuta primero generate_test_set.py")
        return

    test_dir = os.path.join(project_root, "data", "test_dual")
    pose_idx = sample.get("pose_index", 0)

    # 2. Load images
    cen_meta = sample["cameras"]["cenital"]
    lat_meta = sample["cameras"]["lateral"]
    img_cenital = Image.open(os.path.join(test_dir, cen_meta["file_name"])).convert("RGB")
    img_lateral = Image.open(os.path.join(test_dir, lat_meta["file_name"])).convert("RGB")

    # 3. Draw bboxes
    bbox_cen = cen_meta["bbox_norm"]
    bbox_lat = lat_meta["bbox_norm"]
    img_bbox_cenital = draw_bbox_on_image(img_cenital, bbox_cen)
    img_bbox_lateral = draw_bbox_on_image(img_lateral, bbox_lat)

    # 4. Crop pieces
    iw, ih = img_cenital.size
    crop_cen = img_cenital.crop((
        max(0, int(bbox_cen[0] * iw)), max(0, int(bbox_cen[1] * ih)),
        min(iw, int(bbox_cen[2] * iw)), min(ih, int(bbox_cen[3] * ih))
    ))
    liw, lih = img_lateral.size
    crop_lat = img_lateral.crop((
        max(0, int(bbox_lat[0] * liw)), max(0, int(bbox_lat[1] * lih)),
        min(liw, int(bbox_lat[2] * liw)), min(lih, int(bbox_lat[3] * lih))
    ))

    # Generate BBox + Mask overlay and binary mask for cenital image
    mask_cen = segment_crop(crop_cen)

    # 5. Color detection
    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()

    # Detected raw color codes
    color_cen_code = clf._classify_color(crop_cen) if hasattr(clf, '_classify_color') else "?"
    color_lat_code = clf._classify_color(crop_lat) if hasattr(clf, '_classify_color') else "?"

    # Map color codes to hex (approximate)
    colors_lut = {
        "85": "#A0A5A9", "0": "#1B1B1B", "4": "#C91A09", "15": "#F9F9F9",
        "14": "#FAE595", "2": "#E3CC9D"
    }
    color_cen_hex = colors_lut.get(color_cen_code, "#808080")
    color_lat_hex = colors_lut.get(color_lat_code, "#808080")

    # Inferred closest set colors using Lab distance
    nearest_cen = find_nearest_set_color(crop_cen)
    nearest_lat = find_nearest_set_color(crop_lat)

    color_catalog = get_catalog_color(part_ref)
    
    # Match criteria (based on nearest inferred color matching the catalog color code)
    color_match = (nearest_cen["color_code"] == color_catalog["color_code"])

    # ─────────────────────────────────────────────────────────────────────
    # 6. Estimación de superficie cenital (sólo máscara, sin BBox)
    # ─────────────────────────────────────────────────────────────────────
    # Métrica primaria: superficie del crop a partir de la máscara segmentada
    # (conteo de píxeles / px_per_mm²). Refinamos con SAM + filtro de sombras
    # para mayor robustez frente a sombras proyectadas sobre la cinta.
    sam_mask_cen = segment_crop_sam(img_cenital, bbox_cen)
    if sam_mask_cen is not None and sam_mask_cen.size > 0 and np.any(sam_mask_cen):
        sam_mask_cen_clean = filter_shadows_in_mask(crop_cen, sam_mask_cen)
        if np.sum(sam_mask_cen_clean > 0) >= max(20, 0.05 * sam_mask_cen_clean.size):
            mask_for_area = sam_mask_cen_clean
            sam_label_text = "SAM + filtrado HSV de sombras"
        else:
            # Si el filtro deja la máscara casi vacía (p.ej. piezas oscuras),
            # nos quedamos con la SAM cruda.
            mask_for_area = sam_mask_cen
            sam_label_text = "SAM (sin filtro de sombras: pieza muy oscura)"
    else:
        mask_for_area = segment_crop(crop_cen)
        sam_label_text = "segmentación HSV (fallback, SAM no disponible)"
    surface_mask_estimated = float(np.sum(mask_for_area > 0)) / (PX_PER_MM_CENITAL ** 2)

    pose_info = get_stable_pose_info(part_ref, pose_idx)
    cache_lateral_h = pose_info.get("lateral_height") if pose_info else None
    cache_zen_obs   = pose_info.get("zenith_observable_area") if pose_info else None
    # `zenith_silhouette_area` (migration 010): silueta 2D estricta del
    # contorno proyectado a -Z, sin caras laterales. Es el valor que usamos
    # como referencia de BD para la sección "Superficie Cenital" del report.
    cache_zen_silh  = pose_info.get("zenith_silhouette_area") if pose_info else None

    # Silueta cenital real del mesh, según cache (LDraw + populate_stable_pose_dims).
    if cache_zen_obs:
        surface_nominal_raw = float(cache_zen_obs)
        surface_nominal_source_text = "(zenith_observable_area)"
    else:
        # Fallback: AABB nominal de la pieza si la pose no está cacheada.
        dims_fallback = get_part_dimensions(part_ref)
        L_fb, W_fb, H_fb = sorted(dims_fallback, reverse=True)
        surface_nominal_raw = L_fb * W_fb
        surface_nominal_source_text = "(fallback AABB nominal — pose sin cache)"

    # Detección de studs hacia arriba a partir de face_class.
    has_studs_on_top, n_studs_top, studs_reason = detect_studs_top_from_pose(pose_info, part_ref)
    if has_studs_on_top and n_studs_top > 0:
        # Descontamos el área visible de los studs si la pose tiene la base
        # apoyada (cara Bottom). La silueta cenital incluye los studs por
        # ser cilindros que sobresalen hacia +Z.
        studs_area_to_subtract = n_studs_top * STUD_AREA_MM2
        surface_nominal_mask = max(0.0, surface_nominal_raw - studs_area_to_subtract)
        studs_top_text = (f"<strong>Sí</strong> — {studs_reason}. "
                          f"Se restan {n_studs_top} studs × {STUD_AREA_MM2:.1f} mm² "
                          f"= {studs_area_to_subtract:.1f} mm²")
    else:
        surface_nominal_mask = surface_nominal_raw
        studs_top_text = f"No — {studs_reason}"

    # ─── Altura efectiva sobre la superficie de la cinta ───
    # Para la magnificación de perspectiva nos interesa la altura del plano
    # superior de la pieza (donde mira la cámara cenital).
    if cache_lateral_h:
        rest_height = float(cache_lateral_h)
    else:
        rest_height = get_part_dimensions(part_ref)[2]
    if has_studs_on_top:
        # En cenital, los studs sí asoman por encima de la base; la cara
        # superior efectiva está ~0.9 mm más arriba que `lateral_height`.
        face_top_z_mm = rest_height + STUD_HEIGHT_MM
    else:
        face_top_z_mm = rest_height

    # ─── Centro proyectado de la pieza en la imagen cenital → (Δx, Δy) en mm ───
    cx_cen = (bbox_cen[0] + bbox_cen[2]) / 2.0
    cy_cen = (bbox_cen[1] + bbox_cen[3]) / 2.0
    iw_cen, ih_cen = img_cenital.size
    piece_dx_mm = (cx_cen * iw_cen - iw_cen / 2.0) / PX_PER_MM_CENITAL
    piece_dy_mm = (ih_cen / 2.0 - cy_cen * ih_cen) / PX_PER_MM_CENITAL

    # Distancia 3D real desde la cámara cenital (0, 0, 150 mm) al centro de la
    # cara superior de la pieza (corrección de escorzo). El factor de
    # magnificación es proporcional a (h_cam / d_3D)².
    cenital_dist_3d_mm = math.sqrt(
        piece_dx_mm ** 2 + piece_dy_mm ** 2
        + (CAM_CENITAL_HEIGHT_MM - face_top_z_mm) ** 2
    )
    if cenital_dist_3d_mm > 0:
        magnification = (CAM_CENITAL_HEIGHT_MM / cenital_dist_3d_mm) ** 2
    else:
        magnification = 1.0
    surface_apparent_mask = surface_nominal_mask * magnification

    surface_error_mask = (
        abs(surface_mask_estimated - surface_apparent_mask) / surface_apparent_mask * 100
        if surface_apparent_mask > 0 else 0.0
    )
    surface_error_mask_class = "match" if surface_error_mask < 15 else "mismatch"

    # ─── Sección 4 simplificada (BD silueta vs estimación corregida) ───
    # `surface_mask_estimated` se mide con la calibración constante
    # `PX_PER_MM_CENITAL` (≈ válida en el origen del FOV). Para piezas
    # situadas fuera del centro o con altura no nula, la perspectiva
    # **amplifica** los píxeles medidos por un factor `magnification`
    # ((150/d_3D)²). Para obtener la "superficie estimada real" después
    # de corregir perspectiva y distancia, dividimos por ese factor:
    surface_estimated_corrected = (
        surface_mask_estimated / magnification if magnification > 0
        else surface_mask_estimated
    )
    # Referencia de BD: silueta 2D pura (sin caras laterales), si está;
    # en su defecto fallback al área observable o al AABB nominal.
    if cache_zen_silh:
        surface_silhouette_db = float(cache_zen_silh)
    elif cache_zen_obs:
        surface_silhouette_db = float(cache_zen_obs)
    else:
        surface_silhouette_db = surface_nominal_raw
    # Error relativo SIGNED: positivo si sobre-estima, negativo si sub-estima.
    surface_error_silhouette_pct = (
        (surface_estimated_corrected - surface_silhouette_db)
        / surface_silhouette_db * 100.0
        if surface_silhouette_db > 0 else 0.0
    )
    surface_err_silh_class = (
        "match" if abs(surface_error_silhouette_pct) < 15 else "mismatch"
    )

    # ─────────────────────────────────────────────────────────────────────
    # 7. Estimación de altura lateral (mecanismo original — NO modificar)
    # ─────────────────────────────────────────────────────────────────────
    # Altura nominal: tal cual viene del cache. NO sumamos stud_offset_mm
    # adicional aquí (la silueta lateral ya contiene los studs si los hay).
    height_nominal = rest_height

    # Medición tradicional: segmentación HSV del crop lateral + mediana del
    # perfil de columnas de la máscara (mecanismo original del script).
    mask_lat = segment_crop(crop_lat)
    col_heights = []
    for col in range(mask_lat.shape[1]):
        ys = np.where(mask_lat[:, col] > 0)[0]
        if len(ys) > 0:
            col_heights.append(int(ys.max() - ys.min()))
    if col_heights:
        height_measured_px = float(np.median(col_heights))
    else:
        height_measured_px = (bbox_lat[3] - bbox_lat[1]) * img_lateral.height
    height_bbox_lat = height_measured_px / PX_PER_MM_LATERAL

    # ─── Corrección de magnificación lateral ───
    # Cámara lateral en (150, 0, 25) mm. Necesitamos la distancia 3D real
    # desde la cámara al centro de la pieza para corregir el factor.
    d_horiz_lat = math.sqrt(
        (CAM_LATERAL_X_MM - piece_dx_mm) ** 2 + piece_dy_mm ** 2
    )
    cy_lat_norm = (bbox_lat[1] + bbox_lat[3]) / 2.0
    v_px = (img_lateral.height / 2.0) - cy_lat_norm * img_lateral.height
    theta = math.atan(v_px / CAM_FOCAL_PX)
    alpha = math.atan(CAM_LATERAL_Z_MM / CAM_LATERAL_X_MM)
    phi = theta - alpha
    z_piece_center = CAM_LATERAL_Z_MM + d_horiz_lat * math.tan(phi)
    d_act = math.sqrt(d_horiz_lat ** 2 + (CAM_LATERAL_Z_MM - z_piece_center) ** 2)
    d_nom = math.sqrt(CAM_LATERAL_X_MM ** 2 + CAM_LATERAL_Z_MM ** 2)
    mag_lat = d_nom / d_act if d_act > 0 else 1.0

    height_estimated = height_bbox_lat / mag_lat
    height_error = (
        abs(height_estimated - height_nominal) / height_nominal * 100
        if height_nominal > 0 else 0.0
    )
    height_error_class = "match" if height_error < 15 else "mismatch"

    # 8. DINOv2 classification
    valid_refs = SELECTED_PARTS
    topk_cen = classify_with_topk(crop_cen, clf, "cenital", valid_refs, k=3)
    topk_lat = classify_with_topk(crop_lat, clf, "lateral", valid_refs, k=3)

    # Fusion 70/30
    all_refs = set([r for r, _ in topk_cen] + [r for r, _ in topk_lat])
    fusion_scores = {}
    cen_dict = dict(topk_cen)
    lat_dict = dict(topk_lat)
    for ref in all_refs:
        fusion_scores[ref] = 0.7 * cen_dict.get(ref, 0.0) + 0.3 * lat_dict.get(ref, 0.0)
    topk_fusion = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    prediction = topk_fusion[0][0] if topk_fusion else "Desconocido"
    is_correct = (prediction == part_ref)

    # 9. Format top-k cells
    def format_topk_cells(topk):
        cells = []
        for i in range(3):
            if i < len(topk):
                ref, score = topk[i]
                cls = "match" if ref == part_ref else ""
                cells.append(f'<td class="{cls} score">{ref} ({score:.4f})</td>')
            else:
                cells.append('<td>—</td>')
        return "".join(cells)

    # 10. Embeddings count
    from database import supabase_client
    emb_count = supabase_client.count_embeddings()

    # 11. Generate HTML
    html = HTML_TEMPLATE.format(
        part_ref=part_ref,
        pose_index=pose_idx,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        img_cenital_b64=img_to_base64(img_bbox_cenital),
        img_lateral_b64=img_to_base64(img_bbox_lateral),
        
        img_crop_cenital_b64=img_to_base64(crop_cen),
        img_crop_contour_cenital_b64=img_to_base64(get_sam_segmented_crop(crop_cen, segment_crop_sam(img_cenital, bbox_cen))),
        img_crop_lateral_b64=img_to_base64(crop_lat),
        
        color_detected_cenital_code=color_cen_code,
        color_detected_cenital_hex=color_cen_hex,
        color_nearest_cenital_code=nearest_cen["color_code"],
        color_nearest_cenital_name=nearest_cen["color_name"],
        color_nearest_cenital_hex=nearest_cen["color_hex"],
        
        color_detected_lateral_code=color_lat_code,
        color_detected_lateral_hex=color_lat_hex,
        color_nearest_lateral_code=nearest_lat["color_code"],
        color_nearest_lateral_name=nearest_lat["color_name"],
        color_nearest_lateral_hex=nearest_lat["color_hex"],
        
        color_catalog_hex=color_catalog["color_hex"],
        color_catalog_code=color_catalog["color_code"],
        color_catalog_name=color_catalog["color_name"],
        color_match_class="match" if color_match else "mismatch",
        color_match_text="✓ Match (Cenital Inferido)" if color_match else "✗ Mismatch",
        
        # ── Sección 4: superficie cenital (silueta BD vs estimada
        # corregida por perspectiva y distancia) ──
        sam_label_text=sam_label_text,
        surface_estimated_corrected=surface_estimated_corrected,
        surface_silhouette_db=surface_silhouette_db,
        surface_error_silhouette_pct=surface_error_silhouette_pct,
        surface_err_silh_class=surface_err_silh_class,

        # ── Sección 5: altura lateral ──
        height_bbox_lat=height_bbox_lat,
        height_estimated=height_estimated,
        height_nominal=height_nominal,
        mag_lat=mag_lat,
        height_error=height_error,
        height_error_class=height_error_class,
        dinov2_cenital_topk=format_topk_cells(topk_cen),
        dinov2_lateral_topk=format_topk_cells(topk_lat),
        dinov2_fusion_topk=format_topk_cells(topk_fusion),
        prediction=prediction,
        verdict_class="correct" if is_correct else "incorrect",
        verdict_emoji="✅ CORRECTO" if is_correct else "❌ INCORRECTO",
        test_file_cenital=cen_meta["file_name"],
        test_file_lateral=lat_meta["file_name"],
        embeddings_count=emb_count,
        pose_info_json=json.dumps(pose_info, indent=2, default=str)[:500],
    )

    # 12. Save report
    reports_dir = os.path.join(project_root, "data", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"report_{part_ref}_pose{pose_idx:02d}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Report] ✅ Generado: {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera report HTML de diagnóstico para una pieza LEGO.")
    parser.add_argument("--ref", type=str, required=True, help="Referencia de la pieza (ej: 3001)")
    parser.add_argument("--pose", type=int, default=None, help="Índice de pose (opcional)")
    args = parser.parse_args()

    generate_report(args.ref, args.pose)
