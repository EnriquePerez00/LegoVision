# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/run_evaluation.py
==================================================
Pipeline de inferencia y evaluación para el setup de pieza única con 2 cámaras.
Implementa el Algoritmo de Decisión en Cascada:
  - Fase 1: Gating de Color (Cenital) — Estimación Dual CIELAB
  - Fase 2: Gating de Superficie Cenital con Calibración de Perspectiva
  - Fase 3: Gating de Altura Lateral
  - Fase 4: Fusión DINOv2 (Cenital 70% + Lateral 30%)

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


def find_closest_catalog_color(avg_rgb):
    """Busca el color permitido del catálogo más similar a avg_rgb en espacio CIELAB."""
    avg_lab = rgb_to_lab(avg_rgb)
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


def estimate_surface_area_sam_corrected(mask_cen: np.ndarray, bbox_norm: list, rest_h: float = 9.6) -> float:
    """Calcula el área en mm² corrigiendo por perspectiva.
    
    NOTA (Mejoras A y B): rest_h puede recibir la altura de la pose. Para evitar
    la sobrecorrección por perspectiva en piezas con plano inclinado (slopes)
    donde la altura varía, se prefiere pasar la "altura efectiva" (altura media)
    de la pose en lugar de la altura máxima (lateral_height).
    """
    try:
        num_pixels = np.sum(mask_cen > 0)
        cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
        cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
        
        cx_px = cx * 640.0
        cy_px = cy * 640.0
        
        dx_mm = (cx_px - 320.0) / 3.2
        dy_mm = (320.0 - cy_px) / 3.2
        r_mm = math.sqrt(dx_mm**2 + dy_mm**2)
        
        # Angle theta
        theta = math.atan(r_mm / 150.0)
        
        # Pixel-to-mm ratio taking into account distance/angle to camera
        d_cam = math.sqrt(r_mm**2 + (150.0 - rest_h)**2)
        px_per_mm = 480.0 / d_cam
        
        area_raw_mm2 = num_pixels / (px_per_mm ** 2)
        
        # Correct for visible side faces due to perspective angle theta
        w_bbox_mm = (bbox_norm[2] - bbox_norm[0]) * 640.0 / px_per_mm
        h_bbox_mm = (bbox_norm[3] - bbox_norm[1]) * 640.0 / px_per_mm
        perimeter_half = (w_bbox_mm + h_bbox_mm) / 2.0
        
        side_width_projected = (r_mm * rest_h) / (150.0 - rest_h)
        added_side_area_mm2 = perimeter_half * side_width_projected * 0.5
        
        area_corrected = area_raw_mm2 - added_side_area_mm2
        return max(0.1, area_corrected)
    except Exception:
        return np.sum(mask_cen > 0) / (3.2 ** 2)


def measure_lateral_height_mm_sam(mask: np.ndarray) -> float:
    try:
        ys, _ = np.where(mask > 0)
        if len(ys) > 0:
            height_px = max(ys) - min(ys)
            return height_px / PX_PER_MM_LATERAL
    except Exception:
        pass
    return mask.shape[0] / PX_PER_MM_LATERAL


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
    test_dir = os.path.join(project_root, "data", "test_dual")
    metadata_path = os.path.join(test_dir, "test_metadata.json")

    if not os.path.exists(metadata_path):
        log.error(f"Metadata no encontrada: {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    _t_eval_start = _time.perf_counter()
    log_execution_header(log, "run_evaluation.py",
                         test_dir=test_dir,
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

        # Log de estimaciones de color
        log.info(
            f"  [Color] Cenital: est2_seg=[{cen_est2_rgb[0]:.0f},{cen_est2_rgb[1]:.0f},{cen_est2_rgb[2]:.0f}]"
            f"->{cen_est2_catalog['color_name']} (decisión: {color_code_cen})"
        )
        log.info(
            f"  [Color] Lateral: est2_seg=[{lat_est2_rgb[0]:.0f},{lat_est2_rgb[1]:.0f},{lat_est2_rgb[2]:.0f}]"
            f"->{lat_est2_catalog['color_name']}"
        )

        # ── ALGORITMO EN CASCADA ──

        # Phase 1: Color gating (usando estimación dual cenital)
        parts_in_set = [p for p in REAL_SETS["75078-1"]["parts"] if p["ref"] in SELECTED_PARTS]
        valid_by_color = [p["ref"] for p in parts_in_set if p["color_code"] == color_code_cen]
        if not valid_by_color:
            valid_by_color = [p["ref"] for p in parts_in_set]

        # Phase 2: Surface gating (cenital, +/-20% using real mask area)
        valid_by_surface = []
        for ref in valid_by_color:
            dims = get_part_dimensions(ref)
            L, W, H = sorted(dims, reverse=True)
            configs = [(L * W, H), (L * H, W), (W * H, L)]
            for nom_area, rest_h in configs:
                filling_factor = 1.0
                if ref in ["6141", "98138", "4032", "3062", "59900"]:
                    filling_factor = 0.785
                elif ref == "2420":
                    filling_factor = 0.75
                elif ref in ["3039", "3298", "3037", "3665", "85984", "54200", "11477", "15068"]:
                    filling_factor = 0.85
                elif ref in ["2412", "2877"]:
                    filling_factor = 0.92

                obs_surface_real = estimate_surface_area_sam_corrected(mask_cen, [cx1, cy1, cx2, cy2], rest_h)
                nom_target = nom_area * filling_factor
                if sample_idx < 3 and ref == ref_gt:
                    log.info(f"    [DEBUG] Ref={ref}: mask_pixels={np.sum(mask_cen > 0)} | obs_surface_real={obs_surface_real:.2f} | nom_target={nom_target:.2f} | raw_mm2={np.sum(mask_cen > 0)/(3.2**2):.2f}")
                if 0.80 * nom_target <= obs_surface_real <= 1.20 * nom_target:
                    valid_by_surface.append(ref)
                    break
        if not valid_by_surface:
            valid_by_surface = valid_by_color

        # Phase 3: Height gating (lateral, +/-15%)
        measured_height = measure_lateral_height_mm_sam(mask_lat)
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

        results.append({
            "index": sample_idx,
            "ref_gt": ref_gt,
            "consensus_ref": consensus_ref,
            "consensus_score": round(consensus_score, 4),
            "is_correct": is_correct,
            "color_decision": color_code_cen,
            "color_cenital_est2_seg": cen_est2_catalog["color_name"],
            "color_lateral_est2_seg": lat_est2_catalog["color_name"],
            "yolo_conf_cenital": round(cen_yolo_conf, 3),
            "yolo_conf_lateral": round(lat_yolo_conf, 3),
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

    report_path = os.path.join(project_root, "data", "eval_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump({
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": round(accuracy, 2),
            "render_engine": "BLENDER_EEVEE",
            "resolution": "640x640",
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
