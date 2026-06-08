# -*- coding: utf-8 -*-
"""generate_segmentation_report_v2.py
======================================
Genera un report HTML de diagnóstico de segmentación (v2) para piezas específicas.
Implementa 4 estrategias de segmentación, genera fondo sintético y aplica
corrección de perspectiva avanzada basada en la posición de la pieza.

Uso:
    .venv/bin/python 2camaras_pieza_unica/scripts/generate_segmentation_report_v2.py
"""
import os, sys, json, base64, math
from datetime import datetime
from io import BytesIO
import numpy as np
import cv2
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from config_loader import cfg
from ultralytics import YOLO, SAM

# ── Utilidades ──
def img_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def np_to_base64(img_np: np.ndarray, fmt="PNG") -> str:
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return img_to_base64(pil_img, fmt)


# ── Estrategias de Segmentación ──
def segment_color_distance(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 1: Distancia euclidiana al color del fondo azul petróleo."""
    bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
    dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_color, axis=2)
    mask = (dist > 18.0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def segment_hsv_background(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 2: Exclusión del fondo por rango HSV (el fondo tiene Hue cromático ~100-110)."""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Fondo azul petróleo: H~90-120, S~40-220, V~20-140
    lower_bg = np.array([90, 40, 20], dtype=np.uint8)
    upper_bg = np.array([120, 255, 150], dtype=np.uint8)
    mask_bg = cv2.inRange(img_hsv, lower_bg, upper_bg)
    mask_piece = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_piece


def segment_canny_floodfill(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 3: Canny edges + flood-fill desde esquinas (fondo)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    median_val = np.median(blurred)
    low_thresh = int(max(0, 0.5 * median_val))
    high_thresh = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.dilate(edges, kernel, iterations=2)
    h, w = edges_closed.shape
    mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_img = edges_closed.copy()
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for cx, cy in corners:
        if flood_img[cy, cx] == 0:
            cv2.floodFill(flood_img, mask_ff, (cx, cy), 128)
    mask_piece = ((flood_img != 128) & (flood_img != 255)).astype(np.uint8) * 255
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_CLOSE, kernel2, iterations=2)
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_OPEN, kernel2, iterations=1)
    return mask_piece


def segment_combined(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 4: Fusión por votación mayoritaria (2 de 3)."""
    m1 = (segment_color_distance(img_rgb) > 0).astype(np.uint8)
    m2 = (segment_hsv_background(img_rgb) > 0).astype(np.uint8)
    m3 = (segment_canny_floodfill(img_rgb) > 0).astype(np.uint8)
    combined = (m1 + m2 + m3) >= 2
    mask = combined.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def segment_grabcut(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 5: GrabCut interactivo guiado por los bordes del crop (BBox)."""
    h, w = img_rgb.shape[:2]
    if h < 5 or w < 5:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Inicializar la máscara de GrabCut
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:] = cv2.GC_PR_FGD  # Todo es primer plano probable por defecto
    
    # Asumimos que los bordes externos son fondo probable (GC_PR_BGD)
    border_px = max(1, min(w // 10, h // 10, 3))
    mask[0:border_px, :] = cv2.GC_PR_BGD
    mask[-border_px:, :] = cv2.GC_PR_BGD
    mask[:, 0:border_px] = cv2.GC_PR_BGD
    mask[:, -border_px:] = cv2.GC_PR_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        # Ejecutar GrabCut con 5 iteraciones
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # Pixels marcados como GC_FGD (1) o GC_PR_FGD (3) son objeto
        mask_out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return mask_out
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)


def segment_watershed(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 6: Segmentación por Watershed con marcadores de fondo y primer plano."""
    h, w = img_rgb.shape[:2]
    # Usar distancia de color para obtener semillas seguras
    bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
    dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_color, axis=2)
    
    # Semillas de fondo definitivas: muy similares al color de la cinta
    sure_bg = (dist < 10.0).astype(np.uint8) * 255
    # Semillas de primer plano definitivas: muy lejanas al color de la cinta
    sure_fg = (dist > 25.0).astype(np.uint8) * 255
    
    # Área desconocida entre ambos
    unknown = cv2.subtract(cv2.bitwise_not(sure_bg), sure_fg)
    
    # Etiquetar componentes del primer plano seguro
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Añadir 1 a todas las etiquetas para que el fondo seguro sea 1 (no 0)
    markers = markers + 1
    # Marcar el área desconocida con 0
    markers[unknown == 255] = 0
    
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(img_bgr, markers)
        # La pieza es cualquier etiqueta > 1
        mask_out = np.zeros((h, w), dtype=np.uint8)
        mask_out[markers > 1] = 255
        return mask_out
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)



def get_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def apply_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    overlay = img_rgb.copy()
    overlay[mask > 0] = (
        (1 - alpha) * img_rgb[mask > 0] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# ── YOLO Detection ──
def yolo_detect_bbox(model, img_path, conf_threshold=0.25):
    try:
        results = model(img_path, verbose=False, conf=conf_threshold)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            bbox_norm = boxes.xyxyn[best_idx].cpu().numpy().tolist()
            conf = float(boxes.conf[best_idx].cpu().numpy())
            return bbox_norm, conf
    except Exception:
        pass
    return None, 0.0


# ── Corrección de Perspectiva ──
def calculate_perspective_corrected_area(
    num_pixels: float,
    cx_px: float,
    cy_px: float,
    h_mm: float,
    L_nominal: float,
    W_nominal: float,
    px_per_mm_nom: float = 3.2
) -> tuple:
    """
    Calcula la superficie real corregida eliminando la magnificación del plano superior
    y la superficie adicional visible de las caras laterales de la pieza debido al offset radial.
    """
    # Offset desde el centro óptico (320, 320)
    dx_px = cx_px - 320.0
    dy_px = cy_px - 320.0
    
    # En milímetros
    dx_mm = dx_px / px_per_mm_nom
    dy_mm = dy_px / px_per_mm_nom
    r_mm = math.sqrt(dx_mm**2 + dy_mm**2)
    
    # Pixel-to-mm ratio taking into account distance/angle to camera
    d_cam = math.sqrt(r_mm**2 + (150.0 - h_mm)**2)
    px_per_mm = 480.0 / d_cam
    
    area_raw_mm2 = num_pixels / (px_per_mm ** 2)
    
    # Correct for visible side faces due to perspective angle
    perimeter_half = (L_nominal + W_nominal) / 2.0
    side_width_projected = (r_mm * h_mm) / (150.0 - h_mm)
    added_side_area_mm2 = perimeter_half * side_width_projected * 0.5
    
    area_corrected_mm2 = area_raw_mm2 - added_side_area_mm2
    
    # Magnificación equivalente
    magnification = (d_cam / 150.0) ** 2
    
    return max(0.1, area_corrected_mm2), r_mm, added_side_area_mm2, magnification


def segment_sam(img_full_rgb: np.ndarray, bbox_px: list, sam_model) -> np.ndarray:
    """Estrategia 7: Segment Anything Model (SAM) con box prompt."""
    try:
        results = sam_model(img_full_rgb, bboxes=[bbox_px], verbose=False)
        if results and results[0].masks is not None:
            full_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            x1, y1, x2, y2 = bbox_px
            crop_mask = full_mask[y1:y2, x1:x2]
            return crop_mask
    except Exception as e:
        print(f"[SAM ERROR]: {e}")
    h = bbox_px[3] - bbox_px[1]
    w = bbox_px[2] - bbox_px[0]
    return np.zeros((h, w), dtype=np.uint8)


# ── Generación HTML ──
def generate_html_report(piece_ref, pose_idx, crop_img_rgb, bbox_center, h_nominal, L_nominal, W_nominal, nominal_area, strategies_results, output_path):
    crop_b64 = np_to_base64(cv2.cvtColor(crop_img_rgb, cv2.COLOR_RGB2BGR))
    
    cx_px, cy_px = bbox_center
    
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte de Segmentación Avanzada - Pieza {piece_ref} Pose {pose_idx}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #ffd700; margin-top: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background: #16213e; border-radius: 12px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
        .card img {{ width: 100%; border-radius: 8px; image-rendering: pixelated; }}
        .card h3 {{ color: #00d4ff; margin: 10px 0 5px; font-size: 14px; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #ffd700; }}
        .metric-label {{ font-size: 12px; color: #aaa; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; }}
        .stat-box {{ background: #0f3460; padding: 10px 15px; border-radius: 8px; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 8px 12px; text-align: center; border: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        td {{ background: #16213e; }}
        .highlight {{ color: #00ff88; font-weight: bold; }}
        .warning {{ color: #ff9900; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🔍 Diagnóstico de Segmentación v2 — Pieza {piece_ref} (Pose {pose_idx})</h1>
    <p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>📷 Crop de Bounding Box</h2>
    <div class="grid">
        <div class="card">
            <h3>Crop del Bounding Box YOLO / Cenital</h3>
            <img src="data:image/png;base64,{crop_b64}" />
            <div class="stats">
                <div class="stat-box">
                    <div class="metric">{crop_img_rgb.shape[1]}x{crop_img_rgb.shape[0]}</div>
                    <div class="metric-label">Dimensiones (px)</div>
                </div>
                <div class="stat-box">
                    <div class="metric">({cx_px:.1f}, {cy_px:.1f})</div>
                    <div class="metric-label">Centro BBox (px)</div>
                </div>
            </div>
        </div>
    </div>
    
    <h2>🎯 Resultados de Segmentación y Corrección de Perspectiva</h2>
    <table>
        <tr>
            <th>Estrategia</th>
            <th>Área Medida (px)</th>
            <th>Área Medida (mm²)</th>
            <th>Desplazamiento r (mm)</th>
            <th>Magnificación</th>
            <th>Área Corregida (mm²)</th>
            <th>Nominal (mm²)</th>
            <th>Error Relativo Original</th>
            <th>Error Relativo Corregido</th>
        </tr>
""")
    
    for strat_name, mask, overlay in strategies_results:
        area_px = int(np.sum(mask > 0))
        area_raw_mm2 = area_px / (3.2 ** 2)
        
        # Aplicar la corrección matemática
        area_corr, r_mm, added_side, mag = calculate_perspective_corrected_area(
            area_px, cx_px, cy_px, h_nominal, L_nominal, W_nominal
        )
        
        err_orig = abs(area_raw_mm2 - nominal_area) / nominal_area * 100
        err_corr = abs(area_corr - nominal_area) / nominal_area * 100
        
        html_parts.append(f"""
        <tr>
            <td>{strat_name}</td>
            <td class="highlight">{area_px:,}</td>
            <td>{area_raw_mm2:.2f}</td>
            <td>{r_mm:.2f}</td>
            <td>{mag:.3f}</td>
            <td class="highlight">{area_corr:.2f}</td>
            <td>{nominal_area:.2f}</td>
            <td class="warning">{err_orig:.1f}%</td>
            <td class="highlight">{err_corr:.1f}%</td>
        </tr>
""")
        
    html_parts.append("""
    </table>
    
    <h2>🖼️ Visualización de Máscaras</h2>
    <div class="grid">
""")
    
    for strat_name, mask, overlay in strategies_results:
        overlay_b64 = np_to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        html_parts.append(f"""
        <div class="card">
            <h3>{strat_name}</h3>
            <img src="data:image/png;base64,{overlay_b64}" />
        </div>
""")
        
    html_parts.append("""
    </div>
</body>
</html>
""")
    
    html_content = "".join(html_parts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Segmentation Report v2] Generado: {output_path}")


# ── Main ──
def main():
    test_dir = os.path.join(project_root, "data", "test_dual")
    metadata_path = os.path.join(test_dir, "test_metadata.json")
    reports_dir = os.path.join(project_root, "data", "reports")

    # 1. Generar imagen de fondo sintética
    bg_img = np.zeros((640, 640, 3), dtype=np.uint8)
    bg_img[:] = [37, 65, 84]  # Color de la cinta RGB(37,65,84) en BGR: [84, 65, 37]
    bg_path = os.path.join(test_dir, "synthetic_background.png")
    os.makedirs(test_dir, exist_ok=True)
    cv2.imwrite(bg_path, bg_img)
    print(f"[Background] Guardado fondo sintético en: {bg_path}")

    if not os.path.exists(metadata_path):
        print(f"[Segmentation] ERROR: No se encontró test_metadata.json en {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar modelos
    yolo_path = os.path.join(project_root, "models", "yolo_cenital.pt")
    if os.path.exists(yolo_path):
        print(f"[Segmentation] Cargando YOLO cenital: {yolo_path}")
        yolo_model = YOLO(yolo_path)
    else:
        print(f"[Segmentation] YOLO no encontrado, usando bbox de metadata")
        yolo_model = None

    print("[Segmentation] Cargando MobileSAM (mobile_sam.pt)...")
    sam_model = SAM("mobile_sam.pt")

    # Piezas objetivo y sus dimensiones nominales
    # Ref: (L_nominal, W_nominal, H_nominal)
    piece_dims = {
        "3001": (32.0, 16.0, 9.6),
        "4070": (8.0, 8.0, 9.6),
        "59900": (8.0, 8.0, 3.2),  # Variable
    }

    target_pieces = [
        {"ref": "3001", "pose_index": 1, "nominal_area": 512.0},
        {"ref": "4070", "pose_index": 3, "nominal_area": 64.0},
        {"ref": "59900", "pose_index": 0, "nominal_area": 64.0},
    ]

    for target in target_pieces:
        ref_target = target["ref"]
        pose_target = target["pose_index"]
        nominal_area = target["nominal_area"]
        L_nom, W_nom, H_nom = piece_dims[ref_target]

        sample = None
        for entry in meta_data.get("renders", []):
            if entry["ref"] == ref_target and entry["pose_index"] == pose_target:
                sample = entry
                break
        
        if sample is None:
            for entry in meta_data.get("renders", []):
                if entry["ref"] == ref_target:
                    sample = entry
                    break

        if sample is None:
            print(f"[Segmentation] No se encontró muestra para {ref_target} pose {pose_target}")
            continue

        cen_meta = sample["cameras"]["cenital"]
        cen_path = os.path.join(test_dir, cen_meta["file_name"])
        if not os.path.exists(cen_path):
            print(f"[Segmentation] Imagen no encontrada: {cen_path}")
            continue

        img_full = cv2.imread(cen_path)
        img_full_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        ih, iw = img_full.shape[:2]

        # Detectar bbox con YOLO
        bbox = None
        if yolo_model is not None:
            bbox, conf = yolo_detect_bbox(yolo_model, cen_path)

        if bbox is None:
            bbox = cen_meta["bbox_norm"]

        x1, y1, x2, y2 = bbox
        crop_x1 = max(0, int(x1 * iw))
        crop_y1 = max(0, int(y1 * ih))
        crop_x2 = min(iw, int(x2 * iw))
        crop_y2 = min(ih, int(y2 * ih))

        crop_rgb = img_full_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        if crop_rgb.size == 0:
            print(f"[Segmentation] Crop vacío para {ref_target}")
            continue

        bbox_center = ((x1 + x2) / 2.0 * iw, (y1 + y2) / 2.0 * ih)

        print(f"[Segmentation] Procesando {ref_target} pose {pose_target} — crop {crop_rgb.shape[1]}x{crop_rgb.shape[0]}")

        # Aplicar las 7 estrategias de segmentación
        strategies = [
            ("1. Distancia Color (actual)", segment_color_distance),
            ("2. HSV Background Exclusion", segment_hsv_background),
            ("3. Canny + Flood-Fill", segment_canny_floodfill),
            ("4. Fusión Mayoritaria (2/3)", segment_combined),
            ("5. GrabCut (BBox-guided)", segment_grabcut),
            ("6. Watershed con Marcadores", segment_watershed),
            ("7. SAM (Segment Anything)", segment_sam),
        ]

        results = []
        for name, func in strategies:
            if name.startswith("7. SAM"):
                mask_raw = func(img_full_rgb, [crop_x1, crop_y1, crop_x2, crop_y2], sam_model)
            else:
                mask_raw = func(crop_rgb)
            mask_clean = get_largest_component(mask_raw)
            overlay = apply_mask_overlay(crop_rgb, mask_clean, color=(0, 255, 0), alpha=0.35)
            results.append((name, mask_clean, overlay))

        # Generar HTML v2
        output_path = os.path.join(reports_dir, f"segmentation_v2_{ref_target}_pose{pose_target:02d}.html")
        generate_html_report(ref_target, pose_target, crop_rgb, bbox_center, H_nom, L_nom, W_nom, nominal_area, results, output_path)

    print("\n[Segmentation] ✅ Todos los reports v2 generados en:", reports_dir)


if __name__ == "__main__":
    main()
