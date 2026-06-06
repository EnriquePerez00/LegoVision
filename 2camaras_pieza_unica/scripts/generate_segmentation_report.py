# -*- coding: utf-8 -*-
"""generate_segmentation_report.py
====================================
Genera un report HTML de diagnóstico de segmentación para piezas específicas.
Muestra: bounding box original, máscara segmentada, y superficie interior en píxeles.

Uso:
    ../.venv/bin/python scripts/generate_segmentation_report.py
"""
import os, sys, json, base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from ultralytics import YOLO


# ── Utilidades ──
def img_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def np_to_base64(img_np: np.ndarray, fmt="PNG") -> str:
    """Convierte numpy array (BGR o grayscale) a base64 PNG."""
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
    # Fondo azul petróleo: H~95-115, S~80-200, V~30-120
    lower_bg = np.array([90, 40, 20], dtype=np.uint8)
    upper_bg = np.array([120, 220, 140], dtype=np.uint8)
    mask_bg = cv2.inRange(img_hsv, lower_bg, upper_bg)
    # La pieza es todo lo que NO es fondo
    mask_piece = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_piece


def segment_canny_floodfill(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 3: Canny edges + flood-fill desde esquinas (fondo) para detectar contorno cerrado."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    # Canny con umbrales adaptativos
    median_val = np.median(blurred)
    low_thresh = int(max(0, 0.5 * median_val))
    high_thresh = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    # Dilatar bordes para cerrar gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.dilate(edges, kernel, iterations=2)
    # Flood-fill desde las 4 esquinas (que son fondo)
    h, w = edges_closed.shape
    mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_img = edges_closed.copy()
    # Fill desde esquinas
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for cx, cy in corners:
        if flood_img[cy, cx] == 0:
            cv2.floodFill(flood_img, mask_ff, (cx, cy), 128)
    # Todo lo que NO fue llenado (ni es borde) es interior (pieza)
    mask_piece = ((flood_img != 128) & (flood_img != 255)).astype(np.uint8) * 255
    # Morphología para limpiar
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_CLOSE, kernel2, iterations=2)
    mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_OPEN, kernel2, iterations=1)
    return mask_piece


def segment_combined(img_rgb: np.ndarray) -> np.ndarray:
    """Estrategia 4: Fusión por votación mayoritaria de las 3 estrategias."""
    m1 = (segment_color_distance(img_rgb) > 0).astype(np.uint8)
    m2 = (segment_hsv_background(img_rgb) > 0).astype(np.uint8)
    m3 = (segment_canny_floodfill(img_rgb) > 0).astype(np.uint8)
    # Votación: si 2 de 3 dicen "pieza" → es pieza
    combined = (m1 + m2 + m3) >= 2
    mask = combined.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """Retorna solo el mayor componente conectado de la máscara."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # Label 0 es fondo, buscar el mayor entre labels 1..N
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def apply_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Aplica una máscara coloreada semi-transparente sobre la imagen."""
    overlay = img_rgb.copy()
    overlay[mask > 0] = (
        (1 - alpha) * img_rgb[mask > 0] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    # Dibujar contorno
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# ── YOLO Detection ──
def yolo_detect_bbox(model, img_path, conf_threshold=0.25):
    """Detecta bbox YOLO. Retorna [x1,y1,x2,y2] normalizadas o None."""
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


# ── Generación HTML ──
def generate_html_report(piece_ref, pose_idx, crop_img_rgb, strategies_results, output_path):
    """Genera un HTML standalone con visualización de segmentación."""
    
    # Imagen original del crop
    crop_b64 = np_to_base64(cv2.cvtColor(crop_img_rgb, cv2.COLOR_RGB2BGR))
    
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Segmentación - Pieza {piece_ref} Pose {pose_idx}</title>
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
    </style>
</head>
<body>
    <h1>🔍 Diagnóstico de Segmentación — Pieza {piece_ref} (Pose {pose_idx})</h1>
    <p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>📷 Bounding Box Cenital (Crop Original)</h2>
    <div class="grid">
        <div class="card">
            <h3>Crop del Bounding Box YOLO</h3>
            <img src="data:image/png;base64,{crop_b64}" />
            <div class="stats">
                <div class="stat-box">
                    <div class="metric">{crop_img_rgb.shape[1]}x{crop_img_rgb.shape[0]}</div>
                    <div class="metric-label">Dimensiones (px)</div>
                </div>
                <div class="stat-box">
                    <div class="metric">{crop_img_rgb.shape[1] * crop_img_rgb.shape[0]}</div>
                    <div class="metric-label">Total píxeles</div>
                </div>
            </div>
        </div>
    </div>
    
    <h2>🎯 Estrategias de Segmentación</h2>
    <div class="grid">
""")
    
    for strat_name, mask, overlay in strategies_results:
        area_px = int(np.sum(mask > 0))
        total_px = mask.shape[0] * mask.shape[1]
        pct = (area_px / total_px * 100) if total_px > 0 else 0
        
        mask_b64 = np_to_base64(mask)
        overlay_b64 = np_to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        html_parts.append(f"""
        <div class="card">
            <h3>{strat_name}</h3>
            <img src="data:image/png;base64,{overlay_b64}" />
            <div class="stats">
                <div class="stat-box">
                    <div class="metric highlight">{area_px:,}</div>
                    <div class="metric-label">Superficie Interior (px)</div>
                </div>
                <div class="stat-box">
                    <div class="metric">{pct:.1f}%</div>
                    <div class="metric-label">% del Crop</div>
                </div>
            </div>
        </div>
""")
    
    html_parts.append("""
    </div>
    
    <h2>🖼️ Máscaras en Blanco/Negro</h2>
    <div class="grid">
""")
    
    for strat_name, mask, _ in strategies_results:
        mask_b64 = np_to_base64(mask)
        html_parts.append(f"""
        <div class="card">
            <h3>{strat_name}</h3>
            <img src="data:image/png;base64,{mask_b64}" />
        </div>
""")
    
    # Tabla comparativa
    html_parts.append("""
    </div>
    
    <h2>📊 Comparativa de Superficies</h2>
    <table>
        <tr><th>Estrategia</th><th>Superficie (px)</th><th>% del Crop</th></tr>
""")
    
    for strat_name, mask, _ in strategies_results:
        area_px = int(np.sum(mask > 0))
        total_px = mask.shape[0] * mask.shape[1]
        pct = (area_px / total_px * 100) if total_px > 0 else 0
        html_parts.append(f"""
        <tr><td>{strat_name}</td><td class="highlight">{area_px:,}</td><td>{pct:.1f}%</td></tr>
""")
    
    html_parts.append("""
    </table>
</body>
</html>
""")
    
    html_content = "".join(html_parts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Segmentation Report] Generado: {output_path}")


# ── Main ──
def main():
    test_dir = os.path.join(project_root, "data", "test_dual")
    metadata_path = os.path.join(test_dir, "test_metadata.json")
    reports_dir = os.path.join(project_root, "data", "reports")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar modelo YOLO cenital
    yolo_path = os.path.join(project_root, "models", "yolo_cenital.pt")
    if os.path.exists(yolo_path):
        print(f"[Segmentation] Cargando YOLO cenital: {yolo_path}")
        yolo_model = YOLO(yolo_path)
    else:
        print(f"[Segmentation] YOLO no encontrado, usando bbox de metadata")
        yolo_model = None

    # Piezas objetivo: las de los últimos 3 reports
    target_pieces = [
        {"ref": "3001", "pose_index": 1},
        {"ref": "4070", "pose_index": 3},
        {"ref": "59900", "pose_index": 0},
    ]

    for target in target_pieces:
        ref_target = target["ref"]
        pose_target = target["pose_index"]

        # Buscar primera muestra que coincida
        sample = None
        for entry in meta_data.get("renders", []):
            if entry["ref"] == ref_target and entry["pose_index"] == pose_target:
                sample = entry
                break
        
        if sample is None:
            # Fallback: cualquier muestra de esa pieza
            for entry in meta_data.get("renders", []):
                if entry["ref"] == ref_target:
                    sample = entry
                    break

        if sample is None:
            print(f"[Segmentation] No se encontró muestra para {ref_target} pose {pose_target}")
            continue

        # Obtener imagen cenital
        cen_meta = sample["cameras"]["cenital"]
        cen_path = os.path.join(test_dir, cen_meta["file_name"])
        if not os.path.exists(cen_path):
            print(f"[Segmentation] Imagen no encontrada: {cen_path}")
            continue

        img_full = cv2.imread(cen_path)
        img_full_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        ih, iw = img_full.shape[:2]

        # Detectar bbox con YOLO (o fallback a metadata)
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

        print(f"[Segmentation] Procesando {ref_target} pose {pose_target} — crop {crop_rgb.shape[1]}x{crop_rgb.shape[0]}")

        # Aplicar las 4 estrategias de segmentación
        strategies = [
            ("1. Distancia Color (actual)", segment_color_distance),
            ("2. HSV Background Exclusion", segment_hsv_background),
            ("3. Canny + Flood-Fill", segment_canny_floodfill),
            ("4. Fusión Mayoritaria (2/3)", segment_combined),
        ]

        results = []
        for name, func in strategies:
            mask_raw = func(crop_rgb)
            mask_clean = get_largest_component(mask_raw)
            overlay = apply_mask_overlay(crop_rgb, mask_clean, color=(0, 255, 0), alpha=0.35)
            results.append((name, mask_clean, overlay))

        # Generar HTML
        output_path = os.path.join(reports_dir, f"segmentation_{ref_target}_pose{pose_target:02d}.html")
        generate_html_report(ref_target, pose_target, crop_rgb, results, output_path)

    print("\n[Segmentation] ✅ Todos los reports generados en:", reports_dir)


if __name__ == "__main__":
    main()
