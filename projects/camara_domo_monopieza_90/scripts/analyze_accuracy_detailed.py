import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Add virtual environment site-packages to sys.path
venv_site_packages = "/Users/I764690/Code_personal/LegoVision/.venv/lib/python3.13/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compute_iou(boxA, boxB):
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

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

def find_closest_color_code(rgb_est, catalog_colors):
    if not catalog_colors:
        return "0", "Various", "#808080"
    lab_est = rgb_to_lab(rgb_est)
    best_dist = float("inf")
    best_color = catalog_colors[0]
    for c in catalog_colors:
        lab_ref = rgb_to_lab(c["rgb"])
        dL = lab_est[0] - lab_ref[0]
        da = lab_est[1] - lab_ref[1]
        db = lab_est[2] - lab_ref[2]
        dist = np.sqrt(0.2 * (dL ** 2) + (da ** 2) + (db ** 2))
        if dist < best_dist:
            best_dist = dist
            best_color = c
    return best_color["color_code"], best_color["color_name"], best_color["color_hex"]

def estimate_color_bbox(img_path, bbox):
    if not bbox or not os.path.exists(img_path):
        return [128.0, 128.0, 128.0]
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [128.0, 128.0, 128.0]
    cropped = img.crop((x1, y1, x2, y2))
    arr = np.array(cropped)
    return list(arr.mean(axis=(0,1)))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default="new_weights_eval.html", help="Nombre del archivo JSON del reporte en data/reports")
    parser.add_argument("--metadata", type=str, default=os.path.join(project_root, "data", "data100", "simulation_metadata.json"), help="Ruta de simulation_metadata.json")
    args = parser.parse_args()

    if os.path.isabs(args.report):
        eval_json_path = args.report
    else:
        eval_json_path = os.path.join(project_root, "data", "reports", args.report)
    metadata_path = args.metadata
    palette_path = os.path.join(project_root, "data", "color_calibration_palette.json")
    
    print(f"Cargando JSON de evaluación {eval_json_path}...")
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # Cargar paleta de color
    catalog_colors = []
    if os.path.exists(palette_path):
        with open(palette_path, "r", encoding="utf-8") as f:
            palette = json.load(f)
            for item in palette:
                catalog_colors.append({
                    "color_code": str(item.get("color_code", "")),
                    "color_name": item.get("color_name", "Unknown"),
                    "color_hex": item.get("color_hex", "#808080"),
                    "rgb": np.array(item.get("rgb_cenital", [128, 128, 128]), dtype=float)
                })

    # Cargar modelos YOLO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cargando modelos YOLO en {device}...")
    model_cen = YOLO(os.path.join(project_root, "models", "yolo_cenital_pose.pt")).to(device)
    model_lat = YOLO(os.path.join(project_root, "models", "yolo_frontal_pose.pt")).to(device)

    # Agrupar piezas por trayectoria global
    frames_list = meta_data.get("frames", [])
    global_piece_tracks = {}
    for frame in frames_list:
        offset = frame["belt_offset_mm"]
        f_name = frame["file_name"]
        for p in frame["visible_pieces"]:
            x_abs = offset - p["x_belt_local_mm"]
            y_abs = p["y_belt_local_mm"]
            
            matched_key = None
            for key in global_piece_tracks.keys():
                kx, ky = key
                if abs(x_abs - kx) < 1.5 and abs(y_abs - ky) < 1.5:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (x_abs, y_abs)
                global_piece_tracks[matched_key] = []
                
            global_piece_tracks[matched_key].append({
                "ref": p["ref"],
                "color_code": p["color_code"],
                "color_name": p.get("color_name", "Unknown"),
                "file_name": f_name,
                "bbox_cenital_norm": p["bbox_cenital_norm"],
                "bbox_frontal_norm": p.get("bbox_frontal_norm"),
                "x_belt_local_mm": p["x_belt_local_mm"],
                "y_belt_local_mm": p["y_belt_local_mm"]
            })

    iou_cen_list = []
    iou_lat_list = []
    surface_err_list = []
    height_err_list = []
    
    color_cen_matches = 0
    color_lat_matches = 0
    total_samples = len(eval_data["results"])

    print("Procesando muestras...")
    for idx, r in enumerate(eval_data["results"]):
        ref_gt = r["ref_gt"]
        color_code_gt = r["color_code_gt"]
        cenital_file_eval = r["cenital_file"]
        
        # Obtener el color_name_gt real
        color_name_gt = "Unknown"
        for c in catalog_colors:
            if c["color_code"] == str(color_code_gt):
                color_name_gt = c["color_name"]
                break

        # Buscar track coincidente para tener las bboxes de ground truth
        matched_track = None
        for key, track_obs in global_piece_tracks.items():
            has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt and str(o["color_code"]) == str(color_code_gt) for o in track_obs)
            if has_match:
                matched_track = track_obs
                break
        
        if not matched_track:
            for key, track_obs in global_piece_tracks.items():
                has_match = any(o["file_name"] == cenital_file_eval and o["ref"] == ref_gt for o in track_obs)
                if has_match:
                    matched_track = track_obs
                    break
        
        if matched_track:
            best_obs = min(matched_track, key=lambda x: abs(x["x_belt_local_mm"]))
            cen_file = best_obs["file_name"]
            lat_file = cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = best_obs["bbox_cenital_norm"]
            bbox_lat_gt = best_obs["bbox_frontal_norm"]
            if color_name_gt == "Unknown":
                color_name_gt = best_obs.get("color_name", "Unknown")
        else:
            cen_file = cenital_file_eval
            lat_file = r.get("lateral_file") or cen_file.replace(".png", "_frontal.png")
            bbox_cen_gt = None
            bbox_lat_gt = None

        path_cen_img = os.path.join(os.path.dirname(metadata_path), cen_file)
        path_lat_img = os.path.join(os.path.dirname(metadata_path), lat_file)

        # 1. Inferencia Cenital y IoU
        iou_cen = 0.0
        bbox_cen_inf = None
        if os.path.exists(path_cen_img) and bbox_cen_gt:
            res_cen = model_cen(path_cen_img, verbose=False, conf=0.15, imgsz=1024)
            best_iou = 0.0
            best_box = None
            if res_cen and len(res_cen[0].boxes) > 0:
                for box in res_cen[0].boxes:
                    det_box = box.xyxyn[0].cpu().numpy().tolist()
                    iou = compute_iou(bbox_cen_gt, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = det_box
            if best_box and best_iou > 0.1:
                bbox_cen_inf = best_box
                iou_cen = best_iou
        iou_cen_list.append(iou_cen)

        # 2. Inferencia Frontal/Lateral y IoU
        iou_lat = 0.0
        bbox_lat_inf = None
        if os.path.exists(path_lat_img) and bbox_lat_gt:
            res_lat = model_lat(path_lat_img, verbose=False, conf=0.15, imgsz=1024)
            best_iou = 0.0
            best_box = None
            if res_lat and len(res_lat[0].boxes) > 0:
                for box in res_lat[0].boxes:
                    det_box = box.xyxyn[0].cpu().numpy().tolist()
                    iou = compute_iou(bbox_lat_gt, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = det_box
            if best_box and best_iou > 0.1:
                bbox_lat_inf = best_box
                iou_lat = best_iou
        iou_lat_list.append(iou_lat)

        # 3. Color Cenital
        name_cen_inf = r.get("color_name_cen", "Unknown")
        cen_color_match = (str(name_cen_inf).strip().lower() == str(color_name_gt).strip().lower())
        if cen_color_match:
            color_cen_matches += 1

        # 4. Color Lateral
        color_lat_rgb = estimate_color_bbox(path_lat_img, bbox_lat_inf or bbox_lat_gt)
        _, name_lat_inf, _ = find_closest_color_code(color_lat_rgb, catalog_colors)
        lat_color_match = (str(name_lat_inf).strip().lower() == str(color_name_gt).strip().lower())
        if lat_color_match:
            color_lat_matches += 1

        # 5. Superficie / Altura errores relativos
        surface_err_list.append(abs(r.get("surface_error_rel_pct", 0.0)))
        height_err_list.append(abs(r.get("lateral_height_error_rel_pct", 0.0)))

    # Calcular estadísticas de IoU
    def get_stats(vals):
        vals = np.array(vals)
        bins = {
            "[0.0 - 0.1)": int(np.sum(vals < 0.1)),
            "[0.1 - 0.3)": int(np.sum((vals >= 0.1) & (vals < 0.3))),
            "[0.3 - 0.5)": int(np.sum((vals >= 0.3) & (vals < 0.5))),
            "[0.5 - 0.7)": int(np.sum((vals >= 0.5) & (vals < 0.7))),
            "[0.7 - 0.9)": int(np.sum((vals >= 0.7) & (vals < 0.9))),
            "[0.9 - 1.0]": int(np.sum(vals >= 0.9))
        }
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "bins": bins
        }

    def get_stats_error(vals):
        vals = np.array(vals)
        bins = {
            "[0% - 5%)": int(np.sum(vals < 5.0)),
            "[5% - 10%)": int(np.sum((vals >= 5.0) & (vals < 10.0))),
            "[10% - 20%)": int(np.sum((vals >= 10.0) & (vals < 20.0))),
            "[20% - 50%)": int(np.sum((vals >= 20.0) & (vals < 50.0))),
            "[50%+]": int(np.sum(vals >= 50.0))
        }
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "bins": bins
        }

    stats_cen = get_stats(iou_cen_list)
    stats_lat = get_stats(iou_lat_list)
    stats_surface = get_stats_error(surface_err_list)
    stats_height = get_stats_error(height_err_list)

    # Output Results
    print("\n" + "="*50)
    print("RESULTADOS DEL ANÁLISIS DE ACCURACY EN DATA100")
    print("="*50)
    
    print("\n--- 1. Bounding Box Cenital (Domo superior) ---")
    print(f"Mean IoU:   {stats_cen['mean']:.4f}")
    print(f"Median IoU: {stats_cen['median']:.4f}")
    print(f"Min IoU:    {stats_cen['min']:.4f}")
    print(f"Max IoU:    {stats_cen['max']:.4f}")
    print(f"Std Dev:    {stats_cen['std']:.4f}")
    print("Distribución de IoU:")
    for bin_name, count in stats_cen["bins"].items():
        print(f"  {bin_name}: {count} piezas ({count/total_samples*100:.1f}%)")

    print("\n--- 2. Bounding Box Lateral (Frontal) ---")
    print(f"Mean IoU:   {stats_lat['mean']:.4f}")
    print(f"Median IoU: {stats_lat['median']:.4f}")
    print(f"Min IoU:    {stats_lat['min']:.4f}")
    print(f"Max IoU:    {stats_lat['max']:.4f}")
    print(f"Std Dev:    {stats_lat['std']:.4f}")
    print("Distribución de IoU:")
    for bin_name, count in stats_lat["bins"].items():
        print(f"  {bin_name}: {count} piezas ({count/total_samples*100:.1f}%)")

    print("\n--- 3. Inferencia de Color ---")
    print(f"Precisión Color Cenital: {color_cen_matches} / {total_samples} ({color_cen_matches/total_samples*100:.1f}%)")
    print(f"Precisión Color Lateral: {color_lat_matches} / {total_samples} ({color_lat_matches/total_samples*100:.1f}%)")
    
    print("\n--- 4. Inferencia de Superficie Cenital (Área) ---")
    print(f"Mean Abs Error %:   {stats_surface['mean']:.2f}%")
    print(f"Median Abs Error %: {stats_surface['median']:.2f}%")
    print(f"Min Abs Error %:    {stats_surface['min']:.2f}%")
    print(f"Max Abs Error %:    {stats_surface['max']:.2f}%")
    print(f"Std Dev %:          {stats_surface['std']:.2f}%")
    print("Distribución de Error de Superficie:")
    for bin_name, count in stats_surface["bins"].items():
        print(f"  {bin_name}: {count} piezas ({count/total_samples*100:.1f}%)")

    print("\n--- 5. Inferencia de Altura Frontal (Superficie Frontal) ---")
    print(f"Mean Abs Error %:   {stats_height['mean']:.2f}%")
    print(f"Median Abs Error %: {stats_height['median']:.2f}%")
    print(f"Min Abs Error %:    {stats_height['min']:.2f}%")
    print(f"Max Abs Error %:    {stats_height['max']:.2f}%")
    print(f"Std Dev %:          {stats_height['std']:.2f}%")
    print("Distribución de Error de Altura:")
    for bin_name, count in stats_height["bins"].items():
        print(f"  {bin_name}: {count} piezas ({count/total_samples*100:.1f}%)")

    # Save a JSON file with the results for later use or validation
    summary = {
        "cenital_bbox": stats_cen,
        "lateral_bbox": stats_lat,
        "color_accuracy": {
            "cenital": {
                "correct": color_cen_matches,
                "total": total_samples,
                "percentage": color_cen_matches/total_samples*100
            },
            "lateral": {
                "correct": color_lat_matches,
                "total": total_samples,
                "percentage": color_lat_matches/total_samples*100
            }
        },
        "cenital_surface": stats_surface,
        "frontal_height": stats_height
    }
    with open(os.path.join(project_root, "data", "reports", "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen guardado en {os.path.join(project_root, 'data', 'reports', 'metrics_summary.json')}")

if __name__ == "__main__":
    main()
