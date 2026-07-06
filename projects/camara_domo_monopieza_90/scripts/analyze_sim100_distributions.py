import json
import numpy as np

def compute_iou(boxA, boxB):
    # box: [xmin, ymin, xmax, ymax] (normalized)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

with open("projects/camara_domo_monopieza_90/data/simulation_100_inferencia") as f:
    preds = json.load(f)

with open("projects/camara_domo_monopieza_90/data/simulation_100/simulation_metadata.json") as f:
    meta = json.load(f)

# Build a fast lookup for ground truth by frame
gt_by_frame = {}
for frame in meta["frames"]:
    gt_by_frame[frame["file_name"].replace(".png", "")] = frame["visible_pieces"]

iou_cen_list = []
iou_lat_list = []
error_area_list = []
error_height_list = []
height_rel_err_by_gt = {}
area_rel_err_by_gt = {}

for tid, track in preds.items():
    for obs in track["history"]:
        frame_id = obs["frame_id"]
        if frame_id not in gt_by_frame:
            continue
        gts = gt_by_frame[frame_id]
        
        best_iou = 0
        best_gt = None
        for gt in gts:
            iou = compute_iou(obs["bbox_cen"], gt["bbox_cenital_norm"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
                
        if best_gt is None or best_iou < 0.1:
            continue
            
        iou_cen_list.append(best_iou)
        
        if "bbox_lat" in obs and obs["bbox_lat"] != [0,0,1,1]:
            iou_lat = compute_iou(obs["bbox_lat"], best_gt["bbox_frontal_norm"])
            iou_lat_list.append(iou_lat)
            
            gt_h = best_gt["lateral_height_gt"]
            est_h = obs["height"]
            if gt_h > 0:
                err_h = abs(est_h - gt_h)
                error_height_list.append(err_h)
                
                h_key = round(gt_h, 1)
                if h_key not in height_rel_err_by_gt:
                    height_rel_err_by_gt[h_key] = []
                height_rel_err_by_gt[h_key].append(err_h / gt_h)

        gt_area = best_gt["zenith_silhouette_area_gt"]
        est_area = obs["area_cen"]
        if gt_area > 0:
            err_a = abs(est_area - gt_area)
            error_area_list.append(err_a)
            
            a_key = round(gt_area / 100) * 100 # Round to nearest 100
            if a_key not in area_rel_err_by_gt:
                area_rel_err_by_gt[a_key] = []
            area_rel_err_by_gt[a_key].append(err_a / gt_area)

print("=== DISTRIBUCIÓN IOU ===")
print(f"Cenital Mean IoU: {np.mean(iou_cen_list):.4f} (std: {np.std(iou_cen_list):.4f})")
if iou_lat_list:
    print(f"Lateral Mean IoU: {np.mean(iou_lat_list):.4f} (std: {np.std(iou_lat_list):.4f})")

print("\n=== DISTRIBUCIÓN ERROR INFERENCIA ===")
print(f"Area Cenital Error Mean: {np.mean(error_area_list):.1f} mm2 (std: {np.std(error_area_list):.1f})")
if error_height_list:
    print(f"Altura Frontal Error Mean: {np.mean(error_height_list):.2f} mm (std: {np.std(error_height_list):.2f})")

print("\n=== VARIABILIDAD ERROR RELATIVO SEGÚN TAMAÑO (AREA) ===")
for k in sorted(area_rel_err_by_gt.keys()):
    if len(area_rel_err_by_gt[k]) > 5:
        print(f" Area ~{k} mm2: Error Relativo {np.mean(area_rel_err_by_gt[k])*100:.1f}% (muestras: {len(area_rel_err_by_gt[k])})")

print("\n=== VARIABILIDAD ERROR RELATIVO SEGÚN ALTURA ===")
for k in sorted(height_rel_err_by_gt.keys()):
    if len(height_rel_err_by_gt[k]) > 5:
        print(f" Altura ~{k} mm: Error Relativo {np.mean(height_rel_err_by_gt[k])*100:.1f}% (muestras: {len(height_rel_err_by_gt[k])})")
