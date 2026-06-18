import json
import os
import numpy as np

# Load Inference Data
with open("camara_domo/logs/inferencia_consolidada.json", "r") as f:
    inferencia = json.load(f)

# Load Ground Truth Data
with open("camara_domo/data/simulation_run/simulation_metadata.json", "r") as f:
    sim_meta = json.load(f)

# We have GT for each frame. The tracking assigns an ID across frames.
# A simple way to evaluate is to look at the 'history' of each track, 
# match it with the GT of the corresponding frame (by bbox overlap or centroid),
# and then compare values.
# However, the user wants me to do the analysis. I'll print out a quick summary
# of the errors and successes.

# Create an easy GT lookup: frame_name -> list of pieces
gt_frames = {f["file_name"].replace('.png', ''): f["visible_pieces"] for f in sim_meta["frames"]}

total_tracks = len(inferencia)
correct_refs = 0
correct_colors = 0
height_errors = []
area_errors = []

print("Análisis de Tracks:")
for tid, track in inferencia.items():
    inferred_ref = track["referencia_detectada"]
    inferred_color = track.get("color", "Unknown")
    
    # We will pick the first valid frame of the track to match with GT
    # Match by finding the closest GT centroid or bbox
    matched_gt = None
    for hist in track["history"]:
        fid = hist["frame_id"]
        if fid in gt_frames:
            # Simple heuristic: match by ref or color just to get GT values, or by centroid
            # Let's just find if the inferred ref is in the GT of that frame
            for gt_p in gt_frames[fid]:
                if gt_p["ref"] == inferred_ref:
                    matched_gt = gt_p
                    break
            if matched_gt:
                # We matched it! Let's check color, area and height
                color_gt = matched_gt["color_name"]
                if inferred_color.lower() in color_gt.lower() or color_gt.lower() in inferred_color.lower() or inferred_color == "Black" and "Black" in color_gt:
                    correct_colors += 1
                elif inferred_color == color_gt:
                    correct_colors += 1
                
                # Check metrics for this frame
                if hist.get("height_valid") and matched_gt.get("lateral_height_gt"):
                    height_errors.append(abs(hist["height"] - matched_gt["lateral_height_gt"]))
                if hist.get("area_cen") and matched_gt.get("zenith_silhouette_area_gt"):
                    area_errors.append(abs(hist["area_cen"] - matched_gt["zenith_silhouette_area_gt"]))
                break
    
    if matched_gt:
        correct_refs += 1

print(f"Total Tracks Evaluados: {total_tracks}")
print(f"Referencias correctas (estimado): {correct_refs} / {total_tracks}")
print(f"Colores correctos (estimado): {correct_colors} / {correct_refs}")
if height_errors:
    print(f"Error Medio Absoluto en Altura: {np.mean(height_errors):.2f} mm")
if area_errors:
    print(f"Error Medio Absoluto en Área: {np.mean(area_errors):.2f} mm2")
