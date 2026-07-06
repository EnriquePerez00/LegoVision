import os
import json
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modes = ["CLASSIC", "HYBRID", "POSE_ONLY"]
results = {}

for mode in modes:
    with open(os.path.join(project_root, "logs", f"inferencia_consolidada_{mode}.json"), "r") as f:
        results[mode] = json.load(f)

all_tids = set(results["CLASSIC"].keys()).intersection(results["HYBRID"].keys()).intersection(results["POSE_ONLY"].keys())
comparison_data = []

for tid in all_tids:
    r_c = results["CLASSIC"][tid]
    r_h = results["HYBRID"][tid]
    r_p = results["POSE_ONLY"][tid]
    comparison_data.append({
        "tracking_id": tid,
        "ref_CLASSIC": r_c["referencia_detectada"],
        "ref_HYBRID": r_h["referencia_detectada"],
        "ref_POSE": r_p["referencia_detectada"],
        "color_CLASSIC": r_c["color"],
        "color_HYBRID": r_h["color"],
        "color_POSE": r_p["color"],
        "area_cen_CLASSIC": r_c["confidence_details"]["average_area_cen"],
        "area_cen_HYBRID": r_h["confidence_details"]["average_area_cen"],
        "area_cen_POSE": r_p["confidence_details"]["average_area_cen"],
        "height_CLASSIC": r_c["confidence_details"]["average_height"],
        "height_HYBRID": r_h["confidence_details"]["average_height"],
        "height_POSE": r_p["confidence_details"]["average_height"]
    })

os.makedirs(os.path.join(project_root, "reports"), exist_ok=True)
df = pd.DataFrame(comparison_data)
csv_path = os.path.join(project_root, "reports", "comparativa_pipelines.csv")
df.to_csv(csv_path, index=False)

md_path = os.path.join(project_root, "reports", "comparativa_resumen.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Reporte Comparativo de Pipelines de Inferencia\n\n")
    match_h = (df["ref_CLASSIC"] == df["ref_HYBRID"]).sum() / len(df) * 100 if len(df) > 0 else 0
    match_p = (df["ref_CLASSIC"] == df["ref_POSE"]).sum() / len(df) * 100 if len(df) > 0 else 0
    f.write(f"- **HYBRID vs CLASSIC**: {match_h:.1f}% de coincidencia.\n")
    f.write(f"- **POSE_ONLY vs CLASSIC**: {match_p:.1f}% de coincidencia.\n")

print(f"Match POSE_ONLY: {match_p:.1f}%")
