# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(project_root)))

# Import the feature map from the database
from core.db import supabase_client

def main():
    eval_path = os.path.join(project_root, "data", "reports", "sim300_eval.json")
    if not os.path.exists(eval_path):
        print(f"Error: No existe el archivo {eval_path}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)

    # 1. Recuperar catálogo de características de Supabase
    db_features = {}
    try:
        with supabase_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ldraw_id, name, category, topological_features FROM lego_classes;")
                rows = cur.fetchall()
                for row in rows:
                    ldraw_id = row["ldraw_id"]
                    feat = row["topological_features"]
                    if isinstance(feat, str):
                        try:
                            feat = json.loads(feat)
                        except Exception:
                            feat = {}
                    if not isinstance(feat, dict):
                        feat = {}
                    db_features[ldraw_id] = {
                        "name": row["name"],
                        "category": row["category"],
                        "features": feat
                    }
    except Exception as e:
        print(f"Advertencia cargando características de Supabase: {e}")

    # Lista de características topológicas evaluadas
    FEATURE_KEYS = [
        "stud_solid", "stud_hollow", "technic_hole_round", "technic_hole_cross",
        "clip_jaw", "bar_handle", "bottom_tube", "bottom_pin"
    ]

    # Estadísticas
    correct_ref = 0
    correct_color_cen = 0
    correct_color_lat = 0
    
    height_errors = []
    area_errors = []
    
    # Acumuladores de acierto de características morfológicas
    feat_correct = {k: 0 for k in FEATURE_KEYS}
    feat_total_present_gt = {k: 0 for k in FEATURE_KEYS}
    feat_total_present_pred = {k: 0 for k in FEATURE_KEYS}

    # Desglose por familias de piezas
    family_stats = {}

    def get_family(ref):
        ref_str = str(ref).lower()
        if ref_str in ["3001", "3004", "3003", "3005", "3010", "3008", "2465"]:
            return "Bricks"
        elif ref_str in ["3020", "3021", "3022", "3023", "3024", "2445", "3710", "3832", "51739"]:
            return "Plates"
        elif ref_str in ["3040", "3298", "3679", "3680", "85984"]:
            return "Slopes & Wedges"
        elif "technic" in ref_str or ref_str in ["32054", "32000", "6541"]:
            return "Technic"
        else:
            return "Others/Special"

    for r in results:
        ref_gt = r["ref_gt"]
        ref_pred = r["ref_inferred"]
        
        # Color
        color_cen = r["color_name_cen"]
        color_lat = r["color_name_lat"]
        # En run_evaluation.py, el color real se puede inferir buscando en db_features o cargándolo.
        # Busquemos el color real de los frames mapeando con la simulación.
        # Para simplificar, si model_match es True, sabemos el color real.
        # Pero podemos obtener el color_real cruzando con la referencia_real.
        # Vamos a ver si el JSON tiene un color_name_gt o color_gt.
        # Ah, del JSON de sim300_eval.json: "color_code_gt": "86" etc.
        # Mapeemos el color_code_gt al nombre de color real.
        from run_evaluation import CATALOG_COLORS
        color_name_gt = "Unknown"
        for c in CATALOG_COLORS:
            if c["color_code"] == r["color_code_gt"]:
                color_name_gt = c["color_name"]
                break
        
        is_ref_correct = r["model_match"]
        if is_ref_correct:
            correct_ref += 1

        is_color_cen_correct = (color_cen.strip().lower() == color_name_gt.strip().lower())
        if is_color_cen_correct:
            correct_color_cen += 1

        is_color_lat_correct = (color_lat.strip().lower() == color_name_gt.strip().lower())
        if is_color_lat_correct:
            correct_color_lat += 1

        # Áreas y alturas
        if r["surface_db_silhouette_mm2"] is not None and r["surface_db_silhouette_mm2"] > 0:
            area_errors.append(abs(r["surface_obs_apparent_mm2"] - r["surface_db_silhouette_mm2"]) / r["surface_db_silhouette_mm2"] * 100)
        
        if r["lateral_height_db_mm"] is not None and r["lateral_height_db_mm"] > 0:
            height_errors.append(abs(r["lateral_height_meas_mm"] - r["lateral_height_db_mm"]))

        # Familias
        fam = get_family(ref_gt)
        if fam not in family_stats:
            family_stats[fam] = {"total": 0, "correct": 0}
        family_stats[fam]["total"] += 1
        if is_ref_correct:
            family_stats[fam]["correct"] += 1

        # Características Morfológicas
        gt_feat = db_features.get(ref_gt, {}).get("features", {})
        pred_feat = db_features.get(ref_pred, {}).get("features", {})
        for fk in FEATURE_KEYS:
            val_gt = 1 if gt_feat.get(fk, 0) > 0 else 0
            val_pred = 1 if pred_feat.get(fk, 0) > 0 else 0
            if val_gt == val_pred:
                feat_correct[fk] += 1
            if val_gt == 1:
                feat_total_present_gt[fk] += 1
            if val_pred == 1:
                feat_total_present_pred[fk] += 1

    # Imprimir Reporte
    print("\n" + "="*60)
    print("ANÁLISIS DE ACCURACY DETALLADO - SIMULATION_300")
    print("="*60)
    print(f"Total muestras evaluadas: {total}")
    print(f"Precisión Inferencia Final (Referencia): {correct_ref}/{total} ({correct_ref/total*100:.2f}%)")
    print(f"Precisión Color Cenital:                 {correct_color_cen}/{total} ({correct_color_cen/total*100:.2f}%)")
    print(f"Precisión Color Lateral/Frontal:         {correct_color_lat}/{total} ({correct_color_lat/total*100:.2f}%)")

    print("\n--- 1. Distribución de Error de Superficie Cenital ---")
    if area_errors:
        print(f"  Error Medio Absoluto (MAE %):   {np.mean(area_errors):.2f}%")
        print(f"  Error Mediano Absoluto (MedAE %): {np.median(area_errors):.2f}%")
        print(f"  Error Máximo %:                 {np.max(area_errors):.2f}%")
        print(f"  Desviación Estándar %:          {np.std(area_errors):.2f}%")
    else:
        print("  N/A")

    print("\n--- 2. Distribución de Error de Altura Frontal ---")
    if height_errors:
        print(f"  Error Medio Absoluto (MAE):    {np.mean(height_errors):.3f} mm")
        print(f"  Error Mediano Absoluto:        {np.median(height_errors):.3f} mm")
        print(f"  Error Máximo:                  {np.max(height_errors):.3f} mm")
        print(f"  Desviación Estándar:           {np.std(height_errors):.3f} mm")
        
        # Clasificar la altura en rangos de error relativo
        height_errors_np = np.array(height_errors)
        err_rel = (height_errors_np / 6.8) * 100.0
        print(f"  Distribución de Error Relativo (sobre cota media ~6.8mm):")
        print(f"    [0% - 5%):   {np.sum(err_rel < 5.0)} muestras ({np.sum(err_rel < 5.0)/total*100:.1f}%)")
        print(f"    [5% - 10%):  {np.sum((err_rel >= 5.0) & (err_rel < 10.0))} muestras ({np.sum((err_rel >= 5.0) & (err_rel < 10.0))/total*100:.1f}%)")
        print(f"    [10% - 20%): {np.sum((err_rel >= 10.0) & (err_rel < 20.0))} muestras ({np.sum((err_rel >= 10.0) & (err_rel < 20.0))/total*100:.1f}%)")
        print(f"    [20%+]:      {np.sum(err_rel >= 20.0)} muestras ({np.sum(err_rel >= 20.0)/total*100:.1f}%)")
    else:
        print("  N/A")

    print("\n--- 3. Exactitud en Detección de Características Morfológicas (Topología) ---")
    for fk in FEATURE_KEYS:
        acc = feat_correct[fk] / total * 100.0
        print(f"  * {fk:20s} -> Exactitud: {feat_correct[fk]:3d}/{total:3d} ({acc:.2f}%) | Presentes en GT: {feat_total_present_gt[fk]:3d} | Predichos: {feat_total_present_pred[fk]:3d}")

    print("\n--- 4. Desglose de Accuracy por Familias de Piezas ---")
    for fam, stats in sorted(family_stats.items()):
        acc = stats["correct"] / stats["total"] * 100.0
        print(f"  * {fam:18s} -> {stats['correct']:3d}/{stats['total']:3d} ({acc:.2f}%)")

if __name__ == "__main__":
    main()
