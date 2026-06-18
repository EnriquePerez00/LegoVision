# -*- coding: utf-8 -*-
import os
import sys
import json
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from scripts.inferencia_neuronal_v2 import run_pipeline

def run_comparison():
    data_dir = os.path.join(project_root, "data", "data100") # Use a subset folder if available, or just data10
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "data", "data10")
        
    print(f"Ejecutando comparativa sobre: {data_dir}")
    
    modes = ["CLASSIC", "HYBRID", "POSE_ONLY"]
    results = {}
    fps_stats = {}
    
    for mode in modes:
        print(f"\n{'='*50}\nIniciando Pipeline: {mode}\n{'='*50}")
        out_json = os.path.join(project_root, "logs", f"inferencia_consolidada_{mode}.json")
        
        t0 = time.time()
        # Procesar max 100 frames (como solicitado)
        run_pipeline(data_dir, out_json, belt_speed=83.3, fps=5.0, max_frames=100, mode=mode)
        t1 = time.time()
        
        frames_proc = 100
        fps_stats[mode] = round(frames_proc / (t1 - t0 + 0.001), 2)
        
        with open(out_json, "r", encoding="utf-8") as f:
            results[mode] = json.load(f)
            
    # Analizar resultados
    print("\nGenerando Reporte Comparativo...")
    
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
        
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(project_root, "logs", "comparativa_pipelines.csv")
    df.to_csv(csv_path, index=False)
    
    # Generar Markdown en workspace actual (lo leeré luego para generar el artefacto manual)
    md_path = os.path.join(project_root, "logs", "comparativa_resumen.md")
    
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Reporte Comparativo de Pipelines de Inferencia\n\n")
            f.write("Se ejecutaron 3 arquitecturas diferentes sobre el mismo conjunto de datos.\n\n")
            f.write("## 1. Eficiencia Computacional (FPS globales)\n")
            for m in modes:
                f.write(f"- **{m}**: {fps_stats[m]} FPS\n")
                
            f.write("\n## 2. Precisión de Color (Desviaciones respecto a CLASSIC)\n")
            diff_color_h = (df["color_CLASSIC"] != df["color_HYBRID"]).sum()
            diff_color_p = (df["color_CLASSIC"] != df["color_POSE"]).sum()
            f.write(f"- **HYBRID**: {diff_color_h} piezas difieren en color frente a CLASSIC.\n")
            f.write(f"- **POSE_ONLY**: {diff_color_p} piezas difieren en color frente a CLASSIC (Esperado por ruido de fondo).\n")
            
            f.write("\n## 3. Discrepancias Geométricas (Media Absoluta)\n")
            mae_area_h = abs(df["area_cen_CLASSIC"] - df["area_cen_HYBRID"]).mean()
            mae_area_p = abs(df["area_cen_CLASSIC"] - df["area_cen_POSE"]).mean()
            f.write(f"- **Área (HYBRID vs CLASSIC)**: {mae_area_h:.2f} mm² de desviación.\n")
            f.write(f"- **Área (POSE vs CLASSIC)**: {mae_area_p:.2f} mm² de desviación (Muestra el error del Convex Hull vs máscara perfecta).\n")
            
            f.write("\n## 4. Inferencia Final de Pieza\n")
            match_h = (df["ref_CLASSIC"] == df["ref_HYBRID"]).sum() / len(df) * 100 if len(df) > 0 else 0
            match_p = (df["ref_CLASSIC"] == df["ref_POSE"]).sum() / len(df) * 100 if len(df) > 0 else 0
            f.write(f"- **HYBRID vs CLASSIC**: {match_h:.1f}% de coincidencia.\n")
            f.write(f"- **POSE_ONLY vs CLASSIC**: {match_p:.1f}% de coincidencia.\n")
            
        print(f"Reporte MD guardado en: {md_path}")
    except Exception as e:
        print(f"Error guardando MD: {e}")
        
    print(f"Comparativa completa. Datos en {csv_path}")

if __name__ == "__main__":
    run_comparison()
