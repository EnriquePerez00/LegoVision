# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import json
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
sim_script = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
output_sim_dir = os.path.join(project_root, "data", "simulation_100_all")
metadata_path = os.path.join(output_sim_dir, "simulation_metadata.json")
report_path = os.path.join(output_sim_dir, "inferencia_consolidada.json")

def main():
    has_renders = os.path.exists(metadata_path) and len([f for f in os.listdir(output_sim_dir) if f.endswith('.png')]) >= 100
    if has_renders:
        print("--- 1. Renders y metadatos encontrados. Omitiendo fase de simulación (reutilizando datos existentes) ---")
    else:
        print("--- 1. Generando metadatos para la simulación de 100 piezas aleatorias ---")
        # 1.1 Limpiar directorio de salida
        if os.path.exists(output_sim_dir):
            import shutil
            print(f"Limpiando directorio {output_sim_dir}...")
            shutil.rmtree(output_sim_dir)
        os.makedirs(output_sim_dir, exist_ok=True)
        
        # 1.2 Ejecutar metadata_only
        cmd_meta = [
            blender_path, "-b", "-P", sim_script, "--",
            "--num_pieces", "100",
            "--output_dir", output_sim_dir,
            "--speed", "0.083333",
            "--frames_per_piece", "10",
            "--set-id", "random",
            "--metadata_only"
        ]
        subprocess.run(cmd_meta, check=True)
        
        # 1.3 Ejecutar workers en paralelo (10 workers respetando margen de seguridad en M4 12 cores)
        num_workers = 10
        print(f"Lanzando {num_workers} workers de Blender en paralelo para renderizar...")
        
        os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
        processes = []
        for worker_id in range(num_workers):
            cmd = [
                blender_path, "-b", "-P", sim_script, "--",
                "--num_pieces", "100",
                "--output_dir", output_sim_dir,
                "--speed", "0.083333",
                "--frames_per_piece", "10",
                "--set-id", "random",
                "--worker_id", str(worker_id),
                "--num_workers", str(num_workers)
            ]
            log_file = open(os.path.join(project_root, "logs", f"sim_worker_100_all_{worker_id}.log"), "w")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, log_file))
            
        print("Esperando a que los workers terminen...")
        for p, log_file in processes:
            p.wait()
            log_file.close()
            
        print("--- Renders completados para 100 piezas ---")

    print("\n--- 2. Ejecutando inferencia neuronal con pipeline ALL ---")
    eval_script = os.path.join(project_root, "scripts", "run_evaluation_all.py")
    cmd_eval = [
        sys.executable, eval_script,
        "--metadata", metadata_path,
        "--report", report_path,
        "--color-classifier", "all_colors"
    ]
    subprocess.run(cmd_eval, check=True)

    print("\n--- 3. Generando reporte de exactitud ---")
    evaluate_results()

def evaluate_results():
    if not os.path.exists(report_path):
        print(f"Error: No se encontró el reporte de predicciones en {report_path}")
        return
    if not os.path.exists(metadata_path):
        print(f"Error: No se encontró el archivo de metadatos en {metadata_path}")
        return

    with open(report_path) as f:
        preds = json.load(f)
    with open(metadata_path) as f:
        meta = json.load(f)

    # Estructura del reporte consolidado: {"results": [...], "accuracy": ...}
    results = preds.get("results", [])

    # Construir mapa de (ref, color_code) -> color_name a partir de la simulación
    gt_color_map = {}
    for frame in meta.get("frames", []):
        for piece in frame.get("visible_pieces", []):
            ref = piece.get("ref")
            c_code = str(piece.get("color_code", ""))
            c_name = piece.get("color_name")
            if ref and c_code:
                gt_color_map[(ref, c_code)] = c_name

    error_superficie = []
    error_altura = []
    color_cen_correct = 0
    color_lat_correct = 0
    color_fused_correct = 0
    ref_correct = 0

    N = len(results)
    if N == 0:
        print("No predictions found in results list!")
        return

    for r in results:
        ref_gt = r.get("ref_gt")
        c_code_gt = str(r.get("color_code_gt", ""))
        ref_pred = r.get("ref_inferred")
        
        c_name_cen = r.get("color_name_cen")
        c_name_lat = r.get("color_name_lat")
        c_name_fused = r.get("color_name_fused")
        
        c_name_gt = gt_color_map.get((ref_gt, c_code_gt), "Unknown")
        
        h_pred = r.get("measured_height_mm", 9.6)
        h_gt = None
        for frame in meta.get("frames", []):
            for piece in frame.get("visible_pieces", []):
                if piece.get("ref") == ref_gt:
                    h_gt = piece.get("lateral_height_gt")
                    break
            if h_gt is not None:
                break
        
        sup_pred = r.get("apparent_area_mm2", 0)
        sup_gt = None
        for frame in meta.get("frames", []):
            for piece in frame.get("visible_pieces", []):
                if piece.get("ref") == ref_gt:
                    sup_gt = piece.get("zenith_silhouette_area_gt")
                    break
            if sup_gt is not None:
                break

        is_ref_correct = (ref_pred == ref_gt)
        if is_ref_correct:
            ref_correct += 1
            
        if c_name_cen == c_name_gt:
            color_cen_correct += 1
        if c_name_lat == c_name_gt:
            color_lat_correct += 1
        if c_name_fused == c_name_gt:
            color_fused_correct += 1

        if h_gt is not None:
            error_altura.append(abs(h_pred - h_gt))
        if sup_gt is not None and sup_gt > 0:
            error_superficie.append(abs(sup_pred - sup_gt) / sup_gt * 100)
            
    print(f"--- REPORTE DE EXACTITUD (simulation_100_all: {N} piezas) ---")
    print(f"Exactitud de Referencia: {ref_correct/N*100:.2f}%")
    print(f"Exactitud Color Cenital: {color_cen_correct/N*100:.2f}%")
    print(f"Exactitud Color Lateral: {color_lat_correct/N*100:.2f}%")
    print(f"Exactitud Color Fusionado: {color_fused_correct/N*100:.2f}%")
    
    if error_altura:
        print(f"Error Medio Altura Frontal:             {np.mean(error_altura):.2f} mm")
    if error_superficie:
        print(f"Error Relativo Sup. Cenital:            {np.mean(error_superficie):.2f} %")
    else:
        print("Error Relativo Sup. Cenital:            N/A")

    print("\n--- Análisis de Fallos de Color (Combinado/Fusionado) ---")
    fallos = []
    for idx, r in enumerate(results):
        ref_gt = r.get("ref_gt")
        c_code_gt = str(r.get("color_code_gt", ""))
        ref_pred = r.get("ref_inferred")
        c_name_fused = r.get("color_name_fused")
        c_name_gt = gt_color_map.get((ref_gt, c_code_gt), "Unknown")
        
        if c_name_fused != c_name_gt:
            fallos.append(f"Muestra {idx}: Predijo Fused '{c_name_fused}', Real: '{c_name_gt}' (Ref GT: {ref_gt}, Pred: {ref_pred})")
    
    if fallos:
        print(f"Total fallos de color fusionado: {len(fallos)}")
        for f in fallos[:15]:
            print(f"  {f}")
        if len(fallos) > 15:
            print("  ...")
    else:
        print("¡Todos los colores correctos!")

if __name__ == "__main__":
    main()
