# -*- coding: utf-8 -*-
"""run_simulation.py
================================
Genera piezas de un set (o todos), renderizadas a 2048x2048 y empaquetadas
en 1D (fila india) o 2D (ocupando el ancho de la cinta).

Uso:
    python run_simulation.py --set 75078-1 --num_pieces 100 --dimension 1D
"""
import os
import sys
import subprocess
import json
import numpy as np
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description="Pipeline Unificado de Simulación")
    parser.add_argument("--set", type=str, default="75078-1", help="ID del set (ej. 75078-1) o 'all' para aleatorio")
    parser.add_argument("--num_pieces", type=int, default=100, help="Cantidad de piezas a simular")
    parser.add_argument("--dimension", type=str, choices=["1D", "2D"], default="1D", help="Modo de empaquetado")
    args = parser.parse_args()

    project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
    blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
    sim_script = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
    
    # Nombre del directorio de salida dinámico (ej: simulation_100_75078-1_1D)
    out_name = f"simulation_{args.num_pieces}_{args.set}_{args.dimension}"
    output_sim_dir = os.path.join(project_root, "data", out_name)
    metadata_path = os.path.join(output_sim_dir, "simulation_metadata.json")
    report_path = os.path.join(output_sim_dir, "inferencia_consolidada.json")

    has_renders = os.path.exists(metadata_path) and len(
        [f for f in os.listdir(output_sim_dir) if f.endswith('.png')]
    ) >= args.num_pieces if os.path.isdir(output_sim_dir) else False

    set_id_for_blender = "random" if args.set.lower() == "all" else args.set

    if has_renders:
        print(f"--- 1. Renders y metadatos encontrados en {output_sim_dir}. Omitiendo fase de simulación. ---")
    else:
        print(f"--- 1. Generando metadatos para la simulación de {args.num_pieces} piezas del set {args.set} en {args.dimension} ---")
        if os.path.exists(output_sim_dir):
            print(f"Limpiando directorio {output_sim_dir}...")
            shutil.rmtree(output_sim_dir)
        os.makedirs(output_sim_dir, exist_ok=True)

        # 1.2 Ejecutar metadata_only
        cmd_meta = [
            blender_path, "-b", "-P", sim_script, "--",
            "--num_pieces", str(args.num_pieces),
            "--output_dir", output_sim_dir,
            "--speed", "0.083333",
            "--frames_per_piece", "10",
            "--set-id", set_id_for_blender,
            "--dimension", args.dimension,
            "--metadata_only"
        ]
        subprocess.run(cmd_meta, check=True)

        # 1.3 Ejecutar workers en paralelo (10 workers respetando margen de seguridad en M4 12 cores)
        num_workers = max(1, os.cpu_count() - 2)
        print(f"Lanzando {num_workers} workers de Blender en paralelo para renderizar...")

        os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
        processes = []
        for worker_id in range(num_workers):
            cmd = [
                blender_path, "-b", "-P", sim_script, "--",
                "--num_pieces", str(args.num_pieces),
                "--output_dir", output_sim_dir,
                "--speed", "0.083333",
                "--frames_per_piece", "10",
                "--set-id", set_id_for_blender,
                "--dimension", args.dimension,
                "--worker_id", str(worker_id),
                "--num_workers", str(num_workers)
            ]
            log_file = open(
                os.path.join(project_root, "logs", f"sim_worker_{args.num_pieces}_{args.set}_{args.dimension}_{worker_id}.log"),
                "w"
            )
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, log_file))

        print("Esperando a que los workers terminen...")
        for p, log_file in processes:
            p.wait()
            log_file.close()

        print(f"--- Renders completados para {args.num_pieces} piezas del set {args.set} ---")

    print("\n--- 2. Ejecutando inferencia neuronal ---")
    if args.set.lower() == "all":
        eval_script = os.path.join(project_root, "scripts", "run_evaluation_all.py")
        cmd_eval = [
            sys.executable, eval_script,
            "--metadata", metadata_path,
            "--report", report_path
        ]
    else:
        eval_script = os.path.join(project_root, "scripts", "run_evaluation_75078.py")
        cmd_eval = [
            sys.executable, eval_script,
            "--metadata", metadata_path,
            "--report", report_path,
            "--color-classifier", args.set
        ]
    
    subprocess.run(cmd_eval, check=True)

    print("\n--- Pipeline Finalizado ---")

if __name__ == "__main__":
    main()
