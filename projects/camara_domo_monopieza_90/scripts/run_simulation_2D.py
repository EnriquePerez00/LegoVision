# -*- coding: utf-8 -*-
"""run_simulation_2D.py
================================
Genera renders de piezas distribuidas en 2D en la cinta transportadora.
Soporta parametrización de resolución, número de piezas, set-id, velocidad, etc.
Lanza workers en paralelo utilizando Blender y al finalizar ejecuta run_evaluation_2D.py.
"""
import os
import sys
import subprocess
import argparse
import shutil
import json
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
sim_script = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")

def main():
    parser = argparse.ArgumentParser(description="Generador de simulaciones 2D para conveyor.")
    parser.add_argument("--num_pieces", type=int, default=50, help="Número de piezas a simular.")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolución de renderizado (ancho/alto en píxeles).")
    parser.add_argument("--set-id", type=str, default="75078-1", help="ID del set LEGO (ej: '75078-1' o 'random').")
    parser.add_argument("--output_dir", type=str, default=None, help="Directorio de salida de los renders.")
    parser.add_argument("--speed", type=float, default=0.083333, help="Velocidad de la cinta en m/s (default: 0.083333 m/s = 5 m/min).")
    parser.add_argument("--frames_per_piece", type=int, default=10, help="Frames por pieza para regular disparos.")
    parser.add_argument("--num_workers", type=int, default=10, help="Número de workers en paralelo para Blender.")
    args = parser.parse_args()

    # Directorio de salida por defecto
    if args.output_dir is None:
        output_sim_dir = os.path.join(project_root, "data", "simulation_10_2D")
    else:
        output_sim_dir = args.output_dir

    metadata_path = os.path.join(output_sim_dir, "simulation_metadata.json")
    report_path = os.path.join(output_sim_dir, "inferencia_consolidada.json")

    print(f"--- 1. Generando metadatos para simulación 2D de {args.num_pieces} piezas del set {args.set_id} ---")
    
    # 1.1 Limpiar directorio de salida
    if os.path.exists(output_sim_dir):
        print(f"Limpiando directorio {output_sim_dir}...")
        shutil.rmtree(output_sim_dir)
    os.makedirs(output_sim_dir, exist_ok=True)

    # 1.2 Ejecutar metadata_only con Blender
    cmd_meta = [
        blender_path, "-b", "-P", sim_script, "--",
        "--num_pieces", str(args.num_pieces),
        "--output_dir", output_sim_dir,
        "--speed", str(args.speed),
        "--frames_per_piece", str(args.frames_per_piece),
        "--set-id", args.set_id,
        "--dimension", "2D",
        "--resolution", str(args.resolution),
        "--metadata_only"
    ]
    subprocess.run(cmd_meta, check=True)

    # 1.3 Ejecutar workers en paralelo
    num_workers = args.num_workers
    print(f"Lanzando {num_workers} workers de Blender en paralelo para renderizar a {args.resolution}x{args.resolution}...")

    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    processes = []
    for worker_id in range(num_workers):
        cmd = [
            blender_path, "-b", "-P", sim_script, "--",
            "--num_pieces", str(args.num_pieces),
            "--output_dir", output_sim_dir,
            "--speed", str(args.speed),
            "--frames_per_piece", str(args.frames_per_piece),
            "--set-id", args.set_id,
            "--dimension", "2D",
            "--resolution", str(args.resolution),
            "--worker_id", str(worker_id),
            "--num_workers", str(num_workers)
        ]
        log_file_path = os.path.join(project_root, "logs", f"sim_worker_2D_{worker_id}.log")
        log_file = open(log_file_path, "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))

    print("Esperando a que los workers terminen...")
    for p, log_file in processes:
        p.wait()
        log_file.close()

    print(f"--- Renders completados en {output_sim_dir} ---")

    # 2. Inferencia y Evaluación
    print("\n--- 2. Ejecutando inferencia neuronal con run_evaluation_2D ---")
    eval_script = os.path.join(project_root, "scripts", "run_evaluation_2D.py")
    if not os.path.exists(eval_script):
        print(f"Advertencia: El script de evaluación {eval_script} no se encuentra todavía. Omita la evaluación.")
        return

    cmd_eval = [
        sys.executable, eval_script,
        "--metadata", metadata_path,
        "--report", report_path,
        "--color-classifier", "auto"
    ]
    subprocess.run(cmd_eval, check=True)

if __name__ == "__main__":
    main()
