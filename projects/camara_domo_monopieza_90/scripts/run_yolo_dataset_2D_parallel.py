#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_yolo_dataset_2D_parallel.py
===================================
Lanza múltiples instancias de Blender en paralelo para generar el dataset YOLO 2D (multi-pieza)
distribuyendo la carga (sharding) entre todos los cores de CPU disponibles (ej. M4 Pro).
"""
import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces_train", type=int, default=1200, help="Número de piezas para el split train.")
    parser.add_argument("--num_pieces_val", type=int, default=300, help="Número de piezas para el split val.")
    parser.add_argument("--num_workers", type=int, default=10) # 10 workers en un M4 de 12 cores deja 2 libres
    args = parser.parse_args()

    project_root = "/Users/I764690/Code_personal/LegoVision"
    blender_bin = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
    script_path = os.path.join(project_root, "projects", "camara_domo_monopieza_90", "scripts", "generate_yolo_dataset.py")
    output_dir = os.path.join(project_root, "projects", "camara_domo_monopieza_90", "data", "yolo_dataset_2D")

    # Crear directorios
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Procesar train y val secuencialmente para simplificar paralelización
    for split, num_pieces in [("train", args.num_pieces_train), ("val", args.num_pieces_val)]:
        processes = []
        print(f"\n--- Generando {num_pieces} piezas para split '{split}' con {args.num_workers} workers ---")
        
        for worker_id in range(args.num_workers):
            cmd = [
                blender_bin, "-b", "-P", script_path, "--",
                "--num_pieces", str(num_pieces),
                "--output_dir", output_dir,
                "--split", split,
                "--num_workers", str(args.num_workers),
                "--worker_id", str(worker_id)
            ]
            
            log_file = open(os.path.join(project_root, "logs", f"yolo_2D_worker_{split}_{worker_id}.log"), "w")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, log_file, worker_id))
            
        print("Esperando a que terminen los workers...")
        failed = False
        for p, log_file, worker_id in processes:
            p.wait()
            log_file.close()
            if p.returncode != 0:
                print(f"[ERROR] Worker {worker_id} ({split}) falló con código {p.returncode}.")
                failed = True
            
        if failed:
            print(f"Error generando split {split}.")
            sys.exit(1)
        else:
            print(f"[OK] Split {split} generado correctamente.")

    print("\nGeneración del dataset YOLO 2D completada con éxito.")

if __name__ == "__main__":
    main()
