#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_yolo_dataset_parallel.py

Lanza múltiples instancias de Blender en paralelo para generar el dataset YOLO
distribuyendo la carga (sharding) entre todos los cores de CPU disponibles (ej. M4 Pro).

Uso:
    python3 camara_domo/scripts/run_yolo_dataset_parallel.py --num_pieces 10000 --num_workers 10
"""
import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=10) # 10 workers en un M4 de 12 cores deja 2 libres
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    blender_bin = "/opt/homebrew/bin/blender"
    script_path = os.path.join(project_root, "projects", "camara_domo", "scripts", "generate_yolo_dataset.py")

    processes = []
    
    print(f"Lanzando {args.num_workers} workers paralelos para procesar {args.num_pieces} piezas...")
    
    for worker_id in range(args.num_workers):
        cmd = [
            blender_bin, "-b", "-P", script_path, "--",
            "--num_pieces", str(args.num_pieces),
            "--num_workers", str(args.num_workers),
            "--worker_id", str(worker_id),
            "--split", args.split
        ]
        
        log_file = open(os.path.join(project_root, "logs", f"yolo_worker_{worker_id}.log"), "w")
        
        print(f"  Iniciando Worker {worker_id}...")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file, worker_id))
        
    print("Esperando a que todos los workers terminen...")
    
    failed = False
    for p, log_file, worker_id in processes:
        p.wait()
        log_file.close()
        if p.returncode != 0:
            print(f"[ERROR] Worker {worker_id} falló con código {p.returncode}. Revisa logs/yolo_worker_{worker_id}.log")
            failed = True
        else:
            print(f"[OK] Worker {worker_id} completado.")
            
    if failed:
        print("La generación paralela finalizó con errores.")
        sys.exit(1)
    else:
        print("Generación paralela completada con éxito.")

if __name__ == "__main__":
    main()
