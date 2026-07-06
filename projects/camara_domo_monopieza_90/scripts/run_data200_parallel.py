#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_data200_parallel.py

Orchestrates the generation of data200 in parallel to maximize HW resources.
Purges previous data, runs a metadata-only pass to establish conveyor layout,
and splits rendering across parallel Blender workers.

Uso:
    python3 projects/camara_domo/scripts/run_data200_parallel.py --num_pieces 20 --frames_per_piece 6 --num_workers 10
"""
import os
import sys
import shutil
import argparse
import subprocess
import time

def clean_dir(target_dir):
    """Purges the output directory safely."""
    if os.path.exists(target_dir):
        print(f"[Clean] Eliminando contenido anterior en: {target_dir}")
        for item in os.listdir(target_dir):
            path = os.path.join(target_dir, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
            except Exception as e:
                print(f"[Warning] No se pudo borrar {path}: {e}")
    else:
        os.makedirs(target_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=20)
    parser.add_argument("--frames_per_piece", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=10) # 10 workers for 12 cores, leaving 2 free
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    # In test mode, reduce workload
    if args.test_run:
        args.num_pieces = 4
        args.num_workers = 2
        print("[Mode] RUNNING IN TEST MODE (num_pieces=4, num_workers=2)")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    blender_bin = "/opt/homebrew/bin/blender"
    script_path = os.path.join(project_root, "projects", "camara_domo", "scripts", "generate_conveyor_simulation.py")
    output_dir = os.path.join(project_root, "projects", "camara_domo", "data", "data200")
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Limpiar directorio
    clean_dir(output_dir)

    # 2. Ejecutar pase rápido de metadatos (metadata_only) para calcular la cinta y offsets
    print("[Orchestrator] Generando metadatos de la simulación...")
    cmd_meta = [
        blender_bin, "-b", "-P", script_path, "--",
        "--num_pieces", str(args.num_pieces),
        "--output_dir", output_dir,
        "--seed", str(args.seed),
        "--frames_per_piece", str(args.frames_per_piece),
        "--metadata_only"
    ]
    
    meta_log = os.path.join(logs_dir, "data200_meta.log")
    with open(meta_log, "w") as f:
        res = subprocess.run(cmd_meta, stdout=f, stderr=subprocess.STDOUT)
    
    if res.returncode != 0:
        print(f"[ERROR] Falló la generación de metadatos. Revisa {meta_log}")
        sys.exit(1)
    
    print("[Orchestrator] Metadatos generados exitosamente.")

    # 3. Lanzar los workers en paralelo para renderizar
    processes = []
    print(f"[Orchestrator] Lanzando {args.num_workers} workers para renderizado paralelo...")
    t0 = time.time()
    
    for worker_id in range(args.num_workers):
        cmd = [
            blender_bin, "-b", "-P", script_path, "--",
            "--num_pieces", str(args.num_pieces),
            "--output_dir", output_dir,
            "--seed", str(args.seed),
            "--frames_per_piece", str(args.frames_per_piece),
            "--worker_id", str(worker_id),
            "--num_workers", str(args.num_workers)
        ]
        
        log_file = open(os.path.join(logs_dir, f"data200_worker_{worker_id}.log"), "w")
        print(f"  -> Iniciando Worker {worker_id}...")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file, worker_id))

    # 4. Esperar a todos los workers
    print("[Orchestrator] Esperando a que finalicen todos los renderizados...")
    failed = False
    for p, log_file, worker_id in processes:
        p.wait()
        log_file.close()
        if p.returncode != 0:
            print(f"[ERROR] Worker {worker_id} falló (código {p.returncode}). Ver logs/data200_worker_{worker_id}.log")
            failed = True
        else:
            print(f"[OK] Worker {worker_id} completó renderizado.")

    t1 = time.time()
    if failed:
        print("[Orchestrator] La generación paralela finalizó con errores.")
        sys.exit(1)
    else:
        print(f"[Orchestrator] ¡Generación completada con éxito en {t1-t0:.1f} segundos!")

if __name__ == "__main__":
    main()
