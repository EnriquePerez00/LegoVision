#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

BLENDER_BIN = "/opt/homebrew/bin/blender"
SCRIPT_PATH = os.path.join(project_root, "scripts", "generate_1d_continuous.py")
OUTPUT_DIR = os.path.join(project_root, "data", "simulation_1000")

def get_max_workers():
    import psutil
    cores = os.cpu_count() or 4
    # Reserve 2 cores for system responsiveness
    return max(1, cores - 2)

def run_worker(worker_id, num_workers, max_frames=None):
    cmd = [
        BLENDER_BIN, "-b", "-P", SCRIPT_PATH, "--",
        "--num_pieces", "1000",
        "--output_dir", OUTPUT_DIR,
        "--speed", "5.0",
        "--step_mm", "40.0",
        "--worker_id", str(worker_id),
        "--num_workers", str(num_workers)
    ]
    if max_frames:
        cmd.extend(["--max_frames", str(max_frames)])
        
    print(f"[Worker {worker_id}] cmd: {' '.join(cmd)}")
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return worker_id, result.returncode, result.stdout, result.stderr

def main():
    args = sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=None, help="Limita a un número de frames (ej. 15 para pruebas)")
    pa = parser.parse_args(args)

    num_workers = get_max_workers()
    print(f"Iniciando simulación 1D continua con {num_workers} workers paralelos.")
    
    # 1. Metadata Only run
    print("Pre-calculando empaquetado y metadatos...")
    cmd_meta = [
        BLENDER_BIN, "-b", "-P", SCRIPT_PATH, "--",
        "--num_pieces", "1000",
        "--output_dir", OUTPUT_DIR,
        "--metadata_only"
    ]
    if pa.max_frames:
        cmd_meta.extend(["--max_frames", str(pa.max_frames)])
        
    subprocess.run(cmd_meta, check=True)
    print("Metadatos generados exitosamente.")

    # 2. Parallel rendering
    print(f"Ejecutando renderizado en {num_workers} chunks...")
    start_t = time.time()
    
    # Si max_frames es pequeño, limitamos el num_workers a max_frames
    if pa.max_frames and pa.max_frames < num_workers:
        num_workers = pa.max_frames
        
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_worker, w_id, num_workers, pa.max_frames): w_id 
            for w_id in range(num_workers)
        }
        
        for future in as_completed(futures):
            w_id = futures[future]
            try:
                worker_id, rc, out, err = future.result()
                if rc == 0:
                    print(f"[Worker {worker_id}] Completado exitosamente.")
                else:
                    print(f"[Worker {worker_id}] FAILED with rc={rc}")
                    print(f"  STDERR: {err}")
            except Exception as e:
                print(f"[Worker {w_id}] Generó una excepción: {e}")

    elapsed = time.time() - start_t
    print(f"Simulación finalizada en {elapsed:.2f} s")

if __name__ == "__main__":
    main()
