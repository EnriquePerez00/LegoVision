# -*- coding: utf-8 -*-
import os
import sys
import shutil
import subprocess
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
if not os.path.exists(blender_path):
    blender_path = "blender"

script_path = os.path.join(project_root, "scripts", "generate_data200.py")
output_dir = os.path.join(project_root, "data", "data1000")

def clean_dir(target_dir):
    """Safely cleans the target output directory."""
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
    print("=" * 60)
    print("LegoVision - data1000 Parallel Generation (1000 Random Pieces)")
    print("=" * 60)
    
    clean_dir(output_dir)
    
    cpu_cores = os.cpu_count() or 12
    num_workers = max(1, cpu_cores - 2) # Reserva 2 cores para OS y UI
    print(f"[Orchestrator] Cores disponibles: {cpu_cores}. Lanzando {num_workers} workers de Blender...")
    
    t0 = time.time()
    processes = []
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    for worker_id in range(num_workers):
        cmd = [
            blender_path, "-b", "-P", script_path, "--",
            "--num_pieces", "1000",
            "--output_dir", output_dir,
            "--worker_id", str(worker_id),
            "--num_workers", str(num_workers)
        ]
        
        log_file = open(os.path.join(logs_dir, f"data1000_worker_{worker_id}.log"), "w")
        print(f"  -> Iniciando Worker {worker_id}...")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file, worker_id))
        
    print("[Orchestrator] Esperando a que todos los workers terminen...")
    failed = False
    for p, log_file, worker_id in processes:
        p.wait()
        log_file.close()
        if p.returncode != 0:
            print(f"[ERROR] Worker {worker_id} falló con código {p.returncode}. Revisa logs/data1000_worker_{worker_id}.log")
            failed = True
        else:
            print(f"[OK] Worker {worker_id} finalizado.")
            
    t1 = time.time()
    print("=" * 60)
    if failed:
        print("[Orchestrator] La generación de data1000 finalizó con errores.")
        sys.exit(1)
    else:
        print(f"[Orchestrator] ¡data1000 completada con éxito en {t1-t0:.2f} segundos!")
        print(f"Dataset de 1000 piezas guardado en: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
