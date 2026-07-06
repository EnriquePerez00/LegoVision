# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
if not os.path.exists(blender_path):
    blender_path = "blender"

sim_script = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
output_sim_dir = os.path.join(project_root, "data", "simulation_300")

def main():
    print("=" * 60)
    print("LegoVision - 75078 Set Parallel Rendering Only (300 pieces)")
    print("=" * 60)
    
    render_start = time.time()
    
    # 1. Ejecutar pase rápido de metadatos (metadata_only) para calcular la cinta y offsets
    print("[Orchestrator] Generando metadatos globales...")
    cmd_meta = [
        blender_path, "-b", "-P", sim_script, "--",
        "--num_pieces", "300",
        "--output_dir", output_sim_dir,
        "--speed", "5.0",
        "--metadata_only"
    ]
    subprocess.run(cmd_meta, check=True)
    print("[Orchestrator] Metadatos generados exitosamente.")
    
    # 2. Ejecutar workers en paralelo
    cpu_cores = os.cpu_count() or 12
    num_workers = max(1, cpu_cores - 2) # Reserva 2 cores para OS y UX
    print(f"[Orchestrator] Cores disponibles: {cpu_cores}. Lanzando {num_workers} workers de Blender...")
    
    processes = []
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    for worker_id in range(num_workers):
        cmd = [
            blender_path, "-b", "-P", sim_script, "--",
            "--num_pieces", "300",
            "--output_dir", output_sim_dir,
            "--speed", "5.0",
            "--worker_id", str(worker_id),
            "--num_workers", str(num_workers)
        ]
        log_file = open(os.path.join(logs_dir, f"sim_worker_{worker_id}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file, worker_id))
        
    print("[Orchestrator] Esperando renderizados paralelos...")
    failed = False
    for p, log_file, worker_id in processes:
        p.wait()
        log_file.close()
        if p.returncode != 0:
            print(f"[ERROR] Worker {worker_id} falló con código {p.returncode}.")
            failed = True
        else:
            print(f"[OK] Worker {worker_id} finalizado.")
            
    if failed:
        print("[ERROR] Uno o más workers de renderizado fallaron.")
        sys.exit(1)
        
    render_end = time.time()
    render_duration = render_end - render_start
    print("=" * 60)
    print(f"Renderización paralela completada con éxito en {render_duration:.2f} segundos.")
    print(f"Renders guardados en: {output_sim_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
