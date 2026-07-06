# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import time

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
blender_path = "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender"
sim_script = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
output_sim_dir = os.path.join(project_root, "data", "simulation_10")
metadata_path = os.path.join(output_sim_dir, "simulation_metadata.json")

def main():
    print("--- 1. Generando metadatos para la simulación de 10 piezas a 5 m/min ---")
    
    # 1.1 Limpiar directorio de salida
    if os.path.exists(output_sim_dir):
        import shutil
        print(f"Limpiando directorio {output_sim_dir}...")
        shutil.rmtree(output_sim_dir)
    os.makedirs(output_sim_dir, exist_ok=True)
    
    # 1.2 Ejecutar metadata_only
    cmd_meta = [
        blender_path, "-b", "-P", sim_script, "--",
        "--num_pieces", "10",
        "--output_dir", output_sim_dir,
        "--speed", "0.083333",
        "--frames_per_piece", "10",
        "--set-id", "75078-1",
        "--metadata_only"
    ]
    subprocess.run(cmd_meta, check=True)
    
    # 1.3 Ejecutar workers en paralelo
    num_workers = 10
    print(f"Lanzando {num_workers} workers de Blender en paralelo...")
    
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
    processes = []
    for worker_id in range(num_workers):
        cmd = [
            blender_path, "-b", "-P", sim_script, "--",
            "--num_pieces", "10",
            "--output_dir", output_sim_dir,
            "--speed", "0.083333",
            "--frames_per_piece", "10",
            "--set-id", "75078-1",
            "--worker_id", str(worker_id),
            "--num_workers", str(num_workers)
        ]
        log_file = open(os.path.join(project_root, "logs", f"sim_worker_10_{worker_id}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
        
    print("Esperando a que los workers terminen...")
    for p, log_file in processes:
        p.wait()
        log_file.close()
        
    print("--- Renders completados para 10 piezas a 5 m/min ---")

if __name__ == "__main__":
    main()
