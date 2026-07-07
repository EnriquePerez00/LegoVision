import os
import subprocess
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_max_workers():
    import psutil
    cores = os.cpu_count() or 4
    return max(1, cores - 2)

def run_worker(worker_id, num_workers, output_dir, project_root):
    blender_path = "/opt/homebrew/bin/blender"
    script_path = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
    
    cmd = [
        blender_path, "-b", "-P", script_path, "--",
        "--num_pieces", "1000",
        "--output_dir", output_dir,
        "--set-id", "database_all_colors",
        "--dimension", "2D",
        "--resolution", "2048",
        "--randomize_lighting",
        "--worker_id", str(worker_id),
        "--num_workers", str(num_workers)
    ]
    
    print(f"[Worker {worker_id}] cmd: {' '.join(cmd)}")
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return worker_id, result.returncode, result.stdout, result.stderr

def run_simulation():
    project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
    output_dir = os.path.join(project_root, "data", "simulation_1000_database_2D")
    
    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    os.makedirs(output_dir, exist_ok=True)
    
    num_workers = get_max_workers()
    print(f"Iniciando simulación 2D paralela con {num_workers} workers paralelos.")
    
    # 1. Metadata Only run
    print("Pre-calculando empaquetado y metadatos...")
    blender_path = "/opt/homebrew/bin/blender"
    script_path = os.path.join(project_root, "scripts", "generate_conveyor_simulation.py")
    cmd_meta = [
        blender_path, "-b", "-P", script_path, "--",
        "--num_pieces", "1000",
        "--output_dir", output_dir,
        "--set-id", "database_all_colors",
        "--dimension", "2D",
        "--resolution", "2048",
        "--randomize_lighting",
        "--metadata_only"
    ]
    subprocess.run(cmd_meta, check=True)
    print("Metadatos generados exitosamente.")

    # 2. Parallel rendering
    print(f"Ejecutando renderizado en {num_workers} chunks...")
    start_t = time.time()
        
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_worker, w_id, num_workers, output_dir, project_root): w_id 
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
            except Exception as e:
                print(f"[Worker {w_id}] Generó una excepción: {e}")

    elapsed = time.time() - start_t
    print(f"Simulación finalizada en {elapsed:.2f} s")

if __name__ == "__main__":
    run_simulation()
