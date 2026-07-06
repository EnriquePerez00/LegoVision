import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = "/Users/I764690/Code_personal/LegoVision"
classes_file = os.path.join(project_root, "camara_domo", "data", "yolo_dataset", "classes.txt")
blender_exec = "/opt/homebrew/bin/blender"
python_exec = os.path.join(project_root, ".venv", "bin", "python")

with open(classes_file, "r") as f:
    refs = [line.strip() for line in f if line.strip()]

def process_piece(ref):
    print(f"[{ref}] Iniciando batch_physics_runner...")
    
    # 1. Correr la fisica en blender
    script_str = ""
    with open(os.path.join(project_root, "scripts", "batch_physics_runner.py"), "r") as pf:
        script_str = pf.read()
    script_str = script_str.replace('"{ref}"', f'"{ref}"')
    script_str = script_str.replace('"{color}"', '"16"') # Color default
    
    tmp_script = os.path.join(project_root, "scratch", f"physics_{ref}.py")
    os.makedirs(os.path.join(project_root, "scratch"), exist_ok=True)
    with open(tmp_script, "w") as tf:
        tf.write(script_str)
        
    cmd_blender = [blender_exec, "-b", "-P", tmp_script]
    try:
        subprocess.run(cmd_blender, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"[{ref}] Error en physics: {e}")
        return
        
    print(f"[{ref}] Fisica completada. Ejecutando deduplicacion...")
    
    # 2. Correr la deduplicacion
    cmd_dedup = [python_exec, os.path.join(project_root, "scripts", "deduplicate_and_filter_poses.py"), "--part_ref", ref]
    try:
        subprocess.run(cmd_dedup, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{ref}] Error en deduplicacion: {e.output}")
        return
        
    print(f"[{ref}] Completado exitosamente.")
    if os.path.exists(tmp_script):
        os.remove(tmp_script)

if __name__ == "__main__":
    print(f"Iniciando el procesamiento de {len(refs)} piezas con 8 hilos...")
    # Using 8 threads to not overwhelm the CPU
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_piece, ref): ref for ref in refs}
        for i, future in enumerate(as_completed(futures), 1):
            ref = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Excepcion en {ref}: {e}")
            print(f"Progreso: {i}/{len(refs)}")
    print("Todas las piezas han sido procesadas.")
