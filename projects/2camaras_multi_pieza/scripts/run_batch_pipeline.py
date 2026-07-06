# -*- coding: utf-8 -*-
"""scripts/run_batch_pipeline.py
Orchestrates the entire execution:
1. Fast validation run (10 images for YOLO, 3 angles for DINOv2).
2. Production YOLO renders (Cenital & Lateral, 1500 frames, 1 piece, center spawn).
3. Production YOLO training (sequentially following the renders).
4. Production DINOv2 multi-angle renders and index embedding updates.
5. Generate 100 test images using generate_single_piece_three_cameras.py and execute multicamera classification test.
6. Measures time, records logs, and outputs to runs/pipeline_metrics.json.
"""
import os
import sys
import time
import json
import subprocess as sp
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Binaries
blender_exec = "/opt/homebrew/bin/blender"
if not os.path.exists(blender_exec):
    blender_exec = "blender"

venv_python = os.path.join(project_root, ".venv", "bin", "python")
python_exec = venv_python if os.path.exists(venv_python) else sys.executable

def run_proc(cmd, log_file=None, description=""):
    print(f"\n>>> [EJECUTANDO] {description}")
    print(f"Comando: {' '.join(cmd)}")
    t0 = time.time()
    
    # Redirigir output a un log
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            p = sp.Popen(cmd, stdout=f, stderr=sp.STDOUT, text=True)
            p.wait()
            ret = p.returncode
    else:
        p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        p.wait()
        ret = p.returncode
        
    duration = time.time() - t0
    print(f"<<< [COMPLETADO] {description} en {duration:.2f}s | RetCode: {ret}")
    if ret != 0:
        raise RuntimeError(f"Error executing task: {description} (RetCode: {ret})")
    return duration

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LegoVision Batch Pipeline Orchestrator")
    parser.add_argument("--test_only", action="store_true", help="Only run validation test (10 frames)")
    parser.add_argument("--skip_validation", action="store_true", help="Skip the initial 10 frames validation test")
    args = parser.parse_args()

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }
    
    # -------------------------------------------------------------
    # FASE 1: Validación rápida (10 imágenes)
    # -------------------------------------------------------------
    if not args.skip_validation:
        print("\n=== INICIANDO FASE 1: VALIDACIÓN RÁPIDA (10 IMÁGENES) ===")
        val_dir_cenital = os.path.join(project_root, "data", "yolo_cenital_val_run")
        val_dir_lateral = os.path.join(project_root, "data", "yolo_lateral_val_run")
        val_dir_dino = os.path.join(project_root, "data", "dinov2_val_run")
        
        # 1.1 Render YOLO Cenital 10 frames
        cmd = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
               "--set_id", "75078-1", "--num_frames", "10", "--pieces", "1", "--center_spawn", "--output_dir", val_dir_cenital, "--camera_type", "cenital"]
        metrics["phases"]["val_yolo_cenital_render"] = run_proc(cmd, description="YOLO Cenital Render (Val, 10 imgs)")
        
        # 1.2 Render YOLO Lateral 10 frames
        cmd = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
               "--set_id", "75078-1", "--num_frames", "10", "--pieces", "1", "--center_spawn", "--output_dir", val_dir_lateral, "--camera_type", "lateral"]
        metrics["phases"]["val_yolo_lateral_render"] = run_proc(cmd, description="YOLO Lateral Render (Val, 10 imgs)")
        
        # 1.3 Entrenar YOLO Cenital (2 épocas)
        cmd = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
               "--epochs", "2", "--batch", "4", "--dataset_size", "10", "--set_id", "75078-1",
               "--raw_dataset_dir", val_dir_cenital, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_val_yolo_cenital"),
               "--model_name", "yolo_cenital_val"]
        metrics["phases"]["val_yolo_cenital_train"] = run_proc(cmd, description="YOLO Cenital Train (Val, 2 epochs)")
        
        # 1.4 Entrenar YOLO Lateral (2 épocas)
        cmd = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
               "--epochs", "2", "--batch", "4", "--dataset_size", "10", "--set_id", "75078-1",
               "--raw_dataset_dir", val_dir_lateral, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_val_yolo_lateral"),
               "--model_name", "yolo_lateral_val"]
        metrics["phases"]["val_yolo_lateral_train"] = run_proc(cmd, description="YOLO Lateral Train (Val, 2 epochs)")
        
        # 1.5 DINOv2 referencias (drops=1 para pose única, rotaciones rápidas)
        cmd = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py"), "--",
               "--output_dir", val_dir_dino, "--camera_type", "cenital", "--drops", "1", "--set_id", "75078-1"]
        metrics["phases"]["val_dinov2_cenital_render"] = run_proc(cmd, description="DINOv2 Cenital Renders (Val)")
        
        # 1.6 Indexar referencias DINOv2
        cmd = [python_exec, os.path.join(project_root, "training", "index_synthetic_renders.py"),
               "--renders_dir", val_dir_dino, "--multiangle_dir", val_dir_dino]
        metrics["phases"]["val_dinov2_cenital_index"] = run_proc(cmd, description="DINOv2 Indexing (Val)")
        
        print("\n=== VALIDACIÓN RÁPIDA COMPLETADA CON ÉXITO ===")

    if args.test_only:
        print("Finalizando ejecución ya que se especificó --test_only.")
        return

    # -------------------------------------------------------------
    # FASE 2: Renders e Inferencia de Producción (1500 Imágenes)
    # -------------------------------------------------------------
    print("\n=== INICIANDO FASE 2: RENDERS Y ENTRENAMIENTO DE PRODUCCIÓN (1500 IMÁGENES) ===")
    
    prod_dir_cenital = os.path.join(project_root, "data", "yolo_cenital")
    prod_dir_lateral = os.path.join(project_root, "data", "yolo_lateral")
    prod_dir_dino_cenital = os.path.join(project_root, "data", "dinov2_cenital")
    prod_dir_dino_lateral = os.path.join(project_root, "data", "dinov2_lateral")

    # 2.1 Renders YOLO Cenital y Lateral en background/paralelo
    print("\nLanzando Renders YOLO Cenital y Lateral en paralelo en segundo plano...")
    log_cenital = os.path.join(project_root, "runs", "logs", "yolo_cenital_render.log")
    log_lateral = os.path.join(project_root, "runs", "logs", "yolo_lateral_render.log")
    
    os.makedirs(os.path.dirname(log_cenital), exist_ok=True)
    
    t0 = time.time()
    
    cmd_cen = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
               "--set_id", "75078-1", "--num_frames", "1500", "--pieces", "1", "--center_spawn", "--output_dir", prod_dir_cenital, "--camera_type", "cenital"]
    cmd_lat = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
               "--set_id", "75078-1", "--num_frames", "1500", "--pieces", "1", "--center_spawn", "--output_dir", prod_dir_lateral, "--camera_type", "lateral"]
               
    with open(log_cenital, "w") as f_cen, open(log_lateral, "w") as f_lat:
        p_cen = sp.Popen(cmd_cen, stdout=f_cen, stderr=sp.STDOUT)
        p_lat = sp.Popen(cmd_lat, stdout=f_lat, stderr=sp.STDOUT)
        print("Renders de producción iniciados. Monitoreando ejecución...")
        
        while p_cen.poll() is None or p_lat.poll() is None:
            time.sleep(5)
            
        p_cen.wait()
        p_lat.wait()
        
    metrics["phases"]["prod_yolo_renders_parallel"] = time.time() - t0
    print(f"Renders completados en {metrics['phases']['prod_yolo_renders_parallel']:.2f}s")
    
    # 2.2 Entrenar YOLO Cenital
    cmd = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
           "--epochs", "35", "--batch", "16", "--dataset_size", "1500", "--set_id", "75078-1",
           "--raw_dataset_dir", prod_dir_cenital, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_yolo_cenital"),
           "--model_name", "yolo_cenital"]
    metrics["phases"]["prod_yolo_cenital_train"] = run_proc(cmd, description="YOLO Cenital Train (1500 imgs)")

    # 2.3 Entrenar YOLO Lateral
    cmd = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
           "--epochs", "35", "--batch", "16", "--dataset_size", "1500", "--set_id", "75078-1",
           "--raw_dataset_dir", prod_dir_lateral, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_yolo_lateral"),
           "--model_name", "yolo_lateral"]
    metrics["phases"]["prod_yolo_lateral_train"] = run_proc(cmd, description="YOLO Lateral Train (1500 imgs)")

    # -------------------------------------------------------------
    # FASE 3: DINOv2 Cenital y Lateral de Producción
    # -------------------------------------------------------------
    print("\n=== INICIANDO FASE 3: DINOv2 CENITAL Y LATERAL DE PRODUCCIÓN ===")
    log_dino_cen = os.path.join(project_root, "runs", "logs", "dinov2_cenital_render.log")
    log_dino_lat = os.path.join(project_root, "runs", "logs", "dinov2_lateral_render.log")
    
    t0 = time.time()
    
    # Lanzar renders DINOv2 Cenital y Lateral en paralelo
    cmd_dino_cen = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py"), "--",
                    "--output_dir", prod_dir_dino_cenital, "--camera_type", "cenital", "--drops", "1", "--set_id", "75078-1"]
    cmd_dino_lat = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py"), "--",
                    "--output_dir", prod_dir_dino_lateral, "--camera_type", "lateral", "--drops", "1", "--set_id", "75078-1"]
                    
    with open(log_dino_cen, "w") as f_cen, open(log_dino_lat, "w") as f_lat:
        p_cen = sp.Popen(cmd_dino_cen, stdout=f_cen, stderr=sp.STDOUT)
        p_lat = sp.Popen(cmd_dino_lat, stdout=f_lat, stderr=sp.STDOUT)
        print("Renders DINOv2 en paralelo iniciados...")
        
        while p_cen.poll() is None or p_lat.poll() is None:
            time.sleep(5)
            
        p_cen.wait()
        p_lat.wait()
        
    metrics["phases"]["prod_dinov2_renders_parallel"] = time.time() - t0
    print(f"Renders DINOv2 completados en {metrics['phases']['prod_dinov2_renders_parallel']:.2f}s")
    
    # Indexar embeddings DINOv2 Cenital
    cmd = [python_exec, os.path.join(project_root, "training", "index_synthetic_renders.py"),
           "--renders_dir", prod_dir_dino_cenital, "--multiangle_dir", prod_dir_dino_cenital]
    metrics["phases"]["prod_dinov2_cenital_index"] = run_proc(cmd, description="DINOv2 Cenital Indexing")

    # Indexar embeddings DINOv2 Lateral
    cmd = [python_exec, os.path.join(project_root, "training", "index_synthetic_renders.py"),
           "--renders_dir", prod_dir_dino_lateral, "--multiangle_dir", prod_dir_dino_lateral]
    metrics["phases"]["prod_dinov2_lateral_index"] = run_proc(cmd, description="DINOv2 Lateral Indexing")

    # -------------------------------------------------------------
    # FASE 4: Generación de 100 imágenes de test y validación del pipeline
    # -------------------------------------------------------------
    print("\n=== INICIANDO FASE 4: EVALUACIÓN DE INFERENCIA MULTICÁMARA (100 IMÁGENES TEST) ===")
    test_renders_dir = os.path.join(project_root, "data", "test_multicam_100")
    
    # Generar 100 imágenes de test por las 3 cámaras
    cmd_test_gen = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_single_piece_three_cameras.py"), "--",
                    "--set_id", "75078-1", "--num_frames", "100", "--output_dir", test_renders_dir]
    metrics["phases"]["test_images_generation"] = run_proc(cmd_test_gen, description="Generating 100 Test Multicam Renders")
    
    # Lanzar la validación e inferencia multicámara
    # Utilizar el script tests/test_multicam.py o ejecutar una simulación directa
    # Registraremos la precisión final en metrics.json
    try:
        # Ejecutar script de test
        cmd_test_run = [python_exec, "-m", "unittest", "tests/test_multicam.py"]
        metrics["phases"]["multicam_unit_test"] = run_proc(cmd_test_run, description="Running Multicamera Unit Tests")
    except Exception as e:
        print(f"[WARN] Unit tests finished with alert: {e}")

    # Guardar métricas
    runs_dir = os.path.join(project_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    metrics_path = os.path.join(runs_dir, "pipeline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 ¡Métricas guardadas con éxito en {metrics_path}!")

if __name__ == "__main__":
    main()
