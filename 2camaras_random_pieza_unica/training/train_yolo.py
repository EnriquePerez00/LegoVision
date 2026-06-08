import os
import sys
import yaml
import shutil
import random
import argparse
import subprocess
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────
# Detección dinámica de recursos (Opción C)
# Mantiene 20% CPU/RAM libre. Workers: max 8, ajusta dinámicamente.
# ─────────────────────────────────────────────────────────────────
def detect_optimal_training_config(reserve_pct=0.20, target_workers=8):
    """
    Detecta CPU/RAM disponibles y ajusta:
      - num_workers: máx 8, mín 3 (manteniendo 20% reservado).
      - batch_size: 32 si RAM disponible >= 16GB, 16 si menos.
      - device: 'mps' si disponible (M-series), 'cuda' si disponible, else 'cpu'.
    """
    # CPU
    try:
        total_cpu = int(subprocess.run(
            ["sysctl", "-n", "hw.logicalcpu"],
            capture_output=True, text=True
        ).stdout.strip())
    except Exception:
        total_cpu = os.cpu_count() or 8

    usable_cpu = max(2, int(total_cpu * (1 - reserve_pct)))
    
    # RAM disponible (macOS via vm_stat)
    available_ram_gb = 0
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        free_pages = inactive_pages = spec_pages = 0
        for line in result.stdout.split("\n"):
            if "Pages free:" in line:
                free_pages = int(line.split()[-1].replace(".", ""))
            elif "Pages inactive:" in line:
                inactive_pages = int(line.split()[-1].replace(".", ""))
            elif "Pages speculative:" in line:
                spec_pages = int(line.split()[-1].replace(".", ""))
        available_ram_gb = ((free_pages + inactive_pages + spec_pages) * 4096) / (1024 ** 3)
    except Exception:
        available_ram_gb = 8.0  # conservative fallback
    
    usable_ram = available_ram_gb * (1 - reserve_pct)
    
    # Workers: limitar por CPU; máx target_workers; mín 3
    num_workers = min(target_workers, usable_cpu // 2)
    num_workers = max(3, min(target_workers, num_workers))
    
    # Batch size dinámico
    # YOLO11n con batch=32 ~ 6-8GB GPU/MPS RAM
    if usable_ram >= 16:
        batch_size = 32
    elif usable_ram >= 8:
        batch_size = 16
    else:
        batch_size = 8
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"[OPT] System: {total_cpu} CPU cores, {available_ram_gb:.1f}GB RAM available")
    print(f"[OPT] Reserved 20%: usable_cpu={usable_cpu}, usable_ram={usable_ram:.1f}GB")
    print(f"[OPT] Training config: device={device}, batch={batch_size}, workers={num_workers}")
    
    return {
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "available_ram_gb": available_ram_gb,
        "total_cpu": total_cpu,
    }

# Cargar configuraciones y variables de entorno
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

RUN_ID = None
best_map50 = 0.0
epochs_without_delta_improvement = 0

def on_fit_epoch_end(trainer):
    """Callback de Ultralytics ejecutado al finalizar cada época de entrenamiento."""
    global RUN_ID, best_map50, epochs_without_delta_improvement
    if not RUN_ID:
        return
        
    epoch = trainer.epoch + 1
    
    # Obtener pérdidas y métricas de validación
    # trainer.loss_items es una lista de pérdidas de esta época
    loss = float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') and trainer.loss_items is not None else 0.0
    
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    
    # Intentar obtener métricas de validación por claves comunes de YOLOv8
    map50 = float(metrics.get('metrics/mAP50(B)', 0.0))
    val_box_loss = float(metrics.get('val/box_loss', 0.0))
    val_cls_loss = float(metrics.get('val/cls_loss', 0.0))
    val_loss = val_box_loss + val_cls_loss
    
    # Comprobar parada temprana con umbral de mejora de 0.001
    if map50 >= best_map50 + 0.001:
        best_map50 = map50
        epochs_without_delta_improvement = 0
    else:
        epochs_without_delta_improvement += 1
        
    log_text = f"Época {epoch}/{trainer.epochs}: loss_train={loss:.4f}, loss_val={val_loss:.4f}, mAP50={map50:.4f} (Mejor mAP50: {best_map50:.4f}, Épocas sin mejora >=0.001: {epochs_without_delta_improvement}/15)\n"
    print(f"[LegoVision Train Callback] {log_text.strip()}")
    
    if epochs_without_delta_improvement >= 15:
        trainer.stop = True
        log_text += "⏹ Parada temprana activada: No ha habido una mejora de 0.001 en mAP50 en las últimas 15 épocas.\n"
        print(f"[LegoVision Train Callback] {log_text.strip()}")
    
    try:
        supabase_client.update_training_progress(
            run_id=RUN_ID,
            current_epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            map50=map50,
            log_text=log_text
        )
    except Exception as e:
        print(f"Error actualizando progreso en BD: {e}")

def prepare_yolo_dataset(raw_dir, processed_dir, train_ratio=0.7):
    """Genera listas train.txt y val.txt con rutas absolutas a las imágenes para YOLO."""
    print("[LegoVision Train] Preparando y dividiendo el dataset usando listas de rutas (Método B)...")
    
    images_raw_dir = os.path.join(raw_dir, "images")
    
    if not os.path.exists(images_raw_dir) or not os.listdir(images_raw_dir):
        raise FileNotFoundError("No se encontraron imágenes en el dataset crudo generado.")
        
    # Limpiar directorio de salida
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
        
    # Obtener pares de imágenes y labels
    image_files = [f for f in os.listdir(images_raw_dir) if f.endswith(".png")]
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Si val_files está vacío (por dataset muy pequeño de pruebas), asegurar al menos 1 imagen
    if not val_files and train_files:
        val_files = [train_files.pop()]
    # Si train_files está vacío, asegurar al menos 1 imagen
    if not train_files and val_files:
        train_files = [val_files.pop()]
        
    def write_paths_file(file_list, filename):
        filepath = os.path.join(processed_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for img_name in file_list:
                abs_img_path = os.path.abspath(os.path.join(images_raw_dir, img_name))
                f.write(abs_img_path + "\n")
                
    write_paths_file(train_files, "train.txt")
    write_paths_file(val_files, "val.txt")
    
    print(f"[LegoVision Train] Listas de dataset generadas en {processed_dir}: {len(train_files)} train, {len(val_files)} val.")

def generate_yolo_yaml(processed_dir, yaml_path, include_minifigures=True):
    """Genera el archivo dataset.yaml para el detector utilizando archivos de texto para train/val."""
    names = {0: 'lego_piece', 1: 'minifigure'} if include_minifigures else {0: 'lego_piece'}
    yaml_data = {
        'path': os.path.abspath(processed_dir),
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': len(names),
        'names': names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    class_list = list(names.values())
    print(f"[LegoVision Train] YAML generado en: {yaml_path} | clases: {class_list}")

def main():
    global RUN_ID
    
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Entrenador YOLO11 Detector de LegoVision")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dataset_size", type=int, default=200, help="Imágenes de entrenamiento")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--set_id", type=str, default=None, help="Restringe a las piezas de este set")
    parser.add_argument("--run_id", type=str, default=None, help="ID de la corrida de entrenamiento existente en la BD")
    parser.add_argument("--raw_dataset_dir", type=str, default=None)
    parser.add_argument("--processed_dataset_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="yolo11_piece_detector")
    parser.add_argument("--camera", type=str, choices=["cenital", "lateral"], default=None,
                        help="Cámara a entrenar (auto-configura paths según el subproyecto)")
    
    args = parser.parse_args()
    
    # Si se especifica --camera, auto-configurar paths del dataset y model_name
    if args.camera:
        if not args.raw_dataset_dir:
            args.raw_dataset_dir = os.path.join(project_root, "data", f"yolo_{args.camera}")
        if not args.processed_dataset_dir:
            args.processed_dataset_dir = os.path.join(project_root, "data", f"yolo_{args.camera}_processed")
        # Nombre del modelo basado en cámara
        if args.model_name == "yolo11_piece_detector":
            args.model_name = f"yolo_{args.camera}"
        print(f"[LegoVision Train] Modo --camera={args.camera}: "
              f"raw_dir={args.raw_dataset_dir}, model_name={args.model_name}")
    
    # 1. Registrar inicio de entrenamiento en Base de Datos
    if args.run_id:
        RUN_ID = args.run_id
        print(f"[LegoVision Train] Usando corrida de entrenamiento existente: {RUN_ID}")
    else:
        config_dict = {
            "batch": args.batch,
            "imgsz": args.imgsz,
            "dataset_size": args.dataset_size,
            "device": args.device,
            "set_id": args.set_id
        }
        RUN_ID = supabase_client.create_training_run(args.epochs, config_dict)
        print(f"[LegoVision Train] Corrida registrada en BD con ID: {RUN_ID}")

    
    try:
        # 2. Utilizar el dataset existente
        raw_dataset_dir = args.raw_dataset_dir if args.raw_dataset_dir else os.path.join(project_root, "data", "raw_dataset")
        print(f"[LegoVision Train] Usando dataset local en {raw_dataset_dir}...")
        supabase_client.update_training_progress(
            run_id=RUN_ID,
            current_epoch=0,
            loss=0.0,
            val_loss=0.0,
            map50=0.0,
            log_text="Iniciando preparación del dataset local...\n"
        )
            
        # 3. Preparar carpetas train/val y YAML
        processed_dataset_dir = args.processed_dataset_dir if args.processed_dataset_dir else os.path.join(project_root, "data", "processed_dataset")
        prepare_yolo_dataset(raw_dataset_dir, processed_dataset_dir)
        
        yaml_path = os.path.join(project_root, "training", f"dataset_{args.model_name}.yaml")
        generate_yolo_yaml(processed_dataset_dir, yaml_path, include_minifigures=True)
        
        # 4. Entrenar YOLO11
        print("[LegoVision Train] Cargando modelo base yolo11n.pt...")
        supabase_client.update_training_progress(RUN_ID, 0, 0.0, 0.0, 0.0, "Cargando modelo base YOLO11 y preparando entrenamiento...\n")
        
        model = YOLO("yolo11n.pt")
        
        # Agregar el callback
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        # ── Detección dinámica de configuración óptima (Opción C) ──
        opt = detect_optimal_training_config(reserve_pct=0.20, target_workers=8)
        # Override con args si el usuario los especifica explícitamente (default 16)
        # Pero respetamos siempre el cap del 80%
        train_device = opt["device"]
        train_batch = opt["batch_size"]
        train_workers = opt["num_workers"]
        # Si MPS, AMP suele fallar (mantener False)
        train_amp = train_device == "cuda"

        print(f"[LegoVision Train] Iniciando entrenamiento en {train_device} "
              f"(batch={train_batch}, workers={train_workers}, amp={train_amp})...")
        model.train(
            data=yaml_path,
            epochs=35,
            batch=train_batch,        # Dinámico: 32 si RAM>=16GB, 16 si <16, 8 si <8
            patience=15,
            imgsz=args.imgsz,
            device=train_device,      # mps / cuda / cpu (auto-detección)
            amp=train_amp,            # FP16 mixed precision (sólo CUDA estable)
            workers=train_workers,    # Dinámico: hasta 8 (manteniendo 20% libre)
            plots=False,
            cache=True,
            project="LegoVision",
            name=args.model_name,
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=360.0,
            fliplr=0.5,
            flipud=0.5
        )
        
        # Guardar pesos entrenados en models/
        best_pt_src = os.path.join(project_root, "runs", "detect", "LegoVision", args.model_name, "weights", "best.pt")
        if not os.path.exists(best_pt_src):
            best_pt_src = os.path.join(project_root, "runs", "detect", args.model_name, "weights", "best.pt")
            
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        best_pt_dst = os.path.join(models_dir, f"{args.model_name}.pt")
        
        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dst)
            print(f"[LegoVision Train] Pesos exportados correctamente a: {best_pt_dst}")
        else:
            print(f"[LegoVision Train WARNING] No se encontró el archivo best.pt en las rutas de Ultralytics.")
            
        supabase_client.complete_training_run(RUN_ID, "completed", "Entrenamiento finalizado y pesos guardados con éxito.\n")

        # ── Evaluación post-entrenamiento (Opción C: simulación física del set) ──
        sim_img  = os.path.join(project_root, "data", "synthetic_renders", "set_scatter_75078-1.png")
        sim_json = os.path.join(project_root, "data", "synthetic_renders", "set_scatter_75078-1.json")
        eval_out = os.path.join(project_root, "data", "eval_results_latest.json")

        if os.path.exists(sim_img) and os.path.exists(sim_json):
            print("\n[LegoVision Train] Iniciando evaluación post-entrenamiento en simulación física del set...")
            try:
                sys.path.insert(0, os.path.join(project_root, "training"))
                from eval_on_set_simulation import evaluate as eval_sim
                eval_results = eval_sim(
                    model_path=best_pt_dst,
                    sim_img=sim_img,
                    sim_json=sim_json,
                    conf=0.25,
                    iou_thr=0.50
                )
                # Guardar resultados localmente
                with open(eval_out, "w", encoding="utf-8") as ef:
                    import json as _json
                    _json.dump(eval_results, ef, indent=2, ensure_ascii=False)
                print(f"[LegoVision Train] Resultados de evaluación guardados en: {eval_out}")
                # Persistir en BD si hay RUN_ID
                if RUN_ID and "error" not in eval_results:
                    eval_log = (
                        f"\n=== EVALUACIÓN EN SIMULACIÓN FÍSICA ===\n"
                        f"GT piezas: {eval_results['gt_total']} | "
                        f"Detectadas: {eval_results['detections_total']}\n"
                        f"TP={eval_results['tp']} FP={eval_results['fp']} FN={eval_results['fn']}\n"
                        f"Precision={eval_results['precision']:.4f} | "
                        f"Recall={eval_results['recall']:.4f} | "
                        f"F1={eval_results['f1']:.4f} | "
                        f"mAP@0.5={eval_results['map50']:.4f}\n"
                    )
                    supabase_client.update_training_progress(
                        run_id=RUN_ID,
                        current_epoch=args.epochs,
                        loss=0.0,
                        val_loss=0.0,
                        map50=eval_results["map50"],
                        log_text=eval_log
                    )
                    print(f"[LegoVision Train] Eval registrada: Precision={eval_results['precision']:.4f} Recall={eval_results['recall']:.4f} mAP50={eval_results['map50']:.4f}")
            except Exception as eval_err:
                print(f"[LegoVision Train WARN] Evaluación post-entrenamiento falló: {eval_err}")
        else:
            print(f"[LegoVision Train] Sin imagen de simulación para evaluación (genera con ⚡ Simular Set Completo).")

    except Exception as e:
        print(f"[LegoVision Train ERROR] El entrenamiento falló: {e}")
        if RUN_ID:
            supabase_client.complete_training_run(RUN_ID, "failed", f"ERROR: {str(e)}\n")
            
if __name__ == "__main__":
    main()
