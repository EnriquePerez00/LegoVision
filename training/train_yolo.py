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

# Cargar configuraciones y variables de entorno
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

RUN_ID = None
best_map50 = 0.0
epochs_without_delta_improvement = 0

def on_fit_epoch_end(trainer):
    """Callback de Ultralytics ejecutado al finalizar cada época de entrenamiento."""
    global best_map50, epochs_without_delta_improvement
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
    parser.add_argument("--remote", action="store_true", help="Desactiva Supabase y optimiza para entorno T4 remoto")
    
    args = parser.parse_args()
    
    # 1. Registrar inicio de entrenamiento en Base de Datos
    if args.remote:
        print("[LegoVision Train] Remote mode enabled: skipping Supabase database sync.")
        RUN_ID = None
        # Optimize default args for remote T4 if not explicitly passed
        if args.device == ("mps" if torch.backends.mps.is_available() else "cpu"):
            args.device = "cuda"
        if args.batch == 16:
            args.batch = 64
    elif args.run_id:
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
        if RUN_ID:
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
        if RUN_ID:
            supabase_client.update_training_progress(RUN_ID, 0, 0.0, 0.0, 0.0, "Cargando modelo base YOLO11 y preparando entrenamiento...\n")
        
        model = YOLO("yolo11n.pt")
        
        # Agregar el callback
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        print(f"[LegoVision Train] Iniciando entrenamiento en {args.device}...")
        
        # Optimize hyperparams based on device
        is_cuda = args.device == "cuda"
        train_amp = True if is_cuda else False
        train_workers = 4 if is_cuda else 2

        model.train(
            data=yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            patience=15,        # Detención temprana si no mejora durante 15 épocas
            imgsz=args.imgsz,
            device=args.device,
            amp=train_amp,
            workers=train_workers,
            plots=False,        # Evita escrituras redundantes de imágenes en disco
            cache=True,         # Caché de imágenes en RAM para entrenar muchísimo más rápido
            project="LegoVision",
            name=args.model_name,
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=360.0, # Rotación 360 grados para robustez
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
            
        if RUN_ID:
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
