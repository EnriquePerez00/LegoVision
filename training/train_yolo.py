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

def on_fit_epoch_end(trainer):
    """Callback de Ultralytics ejecutado al finalizar cada época de entrenamiento."""
    global RUN_ID
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
    
    log_text = f"Época {epoch}/{trainer.epochs}: loss_train={loss:.4f}, loss_val={val_loss:.4f}, mAP50={map50:.4f}\n"
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
    """Divide las imágenes generadas por Blender en carpetas train/val para YOLO."""
    print("[LegoVision Train] Preparando y dividiendo el dataset...")
    
    images_raw_dir = os.path.join(raw_dir, "images")
    labels_raw_dir = os.path.join(raw_dir, "labels")
    
    if not os.path.exists(images_raw_dir) or not os.listdir(images_raw_dir):
        raise FileNotFoundError("No se encontraron imágenes en el dataset crudo generado.")
        
    # Limpiar directorio de salida
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
        
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(processed_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, subset, "labels"), exist_ok=True)
        
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
        
    def copy_files(file_list, subset):
        for img_name in file_list:
            lbl_name = img_name.replace(".png", ".txt")
            
            src_img = os.path.join(images_raw_dir, img_name)
            dst_img = os.path.join(processed_dir, subset, "images", img_name)
            shutil.copy2(src_img, dst_img)
            
            src_lbl = os.path.join(labels_raw_dir, lbl_name)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(processed_dir, subset, "labels", lbl_name)
                shutil.copy2(src_lbl, dst_lbl)
                
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    
    print(f"[LegoVision Train] Dataset estructurado: {len(train_files)} train, {len(val_files)} val.")

def generate_yolo_yaml(processed_dir, yaml_path):
    """Genera el archivo dataset.yaml para el detector single-class."""
    yaml_data = {
        'path': os.path.abspath(processed_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'lego_piece'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print(f"[LegoVision Train] YAML del dataset único generado en: {yaml_path}")

def main():
    global RUN_ID
    
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Entrenador YOLO11 Detector de LegoVision")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dataset_size", type=int, default=200, help="Imágenes sintéticas a generar")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--set_id", type=str, default=None, help="Restringe a las piezas de este set")
    parser.add_argument("--belt_mode", action="store_true", default=True, help="Usa el modo cinta sin físicas de Blender")
    
    args = parser.parse_args()
    
    # 1. Registrar inicio de entrenamiento en Base de Datos
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
        # 2. Generar dataset sintético vía Blender
        blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
        raw_dataset_dir = os.path.join(project_root, "data", "raw_dataset")
        
        print(f"[LegoVision Train] Lanzando Blender para generar {args.dataset_size} imágenes sintéticas...")
        supabase_client.update_training_progress(
            run_id=RUN_ID,
            current_epoch=0,
            loss=0.0,
            val_loss=0.0,
            map50=0.0,
            log_text="Iniciando generación de dataset sintético en Blender...\n"
        )
        
        cmd = [
            blender_path,
            "--background",
            "--python", "blender_pipeline/generate_dataset.py",
            "--",
            "--num_images", str(args.dataset_size),
            "--pieces_per_image", "6",
            "--single_class"
        ]
        if args.belt_mode:
            cmd.append("--belt_mode")
        if args.set_id:
            cmd.append("--set_id")
            cmd.append(args.set_id)
        
        # Ejecutar e ir canalizando logs a base de datos
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=project_root)
        
        for line in process.stdout:
            # Imprimir en consola y guardar en log de base de datos
            sys.stdout.write(line)
            sys.stdout.flush()
            # Limitar logs a líneas clave de Blender para no saturar
            if "[LegoVision" in line or "Saved:" in line or "Generando Imagen" in line:
                supabase_client.update_training_progress(RUN_ID, 0, 0.0, 0.0, 0.0, line)
                
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"La generación del dataset de Blender falló con código {process.returncode}")
            
        # 3. Preparar carpetas train/val y YAML
        processed_dataset_dir = os.path.join(project_root, "data", "processed_dataset")
        prepare_yolo_dataset(raw_dataset_dir, processed_dataset_dir)
        
        yaml_path = os.path.join(project_root, "training", "dataset.yaml")
        generate_yolo_yaml(processed_dataset_dir, yaml_path)
        
        # 4. Entrenar YOLO11
        print("[LegoVision Train] Cargando modelo base yolo11n.pt...")
        supabase_client.update_training_progress(RUN_ID, 0, 0.0, 0.0, 0.0, "Cargando modelo base YOLO11 y preparando entrenamiento...\n")
        
        model = YOLO("yolo11n.pt")
        
        # Agregar el callback
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        
        print(f"[LegoVision Train] Iniciando entrenamiento en {args.device}...")
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            amp=False if args.device == "mps" else True, # AMP suele fallar en MPS actual
            workers=2,
            project="LegoVision",
            name="yolo11_piece_detector",
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=360.0, # Rotación 360 grados para robustez
            fliplr=0.5,
            flipud=0.5
        )
        
        # Guardar pesos entrenados en models/
        best_pt_src = os.path.join(project_root, "runs", "detect", "LegoVision", "yolo11_piece_detector", "weights", "best.pt")
        if not os.path.exists(best_pt_src):
            best_pt_src = os.path.join(project_root, "runs", "detect", "yolo11_piece_detector", "weights", "best.pt")
            
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        best_pt_dst = os.path.join(models_dir, "best.pt")
        
        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dst)
            print(f"[LegoVision Train] Pesos exportados correctamente a: {best_pt_dst}")
        else:
            print(f"[LegoVision Train WARNING] No se encontró el archivo best.pt en las rutas de Ultralytics.")
            
        supabase_client.complete_training_run(RUN_ID, "completed", "Entrenamiento finalizado y pesos guardados con éxito.\n")
        
    except Exception as e:
        print(f"[LegoVision Train ERROR] El entrenamiento falló: {e}")
        if RUN_ID:
            supabase_client.complete_training_run(RUN_ID, "failed", f"ERROR: {str(e)}\n")
            
if __name__ == "__main__":
    main()
