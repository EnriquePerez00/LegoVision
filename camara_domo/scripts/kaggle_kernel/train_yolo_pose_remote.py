import os
import sys
import subprocess

try:
    import ultralytics
except ImportError:
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "python-dotenv"], check=True)

from ultralytics import YOLO

def create_yaml(dataset_input_path, output_dir, prefix):
    # Generar un YAML class-agnostic (1 sola clase)
    yaml_content = f"""path: {dataset_input_path}
train: images/train
val: images/val

# Keypoints format
kpt_shape: [9, 3]

names:
  0: 'lego_piece'
"""
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{prefix}_dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    return yaml_path

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Detectar entorno Kaggle para usar rutas optimizadas (I/O)
    is_kaggle = os.path.exists("/kaggle/working")
    if is_kaggle:
        import glob
        # En Kaggle, el dataset se monta en /kaggle/input/<nombre_del_dataset>
        input_dirs = glob.glob("/kaggle/input/*")
        if not input_dirs:
            raise FileNotFoundError("No datasets found in /kaggle/input/")
        kaggle_dataset_dir = input_dirs[0]
        
        dataset_cenital = os.path.join(kaggle_dataset_dir, "yolo_dataset_cenital")
        dataset_frontal = os.path.join(kaggle_dataset_dir, "yolo_dataset_frontal")
        output_data_dir = "/kaggle/working/camara_domo/data"
        project_train_runs = "/kaggle/working/camara_domo/models/train_runs"
        models_out_dir = "/kaggle/working/camara_domo/models"
        device = [0, 1] # DDP en Kaggle (2x T4)
        batch_size = 32 # VRAM saturada para dual T4
    else:
        dataset_cenital = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_cenital")
        dataset_frontal = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_frontal")
        output_data_dir = os.path.join(project_root, "camara_domo", "data")
        project_train_runs = os.path.join(project_root, "camara_domo", "models", "train_runs")
        models_out_dir = os.path.join(project_root, "camara_domo", "models")
        device = "cuda"
        batch_size = 16
        
    print("=== Generando YAMLs (Class-Agnostic) ===")
    yaml_cenital = create_yaml(dataset_cenital, output_data_dir, "cenital")
    yaml_frontal = create_yaml(dataset_frontal, output_data_dir, "frontal")
    
    base_model = "yolo11s-pose.pt"
    epochs = 100
    workers = 4
    imgsz = 640
    
    os.makedirs(project_train_runs, exist_ok=True)
    os.makedirs(models_out_dir, exist_ok=True)
    
    # --- Checkpoints Recovery ---
    last_cenital_checkpoint = "/kaggle/input/domo-training-data/last_cenital.pt" if is_kaggle else os.path.join(models_out_dir, "last_cenital.pt")
    last_frontal_checkpoint = "/kaggle/input/domo-training-data/last_frontal.pt" if is_kaggle else os.path.join(models_out_dir, "last_frontal.pt")

    # --- Entrenar Cenital ---
    print("\n" + "="*50)
    print("ENTRENANDO MODELO CENITAL-POSE (REMOTO T4)")
    print("="*50)
    
    if os.path.exists(last_cenital_checkpoint):
        print(f"🔄 Reanudando entrenamiento cenital desde checkpoint: {last_cenital_checkpoint}")
        model_cen = YOLO(last_cenital_checkpoint)
        resume_cen = True
    else:
        model_cen = YOLO(base_model)
        resume_cen = False

    model_cen.train(
        data=yaml_cenital,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=project_train_runs,
        name="domo_cenital_pose",
        exist_ok=True,
        amp=True,
        cache=True,
        pose=12.0,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,
        fliplr=0.5,
        flipud=0.5,
        resume=resume_cen
    )
    
    best_cen = os.path.join(project_train_runs, "domo_cenital_pose", "weights", "best.pt")
    target_cen = os.path.join(models_out_dir, "yolo_cenital_pose.pt")
    if os.path.exists(best_cen):
        os.system(f"cp '{best_cen}' '{target_cen}'")
        print(f"✅ Modelo cenital guardado en {target_cen}")

    # --- Entrenar Frontal ---
    print("\n" + "="*50)
    print("ENTRENANDO MODELO FRONTAL-POSE (REMOTO T4)")
    print("="*50)
    
    if os.path.exists(last_frontal_checkpoint):
        print(f"🔄 Reanudando entrenamiento frontal desde checkpoint: {last_frontal_checkpoint}")
        model_lat = YOLO(last_frontal_checkpoint)
        resume_lat = True
    else:
        model_lat = YOLO(base_model)
        resume_lat = False

    model_lat.train(
        data=yaml_frontal,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=project_train_runs,
        name="domo_frontal_pose",
        exist_ok=True,
        amp=True,
        cache=True,
        pose=12.0,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=36.0,
        fliplr=0.5,
        flipud=0.0,
        resume=resume_lat
    )
    
    best_lat = os.path.join(project_train_runs, "domo_frontal_pose", "weights", "best.pt")
    target_lat = os.path.join(models_out_dir, "yolo_frontal_pose.pt")
    if os.path.exists(best_lat):
        os.system(f"cp '{best_lat}' '{target_lat}'")
        print(f"✅ Modelo frontal guardado en {target_lat}")

    print("\nEntrenamiento remoto finalizado con éxito.")

if __name__ == "__main__":
    main()
