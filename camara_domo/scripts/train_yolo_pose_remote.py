import os
import sys

from ultralytics import YOLO

def create_yaml(dataset_path, classes_file=None):
    # Generar un YAML class-agnostic (1 sola clase)
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val

# Keypoints format
kpt_shape: [9, 3]

names:
  0: 'lego_piece'
"""
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    return yaml_path

def main():
    # Asumimos ejecución remota en Lightning AI (directorio actual será el root de LegoVision)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dataset_cenital = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_cenital")
    dataset_frontal = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_frontal")
    
    # Asegurarnos de que existan las carpetas de dataset
    os.makedirs(dataset_cenital, exist_ok=True)
    os.makedirs(dataset_frontal, exist_ok=True)
        
    print("=== Generando YAMLs (Class-Agnostic) ===")
    yaml_cenital = create_yaml(dataset_cenital)
    yaml_frontal = create_yaml(dataset_frontal)
    
    # Parámetros unificados para hardware remoto (NVIDIA T4)
    base_model = "yolo11s-pose.pt" # Modelo Small unificado
    epochs = 100                   # Unificado a 100 épocas (Early stopping se encargará si convergen antes)
    batch_size = 16                # Reducido a 16 para evitar CUDA Out Of Memory en T4 (16GB)
    workers = 4                    # O 8 si la instancia se actualiza a 8 vCPUs
    imgsz = 640
    device = "cuda"                # Uso explícito de GPU
    
    # --- Entrenar Cenital ---
    print("\n" + "="*50)
    print("ENTRENANDO MODELO CENITAL-POSE (REMOTO T4)")
    print("="*50)
    model_cen = YOLO(base_model)
    model_cen.train(
        data=yaml_cenital,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_root, "camara_domo", "models", "train_runs"),
        name="domo_cenital_pose",
        exist_ok=True,
        amp=True,                   # AMP Activado para aceleración extrema en Tensor Cores
        cache=True,                 # RAM Cache
        pose=12.0,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,
        fliplr=0.5,
        flipud=0.5
    )
    
    best_cen = os.path.join(project_root, "camara_domo", "models", "train_runs", "domo_cenital_pose", "weights", "best.pt")
    target_cen = os.path.join(project_root, "camara_domo", "models", "yolo_cenital_pose.pt")
    if os.path.exists(best_cen):
        os.system(f"cp '{best_cen}' '{target_cen}'")
        print(f"✅ Modelo cenital guardado en {target_cen}")

    # --- Entrenar Frontal ---
    print("\n" + "="*50)
    print("ENTRENANDO MODELO FRONTAL-POSE (REMOTO T4)")
    print("="*50)
    model_lat = YOLO(base_model)
    model_lat.train(
        data=yaml_frontal,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_root, "camara_domo", "models", "train_runs"),
        name="domo_frontal_pose",
        exist_ok=True,
        amp=True,                   # AMP Activado
        cache=True,
        pose=12.0,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=36.0,               # Ajustado de 360 a 36 para la cámara frontal (según original o heurística)
        fliplr=0.5,
        flipud=0.0
    )
    
    best_lat = os.path.join(project_root, "camara_domo", "models", "train_runs", "domo_frontal_pose", "weights", "best.pt")
    target_lat = os.path.join(project_root, "camara_domo", "models", "yolo_frontal_pose.pt")
    if os.path.exists(best_lat):
        os.system(f"cp '{best_lat}' '{target_lat}'")
        print(f"✅ Modelo frontal guardado en {target_lat}")

    print("\nEntrenamiento remoto finalizado con éxito.")

if __name__ == "__main__":
    main()
