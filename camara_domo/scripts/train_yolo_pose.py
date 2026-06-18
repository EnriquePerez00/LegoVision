import os
import sys

# Optimización para Apple Silicon M4
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_cenital = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_cenital")
    dataset_frontal = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_frontal")
    
    # Nos aseguramos de que existan las carpetas de dataset
    os.makedirs(dataset_cenital, exist_ok=True)
    os.makedirs(dataset_frontal, exist_ok=True)
        
    print("=== Generando YAMLs (Class-Agnostic) ===")
    yaml_cenital = create_yaml(dataset_cenital)
    yaml_frontal = create_yaml(dataset_frontal)
    
    base_model = "yolo11s-pose.pt" # Subimos a modelo Small para mayor capacidad espacial en M4
    epochs = 50                    # 50 épocas es óptimo para convergencia de clase única local
    batch_size = 16                # Batch estable para MPS
    workers = 2                    # ¡CRÍTICO! Evita cuellos de botella de transferencia en macOS
    imgsz = 640
    device = "mps"                 # MPS para Apple Silicon M4 GPU
    
    # Entrenar Cenital
    print("\n" + "="*50)
    print("ENTRENANDO MODELO CENITAL-POSE (CLASS-AGNOSTIC)")
    print("="*50)
    model_cen = YOLO(base_model)
    model_cen.train(
        data=yaml_cenital,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_root, "models", "train_runs"),
        name="domo_cenital_pose",
        exist_ok=True,
        amp=False,                  # ¡CRÍTICO! Desactivar en MPS para estabilidad y evitar NaNs
        cache=True,                 # Cachear en RAM (óptimo con memoria unificada M4)
        pose=12.0,                  # ¡CRÍTICO! Subir pérdida de keypoints a 12.0 (default)
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,              # Aumentación de rotación completa para la cinta
        fliplr=0.5,
        flipud=0.5
    )
    
    # Exportar mejor modelo a la raíz
    best_cen = os.path.join(project_root, "models", "train_runs", "domo_cenital_pose", "weights", "best.pt")
    target_cen = os.path.join(project_root, "models", "yolo_cenital_pose.pt")
    if os.path.exists(best_cen):
        os.system(f"cp '{best_cen}' '{target_cen}'")
        print(f"✅ Modelo cenital guardado en {target_cen}")

    # Entrenar Frontal
    print("\n" + "="*50)
    print("ENTRENANDO MODELO FRONTAL-POSE (CLASS-AGNOSTIC)")
    print("="*50)
    model_lat = YOLO(base_model)
    model_lat.train(
        data=yaml_frontal,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_root, "models", "train_runs"),
        name="domo_frontal_pose",
        exist_ok=True,
        amp=False,                  # ¡CRÍTICO! Desactivar en MPS para estabilidad y evitar NaNs
        cache=True,                 # Cachear en RAM (óptimo con memoria unificada M4)
        pose=12.0,                  # ¡CRÍTICO! Subir pérdida de keypoints a 12.0 (default)
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,              # Aumentación de rotación completa para la cinta
        fliplr=0.5,
        flipud=0.5
    )
    
    # Exportar mejor modelo a la raíz
    best_lat = os.path.join(project_root, "models", "train_runs", "domo_frontal_pose", "weights", "best.pt")
    target_lat = os.path.join(project_root, "models", "yolo_frontal_pose.pt")
    if os.path.exists(best_lat):
        os.system(f"cp '{best_lat}' '{target_lat}'")
        print(f"✅ Modelo frontal guardado en {target_lat}")

    print("\nEntrenamiento finalizado. Los modelos consolidados YOLO-Pose están listos.")

if __name__ == "__main__":
    main()
