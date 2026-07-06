import os
import sys
from ultralytics import YOLO

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # projects/camara_domo_monopieza_90
    repo_root = os.path.dirname(os.path.dirname(project_dir))  # LegoVision
    yaml_frontal = os.path.join(project_dir, "data", "frontal_dataset.yaml")
    
    # Base model
    base_model = "yolo11n-pose.pt" # Usar Nano para inferencia super rápida
    epochs = 100
    batch_size = 32
    
    # GPU / CPU Selection (MPS on Apple Silicon)
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    workers = max(1, os.cpu_count() - 2) if hasattr(os, "cpu_count") else 10
    imgsz = 1024
    
    print(f"Configuración de hardware: device={device}, workers={workers}")
    
    # Entrenar Frontal
    print("\n" + "="*50)
    print("ENTRENANDO MODELO FRONTAL-POSE")
    print("="*50)
    model_lat = YOLO(base_model)
    model_lat.train(
        data=yaml_frontal,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_dir, "models", "train_runs"),
        name="domo_frontal_pose",
        exist_ok=True,
        cache=True,
        pose=1.5,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        patience=20
    )
    
    # Exportar mejor modelo a la raíz de modelos del nuevo proyecto
    best_lat = os.path.join(project_dir, "models", "train_runs", "domo_frontal_pose", "weights", "best.pt")
    target_lat = os.path.join(project_dir, "models", "yolo_frontal_pose.pt")
    if os.path.exists(best_lat):
        os.makedirs(os.path.dirname(target_lat), exist_ok=True)
        os.system(f"cp '{best_lat}' '{target_lat}'")
        print(f"✅ Modelo frontal guardado en {target_lat}")

    print("\nEntrenamiento finalizado. El modelo frontal YOLO-Pose está listo.")

if __name__ == "__main__":
    main()
