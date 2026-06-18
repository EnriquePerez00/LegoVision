import os
import sys
from ultralytics import YOLO

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_cenital = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_cenital")
    dataset_frontal = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_frontal")
    
    yaml_frontal = os.path.join(dataset_frontal, "dataset.yaml")
    
    # 1. Recuperar y consolidar el modelo cenital (que llegó hasta el epoch 70 exitosamente)
    best_cen = os.path.join(project_root, "models", "train_runs", "domo_cenital_pose", "weights", "best.pt")
    target_cen = os.path.join(project_root, "models", "yolo_cenital_pose.pt")
    if os.path.exists(best_cen):
        os.system(f"cp '{best_cen}' '{target_cen}'")
        print(f"✅ Modelo cenital recuperado y guardado en {target_cen}")
    else:
        print("⚠️ No se pudo encontrar el best.pt del modelo cenital.")
        
    base_model = "yolo11n-pose.pt" # Usar Nano para inferencia super rápida
    epochs = 100
    batch_size = 32  # Volvemos a 32 ya que usaremos CPU
    workers = 12     # Maximizamos uso de CPU
    imgsz = 640
    device = "cpu"   # Forzamos CPU para eludir 100% el bug del backend MPS de Apple
    
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
        project=os.path.join(project_root, "models", "train_runs"),
        name="domo_frontal_pose",
        exist_ok=True,
        pose=1.5,
        kobj=1.0,
        box=7.5,
        cls=0.5
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
