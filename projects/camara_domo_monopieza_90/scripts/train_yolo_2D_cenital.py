# -*- coding: utf-8 -*-
"""train_yolo_2D_cenital.py
==========================
Entrena un nuevo modelo YOLO-Pose cenital class-agnostic optimizado para el escenario 2D,
partiendo del modelo cenital actual y guardándolo como 'yolo_2D_cenital_pose.pt'.
"""
import os
import sys
from ultralytics import YOLO

# Optimización para Apple Silicon M4
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def create_yaml(dataset_path):
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
    project_root = "/Users/I764690/Code_personal/LegoVision"
    project_models = os.path.join(project_root, "projects", "camara_domo_monopieza_90", "models")
    dataset_cenital = os.path.join(project_root, "projects", "camara_domo_monopieza_90", "data", "yolo_dataset_2D", "yolo_dataset_cenital")
    
    if not os.path.exists(dataset_cenital):
        print(f"[ERROR] No existe el dataset cenital en {dataset_cenital}")
        sys.exit(1)
        
    print("=== Generando YAML (Class-Agnostic) ===")
    yaml_cenital = create_yaml(dataset_cenital)
    
    # Base model: partimos del modelo cenital actual para transferencia de aprendizaje (fine-tuning)
    base_model = os.path.join(project_models, "yolo_cenital_pose.pt")
    if not os.path.exists(base_model):
        print(f"[WARN] No se encontró el modelo base en {base_model}. Usando modelo genérico yolo11s-pose.pt")
        base_model = "yolo11s-pose.pt"
    else:
        print(f"Usando modelo base actual para transferencia: {base_model}")
        
    epochs = 20                    # 20 épocas es óptimo y rápido para fine-tuning en M4
    batch_size = 16                # Batch estable para MPS
    workers = 2                    # Evita cuellos de botella de transferencia en macOS
    imgsz = 1024
    device = "mps"                 # MPS para Apple Silicon M4 GPU
    
    print("\n" + "="*50)
    print("ENTRENANDO MODELO CENITAL 2D (yolo_2D_cenital_pose.pt)")
    print("="*50)
    
    model = YOLO(base_model)
    model.train(
        data=yaml_cenital,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_models, "train_runs"),
        name="domo_2D_cenital_pose",
        exist_ok=True,
        amp=False,                  # Desactivar en MPS para estabilidad y evitar NaNs
        cache=True,                 # Cachear en RAM (óptimo con memoria unificada M4)
        pose=12.0,                  # Pérdida de keypoints
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,              # Aumentación de rotación completa para la cinta
        fliplr=0.5,
        flipud=0.5,
        patience=8
    )
    
    # Copiar el mejor modelo al directorio de modelos canónicos
    best_weights = os.path.join(project_models, "train_runs", "domo_2D_cenital_pose", "weights", "best.pt")
    target_weights = os.path.join(project_models, "yolo_2D_cenital_pose.pt")
    
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\n✅ Nuevo modelo 2D guardado en {target_weights}")
    else:
        print(f"[ERROR] No se pudo encontrar el archivo de pesos entrenado en {best_weights}")
        sys.exit(1)


if __name__ == "__main__":
    main()
