import os
import sys
import yaml
import json
import argparse
from dotenv import load_dotenv
from ultralytics import YOLO

# Cargar configuración y variables de entorno
load_dotenv(override=True)

def generate_yolo_yaml(catalog_path, dataset_path, output_yaml):
    """Genera dinámicamente el archivo dataset.yaml para YOLOv8 en base al catálogo."""
    if not os.path.exists(catalog_path):
        print(f"[LegoVision ERROR] Catálogo no encontrado en: {catalog_path}")
        sys.exit(1)
        
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
        
    # Obtener el mapeo de índice de clase a nombre de pieza
    # catalog es un diccionario: { part_id: { "class_idx": int, "name": str } }
    class_map = {}
    for part_id, info in catalog.items():
        idx = info["class_idx"]
        name = info.get("name", part_id)
        class_map[idx] = f"Part_{part_id}_{name[:15]}" # Nombre descriptivo acotado
        
    # Ordenar por índice
    sorted_classes = [class_map[i] for i in sorted(class_map.keys())]
    
    # Crear estructura del yaml
    yaml_data = {
        'path': os.path.abspath(dataset_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(sorted_classes)}
    }
    
    # Guardar archivo YAML
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"[LegoVision Train] YAML de dataset generado correctamente en: {output_yaml}")
    print(f"  Total clases: {len(sorted_classes)}")

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de YOLOv8 para LegoVision")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=32, help="Tamaño de batch (para GPU T4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de la imagen para entrenamiento")
    parser.add_argument("--device", type=str, default="cuda", help="cuda (T4) | mps (M4) | cpu")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo base: yolov8n.pt | yolov8s.pt | yolov8m.pt")
    
    args = parser.parse_args()
    
    # Rutas base
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, "data", "ldraw", "catalog_index.json")
    dataset_path = os.path.join(project_root, "data", "processed_dataset")
    output_yaml = os.path.join(project_root, "training", "dataset.yaml")
    
    # 1. Generar el YAML del dataset dinámicamente
    generate_yolo_yaml(catalog_path, dataset_path, output_yaml)
    
    # 2. Inicializar modelo YOLOv8
    print(f"[LegoVision Train] Cargando modelo base {args.model}...")
    model = YOLO(args.model)
    
    # 3. Lanzar entrenamiento
    print(f"[LegoVision Train] Iniciando entrenamiento en dispositivo: {args.device}...")
    model.train(
        data=output_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=True,                  # FP16 Automatic Mixed Precision
        workers=4,
        project="LegoVision",
        name="yolov8_lego_run",
        exist_ok=True,
        # Aumentaciones recomendadas para simular variaciones físicas
        hsv_h=0.015,   # Hue
        hsv_s=0.7,     # Saturation
        hsv_v=0.4,     # Brightness/Value
        degrees=180.0, # Rotación completa (piezas sobre la cinta giran en cualquier ángulo)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0
    )
    
    print("[LegoVision Train] Entrenamiento completado con éxito.")
    
    # 4. Exportar el modelo entrenado
    print("[LegoVision Train] Exportando pesos...")
    # Exportar a formato ONNX para máxima velocidad de inferencia posterior
    model.export(format="onnx")

if __name__ == "__main__":
    main()
