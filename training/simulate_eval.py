import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Añadir directorio raíz al path para importar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.detector import LegoDetector
from inference.classifier import LegoClassifier

def calculate_iou(box1, box2):
    """Calcula Intersection over Union (IoU) entre dos bounding boxes en formato [x_center, y_center, width, height]"""
    # Convertir a [x1, y1, x2, y2]
    w1, h1 = box1[2], box1[3]
    x1_min, y1_min = box1[0] - w1/2, box1[1] - h1/2
    x1_max, y1_max = box1[0] + w1/2, box1[1] + h1/2
    
    w2, h2 = box2[2], box2[3]
    x2_min, y2_min = box2[0] - w2/2, box2[1] - h2/2
    x2_max, y2_max = box2[0] + w2/2, box2[1] + h2/2
    
    # Intersección
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    # Unión
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def main():
    print("========================================================")
    # 1. Rutas
    model_path = os.path.join(project_root, "models", "best.pt")
    test_dir = os.path.join(project_root, "data", "test_dataset")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo YOLO en {model_path}. Entrénalo primero.")
        sys.exit(1)
        
    print(f"Cargando detector YOLOv8 desde: {model_path}")
    detector = LegoDetector(model_path=model_path, conf_threshold=0.25)
    
    print("Cargando clasificador DINOv2...")
    classifier = LegoClassifier(top_k=3)
    classifier.load_model()
    classifier.load_reference_embeddings()
    
    if not classifier.is_ready():
        print("[ERROR] Los embeddings de DINOv2 no están listos. Ejecuta primero la indexación.")
        sys.exit(1)
        
    # Buscar imágenes de test
    test_images_dir = os.path.join(test_dir, "images")
    test_labels_dir = os.path.join(test_dir, "labels")
    
    if not os.path.exists(test_images_dir):
        print(f"[ERROR] No existe el dataset de test en {test_images_dir}.")
        print("Sugerencia: Genera imágenes de test ejecutando generate_dataset.py con salida en test_dataset.")
        sys.exit(1)
        
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(".png")]
    print(f"Encontradas {len(image_files)} imágenes de test para simulación.")
    
    correct_count = 0
    total_detections = 0
    detector_misses = 0
    confusions = {}
    correct_by_part = {}
    total_by_part = {}
    
    for img_name in tqdm(image_files, desc="Evaluando simulación"):
        img_path = os.path.join(test_images_dir, img_name)
        meta_path = os.path.join(test_labels_dir, img_name.replace(".png", ".json"))
        
        if not os.path.exists(meta_path):
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        # Cargar imagen y detectar
        img = Image.open(img_path).convert("RGB")
        detections = detector.detect(img)
        
        # Filtrar bounding boxes gigantes corruptas [0.5, 0.5, 1.0, 1.0]
        gt_list = [
            g for g in meta["detections"]
            if not (abs(g["bbox"][0] - 0.5) < 1e-3 and 
                    abs(g["bbox"][1] - 0.5) < 1e-3 and 
                    abs(g["bbox"][2] - 1.0) < 1e-3 and 
                    abs(g["bbox"][3] - 1.0) < 1e-3)
        ]
        
        # DEBUG: Imprimir detalles para los primeros 3 frames
        if img_name in image_files[:3]:
            print(f"\n[DEBUG] Imagen: {img_name}")
            print(f"  Detecciones YOLO ({len(detections)}): {[d['bbox'] for d in detections]}")
            print(f"  Ground Truth ({len(gt_list)}): {[g['bbox'] for g in gt_list]}")
        
        # Emparejar cada detección YOLO con el Ground Truth usando IoU
        matched_gt = set()
        
        for det in detections:
            # bbox en det ya está en formato normalized [x_center, y_center, width, height]
            det_norm = det["bbox"]
            
            # Buscar mejor IoU
            best_iou = 0.0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_list):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(det_norm, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if img_name in image_files[:3]:
                print(f"  Detección {det_norm} -> Mejor GT index: {best_gt_idx} (IoU: {best_iou:.3f})")
            
            if best_gt_idx != -1 and best_iou > 0.25: # Umbral de solapamiento razonable
                matched_gt.add(best_gt_idx)
                gt_part_id = gt_list[best_gt_idx]["part_id"]
                
                # Clasificar crop
                w_img, h_img = img.size
                x_c, y_c, w_b, h_b = det_norm
                x1 = int((x_c - w_b / 2) * w_img)
                y1 = int((y_c - h_b / 2) * h_img)
                x2 = int((x_c + w_b / 2) * w_img)
                y2 = int((y_c + h_b / 2) * h_img)
                
                # Asegurar límites correctos
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                crop = img.crop((x1, y1, x2, y2))
                clf_results = classifier.classify(crop, filter_by_color=True)
                
                if clf_results:
                    pred_part_id = clf_results[0]["part_ref"]
                    is_correct = (pred_part_id == gt_part_id)
                    
                    total_detections += 1
                    total_by_part[gt_part_id] = total_by_part.get(gt_part_id, 0) + 1
                    
                    if is_correct:
                        correct_count += 1
                        correct_by_part[gt_part_id] = correct_by_part.get(gt_part_id, 0) + 1
                    else:
                        confusions[(gt_part_id, pred_part_id)] = confusions.get((gt_part_id, pred_part_id), 0) + 1
            else:
                # Detección falsa de YOLO (o sin ground truth emparejado)
                pass
                
        # Detecciones perdidas por YOLO (False Negatives)
        detector_misses += (len(gt_list) - len(matched_gt))
        
    print("\n========================================================")
    print("           REPORTE DE EVALUACIÓN DE SIMULACIÓN          ")
    print("========================================================")
    if total_detections > 0:
        overall_acc = (correct_count / total_detections) * 100
        print(f"Imágenes evaluadas:       {len(image_files)}")
        print(f"Total piezas detectadas:  {total_detections}")
        print(f"Clasificación correcta:   {correct_count} ({overall_acc:.2f}%)")
        print(f"Piezas omitidas por YOLO: {detector_misses}")
        
        print("\n--- Precisión por Tipo de Pieza (Top 5 con datos) ---")
        sorted_parts = sorted(total_by_part.items(), key=lambda x: x[1], reverse=True)
        for part_id, total in sorted_parts[:10]:
            correct = correct_by_part.get(part_id, 0)
            acc = (correct / total) * 100
            print(f"  Pieza {part_id:8s}: {correct:2d}/{total:2d} correctas ({acc:.1f}%)")
            
        print("\n--- Confusiones más comunes ---")
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        for (gt, pred), count in sorted_confusions[:8]:
            print(f"  Pieza {gt} confundida con {pred}: {count} veces")
    else:
        print("No se realizaron detecciones coincidentes con el ground truth.")
        
    print("========================================================")

if __name__ == "__main__":
    main()
