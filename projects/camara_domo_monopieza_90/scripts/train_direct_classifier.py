# -*- coding: utf-8 -*-
"""
projects/camara_domo_monopieza_90/scripts/train_direct_classifier.py
==================================================================
Script de entrenamiento para el clasificador multimodal directo (Opción A).
Carga el dataset de simulación, extrae cultivos cenitales y métricas,
y entrena el modelo LegoDirectCNNModel en la GPU/MPS del Apple Silicon M4.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from typing import List, Dict, Any, Tuple, Optional

# Configurar paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from direct_cnn_classifier import LegoDirectCNNModel, LegoDirectCNNClassifier
from _belt_mask import filter_out_belt
from rotation_aligner import align_image_by_moments

# Definir transformaciones de aumento para sim-to-real
class SpecularReflections(object):
    def __init__(self, prob=0.4, max_spots=3):
        self.prob = prob
        self.max_spots = max_spots
    def __call__(self, img):
        if torch.rand(1).item() > self.prob:
            return img
        img_np = np.array(img).copy()
        h, w = img_np.shape[:2]
        num_spots = torch.randint(1, self.max_spots + 1, (1,)).item()
        for _ in range(num_spots):
            cx = torch.randint(0, w, (1,)).item()
            cy = torch.randint(0, h, (1,)).item()
            rx = torch.randint(3, 15, (1,)).item()
            ry = torch.randint(3, 15, (1,)).item()
            cv2.ellipse(img_np, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), -1)
        return Image.fromarray(img_np)

class LinearShadow(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img):
        if torch.rand(1).item() > self.prob:
            return img
        img_np = np.array(img).copy()
        h, w = img_np.shape[:2]
        angle = torch.rand(1).item() * 360
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        grad = np.linspace(0.5, 1.0, w, dtype=np.float32)
        grad_grid = np.tile(grad, (h, 1))
        grad_rot = cv2.warpAffine(grad_grid, M, (w, h), borderValue=1.0)
        # Handle grayscale or RGB
        if len(img_np.shape) == 2:
            img_np = np.clip(img_np * grad_rot, 0, 255).astype(np.uint8)
        else:
            for c in range(3):
                img_np[:, :, c] = np.clip(img_np[:, :, c] * grad_rot, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

class LegoDirectTrainingDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        self.samples = samples
        self.transform = transform
        
        # Transformación para normalizar la imagen a Tensor
        self.to_tensor = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Cargar imagen de cultivo y aplicar aumentos si corresponden
        img = Image.open(s["crop_path"])
        
        # Convertir a grises (3 canales idénticos)
        gray_img = img.convert("L")
        img_rgb = Image.merge("RGB", (gray_img, gray_img, gray_img))
        
        if self.transform:
            x_img = self.transform(img_rgb)
        else:
            x_img = self.to_tensor(img_rgb)
            
        # 2. Vector de métricas (14 elementos)
        x_metrics = torch.tensor(s["metrics"], dtype=torch.float32)
        
        # 3. Etiqueta (Class ID)
        y = torch.tensor(s["class_label"], dtype=torch.long)
        
        return x_img, x_metrics, y

def extract_samples_from_metadata(metadata_path: str, classes: List[str], features_clf) -> List[Dict[str, Any]]:
    """
    Parsea los frames del dataset de simulación, recorta las piezas y extrae sus métricas y cultivos.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    data_dir = os.path.dirname(metadata_path)
    class_to_idx = {ref: idx for idx, ref in enumerate(classes)}
    
    samples = []
    
    # Crear carpeta temporal de crops de entrenamiento
    crops_dir = os.path.join(data_dir, "train_crops_temp")
    os.makedirs(crops_dir, exist_ok=True)
    
    total_frames = len(metadata.get("frames", []))
    print(f"[Dataset Extraction] Procesando {total_frames} frames en {data_dir}...")
    
    # Cargar modelos YOLO-Pose locales para extraer studs
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    for f_idx, frame in enumerate(metadata.get("frames", [])):
        img_cen_path = os.path.join(data_dir, frame["file_name"])
        if not os.path.exists(img_cen_path):
            continue
            
        img_cen = Image.open(img_cen_path)
        w_c, h_c = img_cen.size
        px_per_mm = float(w_c) / 196.363636
        
        for p_idx, piece in enumerate(frame.get("visible_pieces", [])):
            ref = piece["ref"]
            if ref not in class_to_idx:
                continue
                
            # Recortar cultivo
            xmin, ymin, xmax, ymax = piece["bbox_cenital_norm"]
            box_px = [int(xmin * w_c), int(ymin * h_c), int(xmax * w_c), int(ymax * h_c)]
            # Evitar crops vacíos o defectuosos en los bordes
            if (box_px[2] - box_px[0]) <= 2 or (box_px[3] - box_px[1]) <= 2:
                continue
            crop_raw = img_cen.crop(box_px)
            
            # Obtener máscara mediante chromakey rápido (azul de la cinta)
            crop_np = np.array(crop_raw)
            hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
            # Rango HSV para cinta azul petróleo (#006064)
            lower_blue = np.array([75, 40, 20])
            upper_blue = np.array([110, 255, 240])
            mask_belt = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_piece = cv2.bitwise_not(mask_belt)
            
            # Limpiar máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_piece = cv2.morphologyEx(mask_piece, cv2.MORPH_OPEN, kernel)
            
            if np.sum(mask_piece) == 0:
                continue
                
            # Guardar el crop con fondo negro
            crop_masked = cv2.bitwise_and(crop_np, crop_np, mask=mask_piece)
            crop_img = Image.fromarray(crop_masked)
            crop_filename = f"crop_{frame['frame_index']}_{p_idx}_{ref}.png"
            crop_path = os.path.join(crops_dir, crop_filename)
            crop_img.save(crop_path)
            
            # Calcular métricas tabulares
            area_cenital = float(np.sum(mask_piece)) / (px_per_mm ** 2)
            
            # Dimensiones
            contours, _ = cv2.findContours(mask_piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(cnt)
                w_box, h_box = rect[1]
                measured_length = max(w_box, h_box) / px_per_mm
                measured_width = min(w_box, h_box) / px_per_mm
            else:
                measured_length = area_cenital ** 0.5
                measured_width = area_cenital ** 0.5
                
            aspect_ratio = (measured_length / measured_width) if measured_width > 0 else 1.0
            
            # Altura (usamos el ground truth con ruido gaussiano del 5% para simular error de cámara lateral)
            gt_h = piece.get("lateral_height_gt", 3.2)
            measured_height = float(np.clip(gt_h + np.random.normal(0.0, 0.25), 1.0, 20.0))
            
            # Firma de studs (Laplacian var)
            gray_crop = cv2.cvtColor(crop_masked, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
            pixels = lap[mask_piece > 0]
            stud_signature = float(np.var(pixels)) if len(pixels) > 10 else 0.0
            
            # Características topológicas (invocar features_cenital si está disponible)
            topological_probs = np.zeros(8)
            if features_clf is not None:
                try:
                    # Preparar imagen para features classifier
                    gray_crop_pil = Image.fromarray(gray_crop)
                    from direct_cnn_classifier import LegoDirectCNNClassifier
                    proc_crop = Image.merge("RGB", (gray_crop_pil, gray_crop_pil, gray_crop_pil))
                    t_crop = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])(proc_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = features_clf(t_crop)
                        topological_probs = torch.sigmoid(logits)[0].cpu().numpy()
                except Exception as e:
                    pass
            
            # Normalizar métricas continuas
            norm_area = area_cenital / 1000.0
            norm_len = measured_length / 50.0
            norm_wid = measured_width / 50.0
            norm_ar = aspect_ratio / 5.0
            norm_height = measured_height / 20.0
            norm_stud_sig = stud_signature / 200.0
            
            metrics = [
                norm_area,
                norm_len,
                norm_wid,
                norm_ar,
                norm_height,
                norm_stud_sig
            ]
            metrics.extend(list(topological_probs))
            
            samples.append({
                "crop_path": crop_path,
                "metrics": metrics,
                "class_label": class_to_idx[ref]
            })
            
        if (f_idx + 1) % 10 == 0:
            print(f"  Procesados {f_idx + 1}/{total_frames} frames...")
            
    print(f"[Dataset Extraction] Extraídas {len(samples)} muestras válidas.")
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="data/simulation_100_75078_2D/simulation_metadata.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    metadata_path = os.path.abspath(args.metadata)
    if not os.path.exists(metadata_path):
        print(f"Error: No existe el metadato en {metadata_path}")
        sys.exit(1)
        
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Iniciando entrenamiento de Clasificador Directo (Opción A) en {device} ===")
    
    # 1. Cargar catálogo de referencias permitidas del set 75078-1
    from core.db.set_catalog import REAL_SETS
    parts = REAL_SETS.get("75078-1", {}).get("parts", [])
    classes = sorted(list(set(p["ref"] for p in parts)))
    num_classes = len(classes)
    
    # Intentar cargar clasificador de features topológicas pre-entrenado
    features_clf = None
    features_weights = os.path.join(project_root, "models", "features_cenital.pt")
    if os.path.exists(features_weights):
        try:
            import timm
            ckpt = torch.load(features_weights, map_location=device)
            model_name = ckpt.get('model_name', 'resnet18')
            features_clf = timm.create_model(model_name, num_classes=8)
            features_clf.load_state_dict(ckpt['model_state_dict'])
            features_clf.to(device)
            features_clf.eval()
            print(f"Cargado clasificador de características de {features_weights} para extracción offline.")
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el extractor de características: {e}")
            
    # 2. Extraer muestras de entrenamiento
    samples = extract_samples_from_metadata(metadata_path, classes, features_clf)
    if not samples:
        print("Error: No se pudieron extraer muestras del dataset de simulación.")
        sys.exit(1)
        
    # Split de train/val simple (80/20)
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # Transformaciones con data augmentation
    train_transform = T.Compose([
        T.Resize((224, 224)),
        SpecularReflections(prob=0.3, max_spots=2),
        LinearShadow(prob=0.4),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = LegoDirectTrainingDataset(train_samples, transform=train_transform)
    val_ds = LegoDirectTrainingDataset(val_samples, transform=None)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Cargador de datos preparado: {len(train_ds)} train, {len(val_ds)} val.")
    
    # 3. Inicializar modelo
    model = LegoDirectCNNModel(num_classes=num_classes, backbone_name='efficientnetv2_rw_s', pretrained=True)
    model.to(device)
    
    # Optimización
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    output_model_dir = os.path.join(project_root, "models")
    os.makedirs(output_model_dir, exist_ok=True)
    output_model_path = os.path.join(output_model_dir, "direct_classifier_75078.pt")
    
    # Guardar mapeo de clases para inferencia
    classes_file = output_model_path + ".classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"Guardadas {num_classes} clases en {classes_file}")
    
    # Early Stopping setup
    patience = 5
    min_delta = 0.05
    patience_counter = 0
    
    # 4. Loop de entrenamiento
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        t0 = time.time()
        for images, metrics, labels in train_loader:
            images, metrics, labels = images.to(device), metrics.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, metrics)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_train += labels.size(0)
            correct_train += preds.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        train_acc = correct_train / total_train * 100.0
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, metrics, labels in val_loader:
                images, metrics, labels = images.to(device), metrics.to(device), labels.to(device)
                outputs = model(images, metrics)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total_val += labels.size(0)
                correct_val += preds.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_ds)
        val_acc = correct_val / total_val * 100.0
        
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {epoch_loss:.4f} ({train_acc:.2f}%) | Val Loss: {epoch_val_loss:.4f} ({val_acc:.2f}%) | Time: {epoch_time:.1f}s")
        
        # Guardar mejor modelo y verificar Early Stopping
        if val_acc > (best_acc + min_delta):
            best_acc = val_acc
            torch.save(model.state_dict(), output_model_path)
            print(f"  [SAVED] Nuevo mejor modelo guardado con Val Acc: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [Early Stopping] Sin mejora significativa >= {min_delta}%. Paciencia: {patience_counter}/{patience}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_model_path)
                print(f"  [SAVED] Modelo actualizado con pequeña mejora de Val Acc: {best_acc:.2f}%")
            if patience_counter >= patience:
                print(f"  [Early Stopping] Parada temprana activada en la época {epoch+1:02d}.")
                break
            
    print(f"Entrenamiento finalizado. Mejor exactitud de validación: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
