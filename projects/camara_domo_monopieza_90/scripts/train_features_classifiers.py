#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projects/camara_domo/scripts/train_features_classifiers.py
==========================================================
Entrena clasificadores multietiqueta (Multi-Label) para detectar 8 características topológicas
en crops cenitales y laterales de piezas LEGO.
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import timm

# Configurar paths
project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, project_root)

# Las 8 características en el orden de salida
CLASSES_FEATURES = [
    "stud_solid",
    "stud_hollow",
    "technic_hole_round",
    "technic_hole_cross",
    "clip_jaw",
    "bar_handle",
    "bottom_tube",
    "bottom_pin"
]

def preprocess_crop_grayscale(crop_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """Consistente con preprocessing de inferencia."""
    gray_img = crop_img.convert("L")
    rgb_gray = Image.merge("RGB", (gray_img, gray_img, gray_img))
    
    margin = 8
    max_dim = canvas_size - 2 * margin
    w, h = rgb_gray.size
    if w > 0 and h > 0:
        scale = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = rgb_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
    return Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

import re

class LegoFeaturesDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None, pose_face_class=None):
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        self.pose_face_class = pose_face_class or {}
        self.pose_pat = re.compile(r"_pose(\d+)")
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.data_dir, item["path"])
        
        # Cargar y preprocesar a escala de grises sobre fondo negro
        try:
            with Image.open(img_path) as img:
                processed = preprocess_crop_grayscale(img, canvas_size=224)
        except Exception as e:
            # Fallback en caso de error de lectura
            processed = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            x = self.transform(processed)
        else:
            x = T.ToTensor()(processed)
            
        y = torch.tensor(item["labels"], dtype=torch.float32)
        
        # Máscara de visibilidad de características por gravedad (1.0 = visible, 0.0 = oculta)
        mask = torch.ones(8, dtype=torch.float32)
        ldraw_id = item.get("ldraw_id")
        
        m = self.pose_pat.search(item["path"])
        if m and ldraw_id:
            pose_idx = int(m.group(1))
            face_class = self.pose_face_class.get((ldraw_id, pose_idx))
            if face_class == "Bottom":
                # bottom_tube y bottom_pin están ocluidos al estar apoyados sobre la base
                mask[6] = 0.0
                mask[7] = 0.0
            elif face_class == "Top":
                # stud_solid y stud_hollow están ocluidos al estar invertidos
                mask[0] = 0.0
                mask[1] = 0.0
                
        return x, y, mask

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss
        return loss

class SpecularReflections(object):
    def __init__(self, prob=0.4, max_spots=3):
        self.prob = prob
        self.max_spots = max_spots
    def __call__(self, img):
        if torch.rand(1).item() > self.prob:
            return img
        import numpy as np
        import cv2
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
        import numpy as np
        import cv2
        img_np = np.array(img).copy()
        h, w = img_np.shape[:2]
        angle = torch.rand(1).item() * 360
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        grad = np.linspace(0.4, 1.0, w, dtype=np.float32)
        grad_grid = np.tile(grad, (h, 1))
        grad_rot = cv2.warpAffine(grad_grid, M, (w, h), borderValue=1.0)
        for c in range(3):
            img_np[:, :, c] = np.clip(img_np[:, :, c] * grad_rot, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_classifier(metadata_path, output_model_path, epochs=10, batch_size=32, model_name="resnet18"):
    device = get_device()
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando entrenamiento en: {device}")
    
    # Cargar clases de caras para cada postura estable desde DB
    pose_face_class = {}
    try:
        from core.db.supabase_client import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT part_ref, pose_index, face_class FROM stable_poses;")
                for row in cur.fetchall():
                    pose_face_class[(row["part_ref"], row["pose_index"])] = row["face_class"]
        print(f"Cargadas {len(pose_face_class)} clases de caras desde la DB.")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar stable_poses desde la DB (usando máscaras por defecto): {e}")
        
    # Cargar metadatos del dataset
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    data_dir = os.path.dirname(metadata_path)
    
    # Transformaciones con aumento de datos básico y avanzado sim-to-real
    train_transform = T.Compose([
        SpecularReflections(prob=0.4, max_spots=2),
        LinearShadow(prob=0.5),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    from core.db.set_catalog import REAL_SETS
    allowed_refs = {p["ref"] for p in REAL_SETS["75078-1"]["parts"]}
    train_filtered = [item for item in metadata["train"] if item["ldraw_id"] in allowed_refs]
    val_filtered = [item for item in metadata["val"] if item["ldraw_id"] in allowed_refs]
    
    train_dataset = LegoFeaturesDataset(train_filtered, data_dir, transform=train_transform, pose_face_class=pose_face_class)
    val_dataset = LegoFeaturesDataset(val_filtered, data_dir, transform=val_transform, pose_face_class=pose_face_class)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Dataset cargado: {len(train_dataset)} train, {len(val_dataset)} val.")
    
    # Crear modelo con salida de 8 clases para clasificación multietiqueta
    print(f"Creando modelo '{model_name}'...")
    model = timm.create_model(model_name, pretrained=True, num_classes=len(CLASSES_FEATURES))
    model = model.to(device)
    
    # Pérdida Focal enmascarada
    criterion = BCEFocalLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        
        for inputs, labels, masks in train_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Pérdida enmascarada por oclusión
            loss_raw = criterion(outputs, labels)
            loss = (loss_raw * masks).sum() / (masks.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct_features = [0] * len(CLASSES_FEATURES)
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels, masks in val_loader:
                inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                outputs = model(inputs)
                
                loss_raw = criterion(outputs, labels)
                loss = (loss_raw * masks).sum() / (masks.sum() + 1e-8)
                val_loss += loss.item() * inputs.size(0)
                
                # Calcular precisión por característica (threshold = 0.5)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                total_samples += labels.size(0)
                for f_idx in range(len(CLASSES_FEATURES)):
                    correct_features[f_idx] += preds[:, f_idx].eq(labels[:, f_idx]).sum().item()
                    
        epoch_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(epoch_val_loss)
        
        # Calcular precisión media de características
        accs = [correct_features[i] / total_samples for i in range(len(CLASSES_FEATURES))]
        mean_acc = sum(accs) / len(accs)
        
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Mean Feat Acc: {mean_acc*100:.2f}% | Time: {epoch_time:.1f}s")
        
        # Guardar mejor modelo y chequear Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'classes_features': CLASSES_FEATURES
            }, output_model_path)
            print(f"  [SAVED] Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  [Early Stopping] Sin mejora. Paciencia: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"  [Early Stopping] Parada temprana activada en época {epoch+1}.")
                break
            
    print(f"Entrenamiento completado. Mejor Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=str, required=True, choices=["cenital", "lateral"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="resnet18")
    args = parser.parse_args()
    
    metadata_filename = f"features_{args.view}_metadata.json"
    metadata_path = os.path.join(project_root, "projects", "camara_domo", "data", metadata_filename)
    output_model_path = os.path.join(project_root, "projects", "camara_domo_75078", "models", f"features_{args.view}.pt")
    
    print(f"=== Entrenando modelo de Características {args.view.upper()} ===")
    train_classifier(metadata_path, output_model_path, epochs=args.epochs, batch_size=args.batch_size, model_name=args.model)
