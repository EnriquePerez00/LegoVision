# -*- coding: utf-8 -*-
"""
projects/camara_domo_monopieza_90/scripts/direct_cnn_classifier.py
==================================================================
Clasificador directo multimodal para la Opción A del set LEGO 75078-1.
Combina características visuales extraídas de cultivos cenitales y métricas geométricas/físicas.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

# Definir la estructura de la red multimodal
class LegoDirectCNNModel(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = 'efficientnetv2_rw_s', pretrained: bool = True):
        super().__init__()
        import timm
        
        # 1. Rama Visual (Backbone para cultivos cenitales)
        # La red timm se carga sin clasificador lineal (num_classes=0) para extraer embeddings
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        # Obtener la dimensión del embedding visual
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.visual_emb_dim = self.backbone(dummy).shape[-1]
            
        # Proyectar el embedding de imagen para reducir su dimensionalidad y equilibrar con las métricas
        self.image_projector = nn.Sequential(
            nn.Linear(self.visual_emb_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # 2. Rama de Métricas Tabulares (MLP)
        # Recibe 14 métricas normalizadas:
        #   [area_cenital, length, width, aspect_ratio, height_frontal, stud_signature, 8 topol_features]
        self.metrics_branch = nn.Sequential(
            nn.Linear(14, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # 3. Cabeza de Fusión y Clasificación
        # Fusión mediante concatenación: 128 (imagen) + 32 (métricas) = 160
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image: torch.Tensor, metrics: torch.Tensor) -> torch.Tensor:
        # Extraer vector visual
        img_feats = self.backbone(image)
        img_proj = self.image_projector(img_feats)
        
        # Extraer vector de métricas
        met_feats = self.metrics_branch(metrics)
        
        # Fusionar y clasificar
        fused = torch.cat([img_proj, met_feats], dim=1)
        logits = self.classifier(fused)
        return logits


class LegoDirectCNNClassifier:
    """Clasificador directo que envuelve a LegoDirectCNNModel e interactúa con el pipeline."""
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.weights_path = weights_path
        self.classes = self._load_classes(weights_path + ".classes.txt")
        self.num_classes = len(self.classes)
        
        # Inicializar modelo
        self.model = LegoDirectCNNModel(num_classes=self.num_classes, pretrained=False)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[Direct CNN Classifier] Cargado modelo de clasificación de {self.weights_path}")
        else:
            print(f"[Direct CNN Classifier Warning] No se encontraron pesos del clasificador en {self.weights_path}. Se usará inicialización aleatoria.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Transformación consistente de imagen cenital
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Reglas físicas simplificadas y robustas para las piezas del set 75078-1
        # Formato: ref -> (max_height_mm, has_studs_bool)
        self.part_rules = {
            "14769": (4.5, False),     # Tile Round 2x2 (No studs)
            "15391": (9.5, False),     # Minifigure Gun Blaster
            "15392": (7.0, False),     # Trigger for gun
            "2335": (5.0, False),      # Flag 2x2
            "2412b": (4.2, False),     # Tile 1x2 Grille (No studs)
            "2419": (4.5, True),       # Wedge Plate 3x6
            "2445": (4.5, True),       # Plate 2x12
            "2449": (32.0, False),     # Slope Inverted 2x1x3 (Very tall)
            "2540": (4.8, True),       # Plate 1x2 with Bar
            "2653": (11.5, True),      # Brick 1x4 with Channel
            "2654": (4.8, False),      # Boat stud round (No studs)
            "2877": (11.5, True),      # Brick 1x2 with Grille
            "3004": (11.5, True),      # Brick 1x2
            "3020": (4.5, True),       # Plate 2x4
            "3022": (4.5, True),       # Plate 2x2
            "3023": (4.5, True),       # Plate 1x2
            "3024": (4.5, True),       # Plate 1x1
            "3040": (11.5, False),     # Slope 45 2x1
            "30414": (11.5, True),     # Brick 1x4 with studs on side
            "3068": (4.5, False),      # Tile 2x2
            "32000": (11.5, True),     # Technic Brick 1x2 with Holes
            "32054": (9.0, False),     # Technic Pin 3L
            "3679": (4.8, True),       # Turntable top (has studs)
            "3680": (4.8, False),      # Turntable base (no studs)
            "3710": (4.5, True),       # Plate 1x4
            "3795": (4.5, True),       # Plate 2x6
            "3832": (4.5, True),       # Plate 2x10
            "3839b": (5.2, True),      # Plate 1x2 with handles
            "4073": (4.8, True),       # Plate Round 1x1
            "4589b": (13.0, False),    # Cone 1x1 (Can be 11.6mm standing or ~6mm lying, no studs)
            "51739": (4.5, True),       # Wedge Plate 2x4
            "60481": (18.0, False),    # Slope 65 2x1x2
            "61184": (9.0, False),     # Flick missile pin
            "61780": (22.0, True),     # Container Box 2x2x2 (Very tall)
            "6541": (11.5, True),      # Technic Brick 1x1 with Hole
            "85984": (8.5, False),     # Slope 30 1x2x2/3
            "87552": (21.0, True),     # Panel 1x2x2
            "87620": (11.5, False),    # Brick Modified Facet
        }

    def _load_classes(self, classes_path: str) -> List[str]:
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                return f.read().splitlines()
        # Fallback de las clases estimadas del set 75078-1 si no existe el fichero
        from core.db.set_catalog import REAL_SETS
        parts = REAL_SETS.get("75078-1", {}).get("parts", [])
        classes = sorted(list(set(p["ref"] for p in parts)))
        return classes

    def preprocess_crop(self, crop: Image.Image) -> Image.Image:
        """Convierte cultivo a escala de grises y lo duplica en 3 canales para compatibilidad."""
        gray_img = crop.convert("L")
        return Image.merge("RGB", (gray_img, gray_img, gray_img))

    def build_metrics_vector(self, 
                             area_cenital: float, 
                             measured_length: float, 
                             measured_width: float, 
                             measured_height: float, 
                             stud_signature: float, 
                             topological_probs: np.ndarray) -> torch.Tensor:
        """
        Construye y normaliza el vector continuo de 14 métricas.
        """
        # Normalizaciones para asegurar estabilidad numérica
        norm_area = area_cenital / 1000.0
        norm_len = measured_length / 50.0
        norm_wid = measured_width / 50.0
        aspect_ratio = (measured_length / measured_width) if measured_width > 0 else 1.0
        norm_ar = aspect_ratio / 5.0
        norm_height = measured_height / 20.0
        norm_stud_sig = (stud_signature / 200.0) if stud_signature >= 0 else 0.0
        
        vec = [
            norm_area,
            norm_len,
            norm_wid,
            norm_ar,
            norm_height,
            norm_stud_sig
        ]
        
        # Agregar las 8 probabilidades de características topológicas predichas
        if topological_probs is not None and len(topological_probs) == 8:
            vec.extend(list(topological_probs))
        else:
            vec.extend([0.0] * 8)
            
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, 
                crop_cen: Image.Image, 
                area_cenital: float, 
                measured_length: float, 
                measured_width: float, 
                measured_height: float, 
                stud_signature: float, 
                topological_probs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Realiza la predicción del modelo híbrido devolviendo una lista ordenada y filtrada por reglas físicas específicas.
        """
        # 1. Preprocesar y transformar imagen cenital
        processed_img = self.preprocess_crop(crop_cen)
        img_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        
        # 2. Generar vector de métricas
        metrics_tensor = self.build_metrics_vector(
            area_cenital=area_cenital,
            measured_length=measured_length,
            measured_width=measured_width,
            measured_height=measured_height,
            stud_signature=stud_signature,
            topological_probs=topological_probs
        )
        
        # 3. Inferencia
        logits = self.model(img_tensor, metrics_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # 4. Construir ranking aplicando reglas de gating específicas y seguras (Evita falsos negativos)
        results = []
        for idx, ref in enumerate(self.classes):
            score = float(probs[idx])
            
            # Regla de Studs específica para el par en conflicto Tile/Plate 1x2 y similares
            if ref == "3023" and stud_signature < 12.0:
                score *= 0.01  # Penalizar la placa si la firma de studs es nula
            elif ref == "2540" and stud_signature < 12.0 and measured_height < 5.0:
                score *= 0.01  # Solo penalizar si la pieza está plana y no reporta studs
            elif ref == "2412b" and stud_signature > 35.0:
                score *= 0.01  # Penalizar la rejilla lisa si tiene alta firma de studs
                
            # Regla de Área específica para evitar clasificar piezas diminutas como pistola Blaster (15391)
            if ref == "15391" and area_cenital < 78.0:
                score *= 0.005  # Una pistola de 17mm x 7mm nunca mide menos de 78mm2

            # Regla de Área para evitar que una pieza pequeña/mediana clasifique como pieza gigante por descarte
            large_parts = ["3832", "2445", "3795", "3020", "2419", "61780", "2653", "30414"]
            if ref in large_parts and area_cenital < 250.0:
                score *= 0.001  # Las piezas grandes/gigantes nunca miden menos de 250 mm2 aparente
            
            results.append({
                "part_ref": ref,
                "score": score,
                "pose_index": 0
            })
            
        # Ordenar por score filtrado
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Re-normalizar
        total_score = sum(x["score"] for x in results)
        if total_score > 0:
            for x in results:
                x["score"] /= total_score
                
        return results
