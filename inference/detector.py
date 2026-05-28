import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class LegoDetector:
    def __init__(self, model_path: str = None, device: str = None, conf_threshold: float = 0.5):
        """
        Wrapper para inferencia con YOLOv8.
        """
        # Determinar dispositivo de aceleración
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Determinar modelo
        if model_path is None or not os.path.exists(model_path):
            print(f"[LegoVision Detector] Pesos del modelo no encontrados en: {model_path}. Cargando yolo11n.pt por defecto.")
            self.model_path = "yolo11n.pt"
        else:
            self.model_path = model_path
            
        print(f"[LegoVision Detector] Inicializando YOLO11 en dispositivo: {self.device}")
        self.model = YOLO(self.model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image: Image.Image, conf: float = None) -> list[dict]:
        """
        Ejecuta inferencia sobre una imagen PIL y retorna la lista de detecciones.
        Detecciones: list[dict] con {"class": str, "name": str, "confidence": float, "bbox": [x, y, w, h]}
        """
        conf_val = conf if conf is not None else self.conf_threshold
        
        # Inferencia
        results = self.model(image, device=self.device, conf=conf_val, verbose=False)
        
        detections = []
        if len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        names = result.names
        
        for box in boxes:
            # Coordenadas de la caja (normalizadas para YOLO)
            # xywhn es: [x_centro, y_centro, ancho, alto] normalizados (0-1)
            xywhn = box.xywhn[0].tolist()
            cls_idx = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            
            detections.append({
                "class": str(cls_idx),
                "name": names[cls_idx],
                "confidence": confidence,
                "bbox": xywhn
            })
            
        return detections
