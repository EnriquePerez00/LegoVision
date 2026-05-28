import os
import sys
from ultralytics import YOLO

def main():
    print("=========================================================")
    print("LegoVision — Test Local de Deep Learning (macOS MPS)")
    print("=========================================================")
    
    # 1. Cargar modelo base YOLO11 nano
    print("Cargando modelo nano yolo11n.pt...")
    model = YOLO("yolo11n.pt")
    
    # 2. Verificar disponibilidad de MPS
    import torch
    print(f"Versión de PyTorch: {torch.__version__}")
    
    mps_available = torch.backends.mps.is_available()
    print(f"¿Aceleración Metal (MPS) disponible?: {mps_available}")
    
    # 3. Mostrar estructura de tests recomendada
    print("\nPara entrenar localmente (prueba rápida):")
    print("  python training/train_lightning.py --epochs 2 --batch 2 --imgsz 640 --device mps")
    print("\nPara producción (Lightning AI - T4 GPU):")
    print("  python training/train_lightning.py --epochs 100 --batch 32 --imgsz 640 --device cuda")
    print("=========================================================")

if __name__ == "__main__":
    main()
