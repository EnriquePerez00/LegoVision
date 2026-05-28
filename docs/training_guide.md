# LegoVision — Guía de Entrenamiento en Lightning AI (NVIDIA T4)

Esta guía detalla el procedimiento para entrenar el modelo YOLOv8 utilizando una máquina virtual con GPU NVIDIA T4 en la plataforma **Lightning AI**.

## 1. Requisitos Previos en Lightning AI
1. Crear un espacio de trabajo (Studio) seleccionando hardware de tipo **GPU (T4)** o superior.
2. Asegurar que los puertos de internet estén abiertos para la descarga de dependencias.

---

## 2. Preparación del Entorno
Instalar las dependencias de GPU requeridas por el pipeline ejecutando:
```bash
pip install -r training/requirements_gpu.txt
```

Para asegurar aceleración de GPU CUDA con PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Estructura del Dataset
Sube tu dataset sintético comprimido o copiado al directorio `/workspace/data/processed_dataset/` con la siguiente estructura:
```
processed_dataset/
├── train/
│   ├── images/  (PNGs generados)
│   └── labels/  (TXTs con formato YOLO)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 4. Ejecución del Entrenamiento
Lanza el script de entrenamiento de la siguiente forma:
```bash
python training/train_lightning.py --epochs 100 --batch 32 --imgsz 640 --device cuda
```

El script se encargará de:
1. Leer `catalog_index.json` y generar el archivo `dataset.yaml` dinámicamente mapeando los índices de clase correctos.
2. Iniciar el entrenamiento con Mixed Precision (FP16) activado para optimizar el rendimiento de la GPU T4.
3. Aplicar aumentaciones espaciales y de brillo (rotación de 180°, cambios de brillo/saturación) para mejorar la robustez de las detecciones en la cinta transportadora.
4. Exportar el modelo final en formato `best.pt` y exportarlo a formato ONNX para una inferencia local de alta velocidad.
