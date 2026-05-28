# LegoVision 🧱

Sistema de **visión artificial** para detectar y clasificar piezas LEGO sobre una cinta transportadora negra, usando dataset sintético generado con Blender + LDraw y modelos YOLOv8.

## 🏗️ Arquitectura

```
LDraw Catalog (~16.500 partes)
    ↓  ldr_tools_blender (Rust)
Blender bpy Pipeline → Dataset Sintético (1000 imgs + YOLO labels)
    ↓  Lightning AI (NVIDIA T4 / FP16)
YOLOv8 Model (.pt weights)
    ↓  FastAPI Inference Server (5 FPS / 200ms latencia)
PyWebView GUI (dark mode, live view, estadísticas)
    ↕  Supabase Local (Docker :5434/:5437)
```

## 📷 Setup de Hardware

| Parámetro | Valor |
|-----------|-------|
| Cámara | Sony IMX264 (Global Shutter, 2/3") |
| Resolución | 5MP (2448 × 2048 px) |
| Lente | 12mm C-mount |
| Working Distance | 355 mm |
| FOV | 250 mm (cinta 200mm + 25mm margen) |
| Velocidad cinta | variable, máx. 5 m/min (83.3 mm/s) |
| Latencia objetivo | < 200 ms (5 FPS) |

## 🚀 Quick Start

### 1. Setup de entorno
```bash
cp .env.example .env
bash scripts/setup_env.sh
```

### 2. Descargar catálogo LDraw
```bash
bash scripts/download_ldraw.sh
```

### 3. Generar dataset sintético (test: 10 imgs)
```bash
$BLENDER_PATH --background --python blender_pipeline/generate_dataset.py -- --num_images 10
```

### 4. Lanzar la aplicación (Base de Datos + API + GUI)
Para arrancar todo el sistema de forma unificada (activación del entorno virtual, base de datos local, servidor API en background y la interfaz gráfica de usuario):
```bash
./run.sh
```
El script cerrará ordenadamente los procesos en segundo plano al salir de la aplicación gráfica.

## 📁 Estructura del Proyecto

```
LegoVision/
├── scripts/          # Scripts de setup y descarga
├── data/
│   ├── ldraw/        # Catálogo LDraw (~500MB, ver .gitignore)
│   ├── raw_dataset/  # Dataset generado por Blender
│   └── processed_dataset/  # Train/Val/Test splits
├── blender_pipeline/ # Script bpy + config hardware
├── training/         # YOLOv8 + Lightning AI
├── inference/        # FastAPI + detector
├── gui/              # PyWebView app
├── database/         # Schema Supabase
└── docs/             # Documentación técnica
```

## 🧪 Tests Locales (macOS M4)

```bash
# Test rápido Blender (10 imágenes)
bash scripts/test_local.sh

# Test inferencia M4 (MPS)
python training/local_test.py --model runs/train/best.pt --device mps
```

## ☁️ Training en Lightning AI (NVIDIA T4)

```bash
lightning run model training/train_lightning.py \
  --strategy ddp --devices 1 --accelerator gpu
```

## 🗄️ Base de Datos

- **Host**: `localhost:5434` (PostgreSQL)
- **API REST**: `localhost:5437`
- **DB**: `legvision`
- Gestión: `docker compose up/down`

## 📄 Documentación

- [Arquitectura y Flujo de Trabajo](docs/architecture_and_workflow.md) — Diagrama de componentes, flujo de datos y variables de entorno
- [Hardware Setup](docs/hardware_setup.md) — Specs Sony IMX264 + cálculos ópticos
- [LDraw Pipeline](docs/ldraw_pipeline.md) — Descarga + addon Blender
- [Training Guide](docs/training_guide.md) — Lightning AI T4
- [Latency Budget](docs/latency_budget.md) — Análisis cinta 5m/min

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|-----------|------------|
| Dataset Gen | Blender 5.1+ bpy + ldr_tools_blender |
| Training | YOLOv8 (Ultralytics) + Lightning AI |
| Inference local | PyTorch MPS (Apple M4) |
| API | FastAPI + Uvicorn + WebSocket |
| GUI | PyWebView + HTML/CSS/JS |
| DB | Supabase (PostgreSQL) en Docker |
| Versioning | Git LFS (para modelos .pt) |
