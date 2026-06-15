# Estructura de Directorios — 2camaras_random_pieza_unica

Este documento describe la organización de directorios del subproyecto, siguiendo el principio de **separación de dominios**.

## 📁 Estructura Principal

```
2camaras_random_pieza_unica/
├── scripts/                   # Solo código fuente (versionado en git)
├── training/                  # Scripts de entrenamiento YOLO
├── config.yaml                # Configuración centralizada
├── config_loader.py           # Cargador de config
├── logger.py                  # Logger centralizado
│
├── data/                      # 📥 DATOS DE ENTRADA (solo lectura)
│   ├── canonical_keypoints.json
│   ├── stable_poses_cache.json
│   ├── color_calibration_palette.json
│   ├── ldraw/                 # Partes LDraw locales
│   └── tmp/                   # Temporales de simulación
│
├── renders/                   # 📤 IMÁGENES GENERADAS (no en git)
│   ├── dinov2_refs/           # Referencias DINOv2 para indexar
│   │   ├── cenital/
│   │   └── lateral/
│   ├── yolo_training/         # Datasets YOLO para entrenamiento
│   │   ├── cenital/
│   │   │   ├── images/train/
│   │   │   └── labels/train/
│   │   └── lateral/
│   │       ├── images/train/
│   │       └── labels/train/
│   ├── test/                  # Imágenes de test
│   │   ├── test_dual/
│   │   ├── test_300/
│   │   ├── test_50_canonical/
│   │   └── inferencia_test_v3_colors/
│   └── canonical/             # Renders de set canónico
│       ├── set_300/
│       └── set_500/
│
├── reports/                   # 📊 INFORMES (no en git)
│   ├── eval_report.json
│   ├── inferencia_test_v3_eval.json
│   ├── inference_300_eval.json
│   ├── inference_300_summary.html
│   ├── inference_300_full.csv
│   ├── color_analysis.md
│   ├── color_analysis.html
│   ├── piece_report/
│   └── color_focus/
│
├── models/                    # 🤖 MODELOS (git: solo .gitkeep)
│   ├── yolo_cenital.pt
│   ├── yolo_lateral.pt
│   ├── yolo_cenital_pose.pt
│   ├── yolo_lateral_pose.pt
│   └── dinov2_index/
│
└── logs/                      # 📋 LOGS (no en git)
    └── pipeline_YYYY-MM-DD.log
```

## 🎯 Convenciones

### 1. Dominio `data/` — Datos de entrada
- **Solo lectura** en la mayoría de scripts
- Archivos pequeños y estables
- **SÍ versionado en git** (excepto `tmp/` y `ldraw/`)
- Ejemplos: caches de poses, keypoints canónicos, paletas de color

### 2. Dominio `renders/` — Imágenes generadas
- **Escritura** por scripts de render (Blender)
- Archivos grandes y regenerables (~GB)
- **NO versionado en git** (excluido en `.gitignore`)
- Ejemplos: datasets YOLO, refs DINOv2, test sets

### 3. Dominio `reports/` — Informes
- **Escritura** por scripts de evaluación y análisis
- JSON, CSV, HTML, Markdown
- **NO versionado en git** (regenerables)
- Ejemplos: eval_report.json, color_analysis.html

### 4. Dominio `models/` — Modelos entrenados
- **Lectura/escritura** por scripts de training e inferencia
- Archivos binarios grandes (.pt, .pkl)
- **NO versionado en git** (usar `git add -f` para modelos validados)
- Ejemplos: yolo_cenital.pt, dinov2_index/

### 5. Dominio `logs/` — Logs
- **Escritura** automática por `logger.py`
- Rotados diariamente
- **NO versionado en git**

## 📝 Uso en Scripts

Los scripts deben usar rutas relativas al `project_root`:

```python
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leer datos de entrada
cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")

# Escribir renders
output_dir = os.path.join(project_root, "renders", "dinov2_refs")

# Escribir reports
report_path = os.path.join(project_root, "reports", "eval_report.json")

# Cargar modelos
model_path = os.path.join(project_root, "models", "yolo_cenital.pt")
```

## 🔧 Configuración en `config.yaml`

La sección `paths` centraliza todas las rutas:

```yaml
paths:
  data_base: "data"
  renders_base: "renders"
  reports_base: "reports"
  models_base: "models"
  logs_base: "logs"
  
  # Rutas específicas
  dinov2_refs: "renders/dinov2_refs"
  yolo_cenital: "renders/yolo_training/cenital"
  eval_report: "reports/eval_report.json"
  # ... etc
```

## 🚀 Migración desde estructura anterior

Si tienes datos en la estructura anterior (`data/yolo_cenital/`, `data/reports/`, etc.):

1. **Mover renders**: `mv data/yolo_cenital renders/yolo_training/cenital`
2. **Mover reports**: `mv data/reports/* reports/`
3. **Mover refs DINOv2**: `mv data/dinov2_refs_* renders/dinov2_refs`

Los scripts actualizados usan las nuevas rutas por defecto, pero siguen aceptando `--output_dir` para rutas personalizadas.