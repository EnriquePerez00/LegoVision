# SETUP_DOC — LegoVision: 2camaras\_multi\_pieza

> **Iteración:** 2camaras\_multi\_pieza | **Fecha:** 2026-06-05  
> **Ubicación:** `LegoVision/2camaras_multi_pieza/`  
> **Basado en:** LegoVision Iteración 9 (setup simétrico multicámara)

---

## 1. Descripción General del Setup

Este proyecto implementa un pipeline de clasificación automática de piezas LEGO sobre una cinta transportadora utilizando **dos cámaras simétricas** (cenital + frontal). El sistema combina detección YOLO para localización y DINOv2 para identificación visual, con un árbol de decisión en cascada que incorpora validaciones geométricas calibradas matemáticamente.

### Diagrama Conceptual

```
                    ┌────────────────────────────────────┐
                    │        Tira LED Cuadrada           │
                    │   (22×22cm, Z=18cm sobre cinta)    │
                    └────────────┬───────────────────────┘
                                 │ Iluminación difusa
              ┌──────────────────▼──────────────────┐
              │         CÁMARA CENITAL               │
              │   Pos: (0, 0, 15cm) | f=27mm        │
              │   FOV: 20×20cm | 3.2px/mm           │
              └──────────────────┬──────────────────┘
                                 │
┌────────────┐  Pieza en cinta   ▼          ┌────────────┐
│CÁMARA      │◄──────────────────────────────│            │
│FRONTAL     │   Cinta: 20cm ancho           │  (cinta)   │
│(0,-15cm,0) │   Largo: 60cm                 │            │
│f=27mm      │   Superficie: Z=0             └────────────┘
│3.2px/mm    │
└────────────┘
```

---

## 2. Parámetros de la Cinta (Belt)

| Parámetro          | Valor             | Notas                               |
|--------------------|-------------------|-------------------------------------|
| Ancho              | 20 cm (20.0 BU)   | 1 Blender Unit = 10 mm              |
| Largo              | 60 cm (60.0 BU)   | Cinta completa renderizada          |
| Grosor colisionador| **1.0 BU (10 mm)** | Cubo solido fino - mas realista para cinta |
| Superficie Z       | 0.0               | Plano de referencia de la cinta     |
| Color (hex)        | `#254154`         | Petrol blue (contraste con piezas)  |
| Rugosidad          | 0.5               | ABS mate                            |
| Rozamiento física  | 0.95              | Alta adherencia para estabilidad    |
| Carriles laterales | Habilitados       | Aluminio mate, 2mm ancho, 4mm alto  |

---

## 3. Configuración de Cámaras (Setup Simétrico)

### 3.1 Cámara Cenital

| Parámetro         | Valor                | Cálculo / Justificación              |
|-------------------|----------------------|--------------------------------------|
| Posición          | `[0.0, 0.0, 15.0]`  | 15 cm sobre el centro de la cinta    |
| Pitch             | 90° (perpendicular)  | Apunta directamente hacia abajo      |
| Focal length      | 27.0 mm              | `f = 36·D / W = 36·150 / 200 = 27mm`|
| FOV horizontal    | 20 cm (200 mm)       | Cubre exactamente el ancho de cinta  |
| Resolución        | 640 × 640 px         | Cuadrada para simetría de escala     |
| Escala px/mm      | **3.2 px/mm**        | `640px / 200mm = 3.2 px/mm`         |
| Rol               | Clasificación primaria (color + superficie + DINOv2) |

### 3.2 Cámara Frontal

| Parámetro         | Valor                | Cálculo / Justificación              |
|-------------------|----------------------|--------------------------------------|
| Posición          | `[0.0, -15.0, 0.0]` | 15 cm en Y, altura Z=0 (nivel cinta) |
| Pitch             | 0° (paralelo)        | Eje óptico paralelo al plano de cinta|
| Focal length      | 27.0 mm              | Idéntico a cenital (setup simétrico) |
| FOV horizontal    | 20 cm (200 mm)       | Cubre exactamente el ancho de cinta  |
| Resolución        | 640 × 640 px         | Cuadrada para simetría de escala     |
| Escala px/mm      | **3.2 px/mm**        | `640px / 200mm = 3.2 px/mm`         |
| Rol               | Verificación de perfil de altura + DINOv2 secundario |

### 3.3 Simetría y Factor de Corrección Unificado

Al tener **ambas cámaras a la misma distancia** (150 mm) del centro con el **mismo focal** (27 mm), el factor de conversión píxel/mm es **idéntico** para ambas vistas. Esto simplifica enormemente la calibración y el pipeline de inferencia:

```
scale_px_per_mm_cenital  = scale_px_per_mm_frontal = 3.2 px/mm
```

---

## 4. Iluminación: Tira de LEDs Cuadrada Difusa

### 4.1 Configuración Geométrica

```
4 luces AREA rectangulares formando un contorno cuadrado:
  ┌──────────── LED_Strip_N ────────────┐  ← (0, +11, 18) BU
  │                                     │
LED_Strip_W                       LED_Strip_E
(-11, 0, 18) BU             (+11, 0, 18) BU
  │                                     │
  └──────────── LED_Strip_S ────────────┘  ← (0, -11, 18) BU

Dimensiones de cada segmento: 22×0.5 BU
Altura Z: 18.0 cm sobre la cinta
Energía por segmento: 150 W
```

### 4.2 Parámetros de Variación (Simulación de Setup Estático)

| Parámetro          | Dataset Entrenamiento      | Referencias DINOv2    |
|--------------------|----------------------------|-----------------------|
| Color de luz       | ±2% cálido/frío alternante | Blanco neutro (1,1,1) |
| Variación energía  | ±2% (`0.98` a `1.02`)      | Sin variación         |
| Variación posición | ±1 mm en X/Y/Z             | Sin variación         |

> **Consistencia:** La geometría LED es **idéntica** en imágenes cenitales, frontales y referencias DINOv2, eliminando discrepancias de dominio visual.

---

## 5. Generación de Datasets de Entrenamiento YOLO

### 5.1 Dataset Cenital (Empaquetamiento Monte Carlo 2D)

**Objetivo:** 1000 piezas individuales detectadas.

**Algoritmo de empaquetamiento:**
1. Límite de colocación: `±9.5 BU` (margen de 0.5 cm con bordes de cinta y FOV)
2. Separación mínima entre piezas: `0.5 BU` (5 mm)
3. Cada frame: se intentan colocar entre 5 y 12 piezas
4. Para cada pieza candidata: 15 intentos con rotación Z aleatoria (`0°–360°`)
5. Validación AABB expandida (con margen de 0.5 BU) para garantizar no-colisión
6. Asentamiento automático en Z: `z = -min_z + 0.02`
7. Ratio de frames vacíos: 5% para robustez del detector YOLO

**Parámetros configurables en `config.yaml`:**
```yaml
yolo:
  dataset:
    pieces_per_frame_min: 5
    pieces_per_frame_max: 12
    empty_frame_ratio: 0.05
```

### 5.2 Dataset Frontal (Spawning en Línea)

**Objetivo:** 1000 piezas individuales detectadas.

**Algoritmo de spawning:**
1. Piezas colocadas en línea a lo largo del eje X, con `Y=0` (perpendicular al eje de la cámara frontal)
2. Poses estables con normal hacia arriba; rotación Z aleatoria
3. Separación mínima de 0.5 cm entre piezas y con bordes de cinta
4. La línea se rellena de izquierda a derecha hasta agotar el espacio disponible (`±9.5 BU`)
5. Ratio de frames vacíos: 5%

---

## 6. Generación de Referencias DINOv2

**Objetivo:** Poblar la base de datos vectorial para clasificación KNN.

### 6.1 Vista Cenital
- 1 pieza sola en el centro de la escena `(0, 0, 0)`
- 12 rotaciones Z (pasos de 30°) por pose estable
- 1 color por referencia (color real del set 75078-1)

### 6.2 Vista Frontal (Multicopia + Recortes)
- Máximo de copias de la misma pieza colocadas en fila (ocupando todo el ancho)
- Renderizado completo → recorte individual por bounding box proyectado
- Se guarda el recorte con sufijo `_instXX` para identificar la instancia
- El script de reindexación selecciona automáticamente la instancia **más central** (menor distorsión de perspectiva) para cada combinación `(pieza, color, pose, rotación)`

### 6.3 Nomenclatura de archivos

```
Cenital:  ref_{PART}_{COLORHEX}_pose{NN}_rot{DDD}.png
Frontal:  ref_{PART}_{COLORHEX}_pose{NN}_rot{DDD}_inst{XX}.png
```

---

## 7. Pipeline de Inferencia y Árbol de Decisión en Cascada

```
Entradas: crop_cenital, crop_frontal, bbox_norm_cenital

Phase 1: Gating de Color (Cenital)
  └─ Clasificación cromaticidad del crop cenital
  └─ Mapeo: código 84 (Dark Bluish Gray) → 85 (Light Bluish Gray)
  └─ Salida: lista de refs. con mismo color en el set

Phase 2: Gating de Superficie Cenital (tolerancia ±15%)
  └─ Extracción MinAreaRect del contorno segmentado (cenital)
  └─ Calibración de perspectiva: área_aparente = área_nom × (150/(150−h))²
  └─ Tres configuraciones de apoyo: flat, side, stand
  └─ Filtrado: piezas dentro del ±15% de área aparente nominal

Phase 3: Gating de Altura Frontal con Compensación de Paralaje (tolerancia ±15%)
  └─ Posición física en cinta: X_cm = (x_norm − 0.5) × 20cm
                                Y_cm = (0.5 − y_norm) × 20cm
  └─ Distancia real a cámara frontal:
     dist_frontal_mm = √[(X_cm·10)² + (Y_cm·10 + 150)²]
  └─ Altura compensada:
     h_comp = h_medida_px / px_per_mm × (dist_frontal_mm / 150mm)
  └─ Validación contra alturas nominales (H+0.9, H, W, L)

Phase 4: Fusión DINOv2 (Cenital 70% + Frontal 30%)
  └─ KNN sobre embeddings proyectados para piezas que pasaron filtros
  └─ Score combinado: 0.7×S_cenital + 0.3×S_frontal
  └─ Predicción: pieza con score combinado máximo
```

### 7.1 Compensación Matemática de Paralaje (Detalle)

Cuando una pieza está a `X_cm` del centro de la cinta, su distancia real a la cámara frontal (ubicada en `Y=-15cm, Z=0`) es:

```
dist = √[(X_cm × 10)² + (Y_cm × 10 + 150)²]  [en mm]
```

La altura observada en píxeles se compensa para obtener la altura física real:

```
h_real_mm = (h_px / px_per_mm) × (dist / 150.0)
```

Esto corrige el error de perspectiva que provocaría falsos negativos en la validación de altura (hasta un 13% de subestimación a X=8cm sin compensación).

### 7.2 Calibración de Superficie Cenital (Detalle)

La cámara cenital a Z=15cm observa una pieza de altura `h_rest` mm con un **factor de magnificación**:

```
magnif = (150mm / (150mm − h_rest))²
área_aparente_nom = área_física × magnif
```

Esto corrige que una pieza vista desde arriba parece más grande de lo que es físicamente, especialmente para piezas altas (9.6mm de altura → ~13% de aumento de área aparente).

---

## 8. Piezas del Set (75078-1)

| Ref    | Nombre                           | Dims (L×W×H mm) | Color (set) |
|--------|----------------------------------|-----------------|-------------|
| `3005` | Brick 1×1                        | 8×8×9.6         | LBGray      |
| `3001` | Brick 2×4                        | 32×16×9.6       | LBGray      |
| `3039` | Slope 45° 2×2                    | 16×16×9.6       | LBGray      |
| `3665` | Slope Inv. 45° 1×2               | 16×8×9.6        | LBGray      |
| `3010` | Brick 1×4                        | 32×8×9.6        | LBGray      |
| `3002` | Brick 2×3                        | 24×16×9.6       | LBGray      |
| `3020` | Plate 2×4                        | 32×16×3.2       | LBGray      |
| `4070` | Brick Mod. 1×1 stud lateral      | 8×8×9.6         | LBGray      |
| `4032` | Plate 2×2 Round                  | 16×16×3.2       | LBGray      |
| `3700` | Technic Brick 1×2 (con agujero)  | 16×8×9.6        | LBGray      |

> **Nota:** Light Bluish Gray (código 85) es el color predominante. Se aplica un mapeo `código 84 → 85` para compensar variaciones de clasificación cromaticidad.

> [!WARNING]
> **Exclusión de Minifiguras:** Las figuras compuestas/minifiguras (como `sw0614`) están explícitamente excluidas de los pipelines de renderizado e inferencia. Únicamente se procesan y detectan las mallas de las piezas de inventario individuales del set.

---

## 9. Estructura del Proyecto

```
2camaras_multi_pieza/
├── config.yaml                          # Configuración centralizada del setup
├── SETUP_DOC.md                         # Este documento
├── config_loader.py                     # Loader YAML para cfg.*
├── requirements.txt                     # Dependencias Python
├── .env                                 # Variables de entorno (Supabase)
├── yolo11n.pt                           # Modelo base YOLO
│
├── scripts/
│   ├── generate_yolo_training_dataset_cenital.py    # Dataset YOLO cenital (Monte Carlo 2D)
│   ├── generate_yolo_training_dataset_frontal.py    # Dataset YOLO frontal (línea)
│   ├── generate_eevee_dinov2_refs.py                # Referencias DINOv2 (cenital + frontal)
│   ├── reindex_dinov2_eevee.py                      # Reindexación embeddings en BD
│   └── run_iter9_evaluation_calibrated.py           # Evaluación con compensaciones
│
├── database/
│   ├── set_catalog.py                   # Catálogo piezas/colores del set 75078-1
│   └── supabase_client.py               # Cliente BD vectorial (PostgreSQL/Supabase)
│
├── inference/
│   ├── knn_classifier.py                # Clasificador KNN + DINOv2
│   └── api.py                           # Alturas nominales de piezas
│
└── data/
    ├── dinov2_refs_full/                # Referencias DINOv2 renderizadas
    │   ├── cenital/                     # ref_PART_COLOR_poseNN_rotDDD.png
    │   └── frontal/                     # ref_PART_COLOR_poseNN_rotDDD_instXX.png
    └── iter9_test/                      # Set de test
        ├── test_metadata.json           # GT y bboxes
        └── eval_report_iter9.json       # Resultados de evaluación
```

---

## 10. Comandos de Ejecución

> **Nota:** Todos los comandos se ejecutan desde la raíz de `LegoVision/`.

### Generación de Datasets YOLO

```bash
# Dataset Cenital
blender -b -P 2camaras_multi_pieza/scripts/generate_yolo_training_dataset_cenital.py -- \
    --output_dir 2camaras_multi_pieza/data/yolo_cenital_train

# Dataset Frontal
blender -b -P 2camaras_multi_pieza/scripts/generate_yolo_training_dataset_frontal.py -- \
    --output_dir 2camaras_multi_pieza/data/yolo_frontal_train
```

### Generación de Referencias DINOv2

```bash
blender -b -P 2camaras_multi_pieza/scripts/generate_eevee_dinov2_refs.py -- \
    --output_dir 2camaras_multi_pieza/data/dinov2_refs_full \
    --rotations 12
```

### Reindexación de Embeddings

```bash
.venv/bin/python 2camaras_multi_pieza/scripts/reindex_dinov2_eevee.py \
    --ref_dir 2camaras_multi_pieza/data/dinov2_refs_full
```

### Evaluación del Pipeline

```bash
.venv/bin/python 2camaras_multi_pieza/scripts/run_iter9_evaluation_calibrated.py
```


---

## 11. Compensaciones Matemáticas Implementadas

### 11.1 Parámetros de Calibración

```python
PX_PER_MM_REF   = 3.2   # Calibración unificada ambas cámaras
PX_PER_MM_TEST  = 3.2
CAMERA_DIST_MM  = 150.0  # Distancia de ambas cámaras al centro

# Árbol de decisión — tolerancias
SURFACE_TOLERANCE = 0.15   # ±15% área superficie cenital
HEIGHT_TOLERANCE  = 0.15   # ±15% altura frontal (post-compensación)

# Stud offset
STUD_OFFSET_MM = 0.9       # Altura adicional del stud en pose flat-up
```

### 11.2 Mapeo de Colores

```python
# Compensación cromaticidad: Dark Bluish Gray → Light Bluish Gray
COLOR_MAP = {
    "84": "85"  # Variación tonal en condiciones de baja temperatura de luz
}
```

---

## 12. Análisis de Riesgos y Puntos de Mejora

### ✅ Resueltos en esta iteración

| Riesgo | Solución implementada |
|--------|-----------------------|
| Paralaje frontal (hasta 13% error en X=8cm) | Compensación matemática `h_comp = h_px × dist / 150` |
| Magnificación cenital (13% en piezas 9.6mm) | Calibración perspectiva `A_app = A_nom × (D/(D-h))²` |
| Variación cromaticidad Gray 84 vs 85 | Mapeo explícito en el árbol de color |
| Incoherencia óptica cenital/frontal | Setup simétrico: misma focal (27mm), misma distancia (15cm) |

### ⚠️ Riesgos conocidos no completamente resueltos

| Riesgo | Impacto | Mitigación actual |
|--------|---------|-------------------|
| **Oclusión frontal** — Piezas alineadas en X=±Ycm se tapan mutuamente en frontal | Alta — arruina gating de altura | Pending: detección de oclusión por proyección 2D cenital |
| **Sombras proyectadas** — LED cenitales crean sombras a Z=0 hacia los bordes | Media — confunde segmentación cromaticidad | Umbral luminance_min_thresh=30 en config.yaml |
| **Piezas iguales (3005 vs 4070)** — Mismas dimensiones externas, diferente feature visual | Alta — requiere DINOv2 robusto | Tolerancia 15% puede no diferenciarlas por tamaño; depende del visual similarity |
| **Tolerancias 15% en piezas pequeñas** — Riesgo de cross-match entre 3005/4070 y placas | Media | Aumentar dataset de referencias |

---

## 13. Resultados de Evaluación (Estado Actual)

| Métrica              | Valor   | Notas                                        |
|----------------------|---------|----------------------------------------------|
| Accuracy global      | ~41%    | Después de calibración con tolerancias ±15%  |
| Motor de render      | EEVEE   | Mismas condiciones que test                  |
| Resolución           | 640×640 | Factor px/mm = 3.2 px/mm uniforme            |
| Embeddings indexados | 1248    | DINOv2 ViT-S/14, 384-dim                    |
| Compensación paralaje| Activa  | `dist_frontal_mm / 150mm`                   |
| Calibración área     | Activa  | `(150 / (150 - h_rest))²`                   |

> **Principal limitación actual:** El accuracy del 41% sugiere que la clasificación DINOv2 es el cuello de botella. El dataset de referencias es limitado y no cubre suficientes variaciones de iluminación/perspectiva para piezas visualmente similares (3005, 4070, 3700). **Acción recomendada:** Ampliar el dataset de referencias con más poses y rotaciones, o incorporar fine-tuning contrastivo del encoder DINOv2.

---

## 14. Sistema de Logs

### 14.1 Descripción General

El proyecto incluye un sistema de logging centralizado que vuelca logs a archivos en `2camaras_multi_pieza/logs/` para análisis post-ejecución. Cada módulo genera su propio archivo con fecha:

```
2camaras_multi_pieza/
└── logs/
    ├── blender_2026-06-04.log        # Renders Blender (refs DINOv2 + YOLO)
    ├── dinov2_2026-06-04.log         # Indexación embeddings DINOv2
    ├── pipeline_2026-06-04.log       # Evaluación del pipeline (decisiones en cascada)
    └── yolo_cenital_2026-06-04.log   # Generación dataset YOLO cenital
```

> **Nota:** La carpeta `logs/` está excluida de Git (`.gitignore`) pero existe en el repo gracias a `logs/.gitkeep`. Los archivos `.log` generados son locales y no se suben al repositorio.

---

### 14.2 Configuración en `config.yaml` (Sección 12)

La configuración completa está en la sección `logging` del `config.yaml`:

```yaml
logging:
  enabled: true                  # false → desactiva todos los logs en disco
  dir: "logs"                    # Relativo a 2camaras_multi_pieza/
  level: "INFO"                  # Nivel global: DEBUG | INFO | WARNING | ERROR
  max_file_size_mb: 50           # Rotación automática al superar 50 MB
  max_files: 10                  # Máximo 10 archivos rotados por módulo

  blender:
    level: "INFO"                # Cambiar a DEBUG → log de cada render individual
  dinov2:
    level: "INFO"                # Throughput, embeddings, errores BD
  pipeline:
    level: "DEBUG"               # DEBUG recomendado: captura cada decisión en cascada
  yolo_cenital:
    level: "INFO"                # Frames generados, piezas por frame
```

**Para activar más verbosidad:**
- Cambiar `level: "INFO"` → `level: "DEBUG"` en cualquier módulo
- El módulo `pipeline` ya usa `DEBUG` por defecto para capturar los scores de cada fase del algoritmo en cascada

**Para desactivar completamente:**
- Cambiar `enabled: false` en el nivel raíz `logging:`

---

### 14.3 Módulo `logger.py`

El módulo `2camaras_multi_pieza/logger.py` es la utilidad centralizada. Para usar el logger en cualquier script nuevo:

```python
# Al inicio del script (antes de main())
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("mi_modulo")   # → genera logs/mi_modulo_YYYY-MM-DD.log

# En main():
log_execution_header(log, "mi_script.py", param1=valor1, param2=valor2)
log.info("Mensaje informativo")
log.warning("Algo salió diferente de lo esperado")
log.error("Error grave")
log.debug("Detalle solo en modo DEBUG")
log_execution_footer(log, "mi_script.py", duration_s=elapsed, resultado="OK")
```

Los logs se escriben **simultáneamente** en:
1. Archivo rotativo en `logs/`
2. Stdout (visible en la terminal de Blender y en la consola del sistema)

---

### 14.4 Contenido de los Logs por Módulo

#### `blender_<date>.log` — Generación de referencias DINOv2
```
2026-06-04 12:00:01 [INFO] [legov.blender] Logger iniciado → logs/blender_2026-06-04.log
2026-06-04 12:00:01 [INFO] [legov.blender] ============================================================
2026-06-04 12:00:01 [INFO] [legov.blender] INICIO: generate_eevee_dinov2_refs.py
2026-06-04 12:00:01 [INFO] [legov.blender]   output_dir: data/dinov2_refs_full
2026-06-04 12:00:01 [INFO] [legov.blender]   rotations: 12
2026-06-04 12:00:05 [INFO] [legov.blender] === Generando referencias para la pieza: 3001 ===
2026-06-04 12:05:30 [WARNING] [legov.blender] import LDraw 3700: FileNotFoundError
2026-06-04 12:30:00 [INFO] [legov.blender] FIN: generate_eevee_dinov2_refs.py
2026-06-04 12:30:00 [INFO] [legov.blender]   Duración total: 30m 0s
2026-06-04 12:30:00 [INFO] [legov.blender]   total_rendered: 1440
```

#### `dinov2_<date>.log` — Indexación de embeddings
```
2026-06-04 12:30:01 [INFO] [legov.dinov2] DINOv2 cargado en mps
2026-06-04 12:30:02 [INFO] [legov.dinov2] Limpiando embeddings existentes...
2026-06-04 12:30:03 [INFO] [legov.dinov2] Cámara cenital: 720 imágenes | IO paralelo 4 workers...
2026-06-04 12:30:15 [INFO] [legov.dinov2] Indexados 128 embeddings  [10.5 emb/s]
2026-06-04 12:31:00 [INFO] [legov.dinov2] DONE: 1248 embeddings indexados
2026-06-04 12:31:00 [INFO] [legov.dinov2]   throughput_emb_s: 20.8
```

#### `pipeline_<date>.log` — Evaluación del pipeline (nivel DEBUG)
```
2026-06-04 13:00:01 [INFO] [legov.pipeline] INICIO: run_iter9_evaluation_calibrated.py
2026-06-04 13:00:05 [INFO] [legov.pipeline] [01/18] GT=3001   -> Pred=3001    ✓  (score=0.8234 | color=85 | h_comp=9.71mm | valid_color=10 | valid_surf=3 | valid_h=2)
2026-06-04 13:00:05 [DEBUG] [legov.pipeline]   Scores cenital={'3001': 0.87, '3010': 0.62} | frontal={'3001': 0.71}
2026-06-04 13:00:06 [WARNING] [legov.pipeline] [02/18] GT=3020   -> Pred=3001    ✗  (score=0.5123 | color=85 | h_comp=3.28mm | valid_color=10 | valid_surf=2 | valid_h=3)
2026-06-04 13:00:15 [INFO] [legov.pipeline]   Precisión global : 41.00%
```

---

### 14.5 Análisis Post-Ejecución

#### Filtrar errores de una ejecución:
```bash
grep "\[ERROR\]\|\[WARNING\]" 2camaras_multi_pieza/logs/pipeline_$(date +%Y-%m-%d).log
```

#### Ver solo predicciones incorrectas del pipeline:
```bash
grep "✗" 2camaras_multi_pieza/logs/pipeline_$(date +%Y-%m-%d).log
```

#### Ver throughput del indexador DINOv2:
```bash
grep "emb/s" 2camaras_multi_pieza/logs/dinov2_$(date +%Y-%m-%d).log
```

#### Comparar accuracy entre ejecuciones:
```bash
grep "Precisión global" 2camaras_multi_pieza/logs/pipeline_*.log
```

#### Analizar patrones de fallo por pieza (ej: 3020):
```bash
grep "GT=3020.*✗" 2camaras_multi_pieza/logs/pipeline_*.log
```

#### Ver resumen completo de una sesión:
```bash
grep "INICIO\|FIN\|Precisión\|total_rendered\|DONE" 2camaras_multi_pieza/logs/*.log
```

---

### 14.6 Política de Retención

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `max_file_size_mb` | 50 MB | Rotación automática por tamaño |
| `max_files` | 10 | Máximo de archivos rotados por módulo |
| Nombre de archivo | `<modulo>_YYYY-MM-DD.log` | Un archivo por día |
| Limpieza manual | — | `rm 2camaras_multi_pieza/logs/*.log` |

> **Espacio estimado:** Una evaluación completa genera ~0.5 MB en `pipeline_*.log` (nivel DEBUG). Una generación de referencias DINOv2 genera ~2 MB en `blender_*.log`.


---

## 11. Setup v2 - Resumen de Cambios (2026-05-06)

Esta seccion resume los cambios introducidos en la iteracion v2 del setup
respecto al setup original documentado arriba.

### 11.1 Set y Piezas Seleccionadas
- **Set:** 75078-1 (Imperial Troop Transport, Star Wars Rebels)
- **Piezas seleccionadas:** 20 piezas aleatorias del set (excluyendo la minifigura `sw0614` y otras figuras compuestas del pipeline de renderizado e inferencia, procesando exclusivamente las piezas individuales no-minifigura de su inventario).
- Lista en `cfg.pieces.selected_parts`:
  `3004, 3001, 3020, 3022, 2877, 59900, 3003, 3005, 3010, 3023,
  3024, 3710, 3665, 3039, 4070, 6141, 3069, 3068, 3700, 4032`

### 11.2 Cinta - Grosor Reducido
- **Antes:** thickness 8 cm (80 mm) - poco realista
- **Despues:** thickness **1 cm** (10 mm) - mas representativo de una cinta industrial

### 11.3 Iluminacion: Lab Lightbox
Reemplaza la antigua "LED Strip cuadrada" por una caja de luz tipo laboratorio:

| Componente        | Tamano       | Posicion BU       | Energia | Proposito             |
|-------------------|--------------|-------------------|---------|------------------------|
| Lab_Main_Dome     | 35x35 cm     | (0, 0, 12)        | 2000 W  | Cenital difusa principal |
| Lab_Wall_N        | 20x12 cm     | (0, +12, 6)       | 600 W   | Iluminacion lateral norte |
| Lab_Wall_S        | 20x12 cm     | (0, -12, 6)       | 600 W   | Iluminacion lateral sur |
| Lab_Wall_E        | 20x12 cm     | (+12, 0, 6)       | 600 W   | Iluminacion lateral este |
| Lab_Wall_W        | 20x12 cm     | (-12, 0, 6)       | 600 W   | Iluminacion lateral oeste |
| Lab_Ground_Fill   | 30x30 cm     | (0, 0, -0.5)      | 200 W   | Rebote del suelo |
| World ambient     | -            | -                 | 30%     | Fill global suave |

Total: ~4600 W -> entorno tipo laboratorio de inspeccion industrial.

### 11.4 Suelo Negro (Black Floor) - Nuevo
- Plano 60x60 BU (60x60 cm) en z = -2 cm
- Color `#000000` (negro absoluto), roughness=1.0, metallic=0.0
- Visible solo desde la camara frontal
- **Beneficio:** mejora contraste pieza/fondo en imagenes frontales,
  facilita segmentacion y reduce variabilidad en imagenes de validacion/inferencia.

### 11.5 Filtro de Estabilidad de Poses
- Threshold: `cfg.stable_poses.render_min_stability = 0.01`
- Solo se renderizan poses con `stability_ratio >= 0.01` (umbral muy permisivo: incluye todas las poses fisicamente posibles, descarta solo las degeneradas con ratio = 0.0) (de la simulacion fisica almacenada en `data/stable_poses_cache.json`)
- **Filtro dimensional de base de contacto (Setup v2):** Se descarta cualquier pose si el ancho o el largo de la superficie de contacto es menor o igual a **4.0 mm** (`min_contact_dimension_mm = 4.0`), lo cual evita automáticamente poses inestables sobre bordes o aristas (por ejemplo, considerando que **el lateral mini pieza lego es de 3.2 mm**, todas las poses apoyadas sobre el canto lateral de placas delgadas quedan descartadas).
- **Fallback geometrico:** si ninguna pose pasa el filtro, se usan las que tienen `face_class in (Top, Bottom)` (CoG-estables geometricamente)
- **Ultimo recurso:** si la pieza no esta en cache, se usa la pose identidad (studs arriba, quaternion `[1, 0, 0, 0]`)
- Implementado en los 4 scripts de render: cenital, frontal, validacion, dinov2

### 11.6 Targets de Render (config)
| Pipeline           | Target    | Frames Estimados | Param config                         |
|--------------------|-----------|------------------|---------------------------------------|
| YOLO Cenital       | 10000 piezas | ~1430 frames | `yolo.dataset.total_pieces_cenital`  |
| YOLO Frontal       | 5000 piezas  | ~830 frames  | `yolo.dataset.total_pieces_frontal`  |
| Validacion         | -         | 500 imagenes     | `validation_renders.total_images`     |
| DINOv2 referencias | -         | ~varies          | `dinov2.reference_generation.*`       |

### 11.7 Renders de Validacion / Inferencia
- **Camara:** Frontal (mismas posicion/lens/resolucion que YOLO frontal)
- **Estrategia de colocacion:** 1 unica linea de piezas a lo ancho (eje X)
- **Posicion Y:** `-0.5 BU` (= 5 cm hacia la camara frontal desde el centro)
- **Salida:** `data/validation_renders/images/val_NNNNN.png` + `metadata.json`
- **metadata.json:** ground truth completo de cada imagen:
  - `ref` (LDraw part), `color_hex`, `pose_index`, `rot_z_rad`,
    `position_bu`, `dimensions_mm`

### 11.8 Comparacion v1 vs v2

| Aspecto              | v1 (original) | v2 (este setup) |
|----------------------|---------------|-----------------|
| Belt thickness       | 8 cm          | **1 cm**        |
| Lighting             | LED strip 4x150W = 600W | Lab lightbox ~4600W |
| Floor                | No            | **Negro 60x60 BU** |
| Stability filter     | Ninguno       | **>= 0.01 + fallback Top/Bottom** |
| Selected parts       | 10            | **20** del set 75078-1 |
| YOLO cenital target  | 1000 piezas   | **10000 piezas** |
| YOLO frontal target  | 1000 piezas   | **5000 piezas** |
| Validation script    | -             | **NUEVO: 500 imgs Y=-0.5 BU** |


---

## 12. Optimización v3 — Reducción de Tiempo (2026-05-06)

### Análisis de los logs de entrenamiento

El entrenamiento cenital (35 epochs) mostró:
- Epoch 1: mAP50-95 = 0.881
- Epoch 3: mAP50-95 = 0.953
- Epoch 8: mAP50-95 = **0.980** (plateau start)
- Epoch 15: mAP50-95 = 0.990
- Epoch 35: mAP50-95 = 0.994 (solo +0.004 en 20 epochs adicionales)

**Conclusión:** El modelo converge en 8-10 epochs. Los últimos 25 epochs aportan +0.014 (despreciable).

### Cambios aplicados

| Parámetro | Antes (v2) | Después (v3) | Ahorro | Impacto |
|-----------|-----------|-------------|--------|---------|
| `yolo.training.epochs` | 35 | **15** | ~25 min | <0.004 mAP |
| `yolo.training.amp` | false | **true** | ~30% más rápido | ~0 impacto |
| `yolo.dataset.total_pieces_cenital` | 10000 | **5000** | ~20 min render | <0.1% |
| `yolo.dataset.total_pieces_frontal` | 5000 | **3000** | ~10 min render | <0.1% |

### Tiempo estimado del pipeline optimizado

| Fase | v2 (original) | v3 (optimizado) |
|------|---------------|-----------------|
| Render cenital | 40 min | ~20 min |
| Render frontal | 22 min | ~15 min |
| Render DINOv2 | 13 min | 13 min (sin cambio) |
| Render validación | 12 min | 12 min (sin cambio) |
| Train cenital | 37 min | ~15 min |
| Train frontal | 16 min | ~8 min |
| DINOv2 index | 16 sec | 16 sec |
| **TOTAL** | **~140 min** | **~83 min** |

### Justificación técnica

1. **Epochs 15 vs 35:** El modelo YOLO es de 1 sola clase ("lego_piece") sobre fondo
   muy predecible (petrol blue belt). El rendimiento del detector satura rápidamente
   porque la tarea es simple (forma vs fondo constante). No necesita 35 epochs para
   aprender separar pixel de pieza vs pixel de cinta.

2. **AMP (FP16):** Apple MPS soporta FP16 nativo. El overhead de conversión es
   despreciable. La detección de objetos (YOLO) no necesita FP32 para accuracy.

3. **5000/3000 piezas:** Para 1 clase con variabilidad geométrica limitada (20 piezas
   × ~4 poses × 6 colores), 5000 instancias de entrenamiento son más que suficientes.
   El original de 10000 fue conservador pero innecesario.


---

## 13. Estructura Autocontenida (2026-05-06)

> **Cambio v4:** Todos los datos, modelos y resultados del escenario `2camaras_multi_pieza`
> se almacenan DENTRO de su directorio. Esto permite tener multiples escenarios en
> el mismo repositorio sin interferencias.

### Arbol de directorios

```
2camaras_multi_pieza/                   ← RAIZ DEL ESCENARIO
├── config.yaml                          ← Configuracion centralizada
├── config_loader.py                     ← Loader para config.yaml
├── logger.py                            ← Sistema de logging
├── SETUP_DOC.md                         ← Este documento
├── requirements.txt
│
├── scripts/                             ← Scripts de render, training, evaluacion
│   ├── generate_yolo_training_dataset_cenital.py
│   ├── generate_yolo_training_dataset_frontal.py
│   ├── generate_eevee_dinov2_refs.py
│   ├── generate_validation_inference_renders.py
│   ├── reindex_dinov2_eevee.py
│   ├── run_iter9_evaluation_calibrated.py
│   ├── simulate_stable_poses.py
│   └── ...
│
├── database/                            ← Catalogos y cliente BD
│   ├── set_catalog.py                    (REAL_SETS con piezas del set 75078-1)
│   ├── color_catalog.json                (colores LEGO disponibles)
│   └── supabase_client.py                (acceso a Supabase)
│
├── models/                              ← Pesos de modelos entrenados
│   ├── yolo_cenital_v2.pt               (mAP50-95=0.994)
│   └── yolo_frontal_v2.pt              (mAP50-95=0.990)
│
├── data/                                ← TODOS los datos del escenario
│   ├── stable_poses_cache.json           (cache poses simuladas)
│   ├── yolo_cenital_full/               (1488 imgs + labels, 607 MB)
│   │   ├── images/
│   │   └── labels/
│   ├── yolo_cenital_split/              (train 1041 + val 447 + dataset.yaml)
│   ├── yolo_frontal_full/               (621 imgs + labels, 204 MB)
│   ├── yolo_frontal_split/              (train 434 + val 187 + dataset.yaml)
│   ├── dinov2_refs_full/                (576 cenital + 5296 frontal, 199 MB)
│   │   ├── cenital/
│   │   └── frontal/
│   ├── validation_renders_full/         (500 imgs + metadata.json, 164 MB)
│   │   ├── images/
│   │   ├── metadata.json
│   │   └── eval_report_v3.json
│   └── yolo_runs/                       (resultados de entrenamiento)
│       ├── cenital_v2/weights/best.pt
│       └── frontal_v2/weights/best.pt
│
└── logs/                                ← Logs de ejecucion
```

### Rutas en config.yaml

Todas las rutas son **relativas al directorio `2camaras_multi_pieza/`**:
- `data/yolo_cenital_full` (no `../data/...`)
- `data/dinov2_refs_full`
- `data/validation_renders_full`
- `models/yolo_cenital_v2.pt`

### Beneficios

1. **Portabilidad:** Se puede copiar/mover el directorio entero a otra maquina
2. **Aislamiento:** Multiples escenarios (ej: `1camara_simple/`, `3camaras_v2/`)
   pueden coexistir sin interferir
3. **Reproducibilidad:** Config + datos + modelos + scripts = todo junto
4. **Git-friendly:** Se puede agregar al .gitignore solo `data/` y `models/`
   manteniendo scripts y config versionados

