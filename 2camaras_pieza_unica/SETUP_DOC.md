# SETUP_DOC — LegoVision: 2camaras\_pieza\_unica

> **Iteración:** 2camaras\_pieza\_unica | **Fecha:** 2026-06-06  
> **Ubicación:** `LegoVision/2camaras_pieza_unica/`  
> **Basado en:** LegoVision `2camaras_multi_pieza` (setup simétrico multicámara)

---

## 1. Descripción General del Setup

Este proyecto implementa un pipeline de clasificación automática de piezas LEGO sobre una cinta transportadora utilizando **dos cámaras** (cenital + lateral) con **una única pieza a la vez** en escena. El sistema combina detección YOLO para localización y DINOv2 para identificación visual, con un árbol de decisión en cascada que incorpora validaciones geométricas calibradas matemáticamente.

### Diferencia clave respecto a `2camaras_multi_pieza`

| Aspecto | `2camaras_multi_pieza` | `2camaras_pieza_unica` |
|---------|------------------------|------------------------|
| Piezas por frame | 5-12 (Monte Carlo 2D) | **1 única pieza** |
| Cámara 2 | Frontal (0, -15, 0) | **Lateral (15, 0, 2.5)** |
| Spawning | Grid/línea multi-pieza | **Centro de cinta, eje cenital** |
| Referencia DINOv2 frontal | Multicopia + recortes | **1 pieza centrada** |

### Diagrama Conceptual

```
                    ┌────────────────────────────────────┐
                    │        Lab Lightbox (Dome)         │
                    │   (35×35cm, Z=12cm sobre cinta)    │
                    └────────────┬───────────────────────┘
                                 │ Iluminación difusa
              ┌──────────────────▼──────────────────┐
              │         CÁMARA CENITAL               │
              │   Pos: (0, 0, 15cm) | f=27mm        │
              │   FOV: 20×20cm | 3.2px/mm           │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼ Pieza ÚNICA centrada
┌────────────┐          ┌──────────────────┐
│CÁMARA      │◄─────────│                  │
│LATERAL     │  15cm    │  (cinta 20×60cm) │
│(15,0,2.5)  │          │  Pieza en (0,0)  │
│f=27mm      │          │                  │
│3.2px/mm    │          └──────────────────┘
└────────────┘
```

---

## 2. Parámetros de la Cinta (Belt)

| Parámetro          | Valor             | Notas                               |
|--------------------|-------------------|-------------------------------------|
| Ancho              | 20 cm (20.0 BU)   | 1 Blender Unit = 10 mm              |
| Largo              | 60 cm (60.0 BU)   | Cinta completa renderizada          |
| Grosor colisionador| **1.0 BU (10 mm)** | Cubo sólido fino - realista         |
| Superficie Z       | 0.0               | Plano de referencia de la cinta     |
| Color (hex)        | `#254154`         | Petrol blue (contraste con piezas)  |
| Rugosidad          | 0.5               | ABS mate                            |
| Rozamiento física  | 0.95              | Alta adherencia para estabilidad    |
| Carriles laterales | Habilitados       | Aluminio mate, 2mm ancho, 4mm alto  |

---

## 3. Configuración de Cámaras

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

### 3.2 Cámara Lateral

| Parámetro         | Valor                | Cálculo / Justificación              |
|-------------------|----------------------|--------------------------------------|
| Posición          | `[15.0, 0.0, 2.5]`  | 15 cm en X, elevación Z=2.5cm       |
| Pitch             | ~0° (casi paralelo)  | Mira hacia el centro de la cinta     |
| Focal length      | 27.0 mm              | Idéntico a cenital (setup simétrico) |
| FOV horizontal    | 20 cm (200 mm)       | Cubre exactamente el ancho de cinta  |
| Resolución        | 640 × 640 px         | Cuadrada para simetría de escala     |
| Escala px/mm      | **3.2 px/mm**        | `640px / 200mm = 3.2 px/mm`         |
| Rol               | Verificación de perfil de altura + DINOv2 secundario |

### 3.3 Factor de Corrección Unificado

Al tener **ambas cámaras a la misma distancia** (~150 mm) del centro con el **mismo focal** (27 mm), el factor de conversión píxel/mm es **idéntico** para ambas vistas:

```
scale_px_per_mm_cenital = scale_px_per_mm_lateral = 3.2 px/mm
```

---

## 4. Iluminación: Lab Lightbox

| Componente        | Tamaño       | Posición BU       | Energía | Propósito             |
|-------------------|--------------|-------------------|---------|------------------------|
| Lab_Main_Dome     | 35×35 cm     | (0, 0, 12)        | 2000 W  | Cenital difusa principal |
| Lab_Wall_N        | 20×12 cm     | (0, +12, 6)       | 600 W   | Iluminación lateral norte |
| Lab_Wall_S        | 20×12 cm     | (0, -12, 6)       | 600 W   | Iluminación lateral sur |
| Lab_Wall_E        | 20×12 cm     | (+12, 0, 6)       | 600 W   | Iluminación lateral este |
| Lab_Wall_W        | 20×12 cm     | (-12, 0, 6)       | 600 W   | Iluminación lateral oeste |
| Lab_Ground_Fill   | 30×30 cm     | (0, 0, -0.5)      | 200 W   | Rebote del suelo |
| World ambient     | -            | -                 | 30%     | Fill global suave |

---

## 5. Generación de Datasets de Entrenamiento YOLO

### 5.1 Estrategia: Pieza Única Centrada

**Principio:** Cada frame contiene exactamente **1 pieza LEGO** posicionada en el centro de la cinta, alineada con el eje óptico de la cámara cenital `(0, 0, Z_pieza)`.

**Algoritmo de colocación:**
1. Se selecciona una pieza aleatoria de `selected_parts`
2. Se selecciona una pose estable filtrada (ver §6)
3. Se aplica rotación Z aleatoria (`0°–360°`)
4. Se posiciona en `(0.0, 0.0, 0.0)` con snap al belt (`z = -min_z + 0.02`)
5. Se renderiza simultáneamente desde ambas cámaras
6. Ratio de frames vacíos: 5% para robustez del detector YOLO

### 5.2 Dataset Cenital

- **Target:** 5000 imágenes de piezas individuales
- **Salida:** `data/yolo_cenital/images/` + `data/yolo_cenital/labels/`

### 5.3 Dataset Lateral

- **Target:** 3000 imágenes de piezas individuales
- **Salida:** `data/yolo_lateral/images/` + `data/yolo_lateral/labels/`

### 5.4 Entrenamiento

Se entrena un modelo YOLOv11-nano **por cámara** (clase única: `lego_piece`):
- `models/yolo_cenital.pt`
- `models/yolo_lateral.pt`

---

## 6. Filtro de Estabilidad de Poses

- **Threshold:** `render_min_stability = 0.01` (umbral muy permisivo)
- **Filtro dimensional de base de contacto:** Se descarta cualquier pose si el ancho o largo de la superficie de contacto es **≤ 4.0 mm** (`min_contact_dimension_mm = 4.0`). Esto elimina poses sobre cantos finos (lateral de placas = 3.2 mm).
- **Fallback geométrico:** Si ninguna pose pasa el filtro, se usan las que tienen `face_class in (Top, Bottom)`
- **Último recurso:** Si la pieza no está en cache, se usa la pose identidad (studs arriba, quaternion `[1, 0, 0, 0]`)

---

## 7. Generación de Referencias DINOv2

**Objetivo:** Poblar la base de datos vectorial para clasificación KNN.

### 7.1 Estrategia: Pieza Única Centrada (ambas cámaras)

Para cada pieza × color × pose estable:
- 1 pieza sola en el centro de la escena `(0, 0, 0)`
- 12 rotaciones Z (pasos de 30°) por pose estable
- Se renderiza **1 imagen por cámara** (cenital + lateral) por rotación
- Color real del set 75078-1

### 7.2 Nomenclatura de archivos

```
Cenital:  data/dinov2_refs/cenital/ref_{PART}_{COLORHEX}_pose{NN}_rot{DDD}.png
Lateral:  data/dinov2_refs/lateral/ref_{PART}_{COLORHEX}_pose{NN}_rot{DDD}.png
```

---

## 8. Pipeline de Inferencia

```
Entrada: frame_cenital, frame_lateral

Paso 1: Detección YOLO
  └─ YOLO cenital → bbox_cenital (1 pieza)
  └─ YOLO lateral → bbox_lateral (1 pieza)

Paso 2: Recorte (Crop)
  └─ crop_cenital = frame_cenital[bbox_cenital]
  └─ crop_lateral = frame_lateral[bbox_lateral]

Paso 3: Extracción de Embeddings DINOv2
  └─ emb_cenital = DINOv2(crop_cenital)
  └─ emb_lateral = DINOv2(crop_lateral)

Paso 4: Algoritmo de Decisión en Cascada
  Phase 1: Gating de Color (Cenital)
    └─ Clasificación cromaticidad del crop cenital
    └─ Mapeo: código 84 → 85 (variación tonal)
    └─ Salida: lista de refs con mismo color en el set

  Phase 2: Gating de Superficie Cenital (±15%)
    └─ MinAreaRect del contorno segmentado (cenital)
    └─ Calibración perspectiva: area_aparente = area_nom × (150/(150−h))²
    └─ Tres configuraciones: flat, side, stand

  Phase 3: Gating de Altura Lateral (±15%)
    └─ Medición de altura en px → conversión a mm
    └─ Validación contra alturas nominales (H+0.9, H, W, L)

  Phase 4: Fusión DINOv2 (Cenital 70% + Lateral 30%)
    └─ KNN sobre embeddings para piezas que pasaron filtros
    └─ Score combinado: 0.7×S_cenital + 0.3×S_lateral
    └─ Predicción: pieza con score combinado máximo
```

---

## 9. Piezas del Set (75078-1)

| Ref    | Nombre                           | Dims (L×W×H mm) | Color (set) |
|--------|----------------------------------|-----------------|-------------|
| `3005` | Brick 1×1                        | 8×8×9.6         | LBGray      |
| `3001` | Brick 2×4                        | 32×16×9.6       | LBGray      |
| `3039` | Slope 45° 2×2                    | 16×16×9.6       | LBGray      |
| `3665` | Slope Inv. 45° 1×2               | 16×8×9.6        | LBGray      |
| `3010` | Brick 1×4                        | 32×8×9.6        | LBGray      |
| `3003` | Brick 2×2                        | 16×16×9.6       | LBGray      |
| `3020` | Plate 2×4                        | 32×16×3.2       | LBGray      |
| `4070` | Brick Mod. 1×1 stud lateral      | 8×8×9.6         | LBGray      |
| `4032` | Plate 2×2 Round                  | 16×16×3.2       | LBGray      |
| `3700` | Technic Brick 1×2 (con agujero)  | 16×8×9.6        | LBGray      |

> **Nota:** Se procesan 20 piezas individuales del inventario del set. Minifiguras excluidas.

---

## 10. Estructura del Proyecto

```
2camaras_pieza_unica/                    ← RAÍZ DEL ESCENARIO
├── config.yaml                          ← Configuración centralizada
├── config_loader.py                     ← Loader YAML para cfg.*
├── logger.py                            ← Sistema de logging
├── SETUP_DOC.md                         ← Este documento
├── requirements.txt
│
├── scripts/
│   ├── generate_yolo_training_dataset.py    # Dataset YOLO (cenital + lateral)
│   ├── generate_eevee_dinov2_refs.py        # Referencias DINOv2 (cenital + lateral)
│   ├── reindex_dinov2_eevee.py              # Reindexación embeddings en BD
│   ├── run_evaluation.py                    # Evaluación del pipeline
│   ├── generate_synthetic_set.py            # Utilidades de escena Blender
│   ├── generate_synthetic_dataset.py        # Utilidades de malla
│   ├── scene_config.py                      # Constantes de escena
│   └── simulate_stable_poses.py            # Simulación de poses estables
│
├── database/
│   ├── set_catalog.py                       # Catálogo piezas/colores del set 75078-1
│   ├── color_catalog.json                   # Colores LEGO disponibles
│   └── supabase_client.py                   # Cliente BD vectorial
│
├── models/                                  ← Pesos de modelos entrenados
│   ├── yolo_cenital.pt
│   └── yolo_lateral.pt
│
├── data/                                    ← TODOS los datos del escenario
│   ├── stable_poses_cache.json              # Cache poses simuladas
│   ├── yolo_cenital/                        # images/ + labels/
│   ├── yolo_lateral/                        # images/ + labels/
│   ├── yolo_cenital_split/                  # train/ + val/ + dataset.yaml
│   ├── yolo_lateral_split/                  # train/ + val/ + dataset.yaml
│   └── dinov2_refs/                         # cenital/ + lateral/
│
└── logs/                                    ← Logs de ejecución
```

---

## 11. Comandos de Ejecución

> **Nota:** Todos los comandos se ejecutan desde la raíz de `LegoVision/`.

### Generación de Dataset YOLO (ambas cámaras simultáneamente)

```bash
blender -b -P 2camaras_pieza_unica/scripts/generate_yolo_training_dataset.py -- \
    --camera cenital --output_dir 2camaras_pieza_unica/data/yolo_cenital

blender -b -P 2camaras_pieza_unica/scripts/generate_yolo_training_dataset.py -- \
    --camera lateral --output_dir 2camaras_pieza_unica/data/yolo_lateral
```

### Generación de Referencias DINOv2

```bash
blender -b -P 2camaras_pieza_unica/scripts/generate_eevee_dinov2_refs.py -- \
    --output_dir 2camaras_pieza_unica/data/dinov2_refs \
    --rotations 12
```

### Reindexación de Embeddings

```bash
.venv/bin/python 2camaras_pieza_unica/scripts/reindex_dinov2_eevee.py \
    --ref_dir 2camaras_pieza_unica/data/dinov2_refs
```

### Evaluación del Pipeline

```bash
.venv/bin/python 2camaras_pieza_unica/scripts/run_evaluation.py
```

---

## 12. Sistema de Logs

Los logs se almacenan en `2camaras_pieza_unica/logs/`:
- `blender_YYYY-MM-DD.log` — Renders Blender
- `dinov2_YYYY-MM-DD.log` — Indexación embeddings
- `pipeline_YYYY-MM-DD.log` — Evaluación del pipeline
- `yolo_YYYY-MM-DD.log` — Generación dataset YOLO

Configuración en `config.yaml` sección `logging`.

---

## 13. Report de Diagnóstico por Pieza

Script: `scripts/generate_piece_report.py`

Genera un report HTML standalone de diagnóstico completo para una pieza específica, usando renders existentes del test set.

### Uso

```bash
# Report para pieza 3001
.venv/bin/python 2camaras_pieza_unica/scripts/generate_piece_report.py --ref 3001

# Report para pieza 3020, pose específica
.venv/bin/python 2camaras_pieza_unica/scripts/generate_piece_report.py --ref 3020 --pose 1
```

### Output
```
data/reports/report_{REF}_pose{NN}.html
```

### Contenido del report (7 secciones)

1. **📷 Renders** — Imagen cenital y lateral de la pieza
2. **🔲 Bounding Boxes** — Cenital y lateral con bbox dibujado
3. **🎨 Color** — Color detectado (cenital/lateral) vs color real del catálogo
4. **📐 Superficie Cenital** — Estimada vs nominal (con magnificación perspectiva)
5. **📏 Altura Lateral** — Estimada vs real de la pose estable
6. **🧠 DINOv2** — Top-3 clasificación cenital, lateral, y fusión 70/30
7. **ℹ️ Metadatos** — Archivos fuente, embeddings en BD, datos de pose

---

## Changelog

| Fecha | Cambio | Archivos |
|-------|--------|----------|
| 2026-06-06 | Creación inicial del setup 2camaras_pieza_unica | config.yaml, SETUP_DOC.md, todos los scripts |
| 2026-06-06 | Pipeline ejecutado: 1000 frames cenital + 1000 lateral + YOLO trained + DINOv2 refs + test set 100 | data/*, models/* |
| 2026-06-06 | Añadido script de report de diagnóstico por pieza | scripts/generate_piece_report.py |
| 2026-06-06 | Creado .clinerules para documentación automática | .clinerules (raíz) |
