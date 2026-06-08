# SETUP_DOC — LegoVision: 2camaras_random_pieza_unica

> **Iteración:** 2camaras_random_pieza_unica | **Fecha:** 2026-08-06
> **Ubicación:** `LegoVision/2camaras_random_pieza_unica/`
> **Basado en:** `2camaras_pieza_unica` (extracción del pipeline `random_position`)

---

## 1. Descripción General

Este proyecto es un **subset focalizado** de `2camaras_pieza_unica` cuyo propósito
exclusivo es:

1. **Renderizar** una pieza única del set 75078-1 en una **posición aleatoria**
   dentro del FOV de la cámara cenital (no centrada), respetando un margen mínimo
   de 5 mm a los bordes y validando que la pieza también sea visible por la cámara
   lateral.
2. **Ejecutar inferencia** sobre esos renders (YOLO + SAM + DINOv2 + cascada de
   gating) y producir reports CSV con métricas de área y altura, además de la
   métrica de acierto/fallo de clasificación.

### Diferencias respecto al proyecto padre

| Aspecto | `2camaras_pieza_unica` | `2camaras_random_pieza_unica` |
|---------|------------------------|-------------------------------|
| Posición pieza | Centro de cinta (0,0) | **Aleatoria en FOV cenital** |
| Foco | Pipeline completo (entrenamiento YOLO, refs DINOv2, eval) | **Solo render+inferencia random_position** |
| Datasets YOLO | Sí (5000+3000 imgs) | No (solo se reutilizan los modelos entrenados) |
| Test set | Múltiples | **76 muestras random_position** |

---

## 2. Estructura del Proyecto

```
2camaras_random_pieza_unica/
├── config.yaml                              # Configuración centralizada
├── config_loader.py                         # Loader YAML (cfg.*)
├── logger.py                                # Sistema de logging
├── requirements.txt
├── mobile_sam.pt                            # Modelo SAM
├── run_pipeline.sh
├── SETUP_DOC.md                             # Este documento
│
├── models/
│   ├── yolo_cenital.pt                      # YOLO entrenado (cenital)
│   ├── yolo_lateral.pt                      # YOLO entrenado (lateral)
│   └── dino_metric_head.pt                  # Head DINOv2 unimodal
│
├── database/
│   ├── color_catalog.json
│   ├── set_catalog.py                       # Catálogo set 75078-1
│   └── supabase_client.py                   # Cliente Supabase (embeddings)
│
├── data/
│   ├── stable_poses_cache.json              # Cache poses estables
│   ├── reports/                             # CSVs salida del report
│   └── random_position/                     # 76 renders + metadata
│       ├── random_position_metadata.json
│       └── sample_<ref>_{cenital,lateral}.png
│
├── scripts/
│   ├── generate_set_random_position.py      # ⭐ Script de RENDER
│   ├── generate_random_position_report.py   # ⭐ Script de INFERENCIA
│   ├── _pose_utils.py                       # Utils TARPS / poses
│   ├── scene_config.py                      # Constantes de escena
│   ├── ldraw_mesh_parser.py                 # Parser meshes LDraw
│   ├── generate_synthetic_set.py            # Materiales, GPU, físicas, fallback meshes
│   ├── generate_synthetic_dataset.py        # Helpers de mesh
│   ├── generate_test_set.py                 # Belt, floor, lightbox, cámaras
│   ├── simulate_stable_poses.py             # (Re)generar cache poses
│   ├── run_evaluation.py                    # Funciones de inferencia (SAM/KNN/etc.)
│   ├── generate_eevee_dinov2_refs.py        # Generar referencias DINOv2
│   └── reindex_dinov2_eevee.py              # Reindexar embeddings
│
└── logs/
```

---

## 3. Comandos de Ejecución

> **Nota:** Todos los comandos se ejecutan desde la raíz de `LegoVision/`.

### 3.1 Generación de renders en posición aleatoria

```bash
/opt/homebrew/bin/blender -b -P \
    2camaras_random_pieza_unica/scripts/generate_set_random_position.py
```

Salida:
- `2camaras_random_pieza_unica/data/random_position/sample_<ref>_cenital.png`
- `2camaras_random_pieza_unica/data/random_position/sample_<ref>_lateral.png`
- `2camaras_random_pieza_unica/data/random_position/random_position_metadata.json`

### 3.2 Inferencia y reports

```bash
.venv/bin/python \
    2camaras_random_pieza_unica/scripts/generate_random_position_report.py
```

Salida:
- `2camaras_random_pieza_unica/data/reports/random_position_areas.csv`
- `2camaras_random_pieza_unica/data/reports/random_position_heights.csv`
- Resumen por consola con aciertos/fallos por pieza.

### 3.3 (Opcional) Regenerar referencias DINOv2

```bash
blender -b -P \
    2camaras_random_pieza_unica/scripts/generate_eevee_dinov2_refs.py -- \
    --output_dir 2camaras_random_pieza_unica/data/dinov2_refs --rotations 12

.venv/bin/python \
    2camaras_random_pieza_unica/scripts/reindex_dinov2_eevee.py \
    --ref_dir 2camaras_random_pieza_unica/data/dinov2_refs
```

### 3.4 (Opcional) Regenerar cache de poses estables

```bash
blender -b -P \
    2camaras_random_pieza_unica/scripts/simulate_stable_poses.py
```

---

## 4. Pipeline de Inferencia (cascada de 4 fases)

Idéntico a `2camaras_pieza_unica` (ver §8 de su SETUP_DOC):

1. **YOLO** detecta bbox cenital + lateral
2. **SAM** segmenta la pieza dentro de cada bbox
3. **Phase 1**: gating de color (cromaticidad, cenital)
4. **Phase 2**: gating de superficie cenital (±20%, con corrección perspectiva)
5. **Phase 3**: gating de altura lateral (±15%)
6. **Phase 4**: fusión DINOv2 (cenital 70% + lateral 30%) sobre KNN

---

## 5. Reports CSV producidos

### 5.1 `random_position_areas.csv`

| Columna | Descripción |
|---------|-------------|
| `pieza` | Ref (e.g. `3001`) |
| `pose` | Índice de pose estable usada |
| `surface_silhouette_mm2` | Área silueta cenital (cache, ground-truth) |
| `surface_convex_hull_mm2` | Área convex hull cenital (cache) |
| `surface_estimated_inference_mm2` | Área estimada por inferencia (SAM + perspectiva) |
| `err_silh_pct` | Error % vs silueta |
| `err_convex_pct` | Error % vs convex hull |

### 5.2 `random_position_heights.csv`

| Columna | Descripción |
|---------|-------------|
| `render_lateral` | Path del PNG lateral |
| `pieza` | Ref |
| `pose` | Índice de pose |
| `height_stable_pose_mm` | Altura ground-truth desde cache |
| `height_estimated_inference_mm` | Altura estimada (SAM, lateral) |
| `err_pct` | Error % |

---

## 6. Dependencias críticas

- **`stable_poses_cache.json`** — necesario tanto para render (selección de
  pose TARPS) como para report (ground-truth de área/altura).
- **`mobile_sam.pt`** — segmentación.
- **`models/yolo_*.pt`** — detección.
- **`models/dino_metric_head.pt`** — head proyectivo DINOv2.
- **Supabase** — la inferencia DINOv2 carga los embeddings de referencia
  desde la tabla `piece_embeddings`. Asegúrate de que las variables `.env` de
  Supabase estén configuradas en la raíz del repo.
- **Inference modules** — `LegoVision/inference/knn_classifier.py`. El script
  `generate_random_position_report.py` añade `LEGOVISION_ROOT` a `sys.path`
  para resolver este import.

---

## Changelog

| Fecha | Cambio |
|-------|--------|
| 2026-08-06 | Creación del proyecto a partir de `2camaras_pieza_unica`: copia de configuración/modelos/database/scripts, y movimiento de `data/random_position/` |