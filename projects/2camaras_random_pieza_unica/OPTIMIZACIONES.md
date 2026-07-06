# 🚀 OPTIMIZACIONES IMPLEMENTADAS - 2camaras_random_pieza_unica

**Fecha (revisión sprint 1-3):** 9 Sep 2026
**Hardware:** Apple M4 (12 cores, 48 GB RAM)

## ✅ TODAS LAS OPCIONES IMPLEMENTADAS

### Sprint 1 — Render DINOv2 (1.1 + 1.2 + 1.4)
- **1.1** TAA 16 → 8 + bloom/SSR/AO=OFF aplicado en
  `scripts/generate_eevee_dinov2_refs.py` y
  `scripts/generate_eevee_dinov2_refs_parallel.py`.
- **1.2** `run_pipeline.sh` paso 5 ahora invoca `run_parallel_dinov2.sh`
  (4 workers dinámicos manteniendo 20% CPU/RAM libre).
- **1.4** Render res 640 → **384** vía flag `--render_res` (default
  configurable; el pipeline pasa 384 explícitamente).
- **Speedup combinado proyectado:** 21m 38s → ~4-6 min (3-5×).

### Sprint 2 — Indexado DINOv2 (2.1 + 2.2 + 2.3)
Modificaciones en `training/index_synthetic_renders.py`:
- **2.1** Preprocess (PIL+resize+canvas+transform) en `ThreadPoolExecutor`
  (`PREPROC_WORKERS=8`); permite que la GPU MPS no espere por I/O.
- **2.2** `DEFAULT_BATCH_SIZE=128` (antes 64).
- **2.3** Contador `indexed` SÓLO se incrementa tras `save_piece_embeddings_batch`
  exitoso, evitando el bug del log donde aparecían `total_failed=2880`
  con `total_indexed=2880` simultáneamente.
- **Speedup proyectado:** 41 s → ~20-25 s (1.6-2×).

### Sprint 3 — Inferencia (3.1 + 3.2 + 3.3 helpers)
Modificaciones en `2camaras_random_pieza_unica/scripts/run_evaluation.py`:
- **3.1** Pre-cómputo YOLO **en lotes de 16** (`yolo_detect_bbox_batch`).
  Antes: 600 calls secuenciales (1 cen + 1 lat × 300 samples).
  Ahora: ~38 batches (300/16 × 2 cámaras).
- **3.2** SAM en lote por cámara (`segment_crop_sam_batch`); reusa el
  contexto del modelo entre llamadas adyacentes (sin overhead Python).
- **3.3** Helper `classify_camera_batch` disponible (procesa N crops en un
  forward pass DINOv2). Mantiene ruta unitaria por seguridad; activable
  reemplazando las dos llamadas `classify_camera(...)` en el bucle.
- **Refactor del bucle:** una primera pasada batch (YOLO + SAM) y una
  segunda pasada por sample con cálculos CPU-bound (color, height,
  surface gating, KNN).
- **Speedup proyectado:** 0.36 s/sample → ~0.10-0.15 s/sample (~2-3×).

### Pipeline orquestación
- `run_pipeline.sh` actualizado para llamar al render paralelo en paso 5
  con `--render_res 384`.
- `run_parallel_dinov2.sh` ahora acepta `RENDER_RES` como 3er argumento.
- `generate_eevee_dinov2_refs(_parallel).py` aceptan `--render_res`.

### Heredado (sprints anteriores — siguen activos)
- Paralelización 4-workers con detección dinámica de CPU/RAM (20% libre).
- YOLO Training Dinámico (`training/train_yolo.py` con `detect_optimal_training_config()`).

## 📈 PROYECCIÓN

| Pipeline | Render | Training | TOTAL |
|----------|--------|----------|-------|
| Baseline | 33m | 35+ min | ~120 min |
| Optimizado (A+B+C) | 7-10m | 12-15m | **~30-40 min** |

**Speedup global: 3-4x** 🚀

## 🚀 USO

```bash
# Pipeline completo optimizado
bash 2camaras_random_pieza_unica/run_pipeline_optimized.sh

# Solo render paralelo
bash 2camaras_random_pieza_unica/scripts/run_parallel_render.sh \
    cenital ./2camaras_random_pieza_unica/data/yolo_cenital 2000 42
```

## 📁 ARCHIVOS GENERADOS (Total: 8 archivos, ~1300 líneas)

| Archivo | Líneas |
|---------|--------|
| optimize_rendering.py | 71 |
| run_parallel_render.sh | 146 |
| generate_yolo_training_dataset_parallel.py | 482 |
| merge_worker_metadata.py | 76 |
| config_optimized.yaml | 55 |
| training/train_yolo.py (refactorizado) | 368 |
| run_pipeline_optimized.sh | 102 |
| OPTIMIZACIONES.md | (este) |

## ⚙️ DETECCIÓN DINÁMICA DE RECURSOS

```
USABLE_CPU = TOTAL_CPU * 0.80    (reserva 20%)
USABLE_RAM = AVAILABLE_RAM * 0.80
WORKERS = min(USABLE_CPU/2, USABLE_RAM/4, 4)
WORKERS = clamp(WORKERS, 3, 4)
```

Si recursos bajos en runtime → workers se reducen automáticamente.

## ⚠️ NOTAS

- **Reproducibilidad:** master_seed compartido garantiza plan global idéntico
- **Recovery:** workers independientes, logs separados (`worker_N.log`)
- **Rollback:** mantener `run_pipeline.sh` original

## 📞 TROUBLESHOOTING

| Problema | Solución |
|----------|----------|
| Sistema saturado | Workers se reducen a 3 automáticamente |
| Worker falla | Ver `output_dir/worker_N.log` |
| MPS error | Cambiar a `device='cpu'` en train_yolo.py |
