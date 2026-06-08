# 🚀 OPTIMIZACIONES IMPLEMENTADAS - 2camaras_random_pieza_unica

**Fecha:** 8 Junio 2026
**Hardware:** Apple M4 Pro (12 cores, 48GB RAM)

## ✅ TODAS LAS OPCIONES IMPLEMENTADAS

### Opción A: Paralelización (4 Workers Dinámicos)
- `scripts/run_parallel_render.sh` (146 líneas)
- `scripts/generate_yolo_training_dataset_parallel.py` (482 líneas)
- `scripts/merge_worker_metadata.py` (76 líneas)
- Detección dinámica de recursos (mantiene 20% CPU/RAM libre)
- Reduce a 3 workers si recursos insuficientes
- **Speedup: 4-5x**

### Opción B: Optimización de Rendering
- B1: TAA samples 16 → 8
- B2: Pre-cargar las 38 meshes UNA VEZ (elimina ~600ms/frame de overhead)
- B3: Desactivar bloom/SSR/AO en EEVEE
- **Speedup combinado: 50-60%**

### Opción C: YOLO Training Dinámico (MPS + Workers)
- `training/train_yolo.py` refactorizado
- Función `detect_optimal_training_config()`:
  - workers=8 (con 12 cores y 20% reservado)
  - batch_size=32 (con 48GB RAM)
  - device='mps' (M4 Pro)
- **Speedup: 2-3x**

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
