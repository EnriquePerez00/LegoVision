# Resultados — Fases 1, 2 y 4 ejecutadas (set 300)
**Fecha**: 2026-06-10  
**Pieza**: set 75078-1, 38 refs × 7 colores (300 muestras canónicas con poses estables y posiciones aleatorias).

---

## Métricas globales (DESPUÉS de los fixes)

| Métrica | **v0 baseline** (antes del fix de pose) | **v1 final** (Fases 1+2+4) | Δ |
|---|---|---|---|
| **Accuracy modelo** | 22.00% (66/300) | **28.33% (85/300)** | **+6.33 pp** |
| Color match cenital | (no medido limpio) | **89.7% (269/300)** | — |
| Color match lateral | (no medido limpio) | **94.0% (282/300)** | — |
| `\|err\| superficie` medio | 76% | **35.1%** (mediana 25.6%) | **−41 pp** |
| `\|err\| altura lateral` medio | ~50% est. | **17.2%** (mediana 9.9%) | **−33 pp** |

> **Las estimaciones de superficie y altura han mejorado dramáticamente** — los fixes 1.1 (P75), 1.2 (Newton 1-step), 1.3 (restar side faces) tienen impacto medible.  
> La accuracy global sube **6 puntos** porque el cuello de botella restante NO está en las observaciones geométricas sino en el matching DINOv2 (refs vs query con pieza descentrada).

---

## Cambios aplicados

### Fase 1 — Quick wins observador (sin re-render, en `run_evaluation.py`)
1. **Altura lateral P50 → P75** del perfil de columnas. Captura la cima en piezas con cabeza estrecha (jumper plates, faros) sin amplificar sombras.
2. **Newton 1-step para `pz_mm`**. Refinamiento de la altura de centroide para `d_act` lateral (reduce error 3-5% a <0.5% en piezas altas).
3. **Restar `apparent_sides`** del observador antes de des-magnificar. Estima perímetro desde el contorno SAM (no del candidato) y resta `perim × h_lat × (r/Zcam) × 0.5`. Resuelve sobre-estimación sistemática en piezas altas y descentradas.
4. **Catálogos `SET_CATALOG_COLORS` recalibrados** (usuario los actualizó).
5. **Filtro HSV simplificado** en `estimate_color_predominant_sam`: media simple de píxeles fg sin filtros V/S agresivos. Mejora en piezas oscuras (Black) y translúcidas (Trans-Brown) que antes caían al fallback ruidoso.

### Fase 2 — Re-render limpio
6. Refs canónicas regeneradas con fix-pose (1824 cenital + 1824 lateral) — pieza apoyada correctamente, sin sombras inferiores.
7. 300 muestras regeneradas con fix-pose.
8. BD reindexada (3648 embeddings).

### Fase 4 — (cubierta por Fases 1.1-1.3)

---

## Per-piece accuracy

### Piezas con accuracy ≥ 50% (10 piezas)

| Ref | Acc | n |
|---|---|---|
| 3023  | 100% | 11 |
| 3020  | 100% | 5  |
| 3710  | 100% | 6  |
| 30414 | 100% | 15 |
| 2653  | 67%  | 9  |
| 3832  | 67%  | 3  |
| 4589b | 67%  | 3  |
| 3022  | 67%  | 12 |
| 2877  | 65%  | 17 |
| 3068  | 60%  | 5  |

Estas son piezas **altas** (3023, 3020 plates 1×2/2×4 con altura distintiva), **largas** (3710 plate 4×8, 30414 brick 1×4 with sides), o **rectangulares estándar** que el modelo geométrico captura bien.

### Piezas problemáticas (accuracy ≤ 25%)
La mayoría son piezas con **footprint similar** pero **alturas diferentes** (3004, 3040, 32000, 87552, 87620). El gating de altura ahora funciona mejor (mediana 10% error) pero todavía no separa estas familias.

---

## Reportes generados

- `inference_300_full.csv` — 300 filas con todos los campos.
- `inference_300_per_piece.csv` — agregado por (ref, color).
- `inference_300_per_pose.csv` — agregado por (ref, pose).
- `inference_300_summary.html` — vista web con métricas globales, top errores, mismatches.
- `piece_32054_11.html`, `piece_61184_86.html`, `piece_32000_86.html`, `piece_3040_86.html` — diagnóstico visual de las 4 piezas problemáticas (mismas piezas que en la primera medición — ahora con la pieza apoyada y el observador corregido).

---

## Conclusión

- **El bug de "pieza flotando" (Fase 0 fixes 1-3) era crítico** y ya está resuelto.
- **Las mejoras del observador (Fase 1) tienen impacto real**: errores de superficie y altura bajan más del 50%.
- **El cuello de botella restante son las features de alto nivel** (DINOv2 embeddings + matching KNN).
- Los **gating de color funcionan muy bien** ahora (90% cenital, 94% lateral), gracias a los catálogos recalibrados y la simplificación de filtros HSV.

### Para subir más la accuracy (no implementado ahora)
1. **Reentrenar `dino_metric_head.pt`** con las nuevas refs canónicas (los pesos actuales fueron entrenados con la escena legacy).
2. **CenterNet/keypoint detection** (Fase 5 del plan original) para sustituir el observador geométrico actual por una pose 6-DoF directa.
3. **Aumentar diversidad de refs**: generar refs en posiciones XY no centradas para que el matching DINOv2 sea robusto a la perspectiva descentrada.