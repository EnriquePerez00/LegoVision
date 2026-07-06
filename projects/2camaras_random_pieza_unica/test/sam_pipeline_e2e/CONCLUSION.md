# SAM Pipeline E2E — Conclusión

**Fecha**: 2026-06-09  
**Pieza**: 3023 (Plate 1×2)  
**Color**: Red `#C30025`  
**Refs canónicas**: 48 entries × 2 cámaras = 96 renders (escena canónica, pieza en (0,0,0)).  
**Queries**: 5 muestras (escena canónica, pieza en posición XY aleatoria).

---

## Pipeline simétrico aplicado a refs y queries

Para CADA imagen (cenital o lateral, ref o query):

```
imagen RGB
  ├─► bbox_norm proyectado en metadata        (refs: get_2d_bbox de Blender;
  │                                             queries: bbox_norm de generate_15_random_focus)
  ├─► expand_bbox_norm(pad_px=4)              (margen para SAM)
  ├─► MobileSAM(image, bbox_px=4_lados)        → mask uint8 HxW
  ├─► crop por bbox + reemplazo de !mask      → fondo CINTA (37,65,84)
  ├─► canvas 224×224 con fondo cinta + 8px margin (FIT-TO-CANVAS)
  └─► DINOv2 ViT-S/14 → CLS token 384-d → L2
```

Comparación: cosine similarity por cámara entre cada query y todas las refs de esa cámara.

---

## Resultados

| Cámara  | n | top1 mean   | top1 min | top1 max | top1 correct | max_sim_mean |
|---------|---|-------------|----------|----------|--------------|--------------|
| Cenital | 5 | **0.8002**  | 0.5963   | 0.8841   | 5/5 (100%)   | 0.8002       |
| Lateral | 5 | **0.8936**  | 0.8229   | 0.9413   | 5/5 (100%)   | 0.8936       |

### Per-sample

| Sample | pos_bu          | cenital top-1 sim | lateral top-1 sim |
|--------|-----------------|-------------------|-------------------|
| 0      | (-0.43, -0.53)  | 0.8412            | 0.8229            |
| 1      | (-0.53,  0.01)  | 0.8307            | **0.9182**        |
| 2      | ( 0.41,  0.38)  | 0.8489            | **0.9251**        |
| 3      | ( 0.17,  0.59)  | 0.8841            | **0.9413**        |
| 4      | (-0.65, -0.15)  | 0.5963 ⚠          | 0.8606            |

---

## Comparativa con experimentos anteriores

| Pipeline                                                                            | top1 mean cen | top1 mean lat |
|-------------------------------------------------------------------------------------|---------------|---------------|
| v0  refs LEGACY (lab MV) + queries V4 + canvas negro scale=0.325                    | (no medido — accuracy global ~62-70% en set 300)  | — |
| v1  refs CANÓNICAS + queries V4 + canvas negro scale=0.325 (sin SAM)                | 0.8129        | **0.4449**    |
| **v2 refs CANÓNICAS + queries V4 + bbox+SAM+cinta+fit-to-canvas (este test)**       | 0.8002        | **0.8936** ✓  |

**Δ lateral**: 0.4449 → 0.8936 (**+0.45**, +101%). El cambio determinante es:

1. **fit-to-canvas** (la pieza se escala para ocupar ≥208 px en el canvas 224, vs los ~10 px que ocupaba con `scale_factor=0.325`).
2. **fondo CINTA uniforme tras SAM** elimina el dominio de la pantalla aluminio mate (que con cinta natural ocupaba el 75% del frame lateral) y la pieza es ahora la señal dominante para DINOv2.
3. **simetría refs↔queries**: el mismo procesamiento garantiza que ambos embeddings vivan en el mismo dominio visual.

## Observaciones puntuales

- **Sample 4 cenital baja a 0.60** (único outlier). Está en `pos=(-0.65, -0.15)` BU, casi en el borde del FOV cenital; la silueta se proyecta con perspectiva más oblicua que cualquier ref (todas ellas en el centro). Es una limitación esperada del setup: para offsets grandes, la pose aparente cambia. La fix natural es generar refs con varios offsets XY en futuras iteraciones.
- **Pose 0 vs Pose 1** del cache son ambas `face=Side` con `contact_normal` opuesto (±Y). El matching tiende a preferir la pose 0 incluso para queries con pose 1 cuando la rotación Z compensa la simetría — esto es correcto: las dos poses producen siluetas casi idénticas.
- **Lateral SUPERA a cenital** ahora porque la pieza es más alargada en lateral (X visible, contraste con cinta) y el fit-to-canvas la escala bien. En cenital, la silueta es prácticamente cuadrada-pequeña y DINOv2 tiene menos features distintivas a procesar.

## Hipótesis del usuario — VERIFICADA

> *"Quiero que en la generación de renders para DINOv2 y en el proceso de inferencia, el proceso sea el mismo: bbox + SAM crop + fondo azul petróleo + vectorización."*

**SÍ**: la simetría de pipeline (mismo procesamiento para refs y queries) sube la similitud lateral de 0.44 a 0.89 y mantiene la cenital en ~0.80. La hipótesis fuerte (≥0.85) **se cumple en lateral** y **roza en cenital** (con un único sample ofensor).

## Recomendaciones para el pipeline real (`run_evaluation.py`)

1. **Refs DINOv2**: regenerar con `generate_canonical_dinov2_refs.py --refs all` (escena canónica + (0,0,0) + metadata.json con bbox).
2. **Indexación**: aplicar el pipeline simétrico SAM al indexar (vs el `index_synthetic_renders.preprocess_render` actual que solo recolora fondo cinta sin máscara). Modificar `index_synthetic_renders.py` para que:
   - Lea el `metadata.json` con bbox por render.
   - Haga SAM crop + cinta + canvas fit-to-canvas, igual que aquí.
3. **Inferencia (`run_evaluation.py`)**: el pipeline ya hace SAM y bbox YOLO. Cambios mínimos:
   - Reemplazar `apply_sam_mask_to_crop` (fondo negro) por una variante con fondo CINTA (37,65,84).
   - Reemplazar el canvas con `scale_factor=0.325` por **fit-to-canvas con margen 8 px** (igual que `index_synthetic_renders.preprocess_render`, ya existe).
4. **Validación masiva**: tras los cambios, ejecutar `run_evaluation.py` sobre el set 300 y comparar accuracy global vs el baseline actual (~62-70%).

## Artefactos producidos en este test

- `refs_sam/{cenital,lateral}/*.png` — 96 canvases procesados (refs).
- `queries_sam/sample_NN_{cenital,lateral}.png` — 10 canvases procesados (queries).
- `embeddings.npz` — embeddings 384-d de las 96 refs (`ref_cen`, `ref_lat`).
- `sam_pipeline_report.json` — top-K + stats por sample/cámara.
- `sam_pipeline_report.html` — visualización lado a lado: query original, query canvas SAM, top-3 refs (canvas SAM + original).
- `CONCLUSION.md` — este documento.