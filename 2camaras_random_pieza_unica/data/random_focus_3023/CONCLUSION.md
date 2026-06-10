# Validación Canonical Match — `3023` (Plate 1×2 Red)

**Fecha**: 2026-06-09  
**Pieza**: 3023 (Plate 1×2)  
**Color**: Red `#C30025`  
**Set**: 75078-1  
**Refs canónicas indexadas**: 96 (48 cenital + 48 lateral) — 2 poses × 2 colores × 12 rotaciones × 2 cámaras  
**Samples de inferencia**: 5 (poses+posiciones aleatorias en FOV)

---

## Resultados

| Cámara  | n  | top1 mean | top1 min | top1 max | top1 correct |
|---------|----|-----------|----------|----------|--------------|
| Cenital | 5  | **0.8129** | 0.7908   | 0.8226   | 5/5 (100%)   |
| Lateral | 5  | **0.4449** | 0.4133   | 0.4878   | 5/5 (100%)   |

- **Cenital ✓** roza la hipótesis fuerte (objetivo 0.85). Las top-1 siempre identifican `3023` correctamente.
- **Lateral ✗** la similitud es BAJA aunque el top-1 también identifica correctamente la pieza (no hay distractor en BD: solo `3023`, así que esto último no es estadísticamente significativo).

## Diagnóstico

### Cenital (OK, ~0.81)
- La pieza ocupa **15-22 px en imagen 384×384** (~0.35-0.42% del frame).
- Las refs (en `0,0`) y queries (en posición XY aleatoria) tienen tamaños comparables porque la cámara cenital es ortográfica-aproximada (Z fijo).
- El fondo dominante es la cinta azul petróleo `(91,114,127)` en ambos casos.
- DINOv2 captura forma y color con suficiente fidelidad.
- **Margen de mejora**: la pose está siempre cerca del centro → la perspectiva entre ref y query difiere poco. Si la pieza está cerca del borde, la magnificación cambia ligeramente y la similitud baja a ~0.79 (sample 4 con `pos=(-0.65,-0.15)`).

### Lateral (DÉBIL, ~0.44)
**El bbox lateral es minúsculo**: 

| Sample | pos_bu       | bbox lateral (norm) | size_px |
|--------|--------------|---------------------|---------|
| 0 | (-0.43, -0.53) | 0.039×0.026 | ~15×10 px |
| 1 | (-0.53, +0.01) | 0.065×0.023 | ~25×9 px  |
| 2 | (+0.41, +0.38) | 0.119×0.055 | ~46×21 px |
| 3 | (+0.17, +0.59) | 0.107×0.036 | ~41×14 px |
| 4 | (-0.65, -0.15) | 0.057×0.020 | ~22×8 px  |

Causas:
1. **Distancia mayor** en lateral cuando la pieza tiene `X<0` (lejos de la cámara lateral en X=+1.5 BU): un plate 1×2 de 16×3.2 mm queda cubierto por solo 8-15 pixeles de altura.
2. **Pantalla aluminio mate detrás** ocupa el 75% del frame, dominando todo lo que DINOv2 percibe del crop+canvas.
3. **Crop por bbox**: el bbox lateral es muy delgado → al pegarlo en canvas 224×224 con `scale_factor=0.325` y fondo cinta, la pieza queda como un puntito de 4-12 px en una zona enorme de cinta. DINOv2 ve sobre todo fondo, no detalle.
4. **Tono de la pantalla aluminio en queries** difiere sutilmente entre samples por las reflejos especulares y la posición XY.

## Conclusión sobre la hipótesis del usuario

> *"Quiero que los renders para los embeddings DINOv2 se generen con las mismas condiciones y parametrización que los que se usan para las imágenes canónicas, exceptuando la posición de la pieza que siempre es en (0,0,0)."*

**Hipótesis CONFIRMADA en cenital**: con escena idéntica, los embeddings DINOv2 cenitales tienen alta similaridad (0.79-0.82) sin necesidad de máscara SAM ni alineación adicional. Dado que el pipeline real anteriormente comparaba refs con escena MV-lab (Ring+Bars + fondo negro) contra queries con escena V4 canónica (cinta + ambient), **la unificación de escena resuelve la mitad principal del desajuste**.

**Hipótesis NECESITA AJUSTES en lateral**: la cámara lateral en su configuración canónica genera bboxes demasiado pequeños cuando la pieza no está en el centro. La similitud queda en torno a 0.44, claramente inferior al objetivo 0.85.

## Recomendaciones (próximos pasos sugeridos)

1. **Regenerar TODAS las refs canónicas** (38 piezas × N poses × 12 rots) y validar accuracy global. Cenital ya gana mucho.
2. **Para mejorar lateral**:
   - Opción L1 — aplicar **máscara SAM también en refs** (`film_transparent` + fondo cinta sólido reinyectado). Así cuando el query también enmascare, el contexto fuera de la pieza es idéntico a la cinta uniforme.
   - Opción L2 — **acercar la cámara lateral** o reducir su FOV para que la pieza ocupe ~50-100 px (vs los 8-15 actuales). Implica recalibrar `px_per_mm_lateral` en config.
   - Opción L3 — **renderizar refs laterales en posiciones múltiples** (no solo `0,0` sino también `±0.5,±0.5` BU) para cubrir el rango de offsets que produce la inferencia.
3. **Validar con más samples** (50+) y con piezas de geometría distinta (3024, 32000, 60481, 87620 que tienen alturas distintas).
4. **Considerar reentrenar el MLP `dino_metric_head.pt`** con las nuevas refs (los pesos actuales fueron entrenados con la escena legacy).

## Artefactos

- `random_focus_metadata.json`  — metadata de los 5 samples generados.
- `canonical_match_report.json`  — top-K + stats por sample y cámara.
- `canonical_match_report.html` — visualización lado a lado query/refs.