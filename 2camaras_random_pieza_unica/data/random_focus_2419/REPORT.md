# Test Random Pieza Única — Pieza `2419` (Light Bluish Gray `#A2A1A3`)

> Pipeline: 2 cámaras (cenital + lateral) · Set 75078-1
> Generado: 2026-06-09
> Fuente: `random_focus_2419/`

---

## 1. Resumen ejecutivo

| Métrica | Resultado |
|---|---|
| Pieza GT | **2419** (Slope, Inverted 33 3 x 1) |
| Color GT | Light Bluish Gray (`#A2A1A3`, code 86) |
| Poses estables disponibles en cache | **1** (única pose) |
| Muestras renderizadas | 15/15 |
| YOLO detecciones cenital | 15/15 |
| YOLO detecciones lateral | 15/15 |
| **Pipeline final accuracy (ref correcta)** | **0/15 = 0.00 %** |
| **DINOv2 puro — GT en top-3 cenital** | **0/15** |
| **DINOv2 puro — GT en top-3 lateral** | **0/15** |
| Pieza 2419 en BD de refs | ✅ Sí (12 embeddings, ambas cámaras, color exacto) |

> ⚠️ **Resultado anómalo:** la pieza GT **nunca aparece** ni en el top-3 DINOv2,
> a pesar de tener 12 refs en la BD con el color exacto.
> El sistema confunde la pieza sistemáticamente con `2445`, `14769` y `2877`.

---

## 2. Distribución de errores

### Predicción final del pipeline (15 muestras)

| Pred | Veces | % |
|---|---|---|
| 2445 (Plate 6×6) | 5 | 33 % |
| 3022 (Plate 2×2) | 3 | 20 % |
| 87552 (Panel 1×2×2) | 3 | 20 % |
| 15392 | 3 | 20 % |
| 3795 (Plate 2×6) | 1 | 7 % |

### DINOv2 puro — top-1 cenital

| Top-1 | Veces |
|---|---|
| 2445 | 8 |
| 14769 | 7 |

### DINOv2 puro — top-1 lateral

| Top-1 | Veces |
|---|---|
| 2877 | 6 |
| 14769 | 4 |
| 2445 | 3 |
| 32054 | 2 |

### Scores de similitud top-1

| Cámara | min | mean | max |
|---|---|---|---|
| cenital | 0.508 | **0.827** | 0.946 |
| lateral | 0.275 | **0.524** | 0.969 |

> Los scores son altos (~0.83 mean cenital), pero contra refs **incorrectas**.
> El embedding de la pieza 2419 está más alejado de su propia ref en el espacio
> DINOv2 que de las refs de 2445 / 14769 / 2877.

---

## 3. Análisis por etapa del pipeline

### 3.1 YOLO (detección de bbox)
- **Cenital**: 15/15 detecciones, conf media ~0.69 (rango 0.55–0.85). ✅
- **Lateral**: 15/15 detecciones, conf media ~0.65 (rango 0.32–0.79). ✅
- No es el cuello de botella.

### 3.2 Color CIELAB (filtro)
GT esperado: code **86** (Light Bluish Gray).

| Decisión cenital | Veces | ¿OK? |
|---|---|---|
| 85 (Dark Bluish Gray) | 8 | ❌ |
| 86 (Light Bluish Gray) | 4 | ✅ |
| 1 (White) | 3 | ❌ |

- Sólo **4/15** detectan el color GT correcto (86).
- En los demás, el color cenital sale más oscuro (85) o más claro (1) por
  efecto del bevel + iluminación de la escena canónica.
- Cuando hay **conflicto cenital≠lateral** (8/15) el filtro se desactiva
  (`none_consensus_fail` → 38 candidatos), por lo que el filtro de color
  acaba abriendo en lugar de restringir.

### 3.3 Superficie + altura
- `mask_pixels` ~9700 (área cenital aparente ~960 mm²).
- Altura lateral medida: 5.46–7.84 mm (rango).
- DB para 2419 pose 0: residual de superficie 572 mm² (alto). El score
  superficie-vs-DB de la propia 2419 es **0.000** en varios samples → no se
  selecciona aunque pase el filtro de color.

### 3.4 DINOv2 (similitud final)
- 1440 embeddings en BD, 38 piezas únicas, **12 de la 2419**.
- 0/15 GT-in-top-3 en ambas cámaras → la pieza 2419 está sistemáticamente
  fuera del top-3 contra las 1440 refs.
- Ganadores recurrentes:
  - **2445** (Plate 6×6) — pieza grande plana de área similar a la
    proyección cenital de 2419 cuando está acostada.
  - **14769** (Tile 2×2 round) — silueta circular cenital, posiblemente
    confundida por el efecto de la única pose estable de 2419.
  - **2877** (Brick 2×2 grille) — en lateral.

> Indicio fuerte: la **única pose estable** registrada para 2419 puede
> producir una silueta cenital muy degenerada respecto a las refs DINOv2
> indexadas. Comparar visualmente con `composites/composite_*.png`.

---

## 4. Hipótesis sobre la causa raíz

1. **Cobertura de poses insuficiente para 2419.**
   En `stable_poses_cache.json` la pieza 2419 sólo tiene **1 pose**. Si las
   refs DINOv2 se generaron con un set de poses pre-canónico (ver TODO
   §7 de SETUP_DOC: las refs aún están con la escena legacy), la única pose
   simulada en el render puede no coincidir con ninguna ref indexada.

2. **Drift escena canónica vs escena de las refs DINOv2.**
   El SETUP_DOC explícitamente dice:
   > *"las refs actuales se generaron con la escena antigua, el factor
   >  BELT_WIDTH override las acerca pero no son idénticas a la escena canónica"*
   Esto explica que los embeddings query (escena canónica) caigan más cerca
   de refs **incorrectas pero geométricamente más cercanas** que de las
   propias refs de 2419.

3. **Color cenital sesgado.**
   8/15 muestras leen Dark Bluish Gray (85) en lugar de Light Bluish Gray (86)
   → probable problema de exposición/iluminación en el render canónico.

---

## 5. Acciones recomendadas

- [ ] **Regenerar refs DINOv2 con `scene_canonical`** (TODO documentado).
      Después: re-indexar BD con `reindex_dinov2_eevee.py --clear`.
- [ ] Revisar `simulate_stable_poses.py` para `2419`: ¿realmente sólo tiene
      1 pose estable? Aumentar tolerancia o generar poses con perturbación.
- [ ] Calibrar exposición/look del render canónico para que el centroide RGB
      cenital de Light Bluish Gray caiga claramente en code 86, no en 85.
- [ ] Probar con otra pieza que tenga más poses (ej. `3023`, `3022`,
      `3024`) para confirmar que el problema es específico de la cobertura
      de poses de 2419 y no del pipeline.

---

## 6. Archivos generados

```
2camaras_random_pieza_unica/data/random_focus_2419/
├── REPORT.md                          ← este documento
├── random_focus_metadata.json         ← metadata de los 15 renders
├── eval_report.json                   ← inferencia pipeline completo
├── focus_similarity.json              ← top-3 DINOv2 puro por sample
├── focus_report.html                  ← visual interactivo (composites + top-3)
├── sample_NN_cenital.png              ← 15 imgs cenital
├── sample_NN_lateral.png              ← 15 imgs lateral
├── masks/sample_NN_*_mask.png         ← máscaras SAM por sample y cámara
└── composites/composite_NN.png        ← un PNG por sample con todo
```

Abrir el HTML para inspección visual:

```bash
open 2camaras_random_pieza_unica/data/random_focus_2419/focus_report.html