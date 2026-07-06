# SETUP_DOC — `2camaras_random_pieza_unica`

> **Iteración:** 2camaras_random_pieza_unica
> **Última actualización:** 2026-09-09 (post-cleanup)
> **Escena canónica:** `scripts/scene_canonical.py`

---

## 1. Descripción general

Pipeline de visión artificial para identificar piezas LEGO del set
**75078-1** sobre una cinta transportadora industrial, usando
**dos cámaras** (cenital + lateral). El proyecto cubre:

1. **Render sintético** (Blender EEVEE) de datasets de entrenamiento
   YOLO + referencias DINOv2 + sets de inferencia.
2. **Inferencia** en cascada: YOLO → SAM → color CIELAB → superficie
   gaussiana → altura lateral → similaridad DINOv2.
3. **Reportes** CSV/HTML con accuracy global, errores por color,
   superficie, altura, focus de similaridad, etc.

---

## 2. Escena canónica (CRÍTICO)

**Toda imagen renderizada en el proyecto** (refs, training, test,
inferencia) usa la misma escena, definida una sola vez en
`scripts/scene_canonical.py`:

| Aspecto | Valor canónico |
|---|---|
| Escala | 1 BU = 0.1 m = 10 cm |
| Cinta | Azul petróleo `(0.145,0.255,0.33)` linear, 20 × 120 × 1 cm |
| Pantalla | Aluminio mate detrás (10 cm alto × 120 cm largo, x = -1 BU) |
| Suelo | Gris claro PVC mate, 60 × 60 BU @ z=-0.5 BU |
| Cám. cenital | (0, 0, 3.0) BU @ focal 55 mm sensor 36 mm (300 mm de altura) |
| Cám. lateral | (1.5, 0, 0.25) BU @ focal 27 mm sensor 36 mm |
| FOV cenital | 200 mm (px/mm = 3.2 a 640px, 1.92 a 384px, 10.24 a 2048px) |
| Iluminación | `variant_V4_overhead_strip_high_ambient`: AREA RECT 0.6×0.3 BU @ z=0.5 BU, 0.6 W + world strength 0.6 |
| Render engine | BLENDER_EEVEE |
| Color management | View `Standard`, Look `None` |

Si necesitas cambiar la escena, **modifica solo `scene_canonical.py`**.
Todos los demás scripts heredan vía `from scene_canonical import …`.

---

## 3. Estructura de scripts (post-cleanup)

```
scripts/
├── scene_canonical.py          ← FUENTE ÚNICA de la escena
├── _pose_utils.py              ← TARPS + apply_stable_pose
│
├── ─── Generación de imágenes (CANONICO) ───
├── generate_inferencia_test_v2.py  ← test forzado (7 muestras x color)
├── generate_15_random_focus.py     ← N renders de 1 pieza al azar (focus)
├── generate_300_random_set.py      ← set 300 random_position (TODO: migrar a scene_canonical)
├── generate_yolo_training_dataset.py        (TODO: migrar a scene_canonical)
├── generate_yolo_training_dataset_parallel.py
├── generate_eevee_dinov2_refs.py            (TODO: migrar a scene_canonical)
├── generate_eevee_dinov2_refs_parallel.py
│
├── ─── Inferencia ───
├── run_evaluation.py              ← pipeline en cascada (1429 LOC)
│
├── ─── Reportes ───
├── report_dinov2.py                        ← top-K cenital+lateral DINOv2 (default K=2)
├── generate_inference_300_report.py       ← reporte HTML/CSV 300 set
├── generate_inference_300_errors_report.py
├── generate_color_analysis_report.py
├── generate_color_focus_report.py
├── analyze_bbox_accuracy.py
│
├── ─── Helpers compartidos ───
├── generate_synthetic_set.py       ← materiales ABS, EEVEE, GPU, fallback mesh
├── generate_synthetic_dataset.py   ← get_single_mesh_object
├── generate_test_set.py            ← (deprecado parcial — solo helpers neutros usados)
├── ldraw_color_resolver.py
├── ldraw_mesh_parser.py
├── simulate_stable_poses.py        ← genera stable_poses_cache.json
├── reindex_dinov2_eevee.py
├── scene_config.py
├── setup_lighting_only.py
├── merge_worker_metadata.py
├── optimize_rendering.py
│
├── ─── Wrappers paralelos ───
├── run_parallel_dinov2.sh
├── run_parallel_render.sh
│
└── _legacy/                       ← scripts deprecados (solo referencia histórica)
    ├── generate_inferencia_test_set.py    (v1, sustituido por _v2)
    ├── generate_lighting_variants.py      (v1)
    ├── generate_lighting_variants_v2.py   (experimental)
    ├── generate_piece_report.py           (1321 LOC, no se usa)
    ├── generate_random_position_report.py (versión 76-set)
    ├── generate_set_random_position.py    (76 muestras viejo)
    └── run_lateral_inference_v3.py        (experimento solo lateral)
```

---

## 4. Estructura de datos (post-cleanup)

```
data/
├── stable_poses_cache.json          ← cache TARPS (336 KB) — CRÍTICO
├── dinov2_refs_v2/                  ← refs DINOv2 indexadas (107 MB)
├── inferencia_test_v3_colors/       ← test forzado canónico (4.7 MB)
├── random_focus_<ref>/              ← outputs de generate_15_random_focus
│   ├── sample_NN_{cenital,lateral}.png
│   ├── random_focus_metadata.json
│   ├── eval_report.json
│   ├── dinov2_report.json
│   ├── dinov2_report.html
│   ├── composites/composite_NN.png
│   └── masks/sample_NN_*_mask.png
├── yolo_cenital/                    ← dataset training cenital (761 MB)
├── yolo_lateral/                    ← dataset training lateral (286 MB)
├── yolo_cenital_processed/          ← split train/val procesado
├── yolo_lateral_processed/
└── reports/                         ← reports CSV+HTML del set 300 (19 MB)
```

**Borrados en cleanup 2026-09-09** (~1.7 GB liberados):
- `dinov2_refs/` (869 MB, v1 sustituido por v2)
- `random_position/` (427 MB, set 300 viejo con escena legacy)
- `test_dual/` (66 MB)
- `inferencia_test/` y `inferencia_test_v2/` (66 MB combinados, sustituidos por `inferencia_test_v3_colors`)
- `lighting_variants/` y `lighting_variants_v2/` (4.6 MB)

---

## 5. Pipeline de ejecución (`run_pipeline.sh`)

```
1. Render YOLO cenital (2000 frames)            generate_yolo_training_dataset.py
2. Render YOLO lateral (1000 frames)            generate_yolo_training_dataset.py
3. Train YOLO cenital                           training/train_yolo.py
4. Train YOLO lateral                           training/train_yolo.py
5. Render refs DINOv2 (paralelo, 384px)         run_parallel_dinov2.sh
6. Indexar embeddings en BD                     reindex_dinov2_eevee.py
7. Generar test forzado V3 (7 colores)          generate_inferencia_test_v2.py
8. Inferencia sobre test forzado                run_evaluation.py
9. Generar set 300 random                       generate_300_random_set.py
10. Inferencia 300 + reporte CSV/HTML           run_evaluation.py + generate_inference_300_report.py
```

---

## 6. Flujos puntuales útiles

### Foco en una pieza (15 muestras + reporte similaridad DINOv2 puro)

```bash
# 1) 15 renders de UNA pieza al azar (o forzar con --ref)
/opt/homebrew/bin/blender -b -P \
    2camaras_random_pieza_unica/scripts/generate_15_random_focus.py

# 2) Inferencia (escribe random_focus_<ref>/eval_report.json)
.venv/bin/python 2camaras_random_pieza_unica/scripts/run_evaluation.py \
    --metadata "$(pwd)/2camaras_random_pieza_unica/data/random_focus_<REF>/random_focus_metadata.json" \
    --report   "$(pwd)/2camaras_random_pieza_unica/data/random_focus_<REF>/eval_report.json"

# 3) Reporte DINOv2 (top-K cenital + lateral; default K=2)
.venv/bin/python 2camaras_random_pieza_unica/scripts/report_dinov2.py \
    --input_dir 2camaras_random_pieza_unica/data/random_focus_<REF>
```

### Solo regenerar refs DINOv2 (paralelo)

```bash
bash 2camaras_random_pieza_unica/scripts/run_parallel_dinov2.sh \
    "$(pwd)/2camaras_random_pieza_unica/data/dinov2_refs_v2" 12 384
.venv/bin/python 2camaras_random_pieza_unica/scripts/reindex_dinov2_eevee.py \
    --ref_dir 2camaras_random_pieza_unica/data/dinov2_refs_v2 --clear
```

---

## 7. Pendientes después del cleanup (TODO)

Los siguientes scripts **funcionan pero todavía construyen su propia
escena** (importando bits de `generate_test_set.py`). Para que el
proyecto sea totalmente coherente con `scene_canonical.py`, hay que
migrarlos en una próxima iteración:

- [ ] `generate_300_random_set.py` → usar `build_scene_canonical()`
- [ ] `generate_yolo_training_dataset.py` → idem
- [ ] `generate_yolo_training_dataset_parallel.py` → idem
- [ ] `generate_eevee_dinov2_refs.py` → idem (las refs actuales se generaron con la escena antigua, el factor BELT_WIDTH override las acerca pero no son idénticas a la escena canónica)
- [ ] `generate_eevee_dinov2_refs_parallel.py` → idem

Después de migrar y regenerar las refs DINOv2, el proyecto tendrá un
único dominio cromático y geométrico para todos los componentes.

---

## 8. Referencias rápidas

- **Inferencia individual** (debug): `scripts/run_evaluation.py`
- **Cache poses estables**: `data/stable_poses_cache.json` (TARPS)
- **Cabezal proyección DINOv2**: `models/dino_metric_head.pt` (unimodal 384→128)
- **Pesos YOLO**: `models/yolo_cenital.pt`, `models/yolo_lateral.pt`
- **Set inventario**: `database/set_catalog.py` → `REAL_SETS["75078-1"]`
