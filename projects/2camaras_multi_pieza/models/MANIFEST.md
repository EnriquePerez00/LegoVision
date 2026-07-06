# Models — `2camaras_multi_pieza`

Pipeline multi-pieza sobre cinta con YOLO detector +
reidentificación DINOv2. **Early stage.**

## Contenido versionado

Actualmente sin modelos ganadores; se entrenan sobre la marcha.
Cuando estabilicen se listarán aquí en el mismo formato que en
`camara_domo_monopieza_90/models/MANIFEST.md`.

## NO trackeados

- Todos los `.pt` intermedios (`runs/`, `train_runs/`).
- Embeddings de galería y refs sintéticas.

## Roadmap

1. Estabilizar dataset multi-pieza en `data/`.
2. Fine-tune YOLO11n con multi-clase.
3. DINOv2 metric head para reidentificación.
4. Métricas objetivo: mAP50 ≥ 0.95, MOTA ≥ 0.85.
