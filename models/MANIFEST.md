# Models — LegoVision (raíz)

Modelos "ganadores" a nivel repo (compartidos entre proyectos).
**Solo los listados aquí se suben a git** (whitelist en `.gitignore`).
El resto de `.pt` / `.pth` / `.mlpackage` se ignora por defecto.

## Contenido

| Archivo | Rol | Arquitectura | Entrenado en | Métrica clave | Origen / Comando |
|---|---|---|---|---|---|
| `best.pt` | Baseline general YOLO | YOLO11n | dataset legacy multi-cámara | — (histórico) | `train_yolo.py` |
| `yolo_cenital_val.pt` | Val YOLO cenital | YOLO11n | training/dataset_yolo_cenital_val.yaml | mAP50 ≈ 0.97 | `training/train_yolo.py` |
| `yolo_lateral_val.pt` | Val YOLO lateral | YOLO11n | training/dataset_yolo_lateral_val.yaml | mAP50 ≈ 0.95 | `training/train_yolo.py` |
| `yolo_cenital_pose.pt` | Pose keypoints cenital | YOLO11n-pose | 2camaras_random_pieza_unica v4 | — | `scripts/train_dino_metric_head.py` |
| `yolo_frontal_pose.pt` | Pose keypoints frontal | YOLO11n-pose | 2camaras_random_pieza_unica v4 | — | idem |
| `dino_metric_head.pt` | Head métrica DINOv2 | Linear+L2 | data100 canónico | recall@1 ≈ 0.94 | `scripts/train_dino_metric_head.py` |

## Modelos base NO trackeados (se descargan on-demand)

Los siguientes archivos NUNCA deben estar en este manifest — se
obtienen del hub oficial o vía `ultralytics`:

- `mobile_sam.pt` — descarga: https://github.com/ChaoningZhang/MobileSAM
- `yolo11n.pt`, `yolo11s-pose.pt`, `yolo26n.pt` — `ultralytics` los cachea la primera vez.
- DINOv2 ViT-B/14, ViT-S/14 — `torch.hub.load('facebookresearch/dinov2', …)`
- EfficientNetV2-B0 (backbone pretrained) — `torchvision.models`

## Cómo añadir un modelo nuevo

1. Entrenar y validar contra el test set canónico del proyecto.
2. Copiar el `.pt` a este directorio.
3. Añadir fila en la tabla con métrica en `evaluate_*` reproducible.
4. Añadir la regla `!models/<archivo>.pt` en `.gitignore` (si no la
   cubre ya el patrón `!models/*.pt`).
5. Commit con mensaje `models: add <archivo> — <métrica>`.