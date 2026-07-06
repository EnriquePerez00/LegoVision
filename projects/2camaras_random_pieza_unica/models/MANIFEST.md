# Models — `2camaras_random_pieza_unica`

Pipeline dual cenital + lateral, 1 pieza por frame, escena canónica v4.

## Contenido versionado

| Archivo | Rol | Arquitectura | Métrica | Comando |
|---|---|---|---|---|
| `yolo_cenital.pt` | Detección cenital | YOLO11n | mAP50 0.97+ | `training/train_yolo.py` |
| `yolo_lateral.pt` | Detección lateral | YOLO11n | mAP50 0.95+ | idem |
| `yolo_cenital_pose.pt` | Pose keypoints cenital | YOLO11n-pose | — | `train_dino_metric_head.py` |
| `yolo_lateral_pose.pt` | Pose keypoints lateral | YOLO11n-pose | — | idem |
| `dino_metric_head.pt` | Head métrica DINOv2 | Linear+L2 (dim 768→256) | recall@1 ≈ 0.94 | `scripts/train_dino_metric_head.py` |

## NO trackeados

- `mobile_sam.pt`, `yolo11n.pt`, DINOv2 pretrained (backbone).
- Embeddings de galería `data/dinov2_refs*/` (regenerables con `generate_canonical_dinov2_refs.py`).
- Carpetas `runs/`, `train_runs/`.

## Export CoreML

```bash
yolo export model=projects/2camaras_random_pieza_unica/models/yolo_cenital.pt \
    format=coreml half=True imgsz=1024
```
