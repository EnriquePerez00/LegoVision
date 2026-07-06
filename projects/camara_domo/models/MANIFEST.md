# Models — `camara_domo` (legacy / referencia)

Iteración previa del monopieza con dome light. **Congelado**; se
mantienen los pesos como baseline histórico para comparar iteraciones
nuevas en `camara_domo_monopieza_90/`.

## Contenido versionado

| Archivo | Rol | Notas |
|---|---|---|
| `yolo_cenital.pt` | Detección cenital baseline | histórico |
| `yolo_frontal.pt` | Detección frontal baseline | histórico |
| `yolo_lateral.pt` | Detección lateral baseline | histórico |
| `yolo_cenital_pose.pt` | Pose keypoints cenital | histórico |
| `yolo_frontal_pose.pt` | Pose keypoints frontal | histórico |
| `efficientnet_cenital.pt` | Embed geometría cenital | histórico |
| `efficientnet_lateral.pt` | Embed geometría lateral | histórico |
| `efficientnet_cenital.pt.classes.txt` | Mapping idx→part_id | — |
| `efficientnet_lateral.pt.classes.txt` | Mapping idx→part_id | — |
| `features_cenital.pt` / `features_lateral.pt` | Feature heads cascada | histórico |
| `color_model_cen.pt` / `color_model_lat.pt` | Clasificadores color legacy | histórico |
| `color_mlp_model.pt` + `color_mlp_metadata.json` | MLP color | histórico |
| `color_ref_embeddings.npz` | Refs DINOv2 (color) | cache |
| `color_classes.txt` | Mapping idx→color_id | — |

## NO trackeados

- `mobile_sam.pt`, `yolo11n.pt`, `yolo11s-pose.pt`, `yolo26n.pt` (bases).
- `train_runs/`, `hierarchical/` (checkpoints y outputs de tuning).
- `domo-yolo-pose-train.log`, `live_training_job.log`.

## Nota

Este proyecto está congelado. Nuevas mejoras van a
`camara_domo_monopieza_90/`. Consulta su MANIFEST para el pipeline
productivo actual.
