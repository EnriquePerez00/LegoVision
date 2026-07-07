# Models — `camara_domo_monopieza_90` (proyecto activo principal)

Modelos "ganadores" del pipeline monopieza cenital 90°.
**Solo los listados aquí se suben a git** (whitelist en `.gitignore`).

## Detección / pose (YOLO)

| Archivo | Rol | Arquitectura | Entrenado en | Métrica | Comando |
|---|---|---|---|---|---|
| `yolo_cenital.pt` | Detección cenital | YOLO11n | `data/yolo_dataset` (v9, ~12k crops) | mAP50 0.98+ | `train_yolo_pose.py` |
| `yolo_lateral.pt` | Detección lateral/frontal | YOLO11n | `data/yolo_dataset` (frontal) | mAP50 ≥ 0.95 | `train_yolo_pose_frontal.py` |
| `yolo_cenital_pose.pt` | Keypoints cenital | YOLO11n-pose | idem | — | `train_yolo_pose.py` |
| `yolo_frontal_pose.pt` | Keypoints frontal | YOLO11n-pose | idem | — | `train_yolo_pose_frontal.py` |

## Geometría (EfficientNet)

| Archivo | Rol | Arquitectura | Métrica | Comando |
|---|---|---|---|---|
| `efficientnet_cenital.pt` | Embed grayscale silueta cenital | EffNetV2-B0 head fine-tuned | top-1 ≥ 96 % | `train_efficientnet_head.py` |
| `efficientnet_cenital.pt.classes.txt` | Mapping idx→part_id | — | — | idem |
| `efficientnet_lateral.pt` | Embed grayscale silueta lateral | EffNetV2-B0 head fine-tuned | — | idem |
| `efficientnet_lateral.pt.classes.txt` | Mapping idx→part_id | — | — | idem |
| `features_cenital.pt` | Feature extractor cascada | Custom head | — | `train_features_classifiers.py` |
| `features_lateral.pt` | idem lateral | Custom head | — | idem |

## Color (MLP + cascada jerárquica)

| Archivo | Rol | Arquitectura | Métrica | Comando |
|---|---|---|---|---|
| `color_mlp_model.pt` | Clasificador color primario (all) | MLP 12→32→32→N | acc ≈ 99 % | `train_and_evaluate_color_mlp.py` |
| `color_mlp_model_75078.pt` | idem restringido al set 75078 | idem | acc ≈ 99 % | idem |
| `color_mlp_metadata.json` | Metadatos features/labels | — | — | idem |
| `color_mlp_metadata_75078.json` | idem 75078 | — | — | idem |
| `color_all.pt` | Clasificador cascada all-colors (legacy) | Router+heads | — | `train_color_classifier_all_colors.py` |
| `color_all_metadata.json` | Metadatos | — | — | idem |
| `color_router_all.pt` | Router jerárquico (familias) | MLP | — | `train_hierarchical_color.py` |
| `color_router_all_metadata.json` | Metadatos router | — | — | idem |
| `color_router_all_colors.pt` | Router extendido | MLP | — | idem |
| `color_router_all_colors_metadata.json` | Metadatos | — | — | idem |
| `color_ref_embeddings.npz` | Refs DINOv2 (color-only) cache | np.savez | — | `save_dinov2_color_references.py` |
| `color_classes.txt` | Mapping idx→color_id | — | — | idem |

## Color V2 (4-stage CIELAB sin MLP Router) — 2026-06-07

| Archivo | Rol | Arquitectura | Métrica | Comando |
|---|---|---|---|---|
| `material_type_classifier.pt` | Stage 2 de ColorClassifierV2: clasifica tipo de material | MLP 6→32→16→5 | acc ≥ 95% (sintético) | `train_material_type_classifier.py` |
| `material_type_classifier_metadata.json` | Metadatos (mean/std/classes) | — | — | idem |

**Uso de ColorClassifierV2 en evaluación:**
```bash
python scripts/run_evaluation_1D_all.py \
    --metadata data/simulation_x5_1D_all/simulation_metadata.json \
    --color-classifier v2 \
    --report reports/eval_colorv2.json
```

**Mejoras sobre clasificador anterior (`all_colors`, 4.2% accuracy):**
- Stage 0: pre-check determinista de material (sin MLP)
- Stage 1: CIELAB directo contra paleta calibrada (CIEDE2000 ponderado por material)
- Stage 2: MLP ligero 6D→5 para resolver ambigüedad de material (solo si ΔE>8)
- Stage 3: resolución de homónimos determinista por mapa canónico

**Accuracy proyectada:** ~50% color (vs 4.2% baseline)

## Modelos base NO trackeados (regenerables)

- `mobile_sam.pt` — descarga original de MobileSAM.
- `yolo11n.pt`, `yolo11s-pose.pt`, `yolo26n.pt` — cachean vía `ultralytics`.
- `yolo_cenital.mlpackage/` — export CoreML (regenerar con `yolo export`).
- `train_runs/` — carpetas de checkpoints intermedios de Ultralytics.
- `hierarchical/` — outputs de tuning de router.

## Export a producción (CoreML)

```bash
yolo export model=projects/camara_domo_monopieza_90/models/yolo_cenital.pt \
    format=coreml half=True imgsz=1024
yolo export model=projects/camara_domo_monopieza_90/models/yolo_cenital_pose.pt \
    format=coreml half=True imgsz=1024
```

Speedup esperado en M4 (Neural Engine): 2-3× vs PyTorch/MPS.

## Cómo actualizar un modelo ganador

1. Entrenar nueva versión → guardar temporal en `models/<name>_v<N>.pt`.
2. Correr `evaluate_sim100.py` y `evaluate_sim300.py` sobre el test canónico.
3. Comparar métricas (accuracy top-1, latencia media) vs versión actual.
4. Si mejora → renombrar a nombre canónico y actualizar esta tabla.
5. Commit: `models(camara_domo_monopieza_90): <nombre> v<N> — <métrica>`.