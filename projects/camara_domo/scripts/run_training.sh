#!/bin/bash
set -e
PROJECT_ROOT="/Users/I764690/Code_personal/LegoVision"

echo "=== Preparando Dataset Sintético ==="
.venv/bin/python $PROJECT_ROOT/camara_domo/scripts/prepare_synthetic_finetuning.py

echo "=== Entrenando EfficientNet Cenital ==="
.venv/bin/python $PROJECT_ROOT/camara_domo/scripts/train_efficientnet_head.py \
    --data_dir $PROJECT_ROOT/camara_domo/data/efficientnet_train/cenital/train \
    --output_model $PROJECT_ROOT/camara_domo/models/efficientnet_cenital.pt \
    --epochs 5

echo "=== Entrenando EfficientNet Lateral ==="
.venv/bin/python $PROJECT_ROOT/camara_domo/scripts/train_efficientnet_head.py \
    --data_dir $PROJECT_ROOT/camara_domo/data/efficientnet_train/lateral/train \
    --output_model $PROJECT_ROOT/camara_domo/models/efficientnet_lateral.pt \
    --epochs 5

echo "[DONE] Modelos entrenados y guardados."

