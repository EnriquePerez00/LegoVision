#!/bin/bash
set -e

echo "=== PASO 1: Renderizando imágenes sintéticas (Fondo Verde) ==="
bash 2camaras_random_pieza_unica/scripts/run_parallel_dinov2.sh /Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/dinov2_refs_green

echo "=== PASO 2: Reindexando EfficientNet ==="
.venv/bin/python camara_domo/scripts/reindex_efficientnet.py \
    --ref_dir /Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/dinov2_refs_green \
    --clear

echo "=== COMPLETADO ==="
