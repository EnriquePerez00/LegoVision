#!/bin/bash
set -e

echo "=== PASO 1: Renderizando imágenes sintéticas CANÓNICAS (Fondo Verde OPACO) ==="
bash 2camaras_random_pieza_unica/scripts/run_parallel_canonical_refs.sh /Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/dinov2_refs_canonical_green

echo "=== PASO 2: Reindexando EfficientNet ==="
.venv/bin/python camara_domo/scripts/reindex_efficientnet.py \
    --ref_dir /Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/dinov2_refs_canonical_green \
    --clear

echo "=== PASO 3: Ejecutando inferencia comparativa ==="
.venv/bin/python camara_domo/scripts/run_comparative_inference.py

echo "=== COMPLETADO ==="
