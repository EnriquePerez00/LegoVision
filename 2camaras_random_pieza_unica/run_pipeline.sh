#!/bin/bash
# =============================================================================
# 2camaras_random_pieza_unica/run_pipeline.sh
# Pipeline completo (escena CANONICA — alineada con scene_canonical.py):
#
#   1. Render dataset YOLO cenital (escena canonica)
#   2. Render dataset YOLO lateral (escena canonica)
#   3. Train YOLO cenital
#   4. Train YOLO lateral
#   5. Render refs DINOv2 (paralelo, res 384)
#   6. Indexar embeddings DINOv2 en BD
#   7. Generar inferencia_test_v3_colors (7 muestras forzadas, escena canonica)
#   8. Evaluar inferencia (cascada color/superficie/altura/DINOv2)
#   9. Generar set 300 random (escena canonica) — balance por pose
#  10. Inferencia 300 + reporte CSV/HTML
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BLENDER="/opt/homebrew/bin/blender"
PYTHON=".venv/bin/python"
SUBPROJECT="2camaras_random_pieza_unica"

echo "============================================================"
echo "  PIPELINE COMPLETO: $SUBPROJECT"
echo "  $(date)"
echo "============================================================"

# ── PASO 1: Render dataset YOLO Cenital (escena canonica) ──
echo ""
echo "▶ [1/10] Renderizando dataset YOLO CENITAL (2000 frames)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_yolo_training_dataset.py -- \
    --camera cenital \
    --output_dir $SUBPROJECT/data/yolo_cenital \
    --num_frames 2000 \
    --seed 42

# ── PASO 2: Render dataset YOLO Lateral (escena canonica) ──
echo ""
echo "▶ [2/10] Renderizando dataset YOLO LATERAL (1000 frames)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_yolo_training_dataset.py -- \
    --camera lateral \
    --output_dir $SUBPROJECT/data/yolo_lateral \
    --num_frames 1000 \
    --seed 43

# ── PASO 3: Train YOLO Cenital ──
echo ""
echo "▶ [3/10] Entrenando modelo YOLO CENITAL..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON training/train_yolo.py --camera cenital
cd "$PROJECT_ROOT"

# ── PASO 4: Train YOLO Lateral ──
echo ""
echo "▶ [4/10] Entrenando modelo YOLO LATERAL..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON training/train_yolo.py --camera lateral
cd "$PROJECT_ROOT"

# ── PASO 5: Render DINOv2 Referencias EN PARALELO (escena canonica, res 384) ──
echo ""
echo "▶ [5/10] Renderizando referencias DINOv2 (4 workers, res=384)..."
bash $SUBPROJECT/scripts/run_parallel_dinov2.sh \
    "$PROJECT_ROOT/$SUBPROJECT/data/dinov2_refs_v2" \
    12 \
    384

# ── PASO 6: Indexar Embeddings DINOv2 ──
echo ""
echo "▶ [6/10] Indexando embeddings DINOv2 en BD..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/reindex_dinov2_eevee.py \
    --ref_dir data/dinov2_refs_v2 --clear
cd "$PROJECT_ROOT"

# ── PASO 7: Inferencia test set forzado (escena canonica V4) ──
# Genera 7 muestras forzadas (3023 x 7 colores del set 75078-1) sobre la
# misma escena canonica que produce las imagenes de inferencia reales.
echo ""
echo "▶ [7/10] Generando inferencia_test_v3_colors (7 forzadas, escena V4)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_inferencia_test_v2.py -- \
    --output_dir $SUBPROJECT/data/inferencia_test_v3_colors \
    --metadata_filename inferencia_test_v3_metadata.json \
    --num_random 0 \
    --seed 42

# ── PASO 8: Evaluar inferencia sobre el test forzado ──
echo ""
echo "▶ [8/10] Evaluando inferencia sobre test forzado..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/run_evaluation.py \
    --metadata data/inferencia_test_v3_colors/inferencia_test_v3_metadata.json \
    --report   data/reports/inferencia_test_v3_eval.json
cd "$PROJECT_ROOT"

# ── PASO 9: Generar set de 300 random (escena canonica) ──
# Balance por pose sobre las 38 refs x 7 colores del set 75078-1, con
# posicion aleatoria valida en cenital+lateral.
echo ""
echo "▶ [9/10] Generando 300 imagenes random (escena canonica)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_300_random_set.py -- \
    --num_samples 300 \
    --output_dir $SUBPROJECT/data/random_300 \
    --metadata_filename random_300_metadata.json \
    --seed 42

# ── PASO 10: Inferencia 300 + reporte CSV/HTML ──
echo ""
echo "▶ [10/10] Inferencia 300 + reporte CSV+HTML..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/run_evaluation.py \
    --metadata data/random_300/random_300_metadata.json \
    --report   data/reports/inference_300_eval.json
$PROJECT_ROOT/$PYTHON scripts/generate_inference_300_report.py \
    --eval data/reports/inference_300_eval.json \
    --out  data/reports/
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  ✅ PIPELINE COMPLETADO"
echo "  $(date)"
echo "  inferencia_test_v3 eval: $SUBPROJECT/data/reports/inferencia_test_v3_eval.json"
echo "  300set eval            : $SUBPROJECT/data/reports/inference_300_eval.json"
echo "  Reporte HTML 300set    : $SUBPROJECT/data/reports/inference_300_summary.html"
echo "  Reporte CSV completo   : $SUBPROJECT/data/reports/inference_300_full.csv"
echo "============================================================"