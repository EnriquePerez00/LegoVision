#!/bin/bash
# =============================================================================
# 2camaras_pieza_unica/run_pipeline.sh
# Pipeline completo: render → train → index → test → evaluate
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BLENDER="/opt/homebrew/bin/blender"
PYTHON=".venv/bin/python"
SUBPROJECT="2camaras_pieza_unica"

echo "============================================================"
echo "  PIPELINE COMPLETO: 2camaras_pieza_unica"
echo "  $(date)"
echo "============================================================"

# ── PASO 1: Render YOLO Cenital (1000 frames, 5% vacíos) ──
echo ""
echo "▶ [1/8] Renderizando dataset YOLO CENITAL (1000 frames)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_yolo_training_dataset.py -- \
    --camera cenital \
    --output_dir $SUBPROJECT/data/yolo_cenital \
    --num_frames 1000

# ── PASO 2: Render YOLO Lateral (1000 frames, 5% vacíos) ──
echo ""
echo "▶ [2/8] Renderizando dataset YOLO LATERAL (1000 frames)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_yolo_training_dataset.py -- \
    --camera lateral \
    --output_dir $SUBPROJECT/data/yolo_lateral \
    --num_frames 1000

# ── PASO 3: Train YOLO Cenital ──
echo ""
echo "▶ [3/8] Entrenando modelo YOLO CENITAL..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON training/train_yolo.py --camera cenital
cd "$PROJECT_ROOT"

# ── PASO 4: Train YOLO Lateral ──
echo ""
echo "▶ [4/8] Entrenando modelo YOLO LATERAL..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON training/train_yolo.py --camera lateral
cd "$PROJECT_ROOT"

# ── PASO 5: Render DINOv2 Referencias (cenital + lateral) ──
echo ""
echo "▶ [5/8] Renderizando referencias DINOv2 (cenital + lateral, 12 rotaciones)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_eevee_dinov2_refs.py -- \
    --output_dir $SUBPROJECT/data/dinov2_refs \
    --rotations 12

# ── PASO 6: Indexar Embeddings DINOv2 ──
echo ""
echo "▶ [6/8] Indexando embeddings DINOv2..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/reindex_dinov2_eevee.py \
    --ref_dir data/dinov2_refs --clear
cd "$PROJECT_ROOT"

# ── PASO 7: Generar Test Set (100 muestras) ──
echo ""
echo "▶ [7/8] Generando test set (100 muestras, pieza única centrada)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_test_set.py -- \
    --output_dir $SUBPROJECT/data/test_dual \
    --num_samples 100

# ── PASO 8: Evaluar Pipeline de Inferencia ──
echo ""
echo "▶ [8/8] Evaluando pipeline de inferencia..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/run_evaluation.py
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  ✅ PIPELINE COMPLETADO"
echo "  $(date)"
echo "  Resultados en: $SUBPROJECT/data/eval_report.json"
echo "============================================================"