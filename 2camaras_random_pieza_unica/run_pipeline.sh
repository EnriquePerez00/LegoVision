#!/bin/bash
# =============================================================================
# 2camaras_random_pieza_unica/run_pipeline.sh
# Pipeline completo: render -> train -> index -> test -> evaluate ->
#                    300set generation -> 300set evaluation -> ad-hoc report
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

# ── PASO 1: Render YOLO Cenital (2000 frames, 5% vacíos, posicion+pose+color
#                                  aleatorios distribuidos uniformemente sobre
#                                  las 42 combos del set 75078-1) ──
echo ""
echo "▶ [1/10] Renderizando dataset YOLO CENITAL (2000 frames, random)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_yolo_training_dataset.py -- \
    --camera cenital \
    --output_dir $SUBPROJECT/data/yolo_cenital \
    --num_frames 2000 \
    --seed 42

# ── PASO 2: Render YOLO Lateral (1000 frames, 5% vacíos, idem distribucion) ──
echo ""
echo "▶ [2/10] Renderizando dataset YOLO LATERAL (1000 frames, random)..."
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

# ── PASO 5: Render DINOv2 Referencias (cenital + lateral) ──
echo ""
echo "▶ [5/10] Renderizando referencias DINOv2 (cenital + lateral, 12 rotaciones)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_eevee_dinov2_refs.py -- \
    --output_dir $SUBPROJECT/data/dinov2_refs \
    --rotations 12

# ── PASO 6: Indexar Embeddings DINOv2 ──
echo ""
echo "▶ [6/10] Indexando embeddings DINOv2..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/reindex_dinov2_eevee.py \
    --ref_dir data/dinov2_refs --clear
cd "$PROJECT_ROOT"

# ── PASO 7: Generar Test Set (100 muestras centradas) ──
echo ""
echo "▶ [7/10] Generando test set (100 muestras, pieza única centrada)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_test_set.py -- \
    --output_dir $SUBPROJECT/data/test_dual \
    --num_samples 100

# ── PASO 8: Evaluar Pipeline (test_dual centrado) ──
echo ""
echo "▶ [8/10] Evaluando pipeline de inferencia (test_dual)..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/run_evaluation.py
cd "$PROJECT_ROOT"

# ── PASO 9: Generar dataset 300 imagenes (random_position) ──
# Distribuye 300 imagenes balanceadas por pose (opcion C) sobre las 38 refs
# y 7 colores del set 75078-1, con posicion aleatoria valida en cenital+lateral.
# Las imagenes existentes "sample_<ref>_*.png" se PRESERVAN; las nuevas usan
# el prefijo "sample300_<idx>_".
echo ""
echo "▶ [9/10] Generando 300 imagenes random_position (balance por pose)..."
$BLENDER -b -P $SUBPROJECT/scripts/generate_300_random_set.py -- \
    --num_samples 300 \
    --output_dir $SUBPROJECT/data/random_position \
    --metadata_filename random_position_300_metadata.json \
    --seed 42

# ── PASO 10: Inferencia + reporte ad-hoc sobre las 300 imagenes ──
echo ""
echo "▶ [10/10] Inferencia 300 + reporte CSV+HTML..."
cd $SUBPROJECT
$PROJECT_ROOT/$PYTHON scripts/run_evaluation.py \
    --metadata data/random_position/random_position_300_metadata.json \
    --report   data/reports/inference_300_eval.json
$PROJECT_ROOT/$PYTHON scripts/generate_inference_300_report.py \
    --eval data/reports/inference_300_eval.json \
    --out  data/reports/
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  ✅ PIPELINE COMPLETADO"
echo "  $(date)"
echo "  Resultados test_dual : $SUBPROJECT/data/eval_report.json"
echo "  Resultados 300set    : $SUBPROJECT/data/reports/inference_300_eval.json"
echo "  Reporte HTML 300set  : $SUBPROJECT/data/reports/inference_300_summary.html"
echo "  Reporte CSV completo : $SUBPROJECT/data/reports/inference_300_full.csv"
echo "============================================================"