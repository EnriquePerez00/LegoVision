#!/bin/bash
# run_test50_pipeline.sh
# Pipeline completo: calibración de color + 50 renders canónicos + inferencia + report
#
# Uso: bash 2camaras_random_pieza_unica/scripts/run_test50_pipeline.sh

set -e

BLENDER="/opt/homebrew/bin/blender"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${PROJECT_ROOT}/scripts"
DATA="${PROJECT_ROOT}/data"
REPORTS="${DATA}/reports/test_500fullhd"

echo "============================================================"
echo " STEP 1: Re-calibrar paleta de color (escena canónica)"
echo "============================================================"
$BLENDER -b -P "${SCRIPTS}/generate_color_calibration_canonical.py" 2>&1 | tail -20

echo ""
echo "============================================================"
echo " STEP 2: Generar 50 renders de test (2048x2048, 30cm, 55mm)"
echo "============================================================"
$BLENDER -b -P "${SCRIPTS}/generate_test50_canonical.py" -- \
    --num_samples 50 \
    --output_dir "${REPORTS}" \
    --seed 2026 2>&1 | tail -30

echo ""
echo "============================================================"
echo " STEP 3: Ejecutar inferencia sobre los 50 renders"
echo "============================================================"
cd "${PROJECT_ROOT}"
python3 "${SCRIPTS}/run_evaluation.py" \
    --test_dir "${REPORTS}" \
    --metadata "${REPORTS}/random_500_metadata.json" \
    --report "${REPORTS}/eval_report.json" \
    2>&1 | tee "${REPORTS}/step3_inference.log" | tail -20

echo ""
echo "============================================================"
echo " STEP 4: Generar report HTML + CSVs"
echo "============================================================"
python3 "${SCRIPTS}/generate_report_inference.py" \
    --eval "${REPORTS}/eval_report.json" \
    --images_dir "${REPORTS}" \
    --out "${REPORTS}" \
    --th_surface 10.0 \
    --th_height 10.0 \
    --top_n 50 \
    2>&1 | tee "${REPORTS}/step4_report.log" | tail -10

echo ""
echo "============================================================"
echo " PIPELINE COMPLETO"
echo " Reports en: ${REPORTS}"
echo "============================================================"
ls -la "${REPORTS}"/*.html "${REPORTS}"/*.csv "${REPORTS}"/*.json 2>/dev/null || true