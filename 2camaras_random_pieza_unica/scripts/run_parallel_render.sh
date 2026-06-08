#!/bin/bash
# Paralelización del render YOLO con detección dinámica de recursos
# Opción A: 4 workers manteniendo 20% CPU/RAM libre
# Si recursos insuficientes, baja dinámicamente a 3 workers

set -e

BLENDER="/opt/homebrew/bin/blender"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SUBPROJECT="2camaras_random_pieza_unica"

# Parámetros
CAMERA=${1:-cenital}
OUTPUT_DIR=${2:-$PROJECT_ROOT/$SUBPROJECT/data/yolo_${CAMERA}}
NUM_FRAMES=${3:-2000}
MASTER_SEED=${4:-42}

# ─── Detección dinámica de recursos (manteniendo 20% libre) ───
TOTAL_CPU=$(sysctl -n hw.logicalcpu)
TOTAL_RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))

# RAM disponible (usando vm_stat)
FREE_PAGES=$(vm_stat | awk '/Pages free:/ {gsub(/\./,""); print $3}')
INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive:/ {gsub(/\./,""); print $3}')
SPECULATIVE_PAGES=$(vm_stat | awk '/Pages speculative:/ {gsub(/\./,""); print $3}')
AVAILABLE_PAGES=$((FREE_PAGES + INACTIVE_PAGES + SPECULATIVE_PAGES))
AVAILABLE_RAM_GB=$((AVAILABLE_PAGES * 4096 / 1024 / 1024 / 1024))

# Calcular workers (mantener 20% reservado para sistema)
USABLE_CPU=$((TOTAL_CPU * 80 / 100))
USABLE_RAM=$((AVAILABLE_RAM_GB * 80 / 100))

# Cada worker Blender EEVEE: ~2 cores, ~4GB RAM
WORKERS_BY_CPU=$((USABLE_CPU / 2))
WORKERS_BY_RAM=$((USABLE_RAM / 4))

# Tomar el mínimo
NUM_WORKERS=$WORKERS_BY_CPU
if [ $WORKERS_BY_RAM -lt $NUM_WORKERS ]; then
    NUM_WORKERS=$WORKERS_BY_RAM
fi

# Limitar entre 3 y 4
if [ $NUM_WORKERS -gt 4 ]; then NUM_WORKERS=4; fi
if [ $NUM_WORKERS -lt 3 ]; then NUM_WORKERS=3; fi

echo "=================================================="
echo "🚀 RENDER PARALELO: $CAMERA"
echo "=================================================="
echo "💻 Sistema:"
echo "   Total CPU: $TOTAL_CPU cores"
echo "   Total RAM: ${TOTAL_RAM_GB}GB | Disponible: ${AVAILABLE_RAM_GB}GB"
echo "   Reserva: 20% (CPU=${USABLE_CPU}, RAM=${USABLE_RAM}GB usables)"
echo ""
echo "📊 Configuración:"
echo "   Workers (dinámico): $NUM_WORKERS"
echo "   Total frames: $NUM_FRAMES"
echo "   Master seed: $MASTER_SEED"
echo "   Output: $OUTPUT_DIR"
echo "=================================================="

mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$OUTPUT_DIR/labels"

# Limpiar metadatas previos de workers
rm -f "$OUTPUT_DIR"/dataset_metadata_worker*.json

FRAMES_PER_WORKER=$((NUM_FRAMES / NUM_WORKERS))
REMAINDER=$((NUM_FRAMES % NUM_WORKERS))

echo ""
echo "📋 Distribución de frames:"

declare -a PIDS
declare -a RANGES

FRAME_START=0
for ((i=0; i<NUM_WORKERS; i++)); do
    FRAMES_THIS_WORKER=$FRAMES_PER_WORKER
    if [ $i -lt $REMAINDER ]; then
        FRAMES_THIS_WORKER=$((FRAMES_THIS_WORKER + 1))
    fi
    FRAME_END=$((FRAME_START + FRAMES_THIS_WORKER))

    echo "   Worker $i: frames [$FRAME_START, $FRAME_END) → $FRAMES_THIS_WORKER frames"

    cd "$PROJECT_ROOT"
    LOG_FILE="$OUTPUT_DIR/worker_${i}.log"
    
    $BLENDER -b -P $SCRIPT_DIR/generate_yolo_training_dataset_parallel.py -- \
        --camera "$CAMERA" \
        --output_dir "$OUTPUT_DIR" \
        --start_frame "$FRAME_START" \
        --end_frame "$FRAME_END" \
        --total_frames "$NUM_FRAMES" \
        --master_seed "$MASTER_SEED" \
        --seed "$((MASTER_SEED + i))" \
        --worker_id "$i" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    RANGES+=("$FRAME_START-$((FRAME_END-1))")

    FRAME_START=$FRAME_END
done

echo ""
echo "⏳ Esperando workers (logs individuales en $OUTPUT_DIR/worker_*.log)..."
START_TIME=$(date +%s)
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    RANGE=${RANGES[$i]}
    if wait $PID 2>/dev/null; then
        echo "✅ Worker $i ($RANGE) completado"
    else
        echo "❌ Worker $i ($RANGE) falló — ver $OUTPUT_DIR/worker_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "⏱️  Tiempo total: $((ELAPSED / 60))m $((ELAPSED % 60))s"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🔗 Consolidando metadata..."
    cd "$PROJECT_ROOT"
    .venv/bin/python3 $SCRIPT_DIR/merge_worker_metadata.py \
        --output_dir "$OUTPUT_DIR" \
        --total_frames "$NUM_FRAMES"
    
    echo ""
    echo "=================================================="
    echo "✅ RENDER PARALELO COMPLETADO"
    echo "   Output: $OUTPUT_DIR"
    echo "   Tiempo: $((ELAPSED / 60))m $((ELAPSED % 60))s"
    echo "=================================================="
else
    echo ""
    echo "⚠️  $FAILED workers fallaron"
    exit 1
fi
