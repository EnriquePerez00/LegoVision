#!/bin/bash
# Paralelización del render DINOv2 con detección dinámica
# 4 workers manteniendo 20% CPU/RAM libre. Reduce a 3 si insuficiente.

set -e

BLENDER="/opt/homebrew/bin/blender"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SUBPROJECT="2camaras_random_pieza_unica"

OUTPUT_DIR=${1:-$PROJECT_ROOT/$SUBPROJECT/data/dinov2_refs}
ROTATIONS=${2:-12}

# Detección dinámica de recursos
TOTAL_CPU=$(sysctl -n hw.logicalcpu)
TOTAL_RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))

FREE_PAGES=$(vm_stat | awk '/Pages free:/ {gsub(/\./,""); print $3}')
INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive:/ {gsub(/\./,""); print $3}')
SPECULATIVE_PAGES=$(vm_stat | awk '/Pages speculative:/ {gsub(/\./,""); print $3}')
AVAILABLE_PAGES=$((FREE_PAGES + INACTIVE_PAGES + SPECULATIVE_PAGES))
AVAILABLE_RAM_GB=$((AVAILABLE_PAGES * 4096 / 1024 / 1024 / 1024))

USABLE_CPU=$((TOTAL_CPU * 80 / 100))
USABLE_RAM=$((AVAILABLE_RAM_GB * 80 / 100))

WORKERS_BY_CPU=$((USABLE_CPU / 2))
WORKERS_BY_RAM=$((USABLE_RAM / 4))

NUM_WORKERS=$WORKERS_BY_CPU
if [ $WORKERS_BY_RAM -lt $NUM_WORKERS ]; then
    NUM_WORKERS=$WORKERS_BY_RAM
fi
if [ $NUM_WORKERS -gt 4 ]; then NUM_WORKERS=4; fi
if [ $NUM_WORKERS -lt 3 ]; then NUM_WORKERS=3; fi

# Contar piezas en stable_poses_cache (usar el del subproyecto, no el root)
CACHE_PATH="$PROJECT_ROOT/$SUBPROJECT/data/stable_poses_cache.json"
TOTAL_PARTS=$(/Users/I764690/Code_personal/LegoVision/.venv/bin/python3 -c "
import json
with open('$CACHE_PATH') as f:
    data = json.load(f)
print(len(data))
" 2>/dev/null || echo 68)

echo "=================================================="
echo "🚀 RENDER DINOv2 PARALELO"
echo "=================================================="
echo "💻 Sistema:"
echo "   CPU: $TOTAL_CPU cores"
echo "   RAM: ${TOTAL_RAM_GB}GB total / ${AVAILABLE_RAM_GB}GB disponible"
echo "   Reserva 20%: usable_cpu=$USABLE_CPU, usable_ram=${USABLE_RAM}GB"
echo ""
echo "📊 Configuración:"
echo "   Workers: $NUM_WORKERS (rango: 3-4 dinámico)"
echo "   Piezas totales: $TOTAL_PARTS"
echo "   Rotaciones por pose: $ROTATIONS"
echo "   Output: $OUTPUT_DIR"
echo "   TAA samples: 16 (calidad mantenida)"
echo "   Skip existing: ON (resume mode)"
echo "=================================================="

mkdir -p "$OUTPUT_DIR/cenital"
mkdir -p "$OUTPUT_DIR/lateral"

# Calcular distribución de piezas
PARTS_PER_WORKER=$((TOTAL_PARTS / NUM_WORKERS))
REMAINDER=$((TOTAL_PARTS % NUM_WORKERS))

echo ""
echo "📋 Distribución de piezas:"

declare -a PIDS
declare -a RANGES

START=0
for ((i=0; i<NUM_WORKERS; i++)); do
    PARTS_THIS=$PARTS_PER_WORKER
    if [ $i -lt $REMAINDER ]; then
        PARTS_THIS=$((PARTS_THIS + 1))
    fi
    END=$((START + PARTS_THIS))

    echo "   Worker $i: piezas [$START, $END) → $PARTS_THIS piezas"

    cd "$PROJECT_ROOT"
    LOG_FILE="$OUTPUT_DIR/worker_dinov2_${i}.log"

    $BLENDER -b -P $SCRIPT_DIR/generate_eevee_dinov2_refs_parallel.py -- \
        --output_dir "$OUTPUT_DIR" \
        --rotations $ROTATIONS \
        --start_idx $START \
        --end_idx $END \
        --worker_id $i \
        --skip_existing \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    RANGES+=("$START-$((END-1))")

    START=$END
done

echo ""
echo "⏳ Esperando workers (logs en $OUTPUT_DIR/worker_dinov2_*.log)..."
START_TIME=$(date +%s)
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    RANGE=${RANGES[$i]}
    if wait $PID 2>/dev/null; then
        echo "✅ Worker $i ($RANGE) completado"
    else
        echo "❌ Worker $i ($RANGE) falló — ver $OUTPUT_DIR/worker_dinov2_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "⏱️  Tiempo total: $((ELAPSED / 60))m $((ELAPSED % 60))s"

if [ $FAILED -eq 0 ]; then
    # Verificar archivos generados
    CEN_COUNT=$(ls "$OUTPUT_DIR/cenital/" 2>/dev/null | wc -l | tr -d ' ')
    LAT_COUNT=$(ls "$OUTPUT_DIR/lateral/" 2>/dev/null | wc -l | tr -d ' ')
    
    echo ""
    echo "=================================================="
    echo "✅ DINOv2 PARALELO COMPLETADO"
    echo "   Renders cenital: $CEN_COUNT"
    echo "   Renders lateral: $LAT_COUNT"
    echo "   Tiempo: $((ELAPSED / 60))m $((ELAPSED % 60))s"
    echo "=================================================="
else
    echo "⚠️ $FAILED workers fallaron"
    exit 1
fi
