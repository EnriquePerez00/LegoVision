#!/bin/bash
# Render paralelo de refs DINOv2 con ESCENA CANONICA (scene_canonical.py)
# Usa 4 workers con deteccion dinamica de CPU/RAM (20% reserva)
# Output: data/dinov2_refs_v4_canonical/

set -e

BLENDER="/opt/homebrew/bin/blender"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBPROJECT="2camaras_random_pieza_unica"

OUTPUT_DIR=${1:-$PROJECT_ROOT/$SUBPROJECT/data/dinov2_refs_v4_canonical}
ROTATIONS=${2:-12}
RENDER_RES=${3:-384}

# Deteccion dinamica de recursos
TOTAL_CPU=$(sysctl -n hw.logicalcpu)
FREE_PAGES=$(vm_stat | awk '/Pages free:/ {gsub(/\./,""); print $3}')
INACTIVE_PAGES=$(vm_stat | awk '/Pages inactive:/ {gsub(/\./,""); print $3}')
SPECULATIVE_PAGES=$(vm_stat | awk '/Pages speculative:/ {gsub(/\./,""); print $3}')
AVAILABLE_PAGES=$((FREE_PAGES + INACTIVE_PAGES + SPECULATIVE_PAGES))
AVAILABLE_RAM_GB=$((AVAILABLE_PAGES * 4096 / 1024 / 1024 / 1024))
USABLE_CPU=$((TOTAL_CPU * 80 / 100))
USABLE_RAM=$((AVAILABLE_RAM_GB * 80 / 100))
NUM_WORKERS=$((USABLE_CPU / 2))
if [ $NUM_WORKERS -gt 4 ]; then NUM_WORKERS=4; fi
if [ $NUM_WORKERS -lt 3 ]; then NUM_WORKERS=3; fi

# Obtener lista de todas las refs del set 75078-1
REFS_JSON=$($PROJECT_ROOT/.venv/bin/python3 -c "
import json, sys
with open('$PROJECT_ROOT/$SUBPROJECT/data/stable_poses_cache.json') as f:
    cache = json.load(f)
sys.path.insert(0,'$PROJECT_ROOT')
from core.db.set_catalog import REAL_SETS
seen = set()
refs = []
for p in REAL_SETS['75078-1']['parts']:
    if p['ref'] not in seen and p['ref'] in cache and cache[p['ref']]:
        seen.add(p['ref']); refs.append(p['ref'])
print(' '.join(refs))
" 2>/dev/null)

ALL_REFS=($REFS_JSON)
TOTAL_REFS=${#ALL_REFS[@]}

echo "=================================================="
echo "🚀 RENDER REFS CANONICOS PARALELO"
echo "   Workers: $NUM_WORKERS | Refs: $TOTAL_REFS | Res: ${RENDER_RES}px"
echo "   Output: $OUTPUT_DIR"
echo "=================================================="

mkdir -p "$OUTPUT_DIR/cenital" "$OUTPUT_DIR/lateral"

# Dividir refs entre workers
PARTS_PER_WORKER=$((TOTAL_REFS / NUM_WORKERS))
REMAINDER=$((TOTAL_REFS % NUM_WORKERS))

declare -a PIDS RANGES
START=0
for ((i=0; i<NUM_WORKERS; i++)); do
    PARTS_THIS=$PARTS_PER_WORKER
    if [ $i -lt $REMAINDER ]; then PARTS_THIS=$((PARTS_THIS + 1)); fi
    END=$((START + PARTS_THIS))
    
    # Slice de refs para este worker
    WORKER_REFS=("${ALL_REFS[@]:$START:$PARTS_THIS}")
    REFS_STR="${WORKER_REFS[*]}"
    
    echo "   Worker $i: refs [$START,$END) = $PARTS_THIS refs"
    LOG_FILE="$OUTPUT_DIR/worker_canonical_${i}.log"
    
    cd "$PROJECT_ROOT"
    $BLENDER -b -P "$SCRIPT_DIR/generate_canonical_dinov2_refs.py" -- \
        --refs $REFS_STR \
        --rotations $ROTATIONS \
        --render_res $RENDER_RES \
        --output_dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
    RANGES+=("$START-$((END-1))")
    START=$END
done

echo ""
echo "⏳ Esperando ${NUM_WORKERS} workers (logs en $OUTPUT_DIR/worker_canonical_*.log)..."
START_TIME=$(date +%s)
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID 2>/dev/null; then
        echo "✅ Worker $i completado"
    else
        echo "❌ Worker $i falló — ver $OUTPUT_DIR/worker_canonical_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
CEN_COUNT=$(ls "$OUTPUT_DIR/cenital/" 2>/dev/null | wc -l | tr -d ' ')
LAT_COUNT=$(ls "$OUTPUT_DIR/lateral/" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=================================================="
echo "✅ COMPLETADO: cenital=$CEN_COUNT lateral=$LAT_COUNT | ${ELAPSED}s"
echo "=================================================="
if [ $FAILED -gt 0 ]; then echo "⚠️ $FAILED workers fallaron"; exit 1; fi
