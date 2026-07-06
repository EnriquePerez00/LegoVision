#!/usr/bin/env bash
# =============================================================
# LegoVision — HW env vars (M4 · 12 CPU · 48 GB · MPS)
# ------------------------------------------------------------
# Se debe hacer `source` (no `bash`) desde cualquier terminal
# antes de lanzar training/inferencia para asegurar que todo el
# stack usa el HW local al máximo.
#
#     source scripts/hw_env.sh
#
# Compatible con macOS Apple Silicon. En Linux/Intel es inocuo.
# =============================================================

# --- Threads CPU ---
# Reserva 2 CPUs para OS/UI (12 - 2 = 10). Override con
# TORCH_NUM_THREADS_OVERRIDE si se desea otro valor.
_HW_CORES="${TORCH_NUM_THREADS_OVERRIDE:-10}"

export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$_HW_CORES}"
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-2}"

# BLAS / NumPy / SciPy / OpenMP
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_HW_CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$_HW_CORES}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$_HW_CORES}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$_HW_CORES}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$_HW_CORES}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$_HW_CORES}"

# --- MPS (Apple Metal) ---
# Fallback a CPU para ops PyTorch aún no soportadas en MPS
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
# Water-mark alto de memoria MPS (0.0 = sin límite, deja al SO gestionar)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

# --- Silenciar warnings ruidosos de HuggingFace / tokenizers ---
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"

# --- Ultralytics YOLO ---
# Evita descargar autoupdate en cada corrida
export YOLO_AUTOINSTALL="${YOLO_AUTOINSTALL:-False}"
export YOLO_VERBOSE="${YOLO_VERBOSE:-False}"

# --- Blender (paralelización externa; cada worker con 1 thread interno) ---
export BLENDER_WORKERS="${BLENDER_WORKERS:-$_HW_CORES}"

# --- Diagnóstico ---
if [ "${LEGOVISION_HW_QUIET:-0}" != "1" ]; then
  echo "[hw_env] cpu_threads=$_HW_CORES  MPS_fallback=$PYTORCH_ENABLE_MPS_FALLBACK  blender_workers=$BLENDER_WORKERS"
fi

unset _HW_CORES