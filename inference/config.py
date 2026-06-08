import os
import sys
import platform
import logging

logger = logging.getLogger("LegoVision.Config")

# 1. Hardware and OS Detection
IS_MAC = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MAC and (platform.machine() == "arm64")

# 2. CPU and Multithreading Defaults
CPU_CORES = os.cpu_count() or 4
# Auto-limit CPU threads to leave at least 20% system capacity free
# On a 10-core M4 chip, this defaults to 8 cores.
AUTO_CPU_THREADS = max(1, CPU_CORES - 2) if CPU_CORES > 4 else CPU_CORES

# 3. PyTorch MPS and GPU Acceleration Defaults
try:
    if IS_APPLE_SILICON:
        # Prevent PyTorch from grabbing more than 70% of macOS unified memory
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.3"
    import torch
    if IS_APPLE_SILICON and torch.backends.mps.is_available():
        DEFAULT_DEVICE = "mps"
        logger.info("Apple Silicon detected. Auto-configured PyTorch to use MPS with unified memory limit.")
    else:
        DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using default torch device: {DEFAULT_DEVICE}")
except ImportError:
    DEFAULT_DEVICE = "cpu"
    logger.warning("Torch not available (likely running in Blender context). Using CPU device fallback.")

# Helper to release memory in MPS/unified RAM architectures
def auto_release_memory():
    if DEFAULT_DEVICE == "mps":
        try:
            import torch
            torch.mps.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()
