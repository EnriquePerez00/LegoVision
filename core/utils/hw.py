"""
core/utils/hw.py — Helpers HW-aware para LegoVision

Centraliza la detección de device y la configuración de threads para
que TODOS los scripts del repo hagan el mismo uso del hardware.

Target de referencia: MacBook Pro M4 (12 CPU · 48 GB · GPU Metal).
Prioridad de device: MPS > CUDA > CPU.

Uso mínimo en scripts:

    from core.utils.hw import get_device, set_torch_threads, log_hw_summary
    set_torch_threads()
    device = get_device()
    log_hw_summary(logger, device, batch_size=32)

Env vars respetadas:
    TORCH_NUM_THREADS   (default 10)
    TORCH_INTEROP_THREADS (default 2)
    LEGOVISION_FORCE_DEVICE  = mps|cuda|cpu  (override manual)
"""
from __future__ import annotations

import os
import platform
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# --------------------------------------------------------------------
#  Constantes por defecto ajustadas al M4 (12 CPU · 48 GB)
# --------------------------------------------------------------------
DEFAULT_TORCH_THREADS = 10       # 12 CPU − 2 reservados para OS/UI
DEFAULT_INTEROP_THREADS = 2
DEFAULT_DATALOADER_WORKERS = 8
DEFAULT_BATCH_SIZES = {
    "yolo": 32,
    "sam": 8,
    "efficientnet_b0": 64,
    "dinov2_vitb_518": 16,
    "dinov2_vits_224": 32,
    "mlp": 256,
}


# --------------------------------------------------------------------
#  Device
# --------------------------------------------------------------------
def get_device(prefer: Optional[str] = None):
    """
    Devuelve `torch.device` óptimo para la máquina actual.

    Prioridad:
        1. env `LEGOVISION_FORCE_DEVICE` (mps|cuda|cpu) si está definida.
        2. Parámetro `prefer` si está soportado.
        3. Auto-detect: mps > cuda > cpu.
    """
    if torch is None:
        raise RuntimeError("PyTorch no está instalado en este entorno.")

    forced = os.environ.get("LEGOVISION_FORCE_DEVICE", "").lower().strip()
    if forced:
        return _resolve_device(forced)

    if prefer:
        try:
            return _resolve_device(prefer)
        except RuntimeError:
            pass  # cae al auto-detect

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_device(name: str):
    name = name.lower().strip()
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS solicitado pero no disponible.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitado pero no disponible.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Device desconocido: {name!r}")


# --------------------------------------------------------------------
#  Threads
# --------------------------------------------------------------------
def set_torch_threads(
    num_threads: Optional[int] = None,
    num_interop_threads: Optional[int] = None,
) -> tuple[int, int]:
    """
    Configura threads de PyTorch respetando env vars.

    Devuelve (num_threads, num_interop_threads) aplicados.
    """
    if torch is None:
        raise RuntimeError("PyTorch no está instalado.")

    n = num_threads or int(
        os.environ.get("TORCH_NUM_THREADS", DEFAULT_TORCH_THREADS)
    )
    interop = num_interop_threads or int(
        os.environ.get("TORCH_INTEROP_THREADS", DEFAULT_INTEROP_THREADS)
    )

    torch.set_num_threads(n)
    try:
        # set_num_interop_threads solo puede llamarse antes de crear
        # el primer op paralelo. Si falla, se ignora silenciosamente.
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    return n, interop


# --------------------------------------------------------------------
#  Precision helpers
# --------------------------------------------------------------------
def recommended_dtype(device) -> "torch.dtype":
    """
    Dtype recomendado para inferencia según device.
    - MPS  → float16 (aprovecha unidades FP16 del Neural Engine / GPU)
    - CUDA → float16 (o bfloat16 en Ampere+, pero no forzamos)
    - CPU  → float32
    """
    if torch is None:
        raise RuntimeError("PyTorch no está instalado.")
    if device.type in ("mps", "cuda"):
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------
#  Presupuesto de recursos
# --------------------------------------------------------------------
def suggested_num_workers(cap: int = DEFAULT_DATALOADER_WORKERS) -> int:
    """Nº de workers razonable para DataLoader (cap ≤ 8 por defecto)."""
    return max(1, min(cap, (os.cpu_count() or 4) - 2))


def suggested_blender_workers() -> int:
    """Nº de procesos Blender en paralelo (reserva 2 CPU al SO)."""
    return max(1, (os.cpu_count() or 4) - 2)


def suggested_batch_size(model_key: str) -> int:
    """Batch size recomendado por modelo (ver DEFAULT_BATCH_SIZES)."""
    return DEFAULT_BATCH_SIZES.get(model_key.lower(), 16)


# --------------------------------------------------------------------
#  Diagnóstico
# --------------------------------------------------------------------
def hw_summary(device=None) -> dict:
    """Devuelve un dict con la config HW efectiva."""
    if torch is None:
        return {"torch": "not installed"}

    dev = device or get_device()
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(dev),
        "torch_num_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "TORCH_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
            )
            if os.environ.get(k)
        },
    }
    return info


def log_hw_summary(logger, device=None, batch_size: Optional[int] = None) -> None:
    """Loggea el resumen HW en el logger dado."""
    info = hw_summary(device)
    if batch_size is not None:
        info["batch_size"] = batch_size
    logger.info("[HW] " + " | ".join(f"{k}={v}" for k, v in info.items()))


# --------------------------------------------------------------------
#  Semilla reproducibilidad
# --------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Fija semillas en random, numpy y torch (incluye MPS/CUDA si aplica)."""
    import random

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
            except AttributeError:
                pass


__all__ = [
    "DEFAULT_TORCH_THREADS",
    "DEFAULT_INTEROP_THREADS",
    "DEFAULT_DATALOADER_WORKERS",
    "DEFAULT_BATCH_SIZES",
    "get_device",
    "set_torch_threads",
    "recommended_dtype",
    "suggested_num_workers",
    "suggested_blender_workers",
    "suggested_batch_size",
    "hw_summary",
    "log_hw_summary",
    "seed_everything",
]