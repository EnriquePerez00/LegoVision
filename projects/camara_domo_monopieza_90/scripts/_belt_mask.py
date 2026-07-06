# -*- coding: utf-8 -*-
"""_belt_mask.py
================================================================================
Utilidad canónica para el chromakey de la cinta transportadora durante la
inferencia. Es la ÚNICA implementación permitida en el proyecto; todos los
scripts de evaluación deben usarla en lugar de hardcodear rangos HSV.

El rango HSV se DERIVA automáticamente de `scripts.scene_config.BELT_COLOR_HEX`
(fuente única de verdad del sistema) vía la función `belt_hsv_range()`.

Uso típico:
    from _belt_mask import filter_out_belt
    pixels_rgb, pixels_hsv = filter_out_belt(pixels_rgb, pixels_hsv)
================================================================================
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np


# ── Import de la fuente única de verdad ─────────────────────────────────────
# Añadir la raíz del repo al sys.path para poder importar `scripts.scene_config`
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_CANDIDATES = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),  # repo root
    os.path.abspath(os.path.join(_HERE, "..", "..", "..", "scripts")),
]
for _p in _REPO_ROOT_CANDIDATES:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts.scene_config import BELT_COLOR_HEX, BELT_COLOR_HSV_OCV, belt_hsv_range
except ImportError:
    # Fallback si el proyecto se ejecuta desde otra ubicación
    from scene_config import BELT_COLOR_HEX, BELT_COLOR_HSV_OCV, belt_hsv_range  # type: ignore


# ── API pública ─────────────────────────────────────────────────────────────
def get_belt_hsv_bounds(
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (60, 255),
    v_bounds: Tuple[int, int] = (30, 220),
) -> Tuple[int, int, int, int, int, int]:
    """Alias de `belt_hsv_range` para la API pública del módulo."""
    return belt_hsv_range(h_tol=h_tol, s_bounds=s_bounds, v_bounds=v_bounds)


def compute_belt_mask(
    pixels_hsv: np.ndarray,
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (60, 255),
    v_bounds: Tuple[int, int] = (30, 220),
) -> np.ndarray:
    """Devuelve una máscara booleana `True` donde el píxel corresponde a la cinta.

    Args:
        pixels_hsv: array (N, 3) de píxeles en HSV OpenCV (H:0-180, S:0-255, V:0-255).
        h_tol: tolerancia ± en H alrededor del H del color de la cinta.
        s_bounds: rango absoluto (s_min, s_max).
        v_bounds: rango absoluto (v_min, v_max).

    Returns:
        Array booleano (N,) con True en píxeles de cinta.
    """
    if pixels_hsv.ndim != 2 or pixels_hsv.shape[1] < 3:
        raise ValueError(
            f"pixels_hsv debe ser (N, 3) en HSV OpenCV. Recibido shape={pixels_hsv.shape}"
        )
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = get_belt_hsv_bounds(h_tol, s_bounds, v_bounds)
    h = pixels_hsv[:, 0]
    s = pixels_hsv[:, 1]
    v = pixels_hsv[:, 2]
    return (
        (h >= h_lo) & (h <= h_hi)
        & (s >= s_lo) & (s <= s_hi)
        & (v >= v_lo) & (v <= v_hi)
    )


def filter_out_belt(
    pixels_rgb: np.ndarray,
    pixels_hsv: np.ndarray,
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (60, 255),
    v_bounds: Tuple[int, int] = (30, 220),
) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra los píxeles de cinta de dos arrays RGB/HSV pareados.

    Devuelve la pareja `(pixels_rgb_sin_cinta, pixels_hsv_sin_cinta)`.
    Si el filtrado dejaría vacío, devuelve los arrays originales sin modificar
    para no romper aguas abajo el cálculo estadístico.
    """
    belt_mask = compute_belt_mask(pixels_hsv, h_tol, s_bounds, v_bounds)
    non_belt_mask = ~belt_mask
    if np.any(non_belt_mask):
        return pixels_rgb[non_belt_mask], pixels_hsv[non_belt_mask]
    return pixels_rgb, pixels_hsv


# ── Diagnóstico rápido si se ejecuta como script ────────────────────────────
if __name__ == "__main__":
    print(f"BELT_COLOR_HEX      = {BELT_COLOR_HEX}")
    print(f"BELT_COLOR_HSV_OCV  = {BELT_COLOR_HSV_OCV}  (H:0-180, S:0-255, V:0-255)")
    print(f"belt_hsv_range()    = {belt_hsv_range()}")