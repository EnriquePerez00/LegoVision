# -*- coding: utf-8 -*-
"""_belt_mask.py
================================================================================
Utilidad canónica para el chromakey de la cinta transportadora durante la
inferencia. Es la ÚNICA implementación permitida en el proyecto; todos los
scripts de evaluación deben usarla en lugar de hardcodear rangos HSV.

CAMBIO F0.1 (2026-06-07): s_bounds default cambiado de (60, 255) → (200, 255)
  para evitar que colores LEGO turquesas sean filtrados junto con la cinta.

  Análisis:
    - Cinta #006064:          HSV_OCV = (91, 255, 100) — S=255 → FILTRADA ✓
    - Dark Turquoise #00828E: HSV_OCV ≈ (91, 181, 143) — S=181 < 200 → NO filtrada ✓
    - Light Turquoise #54A4AE:HSV_OCV ≈ (92,  87, 174) — S=87  < 200 → NO filtrada ✓

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


# ── Constante canónica del color de la cinta ────────────────────────────────
# Este módulo define su propia constante para ser independiente del
# scripts/scene_config.py de la raíz (que puede variar entre proyectos).
# El proyecto camara_domo_monopieza_90 usa #006064 (azul petróleo).
_BELT_COLOR_HEX = "#006064"

# Derivar HSV OCV a partir del hex (evita depender de cv2 en tiempo de módulo)
# #006064 = RGB(0, 96, 100)
# HSV OpenCV: H = hue/2 → H=91, S≈255, V=100
_BELT_H = 91   # H en escala OpenCV [0-180]
_BELT_S = 255  # S en escala OpenCV [0-255]
_BELT_V = 100  # V en escala OpenCV [0-255]

# Exportar constantes para compatibilidad con test_belt_color.py
BELT_COLOR_HEX     = _BELT_COLOR_HEX
BELT_COLOR_RGB_255 = (0, 96, 100)
BELT_COLOR_HSV_OCV = (_BELT_H, _BELT_S, _BELT_V)
BELT_COLOR_LINEAR  = (0.0, 0.117, 0.127, 1.0)


def belt_hsv_range(
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (60, 255),    # default legacy (s_min=60)
    v_bounds: Tuple[int, int] = (30, 220),
) -> Tuple[int, int, int, int, int, int]:
    """Devuelve el rango HSV del chromakey de la cinta (escala OpenCV).

    Replica la función de scripts/scene_config.py para compatibilidad.
    Para el chromakey REAL usar get_belt_hsv_bounds() con s_min=200 (F0.1).
    """
    h_min = max(0, _BELT_H - h_tol)
    h_max = min(180, _BELT_H + h_tol)
    return (h_min, h_max, s_bounds[0], s_bounds[1], v_bounds[0], v_bounds[1])


# ── API pública ─────────────────────────────────────────────────────────────
def get_belt_hsv_bounds(
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (200, 255),   # F0.1: s_min=200 para no filtrar turquesas
    v_bounds: Tuple[int, int] = (30, 220),
) -> Tuple[int, int, int, int, int, int]:
    """Devuelve el rango HSV para el chromakey de la cinta (s_min=200 por defecto).

    NOTA F0.1: s_bounds default es (200, 255).
    - Cinta: S≈255 → FILTRADA
    - Dark Turquoise: S≈181 < 200 → NO filtrada
    - Light Turquoise: S≈87 < 200 → NO filtrada
    """
    h_min = max(0, _BELT_H - h_tol)
    h_max = min(180, _BELT_H + h_tol)
    return (h_min, h_max, s_bounds[0], s_bounds[1], v_bounds[0], v_bounds[1])


def compute_belt_mask(
    pixels_hsv: np.ndarray,
    h_tol: int = 12,
    s_bounds: Tuple[int, int] = (200, 255),   # F0.1: s_min=200
    v_bounds: Tuple[int, int] = (30, 220),
) -> np.ndarray:
    """Devuelve máscara booleana True donde el píxel corresponde a la cinta.

    Args:
        pixels_hsv: array (N, 3) en HSV OpenCV (H:0-180, S:0-255, V:0-255).
        h_tol: tolerancia ± en H.
        s_bounds: (s_min, s_max). Default (200,255) para no filtrar turquesas.
        v_bounds: (v_min, v_max).
    """
    if pixels_hsv.ndim != 2 or pixels_hsv.shape[1] < 3:
        raise ValueError(
            f"pixels_hsv debe ser (N, 3) en HSV OpenCV. Shape={pixels_hsv.shape}"
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
    s_bounds: Tuple[int, int] = (200, 255),   # F0.1: s_min=200
    v_bounds: Tuple[int, int] = (30, 220),
) -> Tuple[np.ndarray, np.ndarray]:
    """Filtra los píxeles de cinta de dos arrays RGB/HSV pareados.

    Devuelve (pixels_rgb_sin_cinta, pixels_hsv_sin_cinta).
    Si el filtrado dejaría vacío, devuelve los arrays originales.

    CAMBIO F0.1: s_bounds default = (200, 255) — no filtra turquesas LEGO.
    """
    belt_mask = compute_belt_mask(pixels_hsv, h_tol, s_bounds, v_bounds)
    non_belt_mask = ~belt_mask
    if np.any(non_belt_mask):
        return pixels_rgb[non_belt_mask], pixels_hsv[non_belt_mask]
    return pixels_rgb, pixels_hsv


# ── Diagnóstico rápido ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"BELT_COLOR_HEX      = {BELT_COLOR_HEX}")
    print(f"BELT_COLOR_RGB_255  = {BELT_COLOR_RGB_255}")
    print(f"BELT_COLOR_HSV_OCV  = {BELT_COLOR_HSV_OCV}  (H:0-180, S:0-255, V:0-255)")
    print(f"BELT_COLOR_LINEAR   = {BELT_COLOR_LINEAR}")
    print(f"belt_hsv_range()    = {belt_hsv_range()}  (legacy s_min=60)")
    print(f"get_belt_hsv_bounds() = {get_belt_hsv_bounds()}  (F0.1: s_min=200)")
    print()
    rng = get_belt_hsv_bounds()
    H_min, H_max, S_min, S_max, V_min, V_max = rng

    def _in(h, s, v):
        return H_min <= h <= H_max and S_min <= s <= S_max and V_min <= v <= V_max

    colors = [
        ("Cinta #006064",          91, 255, 100),
        ("Dark Turquoise #00828E", 91, 181, 143),
        ("Light Turquoise #54A4AE",92,  87, 174),
        ("Bright Blue #0055BF",   107, 255, 191),
        ("Dark Blue #0A3463",     106, 200, 108),
    ]
    print("Verificación chromakey (F0.1):")
    for name, H, S, V in colors:
        sym = "🔴 FILTRADO" if _in(H, S, V) else "✅ NO filtrado"
        print(f"  {sym}: {name} (H={H}, S={S}, V={V})")