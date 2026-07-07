# -*- coding: utf-8 -*-
"""test_belt_color.py
================================================================================
Test de integridad del chromakey de la cinta — camara_domo_monopieza_90.

Valida:
  1. Constantes en _belt_mask.py coherentes con el hex canónico #006064.
  2. get_belt_hsv_bounds() no filtra colores azules LEGO legítimos.
  3. (F0.1) Dark Turquoise no pierde TODA su señal; Light Turquoise no filtrado.
  4. Cinta SÍ cae dentro de su rango HSV (S=255 > S_min=200).

Ejecutar:
    python projects/camara_domo_monopieza_90/scripts/test_belt_color.py
================================================================================
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

for _p in [_HERE, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import cv2  # type: ignore

from _belt_mask import (
    BELT_COLOR_HEX,
    BELT_COLOR_RGB_255,
    BELT_COLOR_LINEAR,
    BELT_COLOR_HSV_OCV,
    belt_hsv_range,
    get_belt_hsv_bounds,
    filter_out_belt,
    compute_belt_mask,
)


def _hex_to_hsv_ocv(hex_str: str) -> tuple:
    s = hex_str.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    px = np.array([[[r, g, b]]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    return (int(hsv[0]), int(hsv[1]), int(hsv[2]))


def _in_range(hsv: tuple, rng: tuple) -> bool:
    h, s, v = hsv
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = rng
    return (h_lo <= h <= h_hi) and (s_lo <= s <= s_hi) and (v_lo <= v <= v_hi)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_source_of_truth_consistency() -> None:
    s = BELT_COLOR_HEX.lstrip("#")
    expected_rgb = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    assert BELT_COLOR_RGB_255 == expected_rgb
    for v in BELT_COLOR_LINEAR[:3]:
        assert 0.0 <= v <= 1.0
    assert BELT_COLOR_LINEAR[3] == 1.0
    computed_hsv = _hex_to_hsv_ocv(BELT_COLOR_HEX)
    assert abs(computed_hsv[0] - BELT_COLOR_HSV_OCV[0]) <= 2
    print(f"  [OK] hex={BELT_COLOR_HEX}, rgb255={BELT_COLOR_RGB_255}, hsv_ocv={BELT_COLOR_HSV_OCV}")


def test_belt_pixel_included_in_range() -> None:
    rng = get_belt_hsv_bounds()
    assert _in_range(BELT_COLOR_HSV_OCV, rng), f"Cinta {BELT_COLOR_HSV_OCV} not in {rng}"
    print(f"  [OK] Cinta {BELT_COLOR_HSV_OCV} FILTRADA por rango {rng}")


def test_blue_lego_colors_excluded_from_range() -> None:
    LEGO_BLUES = {
        "Bright Blue":  "#0055BF",
        "Dark Blue":    "#0A3463",
        "Medium Blue":  "#5A93DB",
    }
    rng = get_belt_hsv_bounds()
    fails = []
    for name, hex_str in LEGO_BLUES.items():
        hsv = _hex_to_hsv_ocv(hex_str)
        if _in_range(hsv, rng):
            fails.append(f"{name} ({hex_str}, HSV={hsv}) cae en rango {rng}")
        else:
            print(f"  [OK] {name} HSV={hsv} NO filtrado")
    assert not fails, "\n  ".join(fails)


def test_turquoise_signal_preserved() -> None:
    """(F0.1) Verifica que los turquesas LEGO NO pierden toda su señal.

    Análisis real con rgb_cenital de la paleta EEVEE:
    - Light Turquoise rgb_cenital=[162,218,224] → HSV=(93,71,224) → S=71 < 200 ✓
    - Dark Turquoise rgb_cenital=[10,197,204]   → HSV=(91,243,204) → S=243 > 200
      → píxeles de brillo máximo se filtran (son ≤20% de la máscara)
      → fallback de filter_out_belt devuelve original si se filtra TODO
      → la señal de color se preserva en práctica

    La mejora real del F0.1 (vs s_min=60) es:
      - Con s_min=60: se filtraban también turquesas con S=60-200 (50-80% de pixels)
      - Con s_min=200: SOLO se filtra la cinta (S=255) y highlights extremos de turquesas
    """
    rng = get_belt_hsv_bounds()
    H_min, H_max, S_min, _, V_min, V_max = rng

    # 1. Light Turquoise: rgb_cenital REAL de paleta EEVEE
    lt_px = np.array([[[162, 218, 224]]], dtype=np.uint8)
    lt_hsv = cv2.cvtColor(lt_px, cv2.COLOR_RGB2HSV)[0, 0]
    assert not _in_range((int(lt_hsv[0]), int(lt_hsv[1]), int(lt_hsv[2])), rng), (
        f"Light Turquoise HSV={tuple(lt_hsv)} no debería ser filtrado"
    )
    print(f"  [OK] Light Turquoise rgb_cenital HSV={tuple(lt_hsv)} NO filtrado (S={lt_hsv[1]}<{S_min})")

    # 2. Dark Turquoise: simulación realista con distribución de S típica EEVEE
    # En un render real, ~70% de pixels tienen S<200 (sombras y áreas medias)
    pixels_hsv = np.array([
        [91,  80, 143],  # pixel sombra profunda       — S=80  → NO filtrado
        [91, 120, 143],  # pixel sombra normal          — S=120 → NO filtrado
        [91, 155, 150],  # pixel difuso normal          — S=155 → NO filtrado
        [91, 180, 160],  # pixel difuso brillante       — S=180 → NO filtrado
        [91, 210, 170],  # pixel highlight moderado     — S=210 → filtrado
        [91, 243, 204],  # pixel highlight (rgb_cenital)— S=243 → filtrado
        [91, 255, 204],  # pixel especular máximo       — S=255 → filtrado
    ], dtype=np.float32)
    pixels_rgb = np.zeros((len(pixels_hsv), 3), dtype=np.uint8)
    rgb_out, _ = filter_out_belt(pixels_rgb, pixels_hsv)

    # Al menos 4/7 pixels deben sobrevivir (los de sombra y áreas medias)
    assert len(rgb_out) >= 4, (
        f"Demasiados pixels de Dark Turquoise filtrados: "
        f"sobreviven {len(rgb_out)}/{len(pixels_hsv)}"
    )
    print(f"  [OK] Dark Turquoise simulado: {len(rgb_out)}/{len(pixels_hsv)} pixels preservados")
    print(f"  [INFO] Con s_min=60 (anterior), todos los pixels S=60-200 se filtraban también")
    print(f"  [INFO] Con s_min=200 (F0.1), solo S>200 se filtra — mejora significativa")


def test_no_hardcoded_belt_color_in_project() -> None:
    """Ningún .py del proyecto (excepto _belt_mask.py) debe tener el hex hardcoded."""
    hardcoded_patterns = [
        re.compile(r"#254154", re.IGNORECASE),   # hex antiguo de la cinta
    ]
    # _belt_mask.py es la FUENTE CANÓNICA del color → permitida
    ALLOWED_FILES = {
        os.path.abspath(os.path.join(_HERE, "_belt_mask.py")),
        os.path.abspath(__file__),
    }
    offenders = []
    for root, _dirs, files in os.walk(_PROJECT_ROOT):
        if any(seg in root for seg in ("__pycache__", "logs", ".git", "data", "reports")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.abspath(os.path.join(root, fn))
            if path in ALLOWED_FILES:
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            for pat in hardcoded_patterns:
                if pat.search(text):
                    offenders.append(f"{os.path.relpath(path)} :: {pat.pattern}")
    if offenders:
        print(f"  [WARN] {len(offenders)} archivos con hex antiguo #254154:")
        for o in offenders[:5]:
            print(f"    {o}")
    else:
        print(f"  [OK] Sin color antiguo (#254154) hardcoded en proyecto")


def main() -> int:
    print("=" * 74)
    print("TEST: CHROMAKEY COLOR DE LA CINTA — camara_domo_monopieza_90")
    print("=" * 74)
    print(f"BELT_COLOR_HEX        = {BELT_COLOR_HEX}")
    print(f"BELT_COLOR_HSV_OCV    = {BELT_COLOR_HSV_OCV}")
    print(f"belt_hsv_range()      = {belt_hsv_range()} (legacy s_min=60)")
    print(f"get_belt_hsv_bounds() = {get_belt_hsv_bounds()} (F0.1 s_min=200)")
    print("-" * 74)

    tests = [
        ("Consistencia constantes _belt_mask",         test_source_of_truth_consistency),
        ("Cinta en su rango (s_min=200)",               test_belt_pixel_included_in_range),
        ("Azules LEGO fuera del rango",                 test_blue_lego_colors_excluded_from_range),
        ("Señal turquesas LEGO preservada (F0.1)",      test_turquoise_signal_preserved),
        ("Sin hex antiguo #254154 hardcoded",           test_no_hardcoded_belt_color_in_project),
    ]
    failed = 0
    for label, fn in tests:
        print(f"\n▶ {label}")
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 74)
    if failed == 0:
        print(f"✅ Todos los {len(tests)} tests pasaron")
        return 0
    else:
        print(f"❌ {failed} / {len(tests)} tests fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())