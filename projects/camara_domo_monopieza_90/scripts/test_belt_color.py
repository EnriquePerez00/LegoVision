# -*- coding: utf-8 -*-
"""test_belt_color.py
================================================================================
Test de integridad de la fuente única de verdad del color de la cinta.

Valida que:
  1. `scripts.scene_config.BELT_COLOR_HEX` y sus derivaciones son coherentes.
  2. `belt_hsv_range()` NO incluye colores azules LEGO comunes (Bright Blue,
     Dark Blue, Medium Blue) para no descartar píxeles legítimos de piezas.
  3. Ningún script del proyecto contiene hardcoded el hex/RGB del color de
     la cinta fuera de `scripts/scene_config.py`.

Ejecutar:
    python projects/camara_domo_monopieza_90/scripts/test_belt_color.py
================================================================================
"""
from __future__ import annotations

import os
import re
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import cv2  # type: ignore

from scripts.scene_config import (  # type: ignore
    BELT_COLOR_HEX,
    BELT_COLOR_RGB_255,
    BELT_COLOR_LINEAR,
    BELT_COLOR_HSV_OCV,
    belt_hsv_range,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
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


# ── Tests ───────────────────────────────────────────────────────────────────
def test_source_of_truth_consistency() -> None:
    """La cadena hex primaria y las derivaciones deben ser coherentes."""
    # RGB255 debe derivarse correctamente de hex
    s = BELT_COLOR_HEX.lstrip("#")
    expected_rgb = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    assert BELT_COLOR_RGB_255 == expected_rgb, (
        f"BELT_COLOR_RGB_255={BELT_COLOR_RGB_255} no coincide con hex {BELT_COLOR_HEX}"
    )

    # Linear debe estar en [0, 1] y tener alpha=1.0
    for v in BELT_COLOR_LINEAR[:3]:
        assert 0.0 <= v <= 1.0, f"BELT_COLOR_LINEAR fuera de rango: {v}"
    assert BELT_COLOR_LINEAR[3] == 1.0, "Alpha de BELT_COLOR_LINEAR debe ser 1.0"

    # HSV OpenCV: H<=180, S<=255, V<=255
    h, s_, v = BELT_COLOR_HSV_OCV
    assert 0 <= h <= 180, f"H fuera de rango: {h}"
    assert 0 <= s_ <= 255, f"S fuera de rango: {s_}"
    assert 0 <= v <= 255, f"V fuera de rango: {v}"

    print(f"  [OK] Consistencia: hex={BELT_COLOR_HEX}, rgb255={BELT_COLOR_RGB_255}, "
          f"hsv_ocv={BELT_COLOR_HSV_OCV}")


def test_belt_pixel_included_in_range() -> None:
    """El propio color de la cinta DEBE caer dentro de belt_hsv_range()."""
    rng = belt_hsv_range()
    assert _in_range(BELT_COLOR_HSV_OCV, rng), (
        f"El color de la cinta {BELT_COLOR_HSV_OCV} no cae en su propio rango {rng}"
    )
    print(f"  [OK] El color de la cinta {BELT_COLOR_HSV_OCV} cae en rango {rng}")


def test_blue_lego_colors_excluded_from_range() -> None:
    """Los azules LEGO habituales NO deben caer en el rango del chromakey.

    De lo contrario el filtro descartaría píxeles legítimos de piezas azules.
    """
    LEGO_BLUES = {
        "Bright Blue":  "#0055BF",   # code 23  (LEGO 7 en Rebrickable)
        "Dark Blue":    "#0A3463",   # code 272 / 63
        "Medium Blue":  "#5A93DB",   # code 42
    }
    rng = belt_hsv_range()
    fails = []
    for name, hex_str in LEGO_BLUES.items():
        hsv = _hex_to_hsv_ocv(hex_str)
        if _in_range(hsv, rng):
            fails.append(f"{name} ({hex_str}, HSV={hsv}) cae en rango {rng}")
        else:
            print(f"  [OK] {name} ({hex_str}, HSV={hsv}) NO cae en rango")
    assert not fails, "Colores LEGO azules incluidos en chromakey:\n  " + "\n  ".join(fails)


def test_no_hardcoded_belt_color_in_project() -> None:
    """Ningún archivo .py del proyecto (excepto scene_config.py) debe contener
    el valor numérico de la cinta hardcoded.
    """
    project_dir = os.path.abspath(os.path.join(_HERE, ".."))
    scene_config_path = os.path.abspath(
        os.path.join(_REPO_ROOT, "scripts", "scene_config.py")
    )
    hardcoded_patterns = [
        re.compile(re.escape(BELT_COLOR_HEX), re.IGNORECASE),
        re.compile(re.escape(BELT_COLOR_HEX.upper()), re.IGNORECASE),
        re.compile(r"[\(\[]\s*0\s*,\s*96\s*,\s*100\s*[\)\]]"),
        re.compile(r"[\(\[]\s*37\s*,\s*65\s*,\s*84\s*[\)\]]"),  # color antiguo
        re.compile(r"#254154", re.IGNORECASE),                   # hex antiguo
    ]
    offenders = []
    for root, _dirs, files in os.walk(project_dir):
        # excluir cachés, logs, backups
        if any(seg in root for seg in ("__pycache__", "logs", ".git", "data", "reports")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.abspath(os.path.join(root, fn))
            if path == scene_config_path or path == os.path.abspath(__file__):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            for pat in hardcoded_patterns:
                if pat.search(text):
                    offenders.append(f"{path} :: {pat.pattern}")
    if offenders:
        print("\n[FAIL] Se encontraron valores hardcoded del color:")
        for o in offenders:
            print(f"  - {o}")
        raise AssertionError(f"{len(offenders)} match(es) hardcoded")
    print(f"  [OK] Ningún archivo .py del proyecto contiene el color hardcoded")


# ── Runner ──────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 74)
    print("TEST DE INTEGRIDAD: FUENTE ÚNICA DE VERDAD DEL COLOR DE LA CINTA")
    print("=" * 74)
    print(f"BELT_COLOR_HEX      = {BELT_COLOR_HEX}")
    print(f"BELT_COLOR_RGB_255  = {BELT_COLOR_RGB_255}")
    print(f"BELT_COLOR_LINEAR   = {BELT_COLOR_LINEAR}")
    print(f"BELT_COLOR_HSV_OCV  = {BELT_COLOR_HSV_OCV}")
    print(f"belt_hsv_range()    = {belt_hsv_range()}")
    print("-" * 74)

    tests = [
        ("Consistencia hex/rgb/linear/hsv", test_source_of_truth_consistency),
        ("Cinta dentro de su rango HSV",    test_belt_pixel_included_in_range),
        ("Azules LEGO fuera del rango",     test_blue_lego_colors_excluded_from_range),
        ("Sin hex hardcoded en proyecto",   test_no_hardcoded_belt_color_in_project),
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
            print(f"  [ERROR] {e}")
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