# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/compute_canonical_keypoints.py
=====================================================================
Genera el archivo `data/canonical_keypoints.json` con 9 keypoints
por cada `ref` del set 75078-1, en frame canonico LDraw (BU = 10 cm).

Los 9 keypoints son las 8 esquinas del AABB del mesh + el centroide:

    KP_0..3 = bottom (Z = -h/2)   (BL, BR, FR, FL)
    KP_4..7 = top    (Z = +h/2)   (BL, BR, FR, FL)
    KP_8    = centroide           (0, 0, 0)

Las dimensiones se leen de `cfg.pieces.dimensions_mm[ref]` =
(L_mm, W_mm, H_mm) y se convierten a BU dividiendo por 100
(1 BU = 10 cm, 1 mm = 0.01 BU). En el frame canonico de la pieza
(post-`normalize_piece`) los ejes son:
    X = largo  L
    Y = ancho  W
    Z = alto   H

NOTA: El frame canonico LDraw de algunas piezas tiene Y como vertical
en lugar de Z (el simulador de poses lo refleja en `contact_normal`).
Por simplicidad asumimos que en el frame post-`normalize_piece`
las dimensiones se mapean (L, W, H) -> (X, Y, Z). El bug residual
de orientacion lo asume el caller cuando aplica `apply_stable_pose`
(que rota la pieza para que `contact_normal` apunte a -Z; los KP se
transforman con la misma matriz mundo).

Uso:
  .venv/bin/python 2camaras_random_pieza_unica/scripts/compute_canonical_keypoints.py
"""
from __future__ import annotations

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from config_loader import cfg


MM_TO_BU = 0.01  # 1 mm = 0.01 BU (1 BU = 10 cm = 100 mm)
OUTPUT_PATH = os.path.join(project_root, "data", "canonical_keypoints.json")


def aabb_to_keypoints(L_mm, W_mm, H_mm):
    """Devuelve 9 keypoints en BU para un AABB centrado en (0,0,0).

    Orden:
      0: bottom_back_left   (-L/2, -W/2, -H/2)
      1: bottom_back_right  (+L/2, -W/2, -H/2)
      2: bottom_front_right (+L/2, +W/2, -H/2)
      3: bottom_front_left  (-L/2, +W/2, -H/2)
      4: top_back_left      (-L/2, -W/2, +H/2)
      5: top_back_right     (+L/2, -W/2, +H/2)
      6: top_front_right    (+L/2, +W/2, +H/2)
      7: top_front_left     (-L/2, +W/2, +H/2)
      8: centroid           (0, 0, 0)
    """
    Lx = L_mm * MM_TO_BU * 0.5
    Wy = W_mm * MM_TO_BU * 0.5
    Hz = H_mm * MM_TO_BU * 0.5
    return [
        [-Lx, -Wy, -Hz],  # 0
        [+Lx, -Wy, -Hz],  # 1
        [+Lx, +Wy, -Hz],  # 2
        [-Lx, +Wy, -Hz],  # 3
        [-Lx, -Wy, +Hz],  # 4
        [+Lx, -Wy, +Hz],  # 5
        [+Lx, +Wy, +Hz],  # 6
        [-Lx, +Wy, +Hz],  # 7
        [ 0.0,  0.0,  0.0],  # 8
    ]


def main():
    refs = list(cfg.pieces.selected_parts)
    dims_cfg = cfg.pieces.dimensions_mm

    out = {
        "format_version": 1,
        "unit": "BU (1 BU = 100 mm)",
        "kpt_shape": [9, 3],
        "kpt_names": [
            "bottom_back_left", "bottom_back_right",
            "bottom_front_right", "bottom_front_left",
            "top_back_left", "top_back_right",
            "top_front_right", "top_front_left",
            "centroid",
        ],
        "flip_idx": [1, 0, 3, 2, 5, 4, 7, 6, 8],
        "pieces": {},
    }
    missing = []
    for ref in refs:
        if not hasattr(dims_cfg, ref):
            missing.append(ref)
            continue
        d = getattr(dims_cfg, ref)
        L, W, H = float(d[0]), float(d[1]), float(d[2])
        out["pieces"][ref] = {
            "dimensions_mm": [L, W, H],
            "keypoints_bu": aabb_to_keypoints(L, W, H),
        }
    if missing:
        print(f"[WARN] {len(missing)} refs sin dimensions_mm: {missing}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {len(out['pieces'])} refs escritas en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()