# -*- coding: utf-8 -*-
"""compute_canonical_keypoints.py
Genera canonical_keypoints.json para TODAS las refs de la BD (todos los sets),
sin hardcoding a ningun set especifico.
"""
from __future__ import annotations
import json, os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from core.utils.config_loader import cfg
from core.db.set_catalog import REAL_SETS

MM_TO_BU = 0.01
OUTPUT_PATH = os.path.join(project_root, "data", "canonical_keypoints.json")


def aabb_to_keypoints(L_mm, W_mm, H_mm):
    Lx = L_mm * MM_TO_BU * 0.5
    Wy = W_mm * MM_TO_BU * 0.5
    Hz = H_mm * MM_TO_BU * 0.5
    return [
        [-Lx, -Wy, -Hz], [+Lx, -Wy, -Hz], [+Lx, +Wy, -Hz], [-Lx, +Wy, -Hz],
        [-Lx, -Wy, +Hz], [+Lx, -Wy, +Hz], [+Lx, +Wy, +Hz], [-Lx, +Wy, +Hz],
        [0.0, 0.0, 0.0],
    ]


def get_all_refs_from_bd():
    """Todas las refs unicas de toda la BD (sin stickers/minifigs)."""
    refs = set()
    for set_id, set_data in REAL_SETS.items():
        for p in set_data.get("parts", []):
            ref = p.get("ref", "")
            if not ref:
                continue
            if "stk" in ref.lower() or ref.lower().startswith("sw") or ref.lower().startswith("fig"):
                continue
            refs.add(ref)
    return sorted(refs)


def main():
    refs = get_all_refs_from_bd()
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
        print(f"[WARN] {len(missing)} refs sin dimensions_mm (omitidas): {missing[:10]}{'...' if len(missing)>10 else ''}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {len(out['pieces'])} refs escritas en {OUTPUT_PATH}")
    print(f"     Total BD: {len(refs)} | Con dims: {len(out['pieces'])} | Sin dims: {len(missing)}")


if __name__ == "__main__":
    main()
