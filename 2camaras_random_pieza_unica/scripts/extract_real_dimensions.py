# -*- coding: utf-8 -*-
"""scripts/extract_real_dimensions.py
Parses LDraw meshes to extract real dimensions in mm, updating config.yaml.
"""
from __future__ import annotations
import os
import sys
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from ldraw_mesh_parser import get_triangles

def main():
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dims = cfg["pieces"]["dimensions_mm"]
    updated_count = 0

    for ref in list(dims.keys()):
        tris = get_triangles(ref)
        if len(tris) > 0:
            av = tris.reshape(-1, 3)
            # LDraw coordinate system: X=width/length, Y=vertical height/thickness, Z=depth/width
            dx = (av[:, 0].max() - av[:, 0].min()) * 0.4
            dy = (av[:, 1].max() - av[:, 1].min()) * 0.4
            dz = (av[:, 2].max() - av[:, 2].min()) * 0.4
            
            # Round to 1 decimal place to align with LEGO standard dimensions (e.g., 3.2, 8.0, 9.6, 16.0, 32.0)
            new_dims = [round(float(dx), 1), round(float(dz), 1), round(float(dy), 1)]
            old_dims = dims[ref]
            
            if new_dims != old_dims:
                print(f"Update '{ref}': {old_dims} -> {new_dims}")
                dims[ref] = new_dims
                updated_count += 1
        else:
            print(f"[WARN] No LDraw file found for '{ref}'")

    if updated_count > 0:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"[OK] Updated {updated_count} part dimensions in config.yaml")
    else:
        print("[OK] All dimensions are already up to date.")

if __name__ == "__main__":
    main()
