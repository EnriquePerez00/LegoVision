# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import math

project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90"
legovic_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from color_classifier_all import ColorClassifierAll, rgb_to_lab, delta_e_ciede2000
from efficientnet_classifier_all import LegoEfficientNetClassifierAll

# Load classifier
clf = LegoEfficientNetClassifierAll()
hierarchical_clf = ColorClassifierAll()

# Load metadata
metadata_path = os.path.join(project_root, "data", "simulation_100_all", "simulation_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

# Read frames
frame = meta_data["frames"][4] # frame_004
piece = frame["visible_pieces"][0]
ref_gt = piece["ref"]
c_code_gt = str(piece["color_code"])
c_name_gt = piece["color_name"]

print(f"GT: Ref={ref_gt}, ColorCode={c_code_gt}, ColorName={c_name_gt}")

# Run first pass geometry to get candidates
# We will use dummy inputs because we just want to see how part_to_colors and gating behaves
allowed_names_direct = clf.part_to_colors.get(ref_gt, set())
print(f"Allowed color names for true ref directly: {allowed_names_direct}")

# Let's check catalog colors
print("\nCatalog colors matches:")
for c in hierarchical_clf.catalog_colors:
    c_name_lower = c["color_name"].strip().lower()
    if c_name_lower in allowed_names_direct:
        hex_str = c["color_hex"].lstrip("#")
        rgb_ref = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
        print(f"  {c['color_name']} (hex={c['color_hex']}, rgb={rgb_ref})")
