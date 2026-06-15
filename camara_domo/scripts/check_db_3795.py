# -*- coding: utf-8 -*-
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from camara_domo.scripts.inferencia_neuronal import load_db_universe

poses_db, colors_db = load_db_universe()

for ref in ["3795", "43121"]:
    if ref in poses_db:
        print(f"\nRef: {ref}")
        for p in poses_db[ref]:
            print(f" - Pose {p.pose_index}: zenith_silhouette_area={p.zenith_silhouette_area}, zenith_observable_area={p.zenith_observable_area}, height={p.lateral_height or p.effective_height}")
    else:
        print(f"\nRef {ref} no encontrada en DB")
