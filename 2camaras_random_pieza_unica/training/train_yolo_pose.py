# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/training/train_yolo_pose.py
============================================================
Entrena YOLO11n-pose sobre el dataset de keypoints canonicos generado
con `generate_yolo_pose_training_dataset.py`.

Uso:
  .venv/bin/python 2camaras_random_pieza_unica/training/train_yolo_pose.py \\
      --camera cenital --epochs 15

Output:
  models/yolo_<camera>_pose.pt   (best.pt copiado de runs/pose/<name>/weights)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["cenital", "lateral"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model_base", default="yolo11n-pose.pt")
    pa = parser.parse_args()

    yaml_path = os.path.join(
        project_root, "training",
        f"dataset_yolo_pose_{pa.camera}.yaml",
    )
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] No existe {yaml_path}")
        sys.exit(1)

    model = YOLO(pa.model_base)
    name = f"yolo_pose_{pa.camera}"
    runs_dir = os.path.join(project_root, "training", "runs_pose")

    print(f"[train] yaml={yaml_path} model_base={pa.model_base} epochs={pa.epochs}")
    model.train(
        data=yaml_path,
        epochs=pa.epochs,
        batch=pa.batch,
        imgsz=pa.imgsz,
        device=pa.device,
        name=name,
        project=runs_dir,
        plots=True,
        verbose=True,
        cache=False,
    )

    src_best = os.path.join(runs_dir, name, "weights", "best.pt")
    dst = os.path.join(project_root, "models", f"yolo_{pa.camera}_pose.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(src_best):
        shutil.copy2(src_best, dst)
        print(f"[train] Pesos copiados a {dst}")
    else:
        print(f"[WARN] no se encontro best.pt en {src_best}")


if __name__ == "__main__":
    main()