# -*- coding: utf-8 -*-
"""train_yolo_pose.py — YOLO11n-pose optimizado para Apple M-series (48GB unified memory)."""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from ultralytics import YOLO


def detect_optimal_config(reserve=0.20, max_workers=8):
    """
    Detecta recursos usando RAM TOTAL (no libre) para M-series con memoria unificada.
    En Apple Silicon la RAM unificada CPU+GPU se gestiona dinámicamente:
    vm_stat solo reporta paginas libres AHORA, pero el sistema puede comprimir/liberar.
    Usamos sysctl hw.memsize para la RAM fisica total y calculamos el 80%.
    """
    # CPU logicos
    try:
        total_cpu = int(subprocess.run(
            ["sysctl", "-n", "hw.logicalcpu"], capture_output=True, text=True
        ).stdout.strip())
    except Exception:
        total_cpu = os.cpu_count() or 12
    usable_cpu = max(2, int(total_cpu * (1 - reserve)))

    # RAM TOTAL fisica (no la libre) — clave para M-series
    total_ram_gb = 8.0
    try:
        mem_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        ).stdout.strip())
        total_ram_gb = mem_bytes / (1024 ** 3)
    except Exception:
        pass

    # Usar 80% de la RAM total para el training
    usable_ram = total_ram_gb * (1 - reserve)

    workers = max(3, min(max_workers, usable_cpu // 2))

    # Batch size agresivo para M-series con memoria unificada
    # YOLO11n-pose: ~150MB VRAM por batch=8 → batch=32 ~600MB → muy seguro con 48GB
    if usable_ram >= 32:
        batch = 32
    elif usable_ram >= 16:
        batch = 16
    else:
        batch = 8

    if torch.backends.mps.is_available():   device = "mps"
    elif torch.cuda.is_available():          device = "cuda"
    else:                                    device = "cpu"

    print(f"[OPT] CPU={total_cpu} cores | RAM total={total_ram_gb:.0f}GB | "
          f"RAM usable={usable_ram:.0f}GB (80%)")
    print(f"[OPT] -> device={device} | batch={batch} | workers={workers}")
    return dict(device=device, batch=batch, workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",     choices=["cenital", "lateral"], required=True)
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--batch",      type=int, default=None,
                    help="Si no se especifica, se auto-detecta (recomendado)")
    ap.add_argument("--imgsz",      type=int, default=640)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--model_base", default="yolo11n-pose.pt")
    pa = ap.parse_args()

    yaml_path = os.path.join(project_root, "training", f"dataset_yolo_pose_{pa.camera}.yaml")
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] {yaml_path}"); sys.exit(1)

    cfg     = detect_optimal_config()
    device  = pa.device if pa.device else cfg["device"]
    batch   = pa.batch  if pa.batch  else cfg["batch"]
    workers = cfg["workers"]
    amp     = (device == "cuda")   # MPS no soporta AMP estable

    runs_dir = os.path.join(project_root, "training", "runs_pose")
    name = f"yolo_pose_{pa.camera}"
    print(f"[train] {pa.camera} | epochs={pa.epochs} batch={batch} "
          f"device={device} amp={amp} workers={workers} cache=True exist_ok=True")

    YOLO(pa.model_base).train(
        data=yaml_path,
        epochs=pa.epochs,
        batch=batch,
        imgsz=pa.imgsz,
        device=device,
        amp=amp,
        workers=workers,
        cache=True,       # dataset en RAM
        patience=15,
        name=name,
        project=runs_dir,
        plots=True,
        verbose=True,
        exist_ok=True,    # continua desde checkpoint si existe
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=360.0, fliplr=0.5, flipud=0.5,
    )

    src = os.path.join(runs_dir, name, "weights", "best.pt")
    dst = os.path.join(project_root, "models", f"yolo_{pa.camera}_pose.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(src):
        shutil.copy2(src, dst); print(f"[train] ✅ -> {dst}")
    else:
        print(f"[WARN] no se encontro {src}")

if __name__ == "__main__":
    main()
