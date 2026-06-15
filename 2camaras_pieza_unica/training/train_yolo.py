# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/training/train_yolo.py
===============================================
Entrena modelos YOLOv11-nano para las cámaras cenital y lateral.
Clase única: lego_piece.
"""
import os, sys, yaml, shutil, random, argparse
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, legovic_root)

from config_loader import cfg
from database import supabase_client
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("yolo")

RUN_ID = None
best_map50 = 0.0
epochs_without_improvement = 0


def on_fit_epoch_end(trainer):
    global best_map50, epochs_without_improvement
    if not RUN_ID:
        return
    epoch = trainer.epoch + 1
    loss = float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') and trainer.loss_items is not None else 0.0
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    map50 = float(metrics.get('metrics/mAP50(B)', 0.0))
    val_box_loss = float(metrics.get('val/box_loss', 0.0))
    val_cls_loss = float(metrics.get('val/cls_loss', 0.0))
    val_loss = val_box_loss + val_cls_loss

    if map50 >= best_map50 + 0.001:
        best_map50 = map50
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    log.info(f"Época {epoch}/{trainer.epochs}: loss={loss:.4f}, val_loss={val_loss:.4f}, mAP50={map50:.4f}")

    if epochs_without_improvement >= 15:
        trainer.stop = True
        log.info("⏹ Parada temprana activada.")

    try:
        supabase_client.update_training_progress(RUN_ID, epoch, loss, val_loss, map50, "")
    except Exception:
        pass


def prepare_dataset(raw_dir, processed_dir, train_ratio=0.7):
    """Split raw dataset into train/val."""
    images_dir = os.path.join(raw_dir, "images")
    labels_dir = os.path.join(raw_dir, "labels")
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        raise FileNotFoundError(f"No images found in {images_dir}")

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    for subset in ["train", "val"]:
        os.makedirs(os.path.join(processed_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, subset, "labels"), exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    if not val_files and train_files:
        val_files = [train_files.pop()]
    if not train_files and val_files:
        train_files = [val_files.pop()]

    def copy_files(file_list, subset):
        for img_name in file_list:
            lbl_name = img_name.replace(".png", ".txt")
            shutil.copy2(os.path.join(images_dir, img_name),
                         os.path.join(processed_dir, subset, "images", img_name))
            src_lbl = os.path.join(labels_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(processed_dir, subset, "labels", lbl_name))

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    log.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val")


def generate_yaml(processed_dir, yaml_path):
    """Generate dataset YAML for YOLO training."""
    yaml_data = {
        'path': os.path.abspath(processed_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': {0: 'lego_piece'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def main():
    global RUN_ID
    load_dotenv(override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="cenital", choices=["cenital", "lateral"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    import time as _time
    _t_start = _time.perf_counter()

    epochs = args.epochs or cfg.yolo.training.epochs
    batch = args.batch or cfg.yolo.training.batch_size
    imgsz = cfg.yolo.training.image_size
    device = cfg.yolo.training.device

    log_execution_header(log, "train_yolo.py",
                         camera=args.camera, epochs=epochs, batch=batch)

    config_dict = {
        "camera": args.camera,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "subproject": "2camaras_pieza_unica"
    }
    RUN_ID = supabase_client.create_training_run(epochs, config_dict)

    try:
        raw_dir = os.path.join(project_root, "data", f"yolo_{args.camera}")
        processed_dir = os.path.join(project_root, "data", f"yolo_{args.camera}_split")

        prepare_dataset(raw_dir, processed_dir)
        yaml_path = os.path.join(processed_dir, "dataset.yaml")
        generate_yaml(processed_dir, yaml_path)

        model = YOLO("yolo11n.pt")
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            patience=15,
            imgsz=imgsz,
            device=device,
            amp=cfg.yolo.training.amp,
            workers=cfg.yolo.training.workers,
            plots=False,
            project=os.path.join(project_root, "data", "yolo_runs"),
            name=f"{args.camera}",
            exist_ok=True,
            degrees=360.0,
            fliplr=0.5,
            flipud=0.5,
        )

        # Copy best weights
        best_pt_src = os.path.join(project_root, "data", "yolo_runs", args.camera, "weights", "best.pt")
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        best_pt_dst = os.path.join(models_dir, f"yolo_{args.camera}.pt")

        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dst)
            log.info(f"Modelo guardado: {best_pt_dst}")

        supabase_client.complete_training_run(RUN_ID, "completed",
                                              f"Training {args.camera} completed.")
    except Exception as e:
        log.error(f"Training failed: {e}")
        supabase_client.complete_training_run(RUN_ID, "failed", f"ERROR: {str(e)}")

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "train_yolo.py", duration_s=_duration, camera=args.camera)


if __name__ == "__main__":
    main()