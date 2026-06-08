# -*- coding: utf-8 -*-
"""scripts/reindex_dinov2_eevee.py
=====================================
Re-indexa los embeddings DINOv2 usando las nuevas imágenes de referencia
renderizadas con EEVEE (mismo dominio visual que las imágenes de test).

Limpia los embeddings actuales (iter7/Cycles) y los reemplaza con los
nuevos (EEVEE+belt+training-lighting).

Optimizaciones M4 Pro (OPT-1):
  - ThreadPoolExecutor(4) para IO paralelo de imágenes
  - Prefetch de imágenes en background mientras DINOv2 procesa

Uso:
  .venv/bin/python scripts/reindex_dinov2_eevee.py \
      --ref_dir data/dinov2_refs_full
"""
import os, sys, re, glob, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)  # LegoVision/ root for inference/, training/ modules
sys.path.insert(0, project_root)   # 2camaras_multi_pieza/ — config_loader, database
sys.path.insert(0, legovic_root)   # LegoVision/ — inference/, training/

from database import supabase_client
from training.index_synthetic_renders import (
    COLOR_HEX_TO_CODE, get_device, load_dinov2,
    get_transform, preprocess_render, extract_embedding,
)
from inference.knn_classifier import LegoProjectionHead

# ── Logging ──────────────────────────────────────────────────────────────────
import sys as _sys
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("dinov2")

# Regex: ref_PARTREF_COLORHEX_poseNN_rotNNN.png (supports optional _instXX)
FILENAME_RE = re.compile(
    r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})_pose(\d+)_rot(\d+)(?:_inst\d+)?\.png",
    re.IGNORECASE,
)

CAMERAS = {
    "cenital":   1,
    "frontal":   2,
}


def filter_centermost_instances(image_paths):
    groups = {}
    for p in image_paths:
        m = FILENAME_RE.match(os.path.basename(p))
        if not m:
            continue
        part_ref     = m.group(1)
        color_hex    = m.group(2).upper()
        pose_idx     = int(m.group(3))
        rotation_deg = int(m.group(4))
        
        filename = os.path.basename(p)
        inst_match = re.search(r"_inst(\d+)\.png", filename, re.IGNORECASE)
        inst_idx = int(inst_match.group(1)) if inst_match else 0
        
        key = (part_ref, color_hex, pose_idx, rotation_deg)
        if key not in groups:
            groups[key] = []
        groups[key].append((inst_idx, p))
        
    filtered_paths = []
    for key, lst in groups.items():
        lst.sort()
        center_idx = len(lst) // 2
        filtered_paths.append(lst[center_idx][1])
    return sorted(filtered_paths)


def main():
    _t0 = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str,
                        default=os.path.join(project_root, "data", "dinov2_refs_full"),
                        help="Directorio raíz con subcarpetas cenital/, lateral_l/, lateral_r/")
    parser.add_argument("--no_clear", action="store_true",
                        help="No limpiar embeddings existentes antes de indexar")
    args = parser.parse_args()

    ref_dir = args.ref_dir

    log_execution_header(log, "reindex_dinov2_eevee.py",
                         ref_dir=ref_dir,
                         clear_before_index=not args.no_clear)

    device = get_device()
    model  = load_dinov2(device)
    transform = get_transform()
    log.info(f"DINOv2 cargado en {device}")

    # ── Cargar Projection Head multimodal ────────────────────────────────────
    proj_head = None
    # Models live in LegoVision root, fall back to project dir
    head_path = os.path.join(legovic_root, "models", "dino_multimodal_head.pt")
    if not os.path.exists(head_path):
        head_path = os.path.join(legovic_root, "models", "dino_metric_head.pt")
    if not os.path.exists(head_path):
        head_path = os.path.join(project_root, "models", "dino_multimodal_head.pt")
    if not os.path.exists(head_path):
        head_path = os.path.join(project_root, "models", "dino_metric_head.pt")

    if os.path.exists(head_path):
        try:
            ckpt = torch.load(head_path, map_location=device)
            # Detectar dimensión de entrada leyendo el primer peso del checkpoint
            first_key = next(iter(ckpt.keys()))
            first_w = ckpt[first_key]
            in_dim = first_w.shape[1]   # shape: [out_features, in_features]
            proj_head = LegoProjectionHead(input_dim=in_dim).to(device)
            proj_head.load_state_dict(ckpt)
            proj_head.eval()
            log.info(f"Projection Head cargado: in_dim={in_dim}")
        except Exception as e:
            log.warning(f"Projection Head fallido: {e}")
            proj_head = None
    else:
        log.warning("No se encontró ningún Projection Head.")

    # ── Limpiar embeddings previos ───────────────────────────────────────────
    if not args.no_clear:
        log.info("Limpiando embeddings existentes...")
        supabase_client.clear_embeddings()
        log.info("BD limpia.")

    # ── Indexar imágenes por cámara ──────────────────────────────────────────
    # OPT-1: ThreadPool(4) para IO paralelo con 4 P-cores del M4 Pro
    IO_WORKERS = 4          # sweet spot medido: 4w > 8w en M4 Pro NVMe
    DB_BATCH   = 128        # tamaño de batch para Supabase

    def _load_image(img_path):
        """Carga y preprocesa una imagen en un worker thread (IO-bound)."""
        m = FILENAME_RE.match(os.path.basename(img_path))
        if not m:
            return None
        img = Image.open(img_path).convert("RGB")
        img_proc = preprocess_render(img)
        return img_path, m, img_proc

    batch  = []
    total  = 0
    t_start = time.perf_counter()

    for cam_name, cam_id in CAMERAS.items():
        cam_dir = os.path.join(ref_dir, cam_name)
        if not os.path.exists(cam_dir):
            log.warning(f"Directorio no encontrado: {cam_dir}")
            continue

        images = sorted(glob.glob(os.path.join(cam_dir, "ref_*.png")))
        images = filter_centermost_instances(images)
        log.info(f"Cámara {cam_name}: {len(images)} imágenes | IO paralelo {IO_WORKERS} workers...")

        # Prefetch: enviar todas las cargas al pool y procesar en orden
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            futures = {pool.submit(_load_image, p): p for p in images}
            # Procesar futuros en orden de finalización (más rápido que iterar)
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    log.warning(f"Nombre no reconocido: {os.path.basename(futures[fut])}")
                    continue

                img_path, m, img_proc = result
                part_ref     = m.group(1)
                color_hex    = m.group(2).upper()
                pose_idx     = int(m.group(3))
                rotation_deg = int(m.group(4))
                color_code   = COLOR_HEX_TO_CODE.get(color_hex, "0")
                stable_face  = pose_idx * 10 + cam_id

                try:
                    # Inferencia DINOv2 en MPS (main thread, GPU)
                    raw_emb = extract_embedding(img_proc, model, transform, device)

                    emb_projected = None
                    if proj_head is not None:
                        with torch.no_grad():
                            t_emb = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0).to(device)
                            if proj_head.net[0].in_features == 386:
                                dummy_size = torch.zeros(1, 2, device=device)
                                t_input = torch.cat([t_emb, dummy_size], dim=1)
                            else:
                                t_input = t_emb
                            t_proj = proj_head(t_input)
                            emb_projected = t_proj[0].cpu().numpy().tolist()

                    batch.append({
                        "part_ref":            part_ref,
                        "stable_face":         stable_face,
                        "rotation_angle":      rotation_deg,
                        "embedding":           raw_emb.tolist(),
                        "color_code":          color_code,
                        "color_hex":           color_hex,
                        "pose_index":          stable_face,
                        "embedding_projected": emb_projected,
                    })
                    total += 1

                    if len(batch) >= DB_BATCH:
                        supabase_client.save_piece_embeddings_batch(batch)
                        batch = []
                        elapsed = time.perf_counter() - t_start
                        rate    = total / elapsed
                        log.info(f"Indexados {total} embeddings  [{rate:.1f} emb/s]")

                except Exception as e:
                    log.error(f"Error embedding {os.path.basename(img_path)}: {e}")

    if batch:
        supabase_client.save_piece_embeddings_batch(batch)

    elapsed = time.perf_counter() - t_start
    rate    = total / elapsed if elapsed > 0 else 0
    log.info(f"DONE: {total} embeddings indexados desde {ref_dir}")
    log_execution_footer(log, "reindex_dinov2_eevee.py",
                         duration_s=elapsed,
                         total_embeddings=total,
                         throughput_emb_s=f"{rate:.1f}")


if __name__ == "__main__":
    main()
