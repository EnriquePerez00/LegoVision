# -*- coding: utf-8 -*-
"""scripts/reindex_dinov2_eevee.py
=====================================
Re-indexa los embeddings DINOv2 usando las nuevas imágenes de referencia
renderizadas con EEVEE (mismo dominio visual que las imágenes de test).

Limpia los embeddings actuales (iter7/Cycles) y los reemplaza con los
nuevos (EEVEE+belt+training-lighting).

Uso:
  .venv/bin/python scripts/reindex_dinov2_eevee.py \
      --ref_dir data/iter9_dinov2_ref
"""
import os, sys, re, glob, argparse
import torch
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client
from training.index_synthetic_renders import (
    COLOR_HEX_TO_CODE, get_device, load_dinov2,
    get_transform, preprocess_render, extract_embedding,
)
from inference.knn_classifier import LegoProjectionHead

# Regex: ref_PARTREF_COLORHEX_poseNN_rotNNN.png
FILENAME_RE = re.compile(
    r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})_pose(\d+)_rot(\d+)\.png",
    re.IGNORECASE,
)

CAMERAS = {
    "cenital":   1,
    "frontal":   2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str,
                        default=os.path.join(project_root, "data", "iter9_dinov2_ref"),
                        help="Directorio raíz con subcarpetas cenital/, lateral_l/, lateral_r/")
    parser.add_argument("--no_clear", action="store_true",
                        help="No limpiar embeddings existentes antes de indexar")
    args = parser.parse_args()

    ref_dir = args.ref_dir

    device = get_device()
    model  = load_dinov2(device)
    transform = get_transform()
    print(f"[Reindexer EEVEE] DINOv2 cargado en {device}")

    # ── Cargar Projection Head multimodal ────────────────────────────────────
    proj_head = None
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
            print(f"[Reindexer EEVEE] Projection Head cargado: in_dim={in_dim}")
        except Exception as e:
            print(f"[Reindexer EEVEE Warning] Projection Head fallido: {e}")
            proj_head = None
    else:
        print("[Reindexer EEVEE Warning] No se encontró ningún Projection Head.")

    # ── Limpiar embeddings previos ───────────────────────────────────────────
    if not args.no_clear:
        print("[Reindexer EEVEE] Limpiando embeddings existentes...")
        supabase_client.clear_embeddings()
        print("[Reindexer EEVEE] BD limpia.")

    # ── Indexar imágenes por cámara ──────────────────────────────────────────
    batch = []
    total = 0

    for cam_name, cam_id in CAMERAS.items():
        cam_dir = os.path.join(ref_dir, cam_name)
        if not os.path.exists(cam_dir):
            print(f"[Reindexer EEVEE Warning] Directorio no encontrado: {cam_dir}")
            continue

        images = sorted(glob.glob(os.path.join(cam_dir, "ref_*.png")))
        print(f"[Reindexer EEVEE] Cámara {cam_name}: {len(images)} imágenes...")

        for img_path in images:
            m = FILENAME_RE.match(os.path.basename(img_path))
            if not m:
                print(f"  [WARN] Nombre no reconocido: {os.path.basename(img_path)}")
                continue

            part_ref     = m.group(1)
            color_hex    = m.group(2).upper()
            pose_idx     = int(m.group(3))
            rotation_deg = int(m.group(4))
            color_code   = COLOR_HEX_TO_CODE.get(color_hex, "0")
            stable_face  = pose_idx * 10 + cam_id

            try:
                img = Image.open(img_path).convert("RGB")
                img_proc = preprocess_render(img)
                raw_emb  = extract_embedding(img_proc, model, transform, device)

                emb_projected = None
                if proj_head is not None:
                    with torch.no_grad():
                        t_emb  = torch.tensor(raw_emb, dtype=torch.float32).unsqueeze(0).to(device)
                        # Si el head espera 386-d (multimodal), concat dummy size
                        if proj_head.net[0].in_features == 386:
                            dummy_size = torch.zeros(1, 2, device=device)
                            t_input = torch.cat([t_emb, dummy_size], dim=1)
                        else:
                            t_input = t_emb
                        t_proj = proj_head(t_input)
                        emb_projected = t_proj[0].cpu().numpy().tolist()

                batch.append({
                    "part_ref":           part_ref,
                    "stable_face":        stable_face,
                    "rotation_angle":     rotation_deg,
                    "embedding":          raw_emb.tolist(),
                    "color_code":         color_code,
                    "color_hex":          color_hex,
                    "pose_index":         stable_face,
                    "embedding_projected": emb_projected,
                })
                total += 1

                if len(batch) >= 128:
                    supabase_client.save_piece_embeddings_batch(batch)
                    batch = []
                    print(f"  Indexados {total}...")

            except Exception as e:
                print(f"  [ERR] {os.path.basename(img_path)}: {e}")

    if batch:
        supabase_client.save_piece_embeddings_batch(batch)

    print(f"\n[Reindexer EEVEE DONE] {total} embeddings indexados desde {ref_dir}")


if __name__ == "__main__":
    main()
