# -*- coding: utf-8 -*-
# scripts/reindex_dinov2_iter9.py

import os
import sys
import re
import glob
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.db import supabase_client
from training.index_synthetic_renders import COLOR_HEX_TO_CODE, get_device, load_dinov2, get_transform, preprocess_render, extract_embedding
from inference.knn_classifier import LegoProjectionHead

def main():
    device = get_device()
    model = load_dinov2(device)
    transform = get_transform()
    
    # Load Projection Head if available
    projection_head = None
    model_path = os.path.join(project_root, "models", "dino_metric_head.pt")
    if os.path.exists(model_path):
        projection_head = LegoProjectionHead(input_dim=384).to(device)
        try:
            projection_head.load_state_dict(torch.load(model_path, map_location=device))
            projection_head.eval()
            print("[Reindexer] MLP Projection Head cargado correctamente (128-d).")
        except Exception as e:
            print(f"[Reindexer Warning] No se pudo cargar el MLP unimodal: {e}. Se ignorará la proyección.")
            projection_head = None

    print("[Reindexer] Limpiando todos los embeddings existentes...")
    supabase_client.clear_embeddings()
    
    dinov2_ref_dir = os.path.join(project_root, "data", "iter7", "dinov2_ref")
    cameras_map = {
        "cenital": 1,
        "lateral1": 2,
        "lateral2": 3
    }
    
    # Regex to parse filenames: ref_PART_COLORHEX_posePOSE_rotROT.png
    regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})_pose(\d+)_rot(\d+)\.png", re.IGNORECASE)
    
    batch_to_save = []
    total_indexed = 0
    
    for cam_name, cam_id in cameras_map.items():
        cam_dir = os.path.join(dinov2_ref_dir, cam_name)
        if not os.path.exists(cam_dir):
            print(f"[Reindexer Warning] Directorio no encontrado: {cam_dir}")
            continue
            
        print(f"[Reindexer] Indexando cámara: {cam_name} (ID: {cam_id})...")
        image_paths = sorted(glob.glob(os.path.join(cam_dir, "ref_*.png")))
        
        for idx, p in enumerate(image_paths):
            m = regex.match(os.path.basename(p))
            if not m:
                print(f"[Reindexer Warning] Filename no coincide con regex: {os.path.basename(p)}")
                continue
                
            part_ref = m.group(1)
            color_hex = m.group(2).upper()
            pose_idx = int(m.group(3))
            rotation_angle = int(m.group(4))
            color_code = COLOR_HEX_TO_CODE.get(color_hex, "0")
            
            stable_face = pose_idx * 10 + cam_id
            
            try:
                img = Image.open(p).convert("RGB")
                img_proc = preprocess_render(img)
                embedding = extract_embedding(img_proc, model, transform, device)
                
                # Unimodal projection (384 -> 128)
                embedding_projected = None
                if projection_head is not None:
                    with torch.no_grad():
                        t_emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
                        t_proj = projection_head(t_emb)
                        embedding_projected = t_proj[0].cpu().numpy().tolist()
                
                batch_to_save.append({
                    "part_ref": part_ref,
                    "stable_face": stable_face,
                    "rotation_angle": rotation_angle,
                    "embedding": embedding.tolist(),
                    "color_code": color_code,
                    "color_hex": color_hex,
                    "pose_index": stable_face,
                    "embedding_projected": embedding_projected
                })
                
                total_indexed += 1
                
                if len(batch_to_save) >= 128:
                    supabase_client.save_piece_embeddings_batch(batch_to_save)
                    batch_to_save = []
                    print(f"Indexados {total_indexed} embeddings...")
                    
            except Exception as e:
                print(f"[Reindexer ERROR] Falló {os.path.basename(p)}: {e}")
                
    if batch_to_save:
        supabase_client.save_piece_embeddings_batch(batch_to_save)
        
    print(f"\n[Reindexer DONE] Reindexación completada con éxito. Total: {total_indexed} embeddings.")

if __name__ == "__main__":
    main()
