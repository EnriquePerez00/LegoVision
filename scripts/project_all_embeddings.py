# -*- coding: utf-8 -*-
# scripts/project_all_embeddings.py
# Batch projects all 384-d embeddings in piece_embeddings to 128-d using the trained projection head.

import os
import sys
import torch
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.append(project_root)

from core.db import supabase_client
from inference.knn_classifier import LegoProjectionHead

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Projector] Using device: {device}")

    # 1. Load trained projection head
    model_path = os.path.join(project_root, "models", "dino_metric_head.pt")
    if not os.path.exists(model_path):
        print(f"[Projector ERROR] Metric Head model file not found at: {model_path}")
        print("Please wait until train_dino_metric_head.py finishes running.")
        sys.exit(1)

    model = LegoProjectionHead().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Projector] Loaded projection head from: {model_path}")

    # 2. Query all embeddings
    print("[Projector] Fetching all rows from piece_embeddings...")
    with supabase_client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT part_ref, stable_face, rotation_angle, color_hex, pose_index, embedding
                FROM piece_embeddings
            """)
            rows = cur.fetchall()

    if not rows:
        print("[Projector ERROR] No embeddings found in database.")
        sys.exit(1)

    print(f"[Projector] Loaded {len(rows)} embeddings. Projecting...")

    # 3. Project all embeddings in batch
    print("[Projector] Projecting embeddings in batch...")
    all_embs = [r['embedding'] for r in rows]
    t_embeddings = torch.tensor(all_embs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        projected_list = []
        chunk_size = 5000
        for i in range(0, len(t_embeddings), chunk_size):
            chunk = t_embeddings[i:i+chunk_size]
            proj_chunk = model(chunk)
            projected_list.append(proj_chunk.cpu().numpy())
        projected = np.vstack(projected_list)

    print("[Projector] Projection completed. Preparing update data...")
    update_data = []
    for idx, r in enumerate(rows):
        part_ref = r['part_ref']
        face = r['stable_face']
        rot = r['rotation_angle']
        color_hex = r['color_hex']
        pose_idx = r.get('pose_index', 0) or 0
        proj_emb = projected[idx].tolist()
        update_data.append((proj_emb, part_ref, face, rot, color_hex, pose_idx))

    # 4. Batch update using execute_batch
    from psycopg2.extras import execute_batch
    print(f"[Projector] Updating database in batches...")
    with supabase_client.get_connection() as conn:
        with conn.cursor() as cur:
            batch_size = 1000
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i:i+batch_size]
                execute_batch(cur, """
                    UPDATE piece_embeddings
                    SET embedding_projected = %s
                    WHERE part_ref = %s AND stable_face = %s AND rotation_angle = %s
                      AND color_hex IS NOT DISTINCT FROM %s AND pose_index IS NOT DISTINCT FROM %s
                """, batch)
                conn.commit()
                print(f"[Projector] Progress: {min(i+batch_size, len(update_data))}/{len(update_data)} updated.")

    print(f"[Projector DONE] Successfully projected and updated {len(update_data)} reference embeddings in DB.")

if __name__ == "__main__":
    main()
