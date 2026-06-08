# -*- coding: utf-8 -*-
# scripts/project_all_embeddings.py
# Batch projects all 384-d embeddings in piece_embeddings to 128-d using the trained projection head.

import os
import sys
import torch
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.append(project_root)

from database import supabase_client
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

    # 3. Project and update database
    updated_count = 0
    with supabase_client.get_connection() as conn:
        with conn.cursor() as cur:
            for idx, r in enumerate(rows):
                part_ref = r['part_ref']
                face = r['stable_face']
                rot = r['rotation_angle']
                color_hex = r['color_hex']
                pose_idx = r.get('pose_index', 0) or 0
                emb = r['embedding']

                # Project
                with torch.no_grad():
                    t_emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    t_proj = model(t_emb)
                    proj_emb = t_proj[0].cpu().numpy().tolist()

                # Update row (PK completa: part_ref, stable_face, rotation_angle, color_hex, pose_index)
                cur.execute("""
                    UPDATE piece_embeddings
                    SET embedding_projected = %s
                    WHERE part_ref = %s AND stable_face = %s AND rotation_angle = %s
                      AND color_hex = %s AND pose_index = %s
                """, (proj_emb, part_ref, face, rot, color_hex, pose_idx))
                
                updated_count += 1
                if updated_count % 100 == 0 or updated_count == len(rows):
                    print(f"[Projector] Progress: {updated_count}/{len(rows)} rows updated.")
                    conn.commit()

            conn.commit()

    print(f"[Projector DONE] Successfully projected and updated {updated_count} reference embeddings in DB.")

if __name__ == "__main__":
    main()
