# -*- coding: utf-8 -*-
# scripts/train_dino_metric_head.py
# PyTorch script to train a projection head (MLP) for DINOv2 embeddings using Triplet Margin Loss.

import os
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.append(project_root)

from database import supabase_client

# Define the Projection Head MLP
class LegoProjectionHead(nn.Module):
    def __init__(self, input_dim=384, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        features = self.net(x)
        # L2 normalize embeddings so that dot product equals cosine similarity
        return F.normalize(features, p=2, dim=1)

# PyTorch dataset to generate triplets
class TripletEmbeddingDataset(Dataset):
    def __init__(self, class_to_embeddings):
        self.class_to_embeddings = class_to_embeddings
        self.classes = list(class_to_embeddings.keys())
        self.triplets = []
        
        print(f"[Dataset] building triplets for {len(self.classes)} classes...")
        for c in self.classes:
            embs = class_to_embeddings[c]
            n = len(embs)
            if n < 2:
                continue
            
            # Create triplet combinations
            # We shuffle or limit to prevent combinatorial explosion if n is very large
            for i in range(n):
                # Pick up to 10 random positives for each anchor to keep dataset size balanced
                pos_indices = list(range(n))
                pos_indices.remove(i)
                random.shuffle(pos_indices)
                for j in pos_indices[:10]:
                    # Pick 2 different negative classes
                    neg_classes = [oc for oc in self.classes if oc != c]
                    if neg_classes:
                        for _ in range(2): # 2 triplets per positive pair
                            neg_c = random.choice(neg_classes)
                            neg_emb = random.choice(class_to_embeddings[neg_c])
                            self.triplets.append((embs[i], embs[j], neg_emb))
                            
        print(f"[Dataset] Generated {len(self.triplets)} triplets total.")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anc, pos, neg = self.triplets[idx]
        return (
            torch.tensor(anc, dtype=torch.float32),
            torch.tensor(pos, dtype=torch.float32),
            torch.tensor(neg, dtype=torch.float32)
        )

def load_data_from_db():
    print("[DB] Loading embeddings from piece_embeddings table...")
    with supabase_client.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT part_ref, embedding 
                FROM piece_embeddings
            """)
            rows = cur.fetchall()
            
    if not rows:
        print("[DB ERROR] No embeddings found in database. Please run indexing first.")
        sys.exit(1)
        
    class_to_embeddings = {}
    for r in rows:
        ref = r['part_ref']
        emb = r['embedding']
        if ref not in class_to_embeddings:
            class_to_embeddings[ref] = []
        class_to_embeddings[ref].append(emb)
        
    print(f"[DB] Loaded {len(rows)} embeddings across {len(class_to_embeddings)} classes.")
    return class_to_embeddings

def evaluate_metrics(model, device, class_to_embeddings):
    """Calcula similitud coseno promedio para pares positivos y negativos."""
    model.eval()
    pos_sims = []
    neg_sims = []
    
    classes = list(class_to_embeddings.keys())
    
    with torch.no_grad():
        # 1. Similitudes Positivas (misma clase)
        for c in classes:
            embs = class_to_embeddings[c]
            if len(embs) < 2:
                continue
            t_embs = torch.tensor(embs, dtype=torch.float32).to(device)
            # Proyectar
            proj_embs = model(t_embs)
            # Calcular similitud coseno de todos los pares
            sim_matrix = torch.mm(proj_embs, proj_embs.t())
            triu_idx = torch.triu_indices(len(embs), len(embs), offset=1)
            pos_sims.extend(sim_matrix[triu_idx[0], triu_idx[1]].cpu().tolist())
            
        # 2. Similitudes Negativas (diferente clase)
        for idx_a, c_a in enumerate(classes):
            for c_b in classes[idx_a + 1:]:
                embs_a = torch.tensor(class_to_embeddings[c_a], dtype=torch.float32).to(device)
                embs_b = torch.tensor(class_to_embeddings[c_b], dtype=torch.float32).to(device)
                
                proj_a = model(embs_a)
                proj_b = model(embs_b)
                
                sim_matrix = torch.mm(proj_a, proj_b.t())
                neg_sims.extend(sim_matrix.flatten().cpu().tolist())
                
    avg_pos = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg = np.mean(neg_sims) if neg_sims else 0.0
    return avg_pos, avg_neg

def evaluate_baseline(class_to_embeddings):
    """Calcula la similitud coseno base antes de entrenar la proyección."""
    pos_sims = []
    neg_sims = []
    classes = list(class_to_embeddings.keys())
    
    # 1. Positivas
    for c in classes:
        embs = np.array(class_to_embeddings[c])
        if len(embs) < 2:
            continue
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-8)
        sim_matrix = np.dot(embs_norm, embs_norm.T)
        triu_idx = np.triu_indices(len(embs), k=1)
        pos_sims.extend(sim_matrix[triu_idx].tolist())
        
    # 2. Negativas
    for idx_a, c_a in enumerate(classes):
        for c_b in classes[idx_a + 1:]:
            embs_a = np.array(class_to_embeddings[c_a])
            embs_b = np.array(class_to_embeddings[c_b])
            
            norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True)
            embs_a_n = embs_a / (norms_a + 1e-8)
            
            norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True)
            embs_b_n = embs_b / (norms_b + 1e-8)
            
            sim_matrix = np.dot(embs_a_n, embs_b_n.T)
            neg_sims.extend(sim_matrix.flatten().tolist())
            
    avg_pos = np.mean(pos_sims) if pos_sims else 0.0
    avg_neg = np.mean(neg_sims) if neg_sims else 0.0
    return avg_pos, avg_neg

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")
    
    # 1. Cargar datos
    class_to_embeddings = load_data_from_db()
    
    # 2. Evaluar Baseline antes del entrenamiento
    base_pos, base_neg = evaluate_baseline(class_to_embeddings)
    print("\n" + "="*50)
    print("BASELINE METRICS (Before Metric Learning):")
    print(f"  Average Positive Similarity (Same Class): {base_pos:.4f}")
    print(f"  Average Negative Similarity (Diff Class): {base_neg:.4f}")
    print(f"  Margin (Pos - Neg): {base_pos - base_neg:.4f}")
    print("="*50 + "\n")
    
    if len(class_to_embeddings) < 2:
        print("[Training ERROR] Need at least 2 classes in the DB to train Triplet Loss. Skipping training.")
        return
        
    # 3. Crear Dataloader
    dataset = TripletEmbeddingDataset(class_to_embeddings)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 4. Inicializar Modelo, Loss y Optimizador
    model = LegoProjectionHead().to(device)
    # Triplet Margin Loss con margen de 1.0
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Bucle de Entrenamiento
    epochs = 60
    print(f"[Training] Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (anc, pos, neg) in enumerate(dataloader):
            anc = anc.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            # Forward
            proj_anc = model(anc)
            proj_pos = model(pos)
            proj_neg = model(neg)
            
            loss = criterion(proj_anc, proj_pos, proj_neg)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            avg_pos, avg_neg = evaluate_metrics(model, device, class_to_embeddings)
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | Sim Pos: {avg_pos:.4f} | Sim Neg: {avg_neg:.4f} | Margin: {avg_pos-avg_neg:.4f}")
            
    # 6. Guardar Modelo
    models_dir = "/Users/I764690/Code_personal/LegoVision/models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "dino_metric_head.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n[Training DONE] Projection head saved to: {model_path}")
    
    # 7. Resumen final
    final_pos, final_neg = evaluate_metrics(model, device, class_to_embeddings)
    print("\n" + "="*50)
    print("FINAL METRICS (After Metric Learning):")
    print(f"  Average Positive Similarity (Same Class): {final_pos:.4f} (was {base_pos:.4f})")
    print(f"  Average Negative Similarity (Diff Class): {final_neg:.4f} (was {base_neg:.4f})")
    print(f"  Final Margin: {final_pos - final_neg:.4f} (was {base_pos - base_neg:.4f})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
