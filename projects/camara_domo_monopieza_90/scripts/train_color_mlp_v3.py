# -*- coding: utf-8 -*-
"""train_color_mlp_v3.py — Entrena ColorMLPV3 con Hard Negative Mining.

Arquitectura:  12D → 256 → 128 → 64 → N_clases  (GELU + BN + Dropout)
               + cabeza embedding 64→32 (triplet loss auxiliar)

Dataset:
  - Base: features extraídos de la paleta calibrada (rgb_cenital → Lab+HSV stats)
  - Augmentación: jitter adaptado por tipo de material
  - Hard Negatives: pares de colores confundidos en Fase 4 con peso 3×

Training:
  - CrossEntropyLoss con label_smoothing=0.05
  - Triplet Loss auxiliar (hard negatives) con margen=5.0
  - 200 épocas, CosineAnnealing LR
  - MPS / CUDA / CPU

Output: models/color_mlp_v3.pt + color_mlp_v3_metadata.json

Uso:
    cd projects/camara_domo_monopieza_90
    python scripts/train_color_mlp_v3.py

M3 del plan de mejora — 2026-06-07
"""
from __future__ import annotations
import json, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from color_classifier_v2 import ColorMLPV3, _rgb_to_lab, _palette_material, FALSE_ATTRACTORS

_MODELS_DIR = os.path.join(_ROOT, "models")

# ── Pares de Hard Negatives (de análisis Fase 4) ──────────────────────────────
# Cada par (GT, Pred_error) recibe peso 3× en el training
HARD_NEGATIVE_PAIRS = [
    # Cromáticos confundidos
    ("Blue",                "Dark Blue"),
    ("Violet",              "Dark Blue"),
    ("Royal Blue(Old Blue-Violet)", "Dark Blue"),
    ("Medium Blue",         "Dark Blue"),
    ("Bright Light Blue",   "Sand Blue"),
    ("Neon Orange",         "Dark Blue"),     # especular contaminado
    ("Glitter Trans-Purple","Dark Blue"),
    ("Chrome Blue",         "Dark Purple"),
    # Metálicos/Pearl confundidos
    ("Flat Silver",         "Pearl Dark Gray"),
    ("Pearl Dark Gray",     "Black"),
    ("Chrome Antique Brass","Chrome Black"),
    ("Chrome Silver",       "Bionicle Silver"),
    ("Speckle Black-Silver","Speckle Black-Copper"),
    ("Speckle Black-Gold",  "Speckle Black-Copper"),
    # Trans → Sólido confundidos
    ("Trans-Red",           "Dark Red"),
    ("Trans-Light Green",   "Green"),
    ("Trans-Neon Yellow",   "Flat Dark Gold"),
    ("Satin Trans-Brown",   "Chrome Black"),
    # Cromáticos similares
    ("Warm Pink",           "Bionicle Copper"),
    ("Orange",              "Reddish Brown"),
    ("Reddish Brown",       "Dark Brown"),
    ("Sand Purple",         "Pearl Brown"),
    ("Light Turquoise",     "Pearl Blue"),
    ("Trans-Yellow",        "Dark Nougat"),
    ("Dark Green",          "Pearl Black"),   # muy importante: 8/8 fallos
    ("Dark Turquoise",      "Pearl Black"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _rgb_to_hsv(rgb):
    r,g,b = rgb[0]/255., rgb[1]/255., rgb[2]/255.
    mx=max(r,g,b); mn=min(r,g,b); df=mx-mn
    if mx==mn: h=0.
    elif mx==r: h=(60.*((g-b)/df)+360.)%360.
    elif mx==g: h=(60.*((b-r)/df)+120.)%360.
    else:       h=(60.*((r-g)/df)+240.)%360.
    s=0. if mx==0. else (df/mx)*255.
    v=mx*255.
    return np.array([h/2., s, v], dtype=float)  # H in [0,180] like OpenCV

def _make_feat_from_rgb(rgb_c, noise_factor=0.0, rng=None):
    """Genera feature vector 12D desde un rgb_cenital."""
    if rng is None: rng = np.random.default_rng()
    rgb = np.array(rgb_c, dtype=float)
    if noise_factor > 0:
        rgb = np.clip(rgb + rng.normal(0, noise_factor*255, 3), 0, 255)
    lab = _rgb_to_lab(rgb)
    hsv = _rgb_to_hsv(rgb)
    # Simular std (desviación por iluminación EEVEE)
    std_lab = np.abs(rng.normal([5., 3., 3.], [2., 1., 1.]))
    std_hsv = np.abs(rng.normal([8., 20., 15.], [3., 8., 6.]))
    return np.array([
        lab[0], std_lab[0], lab[1], std_lab[1], lab[2], std_lab[2],
        hsv[0], std_hsv[0], hsv[1], std_hsv[1], hsv[2], std_hsv[2],
    ], dtype=np.float32)

# ── Triplet Loss con hard negatives ──────────────────────────────────────────
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[train_color_mlp_v3] Device: {device}")

    # Load palette
    palette = json.load(open(os.path.join(_ROOT,"data","color_calibration_palette.json"),encoding="utf-8"))
    color_names = [e["color_name"] for e in palette]
    classes = sorted(set(color_names))
    cls_to_idx = {n:i for i,n in enumerate(classes)}
    n_classes = len(classes)
    print(f"Classes (including attractors): {n_classes}")

    # Build hard negatives set dynamically from failures log
    failures_path = os.path.join(_ROOT, "data", "color_failures.json")
    confused_pairs = []
    loaded_failures = []
    if os.path.exists(failures_path):
        with open(failures_path, encoding="utf-8") as f:
            loaded_failures = json.load(f)
        for fail in loaded_failures:
            gt = fail["color_name_gt"]
            pred = fail["pred_name_v2"]
            if gt and pred:
                confused_pairs.append((gt, pred))
    
    # Extend hard negatives list
    all_hard_pairs = list(HARD_NEGATIVE_PAIRS) + confused_pairs
    hard_neg_set = set()
    for gt,pred in all_hard_pairs:
        hard_neg_set.add(gt.lower())
    print(f"Dynamic hard negatives pairs count: {len(all_hard_pairs)} (added {len(confused_pairs)} from evaluation failures)")

    # Generate training dataset
    N_PER_CLASS = 800
    rng = np.random.default_rng(2026)

    X_list, y_list, weights = [], [], []

    # 1a. Load real failures and oversample them + augment
    if loaded_failures:
        print(f"Loading {len(loaded_failures)} real failures from evaluation for fine-tuning...")
        for fail in loaded_failures:
            if len(fail["features"]) != 19: continue
            gt_name = fail["color_name_gt"]
            if gt_name not in cls_to_idx: continue
            idx = cls_to_idx[gt_name]
            feat = np.array(fail["features"], dtype=np.float32)
            
            # Anchor original failure with high weight
            X_list.append(feat)
            y_list.append(idx)
            weights.append(25.0) # Very strong correction weight
            
            # Generate 60 local jittered copies around this failure
            for _ in range(60):
                # Standard deviations for 19D features (reduced baseline variance)
                noise = rng.normal(
                    [0.0]*19,
                    [0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.02, 0.5, 0.5, 0.5, 0.5]
                )
                feat_noise = feat + noise
                
                # Simulate shadows (asymmetric L-channel shift)
                if rng.uniform() < 0.4:
                    shadow_depth = rng.uniform(2.0, 15.0)
                    feat_noise[0] -= shadow_depth          # mean L drops
                    feat_noise[18] -= shadow_depth * 1.2   # L_p5 drops more
                    feat_noise[12] -= shadow_depth * 0.2   # L_p95 drops very little
                    
                # Simulate belt reflection (affects dark pieces)
                if feat[0] < 35.0 and rng.uniform() < 0.3:
                    feat_noise[2] += rng.uniform(-2.0, 2.0)
                    feat_noise[4] += rng.uniform(-5.0, 0.0)
                
                # Enforce physical constraints: L_p95 >= mean_L >= L_p5
                feat_noise[12] = max(feat_noise[12], feat_noise[0])
                feat_noise[0] = max(feat_noise[0], feat_noise[18])
                
                # Clamp boundaries
                feat_noise[6] = np.clip(feat_noise[6], 0, 180)
                feat_noise[8] = np.clip(feat_noise[8], 0, 255)
                feat_noise[10] = np.clip(feat_noise[10], 0, 255)
                feat_noise[14] = np.clip(feat_noise[14], 0.0, 1.0)
                # Keep std and textures positive
                for k in [1, 3, 5, 7, 9, 11, 13]:
                    feat_noise[k] = max(0.1, feat_noise[k])
                
                X_list.append(feat_noise)
                y_list.append(idx)
                weights.append(8.0) # High-priority augmentation

    # 1b. Load real features from cache if available
    cache_path = os.path.join(_ROOT, "data", "mlp_features_cache.npz")
    if os.path.exists(cache_path):
        print(f"Loading real features from cache: {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True)
        X_real = cache_data["X_cen"]
        y_real = cache_data["y_cen"]
        
        real_counts = {}
        for feat, name in zip(X_real, y_real):
            if name not in cls_to_idx:
                continue
            idx = cls_to_idx[name]
            real_counts[name] = real_counts.get(name, 0) + 1
            
            # Add the original real feature with high weight
            X_list.append(feat)
            y_list.append(idx)
            weights.append(10.0)
            
            # Augment real feature to prevent overfitting
            for _ in range(150):
                # Standard deviations for 19D features (reduced baseline variance)
                noise = rng.normal(
                    [0.0]*19,
                    [0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.02, 0.5, 0.5, 0.5, 0.5]
                )
                feat_noise = feat + noise
                
                # Simulate shadows (asymmetric L-channel shift)
                if rng.uniform() < 0.4:
                    shadow_depth = rng.uniform(2.0, 15.0)
                    feat_noise[0] -= shadow_depth          # mean L drops
                    feat_noise[18] -= shadow_depth * 1.2   # L_p5 drops more
                    feat_noise[12] -= shadow_depth * 0.2   # L_p95 drops very little
                    
                # Simulate belt reflection (affects dark pieces)
                if feat[0] < 35.0 and rng.uniform() < 0.3:
                    feat_noise[2] += rng.uniform(-2.0, 2.0)
                    feat_noise[4] += rng.uniform(-5.0, 0.0)
                
                # Enforce physical constraints: L_p95 >= mean_L >= L_p5
                feat_noise[12] = max(feat_noise[12], feat_noise[0])
                feat_noise[0] = max(feat_noise[0], feat_noise[18])
                
                # Clamp boundaries
                feat_noise[6] = np.clip(feat_noise[6], 0, 180)
                feat_noise[8] = np.clip(feat_noise[8], 0, 255)
                feat_noise[10] = np.clip(feat_noise[10], 0, 255)
                feat_noise[14] = np.clip(feat_noise[14], 0.0, 1.0)
                # Keep std and textures positive
                for k in [1, 3, 5, 7, 9, 11, 13]:
                    feat_noise[k] = max(0.1, feat_noise[k])
                
                X_list.append(feat_noise)
                y_list.append(idx)
                weights.append(5.0)
        print(f"Loaded {len(X_real)} real samples across {len(real_counts)} classes.")

    # 2. Add synthetic baseline palette features only for classes NOT present in the cache
    real_classes = set(y_real) if os.path.exists(cache_path) else set()
    print(f"Bypassing synthetic features for {len(real_classes)} classes that have real cache features.")
    for entry in palette:
        name = entry["color_name"]
        if name not in cls_to_idx: continue
        if name in real_classes: continue  # Skip classes that have real SAM-masked features
        
        idx = cls_to_idx[name]
        rgb_c = entry.get("rgb_cenital",[128,128,128])
        
        # Generate aligned synthetic features for the missing class
        for _ in range(N_PER_CLASS * 2):
            lab_c = _rgb_to_lab(rgb_c)
            lab_c[0] = max(5.0, lab_c[0] - 20.0)  # Systematic L lighting shift
            hsv_c = _rgb_to_hsv(rgb_c)
            hsv_c[2] = max(5.0, hsv_c[2] - 50.0)  # Systematic V lighting shift
            
            std_lab = np.abs(rng.normal([4.33, 1.92, 2.0], [1.0, 0.5, 0.5]))
            std_hsv = np.abs(rng.normal([6.37, 12.10, 12.71], [1.5, 3.0, 3.0]))
            
            # Simulate 19D advanced features
            L_p95 = lab_c[0] + rng.uniform(2.0, 5.0)
            L_p5 = max(0.0, lab_c[0] - rng.uniform(2.0, 5.0))
            
            # Simulate shadows (asymmetric L-channel shift on synthetic)
            if rng.uniform() < 0.4:
                shadow_depth = rng.uniform(2.0, 15.0)
                lab_c[0] -= shadow_depth          # mean L drops
                L_p5 = max(0.0, L_p5 - shadow_depth * 1.2)
                L_p95 = max(0.0, L_p95 - shadow_depth * 0.2)
                
            # Simulate belt reflection (affects dark pieces)
            if lab_c[0] < 35.0 and rng.uniform() < 0.3:
                lab_c[1] += rng.uniform(-2.0, 2.0)
                lab_c[2] += rng.uniform(-5.0, 0.0)
            
            # Enforce physical constraints: L_p95 >= mean_L >= L_p5
            L_p95 = max(L_p95, lab_c[0])
            lab_c[0] = max(lab_c[0], L_p5)
            
            mat_type = entry.get("material", "solid").lower()
            if "trans" in mat_type:
                tex_contrast = np.abs(rng.normal(3.0, 1.0))
                tex_homogeneity = rng.normal(0.85, 0.05)
            elif "metallic" in mat_type or "chrome" in mat_type or "pearl" in mat_type:
                tex_contrast = np.abs(rng.normal(25.0, 5.0))
                tex_homogeneity = rng.normal(0.45, 0.08)
            elif "glitter" in mat_type:
                tex_contrast = np.abs(rng.normal(35.0, 6.0))
                tex_homogeneity = rng.normal(0.35, 0.1)
            else: # Solid matte
                tex_contrast = np.abs(rng.normal(5.0, 1.5))
                tex_homogeneity = rng.normal(0.75, 0.05)
                
            # Simulate 18D background relative calibration (belt Lab is [21.0, 9.0, -37.0])
            diff_L = lab_c[0] - 21.0
            diff_a = lab_c[1] - 9.0
            diff_b = lab_c[2] - (-37.0)
            
            feat = np.array([
                lab_c[0], std_lab[0], lab_c[1], std_lab[1], lab_c[2], std_lab[2],
                hsv_c[0], std_hsv[0], hsv_c[1], std_hsv[1], hsv_c[2], std_hsv[2],
                L_p95, tex_contrast, tex_homogeneity,
                diff_L, diff_a, diff_b, L_p5
            ], dtype=np.float32)
            
            # Add small jitter to means
            feat[0] += rng.normal(0, 3.0)
            feat[2] += rng.normal(0, 1.0)
            feat[4] += rng.normal(0, 1.0)
            feat[6] = np.clip(feat[6] + rng.normal(0, 5.0), 0, 180)
            feat[8] = np.clip(feat[8] + rng.normal(0, 10.0), 0, 255)
            feat[10] = np.clip(feat[10] + rng.normal(0, 10.0), 0, 255)
            
            X_list.append(feat)
            y_list.append(idx)
            weights.append(1.0)

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.int64)
    w_all = np.array(weights, dtype=np.float32)

    # Normalize
    mean = X_all.mean(axis=0); std = X_all.std(axis=0)+1e-8
    X_norm = (X_all - mean)/std

    # Shuffle
    perm = rng.permutation(len(X_norm))
    X_norm = X_norm[perm]; y_all = y_all[perm]; w_all = w_all[perm]

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_all, dtype=torch.long)
    w_t = torch.tensor(w_all, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t, w_t)
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    print(f"Total samples: {len(X_t)} ({N_PER_CLASS}/class base + hard neg extra)")

    # Model + optimizer
    model = ColorMLPV3(input_dim=19, num_classes=n_classes, embed_dim=32).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    triplet_fn = TripletMarginLoss(margin=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Precompute class centroids for triplet mining (updated each epoch)
    def get_class_feats():
        class_feats = defaultdict(list)
        for feat, lbl in zip(X_norm, y_all):
            class_feats[int(lbl)].append(feat)
        return {k: np.mean(v, axis=0) for k,v in class_feats.items()}

    print("Entrenando ColorMLPV3 con Hard Negative Mining...")
    t0 = time.time()
    model.train()

    for epoch in range(200):
        total_loss = 0.; correct = 0; n_batches = 0
        for bx, by, bw in loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            optimizer.zero_grad()
            logits, emb = model(bx, return_embedding=True)
            # Weighted CrossEntropy
            ce = (ce_loss_fn(logits, by) * bw).mean()
            # Triplet loss on batch: vectorized batch-hard triplet mining
            trip_loss = torch.tensor(0., device=device)
            if epoch >= 20:  # start triplet after warmup
                dist_mat = torch.cdist(emb, emb)
                label_equal = (by.unsqueeze(0) == by.unsqueeze(1))
                mask_pos = label_equal.float() - torch.eye(len(by), device=device)
                mask_neg = (~label_equal).float()
                has_pos = (mask_pos.sum(dim=1) > 0)
                if has_pos.sum() > 0:
                    dist_ap = (dist_mat * mask_pos).max(dim=1)[0]
                    dist_an = (dist_mat * mask_neg + (1.0 - mask_neg) * 1e6).min(dim=1)[0]
                    dist_ap = dist_ap[has_pos]
                    dist_an = dist_an[has_pos]
                    trip_loss = F.relu(dist_ap - dist_an + 5.0).mean() * 0.1

            loss = ce + trip_loss
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == by).sum().item()
            n_batches += 1
        scheduler.step()
        if (epoch+1) % 20 == 0:
            acc = 100.*correct/len(X_t)
            elapsed = time.time()-t0
            print(f"  Epoch {epoch+1}/200 | Loss={total_loss/n_batches:.4f} | Acc={acc:.1f}% | {elapsed:.0f}s", flush=True)

    # Final eval
    model.eval()
    with torch.no_grad():
        logits = model(X_t.to(device))
        preds = logits.argmax(1).cpu().numpy()
    final_acc = (preds == y_all).mean()*100
    print(f"Accuracy final: {final_acc:.2f}%")

    # Save
    os.makedirs(_MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(_MODELS_DIR,"color_mlp_v3.pt"))
    meta = {"mean":mean.tolist(),"std":std.tolist(),"classes":classes,
            "n_classes":n_classes,"input_dim":19,"embed_dim":32,
            "train_acc":round(float(final_acc),2),"n_samples":int(len(X_t)),
            "hard_negative_pairs":HARD_NEGATIVE_PAIRS,"date":"2026-06-07",
            "architecture":"19->256->128->64->N (GELU+BN+Dropout)"}
    with open(os.path.join(_MODELS_DIR,"color_mlp_v3_metadata.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Modelo guardado: color_mlp_v3.pt | Metadata: color_mlp_v3_metadata.json")
    print("OK M3 training done.")

if __name__ == "__main__":
    main()
