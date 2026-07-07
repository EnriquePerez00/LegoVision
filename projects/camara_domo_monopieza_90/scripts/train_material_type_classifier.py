# -*- coding: utf-8 -*-
"""train_material_type_classifier.py
================================================================================
Entrena el clasificador ligero de tipo de material para ColorClassifierV2 Stage 2.

Arquitectura: MLP 6D → 32 → 16 → 5 clases
  Clases: [SOLID, TRANSPARENT, METALLIC, PEARL, SPECIAL]

Features de entrada (6D):
  [sigma_L, sigma_H, sigma_V, sigma_S, ratio_chromatic, L_gradient]

Dataset sintético generado a partir de la paleta calibrada + características
estadísticas esperadas por tipo de material según análisis de EEVEE renders.

Tiempo estimado: ~5 min en M4 MPS
Output: models/material_type_classifier.pt + material_type_classifier_metadata.json

Uso:
    cd projects/camara_domo_monopieza_90
    python scripts/train_material_type_classifier.py

Fase 2 del plan de mejora — 2026-06-07
================================================================================
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_ROOT, "models")

# Añadir paths necesarios
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

# ── Clases de material ────────────────────────────────────────────────────────
MAT_SOLID    = "SOLID"
MAT_TRANS    = "TRANSPARENT"
MAT_METALLIC = "METALLIC"
MAT_PEARL    = "PEARL"
MAT_SPECIAL  = "SPECIAL"

MATERIAL_CLASSES = [MAT_SOLID, MAT_TRANS, MAT_METALLIC, MAT_PEARL, MAT_SPECIAL]
MATERIAL_TO_IDX  = {m: i for i, m in enumerate(MATERIAL_CLASSES)}


def _palette_material(name_lower: str) -> str:
    """Clasifica material de color LEGO por nombre."""
    trans_kw    = ("trans-", "glitter trans", "satin trans", "glow in dark trans")
    metallic_kw = ("chrome ", "metallic ", "flat silver", "flat dark gold",
                   "bionicle silver", "bionicle gold", "bionicle copper", "reddish gold")
    pearl_kw    = ("pearl ", "speckle ")
    special_kw  = ("electric_contact", "magnet", "umber", "sienna")
    if any(k in name_lower for k in trans_kw):
        return MAT_TRANS
    if any(k in name_lower for k in metallic_kw):
        return MAT_METALLIC
    if any(k in name_lower for k in pearl_kw):
        return MAT_PEARL
    if any(k in name_lower for k in special_kw):
        return MAT_SPECIAL
    return MAT_SOLID


# ── Parámetros estadísticos por tipo de material ─────────────────────────────
# Basados en análisis de renders EEVEE del dataset simulation_x5_1D_all
# Features: [sigma_L, sigma_H, sigma_V, sigma_S, ratio_chrom, L_gradient]
MATERIAL_PROFILES = {
    MAT_SOLID: {
        # Baja varianza en todos los canales (superficie uniforme mate)
        "sigma_L":   (3.0,  8.0),   # (min, max)
        "sigma_H":   (2.0, 15.0),
        "sigma_V":   (5.0, 20.0),
        "sigma_S":   (5.0, 25.0),
        "ratio":     (0.3,  0.9),
        "L_grad":    (3.0, 12.0),
    },
    MAT_TRANS: {
        # Alta varianza en V y S (transparencia produce rangos amplios de brillo)
        "sigma_L":   (15.0, 45.0),
        "sigma_H":   (5.0,  25.0),
        "sigma_V":   (35.0, 80.0),
        "sigma_S":   (30.0, 80.0),
        "ratio":     (0.2,  0.7),
        "L_grad":    (15.0, 50.0),
    },
    MAT_METALLIC: {
        # Muy baja varianza de hue (superficie especular uniforme)
        "sigma_L":   (2.0, 10.0),
        "sigma_H":   (1.0,  6.0),
        "sigma_V":   (8.0, 25.0),
        "sigma_S":   (2.0, 10.0),
        "ratio":     (0.05, 0.3),
        "L_grad":    (5.0, 20.0),
    },
    MAT_PEARL: {
        # Baja-media varianza, tono estable pero con brillo perlado
        "sigma_L":   (4.0, 12.0),
        "sigma_H":   (2.0, 10.0),
        "sigma_V":   (8.0, 22.0),
        "sigma_S":   (3.0, 12.0),
        "ratio":     (0.1,  0.5),
        "L_grad":    (5.0, 15.0),
    },
    MAT_SPECIAL: {
        # Muy variable — materiales especiales sin patrón claro
        "sigma_L":   (5.0, 30.0),
        "sigma_H":   (3.0, 30.0),
        "sigma_V":   (5.0, 40.0),
        "sigma_S":   (3.0, 35.0),
        "ratio":     (0.05, 0.8),
        "L_grad":    (5.0, 35.0),
    },
}


def _generate_samples(material: str, n: int, seed_offset: int = 0) -> np.ndarray:
    """Genera n muestras sintéticas de features para un tipo de material."""
    rng = np.random.default_rng(42 + seed_offset)
    prof = MATERIAL_PROFILES[material]

    def samp(key):
        lo, hi = prof[key]
        return rng.uniform(lo, hi, n)

    X = np.column_stack([
        samp("sigma_L"),
        samp("sigma_H"),
        samp("sigma_V"),
        samp("sigma_S"),
        samp("ratio"),
        samp("L_grad"),
    ]).astype(np.float32)

    # Añadir jitter gaussiano para robustez
    noise = rng.normal(0, 0.5, X.shape).astype(np.float32)
    X = np.clip(X + noise, 0.0, 255.0)

    return X


class _MatMLP(nn.Module):
    """MLP ligero 6D → 5 clases."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, len(MATERIAL_CLASSES)),
        )

    def forward(self, x):
        return self.net(x)


def main():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[train_material_type_classifier] Device: {device}")

    # ── Generar dataset sintético ──────────────────────────────────────────────
    N_PER_CLASS = 1000
    X_list, y_list = [], []

    for mat_idx, mat in enumerate(MATERIAL_CLASSES):
        X_mat = _generate_samples(mat, N_PER_CLASS, seed_offset=mat_idx)
        y_mat = np.full(N_PER_CLASS, mat_idx, dtype=np.int64)
        X_list.append(X_mat)
        y_list.append(y_mat)
        print(f"  {mat}: {N_PER_CLASS} muestras generadas")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # Normalizar
    mean = X_all.mean(axis=0)
    std  = X_all.std(axis=0) + 1e-8
    X_norm = (X_all - mean) / std

    # ── DataLoader ────────────────────────────────────────────────────────────
    idx = np.random.permutation(len(X_norm))
    X_norm = X_norm[idx]; y_all = y_all[idx]

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_all,  dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = _MatMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print("Entrenando...")
    model.train()
    for epoch in range(100):
        total_loss = 0.0
        correct = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            acc = 100.0 * correct / len(X_norm)
            print(f"  Epoch {epoch+1}/100 | Loss={total_loss/len(loader):.4f} | Train acc={acc:.1f}%")

    # ── Evaluación final ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(X_t.to(device))
        preds = logits.argmax(1).cpu().numpy()
    final_acc = (preds == y_all).mean() * 100
    print(f"\nAccuracy final: {final_acc:.2f}%")

    # ── Guardar modelo ────────────────────────────────────────────────────────
    os.makedirs(_MODELS_DIR, exist_ok=True)
    model_path = os.path.join(_MODELS_DIR, "material_type_classifier.pt")
    meta_path  = os.path.join(_MODELS_DIR, "material_type_classifier_metadata.json")

    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado: {model_path}")

    metadata = {
        "mean":     mean.tolist(),
        "std":      std.tolist(),
        "classes":  MATERIAL_CLASSES,
        "n_features": 6,
        "feature_names": ["sigma_L", "sigma_H", "sigma_V", "sigma_S", "ratio_chromatic", "L_gradient"],
        "train_acc": round(float(final_acc), 2),
        "n_samples": int(len(X_norm)),
        "n_per_class": N_PER_CLASS,
        "date": "2026-06-07",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata guardada: {meta_path}")
    print("\n✅ Entrenamiento completado.")


if __name__ == "__main__":
    main()