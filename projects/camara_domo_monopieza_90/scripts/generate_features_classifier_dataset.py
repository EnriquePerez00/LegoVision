#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projects/camara_domo/scripts/generate_features_classifier_dataset.py
=====================================================================
Genera los metadatos de clasificación multietiqueta para las imágenes/crops existentes.
Mapea cada clase a sus 8 características topológicas consultando la base de datos Supabase.
"""

import os
import sys
import json
import random
import glob

# Configurar paths
project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "projects", "camara_domo", "scripts"))

from core.db.supabase_client import get_connection

# Las 8 clases topológicas en orden indexado
CLASSES_FEATURES = [
    "stud_solid",
    "stud_hollow",
    "technic_hole_round",
    "technic_hole_cross",
    "clip_jaw",
    "bar_handle",
    "bottom_tube",
    "bottom_pin"
]

def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def get_topological_features_map(ldraw_ids):
    """Consulta la base de datos Supabase para obtener las características topológicas de cada pieza."""
    features_map = {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Query topological_features for all parts
            cur.execute("SELECT ldraw_id, topological_features FROM lego_classes;")
            rows = cur.fetchall()
            db_data = {row["ldraw_id"]: row["topological_features"] for row in rows if row["ldraw_id"]}
    except Exception as e:
        print(f"[ERROR] Error al consultar la base de datos: {e}")
        db_data = {}
    finally:
        if conn and not isinstance(conn, (object,)): # Check if it's mock
            try:
                conn.close()
            except Exception:
                pass

    for ldraw_id in ldraw_ids:
        # Default empty features
        features = {feat: 0 for feat in CLASSES_FEATURES}
        if ldraw_id in db_data and db_data[ldraw_id]:
            db_feat = db_data[ldraw_id]
            if isinstance(db_feat, str):
                try:
                    db_feat = json.loads(db_feat)
                except Exception:
                    db_feat = {}
            if isinstance(db_feat, dict):
                for feat in CLASSES_FEATURES:
                    features[feat] = 1 if db_feat.get(feat, 0) > 0 else 0
        
        # Guardar vector binario
        vector = [features[feat] for feat in CLASSES_FEATURES]
        features_map[ldraw_id] = vector
        
    return features_map

def build_dataset(crops_dir, classes, features_map, val_split=0.15, seed=42):
    random.seed(seed)
    
    train_data = []
    val_data = []
    
    # Cada subdirectorio es un ldraw_id (nombre de clase)
    for ldraw_id in sorted(os.listdir(crops_dir)):
        idx_path = os.path.join(crops_dir, ldraw_id)
        if not os.path.isdir(idx_path):
            continue
            
        if ldraw_id not in classes:
            print(f"[WARN] ldraw_id no está en classes.txt: {ldraw_id}")
            continue
            
        class_idx = classes.index(ldraw_id)
            
        # Obtener etiquetas multietiqueta para esta clase
        labels = features_map.get(ldraw_id, [0] * len(CLASSES_FEATURES))
        
        # Buscar todas las imágenes de esta clase
        img_extensions = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for ext in img_extensions:
            image_files.extend(glob.glob(os.path.join(idx_path, ext)))
            
        if not image_files:
            continue
            
        # Shuffle y split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1.0 - val_split))
        
        train_imgs = image_files[:split_idx]
        val_imgs = image_files[split_idx:]
        
        for img_path in train_imgs:
            # Ruta relativa al directorio de camara_domo/data
            rel_path = os.path.relpath(img_path, os.path.join(project_root, "projects", "camara_domo", "data"))
            train_data.append({
                "path": rel_path,
                "class_idx": class_idx,
                "ldraw_id": ldraw_id,
                "labels": labels
            })
            
        for img_path in val_imgs:
            # Ruta relativa al directorio de camara_domo/data
            rel_path = os.path.relpath(img_path, os.path.join(project_root, "projects", "camara_domo", "data"))
            val_data.append({
                "path": rel_path,
                "class_idx": class_idx,
                "ldraw_id": ldraw_id,
                "labels": labels
            })
            
    return train_data, val_data

def main():
    classes_path = os.path.join(project_root, "projects", "camara_domo", "data", "classes.txt")
    if not os.path.exists(classes_path):
        print(f"[ERROR] classes.txt no existe en {classes_path}")
        sys.exit(1)
        
    classes = load_classes(classes_path)
    print(f"[OK] Cargadas {len(classes)} clases de classes.txt")
    
    # Obtener el mapa de características
    features_map = get_topological_features_map(classes)
    print(f"[OK] Mapa de características topológicas construido desde la base de datos.")
    
    # Mostrar resumen de cuántas clases presentan cada característica
    counts = [0] * len(CLASSES_FEATURES)
    for ldraw_id, vector in features_map.items():
        for i, val in enumerate(vector):
            counts[i] += val
    
    print("\nDistribución de características en el catálogo de clases:")
    for feat, count in zip(CLASSES_FEATURES, counts):
        print(f"  - {feat}: {count} / {len(classes)} clases ({count/len(classes)*100:.1f}%)")
    print()

    # Directorios de crops
    cen_dir = os.path.join(project_root, "projects", "camara_domo", "data", "efficientnet_train", "cenital", "train")
    lat_dir = os.path.join(project_root, "projects", "camara_domo", "data", "efficientnet_train", "lateral", "train")
    
    out_dir = os.path.join(project_root, "projects", "camara_domo", "data")
    
    # Cenital
    if os.path.exists(cen_dir):
        print("=== Procesando Dataset Cenital ===")
        train, val = build_dataset(cen_dir, classes, features_map)
        metadata = {
            "classes_features": CLASSES_FEATURES,
            "train": train,
            "val": val
        }
        out_path = os.path.join(out_dir, "features_cenital_metadata.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Guardado dataset Cenital en {out_path} ({len(train)} train, {len(val)} val)")
        
    # Lateral
    if os.path.exists(lat_dir):
        print("=== Procesando Dataset Lateral ===")
        train, val = build_dataset(lat_dir, classes, features_map)
        metadata = {
            "classes_features": CLASSES_FEATURES,
            "train": train,
            "val": val
        }
        out_path = os.path.join(out_dir, "features_lateral_metadata.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Guardado dataset Lateral en {out_path} ({len(train)} train, {len(val)} val)")

if __name__ == "__main__":
    main()
