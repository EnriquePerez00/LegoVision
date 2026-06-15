# -*- coding: utf-8 -*-
"""scripts/pipeline_test10_runner.py
Orchestrates the entire test10 pipeline:
1. Selects 10 random combinations from DB piece_embeddings.
2. Renders 12 DINOv2 reference views (cenital/lateral) for their specific poses.
3. Computes DINOv2 embeddings and updates Supabase database.
4. Renders 10 test images and bboxes in data/test10/.
5. Runs evaluation and generates HTML comparative reports.
"""
from __future__ import annotations
import os
import sys
import json
import random
import subprocess
import torch
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from database import supabase_client
from training.index_synthetic_renders import load_dinov2, get_transform

def select_10_random_combinations():
    print("[test10] Seleccionando 10 combinaciones aleatorias de la BD...")
    conn = supabase_client.get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT part_ref, color_code, color_hex, pose_index 
            FROM (
                SELECT DISTINCT part_ref, color_code, color_hex, pose_index 
                FROM piece_embeddings 
                WHERE color_hex IS NOT NULL AND color_code IS NOT NULL AND pose_index IS NOT NULL
            ) sub
            ORDER BY RANDOM() LIMIT 10
        """)
        rows = cur.fetchall()
    
    selection = []
    for r in rows:
        selection.append({
            "part_ref": r["part_ref"],
            "color_code": r["color_code"],
            "color_hex": r["color_hex"].replace("#", ""),
            "pose_index": r["pose_index"]
        })
    
    os.makedirs(os.path.join(project_root, "data", "test10"), exist_ok=True)
    sel_path = os.path.join(project_root, "data", "test10", "test10_selection.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"[test10] Guardadas 10 combinaciones en {sel_path}")
    return selection

def run_blender_rendering_refs():
    print("[test10] Renderizando vistas de referencia DINOv2 en Blender...")
    cmd = [
        "blender",
        "-b",
        "-P",
        os.path.join(project_root, "scripts", "generate_test10_refs.py")
    ]
    subprocess.run(cmd, check=True)
    print("[test10] Renders de referencia completados.")

def run_blender_rendering_test_dataset():
    print("[test10] Renderizando dataset de test de 10 muestras en Blender...")
    cmd = [
        "blender",
        "-b",
        "-P",
        os.path.join(project_root, "scripts", "generate_test10_dataset.py")
    ]
    subprocess.run(cmd, check=True)
    print("[test10] Dataset de test completado.")

def _parse_and_load_ref(p, cam_name, selection, transform):
    try:
        f = os.path.basename(p)
        parts = f.replace(".png", "").split("_")
        if len(parts) < 5:
            return None
        part_ref = parts[1]
        color_hex = parts[2]
        pose_idx = int(parts[3].replace("pose", ""))
        rot_deg = int(parts[4].replace("rot", ""))
        
        # Find matching selection item to get color code
        color_code = "0"
        for sel in selection:
            if sel["part_ref"] == part_ref and sel["color_hex"] == color_hex:
                color_code = sel["color_code"]
                break
                
        img = Image.open(p).convert("RGB")
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        tensor = transform(img_resized)
        
        meta = {
            "part_ref": part_ref,
            "stable_face": 0 if cam_name == "cenital" else 1,
            "rotation_angle": rot_deg,
            "color_code": color_code,
            "color_hex": "#" + color_hex,
            "pose_index": pose_idx,
        }
        return ("ok", tensor, meta)
    except Exception as e:
        return ("err", os.path.basename(p), str(e))

def compute_and_upload_reference_embeddings(selection):
    print("[test10] Cargando modelo DINOv2 y configurando procesamiento batch/multi-thread...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[test10] Usando dispositivo: {device}")
    model = load_dinov2(device)
    transform = get_transform()
    
    ref_dir = os.path.join(project_root, "data", "test10", "dinov2_refs")
    
    # Collect all image paths to process
    tasks_inputs = []
    for cam_name in ["cenital", "lateral"]:
        d = os.path.join(ref_dir, cam_name)
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".png"):
                tasks_inputs.append((os.path.join(d, f), cam_name))
                
    # Parallel CPU Preprocessing
    from concurrent.futures import ThreadPoolExecutor
    print(f"[test10] Preprocesando {len(tasks_inputs)} imágenes de referencia en paralelo (12 workers)...")
    
    db_rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_parse_and_load_ref, p, cam, selection, transform) for p, cam in tasks_inputs]
        results = [f.result() for f in futures]
        
    imgs_tensors = []
    valid_metas = []
    for r in results:
        if r is None:
            continue
        if r[0] == "err":
            print(f"[test10 Warning] Error preprocesando referencia {r[1]}: {r[2]}")
            continue
        imgs_tensors.append(r[1])
        valid_metas.append(r[2])
        
    if not imgs_tensors:
        print("[test10] No hay imágenes válidas para indexar.")
        return
        
    # Batch inference
    batch_size = 128
    print(f"[test10] Extrayendo embeddings DINOv2 en lotes de {batch_size} sobre {device}...")
    embeddings = []
    for i in range(0, len(imgs_tensors), batch_size):
        batch_tensor = torch.stack(imgs_tensors[i:i+batch_size]).to(device)
        with torch.no_grad():
            features = model(batch_tensor)
            features = torch.nn.functional.normalize(features, dim=-1)
            embeddings.append(features.cpu().numpy())
            
    import numpy as np
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    for idx, meta in enumerate(valid_metas):
        db_rows.append({
            "part_ref": meta["part_ref"],
            "stable_face": meta["stable_face"],
            "rotation_angle": meta["rotation_angle"],
            "embedding": all_embeddings[idx].tolist(),
            "color_code": meta["color_code"],
            "color_hex": meta["color_hex"],
            "pose_index": meta["pose_index"],
            "embedding_projected": None
        })
        
    if db_rows:
        print(f"[test10] Borrando vectores antiguos e insertando {len(db_rows)} nuevos vectores en la BD local...")
        conn = supabase_client.get_connection()
        with conn.cursor() as cur:
            # Delete old references for the exact part_ref and pose_index
            for sel in selection:
                cur.execute("""
                    DELETE FROM piece_embeddings 
                    WHERE part_ref = %s AND pose_index = %s
                """, (sel["part_ref"], sel["pose_index"]))
        
        # Save batch
        supabase_client.save_piece_embeddings_batch(db_rows)
        print("[test10] Inserción de vectores en la BD local completada.")

def main():
    selection = select_10_random_combinations()
    run_blender_rendering_refs()
    compute_and_upload_reference_embeddings(selection)
    run_blender_rendering_test_dataset()
    
    # Run evaluation
    print("[test10] Ejecutando evaluación...")
    eval_cmd = [
        "python3",
        "scripts/run_evaluation.py",
        "--metadata", "data/test10/test10_metadata.json",
        "--report", "data/test10/eval_report.json"
    ]
    subprocess.run(eval_cmd, check=True)
    
    # Generate diagnostics reports
    print("[test10] Generando reportes diagnósticos HTML...")
    # List of pieces: e.g. ref:color
    pieces_args = [f"{sel['part_ref']}:{sel['color_code']}" for sel in selection]
    rep_cmd = [
        "python3",
        "scripts/generate_piece_report.py",
        "--pieces"
    ] + pieces_args + [
        "--eval", "data/test10/eval_report.json",
        "--metadata", "data/test10/test10_metadata.json",
        "--data_dir", "data/test10",
        "--out_dir", "data/test10/report",
        "--out", "piece_report_test10.html"
    ]
    subprocess.run(rep_cmd, check=True)
    print(f"\n[OK] Pipeline test10 completado con éxito. Reportes guardados en data/test10/report/")

if __name__ == "__main__":
    main()
