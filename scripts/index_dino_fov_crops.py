# -*- coding: utf-8 -*-
# scripts/index_dino_fov_crops.py
# Python script to crop renders, extract DINOv2 embeddings, and insert them into PostgreSQL.

import os
import sys
import json
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client
from training.index_synthetic_renders import COLOR_HEX_TO_CODE, get_device, load_dinov2, get_transform, preprocess_render, extract_embedding

def preprocess_render_preserve_scale(img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """
    Preprocesa el recorte de la pieza para DINOv2 conservando su tamaño real en píxeles
    y aislando la silueta de la pieza sobre un fondo negro neutro (0,0,0) mediante su máscara alfa.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")
        
    # Limitar el tamaño si por alguna razón la pieza excede el canvas (con margen de seguridad de 16px)
    max_dim = canvas_size - 16
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
        
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    canvas.paste(img, (paste_x, paste_y))
    return canvas


def preprocess_render_subtraction(img: Image.Image, empty_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """
    Preprocesa el recorte de la pieza para DINOv2 usando la resta de fondo
    con una imagen de referencia de la cinta vacía.
    """
    try:
        import cv2
        img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        empty_bgr = cv2.cvtColor(np.array(empty_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Resta absoluta
        diff = cv2.absdiff(img_bgr, empty_bgr)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Umbral
        _, mask = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
        
        # Limpieza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Aplicar máscara en fondo negro plano
        result = np.zeros_like(img_bgr)
        result[mask > 0] = img_bgr[mask > 0]
        
        # Convertir a PIL
        piece = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Recortar al bounding box de la máscara
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h_c, w_c = result.shape[:2]
            y_min = max(0, y_min - 5)
            x_min = max(0, x_min - 5)
            y_max = min(h_c, y_max + 5)
            x_max = min(w_c, x_max + 5)
            piece = piece.crop((x_min, y_min, x_max, y_max))
    except Exception:
        # Fallback si falla OpenCV
        piece = img.convert("RGB")
        
    # Fit to canvas (preserving scale)
    w, h = piece.size
    if w > 0 and h > 0:
        max_dim = canvas_size - 16
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            piece = piece.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    paste_x = (canvas_size - piece.width) // 2
    paste_y = (canvas_size - piece.height) // 2
    canvas.paste(piece, (paste_x, paste_y))
    return canvas


def run_indexing(part_ref, color_hex, output_dir, clear_previous):
    """Procesa los renders y sube los embeddings a la base de datos."""
    metadata_path = os.path.join(output_dir, f"dino_fov_metadata_{part_ref}.json")
    if not os.path.exists(metadata_path):
        print(f"[Indexer ERROR] No se encontró el JSON de metadatos: {metadata_path}")
        sys.exit(1)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    device = get_device()
    model = load_dinov2(device)
    transform = get_transform()
    
    # Intentar cargar la cabeza de proyección métrica para indexación proyectada
    projection_head = None
    model_path = os.path.join(project_root, "models", "dino_metric_head.pt")
    if os.path.exists(model_path):
        from inference.knn_classifier import LegoProjectionHead
        projection_head = LegoProjectionHead().to(device)
        try:
            projection_head.load_state_dict(torch.load(model_path, map_location=device))
            projection_head.eval()
            print("[Indexer] MLP Projection Head cargado correctamente. Se indexarán embeddings proyectados (128-d).")
        except Exception as e:
            print(f"[Indexer Warning] No se pudo cargar el MLP: {e}. Se ignorará la proyección.")
            projection_head = None
    
    color_code = COLOR_HEX_TO_CODE.get(color_hex.upper(), "0")
    
    # Limpiar embeddings previos de la pieza si se solicitó reemplazar (el usuario comentó "reemplaza")
    if clear_previous:
        print(f"[Indexer] Limpiando embeddings previos para la pieza: {part_ref}")
        supabase_client.clear_piece_embeddings(part_ref)

        
    # Intentar cargar empty_belt.png si existe en el directorio de salida
    empty_belt_path = os.path.join(output_dir, "empty_belt.png")
    empty_belt = None
    if os.path.exists(empty_belt_path):
        try:
            empty_belt = Image.open(empty_belt_path).convert("RGB")
            print(f"[Indexer] Encontrado fondo de referencia vacío: {empty_belt_path}. Usando sustracción de fondo.")
        except Exception as e:
            print(f"[Indexer Warning] No se pudo abrir empty_belt.png: {e}")

    total_indexed = 0
    total_renders = len(metadata.get("renders", []))
    
    print(f"[Indexer] Procesando {total_renders} renders...")
    
    for r_idx, render_entry in enumerate(metadata.get("renders", [])):
        img_path = render_entry["image_path"]
        if not os.path.exists(img_path):
            print(f"[Indexer WARN] No existe la imagen: {img_path}")
            continue
            
        try:
            full_img = Image.open(img_path)
        except Exception as e:
            print(f"[Indexer ERROR] No se pudo abrir {img_path}: {e}")
            continue
            
        for piece in render_entry.get("pieces", []):
            bbox = piece["bbox"]  # [xmin, ymin, xmax, ymax]
            pose_idx = piece["pose_index"]
            rot_angle = int(round(piece["rotation_angle"]))
            
            # Recortar la pieza
            # Bounding box en píxeles: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            
            # Controlar bordes
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(full_img.width, int(xmax))
            ymax = min(full_img.height, int(ymax))
            
            if (xmax - xmin) < 5 or (ymax - ymin) < 5:
                # Demasiado pequeña
                continue
                
            crop_img = full_img.crop((xmin, ymin, xmax, ymax))
            
            # Preprocesar (usar sustracción si está disponible)
            if empty_belt is not None:
                empty_crop = empty_belt.crop((xmin, ymin, xmax, ymax))
                crop_proc = preprocess_render_subtraction(crop_img, empty_crop)
            else:
                crop_proc = preprocess_render_preserve_scale(crop_img)
            
            # Extraer embedding
            embedding = extract_embedding(crop_proc, model, transform, device)
            
            # Proyectar embedding si está disponible
            embedding_projected = None
            if projection_head is not None:
                with torch.no_grad():
                    t_emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
                    t_proj = projection_head(t_emb)
                    embedding_projected = t_proj[0].cpu().numpy().tolist()
            
            # Guardar en base de datos
            # Guardamos stable_face = pose_idx y pose_index = pose_idx para compatibilidad
            supabase_client.save_piece_embedding(
                part_ref=part_ref,
                stable_face=pose_idx,
                rotation_angle=rot_angle,
                embedding=embedding.tolist(),
                color_code=color_code,
                color_hex=color_hex.upper(),
                pose_index=pose_idx,
                embedding_projected=embedding_projected
            )
            
            total_indexed += 1
            
        # Reportar progreso a consola (y por tubería)
        pct = (r_idx + 1) / total_renders * 100
        print(f"PROGRESS:{pct:.1f}% | Procesado render {r_idx + 1}/{total_renders} ({len(render_entry.get('pieces', []))} piezas)")
        sys.stdout.flush()
        
    print(f"[Indexer DONE] Indexación completada. {total_indexed} embeddings guardados en BD.")

def main():
    parser = argparse.ArgumentParser(description="Extrae embeddings DINOv2 de crops del FOV y los inserta en PostgreSQL.")
    parser.add_argument("--part_ref", type=str, required=True, help="Referencia de la pieza LEGO")
    parser.add_argument("--color_hex", type=str, default="A0A5A9", help="Color HEX de la pieza")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio donde están los renders y el JSON de metadatos")
    parser.add_argument("--clear_previous", action="store_true", help="Borrar embeddings previos de esta pieza")
    
    args = parser.parse_args()
    
    run_indexing(
        part_ref=args.part_ref,
        color_hex=args.color_hex,
        output_dir=args.output_dir,
        clear_previous=args.clear_previous
    )

if __name__ == "__main__":
    main()
