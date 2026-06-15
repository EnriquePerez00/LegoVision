# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/reindex_dinov2_eevee.py
=======================================================
Indexa embeddings DINOv2 para las referencias renderizadas (cenital + lateral),
aplicando simetría de crops leyendo bboxes de metadata.json.
"""
import os, sys, re, glob, argparse, json
import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, legovic_root)

from config_loader import cfg
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("dinov2")

from training.index_synthetic_renders import (
    get_device, load_dinov2, get_transform, COLOR_HEX_TO_CODE, PREPROC_WORKERS, DEFAULT_BATCH_SIZE
)
from database import supabase_client

CINTA_BG_RGB = (128, 165, 185)

def _build_clean_canvas(crop_img: Image.Image, canvas_size: int = 224, margin_px: int = 8, bg_color=CINTA_BG_RGB) -> Image.Image:
    w_p, h_p = crop_img.size
    if w_p <= 0 or h_p <= 0:
        return Image.new("RGB", (canvas_size, canvas_size), bg_color)
    max_dim = canvas_size - 2 * margin_px
    scale = min(max_dim / w_p, max_dim / h_p)
    new_w = max(1, int(round(w_p * scale)))
    new_h = max(1, int(round(h_p * scale)))
    resized = crop_img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    canvas.paste(resized, ((canvas_size - new_w) // 2, (canvas_size - new_h) // 2))
    return canvas

def _parse_and_load_cropped(p, regex, transform, metadata_lookup, cam_name):
    m = regex.match(os.path.basename(p))
    if not m:
        return None
    part_ref = m.group(1)
    color_hex = m.group(2).upper()
    if regex.groups >= 4:
        pose_str = m.group(3)
        pose_index = int(pose_str) if pose_str is not None else 0
        rotation_angle = int(m.group(4))
    elif regex.groups == 3:
        pose_index = 0
        rotation_angle = int(m.group(3))
    else:
        pose_index = 0
        rotation_angle = 0
    color_code = COLOR_HEX_TO_CODE.get(color_hex, "0")

    fname = os.path.basename(p)
    try:
        img = Image.open(p).convert("RGB")
        if fname in metadata_lookup:
            bbox = metadata_lookup[fname][cam_name]
            iw, ih = img.size
            cx1, cy1, cx2, cy2 = bbox
            crop_img = img.crop((
                max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
                min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih))
            ))
            
            # --- Symmetric preprocessing (Option A) ---
            from run_evaluation import segment_crop_sam, apply_sam_mask_to_crop
            mask = segment_crop_sam(img, bbox)
            crop_img_masked = apply_sam_mask_to_crop(crop_img, mask, bg_color=CINTA_BG_RGB)
            img_proc = _build_clean_canvas(crop_img_masked)
        else:
            from training.index_synthetic_renders import preprocess_render
            img_proc = preprocess_render(img)
            
        transformed = transform(img_proc)
    except Exception as e:
        return ("err", fname, str(e))

    meta = {
        "part_ref": part_ref,
        "color_hex": color_hex,
        "rotation_angle": rotation_angle,
        "pose_index": pose_index,
        "color_code": color_code,
        "filename": fname,
    }
    return ("ok", transformed, meta)

def index_directory_cropped(image_paths, regex, model, transform, device, face_id, metadata_lookup, cam_name, batch_size=DEFAULT_BATCH_SIZE):
    indexed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            results = list(executor.map(
                lambda p: _parse_and_load_cropped(p, regex, transform, metadata_lookup, cam_name),
                batch_paths,
            ))

            imgs_tensor = []
            valid_metadata = []
            for r in results:
                if r is None:
                    continue
                if r[0] == "err":
                    failed += 1
                    print(f"  ❌ Error cargando {r[1]}: {r[2]}")
                    continue
                imgs_tensor.append(r[1])
                valid_metadata.append(r[2])

            if not imgs_tensor:
                continue

            try:
                batch_tensor = torch.stack(imgs_tensor).to(device)
                with torch.no_grad():
                    features = model(batch_tensor)
                    features = torch.nn.functional.normalize(features, dim=-1)
                    embeddings = features.cpu().numpy()

                batch_to_save = []
                for idx, meta in enumerate(valid_metadata):
                    emb = embeddings[idx].tolist()
                    batch_to_save.append({
                        "part_ref": meta["part_ref"],
                        "stable_face": face_id,
                        "rotation_angle": meta["rotation_angle"],
                        "pose_index": meta.get("pose_index", 0),
                        "embedding": emb,
                        "color_code": meta["color_code"],
                        "color_hex": meta["color_hex"],
                    })

                supabase_client.save_piece_embeddings_batch(batch_to_save)
                indexed += len(batch_to_save)
                print(f"  ✅ batch {i//batch_size + 1}: {len(batch_to_save)} embeddings (cam={cam_name})")
            except Exception as e:
                failed += len(valid_metadata)
                print(f"  ❌ Error en lote {i//batch_size + 1}: {e}")

    return indexed, failed

def main():
    import time as _time
    _t_start = _time.perf_counter()

    parser = argparse.ArgumentParser(description="Reindexar embeddings DINOv2 con simetría de crops.")
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--clear", action="store_true", default=True)
    args = parser.parse_args()

    ref_dir = args.ref_dir or os.path.join(project_root, "data", "dinov2_refs")

    log_execution_header(log, "reindex_dinov2_eevee.py", ref_dir=ref_dir, clear=args.clear)

    device = get_device()
    model = load_dinov2(device)
    transform = get_transform()

    if args.clear:
        log.info("Limpiando embeddings existentes...")
        supabase_client.clear_embeddings()

    # Load all metadata_worker_*.json and metadata.json files
    metadata_lookup = {}
    metadata_files = glob.glob(os.path.join(ref_dir, "metadata_worker_*.json"))
    main_meta = os.path.join(ref_dir, "metadata.json")
    if os.path.isfile(main_meta):
        metadata_files.append(main_meta)
        
    log.info(f"Encontrados {len(metadata_files)} archivos de metadata para unificar.")
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata_db = json.load(f)
            for r in metadata_db.get("renders", []):
                fname = r["file_name"]
                metadata_lookup[fname] = {
                    "cenital": r["cameras"]["cenital"]["bbox_norm"],
                    "lateral": r["cameras"]["lateral"]["bbox_norm"]
                }
        except Exception as e:
            log.warning(f"Error al leer {meta_file}: {e}")
            
    log.info(f"Cargadas {len(metadata_lookup)} entradas totales para bboxes desde metadata.")
    if not metadata_lookup:
        log.warning(f"No se encontró ningún archivo de metadata en {ref_dir}. No se aplicarán crops a las refs.")

    total_indexed = 0
    total_failed = 0
    regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})(?:_pose(\d+))?_rot(\d+)\.png", re.IGNORECASE)

    # Index cenital (cam_id=1)
    cenital_dir = os.path.join(ref_dir, "cenital")
    if os.path.isdir(cenital_dir):
        paths = sorted(glob.glob(os.path.join(cenital_dir, "ref_*.png")))
        log.info(f"Indexando {len(paths)} renders cenital (cam_id=1)...")
        n, f = index_directory_cropped(paths, regex, model, transform, device, face_id=1, metadata_lookup=metadata_lookup, cam_name="cenital")
        total_indexed += n
        total_failed += f
    else:
        log.warning(f"Directorio cenital no encontrado: {cenital_dir}")

    # Index lateral (cam_id=2)
    lateral_dir = os.path.join(ref_dir, "lateral")
    if os.path.isdir(lateral_dir):
        paths = sorted(glob.glob(os.path.join(lateral_dir, "ref_*.png")))
        log.info(f"Indexando {len(paths)} renders lateral (cam_id=2)...")
        n, f = index_directory_cropped(paths, regex, model, transform, device, face_id=2, metadata_lookup=metadata_lookup, cam_name="lateral")
        total_indexed += n
        total_failed += f
    else:
        log.warning(f"Directorio lateral no encontrado: {lateral_dir}")

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "reindex_dinov2_eevee.py",
                          duration_s=_duration,
                          total_indexed=total_indexed,
                          total_failed=total_failed)
    log.info(f"✅ {total_indexed} embeddings guardados | ❌ {total_failed} errores")

if __name__ == "__main__":
    main()