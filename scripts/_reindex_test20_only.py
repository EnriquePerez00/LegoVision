# -*- coding: utf-8 -*-
"""scripts/_reindex_test20_only.py
Re-indexa SOLO los embeddings DINOv2 de las 16 refs presentes en
2camaras_random_pieza_unica/data/test20/random_20_metadata.json.

NO toca embeddings de otras piezas. Idempotente (upsert via
save_piece_embeddings_batch).
"""
from __future__ import annotations
import os, sys, re, glob, json
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from training.index_synthetic_renders import (
    get_device, load_dinov2, get_transform, COLOR_HEX_TO_CODE,
    PREPROC_WORKERS, DEFAULT_BATCH_SIZE,
)
from database import supabase_client

# Color cinta tras tone-mapping bajo Dome Light + Cross-Polarization.
# Medido en esquinas de los renders (2026-06-13). Mismo valor en
# refs DINOv2 canonicas y queries inferencia → mismo dominio visual.
CINTA_BG_RGB = (128, 165, 185)


def _build_clean_canvas(crop_img, canvas_size=224, margin_px=8, bg_color=CINTA_BG_RGB):
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


REGEX = re.compile(
    r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})(?:_pose(\d+))?_rot(\d+)\.png",
    re.IGNORECASE,
)


def main():
    metadata_path = os.path.join(
        PROJECT_ROOT, "2camaras_random_pieza_unica", "data", "test20",
        "random_20_metadata.json",
    )
    with open(metadata_path) as f:
        m = json.load(f)
    target_refs = sorted({r["ref"] for r in m["renders"]})
    print(f"Refs a indexar (solo): {target_refs}")

    ref_dir = os.path.join(
        PROJECT_ROOT, "2camaras_random_pieza_unica", "data",
        "dinov2_refs_v3_canonical",
    )

    # Cargar bboxes desde metadata
    metadata_lookup = {}
    metadata_files = glob.glob(os.path.join(ref_dir, "metadata_worker_*.json"))
    main_meta = os.path.join(ref_dir, "metadata.json")
    if os.path.isfile(main_meta):
        metadata_files.append(main_meta)
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                md = json.load(f)
            for r in md.get("renders", []):
                metadata_lookup[r["file_name"]] = {
                    "cenital": r["cameras"]["cenital"]["bbox_norm"],
                    "lateral": r["cameras"]["lateral"]["bbox_norm"],
                }
        except Exception as e:
            print(f"WARN cargando {meta_file}: {e}")
    print(f"Metadata bboxes cargados: {len(metadata_lookup)}")

    device = get_device()
    print(f"Device: {device}")
    model = load_dinov2(device)
    transform = get_transform()

    def parse_load(p, cam_name):
        match = REGEX.match(os.path.basename(p))
        if not match:
            return None
        part_ref = match.group(1)
        if part_ref not in target_refs:
            return None
        color_hex = match.group(2).upper()
        pose_index = int(match.group(3)) if match.group(3) else 0
        rotation_angle = int(match.group(4))
        color_code = COLOR_HEX_TO_CODE.get(color_hex, "0")
        fname = os.path.basename(p)
        try:
            img = Image.open(p).convert("RGB")
            if fname in metadata_lookup:
                bbox = metadata_lookup[fname][cam_name]
                iw, ih = img.size
                cx1, cy1, cx2, cy2 = bbox
                crop = img.crop((
                    max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
                    min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih)),
                ))
                img_proc = _build_clean_canvas(crop)
            else:
                from training.index_synthetic_renders import preprocess_render
                img_proc = preprocess_render(img)
            return ("ok", transform(img_proc), {
                "part_ref": part_ref, "color_hex": color_hex,
                "rotation_angle": rotation_angle, "pose_index": pose_index,
                "color_code": color_code, "filename": fname,
            })
        except Exception as e:
            return ("err", fname, str(e))

    total_indexed = total_failed = 0
    for cam_name, face_id in [("cenital", 1), ("lateral", 2)]:
        cam_dir = os.path.join(ref_dir, cam_name)
        all_paths = sorted(glob.glob(os.path.join(cam_dir, "ref_*.png")))
        paths = [
            p for p in all_paths
            if any(os.path.basename(p).startswith(f"ref_{r}_") for r in target_refs)
        ]
        print(f"\n{cam_name} (face_id={face_id}): {len(paths)} archivos")

        bs = DEFAULT_BATCH_SIZE
        with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as ex:
            for i in range(0, len(paths), bs):
                batch_paths = paths[i:i + bs]
                results = list(ex.map(lambda p: parse_load(p, cam_name), batch_paths))
                tensors, metas = [], []
                for r in results:
                    if r is None:
                        continue
                    if r[0] == "err":
                        total_failed += 1
                        continue
                    tensors.append(r[1])
                    metas.append(r[2])
                if not tensors:
                    continue
                batch = torch.stack(tensors).to(device)
                with torch.no_grad():
                    feats = model(batch).cpu().numpy()

                rows = []
                for emb, meta in zip(feats, metas):
                    rows.append({
                        "part_ref": meta["part_ref"],
                        "stable_face": face_id,
                        "rotation_angle": meta["rotation_angle"],
                        "embedding": emb.tolist(),
                        "color_code": meta["color_code"],
                        "color_hex": "#" + meta["color_hex"],
                        "pose_index": meta["pose_index"],
                    })
                try:
                    supabase_client.save_piece_embeddings_batch(rows)
                    total_indexed += len(rows)
                except Exception as e:
                    print(f"WARN insert: {e}")
                    total_failed += len(rows)
                if (i // bs) % 10 == 0:
                    print(f"  [{cam_name}] {i+bs}/{len(paths)} indexados...")

    print(f"\nTOTAL indexed={total_indexed}  failed={total_failed}")


if __name__ == "__main__":
    main()