
# =============================================================================
# ALINEACION DE PREPROCESSING (scene_config.py -> pipeline unificado):
#   - Fondo de cinta: DINO_BG_COLOR = (37, 65, 84) = #254154
#   - Tamano canvas: DINO_CANVAS_SIZE = 224 px
#   - Margen: DINO_CANVAS_MARGIN_PX = 8 px
#   - Las referencias generadas por generate_physics_ref_multiangle.py usan
#     exactamente la misma camara ortografica y escena que el dataset YOLO,
#     garantizando alineacion total entre entrenamiento, indexado e inferencia.
# =============================================================================
"""
LegoVision — Re-indexer usando renders sintéticos coloreados.
=============================================================
Re-indexa embeddings DINOv2 usando dos fuentes:
  1. data/synthetic_renders/render_PART_COLORHEX.png   (vista isométrica única)
  2. data/ref_multiangle/ref_PART_COLORHEX_rotANG.png  (12 vistas por pieza)

El preprocessing DEBE ser idéntico al de classify() en classifier.py
para garantizar que los embeddings de referencia y query sean comparables.

Usage:
    python training/index_synthetic_renders.py [--clear]
    --clear: Borra todos los embeddings antes de re-indexar.
"""

import os
import sys
import re
import glob
import json
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.db import supabase_client

# Optimizaciones (sprint 2):
#   - 2.1 ThreadPool de PIL+preprocess+transform (8 hilos CPU) mientras la
#         GPU MPS procesa el batch anterior.
#   - 2.2 batch_size por defecto 128 (antes 64).
PREPROC_WORKERS = 8
DEFAULT_BATCH_SIZE = 128

# COLOR HEX → BrickLink color code mapping (loaded dynamically from color_catalog.json)
COLOR_HEX_TO_CODE = {}
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_catalog_path = os.path.join(_project_root, "database", "color_catalog.json")
if os.path.exists(_catalog_path):
    try:
        with open(_catalog_path, "r", encoding="utf-8") as _f:
            _catalog_data = json.load(_f)
        for _bl_code, _info in _catalog_data.items():
            _hex = _info.get("hex", "").lstrip("#").upper()
            if _hex:
                COLOR_HEX_TO_CODE[_hex] = _bl_code
    except Exception as _e:
        print(f"[Warning] Failed to dynamically load COLOR_HEX_TO_CODE: {_e}")

# Fallback/merge default mappings to ensure basic compatibility
_defaults = {
    "A0A5A9": "86",  # Light Bluish Gray
    "1B1B1B": "11",  # Black
    "C91A09": "5",   # Red
    "5A5A5A": "85",  # Dark Bluish Gray
    "808080": "0",   # Various
    "F2CD37": "3",   # Yellow
    "899395": "297", # Flat Silver
    "720012": "59",  # Dark Red
    "DFD1A5": "2",   # Tan
    "0A3C9F": "7",   # Blue
    "FE8A18": "4",   # Orange
    "254154": "85",  # Petrol (background)
}
for _hex, _code in _defaults.items():
    if _hex not in COLOR_HEX_TO_CODE:
        COLOR_HEX_TO_CODE[_hex] = _code


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dinov2(device):
    print(f"[LegoVision Index] Cargando DINOv2 ViT-S/14 en {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device)
    model.eval()
    print("[LegoVision Index] DINOv2 cargado.")
    return model


def get_transform():
    """DEBE ser idéntico al transform de classifier.py."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_render(img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """
    Preprocesa un render para DINOv2.
    """
    # Si la imagen tiene canal alpha (transparencia), la combinamos con el fondo azul petróleo (#254154)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGBA", img.size, (37, 65, 84, 255))
        img = Image.alpha_composite(background, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")
        
    margin = 8
    max_dim = canvas_size - 2 * margin
    w, h = img.size
    scale = min(max_dim / w, max_dim / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_size, canvas_size), (37, 65, 84))
    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas


def extract_embedding(img: Image.Image, model, transform, device) -> np.ndarray:
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
        if hasattr(features, "last_hidden_state"):
            vec = features.last_hidden_state[:, 0, :]
        else:
            vec = features
        vec = vec[0]
        vec = vec / vec.norm()
    return vec.cpu().numpy().astype(np.float32)


def _parse_and_load(p, regex, transform):
    """Parsea metadata desde el filename y carga PIL+preprocess+transform.
    Diseñado para ejecutarse en ThreadPool (opt 2.1)."""
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

    try:
        img = Image.open(p).convert("RGB")
        img_proc = preprocess_render(img)
        transformed = transform(img_proc)
    except Exception as e:
        return ("err", os.path.basename(p), str(e))

    meta = {
        "part_ref": part_ref,
        "color_hex": color_hex,
        "rotation_angle": rotation_angle,
        "pose_index": pose_index,
        "color_code": color_code,
        "filename": os.path.basename(p),
    }
    return ("ok", transformed, meta)


def index_directory(image_paths, regex, model, transform, device, face_id,
                    batch_size=DEFAULT_BATCH_SIZE):
    """Indexa una lista de imágenes con el regex dado en lotes.
    Retorna (indexed, failed).

    Optimizaciones (sprint 2):
      - 2.1 Preprocess con ThreadPoolExecutor (PREPROC_WORKERS=8) para que
        la GPU MPS no espere por I/O y resize/normalize CPU-bound.
      - 2.2 batch_size por defecto 128.
    """
    indexed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # 2.1 — preprocess en paralelo (CPU)
            results = list(executor.map(
                lambda p: _parse_and_load(p, regex, transform),
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

            # 2.3 — contar SOLO tras éxito del save (antes contaba antes y
            # el log marcaba fallidos+indexados duplicados cuando el insert
            # fallaba pero el bucle ya había sumado indexed).
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
                indexed += len(batch_to_save)  # 2.3 — solo tras save OK
                print(f"  ✅ batch {i//batch_size + 1}: "
                      f"{len(batch_to_save)} embeddings ({face_id=})")
            except Exception as e:
                failed += len(valid_metadata)
                print(f"  ❌ Error en lote {i//batch_size + 1}: {e}")

    return indexed, failed


def main():
    parser = argparse.ArgumentParser(description="Re-indexar embeddings DINOv2.")
    parser.add_argument("--clear", action="store_true", help="Limpiar embeddings antes de indexar.")
    parser.add_argument("--renders_dir", type=str,
                        default=os.path.join(project_root, "data", "synthetic_renders"))
    parser.add_argument("--multiangle_dir", type=str,
                        default=os.path.join(project_root, "data", "ref_multiangle"))
    args = parser.parse_args()

    device = get_device()
    model = load_dinov2(device)
    transform = get_transform()

    if args.clear:
        print("[LegoVision Index] Limpiando embeddings existentes...")
        supabase_client.clear_embeddings()

    total_indexed = 0
    total_failed = 0

    # ── 1. Renders isométricos (render_PART_COLOR.png) ──
    renders_dir = args.renders_dir
    if os.path.isdir(renders_dir):
        pattern = os.path.join(renders_dir, "render_*.png")
        paths = sorted(glob.glob(pattern))
        regex = re.compile(r"render_([a-zA-Z0-9_]+)_([A-F0-9]{6})\.png", re.IGNORECASE)
        print(f"[LegoVision Index] Indexando {len(paths)} renders isométricos...")
        n, f = index_directory(paths, regex, model, transform, device, face_id=0)
        total_indexed += n
        total_failed += f
    else:
        print(f"[WARN] No encontrado: {renders_dir}")

    # ── 2. Renders multi-ángulo (ref_PART_COLOR_rotANG.png) ──
    multiangle_dir = args.multiangle_dir
    if os.path.isdir(multiangle_dir):
        pattern = os.path.join(multiangle_dir, "ref_*.png")
        paths = sorted(glob.glob(pattern))
        regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})(?:_pose\d+)?_rot(\d+)\.png", re.IGNORECASE)
        print(f"\n[LegoVision Index] Indexando {len(paths)} renders multi-ángulo...")
        n, f = index_directory(paths, regex, model, transform, device, face_id=1)
        total_indexed += n
        total_failed += f
    else:
        print(f"[INFO] No encontrado directorio multi-ángulo: {multiangle_dir}")

    print(f"\n[LegoVision Index] Indexación completada:")
    print(f"  ✅ {total_indexed} embeddings guardados")
    print(f"  ❌ {total_failed} errores")


if __name__ == "__main__":
    main()
