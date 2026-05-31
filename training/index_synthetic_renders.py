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
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

# COLOR HEX → LDraw color code mapping
COLOR_HEX_TO_CODE = {
    "A0A5A9": "85",  # Light Bluish Gray
    "1B1B1B": "0",   # Black
    "C91A09": "4",   # Red
    "5A5A5A": "8",   # Dark Gray
    "808080": "8",   # Gray
    "F2CD37": "14",  # Yellow
    "899395": "8",   # Dark Bluish Gray
    "720012": "4",   # Dark Red
    "DFD1A5": "15",  # White
    "0A3C9F": "1",   # Blue
    "FE8A18": "25",  # Orange
    "254154": "85",  # Petrol (background)
}


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


def index_directory(image_paths, regex, model, transform, device, face_id):
    """Indexa una lista de imágenes con el regex dado. Retorna (indexed, failed)."""
    indexed = 0
    failed = 0
    for p in image_paths:
        m = regex.match(os.path.basename(p))
        if not m:
            continue
        part_ref = m.group(1)
        color_hex = m.group(2).upper()
        rotation_angle = int(m.group(3)) if regex.groups >= 3 else 0
        color_code = COLOR_HEX_TO_CODE.get(color_hex, "0")

        try:
            img = Image.open(p).convert("RGB")
            img_proc = preprocess_render(img)
            embedding = extract_embedding(img_proc, model, transform, device)
            supabase_client.save_piece_embedding(
                part_ref=part_ref,
                stable_face=face_id,
                rotation_angle=rotation_angle,
                embedding=embedding.tolist(),
                color_code=color_code,
                color_hex=color_hex,
            )
            indexed += 1
            label = f"rot{rotation_angle:03d}" if rotation_angle else "iso"
            print(f"  ✅ [{label}] {part_ref} ({color_hex})")
        except Exception as e:
            failed += 1
            print(f"  ❌ Error en {os.path.basename(p)}: {e}")
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
        regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})_rot(\d+)\.png", re.IGNORECASE)
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
