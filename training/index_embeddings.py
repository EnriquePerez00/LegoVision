import os
import sys
import glob
import re
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import numpy as np

# Añadir directorio raíz al path para importar database
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_dinov2_model(device):
    print(f"[LegoVision Index] Cargando modelo DINOv2 (ViT-S/14) en {device}...")
    # Usar torch.hub para cargar el modelo pre-entrenado oficial de Meta
    # ViT-S/14 produce un vector de embedding de tamaño 384
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    return model

def get_transform():
    # IMPORTANTE: Idéntico al transform de classifier.py para coherencia de embeddings
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def remove_bg_and_fit_canvas(img_pil: Image.Image, canvas_size: int = 224, preserve_scale: bool = True) -> Image.Image:
    """
    Aplica background removal + tight crop + fit-to-canvas.
    DEBE ser idéntico al preprocesado de LegoClassifier._remove_background_opencv()
    seguido del centrado en lienzo para que los embeddings sean comparables.
    """
    try:
        import cv2
        img_rgb = np.array(img_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Auto-detectar color de fondo (claro/oscuro) por las esquinas
        corner_val = (
            int(img_rgb[0, 0, 0]) + int(img_rgb[0, -1, 0]) +
            int(img_rgb[-1, 0, 0]) + int(img_rgb[-1, -1, 0])
        ) / 4
        if corner_val > 127:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        result = np.ones_like(img_rgb) * 255
        result[mask > 0] = img_rgb[mask > 0]

        # Tight crop al bounding box de la máscara
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h_c, w_c = result.shape[:2]
            y_min = max(0, y_min - 5)
            x_min = max(0, x_min - 5)
            y_max = min(h_c, y_max + 5)
            x_max = min(w_c, x_max + 5)
            result = result[y_min:y_max, x_min:x_max]

        piece = Image.fromarray(result)
    except ImportError:
        piece = img_pil.convert("RGB")

    if not preserve_scale:
        # Fit-to-canvas: redimensionar preservando aspecto con margen de 12px
        margin = 12
        max_dim = canvas_size - 2 * margin
        w, h = piece.size
        if w > 0 and h > 0:
            scale = min(max_dim / w, max_dim / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            piece = piece.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    paste_x = (canvas_size - piece.width) // 2
    paste_y = (canvas_size - piece.height) // 2

    # Seguridad si supera el tamaño del lienzo
    if piece.width > canvas_size or piece.height > canvas_size:
        scale = min(canvas_size / piece.width, canvas_size / piece.height)
        new_w = max(1, int(piece.width * scale))
        new_h = max(1, int(piece.height * scale))
        piece = piece.resize((new_w, new_h), Image.Resampling.LANCZOS)
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2

    canvas.paste(piece, (paste_x, paste_y))
    return canvas

def index_all_renders(renders_dir, device, model, transform):
    # Buscar todos los PNG en la carpeta de renders de referencia
    pattern = os.path.join(renders_dir, "**", "part_*_face*_rot*.png")
    image_paths = glob.glob(pattern, recursive=True)
    
    if not image_paths:
        print(f"[LegoVision Index ERROR] No se encontraron renders en {renders_dir}")
        return
        
    print(f"[LegoVision Index] Indexando {len(image_paths)} imágenes de referencia...")
    
    # Expresión regular para parsear el nombre del archivo
    # e.g., part_3004_face1_rot90.png
    regex = re.compile(r"part_([a-zA-Z0-9_-]+)_face(\d+)_rot(\d+)\.png")
    
    indexed_count = 0
    
    for path in tqdm(image_paths, desc="Procesando embeddings"):
        filename = os.path.basename(path)
        match = regex.match(filename)
        if not match:
            continue
            
        part_ref = match.group(1)
        stable_face = int(match.group(2))
        rotation_angle = int(match.group(3))
        
        try:
            # Cargar imagen, eliminar fondo, tight-crop y fit-to-canvas antes de embedir
            img = Image.open(path).convert('RGB')
            img = remove_bg_and_fit_canvas(img)  # ← mismo preprocesado que el clasificador
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(tensor)[0]
                # Normalizar embedding para poder calcular similitud usando producto escalar directo
                features = features / features.norm(dim=-1, keepdim=True)
                embedding_list = features.cpu().numpy().tolist()
                
            # Guardar en base de datos
            supabase_client.save_piece_embedding(
                part_ref=part_ref,
                stable_face=stable_face,
                rotation_angle=rotation_angle,
                embedding=embedding_list
            )
            indexed_count += 1
            
        except Exception as e:
            print(f"\n[LegoVision Index ERROR] Error procesando {filename}: {e}")
            
    print(f"[LegoVision Index] Indexación completada. {indexed_count} embeddings guardados en la BD.")

def main():
    renders_dir = os.path.join(project_root, "data", "tmp", "ref_renders")
    device = get_device()
    model = load_dinov2_model(device)
    transform = get_transform()
    
    index_all_renders(renders_dir, device, model, transform)

if __name__ == "__main__":
    main()
