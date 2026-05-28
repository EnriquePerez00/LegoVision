"""
LegoVision — Phase 2: DINOv2 Similarity Classifier
===================================================
Given a cropped image of a LEGO piece (from the YOLO bounding box),
this module extracts a DINOv2 embedding and compares it via cosine
similarity against the indexed reference embeddings in the database,
returning the top-K most similar pieces.

Usage:
    from inference.classifier import LegoClassifier
    clf = LegoClassifier()
    results = clf.classify(crop_image_pil)  # PIL Image
    # results -> [{"part_ref": "3001", "score": 0.93, "face": 0, "angle": 90}, ...]
"""

import os
import sys
import io
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client


class LegoClassifier:
    """
    Two-phase LEGO piece classifier.

    Phase 1 (external): YOLOv8 single-class detector produces bounding boxes.
    Phase 2 (this class): Crop → segment background (OpenCV) → DINOv2 embedding
                          → cosine similarity vs. DB reference embeddings → top-K result.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.device = self._get_device()
        self.model = None
        self.transform = self._build_transform()

        # In-memory cache: list of dicts {"part_ref", "face", "angle", "embedding" (np.array)}
        self._ref_embeddings: list[dict] = []

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self):
        """Lazy-load DINOv2 ViT-S/14 model (384-dim embeddings)."""
        if self.model is not None:
            return
        print(f"[LegoClassifier] Cargando DINOv2 ViT-S/14 en {self.device}...")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception:
            # Fallback: try transformers library if torch.hub fails
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model.to(self.device)
        self.model.eval()
        print("[LegoClassifier] DINOv2 cargado correctamente.")

    def load_reference_embeddings(self):
        """Load all reference embeddings from the DB into memory."""
        print("[LegoClassifier] Cargando embeddings de referencia desde la BD...")
        rows = supabase_client.get_all_embeddings()
        self._ref_embeddings = []
        for row in rows:
            self._ref_embeddings.append({
                "part_ref": row["part_ref"],
                "face": row["stable_face"],
                "angle": row["rotation_angle"],
                "embedding": np.array(row["embedding"], dtype=np.float32),
            })
        print(f"[LegoClassifier] {len(self._ref_embeddings)} embeddings de referencia cargados.")

    # ------------------------------------------------------------------
    # Preprocessing: remove background with OpenCV (no SAM2 dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_background_opencv(image: Image.Image) -> Image.Image:
        """
        Simple background removal using adaptive thresholding + GrabCut-style masking.
        Works well for renders on white/uniform backgrounds.
        Returns a RGBA PIL image with background set to white.
        """
        try:
            import cv2
            import numpy as np

            img_rgb = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Convert to grayscale and threshold
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Auto-detectar si el fondo es claro (renders de ref) u oscuro (cinta de simulación) promediando las esquinas
            corner_val = (int(img_rgb[0,0,0]) + int(img_rgb[0,-1,0]) + int(img_rgb[-1,0,0]) + int(img_rgb[-1,-1,0])) / 4
            if corner_val > 127:
                # Fondo claro -> Pieza oscura (usar BINARY_INV)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                # Fondo oscuro -> Pieza clara (usar BINARY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

            # Create white background + paste masked piece
            result = np.ones_like(img_rgb) * 255  # white background
            result[mask > 0] = img_rgb[mask > 0]

            # Recortar al bounding box ajustado (tight crop) de la máscara
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Añadir un margen de 5 píxeles
                h_c, w_c = result.shape[:2]
                y_min = max(0, y_min - 5)
                x_min = max(0, x_min - 5)
                y_max = min(h_c, y_max + 5)
                x_max = min(w_c, x_max + 5)
                
                result = result[y_min:y_max, x_min:x_max]

            return Image.fromarray(result)
        except ImportError:
            # cv2 not available, return original
            return image.convert("RGB")

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract normalized DINOv2 embedding from a PIL image."""
        self.load_model()
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
            # DINOv2 returns CLS token when called directly
            if hasattr(features, "last_hidden_state"):
                # HuggingFace model
                vec = features.last_hidden_state[:, 0, :]
            else:
                vec = features  # torch.hub returns tensor directly
            vec = vec[0]  # Remove batch dim
            vec = vec / vec.norm()  # L2-normalize for cosine similarity via dot product
        return vec.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Color classification & candidate filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_color(clean_image: Image.Image) -> str:
        """
        Calcula el color dominante en espacio HSV para filtrar brillos y bordes blancos,
        y lo mapea al color oficial del set 10692-1 usando Hue para robustez ante la sobreexposición.
        """
        try:
            import cv2
            img_rgb = np.array(clean_image.convert("RGB"))
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            # Máscara de primer plano (no blanco puro de fondo)
            mask_fg = np.any(img_rgb < 250, axis=-1)
            if not np.any(mask_fg):
                return "15" # Blanco por defecto
                
            fg_rgb = img_rgb[mask_fg]
            fg_hsv = img_hsv[mask_fg]
            
            # Separar píxeles cromáticos de los acromáticos usando un umbral de saturación bajo (15)
            # debido a la sobreexposición de los renders que lava el color.
            chromatic_mask = fg_hsv[:, 1] > 15
            
            if np.any(chromatic_mask):
                # Promediamos el Hue de los píxeles cromáticos
                avg_hue = np.median(fg_hsv[chromatic_mask, 0])
                
                # Clasificar por Hue en el rango de OpenCV (0-180)
                if avg_hue < 15 or avg_hue >= 165:
                    return "4"   # Red
                elif 15 <= avg_hue < 45:
                    return "14"  # Yellow
                elif 45 <= avg_hue < 85:
                    return "2"   # Green
                else:
                    return "1"   # Blue
            else:
                # Es acromático (Blanco, Gris, Negro)
                avg_color = fg_rgb.mean(axis=0)
                set_colors_acromatic = {
                    "0": (27, 27, 27),      # Black
                    "15": (255, 255, 255),  # White
                    "85": (160, 165, 169),  # Light Bluish Gray
                    "84": (90, 90, 90)      # Dark Bluish Gray
                }
                best_code = "15"
                min_dist = float("inf")
                for code, rgb in set_colors_acromatic.items():
                    dist = np.linalg.norm(avg_color - rgb)
                    if dist < min_dist:
                        min_dist = dist
                        best_code = code
                return best_code
        except Exception as e:
            print(f"[LegoClassifier Color Error] Error en clasificación HSV: {e}")
            return "15"

    @staticmethod
    def _get_allowed_parts_for_color(color_code: str) -> list[str]:
        """Retorna las referencias de piezas que existen en el catálogo en ese color."""
        try:
            from database.set_catalog import REAL_SETS
            parts = REAL_SETS.get("10692-1", {}).get("parts", [])
            return [p["ref"] for p in parts if p["color_code"] == color_code]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Main classification API
    # ------------------------------------------------------------------

    def classify(self, crop: Image.Image, filter_by_color: bool = True) -> list[dict]:
        """
        Classify a cropped LEGO piece image.

        Args:
            crop: PIL Image of the detected piece region.
            filter_by_color: Si es True, clasifica el color y filtra candidatos de DINOv2.

        Returns:
            List of top-K matches, each as:
            {
                "part_ref": str,        # e.g. "3001"
                "score": float,         # cosine similarity (0-1)
                "face": int,            # reference face (0/1/2)
                "angle": int,           # reference rotation angle (0-330)
                "detected_color": str,  # LDraw color code
                "rank": int             # 1 = best match
            }
        """
        if not self._ref_embeddings:
            self.load_reference_embeddings()

        if not self._ref_embeddings:
            return []

        # Step 2: Remove background + tight crop
        clean_crop = self._remove_background_opencv(crop)

        # Step 2: Redimensionado por escala física real
        scale_factor = 0.3264  # Fallback precalculado para 355mm, 12mm lens, 8.44mm sensor, 2448px

        w_piece, h_piece = clean_crop.size
        if w_piece > 0 and h_piece > 0:
            new_w = max(1, int(w_piece * scale_factor))
            new_h = max(1, int(h_piece * scale_factor))
            
            # Seguridad: si la pieza redimensionada supera el tamaño del lienzo
            if new_w > 224 or new_h > 224:
                scale_sec = min(224 / new_w, 224 / new_h)
                new_w = max(1, int(new_w * scale_sec))
                new_h = max(1, int(new_h * scale_sec))

            clean_crop = clean_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (224, 224), (255, 255, 255))
            paste_x = (224 - new_w) // 2
            paste_y = (224 - new_h) // 2
            canvas.paste(clean_crop, (paste_x, paste_y))
            clean_crop = canvas

        # Step 3: Extract DINOv2 embedding
        query_vec = self._extract_embedding(clean_crop)

        # Step 3: Clasificar color e identificar candidatos permitidos
        filtered_embeddings = self._ref_embeddings
        detected_color = None
        
        if filter_by_color:
            try:
                detected_color = self._classify_color(clean_crop)
                allowed_parts = self._get_allowed_parts_for_color(detected_color)
                if allowed_parts:
                    filtered_embeddings = [
                        r for r in self._ref_embeddings if r["part_ref"] in allowed_parts
                    ]
                    # Si por algún motivo nos quedamos sin candidatos válidos, fallback a todos
                    if not filtered_embeddings:
                        filtered_embeddings = self._ref_embeddings
            except Exception as e:
                print(f"[LegoClassifier Warning] Error filtrando por color: {e}")

        # Step 4: Cosine similarity against allowed embeddings
        ref_matrix = np.stack([r["embedding"] for r in filtered_embeddings])  # (M, 384)
        scores = ref_matrix @ query_vec  # dot product = cosine sim

        # Step 5: Get top-K indices
        top_k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            ref = filtered_embeddings[idx]
            results.append({
                "part_ref": ref["part_ref"],
                "score": float(scores[idx]),
                "face": ref["face"],
                "angle": ref["angle"],
                "detected_color": detected_color,
                "rank": rank,
            })

        return results

    def classify_bytes(self, image_bytes: bytes) -> list[dict]:
        """Convenience wrapper accepting raw image bytes."""
        image = Image.open(io.BytesIO(image_bytes))
        return self.classify(image)

    def is_ready(self) -> bool:
        """True if the model is loaded and reference embeddings are available."""
        return self.model is not None and len(self._ref_embeddings) > 0


# Module-level singleton for API usage
_classifier: LegoClassifier | None = None


def get_classifier() -> LegoClassifier:
    """Return a lazily-initialized shared classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = LegoClassifier()
    return _classifier
