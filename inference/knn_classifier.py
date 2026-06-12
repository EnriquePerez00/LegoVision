"""
LegoVision — Phase 4: DINOv2 Metric Projection & K-NN Consensus Classifier
========================================================================
Extracts DINOv2 features, projects them to a 128-d space using a trained
MLP head (if available), and executes K-NN consensus voting against
reference embeddings in PostgreSQL.
"""

import os
import sys
import json
import io
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

# Fallback physical dimensions (major_axis_mm, minor_axis_mm) for common LEGO pieces
FALLBACK_FOOTPRINT_MM = {
    "3001": (32.0, 16.0),
    "3002": (24.0, 16.0),
    "3003": (16.0, 16.0),
    "3004": (16.0, 8.0),
    "3005": (8.0, 8.0),
    "3010": (32.0, 8.0),
    "3020": (32.0, 16.0),
    "3021": (24.0, 16.0),
    "3022": (16.0, 16.0),
    "3023": (16.0, 8.0),
    "3024": (8.0, 8.0),
    "3068": (16.0, 16.0),
    "3069": (16.0, 8.0),
    "2431": (32.0, 8.0),
    "6636": (48.0, 8.0),
    "3710": (32.0, 8.0),
    "3622": (24.0, 8.0),
    "3039": (16.0, 16.0),
    "3298": (24.0, 16.0),
    "3037": (32.0, 16.0),
    "4070": (8.0, 8.0),
    "6141": (8.0, 8.0),
    "98138": (8.0, 8.0),
    "59900": (8.0, 8.0),
    "2877": (16.0, 8.0),
    "2420": (16.0, 16.0),
    "2412": (16.0, 8.0),
    "15573": (16.0, 8.0),
    "3665": (16.0, 8.0),
    "60478": (16.0, 8.0),
    "48336": (16.0, 8.0),
    "32000": (16.0, 8.0),
    "3700": (16.0, 8.0),
    "3701": (32.0, 8.0),
    "4032": (16.0, 16.0),
    "3062": (8.0, 8.0),
    "85984": (16.0, 8.0),
    "54200": (8.0, 8.0),
    "99206": (16.0, 16.0),
    "11477": (16.0, 8.0),
    "15068": (16.0, 16.0),
}

# Define the Projection Head MLP structure matching the training script
class LegoProjectionHead(nn.Module):
    def __init__(self, input_dim=384, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        features = self.net(x)
        return F.normalize(features, p=2, dim=1)


def map_hex_to_bricklink_code(hex_str):
    if not hex_str:
        return "11"
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return "11"
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        return "11"
    
    ref_rgbs = {
        "1":   [[255, 255, 255], [249, 249, 249]],  # White
        "7":   [[0, 50, 177], [10, 60, 159]],       # Blue
        "6":   [[36, 121, 61], [0, 170, 0]],        # Green
        "156": [[104, 195, 226]],                   # Medium Azure
        "11":  [[32, 32, 32], [27, 27, 27], [0, 0, 0]], # Black
        "86":  [[162, 161, 163], [160, 165, 169], [157, 158, 159]], # Light Bluish Gray
        "85":  [[90, 90, 90], [100, 100, 100], [107, 109, 103], [131, 134, 131]], # Dark Bluish Gray
        "5":   [[195, 0, 37], [201, 26, 9], [172, 46, 90]], # Red
        "15":  [[174, 233, 239]],                   # Trans-Light Blue
        "3":   [[244, 204, 46], [242, 205, 55]],    # Yellow
        "55":  [[94, 116, 140], [88, 112, 131]],    # Sand Blue
        "59":  [[114, 0, 18]],                      # Dark Red
        "2":   [[223, 209, 165], [227, 204, 157]],  # Tan
        "95":  [[136, 134, 135], [137, 147, 149]],  # Flat Silver
        "16":  [[191, 254, 0]],                     # Trans-Neon Green
        "88":  [[95, 49, 9], [92, 30, 15]],         # Reddish Brown
        "14":  [[0, 31, 159]],                      # Trans-Dark Blue
        "98":  [[240, 143, 28], [239, 142, 27]],    # Trans-Orange
        "69":  [[148, 137, 114], [159, 143, 117]],  # Dark Tan
        "297": [[203, 155, 42]],                    # Pearl Gold
        "80":  [[24, 70, 50]],                      # Dark Green
        "12":  [[254, 254, 254]],                   # Trans-Clear
    }
    
    best_code = "11"
    min_dist = float("inf")
    for code, rgbs in ref_rgbs.items():
        for ref_rgb in rgbs:
            dist = (r - ref_rgb[0])**2 + (g - ref_rgb[1])**2 + (b - ref_rgb[2])**2
            if dist < min_dist:
                min_dist = dist
                best_code = code
    return best_code


class LegoKNNClassifier:
    def __init__(self, k: int = 5, top_k_classes: int = 5):
        self.k = k
        self.top_k_classes = top_k_classes
        self.device = self._get_device()
        self.dinov2_model = None
        self.projection_head = None
        self.transform = self._build_transform()
        self._ref_embeddings: list[dict] = []
        self.is_projected_mode = False
        
        # Load exact reference dimensions from metadata JSONs cache
        self.ref_dimensions = {}
        cache_path = os.path.join(project_root, "scratch", "ref_oriented_dimensions.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                for key, val in cache_data.items():
                    parts = key.split(",")
                    ref = parts[0]
                    pose = int(parts[1])
                    angle = int(parts[2])
                    self.ref_dimensions[(ref, pose, angle)] = tuple(val)
                print(f"[KNNClassifier] Carga de dimensiones orientadas exitosa: {len(self.ref_dimensions)} clases.")
            except Exception as e:
                print(f"[KNNClassifier Warning] No se pudo cargar dimensiones orientadas de cache: {e}")

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

    def load_dinov2_model(self):
        """Lazy-load DINOv2 ViT-S/14 model."""
        if self.dinov2_model is not None:
            return
        print(f"[KNNClassifier] Cargando DINOv2 ViT-S/14 en {self.device}...")
        try:
            self.dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception:
            from transformers import AutoModel
            self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.dinov2_model.to(self.device)
        self.dinov2_model.eval()
        print("[KNNClassifier] DINOv2 cargado correctamente.")

    def load_projection_head(self):
        """Loads the trained MLP Projection Head if it exists."""
        model_path = os.path.join(project_root, "models", "dino_multimodal_head.pt")
        if os.path.exists(model_path):
            print(f"[KNNClassifier] Cargando MLP Projection Head multimodal desde {model_path}...")
            self.projection_head = LegoProjectionHead(input_dim=386).to(self.device)
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.projection_head.load_state_dict(state_dict)
                self.projection_head.eval()
                self.is_projected_mode = True
                print("[KNNClassifier] MLP Projection Head cargado y activado (Multimodal 386-d).")
            except Exception as e:
                print(f"[KNNClassifier ERROR] Error cargando pesos del MLP multimodal: {e}. Usando embeddings base.")
                self.projection_head = None
                self.is_projected_mode = False
        else:
            model_path_unimodal = os.path.join(project_root, "models", "dino_metric_head.pt")
            if os.path.exists(model_path_unimodal):
                print(f"[KNNClassifier] Cargando MLP Projection Head unimodal desde {model_path_unimodal}...")
                self.projection_head = LegoProjectionHead(input_dim=384).to(self.device)
                try:
                    state_dict = torch.load(model_path_unimodal, map_location=self.device)
                    self.projection_head.load_state_dict(state_dict)
                    self.projection_head.eval()
                    self.is_projected_mode = True
                    print("[KNNClassifier] MLP Projection Head cargado y activado (Unimodal 384-d).")
                except Exception as e:
                    print(f"[KNNClassifier ERROR] Error cargando pesos del MLP unimodal: {e}. Usando embeddings base.")
                    self.projection_head = None
                    self.is_projected_mode = False
            else:
                print("[KNNClassifier INFO] No se encontró cabezal de proyección. Usando embeddings base (384-d).")
                self.projection_head = None
                self.is_projected_mode = False

    def load_reference_embeddings(self):
        """Loads reference embeddings from PostgreSQL database."""
        if self.projection_head is None and not self.is_projected_mode:
            self.load_projection_head()
        print("[KNNClassifier] Cargando embeddings de referencia desde la BD...")
        rows = supabase_client.get_all_embeddings()
        self._ref_embeddings = []
        
        is_multimodal = False
        if self.is_projected_mode and self.projection_head is not None:
            is_multimodal = (self.projection_head.net[0].in_features == 386)
            
        for row in rows:
            emb = row["embedding"]
            emb_proj = row.get("embedding_projected")
            
            # Decide which embedding vector to store based on model availability
            if self.is_projected_mode:
                if is_multimodal:
                    # Multimodal projection: concatenate visual (384-d) + size (2-d)
                    ref = row["part_ref"]
                    pose = row["stable_face"]
                    angle = row["rotation_angle"]
                    ref_dim = self.ref_dimensions.get((ref, pose, angle))
                    if not ref_dim:
                        ref_dim = FALLBACK_FOOTPRINT_MM.get(ref, (8.0, 8.0))
                    
                    max_mm = max(ref_dim) / 10.0
                    min_mm = min(ref_dim) / 10.0
                    
                    # Normalize raw embedding
                    emb_norm = np.array(emb, dtype=np.float32)
                    emb_norm = emb_norm / (np.linalg.norm(emb_norm) + 1e-8)
                    
                    vec_input = np.concatenate([emb_norm, [max_mm, min_mm]])
                    with torch.no_grad():
                        t_input = torch.tensor(vec_input, dtype=torch.float32).unsqueeze(0).to(self.device)
                        t_proj = self.projection_head(t_input)
                        vec = t_proj[0].cpu().numpy().astype(np.float32)
                else:
                    if emb_proj is not None:
                        vec = np.array(emb_proj, dtype=np.float32)
                    else:
                        # If projection head is loaded but database row lacks projected embedding, project on the fly
                        with torch.no_grad():
                            t_emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                            t_proj = self.projection_head(t_emb)
                            vec = t_proj[0].cpu().numpy().astype(np.float32)
            else:
                vec = np.array(emb, dtype=np.float32)
                
            bl_color_code = map_hex_to_bricklink_code(row.get("color_hex"))
            self._ref_embeddings.append({
                "part_ref": row["part_ref"],
                "face": row["stable_face"],
                "angle": row["rotation_angle"],
                "embedding": vec,
                "color_hex": row.get("color_hex"),
                "color_code": bl_color_code,
            })
        print(f"[KNNClassifier] {len(self._ref_embeddings)} embeddings cargados (Projected={self.is_projected_mode}, Multimodal={is_multimodal}).")

    def _extract_embedding(self, image: Image.Image, size_info: tuple[float, float] = None) -> np.ndarray:
        """Extracts and projects embedding from a PIL Image."""
        self.load_dinov2_model()
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.dinov2_model(tensor)
            if hasattr(features, "last_hidden_state"):
                vec = features.last_hidden_state[:, 0, :]
            else:
                vec = features
            vec = vec[0]
            vec = vec / vec.norm() # 384-d normalized
            
            # Project using MLP if available
            if self.is_projected_mode and self.projection_head is not None:
                is_multimodal = (self.projection_head.net[0].in_features == 386)
                if is_multimodal:
                    if size_info is not None:
                        max_mm = size_info[0] / 10.0
                        min_mm = size_info[1] / 10.0
                    else:
                        max_mm, min_mm = 0.8, 0.8  # dummy fallback
                    
                    vec_np = vec.cpu().numpy()
                    vec_input = np.concatenate([vec_np, [max_mm, min_mm]])
                    t_input = torch.tensor(vec_input, dtype=torch.float32).unsqueeze(0).to(self.device)
                    vec_proj = self.projection_head(t_input)
                    vec = vec_proj[0]
                else:
                    vec_proj = self.projection_head(vec.unsqueeze(0))
                    vec = vec_proj[0]
                
        return vec.cpu().numpy().astype(np.float32)

    def _get_oriented_dimensions(self, img: Image.Image) -> tuple[float, float]:
        """Computes oriented length and width in mm using cv2.minAreaRect on merged contours."""
        try:
            import cv2
            img_np = np.array(img.convert("RGB"))
            bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
            dist = np.linalg.norm(img_np.astype(np.float32) - bg_color, axis=2)
            mask = (dist > 18.0).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
            if valid_contours:
                all_pts = np.vstack(valid_contours)
                rect = cv2.minAreaRect(all_pts)
                (cx, cy), (w_px, h_px), angle = rect
                major_mm = max(w_px, h_px) / 3.2
                minor_mm = min(w_px, h_px) / 3.2
                return major_mm, minor_mm
        except Exception as e:
            print(f"[KNNClassifier Warning] _get_oriented_dimensions failed: {e}")
        w, h = img.size
        return max(w, h) / 3.2, min(w, h) / 3.2

    @staticmethod
    def _classify_color(clean_image: Image.Image) -> str:
        """Helper to classify HSV color with brightness and shadow filtering to avoid lighting distortions."""
        try:
            import cv2
            img_rgb = np.array(clean_image.convert("RGB"))
            
            # Segmentar usando distancia al petrol blue
            bg_color = np.array([37.0, 65.0, 84.0], dtype=np.float32)
            dist = np.linalg.norm(img_rgb.astype(np.float32) - bg_color, axis=-1)
            mask_fg = dist > 18.0
            
            if not np.any(mask_fg):
                return "15" # White fallback
                
            fg_rgb = img_rgb[mask_fg]
            
            # Convertir la imagen completa a HSV para clasificar matices
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            fg_hsv = img_hsv[mask_fg]
            
            # Filtrar píxeles muy oscuros (sombras, bordes) o muy brillantes (brillos especulares, reflejos)
            # El canal V (brillo) en OpenCV va de 0 a 255.
            v_channel = fg_hsv[:, 2]
            valid_brightness_mask = (v_channel >= 40) & (v_channel <= 235)
            
            if np.sum(valid_brightness_mask) > 10:  # Si quedan suficientes píxeles tras filtrar
                filtered_fg_rgb = fg_rgb[valid_brightness_mask]
                filtered_fg_hsv = fg_hsv[valid_brightness_mask]
            else:
                filtered_fg_rgb = fg_rgb
                filtered_fg_hsv = fg_hsv
            
            chromatic_mask = filtered_fg_hsv[:, 1] > 25  # Aumentamos umbral de saturación de 15 a 25 para ser más robusto a ruidos acromáticos
            if np.sum(chromatic_mask) > 0.15 * len(filtered_fg_hsv):
                avg_hue = np.median(filtered_fg_hsv[chromatic_mask, 0])
                if avg_hue < 15 or avg_hue >= 165:
                    return "4"   # Red
                elif 15 <= avg_hue < 45:
                    return "14"  # Yellow
                elif 45 <= avg_hue < 85:
                    return "2"   # Green
                else:
                    return "1"   # Blue
            else:
                avg_color = filtered_fg_rgb.mean(axis=0)
                set_colors_acromatic = {
                    "0": (27, 27, 27),      # Black
                    "15": (255, 255, 255),  # White
                    "85": (80, 85, 90),     # Light Bluish Gray (calibrated for EEVEE renders)
                    "84": (45, 45, 45)      # Dark Bluish Gray (calibrated for EEVEE renders)
                }
                best_code = "15"
                min_dist = float("inf")
                for code, rgb in set_colors_acromatic.items():
                    dist = np.linalg.norm(avg_color - rgb)
                    if dist < min_dist:
                        min_dist = dist
                        best_code = code
                return best_code
        except Exception:
            return "15"

    @staticmethod
    def _get_allowed_parts_for_color(color_code: str, set_id: str = None) -> list[str]:
        """Gets allowed parts based on color selection and set."""
        try:
            if set_id:
                parts = supabase_client.get_set_parts_by_color(set_id, color_code)
                if parts:
                    return parts
                from database.set_catalog import REAL_SETS
                if set_id in REAL_SETS:
                    set_parts = REAL_SETS[set_id].get("parts", [])
                    return [p["ref"] for p in set_parts]
            
            from database.set_catalog import REAL_SETS
            target_set = "10692-1"
            parts = REAL_SETS.get(target_set, {}).get("parts", [])
            return [p["ref"] for p in parts if p["color_code"] == color_code]
        except Exception:
            return []

    def classify(self, crop: Image.Image, filter_by_color: bool = True, set_id: str = None) -> list[dict]:
        """
        Classifies a crop using combined K-NN consensus voting and precise oriented sizing.
        """
        if not self._ref_embeddings:
            self.load_projection_head()
            self.load_reference_embeddings()

        if not self._ref_embeddings:
            return []

        # Fit crop to standard 224x224 canvas (matches references) - PRESERVING SCALE
        canvas_size = 224
        margin = 8
        max_dim = canvas_size - 2 * margin
        w_piece, h_piece = crop.size
        if w_piece > 0 and h_piece > 0:
            if w_piece > max_dim or h_piece > max_dim:
                scale = max_dim / max(w_piece, h_piece)
                new_w = max(1, int(w_piece * scale))
                new_h = max(1, int(h_piece * scale))
                clean_crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                clean_crop = crop
                new_w, new_h = w_piece, h_piece
            canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0)) # Clean black background
            paste_x = (canvas_size - new_w) // 2
            paste_y = (canvas_size - new_h) // 2
            canvas.paste(clean_crop, (paste_x, paste_y))
            clean_crop = canvas
        else:
            clean_crop = crop

        # Classify color and filter candidates
        filtered_embeddings = self._ref_embeddings
        detected_color = None
        
        if filter_by_color:
            try:
                detected_color = self._classify_color(clean_crop)
                allowed_parts = self._get_allowed_parts_for_color(detected_color, set_id=set_id)
                if allowed_parts:
                    filtered_embeddings = [
                        r for r in self._ref_embeddings if r["part_ref"] in allowed_parts
                    ]
                    if not filtered_embeddings:
                        filtered_embeddings = self._ref_embeddings
            except Exception as e:
                print(f"[KNNClassifier Warning] Error filtering by color: {e}")

        # Extract oriented dimensions of the query crop
        max_query, min_query = self._get_oriented_dimensions(crop)

        # Precise size matching and candidates filtering
        valid_candidates = []
        size_scores = []
        
        allowed_parts_set = None
        if set_id:
            try:
                from database.set_catalog import REAL_SETS
                if set_id in REAL_SETS:
                    parts = [p["ref"] for p in REAL_SETS[set_id].get("parts", [])]
                    minifigs = [m["ref"] for m in REAL_SETS[set_id].get("minifigures", [])]
                    allowed_parts_set = parts + minifigs
            except Exception:
                pass

        for r in filtered_embeddings:
            ref = r["part_ref"]
            if allowed_parts_set and ref not in allowed_parts_set:
                continue
                
            pose = r["face"]
            angle = r["angle"]
            
            # Lookup cached oriented dimensions
            ref_dim = self.ref_dimensions.get((ref, pose, angle))
            if not ref_dim:
                ref_dim = FALLBACK_FOOTPRINT_MM.get(ref)
                
            if ref_dim:
                max_ref = max(ref_dim)
                min_ref = min(ref_dim)
                diff_max = abs(max_query - max_ref)
                diff_min = abs(min_query - min_ref)
                
                # Strict size filter: 2.5 mm on major axis, 2.0 mm on minor axis
                if diff_max > 2.5 or diff_min > 2.0:
                    continue
                
                dist_size = math.sqrt(diff_max**2 + diff_min**2)
                size_score = math.exp(-(dist_size**2) / (2 * (1.5**2)))
            else:
                size_score = 1.0
                
            valid_candidates.append(r)
            size_scores.append(size_score)

        if not valid_candidates:
            # Fallback: allow all candidates of the set (or matching color) with low score
            for r in filtered_embeddings:
                ref = r["part_ref"]
                if allowed_parts_set and ref not in allowed_parts_set:
                    continue
                valid_candidates.append(r)
                size_scores.append(0.1)

        if not valid_candidates:
            return []

        # Extract visual embedding
        query_vec = self._extract_embedding(clean_crop, size_info=(max_query, min_query))
        
        # Compute cosine similarity
        ref_matrix = np.stack([r["embedding"] for r in valid_candidates])
        raw_visual_scores = ref_matrix @ query_vec
        
        # Combine visual similarity and size similarity
        combined_scores = raw_visual_scores * np.array(size_scores)

        # Sort and take top K neighbors
        k_neighbors = min(self.k, len(combined_scores))
        top_k_indices = np.argsort(combined_scores)[::-1][:k_neighbors]

        # Consensus majority voting logic
        class_votes = {}
        for idx in top_k_indices:
            ref = valid_candidates[idx]
            part_ref = ref["part_ref"]
            sim_score = float(combined_scores[idx])
            
            if part_ref not in class_votes:
                class_votes[part_ref] = {
                    "votes": 0,
                    "max_sim": -1.0,
                    "sim_sum": 0.0,
                    "face": ref["face"],
                    "angle": ref["angle"]
                }
            
            class_votes[part_ref]["votes"] += 1
            class_votes[part_ref]["sim_sum"] += sim_score
            if sim_score > class_votes[part_ref]["max_sim"]:
                class_votes[part_ref]["max_sim"] = sim_score
                class_votes[part_ref]["face"] = ref["face"]
                class_votes[part_ref]["angle"] = ref["angle"]

        ranking = []
        for part_ref, data in class_votes.items():
            consensus_ratio = data["votes"] / k_neighbors
            raw_score = data["max_sim"]
            
            # Apply consensus weighting
            weighted_score = raw_score * (0.5 + 0.5 * consensus_ratio)
            
            # Normalization scale to match standard GUI confidence thresholds
            if self.is_projected_mode:
                if weighted_score >= 0.50:
                    scaled_score = 0.95 + (weighted_score - 0.50) * (0.049 / 0.50)
                else:
                    scaled_score = 0.10 + max(0.0, weighted_score) * (0.85 / 0.50)
            else:
                if weighted_score >= 0.40:
                    scaled_score = 0.96 + (weighted_score - 0.40) * (0.039 / 0.60)
                else:
                    scaled_score = 0.10 + max(0.0, weighted_score) * (0.86 / 0.40)
            
            scaled_score = min(0.9999, max(0.01, scaled_score))
            
            ranking.append({
                "part_ref": part_ref,
                "score": scaled_score,
                "face": data["face"],
                "angle": data["angle"],
                "detected_color": detected_color,
                "consensus_votes": data["votes"],
                "raw_sim": data["max_sim"]
            })

        ranking.sort(key=lambda x: x["score"], reverse=True)
        return ranking[:self.top_k_classes]

    def classify_bytes(self, image_bytes: bytes) -> list[dict]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.classify(image)

    def is_ready(self) -> bool:
        return len(self._ref_embeddings) > 0


_knn_classifier: LegoKNNClassifier | None = None

def get_knn_classifier() -> LegoKNNClassifier:
    global _knn_classifier
    if _knn_classifier is None:
        _knn_classifier = LegoKNNClassifier()
    return _knn_classifier
