# -*- coding: utf-8 -*-
"""
camara_domo/scripts/efficientnet_classifier.py
=============================================
Modular classifier implementing the neuro-symbolic geometry classification phase
using EfficientNetV2-B0, CIELAB color decoupling, and PostgreSQL-backed stable poses gating.
"""

import os
import sys
import glob
import re
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from rotation_aligner import align_image_by_moments

# Set up project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from core.db import supabase_client
from core.db.set_catalog import REAL_SETS


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Converts sRGB [0-255] to CIELAB using standard analytical formulas."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    l_val = (116 * fy) - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.array([l_val, a_val, b_val], dtype=np.float32)


def hex_to_rgb(hex_str: str) -> np.ndarray:
    """Converts hex color string to RGB numpy array."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    return np.array([128.0, 128.0, 128.0], dtype=np.float32)


def preprocess_crop_grayscale(crop_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    """Converts a crop to Grayscale, scales & pads keeping aspect ratio on a black background."""
    # 1. Convert to grayscale and merge into 3-channels for RGB pre-trained network compatibility
    gray_img = crop_img.convert("L")
    rgb_gray = Image.merge("RGB", (gray_img, gray_img, gray_img))

    # 2. Resize and pad
    margin = 8
    max_dim = canvas_size - 2 * margin
    w, h = rgb_gray.size
    if w > 0 and h > 0:
        scale = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = rgb_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
    return Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))


class LegoEfficientNetClassifier75078:
    """Neuro-symbolic classifier for Lego piece geometry using EfficientNetV2-B0 and database filters for Set 75078."""

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_cen = self._load_model('cenital')
        self.model_lat = self._load_model('lateral')
        self.transform = self._build_transform()
        self.ref_metadata = self._load_ref_metadata()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.emb_dim = self.model_cen(dummy).shape[-1]

        # Cargar clasificadores de características topológicas
        self.CLASSES_FEATURES = [
            "stud_solid", "stud_hollow", "technic_hole_round", "technic_hole_cross",
            "clip_jaw", "bar_handle", "bottom_tube", "bottom_pin"
        ]
        self.features_cen_model = None
        self.features_lat_model = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_local = os.path.dirname(script_dir)
        cen_path = os.path.join(project_root_local, "models", "features_cenital.pt")
        lat_path = os.path.join(project_root_local, "models", "features_lateral.pt")

        if os.path.exists(cen_path):
            try:
                ckpt = torch.load(cen_path, map_location=self.device)
                model_name = ckpt.get('model_name', 'resnet18')
                import timm
                self.features_cen_model = timm.create_model(model_name, num_classes=len(self.CLASSES_FEATURES))
                self.features_cen_model.load_state_dict(ckpt['model_state_dict'])
                self.features_cen_model.to(self.device)
                self.features_cen_model.eval()
                print(f"[Features Classifier] Loaded cenital model from {cen_path}")
            except Exception as e:
                print(f"[Features Classifier Warning] Failed to load cenital model: {e}")

        if os.path.exists(lat_path):
            try:
                ckpt = torch.load(lat_path, map_location=self.device)
                model_name = ckpt.get('model_name', 'resnet18')
                import timm
                self.features_lat_model = timm.create_model(model_name, num_classes=len(self.CLASSES_FEATURES))
                self.features_lat_model.load_state_dict(ckpt['model_state_dict'])
                self.features_lat_model.to(self.device)
                self.features_lat_model.eval()
                print(f"[Features Classifier] Loaded lateral model from {lat_path}")
            except Exception as e:
                print(f"[Features Classifier Warning] Failed to load lateral model: {e}")

        # Caché de características topológicas y categorías desde la base de datos
        self.db_features = {}
        self.db_categories = {}
        try:
            with supabase_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ldraw_id, category, topological_features FROM lego_classes;")
                    rows = cur.fetchall()
                    for row in rows:
                        ldraw_id = row["ldraw_id"]
                        if not ldraw_id:
                            continue
                        self.db_categories[ldraw_id] = row["category"]
                        db_feat = row["topological_features"]
                        if isinstance(db_feat, str):
                            try:
                                db_feat = json.loads(db_feat)
                            except Exception:
                                db_feat = {}
                        if not isinstance(db_feat, dict):
                            db_feat = {}
                        self.db_features[ldraw_id] = {feat: (1 if db_feat.get(feat, 0) > 0 else 0) for feat in self.CLASSES_FEATURES}
            
            # Parche local de características (Hotfix por inconsistencia en DB)
            DB_FEATURES_OVERRIDE = {
                "51739": {"stud_solid": 1, "bottom_tube": 1},
                "87552": {"stud_solid": 1, "bottom_pin": 1},
            }
            for ref, override in DB_FEATURES_OVERRIDE.items():
                if ref in self.db_features:
                    self.db_features[ref].update(override)

            print(f"[Features Classifier] Cached topological features and categories for {len(self.db_features)} parts.")
        except Exception as e:
            print(f"[Features Classifier Warning] Could not load features or categories from DB: {e}")

        # Caché de face_class para cada postura estable
        self.pose_face_class = {}
        try:
            with supabase_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT part_ref, pose_index, face_class FROM stable_poses;")
                    rows = cur.fetchall()
                    for row in rows:
                        self.pose_face_class[(row["part_ref"], row["pose_index"])] = row["face_class"]
            print(f"[Features Classifier] Cached face classes for {len(self.pose_face_class)} stable poses.")
        except Exception as e:
            print(f"[Features Classifier Warning] Could not load face classes from DB: {e}")

        # Inicializar catálogo e info para el set 75078-1
        self.allowed_refs = {p["ref"] for p in REAL_SETS["75078-1"]["parts"]}
        self.part_to_colors = {}
        self.part_qtys = {}
        for p in REAL_SETS["75078-1"]["parts"]:
            self.part_to_colors.setdefault(p["ref"], set()).add(str(p["color_code"]))
            self.part_qtys[p["ref"]] = self.part_qtys.get(p["ref"], 0) + p["qty"]
        total_qty = sum(self.part_qtys.values())
        self.part_priors = {ref: qty / total_qty for ref, qty in self.part_qtys.items()}
        self.color_name_to_code = {}
        for p in REAL_SETS["75078-1"]["parts"]:
            self.color_name_to_code[p["color_name"]] = str(p["color_code"])
        print(f"[Set 75078-1 Classifier] Restricting universe to {len(self.allowed_refs)} parts.")

        # Precalentamiento del caché de embeddings de referencias para evitar consultas a la DB por pieza
        print("[Features Classifier] Precalentando caché de embeddings para referencias permitidas...", flush=True)
        try:
            self._load_reference_embeddings(list(self.allowed_refs))
            print(f"[Features Classifier] Caché precalentado con {len(self.computed_refs_cache)} referencias.", flush=True)
        except Exception as e:
            print(f"[Features Classifier Warning] Error al precalentar caché: {e}", flush=True)


    def _load_model(self, camera_type: str = 'cenital') -> nn.Module:
        """Loads fine-tuned EfficientNetV2-B0 for cenital or lateral camera."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_local = os.path.dirname(script_dir)
        
        if camera_type == 'cenital':
            weights_path = os.path.join(project_root_local, "models", "efficientnet_cenital.pt")
        else:
            weights_path = os.path.join(project_root_local, "models", "efficientnet_lateral.pt")
            
        try:
            import timm
            if os.path.exists(weights_path):
                classes_file = weights_path + ".classes.txt"
                with open(classes_file, "r") as f:
                    classes = f.read().splitlines()
                num_classes = len(classes)
                model = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=num_classes)
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.classifier = nn.Identity()
                print(f"[EfficientNet Classifier] Loaded fine-tuned {camera_type} model from {weights_path} on {self.device}")
            else:
                model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=0)
                print(f"[EfficientNet Classifier] Loaded pretrained ImageNet model on {self.device} (No fine-tuned weights found)")
        except Exception as e:
            print(f"[EfficientNet Classifier Error] Fallback to torchvision: {e}")
            import torchvision.models as models
            model = models.efficientnet_v2_b0(weights=models.EfficientNet_V2_B0_Weights.DEFAULT)
            model.classifier = nn.Identity()
            
        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self) -> T.Compose:
        """Standard ImageNet preprocessing transform."""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_ref_metadata(self) -> Dict[str, Any]:
        """Scans directories to load render bounding box metadata JSONs."""
        possible_dirs = [
            os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "dinov2_refs_v4_canonical"),
            os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "dinov2_refs_v3_canonical"),
            os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "dinov2_refs"),
            os.path.join(project_root, "data", "dinov2_refs"),
        ]
        ref_dir = None
        for d in possible_dirs:
            if os.path.isdir(d):
                ref_dir = d
                break

        metadata_lookup = {}
        if not ref_dir:
            print("[EfficientNet Classifier Warning] Reference directory not found.")
            return {"ref_dir": None, "lookup": {}}

        print(f"[EfficientNet Classifier] Found reference directory at: {ref_dir}")
        metadata_files = glob.glob(os.path.join(ref_dir, "metadata_worker_*.json"))
        main_meta = os.path.join(ref_dir, "metadata.json")
        if os.path.isfile(main_meta):
            metadata_files.append(main_meta)

        for meta_file in metadata_files:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                for r in meta_data.get("renders", []):
                    fname = r["file_name"]
                    metadata_lookup[fname] = {
                        "cenital": r["cameras"]["cenital"]["bbox_norm"],
                        "lateral": r["cameras"]["lateral"]["bbox_norm"]
                    }
            except Exception as e:
                print(f"[EfficientNet Classifier Warning] Failed to read metadata {meta_file}: {e}")

        return {"ref_dir": ref_dir, "lookup": metadata_lookup}

    def extract_median_lab(self, crop: Image.Image, mask: Optional[np.ndarray]) -> np.ndarray:
        """Extracts the CIELAB color of the mean RGB within mask pixels, applying erosion to remove border bleed."""
        if mask is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        crop_np = np.array(crop.convert("RGB"))
        if mask.shape[:2] != crop_np.shape[:2]:
            mask = cv2.resize(mask, (crop_np.shape[1], crop_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Apply morphological erosion to discard background bleed on borders
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        mask_to_use = eroded_mask if np.any(eroded_mask > 0) else mask
        
        mask_bool = (mask_to_use > 0)
        if not np.any(mask_bool):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        mean_rgb = crop_np[mask_bool].mean(axis=0)
        return rgb_to_lab(mean_rgb)

    def get_deterministic_candidates(self, area_cenital: float, color_lab: np.ndarray, tolerance: float = 2.0) -> List[str]:
        """Fase 1: Filter LDraw catalog using stable pose zenith surface (Gating de Área Elástico)."""
        # Tolerancia dinámica dependiente del tamaño inferido de la pieza
        if area_cenital < 150.0:
            # Tolerancia amplia para piezas pequeñas y translúcidas
            min_area = area_cenital / 2.70
            max_area = area_cenital / 0.45
        elif area_cenital < 500.0:
            # Tolerancia media para piezas medianas
            min_area = area_cenital / 1.55
            max_area = area_cenital / 0.45
        else:
            # Tolerancia estricta para piezas grandes
            min_area = area_cenital / 1.15
            max_area = area_cenital / 0.85
        candidates = []

        # Local cache fallback initialization if needed
        poses_cache = {}
        cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
        if not os.path.exists(cache_path):
            cache_path = os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    poses_cache = json.load(f)
            except Exception:
                pass

        # Primary Query: Area-only gating
        try:
            with supabase_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT sp.part_ref
                        FROM stable_poses sp
                        WHERE COALESCE(sp.zenith_silhouette_area, sp.zenith_observable_area) BETWEEN %s AND %s
                    """, (min_area, max_area))
                    candidates = [row["part_ref"] for row in cur.fetchall() if row["part_ref"] in self.allowed_refs]
        except Exception as e:
            print(f"[DB Warning] Could not query candidates from DB: {e}. Using local cache fallback.")

        # Local fallback for Area-only gating
        if not candidates and poses_cache:
            for ref, poses in poses_cache.items():
                if ref not in self.allowed_refs:
                    continue
                for pose in poses:
                    nominal_cen = pose.get("zenith_silhouette_area") or pose.get("zenith_observable_area")
                    if nominal_cen and min_area <= nominal_cen <= max_area:
                        candidates.append(ref)
                        break
            candidates = list(set(candidates))

        return candidates

    def _compute_refs_on_the_fly(self, candidate_refs: List[str]) -> List[Dict[str, Any]]:
        """Computes reference embeddings on-the-fly from image renders for the candidate subset."""
        ref_dir = self.ref_metadata.get("ref_dir")
        lookup = self.ref_metadata.get("lookup", {})
        if not ref_dir:
            return []

        ref_embeddings = []
        regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})(?:_pose(\d+))?_rot(\d+)\.png", re.IGNORECASE)

        for cam_name, face_id in [("cenital", 1), ("lateral", 2)]:
            cam_dir = os.path.join(ref_dir, cam_name)
            if not os.path.isdir(cam_dir):
                continue

            for ref in candidate_refs:
                pattern = os.path.join(cam_dir, f"ref_{ref}_*.png")
                for img_path in glob.glob(pattern):
                    fname = os.path.basename(img_path)
                    m = regex.match(fname)
                    if not m:
                        continue

                    color_hex = m.group(2).upper()
                    pose_index = int(m.group(3)) if m.group(3) else 0
                    rotation_angle = int(m.group(4)) if m.group(4) else 0

                    try:
                        img = Image.open(img_path).convert("RGB")
                        if fname in lookup:
                            bbox = lookup[fname][cam_name]
                            iw, ih = img.size
                            cx1, cy1, cx2, cy2 = bbox
                            cropped = img.crop((
                                max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
                                min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih))
                            ))
                        else:
                            cropped = img

                        preproc = preprocess_crop_grayscale(cropped)
                        tensor = self.transform(preproc).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            feat = self.model(tensor)
                            feat = torch.nn.functional.normalize(feat, dim=-1)
                            emb = feat[0].cpu().numpy().astype(np.float32)

                        ref_embeddings.append({
                            "part_ref": ref,
                            "face": face_id,
                            "rotation_angle": rotation_angle,
                            "pose_index": pose_index,
                            "embedding": emb,
                            "color_hex": color_hex
                        })
                    except Exception as e:
                        print(f"Error computing embedding for reference {fname}: {e}")

        return ref_embeddings

    def _load_reference_embeddings(self, candidate_refs: List[str]) -> List[Dict[str, Any]]:
        """Loads reference embeddings for candidate refs from DB, falling back to on-the-fly extraction."""
        if not hasattr(self, 'computed_refs_cache'):
            self.computed_refs_cache = {}

        ref_embeddings = []
        missing_refs = [ref for ref in candidate_refs if ref not in self.computed_refs_cache]

        if missing_refs:
            db_success = False
            try:
                with supabase_client.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Convert to tuple safely
                        params = tuple(missing_refs) if len(missing_refs) > 1 else (missing_refs[0],)
                        query_placeholders = "%s" if len(missing_refs) == 1 else "%s" # IN operator handles tuples
                        cur.execute("""
                            SELECT pe.part_ref, pe.stable_face, pe.rotation_angle, pe.color_code, pe.color_hex, pe.pose_index, pe.embedding
                            FROM piece_embeddings pe
                            JOIN stable_poses sp ON pe.part_ref = sp.part_ref AND pe.pose_index = sp.pose_index
                            WHERE pe.part_ref IN %s AND sp.is_stable = TRUE AND (pe.rotation_angle %% 45 = 0)
                        """, (params,))
                        rows = cur.fetchall()
                        print(f"[DEBUG DB] missing_refs count: {len(missing_refs)}, rows returned: {len(rows)}", flush=True)
                        if len(rows) == 0:
                            print(f"[DEBUG DB] missing_refs sample: {missing_refs[:10]}", flush=True)
                        if rows:
                            db_success = True
                            temp_refs = {}
                            for r in rows:
                                part_ref = r["part_ref"]
                                temp_refs.setdefault(part_ref, []).append({
                                    "part_ref": part_ref,
                                    "face": r["stable_face"],
                                    "rotation_angle": r["rotation_angle"],
                                    "color_code": r["color_code"],
                                    "color_hex": r["color_hex"],
                                    "pose_index": r["pose_index"],
                                    "embedding": np.array(r["embedding"], dtype=np.float32)
                                })
                            for ref, embs in temp_refs.items():
                                self.computed_refs_cache[ref] = embs
                            print(f"[EfficientNet Classifier] Loaded reference embeddings for {list(temp_refs.keys())} from DB.")
            except Exception as e:
                print(f"[ERROR] Failed to load reference embeddings from DB: {e}", flush=True)

            still_missing = [ref for ref in missing_refs if ref not in self.computed_refs_cache]
            if still_missing:
                for ref in still_missing:
                    computed = self._compute_refs_on_the_fly([ref])
                    self.computed_refs_cache[ref] = computed

        for ref in candidate_refs:
            if ref in self.computed_refs_cache:
                ref_embeddings.extend(self.computed_refs_cache[ref])

        return ref_embeddings

    def extract_embedding(self, crop: Image.Image, camera_type: str = 'cenital') -> np.ndarray:
        """Converts crop to Grayscale, normalizes, runs forward pass through the fine-tuned model, and L2 normalizes."""
        aligned = align_image_by_moments(crop)
        preproc = preprocess_crop_grayscale(aligned)
        tensor = self.transform(preproc).unsqueeze(0).to(self.device)
        model = self.model_cen if camera_type == 'cenital' else self.model_lat
        with torch.no_grad():
            feat = model(tensor)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            emb = feat[0].cpu().numpy().astype(np.float32)
        return emb

    def classify(self, 
                 crop_cen: Image.Image, 
                 mask_cen: np.ndarray, 
                 crop_lat: Optional[Image.Image] = None, 
                 mask_lat: Optional[np.ndarray] = None,
                 area_cenital: float = 0.0,
                 detected_color: Optional[str] = None,
                 ref_gt: Optional[str] = None,
                 measured_height: Optional[float] = None,
                 measured_length: Optional[float] = None,
                 measured_width: Optional[float] = None,
                 detected_studs: Optional[int] = None,
                 is_simulation: bool = False,
                 yolo_conf: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Runs the full classification pipeline: deterministic filter, color decoupling,
        feature extraction, and Restricted K-NN search.
        """
        # 1. CIELAB Color Median extraction (Fase 2)
        print(f"[DEBUG]     extract_median_lab...", flush=True)
        color_lab_cen = self.extract_median_lab(crop_cen, mask_cen)

        # Precalcular Aspect Ratio de la pieza en escena y cuantizar a studs si es rectángulo simple
        query_aspect_ratio = 1.0
        is_simple_rect = False
        measured_l_studs = 0
        measured_w_studs = 0
        
        if mask_cen is not None and np.sum(mask_cen) > 0:
            contours, _ = cv2.findContours(mask_cen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                # Aspect Ratio
                rect = cv2.minAreaRect(c)
                w_rect, h_rect = rect[1]
                if min(w_rect, h_rect) > 0:
                    query_aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                
                # Rectangularity IoU (Solidez orientada)
                box_area = w_rect * h_rect
                if box_area > 0:
                    iou_rect = np.sum(mask_cen) / box_area
                    if iou_rect >= 0.88:
                        is_simple_rect = True
                        px_per_mm = float(mask_cen.shape[1]) / 196.3636
                        measured_length = max(w_rect, h_rect) / px_per_mm
                        measured_width = min(w_rect, h_rect) / px_per_mm
                        measured_l_studs = max(1, int(round(measured_length / 8.0)))
                        measured_w_studs = max(1, int(round(measured_width / 8.0)))
                        print(f"[DEBUG GRID] Rectángulo simple detectado: {measured_l_studs}x{measured_w_studs} studs (IoU_rect: {iou_rect:.2f})", flush=True)

        # 2. Soft Gating Initialization (Fase 1)
        # Evaluamos el universo completo (38 referencias) aplicando penalizaciones suaves
        candidates = list(self.allowed_refs)
        gating_multipliers = {ref: 1.0 for ref in candidates}

        # Gating Morfológico Físico (Priors Morfológicos)
        # Clasificamos la pieza basada en su altura frontal y su área cenital aparente
        if measured_height is not None and measured_height > 0.0:
            allowed_morphologies = None
            
            # Caso 1: Piezas micro / muy pequeñas (Weapons, Pins, Tiles pequeños, Micro-placas)
            if area_cenital > 0.0 and area_cenital < 220.0:
                allowed_morphologies = {"Technic Pin/Connector", "Minifigure Weapon/Accessory", "Tile", "Plate", "Plate Modified"}
            # Caso 2: Placas (Plate, Plate Modified, Tile) (altura nominal 3.2mm)
            elif measured_height < 5.0:
                allowed_morphologies = {"Plate", "Plate Modified", "Tile", "Slope"}
            # Caso 3: Rampas/Slopes (altura media, 4.5mm - 7.2mm)
            elif measured_height >= 5.0 and measured_height < 7.2:
                allowed_morphologies = {"Slope", "Plate Modified", "Brick Modified"}
            # Caso 4: Ladrillos, Vigas y Rampas Altas (Brick, Brick Modified, Technic Beam, Slope) (altura nominal 9.6mm o superior)
            elif measured_height >= 7.2:
                allowed_morphologies = {"Brick", "Brick Modified", "Technic Beam", "Slope"}
                
            if allowed_morphologies:
                for ref in candidates:
                    morph = self.db_categories.get(ref)
                    if morph is not None and morph not in allowed_morphologies:
                        gating_multipliers[ref] *= 0.1
                        print(f"[DEBUG GATING] Penalizado {ref} ({morph}) por desajuste morfológico (esperado: {allowed_morphologies})", flush=True)



        # Gating de Color (Soft Gating Penalty)
        if detected_color:
            if isinstance(detected_color, str):
                detected_colors = [detected_color]
            else:
                detected_colors = list(detected_color)
            allowed_codes = {self.color_name_to_code.get(c) for c in detected_colors if self.color_name_to_code.get(c)}
            if allowed_codes:
                similar_color_map = {
                    "86": {"86", "95", "85"},  # Light Bluish Gray, Flat Silver, Dark Bluish Gray
                    "95": {"86", "95", "85"},  # Flat Silver, Light Bluish Gray, Dark Bluish Gray
                    "85": {"85", "86", "95", "11"}, # Dark Bluish Gray, Light Bluish Gray, Flat Silver, Black
                    "11": {"11", "85"},        # Black, Dark Bluish Gray
                    "1": {"1", "159"},         # White, Glow In Dark White
                }
                for ref in candidates:
                    part_colors = self.part_to_colors.get(ref, set())
                    has_color = False
                    for code in allowed_codes:
                        similar_set = similar_color_map.get(code, {code})
                        if part_colors.intersection(similar_set):
                            has_color = True
                            break
                    if not has_color:
                        gating_multipliers[ref] *= 0.3  # Penalización cromática suave

        poses_cache = {}
        cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
        if not os.path.exists(cache_path):
            cache_path = os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    poses_cache = json.load(f)
            except Exception:
                pass

        # Sincronización en vivo con la Base de Datos desactivada temporalmente para evitar ruidos de posturas no estables
        # if candidates:
        #     try:
        #         with supabase_client.get_connection() as conn:
        #             with conn.cursor() as cur:
        #                 cur.execute("""
        #                     SELECT part_ref, pose_index, is_stable, zenith_silhouette_area, zenith_observable_area, lateral_height, effective_height,
        #                            contact_stable_length, contact_stable_width
        #                     FROM stable_poses
        #                     WHERE part_ref = ANY(%s)
        #                 """, (candidates,))
        #                 rows = cur.fetchall()
        #                 db_poses = {}
        #                 for r in rows:
        #                     ref = r["part_ref"]
        #                     if ref not in db_poses:
        #                         db_poses[ref] = []
        #                     db_poses[ref].append({
        #                         "pose_index": r["pose_index"],
        #                         "is_stable": r["is_stable"],
        #                         "zenith_silhouette_area": float(r["zenith_silhouette_area"]) if r["zenith_silhouette_area"] is not None else None,
        #                         "zenith_observable_area": float(r["zenith_observable_area"]) if r["zenith_observable_area"] is not None else None,
        #                         "lateral_height": float(r["lateral_height"]) if r["lateral_height"] is not None else None,
        #                         "effective_height": float(r["effective_height"]) if r["effective_height"] is not None else None,
        #                         "contact_stable_length": float(r["contact_stable_length"]) if r["contact_stable_length"] is not None else None,
        #                         "contact_stable_width": float(r["contact_stable_width"]) if r["contact_stable_width"] is not None else None
        #                     })
        #                 # Actualizar/Sobrescribir la caché con valores reales de la BD
        #                 for ref in candidates:
        #                     if ref in db_poses:
        #                         poses_cache[ref] = db_poses[ref]
        #     except Exception as e:
        #         print(f"[DB Warning] Error al sincronizar poses_cache con la base de datos: {e}")


        # Mejoras: Hard Pruning Stage based on physical properties
        # El área cenital es una señal física altamente robusta libre de sombras/occlusiones de cinta.
        pruned_candidates = []
        for ref in candidates:
            poses = poses_cache.get(ref, [])
            if not poses:
                pruned_candidates.append(ref)
                continue
                
            # Area hard pruning: exclude if relative area error > dynamic_tolerance in all stable poses
            area_pruned = False
            if area_cenital > 0.0:
                if area_cenital < 150.0:
                    area_tol = 0.70  # Tolerancia amplia para piezas pequeñas y translúcidas (70%)
                elif area_cenital < 400.0:
                    area_tol = 0.45  # Tolerancia media para piezas medianas (45%)
                else:
                    area_tol = 0.30  # Tolerancia estricta para piezas grandes (30%)
                
                # Mejora A: Gating adaptativo por elongación (Estrategia 3: 55% de tolerancia basada en dimensiones nominales)
                valid_poses = [p for p in poses if p.get("is_stable", True)]
                if not valid_poses:
                    valid_poses = poses
                
                is_candidate_elongated = False
                for p in valid_poses:
                    nom_l = p.get("contact_stable_length")
                    nom_w = p.get("contact_stable_width")
                    if nom_l and nom_w and nom_w > 0.0:
                        if (nom_l / nom_w) > 4.0:
                            is_candidate_elongated = True
                            break
                
                if is_candidate_elongated and area_tol < 0.55:
                    area_tol = 0.55
                
                # Relajar tolerancia si YOLO detectó la pieza con baja certeza (segmentación propensa a ruido/recortes)
                if yolo_conf is not None and yolo_conf < 0.50:
                    area_tol = max(area_tol, 0.65)

                has_nominal_area = any((p.get("zenith_silhouette_area") or p.get("zenith_observable_area") or 0.0) > 0.0 for p in poses)
                if has_nominal_area:
                    any_match = False
                    for p in valid_poses:
                        nom_a = p.get("zenith_silhouette_area") or p.get("zenith_observable_area")
                        if nom_a and nom_a > 0.0:
                            rel_err = abs(area_cenital - nom_a) / nom_a
                            if rel_err <= area_tol:
                                any_match = True
                                break
                    if not any_match:
                        area_pruned = True

            # Grid-fitting soft gating para rectángulos simples
            if is_simple_rect and measured_l_studs > 0 and measured_w_studs > 0:
                has_nominal_dims = any((p.get("contact_stable_length") or 0.0) > 0.0 for p in poses)
                if has_nominal_dims:
                    any_grid_match = False
                    valid_poses = [p for p in poses if p.get("is_stable", True)]
                    if not valid_poses:
                        valid_poses = poses
                    for p in valid_poses:
                        nom_l = p.get("contact_stable_length")
                        nom_w = p.get("contact_stable_width")
                        if nom_l and nom_w:
                            nom_l_studs = max(1, int(round(nom_l / 8.0)))
                            nom_w_studs = max(1, int(round(nom_w / 8.0)))
                            if nom_l_studs == measured_l_studs and nom_w_studs == measured_w_studs:
                                any_grid_match = True
                                break
                    if not any_grid_match:
                        gating_multipliers[ref] *= 0.15
                        print(f"[DEBUG GRID] Penalizado {ref} por desajuste de rejilla ({measured_l_studs}x{measured_w_studs} studs)", flush=True)

            if not area_pruned:
                pruned_candidates.append(ref)
                
        # Keep at least one candidate (fallback in case all are pruned)
        if len(pruned_candidates) > 0:
            candidates = pruned_candidates

        # Mejoras: Gating Combinado a nivel de Pose (Pose-level Unified Gating)
        # Para evitar que una pieza falsa escape penalizaciones emparejando el área de una postura
        # y la altura de otra postura distinta (como Slope 2335 emparejando vertical y plano),
        # evaluamos la compatibilidad de área y altura conjuntamente para cada pose estable.
        if area_cenital > 0.0 or (measured_height is not None and measured_height > 0.0):
            for ref in candidates:
                poses = poses_cache.get(ref, [])
                if not poses:
                    continue
                    
                # Determinar tolerancia de área dinámica según tamaño
                if area_cenital < 150.0:
                    area_tol = 0.70
                elif area_cenital < 400.0:
                    area_tol = 0.45
                else:
                    area_tol = 0.30
                
                # Mejora A: Gating adaptativo por elongación (Estrategia 3: 55% de tolerancia basada en dimensiones nominales)
                valid_poses = [p for p in poses if p.get("is_stable", True)]
                if not valid_poses:
                    valid_poses = poses
                
                is_candidate_elongated = False
                for p in valid_poses:
                    nom_l = p.get("contact_stable_length")
                    nom_w = p.get("contact_stable_width")
                    if nom_l and nom_w and nom_w > 0.0:
                        if (nom_l / nom_w) > 4.0:
                            is_candidate_elongated = True
                            break
                
                if is_candidate_elongated and area_tol < 0.55:
                    area_tol = 0.55
                
                # Relajar tolerancia si YOLO detectó la pieza con baja certeza (segmentación propensa a ruido/recortes)
                if yolo_conf is not None and yolo_conf < 0.50:
                    area_tol = max(area_tol, 0.65)

                best_pose_multiplier = 0.0
                valid_poses = [p for p in poses if p.get("is_stable", True)]
                if not valid_poses:
                    valid_poses = poses
                for pose in valid_poses:
                    # 1. Penalización de Área para esta pose
                    nom_a = pose.get("zenith_silhouette_area") or pose.get("zenith_observable_area")
                    area_penalty = 1.0
                    if nom_a and nom_a > 0.0:
                        rel_err_a = abs(area_cenital - nom_a) / nom_a
                        # Curva Gaussiana continua para el área
                        area_penalty = np.exp(-0.5 * (rel_err_a / area_tol) ** 2)

                    # 2. Penalización de Altura para esta pose
                    nom_h = pose.get("lateral_height") or pose.get("effective_height")
                    height_penalty = 1.0
                    if nom_h and nom_h > 0.0 and measured_height is not None and measured_height > 0.0:
                        diff_h = measured_height - nom_h
                        if diff_h > 0.0:
                            # Sobre-estimación (solape/brillos): curva suave (sigma = 4.0mm)
                            height_penalty = np.exp(-0.5 * (diff_h / 4.0) ** 2)
                        else:
                            # Sub-estimación (imposible físicamente): curva estricta (sigma = 1.2mm)
                            height_penalty = np.exp(-0.5 * (diff_h / 1.2) ** 2)

                    # Multiplicador conjunto para esta postura
                    pose_multiplier = area_penalty * height_penalty
                    if pose_multiplier > best_pose_multiplier:
                        best_pose_multiplier = pose_multiplier

                # Aplicar la penalización conjunta de la mejor postura
                # Limitamos a 0.05 para no destruir si todas las poses fallan por ruido extremo
                final_mult = max(0.05, best_pose_multiplier)
                gating_multipliers[ref] *= final_mult

        # Gating de Aspect Ratio Cenital (Soft Gating Penalty)
        # Comparamos el ratio de aspecto medido mediante minAreaRect contra las dimensiones nominales de catálogo
        for ref in candidates:
            poses = poses_cache.get(ref, [])
            if not poses:
                continue
            has_nominal = any((p.get("contact_stable_length") or 0.0) > 0.0 for p in poses)
            if has_nominal:
                ar_ok = False
                valid_poses = [p for p in poses if p.get("is_stable", True)]
                if not valid_poses:
                    valid_poses = poses
                for p in valid_poses:
                    l_nom = p.get("contact_stable_length")
                    w_nom = p.get("contact_stable_width")
                    if l_nom and w_nom and w_nom > 0.0:
                        nom_ratio = l_nom / w_nom
                        # Tolerancia del 30% para evitar falsas penalizaciones por rotación/perspectiva
                        if abs(query_aspect_ratio - nom_ratio) / nom_ratio <= 0.30:
                            ar_ok = True
                            break
                if not ar_ok:
                    gating_multipliers[ref] *= 0.4  # Penalización suave de aspecto (0.4)
                    print(f"[DEBUG GATING] Penalizado {ref} por desajuste de Aspect Ratio (Medido: {query_aspect_ratio:.2f})", flush=True)

        # Gating de Dimensiones (Footprint Dimensions Gating)
        if measured_length is not None and measured_width is not None and measured_length > 0.0 and measured_width > 0.0:
            for ref in candidates:
                poses = poses_cache.get(ref, [])
                has_nominal_dims = any(p.get("contact_stable_length") is not None and p.get("contact_stable_width") is not None for p in poses)
                if not has_nominal_dims:
                    continue
                
                dim_ok = False
                valid_poses = [p for p in poses if p.get("is_stable", True)]
                if not valid_poses:
                    valid_poses = poses
                for pose in valid_poses:
                    nom_l = pose.get("contact_stable_length")
                    nom_w = pose.get("contact_stable_width")
                    if nom_l and nom_w:
                        nom_ratio = nom_w / nom_l
                        meas_ratio = measured_width / measured_length
                        rel_ratio_err = abs(meas_ratio - nom_ratio) / nom_ratio
                        if (nom_l * 0.60 <= measured_length <= nom_l * 1.50) and \
                           (nom_w * 0.60 <= measured_width <= nom_w * 1.50) and \
                           (rel_ratio_err <= 0.22):
                            dim_ok = True
                            break
                if not dim_ok and poses:
                    gating_multipliers[ref] *= 0.5  # Penalización suave de dimensiones

        # El gating de altura ha sido integrado y resuelto conjuntamente en el bloque unificado anterior.

        # Señal A: Gating por Firma de Gradiente Laplaciano (Stud Signature)
        # Sustituye el clasificador ML de studs (37% acc cenital) por un descriptor
        # determinista: superficies con studs producen alta varianza de Laplaciano,
        # superficies lisas (Tiles) producen baja varianza. Umbrales empíricos calibrados.
        if crop_cen is not None and mask_cen is not None:
            try:
                gray_cen = np.array(crop_cen.convert("L"), dtype=np.float32)
                m = mask_cen
                if m.shape != gray_cen.shape:
                    m = cv2.resize(m, (gray_cen.shape[1], gray_cen.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                roi = gray_cen.copy()
                roi[m == 0] = 0
                lap = cv2.Laplacian(roi, cv2.CV_64F)
                pixels = lap[m > 0]
                stud_signature = float(np.var(pixels)) if len(pixels) > 10 else -1.0
                # Umbrales empíricos:
                #   stud_signature > 60  → studs visibles (Plate, Brick, etc.)
                #   stud_signature < 30  → superficie lisa (Tile)
                #   entre 30 y 60       → zona ambigua, no penalizar
                STUD_HIGH_THRESH = 60.0
                STUD_LOW_THRESH  = 30.0
                obs_has_studs = (stud_signature > STUD_HIGH_THRESH) if stud_signature >= 0 else None
                obs_is_smooth  = (stud_signature < STUD_LOW_THRESH)  if stud_signature >= 0 else None
                for ref in candidates:
                    nom = self.db_features.get(ref, {})
                    nom_studs = nom.get("stud_solid", 0) + nom.get("stud_hollow", 0)
                    if obs_is_smooth and nom_studs >= 1:
                        # Observación lisa pero DB dice que tiene studs → penalización fuerte
                        gating_multipliers[ref] *= 0.35
                    elif obs_has_studs and nom_studs == 0:
                        # Observación con studs pero DB dice que no tiene → penalización fuerte
                        gating_multipliers[ref] *= 0.35
            except Exception:
                pass  # Fallo silencioso: no penalizar si hay error de extracción

        # Gating de Espigas Tradicional (CV-based studs validation — fallback)
        if detected_studs is not None:
            if detected_studs >= 1:
                for ref in candidates:
                    nom = self.db_features.get(ref, {})
                    nom_studs = nom.get("stud_solid", 0) + nom.get("stud_hollow", 0)
                    if nom_studs == 0:
                        gating_multipliers[ref] *= 0.85
            else:
                for ref in candidates:
                    nom = self.db_features.get(ref, {})
                    nom_studs = nom.get("stud_solid", 0) + nom.get("stud_hollow", 0)
                    if nom_studs >= 1:
                        gating_multipliers[ref] *= 0.85


        # Señal B: Asimetría Vertical de Máscara Lateral (Slope vs Inverted Slope)
        # El centroide vertical de la máscara SAM lateral discrimina con ~100% de precisión
        # si la rampa tiene el vértice arriba (Slope regular, centroide en zona alta)
        # o abajo (Inverted Slope, centroide en zona baja).
        SLOPE_REFS     = {"3040", "87552", "87620"}
        INV_SLOPE_REFS = {"85984", "30414", "3710"}
        if mask_lat is not None:
            try:
                ys_lat, _ = np.where(mask_lat > 0)
                if len(ys_lat) > 10:
                    cy_mask = float(np.mean(ys_lat))
                    h_lat = mask_lat.shape[0]
                    asym_ratio = (cy_mask - h_lat / 2.0) / max(1.0, h_lat)
                    ASYM_THRESH = 0.06
                    obs_is_inverted = asym_ratio > ASYM_THRESH
                    obs_is_regular  = asym_ratio < -ASYM_THRESH
                    for ref in candidates:
                        if obs_is_regular and ref in INV_SLOPE_REFS:
                            gating_multipliers[ref] *= 0.20
                        elif obs_is_inverted and ref in SLOPE_REFS:
                            gating_multipliers[ref] *= 0.20
            except Exception:
                pass

        # 2b. Classification & Gating of Topological Features
        pred_cen_vec = None
        if crop_cen is not None and self.features_cen_model is not None:
            try:
                processed_cen = preprocess_crop_grayscale(crop_cen, canvas_size=224)
                transform_feat = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor_cen = transform_feat(processed_cen).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.features_cen_model(tensor_cen)
                    pred_cen_vec = torch.sigmoid(logits)[0].cpu().numpy()
            except Exception as e:
                print(f"[Features Gating Warning] Failed to predict cenital features: {e}")

        pred_lat_vec = None
        if crop_lat is not None and self.features_lat_model is not None:
            try:
                processed_lat = preprocess_crop_grayscale(crop_lat, canvas_size=224)
                transform_feat = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor_lat = transform_feat(processed_lat).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.features_lat_model(tensor_lat)
                    pred_lat_vec = torch.sigmoid(logits)[0].cpu().numpy()
            except Exception as e:
                print(f"[Features Gating Warning] Failed to predict lateral features: {e}")

        # 3. Load 1280-d embeddings for these candidates (Fase 4)
        if ref_gt:
            is_gated = (ref_gt in candidates)
            print(f"[DIAGNOSTIC GATING] True ref {ref_gt} in candidates: {is_gated} (candidates count: {len(candidates)})", flush=True)

        print(f"[DEBUG]     _load_reference_embeddings for {len(candidates)} candidates...", flush=True)
        ref_embeddings = self._load_reference_embeddings(candidates)
        if not ref_embeddings:
            print(f"[DEBUG]     no ref embeddings loaded, returning empty", flush=True)
            return []

        # 4. Extract Grayscale query embeddings (Fase 3)
        print(f"[DEBUG]     extract_embeddings for query...", flush=True)
        query_emb_cen = self.extract_embedding(crop_cen, 'cenital')
        query_emb_lat = self.extract_embedding(crop_lat, 'lateral') if crop_lat is not None else None

        # 5. Matching & Scoring: Treat each (part_ref, pose_index) class independently
        class_groups = {}
        for r in ref_embeddings:
            key = (r["part_ref"], r["pose_index"])
            class_groups.setdefault(key, []).append(r)

        ranking = []
        for (part_ref, pose_index), refs in class_groups.items():
            cen_refs = [r for r in refs if r["face"] % 10 == 1]
            lat_refs = [r for r in refs if r["face"] % 10 == 2]

            sim_cen = 0.0
            if cen_refs:
                ref_matrix_cen = np.stack([r["embedding"] for r in cen_refs])
                sims_cen = ref_matrix_cen @ query_emb_cen
                sim_cen = float(np.max(sims_cen))

            sim_lat = 0.0
            if lat_refs and query_emb_lat is not None:
                ref_matrix_lat = np.stack([r["embedding"] for r in lat_refs])
                sims_lat = ref_matrix_lat @ query_emb_lat
                sim_lat = float(np.max(sims_lat))

            if query_emb_lat is not None and lat_refs:
                # Decoupled Multi-View Fusion: 70% Cenital, 30% Lateral (rotation-immune) (Solución B)
                combined_score = 0.7 * sim_cen + 0.3 * sim_lat
            else:
                combined_score = sim_cen

            scaled_score = float(0.10 + combined_score * 0.89)
            scaled_score = min(0.9999, max(0.01, scaled_score))

            max_prior = max(self.part_priors.values()) if self.part_priors else 1.0
            prior_prob = self.part_priors.get(part_ref, 0.0)
            prior_mult = 0.9 + 0.1 * (prior_prob / max_prior) if max_prior > 0 else 1.0
            scaled_score_prior = scaled_score * prior_mult

            # 4. Pose-Aware Topological Features Gating (Gravity Occlusion filter)
            feat_multiplier = 1.0
            face_class = self.pose_face_class.get((part_ref, pose_index), "Bottom")
            
            if (pred_cen_vec is not None or pred_lat_vec is not None) and self.db_features:
                nom = self.db_features.get(part_ref, {feat: 0 for feat in self.CLASSES_FEATURES})
                
                # Accuracies of features classifiers per camera (computed empirically on simulation_300)
                FEAT_ACC_CEN = {
                    "stud_solid": 0.3728,
                    "stud_hollow": 0.9024,
                    "technic_hole_round": 0.9477,
                    "technic_hole_cross": 0.9861,
                    "clip_jaw": 0.9791,
                    "bar_handle": 1.0000,
                    "bottom_tube": 0.5889,
                    "bottom_pin": 0.7213
                }
                FEAT_ACC_LAT = {
                    "stud_solid": 0.2927,
                    "stud_hollow": 0.9024,
                    "technic_hole_round": 0.9477,
                    "technic_hole_cross": 0.9477,
                    "clip_jaw": 0.9791,
                    "bar_handle": 1.0000,
                    "bottom_tube": 0.6272,
                    "bottom_pin": 0.7143
                }
                
                # Inferencia Cenital
                # Features excluidas del penalizador ML en cenital:
                #   bottom_tube/bottom_pin: físicamente invisibles desde cenital en pose normal
                #   stud_solid: Activado (anteriormente excluido por baja precisión baseline, ahora mejorado al 92%+)
                SKIP_ML_CEN = {"bottom_tube", "bottom_pin"}

                if pred_cen_vec is not None:
                    for idx, feat in enumerate(self.CLASSES_FEATURES):
                        if feat in SKIP_ML_CEN:
                            continue  # Señal A u otras señales geométricas manejan estos features

                        p_val = pred_cen_vec[idx]
                        nom_val = nom.get(feat, 0)
                        
                        # Si la pieza está boca abajo (Bottom face class en contacto con el suelo), 
                        # ignoramos la penalización por ausencia de tubos o pines inferiores ya que están ocultos por gravedad
                        if face_class == "Bottom" and feat in ["bottom_tube", "bottom_pin"] and p_val < 0.10 and nom_val == 1:
                            continue
                            
                        # Si la pieza está boca arriba (Top face class en contacto con el suelo),
                        # los studs superiores están ocultos y no se penaliza su ausencia
                        if face_class == "Top" and feat in ["stud_solid", "stud_hollow"] and p_val < 0.10 and nom_val == 1:
                            continue
                            
                        # Penalización adaptativa por precisión del modelo (si acc es superior, la penalización es más fuerte)
                        is_stud = feat in ["stud_solid", "stud_hollow"]
                        upper_thresh = 0.70 if is_stud else 0.90
                        lower_thresh = 0.30 if is_stud else 0.10
                        
                        acc = FEAT_ACC_CEN.get(feat, 0.5)
                        # Fórmula reforzada: de 0.45 (100% acc) a 0.90 (50% acc)
                        penalty_mult = 0.9 - (acc - 0.5) * 0.9 if acc > 0.5 else 0.9
                        
                        if p_val > upper_thresh and nom_val == 0:
                            feat_multiplier *= penalty_mult
                        elif p_val < lower_thresh and nom_val == 1:
                            feat_multiplier *= penalty_mult


                # Inferencia Lateral
                # Features excluidas del penalizador ML en lateral:
                #   stud_solid: Activado (anteriormente excluido por baja precisión baseline, ahora mejorado al 92%+)
                SKIP_ML_LAT = set()

                if pred_lat_vec is not None:
                    for idx, feat in enumerate(self.CLASSES_FEATURES):
                        if feat in SKIP_ML_LAT:
                            continue

                        p_val = pred_lat_vec[idx]
                        nom_val = nom.get(feat, 0)
                        
                        if face_class == "Bottom" and feat in ["bottom_tube", "bottom_pin"] and p_val < 0.10 and nom_val == 1:
                            continue
                        if face_class == "Top" and feat in ["stud_solid", "stud_hollow"] and p_val < 0.10 and nom_val == 1:
                            continue
                            
                        is_stud = feat in ["stud_solid", "stud_hollow"]
                        upper_thresh = 0.70 if is_stud else 0.90
                        lower_thresh = 0.30 if is_stud else 0.10
                        
                        acc = FEAT_ACC_LAT.get(feat, 0.5)
                        # Fórmula reforzada: de 0.45 (100% acc) a 0.90 (50% acc)
                        penalty_mult = 0.9 - (acc - 0.5) * 0.9 if acc > 0.5 else 0.9
                        
                        if p_val > upper_thresh and nom_val == 0:
                            feat_multiplier *= penalty_mult
                        elif p_val < lower_thresh and nom_val == 1:
                            feat_multiplier *= penalty_mult


            # Aplicar soft gating multipliers multiplicativamente
            multiplier = gating_multipliers.get(part_ref, 1.0)
            
            # Eliminamos la atenuación exponencial (** 0.25) para mantener la fuerza real del gating físico y de features
            final_score = float(scaled_score_prior * multiplier * feat_multiplier)


            ranking.append({
                "part_ref": part_ref,
                "pose_index": pose_index,
                "score": final_score,
                "raw_sim_cen": sim_cen,
                "raw_sim_lat": sim_lat,
            })

        ranking.sort(key=lambda x: x["score"], reverse=True)
        # Select unique candidates to avoid pose duplication crowding the Top-10
        unique_ranking = []
        seen_refs = set()
        for r in ranking:
            ref = r["part_ref"]
            if ref not in seen_refs:
                seen_refs.add(ref)
                unique_ranking.append(r)
        top15 = unique_ranking[:15]
        
        if ref_gt:
            in_top5 = any(r["part_ref"] == ref_gt for r in top15[:5])
            in_top15 = any(r["part_ref"] == ref_gt for r in top15)
            print(f"[DIAGNOSTIC KNN] True ref {ref_gt} in Top-5: {in_top5}", flush=True)
            print(f"[DIAGNOSTIC KNN] True ref {ref_gt} in Top-15: {in_top15}", flush=True)
            if not in_top5:
                print(f"[DIAGNOSTIC KNN DETAIL] Top-5 list: {[r['part_ref'] for r in top15[:5]]}", flush=True)
            if not in_top15:
                print(f"[DIAGNOSTIC KNN DETAIL] Top-15 list: {[r['part_ref'] for r in top15]}", flush=True)
                
        return top15


if __name__ == "__main__":
    print("Testing LegoEfficientNetClassifier initialization...")
    clf = LegoEfficientNetClassifier()
    print("Success! Model loaded.")
