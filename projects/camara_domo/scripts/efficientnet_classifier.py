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


class LegoEfficientNetClassifier:
    """Neuro-symbolic classifier for Lego piece geometry using EfficientNetV2-B0 and database filters."""

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()
        self.transform = self._build_transform()
        self.ref_metadata = self._load_ref_metadata()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.emb_dim = self.model(dummy).shape[-1]

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

        # Caché de características topológicas desde la base de datos
        self.db_features = {}
        try:
            with supabase_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ldraw_id, topological_features FROM lego_classes;")
                    rows = cur.fetchall()
                    for row in rows:
                        ldraw_id = row["ldraw_id"]
                        if not ldraw_id:
                            continue
                        db_feat = row["topological_features"]
                        if isinstance(db_feat, str):
                            try:
                                db_feat = json.loads(db_feat)
                            except Exception:
                                db_feat = {}
                        if not isinstance(db_feat, dict):
                            db_feat = {}
                        self.db_features[ldraw_id] = {feat: (1 if db_feat.get(feat, 0) > 0 else 0) for feat in self.CLASSES_FEATURES}
            print(f"[Features Classifier] Cached topological features for {len(self.db_features)} parts.")
        except Exception as e:
            print(f"[Features Classifier Warning] Could not load features from DB: {e}")

    def _load_model(self) -> nn.Module:
        """Loads pretrained EfficientNetV2-B0 and strips the classification head."""
        try:
            import timm
            model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=0)
            print(f"[EfficientNet Classifier] Loaded timm efficientnetv2_rw_s on {self.device}")
        except Exception:
            import torchvision.models as models
            model = models.efficientnet_v2_b0(weights=models.EfficientNet_V2_B0_Weights.DEFAULT)
            model.classifier = nn.Identity()
            print(f"[EfficientNet Classifier] Loaded torchvision efficientnet_v2_b0 on {self.device}")
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
        """Fase 1: Filter LDraw catalog using stable pose zenith surface."""
        min_area = area_cenital * (1.0 - tolerance)
        max_area = area_cenital * (1.0 + tolerance)
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
                    candidates = [row["part_ref"] for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Warning] Could not query candidates from DB: {e}. Using local cache fallback.")

        # Local fallback for Area-only gating
        if not candidates and poses_cache:
            for ref, poses in poses_cache.items():
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
                            WHERE pe.part_ref IN %s AND sp.is_stable = TRUE
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

    def extract_embedding(self, crop: Image.Image) -> np.ndarray:
        """Converts crop to Grayscale, normalizes, runs forward pass through EfficientNetV2-B0, and L2 normalizes."""
        preproc = preprocess_crop_grayscale(crop)
        tensor = self.transform(preproc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            emb = feat[0].cpu().numpy().astype(np.float32)
        return emb

    def classify(self, 
                 crop_cen: Image.Image, 
                 mask_cen: np.ndarray, 
                 crop_lat: Optional[Image.Image] = None, 
                 mask_lat: Optional[np.ndarray] = None,
                 area_cenital: float = 0.0) -> List[Dict[str, Any]]:
        """
        Runs the full classification pipeline: deterministic filter, color decoupling,
        feature extraction, and Restricted K-NN search.
        """
        # 1. CIELAB Color Median extraction (Fase 2)
        print(f"[DEBUG]     extract_median_lab...", flush=True)
        color_lab_cen = self.extract_median_lab(crop_cen, mask_cen)

        # 2. Deterministic Filter (Fase 1)
        print(f"[DEBUG]     get_deterministic_candidates...", flush=True)
        candidates = self.get_deterministic_candidates(area_cenital, color_lab_cen)
        if not candidates:
            # Fallback to all known sets refs if nothing matched
            candidates = list(set(ref for s in REAL_SETS.values() for p in s.get("parts", []) for ref in [p["ref"]]))

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

        if (pred_cen_vec is not None or pred_lat_vec is not None) and self.db_features:
            filtered = []
            for ref in candidates:
                nom = self.db_features.get(ref, {feat: 0 for feat in self.CLASSES_FEATURES})
                compatible = True
                
                # Check Cenital Gating
                if pred_cen_vec is not None:
                    for idx, feat in enumerate(self.CLASSES_FEATURES):
                        p_val = pred_cen_vec[idx]
                        nom_val = nom.get(feat, 0)
                        # Gating thresholds: present > 0.90, absent < 0.10
                        if p_val > 0.90 and nom_val == 0:
                            compatible = False
                            break
                        if p_val < 0.10 and nom_val == 1:
                            compatible = False
                            break
                            
                # Check Lateral Gating
                if compatible and pred_lat_vec is not None:
                    for idx, feat in enumerate(self.CLASSES_FEATURES):
                        p_val = pred_lat_vec[idx]
                        nom_val = nom.get(feat, 0)
                        if p_val > 0.90 and nom_val == 0:
                            compatible = False
                            break
                        if p_val < 0.10 and nom_val == 1:
                            compatible = False
                            break
                            
                if compatible:
                    filtered.append(ref)
            
            if filtered:
                print(f"[DEBUG]     Features Gating: reduced candidates from {len(candidates)} to {len(filtered)}", flush=True)
                candidates = filtered

        # 3. Load 1280-d embeddings for these candidates (Fase 4)
        print(f"[DEBUG]     _load_reference_embeddings for {len(candidates)} candidates...", flush=True)
        ref_embeddings = self._load_reference_embeddings(candidates)
        if not ref_embeddings:
            print(f"[DEBUG]     no ref embeddings loaded, returning empty", flush=True)
            return []

        # 4. Extract Grayscale query embeddings (Fase 3)
        print(f"[DEBUG]     extract_embeddings for query...", flush=True)
        query_emb_cen = self.extract_embedding(crop_cen)
        query_emb_lat = self.extract_embedding(crop_lat) if crop_lat is not None else None

        # 5. Matching & Scoring: Treat each (part_ref, pose_index) class independently
        class_groups = {}
        for r in ref_embeddings:
            key = (r["part_ref"], r["pose_index"])
            class_groups.setdefault(key, []).append(r)

        ranking = []
        for (part_ref, pose_index), refs in class_groups.items():
            # Cenital similarities
            cen_refs = [r for r in refs if r["face"] % 10 == 1]
            # Lateral similarities
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

            # Combine Cenital and Frontal/Lateral scores (0.7 + 0.3)
            if query_emb_lat is not None and lat_refs:
                combined_score = 0.7 * sim_cen + 0.3 * sim_lat
            else:
                combined_score = sim_cen

            # Normalization mapping to fit GUI confidence thresholds (0.0 to 1.0 range)
            scaled_score = float(0.10 + combined_score * 0.89)
            scaled_score = min(0.9999, max(0.01, scaled_score))

            ranking.append({
                "part_ref": part_ref,
                "pose_index": pose_index,
                "score": scaled_score,
                "raw_sim_cen": sim_cen,
                "raw_sim_lat": sim_lat,
            })

        ranking.sort(key=lambda x: x["score"], reverse=True)
        return ranking[:5]


if __name__ == "__main__":
    print("Testing LegoEfficientNetClassifier initialization...")
    clf = LegoEfficientNetClassifier()
    print("Success! Model loaded.")
