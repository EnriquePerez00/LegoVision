# -*- coding: utf-8 -*-
"""
camara_domo/scripts/contour_matcher.py
======================================
Módulo de alineación geométrica de contornos 2D/3D (Gráficos Inversos).
Permite evaluar y desempatar candidatos de piezas LEGO basándose en el
solapamiento (IoU) de la proyección 3D del mesh LDraw nominal sobre la máscara real de SAM.
"""

import os
import sys
import json
import math
import numpy as np
import cv2

# Rutas del proyecto
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_legovic_root = os.path.dirname(os.path.dirname(_project_root))

# Insertar paths para importar dependencias de LegoVision
if _legovic_root not in sys.path:
    sys.path.insert(0, _legovic_root)
if os.path.join(_legovic_root, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_legovic_root, "scripts"))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from ldraw_mesh_parser import get_triangles
except ImportError:
    print("[ContourMatcher Warning] No se pudo importar ldraw_mesh_parser.get_triangles")
    def get_triangles(part_ref):
        return np.empty((0, 3, 3), dtype=np.float64)

# Matrices de proyección de cámara desde el observador de keypoints
try:
    from _kpts_observer import _P_CEN, _P_LAT
except ImportError:
    print("[ContourMatcher Warning] No se pudo importar _P_CEN/_P_LAT desde _kpts_observer.")
    _P_CEN, _P_LAT = None, None

class ContourMatcher:
    def __init__(self, cache_path=None):
        """
        Inicializa el alineador de contornos cargando el caché de poses estables.
        """
        self.cache_path = cache_path or os.path.join(_project_root, "data", "stable_poses_cache.json")
        self.stable_poses_cache = self._load_cache()
        self.mesh_cache = {}  # Caché en memoria para evitar re-lecturas de LDraw

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ContourMatcher Warning] Error leyendo stable_poses_cache: {e}")
        else:
            print(f"[ContourMatcher Warning] No se encontró el archivo de caché en: {self.cache_path}")
        return {}

    def get_pose_metadata(self, part_ref, pose_index):
        """
        Recupera los metadatos de la pose estable de la pieza.
        """
        # Intentar del caché local
        poses = self.stable_poses_cache.get(part_ref, [])
        for pose in poses:
            if pose.get("pose_index") == pose_index:
                return pose
        
        # Fallback de base de datos si está disponible
        try:
            from core.db import supabase_client
            db_poses = supabase_client.get_stable_poses(part_ref)
            for pose in db_poses:
                if pose.get("pose_index") == pose_index:
                    return pose
        except Exception:
            pass
        return None

    def get_oriented_mesh(self, part_ref, pose_metadata):
        """
        Carga, centra, orienta según la pose estable y simplifica la malla LDraw de la pieza.
        Retorna una nube de puntos simplificada (vértices del 3D Convex Hull) para acelerar la proyección.
        """
        cache_key = (part_ref, pose_metadata["pose_index"])
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]

        # 1. Cargar triángulos nominales LDraw
        tris = get_triangles(part_ref)
        if len(tris) == 0:
            return None

        # 2. Escalar LDU a milímetros (1 LDU = 0.4 mm)
        vertices = tris.reshape(-1, 3) * 0.4

        # 3. Centrar la malla en su caja delimitadora
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) * 0.5
        vertices = vertices - bbox_center

        # 4. Rotar usando el cuaternión de la pose estable (orientation_quat: [w, x, y, z])
        quat = pose_metadata.get("orientation_quat")
        if not quat:
            quat = [1.0, 0.0, 0.0, 0.0]

        w, x, y, z = quat
        R = np.array([
            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
        ])
        vertices_rot = (R @ vertices.T).T

        # 5. Mover Z para que la pieza descanse exactamente sobre la cinta transportadora (Z_min = 0)
        z_min = vertices_rot[:, 2].min()
        vertices_rot[:, 2] = vertices_rot[:, 2] - z_min

        # 6. Extraer el Convex Hull 3D para quedarnos con los vértices de contorno mínimo
        # Esto reduce el número de puntos a proyectar de miles a unos 20-40, acelerando el proceso
        try:
            from scipy.spatial import ConvexHull
            hull_3d = ConvexHull(vertices_rot)
            vertices_hull = vertices_rot[hull_3d.vertices]
        except Exception:
            # Fallback seguro si falla Scipy
            if len(vertices_rot) > 150:
                idx = np.linspace(0, len(vertices_rot) - 1, 150, dtype=np.int32)
                vertices_hull = vertices_rot[idx]
            else:
                vertices_hull = vertices_rot

        self.mesh_cache[cache_key] = vertices_hull
        return vertices_hull

    def match_contour(self, 
                      part_ref, 
                      pose_index, 
                      mask_cen, 
                      bbox_cen_norm, 
                      img_res_px_cen,
                      mask_lat=None, 
                      bbox_lat_norm=None, 
                      img_res_px_lat=2048):
        """
        Calcula el score de coincidencia de contornos buscando el ángulo de guiñada (yaw) óptimo.
        
        Parámetros:
            part_ref: Referencia de pieza LDraw (ej: "3004")
            pose_index: Índice de pose estable (0, 1, etc.)
            mask_cen: Máscara SAM cenital (binaria, HxW)
            bbox_cen_norm: Bounding box normalizado cenital [x1, y1, x2, y2]
            img_res_px_cen: Resolución de la imagen cenital (ancho/alto en px)
            mask_lat: Máscara SAM lateral (binaria, HxW) u opcional
            bbox_lat_norm: Bounding box normalizado lateral
            img_res_px_lat: Resolución de la imagen lateral
            
        Retorna:
            (best_score, best_yaw_deg): El score combinado de solapamiento óptimo (0.0 a 1.0)
            y el ángulo yaw en grados correspondiente a ese óptimo.
        """
        # 1. Obtener metadatos de la pose
        pose_meta = self.get_pose_metadata(part_ref, pose_index)
        if not pose_meta:
            return 0.0, 0.0

        # 2. Cargar y orientar el mesh simplificado
        vertices_hull = self.get_oriented_mesh(part_ref, pose_meta)
        if vertices_hull is None or len(vertices_hull) == 0:
            return 0.0, 0.0

        # 3. Estimar la posición 3D real de la pieza en la cinta a partir del centroide cenital
        x1_c, y1_c, x2_c, y2_c = bbox_cen_norm
        cx_px = (x1_c + x2_c) * 0.5 * img_res_px_cen
        cy_px = (y1_c + y2_c) * 0.5 * img_res_px_cen
        center_px_cen = img_res_px_cen * 0.5
        dx_px = cx_px - center_px_cen
        dy_px = cy_px - center_px_cen

        # Escalar la constante nominal px/mm según resolución real
        from core.utils.config_loader import cfg
        PX_PER_MM_NOMINAL = float(cfg.inference.calibration.px_per_mm_cenital)
        px_per_mm_cen = PX_PER_MM_NOMINAL * (img_res_px_cen / 2048.0)

        # Traslación física en mm: en el espacio del observador de keypoints, Y_mundo = -dy_mm
        dx_mm = dx_px / px_per_mm_cen
        dy_mm = dy_px / px_per_mm_cen
        translation = np.array([dx_mm, -dy_mm, 0.0], dtype=np.float32)

        # 4. Recortar la máscara de SAM cenital para hacer el cálculo de IoU súper rápido
        ys_c, xs_c = np.where(mask_cen > 0)
        if len(ys_c) == 0:
            return 0.0, 0.0

        x1_crop_c = max(0, int(xs_c.min() - 20))
        y1_crop_c = max(0, int(ys_c.min() - 20))
        x2_crop_c = min(mask_cen.shape[1], int(xs_c.max() + 20))
        y2_crop_c = min(mask_cen.shape[0], int(ys_c.max() + 20))
        cropped_sam_mask_cen = mask_cen[y1_crop_c:y2_crop_c, x1_crop_c:x2_crop_c]

        # Recortar la máscara lateral (si está activa)
        cropped_sam_mask_lat = None
        x1_crop_l = y1_crop_l = 0
        if mask_lat is not None:
            ys_l, xs_l = np.where(mask_lat > 0)
            if len(ys_l) > 0:
                x1_crop_l = max(0, int(xs_l.min() - 20))
                y1_crop_l = max(0, int(ys_l.min() - 20))
                x2_crop_l = min(mask_lat.shape[1], int(xs_l.max() + 20))
                y2_crop_l = min(mask_lat.shape[0], int(ys_l.max() + 20))
                cropped_sam_mask_lat = mask_lat[y1_crop_l:y2_crop_l, x1_crop_l:x2_crop_l]

        # 5. Búsqueda jerárquica de guiñada (yaw) óptima
        best_angle = 0.0
        best_score = -1.0

        # Fase Coarse: Buscar cada 20 grados (18 orientaciones)
        angles_coarse = np.linspace(0, 2*np.pi, 18, endpoint=False)
        for theta in angles_coarse:
            score, iou_c, iou_l = self._eval_angle(
                vertices_hull, translation, theta,
                cropped_sam_mask_cen, x1_crop_c, y1_crop_c, img_res_px_cen,
                cropped_sam_mask_lat, x1_crop_l, y1_crop_l, img_res_px_lat
            )
            if score > best_score:
                best_score = score
                best_angle = theta

        # Fase Fine: Buscar en pasos de 5 grados en un entorno de +-15 grados
        angles_fine = np.array([-15, -10, -5, 5, 10, 15]) * (np.pi / 180.0) + best_angle
        for theta in angles_fine:
            score, iou_c, iou_l = self._eval_angle(
                vertices_hull, translation, theta,
                cropped_sam_mask_cen, x1_crop_c, y1_crop_c, img_res_px_cen,
                cropped_sam_mask_lat, x1_crop_l, y1_crop_l, img_res_px_lat
            )
            if score > best_score:
                best_score = score
                best_angle = theta

        return float(best_score), float(best_angle * 180.0 / np.pi)

    def _eval_angle(self, vertices_hull, translation, theta,
                    cropped_sam_mask_cen, x1_crop_c, y1_crop_c, img_res_px_cen,
                    cropped_sam_mask_lat=None, x1_crop_l=0, y1_crop_l=0, img_res_px_lat=2048):
        """
        Proyecta la nube de puntos y calcula el solapamiento IoU para un ángulo específico.
        """
        # 1. Aplicar rotación yaw theta alrededor del eje vertical Z
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        vertices_rot = (R_z @ vertices_hull.T).T
        vertices_translated = vertices_rot + translation

        # 2. Proyectar en la Cámara Cenital
        from _kpts_observer import _P_CEN, _P_LAT
        if _P_CEN is None:
            return 0.0, 0.0, 0.0

        ones = np.ones((len(vertices_translated), 1), dtype=np.float32)
        pts_4d = np.hstack([vertices_translated, ones])

        # Proyección Homogénea Cenital
        pts_cen_homg = (_P_CEN @ pts_4d.T).T
        z_c = pts_cen_homg[:, 2]
        z_c[np.abs(z_c) < 1e-6] = 1e-6
        u_cen_img = (pts_cen_homg[:, 0] / z_c) * (img_res_px_cen / 640.0)
        v_cen_img = (pts_cen_homg[:, 1] / z_c) * (img_res_px_cen / 640.0)

        # Desplazar a coordenadas de recorte cenital
        u_cen_crop = u_cen_img - x1_crop_c
        v_cen_crop = v_cen_img - y1_crop_c

        # Generar máscara de proyección cenital via Convex Hull 2D
        pts_cen_crop = np.stack([u_cen_crop, v_cen_crop], axis=1).astype(np.float32)
        hull_c = cv2.convexHull(pts_cen_crop)
        proj_mask_c = np.zeros(cropped_sam_mask_cen.shape, dtype=np.uint8)
        cv2.drawContours(proj_mask_c, [hull_c.astype(np.int32)], -1, 255, -1)

        # Calcular IoU Cenital
        intersection_c = np.sum((proj_mask_c > 0) & (cropped_sam_mask_cen > 0))
        union_c = np.sum((proj_mask_c > 0) | (cropped_sam_mask_cen > 0))
        iou_c = float(intersection_c) / max(1.0, float(union_c))

        # 3. Proyectar en la Cámara Lateral (si está activa)
        iou_l = 0.0
        if cropped_sam_mask_lat is not None and _P_LAT is not None:
            # Proyección Homogénea Lateral
            pts_lat_homg = (_P_LAT @ pts_4d.T).T
            z_l = pts_lat_homg[:, 2]
            z_l[np.abs(z_l) < 1e-6] = 1e-6
            u_lat_img = (pts_lat_homg[:, 0] / z_l) * (img_res_px_lat / 640.0)
            v_lat_img = (pts_lat_homg[:, 1] / z_l) * (img_res_px_lat / 640.0)

            # Desplazar a coordenadas de recorte lateral
            u_lat_crop = u_lat_img - x1_crop_l
            v_lat_crop = v_lat_img - y1_crop_l

            # Generar máscara de proyección lateral via Convex Hull 2D
            pts_lat_crop = np.stack([u_lat_crop, v_lat_crop], axis=1).astype(np.float32)
            hull_l = cv2.convexHull(pts_lat_crop)
            proj_mask_l = np.zeros(cropped_sam_mask_lat.shape, dtype=np.uint8)
            cv2.drawContours(proj_mask_l, [hull_l.astype(np.int32)], -1, 255, -1)

            # Calcular IoU Lateral
            intersection_l = np.sum((proj_mask_l > 0) & (cropped_sam_mask_lat > 0))
            union_l = np.sum((proj_mask_l > 0) | (cropped_sam_mask_lat > 0))
            iou_l = float(intersection_l) / max(1.0, float(union_l))

            # Fusión ponderada: 70% Cenital y 30% Lateral
            score = 0.7 * iou_c + 0.3 * iou_l
        else:
            score = iou_c

        return score, iou_c, iou_l

if __name__ == "__main__":
    print("Probando ContourMatcher...")
    matcher = ContourMatcher()
    print("Matcher instanciado con éxito.")
