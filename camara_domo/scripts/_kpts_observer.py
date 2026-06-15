# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/_kpts_observer.py
=========================================================
Observador 6-DoF basado en YOLO-Pose + triangulacion 2-view.

Configuracion extraida DIRECTAMENTE de Blender via build_scene_canonical()
(script: /tmp/extract_blender_matrices.py, ejecutado 2026-06-10).
Convencion de ejes verificada empiricamente (2026-06-10):

  Blender escena canonica (scene_canonical.py):
    1 BU = 100 mm

  Camara cenital:  pos = (0, 0, 1.5 BU) = (0, 0, 150 mm), mira a -Z
  Camara lateral:  pos = (1.5, 0, 0.25 BU) = (150, 0, 25 mm), mira al origen

  Intrinsicos (focal=27mm, sensor=36mm, 640x640, sensor_fit=AUTO):
    fx = fy = 27/36 * 640 = 480.0 px
    cx = cy = 320.0 px

  Convencion imagen YOLO (y=0 arriba) vs Blender (Y positivo = arriba):
    x_mundo  =  (x_px - cx) * Z_dist / fx        (X igual en ambos)
    y_mundo  = -(y_px - cy) * Z_dist / fy         (Y NEGADO: YOLO-y y Blender-Y opuestos)

  La DLT usa la convencion OpenCV (y_cam=down), que coincide con YOLO.
  El UP_WORLD para _build_camera_matrix se ajusta para que la proyeccion
  sea consistente con la imagen YOLO (y=0 arriba del frame).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from config_loader import cfg


# ─────────────────────────────────────────────────────────────────
# Intrinsicos — calculados dinamicamente desde config.yaml (sensor 36mm)
# ─────────────────────────────────────────────────────────────────
IMG_SIZE_PX  = 640.0
PRINCIPAL_PX = 320.0
FOCAL_CEN_PX = (cfg.cameras.cenital.focal_length_mm / 36.0) * IMG_SIZE_PX
FOCAL_LAT_PX = (cfg.cameras.lateral.focal_length_mm / 36.0) * IMG_SIZE_PX

# ─────────────────────────────────────────────────────────────────
# Extrinsicos — posiciones en mm (1 BU = 10 mm)
# ─────────────────────────────────────────────────────────────────
# bu_per_mm = 0.1, por tanto 1 BU = 10 mm
C_CEN_POS = np.array(cfg.cameras.cenital.position) * 10.0
C_LAT_POS = np.array(cfg.cameras.lateral.position) * 10.0
LOOK_AT   = np.array([0.0, 0.0, 0.0])

# ─────────────────────────────────────────────────────────────────
# UP_WORLD por camara
# Blender UP_Y (+Y mundo = arriba de escena), pero YOLO tiene y=0 arriba
# → para que la proyeccion DLT sea consistente con la imagen YOLO
#   la camara cenital (que mira -Z desde arriba) usa UP = [0, -1, 0]
#   (imagen YOLO: y creciente = hacia abajo = -Y mundo)
# ─────────────────────────────────────────────────────────────────
UP_CEN = np.array([0.0, -1.0, 0.0])   # cenital: y_img↓ corresponde a -Y mundo
UP_LAT = np.array([0.0,  0.0, 1.0])   # lateral: Z is UP in the conveyor system

N_KPS = 9


def _build_camera_matrix(cam_pos: np.ndarray, up_world: np.ndarray):
    """Matriz extrinsica 3x4. Convencion OpenCV: x=right, y=down, z=forward."""
    forward = LOOK_AT - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(up_world, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ cam_pos
    return R, t


def _projection_matrix(R, t, focal_px):
    K = np.array([[focal_px, 0,        PRINCIPAL_PX],
                  [0,        focal_px, PRINCIPAL_PX],
                  [0,        0,        1            ]])
    return K @ np.hstack([R, t.reshape(3, 1)])


_R_CEN, _t_CEN = _build_camera_matrix(C_CEN_POS, UP_CEN)
_R_LAT, _t_LAT = _build_camera_matrix(C_LAT_POS, UP_LAT)
_P_CEN = _projection_matrix(_R_CEN, _t_CEN, FOCAL_CEN_PX)
_P_LAT = _projection_matrix(_R_LAT, _t_LAT, FOCAL_LAT_PX)

# Apply coordinate flips to align with Blender standard camera space
_P_CEN[1] = 640.0 * _P_CEN[2] - _P_CEN[1] # Y-flip
_P_LAT[0] = 640.0 * _P_LAT[2] - _P_LAT[0] # X-flip


def _to_pixel(kp_xy_norm, img_size=IMG_SIZE_PX):
    """(x_norm, y_norm) ∈ [0,1] → pixel. Convencion YOLO: y=0 arriba."""
    return float(kp_xy_norm[0]) * img_size, float(kp_xy_norm[1]) * img_size


def _triangulate_dlt(p_cen_px, p_lat_px):
    x1, y1 = p_cen_px
    x2, y2 = p_lat_px
    A = np.array([
        x1 * _P_CEN[2] - _P_CEN[0],
        y1 * _P_CEN[2] - _P_CEN[1],
        x2 * _P_LAT[2] - _P_LAT[0],
        y2 * _P_LAT[2] - _P_LAT[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    return Xh[:3] / Xh[3]


def triangulate_keypoints(kps_cen, kps_lat, conf_min=0.3):
    n = min(len(kps_cen), len(kps_lat), N_KPS)
    out = []
    for i in range(n):
        cx, cy, cc = kps_cen[i]
        lx, ly, lc = kps_lat[i]
        if cc < conf_min or lc < conf_min:
            out.append(None); continue
        try:
            X = _triangulate_dlt(_to_pixel((cx, cy)), _to_pixel((lx, ly)))
            if X is None or np.any(np.isnan(X)) or not (0.0 <= X[2] <= 50.0):
                out.append(None)
            else:
                out.append(tuple(float(v) for v in X))
        except Exception:
            out.append(None)
    return out, sum(1 for x in out if x is not None)


def derive_observations(kps_3d):
    valid = [p for p in kps_3d if p is not None]
    if len(valid) < 4:
        return {"n_valid": len(valid), "centroid_mm": None,
                "footprint_area_mm2": None, "lateral_height_mm": None,
                "x_mm": None, "y_mm": None, "z_mm": None}
    pts      = np.array(valid)
    centroid = pts.mean(axis=0)
    height   = max(0.0, float(pts[:, 2].max() - pts[:, 2].min()))
    bottom = pts[np.argsort(pts[:, 2])][:4]
    try:
        from scipy.spatial import ConvexHull
        fp = float(ConvexHull(bottom[:, :2]).volume)
    except Exception:
        xs, ys = bottom[:, 0], bottom[:, 1]
        fp = float((xs.max()-xs.min()) * (ys.max()-ys.min()))
    return {"n_valid": len(valid),
            "centroid_mm": list(centroid.astype(float)),
            "footprint_area_mm2": fp,
            "lateral_height_mm": height,
            "x_mm": float(centroid[0]),
            "y_mm": float(centroid[1]),
            "z_mm": float(centroid[2])}


def kpts_observer(kps_cen, kps_lat, conf_min=0.3):
    kps_3d, _ = triangulate_keypoints(kps_cen, kps_lat, conf_min=conf_min)
    obs = derive_observations(kps_3d)
    obs["kps_3d"] = [list(p) if p is not None else None for p in kps_3d]
    return obs


def extract_yolo_pose_keypoints(model, img_path_or_array, conf=0.25):
    try:
        results = model(img_path_or_array, verbose=False, conf=conf)
        if not results:
            return None
        r = results[0]
        if r.keypoints is None or r.keypoints.xyn is None or len(r.keypoints.xyn) == 0:
            return None
        confs = (r.boxes.conf.cpu().numpy()
                 if r.boxes is not None
                 else np.ones(len(r.keypoints.xyn)))
        best  = int(confs.argmax())
        xyn   = r.keypoints.xyn[best].cpu().numpy()
        kconf = (r.keypoints.conf[best].cpu().numpy()
                 if r.keypoints.conf is not None
                 else np.ones(len(xyn)) * float(confs[best]))
        return np.hstack([xyn, kconf.reshape(-1, 1)])
    except Exception:
        return None
