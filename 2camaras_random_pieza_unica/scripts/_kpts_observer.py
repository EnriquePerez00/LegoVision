# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/_kpts_observer.py
=========================================================
Observador 6-DoF basado en YOLO-Pose + triangulacion 2-view.

Para CADA muestra:
  1. YOLO-Pose detecta 9 keypoints 2D en la imagen cenital y lateral.
  2. Para cada keypoint visible en AMBAS camaras (v_cen >= confianza_min
     y v_lat >= confianza_min), triangulamos la posicion 3D mediante
     interseccion de rayos (DLT mid-point).
  3. De los keypoints 3D obtenidos derivamos:
       - centroide 3D (mean de los KPs validos).
       - footprint (mm²) = area del polígono convexo de los 4 KPs
         "bottom" proyectados al plano Z=0.
       - altura lateral (mm) = max(z_kp) - min(z_kp) (sobre los KPs validos).

CAMARAS (escena canonica):
  cenital: pos = (0, 0, 150) mm, mira a (0, 0, 0). Eje optico = -Z.
  lateral: pos = (150, 0, 25) mm, mira a (0, 0, 0). Eje optico = -X aprox.

Intrinsicos compartidos (focal_px=480 @ 640², principal point = (320,320)).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Constantes camara (escena canonica)
# ─────────────────────────────────────────────────────────────────
FOCAL_PX = 480.0
PRINCIPAL_PX = 320.0
IMG_SIZE_PX = 640.0

# Cenital: posicion mundo (mm)
C_CEN_POS = np.array([0.0, 0.0, 150.0])
# Lateral: posicion mundo (mm)
C_LAT_POS = np.array([150.0, 0.0, 25.0])
# Ambas miran al origen (0, 0, 0)
LOOK_AT = np.array([0.0, 0.0, 0.0])
UP_WORLD = np.array([0.0, 1.0, 0.0])  # heuristica para "up"

N_KPS = 9


def _build_camera_matrix(cam_pos):
    """Matriz extrinseca 3x4 (R|t) para una camara que mira a `LOOK_AT`
    con `UP_WORLD` como aprox. up. Convencion: z_cam = backward from scene
    (apuntando hacia la pieza), por eso usamos -forward."""
    forward = LOOK_AT - cam_pos
    forward /= np.linalg.norm(forward)
    # right = up x forward, normalizado
    right = np.cross(UP_WORLD, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    # R: world -> camera. Convencion OpenCV: x_cam=right, y_cam=down, z_cam=forward.
    R = np.stack([right, -up, forward], axis=0)  # 3x3
    t = -R @ cam_pos
    return R, t  # P = K @ [R | t] luego


def _projection_matrix(R, t):
    """Devuelve P (3x4) = K @ [R | t]."""
    K = np.array([[FOCAL_PX, 0, PRINCIPAL_PX],
                  [0, FOCAL_PX, PRINCIPAL_PX],
                  [0, 0, 1]])
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


_R_CEN, _t_CEN = _build_camera_matrix(C_CEN_POS)
_R_LAT, _t_LAT = _build_camera_matrix(C_LAT_POS)
_P_CEN = _projection_matrix(_R_CEN, _t_CEN)
_P_LAT = _projection_matrix(_R_LAT, _t_LAT)


def _normalize_kp_to_pixel(kp_xy_norm, img_size=IMG_SIZE_PX):
    """Convierte (x_norm[0,1], y_norm[0,1]) a pixel (x_px, y_px) con
    convencion YOLO (y down)."""
    return float(kp_xy_norm[0]) * img_size, float(kp_xy_norm[1]) * img_size


def _triangulate_dlt(p_cen_px, p_lat_px):
    """DLT triangulation de un punto desde dos vistas con matrices P_cen y P_lat.
    Devuelve (X, Y, Z) en mundo (mm)."""
    x1, y1 = p_cen_px
    x2, y2 = p_lat_px
    A = np.array([
        x1 * _P_CEN[2] - _P_CEN[0],
        y1 * _P_CEN[2] - _P_CEN[1],
        x2 * _P_LAT[2] - _P_LAT[0],
        y2 * _P_LAT[2] - _P_LAT[1],
    ])
    # Resolver A @ [X,Y,Z,1] = 0  via SVD.
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    Xw = Xh[:3] / Xh[3]
    return Xw


def triangulate_keypoints(kps_cen, kps_lat, conf_min=0.3):
    """Triangula los 9 keypoints. `kps_cen` y `kps_lat` son arrays
    (N_KPS, 3) con (x_norm, y_norm, conf).

    Devuelve:
      kps_3d: lista de (X, Y, Z) o None por cada KP.
      n_valid: numero de keypoints triangulados.
    """
    n = min(len(kps_cen), len(kps_lat), N_KPS)
    out = []
    for i in range(n):
        cx, cy, cc = kps_cen[i]
        lx, ly, lc = kps_lat[i]
        if cc < conf_min or lc < conf_min:
            out.append(None)
            continue
        try:
            p1 = _normalize_kp_to_pixel((cx, cy))
            p2 = _normalize_kp_to_pixel((lx, ly))
            X = _triangulate_dlt(p1, p2)
            if X is None or np.any(np.isnan(X)):
                out.append(None)
            else:
                out.append(tuple(float(v) for v in X))
        except Exception:
            out.append(None)
    n_valid = sum(1 for x in out if x is not None)
    return out, n_valid


def derive_observations(kps_3d):
    """Deriva (centroide, footprint_mm2, altura_lateral_mm, n_valid) de
    los keypoints 3D triangulados."""
    valid_pts = [p for p in kps_3d if p is not None]
    if len(valid_pts) < 4:
        return {
            "n_valid": len(valid_pts),
            "centroid_mm": None,
            "footprint_area_mm2": None,
            "lateral_height_mm": None,
            "x_mm": None, "y_mm": None, "z_mm": None,
        }
    pts = np.array(valid_pts)
    centroid = pts.mean(axis=0)
    z_min = float(pts[:, 2].min())
    z_max = float(pts[:, 2].max())
    height_mm = max(0.0, z_max - z_min)

    # Bottom KPs: los 4 con z mas bajo (independiente del orden canonico
    # porque la pose puede estar girada).
    sorted_by_z = pts[np.argsort(pts[:, 2])]
    bottom = sorted_by_z[:4]
    # Footprint: area del polígono convexo XY de los 4 bottom.
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(bottom[:, :2])
        footprint = float(hull.volume)  # 2D ConvexHull.volume == area
    except Exception:
        # Fallback: bbox
        xs, ys = bottom[:, 0], bottom[:, 1]
        footprint = float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    return {
        "n_valid": len(valid_pts),
        "centroid_mm": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
        "footprint_area_mm2": footprint,
        "lateral_height_mm": height_mm,
        "x_mm": float(centroid[0]),
        "y_mm": float(centroid[1]),
        "z_mm": float(centroid[2]),
    }


def kpts_observer(yolo_pose_cen_results, yolo_pose_lat_results, conf_min=0.3):
    """Wrapper de alto nivel.

    Inputs son los resultados de `model.predict(img)[0].keypoints` de
    ultralytics (`.xyn` y `.conf` ya extraidos como np.array(N_KPS, 3) en cada
    cam, con (x_norm, y_norm, conf)).

    Devuelve dict con observaciones triangulares 2-view + lista de KPs 3D.
    """
    kps_3d, n_valid = triangulate_keypoints(
        yolo_pose_cen_results, yolo_pose_lat_results, conf_min=conf_min,
    )
    obs = derive_observations(kps_3d)
    obs["kps_3d"] = [list(p) if p is not None else None for p in kps_3d]
    return obs


def extract_yolo_pose_keypoints(model, img_path_or_array, conf=0.25):
    """Helper para obtener (N_KPS, 3) keypoints (x_norm, y_norm, conf) de
    una imagen con un modelo ultralytics YOLO-Pose. Si no hay deteccion,
    devuelve None."""
    try:
        results = model(img_path_or_array, verbose=False, conf=conf)
        if not results:
            return None
        r = results[0]
        if r.keypoints is None or r.keypoints.xyn is None or len(r.keypoints.xyn) == 0:
            return None
        # Tomar la deteccion con mayor confianza
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(r.keypoints.xyn))
        best = int(confs.argmax())
        xyn = r.keypoints.xyn[best].cpu().numpy()  # (N_KPS, 2)
        if r.keypoints.conf is not None:
            kconf = r.keypoints.conf[best].cpu().numpy()  # (N_KPS,)
        else:
            kconf = np.ones(len(xyn)) * float(confs[best])
        return np.hstack([xyn, kconf.reshape(-1, 1)])
    except Exception:
        return None