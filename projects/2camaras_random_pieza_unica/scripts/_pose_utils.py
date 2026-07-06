# -*- coding: utf-8 -*-
"""projects/camara_domo_75078/scripts/_pose_utils.py
===================================================
Copia local de _pose_utils.py adaptada para resolver el mismatch
de coordenadas LDraw -> Blender y el filtro analítico TSI.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Iterable, Optional

# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────
TARPS_MIN_TIPPING_DEFAULT = 0.04

# Dimensiones reales de catálogo de piezas en LDraw Units (1 LDU = 0.4 mm)
# dx, dy, dz (donde Y es la vertical longitudinal del cilindro/eje)
PART_DIMENSIONS_LDU = {
    "32054": (20.0, 60.0, 20.0),  # Pin Largo 3L
    "2780":  (20.0, 40.0, 20.0),  # Pin Fricción 2L
    "6558":  (20.0, 60.0, 20.0),  # Pin Largo 3L con fricción y ranura
    "4274":  (20.0, 30.0, 20.0),  # Pin 1/2
    "61184": (20.0, 60.0, 20.0),  # Hub Conector con pin
    "15392": (40.0, 30.0, 40.0),  # Slope 45 2x2 double
    "3024":  (20.0, 8.0, 20.0),   # Plate 1x1
    "3023":  (20.0, 8.0, 40.0),   # Plate 1x2
    "3795":  (40.0, 8.0, 120.0),  # Plate 2x6
    "2412b": (20.0, 8.0, 40.0),   # Grille 1x2
    "3068":  (40.0, 8.0, 40.0),   # Tile 2x2
    "3040":  (20.0, 24.0, 20.0),  # Slope 45 1x2
    "32000": (20.0, 24.0, 40.0),  # Brick 1x2 con agujeros
    "85984": (20.0, 8.0, 20.0),   # Rock 1x1
}


# ─────────────────────────────────────────────────────────────────
# 1) Cálculo determinista de rotación (NO requiere Blender)
# ─────────────────────────────────────────────────────────────────
def rotation_quat_from_contact_normal(
    contact_normal: Iterable[float],
    target_down=(0.0, 0.0, -1.0),
) -> tuple[float, float, float, float]:
    # Normalizar
    nx, ny, nz = float(contact_normal[0]), float(contact_normal[1]), float(contact_normal[2])
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    tx, ty, tz = float(target_down[0]), float(target_down[1]), float(target_down[2])
    tnorm = math.sqrt(tx * tx + ty * ty + tz * tz)
    tx, ty, tz = tx / tnorm, ty / tnorm, tz / tnorm

    dot = max(-1.0, min(1.0, nx * tx + ny * ty + nz * tz))

    # Caso paralelo: identidad
    if dot > 0.999999:
        return (1.0, 0.0, 0.0, 0.0)

    # Caso anti-paralelo: rotación 180°
    if dot < -0.999999:
        if abs(nx) < 0.9:
            ax, ay, az = 1.0, 0.0, 0.0
        else:
            ax, ay, az = 0.0, 1.0, 0.0
        coeff = ax * nx + ay * ny + az * nz
        ax -= coeff * nx
        ay -= coeff * ny
        az -= coeff * nz
        an = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / an, ay / an, az / an
        return (0.0, ax, ay, az)

    # Caso general
    cx = ny * tz - nz * ty
    cy = nz * tx - nx * tz
    cz = nx * ty - ny * tx
    cnorm = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cnorm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = cx / cnorm, cy / cnorm, cz / cnorm

    angle = math.acos(dot)
    half = angle * 0.5
    s = math.sin(half)
    w = math.cos(half)
    return (w, ax * s, ay * s, az * s)


# ─────────────────────────────────────────────────────────────────
# 2) TARPS — Tipping-Aware Random Pose Selection
# ─────────────────────────────────────────────────────────────────
def get_stable_poses_for_ref(part_ref: str, cache_path: str) -> list:
    if not os.path.isfile(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return []
    poses = cache.get(part_ref, [])
    return sorted(
        poses,
        key=lambda p: -(p.get("tipping_energy_ratio") or 0.0),
    )


def select_pose_tarps(
    poses: list,
    min_tipping: float = TARPS_MIN_TIPPING_DEFAULT,
    part_ref: Optional[str] = None,
) -> Optional[dict]:
    if not poses:
        return None
        
    # Filtrar poses inestables mediante el Tipping Stability Index (TSI)
    filtered_poses = []
    
    # Obtener dimensiones en LDU (estándar LDraw)
    dims_ldu = PART_DIMENSIONS_LDU.get(part_ref)
    if not dims_ldu and part_ref:
        # Fallback analítico aproximado si es una pieza desconocida nueva
        # Intentamos estimar a partir de las coordenadas del mesh de ser posible, o AABB cúbica por defecto
        dims_ldu = (20.0, 20.0, 20.0)
        
    for p in poses:
        normal = p.get("contact_normal")
        w_mm = p.get("contact_stable_width")
        
        if normal and len(normal) == 3 and w_mm is not None:
            # Calcular altura proyectada de la pieza en esta pose (convertido a mm)
            # H = sqrt((dx * nx)^2 + (dy * ny)^2 + (dz * nz)^2) * 0.4 mm
            nx, ny, nz = normal[0], normal[1], normal[2]
            dx, dy, dz = dims_ldu[0], dims_ldu[1], dims_ldu[2]
            height_mm = math.sqrt((dx * nx)**2 + (dy * ny)**2 + (dz * nz)**2) * 0.4
            
            # Tipping Stability Index (TSI)
            tsi = w_mm / (height_mm + 1e-8)
            
            # Si el TSI es demasiado bajo (< 0.12), significa que la pieza es muy alta
            # en relación con su base de sustentación, por lo que volcaría con vibración.
            if tsi >= 0.12:
                filtered_poses.append(p)
        else:
            # Si faltan datos de dimensiones de contacto en BD, no filtramos por TSI
            filtered_poses.append(p)
            
    # Si todas las poses son inestables (ej. geometrías complejas), caemos en la que tiene el TSI más alto
    if not filtered_poses:
        def get_tsi(pose):
            n = pose.get("contact_normal")
            w = pose.get("contact_stable_width") or 0.0
            if n and len(n) == 3:
                dx, dy, dz = dims_ldu[0], dims_ldu[1], dims_ldu[2]
                h = math.sqrt((dx * n[0])**2 + (dy * n[1])**2 + (dz * n[2])**2) * 0.4
                return w / (h + 1e-8)
            return 0.0
        filtered_poses = [max(poses, key=get_tsi)]

    candidates = [
        p for p in filtered_poses
        if (p.get("tipping_energy_ratio") or 0.0) >= min_tipping
    ]
    if candidates:
        return random.choice(candidates)
    return max(filtered_poses, key=lambda p: p.get("tipping_energy_ratio") or 0.0)


# ─────────────────────────────────────────────────────────────────
# 3) Aplicación de la pose en Blender
# ─────────────────────────────────────────────────────────────────
def _world_bbox_min_z(part_obj) -> float:
    if not part_obj.data or not hasattr(part_obj.data, "vertices"):
        import mathutils
        bbox_local = [mathutils.Vector(c) for c in part_obj.bound_box]
        return min((part_obj.matrix_world @ v).z for v in bbox_local)
    mw = part_obj.matrix_world
    return min((mw @ v.co).z for v in part_obj.data.vertices)


def apply_stable_pose(
    part_obj,
    pose: dict,
    *,
    random_z: bool = True,
    snap_offset_bu: float = 0.02,
    rng: Optional[random.Random] = None,
):
    import bpy
    import mathutils

    if rng is None:
        rng = random

    # 1) Rotación analítica determinista convertida a Blender Space
    contact_normal_ldraw = pose.get("contact_normal")
    if contact_normal_ldraw is None or len(contact_normal_ldraw) != 3:
        quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    else:
        # Convertir contact_normal de LDraw Space a Blender Space:
        # [x, y, z] en LDraw -> [x, z, -y] en Blender (rotación -90° X)
        contact_normal_bl = [
            contact_normal_ldraw[0],
            contact_normal_ldraw[2],
            -contact_normal_ldraw[1]
        ]
        quat_wxyz = rotation_quat_from_contact_normal(contact_normal_bl)

    part_obj.rotation_mode = "QUATERNION"
    part_obj.rotation_quaternion = mathutils.Quaternion(quat_wxyz)
    bpy.context.view_layer.update()

    # 2) Rotación aleatoria adicional alrededor del eje Z mundial.
    if random_z:
        angle_z = rng.uniform(0.0, 2.0 * math.pi)
        q_z = mathutils.Quaternion((0.0, 0.0, 1.0), angle_z)
        part_obj.rotation_quaternion = q_z @ part_obj.rotation_quaternion
        bpy.context.view_layer.update()

    # 3) Snap a la cinta (z = 0)
    min_z = _world_bbox_min_z(part_obj)
    part_obj.location.z = part_obj.location.z - min_z + snap_offset_bu
    bpy.context.view_layer.update()

    return {
        "rotation_quat": tuple(part_obj.rotation_quaternion),
        "location": tuple(part_obj.location),
        "applied_quat_from_contact_normal": quat_wxyz,
    }
