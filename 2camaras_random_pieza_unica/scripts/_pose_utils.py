# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/_pose_utils.py
=================================================
Fuente única de verdad para la **selección y aplicación de poses
estables** en todos los scripts del subproyecto `2camaras_pieza_unica`.

Por qué este módulo
-------------------
Antes de su existencia, cada script implementaba su propia versión
de la selección de pose y de la aplicación de la rotación, lo que
provocaba tres clases de bugs:

  * Un criterio TARPS canónico (documentado en
    ``docs/stable_pose_selection_rule.md``) y una copia legacy
    convivían en distintos scripts.
  * El campo ``orientation_quat`` de ``stable_poses_cache.json``
    contiene quaternions inconsistentes (hay un bug latente en
    ``simulate_stable_poses.py · orient_piece_on_face`` por el que
    ``bpy.ops.object.transform_apply`` no siempre afecta al objeto
    correcto, dejando el quat con la rotación pre-bake en lugar de
    la pos-bake).
  * Cada consumidor mezclaba ``orientation_quat``,
    ``orientation_euler`` y ``contact_normal`` con criterios distintos.

Este módulo expone:

  * :func:`rotation_quat_from_contact_normal` — cálculo analítico
    (Rodrigues) determinista, sin Blender. Es la rotación que aplicar
    a la mesh canónica para que su ``contact_normal`` apunte a -Z
    (es decir, "esa cara mira hacia la cinta"). Esta es la fuente de
    verdad reproducible: depende SOLO del vector ``contact_normal``.
  * :func:`select_pose_tarps` — el algoritmo canónico **TARPS**.
  * :func:`apply_stable_pose` — aplica una pose al objeto Blender
    (rotación analítica + Z aleatorio + snap a la cinta).
  * :func:`get_stable_poses_for_ref` — lee el cache y devuelve las
    poses ordenadas por ``tipping_energy_ratio`` descendente.

El módulo se puede importar SIN Blender (las funciones que requieren
``bpy`` lo importan de forma lazy), para permitir tests unitarios.
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


# ─────────────────────────────────────────────────────────────────
# 1) Cálculo determinista de rotación (NO requiere Blender)
# ─────────────────────────────────────────────────────────────────
def rotation_quat_from_contact_normal(
    contact_normal: Iterable[float],
    target_down=(0.0, 0.0, -1.0),
) -> tuple[float, float, float, float]:
    """Devuelve el cuaternión ``(w, x, y, z)`` que rota
    ``contact_normal`` para que apunte a ``target_down`` (por
    defecto, hacia el -Z del mundo, es decir, hacia la cinta).

    Implementación del algoritmo de Rodrigues con manejo explícito
    de los casos paralelos. Es determinista, no depende de la mesh,
    de Blender ni de ``transform_apply``. Sólo requiere que
    ``contact_normal`` esté expresado en el mismo frame que la
    mesh canónica (LDraw frame), que es lo que guarda el simulador
    en ``stable_poses_cache.json``.
    """
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

    # Caso anti-paralelo: rotación 180° alrededor de cualquier eje
    # perpendicular a `contact_normal`. Elegimos el eje X si
    # `contact_normal` no es paralelo a X; si lo es, elegimos Y.
    if dot < -0.999999:
        if abs(nx) < 0.9:
            ax, ay, az = 1.0, 0.0, 0.0
        else:
            ax, ay, az = 0.0, 1.0, 0.0
        # Eje perpendicular a `n`: v = axis - (axis·n) n; normalizar
        coeff = ax * nx + ay * ny + az * nz
        ax -= coeff * nx
        ay -= coeff * ny
        az -= coeff * nz
        an = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / an, ay / an, az / an
        # Rotación 180° → quat = (0, ax, ay, az)
        return (0.0, ax, ay, az)

    # Caso general
    # axis = n × t (no normalizado; lo normalizaremos)
    cx = ny * tz - nz * ty
    cy = nz * tx - nx * tz
    cz = nx * ty - ny * tx
    cnorm = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cnorm < 1e-9:
        # No debería ocurrir aquí (lo cubre el dot anti-paralelo)
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
    """Devuelve TODAS las poses estables (``is_stable=True``) del
    cache para ``part_ref`` ordenadas por ``tipping_energy_ratio``
    descendente. El consumidor debe aplicar
    :func:`select_pose_tarps` para escoger UNA pose individual.

    Si el cache no existe o el ref no está, devuelve ``[]``.
    """
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
) -> Optional[dict]:
    """**TARPS — Tipping-Aware Random Pose Selection**.

    Devuelve UNA pose de ``poses`` aplicando la regla canónica
    documentada en ``docs/stable_pose_selection_rule.md``:

        candidates = [p ∈ poses : p.tipping_energy_ratio ≥ min_tipping]
        if candidates:
            return random.choice(candidates)
        else:
            return argmax_{p ∈ poses}(p.tipping_energy_ratio)

    Devuelve ``None`` si ``poses`` está vacío.

    Garantías:
      * cobertura del 100 % de las piezas con poses estables;
      * ``tipping ≥ min_tipping`` cuando hay al menos un candidato.
    """
    if not poses:
        return None
    candidates = [
        p for p in poses
        if (p.get("tipping_energy_ratio") or 0.0) >= min_tipping
    ]
    if candidates:
        return random.choice(candidates)
    # Fallback determinista: pose con máximo tipping_energy_ratio
    return max(poses, key=lambda p: p.get("tipping_energy_ratio") or 0.0)


# ─────────────────────────────────────────────────────────────────
# 3) Aplicación de la pose en Blender (lazy import de bpy/mathutils)
# ─────────────────────────────────────────────────────────────────
def _world_bbox_min_z(part_obj) -> float:
    """Devuelve el menor world-Z de la geometria real de `part_obj`.

    IMPORTANTE: itera sobre `obj.data.vertices` (vertices reales del mesh)
    en lugar de `obj.bound_box` (AABB local del mesh). El AABB local rotado
    da el min-Z de las ESQUINAS del cubo envolvente, que para piezas
    inclinadas (contact_normal oblicuo) puede quedar 1-2 cm POR DEBAJO del
    extremo inferior real del mesh. Eso provocaba que el snap a la cinta
    apoyara las esquinas vacias del bbox y la pieza quedara FLOTANDO sobre
    la cinta.

    Lazy import de mathutils (solo Blender).
    """
    if not part_obj.data or not hasattr(part_obj.data, "vertices"):
        # Fallback: usar bbox si no hay vertices accesibles
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
    """Aplica una pose estable canónica al objeto Blender `part_obj`.

    El procedimiento es:
      1. Calcular la rotación analítica desde ``pose["contact_normal"]``
         con :func:`rotation_quat_from_contact_normal`. Esto orienta la
         pieza para que su cara de contacto apunte a -Z (hacia la cinta),
         de manera **determinista** y **reproducible**, sin depender
         del posiblemente-corrupto ``orientation_quat`` del cache.
      2. Si ``random_z`` es True, añadir una rotación aleatoria sobre
         el eje Z mundial (la cara de apoyo no cambia).
      3. Snap a la cinta (``z = -min_z + snap_offset_bu``).

    La pieza queda en ``location = (x_actual, y_actual, z_apoyo)``
    con la X y la Y inalteradas (el caller decide el placement antes
    o después de llamar a esta función; típicamente se llama en
    ``location = (0, 0, 0)`` y el caller mueve XY tras este paso).

    Devuelve la pose efectiva (``part_obj.rotation_quaternion``,
    ``part_obj.location``) como diccionario.
    """
    import bpy
    import mathutils

    if rng is None:
        rng = random

    # 1) Rotación analítica determinista
    contact_normal = pose.get("contact_normal")
    if contact_normal is None or len(contact_normal) != 3:
        # Fallback: identidad
        quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    else:
        quat_wxyz = rotation_quat_from_contact_normal(contact_normal)

    part_obj.rotation_mode = "QUATERNION"
    part_obj.rotation_quaternion = mathutils.Quaternion(quat_wxyz)
    bpy.context.view_layer.update()

    # 2) Rotación aleatoria adicional alrededor del eje Z mundial.
    #    Conservamos la cara de apoyo (compone el quat actual con la
    #    rotación Z global pre-multiplicando: q_total = q_z * q_pose).
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


# ─────────────────────────────────────────────────────────────────
# 4) Self-tests (sin Blender)
# ─────────────────────────────────────────────────────────────────
def _quat_apply(q, v):
    """Aplica el quaternion (w,x,y,z) al vector v=(x,y,z). Devuelve v'."""
    w, qx, qy, qz = q
    vx, vy, vz = v
    # v' = q v q*  (con v como quat puro)
    # Implementación directa
    # t = 2 * (q.xyz × v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + w*t + q.xyz × t
    rx = vx + w * tx + (qy * tz - qz * ty)
    ry = vy + w * ty + (qz * tx - qx * tz)
    rz = vz + w * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)


def _self_test():
    """Self-tests del módulo. Verifica casos canónicos de
    `rotation_quat_from_contact_normal` y de `select_pose_tarps`."""
    print("[_pose_utils self-test]")
    # Test 1: contact_normal = (0, 0, -1) → identidad
    q = rotation_quat_from_contact_normal((0.0, 0.0, -1.0))
    assert abs(q[0] - 1.0) < 1e-6, f"normal=-Z: {q}"
    print(f"  ✓ normal=(0,0,-1) → identidad: {q}")

    # Test 2: contact_normal = (0, 0, +1) → 180° alrededor de X (o Y)
    q = rotation_quat_from_contact_normal((0.0, 0.0, 1.0))
    v = _quat_apply(q, (0.0, 0.0, 1.0))
    assert abs(v[2] - (-1.0)) < 1e-6, f"normal=+Z: rotated {v}"
    print(f"  ✓ normal=(0,0,+1) → rota +Z a {v}")

    # Test 3: contact_normal = (0, +1, 0)  (LDraw +Y, "Bottom")
    # Debe rotar +Y a -Z
    q = rotation_quat_from_contact_normal((0.0, 1.0, 0.0))
    v = _quat_apply(q, (0.0, 1.0, 0.0))
    assert abs(v[2] - (-1.0)) < 1e-5, f"normal=+Y: rotated {v}"
    print(f"  ✓ normal=(0,+1,0) → rota +Y a {v}")

    # Test 4: contact_normal = (0, -1, 0)
    q = rotation_quat_from_contact_normal((0.0, -1.0, 0.0))
    v = _quat_apply(q, (0.0, -1.0, 0.0))
    assert abs(v[2] - (-1.0)) < 1e-5, f"normal=-Y: rotated {v}"
    print(f"  ✓ normal=(0,-1,0) → rota -Y a {v}")

    # Test 5: contact_normal = (1, 0, 0)
    q = rotation_quat_from_contact_normal((1.0, 0.0, 0.0))
    v = _quat_apply(q, (1.0, 0.0, 0.0))
    assert abs(v[2] - (-1.0)) < 1e-5, f"normal=+X: rotated {v}"
    print(f"  ✓ normal=(+1,0,0) → rota +X a {v}")

    # Test 6: caso oblicuo
    q = rotation_quat_from_contact_normal((0.5, 0.5, 0.5))
    v = _quat_apply(q, (0.5, 0.5, 0.5))
    # v debería tener norm ≈ √(0.75) y apuntar a -Z
    nv = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    target = (0.0, 0.0, -nv)
    err = math.sqrt(sum((v[i] - target[i]) ** 2 for i in range(3)))
    assert err < 1e-5, f"normal=oblique: rotated {v} (err {err})"
    print(f"  ✓ normal=oblicuo → rota correctamente a {v}")

    # Test 7: TARPS principal
    poses = [
        {"pose_index": 0, "tipping_energy_ratio": 0.5},
        {"pose_index": 1, "tipping_energy_ratio": 0.1},
        {"pose_index": 2, "tipping_energy_ratio": 0.02},
    ]
    rng = random.Random(0)
    rs = []
    for _ in range(50):
        random.seed(rng.random())
        rs.append(select_pose_tarps(poses)["pose_index"])
    assert all(r in (0, 1) for r in rs), f"TARPS picked invalid: {set(rs)}"
    assert 2 not in rs, "TARPS no debería elegir pose 2 (tipping<0.04)"
    print(f"  ✓ TARPS: 50 selecciones ⊂ {{0,1}} (excluye pose 2 con tip=0.02)")

    # Test 8: TARPS fallback (todas por debajo del umbral)
    poses_low = [
        {"pose_index": 0, "tipping_energy_ratio": 0.01},
        {"pose_index": 1, "tipping_energy_ratio": 0.03},
        {"pose_index": 2, "tipping_energy_ratio": 0.001},
    ]
    pf = select_pose_tarps(poses_low)
    assert pf["pose_index"] == 1, f"TARPS fallback eligió {pf}"
    print(f"  ✓ TARPS fallback: eligió pose 1 (argmax tip=0.03)")

    # Test 9: TARPS con poses vacías
    assert select_pose_tarps([]) is None
    print("  ✓ TARPS con []: None")

    print("[_pose_utils] TODOS LOS TESTS OK")


if __name__ == "__main__":
    _self_test()
