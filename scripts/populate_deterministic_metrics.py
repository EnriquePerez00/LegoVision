# -*- coding: utf-8 -*-
"""
scripts/populate_deterministic_metrics.py
==========================================
Rellena las metricas DETERMINISTAS de cada pose en `stable_poses`:

    - support_polygon_margin_mm:
        distancia minima (mm) desde la proyeccion del centro de masa al borde
        del poligono de soporte de la cara de contacto. Positiva si el CdM
        esta dentro del poligono (estable estatico). Negativa si esta fuera.

    - tipping_energy_ratio:
        ratio adimensional (sqrt(margin^2 + h_com^2) - h_com) / h_com.
        Mide cuanto tiene que subir el CdM (en relativo a su altura) para
        cruzar la arista mas cercana del poligono de soporte y volcar la
        pieza. Es la "barrera de energia" geometrica de la pose.

Ambas metricas son completamente deterministas: dependen unicamente de la
geometria del mesh LDraw y de `orientation_quat` de la pose. NO dependen
de la simulacion fisica estocastica.

Uso:
    /Users/.../.venv/bin/python scripts/populate_deterministic_metrics.py
        [--part_ref REF] [--set_id ID]

Conversion: 1 LDU = 0.4 mm. La salida de support_polygon_margin esta en mm.
"""
import os
import sys
import argparse
import math

import numpy as np
from scipy.spatial import ConvexHull

# Permitir importar ldraw_mesh_parser desde scripts/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ldraw_mesh_parser import get_triangles  # noqa: E402

import psycopg2
import psycopg2.extras

# Permitir importar desde base de datos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"))

from core.db.supabase_client import get_connection

LDU_TO_MM = 0.4
CONTACT_PLANE_TOL_LDU = 3.0   # 1.2 mm de tolerancia para considerar "vertice en el plano" (soporta piezas inclinadas)
MIN_VERTS_FOR_HULL = 3


def quat_to_matrix(qw, qx, qy, qz):
    """Convierte cuaternion (w,x,y,z) a matriz de rotacion 3x3."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [xy + wz,               1.0 - (xx + zz), yz - wx],
        [xz - wy,               yz + wx,         1.0 - (xx + yy)],
    ])


def rotation_to_align(src, dst):
    """Devuelve la matriz de rotacion 3x3 que lleva el vector unitario `src`
    al vector unitario `dst` por la rotacion mas corta (Rodrigues)."""
    src = src / max(np.linalg.norm(src), 1e-12)
    dst = dst / max(np.linalg.norm(dst), 1e-12)
    d = float(np.dot(src, dst))
    if d > 1.0 - 1e-9:
        return np.eye(3)
    if d < -1.0 + 1e-9:
        # 180 grados: encontrar eje perpendicular cualquiera
        ax = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ax, src)) > 0.9:
            ax = np.array([0.0, 1.0, 0.0])
        ax = ax - np.dot(ax, src) * src
        ax /= max(np.linalg.norm(ax), 1e-12)
        K = np.array([[0, -ax[2], ax[1]],
                      [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    v = np.cross(src, dst)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) * (1.0 / (1.0 + d))


def compute_com_ldu(triangles):
    """Centro de masa estimado como promedio de centroides de triangulos
    ponderado por area. Devuelve coords en LDU."""
    if len(triangles) == 0:
        return np.zeros(3)
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e1, e2)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    centroids = triangles.mean(axis=1)
    total = areas.sum()
    if total < 1e-10:
        return centroids.mean(axis=0)
    return (centroids * areas[:, None]).sum(axis=0) / total


def point_to_polygon_signed_distance(px, py, hull_pts_2d):
    """Distancia firmada del punto (px,py) al poligono convexo (hull_pts_2d
    es la lista de vertices en orden CCW). Positiva si esta dentro,
    negativa si esta fuera. Trabaja en LDU."""
    n = len(hull_pts_2d)
    if n < 3:
        return -1e9
    # Determinar orientacion (CCW vs CW) signando por cross product
    def signed_area():
        a = 0.0
        for i in range(n):
            x1, y1 = hull_pts_2d[i]
            x2, y2 = hull_pts_2d[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return 0.5 * a
    sign = 1.0 if signed_area() > 0 else -1.0

    inside = True
    min_dist = float("inf")
    for i in range(n):
        x1, y1 = hull_pts_2d[i]
        x2, y2 = hull_pts_2d[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        # Vector outward normal (asumiendo CCW => outward = (ey, -ex))
        nx, ny = sign * ey, -sign * ex
        ln = math.hypot(nx, ny)
        if ln < 1e-12:
            continue
        nx /= ln
        ny /= ln
        # Distancia perpendicular (positiva hacia outward)
        d_out = (px - x1) * nx + (py - y1) * ny
        if d_out > 0:
            inside = False
        # Distancia al SEGMENTO (no a la recta) para esquinas:
        t = ((px - x1) * ex + (py - y1) * ey) / max(ex * ex + ey * ey, 1e-12)
        t = max(0.0, min(1.0, t))
        cx, cy = x1 + t * ex, y1 + t * ey
        seg_d = math.hypot(px - cx, py - cy)
        if seg_d < min_dist:
            min_dist = seg_d
    return min_dist if inside else -min_dist


def compute_pose_metrics(triangles, contact_normal):
    """Calcula (support_polygon_margin_mm, tipping_energy_ratio) para la pose.

    Trabaja en el frame nativo del mesh LDraw, sin pasar por Blender. Para
    la pose dada, `contact_normal` es la normal (en LDraw frame) de la cara
    apoyada. Rotamos el mesh para que esa normal apunte hacia +Z (gravedad
    en -Z, suelo en z=0). Esto evita errores con `orientation_quat` que
    estan en frame Blender.

    Si no se puede calcular, devuelve (None, None).
    """
    if len(triangles) == 0:
        return None, None
    cn = np.array(contact_normal, dtype=np.float64)
    if np.linalg.norm(cn) < 1e-9:
        return None, None
    # `contact_normal` es la normal exterior (en frame LDraw nativo) de la
    # cara apoyada, apuntando "fuera" de la pieza. En reposo en la cinta esa
    # normal coincide con la direccion "abajo" (hacia el suelo). En nuestro
    # frame de calculo escogemos +Z = arriba, asi que rotamos -cn -> +Z, lo
    # que equivale a rotar cn -> -Z (suelo en z=zmin, gravedad en -Z).
    target = np.array([0.0, 0.0, -1.0])
    R = rotation_to_align(cn, target)
    verts_world = triangles.reshape(-1, 3) @ R.T
    com_local = compute_com_ldu(triangles)
    com_world = R @ com_local

    # Trasladar para que el suelo (z minima) quede en z=0
    z_min = float(verts_world[:, 2].min())
    verts_world = verts_world.copy()
    verts_world[:, 2] -= z_min
    com_world = com_world.copy()
    com_world[2] -= z_min

    # Vertices proximos al suelo
    contact_mask = verts_world[:, 2] <= CONTACT_PLANE_TOL_LDU
    contact_pts = verts_world[contact_mask][:, :2]
    if len(contact_pts) < MIN_VERTS_FOR_HULL:
        return None, None
    # Eliminar duplicados (rendondeo)
    contact_pts = np.unique(contact_pts.round(3), axis=0)
    if len(contact_pts) < MIN_VERTS_FOR_HULL:
        return None, None
    try:
        hull = ConvexHull(contact_pts)
    except Exception:
        return None, None
    hull_xy = contact_pts[hull.vertices]
    hull_list = [tuple(p) for p in hull_xy]

    cx, cy = float(com_world[0]), float(com_world[1])
    margin_ldu = point_to_polygon_signed_distance(cx, cy, hull_list)
    margin_mm = margin_ldu * LDU_TO_MM

    # Altura del CdM sobre el suelo (LDU); usar mm para el ratio (es adim.).
    h_com_ldu = float(com_world[2])
    if h_com_ldu < 1e-6:
        # CdM bajo suelo => valor no fisico
        return margin_mm, None

    # Para volcar pivotando sobre la arista mas cercana, el CdM debe pasar
    # por encima del eje de pivote. En el peor caso (margin>0), debe ascender
    # hasta sqrt(margin^2 + h_com^2). El ratio relativo es:
    if margin_ldu <= 0:
        # Ya esta fuera del poligono -> energia 0 (caera por si sola)
        return margin_mm, 0.0
    delta_h = math.sqrt(margin_ldu * margin_ldu + h_com_ldu * h_com_ldu) - h_com_ldu
    ratio = delta_h / h_com_ldu
    return margin_mm, ratio


def fetch_poses(part_ref=None, set_id=None):
    where = []
    params = []
    if part_ref:
        where.append("part_ref = %s")
        params.append(part_ref)
    if set_id:
        where.append("COALESCE(set_id,'') = %s")
        params.append(set_id)
    sql = """
        SELECT part_ref, pose_index,
               contact_normal, orientation_quat
        FROM stable_poses
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY part_ref, pose_index"
    with get_connection() as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def update_pose(cur, part_ref, pose_index, margin_mm, tipping_ratio):
    cur.execute(
        """
        UPDATE stable_poses
        SET support_polygon_margin_mm = %s,
            tipping_energy_ratio = %s,
            updated_at = NOW()
        WHERE part_ref = %s AND pose_index = %s
        """,
        (margin_mm, tipping_ratio, part_ref, pose_index),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_ref", type=str, default=None)
    parser.add_argument("--set_id",   type=str, default=None)
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    rows = fetch_poses(args.part_ref, args.set_id)
    print(f"[populate_deterministic_metrics] {len(rows)} poses a procesar")

    # Agrupar por part_ref para no recargar el mesh cada vez
    by_part = {}
    for r in rows:
        by_part.setdefault(r["part_ref"], []).append(r)

    n_ok = 0
    n_skip = 0
    n_fail = 0

    conn = get_connection()
    cur = conn.cursor()

    try:
        for part_ref, poses in sorted(by_part.items()):
            try:
                tris = get_triangles(part_ref)
            except Exception as e:
                print(f"  [{part_ref}] mesh load FAIL: {e}")
                n_skip += len(poses)
                continue
            if tris is None or len(tris) == 0:
                print(f"  [{part_ref}] mesh sin triangulos -> skip {len(poses)} poses")
                n_skip += len(poses)
                continue
            for r in poses:
                cn = r.get("contact_normal")
                if not cn or len(cn) != 3:
                    if args.debug:
                        print(f"    {part_ref} pose {r['pose_index']}: contact_normal invalido")
                    n_skip += 1
                    continue
                try:
                    margin_mm, ratio = compute_pose_metrics(tris, cn)
                except Exception as e:
                    if args.debug:
                        print(f"    {part_ref} pose {r['pose_index']}: error {e}")
                    n_fail += 1
                    continue
                update_pose(
                    cur, part_ref, r["pose_index"], margin_mm, ratio
                )
                n_ok += 1
                if args.debug:
                    print(f"    {part_ref} pose {r['pose_index']:2d}: "
                          f"margin={margin_mm}  ratio={ratio}")
            # Commit after each part_ref to save progress and keep transaction short
            try:
                conn.commit()
            except Exception as e:
                print(f"  [{part_ref}] commit FAIL: {e}")
            print(f"  [{part_ref}] {len(poses)} poses procesadas")
    finally:
        try:
            conn.commit()
            conn.close()
        except Exception:
            pass

    print(
        f"[populate_deterministic_metrics] OK={n_ok}  skip={n_skip}  fail={n_fail}"
    )


if __name__ == "__main__":
    main()
