# -*- coding: utf-8 -*-
"""
scripts/populate_stable_pose_dims.py
=====================================

Calcula y actualiza en la tabla `stable_poses` los siguientes campos
derivados a partir del mesh LDraw y `contact_normal` de cada pose:

    - zenith_observable_area  (mm²)  : área del Convex Hull 2D de la
      proyección de los vértices del mesh sobre el plano perpendicular
      a `contact_normal` (silueta vista por una cámara cenital ideal).

    - lateral_height          (mm)   : extensión del mesh a lo largo de
      `contact_normal`, es decir, qué tan alta queda la pieza cuando
      está apoyada en la pose.

    - contact_stable_length   (mm)   : lado MAYOR del minAreaRect de la
      cara que toca la cinta. Se aproxima como los vértices con
      proyección mínima sobre `contact_normal` (dentro de tolerancia
      `tol_ldu`).

    - contact_stable_width    (mm)   : lado MENOR del mismo minAreaRect.

Convenciones:
    1 LDU = 0.4 mm  →  area_mm² = area_ldu² × 0.16
    contact_normal apunta desde el centro de la pieza hacia AFUERA por
    la cara que toca la cinta (i.e. la cara de contacto está en
    `proj·n = min(proj·n)`).

Uso:
    .venv/bin/python scripts/populate_stable_pose_dims.py
        [--set_id 75078-1]   (None = todas)
        [--tol_ldu 0.5]      (tolerancia para identificar la cara de contacto)
        [--dry_run]
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial import ConvexHull

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, os.path.join(project_root, "database"))

from ldraw_mesh_parser import get_triangles  # noqa: E402
from supabase_client import get_connection   # noqa: E402

LDRAW_TO_MM = 0.4
LDRAW_TO_MM2 = LDRAW_TO_MM ** 2  # 0.16


def build_2d_basis(normal):
    n = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u);   v /= (np.linalg.norm(v) + 1e-12)
    return u, v


def convex_hull_2d_area(points):
    pts = np.unique(points.round(4), axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return ConvexHull(pts).volume  # 2D ConvexHull.volume == area
    except Exception:
        return 0.0


def min_area_rect_2d(points):
    """Returns (length, width) — both in same units as input — using cv2.minAreaRect."""
    if len(points) < 3:
        return 0.0, 0.0
    try:
        import cv2
        rect = cv2.minAreaRect(points.astype(np.float32))
        (_, _), (w, h), _ = rect
        return float(max(w, h)), float(min(w, h))
    except Exception:
        # Fallback: AABB
        if len(points) == 0:
            return 0.0, 0.0
        w = points[:, 0].max() - points[:, 0].min()
        h = points[:, 1].max() - points[:, 1].min()
        return float(max(w, h)), float(min(w, h))


def compute_pose_dims(part_ref, contact_normal, tol_ldu=None):
    """
    Devuelve dict con keys:
      zenith_observable_area, lateral_height, contact_stable_length, contact_stable_width
    Todas en mm / mm². None en valores que no se puedan computar.

    Si `tol_ldu` es None se usa una tolerancia adaptativa:
        tol_ldu = max(0.5, height_ldu * 0.10)
    Esto evita que en piezas curvas (round plates, slopes) la cara de
    contacto se reduzca a una banda de píxeles sin grosor (que da
    minAreaRect width ~0). Una banda del 10 % de la altura total captura
    el contacto real sin "robar" área de las paredes laterales en piezas
    de geometría plana clara (donde 0.5 LDU es suficiente y predomina).
    """
    out = {
        "zenith_observable_area": None,
        "lateral_height": None,
        "contact_stable_length": None,
        "contact_stable_width": None,
    }
    try:
        tris = get_triangles(part_ref)
    except Exception as e:
        print(f"   [WARN] {part_ref}: get_triangles falló: {e}")
        return out
    if tris is None or len(tris) == 0:
        return out

    verts = tris.reshape(-1, 3)
    verts_unique = np.unique(verts.round(2), axis=0)
    if len(verts_unique) < 3:
        return out

    n = np.array(contact_normal, dtype=float)
    nrm = np.linalg.norm(n)
    if nrm < 1e-6:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / nrm

    # Lateral height (extensión total a lo largo de la normal)
    proj_h = verts_unique @ n
    height_ldu = float(proj_h.max() - proj_h.min())
    out["lateral_height"] = round(height_ldu * LDRAW_TO_MM, 2)

    # Zenith observable area: ConvexHull 2D de la proyección al plano _|_ n
    u_ax, v_ax = build_2d_basis(n)
    proj_all_2d = np.column_stack([verts_unique @ u_ax, verts_unique @ v_ax])
    area_ldu2 = convex_hull_2d_area(proj_all_2d)
    out["zenith_observable_area"] = round(float(area_ldu2 * LDRAW_TO_MM2), 2)

    # Tolerancia adaptativa: para piezas de altura pequeña la cara de
    # contacto puede ser una arista finísima del mesh LDraw (round
    # plates, slopes). Una tolerancia fija de 0.5 LDU subestima el
    # ancho real del contacto. Usamos 10 % de la altura con suelo 0.5.
    if tol_ldu is None:
        tol_eff = max(0.5, 0.10 * height_ldu)
    else:
        tol_eff = tol_ldu

    # Cara de contacto: vertices con proj_h cercana al mínimo
    p_min = proj_h.min()
    contact_mask = proj_h <= (p_min + tol_eff)
    contact_pts3d = verts_unique[contact_mask]
    if len(contact_pts3d) >= 3:
        contact_pts2d = np.column_stack([contact_pts3d @ u_ax, contact_pts3d @ v_ax])
        L_ldu, W_ldu = min_area_rect_2d(contact_pts2d)
        out["contact_stable_length"] = round(L_ldu * LDRAW_TO_MM, 2)
        out["contact_stable_width"]  = round(W_ldu * LDRAW_TO_MM, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set_id", default="75078-1",
                        help="Set a procesar (None = todos los stable_poses).")
    parser.add_argument("--tol_ldu", type=float, default=None,
                        help="Tolerancia (LDU) para considerar un vértice parte de la cara de contacto. "
                             "None (default) = adaptativa: max(0.5, 0.10 * height_ldu).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Calcula pero NO escribe en la BD.")
    args = parser.parse_args()

    print("=" * 70)
    print("POPULANDO DIMS ESTABLES (zenith area, lateral height, contact L/W)")
    print(f"  set_id   = {args.set_id}")
    print(f"  tol_ldu  = {args.tol_ldu}")
    print(f"  dry_run  = {args.dry_run}")
    print("=" * 70)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if args.set_id:
                cur.execute("""
                    SELECT id, part_ref, pose_index, contact_normal, face_class
                    FROM stable_poses
                    WHERE set_id = %s
                    ORDER BY part_ref, pose_index;
                """, (args.set_id,))
            else:
                cur.execute("""
                    SELECT id, part_ref, pose_index, contact_normal, face_class
                    FROM stable_poses
                    ORDER BY part_ref, pose_index;
                """)
            rows = cur.fetchall()

        print(f"\n{len(rows)} poses a procesar.\n")

        # Agrupar por part_ref para no recargar la malla repetidamente
        by_part = {}
        for r in rows:
            by_part.setdefault(r["part_ref"], []).append(r)

        updated = 0
        skipped = 0
        with conn.cursor() as cur:
            for i, (ref, plist) in enumerate(by_part.items(), start=1):
                print(f"[{i}/{len(by_part)}] {ref}  ({len(plist)} poses)")
                for p in plist:
                    dims = compute_pose_dims(ref, p["contact_normal"], tol_ldu=args.tol_ldu)
                    if dims["zenith_observable_area"] is None:
                        print(f"    pose {p['pose_index']}: sin malla → skip")
                        skipped += 1
                        continue

                    def _fmt(v, w=6):
                        if v is None:
                            return "None".rjust(w)
                        return f"{v:>{w}.2f}"
                    print(f"    pose {p['pose_index']:>3} {p['face_class']:<6}"
                          f"  L={_fmt(dims['contact_stable_length'])} mm"
                          f"  W={_fmt(dims['contact_stable_width'])} mm"
                          f"  H={_fmt(dims['lateral_height'])} mm"
                          f"  zen_area={_fmt(dims['zenith_observable_area'], 7)} mm²")

                    if not args.dry_run:
                        cur.execute("""
                            UPDATE stable_poses
                            SET zenith_observable_area = %s,
                                lateral_height          = %s,
                                contact_stable_length   = %s,
                                contact_stable_width    = %s,
                                updated_at              = NOW()
                            WHERE id = %s;
                        """, (
                            dims["zenith_observable_area"],
                            dims["lateral_height"],
                            dims["contact_stable_length"],
                            dims["contact_stable_width"],
                            p["id"],
                        ))
                    updated += 1

            if not args.dry_run:
                conn.commit()

        print("\n" + "=" * 70)
        print(f"OK · poses actualizadas = {updated} · skipped = {skipped}"
              + (" · DRY RUN" if args.dry_run else ""))
        print("=" * 70)

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()