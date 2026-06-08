# -*- coding: utf-8 -*-
"""
scripts/populate_silhouette_areas.py

Calcula la **silueta real** (no convexa) del mesh LDraw proyectado sobre el
plano perpendicular a `contact_normal` de cada pose estable, usando
`shapely.ops.unary_union` sobre los triángulos proyectados.

A diferencia de `populate_pose_areas.py` (que usa `ConvexHull`), este método
**SÍ detecta concavidades externas** del contorno. Sin embargo, **no siempre
detecta agujeros pasantes** porque eso depende de cómo el `.dat` de LDraw
modele las paredes interiores: si las tapas superior/inferior cubren la zona
del agujero con triángulos macizos, la unión 2D rellena el hueco.

Pipeline
--------
1. Lee todas las poses de `stable_poses` (id, part_ref, pose_index, contact_normal).
2. Agrupa por `part_ref` para cargar el mesh LDraw una sola vez por pieza.
3. Para cada pose:
   a. Construye base ortonormal (u, v) en el plano perpendicular a contact_normal.
   b. Proyecta cada triangulo a 2D (coords [u . v_i, v . v_i]).
   c. Crea Polygon Shapely por triangulo, repara con `.buffer(0)` los invalidos.
   d. `unary_union(polygons).area` -> area en LDU^2.
   e. Multiplica por 0.16 -> mm^2 (1 LDU^2 = 0.16 mm^2).
4. UPDATE `stable_poses.zenith_silhouette_area`.

Validacion
----------
Al final imprime piezas donde `zenith_silhouette_area > zenith_observable_area`
(violacion del invariante silueta <= convex hull, esperable solo por error
numerico < 1%) y un resumen de las diferencias relativas.

Uso
---
    python scripts/populate_silhouette_areas.py [--dry-run] [--parts 48336,3001,...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "database"))

from ldraw_mesh_parser import get_triangles  # type: ignore  # noqa: E402
from supabase_client import get_connection  # type: ignore  # noqa: E402

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.errors import GEOSException  # type: ignore
except ImportError as e:  # pragma: no cover
    print("ERROR: falta `shapely`. Instala con: pip install shapely>=2.0")
    raise SystemExit(1) from e

LDU2_TO_MM2 = 0.16  # (0.4 mm/LDU)^2


# --------------------------------------------------------------------- helpers
def build_stable_pose_transforms(tris: np.ndarray, cn: np.ndarray) -> np.ndarray:
    """Devuelve los triángulos alineados de forma que contact_normal apunte a [0, 0, -1]
    y la coordenada Z mínima sea exactamente 0.0."""
    n = cn / (np.linalg.norm(cn) + 1e-10)
    target = np.array([0.0, 0.0, -1.0])
    
    if np.allclose(n, target):
        R = np.eye(3)
    elif np.allclose(n, -target):
        R = -np.eye(3)
        R[0,0] = 1.0; R[1,1] = -1.0; R[2,2] = -1.0
    else:
        axis = np.cross(n, target)
        axis /= np.linalg.norm(axis) + 1e-10
        angle = np.acos(np.clip(np.dot(n, target), -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*np.dot(K, K)

    # Convertir a mm (1 LDU = 0.4 mm)
    tris_mm = (tris @ R.T) * 0.4
    
    # Trasladar para que Z mín sea exactamente 0.0 (apoyado en la cinta)
    min_z = tris_mm.reshape(-1, 3)[:, 2].min()
    tris_mm[:, :, 2] -= min_z
    return tris_mm


def calculate_silhouette_and_height(tris: np.ndarray, cn: np.ndarray) -> tuple[float, float]:
    """Calcula la silueta corregida por perspectiva y la altura efectiva en mm.
    
    Improvement A: Altura efectiva (media ponderada por área).
    Improvement B: Silueta proyectada por perspectiva a 150mm y corregida.
    """
    if tris.shape[0] == 0:
        return 0.0, 0.0
        
    tris_mm = build_stable_pose_transforms(tris, cn)
    
    # 1. Calcular altura efectiva (media ponderada por área de la proyección ortográfica)
    total_area = 0.0
    weighted_z_sum = 0.0
    polys_ortho: list[Polygon] = []
    
    for tri in tris_mm:
        try:
            poly = Polygon(tri[:, :2])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-9:
                continue
            if poly.geom_type == "Polygon":
                polys_ortho.append(poly)
                total_area += poly.area
                weighted_z_sum += poly.area * np.mean(tri[:, 2])
            elif poly.geom_type in ("MultiPolygon", "GeometryCollection"):
                for sub in poly.geoms:
                    if sub.geom_type == "Polygon" and sub.area > 1e-9:
                        polys_ortho.append(sub)
                        total_area += sub.area
                        weighted_z_sum += sub.area * np.mean(tri[:, 2])
        except (ValueError, GEOSException):
            continue

    if total_area < 1e-9:
        return 0.0, 0.0

    avg_height = float(weighted_z_sum / total_area)
    
    # 2. Calcular silueta con perspectiva (cámara a 150mm, proyectada sobre Z=0)
    H_c = 150.0
    polys_persp: list[Polygon] = []
    
    for tri in tris_mm:
        try:
            # Proyección de perspectiva de cada vértice
            tri_proj = np.zeros((3, 2))
            for j in range(3):
                x, y, z = tri[j]
                scale = H_c / max(H_c - z, 1.0)
                tri_proj[j, 0] = x * scale
                tri_proj[j, 1] = y * scale
                
            poly = Polygon(tri_proj)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-9:
                continue
            if poly.geom_type == "Polygon":
                polys_persp.append(poly)
            elif poly.geom_type in ("MultiPolygon", "GeometryCollection"):
                for sub in poly.geoms:
                    if sub.geom_type == "Polygon" and sub.area > 1e-9:
                        polys_persp.append(sub)
        except (ValueError, GEOSException):
            continue

    if not polys_persp:
        return 0.0, avg_height

    try:
        union = unary_union(polys_persp)
    except GEOSException:
        union = polys_persp[0]
        for p in polys_persp[1:]:
            try:
                union = union.union(p)
            except GEOSException:
                union = union.buffer(0).union(p.buffer(0))
                
    area_persp = float(union.area)
    
    # 3. Aplicar la corrección de perspectiva usando la altura efectiva (Improvement A & B)
    area_corrected = area_persp * ((H_c - avg_height) / H_c) ** 2
    return round(area_corrected, 4), round(avg_height, 4)


# --------------------------------------------------------------------- DB IO
def fetch_poses(conn, part_filter):
    sql = (
        "SELECT id, part_ref, pose_index, contact_normal, "
        "       zenith_observable_area, lateral_height "
        "FROM stable_poses"
    )
    params: list = []
    if part_filter:
        sql += " WHERE part_ref = ANY(%s)"
        params.append(part_filter)
    sql += " ORDER BY part_ref, pose_index"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def group_by_part(rows):
    out: dict = {}
    for r in rows:
        out.setdefault(r["part_ref"], []).append(r)
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Popula zenith_silhouette_area")
    ap.add_argument("--dry-run", action="store_true", help="No escribe a BD.")
    ap.add_argument("--parts", default=None,
                    help="Lista de part_refs separados por coma.")
    args = ap.parse_args()

    part_filter = (
        [p.strip() for p in args.parts.split(",") if p.strip()]
        if args.parts else None
    )

    print("=" * 72)
    print("POPULAR zenith_silhouette_area (silueta real, no convexa)")
    print("=" * 72)
    print(f"Modo: {'DRY-RUN' if args.dry_run else 'ESCRITURA EN BD'}")
    if part_filter:
        print(f"Filtro piezas: {part_filter}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE stable_poses "
                "ADD COLUMN IF NOT EXISTS zenith_silhouette_area DOUBLE PRECISION;"
            )
            cur.execute(
                "ALTER TABLE stable_poses "
                "ADD COLUMN IF NOT EXISTS effective_height DOUBLE PRECISION;"
            )
            cur.execute(
                "ALTER TABLE stable_poses "
                "ADD COLUMN IF NOT EXISTS efective_height DOUBLE PRECISION;"
            )
        conn.commit()

        rows = fetch_poses(conn, part_filter)
        print(f"Poses a procesar: {len(rows)}")
        groups = group_by_part(rows)
        print(f"Piezas unicas:    {len(groups)}")
        print("-" * 72)

        updated = 0
        skipped_mesh = 0
        violations = []
        report = []

        t0 = time.time()
        with conn.cursor() as cur:
            for i, (part_ref, part_rows) in enumerate(groups.items(), 1):
                tris = get_triangles(part_ref)
                if tris.shape[0] == 0:
                    print(f"[{i}/{len(groups)}] {part_ref}: sin malla LDraw, salto {len(part_rows)} poses.")
                    skipped_mesh += len(part_rows)
                    continue

                for r in part_rows:
                    cn = np.array(r["contact_normal"], dtype=np.float64)
                    area_mm2, eff_height = calculate_silhouette_and_height(tris, cn)

                    conv = r["zenith_observable_area"]
                    if conv is not None and area_mm2 > conv * 1.001:
                        violations.append((part_ref, r["pose_index"], area_mm2, conv))

                    report.append({
                        "part_ref": part_ref,
                        "pose_index": r["pose_index"],
                        "silhouette_mm2": area_mm2,
                        "convex_mm2": conv,
                    })

                    if not args.dry_run:
                        cur.execute(
                            "UPDATE stable_poses "
                            "SET zenith_silhouette_area = %s, "
                            "    effective_height = %s, "
                            "    efective_height = %s, "
                            "    updated_at = NOW() "
                            "WHERE id = %s;",
                            (area_mm2, eff_height, eff_height, r["id"]),
                        )
                    updated += 1

                if i % 25 == 0:
                    print(f"  ... {i}/{len(groups)} piezas procesadas")

        if not args.dry_run:
            conn.commit()

        dt = time.time() - t0
        print("-" * 72)
        ms = dt / max(updated, 1) * 1000
        print(f"Poses actualizadas: {updated} en {dt:.1f}s ({ms:.1f} ms/pose)")
        if skipped_mesh:
            print(f"Poses omitidas (sin malla): {skipped_mesh}")
        if violations:
            print(f"\n[!] {len(violations)} poses con silueta > convex hull (tol 0.1%):")
            for ref, idx, sil, conv in violations[:20]:
                print(f"     {ref} pose#{idx}: sil={sil:.2f} > conv={conv:.2f}")
        else:
            print("\nOK: invariante silueta <= convex hull se cumple en todas las poses.")

        with_both = [r for r in report if r["convex_mm2"]]
        if with_both:
            ratios = np.array([r["silhouette_mm2"] / r["convex_mm2"] for r in with_both])
            i_min = int(np.argmin(ratios))
            print("\nEstadisticas silueta/convex:")
            print(f"   media:    {ratios.mean():.3f}")
            print(f"   min:      {ratios.min():.3f}  "
                  f"({with_both[i_min]['part_ref']} pose#{with_both[i_min]['pose_index']})")
            p25, p50, p75 = np.percentile(ratios, [25, 50, 75])
            print(f"   p25/50/75: {p25:.3f} / {p50:.3f} / {p75:.3f}")

            top_diff = sorted(
                with_both,
                key=lambda r: (r["convex_mm2"] - r["silhouette_mm2"]),
                reverse=True,
            )[:15]
            print("\nTop 15 piezas con mayor diferencia (convex - silueta) [mm^2]:")
            print(f"  {'part_ref':<12} {'pose':>4} {'silhouette':>12} {'convex':>12} {'diff':>10}")
            for r in top_diff:
                diff = r["convex_mm2"] - r["silhouette_mm2"]
                print(f"  {r['part_ref']:<12} {r['pose_index']:>4} "
                      f"{r['silhouette_mm2']:>12.2f} {r['convex_mm2']:>12.2f} "
                      f"{diff:>10.2f}")

        return 0
    except Exception as e:  # pragma: no cover
        conn.rollback()
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
