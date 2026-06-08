# -*- coding: utf-8 -*-
"""
scripts/populate_pose_areas.py

Calcula la proyección cenital observable (en mm²) y la altura física real
(en mm) para todas las poses estables registradas en la tabla `stable_poses`
y actualiza las columnas:

    - zenith_observable_area : DOUBLE PRECISION  (mm²)
    - lateral_height         : DOUBLE PRECISION  (mm)

Convención (la única correcta para representar la pose en la cinta transportadora):
    - `lateral_height` = extensión de la pieza a lo largo de `contact_normal`,
      o sea, qué tan alta queda la pieza cuando está apoyada sobre la cara
      cuya normal es `contact_normal` (eje vertical "real" en la cinta).
        height_ldu = max(verts · n) - min(verts · n)
        height_mm  = round(height_ldu * 0.4, 2)

    - `zenith_observable_area` = área del Convex Hull 2D del mesh proyectado
      sobre el plano perpendicular a `contact_normal` (silueta vista desde
      una cámara cenital ideal).
        area_ldu² = ConvexHull(proj_2D(verts, n)).volume    # 2D convex hull volume == area
        area_mm²  = round(area_ldu² * 0.16, 2)

Importante:
    - NO usar `orientation_quat` aplicada sobre la bounding box nominal
      (eso fue la causa de un bug histórico en el report HTML donde
      altura/área quedaban cruzadas entre las poses 0-1 ↔ 2-3).
      Sólo el mesh LDraw + `contact_normal` da resultados físicos correctos
      independientes del frame interno de la pieza.

    - LDU → mm: factor 0.4 (1 LDU = 0.4 mm), por tanto LDU² → mm² = 0.16.
"""

import os
import sys
import numpy as np
from scipy.spatial import ConvexHull

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, os.path.join(project_root, "database"))

from ldraw_mesh_parser import get_triangles
from supabase_client import get_connection

def build_2d_basis(normal):
    n = normal / (np.linalg.norm(normal) + 1e-10)
    ref = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,1.,0.])
    u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-10)
    v = np.cross(n, u);   v /= (np.linalg.norm(v) + 1e-10)
    return u, v

def convex_hull_2d_area(points):
    # Remueve puntos extremadamente duplicados para evitar problemas numéricos
    pts_unique = np.unique(points.round(4), axis=0)
    if len(pts_unique) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts_unique)
        return hull.volume # En 2D, volume de ConvexHull es su área
    except Exception as e:
        # Fallback si hay colinealidad o error
        return 0.0

def main():
    print("==================================================")
    print("POPULANDO ÁREAS CENITALES Y ALTURAS FÍSICAS EN STABLE_POSES")
    print("==================================================")
    
    # 1. Obtener todas las poses registradas en la BD
    try:
        conn = get_connection()
    except Exception as e:
        print(f"Error conectando a la BD: {e}")
        return
        
    try:
        with conn.cursor() as cur:
            # Asegurar que las columnas existen en la base de datos
            cur.execute("""
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS zenith_observable_area DOUBLE PRECISION;
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS lateral_height DOUBLE PRECISION;
            """)
            conn.commit()
            
            cur.execute("""
                SELECT id, part_ref, pose_index, contact_normal
                FROM stable_poses
                ORDER BY part_ref, pose_index;
            """)
            poses = cur.fetchall()
            
        print(f"Encontradas {len(poses)} poses estables a procesar.")
        
        # Agrupar por pieza para no cargar el mesh de ldraw repetidamente por pose
        parts_dict = {}
        for p in poses:
            ref = p["part_ref"]
            if ref not in parts_dict:
                parts_dict[ref] = []
            parts_dict[ref].append(p)
            
        print(f"Número de piezas únicas: {len(parts_dict)}")
        
        updated_count = 0
        error_count = 0
        
        with conn.cursor() as cur:
            for idx, (ref, part_poses) in enumerate(parts_dict.items()):
                print(f"[{idx+1}/{len(parts_dict)}] Procesando pieza: {ref}...")
                
                # Cargar el mesh una sola vez para esta pieza
                triangles = get_triangles(ref)
                if len(triangles) == 0:
                    print(f"  ⚠️ Advertencia: No se pudo cargar malla para {ref}. Saltando poses.")
                    error_count += len(part_poses)
                    continue
                
                # Vértices únicos para proyección
                verts = triangles.reshape(-1, 3)
                verts_unique = np.unique(verts.round(1), axis=0)
                
                for p in part_poses:
                    p_idx = p["pose_index"]
                    c_norm = p["contact_normal"]
                    
                    if len(verts_unique) < 3:
                        area_mm2 = 0.0
                        height_mm = 0.0
                    else:
                        contact_normal = np.array(c_norm)
                        norm_val = np.linalg.norm(contact_normal)
                        if norm_val < 1e-6:
                            # Evitar división por cero si la normal no es válida
                            contact_normal = np.array([0.0, 0.0, 1.0])
                        else:
                            contact_normal /= norm_val
                            
                        # 1. Calcular área cenital proyectada (Convex Hull 2D)
                        u_ax, v_ax = build_2d_basis(contact_normal)
                        proj_points = np.array([(np.dot(v, u_ax), np.dot(v, v_ax)) for v in verts_unique])
                        area_ldu = convex_hull_2d_area(proj_points)
                        area_mm2 = float(round(area_ldu * 0.16, 2))
                        
                        # 2. Calcular altura física real (diferencia de extremos proyectados sobre la normal)
                        proj_h = np.dot(verts_unique, contact_normal)
                        height_ldu = np.max(proj_h) - np.min(proj_h)
                        height_mm = float(round(height_ldu * 0.4, 2))
                        
                    # Actualizar fila
                    cur.execute("""
                        UPDATE stable_poses
                        SET zenith_observable_area = %s,
                            lateral_height = %s,
                            updated_at = NOW()
                        WHERE part_ref = %s AND pose_index = %s;
                    """, (area_mm2, height_mm, ref, p_idx))
                    updated_count += 1
                    
            conn.commit()
            print("\n==================================================")
            print(f"PROCESO COMPLETADO EXCELENTEMENTE.")
            print(f"Poses actualizadas con éxito: {updated_count}")
            if error_count > 0:
                print(f"Poses omitidas por error de malla: {error_count}")
            print("==================================================")
            
    except Exception as e:
        conn.rollback()
        print(f"Error durante la actualización: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
