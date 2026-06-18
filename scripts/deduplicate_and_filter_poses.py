# -*- coding: utf-8 -*-
"""
scripts/deduplicate_and_filter_poses.py
======================================
1. Filtra poses inestables por relación de aspecto (esbeltez).
   Marca como inestables (is_stable = FALSE) aquellas poses donde:
     tipping_energy_ratio < 0.04 (o el umbral configurable)
   y aquellas poses que tienen una altura de CdM muy superior a su margen de soporte:
     (h_com / margin) > max_aspect_ratio (por defecto 3.0)

2. Deduplica poses simétricas:
   Si dos poses estables tienen normales opuestas (antiparalelas, ej: [1,0,0] vs [-1,0,0])
   o rotacionalmente equivalentes bajo la simetría de la pieza, desactiva la duplicada
   para dejar una única pose representativa (conservando la de mayor área o estabilidad).

Uso:
    .venv/bin/python scripts/deduplicate_and_filter_poses.py --part_ref 61184
"""

import os
import sys
import argparse
import math
import numpy as np
import psycopg2
import psycopg2.extras

# Permitir importar desde base de datos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"))

from supabase_client import get_connection

def deduplicate_and_filter_part(part_ref, max_aspect_ratio=3.0, min_tipping=0.04):
    sql_select = """
        SELECT id, pose_index, contact_normal, face_class, contact_area, 
               stability_ratio, tipping_energy_ratio, support_polygon_margin_mm
        FROM stable_poses
        WHERE part_ref = %s
        ORDER BY pose_index
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_select, (part_ref,))
            poses = cur.fetchall()
            
        if not poses:
            print(f"No poses found in DB for {part_ref}")
            return
            
        print(f"\nEvaluating {len(poses)} poses for {part_ref}:")
        
        # 1. Filtro estático de esbeltez y tipping mínimo
        to_disable = set()
        stable_candidates = []
        
        for p in poses:
            margin = p.get("support_polygon_margin_mm")
            tipping = p.get("tipping_energy_ratio")
            normal = p.get("contact_normal")
            
            is_unstable = False
            reasons = []
            
            if tipping is None:
                # Si no tiene métricas estáticas (tipping es None), filtramos por estabilidad dinámica básica
                sr = p.get("stability_ratio") or 0.0
                if sr < 0.05:
                    is_unstable = True
                    reasons.append(f"stability_ratio ({sr:.3f}) < 0.05")
            else:
                # Si tiene métricas estáticas, aplicamos los filtros estrictos
                # Excepto si cumple con el bypass dinámico de balancín (rocker):
                # Alta tasa de caída dinámica (stability_ratio >= 0.15) y altura de CdM baja (h_com < 4.0 mm)
                sr = p.get("stability_ratio") or 0.0
                h_com = float('inf')
                if margin is not None and tipping > 0:
                    aspect_ratio = 1.0 / math.sqrt(tipping**2 + 2 * tipping)
                    h_com = margin * aspect_ratio
                
                is_rocker = (sr >= 0.15 and h_com < 4.0)
                is_cylinder_roll = (sr >= 0.05 and margin is not None and margin < 1.0)
                
                if is_rocker:
                    print(f"  + Pose #{p['pose_index']} (Normal {normal}) bypassed static check (rocker/balance: sr={sr:.3f}, h_com={h_com:.2f} mm)")
                elif is_cylinder_roll:
                    print(f"  + Pose #{p['pose_index']} (Normal {normal}) bypassed static check (cylinder roll: sr={sr:.3f}, margin={margin:.2f} mm)")
                else:
                    if tipping < min_tipping:
                        is_unstable = True
                        reasons.append(f"tipping_ratio ({tipping:.4f}) < {min_tipping}")
                        
                    # Si el margen es extremadamente bajo (esbeltez inestable, ej. sobre studs o pins estrechos)
                    if margin is not None and margin > 1e-4 and tipping > 0:
                        aspect_ratio = 1.0 / math.sqrt(tipping**2 + 2 * tipping)
                        if aspect_ratio > max_aspect_ratio:
                            is_unstable = True
                            reasons.append(f"aspect_ratio ({aspect_ratio:.2f}) > {max_aspect_ratio}")
            
            if is_unstable:
                to_disable.add(p["id"])
                print(f"  - Pose #{p['pose_index']} (Normal {normal}) marked UNSTABLE: {', '.join(reasons)}")
            else:
                stable_candidates.append(p)
                
        # Fallback de emergencia si todas las poses quedan marcadas como inestables
        if not stable_candidates:
            # Seleccionar la pose de menor altura de Centro de Masas (h_com = margin * aspect_ratio)
            # Esto prioriza las posiciones acostadas (menor energía potencial / menor h_com) sobre las verticales
            print("  [WARN] No stable poses remain under strict criteria. Applying fallback to recover 1 representative pose.")
            
            def get_h_com(p):
                margin = p.get("support_polygon_margin_mm")
                tipping = p.get("tipping_energy_ratio")
                if margin is not None and tipping is not None and tipping > 0:
                    aspect_ratio = 1.0 / math.sqrt(tipping**2 + 2 * tipping)
                    return margin * aspect_ratio
                return float('inf')
                
            # Elegir la pose con menor h_com
            best_fallback = min(poses, key=get_h_com)
            to_disable.discard(best_fallback["id"])
            stable_candidates.append(best_fallback)
            print(f"  + Recovered Pose #{best_fallback['pose_index']} (Normal {best_fallback['contact_normal']}, h_com {get_h_com(best_fallback):.2f} mm) as stable fallback.")
            
        # 2. Deduplicar simetrías (normales antiparalelas / simetría horizontal en cinta)
        # Para normales n1 y n2, si son antiparalelas o muy similares en valor absoluto (eje de simetría)
        unique_stable = []
        for p in stable_candidates:
            n1 = np.array(p["contact_normal"])
            merged = False
            for u in unique_stable:
                n2 = np.array(u["contact_normal"])
                # Comprobar si son similares (mismo sentido de la normal)
                dot = np.dot(n1, n2)
                # Si el ángulo es menor a 15 grados (mismo sentido, normales casi paralelas)
                # No deduplicamos antiparalelas (dot < -0.96) para evitar eliminar caras opuestas en piezas asimétricas (ej: lateral izquierdo vs derecho en cuñas)
                if dot > math.cos(math.radians(15.0)):
                    # Es la misma pose física por simetría local.
                    # Mantener la de mayor estabilidad o mayor área
                    if p["contact_area"] > u["contact_area"]:
                        # Desactivamos la anterior u, y mantenemos p
                        to_disable.add(u["id"])
                        unique_stable.remove(u)
                        unique_stable.append(p)
                    else:
                        to_disable.add(p["id"])
                    merged = True
                    print(f"  - Pose #{p['pose_index']} deduplicated with #{u['pose_index']} (Normal symmetry: {p['contact_normal']} vs {u['contact_normal']})")
                    break
                else:
                    # Comprobación geométrica para simetrías rotacionales (cilindros, pins, lados idénticos)
                    # Si el área de contacto y el margen son casi idénticos (dentro del 5%)
                    a1 = p.get("contact_area") or 0.0
                    a2 = u.get("contact_area") or 0.0
                    m1 = p.get("support_polygon_margin_mm") or 0.0
                    m2 = u.get("support_polygon_margin_mm") or 0.0
                    
                    if a1 > 0 and a2 > 0 and m1 > 0 and m2 > 0:
                        diff_a = abs(a1 - a2) / max(a1, a2)
                        diff_m = abs(m1 - m2) / max(m1, m2)
                        if diff_a < 0.05 and diff_m < 0.05:
                            if p.get("stability_ratio", 0) > u.get("stability_ratio", 0):
                                to_disable.add(u["id"])
                                unique_stable.remove(u)
                                unique_stable.append(p)
                            else:
                                to_disable.add(p["id"])
                            merged = True
                            print(f"  - Pose #{p['pose_index']} deduplicated with #{u['pose_index']} (Geometric symmetry: area {a1:.1f}≈{a2:.1f}, margin {m1:.1f}≈{m2:.1f})")
                            break
            if not merged:
                unique_stable.append(p)
                
        # 3. Aplicar actualizaciones a la base de datos
        with conn.cursor() as cur:
            for p in poses:
                is_stable_val = p["id"] not in to_disable
                cur.execute(
                    "UPDATE stable_poses SET is_stable = %s, updated_at = NOW() WHERE id = %s",
                    (is_stable_val, p["id"])
                )
        print(f"  [DB] {part_ref}: {len(unique_stable)} stable poses remaining, {len(to_disable)} disabled.")

def main():
    parser = argparse.ArgumentParser(description="Deduplicates and filters stable poses by static stability criteria.")
    parser.add_argument("--part_ref", type=str, default=None, help="Ref of the piece to process (e.g. 61184), or omit to process all")
    parser.add_argument("--max_aspect_ratio", type=float, default=3.0, help="Max height/margin ratio allowed")
    parser.add_argument("--min_tipping", type=float, default=0.04, help="Min tipping ratio allowed")
    args = parser.parse_args()
    
    if args.part_ref:
        deduplicate_and_filter_part(args.part_ref, args.max_aspect_ratio, args.min_tipping)
    else:
        print("No --part_ref specified. Fetching all part references from database...")
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT DISTINCT part_ref FROM stable_poses ORDER BY part_ref")
                parts = [r["part_ref"] for r in cur.fetchall()]
        print(f"Found {len(parts)} parts in database: {parts}")
        for part in parts:
            deduplicate_and_filter_part(part, args.max_aspect_ratio, args.min_tipping)

if __name__ == "__main__":
    main()
