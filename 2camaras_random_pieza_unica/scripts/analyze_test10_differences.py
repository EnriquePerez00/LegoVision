#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Análisis de diferencias BD vs Render vs Inferido"""
import json
import sys
import os
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

def main():
    # Cargar datos
    metadata_path = os.path.join(project_root, "data/test10/test10_metadata.json")
    eval_path = os.path.join(project_root, "data/reports/test10_eval.json")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    
    # BD colors lookup: usa _db_helpers (todos los combos ref+color de piece_embeddings)
    # Fallback: REAL_SETS["75078-1"] si BD no disponible
    bd_colors = {}
    try:
        from _db_helpers import get_all_ref_color_combinations_from_db
        db_combos = get_all_ref_color_combinations_from_db()
        if db_combos:
            for combo in db_combos:
                key = (combo["ref"], str(combo["color_code"]))
                bd_colors[key] = {
                    "color_hex": combo["color_hex"] if combo["color_hex"].startswith("#") else "#" + combo["color_hex"],
                    "color_name": combo["color_name"],
                    "color_code": str(combo["color_code"])
                }
            print(f"[BD] {len(bd_colors)} combos (ref, color) cargados desde piece_embeddings")
        else:
            raise Exception("BD vacía")
    except Exception as e:
        print(f"[WARN] BD no disponible ({e}), usando REAL_SETS fallback")
        from database.set_catalog import REAL_SETS
        for part in REAL_SETS["75078-1"]["parts"]:
            key = (part["ref"], str(part["color_code"]))
            bd_colors[key] = {
                "color_hex": part["color_hex"],
                "color_name": part["color_name"],
                "color_code": str(part["color_code"])
            }
    
    print("=" * 100)
    print(" ANÁLISIS DE DIFERENCIAS: BD vs RENDER vs INFERIDO")
    print("=" * 100)
    print()
    
    # COLOR ANALYSIS
    print("SECCIÓN 1: ANÁLISIS DE COLOR")
    print("-" * 100)
    print(f"{'#':<4} {'Ref':<8} {'GT':<4} {'Name':<20} {'BD_Hex':<10} {'Render':<10} {'Cen_Code':<10} {'Lat_Code':<10} {'Status':<15}")
    print("-" * 100)
    
    color_stats = {"total": 0, "bd_render_match": 0, "cenital_match": 0, "lateral_match": 0, "consensus_ok": 0}
    
    for idx, (render_meta, eval_result) in enumerate(zip(metadata["renders"], eval_data["results"])):
        ref = render_meta["ref"]
        color_code_gt = str(render_meta["color_code"])
        color_hex_render = render_meta["color_hex"]
        
        key = (ref, color_code_gt)
        bd_entry = bd_colors.get(key, {"color_hex": "N/A", "color_name": "Unknown"})
        bd_hex = bd_entry["color_hex"]
        bd_name = bd_entry["color_name"]
        
        code_cen = str(eval_result.get("color_cenital_normalized_code", "?"))
        code_lat = str(eval_result.get("color_lateral_normalized_code", "?"))
        
        color_stats["total"] += 1
        if bd_hex.upper() == color_hex_render.upper():
            color_stats["bd_render_match"] += 1
        if code_cen == color_code_gt:
            color_stats["cenital_match"] += 1
        if code_lat == color_code_gt:
            color_stats["lateral_match"] += 1
        if code_cen == code_lat:
            color_stats["consensus_ok"] += 1
        
        cen_match = "✓" if code_cen == color_code_gt else "✗"
        lat_match = "✓" if code_lat == color_code_gt else "✗"
        consensus = "OK" if code_cen == code_lat else "CONFLICT"
        status = f"C:{cen_match} L:{lat_match} {consensus}"
        
        print(f"{idx+1:<4} {ref:<8} {color_code_gt:<4} {bd_name:<20} {bd_hex:<10} {color_hex_render:<10} {code_cen:<10} {code_lat:<10} {status:<15}")
    
    print()
    print("RESUMEN COLOR:")
    print(f"  BD == Render hex         : {color_stats['bd_render_match']}/{color_stats['total']} ({100*color_stats['bd_render_match']/color_stats['total']:.1f}%)")
    print(f"  Cenital inferido == GT   : {color_stats['cenital_match']}/{color_stats['total']} ({100*color_stats['cenital_match']/color_stats['total']:.1f}%)")
    print(f"  Lateral inferido == GT   : {color_stats['lateral_match']}/{color_stats['total']} ({100*color_stats['lateral_match']/color_stats['total']:.1f}%)")
    print(f"  Consenso cen/lat OK      : {color_stats['consensus_ok']}/{color_stats['total']} ({100*color_stats['consensus_ok']/color_stats['total']:.1f}%)")
    print()
    
    # SURFACE ANALYSIS
    print("SECCIÓN 2: SUPERFICIE CENITAL")
    print("-" * 100)
    print(f"{'#':<4} {'Ref':<8} {'Pose':<6} {'Area_DB':<12} {'Area_Obs':<12} {'Error_%':<12} {'Valid':<8}")
    print("-" * 100)
    
    surf_stats = {"total": 0, "valid": 0}
    for idx, (render_meta, eval_result) in enumerate(zip(metadata["renders"], eval_data["results"])):
        ref = render_meta["ref"]
        pose = render_meta.get("pose_index", 0)
        area_db = render_meta.get("zenith_observable_area_gt")
        area_obs = eval_result.get("surface_obs_apparent_mm2")
        error = eval_result.get("surface_error_rel_pct")
        
        surf_stats["total"] += 1
        valid = abs(error) < 10 if error is not None else False
        if valid:
            surf_stats["valid"] += 1
        
        print(f"{idx+1:<4} {ref:<8} {pose:<6} {f'{area_db:.1f}' if area_db else 'N/A':<12} "
              f"{f'{area_obs:.1f}' if area_obs else 'N/A':<12} {f'{error:.1f}' if error else 'N/A':<12} {'✓' if valid else '✗':<8}")
    
    print()
    print(f"RESUMEN SUPERFICIE: {surf_stats['valid']}/{surf_stats['total']} válidas ({100*surf_stats['valid']/surf_stats['total']:.1f}%)")
    print()
    
    # HEIGHT ANALYSIS
    print("SECCIÓN 3: ALTURA LATERAL")
    print("-" * 100)
    print(f"{'#':<4} {'Ref':<8} {'Pose':<6} {'Height_DB':<12} {'Height_Meas':<14} {'Error_%':<12} {'Valid':<8}")
    print("-" * 100)
    
    height_stats = {"total": 0, "valid": 0}
    for idx, (render_meta, eval_result) in enumerate(zip(metadata["renders"], eval_data["results"])):
        ref = render_meta["ref"]
        pose = render_meta.get("pose_index", 0)
        height_db = render_meta.get("lateral_height_gt")
        height_meas = eval_result.get("lateral_height_meas_mm")
        error = eval_result.get("lateral_height_error_rel_pct")
        
        height_stats["total"] += 1
        valid = abs(error) < 15 if error is not None else False
        if valid:
            height_stats["valid"] += 1
        
        print(f"{idx+1:<4} {ref:<8} {pose:<6} {f'{height_db:.1f}' if height_db else 'N/A':<12} "
              f"{f'{height_meas:.2f}' if height_meas else 'N/A':<14} {f'{error:.1f}' if error else 'N/A':<12} {'✓' if valid else '✗':<8}")
    
    print()
    print(f"RESUMEN ALTURA: {height_stats['valid']}/{height_stats['total']} válidas ({100*height_stats['valid']/height_stats['total']:.1f}%)")
    print()
    print("=" * 100)

if __name__ == "__main__":
    main()
