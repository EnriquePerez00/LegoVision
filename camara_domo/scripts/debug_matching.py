# -*- coding: utf-8 -*-
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from camara_domo.scripts.inferencia_neuronal import load_db_universe, match_piece_hypothesis, ColorModel

poses_db, colors_db = load_db_universe()

print(f"Cargados {len(poses_db)} refs en poses_db")

dummy_color = ColorModel(color_code="11", color_name="Black", color_hex="#202020")

# Probar con valores de T004:
area_cen = 948.9
area_lat = 0.0
height_est = 9.6
studs_est = 0
height_is_fallback = True

print(f"\n--- Pruebas de Matching con area_cen={area_cen}, area_lat={area_lat}, height={height_est} (fallback={height_is_fallback}) ---")

# Vamos a ejecutar una versión detallada de match_piece_hypothesis para imprimir debug
candidates = []
rejection_reasons = {
    "no_nominal_cen": 0,
    "diff_cen": 0,
    "diff_lat": 0,
    "diff_h": 0
}

for ref, poses in poses_db.items():
    for pose in poses:
        nominal_cen = pose.zenith_silhouette_area or pose.zenith_observable_area
        if not nominal_cen:
            rejection_reasons["no_nominal_cen"] += 1
            continue
            
        diff_cen = abs(area_cen - nominal_cen) / nominal_cen
        if diff_cen > 0.40: # epsilon
            rejection_reasons["diff_cen"] += 1
            continue
            
        nominal_lat_1 = (pose.lateral_height or 9.6) * (pose.contact_stable_length or 16.0)
        nominal_lat_2 = (pose.lateral_height or 9.6) * (pose.contact_stable_width or 16.0)
        
        lat_is_fallback = (area_lat <= 0.0)
        if lat_is_fallback:
            diff_lat_best = 0.0
        else:
            diff_lat_best = min(
                abs(area_lat - nominal_lat_1) / nominal_lat_1,
                abs(area_lat - nominal_lat_2) / nominal_lat_2
            )
            if diff_lat_best > 0.40: # epsilon_vertical
                rejection_reasons["diff_lat"] += 1
                continue

        nominal_h = pose.lateral_height or pose.effective_height or 9.6
        if height_is_fallback:
            diff_h = 0.0
        else:
            diff_h = abs(height_est - nominal_h) / nominal_h
            if diff_h > 0.40: # epsilon_height
                rejection_reasons["diff_h"] += 1
                continue
                
        # Si llega aquí, es un candidato válido!
        candidates.append((ref, pose.pose_index, diff_cen))

print(f"Total de poses probadas: {sum(len(p) for p in poses_db.values())}")
print(f"Rechazos:")
print(f" - Sin nominal cenital area: {rejection_reasons['no_nominal_cen']}")
print(f" - Error de área cenital > 40%: {rejection_reasons['diff_cen']}")
print(f" - Error de área lateral > 40%: {rejection_reasons['diff_lat']}")
print(f" - Error de altura > 40%: {rejection_reasons['diff_h']}")
print(f"Candidatos encontrados: {len(candidates)}")

if candidates:
    print("\nTop 5 candidatos:")
    candidates.sort(key=lambda x: x[2])
    for c in candidates[:5]:
        print(f" - Ref: {c[0]}, Pose: {c[1]}, diff_cen: {c[2]:.2f}")
else:
    # Imprimir algunas áreas nominales cenitales de la DB
    print("\nEjemplos de áreas nominales en la base de datos:")
    count = 0
    for ref, poses in list(poses_db.items())[:10]:
        for pose in poses:
            nom = pose.zenith_silhouette_area or pose.zenith_observable_area
            print(f" - Ref: {ref}, Pose: {pose.pose_index}, nominal_cen: {nom}")
            count += 1
            if count > 10: break
        if count > 10: break
