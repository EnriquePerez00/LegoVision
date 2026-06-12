#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_test500allhd.py
=======================
Análisis completo del dataset test_500allhd a partir de random_500_metadata.json.
No requiere eval_report.json (análisis pre-inferencia del dataset + geometría óptica).
"""
import json, math, collections, os, statistics

# ── Parámetros del setup (de config.yaml) ──
PX_PER_MM_CENITAL = 10.4296288
CAMERA_DIST_MM    = 300.0
FOCAL_MM          = 27.0
SENSOR_MM         = 36.0
IMG_RES_PX        = 2048.0
FOV_NOMINAL_MM    = 196.0
FOCAL_PX          = FOCAL_MM * IMG_RES_PX / SENSOR_MM  # 1536 px

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
metadata_path = os.path.join(project_root, "data", "test_500allhd", "random_500_metadata.json")

with open(metadata_path, "r", encoding="utf-8") as f:
    data = json.load(f)

renders = data["renders"]
N = len(renders)

SEP = "=" * 72

# ──────────────────────────────────────────────────────────────────────
# SECCION 0: VERIFICACION PARAMETROS
# ──────────────────────────────────────────────────────────────────────
print(SEP)
print("  PARAMETROS OPTICOS VERIFICADOS (config.yaml + geometria)")
print(SEP)
px_check = FOCAL_PX / CAMERA_DIST_MM
print(f"  focal_px  = {FOCAL_MM}mm * {IMG_RES_PX:.0f}px / {SENSOR_MM}mm = {FOCAL_PX:.2f} px")
print(f"  PX/MM calculado  = focal_px / CAM_Z = {FOCAL_PX:.2f} / {CAMERA_DIST_MM:.0f} = {px_check:.7f}")
print(f"  PX/MM en config  = {PX_PER_MM_CENITAL:.7f}")
print(f"  Diferencia       = {abs(px_check - PX_PER_MM_CENITAL):.7f} px/mm")
fov_calc = IMG_RES_PX / PX_PER_MM_CENITAL
print(f"  FOV calculado    = {IMG_RES_PX:.0f} / {PX_PER_MM_CENITAL:.7f} = {fov_calc:.2f} mm")
print(f"  FOV en metadata  = {FOV_NOMINAL_MM} mm")
print()
print(f"  Resolution 2048px @ PX_PER_MM={PX_PER_MM_CENITAL:.4f} → FOV={fov_calc:.1f}mm")
print(f"  (La tarea indicaba FOV=20cm con cam a 30cm; aqui FOV={fov_calc/10:.1f}cm)")
print()

# ──────────────────────────────────────────────────────────────────────
# SECCION 1: DISTRIBUCION GENERAL
# ──────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  DATASET test_500allhd  —  {N} samples  —  set_id={data['set_id']}")
print(SEP)

face_classes     = collections.Counter()
refs_counter     = collections.Counter()
colors_counter   = collections.Counter()
pose_counter     = collections.Counter()
color_code_ctr   = collections.Counter()

for r in renders:
    face_classes[r.get("face_class","?")] += 1
    refs_counter[r.get("ref","?")] += 1
    colors_counter[r.get("color_name","?")] += 1
    pose_counter[r.get("pose_index",-1)] += 1
    color_code_ctr[str(r.get("color_code","?"))] += 1

print("\n--- face_class (poses) ---")
for k, v in sorted(face_classes.items(), key=lambda x: -x[1]):
    bar = "█" * int(v/N*40)
    print(f"  {k:12s}: {v:4d} ({100*v/N:5.1f}%)  {bar}")

print("\n--- refs unicos ---")
print(f"  Total refs unicos: {len(refs_counter)}")
for k, v in sorted(refs_counter.items(), key=lambda x: -x[1]):
    print(f"  {k:10s}: {v:4d} ({100*v/N:4.1f}%)")

print("\n--- colores (color_name) ---")
for k, v in sorted(colors_counter.items(), key=lambda x: -x[1]):
    print(f"  {k:30s}: {v:4d} ({100*v/N:4.1f}%)")

print("\n--- pose_index ---")
for k, v in sorted(pose_counter.items()):
    bar = "█" * int(v/N*30)
    print(f"  pose {k:3d}: {v:3d} ({100*v/N:4.1f}%)  {bar}")

# ──────────────────────────────────────────────────────────────────────
# SECCION 2: ANALISIS OPTICO — ERROR DE PERSPECTIVA TEORICO
# ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  2. ERROR DE PERSPECTIVA TEORICO (camara Z=300mm)")
print(SEP)
print()
print("  mag_lineal = Z_cam / (Z_cam - h)")
print("  mag_area   = mag_lineal^2")
print("  error_pct  = (mag_area - 1) * 100%")
print()
print(f"  {'Caso':<35} {'h_total':>9} {'h_eff=h/2':>10} {'mag_area':>10} {'err_area%':>10}")
print("  " + "-"*76)

casos = [
    ("Plate flat h_total=3.2mm",          3.2),
    ("Plate side h_lateral=4.8mm",        4.8),
    ("  → z_eff=2.4mm",                   2.4),
    ("Brick flat h_total=9.6mm",          9.6),
    ("Brick side h_lateral=9.6mm",        9.6),
    ("  → z_eff=4.8mm",                   4.8),
    ("Slope(3039) side h_lat=11.2mm",    11.2),
    ("  → z_eff=6.0mm",                   6.0),
    ("TechBeam(3700) side h_lat=16mm",   16.0),
    ("  → z_eff=8.0mm",                   8.0),
    ("TechBeam(3700) flat h=9.6mm",       9.6),
    ("60481 Stand h_lat=19.2mm",         19.2),
    ("  → z_eff=9.6mm",                   9.6),
    ("61780/3001 side h_lat=20.8mm",     20.8),
    ("  → z_eff=10.4mm",                 10.4),
]
for label, h in casos:
    ml = CAMERA_DIST_MM / (CAMERA_DIST_MM - h)
    ma = ml * ml
    ep = (ma - 1) * 100
    flag = " <<< CRITICO >25%" if ep >= 25 else (" <<< NOTABLE >10%" if ep >= 10 else "")
    print(f"  {label:<35} {h:>9.1f}   {h/2:>8.1f}  {ma:>10.4f}  {ep:>9.1f}%{flag}")

print()
print("  UMBRAL CRITICO: error_area% >= 25% ocurre cuando z_eff >= 33.5mm")
z_thresh_25 = CAMERA_DIST_MM * (1 - 1/math.sqrt(1.25))
print(f"  → z_eff critico para 25%: {z_thresh_25:.1f}mm → h_total critico: {z_thresh_25*2:.1f}mm")
z_thresh_10 = CAMERA_DIST_MM * (1 - 1/math.sqrt(1.10))
print(f"  → z_eff critico para 10%: {z_thresh_10:.1f}mm → h_total critico: {z_thresh_10*2:.1f}mm")

# ──────────────────────────────────────────────────────────────────────
# SECCION 3: ERROR PERSPECTIVA POR face_class (datos reales GT)
# ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  3. PREDICCION ERROR PERSPECTIVA POR face_class (GT del dataset)")
print(SEP)

face_h    = collections.defaultdict(list)
face_heff = collections.defaultdict(list)
face_area = collections.defaultdict(list)
face_obs  = collections.defaultdict(list)

for r in renders:
    fc = r.get("face_class","?")
    h  = r.get("lateral_height_gt")
    he = r.get("effective_height_gt")
    a  = r.get("zenith_silhouette_area_gt")
    ao = r.get("zenith_observable_area_gt")
    if h  is not None: face_h[fc].append(h)
    if he is not None: face_heff[fc].append(he)
    if a  is not None: face_area[fc].append(a)
    if ao is not None: face_obs[fc].append(ao)

print(f"\n  {'fc':<12} {'N':>5} {'h_lat_mean':>12} {'h_lat_max':>11} {'h_eff_mean':>11} "
      f"{'area_gt_mean':>14} {'obs_area_mean':>14} {'err_persp%':>11}")
print("  " + "-"*98)
for fc in sorted(face_h.keys()):
    hs   = face_h[fc]
    hes  = face_heff[fc]
    ars  = face_area[fc]
    aos  = face_obs[fc]
    n    = len(hs)
    hm   = statistics.mean(hs) if hs else 0
    hmax = max(hs) if hs else 0
    hem  = statistics.mean(hes) if hes else 0
    arm  = statistics.mean(ars) if ars else 0
    aom  = statistics.mean(aos) if aos else 0
    # z_eff = h_eff_mean (ya es la altura efectiva del centro de masa)
    z_eff = hem
    ma   = (CAMERA_DIST_MM / (CAMERA_DIST_MM - z_eff))**2 if z_eff < CAMERA_DIST_MM else 0
    ep   = (ma - 1) * 100
    print(f"  {fc:<12} {n:>5} {hm:>12.2f} {hmax:>11.2f} {hem:>11.3f} "
          f"{arm:>14.1f} {aom:>14.1f} {ep:>10.1f}%")

# ──────────────────────────────────────────────────────────────────────
# SECCION 4: ANALISIS POR REF — dimensiones y poses
# ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  4. ANALISIS POR REF — Alturas, areas y poses representadas")
print(SEP)

ref_d = collections.defaultdict(lambda: {
    "h_lat":[], "h_eff":[], "area":[], "obs_area":[], "face":[]
})
for r in renders:
    ref = r.get("ref","?")
    ref_d[ref]["h_lat"].append(r.get("lateral_height_gt") or 0)
    ref_d[ref]["h_eff"].append(r.get("effective_height_gt") or 0)
    ref_d[ref]["area"].append(r.get("zenith_silhouette_area_gt") or 0)
    ref_d[ref]["obs_area"].append(r.get("zenith_observable_area_gt") or 0)
    ref_d[ref]["face"].append(r.get("face_class","?"))

print(f"\n  {'ref':<10} {'N':>4} {'h_max':>7} {'h_mean':>8} {'area_mean':>10}  {'faces'}")
print("  " + "-"*70)
for ref, d in sorted(ref_d.items(), key=lambda x: -max(x[1]["h_lat"] or [0])):
    n    = len(d["h_lat"])
    hmax = max(d["h_lat"])
    hmean= statistics.mean(d["h_lat"])
    amean= statistics.mean(d["area"])
    fc_c = collections.Counter(d["face"])
    fc_s = ", ".join(f"{k}:{v}" for k,v in sorted(fc_c.items(), key=lambda x:-x[1]))
    # Perspectiva: usar z_eff = h_eff_mean
    he_mean = statistics.mean(d["h_eff"]) if d["h_eff"] else hmean*0.5
    ma = (CAMERA_DIST_MM / (CAMERA_DIST_MM - he_mean))**2 if he_mean < CAMERA_DIST_MM else 0
    ep = (ma-1)*100
    flag = " ***" if ep >= 15 else ""
    print(f"  {ref:<8} {n:>4} {hmax:>7.1f} {hmean:>8.2f} {amean:>10.1f}  {fc_s}  [persp={ep:.1f}%{flag}]")

# ──────────────────────────────────────────────────────────────────────
# SECCION 5: POSICION XY — RADIAL
# ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  5. ANALISIS POSICION XY — Radial (periferica vs central)")
print(SEP)

# position_bu en Blender Units; 1 BU = 10mm (bu_per_mm=0.1)
BU_TO_MM = 10.0

radii_mm = []
bbox_cx_norm = []
for r in renders:
    pos = r.get("position_bu", [0,0,0])
    x_mm = pos[0] * BU_TO_MM
    y_mm = pos[1] * BU_TO_MM
    r_mm = math.sqrt(x_mm**2 + y_mm**2)
    radii_mm.append(r_mm)
    # bbox cenital centroid
    bn = r.