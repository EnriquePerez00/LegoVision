# -*- coding: utf-8 -*-
"""generate_mono_dataset.py
===================================================
Genera un dataset sintético para entrenar YOLO-Pose (frontal) ubicando una única pieza
que avanza por el eje de la cinta (Y=0) desde el extremo del FoV cenital hasta caer.

Uso:
    /opt/homebrew/bin/blender -b -P \
        projects/camara_domo_monopieza_90/scripts/generate_mono_dataset.py -- \
            --num_pieces 100 \
            --output_dir projects/camara_domo_monopieza_90/data/yolo_dataset \
            --split train
"""
import os
import sys
import json
import math
import random
import uuid

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
base_scripts = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "scripts")
if base_scripts not in sys.path:
    sys.path.append(base_scripts)

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import (
    apply_bevel_modifier,
    configure_eevee_for_translucent,
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
)
from _pose_utils import apply_stable_pose, select_pose_tarps
import scene_canonical

# Configuración geométrica
CAM_CEN_Z_MM = 300.0
FOCAL_MM = 55.0
SENSOR_MM = 36.0
FOV_WIDTH_MM = CAM_CEN_Z_MM * (SENSOR_MM / FOCAL_MM)  # ~196.36 mm (1.9636 BU)
FOV_WIDTH_BU = FOV_WIDTH_MM / 100.0

KEYPOINTS_PATH = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "data", "canonical_keypoints.json")

def _world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def compute_bbox_yolo(obj, camera, scene):
    bbox_world = _world_bbox(obj)
    xs, ys = [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(1.0 - co.y)  # YOLO Y down
    x1 = max(0.0, min(xs)); x2 = min(1.0, max(xs))
    y1 = max(0.0, min(ys)); y2 = min(1.0, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 < 0 or x1 > 1 or y2 < 0 or y1 > 1:
        return None
    x1 = max(0.0, min(1.0, x1))
    x2 = max(0.0, min(1.0, x2))
    y1 = max(0.0, min(1.0, y1))
    y2 = max(0.0, min(1.0, y2))
    
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w = x2 - x1; h = y2 - y1
    if w < 1e-3 or h < 1e-3:
        return None
    return (cx, cy, w, h)

def project_keypoints(obj, kps_local_bu, camera, scene):
    out = []
    for kp_local in kps_local_bu:
        v_local = mathutils.Vector(kp_local)
        v_world = obj.matrix_world @ v_local
        co = world_to_camera_view(scene, camera, v_world)
        x_norm = float(co.x)
        y_norm = float(1.0 - co.y)
        depth = float(co.z)
        in_frame = (0.0 <= x_norm <= 1.0) and (0.0 <= y_norm <= 1.0) and (depth > 0)
        v = 2 if in_frame else 0
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        out.append((x_norm, y_norm, v))
    return out

def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "yolo_dataset"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    pa = parser.parse_args(args)

    random.seed(pa.seed + pa.worker_id)

    # Subcarpetas YOLO
    images_dir_cen = os.path.join(pa.output_dir, "yolo_dataset_cenital", "images", pa.split)
    labels_dir_cen = os.path.join(pa.output_dir, "yolo_dataset_cenital", "labels", pa.split)
    images_dir_front = os.path.join(pa.output_dir, "yolo_dataset_frontal", "images", pa.split)
    labels_dir_front = os.path.join(pa.output_dir, "yolo_dataset_frontal", "labels", pa.split)

    os.makedirs(images_dir_cen, exist_ok=True)
    os.makedirs(labels_dir_cen, exist_ok=True)
    os.makedirs(images_dir_front, exist_ok=True)
    os.makedirs(labels_dir_front, exist_ok=True)

    # Cargar keypoints
    if not os.path.exists(KEYPOINTS_PATH):
        print(f"[ERROR] No existe {KEYPOINTS_PATH}")
        sys.exit(1)
    with open(KEYPOINTS_PATH, "r") as f:
        kps_data = json.load(f)
    kps_by_ref = {ref: data["keypoints_bu"] for ref, data in kps_data["pieces"].items()}

    # Cargar poses estables
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    if not os.path.isfile(cache_path):
        print(f"[ERROR] No se encuentra {cache_path}")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        pose_cache = json.load(f)

    from core.db.set_catalog import REAL_SETS
    
    # Recopilar todos los pares del set 75078-1
    real_pairs = []
    set_data = REAL_SETS.get("75078-1", {})
    for part in set_data.get("parts", []):
        ref = part["ref"]
        cc = str(part.get("color_code", ""))
        ch = part.get("color_hex", "")
        cn = part.get("color_name", "Unknown")
        if ref in pose_cache and ref in kps_by_ref and ch and ch.startswith("#") and len(ch) == 7:
            real_pairs.append({
                "ref": ref,
                "color_code": cc,
                "color_hex": ch,
                "color_name": cn
            })

    if not real_pairs:
        print("[ERROR] No se encontraron combinaciones en REAL_SETS para 75078-1 con keypoints.")
        sys.exit(1)

    # Muestrear num_pieces
    selected_samples = []
    shuffled_pairs = real_pairs.copy()
    random.shuffle(shuffled_pairs)
    for pair in shuffled_pairs:
        item = pair.copy()
        poses = pose_cache[item["ref"]]
        item["pose"] = select_pose_tarps(poses, min_tipping=0.05, part_ref=item["ref"])
        selected_samples.append(item)
    while len(selected_samples) < pa.num_pieces:
        item = random.choice(real_pairs).copy()
        poses = pose_cache[item["ref"]]
        item["pose"] = select_pose_tarps(poses, min_tipping=0.05, part_ref=item["ref"])
        selected_samples.append(item)
    if pa.num_pieces < len(selected_samples):
        selected_samples = selected_samples[:pa.num_pieces]

    # Sharding por Worker
    chunk_size = math.ceil(len(selected_samples) / pa.num_workers)
    my_start = pa.worker_id * chunk_size
    my_end = min(my_start + chunk_size, len(selected_samples))
    my_samples = selected_samples[my_start:my_end]

    print(f"Worker {pa.worker_id}/{pa.num_workers} procesará {len(my_samples)} piezas.")

    # Inicializar escena de Blender
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=2048, film_transparent=False)
    scene = bpy.context.scene

    # Generar clases deterministas
    unique_refs = sorted(list(pose_cache.keys()))
    
    # Guardar classes.txt si es el worker 0
    if pa.worker_id == 0:
        with open(os.path.join(pa.output_dir, "classes.txt"), "w") as f:
            for r in unique_refs:
                f.write(f"{r}\n")

    frame_idx = 0
    for s_idx, item in enumerate(my_samples):
        ref = item["ref"]
        pose = item["pose"]
        color_hex = item["color_hex"]

        print(f"  [Piece {s_idx+1}/{len(my_samples)}] Importando {ref} con color {color_hex}...")
        part_obj = scene_canonical.import_part(ref)
        if not part_obj:
            print(f"    [ERROR] No se pudo importar {ref}")
            continue

        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        scene_canonical.normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)
        apply_stable_pose(part_obj, pose)

        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)

        # ------------------------------------------------------------------ #
        # Leer la altura del centroide que apply_stable_pose ya fijó (snap a Z=0)
        # ------------------------------------------------------------------ #
        # apply_stable_pose snaps the bottom of the piece to Z=snap_offset_bu (0.02 BU).
        # We read the centroid Z directly so Phase 1 keeps the piece on the belt.
        z_on_belt = part_obj.location.z   # centroid Z above belt surface

        # For the frontal FOV bottom limit, half_height of the piece is needed
        bbox_corners = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
        z_half = (max(c.z for c in bbox_corners) - min(c.z for c in bbox_corners)) / 2.0

        # ------------------------------------------------------------------ #
        # FASE 1: Recorrido completo del FOV cenital en el eje X
        #   - De X = -HALF_FOV_BU  (entrada en FOV cenital)
        #     a X = +HALF_FOV_BU   (salida de FOV cenital = borde de cinta)
        #   - Z constante = z_on_belt (pieza apoyada sobre la cinta)
        #   - Y = 0.0 (eje de avance central de la cinta)
        # ------------------------------------------------------------------ #
        N_STEPS_BELT = 8  # 8 frames mientras la pieza recorre el FOV cenital

        # Extremos del FOV cenital
        x_fov_left  = -scene_canonical.HALF_FOV_BU
        x_fov_right = +scene_canonical.HALF_FOV_BU

        belt_xs = [x_fov_left + i * (x_fov_right - x_fov_left) / (N_STEPS_BELT - 1)
                   for i in range(N_STEPS_BELT)]

        # ------------------------------------------------------------------ #
        # FASE 2: Caída parabólica tras abandonar la cinta (exactamente 2 frames)
        #   - Para una velocidad de 5 m/min (83.33 mm/s), usamos un dt de caída de
        #     0.033 s (equivalente a 30 fps en caída) para mantener la pieza en el FOV.
        #   - g = 9.81 m/s^2 = 9810 mm/s^2 = 98.1 BU/s^2
        #   - V_x = 83.33 mm/s = 0.8333 BU/s
        # ------------------------------------------------------------------ #
        g_bu = 98.1
        vx_bu = 0.8333
        dt_fall_1 = 0.033
        dt_fall_2 = 0.066

        fall_coords = [
            (x_fov_right + vx_bu * dt_fall_1, z_on_belt - 0.5 * g_bu * (dt_fall_1 ** 2)),
            (x_fov_right + vx_bu * dt_fall_2, z_on_belt - 0.5 * g_bu * (dt_fall_2 ** 2))
        ]

        # ------------------------------------------------------------------ #
        # Generar renders y anotaciones
        # ------------------------------------------------------------------ #
        step_idx = 0

        # --- Fase 1: sobre la cinta ---
        for x_bu in belt_xs:
            part_obj.location.x = x_bu
            part_obj.location.y = 0.0
            part_obj.location.z = z_on_belt
            bpy.context.view_layer.update()

            uuid_str = str(uuid.uuid4())[:8]
            img_name = f"w{pa.worker_id}_p{s_idx:03d}_s{step_idx:02d}_{uuid_str}.png"
            txt_name = f"w{pa.worker_id}_p{s_idx:03d}_s{step_idx:02d}_{uuid_str}.txt"

            # Render Cenital
            scene.camera = cam_cenital
            scene.render.filepath = os.path.join(images_dir_cen, img_name)
            bpy.ops.render.render(write_still=True)
            bbox_cen = compute_bbox_yolo(part_obj, cam_cenital, scene)
            if bbox_cen:
                kps_local = kps_by_ref[ref]
                kps_2d = project_keypoints(part_obj, kps_local, cam_cenital, scene)
                with open(os.path.join(labels_dir_cen, txt_name), "w") as f:
                    line = f"0 {bbox_cen[0]:.6f} {bbox_cen[1]:.6f} {bbox_cen[2]:.6f} {bbox_cen[3]:.6f}"
                    for (kx, ky, kv) in kps_2d:
                        line += f" {kx:.6f} {ky:.6f} {kv}"
                    f.write(line + "\n")

            # Render Frontal
            scene.camera = cam_frontal
            scene.render.filepath = os.path.join(images_dir_front, img_name)
            bpy.ops.render.render(write_still=True)
            bbox_front = compute_bbox_yolo(part_obj, cam_frontal, scene)
            if bbox_front:
                kps_local = kps_by_ref[ref]
                kps_2d = project_keypoints(part_obj, kps_local, cam_frontal, scene)
                with open(os.path.join(labels_dir_front, txt_name), "w") as f:
                    line = f"0 {bbox_front[0]:.6f} {bbox_front[1]:.6f} {bbox_front[2]:.6f} {bbox_front[3]:.6f}"
                    for (kx, ky, kv) in kps_2d:
                        line += f" {kx:.6f} {ky:.6f} {kv}"
                    f.write(line + "\n")

            step_idx += 1

        # --- Fase 2: caída parabólica (fuera de FOV cenital, en FOV frontal) ---
        for x_fall, z_fall in fall_coords:
            part_obj.location.x = x_fall
            part_obj.location.y = 0.0
            part_obj.location.z = z_fall
            bpy.context.view_layer.update()

            uuid_str = str(uuid.uuid4())[:8]
            img_name = f"w{pa.worker_id}_p{s_idx:03d}_s{step_idx:02d}_{uuid_str}.png"
            txt_name = f"w{pa.worker_id}_p{s_idx:03d}_s{step_idx:02d}_{uuid_str}.txt"

            # Cenital: pieza ya fuera del FOV, NO renderizamos para no contaminar el dataset
            # con frames vacíos. Escribimos imagen vacía solo si hace falta por coherencia de steps.
            # En la práctica dejamos que el dataset cenital solo tenga frames útiles (Fase 1).

            # Render Frontal (pieza en caída – muy útil para estimar altura)
            scene.camera = cam_frontal
            scene.render.filepath = os.path.join(images_dir_front, img_name)
            bpy.ops.render.render(write_still=True)
            bbox_front = compute_bbox_yolo(part_obj, cam_frontal, scene)
            if bbox_front:
                kps_local = kps_by_ref[ref]
                kps_2d = project_keypoints(part_obj, kps_local, cam_frontal, scene)
                with open(os.path.join(labels_dir_front, txt_name), "w") as f:
                    line = f"0 {bbox_front[0]:.6f} {bbox_front[1]:.6f} {bbox_front[2]:.6f} {bbox_front[3]:.6f}"
                    for (kx, ky, kv) in kps_2d:
                        line += f" {kx:.6f} {ky:.6f} {kv}"
                    f.write(line + "\n")

            step_idx += 1

        print(f"    [OK] {ref}: {N_STEPS_BELT} frames sobre cinta + {len(fall_frames)} frames en caída = {step_idx} total")

        # Limpiar
        bpy.data.objects.remove(part_obj, do_unlink=True)

    print(f"[Done] Worker {pa.worker_id} completado.")

if __name__ == "__main__":
    main()
