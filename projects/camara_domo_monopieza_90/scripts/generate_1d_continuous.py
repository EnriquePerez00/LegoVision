# -*- coding: utf-8 -*-
"""
camara_domo_monopieza_90/scripts/generate_1d_continuous.py
===================================================
Simula una cinta transportadora a alta velocidad (5 m/s) y realiza empaquetado 1D
de piezas aleatorias con un espacio mínimo de 5 mm. Captura continuamente.

Uso:
    /opt/homebrew/bin/blender -b -P \
        projects/camara_domo_monopieza_90/scripts/generate_1d_continuous.py -- \
            --num_pieces 1000 \
            --output_dir projects/camara_domo_monopieza_90/data/simulation_1000 \
            --speed 5.0 \
            --step_mm 40.0
"""
import os
import sys
import json
import math
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
# Add base scripts folder to sys.path
base_scripts = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "scripts")
if base_scripts not in sys.path:
    sys.path.append(base_scripts)

try:
    import bpy
    import mathutils
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
FOV_WIDTH_MM = CAM_CEN_Z_MM * (SENSOR_MM / FOCAL_MM)  # ~196.36 mm
BELT_WIDTH_MM = 200.0
MARGIN_MM = 5.0

def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "simulation_1000"))
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--step_mm", type=float, default=40.0, help="Desplazamiento de la cinta entre frames (mm).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None, help="Límite máximo de frames a renderizar por proceso (útil para validación inicial).")
    pa = parser.parse_args(args)

    random.seed(pa.seed)
    os.makedirs(pa.output_dir, exist_ok=True)

    # 1. Cargar cache de poses estables (que contiene TODAS las piezas con modelo y pose)
    cache_path = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
    if not os.path.isfile(cache_path):
        print(f"[ERROR] No se encuentra {cache_path}")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        pose_cache = json.load(f)

    # Lista de colores básicos disponibles para asignar aleatoriamente
    COLORS = [
        {"color_code": "1", "color_hex": "#F4F4F4", "color_name": "White"},
        {"color_code": "5", "color_hex": "#D67240", "color_name": "Brick Yellow"},
        {"color_code": "21", "color_hex": "#C91A09", "color_name": "Bright Red"},
        {"color_code": "23", "color_hex": "#11589D", "color_name": "Bright Blue"},
        {"color_code": "24", "color_hex": "#F2CD37", "color_name": "Bright Yellow"},
        {"color_code": "26", "color_hex": "#1B2A34", "color_name": "Black"},
        {"color_code": "28", "color_hex": "#008F9B", "color_name": "Dark Green"},
        {"color_code": "119", "color_hex": "#95B90B", "color_name": "Bright Yellowish Green"},
        {"color_code": "199", "color_hex": "#4C6171", "color_name": "Dark Stone Grey"},
        {"color_code": "208", "color_hex": "#E4ADC8", "color_name": "Light Nougat"},
    ]

    all_refs = list(pose_cache.keys())
    if not all_refs:
        print("[ERROR] No se encontraron referencias en pose_cache.")
        sys.exit(1)

    # Seleccionar aleatoriamente las piezas de cualquier set en la BD
    selected_samples = []
    for _ in range(pa.num_pieces):
        ref = random.choice(all_refs)
        poses = pose_cache[ref]
        pose = select_pose_tarps(poses, min_tipping=0.05, part_ref=ref)
        color = random.choice(COLORS)
        selected_samples.append({
            "ref": ref,
            "pose": pose,
            "color_code": color["color_code"],
            "color_hex": color["color_hex"],
            "color_name": color["color_name"]
        })

    print(f"[Sim] Seleccionadas {len(selected_samples)} piezas aleatorias de toda la base de datos.")

    # Inicializar escena canónica de Blender
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=2048, film_transparent=False)
    scene = bpy.context.scene
    scene.camera = cam_cenital

    enable_metal_gpu_acceleration()

    # 2. Importar y medir dimensiones físicas de cada pieza en su pose estable
    placed_objects = []
    for idx, item in enumerate(selected_samples):
        ref = item["ref"]
        pose = item["pose"]
        color_hex = item["color_hex"]

        # Importar y configurar la pieza en el origen
        part_obj = scene_canonical.import_part(ref)
        if not part_obj:
            print(f"  [WARNING] No se pudo importar la pieza {ref}, se omitirá.")
            continue

        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        scene_canonical.normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)
        apply_stable_pose(part_obj, pose)
        
        # Crear material ABS
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)

        # Medir dimensiones físicas en mm (1 BU = 100 mm)
        bbox_corners = [part_obj.matrix_world @ mathutils.Vector(corner) for corner in part_obj.bound_box]
        xs = [c.x for c in bbox_corners]
        ys = [c.y for c in bbox_corners]
        zs = [c.z for c in bbox_corners]
        
        len_x_mm = (max(xs) - min(xs)) * 100.0
        width_y_mm = (max(ys) - min(ys)) * 100.0
        height_z_mm = (max(zs) - min(zs)) * 100.0

        placed_objects.append({
            "index": idx,
            "ref": ref,
            "obj": part_obj,
            "len_x": len_x_mm,
            "width_y": width_y_mm,
            "height_z": height_z_mm,
            "color_hex": color_hex,
            "color_code": item["color_code"],
            "color_name": item["color_name"],
            "pose": pose
        })

    # Activar CPL Filter (Polarizacion) en todos los materiales
    for mat in bpy.data.materials:
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                if "Specular IOR Level" in bsdf.inputs:
                    bsdf.inputs["Specular IOR Level"].default_value = 0.0
                elif "Specular" in bsdf.inputs:
                    bsdf.inputs["Specular"].default_value = 0.0
                bsdf.inputs["Roughness"].default_value = 1.0
                
    # 3. Algoritmo de empaquetado 1D lineal en la cinta (piezas centradas con 5mm de separación mínima)
    current_x = 0.0
    packed_pieces = []
    
    # La posición inicial en x para la primera pieza (que su borde empiece en 0)
    for i, item in enumerate(placed_objects):
        w_x = item["len_x"]

        if i == 0:
            center_x = w_x * 0.5
        else:
            prev_item = packed_pieces[-1]
            # Borde derecho de la pieza anterior
            prev_right_edge = prev_item["x_belt"] + (prev_item["meta"]["len_x"] * 0.5)
            # El centro de la pieza actual es: borde_derecho_anterior + MARGIN_MM (5mm) + mitad de tamaño actual
            center_x = prev_right_edge + MARGIN_MM + (w_x * 0.5)

        center_y = 0.0 # Centrado en el eje Y
        
        packed_pieces.append({
            "meta": item,
            "x_belt": center_x,
            "y_belt": center_y,
            "z_belt": item["obj"].location.z,
            "obj": item["obj"]
        })

    print(f"\n[Packing] {len(packed_pieces)} piezas empaquetadas en la cinta (gap = {MARGIN_MM}mm).")

    # 4. Calcular los offsets de disparo (Modo de captura continua)
    start_offset = min(p["x_belt"] for p in packed_pieces) - (FOV_WIDTH_MM / 2.0)
    end_offset = max(p["x_belt"] for p in packed_pieces) + (FOV_WIDTH_MM / 2.0)
    
    unique_shoot_offsets = []
    offset = start_offset
    while offset <= end_offset:
        unique_shoot_offsets.append(offset)
        offset += pa.step_mm

    print(f"\n[Sim] Se calcularon {len(unique_shoot_offsets)} offsets de disparo (captura continua a {pa.step_mm}mm).")

    # 5. Particionar los offsets si se ejecuta en paralelo
    total_frames = len(unique_shoot_offsets)
    
    if pa.max_frames and total_frames > pa.max_frames:
        print(f"[Sim] Limitando render a los primeros {pa.max_frames} frames (de {total_frames}).")
        total_frames = pa.max_frames
        
    if pa.num_workers > 1:
        chunk_size = math.ceil(total_frames / pa.num_workers)
        start_idx = pa.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        worker_indices = list(range(start_idx, end_idx))
        print(f"[Sim] Worker {pa.worker_id}/{pa.num_workers} procesará frames del {start_idx} al {end_idx-1}.")
    else:
        worker_indices = list(range(total_frames))

    # 6. Renderizar / Procesar metadatos
    frames_meta = []
    for f_idx in range(total_frames):
        offset = unique_shoot_offsets[f_idx]
        is_my_job = (f_idx in worker_indices)
        
        # Desplazar piezas en X según el offset de la cinta y aplicar física de caída
        for p in packed_pieces:
            x_world_mm = offset - p["x_belt"]
            p["obj"].location.x = x_world_mm / 100.0
            p["obj"].location.y = p["y_belt"] / 100.0
            
            x_fov_right_mm = FOV_WIDTH_MM * 0.5
            if x_world_mm > x_fov_right_mm:
                d_fall_mm = x_world_mm - x_fov_right_mm
                speed_mm_s = pa.speed * 1000.0
                t = d_fall_mm / speed_mm_s
                g_bu = 98.1
                p["obj"].location.z = p["z_belt"] - 0.5 * g_bu * (t ** 2)
            else:
                p["obj"].location.z = p["z_belt"]
        
        if is_my_job:
            bpy.context.view_layer.update()

        visible_pieces_in_frame = []
        for p in packed_pieces:
            x_world_mm = offset - p["x_belt"]
            half_fov = FOV_WIDTH_MM * 0.5
            is_visible = (-half_fov - 50.0 <= x_world_mm <= half_fov + 50.0)
            
            if is_my_job and not pa.metadata_only:
                p["obj"].hide_render = not is_visible

            if is_visible:
                # Actualizar bbox
                bbox_cen = scene_canonical.get_2d_bbox(p["obj"], scene, cam_cenital)
                bbox_front = scene_canonical.get_2d_bbox(p["obj"], scene, cam_frontal)
                visible_pieces_in_frame.append({
                    "ref": p["meta"]["ref"],
                    "color_code": p["meta"]["color_code"],
                    "color_name": p["meta"]["color_name"],
                    "bbox_cenital_norm": bbox_cen,
                    "bbox_frontal_norm": bbox_front,
                    "x_belt_local_mm": x_world_mm,
                    "y_belt_local_mm": p["y_belt"],
                    "zenith_silhouette_area_gt": p["meta"]["pose"].get("zenith_silhouette_area"),
                    "lateral_height_gt": p["meta"]["pose"].get("lateral_height")
                })

        frame_name = f"frame_{f_idx:05d}.png"
        frame_frontal_name = f"frame_{f_idx:05d}_frontal.png"

        if is_my_job and not pa.metadata_only:
            # Renderizar frame cenital
            scene.camera = cam_cenital
            scene.render.filepath = os.path.join(pa.output_dir, frame_name)
            bpy.ops.render.render(write_still=True)

            # Renderizar frame frontal
            scene.camera = cam_frontal
            scene.render.filepath = os.path.join(pa.output_dir, frame_frontal_name)
            bpy.ops.render.render(write_still=True)
            print(f"  Worker {pa.worker_id} Renderizó {frame_name} con {len(visible_pieces_in_frame)} piezas en FoV.")

        frames_meta.append({
            "frame_index": f_idx,
            "belt_offset_mm": offset,
            "file_name": frame_name,
            "file_name_frontal": frame_frontal_name,
            "visible_pieces": visible_pieces_in_frame
        })

    def extract_camera_matrices(cam, scene):
        depsgraph = bpy.context.evaluated_depsgraph_get()
        P = cam.calc_matrix_camera(depsgraph, x=scene.render.resolution_x, y=scene.render.resolution_y)
        Rt = cam.matrix_world.inverted()
        return {"P": [list(row) for row in P], "Rt": [list(row) for row in Rt]}
        
    cam_matrices = {
        "cenital": extract_camera_matrices(cam_cenital, scene),
        "lateral": extract_camera_matrices(cam_frontal, scene)
    }

    if (pa.metadata_only or pa.num_workers == 1) and pa.worker_id == 0:
        meta_output = {
            "belt_speed_m_s": pa.speed,
            "conveyor_width_mm": BELT_WIDTH_MM,
            "fov_width_mm": FOV_WIDTH_MM,
            "capture_step_mm": pa.step_mm,
            "total_renders": len(unique_shoot_offsets),
            "camera_matrices": cam_matrices,
            "frames": frames_meta
        }
        with open(os.path.join(pa.output_dir, "simulation_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_output, f, indent=2, ensure_ascii=False)
        print(f"[Sim] Metadatos escritos en: {os.path.join(pa.output_dir, 'simulation_metadata.json')}")

    print(f"\n[Sim] Simulación completada con éxito para Worker {pa.worker_id}.")

if __name__ == "__main__":
    main()
