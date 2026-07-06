# -*- coding: utf-8 -*-
"""camara_domo/scripts/generate_conveyor_simulation.py
===================================================
Simula una cinta transportadora a alta velocidad (5 m/s) y realiza empaquetado 2D.
Genera la cantidad óptima de renders necesarios para capturar exactamente 5 fotos
equidistantes de cada pieza cuando cruza el FoV cenital.

Uso:
    /opt/homebrew/bin/blender -b -P \
        camara_domo/scripts/generate_conveyor_simulation.py -- \
            --num_pieces 6 \
            --output_dir camara_domo/data/simulation_run \
            --speed 5.0
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
    parser.add_argument("--num_pieces", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "simulation_run"))
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frames_per_piece", type=int, default=0)
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--set-id", type=str, default="75078-1", help="ID del set LEGO para generar las piezas, o 'random' para piezas de múltiples sets.")
    parser.add_argument("--randomize_lighting", action="store_true", help="Si se activa, aleatoriza la iluminación del domo por frame.")
    parser.add_argument("--dimension", type=str, default="1D", choices=["1D", "2D"], help="1D para línea simple, 2D para abarcar ancho de la cinta")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolución de renderizado (ancho/alto en píxeles).")
    pa = parser.parse_args(args)


    random.seed(pa.seed)
    os.makedirs(pa.output_dir, exist_ok=True)

    # 1. Cargar cache de poses estables y base de datos de sets
    cache_path = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
    if not os.path.isfile(cache_path):
        print(f"[ERROR] No se encuentra {cache_path}")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        pose_cache = json.load(f)

    from core.db.set_catalog import REAL_SETS
    
    # Recopilar todos los pares reales disponibles
    real_pairs = []
    if pa.set_id == "random":
        for s_id, s_data in REAL_SETS.items():
            for part in s_data.get("parts", []):
                ref = part["ref"]
                cc = str(part.get("color_code", ""))
                ch = part.get("color_hex", "")
                cn = part.get("color_name", "Unknown")
                if ref in pose_cache and ch and ch.startswith("#") and len(ch) == 7:
                    real_pairs.append({
                        "ref": ref,
                        "color_code": cc,
                        "color_hex": ch,
                        "color_name": cn
                    })
    else:
        set_data = REAL_SETS.get(pa.set_id, {})
        for part in set_data.get("parts", []):
            ref = part["ref"]
            cc = str(part.get("color_code", ""))
            ch = part.get("color_hex", "")
            cn = part.get("color_name", "Unknown")
            if ref in pose_cache and ch and ch.startswith("#") and len(ch) == 7:
                real_pairs.append({
                    "ref": ref,
                    "color_code": cc,
                    "color_hex": ch,
                    "color_name": cn
                })

    if not real_pairs:
        print(f"[ERROR] No se encontraron combinaciones de piezas en REAL_SETS para set-id {pa.set_id}.")
        sys.exit(1)


    # Seleccionar X piezas de manera que estén todos los pares de real_pairs al menos una vez
    selected_samples = []
    
    # Copiar y barajar para que la asignación de poses iniciales sea aleatoria
    shuffled_pairs = real_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Asegurar que al menos una pieza de cada combinación existe
    for pair in shuffled_pairs:
        item = pair.copy()
        poses = pose_cache[item["ref"]]
        item["pose"] = select_pose_tarps(poses, min_tipping=0.05, part_ref=item["ref"])
        selected_samples.append(item)
        
    # Llenar el resto aleatoriamente hasta alcanzar pa.num_pieces (ej. 100)
    while len(selected_samples) < pa.num_pieces:
        item = random.choice(real_pairs).copy()
        poses = pose_cache[item["ref"]]
        item["pose"] = select_pose_tarps(poses, min_tipping=0.05, part_ref=item["ref"])
        selected_samples.append(item)
        
    # Recortar si pa.num_pieces es menor que el total de combinaciones
    if pa.num_pieces < len(selected_samples):
        selected_samples = selected_samples[:pa.num_pieces]

    print(f"[Sim] Seleccionadas {len(selected_samples)} piezas para la simulación del set 75078-1 (garantizando cobertura de combinaciones).")

    # Inicializar escena canónica de Blender
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=pa.resolution, film_transparent=False, randomize_lighting=pa.randomize_lighting)
    scene = bpy.context.scene
    scene.camera = cam_cenital

    # 2. Medir dimensiones físicas de cada pieza en su pose estable
    placed_objects = []
    for idx, item in enumerate(selected_samples):
        ref = item["ref"]
        pose = item["pose"]
        color_hex = item["color_hex"]

        # Importar y configurar la pieza en el origen
        part_obj = scene_canonical.import_part(ref)
        if not part_obj:
            print(f"  [ERROR] No se pudo importar la pieza {ref}")
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


    # 2b. Activar CPL Filter (Polarizacion) en todos los materiales
    for mat in bpy.data.materials:
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                if "Specular IOR Level" in bsdf.inputs:
                    bsdf.inputs["Specular IOR Level"].default_value = 0.0
                elif "Specular" in bsdf.inputs:
                    bsdf.inputs["Specular"].default_value = 0.0
                bsdf.inputs["Roughness"].default_value = 1.0
                
    # 3. Algoritmo de empaquetado (1D lineal o 2D shelf-packing)

    packed_pieces = []

    if pa.dimension == "1D":
        current_x = 0.0
        for item in placed_objects:
            w_x = item["len_x"]

            # Espacio requerido incluyendo margen mínimo de 5.0 mm (0.5 cm) entre extremos
            req_x = w_x + MARGIN_MM

            # Posición del centro de la pieza en la cinta, Y fijo en el centro (0.0)
            center_x = current_x + (w_x * 0.5)
            center_y = 0.0

            packed_pieces.append({
                "meta": item,
                "x_belt": center_x,
                "y_belt": center_y,
                "z_belt": item["obj"].location.z,
                "obj": item["obj"]
            })

            current_x += req_x
    else:
        current_x = 0.0
        current_y_start = -BELT_WIDTH_MM / 2.0 + MARGIN_MM
        current_y = current_y_start
        row_max_x = 0.0

        for item in placed_objects:
            w_x = item["len_x"]
            w_y = item["width_y"]

            req_x = w_x + MARGIN_MM
            req_y = w_y + MARGIN_MM

            if current_y + w_y > BELT_WIDTH_MM / 2.0 - MARGIN_MM:
                if row_max_x == 0.0:
                    row_max_x = req_x
                current_x += row_max_x
                current_y = current_y_start
                row_max_x = 0.0

            center_x = current_x + (w_x * 0.5)
            center_y = current_y + (w_y * 0.5)

            packed_pieces.append({
                "meta": item,
                "x_belt": center_x,
                "y_belt": center_y,
                "z_belt": item["obj"].location.z,
                "obj": item["obj"]
            })

            current_y += req_y
            row_max_x = max(row_max_x, req_x)

    print("\n[Packing] Piezas empaquetadas en la cinta:")
    for p in packed_pieces:
        print(f"  - Ref {p['meta']['ref']}: x_belt={p['x_belt']:.1f} mm, y_belt={p['y_belt']:.1f} mm")

    # 4. Calcular los offsets de disparo de la cinta
    if pa.frames_per_piece > 0:
        # Calcular step_size constante para que cada pieza aparezca en max pa.frames_per_piece frames
        mean_len_x = sum(p["meta"]["len_x"] for p in packed_pieces) / len(packed_pieces)
        visible_range = FOV_WIDTH_MM + mean_len_x
        # step size to get approximately frames_per_piece visibility (using frames_per_piece - 0.5 for safe margin)
        step_size = visible_range / (pa.frames_per_piece - 0.5)
        
        # Rango de offsets: desde que la primera entra hasta que la última sale del FoV
        start_offset = min(p["x_belt"] for p in packed_pieces) - (FOV_WIDTH_MM / 2.0)
        end_offset = max(p["x_belt"] for p in packed_pieces) + (FOV_WIDTH_MM / 2.0)
        
        offset = start_offset
        unique_shoot_offsets = []
        while offset <= end_offset:
            unique_shoot_offsets.append(offset)
            offset += step_size
    else:
        # Calcular los offsets de disparo de la cinta para las 5 fotos por pieza (original)
        max_size = max(p["meta"]["len_x"] for p in packed_pieces)
        d_visible = FOV_WIDTH_MM - max_size
        delta_x = d_visible / 4.0

        offsets_of_capture = [-2 * delta_x, -1 * delta_x, 0.0, 1 * delta_x, 2 * delta_x]
        
        shoot_offsets = []
        for p in packed_pieces:
            for offset in offsets_of_capture:
                shoot_offsets.append(p["x_belt"] - offset)

        shoot_offsets.sort()
        unique_shoot_offsets = []
        for offset in shoot_offsets:
            if not unique_shoot_offsets or abs(offset - unique_shoot_offsets[-1]) > 1.0:
                unique_shoot_offsets.append(offset)

    print(f"\n[Sim] Se calcularon {len(unique_shoot_offsets)} offsets de disparo.")

    # 5. Particionar los offsets si se ejecuta en paralelo
    total_frames = len(unique_shoot_offsets)
    if pa.num_workers > 1:
        chunk_size = math.ceil(total_frames / pa.num_workers)
        start_idx = pa.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        worker_indices = list(range(start_idx, end_idx))
        print(f"[Sim] Worker {pa.worker_id}/{pa.num_workers} procesará frames del {start_idx} al {end_idx-1}.")
    else:
        worker_indices = list(range(total_frames))

    # 6. Renderizar secuencialmente/procesar metadatos
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
        
        bpy.context.view_layer.update()

        visible_pieces_in_frame = []
        for p in packed_pieces:
            x_world_mm = offset - p["x_belt"]
            half_fov = FOV_WIDTH_MM * 0.5
            is_visible = (-half_fov - 50.0 <= x_world_mm <= half_fov + 50.0)
            
            # Solo ocultamos del render si realmente vamos a renderizar en este proceso
            if is_my_job and not pa.metadata_only:
                p["obj"].hide_render = not is_visible

            if is_visible:
                # Obtener bounding box 2D en la imagen
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

        frame_name = f"frame_{f_idx:03d}.png"
        frame_frontal_name = f"frame_{f_idx:03d}_frontal.png"

        # Solo renderizar si es tarea de este worker y no estamos en metadata_only
        if is_my_job and not pa.metadata_only:
            if pa.randomize_lighting:
                scene_canonical.setup_dome_light(randomize=True)
                
            # Renderizar frame cenital
            scene.camera = cam_cenital
            scene.render.filepath = os.path.join(pa.output_dir, frame_name)
            bpy.ops.render.render(write_still=True)

            # Renderizar frame frontal
            scene.camera = cam_frontal
            scene.render.filepath = os.path.join(pa.output_dir, frame_frontal_name)
            bpy.ops.render.render(write_still=True)
            print(f"  Worker {pa.worker_id} Renderizó {frame_name} con {len(visible_pieces_in_frame)} piezas en FoV.")

        # Guardar en metadatos (siempre, ya que metadata_only generará el JSON global con todos los frames)
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

    # Guardar archivo de metadatos de la simulación solo si es el hilo principal / primer paso
    if pa.metadata_only or pa.num_workers == 1:
        meta_output = {
            "associated_set": pa.set_id,
            "belt_speed_m_s": pa.speed,
            "conveyor_width_mm": BELT_WIDTH_MM,
            "fov_width_mm": FOV_WIDTH_MM,
            "resolution": pa.resolution,
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
