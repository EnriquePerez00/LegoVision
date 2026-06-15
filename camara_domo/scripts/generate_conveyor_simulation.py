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
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
# Add base scripts folder to sys.path
base_scripts = os.path.join(legovic_root, "2camaras_random_pieza_unica", "scripts")
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
from _pose_utils import apply_stable_pose
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
    parser.add_argument("--speed", type=type(5.0), default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    pa = parser.parse_args(args)

    random.seed(pa.seed)
    os.makedirs(pa.output_dir, exist_ok=True)

    # 1. Cargar cache de poses estables y base de datos de sets
    cache_path = os.path.join(legovic_root, "2camaras_random_pieza_unica", "data", "stable_poses_cache.json")
    if not os.path.isfile(cache_path):
        print(f"[ERROR] No se encuentra {cache_path}")
        sys.exit(1)
    with open(cache_path, "r", encoding="utf-8") as f:
        pose_cache = json.load(f)

    from database.set_catalog import REAL_SETS
    
    # Recopilar todos los pares reales disponibles
    real_pairs = []
    for set_id, set_data in REAL_SETS.items():
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
        print("[ERROR] No se encontraron combinaciones de piezas en REAL_SETS.")
        sys.exit(1)

    # Seleccionar X piezas aleatorias
    selected_samples = []
    for _ in range(pa.num_pieces):
        item = random.choice(real_pairs).copy()
        poses = pose_cache[item["ref"]]
        item["pose"] = random.choice(poses)
        selected_samples.append(item)

    print(f"[Sim] Seleccionadas {pa.num_pieces} piezas para la simulación.")

    # Inicializar escena canónica de Blender
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=1024, film_transparent=False)
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

    # 3. Algoritmo de empaquetado 2D (Shelf Packing) en la cinta
    # Rango usable en Y (con 5mm de margen a cada borde): [-95, 95]
    y_min_usable = -95.0
    y_max_usable = 95.0
    usable_width_y = y_max_usable - y_min_usable  # 190 mm

    # Ordenar piezas por ancho Y descendente para un empaquetado más eficiente
    placed_objects.sort(key=lambda x: x["width_y"], reverse=True)

    current_shelf_x = 0.0
    current_shelf_y = y_min_usable
    shelf_thickness_x = 0.0

    packed_pieces = []
    for item in placed_objects:
        w_x = item["len_x"]
        h_y = item["width_y"]

        # Espacio requerido incluyendo margen mínimo de 5mm entre piezas
        req_x = w_x + MARGIN_MM
        req_y = h_y + MARGIN_MM

        # Si supera el ancho usable de la cinta, pasar al siguiente "shelf" en X
        if current_shelf_y + req_y > y_max_usable + MARGIN_MM:
            current_shelf_x += shelf_thickness_x
            current_shelf_y = y_min_usable
            shelf_thickness_x = 0.0

        # Posición del centro de la pieza en la cinta
        center_x = current_shelf_x + (w_x * 0.5)
        center_y = current_shelf_y + (h_y * 0.5)

        packed_pieces.append({
            "meta": item,
            "x_belt": center_x,
            "y_belt": center_y,
            "obj": item["obj"]
        })

        shelf_thickness_x = max(shelf_thickness_x, req_x)
        current_shelf_y += req_y

    print("\n[Packing] Piezas empaquetadas en la cinta:")
    for p in packed_pieces:
        print(f"  - Ref {p['meta']['ref']}: x_belt={p['x_belt']:.1f} mm, y_belt={p['y_belt']:.1f} mm")

    # 4. Calcular los offsets de disparo de la cinta para las 5 fotos por pieza
    # El rango de visibilidad cenital total es FOV_WIDTH_MM - max_piece_size
    max_size = max(p["meta"]["len_x"] for p in packed_pieces)
    d_visible = FOV_WIDTH_MM - max_size
    delta_x = d_visible / 4.0

    offsets_of_capture = [-2 * delta_x, -1 * delta_x, 0.0, 1 * delta_x, 2 * delta_x]
    
    shoot_offsets = []
    for p in packed_pieces:
        for offset in offsets_of_capture:
            shoot_offsets.append(p["x_belt"] - offset)

    # Ordenar y consolidar disparos que estén a menos de 1mm (co-disparos eficientes)
    shoot_offsets.sort()
    unique_shoot_offsets = []
    for offset in shoot_offsets:
        if not unique_shoot_offsets or abs(offset - unique_shoot_offsets[-1]) > 1.0:
            unique_shoot_offsets.append(offset)

    print(f"\n[Sim] Se realizarán {len(unique_shoot_offsets)} renders secuenciales para capturar todas las piezas.")

    # 5. Renderizar secuencialmente simulando el avance de la cinta
    frames_meta = []
    for f_idx, offset in enumerate(unique_shoot_offsets):
        # Desplazar piezas en X según el offset de la cinta (y convertir mm a Blender Units)
        visible_pieces_in_frame = []
        
        for p in packed_pieces:
            x_world_mm = p["x_belt"] - offset
            p["obj"].location.x = x_world_mm / 100.0
            p["obj"].location.y = p["y_belt"] / 100.0
            # Asegurarse de que el objeto esté actualizado en la vista de Blender
            bpy.context.view_layer.update()

            # Determinar si la pieza está dentro del FoV (cenital)
            # FoV abarca [-FOV_WIDTH_MM/2, FOV_WIDTH_MM/2] en X e Y
            half_fov = FOV_WIDTH_MM * 0.5
            is_visible = (abs(x_world_mm) <= half_fov)
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

        # Renderizar frame cenital
        frame_name = f"frame_{f_idx:03d}.png"
        out_path = os.path.join(pa.output_dir, frame_name)
        scene.camera = cam_cenital
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)

        # Renderizar frame frontal
        frame_frontal_name = f"frame_{f_idx:03d}_frontal.png"
        out_frontal_path = os.path.join(pa.output_dir, frame_frontal_name)
        scene.camera = cam_frontal
        scene.render.filepath = out_frontal_path
        bpy.ops.render.render(write_still=True)

        frames_meta.append({
            "frame_index": f_idx,
            "belt_offset_mm": offset,
            "file_name": frame_name,
            "file_name_frontal": frame_frontal_name,
            "visible_pieces": visible_pieces_in_frame
        })
        print(f"  Renderizado {frame_name} (offset cinta: {offset:.1f} mm) con {len(visible_pieces_in_frame)} piezas en el FoV.")

    # Guardar archivo de metadatos de la simulación
    meta_output = {
        "belt_speed_m_s": pa.speed,
        "conveyor_width_mm": BELT_WIDTH_MM,
        "fov_width_mm": FOV_WIDTH_MM,
        "total_renders": len(unique_shoot_offsets),
        "frames": frames_meta
    }
    
    with open(os.path.join(pa.output_dir, "simulation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_output, f, indent=2, ensure_ascii=False)

    print(f"\n[Sim] Simulación completada con éxito. Resultados en: {pa.output_dir}")

if __name__ == "__main__":
    main()
