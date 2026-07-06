# -*- coding: utf-8 -*-
"""projects/camara_domo/scripts/generate_data200.py
==================================================
Script de Blender para generar el dataset data200.
Selecciona 200 piezas aleatorias de la base de datos con mesh LDraw local,
simula su trayectoria en la cinta transportadora, tomando al menos 5 fotos
dentro del FoV cenital, y fotos adicionales hasta que la pieza caiga del extremo (+X).
Renderiza vistas Cenital y Frontal/Lateral.
"""
import os
import sys
import json
import math
import random
import argparse

# Configurar paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)

# Añadir scripts del subproyecto base
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
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
    get_ldraw_part_path,
)
from _pose_utils import apply_stable_pose
import scene_canonical
from core.db.supabase_client import get_connection

def query_stable_pieces_from_db():
    """Consulta todas las piezas estables de la base de datos."""
    print("[DB] Consultando piezas estables...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT part_ref 
                FROM stable_poses 
                WHERE is_stable = TRUE
            """)
            rows = cur.fetchall()
    return [r["part_ref"] for r in rows]

def query_poses_for_piece(part_ref):
    """Consulta todas las poses estables para una pieza específica."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pose_index, face_class, contact_normal, orientation_quat, orientation_euler,
                       zenith_observable_area, zenith_silhouette_area, lateral_height
                FROM stable_poses
                WHERE part_ref = %s AND is_stable = TRUE
            """, (part_ref,))
            rows = cur.fetchall()
    return rows

def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=200)
    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data", "data200"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test_run", action="store_true")
    pa = parser.parse_args(args)

    if pa.test_run:
        pa.num_pieces = min(pa.num_pieces, 2)
        pa.res = 256
        print("[Mode] RUNNING IN TEST MODE (res=256, num_pieces=2)")

    random.seed(pa.seed)
    os.makedirs(pa.output_dir, exist_ok=True)

    # 1. Obtener todas las piezas estables y filtrar las que tienen mesh local
    db_parts = query_stable_pieces_from_db()
    valid_parts = []
    for ref in db_parts:
        path = get_ldraw_part_path(ref)
        if path:
            valid_parts.append(ref)

    print(f"[Init] Encontradas {len(valid_parts)} piezas con mesh local de las {len(db_parts)} en DB.")
    if len(valid_parts) < pa.num_pieces:
        print(f"[Warning] Se pidieron {pa.num_pieces} piezas pero solo hay {len(valid_parts)} válidas. Usando todas.")
        pa.num_pieces = len(valid_parts)

    selected_parts = random.sample(valid_parts, pa.num_pieces)
    print(f"[Init] Seleccionadas {len(selected_parts)} piezas aleatorias para renderizar.")

    # 2. Inicializar la escena
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=pa.res, film_transparent=False)
    scene = bpy.context.scene

    # Generar colores aleatorios para el ABS de las piezas (usamos colores brillantes/comunes de LEGO)
    lego_colors = ["#E60012", "#005CAF", "#F39800", "#009944", "#FFF100", "#FFFFFF", "#000000", "#7A7A7A"]

    for idx, ref in enumerate(selected_parts):
        print(f"\n[Piece {idx+1}/{pa.num_pieces}] Procesando pieza {ref}...")
        
        # Consultar poses estables en la DB
        poses = query_poses_for_piece(ref)
        if not poses:
            print(f"  [ERROR] No se encontraron poses para {ref}. Saltando.")
            continue
        
        selected_pose = random.choice(poses)
        color_hex = random.choice(lego_colors)
        
        # Limpiar meshes previas
        scene_canonical.cleanup_piece_objects()
        
        # Importar y configurar la pieza en el origen
        part_obj = scene_canonical.import_part(ref)
        if not part_obj:
            print(f"  [ERROR] No se pudo importar la pieza {ref}. Saltando.")
            continue

        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        scene_canonical.normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)
        
        # Convertir la pose de base de datos a formato esperado por apply_stable_pose
        pose_dict = dict(selected_pose)
        apply_stable_pose(part_obj, pose_dict, random_z=True)

        # Crear material ABS
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)

        # Medir dimensiones físicas en BU (donde 1 BU = 100 mm)
        bbox_corners = [part_obj.matrix_world @ mathutils.Vector(corner) for corner in part_obj.bound_box]
        xs = [c.x for c in bbox_corners]
        ys = [c.y for c in bbox_corners]
        zs = [c.z for c in bbox_corners]
        
        len_x_bu = max(xs) - min(xs)
        width_y_bu = max(ys) - min(ys)
        height_z_bu = max(zs) - min(zs)
        print(f"  Dimensiones físicas (BU): len_x={len_x_bu:.3f}, width_y={width_y_bu:.3f}, height_z={height_z_bu:.3f}")

        # Calcular rango visible en FoV cenital (ancho de FoV es aprox 1.96 BU)
        FOV_WIDTH_BU = 3.0 * (36.0 / 55.0)  # CAM_CEN_LOC.z * (SENSOR_MM / FOCAL_MM) = 3.0 * 36.0 / 55.0 = 1.9636 BU
        half_range = max(0.02, (FOV_WIDTH_BU - len_x_bu) * 0.5)

        # 5 posiciones equidistantes dentro de la zona cenital
        cenital_x_positions = [
            -half_range + i * (2 * half_range / 4.0) for i in range(5)
        ]

        # Posiciones adicionales para simular la salida y la caída de la cinta
        # La cinta mide BELT_L_BU = 12.0 BU, desde -6.0 a +6.0
        # Muestreamos posiciones desde la salida del FoV (half_range) hasta pasada la caída (+6.5 BU)
        falling_x_positions = []
        current_x = cenital_x_positions[-1] + 0.5
        while current_x <= 6.5:
            falling_x_positions.append(current_x)
            current_x += 0.5  # Paso de 5 cm

        trajectory_x = cenital_x_positions + falling_x_positions
        print(f"  Trayectoria X generada ({len(trajectory_x)} posiciones): {[round(x, 2) for x in trajectory_x]}")

        # Definir una coordenada Y constante aleatoria en la cinta
        # Rango usable en Y de la cinta es [-0.95, 0.95] BU
        max_y_offset = max(0.0, 0.95 - (width_y_bu * 0.5))
        trajectory_y = random.uniform(-max_y_offset, max_y_offset)

        piece_dir = os.path.join(pa.output_dir, ref)
        os.makedirs(piece_dir, exist_ok=True)

        frames_meta = []
        for f_idx, x_bu in enumerate(trajectory_x):
            # Posicionar el objeto
            part_obj.location.x = x_bu
            part_obj.location.y = trajectory_y
            bpy.context.view_layer.update()

            # Bounding boxes 2D
            bbox_cen = scene_canonical.get_2d_bbox(part_obj, scene, cam_cenital)
            bbox_lat = scene_canonical.get_2d_bbox(part_obj, scene, cam_frontal)

            # Clasificar si está en la cinta o ha caído
            status = "conveyor"
            if x_bu > 6.0:
                status = "falling"

            # Render Cenital
            scene.camera = cam_cenital
            cen_file = f"frame_{f_idx:02d}_cenital.png"
            scene.render.filepath = os.path.join(piece_dir, cen_file)
            bpy.ops.render.render(write_still=True)

            # Render Lateral (Frontal)
            scene.camera = cam_frontal
            lat_file = f"frame_{f_idx:02d}_lateral.png"
            scene.render.filepath = os.path.join(piece_dir, lat_file)
            bpy.ops.render.render(write_still=True)

            frames_meta.append({
                "frame_index": f_idx,
                "x_bu": x_bu,
                "y_bu": trajectory_y,
                "status": status,
                "file_cenital": cen_file,
                "file_lateral": lat_file,
                "bbox_cenital_norm": bbox_cen,
                "bbox_lateral_norm": bbox_lat
            })

        # Guardar metadatos de la pieza
        piece_meta = {
            "part_ref": ref,
            "color_hex": color_hex,
            "pose_index": selected_pose["pose_index"],
            "face_class": selected_pose["face_class"],
            "contact_normal": list(selected_pose["contact_normal"]),
            "zenith_silhouette_area_gt": selected_pose["zenith_silhouette_area"],
            "lateral_height_gt": selected_pose["lateral_height"],
            "trajectory_y_bu": trajectory_y,
            "frames": frames_meta
        }
        with open(os.path.join(piece_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(piece_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Dataset data200 completado con éxito en: {pa.output_dir}")

if __name__ == "__main__":
    main()
