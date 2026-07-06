# -*- coding: utf-8 -*-
"""camara_domo/scripts/generate_yolo_dataset.py
===================================================
Genera un dataset sintético para entrenar YOLO y YOLO-Pose
para la `camara_domo` (cenital y frontal simultáneamente).

Características:
- Consulta a la base de datos (Supabase) las combinaciones reales de pieza-color.
- Distribuye 10,000 instancias ponderadas por la cantidad de poses estables.
- Empaquetado 2D aleatorio (Random Scatter) para maximizar piezas por frame sin superposición.
- Output en formato Ultralytics YOLO-Pose.

Uso:
    /opt/homebrew/bin/blender -b -P \
        camara_domo/scripts/generate_yolo_dataset.py -- \
            --num_pieces 10000 \
            --output_dir camara_domo/data/yolo_dataset \
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

from core.db.supabase_client import get_connection
from generate_synthetic_set import (
    apply_bevel_modifier,
    configure_eevee_for_translucent,
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
)
from _pose_utils import apply_stable_pose
import scene_canonical

# Configuración óptica
CAM_CEN_Z_MM = 300.0
FOCAL_MM = 55.0
SENSOR_MM = 36.0
FOV_WIDTH_MM = CAM_CEN_Z_MM * (SENSOR_MM / FOCAL_MM)  # ~196.36 mm
# Definimos el área usable para random scatter un poco más pequeña para no cortar
MARGIN_MM = 5.0
HALF_FOV_USABLE = (FOV_WIDTH_MM / 2.0) - MARGIN_MM

KEYPOINTS_PATH = os.path.join(legovic_root, "projects", "2camaras_random_pieza_unica", "data", "canonical_keypoints.json")


def _world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def compute_bbox_yolo(obj, camera, scene):
    """Devuelve (cx, cy, w, h) normalizados [0,1] formato YOLO o None."""
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
    
    # Check if completely outside
    if x2 < 0 or x1 > 1 or y2 < 0 or y1 > 1:
        return None
        
    # Clip to bounds
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
    """Proyecta los 9 keypoints a 2D imagen. v=2 (visible), v=0 (fuera/ocluido)."""
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
        
        # Clamp to 0-1 just to be safe if v=0 but close to edge
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        out.append((x_norm, y_norm, v))
    return out


def intersects(box1, box2):
    """Comprueba si dos bounding boxes 2D alineados a los ejes se intersectan.
    box = (min_x, max_x, min_y, max_y)
    """
    if box1[1] < box2[0] or box1[0] > box2[1]:
        return False
    if box1[3] < box2[2] or box1[2] > box2[3]:
        return False
    return True


def intersects_2d(box1, box2):
    """Comprueba si dos bounding boxes 2D alineados a los ejes se intersectan en coordenadas normalizadas.
    box = [min_x, min_y, max_x, max_y]
    """
    if box1[2] < box2[0] or box1[0] > box2[2]:
        return False
    if box1[3] < box2[1] or box1[1] > box2[3]:
        return False
    return True


def fetch_universe():
    print("[DB] Consultando poses estables y colores en Supabase...")
    valid_poses = {}
    valid_combos = []
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Poses estables
            cur.execute("""
                SELECT part_ref, pose_index, zenith_observable_area, zenith_silhouette_area, 
                       lateral_height, effective_height, contact_stable_length, contact_stable_width,
                       face_class, is_stable, stability_ratio
                FROM stable_poses
                WHERE is_stable = TRUE
            """)
            for row in cur.fetchall():
                ref = row["part_ref"]
                valid_poses.setdefault(ref, []).append(row)
                
            # 2. Combinaciones pieza-color
            cur.execute("""
                SELECT DISTINCT part_ref, color_code, color_hex
                FROM lego_set_parts
                WHERE color_code IS NOT NULL AND color_hex IS NOT NULL
            """)
            for row in cur.fetchall():
                ref = row["part_ref"]
                if ref in valid_poses:
                    valid_combos.append({
                        "ref": ref,
                        "color_code": row["color_code"],
                        "color_hex": row["color_hex"]
                    })
                    
    print(f"[DB] {len(valid_poses)} piezas con poses estables.")
    print(f"[DB] {len(valid_combos)} combinaciones únicas de pieza-color.")
    return valid_poses, valid_combos


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pieces", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, "data"))
    parser.add_argument("--split", type=str, default="train") # 'train' o 'val'
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    pa = parser.parse_args(args)

    random.seed(pa.seed)
    
    # Directorios de salida (estructurados para coincidir exactamente con lo que YOLO espera)
    images_dir_cen = os.path.join(pa.output_dir, "yolo_dataset_cenital", "images", pa.split)
    labels_dir_cen = os.path.join(pa.output_dir, "yolo_dataset_cenital", "labels", pa.split)
    images_dir_lat = os.path.join(pa.output_dir, "yolo_dataset_frontal", "images", pa.split)
    labels_dir_lat = os.path.join(pa.output_dir, "yolo_dataset_frontal", "labels", pa.split)
    
    os.makedirs(images_dir_cen, exist_ok=True)
    os.makedirs(labels_dir_cen, exist_ok=True)
    os.makedirs(images_dir_lat, exist_ok=True)
    os.makedirs(labels_dir_lat, exist_ok=True)

    valid_poses, valid_combos = fetch_universe()
    
    # Asignar un ID numérico de clase a cada referencia (para YOLO)
    # Ordenamos alfabéticamente para que sea determinista
    unique_refs = sorted(list(valid_poses.keys()))
    class_to_idx = {ref: idx for idx, ref in enumerate(unique_refs)}
    
    # Guardar classes.txt
    with open(os.path.join(pa.output_dir, "classes.txt"), "w") as f:
        for ref in unique_refs:
            f.write(f"{ref}\n")
            
    # Load Keypoints
    if not os.path.exists(KEYPOINTS_PATH):
        print(f"[ERROR] No existe {KEYPOINTS_PATH}")
        sys.exit(1)
    with open(KEYPOINTS_PATH, "r") as f:
        kps_data = json.load(f)
    kps_by_ref = {ref: data["keypoints_bu"] for ref, data in kps_data["pieces"].items()}

    # Uniform distribution of geometries (Uniform pieces sampling)
    print("[Plan] Creando distribución uniforme de piezas...")
    # Pre-filtrar combos a solo aquellos que tienen keypoints canónicos
    valid_combos = [c for c in valid_combos if c["ref"] in kps_by_ref]
    print(f"[Plan] {len(valid_combos)} combinaciones tras filtrar piezas sin keypoints.")
    
    # Agrupar combos por color_code para muestreo uniforme de color
    combos_by_color = {}
    for combo in valid_combos:
        combos_by_color.setdefault(combo["color_code"], []).append(combo)
        
    all_colors = list(combos_by_color.keys())
    sampled_instances = []
    for _ in range(pa.num_pieces):
        color = random.choice(all_colors)
        # Seleccionar una geometría aleatoria que posea este color
        combo = random.choice(combos_by_color[color]).copy()
        sampled_instances.append(combo)
    
    # Asignar pose probabilística a cada instancia muestreada
    print("[Plan] Seleccionando poses con espectro inteligente...")
    
    # Piezas cilíndricas conocidas (se comportan igual al rodar)
    cylindrical_refs = {"30554b", "3705", "15462", "62462", "6558", "87994", "2780"}
    
    pose_counts = {ref: {idx: 0 for idx in range(len(valid_poses[ref]))} for ref in valid_poses}
    
    for inst in sampled_instances:
        ref = inst["ref"]
        poses = valid_poses[ref]
        if not poses:
            continue
            
        if ref in cylindrical_refs:
            # Para piezas cilíndricas, todas las poses rodando son equivalentes.
            # Las poses inestables (ej: de pie) tendrán stability_ratio bajo gracias a la simulación.
            # Filtramos solo las que tengan un ratio razonable para evitar las inestables (como la pose 0 y 1 de 3705).
            valid_idx = [i for i, p in enumerate(poses) if p.get("stability_ratio", 0.0) > 0.05]
            if not valid_idx:
                valid_idx = list(range(len(poses)))
            # Escoger aleatoriamente uniformemente entre las poses válidas rodantes
            idx = random.choice(valid_idx)
            inst["pose"] = poses[idx]
            continue

        # Selección inteligente para piezas normales (espectro completo)
        counts = pose_counts[ref]
        total_selected = sum(counts.values())
        
        # Ponderar por stability_ratio (probabilidad real de caída)
        p_weights = [p.get("stability_ratio", 0.0) for p in poses]
        s_sum = sum(p_weights)
        if s_sum <= 0:
            p_weights = [1.0 / len(poses)] * len(poses)
        else:
            p_weights = [w / s_sum for w in p_weights]
            
        adjusted_weights = []
        for i, p in enumerate(poses):
            if total_selected < len(poses) and counts[i] == 0:
                # Forzar que toda pose salga al menos una vez (espectro inicial)
                adjusted_weights.append(100.0)
            else:
                expected = total_selected * p_weights[i]
                actual = counts[i]
                deficit = expected - actual
                # Peso proporcional ajustado por el déficit para balancear
                w = max(0.01, p_weights[i] + deficit * 0.5)
                adjusted_weights.append(w)
                
        idx = random.choices(range(len(poses)), weights=adjusted_weights, k=1)[0]
        counts[idx] += 1
        inst["pose"] = poses[idx]
        
    # Sharding para ejecución paralela
    if pa.num_workers > 1:
        # Usar chunking en lugar de stride para que los logs sean más coherentes
        chunk_size = math.ceil(len(sampled_instances) / pa.num_workers)
        start_idx = pa.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(sampled_instances))
        sampled_instances = sampled_instances[start_idx:end_idx]
        
    print(f"[Plan Worker {pa.worker_id}] Seleccionadas {len(sampled_instances)} instancias para renderizar.")

    # Inicializar Blender
    cam_cenital, cam_frontal = scene_canonical.build_scene_canonical(render_res=1024, film_transparent=False)
    scene = bpy.context.scene

    frames_meta = []
    
    # Limpiar objetos: NO borramos la escena canónica
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.startswith("Plane") and not obj.name.startswith("Conveyor") and not obj.name.startswith("Office") and not obj.name.startswith("Side"):
            obj.select_set(True)
    bpy.ops.object.delete()

    instances_placed = 0
    frame_idx = 0
    
    # Cache de meshes importados para no re-importar todo el tiempo
    loaded_meshes = {}
    
    print("[Render] Iniciando renderizado...")
    
    failed_parts = set()
    while instances_placed < len(sampled_instances):
        placed_objects = []
        bounding_boxes_2d = [] # (min_x, max_x, min_y, max_y)
        placed_bboxes_cenital_2d = []
        placed_bboxes_frontal_2d = []
        
        # Intentar colocar tantas piezas como sea posible (max 6 por frame para evitar densidad)
        consecutive_fails = 0
        
        while instances_placed < len(sampled_instances) and len(placed_objects) < 6 and consecutive_fails < 5:
            item = sampled_instances[instances_placed]
            ref = item["ref"]
            pose = item["pose"]
            color_hex = item["color_hex"]
            
            if ref not in loaded_meshes:
                original_obj = scene_canonical.import_part(ref)
                if not original_obj:
                    failed_parts.add(ref)
                    instances_placed += 1
                    continue
                # Lo movemos fuera de cámara inicialmente
                original_obj.location = (100, 100, 100)
                original_obj.hide_render = True
                loaded_meshes[ref] = original_obj
            
            part_obj = loaded_meshes[ref].copy()
            part_obj.data = loaded_meshes[ref].data.copy()
            part_obj.hide_render = False  # FIXED: Hacer visible la pieza
            scene.collection.objects.link(part_obj)
            
            bpy.context.view_layer.objects.active = part_obj
            scene_canonical.normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            apply_stable_pose(part_obj, pose)
            
            mat = create_abs_plastic_material(color_hex)
            
            # Intentar ubicar aleatoriamente sin solapamiento 3D (XY del mundo) y sin solapamiento 2D en cámara
            placed = False
            for _ in range(50):
                # Generar ubicación aleatoria en el FOV (convertida a Blender Units)
                rx_mm = random.uniform(-HALF_FOV_USABLE, HALF_FOV_USABLE)
                ry_mm = random.uniform(-HALF_FOV_USABLE, HALF_FOV_USABLE)
                rz_rot = random.uniform(0, 2 * math.pi)
                
                part_obj.location.x = rx_mm / 100.0
                part_obj.location.y = ry_mm / 100.0
                part_obj.rotation_euler.z = rz_rot
                
                bpy.context.view_layer.update()
                
                # 1. Chequeo de colisión en 3D (coordenadas XY del mundo)
                xs, ys = [], []
                for v in _world_bbox(part_obj):
                    xs.append(v.x)
                    ys.append(v.y)
                
                # Expansión del BBox como margen de seguridad
                margin = 0.05 # 5mm en BU
                new_box_3d = (min(xs)-margin, max(xs)+margin, min(ys)-margin, max(ys)+margin)
                
                overlap = False
                for box in bounding_boxes_2d:
                    if intersects(new_box_3d, box):
                        overlap = True
                        break
                if overlap:
                    continue
                
                # 2. Chequeo de colisión en 2D en las proyecciones de cámara (Cenital y Frontal)
                bbox_cen_2d = scene_canonical.get_2d_bbox(part_obj, scene, cam_cenital)
                bbox_lat_2d = scene_canonical.get_2d_bbox(part_obj, scene, cam_frontal)
                
                # Evitar cualquier intersección en vista cenital
                for box in placed_bboxes_cenital_2d:
                    if intersects_2d(bbox_cen_2d, box):
                        overlap = True
                        break
                if overlap:
                    continue
                    
                # Evitar cualquier intersección en vista lateral (previene oclusión en perspectiva)
                for box in placed_bboxes_frontal_2d:
                    if intersects_2d(bbox_lat_2d, box):
                        overlap = True
                        break
                if overlap:
                    continue
                
                # Si pasa todos los chequeos, confirmamos posición y guardamos bboxes
                bounding_boxes_2d.append(new_box_3d)
                placed_bboxes_cenital_2d.append(bbox_cen_2d)
                placed_bboxes_frontal_2d.append(bbox_lat_2d)
                placed_objects.append({
                    "obj": part_obj,
                    "ref": ref,
                    "color_code": item["color_code"]
                })
                placed = True
                instances_placed += 1
                consecutive_fails = 0
                break
                    
            if not placed:
                # Si fallamos en ubicarla, la borramos y contamos el fallo
                bpy.data.objects.remove(part_obj, do_unlink=True)
                consecutive_fails += 1
            else:
                # Randomizar material ABS (Domain Randomization)
                mat_unique = mat.copy()
                node_principled = mat_unique.node_tree.nodes.get('Principled BSDF')
                if node_principled:
                    node_principled.inputs['Roughness'].default_value = random.uniform(0.1, 0.4)
                    if 'Specular IOR Level' in node_principled.inputs:
                        node_principled.inputs['Specular IOR Level'].default_value = random.uniform(0.3, 0.7)
                    elif 'Specular' in node_principled.inputs:
                        node_principled.inputs['Specular'].default_value = random.uniform(0.3, 0.7)
                part_obj.data.materials.clear()
                part_obj.data.materials.append(mat_unique)
                
        # --- Render y Anotación ---
        if not placed_objects:
            continue
            
        # Randomización de la iluminación de fondo y EEVEE Ambient Occlusion
        sc = bpy.context.scene
        if sc.world and sc.world.use_nodes:
            bg = sc.world.node_tree.nodes.get("Background")
            if bg:
                bg.inputs["Strength"].default_value = random.uniform(1.2, 1.8)
                bg.inputs["Color"].default_value = (
                    random.uniform(0.95, 1.0),
                    random.uniform(0.95, 1.0),
                    random.uniform(0.95, 1.0),
                    1.0
                )
        try:
            sc.eevee.use_gtao = True
            sc.eevee.gtao_distance = 0.1 # 10mm
        except:
            try:
                sc.eevee.use_ambient_occlusion = True
            except:
                pass
            
        uuid_str = str(uuid.uuid4())[:8]
        img_name = f"{pa.split}_{frame_idx:05d}_{uuid_str}.png"
        txt_name = f"{pa.split}_{frame_idx:05d}_{uuid_str}.txt"
        
        # 1. Render Cenital
        scene.camera = cam_cenital
        scene.render.filepath = os.path.join(images_dir_cen, img_name)
        bpy.ops.render.render(write_still=True)
        
        with open(os.path.join(labels_dir_cen, txt_name), "w") as f_cen:
            for item in placed_objects:
                obj = item["obj"]
                ref = item["ref"]
                
                bbox = compute_bbox_yolo(obj, cam_cenital, scene)
                if bbox:
                    kps_local = kps_by_ref[ref]
                    kps_2d = project_keypoints(obj, kps_local, cam_cenital, scene)
                    
                    # Forzar class_id = 0 para detector class-agnostic
                    line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    for (kx, ky, kv) in kps_2d:
                        line += f" {kx:.6f} {ky:.6f} {kv}"
                    f_cen.write(line + "\n")
                    
        # 2. Render Frontal
        scene.camera = cam_frontal
        scene.render.filepath = os.path.join(images_dir_lat, img_name)
        bpy.ops.render.render(write_still=True)
        
        with open(os.path.join(labels_dir_lat, txt_name), "w") as f_lat:
            for item in placed_objects:
                obj = item["obj"]
                ref = item["ref"]
                
                bbox = compute_bbox_yolo(obj, cam_frontal, scene)
                if bbox:
                    kps_local = kps_by_ref[ref]
                    kps_2d = project_keypoints(obj, kps_local, cam_frontal, scene)
                    
                    # Forzar class_id = 0 para detector class-agnostic
                    line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    for (kx, ky, kv) in kps_2d:
                        line += f" {kx:.6f} {ky:.6f} {kv}"
                    f_lat.write(line + "\n")
                    
        # Limpiar objetos de este frame (excepto originales cacheados)
        for item in placed_objects:
            bpy.data.objects.remove(item["obj"], do_unlink=True)
            
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  [Render] Frame {frame_idx} completado. Instancias procesadas: {instances_placed}/{pa.num_pieces}")

    print(f"[Done] Dataset generado en {pa.output_dir}")
    
    if failed_parts:
        print("\n" + "="*60)
        print("ERROR: No se pudo encontrar ni descargar archivos .dat para:")
        for fp in failed_parts:
            print(f"  - {fp}")
        print("Estas piezas han sido OMITIDAS en las imágenes.")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
