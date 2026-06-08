# -*- coding: utf-8 -*-
# scripts/generate_dino_fov_renders.py
# Blender script to render LEGO pieces packed in the camera FOV at multiple rotation angles.
# Usage: blender -b -P scripts/generate_dino_fov_renders.py -- [args]

import os
import sys
import json
import math
import random
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
sys.path.append(os.path.join(project_root, "scratch"))

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# Import helper functions from existing scene configurations
from generate_synthetic_set import (
    setup_physics_world, setup_studio_lighting, create_abs_plastic_material,
    apply_bevel_modifier, get_ldraw_part_path, generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

from scene_config import (
    BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU, BELT_COLOR_LINEAR,
    CAMERA_Z, CAMERA_ORTHO_SCALE, RENDER_RES_SQUARE,
    TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z,
    WORLD_BG_COLOR, WORLD_BG_STRENGTH, LDRAW_TO_BU, LDRAW_THRESHOLD,
    CORNER_LIGHT_OFFSET_XY, CORNER_LIGHT_Z, CORNER_LIGHT_SIZE, CORNER_LIGHT_ENERGY,
)

# Definir variables de luces de esquina localmente
_OFF = CORNER_LIGHT_OFFSET_XY
_CZ = CORNER_LIGHT_Z
CORNER_LIGHT_NAMES = ["Corner_Light_PP", "Corner_Light_PN", "Corner_Light_NP", "Corner_Light_NN"]
CORNER_LIGHT_POSITIONS = [
    ( _OFF,  _OFF, _CZ),
    ( _OFF, -_OFF, _CZ),
    (-_OFF,  _OFF, _CZ),
    (-_OFF, -_OFF, _CZ),
]


def setup_camera_and_belt():
    """Configura la escena con la misma cámara y cinta de inferencia."""
    # Cámara
    cam_name = "Camera"
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
    else:
        bpy.ops.object.camera_add(location=(0, 0, CAMERA_Z))
        cam = bpy.context.active_object
        cam.name = cam_name
    cam.location = (0.0, 0.0, CAMERA_Z)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = CAMERA_ORTHO_SCALE
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    bpy.context.scene.camera = cam

    # Cinta transportadora de fondo
    belt_name = "Dino_Belt_Plane"
    if belt_name in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects[belt_name].select_set(True)
        bpy.ops.object.delete()
        
    half_t = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_t))
    belt = bpy.context.active_object
    belt.name = belt_name
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    
    mat = bpy.data.materials.get("Belt_Mat_Dino")
    if not mat:
        mat = bpy.data.materials.new("Belt_Mat_Dino")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
            bsdf.inputs["Roughness"].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)

def setup_lights():
    """Configura las condiciones de iluminación de inferencia real con Domain Randomization (variación de luz)."""
    setup_studio_lighting()
    
    # Asegurar luces de esquina para iluminación uniforme con pequeñas variaciones
    for name, pos in zip(CORNER_LIGHT_NAMES, CORNER_LIGHT_POSITIONS):
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
        else:
            bpy.ops.object.light_add(type='AREA', location=pos)
            obj = bpy.context.active_object
            obj.name = name
            
        # Jitter de posición de +-0.4 BU en X/Y e +-0.3 en Z
        rx = pos[0] + random.uniform(-0.4, 0.4)
        ry = pos[1] + random.uniform(-0.4, 0.4)
        rz = pos[2] + random.uniform(-0.3, 0.3)
        obj.location = (rx, ry, rz)
        
        obj.data.size = CORNER_LIGHT_SIZE
        # Variación de energía de +-15%
        obj.data.energy = CORNER_LIGHT_ENERGY * random.uniform(0.85, 1.15)
        obj.rotation_euler = (0.0, 0.0, 0.0)

    # Luz cenital
    top_name = "Top_Diffuse_Light"
    top = bpy.data.objects.get(top_name)
    if not top:
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, TOP_LIGHT_Z))
        top = bpy.context.active_object
        top.name = top_name
        
    # Jitter de posición cenital
    top.location = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), TOP_LIGHT_Z)
    top.data.size = TOP_LIGHT_SIZE
    top.data.energy = TOP_LIGHT_ENERGY * random.uniform(0.85, 1.15)


def _normalize_piece(obj):
    """Normaliza la escala e inicializa el centro de geometría."""
    if not obj.data or not hasattr(obj.data, 'vertices'): return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts: return 1.0
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)
    max_dim = max(dx, dy, dz)
    if max_dim < 1e-6: return 1.0
    
    factor = LDRAW_TO_BU if max_dim > LDRAW_THRESHOLD else 1.0
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    cz = (max(zs) + min(zs)) / 2.0
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) * factor
        v.co.y = (v.co.y - cy) * factor
        v.co.z = (v.co.z - cz) * factor
    obj.data.update()
    obj.scale = (1.0, 1.0, 1.0)
    obj.location = (0.0, 0.0, 0.0)
    return factor

def load_template(part_ref, color_hex):
    """Carga y prepara la plantilla de la pieza LEGO."""
    part_path = get_ldraw_part_path(part_ref)
    existing = set(bpy.context.scene.objects)
    obj = None
    
    if part_path:
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objs = [o for o in bpy.context.scene.objects if o not in existing]
            par = next((o for o in new_objs if o.parent is None), None)
            obj = get_single_mesh_object(par) if par else None
            if not obj:
                generate_detailed_fallback_mesh(part_ref)
                obj = bpy.context.active_object
        except Exception as e:
            print(f"[Blender] Error importando LDraw {part_ref}: {e}. Fallback mesh.")
            generate_detailed_fallback_mesh(part_ref)
            obj = bpy.context.active_object
    else:
        generate_detailed_fallback_mesh(part_ref)
        obj = bpy.context.active_object
        
    if not obj:
        print(f"[Blender] ERROR: No se pudo generar la pieza {part_ref}")
        return None
        
    obj.name = f"Template_{part_ref}"
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    _normalize_piece(obj)
    apply_bevel_modifier(obj)
    
    mat = create_abs_plastic_material(color_hex)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    # Mover al collection de templates y esconder
    if "Templates" not in bpy.data.collections:
        col = bpy.data.collections.new("Templates")
        bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections["Templates"]
    
    col.hide_viewport = True
    col.hide_render = True
    
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    
    return obj

def cleanup_placed_pieces():
    """Limpia los objetos colocados en la simulación."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        if o.name.startswith("Placed_"):
            o.select_set(True)
    bpy.ops.object.delete()

def get_bounding_radius(obj, rotation_quat_list=None):
    """
    Calcula el radio de límite 2D de la pieza de forma robusta e invariante a rotación.
    Utiliza la distancia máxima de cualquier vértice (del objeto y sus hijos) al origen (0,0,0).
    """
    max_dist = 0.0
    
    def traverse(o):
        nonlocal max_dist
        if o.type == 'MESH' and o.data:
            for v in o.data.vertices:
                # Distancia en el plano XY (2D packing)
                dist = math.sqrt(v.co.x**2 + v.co.y**2)
                if dist > max_dist:
                    max_dist = dist
        for child in o.children:
            traverse(child)
            
    traverse(obj)
    # Si no tiene vértices (caso raro), usar valor por defecto de 1.0 BU
    return max(1.0, max_dist)


def compute_bbox_pixel(obj, cam, scene):
    """Calcula la bounding box 2D en píxeles del objeto renderizado."""
    world_verts = []
    if obj.type == 'MESH' and obj.data:
        m = obj.matrix_world
        world_verts = [m @ v.co for v in obj.data.vertices]
        
    if not world_verts:
        world_verts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        
    xs, ys = [], []
    for v in world_verts:
        c = world_to_camera_view(scene, cam, v)
        xs.append(c.x)
        ys.append(c.y)
        
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    
    if x1 <= x0 or y1 <= y0:
        return None
        
    # Convertir a píxeles
    W = RENDER_RES_SQUARE
    H = RENDER_RES_SQUARE
    px_min_x = x0 * W
    px_max_x = x1 * W
    px_min_y = (1.0 - y1) * H
    px_max_y = (1.0 - y0) * H
    
    return [px_min_x, px_min_y, px_max_x, px_max_y]

def pack_pieces_fov(template_obj, stable_poses, count_to_place):
    """
    Empaqueta de forma no solapada hasta `count_to_place` piezas en el FOV.
    Separación mínima entre piezas = 10 mm (1.0 BU).
    Separación con bordes del FOV = 10 mm (1.0 BU).
    """
    placed_pieces = []
    margin_bu = 1.0  # 10 mm margin
    limit = (CAMERA_ORTHO_SCALE / 2.0) - margin_bu  # 10.0 - 1.0 = 9.0 BU
    
    # Calcular radios para cada pose estable para acelerar la comprobación
    pose_radii = {}
    for pose in stable_poses:
        idx = pose["pose_index"]
        pose_radii[idx] = get_bounding_radius(template_obj, pose["orientation_quat"])
        
    attempts = 0
    max_attempts = 1000
    
    while len(placed_pieces) < count_to_place and attempts < max_attempts:
        attempts += 1
        
        # Seleccionar pose estable aleatoria
        pose = random.choice(stable_poses)
        p_idx = pose["pose_index"]
        r = pose_radii[p_idx]
        
        # Generar posición candidata teniendo en cuenta el radio y margen del borde
        # El centro debe estar dentro de [-limit + r, limit - r]
        max_coord = limit - r
        min_coord = -limit + r
        
        if min_coord >= max_coord:
            # La pieza es demasiado grande para caber en el FOV con este margen
            continue
            
        cx = random.uniform(min_coord, max_coord)
        cy = random.uniform(min_coord, max_coord)
        
        # Verificar solapamientos con piezas ya colocadas
        overlap = False
        for px, py, pr, _ in placed_pieces:
            dist = math.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < (r + pr + margin_bu):  # margin_bu = 1.0 BU = 10 mm
                overlap = True
                break
                
        if not overlap:
            placed_pieces.append((cx, cy, r, pose))
            attempts = 0  # reset attempts on success
            
    print(f"[Blender Packing] Empaquetadas {len(placed_pieces)} piezas en el FOV.")
    return placed_pieces

def run_rendering(part_ref, color_hex, num_rotations, num_pieces, stable_poses_path, output_dir):
    """Carga configuraciones, empaqueta piezas, aplica rotaciones y renderiza las imágenes."""
    enable_metal_gpu_acceleration()
    setup_physics_world()
    setup_camera_and_belt()
    setup_lights()
    
    # ------------------------------------------------------------------
    # Render empty belt reference frame
    # ------------------------------------------------------------------
    empty_belt_path = os.path.join(output_dir, "empty_belt.png")
    cleanup_placed_pieces()
    
    scene_tmp = bpy.context.scene
    scene_tmp.render.engine = "CYCLES"
    scene_tmp.render.resolution_x = RENDER_RES_SQUARE
    scene_tmp.render.resolution_y = RENDER_RES_SQUARE
    scene_tmp.render.filepath = empty_belt_path
    
    try:
        bpy.ops.render.render(write_still=True)
        print(f"[Blender] Rendered empty belt reference: {empty_belt_path}")
    except Exception as e:
        print(f"[Blender ERROR] Failed to render empty belt: {e}")
        
    # Cargar JSON de posiciones estables
    with open(stable_poses_path, "r", encoding="utf-8") as f:
        stable_poses = json.load(f)
        
    if not stable_poses:
        print("[Blender ERROR] No hay posiciones estables para esta pieza.")
        sys.exit(1)
        
    template = load_template(part_ref, color_hex)
    if not template:
        sys.exit(1)
        
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8  # Suficiente calidad para DINOv2
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE
    scene.render.film_transparent = True

    
    # Determinar cuántas imágenes necesitamos renderizar
    total_placed = 0
    frame_idx = 0
    metadata = {"renders": []}
    
    camera = bpy.data.objects["Camera"]
    
    while total_placed < num_pieces:
        cleanup_placed_pieces()
        
        # Empaquetar el máximo número de piezas posibles de las restantes
        remaining = num_pieces - total_placed
        layout = pack_pieces_fov(template, stable_poses, remaining)
        
        if not layout:
            print("[Blender ERROR] No se pudieron empaquetar más piezas. La geometría podría ser demasiado grande.")
            break
            
        # Instanciar las piezas en sus posiciones y orientaciones estables base
        placed_objs = []
        for i, (x, y, r, pose) in enumerate(layout):
            oc = template.copy()
            oc.data = template.data.copy()
            oc.name = f"Placed_Piece_{i}"
            
            # Añadir a la escena
            bpy.context.scene.collection.objects.link(oc)
            oc.hide_viewport = False
            oc.hide_render = False
            
            # Orientar con la pose estable
            oc.rotation_mode = 'QUATERNION'
            oc.rotation_quaternion = mathutils.Quaternion(pose["orientation_quat"])
            bpy.context.view_layer.update()
            
            # Posicionar sobre la cinta
            oc.location = (x, y, 0.0)
            bpy.ops.object.select_all(action='DESELECT')
            oc.select_set(True)
            bpy.context.view_layer.objects.active = oc
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            bbox = [oc.matrix_world @ mathutils.Vector(c) for c in oc.bound_box]
            min_z = min(pt.z for pt in bbox)
            oc.location.z = -min_z + 0.02
            
            placed_objs.append((oc, pose))
            
        total_placed += len(layout)
        
        # Bucle de rotaciones horizontales
        for rot_step in range(num_rotations):
            angle_deg = rot_step * (360.0 / num_rotations)
            angle_rad = math.radians(angle_deg)
            
            # Aplicar rotación horizontal y re-ajustar altura
            for oc, pose in placed_objs:
                # Rotación en Z global
                rot_z = mathutils.Matrix.Rotation(angle_rad, 4, 'Z')
                base_rot = mathutils.Quaternion(pose["orientation_quat"]).to_matrix().to_4x4()
                final_rot = rot_z @ base_rot
                
                oc.rotation_quaternion = final_rot.to_quaternion()
                bpy.context.view_layer.update()
                
                # Re-alinear altura z
                bbox = [oc.matrix_world @ mathutils.Vector(c) for c in oc.bound_box]
                min_z = min(pt.z for pt in bbox)
                oc.location.z = oc.location.z - min_z + 0.02
                
            bpy.context.view_layer.update()
            bpy.context.evaluated_depsgraph_get().update()
            
            # Renderizar escena
            img_filename = f"dino_fov_{part_ref}_f{frame_idx:03d}_r{rot_step:03d}.png"
            img_path = os.path.join(output_dir, img_filename)
            scene.render.filepath = img_path
            
            try:
                bpy.ops.render.render(write_still=True)
                print(f"[Blender] Renderizado: {img_filename}")
            except Exception as e:
                print(f"[Blender ERROR] Error en render: {e}")
                continue
                
            # Calcular bounding boxes en píxeles
            pieces_meta = []
            for idx, (oc, pose) in enumerate(placed_objs):
                bbox_px = compute_bbox_pixel(oc, camera, scene)
                if bbox_px:
                    pieces_meta.append({
                        "instance_id": idx,
                        "pose_index": pose["pose_index"],
                        "rotation_angle": angle_deg,
                        "bbox": bbox_px  # [xmin, ymin, xmax, ymax]
                    })
                    
            metadata["renders"].append({
                "image_path": img_path,
                "pieces": pieces_meta
            })
            
        frame_idx += 1
        
    cleanup_placed_pieces()
    
    # Escribir metadatos JSON
    metadata_path = os.path.join(output_dir, f"dino_fov_metadata_{part_ref}.json")
    with open(metadata_path, "w", encoding="utf-8") as fm:
        json.dump(metadata, fm, indent=2)
    print(f"[Blender DONE] Generados todos los renders. Metadatos guardados en {metadata_path}")

def main():
    if not IN_BLENDER:
        print("[ERROR] Este script debe ejecutarse dentro de Blender.")
        sys.exit(1)
        
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
        
    parser = argparse.ArgumentParser(description="Renderiza piezas LEGO para DINOv2 en el FOV.")
    parser.add_argument("--part_ref", type=str, required=True, help="Referencia de la pieza LEGO")
    parser.add_argument("--color_hex", type=str, default="A0A5A9", help="Color HEX de la pieza")
    parser.add_argument("--num_rotations", type=int, default=12, help="Número de rotaciones horizontales")
    parser.add_argument("--num_pieces", type=int, default=30, help="Número total de piezas a renderizar")
    parser.add_argument("--stable_poses_json", type=str, required=True, help="Ruta al JSON de poses estables de la pieza")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio de salida para los renders")
    
    parsed = parser.parse_known_args(args_raw)[0]
    
    run_rendering(
        part_ref=parsed.part_ref,
        color_hex=parsed.color_hex,
        num_rotations=parsed.num_rotations,
        num_pieces=parsed.num_pieces,
        stable_poses_path=parsed.stable_poses_json,
        output_dir=parsed.output_dir
    )

if __name__ == "__main__":
    main()
