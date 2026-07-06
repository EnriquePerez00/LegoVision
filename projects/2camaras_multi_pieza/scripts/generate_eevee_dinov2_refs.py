# -*- coding: utf-8 -*-
"""scripts/generate_eevee_dinov2_refs.py
=========================================
Renderiza imágenes de referencia DINOv2 para las 10 piezas de test
usando el setup simétrico de 2camaras_multi_pieza:
  - Motor:          BLENDER_EEVEE
  - Resolución:     640 × 640
  - Cámaras:         PERSP lens=27.0mm Z/Y=15BU
  - Iluminación:    LED Square Strip (tira de LEDs difusa)
"""
import os, sys, random, math, json

# Add user site-packages for Blender isolated environment
user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'scratch'))

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from core.utils.config_loader import cfg
from generate_synthetic_set import (
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

# ── Logging ──────────────────────────────────────────────────────────────────
import sys as _sys
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in _sys.path:
    _sys.path.insert(0, _proj)
from core.utils.logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("blender")

# Parámetros cargados del config.yaml
SELECTED_PARTS = cfg.pieces.selected_parts
PART_COLORS_HEX = cfg.pieces.reference_colors_hex
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)

RENDER_RES_SQUARE = cfg.render.resolution.width
TOP_LIGHT_Z = cfg.scene.lighting.led_strip.z_bu
TOP_LIGHT_SIZE_X = cfg.scene.lighting.led_strip.size_x_bu
TOP_LIGHT_SIZE_Y = cfg.scene.lighting.led_strip.size_y_bu
TOP_LIGHT_ENERGY = cfg.scene.lighting.led_strip.energy_w

WORLD_BG_STRENGTH = cfg.scene.world.strength
WORLD_BG_COLOR = tuple(cfg.scene.world.color)


def setup_led_strip_lights():
    """Setup laboratory-style lightbox lighting (canonical, no randomization).

    Same as the training scripts but no randomization, for DINOv2 references.
    """
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Frontal", "Lab_Floor"}
    for o in list(bpy.context.scene.objects):
        if o.type == 'LIGHT' and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.3

    neutral_color = (1.0, 1.0, 1.0)

    # 1. MAIN DOME
    bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, TOP_LIGHT_Z))
    main = bpy.context.active_object
    main.name = "Lab_Main_Dome"
    main.data.size = TOP_LIGHT_SIZE_X
    main.data.size_y = TOP_LIGHT_SIZE_Y
    main.data.shape = 'RECTANGLE'
    main.data.color = neutral_color
    main.data.energy = TOP_LIGHT_ENERGY

    # 2. WALL PANELS
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    wall_panels = [
        ("Lab_Wall_N", (0.0, +12.0, 6.0)),
        ("Lab_Wall_S", (0.0, -12.0, 6.0)),
        ("Lab_Wall_E", (+12.0, 0.0, 6.0)),
        ("Lab_Wall_W", (-12.0, 0.0, 6.0)),
    ]
    for wname, wloc in wall_panels:
        bpy.ops.object.light_add(type='AREA', location=wloc)
        wp = bpy.context.active_object
        wp.name = wname
        wp.data.size = 20.0
        wp.data.size_y = 12.0
        wp.data.shape = 'RECTANGLE'
        wp.data.color = neutral_color
        wp.data.energy = 600.0
        track = wp.constraints.new(type='TRACK_TO')
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

    # 3. GROUND FILL
    bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, -0.5))
    gf = bpy.context.active_object
    gf.name = "Lab_Ground_Fill"
    gf.data.size = 30.0
    gf.data.size_y = 30.0
    gf.data.shape = 'RECTANGLE'
    gf.data.color = neutral_color
    gf.data.energy = 200.0
    gf.rotation_euler = (3.14159, 0.0, 0.0)



def create_floor():
    """Create a large black floor below the belt for frontal camera background.

    Position: z = -2 BU (below the belt which is at z = -0.5 to 0).
    Size: 60x60 BU - extends far beyond the belt to fill frontal camera FOV.
    Material: pure black, fully matte (roughness=1.0).
    Purpose: provides high-contrast background so frontal camera sees pieces
             as silhouettes against black, improving segmentation/detection.
    """
    if "Lab_Floor" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects["Lab_Floor"].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, -2.0))
    floor = bpy.context.active_object
    floor.name = "Lab_Floor"
    floor.scale = (60.0, 60.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.get("Lab_Floor_Black")
    if not mat:
        mat = bpy.data.materials.new("Lab_Floor_Black")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            bsdf.inputs['Roughness'].default_value = 1.0
            if 'Metallic' in bsdf.inputs:
                bsdf.inputs['Metallic'].default_value = 0.0
    floor.data.materials.clear()
    floor.data.materials.append(mat)
    return floor


def create_belt_collider():
    """Crea el belt y los carriles laterales metálicos."""
    if 'Conveyor_Belt_Plane' in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Conveyor_Belt_Plane'].select_set(True)
        bpy.ops.object.delete()

    ht = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = 'Conveyor_Belt_Plane'
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    
    mat = bpy.data.materials.get('Belt_Material')
    if not mat:
        mat = bpy.data.materials.new('Belt_Material')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = BELT_COLOR_LINEAR
            bsdf.inputs['Roughness'].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)

    for name in ["Side_Rail_L", "Side_Rail_R"]:
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.delete()
            
    rail_w = 0.2
    rail_h = 0.4
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-BELT_WIDTH_BU/2.0 + rail_w/2.0, 0.0, rail_h/2.0))
    rail_l = bpy.context.active_object
    rail_l.name = "Side_Rail_L"
    rail_l.scale = (rail_w, BELT_LENGTH_BU, rail_h)
    bpy.ops.object.transform_apply(scale=True)
    
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(BELT_WIDTH_BU/2.0 - rail_w/2.0, 0.0, rail_h/2.0))
    rail_r = bpy.context.active_object
    rail_r.name = "Side_Rail_R"
    rail_r.scale = (rail_w, BELT_LENGTH_BU, rail_h)
    bpy.ops.object.transform_apply(scale=True)
    
    mat_metal = bpy.data.materials.get("Rail_Metal_Mat")
    if not mat_metal:
        mat_metal = bpy.data.materials.new("Rail_Metal_Mat")
        mat_metal.use_nodes = True
        bsdf = mat_metal.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.55, 0.55, 0.55, 1.0)
            bsdf.inputs['Metallic'].default_value = 0.9
            bsdf.inputs['Roughness'].default_value = 0.5
            
    for rail in [rail_l, rail_r]:
        rail.data.materials.clear()
        rail.data.materials.append(mat_metal)


def setup_cameras():
    """Configura las cámaras simétricas cenital y frontal."""
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    # Cenital
    cam_c_name = "Cam_Cenital"
    if cam_c_name in bpy.data.objects:
        cam_c = bpy.data.objects[cam_c_name]
    else:
        bpy.ops.object.camera_add(location=(0, 0, 15.0))
        cam_c = bpy.context.active_object
        cam_c.name = cam_c_name
    cam_c.location = (0.0, 0.0, 15.0)
    cam_c.constraints.clear()
    track_c = cam_c.constraints.new(type='TRACK_TO')
    track_c.target = target
    track_c.track_axis = 'TRACK_NEGATIVE_Z'
    track_c.up_axis = 'UP_Y'
    cam_c.data.type = 'PERSP'
    cam_c.data.lens = 27.0
    cam_c.data.clip_start = 0.01

    # Frontal
    cam_f_name = "Cam_Frontal"
    if cam_f_name in bpy.data.objects:
        cam_f = bpy.data.objects[cam_f_name]
    else:
        bpy.ops.object.camera_add(location=(0.0, -15.0, 0.0))
        cam_f = bpy.context.active_object
        cam_f.name = cam_f_name
    cam_f.location = (0.0, -15.0, 0.0)
    cam_f.constraints.clear()
    track_f = cam_f.constraints.new(type='TRACK_TO')
    track_f.target = target
    track_f.track_axis = 'TRACK_NEGATIVE_Z'
    track_f.up_axis = 'UP_Y'
    cam_f.data.type = 'PERSP'
    cam_f.data.lens = 27.0
    cam_f.data.clip_start = 0.01

    return cam_c, cam_f


def _normalize_piece(obj):
    if not obj.data or not hasattr(obj.data, 'vertices'): return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts: return 1.0
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if mx < 1e-6: return 1.0
    factor = 0.04 if mx > 5.0 else 1.0
    cx = (max(xs)+min(xs))/2.0; cy = (max(ys)+min(ys))/2.0; cz = (max(zs)+min(zs))/2.0
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) * factor
        v.co.y = (v.co.y - cy) * factor
        v.co.z = (v.co.z - cz) * factor
    obj.data.update()
    obj.scale = (1.0, 1.0, 1.0)
    obj.location = (0.0, 0.0, 0.0)
    return factor


def cleanup_piece():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R", "Cam_Cenital", "Cam_Frontal", "Lab_Floor"}
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try: o.select_set(True)
            except: pass
    bpy.ops.object.delete()


def compute_bbox_pixel(obj, cam, scene):
    """Calcula bounding box en píxeles [xmin, ymin, xmax, ymax] de forma exacta."""
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
        
    W = RENDER_RES_SQUARE
    H = RENDER_RES_SQUARE
    px_min_x = max(0, int(x0 * W))
    px_max_x = min(W, int(x1 * W))
    px_min_y = max(0, int((1.0 - y1) * H))
    px_max_y = min(H, int((1.0 - y0) * H))
    
    return [px_min_x, px_min_y, px_max_x, px_max_y]


def get_stable_poses_from_db_subprocess(part_ref):
    """Load stable poses from cache, filtered by stability_ratio >= MIN_STABILITY.

    Returns only poses that survive at least MIN_STABILITY (default 0.5 = 50%) of
    perturbation tests. This filter ensures rendered pieces are in genuinely
    stable orientations that resist conveyor belt vibrations.
    """
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    min_stability = 0.5
    try:
        # Try to load threshold from config
        from core.utils.config_loader import cfg as _cfg
        min_stability = getattr(_cfg.stable_poses, "render_min_stability", 0.5)
    except Exception:
        pass
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if part_ref in cache:
                all_poses = cache[part_ref]
                # Filter: keep only poses with stability_ratio >= threshold
                stable = [p for p in all_poses if p.get("stability_ratio", 0.0) >= min_stability]
                if stable:
                    return stable
                # Fallback: if no poses pass threshold, prefer Top/Bottom faces (CoG-stable)
                top_bottom = [p for p in all_poses if p.get("face_class") in ("Top", "Bottom")]
                if top_bottom:
                    return top_bottom
                # Last resort: return all poses (caller will pick one)
                return all_poses
        except Exception as e:
            pass
    return []



def main():
    import time as _time
    _t_start = _time.perf_counter()

    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rotations", type=int, default=12)
    pa = parser.parse_known_args(args_raw)[0]
    out_dir = pa.output_dir

    log_execution_header(log, "generate_eevee_dinov2_refs.py",
                         output_dir=out_dir,
                         rotations=pa.rotations,
                         selected_parts=SELECTED_PARTS)

    for c in ["cenital", "frontal"]:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_led_strip_lights()
    cam_c, cam_f = setup_cameras()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    # Habilitar transparencia para poder recortar de forma limpia
    scene.render.film_transparent = True
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE

    total_rendered = 0
    import mathutils

    for part_ref in SELECTED_PARTS:
        log.info(f"=== Generando referencias para la pieza: {part_ref} ===")
        # Obtener los colores reales de este part_ref en el set 75078-1
        from core.db.set_catalog import REAL_SETS
        allowed_colors = []
        for p in REAL_SETS["75078-1"]["parts"]:
            if p["ref"] == part_ref:
                allowed_colors.append(p["color_hex"].replace("#", "").upper())
        if not allowed_colors:
            allowed_colors = [c.replace("#", "").upper() for c in PART_COLORS_HEX]

        poses = get_stable_poses_from_db_subprocess(part_ref)
        if not poses:
            poses = [{"pose_index": 0, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}]

        for pose in poses:
            pose_idx = pose.get("pose_index", 0)

            # Cargar malla
            part_path = get_ldraw_part_path(part_ref)
            existing_objects = set(bpy.context.scene.objects)
            part_obj = None

            if part_path:
                try:
                    bpy.ops.import_scene.importldr(filepath=part_path)
                    new_objs = [o for o in bpy.context.scene.objects if o not in existing_objects]
                    par = next((o for o in new_objs if o.parent is None), None)
                    if par:
                        part_obj = get_single_mesh_object(par)
                except Exception as e:
                    log.warning(f"import LDraw {part_ref}: {e}")

            if not part_obj:
                generate_detailed_fallback_mesh(part_ref)
                part_obj = bpy.context.active_object

            if not part_obj:
                log.error(f"No se pudo cargar mesh para {part_ref}")
                continue

            bpy.ops.object.select_all(action='DESELECT')
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)

            # Para cada una de las rotaciones
            n_rots = pa.rotations
            rot_step = (2 * math.pi) / n_rots

            for rot_i in range(n_rots):
                rot_deg = int(round(rot_i * (360.0 / n_rots)))
                rot_rad = rot_i * rot_step

                # ── 1. RENDER CENITAL (Estrategia Iter9: Pieza en el centro) ──
                # Limpiar escena de otras piezas temporales
                for o in list(bpy.context.scene.objects):
                    if o.name.startswith("Placed_Copy_"):
                        bpy.data.objects.remove(o, do_unlink=True)

                # Colocar pieza central
                part_obj.hide_render = False
                quat = pose.get("orientation_quat")
                if quat and len(quat) == 4:
                    part_obj.rotation_mode = 'QUATERNION'
                    part_obj.rotation_quaternion = mathutils.Quaternion(quat)
                else:
                    part_obj.rotation_mode = 'XYZ'
                    part_obj.rotation_euler = mathutils.Euler(pose.get("orientation_euler", [0,0,0]))
                
                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler.z += rot_rad
                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                
                # Snap to belt
                bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
                min_z = min(pt.z for pt in bbox_world)
                part_obj.location.z = -min_z + 0.02
                bpy.context.view_layer.update()

                for color_hex in allowed_colors:
                    mat = create_abs_plastic_material(f"#{color_hex}")
                    part_obj.data.materials.clear()
                    part_obj.data.materials.append(mat)
                    bpy.context.view_layer.update()

                    # Render Cenital
                    scene.camera = cam_c
                    scene.render.filepath = os.path.join(out_dir, "cenital", f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png")
                    try:
                        bpy.ops.render.render(write_still=True)
                        total_rendered += 1
                    except Exception as e:
                        print(f"  [WARN] Render cenital fallido: {e}")

                # ── 2. RENDER FRONTAL (Multicopia en línea + Recortes) ──
                # Ocultamos la pieza original del render
                part_obj.hide_render = True
                
                # Determinamos el tamaño de la pieza rotada para saber cuántas caben
                # Instanciamos una pieza de test para medir
                measure_obj = part_obj.copy()
                measure_obj.data = part_obj.data.copy()
                bpy.context.scene.collection.objects.link(measure_obj)
                measure_obj.hide_render = True
                measure_obj.location = (0.0, 0.0, 0.0)
                
                quat = pose.get("orientation_quat")
                if quat and len(quat) == 4:
                    measure_obj.rotation_mode = 'QUATERNION'
                    measure_obj.rotation_quaternion = mathutils.Quaternion(quat)
                else:
                    measure_obj.rotation_mode = 'XYZ'
                    measure_obj.rotation_euler = mathutils.Euler(pose.get("orientation_euler", [0,0,0]))
                measure_obj.rotation_mode = 'XYZ'
                measure_obj.rotation_euler.z += rot_rad
                bpy.context.view_layer.update()
                
                m_bbox = [measure_obj.matrix_world @ mathutils.Vector(c) for c in measure_obj.bound_box]
                w_piece = max(pt.x for pt in m_bbox) - min(pt.x for pt in m_bbox)
                bpy.data.objects.remove(measure_obj, do_unlink=True)

                # Calcular N_max copias en fila a lo largo de X (ancho 20cm, margen 0.5cm)
                # Formula: N_max = floor(19.5 / (w_piece + 0.5))
                # En Blender units, w_piece es en BU
                n_copies = int(math.floor(19.5 / (w_piece + 0.5)))
                n_copies = max(1, n_copies) # Al menos 1 copia en el centro

                # Posiciones de los centros de las copias
                if n_copies > 1:
                    g = (19.0 - n_copies * w_piece) / (n_copies - 1)
                    centers_x = [-9.5 + w_piece/2.0 + i * (w_piece + g) for i in range(n_copies)]
                else:
                    centers_x = [0.0]

                # Instanciar copias en escena
                placed_copies = []
                for i, cx in enumerate(centers_x):
                    oc = part_obj.copy()
                    oc.data = part_obj.data.copy()
                    oc.name = f"Placed_Copy_{i}"
                    bpy.context.scene.collection.objects.link(oc)
                    oc.hide_render = False
                    
                    quat = pose.get("orientation_quat")
                    if quat and len(quat) == 4:
                        oc.rotation_mode = 'QUATERNION'
                        oc.rotation_quaternion = mathutils.Quaternion(quat)
                    else:
                        oc.rotation_mode = 'XYZ'
                        oc.rotation_euler = mathutils.Euler(pose.get("orientation_euler", [0,0,0]))
                    oc.rotation_mode = 'XYZ'
                    oc.rotation_euler.z += rot_rad
                    oc.location = (cx, 0.0, 0.0)
                    bpy.context.view_layer.update()
                    
                    bbox_copy = [oc.matrix_world @ mathutils.Vector(c) for c in oc.bound_box]
                    min_z = min(pt.z for pt in bbox_copy)
                    oc.location.z = -min_z + 0.02
                    placed_copies.append(oc)

                bpy.context.view_layer.update()

                # Renderizar para cada color y recortar
                for color_hex in allowed_colors:
                    mat = create_abs_plastic_material(f"#{color_hex}")
                    for oc in placed_copies:
                        oc.data.materials.clear()
                        oc.data.materials.append(mat)
                    
                    bpy.context.view_layer.update()

                    # Render Frontal completo a archivo temporal
                    scene.camera = cam_f
                    temp_fpath = os.path.join(out_dir, "frontal", "temp_full_frontal.png")
                    scene.render.filepath = temp_fpath
                    
                    try:
                        bpy.ops.render.render(write_still=True)
                        
                        # Abrir la imagen renderizada y recortar por cada copia
                        full_img = Image.open(temp_fpath)
                        for inst_idx, oc in enumerate(placed_copies):
                            bbox = compute_bbox_pixel(oc, cam_f, scene)
                            if bbox:
                                crop_img = full_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                                fname = f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}_inst{inst_idx:02d}.png"
                                fpath = os.path.join(out_dir, "frontal", fname)
                                crop_img.save(fpath)
                                total_rendered += 1
                                
                        if os.path.exists(temp_fpath):
                            os.remove(temp_fpath)
                    except Exception as e:
                        log.warning(f"Render frontal o recortes fallidos ({part_ref} pose{pose_idx} rot{rot_deg}): {e}")

                # Limpieza de las copias
                for oc in placed_copies:
                    bpy.data.objects.remove(oc, do_unlink=True)

            cleanup_piece()

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "generate_eevee_dinov2_refs.py",
                         duration_s=_duration,
                         total_rendered=total_rendered,
                         output_dir=out_dir)
    log.info(f"Generadas {total_rendered} imágenes de referencia DINOv2.")


if __name__ == "__main__":
    main()
