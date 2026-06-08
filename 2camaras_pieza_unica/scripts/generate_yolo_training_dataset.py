# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_yolo_training_dataset.py
=================================================================
Genera dataset YOLO para el setup de pieza única centrada.
Cada frame contiene exactamente 1 pieza LEGO en el centro de la cinta,
renderizada simultáneamente desde cámara cenital y lateral.
"""
import os, sys, random, math, argparse, json

# Blender-compatible user site-packages
user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from config_loader import cfg
from generate_synthetic_set import (
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

# Fuente única de verdad: TARPS + rotación analítica desde
# contact_normal. Ver _pose_utils.py y docs/stable_pose_selection_rule.md.
from _pose_utils import (
    apply_stable_pose,
    get_stable_poses_for_ref,
    select_pose_tarps,
    TARPS_MIN_TIPPING_DEFAULT,
)

# ── Logging ──
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("yolo")

# ── Config ──
SELECTED_PARTS = cfg.pieces.selected_parts
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
RENDER_RES = cfg.render.resolution.width
MIN_CONTACT_DIM_MM = cfg.stable_poses.min_contact_dimension_mm   # legacy
MIN_STABILITY = cfg.stable_poses.render_min_stability             # legacy
# TARPS - Tipping-Aware Random Pose Selection
# (ver docs/stable_pose_selection_rule.md). El umbral se centraliza
# en _pose_utils.TARPS_MIN_TIPPING_DEFAULT, pero permitimos override
# desde config.yaml para experimentos.
TARPS_MIN_TIPPING = getattr(
    cfg.stable_poses, "tarps_min_tipping", TARPS_MIN_TIPPING_DEFAULT
)


def load_lego_color_palette():
    path = os.path.join(project_root, 'database', 'color_catalog.json')
    fallback = ['#A0A5A9', '#1B1B1B', '#C91A09', '#F2F3F2', '#FE8A18',
                '#0A3C9F', '#5A5A5A', '#3B5E28', '#F2CD37', '#FF7E14']
    if not os.path.exists(path):
        return fallback
    with open(path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    pal = []
    for _, info in catalog.items():
        hx = info.get('hex', '')
        if hx and info.get('alpha', 1.0) >= 0.6 and info.get('material_type', 'solid') in ('solid', 'metallic', 'rubber'):
            pal.append(hx if hx.startswith('#') else '#' + hx)
    return list(set(pal)) or fallback


def get_stable_poses(part_ref):
    """Wrapper compatibilidad: delega en `_pose_utils.get_stable_poses_for_ref`.
    Devuelve poses ordenadas por tipping_energy_ratio descendente.
    Ver docs/stable_pose_selection_rule.md."""
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    return get_stable_poses_for_ref(part_ref, cache_path)


# `select_pose_tarps` ya está importado de `_pose_utils` arriba; lo
# re-exportamos aquí para compatibilidad con scripts que hagan
#   from generate_yolo_training_dataset import select_pose_tarps
# pero la fuente de verdad vive en `_pose_utils.py`.


def setup_lab_lightbox():
    """Setup laboratory lightbox lighting."""
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral", "Lab_Floor"}
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

    # Main Dome
    bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, 12.0))
    main = bpy.context.active_object
    main.name = "Lab_Main_Dome"
    main.data.size = 35.0
    main.data.size_y = 35.0
    main.data.shape = 'RECTANGLE'
    main.data.color = neutral_color
    main.data.energy = 2000.0

    # Wall Panels
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

    # Ground Fill
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
    """Create black floor for lateral camera background."""
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
    floor.data.materials.clear()
    floor.data.materials.append(mat)


def create_belt_collider():
    """Create belt with side rails."""
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

    # Side rails
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
    """Configure cenital and lateral cameras."""
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
    cam_c.data.clip_end = 100.0

    # Lateral
    cam_l_name = "Cam_Lateral"
    if cam_l_name in bpy.data.objects:
        cam_l = bpy.data.objects[cam_l_name]
    else:
        bpy.ops.object.camera_add(location=(15.0, 0.0, 2.5))
        cam_l = bpy.context.active_object
        cam_l.name = cam_l_name
    cam_l.location = (15.0, 0.0, 2.5)
    cam_l.constraints.clear()
    track_l = cam_l.constraints.new(type='TRACK_TO')
    track_l.target = target
    track_l.track_axis = 'TRACK_NEGATIVE_Z'
    track_l.up_axis = 'UP_Y'
    cam_l.data.type = 'PERSP'
    cam_l.data.lens = 27.0
    cam_l.data.clip_start = 0.01
    cam_l.data.clip_end = 100.0

    return cam_c, cam_l


def _normalize_piece(obj):
    """Normalize LDraw piece to BU scale."""
    if not obj.data or not hasattr(obj.data, 'vertices'):
        return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return 1.0
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if mx < 1e-6:
        return 1.0
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
    """Remove all temporary piece objects."""
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral", "Lab_Floor",
            "Lab_Main_Dome", "Lab_Wall_N", "Lab_Wall_S", "Lab_Wall_E", "Lab_Wall_W", "Lab_Ground_Fill"}
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and o.type not in ('CAMERA', 'LIGHT', 'EMPTY'):
            try:
                o.select_set(True)
            except:
                pass
    bpy.ops.object.delete()


def compute_bbox_yolo(obj, cam, scene):
    """Compute YOLO-format bbox [cx, cy, w, h] normalized."""
    world_verts = []
    if obj.type == 'MESH' and obj.data:
        m = obj.matrix_world
        world_verts = [m @ v.co for v in obj.data.vertices]
    if not world_verts:
        world_verts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    if not world_verts:
        return None
    xs, ys = [], []
    for v in world_verts:
        c = world_to_camera_view(scene, cam, v)
        xs.append(c.x)
        ys.append(c.y)
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0
    if w < 0.005 or h < 0.005:
        return None
    return [x0 + w / 2.0, 1.0 - (y0 + h / 2.0), w, h]


def generate_dataset(camera_type, output_dir, num_frames):
    """Main generation function."""
    import time as _time
    _t_start = _time.perf_counter()

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    log_execution_header(log, "generate_yolo_training_dataset.py",
                         camera_type=camera_type, output_dir=output_dir,
                         num_frames=num_frames)

    color_palette = load_lego_color_palette()

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    cam_c, cam_l = setup_cameras()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES

    # Select active camera
    active_cam = cam_c if camera_type == "cenital" else cam_l

    empty_ratio = cfg.yolo.dataset.empty_frame_ratio
    num_empty = max(1, int(num_frames * empty_ratio))
    num_piece_frames = num_frames - num_empty
    frame_types = ["piece"] * num_piece_frames + ["empty"] * num_empty
    random.shuffle(frame_types)

    saved = 0
    for fi, ftype in enumerate(frame_types):
        cleanup_piece()

        if ftype == "empty":
            # Empty frame (no piece)
            scene.camera = active_cam
            img_fn = f"train_{fi:05d}.png"
            lbl_fn = f"train_{fi:05d}.txt"
            scene.render.filepath = os.path.join(images_dir, img_fn)
            bpy.ops.render.render(write_still=True)
            open(os.path.join(labels_dir, lbl_fn), "w").close()
            saved += 1
            continue

        # Select random piece and pose (regla TARPS - ver docs/stable_pose_selection_rule.md)
        part_ref = random.choice(SELECTED_PARTS)
        poses = get_stable_poses(part_ref)
        pose = select_pose_tarps(poses)
        if pose is None:
            pose = {"orientation_quat": [1.0, 0.0, 0.0, 0.0]}

        # Load mesh
        part_path = get_ldraw_part_path(part_ref)
        existing = set(bpy.context.scene.objects)
        part_obj = None

        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing]
                par = next((o for o in new_objs if o.parent is None), None)
                if par:
                    part_obj = get_single_mesh_object(par)
            except Exception as e:
                log.warning(f"import LDraw {part_ref}: {e}")

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            continue

        # Prepare piece
        bpy.ops.object.select_all(action='DESELECT')
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        # Aplicación canónica de pose estable: rotación analítica
        # desde `contact_normal` (determinista) + Z aleatorio + snap
        # a la cinta. Reemplaza el antiguo bloque que usaba el
        # `orientation_quat` del cache, que en algunas piezas estaba
        # corrupto por un bug en `simulate_stable_poses · transform_apply`.
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Apply random color
        color_hex = random.choice(color_palette)
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)

        # Render and compute bbox
        scene.camera = active_cam
        bpy.context.view_layer.update()

        bb = compute_bbox_yolo(part_obj, active_cam, scene)
        if not bb:
            continue

        img_fn = f"train_{fi:05d}.png"
        lbl_fn = f"train_{fi:05d}.txt"
        scene.render.filepath = os.path.join(images_dir, img_fn)

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log.warning(f"Render failed frame {fi}: {e}")
            continue

        with open(os.path.join(labels_dir, lbl_fn), "w") as lf:
            lf.write(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        saved += 1
        if saved % 100 == 0:
            log.info(f"  [{saved}/{num_frames}] frames guardados...")

    cleanup_piece()
    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "generate_yolo_training_dataset.py",
                         duration_s=_duration,
                         total_saved=saved,
                         camera=camera_type)


if __name__ == "__main__":
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="cenital", choices=["cenital", "lateral"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=5000)
    pa = parser.parse_known_args(args_raw)[0]

    out = pa.output_dir or os.path.join(project_root, "data", f"yolo_{pa.camera}")
    generate_dataset(pa.camera, out, pa.num_frames)
