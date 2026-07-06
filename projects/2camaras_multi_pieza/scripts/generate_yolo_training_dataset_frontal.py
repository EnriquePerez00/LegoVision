# -*- coding: utf-8 -*-
"""scripts/generate_yolo_training_dataset_frontal.py
===================================================
Genera el dataset de entrenamiento YOLO desde la cámara frontal
posicionando piezas en fila a lo largo del eje X (línea Y = 0).
"""
import os, sys, random, math, argparse, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'scratch'))

# Add user site-packages for Blender isolated environment
user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

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
    setup_studio_lighting,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object
from generate_yolo_training_dataset import create_abs_plastic_material_dr, load_lego_color_palette, load_set_geometries

# Parámetros desde config.yaml
SELECTED_PARTS = cfg.pieces.selected_parts
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
BELT_FRICTION = cfg.scene.belt.roughness
BELT_RESTITUTION = cfg.scene.belt.restitution

RENDER_RES_SQUARE = cfg.render.resolution.width
TOP_LIGHT_Z = cfg.scene.lighting.led_strip.z_bu
TOP_LIGHT_SIZE_X = cfg.scene.lighting.led_strip.size_x_bu
TOP_LIGHT_SIZE_Y = cfg.scene.lighting.led_strip.size_y_bu
TOP_LIGHT_ENERGY = cfg.scene.lighting.led_strip.energy_w

WORLD_BG_STRENGTH = cfg.scene.world.strength
WORLD_BG_COLOR = tuple(cfg.scene.world.color)

YOLO_PIECES_PER_FRAME_MIN = cfg.yolo.dataset.pieces_per_frame_min
YOLO_PIECES_PER_FRAME_MAX = cfg.yolo.dataset.pieces_per_frame_max
YOLO_EMPTY_FRAME_RATIO = cfg.yolo.dataset.empty_frame_ratio
# Total piezas objetivo (configurable por config.yaml; fallback 1000 para retro-compat)
YOLO_TOTAL_PIECES_TARGET = getattr(cfg.yolo.dataset, "total_pieces_frontal", 1000)


def setup_led_strip_lights(randomize=True):
    """Setup laboratory-style lightbox lighting.

    - 1 large main dome cenital (35x35cm @ z=12cm, 2000W)
    - 4 wall panels lateral (N/S/E/W @ z=6cm, 600W each)
    - 1 ground fill below belt (-0.5cm, 200W)
    - World ambient at 30% strength
    Total ~5000W -> simulates a high-clarity laboratory inspection lightbox.
    """
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Frontal", "Lab_Floor"}
    for o in list(bpy.context.scene.objects):
        if o.type == 'LIGHT' and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    # World ambient (30% white)
    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.3

    # Variacion calida/fria suave
    if randomize and random.choice([True, False]):
        base_color = (1.0, 0.97 * random.uniform(0.99, 1.01), 0.94 * random.uniform(0.99, 1.01))
    elif randomize:
        base_color = (0.96 * random.uniform(0.99, 1.01), 0.98 * random.uniform(0.99, 1.01), 1.0)
    else:
        base_color = (1.0, 1.0, 1.0)

    # 1. MAIN DOME (cenital grande)
    bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, TOP_LIGHT_Z))
    main = bpy.context.active_object
    main.name = "Lab_Main_Dome"
    main.data.size = TOP_LIGHT_SIZE_X
    main.data.size_y = TOP_LIGHT_SIZE_Y
    main.data.shape = 'RECTANGLE'
    main.data.color = base_color
    main.data.energy = TOP_LIGHT_ENERGY * (random.uniform(0.97, 1.03) if randomize else 1.0)

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
        wp.data.color = base_color
        wp.data.energy = 600.0 * (random.uniform(0.95, 1.05) if randomize else 1.0)
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
    gf.data.color = base_color
    gf.data.energy = 200.0 * (random.uniform(0.95, 1.05) if randomize else 1.0)
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
    """Crea la cinta y los carriles laterales."""
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


def setup_camera():
    """Configura la cámara frontal simétrica (Y = -15.0, Z = 0.0)."""
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    cam_name = "Cam_Frontal"
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
    else:
        bpy.ops.object.camera_add(location=(0.0, -15.0, 0.0))
        cam = bpy.context.active_object
        cam.name = cam_name

    cam.location = (0.0, -15.0, 0.0)
    cam.constraints.clear()
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    cam.data.type = 'PERSP'
    cam.data.lens = 27.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0

    bpy.context.scene.camera = cam
    return cam


def create_templates(geom_list):
    """Carga plantillas ocultas."""
    if 'Templates' not in bpy.data.collections:
        col = bpy.data.collections.new('Templates')
        bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections['Templates']
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    col.hide_viewport = True
    col.hide_render = True
    tmap = {}

    for part in geom_list:
        ref = part['ref']
        part_path = get_ldraw_part_path(ref)
        existing = set(bpy.context.scene.objects)
        obj = None
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs = [o for o in bpy.context.scene.objects if o not in existing]
                par = next((o for o in new_objs if o.parent is None), None)
                obj = get_single_mesh_object(par) if par else None
            except Exception as e:
                print(f"[FrontalGen] Error importando {ref}: {e}")
        if not obj:
            generate_detailed_fallback_mesh(ref)
            obj = bpy.context.active_object

        obj.name = f"Template_{ref}"
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        # Normalizar escala
        if obj.data and hasattr(obj.data, 'vertices'):
            verts = [v.co for v in obj.data.vertices]
            if verts:
                xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
                mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
                factor = 0.04 if mx > 5.0 else 1.0
                for v in obj.data.vertices:
                    v.co.x = (v.co.x - (max(xs)+min(xs))/2) * factor
                    v.co.y = (v.co.y - (max(ys)+min(ys))/2) * factor
                    v.co.z = (v.co.z - (max(zs)+min(zs))/2) * factor
                obj.data.update()
        obj.scale = (1.0, 1.0, 1.0)
        apply_bevel_modifier(obj)

        for c2 in list(obj.users_collection):
            c2.objects.unlink(obj)
        col.objects.link(obj)
        tmap[ref] = obj
    return tmap


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



def spawn_pieces_frontal_line(tmap, geom_list, color_palette):
    """Spawnea una fila de piezas a lo largo del eje X (línea Y = 0)."""
    placed = []
    limit_bu = 9.5
    margin_bu = 0.5  # 0.5 cm de separación mínima

    current_x = -limit_bu
    selected_parts = random.choices(geom_list, k=15)
    import mathutils

    for part in selected_parts:
        if current_x >= limit_bu:
            break

        ref = part['ref']
        tmpl = tmap.get(ref)
        if not tmpl:
            continue

        poses = get_stable_poses_from_db_subprocess(ref)
        if not poses:
            poses = [{"pose_index": 0, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}]

        pose = random.choice(poses)
        rot_z = random.uniform(0.0, 2 * math.pi)

        # Instanciar temporalmente
        oc = tmpl.copy()
        oc.data = tmpl.data.copy()
        oc.name = f"Piece_Spawned_{ref}_{len(placed)}"
        bpy.context.scene.collection.objects.link(oc)

        # Orientación
        quat = pose.get("orientation_quat")
        if quat and len(quat) == 4:
            oc.rotation_mode = 'QUATERNION'
            oc.rotation_quaternion = mathutils.Quaternion(quat)
        else:
            euler = pose.get("orientation_euler")
            if euler and len(euler) == 3:
                oc.rotation_mode = 'XYZ'
                oc.rotation_euler = mathutils.Euler(euler)

        oc.rotation_mode = 'XYZ'
        oc.rotation_euler.z += rot_z
        bpy.context.view_layer.update()

        # Bounding box
        bbox = [oc.matrix_world @ mathutils.Vector(c) for c in oc.bound_box]
        min_x = min(pt.x for pt in bbox)
        max_x = max(pt.x for pt in bbox)
        w_x = max_x - min_x

        cx = current_x + w_x/2.0

        if cx + w_x/2.0 > limit_bu:
            bpy.data.objects.remove(oc, do_unlink=True)
            break

        oc.location = (cx, 0.0, 0.0)
        bpy.context.view_layer.update()

        bbox_world = [oc.matrix_world @ mathutils.Vector(c) for c in oc.bound_box]
        min_z = min(pt.z for pt in bbox_world)
        oc.location.z = -min_z + 0.02

        color_hex = random.choice(color_palette)
        mat = create_abs_plastic_material_dr(color_hex)
        oc.data.materials.clear()
        oc.data.materials.append(mat)

        placed.append((oc, ref))
        current_x = cx + w_x/2.0 + margin_bu
        
    return placed


def cleanup_placed_pieces():
    bpy.ops.object.select_all(action='DESELECT')
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R", "Cam_Frontal", "Lab_Floor"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and not o.name.startswith("Template_"):
            try:
                o.select_set(True)
            except:
                pass
    bpy.ops.object.delete()


def compute_bbox_yolo(obj, cam, scene):
    """Calcula la bounding box 2D normalizada en el formato YOLO."""
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
        
    w, h = x1 - x0, y1 - y0
    if w < 0.008 or h < 0.008:
        return None
    return [x0 + w/2.0, 1.0 - (y0 + h/2.0), w, h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    pa = parser.parse_known_args(args_raw)[0]

    output_dir = pa.output_dir
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    geom_list = load_set_geometries("75078-1")
    color_palette = load_lego_color_palette()

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    camera = setup_camera()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE

    tmap = create_templates(geom_list)

    total_pieces_needed = YOLO_TOTAL_PIECES_TARGET
    total_pieces_placed = 0
    frame_idx = 0

    print(f"[FrontalGen] Generando dataset YOLO Frontal para un total de {total_pieces_needed} piezas (config.yolo.dataset.total_pieces_frontal)...")

    while total_pieces_placed < total_pieces_needed:
        cleanup_placed_pieces()
        
        # Frame vacío aleatorio (5% ratio)
        if random.random() < YOLO_EMPTY_FRAME_RATIO:
            img_fn = f"train_frontal_{frame_idx:05d}.png"
            lbl_fn = f"train_frontal_{frame_idx:05d}.txt"
            scene.render.filepath = os.path.join(images_dir, img_fn)
            setup_led_strip_lights(randomize=True)
            bpy.ops.render.render(write_still=True)
            open(os.path.join(labels_dir, lbl_fn), "w").close()
            print(f"[Empty] frame frontal {frame_idx+1}")
            frame_idx += 1
            continue

        placed = spawn_pieces_frontal_line(tmap, geom_list, color_palette)
        
        if not placed:
            continue

        labels = []
        bpy.context.view_layer.update()
        
        for obj, ref in placed:
            bb = compute_bbox_yolo(obj, camera, scene)
            if bb:
                labels.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

        if labels:
            img_fn = f"train_frontal_{frame_idx:05d}.png"
            lbl_fn = f"train_frontal_{frame_idx:05d}.txt"
            scene.render.filepath = os.path.join(images_dir, img_fn)
            setup_led_strip_lights(randomize=True)
            
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"[WARN] Render fallido frame {frame_idx}: {e}")
                continue
                
            with open(os.path.join(labels_dir, lbl_fn), "w") as lf:
                lf.write("\n".join(labels) + "\n")
                
            total_pieces_placed += len(labels)
            print(f"[OK] frame frontal {frame_idx+1} - {len(labels)} piezas colocadas (Total: {total_pieces_placed}/{total_pieces_needed})")
            frame_idx += 1
        else:
            print(f"[SKIP] frame frontal {frame_idx+1} sin piezas detectadas en FOV")

    cleanup_placed_pieces()
    for obj in list(tmap.values()):
        bpy.data.objects.remove(obj)
    print(f"[FrontalGen DONE] Dataset generado en {output_dir}")


if __name__ == "__main__":
    main()
