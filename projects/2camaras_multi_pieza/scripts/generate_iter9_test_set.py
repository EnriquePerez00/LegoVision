# -*- coding: utf-8 -*-
"""scripts/generate_validation_renders_full_set.py
======================================
Genera el set de test para la configuración 2camaras_multi_pieza.
"""
import os, sys, random, math, json

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
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

SELECTED_PARTS = cfg.pieces.selected_parts
PART_COLOR_HEX = cfg.pieces.test_color_hex
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


def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


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


def get_stable_poses_from_db_subprocess(part_ref):
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if part_ref in cache:
                return cache[part_ref]
        except Exception as e:
            pass
    return []


def get_2d_bbox(obj, scene, camera):
    bbox_coords = _get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_coords:
        co_2d = world_to_camera_view(scene, camera, v)
        xs.append(co_2d.x)
        ys.append(co_2d.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0)),
    ]


def setup_test_camera(cam_name, location):
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
        cam.location = location
    else:
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object
        cam.name = cam_name

    cam.constraints.clear()
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    cam.data.type = 'PERSP'
    cam.data.lens = 27.0
    cam.data.clip_start = 0.01
    return cam


def setup_led_strip_lights():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R", "Cam_Cenital", "Cam_Frontal"}
    for o in list(bpy.context.scene.objects):
        if o.type == 'LIGHT' and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    z = TOP_LIGHT_Z
    sx = TOP_LIGHT_SIZE_X
    sy = TOP_LIGHT_SIZE_Y
    energy = TOP_LIGHT_ENERGY

    neutral_color = (1.0, 1.0, 1.0)

    segments = [
        ("LED_Strip_S", (0.0, -sy/2.0, z), (sx, 0.5)),
        ("LED_Strip_N", (0.0, sy/2.0, z), (sx, 0.5)),
        ("LED_Strip_W", (-sx/2.0, 0.0, z), (0.5, sy)),
        ("LED_Strip_E", (sx/2.0, 0.0, z), (0.5, sy)),
    ]

    for name, loc, size in segments:
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.active_object
        light.name = name
        light.data.size = size[0]
        light.data.size_y = size[1]
        light.data.shape = 'RECTANGLE'
        light.data.color = neutral_color
        light.data.energy = energy


def create_belt_collider():
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


def cleanup_piece_objects():
    keep = {
        "Conveyor_Belt_Plane", "Camera_Target",
        "Side_Rail_L", "Side_Rail_R",
        "Cam_Cenital", "Cam_Frontal",
        "LED_Strip_S", "LED_Strip_N", "LED_Strip_W", "LED_Strip_E"
    }
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try: o.select_set(True)
            except: pass
    bpy.ops.object.delete()


def build_scene():
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        try: o.select_set(True)
        except: pass
    bpy.ops.object.delete()

    setup_physics_world()
    create_belt_collider()
    setup_led_strip_lights()

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = WORLD_BG_COLOR
            bg.inputs["Strength"].default_value = WORLD_BG_STRENGTH

    enable_metal_gpu_acceleration()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE

    cam_cenital = setup_test_camera("Cam_Cenital",  (0.0, 0.0, 15.0))
    cam_frontal = setup_test_camera("Cam_Frontal",  (0.0, -15.0, 0.0))

    return cam_cenital, cam_frontal


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parsed_args = parser.parse_known_args(args)[0]

    output_dir = parsed_args.output_dir
    num_samples = parsed_args.num_samples
    os.makedirs(output_dir, exist_ok=True)

    cam_cenital, cam_frontal = build_scene()
    scene = bpy.context.scene

    cameras = {
        "cenital":   cam_cenital,
        "frontal":   cam_frontal,
    }

    stable_poses_cache = {}
    for part in SELECTED_PARTS:
        poses = get_stable_poses_from_db_subprocess(part)
        if poses:
            stable_poses_cache[part] = poses
        else:
            print(f"[TestGen Warning] Sin poses estables para {part}")

    results_meta = []

    for i in range(num_samples):
        part_ref = random.choice(SELECTED_PARTS)
        print(f"\n[{i+1}/{num_samples}] Generando muestra de test: {part_ref}")

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
                pass

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            continue

        bpy.ops.object.select_all(action='DESELECT')
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        poses = stable_poses_cache.get(part_ref, [])
        selected_pose_idx = 0
        if poses:
            pose = random.choice(poses)
            selected_pose_idx = pose.get("pose_index", 0)
            quat = pose.get("orientation_quat")
            if quat and len(quat) == 4:
                part_obj.rotation_mode = 'QUATERNION'
                part_obj.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                euler = pose.get("orientation_euler")
                if euler and len(euler) == 3:
                    part_obj.rotation_mode = 'XYZ'
                    part_obj.rotation_euler = mathutils.Euler(euler)
        else:
            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler = (0, 0, 0)

        # Rotación Z aleatoria y offset aleatorio en X, Y para simular cintas con flujo real
        # ya que ahora tenemos compensaciones matemáticas
        part_obj.rotation_mode = 'XYZ'
        part_obj.rotation_euler.z += random.uniform(0.0, math.pi * 2)

        # Ubicar aleatoriamente en el FOV útil de la cinta
        part_obj.location = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 0.0)
        bpy.context.view_layer.update()
        
        bbox_world = _get_world_bbox(part_obj)
        min_z = min(pt.z for pt in bbox_world)
        part_obj.location.z = -min_z + 0.02

        # Buscar el color real en el catálogo para ser consistentes con el filtro de color
        from core.db.set_catalog import REAL_SETS
        part_color_hex = f"#{PART_COLOR_HEX}" if not PART_COLOR_HEX.startswith("#") else PART_COLOR_HEX
        for p in REAL_SETS["75078-1"]["parts"]:
            if p["ref"] == part_ref:
                part_color_hex = p["color_hex"]
                break
        
        mat_abs = create_abs_plastic_material(part_color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat_abs)

        bpy.context.view_layer.update()

        sample_meta = {
            "ref":         part_ref,
            "pose_index":  selected_pose_idx,
            "color_hex":   part_color_hex,
            "cameras":     {},
        }

        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()

            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"sample_{i:03d}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path

            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"[WARN] Render fallido {cam_name} muestra {i}: {e}")
                continue

            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "bbox_norm": bbox_norm,
                "image_path": file_path,
            }

        if len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"[OK] Muestra {i+1}: {part_ref} | pose={selected_pose_idx}")
        else:
            print(f"[WARN] Muestra {i+1} incompleta, descartada.")

        cleanup_piece_objects()

    meta_path = os.path.join(output_dir, "test_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id":          "75078-1",
            "render_engine":   "BLENDER_EEVEE",
            "resolution":      f"{RENDER_RES_SQUARE}x{RENDER_RES_SQUARE}",
            "samples_count":   len(results_meta),
            "renders":         results_meta,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[TestGen DONE] {len(results_meta)} muestras en {output_dir}")


if __name__ == "__main__":
    main()
