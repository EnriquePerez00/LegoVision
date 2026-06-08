# -*- coding: utf-8 -*-
"""scripts/generate_iter9_test_set.py
======================================
Genera el set de test de Iteración 9 usando EXACTAMENTE el mismo pipeline
de render que el training YOLO (source of truth):

  - Motor:          BLENDER_EEVEE  (= idéntico a training)
  - Resolución:     640 × 640      (= RENDER_RES_SQUARE de scene_config)
  - Belt:           mismo material, mismas dimensiones, mismo color
  - Iluminación:    Top_Diffuse_Light + 4 Corner Lights (= training)
  - Cámara:         PERSP lens=52.5mm Z=15BU  (= setup_ortho_camera del training)

Una pieza por frame, 3 cámaras (cenital, lateral_l, lateral_r).

Uso:
  blender -b -P scripts/generate_iter9_test_set.py -- \
      --output_dir data/iter9_test --num_samples 100
"""
import os, sys, random, math, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'scratch'))

try:
    import bpy
    import bpy_extras
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

# ── Funciones compartidas con el pipeline de training ────────────────────────
# Exactamente las mismas que usa generate_yolo_training_dataset.py
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
from generate_yolo_training_dataset import (
    setup_corner_lights,
    create_belt_collider,
    CORNER_LIGHT_NAMES,
    CORNER_LIGHT_POSITIONS,
)
from scene_config import (
    RENDER_RES_SQUARE,          # 640
    LDRAW_TO_BU, LDRAW_THRESHOLD,
    TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z,
    WORLD_BG_STRENGTH, WORLD_BG_COLOR,
    BELT_COLOR_LINEAR,
)

# ── Piezas y colores ─────────────────────────────────────────────────────────
SELECTED_PARTS = ["3005", "3001", "3039", "3665", "3010",
                  "3002", "3020", "4070", "4032", "3700"]

# Color uniforme Light Bluish Gray (A0A5A9 / código 85) — igual que iter7/8
PART_COLOR_HEX = "A0A5A9"


# ── Helpers compartidos ──────────────────────────────────────────────────────

def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece(obj):
    """Misma lógica que _normalize_piece() de generate_yolo_training_dataset."""
    if not obj.data or not hasattr(obj.data, 'vertices'):
        return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return 1.0
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if mx < 1e-6:
        return 1.0
    factor = LDRAW_TO_BU if mx > LDRAW_THRESHOLD else 1.0
    cx = (max(xs)+min(xs))/2; cy = (max(ys)+min(ys))/2; cz = (max(zs)+min(zs))/2
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) * factor
        v.co.y = (v.co.y - cy) * factor
        v.co.z = (v.co.z - cz) * factor
    obj.data.update(); obj.scale = (1.0, 1.0, 1.0); obj.location = (0.0, 0.0, 0.0)
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
            print(f"[WARN] Loading local cache: {e}")
    return []


def get_2d_bbox(obj, scene, camera):
    """Bounding box 2D normalizado [x1, y1, x2, y2] en coordenadas imagen (Y invertido)."""
    bbox_coords = _get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_coords:
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, v)
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
    """
    Crea o reutiliza una cámara PERSP lens=52.5mm apuntando a (0,0,0).
    Misma configuración que setup_ortho_camera('lateral'/'cenital') del training.
    """
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
    track.name = "Track_To"
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    cam.data.type = 'PERSP'
    cam.data.lens = 52.5          # Mismo focal que training (lens 52.5mm)
    cam.data.clip_start = 0.01
    return cam

def cleanup_piece_objects():
    """Elimina sólo los objetos de piezas, manteniendo belt, luces y cámaras."""
    keep = {
        "Conveyor_Belt_Plane", "Camera_Target",
        "Top_Diffuse_Light", "Key_Light", "Fill_Light", "Rim_Light",
        "Side_Rail_R",
        "Cam_Cenital", "Cam_Frontal",
    }
    keep.update(CORNER_LIGHT_NAMES)
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()
    # Limpiar materiales y meshes huérfanos
    for mat in list(bpy.data.materials):
        if mat.name.startswith('DR_') and mat.users == 0:
            bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def build_scene():
    """
    Construye la escena base para 2 cámaras:
      Cenital + Frontal (perfil longitudinal bajo).
    """
    # Limpiar escena completa al inicio
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()

    # ── 1. Física (gravitación) ───────────────────────────────────────────
    setup_physics_world()

    # ── 2. Belt y Colisiones ──────────────────────────────────────────────
    create_belt_collider()
    
    # Remover Side_Rail_L para que no bloquee el plano de fondo de la cámara lateral
    rail_l = bpy.data.objects.get("Side_Rail_L")
    if rail_l:
        bpy.ops.object.select_all(action='DESELECT')
        rail_l.select_set(True)
        bpy.ops.object.delete()

    # ── 3. Iluminación estudio ────────────────────────────────────────────
    setup_studio_lighting()

    # ── 4. Luz cenital principal ──────────────────────────────────────────
    top = bpy.data.objects.get("Top_Diffuse_Light")
    if not top:
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, TOP_LIGHT_Z))
        top = bpy.context.active_object
        top.name = "Top_Diffuse_Light"
    top.location = (0.0, 0.0, TOP_LIGHT_Z)
    top.data.size = TOP_LIGHT_SIZE
    top.data.energy = TOP_LIGHT_ENERGY

    # ── 5. 4 Corner Lights ────────────────────────────────────────────────
    setup_corner_lights()

    # ── 6. World background ───────────────────────────────────────────────
    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = WORLD_BG_COLOR
            bg.inputs["Strength"].default_value = WORLD_BG_STRENGTH

    # ── 7. Motor EEVEE + resolución 640×640 ────────────────────────────────
    enable_metal_gpu_acceleration()
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES_SQUARE   # 640
    scene.render.resolution_y = RENDER_RES_SQUARE   # 640

    print(f"[TestGen] Escena construida: EEVEE, {RENDER_RES_SQUARE}×{RENDER_RES_SQUARE}")

    # ── 8. Cámaras (Nuevo setup: Cenital + Frontal perfil) ────────
    cam_cenital = setup_test_camera("Cam_Cenital",  ( 0.0,  0.0, 15.0))
    cam_frontal = setup_test_camera("Cam_Frontal",  ( 0.0, -15.0,  2.5))

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

    # ── 1. Escena base ────────────────────────────────────────────────────
    cam_cenital, cam_frontal = build_scene()
    scene = bpy.context.scene

    cameras = {
        "cenital":   cam_cenital,
        "frontal":   cam_frontal,
    }

    # ── 2. Cache de poses estables para las 10 piezas ─────────────────────
    stable_poses_cache = {}
    for part in SELECTED_PARTS:
        poses = get_stable_poses_from_db_subprocess(part)
        if poses:
            stable_poses_cache[part] = poses
            print(f"[TestGen] {len(poses)} poses estables para pieza {part}")
        else:
            print(f"[TestGen Warning] Sin poses estables para {part}")

    results_meta = []

    # ── 3. Loop de render ─────────────────────────────────────────────────
    for i in range(num_samples):
        part_ref = random.choice(SELECTED_PARTS)
        print(f"\n[{i+1}/{num_samples}] Generando muestra: {part_ref}")

        # ── Cargar malla LDraw / fallback ─────────────────────────────────
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
                print(f"[WARN] import LDraw {part_ref}: {e}")

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            print(f"[ERR] Sin mesh para {part_ref}. Saltando.")
            continue

        # ── Origin + normalizar escala (= training) ───────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        _normalize_piece(part_obj)

        # ── Aplicar bevel (= training) ────────────────────────────────────
        apply_bevel_modifier(part_obj)

        # ── Pose estable aleatoria ────────────────────────────────────────
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

        # Rotación horizontal aleatoria (= training spawn_pieces)
        part_obj.rotation_mode = 'XYZ'
        part_obj.rotation_euler.z += random.uniform(0.0, math.pi * 2)

        # Centrar en (0,0) y ajustar Z sobre el belt
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        bbox_world = _get_world_bbox(part_obj)
        min_z = min(pt.z for pt in bbox_world)
        part_obj.location.z = -min_z + 0.02

        # ── Material ABS plástico (= training) ───────────────────────────
        mat_abs = create_abs_plastic_material(PART_COLOR_HEX)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat_abs)

        bpy.context.view_layer.update()

        # ── Render 3 cámaras ──────────────────────────────────────────────
        sample_meta = {
            "ref":         part_ref,
            "pose_index":  selected_pose_idx,
            "color_hex":   PART_COLOR_HEX,
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

        # ── Limpiar pieza ─────────────────────────────────────────────────
        cleanup_piece_objects()

    # ── 4. Guardar metadata ───────────────────────────────────────────────
    meta_path = os.path.join(output_dir, "test_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id":          "75078-1",
            "render_engine":   "BLENDER_EEVEE",
            "resolution":      f"{RENDER_RES_SQUARE}x{RENDER_RES_SQUARE}",
            "samples_count":   len(results_meta),
            "renders":         results_meta,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[TestGen DONE] {len(results_meta)} muestras generadas en {output_dir}")
    print(f"[TestGen] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
