# -*- coding: utf-8 -*-
"""scripts/generate_eevee_dinov2_refs.py
=========================================
Renderiza imágenes de referencia DINOv2 para las 10 piezas de test
usando EXACTAMENTE el mismo pipeline de render que el training YOLO
(EEVEE, 640×640, belt + corner lights).

Genera N vistas por pieza (todas las poses estables × varios ángulos de rotación),
desde las 3 cámaras (cenital, lateral_l, lateral_r).

Uso:
  blender -b -P scripts/generate_eevee_dinov2_refs.py -- \
      --output_dir data/iter9_dinov2_ref --rotations 8
"""
import os, sys, random, math, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))
sys.path.append(os.path.join(project_root, 'scratch'))

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

# ── Funciones compartidas con pipeline de training (source of truth) ─────────
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
)
from scene_config import (
    RENDER_RES_SQUARE,
    LDRAW_TO_BU, LDRAW_THRESHOLD,
    TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z,
    WORLD_BG_STRENGTH, WORLD_BG_COLOR,
)

# ── Las mismas 10 piezas del test ────────────────────────────────────────────
SELECTED_PARTS = ["3005", "3001", "3039", "3665", "3010",
                  "3002", "3020", "4070", "4032", "3700",
                  "2412b", "98138", "2335"]

PART_COLORS_HEX = ["A0A5A9", "1B1B1B", "C91A09", "F2CD37", "DFD1A5", "0A3C9F"]


def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece(obj):
    if not obj.data or not hasattr(obj.data, 'vertices'):
        return 1.0
    import bpy
    # Bake the import rotation/scale transforms into the raw vertices
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
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


def setup_camera(cam_name, location):
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
    cam.data.lens = 52.5   # Mismo focal que training
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    return cam


def cleanup_piece():
    keep = {
        "Conveyor_Belt_Plane", "Camera_Target",
        "Top_Diffuse_Light", "Key_Light", "Fill_Light", "Rim_Light",
        "Side_Rail_L", "Side_Rail_R",
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
    for mat in list(bpy.data.materials):
        if mat.name.startswith('DR_') and mat.users == 0:
            bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def build_scene():
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()

    setup_physics_world()
    create_belt_collider()
    setup_studio_lighting()

    top = bpy.data.objects.get("Top_Diffuse_Light")
    if not top:
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, TOP_LIGHT_Z))
        top = bpy.context.active_object
        top.name = "Top_Diffuse_Light"
    top.location = (0.0, 0.0, TOP_LIGHT_Z)
    top.data.size = TOP_LIGHT_SIZE
    top.data.energy = TOP_LIGHT_ENERGY

    setup_corner_lights()

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = WORLD_BG_COLOR
            bg.inputs["Strength"].default_value = WORLD_BG_STRENGTH

    enable_metal_gpu_acceleration()
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE

    cam_c  = setup_camera("Cam_Cenital",  ( 0.0,  0.0, 15.0))
    cam_f  = setup_camera("Cam_Frontal",  ( 0.0, -15.0,  2.5))

    print(f"[RefGen] Escena lista: EEVEE {RENDER_RES_SQUARE}×{RENDER_RES_SQUARE}")
    return scene, cam_c, cam_f


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rotations",  type=int, default=8,
                        help="Número de rotaciones Z uniformes por pose")
    pa = parser.parse_known_args(args)[0]

    out_dir  = pa.output_dir
    n_rots   = pa.rotations
    rot_step = (2 * math.pi) / n_rots

    for cam_name in ["cenital", "frontal"]:
        os.makedirs(os.path.join(out_dir, cam_name), exist_ok=True)

    scene, cam_c, cam_f = build_scene()

    cameras = {
        "cenital":   cam_c,
        "frontal":   cam_f,
    }

    total_rendered = 0

    for part_ref in SELECTED_PARTS:
        print(f"\n[RefGen] === Pieza {part_ref} ===")

        poses = get_stable_poses_from_db_subprocess(part_ref)
        if not poses:
            poses = [{"pose_index": 0, "orientation_quat": None, "orientation_euler": None}]

        for pose in poses:
            pose_idx = pose.get("pose_index", 0)

            # ── Cargar malla ──────────────────────────────────────────
            part_path = get_ldraw_part_path(part_ref)
            existing_objects = set(bpy.context.scene.objects)
            part_obj = None

            if part_path:
                try:
                    bpy.ops.import_scene.importldr(filepath=part_path)
                    new_objs = [o for o in bpy.context.scene.objects
                                if o not in existing_objects]
                    par = next((o for o in new_objs if o.parent is None), None)
                    if par:
                        part_obj = get_single_mesh_object(par)
                except Exception as e:
                    print(f"  [WARN] import {part_ref}: {e}")

            if not part_obj:
                generate_detailed_fallback_mesh(part_ref)
                part_obj = bpy.context.active_object

            if not part_obj:
                print(f"  [ERR] Sin mesh para {part_ref}. Saltando.")
                continue

            # ── Normalizar + bevel (= training) ───────────────────────
            bpy.ops.object.select_all(action='DESELECT')
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)

            for rot_i in range(n_rots):
                rot_z = rot_i * rot_step

                # ── Pose estable ──────────────────────────────────────────
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

                # ── Rotación Z uniforme ───────────────────────────────────
                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler.z += rot_z

                # ── Centrar y ajustar Z ───────────────────────────────────
                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                bbox_world = [part_obj.matrix_world @ mathutils.Vector(c)
                              for c in part_obj.bound_box]
                min_z = min(pt.z for pt in bbox_world)
                part_obj.location.z = -min_z + 0.02

                # ── Loop over multiple colors ─────────────────────────────
                for color_hex in PART_COLORS_HEX:
                    # ── Material ABS ──────────────────────────────────────────
                    mat = create_abs_plastic_material(color_hex)
                    part_obj.data.materials.clear()
                    part_obj.data.materials.append(mat)

                    bpy.context.view_layer.update()

                    # ── Render 3 cámaras ──────────────────────────────────────
                    for cam_name, cam_obj in cameras.items():
                        scene.camera = cam_obj
                        bpy.context.view_layer.update()

                        rot_deg = int(math.degrees(rot_z))
                        fname = f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png"
                        fpath = os.path.join(out_dir, cam_name, fname)
                        scene.render.filepath = fpath

                        try:
                            bpy.ops.render.render(write_still=True)
                            total_rendered += 1
                        except Exception as e:
                            print(f"  [WARN] Render fallido {cam_name}: {e}")

                print(f"  [OK] {part_ref} pose={pose_idx} rot={rot_deg}° (rendered {len(PART_COLORS_HEX)} colors)")

            cleanup_piece()


    print(f"\n[RefGen DONE] {total_rendered} imágenes de referencia en {out_dir}")


if __name__ == "__main__":
    main()
