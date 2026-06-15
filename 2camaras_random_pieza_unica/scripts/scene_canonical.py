# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/scene_canonical.py
==========================================================
Modulo CANONICO de la escena del proyecto. Toda la geometria,
camaras, iluminacion y helpers de posicionamiento viven aqui
para que TODOS los scripts de render (test, refs DINOv2, YOLO
training, focus, etc.) compartan exactamente el mismo dominio.

ESCALA: 1 BU = 0.1 m = 10 cm.

Geometria:
  * Cinta azul petroleo 20 x 120 cm x 1 cm espesor (2.0 x 12.0 x 0.1 BU)
  * Pantalla aluminio mate detras (10 cm alto x 120 cm largo, en x=-1.0 BU)
  * Suelo oficina gris claro (60 x 60 BU @ z=-0.5 BU)

Camaras (focal 27 mm, sensor 36 mm, persp):
  * Cenital  (0.0, 0.0, 3.0) BU = 30 cm de altura (focal 55mm)
  * Lateral  (1.5, 0.0, 0.25) BU = (15 cm, 0, 2.5 cm sobre cinta) (focal 27mm)

Iluminacion canonica (Dome Light + Cross-Polarization, 2026-06-13):
  * NO area lights, NO directional lights.
  * Iluminacion EXCLUSIVAMENTE via World Background (Dome Light perfecto).
  * WORLD_BG_STRENGTH = 1.5, WORLD_BG_COLOR = blanco puro (1,1,1,1).
  * Piezas: Specular = 0.05, Roughness = 0.75 (cross-polarization simulada).
  * Cinta: Specular = 0.0, Roughness = 1.0 (mate perfecto, sin reflejos).

Origen: extraido de generate_inferencia_test_v2.py (que produce
data/inferencia_test_v3_colors/ en 2026-09-09), luego actualizado
para Dome Light + Cross-Polarization (2026-06-13).
"""
from __future__ import annotations

import math
import os
import random
import sys

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
except ImportError:
    raise ImportError("scene_canonical.py debe importarse dentro de Blender (-b -P)")


# Constantes geometricas (ESCALA 1 BU = 10 cm)
BELT_W_BU = 2.0
BELT_L_BU = 12.0
BELT_T_BU = 0.1

SCREEN_T_BU = 0.05
SCREEN_H_BU = 1.0
SCREEN_L_BU = 12.0

CAM_CEN_LOC = (0.0, 0.0, 3.0)
CAM_LAT_LOC = (1.5, 0.0, 0.25)
CAM_FOCAL_MM = 27.0
CAM_SENSOR_MM = 36.0

HALF_FOV_BU = 3.0 * (18.0 / 55.0)
FOV_FULL_MM = int(2 * HALF_FOV_BU * 100)
MARGIN_BU_DEFAULT = 0.05
MAX_PLACEMENT_ATTEMPTS_DEFAULT = 200

# Dome Light + Cross-Polarization parameters (from scene_config.py)
try:
    from scene_config import (
        WORLD_BG_STRENGTH,
        WORLD_BG_COLOR,
        PIECE_SPECULAR,
        PIECE_ROUGHNESS,
        BELT_SPECULAR,
        BELT_ROUGHNESS,
        BELT_COLOR_LINEAR,
    )
except ImportError:
    WORLD_BG_STRENGTH = 1.5
    WORLD_BG_COLOR = (1.0, 1.0, 1.0, 1.0)
    PIECE_SPECULAR = 0.05
    PIECE_ROUGHNESS = 0.75
    BELT_SPECULAR = 0.0
    BELT_ROUGHNESS = 1.0
    BELT_COLOR_LINEAR = (0.145, 0.255, 0.33, 1.0)


def get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def normalize_piece(obj, ldu_to_bu=0.004):
    if not obj.data or not hasattr(obj.data, "vertices"):
        return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return 1.0
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    mx = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if mx < 1e-6:
        return 1.0
    factor = ldu_to_bu if mx > 5.0 else 1.0
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


def cleanup_piece_objects():
    keep = {
        "Conveyor_Belt_Plane", "Side_Screen_AL", "Office_Floor",
        "Camera_Target", "Cam_Cenital", "Cam_Lateral",
        "V1_Dome_Cenital", "V2_Sun", "V2_Dome_Cenital",
        "V3_Dome_Cenital", "V3_Panel_N", "V3_Panel_S", "V3_Panel_E", "V3_Panel_W",
        "V4_Overhead_Strip",
    }
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and o.type not in ("CAMERA", "LIGHT", "EMPTY"):
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def setup_camera(name, location):
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"
    if name in bpy.data.objects:
        cam = bpy.data.objects[name]
        cam.location = location
    else:
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object
        cam.name = name
    cam.constraints.clear()
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    cam.data.type = "PERSP"
    if name == "Cam_Cenital" or "Cenital" in name:
        cam.data.lens = 55.0
    else:
        cam.data.lens = CAM_FOCAL_MM
    cam.data.clip_start = 0.001
    cam.data.clip_end = 100.0
    return cam


def _make_belt_blue_petroleum_material(name="Belt_Blue_Petroleum"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = BELT_ROUGHNESS
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = BELT_SPECULAR
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = BELT_SPECULAR
    return mat


def _make_aluminum_mate_material(name="Aluminum_Mate"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.62, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.85
        bsdf.inputs["Roughness"].default_value = 0.9
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.1
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.1
    return mat


def _make_office_floor_material(name="Office_Floor"):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.58, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
    return mat


def create_belt_blue_petroleum():
    name = "Conveyor_Belt_Plane"
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    ht = BELT_T_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = name
    belt.scale = (BELT_W_BU, BELT_L_BU, BELT_T_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_belt_blue_petroleum_material("Belt_Blue_Petroleum")
    belt.data.materials.clear()
    belt.data.materials.append(mat)
    return belt


def create_side_screen_aluminum():
    name = "Side_Screen_AL"
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    x_screen = -BELT_W_BU / 2.0 - SCREEN_T_BU / 2.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_screen, 0.0, SCREEN_H_BU / 2.0))
    sc = bpy.context.active_object
    sc.name = name
    sc.scale = (SCREEN_T_BU, SCREEN_L_BU, SCREEN_H_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_aluminum_mate_material("Aluminum_Mate_Screen")
    sc.data.materials.clear()
    sc.data.materials.append(mat)
    return sc


def create_office_floor():
    name = "Office_Floor"
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, -0.5))
    fl = bpy.context.active_object
    fl.name = name
    fl.scale = (60.0, 60.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_office_floor_material()
    fl.data.materials.clear()
    fl.data.materials.append(mat)
    return fl


def _clear_lights():
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def _set_world_dome_light():
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = WORLD_BG_COLOR
        bg.inputs["Strength"].default_value = float(WORLD_BG_STRENGTH)


def setup_dome_light():
    _clear_lights()
    _set_world_dome_light()


def build_scene_canonical(render_res=640, film_transparent=False, enable_metal_gpu=True, configure_eevee=True):
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()
    try:
        from generate_synthetic_set import setup_physics_world
        setup_physics_world()
    except Exception:
        pass
    create_belt_blue_petroleum()
    create_side_screen_aluminum()
    create_office_floor()
    if enable_metal_gpu:
        try:
            from generate_synthetic_set import enable_metal_gpu_acceleration
            enable_metal_gpu_acceleration()
        except Exception:
            pass
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = bool(film_transparent)
    scene.render.resolution_x = int(render_res)
    scene.render.resolution_y = int(render_res)
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    cam_cenital = setup_camera("Cam_Cenital", CAM_CEN_LOC)
    cam_lateral = setup_camera("Cam_Lateral", CAM_LAT_LOC)
    setup_dome_light()
    if configure_eevee:
        try:
            from generate_synthetic_set import configure_eevee_for_translucent
            configure_eevee_for_translucent(scene)
        except Exception:
            pass
    return cam_cenital, cam_lateral


def get_2d_bbox(obj, scene, camera):
    bbox_world = get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(co.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0)),
    ]


def _project_bbox_norm(obj, scene, camera):
    bbox_world = get_world_bbox(obj)
    xs, ys, zs = [], [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(co.y)
        zs.append(co.z)
    return min(xs), min(ys), max(xs), max(ys), min(zs)


def _bbox_within_margin(b, m):
    x1, y1, x2, y2, depth_min = b
    if depth_min <= 0:
        return False
    return (x1 >= m and y1 >= m and x2 <= 1.0 - m and y2 <= 1.0 - m)


def _world_min_z_mesh(part_obj):
    if not part_obj.data or not hasattr(part_obj.data, "vertices"):
        return min(pt.z for pt in get_world_bbox(part_obj))
    mw = part_obj.matrix_world
    return min((mw @ v.co).z for v in part_obj.data.vertices)


def sample_valid_position(part_obj, scene, cam_cen, cam_lat,
                          margin_bu=MARGIN_BU_DEFAULT,
                          max_attempts=MAX_PLACEMENT_ATTEMPTS_DEFAULT,
                          snap_air_bu=0.005):
    margin_norm = margin_bu / (2.0 * HALF_FOV_BU)
    sample_range = HALF_FOV_BU - 0.05
    for _ in range(max_attempts):
        rx = random.uniform(-sample_range, sample_range)
        ry = random.uniform(-sample_range, sample_range)
        part_obj.location = (rx, ry, 0.0)
        bpy.context.view_layer.update()
        min_z = _world_min_z_mesh(part_obj)
        part_obj.location.z = -min_z + snap_air_bu
        bpy.context.view_layer.update()
        bbox_cen = _project_bbox_norm(part_obj, scene, cam_cen)
        if not _bbox_within_margin(bbox_cen, margin_norm):
            continue
        bbox_lat = _project_bbox_norm(part_obj, scene, cam_lat)
        if not _bbox_within_margin(bbox_lat, margin_norm):
            continue
        return (
            float(part_obj.location.x),
            float(part_obj.location.y),
            float(part_obj.location.z),
        )
    return None


def import_part(part_ref):
    from generate_synthetic_set import (
        get_ldraw_part_path, generate_detailed_fallback_mesh,
    )
    from generate_synthetic_dataset import get_single_mesh_object
    part_path = get_ldraw_part_path(part_ref)
    existing = set(bpy.context.scene.objects)
    obj = None
    if part_path:
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objs = [o for o in bpy.context.scene.objects if o not in existing]
            par = next((o for o in new_objs if o.parent is None), None)
            if par:
                obj = get_single_mesh_object(par)
        except Exception:
            obj = None
    if not obj:
        generate_detailed_fallback_mesh(part_ref)
        obj = bpy.context.active_object
    return obj


def hide_render_environment(hide=True):
    targets = ["Conveyor_Belt_Plane", "Side_Screen_AL", "Office_Floor"]
    prev = {}
    for n in targets:
        o = bpy.data.objects.get(n)
        if o is not None:
            prev[n] = o.hide_render
            o.hide_render = bool(hide)
    return prev


def restore_render_environment(prev):
    for n, p in (prev or {}).items():
        o = bpy.data.objects.get(n)
        if o is not None:
            o.hide_render = p


# Legacy alias for backward compatibility
variant_V4_overhead_strip_high_ambient = setup_dome_light
