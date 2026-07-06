# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/scene_canonical.py
==========================================================
Modulo CANONICO de la escena del proyecto. Toda la geometria,
camaras, iluminacion y helpers de posicionamiento viven aqui
para que TODOS los scripts de render (test, refs DINOv2, YOLO
training, focus, etc.) compartan exactamente el mismo dominio.

ESCALA: 1 BU = 0.1 m = 10 cm.

Geometria:
  * Cinta azul petroleo (color definido por scripts.scene_config.BELT_COLOR_HEX,
    fuente ÚNICA de verdad del sistema) 20 x 120 cm x 1 cm espesor (2.0 x 12.0 x 0.1 BU)
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

# ── Cámara Cenital ──────────────────────────────────────────────────────────
CAM_CEN_LOC       = (0.0, 0.0, 3.0)    # 30 cm sobre la superficie
CAM_CEN_FOCAL_MM  = 55.0               # focal cenital
CAM_SENSOR_MM     = 36.0               # sensor común

# HALF_FOV_BU: semi-ancho del FOV cenital (1 BU = 10 cm)
# FOV_WIDTH = 300 mm * (36/55) = 196.36 mm = 1.9636 BU  →  half ≈ 0.9818 BU
HALF_FOV_BU  = 3.0 * (18.0 / 55.0)   # ≈ 0.9818 BU
FOV_FULL_MM  = int(2 * HALF_FOV_BU * 100)  # ≈ 196 mm
MARGIN_BU_DEFAULT = 0.05
MAX_PLACEMENT_ATTEMPTS_DEFAULT = 200

# Belt runs along X-axis; right edge at HALF_FOV_BU (belt drops here)
BELT_END_X_BU   = HALF_FOV_BU
BELT_START_X_BU = BELT_END_X_BU - BELT_L_BU

# ── Cámara Frontal ───────────────────────────────────────────────────────────
# Posición:  15 cm detrás del borde de caída en X, 2 cm de altura
# La cámara apunta al borde de caída (BELT_END_X_BU, 0, 0), NO al origen.
#
# Cálculo focal:  focal = sensor * dist / fov_coverage
#   dist = 1.5 BU = 150 mm  (distancia horizontal al punto de caída)
#   fov_coverage = 20 cm = 200 mm  (ancho de la cinta en Y)
#   focal = 36 * 150 / 200 = 27 mm
#
# scale_px_per_mm (a 2048 px):  2048 / 200 = 10.24 px/mm
CAM_FRONTAL_DIST_BU   = 1.5    # distancia horizontal al punto de caída (BU)
CAM_FRONTAL_HEIGHT_BU = 0.02   # altura sobre la superficie de la cinta (2 cm)
CAM_FRONTAL_FOCAL_MM  = 27.0   # cubre exactamente 20 cm en Y a 15 cm
CAM_FRONTAL_SCALE_PX_MM = 10.24  # píxeles/mm a resolución 2048 × 2048

# Posición absoluta (X = borde_cinta + distancia)
CAM_FRONTAL_LOC    = (BELT_END_X_BU + CAM_FRONTAL_DIST_BU, 0.0, CAM_FRONTAL_HEIGHT_BU)
# Punto de mira: borde de caída a nivel de la superficie
CAM_FRONTAL_TARGET = (BELT_END_X_BU, 0.0, 0.0)

# Alias genérico (usado por código legacy que lee CAM_FOCAL_MM)
CAM_FOCAL_MM = CAM_CEN_FOCAL_MM

# ── Import parámetros de material desde la fuente única de verdad ────────────
# El color de la cinta (BELT_COLOR_LINEAR) proviene de scripts/scene_config.py,
# derivado de BELT_COLOR_HEX. NO se permite fallback con valores hardcoded.
_SCENE_CONFIG_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"),
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
]
for _p in _SCENE_CONFIG_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts.scene_config import (  # type: ignore
        WORLD_BG_STRENGTH,
        WORLD_BG_COLOR,
        PIECE_SPECULAR,
        PIECE_ROUGHNESS,
        BELT_SPECULAR,
        BELT_ROUGHNESS,
        BELT_COLOR_LINEAR,
        BELT_COLOR_HEX,
    )
except ImportError:
    # Fallback secundario: import plano si el proyecto se ejecuta desde scripts/
    from scene_config import (  # type: ignore
        WORLD_BG_STRENGTH,
        WORLD_BG_COLOR,
        PIECE_SPECULAR,
        PIECE_ROUGHNESS,
        BELT_SPECULAR,
        BELT_ROUGHNESS,
        BELT_COLOR_LINEAR,
        BELT_COLOR_HEX,
    )

# Sobrescribir los valores importados de scene_config con la iluminación canónica del domo
WORLD_BG_STRENGTH = 0.6
WORLD_BG_COLOR = (1.0, 1.0, 1.0, 1.0)


def get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def normalize_piece(obj, ldu_to_bu=0.004):
    if not obj.data or not hasattr(obj.data, "vertices"):
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
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    mx = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if mx < 1e-6:
        return 1.0
    if mx > 5.0:
        factor = ldu_to_bu  # Imported at LDraw scale (1 LDU = 1 unit)
    elif mx > 0.15:
        factor = 0.4        # Imported at scale 0.01 (needs conversion to 0.004 BU)
    else:
        factor = 1.0        # Already imported at 0.004 BU scale
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
    """Create or update a camera.
    Cenital  → tracks shared Camera_Target at (0,0,0), focal 55 mm.
    Frontal  → tracks Camera_Frontal_Target at (BELT_END_X_BU, 0, 0),
               focal 27 mm (covers 20 cm belt width at 15 cm distance),
               positioned at CAM_FRONTAL_LOC with 2 cm height.
    """
    is_frontal = ("Frontal" in name or "frontal" in name or name == "Cam_Frontal")

    if is_frontal:
        # Dedicated target at the belt-fall point so the camera looks straight at it
        tgt_name = "Camera_Frontal_Target"
        tgt_loc  = CAM_FRONTAL_TARGET
        focal    = CAM_FRONTAL_FOCAL_MM
    else:
        tgt_name = "Camera_Target"
        tgt_loc  = (0.0, 0.0, 0.0)
        focal    = CAM_CEN_FOCAL_MM

    # Create or retrieve the target empty
    target = bpy.data.objects.get(tgt_name)
    if not target:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=tgt_loc)
        target = bpy.context.active_object
        target.name = tgt_name
    else:
        target.location = mathutils.Vector(tgt_loc)

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
    track.up_axis   = "UP_Y"
    cam.data.type       = "PERSP"
    cam.data.lens       = focal
    cam.data.sensor_width = CAM_SENSOR_MM
    cam.data.clip_start = 0.001
    cam.data.clip_end   = 100.0
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
    """Belt runs along the X-axis from BELT_START_X_BU to BELT_END_X_BU.
    The right edge terminates at HALF_FOV_BU so the piece falls off exactly
    at the right boundary of the cenital camera's field of view.
    """
    name = "Conveyor_Belt_Plane"
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    ht = BELT_T_BU * 0.5
    # Center of belt along X: midpoint between start and end
    belt_center_x = (BELT_START_X_BU + BELT_END_X_BU) / 2.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(belt_center_x, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = name
    # X = belt length, Y = belt width, Z = belt thickness
    belt.scale = (BELT_L_BU, BELT_W_BU, BELT_T_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_belt_blue_petroleum_material("Belt_Blue_Petroleum")
    belt.data.materials.clear()
    belt.data.materials.append(mat)
    return belt


def create_side_screen_aluminum():
    """Aluminum back screen running parallel to the belt along X-axis.
    Placed at Y = -(BELT_W_BU/2 + SCREEN_T_BU/2) so it backs the belt.
    """
    name = "Side_Screen_AL"
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    belt_center_x = (BELT_START_X_BU + BELT_END_X_BU) / 2.0
    y_screen = -(BELT_W_BU / 2.0 + SCREEN_T_BU / 2.0)
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(belt_center_x, y_screen, SCREEN_H_BU / 2.0)
    )
    sc = bpy.context.active_object
    sc.name = name
    # X = screen length (same as belt), Y = screen thickness, Z = screen height
    sc.scale = (BELT_L_BU, SCREEN_T_BU, SCREEN_H_BU)
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


def _set_world_dome_light(strength=None, color=None):
    sc = bpy.context.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = color if color is not None else WORLD_BG_COLOR
        bg.inputs["Strength"].default_value = float(strength) if strength is not None else float(WORLD_BG_STRENGTH)


def setup_dome_light(randomize=False):
    _clear_lights()
    if randomize:
        import random
        # Aleatorizar intensidad alrededor de 0.6 (rango 0.45 a 0.75)
        strength = random.uniform(0.45, 0.75)
        # Aleatorizar color (desvío de temperatura de color simulado por desvío rojo/azul)
        r_shift = random.uniform(-0.08, 0.08)
        b_shift = -r_shift
        color = (1.0 + r_shift, 1.0, 1.0 + b_shift, 1.0)
    else:
        strength = WORLD_BG_STRENGTH
        color = WORLD_BG_COLOR
    _set_world_dome_light(strength, color)


def build_scene_canonical(render_res=640, film_transparent=False, enable_metal_gpu=True, configure_eevee=True, randomize_lighting=False):
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
    if not film_transparent:
        create_belt_blue_petroleum()
        # create_side_screen_aluminum()
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
    if film_transparent:
        scene.render.image_settings.color_mode = 'RGBA'
    else:
        scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_x = int(render_res)
    scene.render.resolution_y = int(render_res)
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    cam_cenital = setup_camera("Cam_Cenital", CAM_CEN_LOC)
    cam_frontal = setup_camera("Cam_Frontal", CAM_FRONTAL_LOC)  # 15 cm from fall point, 2 cm height
    setup_dome_light(randomize=randomize_lighting)
    if configure_eevee:
        try:
            from generate_synthetic_set import configure_eevee_for_translucent
            configure_eevee_for_translucent(scene)
        except Exception:
            pass
    return cam_cenital, cam_frontal


def get_2d_bbox(obj, scene, camera):
    if obj.type == 'MESH' and hasattr(obj, "data") and hasattr(obj.data, "vertices") and len(obj.data.vertices) > 0:
        matrix_world = obj.matrix_world
        xs, ys = [], []
        for v in obj.data.vertices:
            v_world = matrix_world @ v.co
            co = world_to_camera_view(scene, camera, v_world)
            xs.append(co.x)
            ys.append(co.y)
    else:
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


def apply_cross_polarization(obj):
    if not obj:
        return
    objs_to_check = [obj]
    # Recorrer jerarquía recursiva si es necesario
    def add_children(o):
        for child in o.children:
            objs_to_check.append(child)
            add_children(child)
    add_children(obj)
    
    for o in objs_to_check:
        if not hasattr(o, "material_slots"):
            continue
        for slot in o.material_slots:
            mat = slot.material
            if mat and mat.use_nodes:
                nodes = mat.node_tree.nodes
                principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if principled:
                    for spec_name in ["Specular", "Specular IOR Level"]:
                        if spec_name in principled.inputs:
                            principled.inputs[spec_name].default_value = 0.0
                    if "Roughness" in principled.inputs:
                        principled.inputs["Roughness"].default_value = 1.0

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
                apply_cross_polarization(par)
        except Exception:
            obj = None
            
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
