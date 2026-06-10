# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_inferencia_test_v2.py
Set de inferencia v2: N piezas aleatorias del set 75078-1 (default 100)
+ 2 muestras forzadas (3023 Trans-Brown / 3023 Red), sobre la escena
nueva con cinta aluminio 20x120 cm, pantalla lateral 10 cm, suelo
oficina, camaras (cenital z=15cm, lateral 15cm/2.5cm) e iluminacion
V4 (Overhead Strip + Ambient).

Las posiciones, poses y rotaciones Z son aleatorias dentro del FOV
de ambas camaras.

Uso:
    /opt/homebrew/bin/blender -b -P \\
        2camaras_random_pieza_unica/scripts/generate_inferencia_test_v2.py -- \\
            --num_random 100 \\
            --output_dir 2camaras_random_pieza_unica/data/inferencia_test_v2 \\
            --metadata_filename inferencia_test_v2_metadata.json \\
            --seed 42
"""
from __future__ import annotations
import json, math, os, random, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
# `database/` y otros módulos compartidos viven en la raíz LegoVision/
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
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

from config_loader import cfg
from generate_synthetic_set import (
    apply_bevel_modifier,
    configure_eevee_for_translucent,
    create_abs_plastic_material,
    enable_metal_gpu_acceleration,
    generate_detailed_fallback_mesh,
    get_ldraw_part_path,
    setup_physics_world,
)
from generate_synthetic_dataset import get_single_mesh_object
from generate_test_set import (
    _get_world_bbox,
    _normalize_piece,
    cleanup_piece_objects,
    create_belt_collider,
    create_floor,
    setup_camera,
    setup_lab_lightbox,
)
from _pose_utils import (
    get_stable_poses_for_ref,
    apply_stable_pose,
)
from database.set_catalog import REAL_SETS

SET_ID = "75078-1"
RENDER_RES = cfg.render.resolution.width
# Escala 1 BU = 10 cm. FOV cenital con cam @ z=1.5 BU, focal 27 mm,
# sensor 36 mm => half_FOV = 1.5 * (18/27) = 1.0 BU = 10 cm.
HALF_FOV_BU = 1.0
MARGIN_BU = 0.05  # 5 mm de margen
MARGIN_NORM = MARGIN_BU / (2.0 * HALF_FOV_BU)
MAX_PLACEMENT_ATTEMPTS = 200
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
OUTPUT_DIR_DEFAULT = os.path.join(project_root, "data", "inferencia_test_v2")
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")

# Catalogo de los 7 colores reales del set 75078-1 (BrickLink codes).
SET_COLORS = {
    "1":  {"name": "White",             "hex": "#FFFFFF"},
    "5":  {"name": "Red",               "hex": "#C91A09"},
    "11": {"name": "Black",             "hex": "#1B1B1B"},
    "13": {"name": "Trans-Brown",       "hex": "#583927"},
    "17": {"name": "Trans-Red",         "hex": "#C91A09"},
    "85": {"name": "Dark Bluish Gray",  "hex": "#646464"},
    "86": {"name": "Light Bluish Gray", "hex": "#A0A5A9"},
}

# Muestras forzadas: 1 pieza (3023) x 7 colores del set (incluye semi-trans).
# Cubre todos los colores del catalogo BrickLink del set 75078-1.
FORCED_SAMPLES = [
    {"ref": "3023", "pose_index": 1, "color_code": "1"},   # White
    {"ref": "3023", "pose_index": 1, "color_code": "5"},   # Red
    {"ref": "3023", "pose_index": 1, "color_code": "11"},  # Black
    {"ref": "3023", "pose_index": 1, "color_code": "13"},  # Trans-Brown
    {"ref": "3023", "pose_index": 1, "color_code": "17"},  # Trans-Red
    {"ref": "3023", "pose_index": 1, "color_code": "85"},  # Dark Bluish Gray
    {"ref": "3023", "pose_index": 1, "color_code": "86"},  # Light Bluish Gray
]


# ------------------------- Geometry helpers -------------------------
def get_2d_bbox(obj, scene, camera):
    bbox_world = _get_world_bbox(obj)
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
    bbox_world = _get_world_bbox(obj)
    xs, ys, zs = [], [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(co.y)
        zs.append(co.z)
    return min(xs), min(ys), max(xs), max(ys), min(zs)


def _bbox_within_margin(bbox_norm, margin):
    x1, y1, x2, y2, depth_min = bbox_norm
    if depth_min <= 0:
        return False
    return (x1 >= margin and y1 >= margin
            and x2 <= 1.0 - margin and y2 <= 1.0 - margin)


def sample_valid_position(part_obj, scene, cam_cen, cam_lat,
                          margin_norm=MARGIN_NORM,
                          max_attempts=MAX_PLACEMENT_ATTEMPTS):
    """Rejection sampling: posicion (x,y) valida en CENITAL y LATERAL."""
    sample_range = HALF_FOV_BU - 0.05
    for _ in range(max_attempts):
        rx = random.uniform(-sample_range, sample_range)
        ry = random.uniform(-sample_range, sample_range)
        part_obj.location = (rx, ry, 0.0)
        bpy.context.view_layer.update()
        bbox_world = _get_world_bbox(part_obj)
        min_z = min(pt.z for pt in bbox_world)
        part_obj.location.z = -min_z + 0.02
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


# ------------------------- Scene setup (CINTA 3 cm + PANTALLA + SUELO OFICINA) -
def _make_aluminum_mate_material(name="Aluminum_Mate"):
    """Material aluminio mate compartido por cinta y pantalla lateral."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.62, 0.62, 0.62, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.85
        bsdf.inputs['Roughness'].default_value = 0.45
    return mat


def _make_office_floor_material(name="Office_Floor"):
    """Suelo de oficina: gris claro pizarra ligeramente azulado, mate."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.55, 0.55, 0.58, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.6
    return mat


# Geometria definitiva (escala 1 BU = 0.1 m = 10 cm):
#   Cinta:    20 cm ancho x 120 cm largo x 1 cm espesor (2.0 x 12.0 x 0.1 BU)
#   Pantalla: 10 cm alto x 120 cm largo x 0.5 cm espesor (1.0 x 12.0 x 0.05 BU)
#             posicionada en el borde -X de la cinta.
#   Suelo:    60 x 60 BU @ z=-0.5 BU (gris claro PVC, mate).
BELT_W_BU = 2.0   # 20 cm
BELT_L_BU = 12.0  # 120 cm
BELT_T_BU = 0.1   # 1 cm

SCREEN_T_BU = 0.05  # 0.5 cm espesor
SCREEN_H_BU = 1.0   # 10 cm alto
SCREEN_L_BU = 12.0  # 120 cm largo

# Camaras (escala 1 BU = 10 cm):
#   Cenital: (0, 0, 1.5) BU = (0, 0, 15 cm)
#   Lateral: (1.5, 0, 0.25) BU = (15 cm, 0, 2.5 cm sobre superficie cinta)
CAM_CEN_LOC = (0.0, 0.0, 1.5)
CAM_LAT_LOC = (1.5, 0.0, 0.25)


def _make_belt_blue_petroleum_material(name="Belt_Blue_Petroleum"):
    """Cinta azul petroleo (color canonico del proyecto, optimizado para
    chromaticity-based segmentation): RGB linear (0.145, 0.255, 0.33),
    roughness 0.5, no metallic."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.145, 0.255, 0.33, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.5
    return mat


def create_belt_blue_petroleum():
    """Cinta 20x120x1 cm en azul petroleo (canonico para chromaticity-segmentation)."""
    name = 'Conveyor_Belt_Plane'
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


def create_belt_aluminum():
    """[Legacy] Cinta aluminio mate. Mantenido por compat. NO se usa."""
    name = 'Conveyor_Belt_Plane'
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    ht = BELT_T_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = name
    belt.scale = (BELT_W_BU, BELT_L_BU, BELT_T_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_aluminum_mate_material("Aluminum_Mate_Belt")
    belt.data.materials.clear()
    belt.data.materials.append(mat)
    return belt


def create_side_screen_aluminum():
    """Pantalla aluminio mate 10 cm alto x 120 cm largo, en x=-(BELT_W/2 + T/2)
    (borde -X de la cinta, lado opuesto a cam_lateral en x=+1.5)."""
    name = 'Side_Screen_AL'
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    x_screen = -BELT_W_BU/2.0 - SCREEN_T_BU/2.0
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(x_screen, 0.0, SCREEN_H_BU/2.0))
    sc = bpy.context.active_object
    sc.name = name
    sc.scale = (SCREEN_T_BU, SCREEN_L_BU, SCREEN_H_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = _make_aluminum_mate_material("Aluminum_Mate_Screen")
    sc.data.materials.clear()
    sc.data.materials.append(mat)
    return sc


def create_office_floor():
    """Suelo oficina (PVC gris claro) debajo de la cinta."""
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


def build_scene():
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()
    setup_physics_world()
    # Geometria: cinta azul petroleo (canonico) + pantalla aluminio + suelo oficina
    create_belt_blue_petroleum()
    create_side_screen_aluminum()
    create_office_floor()
    # Luces se aplican luego por cada variante (variant_*())
    enable_metal_gpu_acceleration()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES

    # Color management
    scene.view_settings.view_transform = 'Standard'
    try:
        scene.view_settings.look = 'None'
    except TypeError:
        pass

    cam_cenital = setup_camera("Cam_Cenital", CAM_CEN_LOC)
    cam_lateral = setup_camera("Cam_Lateral", CAM_LAT_LOC)
    configure_eevee_for_translucent(scene)
    return cam_cenital, cam_lateral


def import_part(part_ref):
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


# ------------------------- Lighting variants -------------------------
def _set_blackbody(light_obj, temp_k=5500.0, energy_w=20.0):
    light = light_obj.data
    light.use_nodes = True
    nt = light.node_tree
    nt.nodes.clear()
    nb = nt.nodes.new(type="ShaderNodeBlackbody")
    nb.inputs["Temperature"].default_value = float(temp_k)
    ne = nt.nodes.new(type="ShaderNodeEmission")
    ne.inputs["Strength"].default_value = float(energy_w)
    no = nt.nodes.new(type="ShaderNodeOutputLight")
    nt.links.new(nb.outputs["Color"], ne.inputs["Color"])
    nt.links.new(ne.outputs["Emission"], no.inputs["Surface"])
    light.energy = float(energy_w)


def _add_area_light(name, location, shape, size, size_y, energy_w,
                    track_target=True, specular=1.0, temp_k=5500.0):
    bpy.ops.object.light_add(type="AREA", location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.shape = shape
    obj.data.size = size
    if shape == "RECTANGLE":
        obj.data.size_y = size_y
    if hasattr(obj.data, "specular_factor"):
        obj.data.specular_factor = specular
    _set_blackbody(obj, temp_k=temp_k, energy_w=energy_w)
    if track_target:
        target = bpy.data.objects.get("Camera_Target")
        if target:
            tr = obj.constraints.new(type="TRACK_TO")
            tr.target = target
            tr.track_axis = "TRACK_NEGATIVE_Z"
            tr.up_axis = "UP_Y"
    return obj


def _clear_lights():
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def _set_world_strength(s):
    sc = bpy.context.scene
    if sc.world:
        sc.world.use_nodes = True
        bg = sc.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = float(s)


def variant_V1_studio_soft():
    """V1 - Studio Soft: domo grande + ambiente moderado.
    DISK 2.0 BU @ z=3 BU 18W + world 0.5."""
    _clear_lights()
    _set_world_strength(0.5)
    _add_area_light("V1_Dome_Cenital", (0.0, 0.0, 3.0),
                    "DISK", 2.0, 2.0, 18.0, specular=1.0,
                    track_target=False)
    obj = bpy.data.objects.get("V1_Dome_Cenital")
    if obj:
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, 0.0)


def variant_V2_daylight_dome():
    """V2 - Daylight + Dome: SUN suave + domo cenital + ambiente medio.
    SUN 0.8 + DISK 1.5 BU @ z=2.5 12W + world 0.4."""
    _clear_lights()
    _set_world_strength(0.4)
    bpy.ops.object.light_add(type='SUN', location=(-3.0, 0.0, 5.0))
    sun = bpy.context.active_object
    sun.name = "V2_Sun"
    sun.data.energy = 0.8
    sun.data.angle = 0.05
    sun.data.color = (1.0, 1.0, 1.0)
    if hasattr(sun.data, 'use_nodes'):
        sun.data.use_nodes = False
    sun.rotation_mode = 'XYZ'
    sun.rotation_euler = (math.radians(-45), math.radians(-30), 0.0)
    _add_area_light("V2_Dome_Cenital", (0.0, 0.0, 2.5),
                    "DISK", 1.5, 1.5, 12.0, specular=1.0,
                    track_target=False)
    obj = bpy.data.objects.get("V2_Dome_Cenital")
    if obj:
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, 0.0)


def variant_V3_lightbox_industrial():
    """V3 - Lightbox Industrial: domo + 4 paneles laterales (caja de luz envolvente).
    DISK 1.6 BU @ z=2.8 14W + 4 panels 2.0x1.0 @ z=1.5 4W cada + world 0.4."""
    _clear_lights()
    _set_world_strength(0.4)
    _add_area_light("V3_Dome_Cenital", (0.0, 0.0, 2.8),
                    "DISK", 1.6, 1.6, 14.0, specular=1.0,
                    track_target=False)
    obj = bpy.data.objects.get("V3_Dome_Cenital")
    if obj:
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, 0.0)
    panel_specs = [
        ("V3_Panel_N", (0.0,  3.5, 1.5), 'RECTANGLE', 2.0, 1.0),
        ("V3_Panel_S", (0.0, -3.5, 1.5), 'RECTANGLE', 2.0, 1.0),
        ("V3_Panel_E", (3.5,  0.0, 1.5), 'RECTANGLE', 1.0, 2.0),
        ("V3_Panel_W", (-3.5, 0.0, 1.5), 'RECTANGLE', 1.0, 2.0),
    ]
    for name, loc, shape, sx, sy in panel_specs:
        _add_area_light(name, loc, shape, sx, sy, 4.0, specular=1.0,
                        track_target=True)


def variant_V4_overhead_strip_high_ambient():
    """V4 - Overhead Strip + Ambient (escala nueva: cinta 20x120, cam cen z=15cm).
    RECT 0.6x0.3 BU (6x3 cm) @ z=0.5 BU (5 cm sobre cinta), 0.6W + world 0.6.
    A 5 cm de la pieza la potencia se reduce ~36x vs escala antigua (luz a 30 cm)."""
    _clear_lights()
    _set_world_strength(0.6)
    _add_area_light("V4_Overhead_Strip", (0.0, 0.0, 0.5),
                    "RECTANGLE", 0.6, 0.3, 0.6, specular=1.0,
                    track_target=False)
    obj = bpy.data.objects.get("V4_Overhead_Strip")
    if obj:
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (0.0, 0.0, 0.0)


VARIANTS = [
    ("V4_overhead_strip_high_ambient",
     "Overhead Strip + Ambient (escala nueva): RECT 0.6x0.3 BU @ z=0.5 (0.6W) + world 0.6",
     variant_V4_overhead_strip_high_ambient),
]


def find_pose_by_index(poses, target_index):
    """Busca pose por pose_index o original_pose_index."""
    for p in poses:
        if p.get("pose_index") == target_index or p.get("original_pose_index") == target_index:
            return p
    return None


# ------------------------- Plan helpers -------------------------
def get_unique_ref_color_combinations(set_id):
    seen, combos = set(), []
    for p in REAL_SETS[set_id]["parts"]:
        key = (p["ref"], p["color_code"])
        if key in seen: continue
        seen.add(key)
        combos.append({
            "ref": p["ref"],
            "color_code": p["color_code"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_name": p.get("color_name", "Unknown"),
        })
    return combos


def filter_stable_poses(poses):
    if not poses: return []
    stable = [p for p in poses
              if float(p.get("tipping_energy_ratio", 0.0)) >= TARPS_MIN_TIPPING]
    if stable: return stable
    best = max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))
    return [best]


def build_random_samples(combos, cache, num_random):
    universe = []
    for combo in combos:
        poses = cache.get(combo["ref"], [])
        stable = filter_stable_poses(poses)
        if not stable: continue
        for pose in stable:
            universe.append({
                "ref": combo["ref"],
                "color_code": combo["color_code"],
                "color_hex": combo["color_hex"],
                "color_name": combo["color_name"],
                "pose": pose,
                "mode": "random",
            })
    if not universe:
        return []
    if num_random <= len(universe):
        return random.sample(universe, num_random)
    return [random.choice(universe) for _ in range(num_random)]


def build_forced_samples(cache):
    out = []
    for spec in FORCED_SAMPLES:
        ref = spec["ref"]
        poses = cache.get(ref, [])
        if not poses: continue
        pose = find_pose_by_index(poses, spec["pose_index"]) or poses[0]
        cc = spec["color_code"]
        cinfo = SET_COLORS.get(cc, {"name": "Unknown", "hex": "#A0A5A9"})
        out.append({
            "ref": ref, "color_code": cc,
            "color_hex": cinfo["hex"], "color_name": cinfo["name"],
            "pose": pose, "mode": "forced",
        })
    return out


# ------------------------- Main -------------------------
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_random", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--metadata_filename", type=str,
                        default="inferencia_test_v2_metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    pa = parser.parse_known_args(args)[0]

    random.seed(pa.seed)
    output_dir = pa.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}"); sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    combos = get_unique_ref_color_combinations(SET_ID)
    print(f"[infTestV2] Set {SET_ID}: {len(combos)} combinaciones (ref,color)")

    plan = build_random_samples(combos, cache, pa.num_random) + build_forced_samples(cache)
    print(f"[infTestV2] Plan: {len(plan)} muestras")

    cam_cenital, cam_lateral = build_scene()
    # Aplicar V4 una sola vez (las luces no cambian entre samples)
    variant_V4_overhead_strip_high_ambient()
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    results_meta = []
    skipped = []
    current_ref = None
    part_obj = None

    for idx, item in enumerate(plan):
        ref = item["ref"]
        color_code = item["color_code"]
        color_hex = item["color_hex"]
        color_name = item["color_name"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))
        mode_tag = item.get("mode", "random")

        print(f"\n[{idx+1:03d}/{len(plan)}] {mode_tag} ref={ref} color={color_code}({color_name}) pose={pose_index}")

        if ref != current_ref:
            cleanup_piece_objects()
            part_obj = import_part(ref)
            if not part_obj:
                print(f"   [SKIP] no se pudo importar mesh de {ref}")
                skipped.append({"idx": idx, "ref": ref, "reason": "import_failed"})
                current_ref = None
                continue
            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            current_ref = ref

        # Pose + Z aleatoria + snap
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)
        z_rotation_rad = float(part_obj.rotation_euler.z)

        # Posicion aleatoria valida
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_norm=MARGIN_NORM,
            max_attempts=MAX_PLACEMENT_ATTEMPTS,
        )
        if valid_pos is None:
            print(f"   [SKIP] no hay posicion valida")
            skipped.append({"idx": idx, "ref": ref, "reason": "no_valid_placement"})
            continue
        rx, ry, rz = valid_pos

        # Color
        color_hex_full = color_hex if color_hex.startswith("#") else "#" + color_hex
        mat = create_abs_plastic_material(color_hex_full)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "index": idx,
            "mode": mode_tag,
            "ref": ref,
            "pose_index": pose_index,
            "original_pose_index": pose.get("original_pose_index"),
            "face_class": pose.get("face_class"),
            "color_code": color_code,
            "color_name": color_name,
            "color_hex": color_hex_full,
            "lateral_height_gt": pose.get("lateral_height"),
            "effective_height_gt": pose.get("effective_height"),
            "zenith_silhouette_area_gt": pose.get("zenith_silhouette_area"),
            "zenith_observable_area_gt": pose.get("zenith_observable_area"),
            "contact_normal_gt": pose.get("contact_normal"),
            "tipping_energy_ratio_gt": pose.get("tipping_energy_ratio"),
            "stability_ratio_gt": pose.get("stability_ratio"),
            "position_bu": [round(rx, 4), round(ry, 4), round(rz, 4)],
            "z_rotation_rad": round(z_rotation_rad, 6),
            "cameras": {},
        }

        ok = True
        prefix = f"inftestv2_{idx:03d}_{mode_tag}_{ref}_{color_code}_p{pose_index}"
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"{prefix}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"   [WARN] render fallido {cam_name}: {e}")
                ok = False
                break
            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "image_path": file_path,
                "bbox_norm": bbox_norm,
            }

        if ok and len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"   [OK] pos=({rx:.2f},{ry:.2f}) BU")
        else:
            skipped.append({"idx": idx, "ref": ref, "reason": "render_failed"})

    cleanup_piece_objects()

    meta_path = os.path.join(output_dir, pa.metadata_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": SET_ID,
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{RENDER_RES}x{RENDER_RES}",
            "fov_cenital_mm": int(2 * HALF_FOV_BU * 100),
            "margin_mm": int(MARGIN_BU * 100),
            "samples_count": len(results_meta),
            "samples_planned": len(plan),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[infTestV2 DONE] {len(results_meta)} muestras renderizadas (skipped={len(skipped)})")
    print(f"[infTestV2] Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
