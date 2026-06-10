# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_lighting_variants.py
Renderiza UNA pieza con 4 variantes de iluminacion (A/B/C/D) para
elegir visualmente la mejor. Genera 8 PNGs (cenital+lateral x 4) +
HTML comparativo.

Uso:
    /opt/homebrew/bin/blender -b -P \\
        2camaras_random_pieza_unica/scripts/generate_lighting_variants.py -- \\
            --output_dir 2camaras_random_pieza_unica/data/lighting_variants \\
            --ref 3023 --pose_index 1 --color_code 5
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
HALF_FOV_BU = 10.0
MARGIN_BU = 0.5
MARGIN_NORM = MARGIN_BU / (2.0 * HALF_FOV_BU)
MAX_PLACEMENT_ATTEMPTS = 200
TARPS_MIN_TIPPING = float(getattr(cfg.stable_poses, "tarps_min_tipping", 0.04))
OUTPUT_DIR_DEFAULT = os.path.join(project_root, "data", "inferencia_test")
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

# Muestras forzadas (foco en stress-test del color).
FORCED_SAMPLES = [
    {"ref": "3023", "pose_index": 1, "color_code": "13"},  # Trans-Brown
    {"ref": "3023", "pose_index": 1, "color_code": "5"},   # Red
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


# ------------------------- Scene setup -------------------------
def build_scene():
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    enable_metal_gpu_acceleration()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    cam_cenital = setup_camera("Cam_Cenital", (0.0, 0.0, 15.0))
    cam_lateral = setup_camera("Cam_Lateral", (15.0, 0.0, 2.5))

    # Activar SSR + Refraction (o Raytracing en EEVEE Next 4.2+) globalmente.
    # Es idempotente y NO modifica luces/cámara/AO/bloom: solo permite que
    # los materiales translúcidos (Trans-Brown, Trans-Clear, …) refracten
    # correctamente y proyecten sombras coloreadas. Para piezas sólidas es
    # un no-op visual.
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


def variant_A_softbox_fill():
    """3 softboxes grandes + relleno hemisferico (60+40+40+25 W, world 0.20)."""
    _clear_lights()
    _set_world_strength(0.20)
    _add_area_light("A_Softbox_Cenital", (0.0, 0.0, 1.6),
                    "DISK", 1.5, 1.5, 60.0, specular=1.0)
    _add_area_light("A_Softbox_Lat_L", (1.5, -1.0, 1.0),
                    "RECTANGLE", 1.5, 1.0, 40.0, specular=1.0)
    _add_area_light("A_Softbox_Lat_R", (1.5, +1.0, 1.0),
                    "RECTANGLE", 1.5, 1.0, 40.0, specular=1.0)
    _add_area_light("A_Softbox_Trasera", (-1.5, 0.0, 1.0),
                    "RECTANGLE", 1.0, 1.0, 25.0, specular=1.0)


def variant_B_ring_3point():
    """Anillo cenital grande + 3 fills triangulares (50+30x3 W, world 0.15)."""
    _clear_lights()
    _set_world_strength(0.15)
    _add_area_light("B_Ring_Cenital", (0.0, 0.0, 1.6),
                    "DISK", 1.2, 1.2, 50.0, specular=1.0)
    angles = [0.0, 2.0944, 4.1888]  # 0, 120, 240 grados
    for i, a in enumerate(angles):
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        _add_area_light(f"B_Fill_{i}", (x, y, 0.6),
                        "RECTANGLE", 1.0, 0.6, 30.0, specular=1.0)


def variant_C_led_strips():
    """Tiras LED cruzadas (35x2 cen + 30x2 lat W, world 0.10)."""
    _clear_lights()
    _set_world_strength(0.10)
    _add_area_light("C_StripCen_X", (0.0, 0.0, 1.5),
                    "RECTANGLE", 2.0, 0.2, 35.0, specular=1.0)
    _add_area_light("C_StripCen_Y", (0.0, 0.0, 1.5),
                    "RECTANGLE", 0.2, 2.0, 35.0, specular=1.0)
    _add_area_light("C_StripLat_L", (1.5, -0.8, 0.8),
                    "RECTANGLE", 1.5, 0.2, 30.0, specular=1.0)
    _add_area_light("C_StripLat_R", (1.5, +0.8, 0.8),
                    "RECTANGLE", 1.5, 0.2, 30.0, specular=1.0)


def variant_D_bright_dome():
    """Domo grande tipo lightbox (80+25+25 W, world 0.30)."""
    _clear_lights()
    _set_world_strength(0.30)
    _add_area_light("D_Dome", (0.0, 0.0, 2.0),
                    "DISK", 2.5, 2.5, 80.0, specular=1.0)
    _add_area_light("D_Fill_L", (1.5, -1.0, 0.8),
                    "RECTANGLE", 1.0, 0.8, 25.0, specular=1.0)
    _add_area_light("D_Fill_R", (1.5, +1.0, 0.8),
                    "RECTANGLE", 1.0, 0.8, 25.0, specular=1.0)


VARIANTS = [
    ("A_softbox_fill", "3 softboxes grandes + fill hemisferico (60+40+40+25 W, world 0.20)",
     variant_A_softbox_fill),
    ("B_ring_3point",  "Anillo cenital grande + 3 fills triangulares (50+30x3 W, world 0.15)",
     variant_B_ring_3point),
    ("C_led_strips",   "Tiras LED cruzadas (35x2 cen + 30x2 lat W, world 0.10)",
     variant_C_led_strips),
    ("D_bright_dome",  "Domo grande tipo lightbox (80+25+25 W, world 0.30)",
     variant_D_bright_dome),
]


def find_pose_by_index(poses, target_index):
    """Busca pose por pose_index o original_pose_index."""
    for p in poses:
        if p.get("pose_index") == target_index or p.get("original_pose_index") == target_index:
            return p
    return None


# ------------------------- HTML compare report -------------------------
def write_compare_html(out_path, ref, pose_index, color_code, color_name,
                       color_hex, output_dir, variant_results):
    """Genera comparativa HTML con thumbnails cenital+lateral por variante."""
    head = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Lighting variants - comparativa</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#f4f6f8;padding:24px;color:#222}
  h1{margin-top:0}
  .card{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:18px;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  img{display:block;width:100%;max-width:480px;border:1px solid #ddd;border-radius:4px}
  .lbl{color:#64748b;font-size:12px;margin-bottom:4px}
  small{color:#64748b}
  .swatch{display:inline-block;width:14px;height:14px;border:1px solid #888;vertical-align:middle;margin-right:4px}
</style></head><body>
"""
    body = [head]
    body.append(f"<h1>Lighting variants &mdash; pieza {ref} (pose {pose_index})</h1>")
    body.append(
        f"<p>Color GT: <span class='swatch' style='background:{color_hex}'></span>"
        f"<b>{color_code}</b> &mdash; {color_name} (<code>{color_hex}</code>)</p>"
    )
    body.append("<p>Renders generados con Blender EEVEE, blackbody 5500 K, "
                "specular_factor=1.0 (sin polarizador). El setup actual "
                "(<code>setup_machine_vision_lighting</code>) sirve como "
                "baseline implicito (61 W, world 0.05).</p>")
    for vid, vdesc, cen_file, lat_file in variant_results:
        body.append(f"<div class='card'><h2>Variante {vid}</h2>")
        body.append(f"<small>{vdesc}</small>")
        body.append("<div class='grid' style='margin-top:8px'>")
        body.append(f"<div><div class='lbl'>Cenital</div>"
                    f"<img src='{cen_file}'/></div>")
        body.append(f"<div><div class='lbl'>Lateral</div>"
                    f"<img src='{lat_file}'/></div>")
        body.append("</div></div>")
    body.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("".join(body))


# ------------------------- Main -------------------------
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(project_root, "data", "lighting_variants"))
    parser.add_argument("--ref", type=str, default="3023")
    parser.add_argument("--pose_index", type=int, default=1)
    parser.add_argument("--color_code", type=str, default="5")
    pa = parser.parse_known_args(args)[0]

    output_dir = pa.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Cargar pose desde cache
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    poses = cache.get(pa.ref, [])
    if not poses:
        print(f"[ERROR] {pa.ref} sin poses en cache")
        sys.exit(1)
    pose = find_pose_by_index(poses, pa.pose_index) or poses[0]
    pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))
    cinfo = SET_COLORS.get(pa.color_code, {"name": "Unknown", "hex": "#A0A5A9"})
    color_hex = cinfo["hex"]
    color_name = cinfo["name"]
    print(f"[lightVar] ref={pa.ref} pose={pose_index} color={pa.color_code}({color_name}) hex={color_hex}")

    # Construir escena base
    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    # Importar mesh + normalizar + bevel + material
    cleanup_piece_objects()
    part_obj = import_part(pa.ref)
    if not part_obj:
        print(f"[ERROR] no se pudo importar mesh de {pa.ref}")
        sys.exit(1)
    bpy.ops.object.select_all(action="DESELECT")
    part_obj.select_set(True)
    bpy.context.view_layer.objects.active = part_obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    _normalize_piece(part_obj)
    apply_bevel_modifier(part_obj)

    # Aplicar pose y posicion centrada (deterministica para todas variantes)
    random.seed(42)
    part_obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    apply_stable_pose(part_obj, pose, random_z=False)
    bbox_world = _get_world_bbox(part_obj)
    min_z = min(pt.z for pt in bbox_world)
    part_obj.location = (-2.0, 0.0, -min_z + 0.02)  # offset para no estar en (0,0)
    bpy.context.view_layer.update()

    # Aplicar color real
    mat = create_abs_plastic_material(color_hex if color_hex.startswith("#") else "#" + color_hex)
    part_obj.data.materials.clear()
    part_obj.data.materials.append(mat)
    bpy.context.view_layer.update()

    # Renderizar cada variante
    variant_results = []
    for vid, vdesc, vfn in VARIANTS:
        print(f"\n[variant {vid}] {vdesc}")
        vfn()  # aplica las luces de la variante
        bpy.context.view_layer.update()
        cen_file = f"variant_{vid}_cenital.png"
        lat_file = f"variant_{vid}_lateral.png"
        for cam_name, cam_obj, fname in [
            ("cenital", cam_cenital, cen_file),
            ("lateral", cam_lateral, lat_file),
        ]:
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            scene.render.filepath = os.path.join(output_dir, fname)
            try:
                bpy.ops.render.render(write_still=True)
                print(f"   [OK] {fname}")
            except Exception as e:
                print(f"   [WARN] render fallido {cam_name}: {e}")
        variant_results.append((vid, vdesc, cen_file, lat_file))

    cleanup_piece_objects()

    # HTML comparativa
    html_path = os.path.join(output_dir, "comparison.html")
    write_compare_html(html_path, pa.ref, pose_index, pa.color_code,
                       color_name, color_hex, output_dir, variant_results)

    print("\n" + "=" * 60)
    print(f"[lightVar DONE] {len(variant_results)} variantes renderizadas")
    print(f"[lightVar] HTML comparativa: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
