# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_lateral0_renders.py
=============================================================
Phase 1: Re-renderiza las imágenes laterales del test set usando la
misma pieza, pose y color que `data/test_dual/test_metadata.json`,
pero con la cámara lateral a Z=0.05 BU (casi a ras del suelo).

El objetivo es obtener renders de perfil puro para mejorar la
estimación de altura (MAPE actual ~25% con Z=2.5 BU).

Output
------
- data/dual_test_lateral0/sample_NNN_lateral.png   (100 renders)
- data/dual_test_lateral0/lateral0_metadata.json   (bboxes recalculadas)

Uso
---
    blender -b -P 2camaras_pieza_unica/scripts/generate_lateral0_renders.py -- [--num_samples N]
"""
from __future__ import annotations

import json
import math
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovision_root = os.path.dirname(project_root)
sys.path.insert(0, legovision_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

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

from config_loader import cfg  # noqa: E402
from generate_synthetic_set import (  # noqa: E402
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object  # noqa: E402
from _pose_utils import apply_stable_pose, get_stable_poses_for_ref  # noqa: E402

# ── Constantes ────────────────────────────────────────────────────────────────
RENDER_RES = cfg.render.resolution.width
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)

TEST_DUAL_DIR = os.path.join(project_root, "data", "test_dual")
SOURCE_META_PATH = os.path.join(TEST_DUAL_DIR, "test_metadata.json")
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")

OUTPUT_DIR = os.path.join(project_root, "data", "dual_test_lateral0")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "lateral0_metadata.json")

# ── Posición de cámara lateral a ras del suelo ───────────────────────────────
# Z=0.05 BU ≈ 1.5 mm — vista horizontal pura, evita z-fighting con la cinta
CAM_LATERAL_Z0_LOC = (15.0, 0.0, 0.05)


# ── Helpers escena ─────────────────────────────────────────────────────────────
def setup_lab_lightbox():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Lateral0", "Lab_Floor"}
    for o in list(bpy.context.scene.objects):
        if o.type == "LIGHT" and o.name not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.3

    neutral = (1.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 12.0))
    m = bpy.context.active_object
    m.name = "Lab_Main_Dome"
    m.data.size = 35.0; m.data.size_y = 35.0
    m.data.shape = "RECTANGLE"; m.data.color = neutral; m.data.energy = 2000.0

    target = bpy.data.objects.get("Camera_Target")
    for wname, wloc in [
        ("Lab_Wall_N", (0.0, +12.0, 6.0)),
        ("Lab_Wall_S", (0.0, -12.0, 6.0)),
        ("Lab_Wall_E", (+12.0, 0.0, 6.0)),
        ("Lab_Wall_W", (-12.0, 0.0, 6.0)),
    ]:
        bpy.ops.object.light_add(type="AREA", location=wloc)
        wp = bpy.context.active_object
        wp.name = wname
        wp.data.size = 20.0; wp.data.size_y = 12.0
        wp.data.shape = "RECTANGLE"; wp.data.color = neutral; wp.data.energy = 600.0
        if target:
            tr = wp.constraints.new(type="TRACK_TO")
            tr.target = target; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"

    bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, -0.5))
    gf = bpy.context.active_object
    gf.name = "Lab_Ground_Fill"
    gf.data.size = 30.0; gf.data.size_y = 30.0
    gf.data.shape = "RECTANGLE"; gf.data.color = neutral; gf.data.energy = 200.0
    gf.rotation_euler = (math.pi, 0.0, 0.0)


def create_floor():
    if "Lab_Floor" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
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
            bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Roughness"].default_value = 1.0
    floor.data.materials.clear()
    floor.data.materials.append(mat)


def create_belt():
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    ht = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.get("Belt_Material")
    if not mat:
        mat = bpy.data.materials.new("Belt_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
            bsdf.inputs["Roughness"].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)

    for name in ["Side_Rail_L", "Side_Rail_R"]:
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action="DESELECT")
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.delete()
    rail_w, rail_h = 0.2, 0.4
    for xoff, rname in [(-BELT_WIDTH_BU / 2.0 + rail_w / 2.0, "Side_Rail_L"),
                         ( BELT_WIDTH_BU / 2.0 - rail_w / 2.0, "Side_Rail_R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(xoff, 0.0, rail_h / 2.0))
        rail = bpy.context.active_object
        rail.name = rname
        rail.scale = (rail_w, BELT_LENGTH_BU, rail_h)
        bpy.ops.object.transform_apply(scale=True)


def setup_lateral0_camera():
    """Crea/configura la cámara lateral a Z=0.05 BU."""
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    cam_name = "Cam_Lateral0"
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
        cam.location = CAM_LATERAL_Z0_LOC
    else:
        bpy.ops.object.camera_add(location=CAM_LATERAL_Z0_LOC)
        cam = bpy.context.active_object
        cam.name = cam_name

    cam.constraints.clear()
    tr = cam.constraints.new(type="TRACK_TO")
    tr.target = target
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"
    cam.data.type = "PERSP"
    cam.data.lens = 27.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    return cam


def cleanup_piece():
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Lateral0", "Lab_Floor",
            "Lab_Main_Dome", "Lab_Wall_N", "Lab_Wall_S", "Lab_Wall_E", "Lab_Wall_W",
            "Lab_Ground_Fill"}
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and o.type not in ("CAMERA", "LIGHT", "EMPTY"):
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def get_2d_bbox(obj, scene, camera) -> list[float]:
    world_verts = []
    if obj.type == "MESH" and obj.data:
        m = obj.matrix_world
        world_verts = [m @ v.co for v in obj.data.vertices]
    if not world_verts:
        world_verts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs, ys = [], []
    for v in world_verts:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x); ys.append(co.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0)),
    ]


def normalize_piece(obj):
    if not obj.data or not hasattr(obj.data, "vertices"):
        return
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if mx < 1e-6:
        return
    factor = 0.04 if mx > 5.0 else 1.0
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    import argparse
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Número de muestras (por defecto: todas las del metadata)")
    pa = parser.parse_known_args(raw)[0]

    print("=" * 72)
    print("LATERAL-0 RENDERS — cámara lateral Z=0.05 BU")
    print(f"  Source metadata : {SOURCE_META_PATH}")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"  Cam location    : {CAM_LATERAL_Z0_LOC}")
    print("=" * 72)

    if not os.path.isfile(SOURCE_META_PATH):
        print(f"[ERROR] No se encuentra {SOURCE_META_PATH}")
        return 1
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra cache de poses: {CACHE_PATH}")
        return 1

    with open(SOURCE_META_PATH, "r", encoding="utf-8") as f:
        source_meta = json.load(f)

    renders = source_meta["renders"]
    if pa.num_samples is not None:
        renders = renders[: pa.num_samples]
    print(f"Muestras a procesar: {len(renders)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Build escena base ────────────────────────────────────────────────────
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except Exception:
            pass
    bpy.ops.object.delete()

    setup_physics_world()
    create_belt()
    create_floor()
    setup_lab_lightbox()
    enable_metal_gpu_acceleration()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES

    cam_lat0 = setup_lateral0_camera()
    scene.camera = cam_lat0

    # ── Pre-cargar cache poses ───────────────────────────────────────────────
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        full_cache = json.load(f)

    output_renders = []
    skipped = 0

    for i, r in enumerate(renders):
        ref = r["ref"]
        pose_idx = int(r["pose_index"])
        color_hex = r.get("color_hex", "#A0A5A9")

        print(f"\n[{i:03d}/{len(renders)}] ref={ref}  pose={pose_idx}  color={color_hex}")

        # Lookup pose en cache
        poses_for_ref = full_cache.get(ref, [])
        if not poses_for_ref:
            for suf in ("b", "a", "c"):
                alt = ref + suf
                if alt in full_cache:
                    poses_for_ref = full_cache[alt]
                    break
        pose_data = None
        for p in poses_for_ref:
            if p.get("original_pose_index") == pose_idx:
                pose_data = p
                break
        if pose_data is None and 0 <= pose_idx < len(poses_for_ref):
            pose_data = poses_for_ref[pose_idx]
        if pose_data is None:
            print(f"  [WARN] Pose {pose_idx} no encontrada para {ref}, skip")
            skipped += 1
            continue

        # Cargar malla
        part_path = get_ldraw_part_path(ref)
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
                print(f"  [WARN] Import LDraw {ref}: {e}")

        if not part_obj:
            generate_detailed_fallback_mesh(ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            print(f"  [WARN] No se pudo cargar {ref}, skip")
            skipped += 1
            continue

        # Preparar pieza
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        # Aplicar pose estable (misma lógica que generate_test_set.py)
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        # random_z=False para reproducir exactamente la misma rotación Z que
        # la muestra original. Si quieres variación, cambia a True.
        apply_stable_pose(part_obj, pose_data, random_z=False)

        # Material
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        # Bbox 2D con cámara lateral0
        scene.camera = cam_lat0
        bpy.context.view_layer.update()
        bbox_norm = get_2d_bbox(part_obj, scene, cam_lat0)

        # Render
        file_name = f"sample_{i:03d}_lateral.png"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        scene.render.filepath = file_path
        try:
            bpy.ops.render.render(write_still=True)
            print(f"  [OK] → {file_name}  bbox={[f'{v:.3f}' for v in bbox_norm]}")
        except Exception as e:
            print(f"  [ERROR] Render {file_name}: {e}")
            cleanup_piece()
            skipped += 1
            continue

        output_renders.append({
            "sample_index": i,
            "ref": ref,
            "pose_index": pose_idx,
            "color_hex": color_hex,
            "file_name": file_name,
            "image_path": os.path.join("2camaras_pieza_unica", "data",
                                       "dual_test_lateral0", file_name),
            "bbox_norm": bbox_norm,
            "cam_lateral_loc": list(CAM_LATERAL_Z0_LOC),
        })

        cleanup_piece()

    # ── Guardar metadata ─────────────────────────────────────────────────────
    out_meta = {
        "camera_config": {
            "name": "Cam_Lateral0",
            "location": list(CAM_LATERAL_Z0_LOC),
            "description": "Vista lateral a ras del suelo Z=0.05BU (~1.5mm)",
        },
        "source_metadata": SOURCE_META_PATH,
        "renders_count": len(output_renders),
        "skipped": skipped,
        "renders": output_renders,
    }
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"DONE — {len(output_renders)} renders generados  ({skipped} omitidos)")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Metadata   : {OUTPUT_META_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
