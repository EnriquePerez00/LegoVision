# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_set_random_position.py
==================================================================
Genera un render cenital + lateral por cada pieza ÚNICA del set
75078-1, usando una pose estable aleatoria, y ubicando la pieza en
una posición aleatoria del FOV de la cámara cenital (no en el
centro de la cinta), respetando un margen mínimo de 5 mm a los
bordes del FOV.

FOV cenital: cámara en (0,0,15 BU) con focal 27 mm y sensor 36 mm
            => half-FOV en el plano de la cinta = 100 mm = 10 BU.
            (coherente con cfg.inference.calibration.px_per_mm_cenital
             = 3.2 → 640 px / 3.2 = 200 mm de FOV total.)

Uso:
    /opt/homebrew/bin/blender -b -P \\
        2camaras_pieza_unica/scripts/generate_set_random_position.py

Salida:
    2camaras_pieza_unica/data/random_position/
        sample_<ref>_cenital.png
        sample_<ref>_lateral.png
        random_position_metadata.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

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
# Fuente única de verdad para selección y aplicación de pose
# estable — TARPS canónico + rotación analítica desde contact_normal.
# Ver docs/stable_pose_selection_rule.md y _pose_utils.py.
from _pose_utils import (
    get_stable_poses_for_ref,
    select_pose_tarps,
    apply_stable_pose,
)
from database.set_catalog import REAL_SETS

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
SET_ID = "75078-1"
RENDER_RES = cfg.render.resolution.width
MIN_CONTACT_DIM_MM = cfg.stable_poses.min_contact_dimension_mm
MIN_STABILITY = cfg.stable_poses.render_min_stability

# FOV cenital en BU: cámara a 15 BU de altura, focal 27 mm, sensor 36 mm
#   half_FOV = z * (sensor/2 / focal) = 15 * (18/27) = 10 BU = 100 mm
HALF_FOV_BU = 10.0
MARGIN_BU = 0.5          # 0.5 BU = 5 mm = 0.5 cm
MARGIN_NORM = MARGIN_BU / (2.0 * HALF_FOV_BU)   # = 0.025 (5 mm sobre 200 mm)

# Cuántas posiciones aleatorias intentamos por pieza antes de rendirnos.
MAX_PLACEMENT_ATTEMPTS = 200

OUTPUT_DIR = os.path.join(project_root, "data", "random_position")
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")


# ─────────────────────────────────────────────────────────────────
# Stable-pose loading (delegado a la fuente canónica TARPS)
# ─────────────────────────────────────────────────────────────────
# `get_stable_poses_tarps(ref)` y `select_pose_tarps(poses)` se
# importan de `generate_yolo_training_dataset.py` para garantizar
# que el criterio de selección sea idéntico al canónico documentado
# en docs/stable_pose_selection_rule.md (TARPS:
# `tipping_energy_ratio >= 0.04`, fallback `argmax(tipping)`).


# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────
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
    """Proyecta los 8 vértices del world-bbox de `obj` con la cámara dada
    y devuelve (x1, y1, x2, y2, depth_min) en coords normalizadas [0..1].
    `depth_min` permite detectar vértices detrás de la cámara."""
    bbox_world = _get_world_bbox(obj)
    xs, ys, zs = [], [], []
    for v in bbox_world:
        co = world_to_camera_view(scene, camera, v)
        xs.append(co.x)
        ys.append(co.y)
        zs.append(co.z)
    return min(xs), min(ys), max(xs), max(ys), min(zs)


def _bbox_within_margin(bbox_norm, margin) -> bool:
    x1, y1, x2, y2, depth_min = bbox_norm
    if depth_min <= 0:                       # algún vértice detrás del cam
        return False
    return (
        x1 >= margin and y1 >= margin
        and x2 <= 1.0 - margin and y2 <= 1.0 - margin
    )


def sample_valid_position(part_obj, scene, cam_cen, cam_lat,
                          margin_norm=MARGIN_NORM,
                          max_attempts=MAX_PLACEMENT_ATTEMPTS):
    """Rejection sampling: prueba posiciones (x, y) aleatorias dentro
    de la cinta y devuelve la primera que satisface, en AMBAS cámaras
    (cenital y lateral), `bbox_norm ⊂ [margin, 1-margin]` con `depth>0`.

    La pose y la rotación Z deben estar ya aplicadas al `part_obj`
    antes de llamar a esta función. La función modifica
    `part_obj.location` durante la búsqueda; al retornar, deja la
    pieza en la posición ganadora (ya con `snap-to-belt` aplicado).

    Devuelve `(x_bu, y_bu, z_bu)` o None si tras `max_attempts` no se
    encuentra ninguna posición válida.
    """
    # Rango holgado en el plano de la cinta. La validación real la
    # hace `_bbox_within_margin` con world_to_camera_view.
    sample_range = HALF_FOV_BU - 0.05  # casi todo el FOV cenital

    for _ in range(max_attempts):
        rx = random.uniform(-sample_range, sample_range)
        ry = random.uniform(-sample_range, sample_range)
        part_obj.location = (rx, ry, 0.0)
        bpy.context.view_layer.update()

        # Snap al plano de la cinta para esta XY candidata
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


# ─────────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────────
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
    return cam_cenital, cam_lateral


# ─────────────────────────────────────────────────────────────────
# Catalog helpers
# ─────────────────────────────────────────────────────────────────
def unique_refs_of_set(set_id: str) -> list:
    seen, refs = set(), []
    for p in REAL_SETS[set_id]["parts"]:
        ref = p["ref"]
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def get_color_for_ref(set_id: str, ref: str):
    for p in REAL_SETS[set_id]["parts"]:
        if p["ref"] == ref:
            return p.get("color_hex", "#A0A5A9"), p.get("color_name", "Unknown")
    return "#A0A5A9", "Light Bluish Gray"


def import_part(part_ref: str):
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


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla aleatoria (opcional)")
    parsed_args = parser.parse_known_args(args)[0]

    if parsed_args.seed is not None:
        random.seed(parsed_args.seed)

    output_dir = parsed_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Cache de poses
    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Refs únicas del set
    refs = unique_refs_of_set(SET_ID)
    print(f"[RandomPos] Set {SET_ID}: {len(refs)} piezas únicas")

    # Construir escena (una sola vez)
    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    results_meta = []
    skipped = []

    for i, ref in enumerate(refs):
        print(f"\n[{i+1}/{len(refs)}] Pieza {ref}")
        # TARPS canónico (docs/stable_pose_selection_rule.md): el cache
        # no pre-filtra; es el consumidor el que aplica TARPS.
        poses = get_stable_poses_for_ref(ref, CACHE_PATH)
        if not poses:
            print(f"   ↳ [SKIP] Sin poses estables en cache para {ref}")
            skipped.append({"ref": ref, "reason": "no_stable_poses"})
            continue

        # Importar mesh
        cleanup_piece_objects()
        part_obj = import_part(ref)
        if not part_obj:
            print(f"   ↳ [SKIP] No se pudo importar mesh de {ref}")
            skipped.append({"ref": ref, "reason": "import_failed"})
            continue

        # Normalizar
        bpy.ops.object.select_all(action="DESELECT")
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        # Pose estable seleccionada por TARPS (criterio canónico)
        pose = select_pose_tarps(poses)
        if pose is None:
            skipped.append({"ref": ref, "reason": "tarps_no_pose"})
            continue
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        # Aplicación analítica determinista (rotación derivada de
        # `contact_normal`, no del posiblemente-corrupto orientation_quat
        # del cache) + rotación Z aleatoria + snap a la cinta.
        # La pieza queda apoyada en la cara cuyo normal apunta a -Z.
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Posición aleatoria con rejection sampling: válida en CENITAL
        # y LATERAL al mismo tiempo, respetando margen MARGIN_NORM en
        # ambas (proyección perspective real, vía world_to_camera_view).
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_norm=MARGIN_NORM,
            max_attempts=MAX_PLACEMENT_ATTEMPTS,
        )
        if valid_pos is None:
            print(f"   ↳ [SKIP] {ref}: no hay posición válida en "
                  f"{MAX_PLACEMENT_ATTEMPTS} intentos (FOV cenital + lateral).")
            skipped.append({"ref": ref, "reason": "no_valid_placement"})
            continue
        rx, ry, rz = valid_pos

        # Color real del catálogo
        color_hex, color_name = get_color_for_ref(SET_ID, ref)
        if not color_hex.startswith("#"):
            color_hex = "#" + color_hex
        mat = create_abs_plastic_material(color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "ref": ref,
            "pose_index": pose_index,
            # FIX A: persistir original_pose_index + GT desde cache para
            # que la inferencia pueda hacer lookup canónico sin ambigüedad.
            "original_pose_index": pose.get("original_pose_index"),
            "face_class": pose.get("face_class"),
            "lateral_height_gt": pose.get("lateral_height"),
            "zenith_silhouette_area_gt": pose.get("zenith_silhouette_area"),
            "zenith_observable_area_gt": pose.get("zenith_observable_area"),
            "contact_normal_gt": pose.get("contact_normal"),
            "color_hex": color_hex,
            "color_name": color_name,
            "position_bu": [
                round(part_obj.location.x, 4),
                round(part_obj.location.y, 4),
                round(part_obj.location.z, 4),
            ],
            "cameras": {},
        }

        ok = True
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()
            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"sample_{ref}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"   ↳ [WARN] Render fallido {cam_name} {ref}: {e}")
                ok = False
                break
            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "image_path": file_path,
                "bbox_norm": bbox_norm,
            }

        if ok and len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"   ↳ [OK] {ref} pose={pose_index} "
                  f"pos=({rx:.2f},{ry:.2f}) BU color={color_name}")
        else:
            skipped.append({"ref": ref, "reason": "render_failed"})

    cleanup_piece_objects()

    # Guardar metadata
    meta_path = os.path.join(output_dir, "random_position_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": SET_ID,
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{RENDER_RES}x{RENDER_RES}",
            "fov_cenital_mm": int(2 * HALF_FOV_BU * 10),
            "margin_mm": int(MARGIN_BU * 10),
            "samples_count": len(results_meta),
            "skipped": skipped,
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[RandomPos DONE] {len(results_meta)} muestras renderizadas")
    if skipped:
        print(f"[RandomPos]      {len(skipped)} omitidas:")
        for s in skipped:
            print(f"    - {s['ref']}: {s['reason']}")
    print(f"[RandomPos] Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
