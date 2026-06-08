# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_300_random_set.py
Genera N (default 300) imagenes (cenital + lateral) del set 75078-1
con balanceado por pose (opcion C):

  - Cada (ref, color, pose_estable) aparece >= 1 vez (TARPS).
  - El resto se reparte round-robin barajado hasta num_samples.
  - Posicion XY uniforme aleatoria DENTRO del FOV de AMBAS camaras
    (cenital + lateral; rejection sampling con margen 5 mm).
  - Rotacion Z aleatoria.

Uso:
    /opt/homebrew/bin/blender -b -P \\
        2camaras_random_pieza_unica/scripts/generate_300_random_set.py -- \\
            --num_samples 300 \\
            --output_dir 2camaras_random_pieza_unica/data/random_position \\
            --metadata_filename random_position_300_metadata.json \\
            --seed 42

Las imagenes existentes "sample_<ref>_*.png" se PRESERVAN; las nuevas
usan el prefijo "sample300_<idx>_".
"""
from __future__ import annotations
import json, math, os, random, sys

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
OUTPUT_DIR_DEFAULT = os.path.join(project_root, "data", "random_position")
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")


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


# ------------------------- Plan / catalog -------------------------
def get_unique_ref_color_combinations(set_id):
    """Devuelve dicts con todas las (ref, color_code) unicas del set."""
    seen = set()
    combos = []
    for p in REAL_SETS[set_id]["parts"]:
        key = (p["ref"], p["color_code"])
        if key in seen:
            continue
        seen.add(key)
        combos.append({
            "ref": p["ref"],
            "color_code": p["color_code"],
            "color_hex": p.get("color_hex", "#A0A5A9"),
            "color_name": p.get("color_name", "Unknown"),
            "qty": p.get("qty", 1),
        })
    return combos


def filter_stable_poses(poses):
    if not poses:
        return []
    stable = [p for p in poses
              if float(p.get("tipping_energy_ratio", 0.0)) >= TARPS_MIN_TIPPING]
    if stable:
        return stable
    best = max(poses, key=lambda p: float(p.get("tipping_energy_ratio", 0.0)))
    return [best]


def build_sample_plan(combos, cache, num_samples):
    """Opcion C: garantiza 1 muestra por (ref, color, pose_estable);
    rellena round-robin barajado hasta num_samples."""
    universe = []
    for combo in combos:
        ref = combo["ref"]
        poses = cache.get(ref, [])
        stable_poses = filter_stable_poses(poses)
        if not stable_poses:
            print(f"[plan] WARN: {ref} sin poses estables")
            if poses:
                stable_poses = [poses[0]]
            else:
                continue
        for pose in stable_poses:
            universe.append({
                "ref": ref,
                "color_code": combo["color_code"],
                "color_hex": combo["color_hex"],
                "color_name": combo["color_name"],
                "qty": combo["qty"],
                "pose": pose,
            })
    n_universe = len(universe)
    print(f"[plan] Universo (combo x pose): {n_universe} items "
          f"(combos={len(combos)})")

    plan = []
    if num_samples <= n_universe:
        idxs = list(range(n_universe))
        random.shuffle(idxs)
        plan = [universe[i] for i in idxs[:num_samples]]
    else:
        plan = list(universe)
        pool = list(universe)
        random.shuffle(pool)
        while len(plan) < num_samples:
            for item in pool:
                if len(plan) >= num_samples:
                    break
                plan.append(item)
    random.shuffle(plan)
    return plan


# ------------------------- Main -------------------------
def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--metadata_filename", type=str,
                        default="random_position_300_metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    parsed_args = parser.parse_known_args(args)[0]

    random.seed(parsed_args.seed)
    output_dir = parsed_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(CACHE_PATH):
        print(f"[ERROR] No se encuentra {CACHE_PATH}")
        sys.exit(1)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    combos = get_unique_ref_color_combinations(SET_ID)
    print(f"[300set] Set {SET_ID}: {len(combos)} combinaciones (ref,color)")

    plan = build_sample_plan(combos, cache, parsed_args.num_samples)
    print(f"[300set] Plan: {len(plan)} muestras")

    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene
    cameras = {"cenital": cam_cenital, "lateral": cam_lateral}

    results_meta = []
    skipped = []

    # Cache de meshes ya importados/normalizados (key=ref) para no
    # re-importar la misma pieza varias veces. El color y la pose se
    # aplican en cada iteracion.
    current_ref = None
    part_obj = None

    for idx, item in enumerate(plan):
        ref = item["ref"]
        color_code = item["color_code"]
        color_hex = item["color_hex"]
        color_name = item["color_name"]
        pose = item["pose"]
        pose_index = pose.get("pose_index", pose.get("original_pose_index", 0))

        print(f"\n[{idx+1:03d}/{len(plan)}] ref={ref} color={color_code}({color_name}) pose={pose_index}")

        # Re-importar mesh solo si cambia la ref
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

        # Aplicar pose + rotacion Z aleatoria + snap a cinta
        part_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        apply_stable_pose(part_obj, pose, random_z=True)

        # Posicion aleatoria valida en CENITAL y LATERAL simultaneamente
        valid_pos = sample_valid_position(
            part_obj, scene, cam_cenital, cam_lateral,
            margin_norm=MARGIN_NORM,
            max_attempts=MAX_PLACEMENT_ATTEMPTS,
        )
        if valid_pos is None:
            print(f"   [SKIP] no hay posicion valida en {MAX_PLACEMENT_ATTEMPTS} intentos")
            skipped.append({"idx": idx, "ref": ref, "reason": "no_valid_placement"})
            continue
        rx, ry, rz = valid_pos

        # Aplicar color real del catalogo
        if not color_hex.startswith("#"):
            color_hex_full = "#" + color_hex
        else:
            color_hex_full = color_hex
        mat = create_abs_plastic_material(color_hex_full)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat)
        bpy.context.view_layer.update()

        sample_meta = {
            "index": idx,
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
            "cameras": {},
        }

        # Render por camara
        ok = True
        prefix = f"sample300_{idx:03d}_{ref}_{color_code}_p{pose_index}"
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

    meta_path = os.path.join(output_dir, parsed_args.metadata_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": SET_ID,
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{RENDER_RES}x{RENDER_RES}",
            "fov_cenital_mm": int(2 * HALF_FOV_BU * 10),
            "margin_mm": int(MARGIN_BU * 10),
            "samples_count": len(results_meta),
            "samples_planned": len(plan),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[300set DONE] {len(results_meta)} muestras renderizadas "
          f"(plan={len(plan)}, skipped={len(skipped)})")
    print(f"[300set] Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
