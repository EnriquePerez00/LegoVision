# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/generate_canonical_dinov2_refs.py
========================================================================
Genera referencias DINOv2 (cenital + lateral) usando la ESCENA CANONICA
(`scene_canonical.build_scene_canonical`) con la pieza siempre en (0,0,0)
(snap a cinta).

Diferencia clave vs `generate_eevee_dinov2_refs.py`:
  * Usa la MISMA escena que la inferencia (cinta azul petroleo, pantalla
    aluminio, suelo oficina, luz V4 overhead strip + world strength 0.6).
  * NO oculta el entorno: el render lleva la cinta como fondo natural.
  * NO usa `film_transparent=True`. PNG 8-bit RGB con cinta como fondo.

Para cada (part_ref, pose, color_hex, rotation_z):
  1. cleanup_piece_objects + import_part + normalize_piece + bevel.
  2. rotacion analitica (contact_normal) + rot Z determinista i*30 deg.
  3. snap a cinta (z = -min_z + 0.005 BU). Posicion XY = (0,0).
  4. material ABS con color real.
  5. render cenital + lateral.

Salida (default):
  data/dinov2_refs_v3_canonical/{cenital,lateral}/
    ref_<part>_<COLORHEX>_pose<NN>_rot<DDD>.png

Uso:
  /opt/homebrew/bin/blender -b -P \\
      2camaras_random_pieza_unica/scripts/generate_canonical_dinov2_refs.py -- \\
      --refs 3023 --rotations 12 --render_res 384

  # Todas las refs del set 75078-1:
  /opt/homebrew/bin/blender -b -P \\
      2camaras_random_pieza_unica/scripts/generate_canonical_dinov2_refs.py -- \\
      --refs all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "scripts"))

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import (
    apply_bevel_modifier,
    create_abs_plastic_material,
)
from scene_canonical import (
    build_scene_canonical,
    cleanup_piece_objects,
    get_2d_bbox,
    import_part,
    normalize_piece,
)
from _pose_utils import rotation_quat_from_contact_normal
from database.set_catalog import REAL_SETS

try:
    from logger import get_logger, log_execution_header, log_execution_footer
    log = get_logger("blender")
    HAS_LOGGER = True
except Exception:
    HAS_LOGGER = False
    class _DummyLog:
        def info(self, msg): print(msg)
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    log = _DummyLog()


SET_ID = "75078-1"
DEFAULT_RENDER_RES = 384
DEFAULT_ROTATIONS = 12
CACHE_PATH = os.path.join(project_root, "data", "stable_poses_cache.json")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "data", "dinov2_refs_v3_canonical")
SNAP_OFFSET_BU = 0.005


def _load_stable_cache():
    if not os.path.isfile(CACHE_PATH):
        log.error(f"No existe cache de poses estables: {CACHE_PATH}")
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_refs(args_refs, cache):
    set_refs = []
    seen = set()
    for p in REAL_SETS[SET_ID]["parts"]:
        ref = p["ref"]
        if ref not in seen:
            seen.add(ref)
            set_refs.append(ref)
    if not args_refs or args_refs == ["all"]:
        return [r for r in set_refs if r in cache and cache[r]]
    chosen = [r for r in args_refs if r in cache and cache[r]]
    missing = [r for r in args_refs if r not in cache or not cache[r]]
    if missing:
        log.warning(f"Refs sin poses en cache: {missing}")
    return chosen


def _allowed_colors_for_ref(ref):
    out = []
    for p in REAL_SETS[SET_ID]["parts"]:
        if p["ref"] == ref:
            ch = (p.get("color_hex") or "").lstrip("#").upper()
            if ch and ch not in out:
                out.append(ch)
    if not out:
        out = ["A0A5A9"]
    return out


def _apply_pose_no_random_z(part_obj, pose):
    contact = pose.get("contact_normal")
    if contact and len(contact) == 3:
        quat = rotation_quat_from_contact_normal(contact)
    else:
        quat = (1.0, 0.0, 0.0, 0.0)
    part_obj.rotation_mode = "QUATERNION"
    part_obj.rotation_quaternion = mathutils.Quaternion(quat)
    bpy.context.view_layer.update()


def _apply_z_rotation(part_obj, angle_rad):
    q_z = mathutils.Quaternion((0.0, 0.0, 1.0), angle_rad)
    part_obj.rotation_quaternion = q_z @ part_obj.rotation_quaternion
    bpy.context.view_layer.update()


def _snap_to_belt(part_obj):
    """Snap del MESH REAL a la cinta (NO del AABB local rotado, que para
    piezas inclinadas dejaba la pieza flotando). Iteramos sobre los
    vertices reales del mesh para garantizar que el extremo inferior real
    queda en z=SNAP_OFFSET_BU."""
    if part_obj.data and hasattr(part_obj.data, "vertices"):
        mw = part_obj.matrix_world
        min_z = min((mw @ v.co).z for v in part_obj.data.vertices)
    else:
        bbox_local = [mathutils.Vector(c) for c in part_obj.bound_box]
        min_z = min((part_obj.matrix_world @ v).z for v in bbox_local)
    part_obj.location.z = part_obj.location.z - min_z + SNAP_OFFSET_BU
    bpy.context.view_layer.update()


def main():
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="Generar refs DINOv2 con escena canonica.")
    parser.add_argument("--refs", nargs="+", default=["all"])
    parser.add_argument("--rotations", type=int, default=DEFAULT_ROTATIONS)
    parser.add_argument("--render_res", type=int, default=DEFAULT_RENDER_RES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--colors", nargs="+", default=None,
                        help="Lista de hex (sin #) que sustituye a los del set.")
    parser.add_argument("--max_poses_per_ref", type=int, default=None)
    pa = parser.parse_known_args(args_raw)[0]

    t0 = time.perf_counter()
    if HAS_LOGGER:
        log_execution_header(
            log, "generate_canonical_dinov2_refs.py",
            refs=pa.refs, rotations=pa.rotations,
            render_res=pa.render_res, output_dir=pa.output_dir,
        )

    cache = _load_stable_cache()
    refs = _select_refs(pa.refs, cache)
    if not refs:
        log.error("No hay refs para procesar. Abortando.")
        sys.exit(1)
    log.info(f"Refs a procesar: {len(refs)} -> {refs}")

    out_cen = os.path.join(pa.output_dir, "cenital")
    out_lat = os.path.join(pa.output_dir, "lateral")
    os.makedirs(out_cen, exist_ok=True)
    os.makedirs(out_lat, exist_ok=True)

    cam_cenital, cam_lateral = build_scene_canonical(
        render_res=pa.render_res, film_transparent=False,
    )
    scene = bpy.context.scene

    total_rendered = 0
    total_failed = 0
    n_rots = int(pa.rotations)
    rot_step_deg = 360.0 / n_rots if n_rots > 0 else 0.0

    # Metadata por render: { fname: { "cenital": {bbox_norm, ...}, "lateral": {...},
    #                                  ref, color_hex, pose_index, rotation_deg } }
    metadata = {
        "scene": "canonical (scene_canonical.py)",
        "render_engine": "BLENDER_EEVEE",
        "resolution": f"{pa.render_res}x{pa.render_res}",
        "snap_offset_bu": SNAP_OFFSET_BU,
        "rotations": n_rots,
        "renders": [],
    }

    for ref in refs:
        poses = cache.get(ref, [])
        if pa.max_poses_per_ref is not None:
            poses = poses[: pa.max_poses_per_ref]
        if not poses:
            log.warning(f"Sin poses para {ref}. Skip.")
            continue
        colors = (
            [c.upper() for c in pa.colors] if pa.colors
            else _allowed_colors_for_ref(ref)
        )
        log.info(
            f"=== {ref}: {len(poses)} poses x {len(colors)} colores x {n_rots} rots "
            f"= {len(poses) * len(colors) * n_rots * 2} renders ==="
        )

        for pose in poses:
            pose_idx = pose.get("pose_index", pose.get("original_pose_index", 0))

            cleanup_piece_objects()
            part_obj = import_part(ref)
            if part_obj is None:
                log.warning(f"   No se pudo importar mesh para {ref}. Skip pose {pose_idx}.")
                continue

            bpy.ops.object.select_all(action="DESELECT")
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)
            part_obj.location = (0.0, 0.0, 0.0)

            for rot_i in range(n_rots):
                rot_deg = int(round(rot_i * rot_step_deg))
                rot_rad = math.radians(rot_deg)

                # Reposicionar pose desde cero para cada rotacion (evita
                # acumulacion de errores de quaternion).
                part_obj.location = (0.0, 0.0, 0.0)
                _apply_pose_no_random_z(part_obj, pose)
                _apply_z_rotation(part_obj, rot_rad)
                _snap_to_belt(part_obj)

                for color_hex in colors:
                    mat = create_abs_plastic_material(f"#{color_hex}")
                    part_obj.data.materials.clear()
                    part_obj.data.materials.append(mat)
                    bpy.context.view_layer.update()

                    fname = f"ref_{ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png"
                    sample_meta = {
                        "file_name": fname,
                        "ref": ref,
                        "color_hex": f"#{color_hex}",
                        "pose_index": int(pose_idx),
                        "rotation_deg": int(rot_deg),
                        "cameras": {},
                    }

                    # Cenital
                    scene.camera = cam_cenital
                    bpy.context.view_layer.update()
                    bbox_cen = get_2d_bbox(part_obj, scene, cam_cenital)
                    sample_meta["cameras"]["cenital"] = {
                        "file_name": fname,
                        "bbox_norm": [round(float(v), 6) for v in bbox_cen],
                    }
                    scene.render.filepath = os.path.join(out_cen, fname)
                    try:
                        bpy.ops.render.render(write_still=True)
                        total_rendered += 1
                    except Exception as e:
                        log.warning(f"Render cenital fallido ({fname}): {e}")
                        total_failed += 1

                    # Lateral
                    scene.camera = cam_lateral
                    bpy.context.view_layer.update()
                    bbox_lat = get_2d_bbox(part_obj, scene, cam_lateral)
                    sample_meta["cameras"]["lateral"] = {
                        "file_name": fname,
                        "bbox_norm": [round(float(v), 6) for v in bbox_lat],
                    }
                    scene.render.filepath = os.path.join(out_lat, fname)
                    try:
                        bpy.ops.render.render(write_still=True)
                        total_rendered += 1
                    except Exception as e:
                        log.warning(f"Render lateral fallido ({fname}): {e}")
                        total_failed += 1

                    metadata["renders"].append(sample_meta)

            cleanup_piece_objects()

    metadata["total_rendered"] = total_rendered
    metadata["total_failed"] = total_failed
    metadata_path = os.path.join(pa.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info(f"Metadata escrito: {metadata_path} "
             f"({len(metadata['renders'])} entries)")

    duration = time.perf_counter() - t0
    log.info(
        f"DONE. Rendered={total_rendered} | Failed={total_failed} | "
        f"Duration={duration:.1f}s | Output={pa.output_dir}"
    )
    if HAS_LOGGER:
        log_execution_footer(
            log, "generate_canonical_dinov2_refs.py",
            duration_s=duration,
            total_rendered=total_rendered,
            total_failed=total_failed,
            output_dir=pa.output_dir,
        )


if __name__ == "__main__":
    main()
