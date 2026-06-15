# -*- coding: utf-8 -*-
"""generate_eevee_dinov2_refs_parallel.py

Versión PARALELIZABLE de generate_eevee_dinov2_refs.py.

Importa las funciones del script original y solo modifica el main()
para procesar SOLO un slice de piezas [start_idx, end_idx).

CAMBIOS:
  - Acepta --start_idx, --end_idx, --worker_id, --skip_existing
  - Cada worker procesa un subconjunto de piezas
  - --skip_existing: salta archivos ya renderizados (resume)
  - TAA samples = 16 (calidad MANTENIDA, no reducir)
  - Optimizaciones EEVEE B3: bloom/SSR/AO desactivados
"""
import os
import sys
import math
import json
import time as _time

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] Ejecutar dentro de Blender (-b -P)")
    sys.exit(1)

from config_loader import cfg
from generate_synthetic_set import (
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

# Importar las funciones del script original (mismo directorio)
from generate_eevee_dinov2_refs import (
    get_stable_poses,
    setup_lab_lightbox,
    create_belt_collider,
    create_floor,
    setup_cameras,
    cleanup_piece,
    _normalize_piece,
    calculate_adaptive_rotations,
)

from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("blender_parallel")

SELECTED_PARTS = cfg.pieces.selected_parts
RENDER_RES_DEFAULT = cfg.render.resolution.width

# Optimizaciones EEVEE (sprint 1 — render refs DINOv2 reducido):
#   - 1.1 (TAA 16→8): ganancia ~25-30% sin pérdida visible para refs DINOv2.
#   - B3: bloom/SSR/AO desactivados (innecesarios para refs canónicas).
TAA_SAMPLES = 8
DISABLE_BLOOM = True
DISABLE_SSR = True
DISABLE_AO = True


def apply_eevee_optimizations(scene):
    """Aplica B3: desactivar features EEVEE no críticos. TAA mantenido en 16."""
    try:
        scene.eevee.taa_render_samples = TAA_SAMPLES
        if DISABLE_BLOOM and hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if DISABLE_SSR and hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if DISABLE_AO and hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
        log.info(f"[opt] EEVEE: TAA={TAA_SAMPLES}, bloom/SSR/AO=False")
    except Exception as e:
        log.warning(f"[opt] EEVEE optim parcial: {e}")


def main():
    _t_start = _time.perf_counter()

    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rotations", type=int, default=12, help="Rotaciones por defecto si no se usa heurística.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--render_res", type=int, default=384,
                        help="Resolución cuadrada (px) del render. "
                             "Para refs DINOv2 se recomienda 384 (1.4).")
    pa = parser.parse_known_args(args_raw)[0]
    out_dir = pa.output_dir
    render_res = int(pa.render_res)

    log_execution_header(log, "generate_eevee_dinov2_refs_parallel.py",
                         worker_id=pa.worker_id,
                         start_idx=pa.start_idx, end_idx=pa.end_idx,
                         output_dir=out_dir, rotations=pa.rotations,
                         render_res=render_res,
                         skip_existing=pa.skip_existing)

    for c in ["cenital", "lateral"]:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    enable_metal_gpu_acceleration()
    setup_physics_world()
    create_belt_collider()
    create_floor()
    setup_lab_lightbox()
    cam_c, cam_l = setup_cameras()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.render.resolution_x = render_res
    scene.render.resolution_y = render_res
    apply_eevee_optimizations(scene)

    total_rendered = 0
    total_skipped = 0

    from database.set_catalog import REAL_SETS
    PART_COLORS_HEX = cfg.pieces.reference_colors_hex

    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            ALL_PARTS = sorted(json.load(f).keys())
    except Exception:
        ALL_PARTS = list(SELECTED_PARTS)

    end_idx = len(ALL_PARTS) if pa.end_idx < 0 else min(pa.end_idx, len(ALL_PARTS))
    start_idx = max(0, pa.start_idx)
    PARTS_SLICE = ALL_PARTS[start_idx:end_idx]
    log.info(f"[w{pa.worker_id}] Slice [{start_idx},{end_idx}): "
             f"{len(PARTS_SLICE)} piezas (de {len(ALL_PARTS)} totales)")

    for part_ref in PARTS_SLICE:
        log.info(f"[w{pa.worker_id}] === Pieza: {part_ref} ===")

        allowed_colors = []
        try:
            from database import supabase_client
            with supabase_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT color_hex FROM lego_set_parts WHERE part_ref = %s AND color_hex IS NOT NULL", (part_ref,))
                    for row in cur.fetchall():
                        color_h = row.get("color_hex") if isinstance(row, dict) else row[0]
                        if color_h:
                            allowed_colors.append(color_h.replace("#", "").upper())
        except Exception as e:
            log.warning(f"[w{pa.worker_id}] Failed to fetch colors from DB for {part_ref}: {e}")

        if not allowed_colors:
            # Fallback to searching REAL_SETS (all sets in the catalog)
            for s_id, s_data in REAL_SETS.items():
                for p in s_data.get("parts", []):
                    if p["ref"] == part_ref and p.get("color_hex"):
                        allowed_colors.append(p["color_hex"].replace("#", "").upper())

        # Deduplicate and sort
        allowed_colors = sorted(list(set(allowed_colors)))

        if not allowed_colors:
            allowed_colors = [str(c).replace("#", "").upper() for c in PART_COLORS_HEX]

        poses = get_stable_poses(part_ref)
        if not poses:
            poses = [{"pose_index": 0, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}]

        for pose in poses:
            pose_idx = pose.get("pose_index", 0)

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
                    log.warning(f"[w{pa.worker_id}] import LDraw {part_ref}: {e}")

            if not part_obj:
                generate_detailed_fallback_mesh(part_ref)
                part_obj = bpy.context.active_object

            if not part_obj:
                log.error(f"[w{pa.worker_id}] No mesh: {part_ref}")
                continue

            bpy.ops.object.select_all(action='DESELECT')
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)

            n_rots, rot_step = calculate_adaptive_rotations(part_obj, pose)

            for rot_i in range(n_rots):
                rot_rad = rot_i * rot_step
                rot_deg = int(round(math.degrees(rot_rad)))

                quat = pose.get("orientation_quat")
                if quat and len(quat) == 4:
                    part_obj.rotation_mode = 'QUATERNION'
                    part_obj.rotation_quaternion = mathutils.Quaternion(quat)
                else:
                    part_obj.rotation_mode = 'XYZ'
                    part_obj.rotation_euler = mathutils.Euler(pose.get("orientation_euler", [0, 0, 0]))

                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler.z += rot_rad

                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
                min_z = min(pt.z for pt in bbox_world)
                part_obj.location.z = -min_z + 0.02
                bpy.context.view_layer.update()

                for color_hex in allowed_colors:
                    fname = f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png"
                    cenital_path = os.path.join(out_dir, "cenital", fname)
                    lateral_path = os.path.join(out_dir, "lateral", fname)

                    skip_cen = pa.skip_existing and os.path.exists(cenital_path)
                    skip_lat = pa.skip_existing and os.path.exists(lateral_path)

                    if skip_cen and skip_lat:
                        total_skipped += 2
                        continue

                    mat = create_abs_plastic_material(f"#{color_hex}")
                    part_obj.data.materials.clear()
                    part_obj.data.materials.append(mat)
                    bpy.context.view_layer.update()

                    if not skip_cen:
                        # ── Render Cenital ──
                        # Ocultamos del frame los planos opacos que la cámara cenital
                        # ve por debajo de la pieza (Lab_Floor, Conveyor_Belt_Plane,
                        # Side_Rail_L/R) para que `film_transparent=True` produzca
                        # alpha=0 en el fondo, igual que ya ocurre con el lateral.
                        # Crítico para colores translúcidos (Trans-Brown, Trans-Red…).
                        _hide_targets = ["Lab_Floor", "Conveyor_Belt_Plane",
                                         "Side_Rail_L", "Side_Rail_R"]
                        _prev_hide = {}
                        for _n in _hide_targets:
                            _o = bpy.data.objects.get(_n)
                            if _o is not None:
                                _prev_hide[_n] = _o.hide_render
                                _o.hide_render = True

                        scene.camera = cam_c
                        scene.render.filepath = cenital_path
                        try:
                            bpy.ops.render.render(write_still=True)
                            total_rendered += 1
                        except Exception as e:
                            log.warning(f"[w{pa.worker_id}] Cen fallido {fname}: {e}")
                        finally:
                            # Restaurar visibilidad antes del render lateral.
                            for _n, _prev in _prev_hide.items():
                                _o = bpy.data.objects.get(_n)
                                if _o is not None:
                                    _o.hide_render = _prev
                    else:
                        total_skipped += 1

                    if not skip_lat:
                        # ── Render Lateral ──
                        scene.camera = cam_l
                        scene.render.filepath = lateral_path
                        try:
                            bpy.ops.render.render(write_still=True)
                            total_rendered += 1
                        except Exception as e:
                            log.warning(f"[w{pa.worker_id}] Lat fallido {fname}: {e}")
                    else:
                        total_skipped += 1

            cleanup_piece()

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "generate_eevee_dinov2_refs_parallel.py",
                         worker_id=pa.worker_id,
                         total_rendered=total_rendered,
                         total_skipped=total_skipped,
                         duration_seconds=round(_duration, 2),
                         num_parts_processed=len(PARTS_SLICE))


if __name__ == "__main__":
    main()
