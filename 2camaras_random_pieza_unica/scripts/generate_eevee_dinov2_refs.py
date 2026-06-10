# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_eevee_dinov2_refs.py
=============================================================
Renderiza imágenes de referencia DINOv2 para las 20 piezas del set 75078-1
usando el setup de pieza única centrada con 2 cámaras:
  - Motor:      BLENDER_EEVEE
  - Resolución: 640 × 640
  - Cámaras:    Cenital (0,0,15) + Lateral (15,0,2.5) PERSP f=27mm
  - Estrategia: 1 pieza centrada en (0,0,0), 12 rotaciones Z por pose estable
"""
import os, sys, random, math, json

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
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
# Reusar el setup canónico Machine Vision desde generate_test_set para que
# todos los scripts (test, 300_random, yolo_training, dinov2_refs)
# compartan exactamente la misma iluminación. Si el setup cambia, basta con
# tocarlo una vez en generate_test_set.py.
from generate_test_set import setup_lab_lightbox

from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("blender")

# ── Config ──
SELECTED_PARTS = cfg.pieces.selected_parts
# OVERRIDE de escala nueva (1 BU = 10 cm). El config sigue cargando los
# valores legacy (BELT_WIDTH_BU=20 = 200 cm) pero aqui los reescalamos
# a 2.0 BU = 20 cm de ancho, coherente con generate_inferencia_test_v2.
# La pieza viene normalizada con LDU_TO_BU=0.004 (3023 plate 1x2 = 0.128 BU
# = 1.28 cm), asi que ocupa ~6 % del FOV cenital de 20 cm. Sin este
# override, el ancho seria 200 cm y la pieza solo ocuparia 0.6 % del FOV.
BELT_WIDTH_BU = 2.0      # 20 cm (override de cfg.scene.belt.width_bu = 20.0)
BELT_LENGTH_BU = 12.0    # 120 cm
BELT_THICKNESS_BU = 0.1  # 1 cm
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
RENDER_RES_DEFAULT = cfg.render.resolution.width
MIN_CONTACT_DIM_MM = cfg.stable_poses.min_contact_dimension_mm
MIN_STABILITY = cfg.stable_poses.render_min_stability

# Optimizaciones EEVEE (sprint 1):
#   - 1.1 (TAA 16→8): ganancia ~25-30% sin pérdida visible para refs.
#   - B3: bloom/SSR/AO desactivados.
TAA_SAMPLES_OPT = 8


def apply_eevee_optimizations(scene):
    """Configura EEVEE: menos samples y sin efectos costosos (B1+B3)."""
    try:
        scene.eevee.taa_render_samples = TAA_SAMPLES_OPT
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = False
        log.info(f"[opt] EEVEE: TAA={TAA_SAMPLES_OPT}, bloom/SSR/AO=False")
    except Exception as e:
        log.warning(f"[opt] EEVEE optim parcial: {e}")


def get_stable_poses(part_ref):
    """Load stable poses from cache.

    El cache lo construye sync_stable_poses_cache.py y ya viene filtrado
    (stability_ratio>0.05 + contact_stable_width>=4mm). Aquí sólo
    aplicamos el filtro de stability_ratio si MIN_STABILITY > 0.05 (más
    estricto que el del cache) y leemos los nombres de campos modernos:
    contact_stable_width / contact_stable_length.
    """
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if part_ref not in cache:
            return []
        all_poses = cache[part_ref]
        stable = [p for p in all_poses if (p.get("stability_ratio") or 0.0) >= MIN_STABILITY]
        filtered = []
        for p in (stable or all_poses):
            cw = p.get("contact_stable_width", p.get("contact_width_mm"))
            cl = p.get("contact_stable_length", p.get("contact_length_mm"))
            # None passes (cara curva sin minAreaRect): el cache ya validó
            # que la pose pasa los filtros físicos básicos.
            if cw is None or cl is None:
                filtered.append(p)
            elif min(cw, cl) >= MIN_CONTACT_DIM_MM:
                filtered.append(p)
        if filtered:
            return filtered
        top_bottom = [p for p in all_poses if p.get("face_class") in ("Top", "Bottom")]
        if top_bottom:
            return top_bottom
        return all_poses if all_poses else []
    except Exception:
        return []


# La definición local de setup_lab_lightbox() se eliminó. Ahora se reusa
# el setup canónico Machine Vision desde generate_test_set (importado
# arriba). Si necesitas modificar la iluminación, hazlo en
# generate_test_set.setup_machine_vision_lighting().


def create_floor():
    if "Lab_Floor" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
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
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            bsdf.inputs['Roughness'].default_value = 1.0
    floor.data.materials.clear()
    floor.data.materials.append(mat)


def create_belt_collider():
    if 'Conveyor_Belt_Plane' in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Conveyor_Belt_Plane'].select_set(True)
        bpy.ops.object.delete()
    ht = BELT_THICKNESS_BU * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -ht))
    belt = bpy.context.active_object
    belt.name = 'Conveyor_Belt_Plane'
    belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    mat = bpy.data.materials.get('Belt_Material')
    if not mat:
        mat = bpy.data.materials.new('Belt_Material')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = BELT_COLOR_LINEAR
            bsdf.inputs['Roughness'].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)

    # Side rails
    for name in ["Side_Rail_L", "Side_Rail_R"]:
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.delete()
    rail_w = 0.2
    rail_h = 0.4
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-BELT_WIDTH_BU/2.0 + rail_w/2.0, 0.0, rail_h/2.0))
    rail_l = bpy.context.active_object
    rail_l.name = "Side_Rail_L"
    rail_l.scale = (rail_w, BELT_LENGTH_BU, rail_h)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(BELT_WIDTH_BU/2.0 - rail_w/2.0, 0.0, rail_h/2.0))
    rail_r = bpy.context.active_object
    rail_r.name = "Side_Rail_R"
    rail_r.scale = (rail_w, BELT_LENGTH_BU, rail_h)
    bpy.ops.object.transform_apply(scale=True)
    mat_metal = bpy.data.materials.get("Rail_Metal_Mat")
    if not mat_metal:
        mat_metal = bpy.data.materials.new("Rail_Metal_Mat")
        mat_metal.use_nodes = True
        bsdf = mat_metal.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.55, 0.55, 0.55, 1.0)
            bsdf.inputs['Metallic'].default_value = 0.9
            bsdf.inputs['Roughness'].default_value = 0.5
    for rail in [rail_l, rail_r]:
        rail.data.materials.clear()
        rail.data.materials.append(mat_metal)


def setup_cameras():
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    # Cenital
    cam_c_name = "Cam_Cenital"
    if cam_c_name in bpy.data.objects:
        cam_c = bpy.data.objects[cam_c_name]
    else:
        bpy.ops.object.camera_add(location=(0, 0, 15.0))
        cam_c = bpy.context.active_object
        cam_c.name = cam_c_name
    cam_c.location = (0.0, 0.0, 15.0)
    cam_c.constraints.clear()
    track_c = cam_c.constraints.new(type='TRACK_TO')
    track_c.target = target
    track_c.track_axis = 'TRACK_NEGATIVE_Z'
    track_c.up_axis = 'UP_Y'
    cam_c.data.type = 'PERSP'
    cam_c.data.lens = 27.0
    cam_c.data.clip_start = 0.01
    cam_c.data.clip_end = 100.0

    # Lateral
    cam_l_name = "Cam_Lateral"
    if cam_l_name in bpy.data.objects:
        cam_l = bpy.data.objects[cam_l_name]
    else:
        bpy.ops.object.camera_add(location=(15.0, 0.0, 2.5))
        cam_l = bpy.context.active_object
        cam_l.name = cam_l_name
    cam_l.location = (15.0, 0.0, 2.5)
    cam_l.constraints.clear()
    track_l = cam_l.constraints.new(type='TRACK_TO')
    track_l.target = target
    track_l.track_axis = 'TRACK_NEGATIVE_Z'
    track_l.up_axis = 'UP_Y'
    cam_l.data.type = 'PERSP'
    cam_l.data.lens = 27.0
    cam_l.data.clip_start = 0.01
    cam_l.data.clip_end = 100.0

    return cam_c, cam_l


def _normalize_piece(obj):
    """Misma logica que generate_test_set._normalize_piece (LDU_TO_BU=0.004
    en la nueva escala 1 BU = 10 cm)."""
    if not obj.data or not hasattr(obj.data, 'vertices'):
        return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return 1.0
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if mx < 1e-6:
        return 1.0
    LDU_TO_BU = 0.004
    factor = LDU_TO_BU if mx > 5.0 else 1.0
    cx = (max(xs)+min(xs))/2.0; cy = (max(ys)+min(ys))/2.0; cz = (max(zs)+min(zs))/2.0
    for v in obj.data.vertices:
        v.co.x = (v.co.x - cx) * factor
        v.co.y = (v.co.y - cy) * factor
        v.co.z = (v.co.z - cz) * factor
    obj.data.update()
    obj.scale = (1.0, 1.0, 1.0)
    obj.location = (0.0, 0.0, 0.0)
    return factor


def cleanup_piece():
    # Setup canónico Machine Vision (MV_Ring_Cenital + MV_Bar_Lateral_L/R).
    # Se mantienen los nombres legacy (Lab_*) por compatibilidad si algún
    # script externo aún los crea; son inocuos.
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral", "Lab_Floor",
            # Setup MV (actual)
            "MV_Ring_Cenital", "MV_Bar_Lateral_L", "MV_Bar_Lateral_R",
            # Setup legacy lab_lightbox (compatibilidad)
            "Lab_Main_Dome", "Lab_Wall_N", "Lab_Wall_S", "Lab_Wall_E", "Lab_Wall_W", "Lab_Ground_Fill"}
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and o.type not in ('CAMERA', 'LIGHT', 'EMPTY'):
            try:
                o.select_set(True)
            except:
                pass
    bpy.ops.object.delete()


def main():
    import time as _time
    _t_start = _time.perf_counter()

    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rotations", type=int, default=12)
    parser.add_argument("--render_res", type=int, default=RENDER_RES_DEFAULT,
                        help="Resolución cuadrada (px). Para refs DINOv2 se "
                             "recomienda 384 (opt 1.4) si la red trabaja a 224.")
    pa = parser.parse_known_args(args_raw)[0]
    out_dir = pa.output_dir
    render_res = int(pa.render_res)

    log_execution_header(log, "generate_eevee_dinov2_refs.py",
                         output_dir=out_dir, rotations=pa.rotations,
                         render_res=render_res,
                         selected_parts=SELECTED_PARTS)

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

    # Get colors for each part from the set catalog
    from database.set_catalog import REAL_SETS
    PART_COLORS_HEX = cfg.pieces.reference_colors_hex

    # Iterar SOLO las piezas del set 75078-1 (38 refs) que tengan poses
    # estables en el cache. El cache trae 68 piezas (incluye otros sets);
    # filtramos para acotar tiempo de render a ~22 min vs ~102 min.
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    set_refs = sorted({p["ref"] for p in REAL_SETS["75078-1"]["parts"]})
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_keys = set(json.load(f).keys())
        ALL_PARTS = sorted([r for r in set_refs if r in cache_keys])
    except Exception:
        ALL_PARTS = list(SELECTED_PARTS)
    log.info(f"Procesando {len(ALL_PARTS)} piezas del set 75078-1 con poses en cache.")

    for part_ref in ALL_PARTS:
        log.info(f"=== Generando referencias para pieza: {part_ref} ===")

        # Get real colors for this part in the set
        allowed_colors = []
        for p in REAL_SETS["75078-1"]["parts"]:
            if p["ref"] == part_ref:
                allowed_colors.append(p["color_hex"].replace("#", "").upper())
        if not allowed_colors:
            allowed_colors = [c.replace("#", "").upper() for c in PART_COLORS_HEX]

        poses = get_stable_poses(part_ref)
        if not poses:
            poses = [{"pose_index": 0, "orientation_quat": [1.0, 0.0, 0.0, 0.0]}]

        for pose in poses:
            pose_idx = pose.get("pose_index", 0)

            # Load mesh
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
                    log.warning(f"import LDraw {part_ref}: {e}")

            if not part_obj:
                generate_detailed_fallback_mesh(part_ref)
                part_obj = bpy.context.active_object

            if not part_obj:
                log.error(f"No se pudo cargar mesh para {part_ref}")
                continue

            bpy.ops.object.select_all(action='DESELECT')
            part_obj.select_set(True)
            bpy.context.view_layer.objects.active = part_obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            _normalize_piece(part_obj)
            apply_bevel_modifier(part_obj)

            # Rotations
            n_rots = pa.rotations
            rot_step = (2 * math.pi) / n_rots

            for rot_i in range(n_rots):
                rot_deg = int(round(rot_i * (360.0 / n_rots)))
                rot_rad = rot_i * rot_step

                # Apply pose orientation
                quat = pose.get("orientation_quat")
                if quat and len(quat) == 4:
                    part_obj.rotation_mode = 'QUATERNION'
                    part_obj.rotation_quaternion = mathutils.Quaternion(quat)
                else:
                    part_obj.rotation_mode = 'XYZ'
                    part_obj.rotation_euler = mathutils.Euler(pose.get("orientation_euler", [0, 0, 0]))

                # Add Z rotation
                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler.z += rot_rad

                # Position at center, snap to belt
                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                bbox_world = [part_obj.matrix_world @ mathutils.Vector(c) for c in part_obj.bound_box]
                min_z = min(pt.z for pt in bbox_world)
                part_obj.location.z = -min_z + 0.02
                bpy.context.view_layer.update()

                for color_hex in allowed_colors:
                    mat = create_abs_plastic_material(f"#{color_hex}")
                    part_obj.data.materials.clear()
                    part_obj.data.materials.append(mat)
                    bpy.context.view_layer.update()

                    fname = f"ref_{part_ref}_{color_hex}_pose{pose_idx:02d}_rot{rot_deg:03d}.png"

                    # ── Render Cenital ──
                    # Ocultamos del frame los planos opacos que la cámara cenital
                    # ve por debajo de la pieza (Lab_Floor, Conveyor_Belt_Plane,
                    # Side_Rail_L/R). De este modo `film_transparent=True` produce
                    # alpha=0 en el fondo, igual que ya ocurre con la cámara lateral
                    # (que mira hacia el horizonte y no ve esos planos).
                    # Esto es CRÍTICO para los colores translúcidos (Trans-Brown,
                    # Trans-Red…) cuyo matiz queda lavado contra fondo opaco.
                    _hide_targets = ["Lab_Floor", "Conveyor_Belt_Plane",
                                     "Side_Rail_L", "Side_Rail_R"]
                    _prev_hide = {}
                    for _n in _hide_targets:
                        _o = bpy.data.objects.get(_n)
                        if _o is not None:
                            _prev_hide[_n] = _o.hide_render
                            _o.hide_render = True

                    scene.camera = cam_c
                    scene.render.filepath = os.path.join(out_dir, "cenital", fname)
                    try:
                        bpy.ops.render.render(write_still=True)
                        total_rendered += 1
                    except Exception as e:
                        log.warning(f"Render cenital fallido: {e}")
                    finally:
                        # Restauramos visibilidad para el render lateral, que SÍ
                        # debe ver la cinta como suelo/horizon de referencia.
                        for _n, _prev in _prev_hide.items():
                            _o = bpy.data.objects.get(_n)
                            if _o is not None:
                                _o.hide_render = _prev

                    # ── Render Lateral ──
                    # Tambien ocultamos la geometria del entorno para que el
                    # render lateral salga con fondo transparente (= negro al
                    # convertir RGB), simetrico al cenital. Esto se alinea con
                    # el masking SAM que aplica el pipeline de inferencia
                    # (apply_sam_mask_to_crop) sobre el query lateral.
                    _prev_hide_lat = {}
                    for _n in _hide_targets:
                        _o = bpy.data.objects.get(_n)
                        if _o is not None:
                            _prev_hide_lat[_n] = _o.hide_render
                            _o.hide_render = True

                    scene.camera = cam_l
                    scene.render.filepath = os.path.join(out_dir, "lateral", fname)
                    try:
                        bpy.ops.render.render(write_still=True)
                        total_rendered += 1
                    except Exception as e:
                        log.warning(f"Render lateral fallido: {e}")
                    finally:
                        # Restaurar visibilidad para no afectar siguiente sample.
                        for _n, _prev in _prev_hide_lat.items():
                            _o = bpy.data.objects.get(_n)
                            if _o is not None:
                                _o.hide_render = _prev

            cleanup_piece()

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "generate_eevee_dinov2_refs.py",
                         duration_s=_duration,
                         total_rendered=total_rendered,
                         output_dir=out_dir)
    log.info(f"Generadas {total_rendered} imágenes de referencia DINOv2.")


if __name__ == "__main__":
    main()
