# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/generate_test_set.py
=====================================================
Genera el set de test (100 muestras) para el pipeline 2camaras_pieza_unica.
Cada muestra: 1 pieza centrada, renderizada desde cenital + lateral.
"""
import os, sys, random, math, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts'))

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
    setup_physics_world,
    create_abs_plastic_material,
    apply_bevel_modifier,
    get_ldraw_part_path,
    generate_detailed_fallback_mesh,
    enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object
# Fuente única de verdad: TARPS + rotación analítica desde
# contact_normal. Ver _pose_utils.py y docs/stable_pose_selection_rule.md.
from _pose_utils import (
    apply_stable_pose,
    get_stable_poses_for_ref,
    select_pose_tarps,
)

SELECTED_PARTS = cfg.pieces.selected_parts
PART_COLOR_HEX = cfg.pieces.test_color_hex
BELT_WIDTH_BU = cfg.scene.belt.width_bu
BELT_LENGTH_BU = cfg.scene.belt.length_bu
BELT_THICKNESS_BU = cfg.scene.belt.thickness_bu
BELT_COLOR_LINEAR = tuple(cfg.scene.belt.color_linear)
RENDER_RES = cfg.render.resolution.width
MIN_CONTACT_DIM_MM = cfg.stable_poses.min_contact_dimension_mm
MIN_STABILITY = cfg.stable_poses.render_min_stability


def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece(obj):
    """Normaliza una pieza importada de LDraw centrandola en el origen y
    escalandola al sistema de Blender Units del proyecto.

    Convencion: en la escala nueva (1 BU = 10 cm = 100 mm), 1 LDU
    (LDraw Unit) = 0.4 mm = 0.004 BU. El factor 0.004 transforma
    los vertices LDraw directos a BU del proyecto.

    Para la escala vieja (1 BU = 1 cm = 10 mm) se usaba factor 0.04
    (1 LDU = 0.04 BU). Si en algun momento se restaura esa escala,
    cambiar `LDU_TO_BU` a 0.04.
    """
    if not obj.data or not hasattr(obj.data, 'vertices'):
        return 1.0
    verts = [v.co for v in obj.data.vertices]
    if not verts:
        return 1.0
    xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
    mx = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    if mx < 1e-6:
        return 1.0
    # 0.004 = 1 LDU (0.4 mm) -> 1 BU (10 cm) cuando 1 BU = 10 cm.
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


def get_stable_poses(part_ref):
    """Wrapper compatibilidad: delega en `_pose_utils.get_stable_poses_for_ref`.
    Devuelve TODAS las poses estables ordenadas por tipping_energy_ratio
    descendente (sin filtros legacy). El consumidor debe aplicar
    `select_pose_tarps()` para escoger una pose individual.

    Ver docs/stable_pose_selection_rule.md para la regla TARPS oficial."""
    cache_path = os.path.join(project_root, "data", "stable_poses_cache.json")
    return get_stable_poses_for_ref(part_ref, cache_path)


def get_2d_bbox(obj, scene, camera):
    bbox_coords = _get_world_bbox(obj)
    xs, ys = [], []
    for v in bbox_coords:
        co_2d = world_to_camera_view(scene, camera, v)
        xs.append(co_2d.x)
        ys.append(co_2d.y)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [
        max(0.0, min(x1, 1.0)),
        max(0.0, min(1.0 - y2, 1.0)),
        max(0.0, min(x2, 1.0)),
        max(0.0, min(1.0 - y1, 1.0)),
    ]


def setup_camera(cam_name, location):
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"

    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
        cam.location = location
    else:
        bpy.ops.object.camera_add(location=location)
        cam = bpy.context.active_object
        cam.name = cam_name

    cam.constraints.clear()
    track = cam.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    cam.data.type = 'PERSP'
    if "Cenital" in cam_name:
        cam.data.lens = 55.0
    else:
        cam.data.lens = 27.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    return cam


def _set_light_blackbody(light_obj, temp_k=5500.0, energy_w=20.0):
    """Configura una luz para que emita luz blanca neutra mediante un nodo
    Blackbody @ ``temp_k`` Kelvin (5500K = luz de día neutra industrial).

    Habilita ``use_nodes=True`` y reemplaza el árbol por:

        Blackbody → Emission(strength=energy_w) → Light Output

    Esto evita que el color por defecto del shader interno tiña la imagen
    con tonos cálidos (4100K) o azulados, garantizando neutralidad cromática
    para los plásticos ABS (negros, blancos, transparentes).
    """
    light = light_obj.data
    light.use_nodes = True
    nt = light.node_tree
    nt.nodes.clear()

    n_blackbody = nt.nodes.new(type='ShaderNodeBlackbody')
    n_blackbody.location = (-300, 0)
    n_blackbody.inputs['Temperature'].default_value = float(temp_k)

    n_emission = nt.nodes.new(type='ShaderNodeEmission')
    n_emission.location = (-50, 0)
    n_emission.inputs['Strength'].default_value = float(energy_w)

    n_output = nt.nodes.new(type='ShaderNodeOutputLight')
    n_output.location = (200, 0)

    nt.links.new(n_blackbody.outputs['Color'], n_emission.inputs['Color'])
    nt.links.new(n_emission.outputs['Emission'], n_output.inputs['Surface'])

    # Mantener también la energía a nivel de objeto (Blender la usa como
    # fallback / para previews). El strength del nodo Emission domina el
    # render final cuando use_nodes=True.
    light.energy = float(energy_w)


def setup_machine_vision_lighting():
    """Setup Dome Light + Cross-Polarization (canonical desde 2026-06-13).

    Dome Light perfecto usando exclusivamente el World Background:
      - NO area lights, NO directional lights.
      - WORLD_BG_STRENGTH = 1.5 (iluminación uniforme desde todas direcciones).
      - WORLD_BG_COLOR = blanco puro (1,1,1,1) para no contaminar CIELAB.

    Cross-Polarization simulada en los materiales:
      - Piezas: Specular = 0.05, Roughness = 0.75 (en create_abs_plastic_material).
      - Cinta: Specular = 0.0, Roughness = 1.0 (mate perfecto).

    Esto elimina los brillos especulares blancos del plástico ABS,
    dejando solo el color difuso (albedo) visible, ideal para Machine Vision.

    Color management: View=Standard, Look=None (fidelidad de color máxima).
    """
    # Importar parámetros desde scene_config
    try:
        from scene_config import WORLD_BG_STRENGTH, WORLD_BG_COLOR
    except ImportError:
        WORLD_BG_STRENGTH = 1.5
        WORLD_BG_COLOR = (1.0, 1.0, 1.0, 1.0)

    # Limpiar TODAS las luces previas (Dome Light no usa area lights).
    for o in list(bpy.context.scene.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o, do_unlink=True)
    # Eliminar data-blocks de luz huérfanos.
    for ld in [l for l in bpy.data.lights if l.users == 0]:
        bpy.data.lights.remove(ld)

    # ══════════════════════════════════════════════════════════════════════
    # DOME LIGHT via World Background
    # ══════════════════════════════════════════════════════════════════════
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        # Blanco puro para no contaminar espacio de color CIELAB
        bg.inputs["Color"].default_value = WORLD_BG_COLOR
        # Strength alto para iluminación uniforme
        bg.inputs["Strength"].default_value = float(WORLD_BG_STRENGTH)

    # Color management para vision artificial: View=Standard, Look=None
    # (máxima fidelidad de color, sin transformaciones artísticas).
    scene.view_settings.view_transform = 'Standard'
    try:
        scene.view_settings.look = 'None'
    except TypeError:
        pass


# Alias retrocompatible: scripts existentes (generate_300_random_set,
# generate_yolo_training_dataset, generate_eevee_dinov2_refs) importan
# `setup_lab_lightbox` desde este módulo. Mantener el nombre evita romper
# imports y permite cambiar el comportamiento de iluminación globalmente.
setup_lab_lightbox = setup_machine_vision_lighting


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
    """Crea la cinta transportadora con material mate (cross-polarization)."""
    # Importar parámetros de cross-polarization desde scene_config
    try:
        from scene_config import BELT_SPECULAR, BELT_ROUGHNESS
    except ImportError:
        BELT_SPECULAR = 0.0
        BELT_ROUGHNESS = 1.0

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
        # Color azul petróleo canónico (NO modificar)
        bsdf.inputs['Base Color'].default_value = BELT_COLOR_LINEAR
        # Cross-polarization: mate perfecto para no reflejar Dome Light
        bsdf.inputs['Roughness'].default_value = BELT_ROUGHNESS  # 1.0
        # Specular = 0.0 (sin brillo especular)
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = BELT_SPECULAR  # 0.0
        elif 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = BELT_SPECULAR  # 0.0
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


def cleanup_piece_objects():
    # Setup canónico Machine Vision (MV_Ring_Cenital + MV_Bar_Lateral_L/R).
    # Se mantienen los nombres antiguos (Lab_*) en `keep` por seguridad: si
    # algún script externo aún crea esas luces, no se eliminarán por error.
    keep = {"Conveyor_Belt_Plane", "Camera_Target", "Side_Rail_L", "Side_Rail_R",
            "Cam_Cenital", "Cam_Lateral", "Lab_Floor",
            # Setup Domo Industrial (actual, 2026-09-06b)
            "Key_Cenital", "Fill_Lateral_+Y", "Fill_Lateral_-Y",
            # Setup MV Ring+Bars (legacy intermedio)
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


def build_scene():
    """Build full scene: belt, floor, lights, cameras."""
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.context.scene.objects):
        try:
            o.select_set(True)
        except:
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

    cam_cenital = setup_camera("Cam_Cenital", (0.0, 0.0, 30.0))
    cam_lateral = setup_camera("Cam_Lateral", (15.0, 0.0, 2.5))

    return cam_cenital, cam_lateral


def main():
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parsed_args = parser.parse_known_args(args)[0]

    # Default: renders/test/test_dual (separación de dominios)
    output_dir = parsed_args.output_dir or os.path.join(project_root, "renders", "test", "test_dual")
    num_samples = parsed_args.num_samples
    os.makedirs(output_dir, exist_ok=True)

    cam_cenital, cam_lateral = build_scene()
    scene = bpy.context.scene

    cameras = {
        "cenital": cam_cenital,
        "lateral": cam_lateral,
    }

    # Iterar sobre TODAS las piezas que tengan poses estables en el cache,
    # no solo SELECTED_PARTS. Garantiza que el test set cubre todo el set.
    cache_path_full = os.path.join(project_root, "data", "stable_poses_cache.json")
    try:
        with open(cache_path_full, "r", encoding="utf-8") as fcache:
            ALL_PARTS = sorted(json.load(fcache).keys())
    except Exception:
        ALL_PARTS = list(SELECTED_PARTS)
    print(f"[TestGen] Procesando {len(ALL_PARTS)} piezas (todas las del cache stable_poses).")

    # Pre-load stable poses cache
    stable_poses_cache = {}
    for part in ALL_PARTS:
        poses = get_stable_poses(part)
        if poses:
            stable_poses_cache[part] = poses
        else:
            print(f"[TestGen Warning] Sin poses estables para {part}")

    results_meta = []

    for i in range(num_samples):
        part_ref = random.choice(ALL_PARTS)
        print(f"\n[{i+1}/{num_samples}] Generando muestra de test: {part_ref}")

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
                pass

        if not part_obj:
            generate_detailed_fallback_mesh(part_ref)
            part_obj = bpy.context.active_object

        if not part_obj:
            continue

        # Prepare piece
        bpy.ops.object.select_all(action='DESELECT')
        part_obj.select_set(True)
        bpy.context.view_layer.objects.active = part_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        _normalize_piece(part_obj)
        apply_bevel_modifier(part_obj)

        # Apply stable pose using TARPS (canónico). La rotación se
        # deriva analíticamente de `contact_normal` (determinista),
        # incluyendo Z aleatorio y snap a la cinta.
        # Ver _pose_utils.py y docs/stable_pose_selection_rule.md.
        poses = stable_poses_cache.get(part_ref, [])
        selected_pose_idx = 0
        if poses:
            pose = select_pose_tarps(poses)
            if pose is not None:
                selected_pose_idx = pose.get("pose_index", 0)
                # Posicionar en el centro y aplicar pose+snap.
                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                apply_stable_pose(part_obj, pose, random_z=True)
            else:
                part_obj.rotation_mode = 'XYZ'
                part_obj.rotation_euler = (0, 0, 0)
                part_obj.location = (0.0, 0.0, 0.0)
                bpy.context.view_layer.update()
                bbox_world = _get_world_bbox(part_obj)
                min_z = min(pt.z for pt in bbox_world)
                part_obj.location.z = -min_z + 0.02
        else:
            # Sin pose en cache: identidad + snap (caso degenerado).
            part_obj.rotation_mode = 'XYZ'
            part_obj.rotation_euler = (0, 0, 0)
            part_obj.location = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            bbox_world = _get_world_bbox(part_obj)
            min_z = min(pt.z for pt in bbox_world)
            part_obj.location.z = -min_z + 0.02

        # Get real color from set catalog
        from database.set_catalog import REAL_SETS
        part_color_hex = f"#{PART_COLOR_HEX}" if not PART_COLOR_HEX.startswith("#") else PART_COLOR_HEX
        for p in REAL_SETS["75078-1"]["parts"]:
            if p["ref"] == part_ref:
                part_color_hex = p["color_hex"]
                break

        mat_abs = create_abs_plastic_material(part_color_hex)
        part_obj.data.materials.clear()
        part_obj.data.materials.append(mat_abs)
        bpy.context.view_layer.update()

        sample_meta = {
            "ref": part_ref,
            "pose_index": selected_pose_idx,
            "color_hex": part_color_hex,
            "cameras": {},
        }

        # Render from both cameras
        for cam_name, cam_obj in cameras.items():
            scene.camera = cam_obj
            bpy.context.view_layer.update()

            bbox_norm = get_2d_bbox(part_obj, scene, cam_obj)
            file_name = f"sample_{i:03d}_{cam_name}.png"
            file_path = os.path.join(output_dir, file_name)
            scene.render.filepath = file_path

            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"[WARN] Render fallido {cam_name} muestra {i}: {e}")
                continue

            sample_meta["cameras"][cam_name] = {
                "file_name": file_name,
                "bbox_norm": bbox_norm,
                "image_path": file_path,
            }

        if len(sample_meta["cameras"]) == 2:
            results_meta.append(sample_meta)
            print(f"[OK] Muestra {i+1}: {part_ref} | pose={selected_pose_idx}")
        else:
            print(f"[WARN] Muestra {i+1} incompleta, descartada.")

        cleanup_piece_objects()

    # Save metadata
    meta_path = os.path.join(output_dir, "test_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": "75078-1",
            "render_engine": "BLENDER_EEVEE",
            "resolution": f"{RENDER_RES}x{RENDER_RES}",
            "samples_count": len(results_meta),
            "renders": results_meta,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[TestGen DONE] {len(results_meta)} muestras generadas en {output_dir}")


if __name__ == "__main__":
    main()
