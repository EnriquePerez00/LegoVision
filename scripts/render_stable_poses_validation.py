# -*- coding: utf-8 -*-
# scripts/render_stable_poses_validation.py
# Renderiza las posiciones estables encontradas en la simulacion experimental
# para cada pieza del set. Directorio de salida: data/validation_renders/
# SEPARADO de ref_multiangle/ y demas dirs de Blender para no interferir.
# Uso: blender -b -P scripts/render_stable_poses_validation.py -- --set_id 75078-1
import os
import sys
import json
import math
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
sys.path.append(os.path.join(project_root, "scratch"))

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# Directorio EXCLUSIVO para renders de validacion - no interfiere con otros pipelines
VALIDATION_RENDERS_DIR = os.path.join(project_root, "data", "validation_renders")


def _face_label(face_id):
    return {0: "Top", 1: "Side", 2: "Bottom"}.get(face_id, "Face" + str(face_id))


def setup_validation_scene():
    if not IN_BLENDER:
        return
    from generate_synthetic_set import (
        setup_physics_world, setup_studio_lighting,
        create_abs_plastic_material, apply_bevel_modifier,
        enable_metal_gpu_acceleration,
    )
    from scene_config import (
        CAMERA_ORTHO_SCALE, CAMERA_Z, RENDER_RES_SQUARE,
        BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU,
        BELT_COLOR_LINEAR, TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z,
        WORLD_BG_COLOR, WORLD_BG_STRENGTH,
    )
    enable_metal_gpu_acceleration()
    setup_physics_world()
    setup_studio_lighting()

    # Camara ortografica cenital (mismos parametros que ref_multiangle)
    cam_name = "Camera"
    if cam_name in bpy.data.objects:
        cam = bpy.data.objects[cam_name]
    else:
        bpy.ops.object.camera_add(location=(0, 0, CAMERA_Z))
        cam = bpy.context.active_object
        cam.name = cam_name
    cam.location = (0.0, 0.0, CAMERA_Z)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = CAMERA_ORTHO_SCALE
    bpy.context.scene.camera = cam

    # Plano de fondo con nombre unico para no interferir con Conveyor_Belt_Plane
    belt_name = "Validation_Belt_Plane"
    if belt_name not in bpy.data.objects:
        half_t = BELT_THICKNESS_BU * 0.5
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_t))
        belt = bpy.context.active_object
        belt.name = belt_name
        belt.scale = (BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU)
        bpy.ops.object.transform_apply(scale=True)
        mat = bpy.data.materials.get("Belt_Mat_Val")
        if not mat:
            mat = bpy.data.materials.new("Belt_Mat_Val")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = BELT_COLOR_LINEAR
                bsdf.inputs["Roughness"].default_value = 0.5
        belt.data.materials.clear()
        belt.data.materials.append(mat)

    # Luz cenital
    light_name = "Val_Top_Light"
    top = bpy.data.objects.get(light_name)
    if not top:
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, TOP_LIGHT_Z))
        top = bpy.context.active_object
        top.name = light_name
    top.location = (0.0, 0.0, TOP_LIGHT_Z)
    top.data.size = TOP_LIGHT_SIZE
    top.data.energy = TOP_LIGHT_ENERGY

    scene = bpy.context.scene
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = WORLD_BG_COLOR
            bg.inputs["Strength"].default_value = WORLD_BG_STRENGTH

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_RES_SQUARE
    scene.render.resolution_y = RENDER_RES_SQUARE


def cleanup_piece_objects():
    if not IN_BLENDER:
        return
    keep = {"Camera", "Validation_Belt_Plane", "Val_Top_Light",
            "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light", "Top_Diffuse_Light"}
    bpy.ops.object.select_all(action="DESELECT")
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def local_up_to_rotation(local_up_vec):
    # Calcula la rotacion Euler necesaria para que el vector local_up
    # quede alineado con Z+ global (orientacion de reposo simulada)
    up = mathutils.Vector(local_up_vec).normalized()
    z_global = mathutils.Vector((0.0, 0.0, 1.0))
    axis = up.cross(z_global)
    if axis.length < 1e-6:
        if up.dot(z_global) > 0:
            return mathutils.Euler((0, 0, 0), "XYZ")
        else:
            return mathutils.Euler((math.pi, 0, 0), "XYZ")
    axis.normalize()
    angle = up.angle(z_global)
    rot_mat = mathutils.Matrix.Rotation(angle, 4, axis)
    return rot_mat.to_euler("XYZ")


def render_validation_poses(set_id, output_dir, validation_json_path):
    if not IN_BLENDER:
        print("[ERROR] Este script debe ejecutarse en Blender")
        return

    from generate_synthetic_set import (
        get_ldraw_part_path, generate_detailed_fallback_mesh,
        create_abs_plastic_material, apply_bevel_modifier,
    )
    from generate_synthetic_dataset import get_single_mesh_object
    from scene_config import LDRAW_TO_BU, LDRAW_THRESHOLD

    os.makedirs(output_dir, exist_ok=True)
    setup_validation_scene()

    if not os.path.exists(validation_json_path):
        print("[ERROR] No se encontro el JSON de validacion: " + validation_json_path)
        return

    with open(validation_json_path, "r", encoding="utf-8") as fh:
        val_data = json.load(fh)

    report = val_data.get("report", [])
    total_pieces = len(report)
    print("[RenderValidation] Procesando " + str(total_pieces) + " piezas del set " + set_id + "...")

    sys.path.insert(0, os.path.join(project_root, "database"))
    from set_catalog import REAL_SETS
    set_data = REAL_SETS.get(set_id, {})
    color_map = {}
    for part_info in set_data.get("parts", []):
        color_map[part_info["ref"]] = part_info.get("color_hex", "#A0A5A9")

    scene = bpy.context.scene
    rendered_count = 0

    for idx, item in enumerate(report):
        ref = item["part_ref"]
        poses = item.get("poses", [])
        color_hex = color_map.get(ref, "#A0A5A9")
        print("\n[" + str(idx + 1) + "/" + str(total_pieces) + "] Pieza " + ref +
              " (" + str(len(poses)) + " poses simuladas)...")

        for pose_idx, pose in enumerate(poses):
            face_id = pose["face"]
            local_up = pose["local_up"]
            out_filename = ("validation_" + ref + "_pose" + str(pose_idx) +
                            "_face" + str(face_id) + ".png")
            out_path = os.path.join(output_dir, out_filename)

            if os.path.exists(out_path):
                print("  Pose " + str(pose_idx) + " ya existe, saltando.")
                rendered_count += 1
                continue

            cleanup_piece_objects()

            part_path = get_ldraw_part_path(ref)
            existing = set(bpy.context.scene.objects)
            obj = None

            if part_path:
                try:
                    bpy.ops.import_scene.importldr(filepath=part_path)
                    new_objs = [o for o in bpy.context.scene.objects if o not in existing]
                    par = next((o for o in new_objs if o.parent is None), None)
                    if par:
                        obj = get_single_mesh_object(par)
                    if not obj:
                        generate_detailed_fallback_mesh(ref)
                        obj = bpy.context.active_object
                except Exception as exc:
                    print("  Error importando " + ref + ": " + str(exc))
                    generate_detailed_fallback_mesh(ref)
                    obj = bpy.context.active_object
            else:
                generate_detailed_fallback_mesh(ref)
                obj = bpy.context.active_object

            if not obj:
                print("  No se pudo cargar " + ref)
                continue

            # Normalizar escala
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            dims = [max(pt.x for pt in bbox) - min(pt.x for pt in bbox),
                    max(pt.y for pt in bbox) - min(pt.y for pt in bbox),
                    max(pt.z for pt in bbox) - min(pt.z for pt in bbox)]
            max_dim = max(dims)
            factor = LDRAW_TO_BU if max_dim > LDRAW_THRESHOLD else 1.0
            obj.scale = (factor, factor, factor)
            bpy.ops.object.transform_apply(scale=True)

            # Material y bisel
            apply_bevel_modifier(obj)
            mat_piece = create_abs_plastic_material(color_hex)
            obj.data.materials.clear()
            obj.data.materials.append(mat_piece)

            # Orientar segun la posicion de reposo simulada
            euler = local_up_to_rotation(local_up)
            obj.rotation_euler = euler
            bpy.ops.object.transform_apply(rotation=True)

            # Posicionar sobre el plano
            obj.location = (0.0, 0.0, 0.0)
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            bbox2 = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            min_z = min(pt.z for pt in bbox2)
            obj.location.z = -min_z + 0.02

            # Renderizar
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            rendered_count += 1
            print("  Pose " + str(pose_idx) + " (" + _face_label(face_id) + ") -> " + out_filename)

    print("\n[RenderValidation] Completado: " + str(rendered_count) + " renders en " + output_dir)


def main():
    if not IN_BLENDER:
        print("[ERROR] Ejecutar con: blender -b -P scripts/render_stable_poses_validation.py")
        return
    args_raw = []
    if "--" in sys.argv:
        args_raw = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(project_root, "data", "validation_renders"))
    parser.add_argument("--validation_json", type=str,
                        default=os.path.join(project_root, "data", "tmp",
                                             "stability_validation_results.json"))
    parsed = parser.parse_known_args(args_raw)[0]
    render_validation_poses(parsed.set_id, parsed.output_dir, parsed.validation_json)


if __name__ == "__main__":
    main()
