# -*- coding: utf-8 -*-
"""scripts/physics_belt_generator.py
FIX-A: Normalizar pieza a TARGET_SIZE BU (escala LDraw independiente).
FIX-B: spawn_z garantiza que la pieza no penetre el suelo en frame 1.
FIX-C: camara adaptativa cam_z = piece_z_top + piece_xy * 3.
FIX-D: Cache hit - si 15 crops ya existen, no re-simular.
FIX-E: Grid proporcional a TARGET_SIZE.
FIX-F: g=981 BU/s^2 con escala 1BU=10mm.
"""
import os, sys, random, math

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world, create_conveyor_belt_collider, setup_studio_lighting,
        create_abs_plastic_material, apply_bevel_modifier, apply_rigid_body_physics,
        get_ldraw_part_path, generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
else:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")

BELT_SURFACE_Z = 0.0
TARGET_SIZE = 1.6  # BU; Brick 1x2=16mm, 1BU=10mm -> 1.6 BU


def _get_world_bbox(obj):
    import mathutils
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece_to_target_size(obj):
    """FIX-A: escala obj a TARGET_SIZE independientemente del addon de importacion."""
    bbox = _get_world_bbox(obj)
    dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
    dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
    dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
    max_dim = max(dim_x, dim_y, dim_z)
    if max_dim < 1e-6:
        print(f"[WARN] {obj.name} dim~0, no se escala.")
        return 1.0
    factor = TARGET_SIZE / max_dim
    obj.scale = (factor, factor, factor)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"[Scale] max_dim={max_dim:.4f} factor={factor:.6f} TARGET={TARGET_SIZE}")
    return factor


def _crops_already_exist(output_dir, part_ref, clean_color, num=15):
    """FIX-D: True si todos los crops ya estan en disco."""
    for i in range(num):
        p = os.path.join(output_dir, f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png")
        if not os.path.exists(p):
            return False
    return True


def cleanup_pieces():
    if not IN_BLENDER:
        return
    bpy.ops.object.select_all(action="DESELECT")
    keep = {"Conveyor_Belt_Plane", "Camera", "Camera_Target",
            "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and not o.name.startswith("Template_"):
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def setup_camera():
    if not IN_BLENDER:
        return None
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add(location=(0, 0, TARGET_SIZE * 20))
        camera = bpy.context.active_object
        camera.name = "Camera"
    camera.location = (0.0, 0.0, TARGET_SIZE * 20)
    camera.data.sensor_width = 36.0
    camera.data.lens = 50.0
    if "Camera_Target" not in bpy.data.objects:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        target = bpy.context.active_object
        target.name = "Camera_Target"
    else:
        target = bpy.data.objects["Camera_Target"]
    target.location = (0, 0, 0)
    camera.constraints.clear()
    c = camera.constraints.new(type="TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    bpy.context.scene.camera = camera
    return camera


def create_thick_belt_collider():
    """Belt collider escalado a TARGET_SIZE. Cara superior en BELT_SURFACE_Z=0."""
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    half_thick = TARGET_SIZE * 5.0
    belt_extent = TARGET_SIZE * 30.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_thick))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (belt_extent, belt_extent, half_thick)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type = "PASSIVE"
    belt.rigid_body.collision_shape = "BOX"
    belt.rigid_body.friction = 0.9
    belt.rigid_body.restitution = 0.02
    belt.rigid_body.use_margin = True
    belt.rigid_body.collision_margin = 0.0
    mat = bpy.data.materials.get("Light_Petrol_Blue_Belt")
    if not mat:
        mat = bpy.data.materials.new(name="Light_Petrol_Blue_Belt")
        mat.use_nodes = True
        p = mat.node_tree.nodes.get("Principled BSDF")
        if p:
            p.inputs["Base Color"].default_value = (0.145, 0.255, 0.33, 1.0)
            p.inputs["Roughness"].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)
    return belt


def build_physics_scatter(part_ref, color_hex, output_path):
    print(f"--- Simulacion de Fisica ({part_ref}, {color_hex}) ---")
    clean_color = color_hex.replace("#", "")
    output_dir = os.path.dirname(output_path)
    num_pieces = 15

    # FIX-D: cache hit
    if _crops_already_exist(output_dir, part_ref, clean_color, num_pieces):
        print(f"[Cache] {num_pieces} crops ya existen. Saltando simulacion.")
        return True

    if not IN_BLENDER:
        print("[ERROR] Se requiere Blender.")
        return False

    import mathutils
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity = (0.0, 0.0, -981.0)  # FIX-F

    belt = create_thick_belt_collider()
    setup_studio_lighting()
    setup_camera()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.render.film_transparent = False
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    cleanup_pieces()

    part_path = get_ldraw_part_path(part_ref)
    existing_objects = set(bpy.context.scene.objects)
    if part_path:
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objects = [o for o in bpy.context.scene.objects if o not in existing_objects]
            parent_obj = next((o for o in new_objects if o.parent is None), None)
            if parent_obj:
                template_obj = get_single_mesh_object(parent_obj)
            else:
                generate_detailed_fallback_mesh(part_ref)
                template_obj = bpy.context.active_object
        except Exception as e:
            print(f"Error importando {part_ref}: {e}. Fallback...")
            generate_detailed_fallback_mesh(part_ref)
            template_obj = bpy.context.active_object
    else:
        generate_detailed_fallback_mesh(part_ref)
        template_obj = bpy.context.active_object

    if not template_obj:
        print(f"Error critico: no se pudo cargar {part_ref}")
        return False

    template_obj.name = f"Template_{part_ref}"
    bpy.ops.object.select_all(action="DESELECT")
    template_obj.select_set(True)
    bpy.context.view_layer.objects.active = template_obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    _normalize_piece_to_target_size(template_obj)  # FIX-A

    bbox_tmpl = _get_world_bbox(template_obj)
    tmpl_dim_z = max(p.z for p in bbox_tmpl) - min(p.z for p in bbox_tmpl)
    tmpl_max_xy = max(
        max(p.x for p in bbox_tmpl) - min(p.x for p in bbox_tmpl),
        max(p.y for p in bbox_tmpl) - min(p.y for p in bbox_tmpl),
    )
    spawn_z = BELT_SURFACE_Z + tmpl_dim_z * 0.5 + TARGET_SIZE * 3.0  # FIX-B
    print(f"[Spawn] dim_z={tmpl_dim_z:.3f} max_xy={tmpl_max_xy:.3f} spawn_z={spawn_z:.3f}")

    apply_bevel_modifier(template_obj)
    mat = create_abs_plastic_material(color_hex)
    template_obj.data.materials.clear()
    template_obj.data.materials.append(mat)
    template_obj.hide_viewport = True
    template_obj.hide_render = True

    # FIX-E: grid escalado a TARGET_SIZE
    grid_step = TARGET_SIZE * 3.5
    grid_coords = [(xi * grid_step, yi * grid_step)
                   for xi in [-2, -1, 1, 2] for yi in [-2, -1, 1, 2]]
    grid_coords.append((0.0, 0.0))
    random.shuffle(grid_coords)

    active_col = bpy.context.scene.collection
    pieces = []
    jitter = TARGET_SIZE * 0.3

    for i in range(num_pieces):
        gx, gy = grid_coords[i]
        obj_copy = template_obj.copy()
        obj_copy.data = template_obj.data.copy()
        active_col.objects.link(obj_copy)
        obj_copy.name = f"Lego_Scatter_{part_ref}_{i}"
        obj_copy.location = (
            gx + random.uniform(-jitter, jitter),
            gy + random.uniform(-jitter, jitter),
            spawn_z + random.uniform(-jitter * 0.3, jitter * 0.3),
        )
        obj_copy.rotation_euler = (
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
        )
        obj_copy.hide_viewport = False
        obj_copy.hide_render = False
        apply_rigid_body_physics(obj_copy, mass=0.008)
        obj_copy.rigid_body.restitution = 0.01
        obj_copy.rigid_body.friction = 0.9
        obj_copy.rigid_body.use_margin = True
        obj_copy.rigid_body.collision_margin = 0.0
        pieces.append(obj_copy)

    scene.frame_start = 1
    scene.frame_end = 100
    print("Simulando caida y colisiones fisicas...")
    for f in range(1, 101):
        scene.frame_set(f)
        bpy.context.view_layer.update()

    for obj in pieces:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.visual_transform_apply()
        bpy.ops.rigidbody.object_remove()

    bpy.data.objects.remove(template_obj)
    scene.frame_set(100)
    bpy.context.view_layer.update()

    camera = scene.camera
    target = bpy.data.objects.get("Camera_Target")

    for i, obj in enumerate(pieces):
        print(f"Renderizando pieza {i + 1}/15...")

        # FIX-C: camara adaptativa basada en bbox real post-simulacion
        bbox = _get_world_bbox(obj)
        piece_z_top = max(p.z for p in bbox)
        piece_xy = max(
            max(p.x for p in bbox) - min(p.x for p in bbox),
            max(p.y for p in bbox) - min(p.y for p in bbox),
        )
        # Camara a 3x la dimension XY sobre la cara superior de la pieza.
        # Minimo TARGET_SIZE*1.5 para evitar cam por debajo del suelo.
        cam_z = max(piece_z_top + piece_xy * 3.0, BELT_SURFACE_Z + TARGET_SIZE * 1.5)
        # Focal fija 50mm con zoom via distancia (mas predecible que cambiar lens)
        camera.data.lens = 50.0
        camera.location = (obj.location.x, obj.location.y, cam_z)
        if target:
            target.location = (obj.location.x, obj.location.y, piece_z_top * 0.5)
        bpy.context.view_layer.update()

        crop_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        scene.render.filepath = crop_path
        bpy.ops.render.render(write_still=True)

    print("Renderizado de 15 vistas completado.")
    return True


if __name__ == "__main__":
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_ref", type=str, required=True)
    parser.add_argument("--color_hex", type=str, default="#A0A5A9")
    parser.add_argument("--output_path", type=str, required=True)
    parsed_args = parser.parse_known_args(args)[0]
    if IN_BLENDER:
        build_physics_scatter(parsed_args.part_ref, parsed_args.color_hex, parsed_args.output_path)
