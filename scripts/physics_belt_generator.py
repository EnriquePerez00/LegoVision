# -*- coding: utf-8 -*-
"""
scripts/physics_belt_generator.py
"""

import os
import sys
import random
import math

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scratch"))

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world,
        create_conveyor_belt_collider,
        setup_studio_lighting,
        create_abs_plastic_material,
        apply_bevel_modifier,
        apply_rigid_body_physics,
        get_ldraw_part_path,
        generate_detailed_fallback_mesh,
        enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
else:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")

# FIX #3: superficie del belt esta en Z=0.0 (verificado con vertices world).
# La camara de primer plano se posiciona relativa a esta constante.
BELT_SURFACE_Z = 0.0


def cleanup_pieces():
    if not IN_BLENDER:
        return
    bpy.ops.object.select_all(action='DESELECT')
    keep_names = {"Conveyor_Belt_Plane", "Camera", "Camera_Target",
                  "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep_names and not o.name.startswith("Template_"):
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
        bpy.ops.object.camera_add(location=(0, 0, 37.0))
        camera = bpy.context.active_object
        camera.name = "Camera"
    camera.location = (0.0, 0.0, 37.0)
    camera.data.sensor_width = 36.0
    camera.data.lens = 50.0
    if "Camera_Target" not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        target = bpy.context.active_object
        target.name = "Camera_Target"
    else:
        target = bpy.data.objects["Camera_Target"]
    target.location = (0, 0, 0)
    camera.constraints.clear()
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    bpy.context.scene.camera = camera
    return camera


def create_thick_belt_collider():
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -10.0))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (50.0, 50.0, 20.0)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    belt.rigid_body.type = 'PASSIVE'
    belt.rigid_body.collision_shape = 'BOX'
    belt.rigid_body.friction = 0.8
    belt.rigid_body.restitution = 0.05
    belt.rigid_body.use_margin = True
    belt.rigid_body.collision_margin = 0.0
    mat_name = "Light_Petrol_Blue_Belt"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        principled = nodes.get("Principled BSDF")
        if principled:
            principled.inputs['Base Color'].default_value = (0.145, 0.255, 0.33, 1.0)
            principled.inputs['Roughness'].default_value = 0.5
    belt.data.materials.clear()
    belt.data.materials.append(mat)
    return belt


def build_physics_scatter(part_ref, color_hex, output_path):
    print(f"\n--- Iniciando Simulacion de Fisica ({part_ref}, {color_hex}) ---")

    enable_metal_gpu_acceleration()
    setup_physics_world()
    if IN_BLENDER:
        bpy.context.scene.gravity = (0.0, 0.0, -981.0)

    belt = create_thick_belt_collider()
    if IN_BLENDER:
        belt.rigid_body.friction = 0.9

    setup_studio_lighting()
    setup_camera()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
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
            parent_obj = None
            for o in new_objects:
                if o.parent is None:
                    parent_obj = o
                    break
            if parent_obj:
                template_obj = get_single_mesh_object(parent_obj)
            else:
                generate_detailed_fallback_mesh(part_ref)
                template_obj = bpy.context.active_object
        except Exception as e:
            print(f"Error importando {part_ref}: {e}. Usando fallback...")
            generate_detailed_fallback_mesh(part_ref)
            template_obj = bpy.context.active_object
    else:
        generate_detailed_fallback_mesh(part_ref)
        template_obj = bpy.context.active_object

    if not template_obj:
        print(f"Error critico: no se pudo cargar la pieza {part_ref}")
        return False

    template_obj.name = f"Template_{part_ref}"

    if IN_BLENDER:
        bpy.ops.object.select_all(action='DESELECT')
        template_obj.select_set(True)
        bpy.context.view_layer.objects.active = template_obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    apply_bevel_modifier(template_obj)
    mat = create_abs_plastic_material(color_hex)
    template_obj.data.materials.clear()
    template_obj.data.materials.append(mat)
    template_obj.hide_viewport = True
    template_obj.hide_render = True

    num_pieces = 15
    grid_coords = []
    for x in [-8.0, -2.5, 2.5, 8.0]:
        for y in [-6.0, -2.0, 2.0, 6.0]:
            grid_coords.append((x, y))
    random.shuffle(grid_coords)

    active_col = bpy.context.scene.collection
    pieces = []

    for i in range(num_pieces):
        gx, gy = grid_coords[i]
        obj_copy = template_obj.copy()
        obj_copy.data = template_obj.data.copy()
        active_col.objects.link(obj_copy)
        obj_copy.name = f"Lego_Scatter_{part_ref}_{i}"
        obj_copy.location = (
            gx + random.uniform(-0.3, 0.3),
            gy + random.uniform(-0.3, 0.3),
            5.0 + random.uniform(-0.1, 0.1)
        )
        obj_copy.rotation_euler = (
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2)
        )
        obj_copy.hide_viewport = False
        obj_copy.hide_render = False
        apply_rigid_body_physics(obj_copy, mass=0.008)
        obj_copy.rigid_body.restitution = 0.01
        obj_copy.rigid_body.friction = 0.9
        obj_copy.rigid_body.use_margin = True
        obj_copy.rigid_body.collision_margin = 0.0
        pieces.append(obj_copy)

    # FIX #1: NO llamar frame_set(1) antes del loop.
    # Llamarlo antes y luego repetirlo en range(1,101) resetea el point_cache
    # del rigid body world, congelando piezas en Z~5 en lugar de simularlas.
    # Patron correcto: iterar directamente. Verificado: pieza cae en 4 frames.
    scene.frame_start = 1
    scene.frame_end = 100
    print("Simulando caida y colisiones fisicas...")
    for f in range(1, 101):
        scene.frame_set(f)
        bpy.context.view_layer.update()

    # FIX #2: obj.location NO se actualiza en background (-b), pero
    # obj.matrix_world SI refleja la posicion fisica real.
    # visual_transform_apply() copia matrix_world -> location.
    import mathutils
    for obj in pieces:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.visual_transform_apply()
        bpy.ops.rigidbody.object_remove()

    bpy.data.objects.remove(template_obj)
    scene.frame_set(100)
    bpy.context.view_layer.update()

    camera = scene.camera
    target = bpy.data.objects.get("Camera_Target")
    clean_color = color_hex.replace("#", "")

    for i, obj in enumerate(pieces):
        print(f"Renderizando pieza {i+1}/15...")
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        size_x = max(p.x for p in bbox) - min(p.x for p in bbox)
        size_y = max(p.y for p in bbox) - min(p.y for p in bbox)
        max_size = max(size_x, size_y)
        camera.data.lens = max(15.0, min(300.0, 324.0 / max(0.1, max_size)))

        # FIX #3: camara a 10cm sobre BELT_SURFACE_Z (Z=0), NO sobre obj.location.z.
        # Antes: cam_z = obj.location.z + 10.0  (INCORRECTO: origen != suelo)
        # Ahora: cam_z = BELT_SURFACE_Z + 10.0  (CORRECTO: siempre 10cm del suelo)
        cam_z = BELT_SURFACE_Z + 10.0
        camera.location = (obj.location.x, obj.location.y, cam_z)
        if target:
            target.location = obj.location
        bpy.context.view_layer.update()

        crop_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png"
        crop_path = os.path.join(os.path.dirname(output_path), crop_filename)
        scene.render.filepath = crop_path
        bpy.ops.render.render(write_still=True)

    print("Renderizado de 15 vistas completado con exito.")
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
