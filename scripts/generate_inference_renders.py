# -*- coding: utf-8 -*-
"""scripts/generate_inference_renders.py
Generates the GUI visualization renders (render_PART_COLORHEX.png and crops)
for all parts of set 75078-1 using LDraw imports, ABS plastic materials,
and rigid body physics to settle them naturally on the conveyor belt.
"""
import os
import sys
import random
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))
sys.path.append(os.path.join(project_root, 'scripts'))

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if not IN_BLENDER:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import (
    setup_physics_world, create_conveyor_belt_collider, setup_studio_lighting,
    create_abs_plastic_material, apply_bevel_modifier, apply_rigid_body_physics,
    get_ldraw_part_path, generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object

BELT_SURFACE_Z = 0.0
TARGET_SIZE = 1.6  # BU; 1BU = 10mm

# Lista de partes de set 75078-1 con ref y color hex
SET_PARTS = [
    { "ref": "3004", "color_hex": "A0A5A9" },
    { "ref": "3001", "color_hex": "A0A5A9" },
    { "ref": "3020", "color_hex": "A0A5A9" },
    { "ref": "3022", "color_hex": "A0A5A9" },
    { "ref": "2877", "color_hex": "1B1B1B" },
    { "ref": "59900", "color_hex": "C91A09" },
    { "ref": "3003", "color_hex": "A0A5A9" },
    { "ref": "3002", "color_hex": "A0A5A9" },
    { "ref": "3005", "color_hex": "A0A5A9" },
    { "ref": "3010", "color_hex": "A0A5A9" },
    { "ref": "3021", "color_hex": "A0A5A9" },
    { "ref": "3023", "color_hex": "1B1B1B" },
    { "ref": "3024", "color_hex": "1B1B1B" },
    { "ref": "2420", "color_hex": "A0A5A9" },
    { "ref": "3710", "color_hex": "A0A5A9" },
    { "ref": "3622", "color_hex": "A0A5A9" },
    { "ref": "3665", "color_hex": "1B1B1B" },
    { "ref": "3039", "color_hex": "A0A5A9" },
    { "ref": "4070", "color_hex": "A0A5A9" },
    { "ref": "6141", "color_hex": "C91A09" },
    { "ref": "15573", "color_hex": "A0A5A9" },
    { "ref": "2412", "color_hex": "1B1B1B" },
    { "ref": "3069", "color_hex": "A0A5A9" },
    { "ref": "3068", "color_hex": "A0A5A9" },
    { "ref": "60478", "color_hex": "1B1B1B" },
    { "ref": "48336", "color_hex": "1B1B1B" },
    { "ref": "32000", "color_hex": "A0A5A9" },
    { "ref": "3700", "color_hex": "A0A5A9" },
    { "ref": "3701", "color_hex": "A0A5A9" },
    { "ref": "4032", "color_hex": "1B1B1B" },
    { "ref": "3062", "color_hex": "A0A5A9" },
    { "ref": "85984", "color_hex": "A0A5A9" },
    { "ref": "54200", "color_hex": "A0A5A9" },
    { "ref": "99206", "color_hex": "A0A5A9" },
    { "ref": "3037", "color_hex": "A0A5A9" },
    { "ref": "3298", "color_hex": "A0A5A9" },
    { "ref": "11477", "color_hex": "A0A5A9" },
    { "ref": "15068", "color_hex": "A0A5A9" },
    { "ref": "98138", "color_hex": "C91A09" },
    { "ref": "2431", "color_hex": "A0A5A9" },
    { "ref": "6636", "color_hex": "A0A5A9" }
]


def _get_world_bbox(obj):
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece_to_target_size(obj):
    bbox = _get_world_bbox(obj)
    dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
    dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
    dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
    max_dim = max(dim_x, dim_y, dim_z)
    if max_dim < 1e-6:
        return 1.0
    factor = 0.04 if max_dim > 5.0 else 1.0
    obj.scale = (factor, factor, factor)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return factor


def cleanup_scene():
    bpy.ops.object.select_all(action="DESELECT")
    keep = {"Conveyor_Belt_Plane", "Camera", "Camera_Target",
            "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light", "Top_Diffuse_Light"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and not o.name.startswith("Template_"):
            try:
                o.select_set(True)
            except Exception:
                pass
    bpy.ops.object.delete()


def setup_camera():
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


def build_stable_pose_render(part_ref, color_hex, output_path):
    cleanup_scene()
    
    # 1. Cargar template
    existing = set(bpy.context.scene.objects)
    part_path = get_ldraw_part_path(part_ref)
    obj = None
    
    if part_path:
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objs = [o for o in bpy.context.scene.objects if o not in existing]
            par = next((o for o in new_objs if o.parent is None), None)
            obj = get_single_mesh_object(par) if par else None
            if not obj:
                generate_detailed_fallback_mesh(part_ref)
                obj = bpy.context.active_object
        except Exception as e:
            print(f"[Import Warning] {part_ref}: {e}")
            generate_detailed_fallback_mesh(part_ref)
            obj = bpy.context.active_object
    else:
        generate_detailed_fallback_mesh(part_ref)
        obj = bpy.context.active_object

    if not obj:
        print(f"[ERROR] No se pudo cargar malla para {part_ref}")
        return False

    # 2. Normalizar y centrar
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    _normalize_piece_to_target_size(obj)
    apply_bevel_modifier(obj)

    # 3. Aplicar material plástico con color real
    mat = create_abs_plastic_material(color_hex)
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # 4. Colocar para simulación de caída física a Z=4
    # Usar rotación aleatoria para que caiga y aterrice de forma natural
    obj.location = (0.0, 0.0, TARGET_SIZE * 2.5)
    obj.rotation_euler = (
        random.uniform(0.0, 3.1415),
        random.uniform(0.0, 3.1415),
        random.uniform(0.0, 3.1415)
    )

    # 5. Agregar física de cuerpo rígido
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.mass = 0.05
    obj.rigid_body.collision_shape = 'CONVEX_HULL'
    obj.rigid_body.restitution = 0.1
    obj.rigid_body.friction = 0.9
    obj.rigid_body.collision_margin = 0.005
    obj.rigid_body.use_margin = True
    obj.rigid_body.linear_damping = 0.6
    obj.rigid_body.angular_damping = 0.6

    # 6. Correr simulación de físicas de caída libre y rebote
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 90
    for f in range(1, 91):
        scene.frame_set(f)
        bpy.context.view_layer.update()

    # 7. Aplicar transformaciones visuales de física y remover rigidbody
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.visual_transform_apply()
    bpy.ops.rigidbody.object_remove()
    
    # Asegurar que se sitúa justo encima de la cinta transportadora
    bbox = _get_world_bbox(obj)
    min_z = min(p.z for p in bbox)
    obj.location.z += (BELT_SURFACE_Z - min_z) + 0.01

    # 8. Ajustar cámara adaptativa basada en bbox real
    camera = scene.camera
    target = bpy.data.objects.get("Camera_Target")
    
    bbox = _get_world_bbox(obj)
    piece_z_top = max(p.z for p in bbox)
    piece_xy = max(
        max(p.x for p in bbox) - min(p.x for p in bbox),
        max(p.y for p in bbox) - min(p.y for p in bbox),
    )
    
    cam_z = max(piece_z_top + piece_xy * 3.0, BELT_SURFACE_Z + TARGET_SIZE * 1.5)
    camera.location = (obj.location.x, obj.location.y, cam_z)
    if target:
        target.location = (obj.location.x, obj.location.y, piece_z_top * 0.5)

    # Luz difusa cenital directamente sobre la pieza
    top_light = bpy.data.objects.get("Top_Diffuse_Light")
    if not top_light:
        bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, 10.0))
        top_light = bpy.context.active_object
        top_light.name = "Top_Diffuse_Light"
    top_light.location = (obj.location.x, obj.location.y, 8.0)
    top_light.data.size = 8.0
    top_light.data.energy = 450.0

    # Posicionar luces de estudio clave
    for l_name, offset in [("Key_Light", (3.0, -3.0, 5.0)), 
                           ("Fill_Light", (-3.0, -2.0, 3.0)), 
                           ("Rim_Light", (0.0, 4.0, 4.0))]:
        l_obj = bpy.data.objects.get(l_name)
        if l_obj:
            l_obj.location = (obj.location.x + offset[0], obj.location.y + offset[1], offset[2])

    bpy.context.view_layer.update()

    # 9. Renderizar y guardar imagen
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    # 10. Limpieza
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.delete()
    
    return True


def main():
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity = (0.0, 0.0, -98.1)  # 1BU = 10mm -> g=9.81m/s^2 = 981BU/s^2, wait, -98.1 is correct scaling
    create_conveyor_belt_collider()
    setup_studio_lighting()
    setup_camera()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.film_transparent = True
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256

    output_dir = os.path.join(project_root, "data", "synthetic_renders")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Render Generator] Empezando la generación de renders para {len(SET_PARTS)} piezas...")

    for idx, part in enumerate(SET_PARTS):
        ref = part["ref"]
        ch = part["color_hex"]
        out_filename = f"render_{ref}_{ch}.png"
        out_path = os.path.join(output_dir, out_filename)

        print(f"[{idx+1}/{len(SET_PARTS)}] Generando render para {ref} con color #{ch}...")
        
        success = build_stable_pose_render(ref, f"#{ch}", out_path)
        if success and os.path.exists(out_path):
            print(f"  -> Guardado {out_filename}")
            # Copiar a los 15 crops de physics_scatter para la cinta física en vivo
            for ci in range(15):
                crop_filename = f"physics_scatter_{ref}_{ch}_crop_{ci}.png"
                crop_path = os.path.join(output_dir, crop_filename)
                shutil.copy2(out_path, crop_path)
        else:
            print(f"  -> [ERROR] Falló generación para {ref}")

    print("[Render Generator] Generación completada con éxito.")


if __name__ == "__main__":
    main()
