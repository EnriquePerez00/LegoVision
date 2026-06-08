# -*- coding: utf-8 -*-
"""scripts/physics_belt_generator.py
FIX-A: Normalizar pieza a TARGET_SIZE BU (escala LDraw independiente).
FIX-B: spawn_z garantiza que la pieza no penetre el suelo en frame 1.
FIX-C: camara adaptativa cam_z = piece_z_top + piece_xy * 3.
FIX-D: Cache hit - si 15 crops ya existen, no re-simular.
FIX-E: Grid proporcional a TARGET_SIZE.
FIX-F: g=981 BU/s^2 con escala 1BU=10mm.
"""
import os, sys, random, math, time, json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))
sys.path.append(os.path.join(project_root, 'scripts'))

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
    from scene_config import BELT_FRICTION, BELT_RESTITUTION, PIECE_FRICTION, PIECE_RESTITUTION
else:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    BELT_FRICTION, BELT_RESTITUTION, PIECE_FRICTION, PIECE_RESTITUTION = 0.95, 0.02, 0.95, 0.02

BELT_SURFACE_Z = 0.0
TARGET_SIZE = 1.6  # BU; Brick 1x2=16mm, 1BU=10mm -> 1.6 BU


def _get_world_bbox(obj):
    import mathutils
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece_to_target_size(obj):
    """FIX-A: escala obj de forma realista (0.04 para LDraw en LDU, 1.0 para fallbacks en BU)."""
    bbox = _get_world_bbox(obj)
    dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
    dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
    dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
    max_dim = max(dim_x, dim_y, dim_z)
    if max_dim < 1e-6:
        print(f"[WARN] {obj.name} dim~0, no se escala.")
        return 1.0
    
    # Si la dimension maxima es grande (> 5.0), asumimos que esta en LDU (LDraw Units)
    # y la multiplicamos por 0.04 para llevarla a BU (donde 1 BU = 10mm).
    # Si es pequeña, ya esta generada en BU como fallback.
    if max_dim > 5.0:
        factor = 0.04
    else:
        factor = 1.0
        
    obj.scale = (factor, factor, factor)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print(f"[Scale] max_dim={max_dim:.4f} factor={factor:.6f} RealScale applied")
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
            "Sun_Light", "Rim_Light", "Fill_Light", "Key_Light", "Top_Diffuse_Light"}
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
    # FIX: Para que la cara superior este en Z = 0, el origen del cubo (de altura half_thick) debe estar en -half_thick * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_thick * 0.5))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (belt_extent, belt_extent, half_thick)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type = "PASSIVE"
    belt.rigid_body.collision_shape = "BOX"
    belt.rigid_body.friction = BELT_FRICTION
    belt.rigid_body.restitution = BELT_RESTITUTION
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


def build_physics_scatter(part_ref, color_hex, output_path, num_simulations=15):
    print(f"--- Simulacion de Fisica ({part_ref}, {color_hex}, {num_simulations} runs) ---")
    clean_color = color_hex.replace("#", "")
    output_dir = os.path.dirname(output_path)
    stats_path = os.path.join(output_dir, f"physics_scatter_{part_ref}_{clean_color}_stats.json")

    # Lógica de Cache Hit
    if os.path.exists(stats_path) and _crops_already_exist(output_dir, part_ref, clean_color, num_simulations):
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                stats = json.load(sf)
            if stats.get("num_simulations") == num_simulations:
                print(f"[Cache] {num_simulations} crops y stats JSON ya existen. Saltando simulacion.")
                return True
        except Exception:
            pass

    if not IN_BLENDER:
        print("[ERROR] Se requiere Blender.")
        return False

    import mathutils
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity = (0.0, 0.0, -9.81)  # FIX-F

    belt = create_thick_belt_collider()
    setup_studio_lighting()
    
    scene = bpy.context.scene
    # Reducir la fuerza del fondo del mundo a 0.1 para evitar que emita luz desde abajo de la cinta.
    if scene.world:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1.0)
            bg.inputs["Strength"].default_value = 0.1
            
    setup_camera()
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.render.film_transparent = False
    # FIX-RES: resolución cuadrada 640x640 (igual que dataset YOLO)
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
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
    
    apply_bevel_modifier(template_obj)
    mat = create_abs_plastic_material(color_hex)
    template_obj.data.materials.clear()
    template_obj.data.materials.append(mat)
    template_obj.hide_viewport = True
    template_obj.hide_render = True

    # Generar coordenadas de cuadrícula no colisionable dinámicamente
    def generate_grid_coordinates(n, step):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        coords = []
        for r in range(rows):
            for c in range(cols):
                if len(coords) < n:
                    x = (c - (cols - 1) / 2) * step
                    y = (r - (rows - 1) / 2) * step
                    coords.append((x, y))
        return coords

    grid_step = TARGET_SIZE * 4.0
    grid_coords = generate_grid_coordinates(num_simulations, grid_step)
    random.shuffle(grid_coords)

    active_col = bpy.context.scene.collection
    pieces = []
    jitter = TARGET_SIZE * 0.3

    # Medir tiempo de simulación de físicas
    t_phys_start = time.time()

    for i in range(num_simulations):
        gx, gy = grid_coords[i]
        obj_copy = template_obj.copy()
        obj_copy.data = template_obj.data.copy()
        active_col.objects.link(obj_copy)
        obj_copy.name = f"Lego_Scatter_{part_ref}_{i}"
        
        # Posición inicial Z alta temporal
        obj_copy.location = (
            gx + random.uniform(-jitter, jitter),
            gy + random.uniform(-jitter, jitter),
            10.0
        )
        # Rotación aleatoria de 360 grados
        obj_copy.rotation_euler = (
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
        )
        bpy.context.view_layer.update()
        
        # Calcular world bbox del objeto rotado para determinar el vértice inferior
        bbox = _get_world_bbox(obj_copy)
        min_z = min(p.z for p in bbox)
        
        # Ajustar ubicación Z para que el vértice inferior esté a exactamente 5.0 BU (5 cm) de la superficie (Z=0.0)
        obj_copy.location.z = obj_copy.location.z - min_z + 5.0
        
        obj_copy.hide_viewport = False
        obj_copy.hide_render = False
        
        apply_rigid_body_physics(obj_copy, mass=0.008)
        obj_copy.rigid_body.restitution = PIECE_RESTITUTION
        obj_copy.rigid_body.friction = PIECE_FRICTION
        obj_copy.rigid_body.use_margin = True
        obj_copy.rigid_body.collision_margin = 0.0
        pieces.append(obj_copy)

    scene.frame_start = 1
    scene.frame_end = 120  # Aumentado a 120 frames para garantizar estabilidad completa
    print(f"Simulando caída y colisiones físicas para {num_simulations} piezas...")
    for f in range(1, 121):
        scene.frame_set(f)
        bpy.context.view_layer.update()

    for obj in pieces:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.visual_transform_apply()
        bpy.ops.rigidbody.object_remove()

    bpy.data.objects.remove(template_obj)
    scene.frame_set(120)
    bpy.context.view_layer.update()

    t_phys_end = time.time()
    phys_time = t_phys_end - t_phys_start

    # Desactivar sombras en todas las luces del mundo para una iluminación difusa sin sombras
    for light in bpy.data.lights:
        light.use_shadow = False

    # Medir tiempo de renderizado
    t_render_start = time.time()

    camera = scene.camera
    target = bpy.data.objects.get("Camera_Target")

    for i, obj in enumerate(pieces):
        print(f"Renderizando pieza {i + 1}/{num_simulations}...")

        # FIX-C: camara adaptativa basada en bbox real post-simulacion
        bbox = _get_world_bbox(obj)
        piece_z_top = max(p.z for p in bbox)
        piece_xy = max(
            max(p.x for p in bbox) - min(p.x for p in bbox),
            max(p.y for p in bbox) - min(p.y for p in bbox),
        )
        cam_z = max(piece_z_top + piece_xy * 3.0, BELT_SURFACE_Z + TARGET_SIZE * 1.5)
        camera.data.lens = 50.0
        camera.location = (obj.location.x, obj.location.y, cam_z)
        if target:
            target.location = (obj.location.x, obj.location.y, piece_z_top * 0.5)

        # Asegurar la existencia de una luz difusa cenital gigante directamente sobre la pieza
        top_light = bpy.data.objects.get("Top_Diffuse_Light")
        if not top_light:
            bpy.ops.object.light_add(type='AREA', location=(0.0, 0.0, 10.0))
            top_light = bpy.context.active_object
            top_light.name = "Top_Diffuse_Light"
        top_light.location = (obj.location.x, obj.location.y, 8.0)
        top_light.data.size = 8.0
        top_light.data.energy = 450.0

        # Reposicionar luces de estudio relativas a la pieza
        for l_name, offset in [("Key_Light", (3.0, -3.0, 5.0)), 
                               ("Fill_Light", (-3.0, -2.0, 3.0)), 
                               ("Rim_Light", (0.0, 4.0, 4.0))]:
            l_obj = bpy.data.objects.get(l_name)
            if l_obj:
                l_obj.location = (obj.location.x + offset[0], obj.location.y + offset[1], offset[2])
                l_obj.data.size = 4.0
                if l_name == "Key_Light":
                    l_obj.data.energy = 250.0
                elif l_name == "Rim_Light":
                    l_obj.data.energy = 150.0
                elif l_name == "Fill_Light":
                    l_obj.data.energy = 100.0

        bpy.context.view_layer.update()

        crop_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png"
        crop_path = os.path.join(output_dir, crop_filename)
        scene.render.filepath = crop_path
        bpy.ops.render.render(write_still=True)

    t_render_end = time.time()
    render_time = t_render_end - t_render_start

    # Guardar estadísticas de tiempos de ejecución
    stats_data = {
        "total_physics_time": phys_time,
        "total_render_time": render_time,
        "physics_time_per_piece": phys_time / num_simulations,
        "render_time_per_piece": render_time / num_simulations,
        "num_simulations": num_simulations
    }
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats_data, sf, indent=2)
    print(f"Estadísticas de simulación y renderizado guardadas en: {stats_path}")

    print(f"Renderizado de {num_simulations} vistas completado.")
    return True


if __name__ == "__main__":
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_ref", type=str, required=True)
    parser.add_argument("--color_hex", type=str, default="#A0A5A9")
    parser.add_argument("--num_simulations", type=int, default=15)
    parser.add_argument("--output_path", type=str, default=None,
                        help="Ruta completa de salida. Si no se especifica, usa data/dino_scatter/")
    parsed_args = parser.parse_known_args(args)[0]
    # FIX-DIR: output separado del dataset YOLO
    if parsed_args.output_path is None:
        sys.path.insert(0, os.path.join(project_root, "scripts"))
        from scene_config import DINO_SCATTER_SUBDIR
        dino_dir = os.path.join(project_root, "data", DINO_SCATTER_SUBDIR)
        os.makedirs(dino_dir, exist_ok=True)
        out_path = os.path.join(dino_dir,
            f"physics_scatter_{parsed_args.part_ref}_{parsed_args.color_hex.replace('#','')}_crop_0.png")
    else:
        out_path = parsed_args.output_path
    if IN_BLENDER:
        build_physics_scatter(parsed_args.part_ref, parsed_args.color_hex, out_path, parsed_args.num_simulations)
