# -*- coding: utf-8 -*-
import os
import sys
import bpy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
base_scripts = os.path.join(legovic_root, "2camaras_random_pieza_unica", "scripts")
if base_scripts not in sys.path:
    sys.path.append(base_scripts)

import scene_canonical
from generate_synthetic_set import apply_bevel_modifier, create_abs_plastic_material

def render_test():
    # Build scene
    cam_cen, cam_front = scene_canonical.build_scene_canonical(render_res=1024, film_transparent=False)
    scene = bpy.context.scene
    
    # Import part
    ref = "3003" # 2x2 Brick
    part_obj = scene_canonical.import_part(ref)
    bpy.ops.object.select_all(action='DESELECT')
    part_obj.select_set(True)
    bpy.context.view_layer.objects.active = part_obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    scene_canonical.normalize_piece(part_obj)
    apply_bevel_modifier(part_obj)
    
    # Position at center
    part_obj.location = (0, 0, 0)
    
    # Material
    mat = create_abs_plastic_material("#A2A1A3") # Light Bluish Gray
    part_obj.data.materials.clear()
    part_obj.data.materials.append(mat)
    bpy.context.view_layer.update()
    
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    
    # Render Cenital
    scene.camera = cam_cen
    bpy.context.view_layer.update()
    cen_path = os.path.join(project_root, "data", "test_cenital.png")
    scene.render.filepath = cen_path
    bpy.ops.render.render(write_still=True)
    print(f"Cenital rendered: {cen_path}")
    
    # Render Frontal
    scene.camera = cam_front
    bpy.context.view_layer.update()
    front_path = os.path.join(project_root, "data", "test_frontal.png")
    scene.render.filepath = front_path
    bpy.ops.render.render(write_still=True)
    print(f"Frontal rendered: {front_path}")

if __name__ == "__main__":
    render_test()
