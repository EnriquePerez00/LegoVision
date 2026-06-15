import os
import sys
import json
import numpy as np

try:
    import bpy
    import mathutils
except ImportError:
    print("This script must be run inside Blender: blender -b -P scripts/generate_color_calibration.py")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import scene_canonical
import generate_synthetic_set
def main():

    db_path = os.path.join(current_dir, "..", "..", "database", "color_catalog.json")
    with open(db_path, "r") as f:
        catalog = json.load(f)
        
    out_dir = os.path.join(current_dir, "..", "data", "color_calibration")
    os.makedirs(out_dir, exist_ok=True)
    
    scene_canonical.cleanup_piece_objects()
    cam_cen, cam_lat = scene_canonical.build_scene_canonical(render_res=256, film_transparent=True)
    
    # Delete Belt and Screen to ensure transparent background
    # Delete Belt, Screen, and Floor to ensure transparent background and no color contamination
    for name in ["Conveyor_Belt_Plane", "Side_Screen_AL", "Office_Floor"]:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
        
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0.05))
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    
    calibration_data = []
    
    for code, info in catalog.items():
        hex_color = info.get("hex", "#000000")
        name = info.get("name", "Unknown")
        
        mat = generate_synthetic_set.create_abs_plastic_material(code)
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
            
        print(f"Rendering color {code}: {name} ({hex_color})")
        
        cen_path = os.path.join(out_dir, f"calib_{code}_cen.png")
        lat_path = os.path.join(out_dir, f"calib_{code}_lat.png")
        
        bpy.context.scene.camera = cam_cen
        bpy.context.scene.render.filepath = cen_path
        bpy.ops.render.render(write_still=True)
        
        bpy.context.scene.camera = cam_lat
        bpy.context.scene.render.filepath = lat_path
        bpy.ops.render.render(write_still=True)
        
    print(f"Blender render completed! Now run extract_color_calibration.py")

if __name__ == "__main__":
    main()
