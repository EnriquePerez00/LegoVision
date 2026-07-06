import bpy
import sys
import mathutils

def check_scene():
    import scene_canonical
    scene = scene_canonical.build_scene_canonical()
    
    # Load the 5 parts
    parts = ["48170", "30350b", "60484", "2489", "32000"]
    for p in parts:
        obj = scene_canonical.import_part(p)
        if obj:
            bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            xs = [v.x for v in bbox]
            ys = [v.y for v in bbox]
            zs = [v.z for v in bbox]
            size = (max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
            print(f"Part {p}: size = {size}")
        else:
            print(f"Part {p}: Failed to load")

check_scene()
