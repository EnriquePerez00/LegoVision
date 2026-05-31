import bpy, os, math, sys

sys.path.insert(0, '/Users/I764690/Code_personal/LegoVision')

OUTPUT_DIR = '/Users/I764690/Code_personal/LegoVision/data/synthetic_renders'
LDRAW_BASE = '/Applications/Studio 2.0/ldraw/parts'
RENDER_SIZE = 512

def hex_to_rgba(hx):
    hx = hx.lstrip('#')
    return (int(hx[0:2],16)/255.0, int(hx[2:4],16)/255.0, int(hx[4:6],16)/255.0, 1.0)

def setup_render():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = RENDER_SIZE
    scene.render.resolution_y = RENDER_SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.active_object
    cam.name = 'RenderCam'
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.0
    cam.rotation_euler = (0, 0, 0)
    scene.camera = cam

    # Lights
    bpy.ops.object.light_add(type='SUN', location=(3, 3, 8))
    sun = bpy.context.active_object
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))

    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
    fill = bpy.context.active_object
    fill.data.energy = 2.0
    fill.data.size = 4

    return cam

def render_one(part_file, color_hex, out_path):
    dat_path = os.path.join(LDRAW_BASE, part_file)
    if not os.path.exists(dat_path):
        print(f'  MISS .dat: {dat_path}')
        return False

    # Remove previous part objects
    for o in list(bpy.context.scene.objects):
        if o.type not in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(o, do_unlink=True)

    # Import
    bpy.ops.import_scene.importldr(filepath=dat_path)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        print(f'  No meshes after import: {part_file}')
        return False

    # Override material with correct color
    rgba = hex_to_rgba(color_hex)
    mat = bpy.data.materials.new(name='PartColor')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = 0.2
    nt.links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    # Auto-fit camera orthographic scale
    bpy.context.view_layer.update()
    all_verts = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_verts.append(obj.matrix_world @ v.co)
    if all_verts:
        xs = [v.x for v in all_verts]
        ys = [v.y for v in all_verts]
        zs = [v.z for v in all_verts]
        cx = (max(xs)+min(xs))/2
        cy = (max(ys)+min(ys))/2
        cz = (max(zs)+min(zs))/2
        span = max(max(xs)-min(xs), max(ys)-min(ys)) * 1.3
        cam = bpy.context.scene.camera
        cam.location = (cx, cy, cz + 5)
        cam.data.ortho_scale = max(span, 0.5)

    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f'  DONE: {os.path.basename(out_path)}')
    return True

from scripts.assemble_minifig import MINIFIG_DATABASE
os.makedirs(OUTPUT_DIR, exist_ok=True)
setup_render()
seen = {}
for config in MINIFIG_DATABASE.values():
    for comp in config['components']:
        key = (comp['part_file'], comp['color_hex'])
        if key not in seen:
            seen[key] = comp

total = len(seen)
ok = skip = 0
for i, ((pf, ch), comp) in enumerate(seen.items()):
    ref = pf.replace('.dat','')
    clean = ch.lstrip('#').upper()
    out = os.path.join(OUTPUT_DIR, f'render_{ref}_{clean}.png')
    if os.path.exists(out):
        print(f'  [{i+1}/{total}] SKIP {os.path.basename(out)}')
        skip += 1
        continue
    print(f'  [{i+1}/{total}] Rendering {ref} {ch} ({comp["color_name"]})...')
    if render_one(pf, ch, out):
        ok += 1

print(f'Done: {ok} rendered, {skip} skipped out of {total}')
