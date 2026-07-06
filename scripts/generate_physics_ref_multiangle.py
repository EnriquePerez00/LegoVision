# -*- coding: utf-8 -*-
# scripts/generate_physics_ref_multiangle.py
# Genera referencias DINOv2 usando simulacion de fisica real y camara ortografica.
# Cada pieza cae sobre la cinta, estabiliza y se renderiza con la misma camara que YOLO.
# Alineacion total: referencias DINOv2 == dataset YOLO == inferencia real.
# Uso: blender -b -P scripts/generate_physics_ref_multiangle.py
import os, sys, random, math, json

try:
    import bpy, bpy_extras
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scratch"))
sys.path.insert(0, os.path.join(project_root, "scripts"))

from generate_synthetic_set import (
    setup_physics_world, setup_studio_lighting, create_abs_plastic_material,
    apply_bevel_modifier, apply_rigid_body_physics, get_ldraw_part_path,
    generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
)
from generate_synthetic_dataset import get_single_mesh_object
from scene_config import (
    BELT_SURFACE_Z, BELT_WIDTH_BU, BELT_LENGTH_BU, BELT_THICKNESS_BU,
    BELT_COLOR_LINEAR, BELT_FRICTION, BELT_RESTITUTION,
    CAMERA_ORTHO_SCALE, CAMERA_Z, RENDER_RES_SQUARE, CYCLES_SAMPLES_REF,
    GRAVITY_Z, PHYSICS_FRAMES, PIECE_MASS_KG, PIECE_FRICTION, PIECE_RESTITUTION,
    LDRAW_TO_BU, LDRAW_THRESHOLD, DEFAULT_SPAWN_Z,
    TOP_LIGHT_SIZE, TOP_LIGHT_ENERGY, TOP_LIGHT_Z, WORLD_BG_STRENGTH, WORLD_BG_COLOR,
    DINO_CANVAS_SIZE, DINO_CANVAS_MARGIN_PX, DINO_BG_COLOR,
)

def _load_all_part_colors_for_dino(set_id=None):
    """Carga pares (ref, color_hex) unicos de todos los sets o uno especifico para DINOv2.
    Para DINOv2 si importa el color - cada (ref,color) es una entrada diferente.
    Excluye stickers y piezas con decoraciones de impresion.
    """
    sys.path.insert(0, os.path.join(project_root, "database"))
    try:
        from set_catalog import REAL_SETS
    except ImportError:
        return []
    seen = {}
    sets_to_process = {set_id: REAL_SETS[set_id]} if (set_id and set_id in REAL_SETS) else REAL_SETS
    for sid, set_data in sets_to_process.items():
        for part in set_data.get("parts", []):
            ref = part["ref"]
            color = part.get("color_hex", "#A0A5A9")
            # Excluir stickers, prints y refs muy largos
            if "stk" not in ref.lower() and "pb" not in ref.lower() and len(ref) < 15:
                key = (ref, color)
                if key not in seen:
                    seen[key] = {"ref": ref, "color": color}
    parts_list = list(seen.values())
    print(f"[DINOv2 Refs] {len(parts_list)} pares (ref,color) unicos.")
    return parts_list

# Numero de caidas fisicas por pieza - leer de scene_config si disponible
try:
    from scene_config import NUM_PHYSICS_DROPS as NUM_DROPS_PER_PIECE
except ImportError:
    NUM_DROPS_PER_PIECE = 20  # numero de caidas por pieza (equivale a 12 angulos)
OUTPUT_DIR = os.path.join(project_root, "data", "ref_multiangle")


def _get_world_bbox(obj):
    import mathutils
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]


def _normalize_piece_to_scale(obj):
    bbox=_get_world_bbox(obj)
    dims=[max(b.x for b in bbox)-min(b.x for b in bbox),
          max(b.y for b in bbox)-min(b.y for b in bbox),
          max(b.z for b in bbox)-min(b.z for b in bbox)]
    mx=max(dims)
    if mx<1e-6:return 1.0
    if mx>LDRAW_THRESHOLD:
        factor=LDRAW_TO_BU
        obj.scale=(factor,factor,factor)
    else:
        factor=1.0
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active=obj
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return factor


def cleanup_scene():
    if not IN_BLENDER:return
    bpy.ops.object.select_all(action="DESELECT")
    keep={"Conveyor_Belt_Plane","Camera","Sun_Light","Rim_Light","Fill_Light",
          "Key_Light","Top_Diffuse_Light"}
    for o in list(bpy.context.scene.objects):
        if o.name not in keep:
            try:o.select_set(True)
            except:pass
    bpy.ops.object.delete()


def create_belt_collider():
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    half_t=BELT_THICKNESS_BU*0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0,location=(0.0,0.0,-half_t))
    belt=bpy.context.active_object
    belt.name="Conveyor_Belt_Plane"
    belt.scale=(BELT_WIDTH_BU,BELT_LENGTH_BU,BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type="PASSIVE";belt.rigid_body.collision_shape="BOX"
    belt.rigid_body.friction=BELT_FRICTION;belt.rigid_body.restitution=BELT_RESTITUTION
    belt.rigid_body.use_margin=True;belt.rigid_body.collision_margin=0.0
    mat=bpy.data.materials.get("Belt_Mat")
    if not mat:
        mat=bpy.data.materials.new("Belt_Mat")
        mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value=BELT_COLOR_LINEAR
            bsdf.inputs["Roughness"].default_value=0.5
    belt.data.materials.clear();belt.data.materials.append(mat)
    
    # Simular barras laterales metálicas de 2mm (0.2 BU) corriendo paralelas a la cinta
    for name in ["Side_Rail_L", "Side_Rail_R"]:
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.delete()
            
    rail_w = 0.2 # 2 mm
    rail_h = 0.4 # 4 mm altura
    # Carril izquierdo a X = -10 (límite del ancho 20)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-BELT_WIDTH_BU/2.0 + rail_w/2.0, 0.0, rail_h/2.0))
    rail_l = bpy.context.active_object
    rail_l.name = "Side_Rail_L"
    rail_l.scale = (rail_w, BELT_LENGTH_BU, rail_h)
    bpy.ops.object.transform_apply(scale=True)
    
    # Carril derecho a X = 10 (límite del ancho 20)
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
            bsdf.inputs['Base Color'].default_value = (0.55, 0.55, 0.55, 1.0) # Aluminio mate
            bsdf.inputs['Metallic'].default_value = 0.9
            bsdf.inputs['Roughness'].default_value = 0.5
            
    for rail in [rail_l, rail_r]:
        rail.data.materials.clear()
        rail.data.materials.append(mat_metal)
        
    return belt


def setup_ortho_camera(camera_type="cenital"):
    if "Camera" in bpy.data.objects:
        cam=bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add(location=(0,0,CAMERA_Z))
        cam=bpy.context.active_object;cam.name="Camera"
        
    target = bpy.data.objects.get("Camera_Target")
    if not target:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
        target = bpy.context.active_object
        target.name = "Camera_Target"
        
    cam.constraints.clear()
    
    if camera_type == "lateral":
        is_left = random.choice([True, False])
        if is_left:
            cam.location = (-5.0, 0.0, 15.0)
        else:
            cam.location = (5.0, 0.0, 15.0)
            
        track = cam.constraints.new(type='TRACK_TO')
        track.name = "Track_To"
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        cam.data.type = 'PERSP'
        cam.data.lens = 52.5  # 50% zoom (35mm -> 52.5mm)
    else:
        # Cenital camera at 15cm height with 50% zoom
        cam.location = (0.0, 0.0, 15.0)
        track = cam.constraints.new(type='TRACK_TO')
        track.name = "Track_To"
        track.target = target
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        cam.data.type = 'PERSP'
        cam.data.lens = 52.5  # 50% zoom (35mm -> 52.5mm)
        
    cam.data.clip_start=0.01; cam.data.clip_end=100.0
    bpy.context.scene.camera=cam; return cam


def get_stable_poses_from_db_subprocess(part_ref):
    import subprocess
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable
    code = f"""
import sys, json
sys.path.append('{project_root}')
try:
    from core.db import supabase_client
    poses = supabase_client.get_stable_poses('{part_ref}')
    print(json.dumps(poses))
except Exception as e:
    print(json.dumps([]))
"""
    try:
        res = subprocess.run([python_exec, "-c", code], capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            return json.loads(res.stdout.strip())
    except Exception as e:
        print(f"[WARN] Error consultando DB en subproceso: {e}")
    return []


def generate_physics_refs(output_dir=None, camera_type="cenital", set_id="75078-1"):
    if output_dir is None:output_dir=OUTPUT_DIR
    os.makedirs(output_dir,exist_ok=True)
    if not IN_BLENDER:print("[ERROR] Debe ejecutarse dentro de Blender");return
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity=(0.0,0.0,GRAVITY_Z)
    create_belt_collider()
    setup_studio_lighting()
    scene=bpy.context.scene
    if scene.world:
        scene.world.use_nodes=True
        bg=scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value=WORLD_BG_COLOR
            bg.inputs["Strength"].default_value=WORLD_BG_STRENGTH
    top=bpy.data.objects.get("Top_Diffuse_Light")
    if not top:
        bpy.ops.object.light_add(type="AREA",location=(0.0,0.0,TOP_LIGHT_Z))
        top=bpy.context.active_object;top.name="Top_Diffuse_Light"
    top.location=(0.0,0.0,TOP_LIGHT_Z)
    top.data.size=TOP_LIGHT_SIZE;top.data.energy=TOP_LIGHT_ENERGY
    engine_type = getattr(pa, 'engine', 'BLENDER_EEVEE')
    scene.render.engine = engine_type
    if engine_type == "CYCLES":
        scene.cycles.samples=CYCLES_SAMPLES_REF
    scene.render.film_transparent=False
    scene.render.resolution_x=RENDER_RES_SQUARE
    scene.render.resolution_y=RENDER_RES_SQUARE
    
    import mathutils
    
    # Cargar partes correspondientes al set
    parts_list = _load_all_part_colors_for_dino(set_id)
    if not parts_list:
        print(f"[ERROR] No se encontraron piezas para el set {set_id}")
        return
        
    # Pre-cargar todas las poses para estimar el total real de renders
    all_piece_poses = {}
    total_renders = 0
    for part in parts_list:
        ref = part["ref"]
        poses = get_stable_poses_from_db_subprocess(ref)
        if not poses:
            # Si no hay registradas en BD, al menos asumimos 1 pose por defecto
            poses = [{"orientation_euler": [0.0, 0.0, 0.0]}]
        all_piece_poses[ref] = poses
        total_renders += len(poses) * 24
        
    done = 0
    print(f"[DINOv2] Iniciando generación de renders para el set {set_id}. Total estimado: {total_renders} imágenes...")
    
    for part in parts_list:
        ref=part["ref"];color=part["color"]
        color_hex=color.lstrip("#")
        poses = all_piece_poses.get(ref, [{"orientation_euler": [0.0, 0.0, 0.0]}])
        print(f"Procesando {ref} ({color}) — {len(poses)} poses estables...")
        
        for pose_idx, pose in enumerate(poses):
            cleanup_scene()
            cam=setup_ortho_camera(camera_type)
            part_path=get_ldraw_part_path(ref)
            existing=set(bpy.context.scene.objects)
            if part_path:
                try:
                    bpy.ops.import_scene.importldr(filepath=part_path)
                    new_objs=[o for o in bpy.context.scene.objects if o not in existing]
                    par=next((o for o in new_objs if o.parent is None),None)
                    obj=get_single_mesh_object(par) if par else None
                    if not obj:
                        generate_detailed_fallback_mesh(ref)
                        obj=bpy.context.active_object
                except Exception as e:
                    print(f"Error importando {ref}: {e}")
                    generate_detailed_fallback_mesh(ref)
                    obj=bpy.context.active_object
            else:
                generate_detailed_fallback_mesh(ref)
                obj=bpy.context.active_object
            if not obj:
                done+=24
                continue
                
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True);bpy.context.view_layer.objects.active=obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY",center="BOUNDS")
            _normalize_piece_to_scale(obj)
            apply_bevel_modifier(obj)
            mat=create_abs_plastic_material(color)
            obj.data.materials.clear();obj.data.materials.append(mat)
            
            # Aplicar rotación de la pose estable
            quat = pose.get("orientation_quat")
            if quat and len(quat) == 4:
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                euler = pose.get("orientation_euler")
                if euler and len(euler) == 3:
                    obj.rotation_mode = 'XYZ'
                    obj.rotation_euler = mathutils.Euler(euler)
            
            # Posicionar centrado en X, Y (0,0)
            obj.location=(0.0,0.0,0.0)
            bpy.context.view_layer.update()
            
            # Asentar sobre la cinta (Z = 0)
            bbox = _get_world_bbox(obj)
            min_z = min(pt.z for pt in bbox)
            obj.location.z = -min_z + 0.02
            bpy.context.view_layer.update()
            
            initial_rot_z = obj.rotation_euler.z
            for angle_idx in range(24):
                angle_deg = angle_idx * 15
                angle_rad = math.radians(angle_deg)
                obj.rotation_euler.z = initial_rot_z + angle_rad
                bpy.context.view_layer.update()
                
                out_name=f"ref_{ref}_{color_hex}_pose{pose_idx:02d}_rot{angle_deg:03d}.png"
                out_path=os.path.join(output_dir,out_name)
                if os.path.exists(out_path):
                    done+=1;continue
                
                scene.render.filepath=out_path
                bpy.ops.render.render(write_still=True)
                done+=1
                print(f"  [{done}/{total_renders}] Guardado: {out_name}")
    print(f"OK: {done}/{total_renders} referencias generadas en {output_dir}")


if __name__=="__main__":
    import argparse
    args=[]
    if "--" in sys.argv: args=sys.argv[sys.argv.index("--")+1:]
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str,default=None)
    parser.add_argument("--camera_type",type=str,default="cenital")
    parser.add_argument("--drops",type=int,default=None)
    parser.add_argument("--set_id",type=str,default="75078-1")
    parser.add_argument("--engine",type=str,default="BLENDER_EEVEE",choices=["CYCLES", "BLENDER_EEVEE"])
    pa=parser.parse_known_args(args)[0]
    if pa.drops is not None:
        NUM_DROPS_PER_PIECE = pa.drops
    generate_physics_refs(output_dir=pa.output_dir, camera_type=pa.camera_type, set_id=pa.set_id)

