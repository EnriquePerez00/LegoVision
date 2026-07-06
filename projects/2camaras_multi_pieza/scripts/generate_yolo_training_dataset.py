# -*- coding: utf-8 -*-
import os,sys,random,math,argparse,json
project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root,'scratch'))
sys.path.append(os.path.join(project_root,'scripts'))
try:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER=True
except ImportError:
    IN_BLENDER=False
if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world,setup_studio_lighting,create_abs_plastic_material,
        apply_bevel_modifier,apply_rigid_body_physics,get_ldraw_part_path,
        generate_detailed_fallback_mesh,enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
from scene_config import (
    BELT_WIDTH_BU,BELT_LENGTH_BU,BELT_THICKNESS_BU,BELT_COLOR_LINEAR,
    BELT_FRICTION,BELT_RESTITUTION,CAMERA_ORTHO_SCALE,CAMERA_Z,
    RENDER_RES_SQUARE,CYCLES_SAMPLES_DATASET,GRAVITY_Z,PIECE_MASS_KG,
    PIECE_FRICTION,PIECE_RESTITUTION,LDRAW_TO_BU,LDRAW_THRESHOLD,DEFAULT_SPAWN_Z,
    TOP_LIGHT_SIZE,TOP_LIGHT_ENERGY,TOP_LIGHT_Z,WORLD_BG_STRENGTH,WORLD_BG_COLOR,
    YOLO_PIECES_PER_FRAME_MIN,YOLO_PIECES_PER_FRAME_MAX,
    YOLO_EMPTY_FRAME_RATIO,YOLO_GRID_COLS,
    CORNER_LIGHT_OFFSET_XY,CORNER_LIGHT_Z,CORNER_LIGHT_SIZE,CORNER_LIGHT_ENERGY,
)

_OFF = CORNER_LIGHT_OFFSET_XY
_CZ  = CORNER_LIGHT_Z
CORNER_LIGHT_NAMES = ["Corner_Light_PP","Corner_Light_PN","Corner_Light_NP","Corner_Light_NN"]
CORNER_LIGHT_POSITIONS = [
    ( _OFF,  _OFF, _CZ),
    ( _OFF, -_OFF, _CZ),
    (-_OFF,  _OFF, _CZ),
    (-_OFF, -_OFF, _CZ),
]


def load_lego_color_palette():
    path=os.path.join(project_root,'database','color_catalog.json')
    fallback=['#A0A5A9','#1B1B1B','#C91A09','#F2F3F2','#FE8A18','#0A3C9F','#5A5A5A','#3B5E28','#F2CD37','#FF7E14']
    if not os.path.exists(path): return fallback
    with open(path,'r',encoding='utf-8') as f: catalog=json.load(f)
    pal=[]
    for _,info in catalog.items():
        hx=info.get('hex','')
        if hx and info.get('alpha',1.0)>=0.6 and info.get('material_type','solid') in ('solid','metallic','rubber'):
            pal.append(hx if hx.startswith('#') else '#'+hx)
    pal=list(set(pal)) or fallback
    print(f'[Color Palette] {len(pal)} colores LEGO.')
    return pal


def load_set_geometries(set_id):
    sys.path.insert(0,os.path.join(project_root,'database'))
    try: from set_catalog import REAL_SETS
    except ImportError: return [{'ref':'3004','is_minifig':False}]
    if set_id not in REAL_SETS: print(f'[ERROR] Set {set_id} no encontrado'); return []
    data=REAL_SETS[set_id]; seen={}
    for p in data.get('parts',[]):
        r=p['ref']
        if 'stk' not in r.lower() and len(r)<15 and r not in seen:
            seen[r]={'ref':r,'is_minifig':False}
    for fig in data.get('minifigures',[]):
        r=fig['ref']
        if r not in seen: seen[r]={'ref':r,'is_minifig':True}
    geoms=list(seen.values())
    print(f'[Universe] {set_id}: {len(geoms)} geometrias: '+', '.join(g['ref'] for g in geoms))
    return geoms


def _get_world_bbox(obj):
    import mathutils
    return [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

def _normalize_piece(obj):
    if not obj.data or not hasattr(obj.data,'vertices'): return 1.0
    verts=[v.co for v in obj.data.vertices]
    if not verts: return 1.0
    xs=[v.x for v in verts]; ys=[v.y for v in verts]; zs=[v.z for v in verts]
    dx=max(xs)-min(xs); dy=max(ys)-min(ys); dz=max(zs)-min(zs)
    mx=max(dx,dy,dz)
    if mx<1e-6: return 1.0
    factor=LDRAW_TO_BU if mx>LDRAW_THRESHOLD else 1.0
    cx=(max(xs)+min(xs))/2; cy=(max(ys)+min(ys))/2; cz=(max(zs)+min(zs))/2
    for v in obj.data.vertices:
        v.co.x=(v.co.x-cx)*factor; v.co.y=(v.co.y-cy)*factor; v.co.z=(v.co.z-cz)*factor
    obj.data.update(); obj.scale=(1.0,1.0,1.0); obj.location=(0.0,0.0,0.0)
    print(f'    [scale] {obj.name}: max={mx:.1f}LDU -> {mx*factor:.3f}BU (factor={factor})')
    return factor

def _all_mesh_verts_world(obj):
    verts=[]
    if obj.type=='MESH' and obj.data:
        m=obj.matrix_world; verts.extend([m@v.co for v in obj.data.vertices])
    return verts

def _bbox_yolo(obj,cam,scene):
    import mathutils
    if obj.location.z<-3.0: return None
    half_fov=CAMERA_ORTHO_SCALE/2.0
    if abs(obj.location.x)>half_fov+2.0 or abs(obj.location.y)>half_fov+2.0: return None
    world_verts=_all_mesh_verts_world(obj)
    if not world_verts:
        try:
            dep=bpy.context.evaluated_depsgraph_get(); oe=obj.evaluated_get(dep)
            world_verts=[oe.matrix_world@mathutils.Vector(c) for c in oe.bound_box]
        except Exception: world_verts=[obj.matrix_world@mathutils.Vector(c) for c in obj.bound_box]
    if not world_verts: return None
    xs,ys=[],[]
    for v in world_verts:
        c=world_to_camera_view(scene,cam,v); xs.append(c.x); ys.append(c.y)
    x0,x1=max(0.0,min(xs)),min(1.0,max(xs)); y0,y1=max(0.0,min(ys)),min(1.0,max(ys))
    if x1<=x0 or y1<=y0: return None
    w,h=x1-x0,y1-y0
    if w<0.008 or h<0.008: return None
    return [x0+w/2.0, 1.0-(y0+h/2.0), w, h]


def cleanup_pieces():
    if not IN_BLENDER: return
    bpy.ops.object.select_all(action='DESELECT')
    keep={'Conveyor_Belt_Plane','Camera','Camera_Target',
          'Sun_Light','Rim_Light','Fill_Light','Key_Light','Top_Diffuse_Light'}
    keep.update(CORNER_LIGHT_NAMES)
    for o in list(bpy.context.scene.objects):
        if o.name not in keep and not o.name.startswith('Template_'):
            try: o.select_set(True)
            except: pass
    bpy.ops.object.delete()
    for mat in list(bpy.data.materials):
        if mat.name.startswith('DR_') and mat.users==0: bpy.data.materials.remove(mat)
    for mesh in list(bpy.data.meshes):
        if mesh.users==0: bpy.data.meshes.remove(mesh)


def create_abs_plastic_material_dr(color_hex):
    roughness=random.uniform(0.1,0.3)
    mat_name=f'DR_{color_hex}_{roughness:.2f}'
    if mat_name in bpy.data.materials: return bpy.data.materials[mat_name]
    mat=bpy.data.materials.new(name=mat_name); mat.use_nodes=True
    nodes=mat.node_tree.nodes; links=mat.node_tree.links; nodes.clear()
    n=nodes.new(type='ShaderNodeBsdfPrincipled'); n.location=(0,0)
    hex_val=color_hex.lstrip('#')
    try: rgba=[int(hex_val[i:i+2],16)/255.0 for i in (0,2,4)]+[1.0]
    except: rgba=[0.6,0.6,0.6,1.0]
    n.inputs['Base Color'].default_value=rgba; n.inputs['Roughness'].default_value=roughness
    if 'IOR' in n.inputs: n.inputs['IOR'].default_value=1.55
    if 'Metallic' in n.inputs: n.inputs['Metallic'].default_value=0.0
    if 'Subsurface Weight' in n.inputs: n.inputs['Subsurface Weight'].default_value=0.06
    elif 'Subsurface' in n.inputs: n.inputs['Subsurface'].default_value=0.06
    out=nodes.new(type='ShaderNodeOutputMaterial'); out.location=(300,0)
    links.new(n.outputs['BSDF'],out.inputs['Surface'])
    return mat


def setup_corner_lights():
    for name,pos in zip(CORNER_LIGHT_NAMES,CORNER_LIGHT_POSITIONS):
        if name in bpy.data.objects:
            obj=bpy.data.objects[name]; obj.location=pos
        else:
            bpy.ops.object.light_add(type='AREA',location=pos)
            obj=bpy.context.active_object; obj.name=name
        obj.data.size=CORNER_LIGHT_SIZE; obj.data.energy=CORNER_LIGHT_ENERGY
        obj.rotation_euler=(0.0,0.0,0.0)
    print(f'[Lights] 4 luces de esquina OK')


def randomize_lights():
    configs={
        'Top_Diffuse_Light':{'base_energy':TOP_LIGHT_ENERGY,'base_z':TOP_LIGHT_Z,'bx':0.0,'by':0.0},
        'Key_Light':        {'base_energy':250.0,'base_z':5.0,'bx':3.0,'by':-3.0},
        'Fill_Light':       {'base_energy':100.0,'base_z':3.0,'bx':-3.0,'by':-2.0},
        'Rim_Light':        {'base_energy':150.0,'base_z':4.0,'bx':0.0,'by':4.0},
    }
    for name,pos in zip(CORNER_LIGHT_NAMES,CORNER_LIGHT_POSITIONS):
        configs[name]={'base_energy':CORNER_LIGHT_ENERGY,'base_z':pos[2],'bx':pos[0],'by':pos[1]}
    for lname,cfg in configs.items():
        obj=bpy.data.objects.get(lname)
        if obj and obj.type=='LIGHT':
            obj.location.x=cfg['bx']+random.uniform(-1.0,1.0)
            obj.location.y=cfg['by']+random.uniform(-1.0,1.0)
            obj.location.z=max(3.0,cfg['base_z']+random.uniform(-1.0,1.0))
            obj.data.energy=cfg['base_energy']*random.uniform(0.90,1.10)


def create_belt_collider():
    if 'Conveyor_Belt_Plane' in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Conveyor_Belt_Plane'].select_set(True)
        bpy.ops.object.delete()
    ht=BELT_THICKNESS_BU*0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0,location=(0.0,0.0,-ht))
    belt=bpy.context.active_object; belt.name='Conveyor_Belt_Plane'
    belt.scale=(BELT_WIDTH_BU,BELT_LENGTH_BU,BELT_THICKNESS_BU)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    belt.rigid_body.type='PASSIVE'; belt.rigid_body.collision_shape='BOX'
    belt.rigid_body.friction=BELT_FRICTION; belt.rigid_body.restitution=BELT_RESTITUTION
    belt.rigid_body.use_margin=True; belt.rigid_body.collision_margin=0.0
    mat=bpy.data.materials.get('Belt_Material')
    if not mat:
        mat=bpy.data.materials.new('Belt_Material'); mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value=BELT_COLOR_LINEAR
            bsdf.inputs['Roughness'].default_value=0.5
    belt.data.materials.clear(); belt.data.materials.append(mat)
    
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
    if 'Camera' in bpy.data.objects: cam=bpy.data.objects['Camera']
    else:
        bpy.ops.object.camera_add(location=(0,0,CAMERA_Z))
        cam=bpy.context.active_object; cam.name='Camera'
        
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


def create_templates(geom_list):
    if 'Templates' not in bpy.data.collections:
        col=bpy.data.collections.new('Templates')
        bpy.context.scene.collection.children.link(col)
    else:
        col=bpy.data.collections['Templates']
        for obj in list(col.objects): bpy.data.objects.remove(obj,do_unlink=True)
    col.hide_viewport=True; col.hide_render=True; tmap={}
    for part in geom_list:
        ref=part['ref']; part_path=get_ldraw_part_path(ref)
        if part.get('is_minifig') and not part_path:
            try:
                from scripts.assemble_minifig import build_minifig
                build_minifig(ref); part_path=get_ldraw_part_path(ref)
            except Exception as e: print(f'[WARN] minifig {ref}: {e}')
        existing=set(bpy.context.scene.objects); obj=None
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objs=[o for o in bpy.context.scene.objects if o not in existing]
                par=next((o for o in new_objs if o.parent is None),None)
                obj=get_single_mesh_object(par) if par else None
                if not obj: generate_detailed_fallback_mesh(ref); obj=bpy.context.active_object
            except Exception as e:
                print(f'[WARN] import {ref}: {e}')
                generate_detailed_fallback_mesh(ref); obj=bpy.context.active_object
        else:
            generate_detailed_fallback_mesh(ref); obj=bpy.context.active_object
        if not obj: print(f'[ERR] no template {ref}'); continue
        obj.name=f'Template_{ref}'
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True); bpy.context.view_layer.objects.active=obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
        _normalize_piece(obj); apply_bevel_modifier(obj)
        for c2 in list(obj.users_collection): c2.objects.unlink(obj)
        col.objects.link(obj); tmap[ref]=obj
        print(f'  [OK] Template {ref}')
    print(f'[Templates] {len(tmap)}/{len(geom_list)} cargadas.')
    return tmap


def build_spawn_grid(num_pieces):
    cols=[-8.0,-4.8,-1.6,1.6,4.8,8.0]
    rows=math.ceil(num_pieces/len(cols))
    y_half=min(8.5,rows*1.8)
    if rows>1: y_pos=[-y_half+i*(2*y_half/(rows-1)) for i in range(rows)]
    else: y_pos=[0.0]
    coords=[(cx,cy) for cy in y_pos for cx in cols]
    random.shuffle(coords)
    return coords

def run_physics(pieces_list,scene):
    if scene.rigidbody_world and scene.rigidbody_world.point_cache:
        scene.rigidbody_world.point_cache.frame_start=1
        scene.rigidbody_world.point_cache.frame_end=200
        bpy.ops.ptcache.free_bake_all()
    scene.frame_start=1; scene.frame_end=150; scene.frame_set(1)
    bpy.context.view_layer.update()
    for fr in range(1,151):
        scene.frame_set(fr); bpy.context.view_layer.update()
    for obj,_ in pieces_list:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True); bpy.context.view_layer.objects.active=obj
        try: bpy.ops.object.visual_transform_apply()
        except Exception: pass
        try: bpy.ops.rigidbody.object_remove()
        except Exception: pass
    scene.frame_set(150); bpy.context.view_layer.update()

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

def spawn_pieces(geom_list,color_palette,tmap,fi,num_pieces,center_spawn=False):
    coords=build_spawn_grid(num_pieces)
    selected=random.choices(geom_list,k=num_pieces)
    acol=bpy.context.scene.collection; placed=[]; jitter=0.5
    import mathutils
    for i,part in enumerate(selected):
        ref=part['ref']; tmpl=tmap.get(ref)
        if not tmpl or i>=len(coords): continue
        if center_spawn:
            gx, gy = 0.0, 0.0
            jitter_val = 0.05
        else:
            gx, gy = coords[i]
            jitter_val = jitter
        color_hex=random.choice(color_palette)
        oc=tmpl.copy(); oc.data=tmpl.data.copy()
        oc.name = f"Piece_Spawned_{ref}_{i}"
        acol.objects.link(oc); oc.parent=None
        oc.hide_viewport=False; oc.hide_render=False
        bpy.ops.object.select_all(action='DESELECT')
        oc.select_set(True); bpy.context.view_layer.objects.active=oc
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY',center='BOUNDS')
        
        # Load stable poses from DB
        poses = get_stable_poses_from_db_subprocess(ref)
        if poses:
            pose = random.choice(poses)
            quat = pose.get("orientation_quat")
            if quat and len(quat) == 4:
                oc.rotation_mode = 'QUATERNION'
                oc.rotation_quaternion = mathutils.Quaternion(quat)
            else:
                euler = pose.get("orientation_euler")
                if euler and len(euler) == 3:
                    oc.rotation_mode = 'XYZ'
                    oc.rotation_euler = mathutils.Euler(euler)
        else:
            oc.rotation_mode = 'XYZ'
            oc.rotation_euler = (random.uniform(0,6.283),random.uniform(0,6.283),random.uniform(0,6.283))
            
        # Additional random horizontal rotation
        oc.rotation_mode = 'XYZ'
        oc.rotation_euler.z += random.uniform(0.0, math.pi * 2)
        
        # Positions
        oc.location=(gx+random.uniform(-jitter_val,jitter_val),gy+random.uniform(-jitter_val,jitter_val),0.0)
        bpy.context.view_layer.update()
        
        # Snap to belt surface
        bbox = _get_world_bbox(oc)
        min_z = min(pt.z for pt in bbox)
        oc.location.z = -min_z + 0.02
        
        mat=create_abs_plastic_material_dr(color_hex)
        oc.data.materials.clear(); oc.data.materials.append(mat)
        placed.append((oc,ref))
    return placed


def generate_dataset(set_id,num_frames,output_dir,start_frame=0,pieces_exact=None,empty_ratio_override=None,camera_type="cenital",center_spawn=False,engine="BLENDER_EEVEE"):
    images_dir=os.path.join(output_dir,"images")
    labels_dir=os.path.join(output_dir,"labels")
    os.makedirs(images_dir,exist_ok=True); os.makedirs(labels_dir,exist_ok=True)
    if not IN_BLENDER: print("[ERROR] Debe ejecutarse en Blender."); return
    geom_list=load_set_geometries(set_id)
    if not geom_list: print("[ERROR] Sin geometrias."); return
    color_palette=load_lego_color_palette()
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity=(0.0,0.0,GRAVITY_Z)
    create_belt_collider()
    setup_studio_lighting()
    # Reemplazar luces de estudio con configuracion optimizada para FOV 20x20 BU
    # Luz cenital reducida (evita hotspot) + 4 luces de esquina (iluminacion uniforme)
    top=bpy.data.objects.get("Top_Diffuse_Light")
    if not top:
        bpy.ops.object.light_add(type="AREA",location=(0.0,0.0,TOP_LIGHT_Z))
        top=bpy.context.active_object; top.name="Top_Diffuse_Light"
    top.location=(0.0,0.0,TOP_LIGHT_Z); top.data.size=TOP_LIGHT_SIZE; top.data.energy=TOP_LIGHT_ENERGY
    setup_corner_lights()
    scene=bpy.context.scene
    if scene.world:
        scene.world.use_nodes=True
        bg=scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value=WORLD_BG_COLOR
            bg.inputs["Strength"].default_value=WORLD_BG_STRENGTH
    scene.render.engine=engine
    if engine=="CYCLES":
        scene.cycles.samples=CYCLES_SAMPLES_DATASET
        scene.cycles.max_bounces=2; scene.cycles.diffuse_bounces=1; scene.cycles.glossy_bounces=1
    scene.render.film_transparent=False
    scene.render.resolution_x=RENDER_RES_SQUARE; scene.render.resolution_y=RENDER_RES_SQUARE
    tmap=create_templates(geom_list)
    effective_empty_ratio = empty_ratio_override if empty_ratio_override is not None else YOLO_EMPTY_FRAME_RATIO
    num_empty=max(1,int(num_frames*effective_empty_ratio))
    num_pieces_frames=num_frames-num_empty
    frame_types=["pieces"]*num_pieces_frames+["empty"]*num_empty
    random.shuffle(frame_types)
    saved=0
    for fi,ftype in enumerate(frame_types):
        if fi<start_frame: continue
        cleanup_pieces()
        camera=setup_ortho_camera(camera_type)
        if ftype=="empty":
            img_fn=f"train_{fi:05d}.png"; lbl_fn=f"train_{fi:05d}.txt"
            scene.render.filepath=os.path.join(images_dir,img_fn)
            bpy.ops.render.render(write_still=True)
            open(os.path.join(labels_dir,lbl_fn),"w").close()
            print(f"[Empty] frame {fi+1}/{num_frames}")
            saved+=1; continue
        num_p=pieces_exact if pieces_exact is not None else random.randint(YOLO_PIECES_PER_FRAME_MIN,YOLO_PIECES_PER_FRAME_MAX)
        placed=spawn_pieces(geom_list,color_palette,tmap,fi,num_p,center_spawn=center_spawn)
        # Volver a configurar cámara para asegurar aleatoriedad en lateral
        camera=setup_ortho_camera(camera_type)
        labels=[]
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        for obj,ref in placed:
            bb=_bbox_yolo(obj,camera,scene)
            if bb: labels.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
        if labels:
            img_fn=f"train_{fi:05d}.png"; lbl_fn=f"train_{fi:05d}.txt"
            scene.render.filepath=os.path.join(images_dir,img_fn)
            randomize_lights()
            try: bpy.ops.render.render(write_still=True)
            except Exception as render_err:
                print(f"[WARN] Render fallido frame {fi}: {render_err}. Continuando..."); continue
            with open(os.path.join(labels_dir,lbl_fn),"w") as lf: lf.write(chr(10).join(labels)+chr(10))
            print(f"[OK] frame {fi+1}/{num_frames} - {len(labels)} piezas")
            saved+=1
        else:
            print(f"[SKIP] frame {fi+1} sin piezas en FOV")
    cleanup_pieces()
    for obj in list(tmap.values()): bpy.data.objects.remove(obj)
    print(f"[DONE] {saved}/{num_frames} imagenes generadas en {output_dir}")


if __name__=="__main__":
    args=[]
    if "--" in sys.argv: args=sys.argv[sys.argv.index("--")+1:]
    parser=argparse.ArgumentParser()
    parser.add_argument("--set_id",type=str,default="75078-1")
    parser.add_argument("--num_frames",type=int,default=500)
    parser.add_argument("--output_dir",type=str,default=None)
    parser.add_argument("--pieces",type=int,default=None,help="Numero exacto de piezas por frame (sobreescribe scene_config)")
    parser.add_argument("--start_frame",type=int,default=0)
    parser.add_argument("--empty_ratio",type=float,default=None)
    parser.add_argument("--camera_type",type=str,default="cenital")
    parser.add_argument("--center_spawn",action="store_true",help="Spawn pieces at the center of the belt (0,0)")
    parser.add_argument("--engine",type=str,default="BLENDER_EEVEE",choices=["CYCLES", "BLENDER_EEVEE"])
    pa=parser.parse_known_args(args)[0]
    out=pa.output_dir or os.path.join(project_root,"data","raw_dataset")
    import time; t0=time.time()
    pieces_exact=None
    if pa.pieces is not None:
        pieces_exact=max(1,pa.pieces)
        print(f"[Config] Piezas por frame (exacto): {pieces_exact}")
    empty_ratio_val=None
    if pa.empty_ratio is not None:
        empty_ratio_val=max(0.0,min(0.5,float(pa.empty_ratio)))
        print(f"[Config] YOLO_EMPTY_FRAME_RATIO={empty_ratio_val:.3f}")
    generate_dataset(pa.set_id,pa.num_frames,out,start_frame=pa.start_frame,pieces_exact=pieces_exact,empty_ratio_override=empty_ratio_val,camera_type=pa.camera_type,center_spawn=pa.center_spawn,engine=pa.engine)
    t1=time.time(); elapsed=t1-t0; n=pa.num_frames
    per_img=elapsed/n if n>0 else 0
    print(f"[TIMING] Total: {elapsed:.1f}s | Por imagen: {per_img:.1f}s | Estimado 500 imgs: {per_img*500/60:.1f} min")
