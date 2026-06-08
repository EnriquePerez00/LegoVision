# -*- coding: utf-8 -*-
# scripts/validate_stable_poses.py
import os
import sys
import random
import math
import json
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
sys.path.append(os.path.join(project_root, "scratch"))

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if IN_BLENDER:
    from generate_synthetic_set import (
        setup_physics_world, setup_studio_lighting,
        create_abs_plastic_material, apply_bevel_modifier, apply_rigid_body_physics,
        get_ldraw_part_path, generate_detailed_fallback_mesh, enable_metal_gpu_acceleration,
    )
    from generate_synthetic_dataset import get_single_mesh_object
    from scene_config import (
        BELT_SURFACE_Z, GRAVITY_Z, PIECE_MASS_KG, PIECE_FRICTION, PIECE_RESTITUTION,
        LDRAW_TO_BU, LDRAW_THRESHOLD, DEFAULT_SPAWN_Z
    )

def main():
    if not IN_BLENDER:
        print("[ERROR] Este script debe ejecutarse en Blender")
        return
        
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output_path", type=str, default=None)
    parsed_args = parser.parse_known_args(args)[0]
    
    set_id = parsed_args.set_id
    runs = parsed_args.runs
    
    # 1. Obtener inventario del catálogo
    from database.set_catalog import REAL_SETS
    if set_id not in REAL_SETS:
        print(f"[ERROR] Set {set_id} no encontrado en el catálogo.")
        return
        
    set_data = REAL_SETS[set_id]
    parts_list = []
    for p in set_data.get("parts", []):
        ref = p["ref"]
        # Excluir stickers, prints
        if "stk" not in ref.lower() and "pb" not in ref.lower() and len(ref) < 15:
            parts_list.append({
                "ref": ref,
                "name": p.get("name", "Pieza Lego")
            })
            
    print(f"Validando posiciones estables para {len(parts_list)} geometrías únicas del set {set_id}...")
    
    enable_metal_gpu_acceleration()
    setup_physics_world()
    bpy.context.scene.gravity = (0.0, 0.0, -9.81)
    
    # Crear plano de la cinta como colisionador pasivo
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
    half_t = 8.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_t * 0.5))
    belt = bpy.context.active_object
    belt.name = "Conveyor_Belt_Plane"
    belt.scale = (40.0, 40.0, half_t)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type = "PASSIVE"
    belt.rigid_body.collision_shape = "BOX"
    belt.rigid_body.friction = 0.95
    belt.rigid_body.restitution = 0.02
    
    results = {}
    
    for idx, p_info in enumerate(parts_list):
        ref = p_info["ref"]
        name = p_info["name"]
        print(f"\nPiece {idx+1}/{len(parts_list)}: {ref} ({name})")
        
        # Limpiar escena de otras piezas
        bpy.ops.object.select_all(action="DESELECT")
        for o in list(bpy.context.scene.objects):
            if o.name not in {"Conveyor_Belt_Plane", "Camera", "Camera_Target"}:
                try: o.select_set(True)
                except: pass
        bpy.ops.object.delete()
        
        # Importar pieza
        part_path = get_ldraw_part_path(ref)
        existing_objects = set(bpy.context.scene.objects)
        if part_path:
            try:
                bpy.ops.import_scene.importldr(filepath=part_path)
                new_objects = [o for o in bpy.context.scene.objects if o not in existing_objects]
                parent_obj = next((o for o in new_objects if o.parent is None), None)
                if parent_obj:
                    obj = get_single_mesh_object(parent_obj)
                else:
                    generate_detailed_fallback_mesh(ref)
                    obj = bpy.context.active_object
            except Exception as e:
                generate_detailed_fallback_mesh(ref)
                obj = bpy.context.active_object
        else:
            generate_detailed_fallback_mesh(ref)
            obj = bpy.context.active_object
            
        if not obj:
            print(f"Error cargando {ref}")
            continue
            
        # Normalizar
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        
        # Llevar a escala
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        dim_x = max(p.x for p in bbox) - min(p.x for p in bbox)
        dim_y = max(p.y for p in bbox) - min(p.y for p in bbox)
        dim_z = max(p.z for p in bbox) - min(p.z for p in bbox)
        max_dim = max(dim_x, dim_y, dim_z)
        factor = 0.04 if max_dim > 5.0 else 1.0
        obj.scale = (factor, factor, factor)
        bpy.ops.object.transform_apply(scale=True)
        
        # Configurar físicas
        apply_bevel_modifier(obj)
        
        # Haremos los lanzamientos
        stable_poses_found = []  # lista de vectores local_up
        stable_poses_counts = []  # conteo por pose
        stable_poses_details = []  # lista de diccionarios con info de cada pose
        
        scene = bpy.context.scene
        
        for r_idx in range(runs):
            # Posición aleatoria arriba
            obj.location = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), 4.0)
            obj.rotation_euler = (random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi))
            
            # Asegurar cuerpo rígido activo
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            try: bpy.ops.rigidbody.object_remove()
            except: pass
            
            bpy.ops.rigidbody.object_add(type="ACTIVE")
            obj.rigid_body.mass = 0.008
            obj.rigid_body.friction = 0.95
            obj.rigid_body.restitution = 0.02
            obj.rigid_body.collision_shape = "CONVEX_HULL"
            
            # Ejecutar simulación de caída (100 frames)
            for f in range(1, 101):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                
            # Aplicar transform y remover físicas
            bpy.ops.object.visual_transform_apply()
            bpy.ops.rigidbody.object_remove()
            
            scene.frame_set(100)
            bpy.context.view_layer.update()
            
            # Analizar orientación
            R = obj.matrix_world.to_3x3()
            # Vector UP global es (0,0,1). En coordenadas locales del objeto:
            local_up = R.inverted() @ mathutils.Vector((0, 0, 1))
            local_up.normalize()
            
            # Determinar a qué pose teórica corresponde
            # 0 (Top) -> local_up.y ~ -1
            # 2 (Bottom) -> local_up.y ~ 1
            # 1 (Side) -> local_up.x or local_up.z ~ 1/-1
            face_lbl = 1  # default Side
            if local_up.y < -0.85:
                face_lbl = 0  # Top
            elif local_up.y > 0.85:
                face_lbl = 2  # Bottom
                
            # Agrupar orientaciones similares para contar posiciones estables distintas
            matched = False
            for p_idx, prev_up in enumerate(stable_poses_found):
                # Ángulo entre los vectores menor a 15 grados
                angle = prev_up.angle(local_up)
                if angle < math.radians(15.0):
                    stable_poses_counts[p_idx] += 1
                    matched = True
                    break
                    
            if not matched:
                stable_poses_found.append(local_up.copy())
                stable_poses_counts.append(1)
                stable_poses_details.append({
                    "local_up": [local_up.x, local_up.y, local_up.z],
                    "face": face_lbl
                })
                
        # Guardar resultados experimentales de esta pieza
        results[ref] = {
            "name": name,
            "simulations": runs,
            "unique_poses_count": len(stable_poses_found),
            "poses": [
                {
                    "face": p["face"],
                    "count": stable_poses_counts[p_idx],
                    "local_up": p["local_up"]
                }
                for p_idx, p in enumerate(stable_poses_details)
            ]
        }
        
        print(f"  Posiciones estables identificadas: {len(stable_poses_found)}")
        for p_idx, count in enumerate(stable_poses_counts):
            pose = stable_poses_details[p_idx]
            print(f"    Pose {p_idx+1}: Cara {pose['face']} ({count} veces, local_up={pose['local_up']})")
            
    # Comparar con Base de Datos
    db_faces = {}
    try:
        from database import supabase_client
        db_embeddings = supabase_client.get_all_embeddings()
        for emb in db_embeddings:
            r = emb["part_ref"]
            f = emb["stable_face"]
            if r not in db_faces:
                db_faces[r] = set()
            db_faces[r].add(f)
    except Exception as e:
        print(f"[WARN] No se pudo conectar a la base de datos desde Blender ({e}). Se delegará la comparación a la GUI.")
        
    validation_report = []
    
    for ref, sim_info in results.items():
        db_f = list(db_faces.get(ref, []))
        exp_f = list(set(p["face"] for p in sim_info["poses"]))
        
        # Discrepancias
        missing_in_db = [f for f in exp_f if f not in db_f]
        extra_in_db = [f for f in db_f if f not in exp_f]
        
        discrepancy = len(missing_in_db) > 0 or len(extra_in_db) > 0
        
        report_item = {
            "part_ref": ref,
            "name": sim_info["name"],
            "simulations": sim_info["simulations"],
            "experimental_poses_count": sim_info["unique_poses_count"],
            "experimental_faces": exp_f,
            "database_faces": db_f,
            "missing_in_db": missing_in_db,
            "extra_in_db": extra_in_db,
            "discrepancy": discrepancy,
            "poses": sim_info["poses"]
        }
        validation_report.append(report_item)
        
    # Guardar en archivo temporal
    out_path = parsed_args.output_path or os.path.join(project_root, "data", "tmp", "stability_validation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "set_id": set_id,
            "runs_per_piece": runs,
            "report": validation_report
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\nReporte de validación de estabilidad guardado en: {out_path}")

if __name__ == "__main__":
    main()
