# -*- coding: utf-8 -*-
"""projects/camara_domo_75078/scripts/update_db_effective_heights.py
===================================================================
Ejecutar dentro de Blender para corregir la altura real de las poses.
"""
import os
import sys
import math
import numpy as np

project_root = "/Users/I764690/Code_personal/LegoVision"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
sys.path.append(os.path.join(project_root, "core", "db"))

import bpy
import mathutils
from supabase_client import get_connection
from generate_synthetic_set import get_ldraw_part_path
from generate_synthetic_dataset import get_single_mesh_object

def main():
    print("Iniciando actualización de alturas estables en Blender...")
    
    # 1. Obtener partes únicas de la BD del set 75078-1
    part_refs = [
        '32054', '2780', '6558', '4274', '3024', '3023', '3795', 
        '2412b', '3068', '3040', '32000', '61184', '15392', '85984'
    ]
    
    conn = get_connection()
    cur = conn.cursor()
    
    for ref in part_refs:
        print(f"\nProcesando pieza {ref}...")
        
        # Limpiar escena de otras piezas
        bpy.ops.object.select_all(action="DESELECT")
        for o in list(bpy.context.scene.objects):
            if o.name not in {"Camera", "Camera_Target"}:
                try: o.select_set(True)
                except: pass
        bpy.ops.object.delete()
        
        # Guardar objetos existentes
        existing_objects = set(bpy.context.scene.objects)
        
        # Importar pieza con importldr
        part_path = get_ldraw_part_path(ref)
        if not part_path:
            print(f"  [Error] No se encontró ruta LDraw para {ref}")
            continue
            
        try:
            bpy.ops.import_scene.importldr(filepath=part_path)
            new_objs = [o for o in bpy.context.scene.objects if o not in existing_objects]
            parent_obj = next((o for o in new_objs if o.parent is None), None)
            if parent_obj:
                obj = get_single_mesh_object(parent_obj)
            else:
                print(f"  [Error] No se encontró parent_obj para {ref}")
                continue
        except Exception as e:
            print(f"  [Error] Falló importación de {ref}: {e}")
            continue
            
        if not obj or obj.type != 'MESH':
            print(f"  [Error] El objeto no es una malla de tipo MESH: {obj}")
            continue
            
        # Centrar origen en BOUNDS
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        
        verts_local = np.array([v.co for v in obj.data.vertices])
        
        # Query todas las poses de esta pieza en la BD
        cur.execute("""
            SELECT id, pose_index, contact_normal 
            FROM stable_poses 
            WHERE part_ref = %s AND is_stable = true;
        """, (ref,))
        poses = cur.fetchall()
        
        for pose in poses:
            pose_id = pose["id"]
            pose_idx = pose["pose_index"]
            normal_ldraw = pose["contact_normal"]
            
            # En LDraw el normal está en LDraw space.
            # Convertir a Blender local space: [x, z, -y]
            normal_bl = np.array([
                normal_ldraw[0],
                normal_ldraw[2],
                -normal_ldraw[1]
            ], dtype=float)
            normal_bl /= np.linalg.norm(normal_bl)
            
            # Proyectar vértices sobre la normal
            proj = verts_local @ normal_bl
            height_ldu = float(proj.max() - proj.min())
            
            # Convertir a milímetros (1 LDU = 0.4 mm)
            height_mm = height_ldu * 0.4
            
            print(f"  Pose {pose_idx}: normal={normal_ldraw} -> height_ldu={height_ldu:.2f} -> height_mm={height_mm:.2f} mm")
            
            # Actualizar base de datos
            cur.execute("""
                UPDATE stable_poses 
                SET lateral_height = %s, effective_height = %s, efective_height = %s 
                WHERE id = %s;
            """, (height_mm, height_mm, height_mm, pose_id))
            
    conn.commit()
    cur.close()
    conn.close()
    print("Actualización completada exitosamente.")

if __name__ == "__main__":
    main()
