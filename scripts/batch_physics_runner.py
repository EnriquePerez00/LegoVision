# -*- coding: utf-8 -*-
"""
scripts/batch_physics_runner.py
Orquesta la simulación física en lote en Blender sin renderizado de imágenes.
Vacia la tabla stable_poses y actualiza la BD local en tiempo real.
Usa multiprocessing para acelerar el cálculo físico en paralelo.
"""

import os
import sys
import json
import math
import subprocess
import time
from multiprocessing import Pool
import psycopg2
from psycopg2.extras import execute_values

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "database"))

from dotenv import load_dotenv
load_dotenv(override=True)
from supabase_client import DB_CONFIG
DB_CONFIG = DB_CONFIG.copy()
DB_CONFIG["gssencmode"] = "disable"

BLENDER_PATH = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")

# Piezas del set 75078-1 con sus colores
SET_75078_1_PARTS = [
    {"ref": "sw0614", "color": "#FFFFFF"},
    {"ref": "3004", "color": "#A0A5A9"},
    {"ref": "3001", "color": "#A0A5A9"},
    {"ref": "3020", "color": "#A0A5A9"},
    {"ref": "3022", "color": "#A0A5A9"},
    {"ref": "2877", "color": "#1B1B1B"},
    {"ref": "59900", "color": "#C91A09"},
    {"ref": "3003", "color": "#A0A5A9"},
    {"ref": "3002", "color": "#A0A5A9"},
    {"ref": "3005", "color": "#A0A5A9"},
    {"ref": "3010", "color": "#A0A5A9"},
    {"ref": "3021", "color": "#A0A5A9"},
    {"ref": "3023", "color": "#1B1B1B"},
    {"ref": "3024", "color": "#1B1B1B"},
    {"ref": "2420", "color": "#A0A5A9"},
    {"ref": "3710", "color": "#A0A5A9"},
    {"ref": "3622", "color": "#A0A5A9"},
    {"ref": "3665", "color": "#1B1B1B"},
    {"ref": "3039", "color": "#A0A5A9"},
    {"ref": "4070", "color": "#A0A5A9"},
    {"ref": "6141", "color": "#C91A09"},
    {"ref": "15573", "color": "#A0A5A9"},
    {"ref": "2412", "color": "#1B1B1B"},
    {"ref": "3069", "color": "#A0A5A9"},
    {"ref": "3068", "color": "#A0A5A9"},
    {"ref": "60478", "color": "#1B1B1B"},
    {"ref": "48336", "color": "#1B1B1B"},
    {"ref": "32000", "color": "#A0A5A9"},
    {"ref": "3700", "color": "#A0A5A9"},
    {"ref": "3701", "color": "#A0A5A9"},
    {"ref": "4032", "color": "#1B1B1B"},
    {"ref": "3062", "color": "#A0A5A9"},
    {"ref": "85984", "color": "#A0A5A9"},
    {"ref": "54200", "color": "#A0A5A9"},
    {"ref": "99206", "color": "#A0A5A9"},
    {"ref": "3037", "color": "#A0A5A9"},
    {"ref": "3298", "color": "#A0A5A9"},
    {"ref": "11477", "color": "#A0A5A9"},
    {"ref": "15068", "color": "#A0A5A9"},
    {"ref": "98138", "color": "#C91A09"},
    {"ref": "2431", "color": "#A0A5A9"},
    {"ref": "6636", "color": "#A0A5A9"}
]


def alter_db_schema():
    """Ejecuta los ALTERS necesarios para asegurar que los nuevos campos existen."""
    print("[DB] Configurando esquema de base de datos...")
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS energy_barrier_min DOUBLE PRECISION;
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS com_distance_to_boundary DOUBLE PRECISION;
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS zenith_observable_area DOUBLE PRECISION;
                ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS lateral_height DOUBLE PRECISION;
            """)
            conn.commit()
        print("[DB] Esquema verificado con éxito.")
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] Error modificando esquema: {e}")
    finally:
        conn.close()


def clear_stable_poses():
    """Vacía por completo la tabla stable_poses antes del cálculo masivo."""
    print("[DB] Vaciando la tabla stable_poses...")
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE stable_poses CASCADE;")
            conn.commit()
        print("[DB] Tabla stable_poses vaciada con éxito.")
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] Error al vaciar la tabla: {e}")
    finally:
        conn.close()


def run_single_part_sim(part_info):
    """Ejecuta la simulación en Blender para una pieza individual (se ejecuta en paralelo)."""
    ref = part_info["ref"]
    color = part_info["color"]
    print(f"🚀 Iniciando simulación de caída para {ref}...")
    
    # Crear script temporal para que Blender lo ejecute en segundo plano
    sim_script_path = os.path.join(project_root, "data", "tmp", f"sim_{ref}.py")
    os.makedirs(os.path.dirname(sim_script_path), exist_ok=True)
    
    script_content = f"""# -*- coding: utf-8 -*-
import os, sys, json, math, random
import bpy, mathutils

project_root = "{project_root}"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scratch'))
sys.path.append(os.path.join(project_root, 'scripts'))

from generate_synthetic_set import (
    setup_physics_world, create_conveyor_belt_collider,
    create_abs_plastic_material, apply_bevel_modifier, apply_rigid_body_physics,
    get_ldraw_part_path, generate_detailed_fallback_mesh,
)
from generate_synthetic_dataset import get_single_mesh_object

part_ref = "{ref}"
color_hex = "{color}"
num_simulations = 250
TARGET_SIZE = 1.6

setup_physics_world()
bpy.context.scene.gravity = (0.0, 0.0, -9.81)

# Crear plano/cinta colisionador grueso
if "Conveyor_Belt_Plane" in bpy.data.objects:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
    bpy.ops.object.delete()

cols = math.ceil(math.sqrt(num_simulations))
grid_step = TARGET_SIZE * 4.0
half_thick = TARGET_SIZE * 5.0
belt_extent = max(TARGET_SIZE * 30.0, cols * grid_step * 1.2)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -half_thick * 0.5))
belt = bpy.context.active_object
belt.name = "Conveyor_Belt_Plane"
belt.scale = (belt_extent, belt_extent, half_thick)
bpy.ops.object.transform_apply(scale=True)
bpy.ops.rigidbody.object_add(type="PASSIVE")
belt.rigid_body.type = "PASSIVE"
belt.rigid_body.collision_shape = "BOX"
belt.rigid_body.friction = 0.95
belt.rigid_body.restitution = 0.02
belt.rigid_body.use_margin = True
belt.rigid_body.collision_margin = 0.0

# Eliminar restos
for o in list(bpy.context.scene.objects):
    if o.name != "Conveyor_Belt_Plane" and not o.name.startswith("Template_"):
        bpy.data.objects.remove(o, do_unlink=True)

# Cargar pieza
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
    except Exception:
        generate_detailed_fallback_mesh(part_ref)
        template_obj = bpy.context.active_object
else:
    generate_detailed_fallback_mesh(part_ref)
    template_obj = bpy.context.active_object

template_obj.name = "Template"
bpy.ops.object.select_all(action="DESELECT")
template_obj.select_set(True)
bpy.context.view_layer.objects.active = template_obj
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Escalar a tamaño objetivo
bbox_tmpl = [template_obj.matrix_world @ mathutils.Vector(c) for c in template_obj.bound_box]
dim_x = max(p.x for p in bbox_tmpl) - min(p.x for p in bbox_tmpl)
dim_y = max(p.y for p in bbox_tmpl) - min(p.y for p in bbox_tmpl)
dim_z = max(p.z for p in bbox_tmpl) - min(p.z for p in bbox_tmpl)
max_dim = max(dim_x, dim_y, dim_z)
factor = 0.04 if max_dim > 5.0 else 1.0
template_obj.scale = (factor, factor, factor)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Crear copias en grilla
cols = math.ceil(math.sqrt(num_simulations))
rows = math.ceil(num_simulations / cols)
grid_step = TARGET_SIZE * 4.0
coords = []
for r in range(rows):
    for c in range(cols):
        if len(coords) < num_simulations:
            coords.append(((c - (cols - 1) / 2) * grid_step, (r - (rows - 1) / 2) * grid_step))

active_col = bpy.context.scene.collection
pieces = []
jitter = TARGET_SIZE * 0.3

for i in range(num_simulations):
    gx, gy = coords[i]
    obj_copy = template_obj.copy()
    obj_copy.data = template_obj.data.copy()
    active_col.objects.link(obj_copy)
    obj_copy.name = f"Piece_{{i}}"
    obj_copy.location = (gx + random.uniform(-jitter, jitter), gy + random.uniform(-jitter, jitter), 10.0)
    obj_copy.rotation_euler = (random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2), random.uniform(0, math.pi * 2))
    bpy.context.view_layer.update()
    
    # Caída exacta desde 5 cm
    bbox = [obj_copy.matrix_world @ mathutils.Vector(c) for c in obj_copy.bound_box]
    min_z = min(p.z for p in bbox)
    obj_copy.location.z = obj_copy.location.z - min_z + 5.0
    
    apply_rigid_body_physics(obj_copy, mass=0.008)
    obj_copy.rigid_body.restitution = 0.02
    obj_copy.rigid_body.friction = 0.95
    pieces.append(obj_copy)

# Simular físicas hasta que se estabilicen por completo
for f in range(1, 121):
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()

# Aplicar transformaciones finales
for obj in pieces:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.visual_transform_apply()
    bpy.ops.rigidbody.object_remove()

# Identificar y agrupar posiciones estables por la normal vertical final
def get_down_vector(o):
    v = o.matrix_world.to_3x3().inverted() @ mathutils.Vector((0.0, 0.0, -1.0))
    v.normalize()
    return v

poses_found = []
for o in pieces:
    d_vec = get_down_vector(o)
    
    # Clasificación de caras básica
    # La normal con componente Z más alta indica la dirección
    abs_v = [abs(d_vec.x), abs(d_vec.y), abs(d_vec.z)]
    max_idx = abs_v.index(max(abs_v))
    face_cls = "Side"
    if max_idx == 2:
        face_cls = "Top" if d_vec.z < 0 else "Bottom"
    
    # Calcular centro de masas (CoM) y polígono de sustentación
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    # Establecer origen en el Centro de Masas
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    
    com_world = o.location.copy()
    com_2d = mathutils.Vector((com_world.x, com_world.y))
    
    # Encontrar puntos de contacto con el suelo (Z < 0.05 relativo a la base)
    world_verts = [o.matrix_world @ v.co for v in o.data.vertices]
    min_z = min(v.z for v in world_verts)
    contact_pts = [v for v in world_verts if (v.z - min_z) < 0.05]
    
    # Envoltura convexa simple de puntos proyectados en XY
    pts_2d = [mathutils.Vector((p.x, p.y)) for p in contact_pts]
    
    # Algoritmo de Graham Scan simplificado para Convex Hull en 2D
    def get_convex_hull(points):
        if len(points) <= 3:
            return points
        # Ordenar por coordenada X, luego Y
        sorted_pts = sorted(points, key=lambda p: (p.x, p.y))
        
        def cross_product(o, a, b):
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
            
        lower = []
        for p in sorted_pts:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
            
        upper = []
        for p in reversed(sorted_pts):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        return lower[:-1] + upper[:-1]

    hull = get_convex_hull(pts_2d)
    
    # 1. Calcular distancia mínima al borde (com_distance_to_boundary)
    min_dist = float('inf')
    if len(hull) >= 2:
        for j in range(len(hull)):
            p1 = hull[j]
            p2 = hull[(j + 1) % len(hull)]
            
            # Vector del segmento
            edge = p2 - p1
            edge_len = edge.length
            if edge_len < 1e-6:
                continue
            edge_unit = edge / edge_len
            
            # Vector desde p1 al com_2d
            to_com = com_2d - p1
            
            # Proyección
            proj = to_com.dot(edge_unit)
            proj = max(0.0, min(edge_len, proj)) # Sujetar al segmento
            
            closest_pt = p1 + edge_unit * proj
            dist = (com_2d - closest_pt).length
            if dist < min_dist:
                min_dist = dist
    else:
        min_dist = 0.0
        
    # 2. Calcular barrera de energía potencial mínima (energy_barrier_min)
    # m = 0.008 kg, g = 9.81 m/s^2
    m = 0.008
    g = 9.81
    com_height_z = com_world.z
    min_energy_barrier = float('inf')
    
    if len(hull) >= 2:
        for j in range(len(hull)):
            p1 = hull[j]
            p2 = hull[(j + 1) % len(hull)]
            edge = p2 - p1
            edge_len = edge.length
            if edge_len < 1e-6:
                continue
            edge_unit = edge / edge_len
            to_com = com_2d - p1
            proj = to_com.dot(edge_unit)
            proj = max(0.0, min(edge_len, proj))
            closest_pt = p1 + edge_unit * proj
            
            # Distancia horizontal al eje de giro en 2D
            d_horiz = (com_2d - closest_pt).length
            # Distancia en 3D (radio del giro)
            r_3d = math.sqrt(d_horiz**2 + com_height_z**2)
            # Diferencia de altura al pivotar
            delta_h = r_3d - com_height_z
            energy = m * g * delta_h
            if energy < min_energy_barrier:
                min_energy_barrier = energy
    else:
        min_energy_barrier = 0.0

    quat = o.matrix_world.to_quaternion()
    euler = o.matrix_world.to_euler()
    
    poses_found.append({{
        "normal": [d_vec.x, d_vec.y, d_vec.z],
        "face_class": face_cls,
        "quat": [quat.w, quat.x, quat.y, quat.z],
        "euler": [euler.x, euler.y, euler.z],
        "com_distance": min_dist,
        "energy_barrier": min_energy_barrier
    }})

# Agrupar poses similares
grouped = []
for p in poses_found:
    merged = False
    p_norm = mathutils.Vector(p["normal"])
    for g in grouped:
        g_norm = mathutils.Vector(g["normal"])
        angle = p_norm.angle(g_norm)
        if angle < math.radians(15.0):
            g["count"] += 1
            # Acumular/promediar métricas
            g["com_distance"] = (g["com_distance"] + p["com_distance"]) / 2
            g["energy_barrier"] = (g["energy_barrier"] + p["energy_barrier"]) / 2
            merged = True
            break
    if not merged:
        grouped.append({{
            "normal": p["normal"],
            "face_class": p["face_class"],
            "quat": p["quat"],
            "euler": p["euler"],
            "com_distance": p["com_distance"],
            "energy_barrier": p["energy_barrier"],
            "count": 1
        }})

# Ordenar por frecuencia
grouped = sorted(grouped, key=lambda x: x["count"], reverse=True)

# Guardar salida JSON
out_data = []
for idx, g in enumerate(grouped):
    out_data.append({{
        "pose_index": idx,
        "contact_normal": g["normal"],
        "face_class": g["face_class"],
        "orientation_quat": g["quat"],
        "orientation_euler": g["euler"],
        "simulation_passes": g["count"],
        "simulation_total": num_simulations,
        "stability_ratio": g["count"] / float(num_simulations),
        "com_distance_to_boundary": g["com_distance"],
        "energy_barrier_min": g["energy_barrier"]
    }})

with open("{sim_script_path}.json", "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

"""
    
    with open(sim_script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
        
    # Ejecutar Blender
    cmd = [
        BLENDER_PATH,
        "-b",
        "-P",
        sim_script_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        # Leer JSON resultante
        result_json_path = sim_script_path + ".json"
        if os.path.exists(result_json_path):
            with open(result_json_path, "r", encoding="utf-8") as jf:
                poses = json.load(jf)
            # Limpiar archivos temporales
            os.remove(sim_script_path)
            os.remove(result_json_path)
            print(f"✅ Simulación completada para {ref}. Detectadas {len(poses)} poses estables.")
            return {"part_ref": ref, "status": "success", "poses": poses}
        else:
            print(f"❌ Error en simulación de {ref}: No se generó archivo de resultados. {res.stderr}")
            return {"part_ref": ref, "status": "failed", "error": "No output file"}
    except subprocess.TimeoutExpired:
        print(f"⌛ Tiempo de espera agotado para {ref}.")
        return {"part_ref": ref, "status": "timeout"}
    except Exception as e:
        print(f"❌ Excepción ejecutando {ref}: {e}")
        return {"part_ref": ref, "status": "error", "error": str(e)}


def save_poses_to_db(part_ref, poses, set_id="75078-1"):
    """Guarda inmediatamente las poses calculadas en la base de datos local."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from ldraw_mesh_parser import get_triangles

    # Cargar mesh
    triangles = get_triangles(part_ref)
    verts_unique = np.empty((0, 3))
    if len(triangles) > 0:
        verts = triangles.reshape(-1, 3)
        verts_unique = np.unique(verts.round(1), axis=0)

    def build_2d_basis(normal):
        n = normal / (np.linalg.norm(normal) + 1e-10)
        ref = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,1.,0.])
        u = np.cross(n, ref); u /= (np.linalg.norm(u) + 1e-10)
        v = np.cross(n, u);   v /= (np.linalg.norm(v) + 1e-10)
        return u, v

    def convex_hull_2d_area(points):
        pts_unique = np.unique(points.round(4), axis=0)
        if len(pts_unique) < 3: return 0.0
        try:
            hull = ConvexHull(pts_unique)
            return hull.volume
        except:
            return 0.0

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Eliminar previas
            cur.execute("DELETE FROM stable_poses WHERE part_ref = %s;", (part_ref,))
            
            # Insertar en lote
            values = []
            for p in poses:
                # Calcular zenith area y altura física
                area_mm2 = 0.0
                height_mm = 0.0
                if len(verts_unique) >= 3:
                    c_norm = np.array(p["contact_normal"])
                    norm_val = np.linalg.norm(c_norm)
                    if norm_val > 1e-6:
                        c_norm /= norm_val
                    else:
                        c_norm = np.array([0.0, 0.0, 1.0])
                    u_ax, v_ax = build_2d_basis(c_norm)
                    proj_points = np.array([(np.dot(v, u_ax), np.dot(v, v_ax)) for v in verts_unique])
                    area_ldu = convex_hull_2d_area(proj_points)
                    area_mm2 = float(round(area_ldu * 0.16, 2))
                    
                    # Calcular altura física real
                    proj_h = np.dot(verts_unique, c_norm)
                    height_ldu = np.max(proj_h) - np.min(proj_h)
                    height_mm = float(round(height_ldu * 0.4, 2))

                values.append((
                    part_ref,
                    p["pose_index"],
                    p["contact_normal"],
                    p["face_class"],
                    p.get("contact_area", 28.3), # fallback area
                    p["orientation_quat"],
                    p["orientation_euler"],
                    p["simulation_passes"],
                    p["simulation_total"],
                    p["stability_ratio"],
                    True,
                    set_id,
                    p["energy_barrier_min"],
                    p["com_distance_to_boundary"],
                    area_mm2,
                    height_mm
                ))
            
            if values:
                execute_values(cur, """
                    INSERT INTO stable_poses (
                        part_ref, pose_index, contact_normal, face_class, contact_area,
                        orientation_quat, orientation_euler, simulation_passes,
                        simulation_total, stability_ratio, is_stable, set_id,
                        energy_barrier_min, com_distance_to_boundary, zenith_observable_area,
                        lateral_height
                    ) VALUES %s;
                """, values)
                
            conn.commit()
            print(f"💾 Base de Datos actualizada para {part_ref} ({len(poses)} poses guardadas).")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error guardando poses para {part_ref} en la BD: {e}")
    finally:
        conn.close()


def main():
    t_start = time.time()
    print("==================================================")
    print("INICIANDO EJECUCIÓN MULTIPROCESO DE SIMULACIÓN FÍSICA")
    print("==================================================")
    
    # 1. Configurar DB
    alter_db_schema()
    # clear_stable_poses()
    
    # 2. Configurar Workers (Límite del 70% de hilos de CPU. En 12 cores, usamos 8 workers)
    num_cores = 12
    num_workers = int(num_cores * 0.7)
    print(f"Configurando Pool con {num_workers} procesos concurrentes...")
    
    # 3. Lanzar ejecuciones concurrentes
    pool = Pool(processes=num_workers)
    
    # Seguir el progreso en tiempo real
    results = pool.imap_unordered(run_single_part_sim, SET_75078_1_PARTS)
    
    completed = 0
    total = len(SET_75078_1_PARTS)
    
    for r in results:
        completed += 1
        ref = r["part_ref"]
        print(f"Progreso: {completed}/{total} piezas procesadas ({(completed/total)*100:.1f}%).")
        
        if r["status"] == "success":
            save_poses_to_db(ref, r["poses"])
        else:
            print(f"⚠️ Saltando actualización de DB para {ref} debido a error en la simulación.")
            
    pool.close()
    pool.join()
    
    t_end = time.time()
    print("==================================================")
    print(f"PROCESO COMPLETADO EN {t_end - t_start:.2f} SEGUNDOS.")
    print("==================================================")


if __name__ == "__main__":
    main()
