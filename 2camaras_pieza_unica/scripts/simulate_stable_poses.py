# -*- coding: utf-8 -*-
# scripts/simulate_stable_poses.py
# Simulacion fisica de posiciones estables de piezas LEGO en Blender/Bullet.
import os, sys, json, math, random, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root    = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, os.path.join(project_root, "database"))
# También añadir la raíz del repo y su carpeta scripts/, donde vive
# `analyze_stable_poses_ldraw.py`.
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "scripts"))
try:
    import bpy, mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
_C = {"belt_friction": 0.95, "belt_restitution": 0.02, "belt_speed_ms": 0.0833,
      "settle_frames": 20, "impulse_frames": 6, "eval_frames": 40, "n_dirs": 8,
      "stab_thresh": 0.875, "angle_thresh_deg": 15.0, "min_area": 28.3,
      "settle_z": 0.3, "ldraw_to_bu": 0.04}
try:
    import scene_config as sc
    _C["belt_friction"] = sc.BELT_FRICTION; _C["belt_restitution"] = sc.BELT_RESTITUTION
    _C["belt_speed_ms"] = sc.STABLE_POSE_BELT_SPEED_MS
    _C["settle_frames"] = sc.STABLE_POSE_SETTLE_FRAMES
    _C["impulse_frames"] = sc.STABLE_POSE_IMPULSE_FRAMES
    _C["eval_frames"] = sc.STABLE_POSE_EVAL_FRAMES
    _C["n_dirs"] = sc.STABLE_POSE_N_DIRECTIONS
    _C["stab_thresh"] = sc.STABLE_POSE_STABILITY_THRESHOLD
    _C["angle_thresh_deg"] = sc.STABLE_POSE_ANGLE_THRESHOLD_DEG
    _C["min_area"] = sc.STABLE_POSE_MIN_FACE_AREA_LDU2
    _C["settle_z"] = sc.STABLE_POSE_SETTLE_Z_OFFSET
    _C["ldraw_to_bu"] = sc.LDRAW_TO_BU
except Exception:
    pass
try:
    from ldraw_mesh_parser import get_triangles
    from analyze_stable_poses_ldraw import detect_stable_faces
    HAS_LDRAW = True
except ImportError:
    HAS_LDRAW = False


_PRECOMPUTED_CANDIDATES = {}  # loaded from --candidates_json

def get_candidate_faces(part_ref, min_area=None):
    if part_ref in _PRECOMPUTED_CANDIDATES:
        return _PRECOMPUTED_CANDIDATES[part_ref]
    if min_area is None: min_area = _C["min_area"]
    if not HAS_LDRAW: return []
    tris = get_triangles(part_ref)
    if len(tris) == 0: return []
    import analyze_stable_poses_ldraw as asp
    orig = asp.MIN_FACE_AREA_LDU2; asp.MIN_FACE_AREA_LDU2 = min_area
    try: faces = detect_stable_faces(tris)
    finally: asp.MIN_FACE_AREA_LDU2 = orig
    return [f for f in faces if f["area"] >= min_area]

def setup_physics_world():
    bpy.context.scene.use_gravity = True
    bpy.context.scene.gravity = (0.0, 0.0, -9.81)
    if bpy.context.scene.rigidbody_world is None: bpy.ops.rigidbody.world_add()
    rw = bpy.context.scene.rigidbody_world
    rw.time_scale = 1.0; rw.substeps_per_frame = 10; rw.solver_iterations = 20
    bpy.context.scene.render.fps = 60

def setup_belt():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.5))
    belt = bpy.context.active_object; belt.name = "StabilityBelt"
    belt.scale = (20.0, 20.0, 1.0); bpy.ops.object.transform_apply(scale=True)
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    belt.rigid_body.type = "PASSIVE"; belt.rigid_body.collision_shape = "BOX"
    belt.rigid_body.friction = _C["belt_friction"]
    belt.rigid_body.restitution = _C["belt_restitution"]
    belt.rigid_body.kinematic = True
    return belt

def load_ldraw_part(part_ref):
    bases = [
        os.path.join(project_root, "data", "ldraw"),
        os.path.expanduser("~/Library/Application Support/LDraw"),
        os.path.expanduser("~/ldraw"),
        "/Applications/Studio 2.0/ldraw",
        os.path.expanduser("~/Library/Application Support/BrickLink Studio 2.0/ldraw"),
    ]
    part_file = None
    for base in bases:
        for sub in ["UnOfficial/parts", "parts", "UnOfficial/p", "p"]:
            c = os.path.join(base, sub, part_ref + ".dat")
            if os.path.exists(c): part_file = c; break
        if part_file: break
    before = set(bpy.data.objects)
    if part_file:
        try: bpy.ops.import_scene.importldr(filepath=part_file)
        except Exception as e: print("    [WARN] " + part_ref + ": " + str(e)); part_file = None
    new_objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new_objs:
        bpy.ops.mesh.primitive_cube_add(size=0.8); new_objs = [bpy.context.active_object]
    bpy.ops.object.select_all(action="DESELECT")
    for o in new_objs: o.select_set(True)
    bpy.context.view_layer.objects.active = new_objs[0]
    if len(new_objs) > 1: bpy.ops.object.join()
    obj = bpy.context.active_object; obj.name = "Piece_" + part_ref
    s = _C["ldraw_to_bu"]; obj.scale = (s, s, s)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return obj


def orient_piece_on_face(obj, contact_normal):
    """Orienta `obj` para que la cara con normal `contact_normal`
    apunte hacia -Z (es decir, esa cara descanse sobre la cinta).

    NOTA HISTÓRICA: hasta esta versión, esta función tenía un bug
    silencioso: `bpy.ops.object.transform_apply(rotation=True)` se
    ejecutaba sin garantizar que `obj` fuera el ACTIVE object, lo
    que en algunos contextos (cuando ya hay otra mesh activa)
    aplicaba el bake al objeto incorrecto y dejaba `obj.rotation_euler`
    intacto, generando ``orientation_quat`` y ``orientation_euler``
    inconsistentes en la BD `stable_poses`. La fuente de verdad
    moderna para la rotación al renderizar es
    `_pose_utils.rotation_quat_from_contact_normal(contact_normal)`,
    determinista y sin Blender; aquí dejamos el bake correcto sólo
    para que el simulador físico pueda evaluar la estabilidad de la
    pose tras desplazamientos de cinta.
    """
    n = mathutils.Vector(contact_normal).normalized()
    target = mathutils.Vector((0.0, 0.0, -1.0))
    dot = max(-1.0, min(1.0, n.dot(target)))
    if dot > 0.9999: obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0))
    elif dot < -0.9999: obj.rotation_euler = mathutils.Euler((math.pi, 0.0, 0.0))
    else:
        axis = n.cross(target).normalized()
        obj.rotation_euler = mathutils.Matrix.Rotation(math.acos(dot), 4, axis).to_euler()
    # FIX: Garantizar que `obj` es el active object antes de
    # `transform_apply`, si no `bpy.ops.object.transform_apply` aplica
    # el bake a un objeto distinto en silencio.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(rotation=True)
    bpy.context.view_layer.update()
    # Sanity check: tras el bake, rotation_euler debe ser ~(0,0,0).
    re = obj.rotation_euler
    if max(abs(re.x), abs(re.y), abs(re.z)) > 1e-4:
        raise RuntimeError(
            f"transform_apply(rotation=True) NO se aplicó a {obj.name}; "
            f"rotation_euler={list(re)}. Revisar contexto activo."
        )

    bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    obj.location = (0.0, 0.0, -min(v.z for v in bbox) + _C["settle_z"])
    bpy.context.view_layer.update()
    q = obj.matrix_world.to_quaternion(); e = list(obj.matrix_world.to_euler())
    return [q.w, q.x, q.y, q.z], [e[0], e[1], e[2]]

def add_rigidbody(obj):
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try: bpy.ops.rigidbody.object_remove()
    except Exception: pass
    bpy.ops.rigidbody.object_add(type="ACTIVE"); rb = obj.rigid_body
    rb.mass = 0.008; rb.friction = _C["belt_friction"]; rb.restitution = _C["belt_restitution"]
    rb.collision_shape = "CONVEX_HULL"; rb.use_deactivation = False
    rb.linear_damping = 0.15; rb.angular_damping = 0.15

def clear_belt_animation(belt):
    if belt.animation_data: belt.animation_data_clear()
    belt.location = (0.0, 0.0, belt.location.z)

def apply_belt_impulse(belt, direction_rad, frame_start):
    fps = bpy.context.scene.render.fps
    v_max_bu_s = _C["belt_speed_ms"] / 0.01
    disp_bu = 2.0 * v_max_bu_s * (_C["impulse_frames"] / float(fps)) / math.pi
    dx = math.cos(direction_rad); dy = math.sin(direction_rad)
    loc0_z = belt.location.z; f0 = frame_start
    f1 = f0 + _C["impulse_frames"]; f2 = f1 + _C["eval_frames"]
    belt.location = (0.0, 0.0, loc0_z); belt.keyframe_insert(data_path="location", frame=f0)
    belt.location = (dx * disp_bu, dy * disp_bu, loc0_z)
    belt.keyframe_insert(data_path="location", frame=f1)
    belt.keyframe_insert(data_path="location", frame=f2)
    belt.location = (0.0, 0.0, loc0_z)
    if belt.animation_data and belt.animation_data.action:
        act = belt.animation_data.action
        all_fcs = []
        if hasattr(act, 'fcurves'):
            all_fcs = act.fcurves
        elif hasattr(act, 'layers'):
            for layer in act.layers:
                for strip in layer.strips:
                    for cb in getattr(strip, 'channelbags', []):
                        all_fcs += list(getattr(cb, 'fcurves', []))
        for fc in all_fcs:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'

def get_face_down_vector(obj):
    v = obj.matrix_world.to_3x3().inverted() @ mathutils.Vector((0.0, 0.0, -1.0))
    v.normalize(); return v

def same_pose(v1, v2):
    return math.degrees(math.acos(max(-1.0, min(1.0, v1.dot(v2))))) < _C["angle_thresh_deg"]


def simulate_face_stability(obj, belt, contact_normal, n_dirs, debug=False):
    scene = bpy.context.scene
    quat_init, euler_init = orient_piece_on_face(obj, contact_normal)
    loc_init = obj.location.copy(); rot_init = obj.rotation_euler.copy()
    add_rigidbody(obj)
    # Settle phase: run settle_frames to let piece rest on surface before measuring initial pose
    for sf in range(1, _C["settle_frames"] + 1):
        scene.frame_set(sf); bpy.context.view_layer.update()
    v_initial = get_face_down_vector(obj)
    passes = 0
    dirs = [2.0 * math.pi * i / n_dirs + random.uniform(-0.3, 0.3) for i in range(n_dirs)]
    random.shuffle(dirs)
    total_f = _C["settle_frames"] + _C["impulse_frames"] + _C["eval_frames"]
    for i, ang in enumerate(dirs):
        try: bpy.ops.rigidbody.object_remove()
        except Exception: pass
        # Restore piece to initial settled position
        obj.location = loc_init.copy()
        obj.rotation_euler = rot_init.copy()
        bpy.context.view_layer.update()
        clear_belt_animation(belt)
        add_rigidbody(obj)
        # Reset physics cache to force re-simulation from frame 1
        if scene.rigidbody_world and scene.rigidbody_world.point_cache:
            scene.rigidbody_world.point_cache.frame_start = 1
            bpy.ops.ptcache.free_bake_all()
        apply_belt_impulse(belt, ang, frame_start=_C["settle_frames"])
        for f in range(1, total_f + 1):
            scene.frame_set(f); bpy.context.view_layer.update()
        ok = same_pose(v_initial, get_face_down_vector(obj))
        if ok: passes += 1
        if debug:
            vafter = get_face_down_vector(obj)
            print("    dir=" + str(round(math.degrees(ang), 1)) +
                  " v_init=" + str([round(x,2) for x in v_initial]) +
                  " v_after=" + str([round(x,2) for x in vafter]) +
                  " ok=" + str(ok) + " " + str(passes) + "/" + str(i + 1))
    ratio = passes / float(n_dirs)
    return {"stable": ratio >= _C["stab_thresh"], "passes": passes, "total": n_dirs,
            "stability_ratio": ratio, "orientation_quat": quat_init, "orientation_euler": euler_init}

def analyze_part(part_ref, belt, n_dirs, debug=False):
    candidates = get_candidate_faces(part_ref)
    print("  [" + part_ref + "] " + str(len(candidates)) + " candidatas")
    if not candidates:
        return {"part_ref": part_ref, "error": "no candidates", "stable_poses": [], "n_poses": 0}
    for o in list(bpy.data.objects):
        if o.name.startswith("Piece_"): bpy.data.objects.remove(o, do_unlink=True)
    obj = load_ldraw_part(part_ref); stable_poses = []
    for fi, face in enumerate(candidates):
        normal = face["normal"]; fc = face["face_class"]; area = face["area"]
        print("    " + str(fi+1) + "/" + str(len(candidates)) + " " + fc + " area=" + str(round(area,1)))
        result = simulate_face_stability(obj, belt, normal, n_dirs, debug=debug)
        s = "ESTABLE" if result["stable"] else "inestable"
        print("      " + s + " " + str(result["passes"]) + "/" + str(result["total"]))
        # Guardar TODAS las caras candidatas (no solo las que pasan threshold).
        # is_stable = result["stable"] preserva la informacion del filtro,
        # pero stability_ratio queda como medida continua y comparable.
        stable_poses.append({"pose_index": len(stable_poses), "contact_normal": normal,
            "face_class": fc, "contact_area": area,
            "orientation_quat": result["orientation_quat"],
            "orientation_euler": result["orientation_euler"],
            "stability_ratio": result["stability_ratio"],
            "is_stable": bool(result["stable"]),
            "passes": result["passes"], "total": result["total"]})
    # Deduplicar poses con normal similar (angulo < ANGLE_THRESH)
    # Para cada normal duplicada, mantener la de mayor area de contacto
    import math as _math
    deduped = []
    for pose in stable_poses:
        n = pose["contact_normal"]
        merged = False
        for ex in deduped:
            en = ex["contact_normal"]
            dot = sum(n[k]*en[k] for k in range(3))
            dot = max(-1.0, min(1.0, dot))
            angle_deg = _math.degrees(_math.acos(dot))
            if angle_deg < _C["angle_thresh_deg"]:
                # Misma direccion: mantener la de mayor area
                if pose["contact_area"] > ex["contact_area"]:
                    ex.update(pose)
                merged = True
                break
        if not merged:
            deduped.append(dict(pose))
    # Renumerar pose_index
    for idx, pose in enumerate(deduped):
        pose["pose_index"] = idx
    if len(stable_poses) != len(deduped):
        print("  Dedup: " + str(len(stable_poses)) + " -> " + str(len(deduped)) + " poses")
    return {"part_ref": part_ref, "n_poses": len(deduped), "stable_poses": deduped}


def save_to_db(part_ref, stable_poses):
    """Guarda en BD (PostgreSQL local vía psycopg2). Borra previas y reinserta."""
    try:
        from supabase_client import get_connection
        with get_connection() as conn, conn.cursor() as cur:
            # Borrar poses previas para esta pieza
            cur.execute(
                "DELETE FROM stable_poses WHERE part_ref = %s", (part_ref,)
            )
            for pose in stable_poses:
                cur.execute(
                    """
                    INSERT INTO stable_poses
                        (part_ref, pose_index, contact_normal, face_class,
                         contact_area, orientation_quat, orientation_euler,
                         simulation_passes, simulation_total, stability_ratio,
                         is_stable)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        part_ref,
                        pose["pose_index"],
                        pose["contact_normal"],
                        pose["face_class"],
                        pose["contact_area"],
                        pose["orientation_quat"],
                        pose["orientation_euler"],
                        pose["passes"],
                        pose["total"],
                        pose["stability_ratio"],
                        bool(pose.get("is_stable", True)),
                    ),
                )
            # Recalcular stability_ratio_normalized para esta pieza
            cur.execute(
                """
                WITH max_sr AS (
                    SELECT GREATEST(MAX(stability_ratio), 1e-6) AS m
                    FROM stable_poses WHERE part_ref = %s
                )
                UPDATE stable_poses sp
                SET stability_ratio_normalized =
                    CASE WHEN sp.stability_ratio IS NULL THEN NULL
                         ELSE sp.stability_ratio / (SELECT m FROM max_sr)
                    END
                WHERE sp.part_ref = %s
                """,
                (part_ref, part_ref),
            )
        print("  [DB] " + part_ref + ": " + str(len(stable_poses)) + " poses guardadas")
    except Exception as e:
        print("  [DB WARN] " + part_ref + ": " + str(e))


def main():
    if not IN_BLENDER:
        print("[ERROR] Este script debe ejecutarse en Blender"); return
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--part", type=str, default="")
    parser.add_argument("--n_dirs", type=int, default=_C["n_dirs"])
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--save_db", action="store_true")
    parser.add_argument("--candidates_json", type=str, default="", help="JSON with precomputed face candidates")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_known_args(args_list)[0]
    # Load precomputed candidates if provided
    if args.candidates_json and os.path.exists(args.candidates_json):
        import json as _json
        with open(args.candidates_json) as _f:
            _data = _json.load(_f)
        _PRECOMPUTED_CANDIDATES.update(_data)
        print("[SimStablePoses] Loaded candidates for:", list(_data.keys()))
    setup_physics_world()
    for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
    belt = setup_belt()
    if args.part:
        parts = [args.part]
    else:
        sys.path.insert(0, os.path.join(project_root, "database"))
        from set_catalog import REAL_SETS
        set_data = REAL_SETS.get(args.set_id, {})
        parts = list(dict.fromkeys(
            p["ref"] for p in set_data.get("parts", [])
            if "stk" not in p["ref"].lower() and "pb" not in p["ref"].lower() and len(p["ref"]) < 15
        ))
    print("[SimStablePoses] " + str(len(parts)) + " piezas del set " + args.set_id)
    results = []
    for i, ref in enumerate(parts):
        print("[" + str(i+1) + "/" + str(len(parts)) + "] " + ref)
        r = analyze_part(ref, belt, args.n_dirs, debug=args.debug)
        results.append(r)
        if args.save_db and r.get("stable_poses") is not None:
            save_to_db(ref, r["stable_poses"])
    out = args.output or os.path.join(project_root, "data", "tmp",
        "stable_poses_sim_" + args.set_id.replace("-", "") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"set_id": args.set_id, "n_dirs": args.n_dirs, "results": results}, fh, indent=2)
    print("[SimStablePoses] Guardado en: " + out)
    total_stable = sum(r["n_poses"] for r in results)
    print("[SimStablePoses] Total poses estables: " + str(total_stable))


if __name__ == "__main__":
    main()
