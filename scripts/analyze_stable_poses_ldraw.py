# -*- coding: utf-8 -*-
# scripts/analyze_stable_poses_ldraw.py  v3
# Algoritmo basado en Convex Hull 3D. Usa scipy ConvexHull para obtener
# caras EXTERIORES reales sin depender de normales del mesh LDraw.

import os, sys, json, argparse
import numpy as np
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldraw_mesh_parser import get_triangles

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_FACE_AREA_LDU2 = 28.3
GRAVITY_DIR        = np.array([0., 1., 0.])
NORMAL_DOT_THRESH  = 0.98
PLANE_D_TOL        = 1.0

def tri_area(tri):
    return np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) * 0.5

def build_2d_basis(normal):
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.,0.,0.]) if abs(n[0]) < 0.9 else np.array([0.,1.,0.])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u);   v /= np.linalg.norm(v)
    return u, v

def convex_hull_2d(points):
    pts = sorted(set(tuple(np.round(p,4)) for p in points))
    if len(pts) < 3: return pts
    def cross(O,A,B): return (A[0]-O[0])*(B[1]-O[1])-(A[1]-O[1])*(B[0]-O[0])
    lower = []
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def polygon_area_2d(hull):
    n = len(hull)
    if n < 3: return 0.0
    area = 0.0
    for i in range(n):
        j = (i+1) % n
        area += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]
    return abs(area) * 0.5

def point_in_polygon_2d(px, py, hull):
    inside = False; n = len(hull); j = n-1
    for i in range(n):
        xi,yi = hull[i]; xj,yj = hull[j]
        if (yi>py)!=(yj>py):
            if px < (xj-xi)*(py-yi)/(yj-yi+1e-12)+xi:
                inside = not inside
        j = i
    return inside

def face_name(avg_n):
    x,y,z = avg_n; ax,ay,az = abs(x),abs(y),abs(z)
    if ax>=ay and ax>=az: return "Right" if x>0 else "Left"
    if ay>=ax and ay>=az: return "Bottom" if y>0 else "Top"
    return "Front" if z>0 else "Back"

def face_class_from_normal(avg_n):
    gc = float(np.dot(avg_n, GRAVITY_DIR))
    if gc>0.7:  return "Bottom"
    if gc<-0.7: return "Top"
    return "Side"

def get_com(triangles):
    areas = np.array([tri_area(t) for t in triangles])
    total = areas.sum()
    centroids = triangles.mean(axis=1)
    return (centroids*areas[:,np.newaxis]).sum(axis=0) / (total+1e-10)


def detect_stable_faces_convexhull(triangles, debug=False):
    if not HAS_SCIPY:
        return detect_stable_faces_fallback(triangles, debug)
    verts = triangles.reshape(-1,3)
    verts_unique = np.unique(verts.round(1), axis=0)
    if len(verts_unique) < 4: return []
    try:
        hull = ConvexHull(verts_unique)
    except Exception as e:
        if debug: print("  ConvexHull error:", e)
        return detect_stable_faces_fallback(triangles, debug)
    equations = hull.equations
    normals = equations[:,:3]
    offsets = equations[:,3]
    used = set(); groups = []
    for i in range(len(normals)):
        if i in used: continue
        ni = normals[i]; di = offsets[i]
        grp = [i]; used.add(i)
        for j in range(i+1, len(normals)):
            if j in used: continue
            if np.dot(ni, normals[j]) < NORMAL_DOT_THRESH: continue
            if abs(offsets[j]-di) > PLANE_D_TOL: continue
            grp.append(j); used.add(j)
        groups.append(grp)
    com = get_com(triangles)
    if debug: print("  CdM=", np.round(com,2), "hull groups:", len(groups))
    stable = []
    for grp in groups:
        avg_n = normals[grp[0]].copy()
        avg_n /= (np.linalg.norm(avg_n)+1e-10)
        face_vi = set()
        for fi in grp:
            for vi in hull.simplices[fi]:
                face_vi.add(vi)
        fv = verts_unique[list(face_vi)]
        u_ax, v_ax = build_2d_basis(avg_n)
        p0 = fv[0]
        proj_2d = [(float(np.dot(v-p0, u_ax)), float(np.dot(v-p0, v_ax))) for v in fv]
        hull_2d = convex_hull_2d(proj_2d)
        if len(hull_2d) < 3: continue
        area = polygon_area_2d(hull_2d)
        if area < MIN_FACE_AREA_LDU2: continue
        pd = float(np.dot(avg_n, p0))
        com_proj = com - (float(np.dot(avg_n,com))-pd)*avg_n
        cu = float(np.dot(com_proj-p0, u_ax))
        cv = float(np.dot(com_proj-p0, v_ax))
        if not point_in_polygon_2d(cu, cv, hull_2d):
            if debug: print("  CdM fuera: area=",round(area,1),"n=",np.round(avg_n,3))
            continue
        gc = float(np.dot(avg_n, GRAVITY_DIR))
        is_side = abs(gc) < 0.7
        if is_side:
            hull_arr = np.array(hull_2d)
            w = hull_arr[:,0].max()-hull_arr[:,0].min()
            h = hull_arr[:,1].max()-hull_arr[:,1].min()
            if min(w,h) < 9.0:
                if debug: print("  Cara lateral estrecha: area=",round(area,1))
                continue
        fc = face_class_from_normal(avg_n)
        fname = face_name(avg_n)
        stable.append({
            "normal": avg_n.tolist(), "area": float(area),
            "face_class": fc, "face_name": fname,
            "gravity_component": float(gc),
        })
        if debug: print("  ESTABLE:", fc, fname, "area=", round(area,1), "n=", np.round(avg_n,3))
    return stable


def detect_stable_faces_fallback(triangles, debug=False):
    if debug: print("  [FALLBACK sin scipy]")
    N = len(triangles)
    normals_arr = np.zeros((N,3)); areas_arr = np.zeros(N); valid = np.zeros(N, dtype=bool)
    for i,tri in enumerate(triangles):
        e1=tri[1]-tri[0]; e2=tri[2]-tri[0]; n=np.cross(e1,e2); nm=np.linalg.norm(n)
        a=nm*0.5
        if nm>1e-10 and a>1e-8: normals_arr[i]=n/nm; areas_arr[i]=a; valid[i]=True
    total=areas_arr.sum()
    com=(triangles.mean(axis=1)*areas_arr[:,np.newaxis]).sum(0)/(total+1e-10)
    stable=[]
    for i in range(N):
        if not valid[i]: continue
        n=normals_arr[i]; p=triangles[i][0]
        if float(np.dot(n,com))-float(np.dot(n,p))>2.0: continue
        if areas_arr[i]<MIN_FACE_AREA_LDU2: continue
        gc=float(np.dot(n,GRAVITY_DIR))
        stable.append({"normal":n.tolist(),"area":float(areas_arr[i]),
                       "face_class":face_class_from_normal(n),"face_name":face_name(n),
                       "gravity_component":float(gc)})
    return stable


def detect_stable_faces(triangles, debug=False):
    return detect_stable_faces_convexhull(triangles, debug)


DEDUP_ANGLE_DEG = 50.0  # agrupar caras con normal a menos de 50 grados

def analyze_part(part_ref, debug=False):
    import math as _math
    tris = get_triangles(part_ref)
    if len(tris)==0:
        return {"part_ref":part_ref,"error":"no mesh","stable_poses":[],"n_poses":0}
    stable = detect_stable_faces(tris, debug=debug)
    # Deduplicar por normal similar a <= DEDUP_ANGLE_DEG grados
    # Mantener la cara de mayor area en cada grupo
    # Iterar dedup hasta convergencia (para caras a 45 grados encadenadas)
    changed = True
    deduped = [dict(s) for s in stable]
    while changed:
        changed = False
        new_deduped = []
        for s in deduped:
            n=np.array(s["normal"]); merged=False
            for ex in new_deduped:
                if ex["face_class"] != s["face_class"]: continue
                dot = max(-1.0, min(1.0, float(np.dot(n, np.array(ex["normal"])))))
                if _math.degrees(_math.acos(dot)) < DEDUP_ANGLE_DEG:
                    if s["area"]>ex["area"]:
                        ex.update(s)
                        changed = True
                    merged=True; break
            if not merged: new_deduped.append(dict(s))
        if len(new_deduped) < len(deduped): changed = True
        deduped = new_deduped
    return {"part_ref":part_ref,"n_triangles":int(len(tris)),
            "n_poses":len(deduped),"stable_poses":deduped}


def analyze_set(set_id, output_path=None, debug=False):
    sys.path.insert(0, os.path.join(_PROJECT_ROOT,"database"))
    from set_catalog import REAL_SETS
    set_data = REAL_SETS.get(set_id,{})
    parts = list(dict.fromkeys(
        p["ref"] for p in set_data.get("parts",[])
        if "stk" not in p["ref"].lower() and "pb" not in p["ref"].lower() and len(p["ref"])<15
    ))
    print("[SemanticPoses v3] Analizando",len(parts),"piezas del set",set_id)
    results=[]
    for i,ref in enumerate(parts):
        r=analyze_part(ref,debug=debug); results.append(r)
        faces_str=", ".join(p["face_name"]+"("+p["face_class"]+")" for p in r["stable_poses"])
        print("  ["+str(i+1)+"/"+str(len(parts))+"] "+ref+": "+str(r["n_poses"])+" poses ("+faces_str+")")
    if output_path:
        os.makedirs(os.path.dirname(output_path),exist_ok=True)
        with open(output_path,"w",encoding="utf-8") as fh:
            json.dump({"set_id":set_id,"results":results},fh,indent=2)
        print("[SemanticPoses v3] Guardado en:",output_path)
    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default="")
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if args.part:
        r = analyze_part(args.part, debug=args.debug)
        print("Part:", args.part, "| Triangles:", r.get("n_triangles",0), "| Stable poses:", r["n_poses"])
        for p in r["stable_poses"]:
            print("  ", p["face_class"], p["face_name"],
                  "area="+str(round(p["area"],1)), "n="+str([round(x,3) for x in p["normal"]]))
    else:
        out = args.output or os.path.join(_PROJECT_ROOT,"data","tmp",
            "semantic_poses_"+args.set_id.replace("-","")+".json")
        analyze_set(args.set_id, output_path=out, debug=args.debug)
