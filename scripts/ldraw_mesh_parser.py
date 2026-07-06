# -*- coding: utf-8 -*-
# scripts/ldraw_mesh_parser.py
# Parser LDraw recursivo: convierte .dat en lista de triangulos 3D
# resolviendo recursivamente sub-partes y primitivas con sus matrices.
#
# Coordenadas LDraw: X=derecha, Y=abajo (gravedad +Y), Z=camara
# 1 LDU = 0.4mm | stud pitch = 20 LDU = 8mm
#
# Studs: discos superiores (4-4disc.dat) = triangulos coplanares en Y=cte
# con normal (0,-1,0). SE INCLUYEN en la malla como cara plana real.
import os, sys
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LDRAW_ROOTS = []
_SUBDIRS = ["UnOfficial/parts","UnOfficial/p","Unofficial/parts","Unofficial/p",
            "parts/s","parts","p","models"]

def _init():
    global _LDRAW_ROOTS
    _LDRAW_ROOTS = []
    for d in [os.path.join(_PROJECT_ROOT,"data","ldraw"),
              "/Applications/Studio 2.0/ldraw",
              os.path.expanduser("~/ldraw")]:
        if os.path.isdir(d): _LDRAW_ROOTS.append(d)
_init()

_find_ldraw_file_cache = {}
_dir_contents_cache = {}

def find_ldraw_file(filename):
    fn = filename.replace("/", os.sep)
    # handle windows-style backslash in sub-paths from .dat files
    fn = fn.replace(chr(92)+chr(92), os.sep).replace(chr(92), os.sep)
    fn_low = fn.lower()
    
    if fn_low in _find_ldraw_file_cache:
        return _find_ldraw_file_cache[fn_low]
        
    bn_low = os.path.basename(fn_low)
    for root in _LDRAW_ROOTS:
        for sub in _SUBDIRS:
            for nm in [fn, fn_low]:
                c = os.path.join(root, sub, nm)
                if os.path.isfile(c):
                    _find_ldraw_file_cache[fn_low] = c
                    return c
        for sub in _SUBDIRS:
            d = os.path.join(root, sub)
            if not os.path.isdir(d): continue
            if d not in _dir_contents_cache:
                try:
                    _dir_contents_cache[d] = {f.lower(): f for f in os.listdir(d)}
                except Exception:
                    _dir_contents_cache[d] = {}
            contents = _dir_contents_cache[d]
            if bn_low in contents:
                c = os.path.join(d, contents[bn_low])
                _find_ldraw_file_cache[fn_low] = c
                return c
    _find_ldraw_file_cache[fn_low] = None
    return None

def _mat(tok):
    t = [float(x) for x in tok[:12]]
    return np.array([[t[3],t[4],t[5],t[0]],
                     [t[6],t[7],t[8],t[1]],
                     [t[9],t[10],t[11],t[2]],
                     [0.,0.,0.,1.]], dtype=np.float64)

def _xf(verts, M):
    ones = np.ones((len(verts),1))
    return (M @ np.hstack([verts,ones]).T).T[:,:3]

def parse_dat(path, M=None, depth=0, seen=None):
    if depth > 25: return []
    if seen is None: seen = set()
    rp = os.path.realpath(path)
    if rp in seen: return []
    seen = seen | {rp}
    if M is None: M = np.eye(4)
    tris = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except: return []
    for raw in lines:
        p = raw.strip().split()
        if not p: continue
        tp = p[0]
        if tp == "1" and len(p) >= 15:
            sub_name = " ".join(p[14:])
            Mc = M @ _mat(p[2:14])
            sp = find_ldraw_file(sub_name)
            if sp: tris.extend(parse_dat(sp, Mc, depth+1, seen))
        elif tp == "3" and len(p) >= 11:
            try:
                v = np.array([[float(p[i]) for i in range(2,11)]],dtype=np.float64).reshape(3,3)
                tris.append(_xf(v, M))
            except: pass
        elif tp == "4" and len(p) >= 14:
            try:
                v = np.array([[float(p[i]) for i in range(2,14)]],dtype=np.float64).reshape(4,3)
                v = _xf(v, M)
                tris.append(np.array([v[0],v[1],v[2]]))
                tris.append(np.array([v[0],v[2],v[3]]))
            except: pass
    return tris

def _ref_fallback_candidates(part_ref):
    """Genera candidatos de fallback para piezas cuyo ref BrickLink incluye
    sufijo de variante (ej. '4589b') pero el .dat oficial sólo está como
    base ('4589.dat'). Devuelve la lista en orden de preferencia.

    Reglas de fallback aplicadas en cascada:
      1. <ref>                                     (tal cual)
      2. <ref>.dat / <ref>.lower().dat             (con extension)
      3. quitar 1 letra final si va precedida de digito
         ('4589b' -> '4589', '2412b' -> '2412')
      4. quitar 2 letras finales si tras eso queda otra letra precedida
         de digito (cubre '973pb1810c01' -> '973pb1810c0' no aplica
         porque acaba en digito; vale para cosas como '4070a' -> '4070').
    """
    cands = [part_ref, part_ref + ".dat", part_ref.lower() + ".dat"]
    base = part_ref
    if len(base) > 1 and base[-1].isalpha() and base[-2].isdigit():
        base = base[:-1]
        cands += [base, base + ".dat", base.lower() + ".dat"]
    return cands


def get_triangles(part_ref):
    for c in _ref_fallback_candidates(part_ref):
        path = find_ldraw_file(c)
        if path:
            tris = parse_dat(path)
            if tris:
                return np.array(tris, dtype=np.float64)
    return np.empty((0, 3, 3), dtype=np.float64)

if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "3004"
    tris = get_triangles(ref)
    print("Part:", ref, "triangles:", len(tris))
    if len(tris) > 0:
        av = tris.reshape(-1,3)
        print("  Y range: [{:.2f}, {:.2f}] LDU".format(av[:,1].min(), av[:,1].max()))
        print("  X range: [{:.2f}, {:.2f}] LDU".format(av[:,0].min(), av[:,0].max()))
        print("  Z range: [{:.2f}, {:.2f}] LDU".format(av[:,2].min(), av[:,2].max()))
