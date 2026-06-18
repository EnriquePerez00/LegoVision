# -*- coding: utf-8 -*-
"""
generate_3dposes_reports.py
===========================
Genera un fichero HTML por pieza del set 75078-1 con:
  - Vista 3D (proyección isométrica SVG — malla LDraw en Blender-space + quat)
  - Número de pose y todos los campos de stable_pose
  - Muestra TODAS las poses de la BD (sin filtro stability/contact_dim)

Correcciones respecto a v1:
  1. Lee directamente de la BD (sin filtrar): muestra todas las poses is_stable=TRUE
  2. Convierte LDraw→Blender space ANTES de aplicar el quaternión
     LDraw:   X=right, Y=down, Z=toward-camera
     Blender: X=right, Y=forward, Z=up
     Transform: (x,y,z)_ldr  →  (x, -z, -y)_blender
  3. Backface culling correcto en Blender space

Salida: 2camaras_pieza_unica/reports/3dposes/stable_poses_{ref}.html
"""

import os, sys, json, math, argparse
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LEGO_ROOT    = os.path.dirname(PROJECT_ROOT)

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LEGO_ROOT)
sys.path.insert(0, os.path.join(LEGO_ROOT, "database"))

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "3dposes")
SET_ID  = "75078-1"

# ── Nombres conocidos de piezas ────────────────────────────────────────────────
PART_NAMES = {
    "11477": "Slope Curved 2x1",
    "15068": "Slope Curved 2x2",
    "15573": "Plate Special 1x2 Groove",
    "2412":  "Tile Special 1x2 Grille",
    "2420":  "Plate Corner 2x2",
    "2431":  "Tile 1x4",
    "2877":  "Plate Special 1x2 Studs",
    "3001":  "Brick 2x4",
    "3002":  "Brick 2x3",
    "3003":  "Brick 2x2",
    "3004":  "Brick 1x2",
    "3005":  "Brick 1x1",
    "3010":  "Brick 1x4",
    "3020":  "Plate 2x4",
    "3021":  "Plate 2x3",
    "3022":  "Plate 2x2",
    "3023":  "Plate 1x2",
    "3024":  "Plate 1x1",
    "3037":  "Slope 45° 2x4",
    "3039":  "Slope 45° 2x2",
    "3062":  "Brick Round 1x1",
    "3068":  "Tile 2x2",
    "3069":  "Tile 1x2",
    "32000": "Technic Brick 1x2 Holes",
    "3298":  "Slope 33° 3x2",
    "3622":  "Brick 1x3",
    "3665":  "Slope Inverted 45° 2x1",
    "3700":  "Technic Brick 1x2 Hole",
    "3701":  "Technic Brick 1x4 Holes",
    "3710":  "Plate 1x4",
    "4032":  "Plate Round 2x2",
    "4070":  "Brick Special 1x1 Headlight",
    "48336": "Plate Special 1x2 Handle",
    "60478": "Plate Special 1x2 Handle Side",
    "6141":  "Plate Round 1x1",
    "6636":  "Tile 1x6",
    "98138": "Tile Round 1x1",
    "99206": "Bracket 2x2-2x2",
    "59900": "Cone 1x1",
    "85984": "Slope 31° 1x2",
    "54200": "Slope 31° 1x1",
}

# ── LDraw loader ───────────────────────────────────────────────────────────────
try:
    from ldraw_mesh_parser import get_triangles
    HAS_LDRAW = True
except ImportError:
    HAS_LDRAW = False
    print("[WARN] ldraw_mesh_parser no disponible; se usará representación de caja.")


# ══════════════════════════════════════════════════════════════════════════════
#  ACCESO A BASE DE DATOS — todas las poses sin filtrar
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_poses_from_db(set_id: str) -> dict:
    """
    Lee TODAS las poses is_stable=TRUE del set desde la BD.
    No aplica ningún filtro de stability_ratio ni contact_stable_width.
    Retorna dict {part_ref: [pose_dict, ...]} ordenado por part_ref, pose_index.
    """
    sql = """
        SELECT part_ref, pose_index, contact_normal, face_class, contact_area,
               orientation_quat, orientation_euler, stability_ratio,
               zenith_observable_area, zenith_bbox_area, lateral_height,
               contact_stable_length, contact_stable_width,
               is_stable
        FROM stable_poses
        WHERE is_stable = TRUE
        ORDER BY part_ref, pose_index
    """
    from supabase_client import get_connection
    out = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            ref = row["part_ref"]
            pose = {
                "pose_index":             int(row["pose_index"]),
                "original_pose_index":    int(row["pose_index"]),
                "contact_normal":         list(row["contact_normal"]) if row["contact_normal"] else None,
                "face_class":             row["face_class"],
                "contact_area":           float(row["contact_area"]) if row["contact_area"] is not None else None,
                "orientation_quat":       list(row["orientation_quat"]) if row["orientation_quat"] else None,
                "orientation_euler":      list(row["orientation_euler"]) if row["orientation_euler"] else None,
                "stability_ratio":        float(row["stability_ratio"]) if row["stability_ratio"] is not None else None,
                "zenith_observable_area": float(row["zenith_observable_area"]) if row["zenith_observable_area"] is not None else None,
                "zenith_bbox_area":       float(row["zenith_bbox_area"]) if row["zenith_bbox_area"] is not None else None,
                "lateral_height":         float(row["lateral_height"]) if row["lateral_height"] is not None else None,
                "contact_stable_length":  float(row["contact_stable_length"]) if row["contact_stable_length"] is not None else None,
                "contact_stable_width":   float(row["contact_stable_width"]) if row["contact_stable_width"] is not None else None,
            }
            out.setdefault(ref, []).append(pose)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRÍA 3D → SVG   (con sistema de coordenadas correcto)
# ══════════════════════════════════════════════════════════════════════════════

def ldraw_to_blender(pts: np.ndarray) -> np.ndarray:
    """
    Convierte vértices de LDraw space a Blender space.
      LDraw:   X=right,   Y=down (gravity+Y),  Z=toward-camera
      Blender: X=right,   Y=forward,            Z=up   (gravity -Z)
    Transformación estándar del importador LDraw de Blender:
      X_bl =  X_ldr
      Y_bl = -Z_ldr
      Z_bl = -Y_ldr
    """
    return np.column_stack([pts[:, 0], -pts[:, 2], -pts[:, 1]])


def quat_to_matrix(q) -> np.ndarray:
    """Cuaternión [w,x,y,z] → matriz de rotación 3×3."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,    2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,    1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def isometric_project(pts3d: np.ndarray) -> np.ndarray:
    """
    Proyección isométrica en Blender space (Z-up).
    Ángulo de azimut 30°, elevación 35.26°.
    Retorna (N, 2) — coordenadas de pantalla (X, Y con Y hacia abajo).
    """
    az = math.radians(30)
    el = math.radians(35.264)
    cos_az, sin_az = math.cos(az), math.sin(az)
    cos_el, sin_el = math.cos(el), math.sin(el)

    # Blender: X=right, Y=forward, Z=up
    # Isometric screen:  sx = X*cos_az - Y*sin_az
    #                    sy = -(X*sin_az + Y*cos_az)*sin_el - Z*cos_el
    # (invertimos Y de pantalla para que +Z vaya hacia arriba)
    sx =  pts3d[:, 0] * cos_az - pts3d[:, 1] * sin_az
    sy = -(pts3d[:, 0] * sin_az + pts3d[:, 1] * cos_az) * sin_el + pts3d[:, 2] * cos_el
    # Flip Y para coordenadas SVG (SVG Y crece hacia abajo)
    return np.column_stack([sx, -sy])


# Dirección de vista isométrica en Blender space (normalizada)
_AZ = math.radians(30)
_EL = math.radians(35.264)
ISO_VIEW_DIR_BL = np.array([
    math.cos(_EL) * math.cos(_AZ),
    math.cos(_EL) * math.sin(_AZ),   # componente Y (hacia el observador isométrico)
    math.sin(_EL),
], dtype=np.float64)
ISO_VIEW_DIR_BL /= np.linalg.norm(ISO_VIEW_DIR_BL)

# Dirección de luz en Blender space
LIGHT_DIR_BL = np.array([0.4, -0.6, 0.7], dtype=np.float64)
LIGHT_DIR_BL /= np.linalg.norm(LIGHT_DIR_BL)


def get_box_triangles_blender(dims_ldu):
    """Caja centrada en origen en Blender space (dims en LDU)."""
    lx, ly, lz = dims_ldu
    # En Blender space: X=right, Y=depth, Z=up
    hx, hy, hz = lx/2, ly/2, lz/2
    v = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ])
    faces = [
        (3,2,1),(3,1,0),  # -Z (suelo)
        (4,5,6),(4,6,7),  # +Z (techo)
        (0,1,5),(0,5,4),  # -Y
        (2,3,7),(2,7,6),  # +Y
        (0,3,7),(0,7,4),  # -X
        (1,2,6),(1,6,5),  # +X
    ]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float64)


def get_part_triangles_blender(ref: str) -> np.ndarray:
    """
    Devuelve triángulos de la pieza en Blender space (N,3,3).
    Primero intenta LDraw (convierte LDraw→Blender), si falla usa caja.
    """
    if HAS_LDRAW:
        tris_ldr = get_triangles(ref)
        if tris_ldr is not None and len(tris_ldr) > 0:
            flat_ldr = tris_ldr.reshape(-1, 3)
            flat_bl  = ldraw_to_blender(flat_ldr)
            return flat_bl.reshape(-1, 3, 3)
    # Fallback a caja si no hay malla LDraw
    # Usamos dimensiones por defecto de 20x20x20 LDU (1 stud) si no se puede determinar
    dims = (20.0, 20.0, 20.0)
    return get_box_triangles_blender(dims)


def project_top(pts3d: np.ndarray) -> np.ndarray:
    # Cenital pura: el centro (0,0) es la mitad de la imagen, no depende del bounding box
    return np.column_stack([pts3d[:, 0], -pts3d[:, 1]])


def project_side(pts3d: np.ndarray) -> np.ndarray:
    # Vista lateral desde la superficie de apoyo (Z=0)
    # sx = X, sy = -Z
    return np.column_stack([pts3d[:, 0], -pts3d[:, 2]])

# Direcciones de vista (hacia el observador) en Blender space
TOP_VIEW_DIR_BL  = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # mira desde +Z
SIDE_VIEW_DIR_BL = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # mira desde +Y


def _prepare_tris(ref: str, pose: dict) -> np.ndarray:
    """
    Carga la malla de la pieza, aplica el quaternión, deja la pieza reposando
    sobre Z=0 y centra en X,Y. Devuelve array (N, 3, 3) de triángulos en Blender space.
    """
    tris_bl = get_part_triangles_blender(ref)

    # Corregir error de quaternión identidad inconsistente en la base de datos
    # Calculando el quaternión analítico determinista a partir de contact_normal
    cn = pose.get("contact_normal")
    if cn and len(cn) == 3:
        # En LDraw space la normal apunta hacia el plano de contacto. Queremos rotarla
        # para que apunte hacia abajo (-Y en LDraw space).
        # Pero estamos en Blender space. La función rotation_quat_from_contact_normal
        # está pensada para la mesh en LDraw space (donde target_down es -Y_ldr, o sea,
        # (0, -1, 0) o similar).
        # Vamos a importarla de _pose_utils que ya tiene la lógica correcta de Rodrigues.
        from _pose_utils import rotation_quat_from_contact_normal
        # contact_normal está en LDraw space en la base de datos.
        # Queremos el cuaternión en Blender space para rotar los triángulos ya convertidos a Blender space.
        # Alternativamente, podemos aplicar la rotación determinista en LDraw space y luego
        # convertir a Blender space. ¡Eso es más robusto!
        # Vamos a hacerlo:
        tris_ldr = get_triangles(ref)
        if tris_ldr is None or len(tris_ldr) == 0:
            dims = (20.0, 20.0, 20.0)
            tris_bl = get_box_triangles_blender(dims)
            tris_rot = tris_bl.copy()
        else:
            # 1. Rotar en LDraw space usando el quaternión de Rodrigues determinista
            # En LDraw space, la normal de contacto debe acabar apuntando a (0, 1, 0) (down en LDraw)
            q_ldr = rotation_quat_from_contact_normal(cn, target_down=(0.0, 1.0, 0.0))
            R_ldr = quat_to_matrix(q_ldr)
            flat_ldr = tris_ldr.reshape(-1, 3)
            flat_rot_ldr = (R_ldr @ flat_ldr.T).T
            # 2. Convertir los vértices rotados a Blender space
            flat_rot_bl = ldraw_to_blender(flat_rot_ldr)
            tris_rot = flat_rot_bl.reshape(-1, 3, 3)
    else:
        tris_rot = tris_bl.copy()

    # Apoyar en Z=0 (plano de contacto de la pose estable)
    all_verts = tris_rot.reshape(-1, 3)
    z_min = all_verts[:, 2].min()
    tris_rot = tris_rot - np.array([0.0, 0.0, z_min])

    # Centrar X,Y en 0.0, 0.0 Blender-space
    all_verts = tris_rot.reshape(-1, 3)
    cx = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
    cy = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
    tris_rot = tris_rot - np.array([cx, cy, 0.0])
    return tris_rot


def _compute_shared_scale(tris_rot: np.ndarray, size: int, margin: int = 20) -> float:
    """
    Calcula UN ÚNICO factor de escala (px/mm o px/LDU) que se aplicará a las dos
    vistas (cenital y lateral) para que las dimensiones físicas de la pieza
    se preserven entre ellas.
    """
    canvas = size - 2 * margin
    all_verts = tris_rot.reshape(-1, 3)

    proj_top  = project_top(all_verts)
    proj_side = project_side(all_verts)

    span_top  = proj_top.max(axis=0)  - proj_top.min(axis=0)
    span_side = proj_side.max(axis=0) - proj_side.min(axis=0)

    # Mayor extensión entre las dos vistas (en cualquiera de los dos ejes)
    max_span = float(max(span_top[0], span_top[1], span_side[0], span_side[1]))
    if max_span < 1e-6:
        return 1.0

    return (canvas / max_span) * 0.90


def _render_orthographic_svg(tris_rot: np.ndarray,
                             project_fn,
                             view_dir: np.ndarray,
                             base_rgb: tuple,
                             size: int,
                             label: str,
                             cn_str: str = "",
                             draw_ground: bool = False,
                             scale: float = None) -> str:
    """
    Renderiza una vista ortográfica (cenital o lateral) en SVG.
    """
    margin = 20
    canvas = size - 2 * margin

    all_verts = tris_rot.reshape(-1, 3)
    all_proj  = project_fn(all_verts)

    if len(all_proj) == 0:
        return _empty_svg(size)

    mn, mx = all_proj.min(axis=0), all_proj.max(axis=0)
    span = mx - mn
    if span[0] < 1e-6 or span[1] < 1e-6:
        return _empty_svg(size)

    if scale is None:
        scale = min(canvas / span[0], canvas / span[1]) * 0.90

    # Configuración de offsets específicos
    if draw_ground:
        # Vista lateral: Alinear plano Z=0 (suelo) exactamente al nivel inferior del canvas (size - margin)
        # sy = -Z -> min Z es 0 -> max Z proyectada (-Z) es 0.
        # Deseamos que Z=0 (que se proyecta a 0) se dibuje en y = size - margin.
        # Por tanto, offset_y = size - margin
        offset = np.array([
            size / 2.0,                  # Centrado en X
            size - margin - 10.0         # Suelo alineado a la base del canvas
        ])
    else:
        # Vista cenital (TOP): Forzar centro en el punto 0.0, 0.0 de Blender-space
        # La pieza ya está posicionada con el centro de la pose estable en 0,0,0
        # Así que proyectar 0.0, 0.0 da 0.0, 0.0. Queremos que este origen 3D se sitúe
        # exactamente en el centro de la imagen (size/2, size/2).
        offset = np.array([
            size / 2.0,
            size / 2.0
        ])

    # Painter's algorithm: ordenar por profundidad respecto a la dir. de vista
    n_tris = tris_rot.shape[0]
    tri_depth_list = [
        (tris_rot[i], float(np.dot(tris_rot[i].mean(axis=0), view_dir)))
        for i in range(n_tris)
    ]
    tri_depth_list.sort(key=lambda x: x[1])  # de fondo hacia frente

    # Suelo (solo en vista lateral): línea Z=0
    ground_svg = ""
    if draw_ground:
        all_v = tris_rot.reshape(-1, 3)
        x_ext = max(abs(all_v[:, 0].max()), abs(all_v[:, 0].min())) * 1.50
        # En project_side, Y_svg = -Z * scale + offset_y
        # Para Z=0, Y_svg es simplemente offset[1]
        y_ground = offset[1]
        x0_ground = -x_ext * scale + offset[0]
        x1_ground = x_ext * scale + offset[0]
        ground_svg = (f'<line x1="{x0_ground:.1f}" y1="{y_ground:.1f}" '
                      f'x2="{x1_ground:.1f}" y2="{y_ground:.1f}" '
                      f'stroke="#b0bec5" stroke-width="1.5" '
                      f'stroke-dasharray="3,3"/>')

    polygons = []
    for tri_3d, _ in tri_depth_list:
        # Si es vista lateral, filtrar triángulos que queden por debajo del suelo (Z < 0)
        # En Blender space, la altura es Z (tri_3d[:, 2]). Si algún vértice o el baricentro
        # está por debajo de 0, o si hacemos un recorte estricto.
        # El usuario dice: "muestra unicamente la parte por encima de la superficie".
        # Si algún vértice de un triángulo está por debajo de 0, podemos ignorarlo por completo
        # o recortarlo. Omitir el triángulo completo si tiene vértices por debajo de Z < -1e-3 es simple y efectivo.
        if draw_ground and np.any(tri_3d[:, 2] < -1e-3):
            continue # Omitir partes sumergidas bajo el suelo

        e1 = tri_3d[1] - tri_3d[0]
        e2 = tri_3d[2] - tri_3d[0]
        n  = np.cross(e1, e2)
        n_len = np.linalg.norm(n)
        if n_len < 1e-10:
            continue
        n /= n_len

        # Backface culling
        if float(np.dot(n, view_dir)) < 0.0:
            continue

        # Iluminación
        diff    = max(0.0, float(np.dot(n, LIGHT_DIR_BL)))
        ambient = 0.55
        intensity = ambient + (1.0 - ambient) * diff

        r = min(255, int(base_rgb[0] * intensity))
        g = min(255, int(base_rgb[1] * intensity))
        b = min(255, int(base_rgb[2] * intensity))
        
        # Borde luminoso para resaltar studs y huecos
        stroke_str = "rgba(255,255,255,0.75)"

        # Proyectar
        pts2d = []
        for v in tri_3d:
            if draw_ground:
                # Vista lateral: X = v[0]*scale + offset_x, Y = -v[2]*scale + offset_y
                pts2d.append([v[0], -v[2]])
            else:
                # Vista cenital: X = v[0]*scale + offset_x, Y = -v[1]*scale + offset_y
                pts2d.append([v[0], -v[1]])
        pts_svg = np.array(pts2d) * scale + offset
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_svg)

        polygons.append(
            f'<polygon points="{pts_str}" '
            f'fill="rgb({r},{g},{b})" '
            f'stroke="{stroke_str}" '
            f'stroke-width="0.5" stroke-linejoin="round"/>'
        )

    return (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:linear-gradient(145deg,#f2f5fa,#e4eaf4);'
        f'border-radius:10px;border:1px solid #c5cedc;">\n'
        f'{ground_svg}\n'
        f'{"".join(polygons)}\n'
        f'<text x="4" y="{size-14}" font-size="8.5" fill="#7a8a9e" '
        f'font-family="JetBrains Mono,monospace">{cn_str}</text>\n'
        f'<text x="4" y="{size-4}" font-size="8" fill="#9aabb8" '
        f'font-family="monospace">{label}</text>\n'
        f'</svg>'
    )


def render_top_svg(ref: str, pose: dict, size: int = 200,
                   scale: float = None, tris_rot: np.ndarray = None) -> str:
    """Vista cenital ortográfica (mirando desde +Z hacia abajo)."""
    if tris_rot is None:
        tris_rot = _prepare_tris(ref, pose)
    if scale is None:
        scale = _compute_shared_scale(tris_rot, size)
    fc_low = (pose.get("face_class") or "side").lower()
    color_map = {
        "top":    (74, 185, 149),     # Verde menta/esmeralda luminoso
        "bottom": (160, 116, 225),    # Violeta/lavanda claro
        "side":   (91, 150, 241),     # Azul coral/celeste vivo
    }
    base_rgb = color_map.get(fc_low, (91, 150, 241))
    cn = pose.get("contact_normal") or []
    cn_str = f"n=[{','.join(f'{v:.2f}' for v in cn)}]" if cn else ""
    return _render_orthographic_svg(
        tris_rot, project_top, TOP_VIEW_DIR_BL, base_rgb, size,
        label="TOP (cenital, +Z)", cn_str=cn_str, draw_ground=False, scale=scale,
    )


def render_side_svg(ref: str, pose: dict, size: int = 200,
                    scale: float = None, tris_rot: np.ndarray = None) -> str:
    """Vista lateral ortográfica (mirando desde +Y hacia -Y, plano XZ)."""
    if tris_rot is None:
        tris_rot = _prepare_tris(ref, pose)
    if scale is None:
        scale = _compute_shared_scale(tris_rot, size)
    fc_low = (pose.get("face_class") or "side").lower()
    color_map = {
        "top":    (74, 185, 149),     # Verde menta/esmeralda luminoso
        "bottom": (160, 116, 225),    # Violeta/lavanda claro
        "side":   (91, 150, 241),     # Azul coral/celeste vivo
    }
    base_rgb = color_map.get(fc_low, (91, 150, 241))
    cn = pose.get("contact_normal") or []
    cn_str = f"n=[{','.join(f'{v:.2f}' for v in cn)}]" if cn else ""
    return _render_orthographic_svg(
        tris_rot, project_side, SIDE_VIEW_DIR_BL, base_rgb, size,
        label="SIDE (lateral, +Y)", cn_str=cn_str, draw_ground=True, scale=scale,
    )


def render_two_views(ref: str, pose: dict, size: int = 200) -> tuple:
    """
    Renderiza la pose con DOS vistas ortográficas (cenital y lateral) que
    comparten el mismo factor de escala, para que las dimensiones físicas
    de la pieza se conserven entre ambas imágenes.
    Retorna (svg_top, svg_side, scale_px_per_unit).
    """
    tris_rot = _prepare_tris(ref, pose)
    scale    = _compute_shared_scale(tris_rot, size)
    svg_top  = render_top_svg(ref, pose, size=size, scale=scale, tris_rot=tris_rot)
    svg_side = render_side_svg(ref, pose, size=size, scale=scale, tris_rot=tris_rot)
    return svg_top, svg_side, scale


def render_3d_svg(ref: str, pose: dict, size: int = 240) -> str:
    """
    Renderiza la pieza en Blender space con el quaternión de la pose.
    El quaternión ya está en Blender space (generado por Blender physics).
    La pieza se muestra reposando: Z mínimo = plano de contacto.
    """
    # Usar _prepare_tris que ya calcula correctamente la rotación determinista y la apoya en Z=0 centrada en X,Y.
    tris_rot = _prepare_tris(ref, pose)

    # 4. Centrar en XY
    all_verts = tris_rot.reshape(-1, 3)
    cx = (all_verts[:, 0].max() + all_verts[:, 0].min()) / 2
    cy = (all_verts[:, 1].max() + all_verts[:, 1].min()) / 2
    tris_rot = tris_rot - np.array([cx, cy, 0.0])

    # 5. Calcular escala para encajar en canvas
    margin = 20
    canvas = size - 2 * margin
    all_verts = tris_rot.reshape(-1, 3)
    all_proj  = isometric_project(all_verts)

    if len(all_proj) == 0:
        return _empty_svg(size)

    mn, mx = all_proj.min(axis=0), all_proj.max(axis=0)
    span = mx - mn
    if span[0] < 1e-6 or span[1] < 1e-6:
        return _empty_svg(size)

    scale  = min(canvas / span[0], canvas / span[1]) * 0.90
    offset = np.array([
        margin + (canvas - span[0] * scale) / 2 - mn[0] * scale,
        margin + (canvas - span[1] * scale) / 2 - mn[1] * scale,
    ])

    # 6. Color base según face_class
    fc_low = (pose.get("face_class") or "side").lower()
    color_map = {
        "top":    (74, 185, 149),     # Verde menta/esmeralda luminoso
        "bottom": (160, 116, 225),    # Violeta/lavanda claro
        "side":   (91, 150, 241),     # Azul coral/celeste vivo
    }
    base_rgb = color_map.get(fc_low, (91, 150, 241))

    # 7. Painter's algorithm — ordenar por profundidad media (Y en Blender ≈ depth)
    def tri_depth(tri):
        # Profundidad en dirección de vista (proyección sobre view_dir)
        centroid = tri.mean(axis=0)
        return float(np.dot(centroid, ISO_VIEW_DIR_BL))

    # Construir lista (triangulo_blender, depth)
    n_tris = tris_rot.shape[0]
    tri_depth_list = [(tris_rot[i], tri_depth(tris_rot[i])) for i in range(n_tris)]
    tri_depth_list.sort(key=lambda x: x[1])  # de fondo hacia frente

    # 8. Renderizar triángulos visibles
    polygons = []
    for tri_3d, _ in tri_depth_list:
        # Normal de la cara en Blender space
        e1 = tri_3d[1] - tri_3d[0]
        e2 = tri_3d[2] - tri_3d[0]
        n  = np.cross(e1, e2)
        n_len = np.linalg.norm(n)
        if n_len < 1e-10:
            continue
        n /= n_len

        # Backface culling correcto en Blender space
        dot_view = float(np.dot(n, ISO_VIEW_DIR_BL))
        if dot_view < 0.0:
            continue  # cara trasera: descartar

        # Iluminación difusa + ambiental
        diff    = max(0.0, float(np.dot(n, LIGHT_DIR_BL)))
        ambient = 0.55
        intensity = ambient + (1.0 - ambient) * diff

        r = min(255, int(base_rgb[0] * intensity))
        g = min(255, int(base_rgb[1] * intensity))
        b = min(255, int(base_rgb[2] * intensity))

        # Borde luminoso para resaltar studs y huecos en 3D
        stroke_str = "rgba(255,255,255,0.75)"

        # Proyectar y escalar a SVG
        pts2d   = isometric_project(tri_3d)
        pts_svg = pts2d * scale + offset
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_svg)

        polygons.append(
            f'<polygon points="{pts_str}" '
            f'fill="rgb({r},{g},{b})" '
            f'stroke="{stroke_str}" '
            f'stroke-width="0.5" stroke-linejoin="round"/>'
        )

    # 9. Plano de suelo (grid sutil)
    ground_y_svg = _ground_line_svg(tris_rot, scale, offset, size)

    # 10. Label contact_normal
    cn = pose.get("contact_normal") or []
    cn_str = f"n=[{','.join(f'{v:.2f}' for v in cn)}]" if cn else ""

    svg = (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:linear-gradient(145deg,#f2f5fa,#e4eaf4);'
        f'border-radius:10px;border:1px solid #c5cedc;">\n'
        f'{ground_y_svg}\n'
        f'{"".join(polygons)}\n'
        f'<text x="4" y="{size-14}" font-size="8.5" fill="#7a8a9e" '
        f'font-family="JetBrains Mono,monospace">{cn_str}</text>\n'
        f'<text x="4" y="{size-4}" font-size="8" fill="#9aabb8" '
        f'font-family="monospace">Blender-space isometric</text>\n'
        f'</svg>'
    )
    return svg


def _ground_line_svg(tris_rot, scale, offset, size):
    """Dibuja una línea de suelo sutil (Z=0) proyectada."""
    try:
        # Proyectar 4 esquinas del plano Z=0
        all_v  = tris_rot.reshape(-1, 3)
        x_ext  = max(abs(all_v[:, 0].max()), abs(all_v[:, 0].min())) * 1.1
        y_ext  = max(abs(all_v[:, 1].max()), abs(all_v[:, 1].min())) * 1.1
        corners = np.array([
            [-x_ext, -y_ext, 0], [x_ext, -y_ext, 0],
            [x_ext,  y_ext, 0], [-x_ext,  y_ext, 0],
        ])
        proj = isometric_project(corners) * scale + offset
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in proj)
        return (f'<polygon points="{pts_str}" '
                f'fill="rgba(200,210,225,0.35)" '
                f'stroke="rgba(150,170,190,0.5)" stroke-width="0.8"/>')
    except Exception:
        return ""


def _empty_svg(size):
    return (
        f'<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:#f2f5fa;border-radius:10px;border:1px solid #ccc;">'
        f'<text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" '
        f'fill="#aab" font-size="11" font-family="sans-serif">Sin malla LDraw</text></svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATEO DE CAMPOS
# ══════════════════════════════════════════════════════════════════════════════

FIELD_LABELS = {
    "pose_index":             "pose_index (BD)",
    "face_class":             "face_class",
    "contact_normal":         "contact_normal",
    "contact_area":           "contact_area (mm²)",
    "contact_stable_length":  "contact_stable_length (mm)",
    "contact_stable_width":   "contact_stable_width (mm)",
    "stability_ratio":        "stability_ratio",
    "lateral_height":         "lateral_height (mm)",
    "zenith_observable_area": "zenith_observable_area (mm²)",
    "zenith_bbox_area":       "zenith_bbox_area (mm²)",
    "orientation_quat":       "orientation_quat [w,x,y,z]",
    "orientation_euler":      "orientation_euler [rx,ry,rz] rad",
    "set_id":                 "set_id",
}

FIELD_ORDER = [
    "pose_index", "face_class",
    "contact_normal", "contact_area",
    "contact_stable_length", "contact_stable_width",
    "stability_ratio",
    "lateral_height", "zenith_observable_area", "zenith_bbox_area",
    "orientation_quat", "orientation_euler",
    "set_id",
]


def fmt_value(key, val):
    if val is None:
        return '<span style="color:#bbb;font-style:italic">null</span>'
    if isinstance(val, list):
        rounded = []
        for v in val:
            if isinstance(v, float):
                rounded.append(f"{v:.4f}")
            else:
                rounded.append(str(v))
        return f'<code>[{", ".join(rounded)}]</code>'
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def face_badge(fc):
    fc_low = (fc or "").lower()
    styles = {
        "top":    ("background:#e0f7fa;color:#006064", "TOP"),
        "bottom": ("background:#ede7f6;color:#4a148c", "BOTTOM"),
        "side":   ("background:#fff3e0;color:#bf360c", "SIDE"),
    }
    style, label = styles.get(fc_low, ("background:#f5f5f5;color:#555", fc.upper()))
    return (f'<span style="{style};padding:3px 9px;border-radius:12px;'
            f'font-size:0.75em;font-weight:700;letter-spacing:0.06em">{label}</span>')


def stability_badge(ratio):
    """Badge de estabilidad con color semántico y barra visual."""
    if ratio is None:
        return '<span style="color:#bbb">—</span>'
    pct = min(100, int(ratio * 100))
    if pct >= 70:
        color = "#2e7d32"
    elif pct >= 30:
        color = "#e65100"
    else:
        color = "#b71c1c"
    return (
        f'<span style="color:{color};font-weight:700">{ratio:.3f}</span>'
        f'&nbsp;<span style="display:inline-block;width:50px;height:5px;'
        f'border-radius:3px;background:#e0e0e0;vertical-align:middle;">'
        f'<span style="display:block;width:{pct}%;height:100%;border-radius:3px;'
        f'background:{color}"></span></span>'
    )


def in_pipeline_badge(pose):
    """Indica si esta pose pasaría los filtros del pipeline YOLO."""
    sr  = pose.get("stability_ratio") or 0.0
    cw  = pose.get("contact_stable_width")
    ok  = sr > 0.05 and (cw is None or cw >= 4.0)
    if ok:
        return ('<span style="background:#e8f5e9;color:#2e7d32;padding:2px 7px;'
                'border-radius:8px;font-size:0.72em;font-weight:600">✓ pipeline</span>')
    reasons = []
    if sr <= 0.05:
        reasons.append(f"stability={sr:.3f}≤0.05")
    if cw is not None and cw < 4.0:
        reasons.append(f"cw={cw:.1f}<4mm")
    tip = "; ".join(reasons)
    return (f'<span style="background:#fce4ec;color:#b71c1c;padding:2px 7px;'
            f'border-radius:8px;font-size:0.72em;font-weight:600" title="{tip}">'
            f'✗ filtrada ({tip})</span>')


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE HTML
# ══════════════════════════════════════════════════════════════════════════════

PAGE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, sans-serif;
    margin: 0; padding: 32px 24px 60px;
    background: radial-gradient(ellipse at top left, #e8edf5 0%, #f0f3f9 60%, #e4e9f2 100%);
    color: #1e2a3a; min-height: 100vh;
}
.page-header {
    max-width: 1300px; margin: 0 auto 32px; padding: 24px 32px;
    background: linear-gradient(135deg, #1a237e 0%, #283593 55%, #1565c0 100%);
    border-radius: 16px; color: white;
    box-shadow: 0 8px 32px rgba(26,35,126,0.22);
    display: flex; align-items: center; gap: 18px;
}
.page-header .emoji { font-size: 2.6em; }
.page-header h1 { margin: 0 0 5px; font-size: 1.7em; font-weight: 700; letter-spacing: -0.02em; }
.page-header .subtitle { margin: 0; font-size: 0.9em; opacity: 0.8; font-weight: 300; }

.poses-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(370px, 1fr));
    gap: 24px; max-width: 1300px; margin: 0 auto;
}

.pose-card {
    background: rgba(255,255,255,0.97); border-radius: 14px; overflow: hidden;
    box-shadow: 0 3px 18px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04);
    border: 1px solid rgba(200,212,230,0.55);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.pose-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 36px rgba(0,0,0,0.11);
}
.pose-card.filtered {
    border-left: 4px solid #ef9a9a;
    opacity: 0.88;
}

.card-header {
    background: linear-gradient(90deg, #f7f8ff, #eef0fb);
    padding: 12px 18px; border-bottom: 1px solid #e8ebf5;
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.pose-number { font-size: 1.1em; font-weight: 700; color: #1a237e; }
.badges { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }

.card-3d {
    display: flex; justify-content: center; align-items: center;
    padding: 16px 14px 8px;
    background: linear-gradient(180deg, #f9faff, #f3f5fb);
}
.card-2views {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
    padding: 14px 12px 10px;
    background: linear-gradient(180deg, #f9faff, #f3f5fb);
}
.view {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
}
.view-label {
    font-size: 0.68em; color: #607d8b; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    font-family: 'JetBrains Mono', monospace;
}
.view svg { width: 100%; height: auto; max-width: 180px; }

.fields-table { width: 100%; border-collapse: collapse; font-size: 0.84em; }
.fields-table td { padding: 6px 15px; border-bottom: 1px solid #f2f4f9; vertical-align: top; }
.fields-table tr:last-child td { border-bottom: none; }
.fields-table .fk { color: #607d8b; font-weight: 500; width: 50%; white-space: nowrap; }
.fields-table .fv { color: #1e2a3a; font-weight: 600;
                     font-family: 'JetBrains Mono', monospace; font-size: 0.9em; word-break: break-all; }
.fields-table tr:hover td { background: #f6f9ff; }
code { background: #eef2f8; padding: 1px 4px; border-radius: 3px; font-size: 0.88em; }
"""


def build_html(ref: str, part_name: str, poses: list) -> str:
    cards = []
    for pose in poses:
        pose_idx = pose.get("pose_index", "?")
        fc       = pose.get("face_class", "?")
        sr       = pose.get("stability_ratio")

        svg_top, svg_side, _scale_px = render_two_views(ref, pose, size=200)

        # Tabla de campos
        rows = []
        for key in FIELD_ORDER:
            if key not in pose:
                continue
            label = FIELD_LABELS.get(key, key)
            val   = pose[key]
            if key == "stability_ratio":
                val_html = stability_badge(val)
            else:
                val_html = fmt_value(key, val)
            rows.append(
                f'<tr><td class="fk">{label}</td><td class="fv">{val_html}</td></tr>'
            )

        # ¿Pasaría el filtro del pipeline?
        pip_badge = in_pipeline_badge(pose)

        # Tarjeta filtrada (borde rojo) si no pasa pipeline
        cw = pose.get("contact_stable_width")
        filtered_cls = "" if ((sr or 0) > 0.05 and (cw is None or cw >= 4.0)) else " filtered"

        cards.append(f"""
<div class="pose-card{filtered_cls}" id="pose-{pose_idx}">
  <div class="card-header">
    <span class="pose-number">Pose #{pose_idx}</span>
    <div class="badges">{face_badge(fc)}{pip_badge}</div>
  </div>
  <div class="card-2views">
    <div class="view"><div class="view-label">Cenital (top)</div>{svg_top}</div>
    <div class="view"><div class="view-label">Lateral (side)</div>{svg_side}</div>
  </div>
  <div>
    <table class="fields-table">{"".join(rows)}</table>
  </div>
</div>
""")

    n = len(poses)
    fc_counts = {}
    for p in poses:
        fc_counts[p.get("face_class", "?")] = fc_counts.get(p.get("face_class", "?"), 0) + 1
    fc_summary = " · ".join(f"{k}: {v}" for k, v in sorted(fc_counts.items()))
    n_pipeline = sum(1 for p in poses
                     if (p.get("stability_ratio") or 0) > 0.05
                     and (p.get("contact_stable_width") is None or p.get("contact_stable_width", 0) >= 4.0))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poses Estables — Pieza {ref} ({part_name})</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header class="page-header">
  <div class="emoji">🧱</div>
  <div>
    <h1>Poses Estables — Pieza {ref}: {part_name}</h1>
    <p class="subtitle">
      Set {SET_ID} &nbsp;·&nbsp; <strong>{n}</strong> poses totales en BD
      &nbsp;·&nbsp; {fc_summary}
      &nbsp;·&nbsp; <span style="color:#a5d6a7">{n_pipeline} pasan filtro pipeline</span>
      &nbsp;·&nbsp; <span style="color:#ef9a9a">{n - n_pipeline} filtradas (borde rojo)</span>
    </p>
  </div>
</header>
<div class="poses-grid">{"".join(cards)}</div>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genera reports HTML de poses estables")
    parser.add_argument("--ref", default=None,
                        help="Generar solo el report de una pieza concreta (ej. 2877)")
    parser.add_argument("--no-index", action="store_true",
                        help="No regenerar el índice (útil cuando se procesa una sola pieza)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limitar a un número aleatorio de piezas procesadas")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[3DPoses] Leyendo TODAS las poses de BD para set {SET_ID} (sin filtro)...")
    poses_by_ref = fetch_all_poses_from_db(SET_ID)

    total_refs  = len(poses_by_ref)
    total_poses = sum(len(v) for v in poses_by_ref.values())
    print(f"[3DPoses] {total_refs} piezas, {total_poses} poses totales")

    if args.ref:
        if args.ref not in poses_by_ref:
            print(f"[ERROR] No hay poses en BD para la pieza '{args.ref}' en set {SET_ID}")
            print(f"        Refs disponibles: {sorted(poses_by_ref.keys())}")
            sys.exit(1)
        refs_to_process = [args.ref]
    else:
        refs_to_process = sorted(poses_by_ref.keys())
        if args.limit > 0 and len(refs_to_process) > args.limit:
            import random
            refs_to_process = random.sample(refs_to_process, args.limit)
            print(f"[INFO] Limitando a {args.limit} piezas seleccionadas al azar.")

    for i, ref in enumerate(refs_to_process, 1):
        poses     = poses_by_ref[ref]
        part_name = PART_NAMES.get(ref, f"Pieza {ref}")
        print(f"  [{i:02d}/{len(refs_to_process)}] {ref} — {part_name} ({len(poses)} poses)")
        html = build_html(ref, part_name, poses)
        out  = os.path.join(OUT_DIR, f"stable_poses_{ref}.html")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"         → {out}")

    if not args.ref and not args.no_index:
        # Filtrar poses_by_ref para incluir solo las procesadas en el índice si hay un límite
        filtered_poses_by_ref = {k: poses_by_ref[k] for k in refs_to_process}
        _generate_index(filtered_poses_by_ref)
    print(f"\n[3DPoses] ✓ {len(refs_to_process)} report(s) generado(s) en: {OUT_DIR}")


def _generate_index(poses_by_ref: dict):
    rows = []
    for ref in sorted(poses_by_ref.keys()):
        poses   = poses_by_ref[ref]
        name    = PART_NAMES.get(ref, f"Pieza {ref}")
        n       = len(poses)
        fcs     = sorted({p.get("face_class", "?") for p in poses})
        fc_html = " ".join(face_badge(fc) for fc in fcs)
        n_pip   = sum(1 for p in poses
                      if (p.get("stability_ratio") or 0) > 0.05
                      and (p.get("contact_stable_width") is None or p.get("contact_stable_width", 0) >= 4.0))
        rows.append(
            f'<tr onclick="window.location=\'stable_poses_{ref}.html\'" style="cursor:pointer">'
            f'<td style="font-weight:700;color:#1a237e">{ref}</td>'
            f'<td>{name}</td>'
            f'<td style="text-align:center">{n}</td>'
            f'<td style="text-align:center"><span style="color:#2e7d32;font-weight:700">{n_pip}</span>'
            f' / <span style="color:#b71c1c">{n-n_pip}</span></td>'
            f'<td>{fc_html}</td>'
            f'<td><a href="stable_poses_{ref}.html" style="color:#1565c0;font-weight:600">Ver →</a></td>'
            f'</tr>'
        )

    total_poses = sum(len(v) for v in poses_by_ref.values())
    idx = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Índice 3D Poses — Set {SET_ID}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
body {{ font-family:'Inter',sans-serif; margin:0; padding:32px 24px;
       background:radial-gradient(ellipse at top,#e8edf5,#f0f3f9); color:#1e2a3a; }}
.hdr {{ max-width:980px; margin:0 auto 26px; padding:22px 30px;
        background:linear-gradient(135deg,#1a237e,#1565c0); color:white;
        border-radius:14px; box-shadow:0 8px 32px rgba(26,35,126,.22); }}
.hdr h1 {{ margin:0 0 5px; font-size:1.5em; font-weight:700; }}
.hdr p  {{ margin:0; opacity:.8; font-size:.88em; }}
table {{ max-width:980px; margin:0 auto; width:100%; border-collapse:collapse;
         background:white; border-radius:12px; overflow:hidden;
         box-shadow:0 4px 18px rgba(0,0,0,.07); }}
th {{ background:#f0f2fc; color:#283593; font-weight:700; padding:11px 16px;
      text-align:left; font-size:.8em; text-transform:uppercase; letter-spacing:.05em; }}
td {{ padding:10px 16px; border-bottom:1px solid #f0f2f9; font-size:.88em; }}
tr:last-child td {{ border-bottom:none; }}
tr:hover td {{ background:#f7f9ff; }}
</style>
</head>
<body>
<div class="hdr">
  <h1>🧱 Índice de Poses 3D — Set {SET_ID}</h1>
  <p>Imperial Troop Transport (Star Wars Rebels)
     &nbsp;·&nbsp; {len(poses_by_ref)} piezas
     &nbsp;·&nbsp; {total_poses} poses totales en BD
  </p>
</div>
<table>
<thead><tr>
  <th>Ref.</th><th>Nombre</th>
  <th style="text-align:center">Total poses</th>
  <th style="text-align:center">Pipeline ✓ / ✗</th>
  <th>Tipos de cara</th><th>Report</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</body>
</html>
"""
    idx_path = os.path.join(OUT_DIR, "index.html")
    with open(idx_path, "w", encoding="utf-8") as fh:
        fh.write(idx)
    print(f"  [Índice] → {idx_path}")


if __name__ == "__main__":
    main()
