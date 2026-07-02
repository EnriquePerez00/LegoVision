# -*- coding: utf-8 -*-
"""
ldraw_color_resolver.py
=======================
Resolver de colores BrickLink para piezas compuestas (LDraw shortcuts cXX).

Aplica la jerarquía pedida por el usuario:

    PARA cada subparte S referenciada en el .dat de la pieza padre P:
      1) Si S tiene LDraw color 16 (placeholder heredable):
           color(S) = color BrickLink del padre (el del inventario)
      2) Si S tiene un color LDraw hardcoded ≠ 16:
           a) Buscar (S.ref) en lego_set_parts (BD Postgres) para set_code → usar su color BL
           b) Si no está → re-scrape on-demand de BrickLink y persistir en lego_set_parts
           c) Si tampoco se encuentra → mapear LDraw color_id → BrickLink color via
              inverso de database/color_catalog.json (campo `ldraw_code`)
           d) Fallback final: color del padre

Cachea cada resolución en la tabla `subpart_color_overrides` para evitar
re-trabajo (lookup determinístico la próxima vez).

Excepción: cualquier ref que pertenezca a una minifigura (sw*, fig-*, etc.)
queda fuera de este flujo — se gestiona en el pipeline propio de minifigs.

Uso:
    from ldraw_color_resolver import resolve_subpart_colors

    color_map = resolve_subpart_colors(
        part_ref="15391",
        parent_bl_color="11",
        set_code="75078-1",
        allow_scrape=True,
    )
    # → {"15391": "11", "15392": "85", "4073": "98"}
"""
from __future__ import annotations

import os
import re
import json
import sys
from typing import Optional, Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────
#  Rutas estándar
# ─────────────────────────────────────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

DEFAULT_LDRAW_ROOTS = [
    os.path.join(_PROJECT_ROOT, "data", "ldraw"),
    "/Applications/Studio 2.0/ldraw",
    os.path.expanduser("~/ldraw"),
]
DEFAULT_LDRAW_SUBDIRS = [
    "UnOfficial/parts", "UnOfficial/p",
    "Unofficial/parts", "Unofficial/p",
    "parts/s", "parts", "p", "models",
]

DEFAULT_COLOR_CATALOG = os.path.join(
    _PROJECT_ROOT, "core", "db", "color_catalog.json"
)

# Refs que NUNCA se procesan por este resolver (se delegan al pipeline minifig).
_MINIFIG_REF_PATTERNS = (
    re.compile(r"^sw\d", re.I),
    re.compile(r"^fig-?\d", re.I),
)


# ═════════════════════════════════════════════════════════════════════════
#  Utilidades de filesystem LDraw
# ═════════════════════════════════════════════════════════════════════════

def find_ldraw_dat(part_ref: str,
                   ldraw_roots: Optional[List[str]] = None) -> Optional[str]:
    """Localiza el .dat de una pieza dada su ref (sin extensión)."""
    roots = [r for r in (ldraw_roots or DEFAULT_LDRAW_ROOTS) if os.path.isdir(r)]
    target = f"{part_ref}.dat".lower()
    fallbacks = [target]
    if len(part_ref) > 1 and part_ref[-1].isalpha() and part_ref[-2].isdigit():
        fallbacks.append(f"{part_ref[:-1]}.dat".lower())

    for root in roots:
        for sub in DEFAULT_LDRAW_SUBDIRS:
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            try:
                files_lc = {f.lower(): f for f in os.listdir(d)}
            except OSError:
                continue
            for fb in fallbacks:
                if fb in files_lc:
                    return os.path.join(d, files_lc[fb])
    return None


_RE_NORMALIZE_BACKSLASH = re.compile(r"[\\/]+")


_PRIMITIVE_PATTERNS = (
    re.compile(r"^\d+-\d+", re.I),    # 4-4cyli, 1-4edge, 2-4disc...
    re.compile(r"^stud\d*[a-z]?$", re.I),   # stud, stud2, stud3, stud2a...
    re.compile(r"^box\d*$", re.I),
    re.compile(r"^rect\d*p?$", re.I),
    re.compile(r"^tri\d*$", re.I),
    re.compile(r"^axl", re.I),
    re.compile(r"^connect", re.I),
    re.compile(r"^logo\d*$", re.I),
)


def _is_ldraw_primitive(ref: str) -> bool:
    """Detecta primitivas geométricas LDraw (no son piezas LEGO con ID BrickLink)."""
    rl = ref.lower()
    return any(pat.match(rl) for pat in _PRIMITIVE_PATTERNS)


def parse_dat_subreferences(dat_path: str) -> List[Tuple[str, str]]:
    """
    Devuelve [(ldraw_color, sub_ref_sin_extension), ...] sólo para subreferencias
    a otras piezas reales (descartando primitivas geométricas como 4-4cyli, stud2…).
    """
    if not dat_path or not os.path.isfile(dat_path):
        return []
    out: List[Tuple[str, str]] = []
    seen: set = set()
    try:
        with open(dat_path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                p = raw.strip().split()
                if not p or p[0] != "1" or len(p) < 15:
                    continue
                ldraw_color = p[1]
                sub_name = " ".join(p[14:])
                clean = _RE_NORMALIZE_BACKSLASH.sub("/", sub_name).strip()
                base = clean.split("/")[-1]
                if not base.lower().endswith(".dat"):
                    continue
                ref = base[:-4]
                if _is_ldraw_primitive(ref):
                    continue
                key = (ldraw_color, ref.lower())
                if key in seen:
                    continue
                seen.add(key)
                out.append((ldraw_color, ref))
    except Exception:
        pass
    return out


def is_minifig_ref(ref: str) -> bool:
    """True si la ref es una minifigura (excluida del resolver)."""
    return any(pat.match(ref) for pat in _MINIFIG_REF_PATTERNS)


# ═════════════════════════════════════════════════════════════════════════
#  Catálogo de colores e índice inverso LDraw → BrickLink
# ═════════════════════════════════════════════════════════════════════════

def load_color_catalog(path: Optional[str] = None) -> dict:
    """Carga `database/color_catalog.json` indexado por BL id (str)."""
    p = path or DEFAULT_COLOR_CATALOG
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print(f"[ldraw_color_resolver] No se pudo leer {p}: {e}")
        return {}


def build_ldraw_to_bl_map(color_catalog: dict) -> Dict[str, str]:
    """Construye índice inverso ldraw_code → bl_id (primer hit gana en colisión)."""
    inv: Dict[str, str] = {}
    for bl_id, info in color_catalog.items():
        ldraw_code = str(info.get("ldraw_code", "")).strip()
        if ldraw_code and ldraw_code not in inv:
            inv[ldraw_code] = str(bl_id)
    return inv


def get_color_info(bl_id: str, color_catalog: dict) -> Dict[str, str]:
    """Devuelve {hex, name, ldraw_code, material_type} para un BL id, o vacío."""
    info = color_catalog.get(str(bl_id), {}) or {}
    hx = info.get("hex", "") or ""
    if hx and not hx.startswith("#"):
        hx = "#" + hx
    return {
        "hex": hx,
        "name": info.get("name", ""),
        "ldraw_code": str(info.get("ldraw_code", "")),
        "material_type": info.get("material_type", "solid"),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Acceso a BD (Postgres / Supabase)
# ═════════════════════════════════════════════════════════════════════════

def _try_import_db():
    """Importa el módulo supabase_client si está disponible."""
    try:
        from core.db import supabase_client as _sc  # type: ignore
        return _sc
    except Exception:
        pass
    try:
        sys.path.insert(0, _PROJECT_ROOT)
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "database"))
        from core.db import supabase_client as _sc  # type: ignore
        return _sc
    except Exception:
        try:
            import supabase_client as _sc  # type: ignore
            return _sc
        except Exception:
            return None


def _normalize_set_code(set_id: str) -> str:
    s = set_id.strip()
    return f"{s}-1" if "-" not in s else s


def get_override_from_db(parent_ref: str, parent_color: str,
                         sub_ref: str) -> Optional[Dict[str, str]]:
    """Lookup en `subpart_color_overrides` (cache de decisiones previas)."""
    db = _try_import_db()
    if db is None:
        return None
    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT sub_color, sub_color_hex, sub_color_name, source
                FROM subpart_color_overrides
                WHERE parent_ref=%s AND parent_color=%s AND sub_ref=%s
                LIMIT 1
            """, (parent_ref, str(parent_color), sub_ref))
            row = cur.fetchone()
            if row:
                return dict(row)
    except Exception as e:
        print(f"[ldraw_color_resolver] get_override_from_db: {e}")
    return None


def save_override_to_db(parent_ref: str, parent_color: str, sub_ref: str,
                        sub_color: str, sub_hex: Optional[str],
                        sub_name: Optional[str], source: str,
                        set_code: Optional[str] = None,
                        minifig_ref: Optional[str] = None) -> None:
    """Persiste/actualiza una resolución en `subpart_color_overrides`."""
    db = _try_import_db()
    if db is None:
        return
    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO subpart_color_overrides
                    (parent_ref, parent_color, sub_ref, sub_color,
                     sub_color_hex, sub_color_name, source, set_code, minifig_ref)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (parent_ref, parent_color, sub_ref) DO UPDATE SET
                    sub_color      = EXCLUDED.sub_color,
                    sub_color_hex  = EXCLUDED.sub_color_hex,
                    sub_color_name = EXCLUDED.sub_color_name,
                    source         = EXCLUDED.source,
                    set_code       = COALESCE(EXCLUDED.set_code, subpart_color_overrides.set_code),
                    minifig_ref    = COALESCE(EXCLUDED.minifig_ref, subpart_color_overrides.minifig_ref),
                    updated_at     = NOW()
            """, (parent_ref, str(parent_color), sub_ref, str(sub_color),
                  sub_hex, sub_name, source, set_code, minifig_ref))
    except Exception as e:
        print(f"[ldraw_color_resolver] save_override_to_db: {e}")


def find_subpart_in_set(sub_ref: str, set_code: str) -> Optional[Dict[str, str]]:
    """Busca (sub_ref, *) en lego_set_parts del set indicado.

    Devuelve {color_code, color_hex, color_name} o None. Si la pieza aparece
    con varios colores en el set, devuelve uno cualquiera (el primero que
    Postgres ordene); en la práctica casi nunca colisiona y no es crítico.
    """
    db = _try_import_db()
    if db is None or not set_code:
        return None
    code = _normalize_set_code(set_code)
    try:
        with db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT color_code, color_hex, color_name
                FROM lego_set_parts
                WHERE set_code = %s AND part_ref = %s
                ORDER BY color_code
                LIMIT 1
            """, (code, sub_ref))
            row = cur.fetchone()
            if row:
                return dict(row)
            # También buscar en piezas de minifiguras del set (minifig_parts).
            cur.execute("""
                SELECT mp.color_code, mp.color_hex, mp.color_name
                FROM minifig_parts mp
                JOIN lego_set_minifigures lsm
                  ON lsm.minifig_ref = mp.minifig_ref
                WHERE lsm.set_code = %s AND mp.part_ref = %s
                ORDER BY mp.color_code
                LIMIT 1
            """, (code, sub_ref))
            row = cur.fetchone()
            if row:
                return dict(row)
    except Exception as e:
        # La tabla minifig_parts puede no existir en BDs antiguas; ignoramos.
        print(f"[ldraw_color_resolver] find_subpart_in_set: {e}")
    return None


# ═════════════════════════════════════════════════════════════════════════
#  Re-scrape on-demand de BrickLink (sólo si allow_scrape=True)
# ═════════════════════════════════════════════════════════════════════════

def rescrape_set_inventory(set_code: str) -> bool:
    """
    Lanza un re-scrape del inventario de BrickLink para `set_code` y persiste
    en BD. Devuelve True si tuvo éxito.
    """
    code = _normalize_set_code(set_code)
    try:
        # Usamos el script existente del proyecto 2camaras_pieza_unica/.
        candidate_modules = [
            "2camaras_pieza_unica.scripts.scrape_bricklink_inventory",
            "scripts.scrape_bricklink_inventory",
        ]
        scraper = None
        for mod in candidate_modules:
            try:
                scraper = __import__(mod, fromlist=["scrape_set", "save_to_db"])
                if hasattr(scraper, "scrape_set"):
                    break
            except Exception:
                continue
        if scraper is None:
            # Probar import directo del fichero por path (solución robusta).
            import importlib.util
            for rel in [
                "2camaras_pieza_unica/scripts/scrape_bricklink_inventory.py",
            ]:
                p = os.path.join(_PROJECT_ROOT, rel)
                if os.path.isfile(p):
                    spec = importlib.util.spec_from_file_location(
                        "scrape_bricklink_inventory", p
                    )
                    if spec and spec.loader:
                        scraper = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(scraper)
                        break
        if scraper is None:
            print("[ldraw_color_resolver] No se encontró el scraper de BrickLink.")
            return False

        data = scraper.scrape_set(code)
        # Persistir en BD (usa la firma original del scraper).
        try:
            scraper.save_to_db(
                code, data["set_name"], data["parts"],
                data["minifigures"], data["minifig_parts"]
            )
        except Exception as e:
            print(f"[ldraw_color_resolver] save_to_db en re-scrape: {e}")
        return True
    except Exception as e:
        print(f"[ldraw_color_resolver] rescrape_set_inventory: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════
#  Función principal: resolve_subpart_colors
# ═════════════════════════════════════════════════════════════════════════

def resolve_subpart_colors(part_ref: str,
                           parent_bl_color: str,
                           set_code: Optional[str] = None,
                           color_catalog: Optional[dict] = None,
                           allow_scrape: bool = True,
                           ldraw_roots: Optional[List[str]] = None,
                           verbose: bool = False) -> Dict[str, str]:
    """
    Devuelve `{sub_ref → bl_color_id}` aplicando la jerarquía completa.

    Siempre incluye el `part_ref` raíz mapeado al `parent_bl_color`.
    Si la pieza no es un shortcut (no tiene subreferencias significativas),
    devuelve simplemente `{part_ref: parent_bl_color}`.
    """
    parent_bl_color = str(parent_bl_color)
    catalog = color_catalog if color_catalog is not None else load_color_catalog()
    inv_map = build_ldraw_to_bl_map(catalog)

    color_map: Dict[str, str] = {part_ref: parent_bl_color}

    # Si es minifig, no procesamos subpartes aquí.
    if is_minifig_ref(part_ref):
        return color_map

    dat_path = find_ldraw_dat(part_ref, ldraw_roots=ldraw_roots)
    if not dat_path:
        return color_map

    subrefs = parse_dat_subreferences(dat_path)
    if not subrefs:
        return color_map  # pieza simple sin subpartes relevantes

    rescrape_attempted = False

    for ldraw_color, sub_ref in subrefs:
        # Regla 1: heredable
        if str(ldraw_color) == "16":
            color_map[sub_ref] = parent_bl_color
            if verbose:
                print(f"  [hereda] {sub_ref} ← {parent_bl_color} (LDraw 16)")
            continue

        # 2.0) Cache previa
        cached = get_override_from_db(part_ref, parent_bl_color, sub_ref)
        if cached:
            color_map[sub_ref] = str(cached["sub_color"])
            if verbose:
                print(f"  [cache] {sub_ref} = {cached['sub_color']} ({cached['source']})")
            continue

        resolved_bl: Optional[str] = None
        source: str = ""

        # 2a) Buscar en lego_set_parts del set
        if set_code:
            inv_hit = find_subpart_in_set(sub_ref, set_code)
            if inv_hit:
                resolved_bl = str(inv_hit["color_code"])
                source = "inventory"

        # 2b) Re-scrape on-demand
        if resolved_bl is None and set_code and allow_scrape and not rescrape_attempted:
            rescrape_attempted = True
            ok = rescrape_set_inventory(set_code)
            if ok:
                inv_hit = find_subpart_in_set(sub_ref, set_code)
                if inv_hit:
                    resolved_bl = str(inv_hit["color_code"])
                    source = "bricklink"

        # 2c) Mapeo LDraw → BL via catálogo
        if resolved_bl is None:
            mapped = inv_map.get(str(ldraw_color))
            if mapped:
                resolved_bl = mapped
                source = "ldraw_map"

        # 2d) Fallback definitivo: color del padre
        if resolved_bl is None:
            resolved_bl = parent_bl_color
            source = "ldraw_map"  # mejor catalogación: era LDraw_map fallido

        color_map[sub_ref] = resolved_bl

        # Guardar override (audit + cache para próximas invocaciones)
        cinfo = get_color_info(resolved_bl, catalog)
        save_override_to_db(
            parent_ref=part_ref,
            parent_color=parent_bl_color,
            sub_ref=sub_ref,
            sub_color=resolved_bl,
            sub_hex=cinfo.get("hex"),
            sub_name=cinfo.get("name"),
            source=source,
            set_code=_normalize_set_code(set_code) if set_code else None,
        )
        if verbose:
            print(f"  [{source}] {sub_ref} ← BL {resolved_bl} (LDraw {ldraw_color})")

    return color_map


# ═════════════════════════════════════════════════════════════════════════
#  CLI de pruebas
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Resuelve colores BrickLink para una pieza compuesta LDraw."
    )
    parser.add_argument("--part_ref", required=True,
                        help="Ref BrickLink de la pieza padre, e.g. 15391")
    parser.add_argument("--color", required=True,
                        help="Color BrickLink del padre (ID), e.g. 11")
    parser.add_argument("--set_code", default=None,
                        help="Código del set BrickLink, e.g. 75078-1 (opcional)")
    parser.add_argument("--no-scrape", action="store_true",
                        help="No realizar re-scrape on-demand de BrickLink")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"\n[Resolver] part={args.part_ref}  color BL={args.color}  "
          f"set={args.set_code or '(ninguno)'}\n")
    color_map = resolve_subpart_colors(
        part_ref=args.part_ref,
        parent_bl_color=args.color,
        set_code=args.set_code,
        allow_scrape=not args.no_scrape,
        verbose=args.verbose,
    )
    print("\n=== RESOLUCIÓN ===")
    for sub_ref, bl_col in color_map.items():
        cinfo = get_color_info(bl_col, load_color_catalog())
        hexv = cinfo.get("hex", "?")
        name = cinfo.get("name", "?")
        print(f"  {sub_ref:<12s} → BL {bl_col:<5s}  {hexv:<8s}  {name}")
