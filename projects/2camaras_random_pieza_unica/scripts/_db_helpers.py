#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_db_helpers.py - Funciones para consultar BD dinámicamente
Los nombres de colores se resuelven desde color_catalog.json (fuente única de verdad).
"""
import os, sys, json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
from core.db.supabase_client import get_connection

# Cargar color_catalog.json una vez
_COLOR_CATALOG = None
def _get_color_catalog():
    global _COLOR_CATALOG
    if _COLOR_CATALOG is None:
        cat_path = os.path.join(legovic_root, "database", "color_catalog.json")
        try:
            with open(cat_path, "r") as f:
                _COLOR_CATALOG = json.load(f)
        except Exception as e:
            print(f"[_db_helpers] No se pudo cargar color_catalog.json: {e}")
            _COLOR_CATALOG = {}
    return _COLOR_CATALOG

def get_color_name(color_code):
    """Obtiene el nombre del color desde color_catalog.json por su código."""
    cat = _get_color_catalog()
    code_str = str(color_code)
    if code_str in cat:
        return cat[code_str].get("name", f"Color {code_str}")
    return f"Color {code_str}"

def get_color_hex_from_catalog(color_code):
    """Obtiene el hex canónico del color desde color_catalog.json."""
    cat = _get_color_catalog()
    code_str = str(color_code)
    if code_str in cat:
        return cat[code_str].get("hex", "").lstrip("#")
    return ""

def get_all_ref_color_combinations_from_db():
    """Obtiene combos (ref, color_code, color_hex) de piece_embeddings.
    Nombres resueltos desde color_catalog.json."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT part_ref as ref, color_code, color_hex
                    FROM piece_embeddings
                    ORDER BY part_ref, color_code
                """)
                rows = cur.fetchall()
                return [{
                    "ref": r["ref"],
                    "color_code": r["color_code"],
                    "color_hex": r["color_hex"],
                    "color_name": get_color_name(r["color_code"])
                } for r in rows]
    except Exception as e:
        print(f"[_db_helpers] Error: {e}")
        return []

def get_unique_colors_from_db():
    """Obtiene colores únicos (code, hex) de piece_embeddings.
    Nombres resueltos desde color_catalog.json."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT color_code as code, color_hex as hex, COUNT(DISTINCT part_ref) as n_refs
                    FROM piece_embeddings
                    GROUP BY color_code, color_hex ORDER BY color_code
                """)
                rows = cur.fetchall()
                return [{
                    "code": r["code"], "hex": r["hex"],
                    "name": get_color_name(r["code"]), "n_refs": r["n_refs"]
                } for r in rows]
    except Exception as e:
        print(f"[_db_helpers] Error: {e}")
        return []

if __name__ == "__main__":
    combos = get_all_ref_color_combinations_from_db()
    print(f"Total combos: {len(combos)}")
    colors = get_unique_colors_from_db()
    print(f"Total colors: {len(colors)}")
    # Mostrar top colores
    from collections import Counter
    cnt = Counter(c['code'] for c in colors)
    for code, n in sorted(cnt.items(), key=lambda x: x[0])[:10]:
        name = get_color_name(code)
        print(f"  code={code:5s} ({name:30s}): {n} hex variantes")
