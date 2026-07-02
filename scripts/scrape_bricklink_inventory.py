# -*- coding: utf-8 -*-
"""
scrape_bricklink_inventory.py
=============================
Scrapea el inventario de un set en BrickLink (sin login) desde la URL pública
`https://www.bricklink.com/catalogItemInv.asp?S={set_id}` y, opcionalmente, el
inventario de cada minifigura del set desde
`https://www.bricklink.com/catalogItemInv.asp?M={minifig_ref}`.

⚠️  IMPORTANTE
    Sólo se consideran los ítems bajo la sección **"Regular Items"**.
    Las secciones **"Extra Items"**, **"Counterparts"** y **"Alternate Items"**
    se ignoran por completo (no son parte real del inventario del set).

Para cada pieza extrae: ref, color_code (BrickLink ID), color_hex, color_name,
qty y nombre. Mapea el color usando `database/color_catalog.json`.

Salida:
  - Persiste en BD (lego_sets, lego_set_parts, lego_set_minifigures, minifig_parts)
  - Marca en lego_set_parts las piezas que también figuran en alguna minifig
    con `is_minifig_part=TRUE` y `minifig_ref=<ref>`
  - Reescribe el bloque del set en `database/set_catalog.py` (raíz LegoVision/)

Uso:
  .venv/bin/python scripts/scrape_bricklink_inventory.py \\
      --set 75078-1 [--no-db] [--no-update-catalog]
"""

import os
import sys
import re
import json
import time
import argparse
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
# Este script vive en `scripts/` del repositorio raíz (LegoVision/) y siempre
# escribe sobre `database/set_catalog.py` y `database/color_catalog.json` de la
# raíz como única fuente de verdad para inventarios de sets / piezas / colores.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)            # repo raíz LegoVision/

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "database"))

import requests
from bs4 import BeautifulSoup

from core.db import supabase_client  # noqa: E402
from core.db.supabase_client import get_connection  # type: ignore  # noqa: E402

CATALOG_PY_PATH = os.path.join(PROJECT_ROOT, "core", "db", "set_catalog.py")
COLOR_CATALOG_PATH = os.path.join(PROJECT_ROOT, "core", "db", "color_catalog.json")
COLORS_CSV_PATH = os.path.join(PROJECT_ROOT, "database", "colors.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

INV_URL_SET    = "https://www.bricklink.com/catalogItemInv.asp?S={code}"
INV_URL_MINFIG = "https://www.bricklink.com/catalogItemInv.asp?M={ref}"
SET_PAGE_URL   = "https://www.bricklink.com/v2/catalog/catalogitem.page?S={code}"


# ══════════════════════════════════════════════════════════════════════════════
#  Color catalog
# ══════════════════════════════════════════════════════════════════════════════

def load_color_catalog() -> dict:
    """Devuelve dict {bl_id: {name, hex, ...}} desde color_catalog.json."""
    try:
        with open(COLOR_CATALOG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print(f"[WARN] No se pudo leer {COLOR_CATALOG_PATH}: {e}")
        return {}


def load_colors_csv() -> dict:
    """Carga colors.csv de Rebrickable {id: {name, hex}}."""
    import csv
    colors = {}
    if not os.path.exists(COLORS_CSV_PATH):
        return colors
    try:
        with open(COLORS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                colors[row["id"]] = {
                    "name": row["name"],
                    "hex": f"#{row['rgb']}"
                }
    except Exception as e:
        print(f"[WARN] No se pudo leer {COLORS_CSV_PATH}: {e}")
    return colors



# ══════════════════════════════════════════════════════════════════════════════
#  Scraping
# ══════════════════════════════════════════════════════════════════════════════

def fetch_html(url: str, retries: int = 3, delay: float = 1.5) -> str:
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(delay)
    raise RuntimeError(f"fetch_html falló para {url}: {last_err}")


# Regex para los enlaces a piezas/minifigs en la página de inventario.
#   /v2/catalog/catalogitem.page?P=<ref>&idColor=<id>      (parts)
#   /v2/catalog/catalogitem.page?M=<ref>                   (minifigs)
RE_PART_LINK    = re.compile(r"P=([A-Za-z0-9_]+)&idColor=([0-9]+)", re.I)
RE_MINIFIG_LINK = re.compile(r"M=([A-Za-z0-9_]+)", re.I)


def _select_inventory_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Devuelve la <table> con más enlaces a `catalogitem.page?P=...&idColor=...`
    o `?M=...` (heurística para localizar la tabla de inventario).
    """
    best, best_n = None, 0
    for t in soup.find_all("table"):
        n = 0
        for a in t.find_all("a", href=True):
            h = a["href"]
            if "catalogitem.page" in h.lower() and ("idColor" in h or "M=" in h):
                n += 1
        if n > best_n:
            best, best_n = t, n
    return best


def _classify_section(text: str, prev: str) -> str:
    """
    Determina la sección top-level a la que pertenece la fila siguiente,
    a partir del texto de un heading.
        'Regular Items:'   → 'regular'
        'Extra Items:'     → 'extra'
        'Counterparts:'    → 'counterparts'
        'Alternate Items:' → 'alternate'
    Sub-headings ('Parts:', 'Minifigures:') no alteran el estado.
    """
    low = (text or "").strip().lower().rstrip(":")
    if low == "regular items":
        return "regular"
    if low == "extra items":
        return "extra"
    if low == "counterparts":
        return "counterparts"
    if low == "alternate items":
        return "alternate"
    return prev


def _iter_inventory_rows(table) -> list:
    """
    Itera la tabla de inventario y devuelve sólo las filas que pertenecen a
    "Regular Items". Cada elemento es (kind, ref, color_id, qty, desc_clean)
    donde kind in {'part', 'minifig'}.
    """
    out = []
    section = "unknown"
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        text = tr.get_text(" ", strip=True)
        if len(cells) <= 2:
            section = _classify_section(text, section)
            continue
        if section != "regular":
            continue
        if len(cells) < 4:
            continue
        item_a = None
        for a in tr.find_all("a", href=True):
            if "catalogitem.page" in a["href"].lower():
                item_a = a
                break
        if not item_a:
            continue
        href = item_a["href"]
        m_part = RE_PART_LINK.search(href)
        m_minfig = RE_MINIFIG_LINK.search(href) if not m_part else None
        try:
            qty = int(cells[1].get_text(strip=True) or "0")
        except ValueError:
            qty = 0
        desc = cells[3].get_text(" ", strip=True)
        desc_clean = re.split(r"\s+Catalog\s*:", desc, maxsplit=1)[0].strip()
        if m_part:
            out.append(("part", m_part.group(1), m_part.group(2), qty, desc_clean))
        elif m_minfig:
            out.append(("minifig", m_minfig.group(1), None, qty, desc_clean))
    return out


def parse_set_inventory(html: str) -> dict:
    """Parsea inventario del set; devuelve {'parts':[...], 'minifigures':[...]}."""
    soup = BeautifulSoup(html, "lxml")
    table = _select_inventory_table(soup)
    if table is None:
        raise RuntimeError("No se encontró la tabla de inventario.")
    parts: list = []
    minifigs: list = []
    for kind, ref, color_id, qty, name in _iter_inventory_rows(table):
        if kind == "part":
            parts.append({"ref": ref, "color_code": color_id, "qty": qty, "name": name})
        elif kind == "minifig":
            minifigs.append({"ref": ref, "qty": qty, "name": name})
    return {"parts": parts, "minifigures": minifigs}


def parse_minifig_inventory(html: str) -> list:
    """Parsea inventario de una minifig; devuelve sólo parts."""
    soup = BeautifulSoup(html, "lxml")
    table = _select_inventory_table(soup)
    if table is None:
        return []
    out = []
    for kind, ref, color_id, qty, name in _iter_inventory_rows(table):
        if kind == "part":
            out.append({"ref": ref, "color_code": color_id, "qty": qty, "name": name})
    return out


def get_set_name(html: str, fallback: str = "") -> str:
    """
    Intenta extraer el nombre del set desde la página de inventario:
      - <title> del estilo "Catalog : Set : ... : <name> | BrickLink"
      - atributo TITLE de la imagen del set: 'Set No: NNNN  Name: <name>'
    """
    # 1) Atributo TITLE de la imagen (es muy fiable)
    m = re.search(r'TITLE\s*=\s*["\']\s*Set No:\s*\S+\s+Name:\s*([^"\']+)["\']',
                  html, flags=re.I)
    if m:
        return m.group(1).strip()
    # 2) <title> de la página
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("title")
    if title:
        t = title.get_text()
        m = re.search(r":\s*(.+?)\s*-\s*BrickLink", t)
        if m:
            return m.group(1).strip()
        m = re.search(r":\s*(.+?)\s*\|\s*BrickLink", t)
        if m:
            return m.group(1).strip()
    return fallback


def get_minifig_name(html: str, fallback: str = "") -> str:
    return get_set_name(html, fallback)


# ══════════════════════════════════════════════════════════════════════════════
#  Enriquecimiento (color_hex / color_name) y limpieza de nombre
# ══════════════════════════════════════════════════════════════════════════════

def enrich_part(part: dict, color_catalog: dict, rebrickable_colors: dict = None) -> dict:
    """Añade color_hex/color_name oficiales y quita el prefijo de color del name."""
    bl_id = str(part.get("color_code", ""))
    info = color_catalog.get(bl_id) or {}
    color_name = info.get("name", "") or ""
    color_hex  = info.get("hex", "") or ""

    # Si el color es desconocido o genérico en color_catalog, inferir de colors.csv
    if (not color_name or color_name == "Unknown Color" or color_name == "Various") and rebrickable_colors:
        if bl_id in rebrickable_colors:
            info_rb = rebrickable_colors[bl_id]
            color_name = info_rb["name"]
            color_hex = info_rb["hex"]

    raw_name = (part.get("name") or "").strip()
    cleaned  = raw_name
    if color_name and raw_name.lower().startswith(color_name.lower()):
        cleaned = raw_name[len(color_name):].strip()


    return {
        "ref":        part["ref"],
        "color_code": bl_id,
        "color_hex":  color_hex,
        "color_name": color_name,
        "qty":        int(part.get("qty", 1)),
        "name":       cleaned,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Persistencia en BD
# ══════════════════════════════════════════════════════════════════════════════

def save_to_db(set_id: str, set_name: str, parts: list, minifigs: list,
               minifig_parts_map: dict):
    """
    Guarda en la BD:
      - lego_sets               (UPSERT)
      - lego_set_parts          (DELETE + INSERT, marcando is_minifig_part)
      - lego_set_minifigures    (DELETE + INSERT)
      - minifig_parts           (DELETE + INSERT por minifig_ref)
    """
    minifig_ref_by_part = {}
    for mref, mparts in minifig_parts_map.items():
        for mp in mparts:
            key = (str(mp["ref"]), str(mp["color_code"]))
            minifig_ref_by_part.setdefault(key, mref)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lego_sets (code, name)
            VALUES (%s, %s)
            ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
        """, (set_id, set_name))

        cur.execute("DELETE FROM lego_set_parts WHERE set_code = %s", (set_id,))
        for p in parts:
            key = (str(p["ref"]), str(p["color_code"]))
            mref = minifig_ref_by_part.get(key)
            cur.execute("""
                INSERT INTO lego_set_parts
                    (set_code, part_ref, color_code, color_hex, color_name, qty,
                     is_minifig_part, minifig_ref)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (set_code, part_ref, color_code) DO UPDATE SET
                    color_hex       = EXCLUDED.color_hex,
                    color_name      = EXCLUDED.color_name,
                    qty             = EXCLUDED.qty,
                    is_minifig_part = EXCLUDED.is_minifig_part,
                    minifig_ref     = EXCLUDED.minifig_ref
            """, (set_id, p["ref"], p["color_code"], p.get("color_hex"),
                  p.get("color_name"), p["qty"], bool(mref), mref))

        cur.execute("DELETE FROM lego_set_minifigures WHERE set_code = %s",
                    (set_id,))
        for m in minifigs:
            cur.execute("""
                INSERT INTO lego_set_minifigures (set_code, minifig_ref, name, qty)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (set_code, minifig_ref) DO UPDATE SET
                    name = EXCLUDED.name, qty = EXCLUDED.qty
            """, (set_id, m["ref"], m["name"], m["qty"]))

        for mref, mparts in minifig_parts_map.items():
            cur.execute("DELETE FROM minifig_parts WHERE minifig_ref = %s",
                        (mref,))
            for mp in mparts:
                cur.execute("""
                    INSERT INTO minifig_parts
                        (minifig_ref, part_ref, color_code, color_hex,
                         color_name, qty, name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (minifig_ref, part_ref, color_code) DO UPDATE SET
                        color_hex  = EXCLUDED.color_hex,
                        color_name = EXCLUDED.color_name,
                        qty        = EXCLUDED.qty,
                        name       = EXCLUDED.name
                """, (mref, mp["ref"], mp["color_code"], mp.get("color_hex"),
                      mp.get("color_name"), mp["qty"], mp.get("name")))

    print(f"[DB] Guardado set {set_id}: {len(parts)} parts, "
          f"{len(minifigs)} minifigs, "
          f"{sum(len(v) for v in minifig_parts_map.values())} piezas de minifig")


# ══════════════════════════════════════════════════════════════════════════════
#  Actualización del fichero set_catalog.py
# ══════════════════════════════════════════════════════════════════════════════

def update_set_catalog_py(set_id: str, set_name: str, parts: list,
                          minifigs: list, minifig_parts_map: dict):
    """
    Reemplaza el bloque del set indicado en `set_catalog.py` por el nuevo
    inventario, preservando el resto de sets. La estrategia es localizar
    `"<set_id>": {` y balancear llaves para encontrar el cierre.
    """
    src = open(CATALOG_PY_PATH, "r", encoding="utf-8").read()

    minifig_ref_by_part = {}
    for mref, mparts in minifig_parts_map.items():
        for mp in mparts:
            minifig_ref_by_part.setdefault((mp["ref"], mp["color_code"]), mref)

    new_set = {
        "name":        set_name,
        "minifigures": [
            {"ref": m["ref"], "name": m["name"], "qty": m["qty"]}
            for m in minifigs
        ],
        "parts": [
            {
                "ref":        p["ref"],
                "color_code": p["color_code"],
                "color_hex":  p.get("color_hex", ""),
                "color_name": p.get("color_name", ""),
                "qty":        p["qty"],
                "name":       p.get("name", ""),
                **(
                    {"is_minifig_part": True,
                     "minifig_ref": minifig_ref_by_part[(p["ref"], p["color_code"])]}
                    if (p["ref"], p["color_code"]) in minifig_ref_by_part
                    else {}
                ),
            }
            for p in parts
        ],
    }

    block_repr = json.dumps(new_set, indent=4, ensure_ascii=False)
    block_repr = (block_repr
                  .replace(": null", ": None")
                  .replace(": true", ": True")
                  .replace(": false", ": False"))

    pattern = re.compile(r'("' + re.escape(set_id) + r'"\s*:\s*)\{', re.M)
    m = pattern.search(src)
    if not m:
        raise RuntimeError(
            f"No se encontró el bloque del set {set_id} en {CATALOG_PY_PATH}"
        )

    start = m.end() - 1
    depth = 0
    end = None
    in_str = False
    str_ch = None
    i = start
    while i < len(src):
        ch = src[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == str_ch:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                str_ch = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1
    if end is None:
        raise RuntimeError(
            f"No se pudo balancear el bloque del set {set_id}"
        )

    new_src = src[:start] + block_repr + src[end:]
    with open(CATALOG_PY_PATH, "w", encoding="utf-8") as fh:
        fh.write(new_src)
    print(f"[Catalog] {CATALOG_PY_PATH} actualizado para el set {set_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def scrape_set(set_id: str) -> dict:
    """Orquesta scraping del set y de cada minifig. Devuelve dict completo."""
    color_catalog = load_color_catalog()
    rebrickable_colors = load_colors_csv()

    print(f"[Scrape] Set {set_id} → {INV_URL_SET.format(code=set_id)}")
    html_set = fetch_html(INV_URL_SET.format(code=set_id))
    raw      = parse_set_inventory(html_set)
    set_name = get_set_name(html_set, fallback=f"Set {set_id}")
    if not set_name or set_name.startswith("Set "):
        try:
            html2 = fetch_html(SET_PAGE_URL.format(code=set_id))
            set_name = get_set_name(html2, fallback=set_name)
        except Exception:
            pass

    print(f"[Scrape] Set name: {set_name!r}")
    print(f"[Scrape] Regular Items → {len(raw['parts'])} parts, "
          f"{len(raw['minifigures'])} minifigs")

    parts_enriched = [enrich_part(p, color_catalog, rebrickable_colors) for p in raw["parts"]]
    minifigs       = list(raw["minifigures"])

    # Inventario de cada minifig
    minifig_parts_map: dict = {}
    for mfg in minifigs:
        mref = mfg["ref"]
        try:
            print(f"[Scrape] Minifig {mref} → {INV_URL_MINFIG.format(ref=mref)}")
            html_m = fetch_html(INV_URL_MINFIG.format(ref=mref))
            raw_m  = parse_minifig_inventory(html_m)
            mparts = [enrich_part(p, color_catalog, rebrickable_colors) for p in raw_m]
            minifig_parts_map[mref] = mparts
            print(f"[Scrape]   {mref}: {len(mparts)} piezas")
        except Exception as e:
            print(f"[WARN] Minifig {mref}: {e}")
            minifig_parts_map[mref] = []
        time.sleep(0.5)

    return {
        "set_id":       set_id,
        "set_name":     set_name,
        "parts":        parts_enriched,
        "minifigures":  minifigs,
        "minifig_parts": minifig_parts_map,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="75078-1", help="Código del set")
    parser.add_argument("--no-db", action="store_true",
                        help="No guardar en BD")
    parser.add_argument("--no-update-catalog", action="store_true",
                        help="No reescribir set_catalog.py")
    parser.add_argument("--dump-json", default="",
                        help="Volcar el resultado a este JSON")
    args = parser.parse_args()

    data = scrape_set(args.set)

    if args.dump_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_json)) or ".",
                    exist_ok=True)
        with open(args.dump_json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"[Dump] JSON → {args.dump_json}")

    if not args.no_db:
        save_to_db(args.set, data["set_name"], data["parts"],
                   data["minifigures"], data["minifig_parts"])

    if not args.no_update_catalog:
        update_set_catalog_py(args.set, data["set_name"], data["parts"],
                              data["minifigures"], data["minifig_parts"])

    print("\n[Scrape] ✓ Completado")


if __name__ == "__main__":
    main()
