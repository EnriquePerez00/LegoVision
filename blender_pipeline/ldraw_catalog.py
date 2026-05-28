"""
blender_pipeline/ldraw_catalog.py
Indexa el catálogo LDraw completo y genera catalog_index.json
con las piezas relevantes para el training (excluye minifiguras,
pegatinas, etc.)

Uso:
    python blender_pipeline/ldraw_catalog.py --catalog-dir data/ldraw --output data/ldraw/catalog_index.json
    python blender_pipeline/ldraw_catalog.py  # usa defaults de config.py
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Categorías LDraw a INCLUIR (piezas físicas detectables) ────────────────
INCLUDE_CATEGORIES = {
    "Brick", "Plate", "Tile", "Slope", "Wedge", "Arch",
    "Technic", "Panel", "Container", "Door", "Window",
    "Cylinder", "Cone", "Sphere", "Round", "Bracket",
    "Modified", "Train", "Wheel", "Antenna",
}

# ── Categorías / patrones a EXCLUIR ────────────────────────────────────────
EXCLUDE_PATTERNS = [
    r"minifig", r"figure", r"sticker", r"decal",
    r"string", r"rubber", r"band", r"hose",
    r"electric", r"technic\s+pin", r"duplo",
    r"znap", r"scala",
]

# ── Piezas más comunes en sets LEGO (ponderación) ──────────────────────────
# Basado en análisis de sets oficiales LDraw
COMMON_PARTS_WEIGHT = {
    "3001": 10,  # Brick 2x4
    "3002": 9,   # Brick 2x3
    "3003": 9,   # Brick 2x2
    "3004": 8,   # Brick 1x2
    "3005": 7,   # Brick 1x1
    "3010": 8,   # Brick 1x4
    "3009": 7,   # Brick 1x6
    "3008": 6,   # Brick 1x8
    "3020": 9,   # Plate 2x4
    "3021": 8,   # Plate 2x3
    "3022": 8,   # Plate 2x2
    "3023": 7,   # Plate 1x2
    "3024": 6,   # Plate 1x1
    "3034": 7,   # Plate 2x8
    "3035": 6,   # Plate 4x8
    "3031": 7,   # Plate 4x4
    "2420": 6,   # Plate 2x2 Corner
    "3068": 8,   # Tile 2x2
    "3069": 7,   # Tile 1x2
    "3070": 6,   # Tile 1x1
    "6636": 6,   # Tile 1x6
    "4162": 5,   # Tile 1x8
    "3038": 6,   # Slope 45° 2x3
    "3039": 6,   # Slope 45° 2x2
    "3040": 5,   # Slope 45° 1x2
    "3298": 5,   # Slope 33° 3x2
    "3037": 5,   # Slope 45° 2x4
}


def parse_dat_header(filepath: Path) -> dict:
    """
    Lee el header de un archivo .dat de LDraw y extrae:
    - descripción (primera línea, comenta con '0 ')
    - categoría (línea '0 !CATEGORY')
    - palabras clave ('0 !KEYWORDS')
    """
    info = {"description": "", "category": "", "keywords": []}
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("0"):
                    break
                if line.startswith("0 !CATEGORY"):
                    info["category"] = line.replace("0 !CATEGORY", "").strip()
                elif line.startswith("0 !KEYWORDS"):
                    kw = line.replace("0 !KEYWORDS", "").strip()
                    info["keywords"] = [k.strip() for k in kw.split(",")]
                elif line.startswith("0 ") and not info["description"]:
                    desc = line[2:].strip()
                    if desc and not desc.startswith("!"):
                        info["description"] = desc
    except Exception:
        pass
    return info


def is_excluded(name: str, description: str, category: str) -> bool:
    """Devuelve True si la pieza debe excluirse del catálogo de training."""
    text = f"{name} {description} {category}".lower()
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def build_catalog(catalog_dir: str, output_path: str, max_parts: int = 5000):
    """
    Genera catalog_index.json con las piezas más relevantes.

    Estructura del JSON:
    {
        "metadata": {...},
        "classes": [
            {
                "idx": 0,
                "ldraw_id": "3001",
                "name": "Brick 2x4",
                "category": "Brick",
                "dat_path": "parts/3001.dat",
                "weight": 10
            },
            ...
        ]
    }
    """
    catalog_dir = Path(catalog_dir)
    parts_dir = catalog_dir / "parts"

    if not parts_dir.exists():
        raise FileNotFoundError(f"No se encontró parts/ en {catalog_dir}")

    dat_files = list(parts_dir.glob("*.dat"))
    print(f"[LegoVision] Indexando {len(dat_files)} archivos .dat...")

    parts = []
    excluded = 0

    for dat_file in tqdm(dat_files, desc="Indexando"):
        part_id = dat_file.stem  # nombre sin extensión, e.g. "3001"

        # Saltar sub-partes (contienen 's' o 'd' en el nombre)
        if re.search(r'[a-z]', part_id) and part_id not in COMMON_PARTS_WEIGHT:
            # Solo incluir sub-partes si son muy comunes
            continue

        header = parse_dat_header(dat_file)
        description = header["description"]
        category    = header["category"]

        # Excluir piezas no deseadas
        if is_excluded(part_id, description, category):
            excluded += 1
            continue

        # Peso de importancia (más común = mayor weight)
        weight = COMMON_PARTS_WEIGHT.get(part_id, 1)

        parts.append({
            "ldraw_id":   part_id,
            "name":       description or f"Part {part_id}",
            "category":   category or "Unknown",
            "dat_path":   f"parts/{dat_file.name}",
            "keywords":   header["keywords"],
            "weight":     weight,
        })

    # Ordenar por peso (primero las más comunes) y luego por ID
    parts.sort(key=lambda p: (-p["weight"], p["ldraw_id"]))

    # Limitar al máximo configurado
    parts = parts[:max_parts]

    # Asignar índices YOLO correlativos
    for idx, part in enumerate(parts):
        part["idx"] = idx

    # Generar JSON
    catalog = {
        "metadata": {
            "total_classes":  len(parts),
            "total_scanned":  len(dat_files),
            "excluded":       excluded,
            "catalog_dir":    str(catalog_dir),
            "generated_with": "legvision/blender_pipeline/ldraw_catalog.py",
        },
        "classes": parts,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n[LegoVision] ✅ Catálogo generado:")
    print(f"   Clases:   {len(parts)}")
    print(f"   Excluidas: {excluded}")
    print(f"   Output:   {output_path}")

    # Mostrar top-10
    print("\n  Top 10 piezas por frecuencia:")
    for p in parts[:10]:
        print(f"    [{p['idx']:4d}] {p['ldraw_id']:12s} {p['name'][:40]:<40} (w={p['weight']})")

    return catalog


def load_catalog(catalog_path: str) -> dict:
    """Carga el catalog_index.json y devuelve el dict."""
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def get_random_part(catalog: dict, rng=None) -> dict:
    """
    Selecciona una parte aleatoria del catálogo, ponderada por frecuencia.
    """
    import random
    rng = rng or random
    parts   = catalog["classes"]
    weights = [p["weight"] for p in parts]
    return rng.choices(parts, weights=weights, k=1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar catálogo LDraw")
    parser.add_argument("--catalog-dir", default="data/ldraw",
                        help="Directorio raíz del catálogo LDraw")
    parser.add_argument("--output", default="data/ldraw/catalog_index.json",
                        help="Ruta de salida del JSON")
    parser.add_argument("--max-parts", type=int, default=5000,
                        help="Número máximo de clases a incluir")
    args = parser.parse_args()

    build_catalog(args.catalog_dir, args.output, args.max_parts)
