# -*- coding: utf-8 -*-
"""
import_rebrickable_inventory.py
===============================
Imports LEGO set inventories from Rebrickable CSV downloads into the database.
Handles color code mapping between Rebrickable and BrickLink.
Updates both PostgreSQL and core/db/set_catalog.py.
"""

import os
import sys
import gzip
import csv
import json

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.scrape_bricklink_inventory import save_to_db, load_color_catalog, load_colors_csv
from core.db.set_catalog import REAL_SETS

CACHE_DIR = os.path.join(PROJECT_ROOT, "database", "rebrickable_cache")
CATALOG_PY_PATH = os.path.join(PROJECT_ROOT, "core", "db", "set_catalog.py")

SET_IDS = [
    "79006-1", "75075-1", "31037-1", "79012-1", "60008-1", "42075-1", 
    "75052-1", "75105-1", "60093-1", "8088-1", "41104-1", "60082-1", 
    "6212-1", "75072-1"
]

def load_gzip_csv(filename):
    path = os.path.join(CACHE_DIR, filename)
    print(f"Loading {filename}...")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    color_catalog = load_color_catalog() # BrickLink catalog: {bl_id: {name, hex}}
    
    # Map Rebrickable names / hexes to BrickLink IDs
    bl_name_map = {}
    bl_hex_map = {}
    for bl_id, info in color_catalog.items():
        name = info.get("name")
        hex_val = info.get("hex")
        if name:
            bl_name_map[name.lower()] = bl_id
        if hex_val:
            bl_hex_map[hex_val.lower()] = bl_id

    # Add explicit manual overrides for colors to ensure perfect alignment
    color_overrides = {
        "trans-light blue": "15", # Trans-Light Blue
        "trans-orange": "98",     # Trans-Orange
        "trans-red": "17",        # Trans-Red
        "trans-green": "20",      # Trans-Green
        "trans-yellow": "19",     # Trans-Yellow
        "trans-clear": "12",      # Trans-Clear
        "reddish brown": "88",    # Reddish Brown
        "dark bluish gray": "85", # Dark Bluish Gray
        "light bluish gray": "86",# Light Bluish Gray
        "dark gray": "48",        # Dark Gray
        "light gray": "9",        # Light Gray
        "bright light orange": "110", # Bright Light Orange
        "bright light blue": "105", # Bright Light Blue
        "medium blue": "42",      # Medium Blue
        "dark blue": "63",        # Dark Blue
        "lime": "34",             # Lime
        "dark tan": "69",         # Dark Tan
        "flat silver": "95",      # Flat Silver
        "pearl gold": "115",      # Pearl Gold
        "glow in dark opaque": "159", # Glow In Dark Opaque
    }

    # Load Rebrickable datasets
    rb_colors = load_gzip_csv("colors.csv.gz")
    rb_sets = load_gzip_csv("sets.csv.gz")
    rb_inventories = load_gzip_csv("inventories.csv.gz")
    rb_inv_parts = load_gzip_csv("inventory_parts.csv.gz")
    rb_minifigs = load_gzip_csv("minifigs.csv.gz")
    rb_inv_minifigs = load_gzip_csv("inventory_minifigs.csv.gz")

    # Build Rebrickable index structures
    color_by_id = {c["id"]: c for c in rb_colors}
    set_by_num = {s["set_num"]: s for s in rb_sets}
    
    # Get standard inventory id for each set
    inv_by_set = {}
    for inv in rb_inventories:
        set_num = inv["set_num"]
        # Prefer version 1 or first one
        if set_num not in inv_by_set or int(inv["version"]) < int(inv_by_set[set_num]["version"]):
            inv_by_set[set_num] = inv

    # Map fig_num to name
    minifig_names = {m["fig_num"]: m["name"] for m in rb_minifigs}

    print("\nProcessing sets...")
    imported_sets = {}
    for set_id in SET_IDS:
        if set_id not in set_by_num:
            print(f"⚠️ Set {set_id} not found in Rebrickable sets.csv")
            continue
        
        set_meta = set_by_num[set_id]
        set_name = set_meta["name"]
        print(f"\n--- Set {set_id}: {set_name} ---")

        if set_id not in inv_by_set:
            print(f"⚠️ No inventory found for set {set_id}")
            continue

        inv_id = inv_by_set[set_id]["id"]
        
        # 1. Get Parts
        set_parts = []
        for row in rb_inv_parts:
            if row["inventory_id"] != inv_id:
                continue
            
            part_num = row["part_num"]
            rb_color_id = row["color_id"]
            qty = int(row["quantity"])
            
            # Map Color
            color_meta = color_by_id.get(rb_color_id, {})
            rb_color_name = color_meta.get("name", "Unknown")
            rb_color_rgb = color_meta.get("rgb", "FFFFFF")
            rb_color_hex = f"#{rb_color_rgb}"
            
            # Find BL color code
            bl_id = None
            name_lower = rb_color_name.lower()
            if name_lower in color_overrides:
                bl_id = color_overrides[name_lower]
            elif name_lower in bl_name_map:
                bl_id = bl_name_map[name_lower]
            elif rb_color_hex.lower() in bl_hex_map:
                bl_id = bl_hex_map[rb_color_hex.lower()]
            else:
                # Fallback to closest by name
                for bl_name, bid in bl_name_map.items():
                    if name_lower in bl_name or bl_name in name_lower:
                        bl_id = bid
                        break
                if not bl_id:
                    # Fallback to Rebrickable ID
                    bl_id = rb_color_id

            # Enrich part name if needed (from Rebrickable we just keep it simple)
            set_parts.append({
                "ref": part_num,
                "color_code": str(bl_id),
                "color_hex": rb_color_hex,
                "color_name": rb_color_name,
                "qty": qty,
                "name": f"Part {part_num}" # Fallback generic name
            })

        # 2. Get Minifigures
        set_minifigs = []
        minifig_parts_map = {} # We don't have full minifig parts inventory easily in the CSVs, so leave empty/placeholder
        for row in rb_inv_minifigs:
            if row["inventory_id"] != inv_id:
                continue
            mfg_num = row["fig_num"]
            qty = int(row["quantity"])
            name = minifig_names.get(mfg_num, f"Minifigure {mfg_num}")
            set_minifigs.append({
                "ref": mfg_num,
                "name": name,
                "qty": qty
            })
            minifig_parts_map[mfg_num] = []

        # 3. Save to PostgreSQL Database
        save_to_db(set_id, set_name, set_parts, set_minifigs, minifig_parts_map)
        
        # Keep track for updating catalog file
        imported_sets[set_id] = {
            "name": set_name,
            "minifigures": set_minifigs,
            "parts": set_parts
        }

    # 4. Update core/db/set_catalog.py
    print("\nUpdating core/db/set_catalog.py...")
    # Update REAL_SETS dictionary in memory
    for set_id, set_data in imported_sets.items():
        REAL_SETS[set_id] = set_data

    # Serialize REAL_SETS back to set_catalog.py
    catalog_content = "import random\n\n# Base de datos local estática de sets de LEGO para demostración y consistencia\nREAL_SETS = "
    
    # We can format it nicely
    def format_set_catalog(data):
        return json.dumps(data, indent=4, ensure_ascii=False) \
            .replace(": null", ": None") \
            .replace(": true", ": True") \
            .replace(": false", ": False")
            
    catalog_content += format_set_catalog(REAL_SETS) + "\n"
    
    with open(CATALOG_PY_PATH, "w", encoding="utf-8") as f:
        f.write(catalog_content)
    
    print("✓ Successfully updated set_catalog.py with 14 new sets.")

if __name__ == "__main__":
    main()
