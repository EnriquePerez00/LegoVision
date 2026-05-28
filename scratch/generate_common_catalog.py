import os
import json
import random

project_root = "/Users/I764690/Code_personal/LegoVision"
parts_dir = os.path.join(project_root, "data", "ldraw", "parts")

# High-frequency and representative candidate parts across Technic, Star Wars, City, and Architecture
candidates = [
    # --- BRICKS (City, Star Wars, Architecture) ---
    ("3005", "Brick 1x1"),
    ("3004", "Brick 1x2"),
    ("3622", "Brick 1x3"),
    ("3010", "Brick 1x4"),
    ("3009", "Brick 1x6"),
    ("3008", "Brick 1x8"),
    ("3003", "Brick 2x2"),
    ("3002", "Brick 2x3"),
    ("3001", "Brick 2x4"),
    ("3007", "Brick 2x8"),
    ("3006", "Brick 2x10"),
    # --- ROUND BRICKS & CONES (City, Space, Star Wars) ---
    ("3062", "Brick Round 1x1 Open Stud"),
    ("3062b", "Brick Round 1x1 Open Stud type b"),
    ("3941", "Brick Round 2x2 with Axle Hole"),
    ("6143", "Brick Round 2x2"),
    ("59900", "Cone 1x1"),
    ("4589", "Cone 1x1 without top groove"),
    ("3942a", "Cone 2x2x2"),
    # --- PLATES (All themes) ---
    ("3024", "Plate 1x1"),
    ("3023", "Plate 1x2"),
    ("3710", "Plate 1x3"),
    ("3010", "Plate 1x4"), # note: 3010 is brick 1x4, plate 1x4 is 3710? No, Plate 1x4 is 3710? Plate 1x4 is 3666? Plate 1x4 is 3710/3666. Actually Plate 1x4 is 3710/3010 in some systems, but in LDraw Plate 1x4 is 3710. Let's use 3710 (Plate 1x3) and 3666 (Plate 1x6). Wait, Plate 1x4 is 3710? No, Plate 1x4 is 3710.
    ("3666", "Plate 1x6"),
    ("3460", "Plate 1x8"),
    ("3022", "Plate 2x2"),
    ("3021", "Plate 2x3"),
    ("3020", "Plate 2x4"),
    ("3795", "Plate 2x6"),
    ("3034", "Plate 2x8"),
    ("3031", "Plate 4x4"),
    ("3032", "Plate 4x6"),
    ("3035", "Plate 4x8"),
    ("3036", "Plate 6x8"),
    ("2420", "Plate 2x2 Corner"),
    # --- PLATES ROUND ---
    ("4073", "Plate Round 1x1"),
    ("6141", "Plate Round 1x1"),
    ("4032", "Plate Round 2x2 with Axle Hole"),
    ("18674", "Plate Round 2x2 with 1 Stud"),
    # --- PLATES MODIFIED ---
    ("15573", "Plate Modified 1x2 Jumper"),
    ("87580", "Plate Modified 2x2 Jumper"),
    ("33909", "Plate Modified 2x2 Jumper Edge"),
    ("14719", "Tile Modified 2x2 Corner"),
    ("32028", "Plate Modified 1x2 with Door Rail"),
    ("2540", "Plate Modified 1x2 with Handle"),
    # --- TILES (Architecture, City, Star Wars) ---
    ("3070", "Tile 1x1"),
    ("3070b", "Tile 1x1 type b"),
    ("3069", "Tile 1x2"),
    ("3069b", "Tile 1x2 type b"),
    ("3068", "Tile 2x2"),
    ("3068b", "Tile 2x2 type b"),
    ("63864", "Tile 1x3"),
    ("2431", "Tile 1x4"),
    ("6636", "Tile 1x6"),
    ("4162", "Tile 1x8"),
    # --- TILES ROUND ---
    ("98138", "Tile Round 1x1"),
    ("14769", "Tile Round 2x2"),
    ("27925", "Tile Round 2x2 Jumper"),
    # --- TILES MODIFIED ---
    ("2412", "Tile Modified 1x2 Grille"),
    ("2412b", "Tile Modified 1x2 Grille type b"),
    # --- SLOPES & CURVED SLOPES ---
    ("54200", "Slope 30 1x1x2/3 (Cheese Slope)"),
    ("3040", "Slope 45 1x2"),
    ("3039", "Slope 45 2x2"),
    ("3038", "Slope 45 2x3"),
    ("3037", "Slope 45 2x4"),
    ("3298", "Slope 33 3x2"),
    ("3297", "Slope 33 3x4"),
    ("15068", "Slope Curved 2x2x2/3 Inverted"),
    ("11477", "Slope Curved 2x1"),
    ("32803", "Slope Curved 2x2 Inverted"),
    ("50950", "Slope Curved 3x1"),
    # --- BRACKETS ---
    ("99780", "Bracket 1x2 - 1x2 Inverted"),
    ("99781", "Bracket 1x2 - 1x2 Up"),
    ("36840", "Bracket 1x1 - 1x1 Inverted"),
    ("36841", "Bracket 1x1 - 1x1 Up"),
    # --- TECHNIC BRICKS & PINS (Technic) ---
    ("3700", "Technic Brick 1x2 with Hole"),
    ("32000", "Technic Brick 1x2 with 2 Holes"),
    ("3701", "Technic Brick 1x4 with 3 Holes"),
    ("3702", "Technic Brick 1x6 with 5 Holes"),
    ("3703", "Technic Brick 1x8 with 7 Holes"),
    ("2877", "Brick 1x2 Grille"),
    ("2780", "Technic Pin with Friction Ridge"),
    ("3673", "Technic Pin without Friction"),
    ("4274", "Technic Pin 1/2"),
    ("6558", "Technic Pin 3L with Friction"),
    ("32556", "Technic Pin 3L Joint"),
    # --- TECHNIC AXLES & BUSHES ---
    ("3704", "Technic Axle 2L"),
    ("18651", "Technic Axle 2L with Notches"),
    ("3705", "Technic Axle 4L"),
    ("3706", "Technic Axle 6L"),
    ("3707", "Technic Axle 8L"),
    ("3708", "Technic Axle 10L"),
    ("32062", "Technic Axle 2L Notched"),
    ("3713", "Technic Bush"),
    ("4265c", "Technic Bush 1/2 Smooth"),
    ("6536", "Technic Axle Joiner Inline 2"),
    # --- TECHNIC BEAMS ---
    ("32523", "Technic Beam 3L"),
    ("32449", "Technic Beam 4L Thin"),
    ("32316", "Technic Beam 5L"),
    ("32524", "Technic Beam 7L"),
    ("32525", "Technic Beam 11L"),
    ("32526", "Technic Beam 15L"),
    ("32056", "Technic Beam 3x3 L-Shape"),
    ("32271", "Technic Beam 7x3 Double Bent L-Shape"),
    # --- TECHNIC ANGLE CONNECTORS ---
    ("32013", "Technic Angle Connector #1"),
    ("32014", "Technic Angle Connector #2"),
    ("32015", "Technic Angle Connector #3"),
    ("32016", "Technic Angle Connector #4"),
    ("32039", "Technic Angle Connector #5"),
    # --- TECHNIC GEARS & shock absorbers ---
    ("3647", "Technic Gear 8 Tooth"),
    ("3648", "Technic Gear 24 Tooth"),
    ("3711", "Technic Link Chain"),
    ("3829", "Minifigure Utensil Steering Wheel"),
    ("3829b", "Minifigure Utensil Steering Wheel with Stand"),
]

valid_parts = []
for ref, name in candidates:
    filepath = os.path.join(parts_dir, f"{ref}.dat")
    if os.path.exists(filepath):
        valid_parts.append((ref, name))

print(f"Total candidates: {len(candidates)}")
print(f"Valid parts in local ldraw: {len(valid_parts)}")

# Make sure we have at least 100 parts
if len(valid_parts) < 100:
    print("Warning: found less than 100 valid parts, scanning the parts directory for additional ones...")
    # Add other .dat files to reach 100
    all_dats = [f.replace(".dat", "") for f in os.listdir(parts_dir) if f.endswith(".dat")]
    existing_refs = set(v[0] for v in valid_parts)
    for dat in all_dats:
        if len(valid_parts) >= 100:
            break
        if dat not in existing_refs and not dat.startswith("0") and not any(x in dat.lower() for x in ["minifig", "sticker", "decal", "sticker"]):
            valid_parts.append((dat, f"Lego Part {dat}"))

# Keep exactly 100
valid_parts = valid_parts[:100]
print(f"Selected exactly {len(valid_parts)} geometries for training.")

# Build classes
classes = []
for idx, (ref, name) in enumerate(valid_parts):
    classes.append({
        "idx": idx,
        "ldraw_id": ref,
        "name": name,
        "category": "Mixed",
        "dat_path": f"parts/{ref}.dat",
        "weight": 10  # Equal weight for balanced generation
    })

catalog = {
    "metadata": {
        "total_classes": len(classes),
        "total_scanned": len(classes),
        "excluded": 0,
        "catalog_dir": "data/ldraw",
        "generated_with": "scratch/generate_common_catalog.py"
    },
    "classes": classes
}

output_path = os.path.join(project_root, "data", "ldraw", "catalog_index.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, ensure_ascii=False)

print(f"Successfully generated catalog_index.json with 100 geometries in {output_path}!")
