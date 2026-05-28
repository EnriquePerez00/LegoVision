import json
import os

parsed_json_path = "/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/scratch/parsed_sets.json"
catalog_path = "/Users/I764690/Code_personal/LegoVision/database/set_catalog.py"

with open(parsed_json_path, "r") as f:
    parsed_data = json.load(f)

# Read the original file
with open(catalog_path, "r") as f:
    lines = f.readlines()

# Let's rebuild REAL_SETS programmatically or replace it.
# To keep it extremely simple, let's write the entire database/set_catalog.py file from scratch with the clean REAL_SETS.
# We will keep set "75078-1" and set "10692-1" as well.

original_75078 = {
    "name": "Imperial Troop Transport (Star Wars Rebels)",
    "minifigures": [
        {"ref": "sw0614", "name": "Stormtrooper (Rebels) with Azure Vents", "qty": 4}
    ],
    "parts": [
        {"ref": "3004", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 8},
        {"ref": "3001", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 2},
        {"ref": "3020", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 4},
        {"ref": "3022", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 12},
        {"ref": "2877", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 4},
        {"ref": "59900", "color_code": "36", "color_hex": "#C91A09", "color_name": "Trans-Red", "qty": 4},
        {"ref": "3003", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 2}
    ]
}

# Combine
all_sets = {
    "75078-1": original_75078,
}

for set_id, data in parsed_data.items():
    all_sets[set_id] = data

# Write the new set_catalog.py
new_content = """import random

# Base de datos local estática de sets de LEGO para demostración y consistencia
REAL_SETS = {
"""

for set_id, data in all_sets.items():
    new_content += f'    "{set_id}": {{\n'
    new_content += f'        "name": {repr(data["name"])},\n'
    new_content += f'        "minifigures": {json.dumps(data["minifigures"], indent=12).replace("null", "None")},\n'
    new_content += f'        "parts": {json.dumps(data["parts"], indent=12).replace("null", "None")}\n'
    new_content += '    },\n'

# Add the 10692-1 classic set as well
classic_parts = [
    {"ref": "3001", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 4},
    {"ref": "3002", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 4},
    {"ref": "3003", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 4},
    {"ref": "3004", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 6},
    {"ref": "3005", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 10},
    {"ref": "3010", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 4},
    {"ref": "3009", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 2},
    {"ref": "3008", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 2},
    {"ref": "3020", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 6},
    {"ref": "3021", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 6},
    {"ref": "3022", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 8},
    {"ref": "3023", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 12},
    {"ref": "3024", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 10},
    {"ref": "3034", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 2},
    {"ref": "3035", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 2},
    {"ref": "3031", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 4},
    {"ref": "2420", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 4},
    {"ref": "3068", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 4},
    {"ref": "3069", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 6},
    {"ref": "3070", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 8},
    {"ref": "6636", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 4},
    {"ref": "4162", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 2},
    {"ref": "3038", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 4},
    {"ref": "3039", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 6},
    {"ref": "3040", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 6},
    {"ref": "3298", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 4},
    {"ref": "3037", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 4},
    {"ref": "2412", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 8},
    {"ref": "3710", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 10},
    {"ref": "3622", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 4},
    {"ref": "3666", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 6},
    {"ref": "3795", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 4},
    {"ref": "4073", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 12},
    {"ref": "3062", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 6},
    {"ref": "22885", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 4},
    {"ref": "32000", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 4},
    {"ref": "3700", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 4},
    {"ref": "2877", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 6},
    {"ref": "3659", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 2},
    {"ref": "6141", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 12},
    {"ref": "15573", "color_code": "15", "color_hex": "#FFFFFF", "color_name": "White", "qty": 8},
    {"ref": "14719", "color_code": "2", "color_hex": "#00AA00", "color_name": "Green", "qty": 4},
    {"ref": "18674", "color_code": "4", "color_hex": "#C91A09", "color_name": "Red", "qty": 4},
    {"ref": "32013", "color_code": "1", "color_hex": "#0A3C9F", "color_name": "Blue", "qty": 4},
    {"ref": "6536", "color_code": "14", "color_hex": "#F2CD37", "color_name": "Yellow", "qty": 4},
    {"ref": "4274", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 10},
    {"ref": "3673", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 10},
    {"ref": "2780", "color_code": "0", "color_hex": "#1B1B1B", "color_name": "Black", "qty": 15},
    {"ref": "3705", "color_code": "84", "color_hex": "#5A5A5A", "color_name": "Dark Bluish Gray", "qty": 4},
    {"ref": "3003", "color_code": "85", "color_hex": "#A0A5A9", "color_name": "Light Bluish Gray", "qty": 4}
]

new_content += '    "10692-1": {\n'
new_content += '        "name": "LEGO Classic Creative Bricks (50 Simple Parts Edition)",\n'
new_content += '        "minifigures": [],\n'
new_content += f'        "parts": {json.dumps(classic_parts, indent=12)}\n'
new_content += '    }\n'
new_content += '}\n'

# Append helper functions from original set_catalog.py
new_content += """
# Paleta de colores comunes de LDraw para el generador dinámico
LDRAW_COLORS = [
    {"code": "85", "hex": "#A0A5A9", "name": "Light Bluish Gray"},
    {"code": "84", "hex": "#5A5A5A", "name": "Dark Bluish Gray"},
    {"code": "0", "hex": "#1B1B1B", "name": "Black"},
    {"code": "4", "hex": "#C91A09", "name": "Red"},
    {"code": "1", "hex": "#0A3C9F", "name": "Blue"},
    {"code": "14", "hex": "#F2CD37", "name": "Yellow"},
]

# Piezas LDraw comunes disponibles
LDRAW_PARTS = [
    {"ref": "3001", "name": "Brick 2x4"},
    {"ref": "3002", "name": "Brick 2x3"},
    {"ref": "3003", "name": "Brick 2x2"},
    {"ref": "3004", "name": "Brick 1x2"},
    {"ref": "3005", "name": "Brick 1x1"},
    {"ref": "3010", "name": "Brick 1x4"},
    {"ref": "3020", "name": "Plate 2x4"},
    {"ref": "3021", "name": "Plate 2x3"},
    {"ref": "3022", "name": "Plate 2x2"},
    {"ref": "3023", "name": "Plate 1x2"},
]

def get_set_data(set_id: str) -> dict:
    \"\"\"
    Retorna el inventario de un set. Si el set existe en REAL_SETS, lo devuelve directamente.
    Si no, genera dinámicamente un set de LEGO realista para demostración de búsqueda bajo demanda.
    \"\"\"
    # Limpiar formato de entrada (e.g. 75078 -> 75078-1)
    clean_id = set_id.strip()
    if "-" not in clean_id:
        clean_id = f"{clean_id}-1"
        
    if clean_id in REAL_SETS:
        return REAL_SETS[clean_id]
        
    # Inicializar generador determinista a partir del hash del Set ID para que devuelva
    # siempre el mismo inventario para un set concreto
    random.seed(hash(clean_id))
    
    num_parts = random.randint(5, 12)
    generated_parts = []
    
    # Seleccionar partes aleatorias del catálogo LDraw
    for _ in range(num_parts):
        part = random.choice(LDRAW_PARTS)
        color = random.choice(LDRAW_COLORS)
        qty = random.randint(2, 24)
        
        generated_parts.append({
            "ref": part["ref"],
            "color_code": color["code"],
            "color_hex": color["hex"],
            "color_name": color["name"],
            "qty": qty
        })
        
    # Minifiguras generadas
    minifigs = []
    num_minifigs = random.randint(0, 4)
    for i in range(num_minifigs):
        fig_id = f"fig-{random.randint(100, 999)}"
        minifigs.append({
            "ref": fig_id,
            "name": f"Minifigura Especial {fig_id.upper()}",
            "qty": random.randint(1, 2)
        })
        
    return {
        "name": f"Set Genérico Lego #{clean_id}",
        "minifigures": minifigs,
        "parts": generated_parts
    }
"""

with open(catalog_path, "w") as f:
    f.write(new_content)

print("Updated set_catalog.py successfully!")
