import re
import json

# Mapping of BrickLink Color ID -> (LDraw Code, Hex, Name)
COLOR_MAP = {
    "11": ("0", "#1B1B1B", "Black"),
    "1": ("15", "#FFFFFF", "White"),
    "86": ("85", "#A0A5A9", "Light Bluish Gray"),
    "85": ("84", "#5A5A5A", "Dark Bluish Gray"),
    "7": ("1", "#0A3C9F", "Blue"),
    "5": ("4", "#C91A09", "Red"),
    "3": ("14", "#F2CD37", "Yellow"),
    "18": ("36", "#C91A09", "Trans-Red"),
    "59": ("320", "#720012", "Dark Red"),
    "55": ("73", "#5E748C", "Sand Blue"),
    "4": ("25", "#FE8A18", "Orange"),
    "88": ("70", "#5C1E0F", "Reddish Brown"),
    "95": ("297", "#899395", "Flat Silver"),
    "77": ("148", "#575857", "Pearl Dark Gray"),
    "2": ("19", "#DFD1A5", "Tan"),
    "69": ("28", "#9F8F75", "Dark Tan"),
    "34": ("27", "#BBE90B", "Lime"),
    "15": ("33", "#A5DBF5", "Trans-Light Blue"),
    "19": ("38", "#F08F1C", "Trans-Orange"),
    "16": ("37", "#C0F010", "Trans-Neon Green"),
    "110": ("68", "#F9A725", "Bright Light Orange"),
    "156": ("72", "#36AEBF", "Medium Azure"),
    "155": ("326", "#7C9051", "Olive Green"),
    "80": ("288", "#184632", "Dark Green"),
    "63": ("272", "#0D2654", "Dark Blue"),
    "71": ("73", "#B30006", "Magenta"),
}

# Sets to parse
SETS = {
    "75280-1": {
        "file": "/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/.system_generated/steps/2573/content.md",
        "name": "501st Legion Clone Troopers"
    },
    "75218-1": {
        "file": "/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/.system_generated/steps/2591/content.md",
        "name": "X-Wing Starfighter"
    },
    "75337-1": {
        "file": "/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/.system_generated/steps/2593/content.md",
        "name": "AT-TE Walker"
    }
}

output_dict = {}

for set_id, set_info in SETS.items():
    with open(set_info["file"], "r", encoding="utf-8") as f:
        html = f.read()

    regular_start = html.find("Regular Items:")
    extra_start = html.find("Extra Items:")

    if regular_start == -1:
        regular_start = html.lower().find("regular items:")
    if extra_start == -1:
        extra_start = html.lower().find("extra items:")

    if extra_start != -1:
        relevant_html = html[regular_start:extra_start]
    else:
        relevant_html = html[regular_start:]

    row_matches = re.finditer(r'<TR class="([^"]+)\s+IV_ITEM"[^>]*>(.*?)</TR>', relevant_html, re.DOTALL)

    parts = []
    minifigures = []

    for match in row_matches:
        class_attr = match.group(1)
        row_content = match.group(2)
        
        tds = re.findall(r'<TD[^>]*>(.*?)</TD>', row_content, re.DOTALL)
        if len(tds) < 4:
            continue
            
        # Qty
        qty_text = re.sub(r'&nbsp;|\s', '', tds[1])
        try:
            qty = int(qty_text)
        except:
            continue
            
        # Item No
        # We need to distinguish between Minifigures (M=...) and Parts (P=...)
        # Link: <A HREF="/v2/catalog/catalogitem.page?M=sw1093&idColor=0">sw1093</A>
        item_type = "part"
        ref_m = re.search(r'(?:P|S|M)=([^&"]+).*?idColor=(\d+)', tds[2])
        if not ref_m:
            ref_m = re.search(r'href="[^"]*(?:P|S|M)=([^&"]+).*?idColor=(\d+)"', tds[2], re.IGNORECASE)
            if not ref_m:
                continue
        
        # Check if minifig or part
        link_html = tds[2]
        if "catalogitem.page?M=" in link_html or "?M=" in link_html:
            item_type = "minifig"
            
        part_ref = ref_m.group(1)
        bl_color = ref_m.group(2)
        
        # Description
        desc_html = tds[3]
        desc_m = re.search(r'<B>(.*?)</B>', desc_html, re.DOTALL)
        if desc_m:
            desc = desc_m.group(1).strip()
        else:
            desc = re.sub(r'<[^>]*>', '', desc_html).strip()
            
        desc = re.sub(r'\s+', ' ', desc)
        desc = re.sub(r'&#\d+;', '', desc)
        desc = desc.replace("&nbsp;", " ")

        if item_type == "minifig":
            minifigures.append({
                "ref": part_ref,
                "name": desc,
                "qty": qty
            })
        else:
            # Map color
            ld_color, ld_hex, ld_name = COLOR_MAP.get(bl_color, (bl_color, "#808080", "Unknown Color"))
            parts.append({
                "ref": part_ref,
                "color_code": ld_color,
                "color_hex": ld_hex,
                "color_name": ld_name,
                "qty": qty
            })

    output_dict[set_id] = {
        "name": set_info["name"],
        "minifigures": minifigures,
        "parts": parts
    }
    
    total_pcs = sum(p["qty"] for p in parts) + sum(m["qty"] for m in minifigures)
    print(f"Set {set_id} ({set_info['name']}): {total_pcs} pieces, {len(parts)} unique parts, {len(minifigures)} minifigs.")

# Output Python dict to paste in set_catalog.py
print("\n--- PYTHON REAL_SETS ADDITIONS ---")
for set_id, data in output_dict.items():
    print(f"    {repr(set_id)}: {repr(data)},")

# Also write to a file in scratch
with open("/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/scratch/parsed_sets.json", "w") as out:
    json.dump(output_dict, out, indent=2)
