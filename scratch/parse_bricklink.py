import re
import json

file_path = "/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/.system_generated/steps/2573/content.md"

with open(file_path, "r", encoding="utf-8") as f:
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

items = []
for match in row_matches:
    class_attr = match.group(1)
    row_content = match.group(2)
    
    tds = re.findall(r'<TD[^>]*>(.*?)</TD>', row_content, re.DOTALL)
    if len(tds) < 4:
        continue
        
    # tds[0]: Image
    # tds[1]: Qty
    # tds[2]: Item No
    # tds[3]: Description
    
    # Qty
    qty_text = re.sub(r'&nbsp;|\s', '', tds[1])
    try:
        qty = int(qty_text)
    except:
        continue
        
    # Item No & Color ID
    # Link: <A HREF="/v2/catalog/catalogitem.page?P=87994&idColor=11">87994</A>
    ref_m = re.search(r'(?:P|S|M)=([^&"]+).*?idColor=(\d+)', tds[2])
    if not ref_m:
        ref_m = re.search(r'href="[^"]*(?:P|S|M)=([^&"]+).*?idColor=(\d+)"', tds[2], re.IGNORECASE)
        if not ref_m:
            continue
            
    part_ref = ref_m.group(1)
    color_code = ref_m.group(2)
    
    # Description
    desc_html = tds[3]
    # Clean description from <B>...</B>
    desc_m = re.search(r'<B>(.*?)</B>', desc_html, re.DOTALL)
    if desc_m:
        desc = desc_m.group(1).strip()
    else:
        desc = re.sub(r'<[^>]*>', '', desc_html).strip()
        
    # Clean up double spaces, HTML entities, etc.
    desc = re.sub(r'\s+', ' ', desc)
    desc = re.sub(r'&#\d+;', '', desc)
    desc = desc.replace("&nbsp;", " ")
    
    items.append({
        "ref": part_ref,
        "color_code": color_code,
        "qty": qty,
        "desc": desc
    })

print(f"Parsed {len(items)} regular items")
print(json.dumps(items[:10], indent=2))

with open("/Users/I764690/.gemini/antigravity/brain/cea48a87-7be6-4f05-8c54-cfd39b5b05c1/scratch/parsed_75280.json", "w") as out:
    json.dump(items, out, indent=2)
