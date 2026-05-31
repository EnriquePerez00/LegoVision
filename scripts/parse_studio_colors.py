import os
import json

def parse_studio_colors():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    studio_color_file = "/Applications/Studio 2.0/data/StudioColorDefinition.txt"
    output_json = os.path.join(project_root, "database", "color_catalog.json")
    
    if not os.path.exists(studio_color_file):
        print(f"Error: No se encontró el archivo de colores en: {studio_color_file}")
        return
        
    print(f"Leyendo definiciones de color desde: {studio_color_file}")
    color_map = {}
    
    with open(studio_color_file, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split("\t")
        
        # Encontrar los índices de las columnas relevantes
        try:
            bl_code_idx = header.index("BL Color Code")
            ldraw_code_idx = header.index("LDraw Color Code")
            bl_name_idx = header.index("BL Color Name")
            rgb_idx = header.index("RGB value")
            alpha_idx = header.index("Alpha")
            category_idx = header.index("CategoryName")
        except ValueError as e:
            print("Error en el formato de cabecera del archivo de colores:", e)
            return
            
        for line in f:
            fields = line.strip("\n").split("\t")
            if len(fields) <= max(bl_code_idx, rgb_idx, alpha_idx, category_idx):
                continue
                
            bl_code = fields[bl_code_idx].strip()
            ldraw_code = fields[ldraw_code_idx].strip()
            bl_name = fields[bl_name_idx].strip()
            rgb_val = fields[rgb_idx].strip()
            alpha_val = fields[alpha_idx].strip()
            category = fields[category_idx].strip()
            
            # Saltarse registros vacíos
            if not bl_code or not rgb_val:
                continue
                
            # Determinar tipo de material
            alpha = 1.0
            try:
                alpha = float(alpha_val)
            except ValueError:
                pass
                
            category_lower = category.lower()
            if alpha < 1.0 or "trans" in category_lower or "transparent" in category_lower:
                material_type = "transparent"
            elif "chrome" in category_lower or "metallic" in category_lower or "metal" in category_lower:
                material_type = "metallic"
            elif "rubber" in category_lower:
                material_type = "rubber"
            else:
                material_type = "solid"
                
            color_map[bl_code] = {
                "name": bl_name,
                "ldraw_code": ldraw_code,
                "hex": rgb_val,
                "alpha": alpha,
                "category": category,
                "material_type": material_type
            }
            
    # Guardar a JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(color_map, out, indent=4)
        
    print(f"Catálogo de colores generado con éxito: {len(color_map)} colores guardados en {output_json}")

if __name__ == "__main__":
    parse_studio_colors()
