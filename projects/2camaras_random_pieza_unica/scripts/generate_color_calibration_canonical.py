# -*- coding: utf-8 -*-
"""generate_color_calibration_canonical.py
Re-calibra la paleta de color usando la escena CANONICA (30cm, 55mm, 2048px,
iluminacion V4, cinta azul petroleo). Genera renders de un plano coloreado
con cada color del catalogo y extrae los RGB medios para cenital y lateral.

Uso:
    /opt/homebrew/bin/blender -b -P \
        2camaras_random_pieza_unica/scripts/generate_color_calibration_canonical.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))
if legovic_root not in sys.path:
    sys.path.append(legovic_root)
user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    import bpy
    import mathutils
except ImportError:
    print("[ERROR] Este script debe ejecutarse dentro de Blender (-b -P)")
    sys.exit(1)

from generate_synthetic_set import create_abs_plastic_material
import scene_canonical

# ─── Config ───────────────────────────────────────────────────────────────────
RENDER_RES = 2048
COLOR_CATALOG_PATH = os.path.join(legovic_root, "database", "color_catalog.json")
OUT_DIR = os.path.join(project_root, "data", "color_calibration_canonical")
OUT_PALETTE = os.path.join(project_root, "data", "color_calibration_palette.json")


def extract_mean_rgb_from_render(filepath):
    """Lee una imagen renderizada y extrae el RGB medio usando la máscara de alpha."""
    # Load image into Blender and read pixels
    img = bpy.data.images.load(filepath)
    w, h = img.size[0], img.size[1]
    pixels = np.array(img.pixels[:]).reshape(h, w, 4)  # RGBA float [0,1]
    bpy.data.images.remove(img)
    
    # Convert to 0-255
    rgb = pixels[:, :, :3] * 255.0
    alpha = pixels[:, :, 3]
    
    # El plano está en el centro. Filtramos los píxeles donde el alpha sea > 0.9 (foreground)
    mask = alpha > 0.9
    
    if not np.any(mask):
        # Fallback si no hay máscara (por ejemplo, si no se habilitó film_transparent)
        print(f"[WARN] No foreground detected via alpha in {filepath}, falling back to center median")
        h8, w8 = h * 3 // 8, w * 3 // 8
        piece_region = rgb[h8:h-h8, w8:w-w8, :]
        pixels_fg = piece_region.reshape(-1, 3)
    else:
        pixels_fg = rgb[mask]
    
    # Recalculamos luminancia para descartar brillos especulares (top 10%)
    lum = pixels_fg[:, 0] * 0.299 + pixels_fg[:, 1] * 0.587 + pixels_fg[:, 2] * 0.114
    thresh = np.percentile(lum, 90)
    keep = lum <= thresh
    
    if np.any(keep):
        avg_rgb = pixels_fg[keep].mean(axis=0)
    else:
        avg_rgb = pixels_fg.mean(axis=0)
        
    return [round(float(avg_rgb[0]), 1), round(float(avg_rgb[1]), 1), round(float(avg_rgb[2]), 1)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load color catalog
    with open(COLOR_CATALOG_PATH, "r", encoding="utf-8") as f:
        color_catalog = json.load(f)

    # Build canonical scene (2048x2048, cam at 30cm, focal 55mm, V4 lighting)
    cam_cen, cam_lat = scene_canonical.build_scene_canonical(
        render_res=RENDER_RES, film_transparent=True
    )
    
    # Remove background objects so that alpha channel only contains the calibration plane
    for name in ["Conveyor_Belt_Plane", "Side_Screen_AL", "Office_Floor"]:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES

    # Create a flat plane (like a LEGO plate top) at z=0.005 (slightly above belt)
    # Size: ~2cm x 2cm (0.2 x 0.2 BU) centered at origin
    bpy.ops.mesh.primitive_plane_add(size=0.4, location=(0.0, 0.0, 0.005))
    plane = bpy.context.active_object
    plane.name = "Color_Calibration_Plane"
    bpy.ops.object.shade_smooth()

    calibration_data = []
    total = len(color_catalog)

    for i, (code, info) in enumerate(color_catalog.items()):
        hex_color = info.get("hex", "#808080")
        name = info.get("name", "Unknown")

        if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
            continue

        print(f"[{i+1}/{total}] Calibrating color {code}: {name} ({hex_color})")

        # Apply material
        mat = create_abs_plastic_material(hex_color)
        if plane.data.materials:
            plane.data.materials[0] = mat
        else:
            plane.data.materials.append(mat)
        bpy.context.view_layer.update()

        # Render cenital
        cen_path = os.path.join(OUT_DIR, f"calib_{code}_cen.png")
        scene.camera = cam_cen
        scene.render.filepath = cen_path
        bpy.ops.render.render(write_still=True)

        # Render lateral
        lat_path = os.path.join(OUT_DIR, f"calib_{code}_lat.png")
        scene.camera = cam_lat
        scene.render.filepath = lat_path
        bpy.ops.render.render(write_still=True)

        # Extract RGB
        rgb_cen = extract_mean_rgb_from_render(cen_path)
        rgb_lat = extract_mean_rgb_from_render(lat_path)

        calibration_data.append({
            "color_code": int(code) if code.isdigit() else code,
            "color_name": name,
            "color_hex": hex_color,
            "rgb_cenital": rgb_cen,
            "rgb_lateral": rgb_lat,
        })

        print(f"   cen={rgb_cen}, lat={rgb_lat}")

    # Remove calibration plane
    bpy.data.objects.remove(plane, do_unlink=True)

    # Save palette
    with open(OUT_PALETTE, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[CalibCanonical DONE] {len(calibration_data)} colores calibrados")
    print(f"[CalibCanonical] Palette: {OUT_PALETTE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()