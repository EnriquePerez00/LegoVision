"""
build_math_palette.py
Genera data/color_calibration_palette.json aplicando los factores lineales
de color correction (CCM) sobre los HEX del catálogo de colores.
No requiere Blender ni renders.
"""
import os
import sys
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "..", "..", "database", "color_catalog.json")

# CCM matrices (cenital / lateral) — matrices reales del pipeline (run_evaluation.py)
CCM_CEN = np.array([
    [ 1.33115375, -0.14676734, -0.19946276],
    [-0.0920857 ,  1.08937858, -0.03107387],
    [-0.34741085,  0.91087719,  0.3857221 ]
])

CCM_LAT = np.array([
    [ 1.5373125 , -0.40934897,  0.06302813],
    [-0.09606269,  1.32641158, -0.09466379],
    [-0.37601142,  1.19138109,  0.28907606]
])


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6:
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)
    return np.array([128.0, 128.0, 128.0])


def main():
    if not os.path.exists(db_path):
        sys.exit(f"[ERROR] color_catalog.json not found at: {db_path}")

    with open(db_path, "r") as f:
        catalog = json.load(f)

    calibration_data = []

    for str_code, info in catalog.items():
        hex_color = info.get("hex", info.get("color_hex", "#808080"))
        name      = info.get("name", info.get("color_name", f"Color {str_code}"))
        
        # Convertir código — mantenerlo como entero si es posible
        try:
            code = int(str_code)
        except ValueError:
            code = str_code

        rgb = hex_to_rgb(hex_color)  # HEX teórico [0-255]
        # La paleta almacena CCM(HEX) → predicción de lo que verá la cámara
        # La comparación será: raw_image vs CCM(HEX), sin transformar el query
        rgb_cen = np.clip(CCM_CEN @ rgb, 0.0, 255.0).tolist()
        rgb_lat = np.clip(CCM_LAT @ rgb, 0.0, 255.0).tolist()

        calibration_data.append({
            "color_code": code,
            "color_name": name,
            "color_hex":  hex_color,
            "rgb_cenital":  rgb_cen,
            "rgb_lateral":  rgb_lat,
        })

    out_json = os.path.join(current_dir, "..", "data", "color_calibration_palette.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Paleta matemática generada: {len(calibration_data)} colores → {out_json}")
    # Sanity check: mostramos 3 colores típicos
    for entry in calibration_data[:3]:
        print(f"  {entry['color_code']:>4} {entry['color_name']:<25} HEX={entry['color_hex']}  "
              f"cen={[round(x,1) for x in entry['rgb_cenital']]}  "
              f"lat={[round(x,1) for x in entry['rgb_lateral']]}")


if __name__ == "__main__":
    main()
