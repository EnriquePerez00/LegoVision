import os
import sys
import json
from PIL import Image
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import run_evaluation

def main():
    db_path = os.path.join(current_dir, "..", "..", "database", "color_catalog.json")
    with open(db_path, "r") as f:
        catalog = json.load(f)
        
    out_dir = os.path.join(current_dir, "..", "data", "color_calibration")
    
    calibration_data = []
    
    for code, info in catalog.items():
        hex_color = info.get("hex", "#000000")
        name = info.get("name", "Unknown")
        
        cen_path = os.path.join(out_dir, f"calib_{code}_cen.png")
        lat_path = os.path.join(out_dir, f"calib_{code}_lat.png")
        
        if not os.path.exists(cen_path) or not os.path.exists(lat_path):
            print(f"Skipping {code}: images not found")
            continue
            
        print(f"Extracting {code} ({name})...")
        
        img_cen = Image.open(cen_path).convert("RGBA")
        alpha_cen = np.array(img_cen)[:, :, 3]
        rgb_cen = run_evaluation.estimate_color_predominant_sam(img_cen, alpha_cen)
        
        img_lat = Image.open(lat_path).convert("RGBA")
        alpha_lat = np.array(img_lat)[:, :, 3]
        rgb_lat = run_evaluation.estimate_color_predominant_sam(img_lat, alpha_lat)
        
        calibration_data.append({
            "color_code": int(code) if code.isdigit() else code,
            "color_name": name,
            "color_hex": hex_color,
            "rgb_cenital": rgb_cen.tolist(),
            "rgb_lateral": rgb_lat.tolist()
        })
        
    out_json = os.path.join(current_dir, "..", "data", "color_calibration_palette.json")
    with open(out_json, "w") as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"Extraction completed! Saved {len(calibration_data)} colors to {out_json}")

if __name__ == "__main__":
    main()
