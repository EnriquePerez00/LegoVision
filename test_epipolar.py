import subprocess
import json
import re

script_path = "camara_domo/scripts/inferencia_neuronal.py"

with open(script_path, "r", encoding="utf-8") as f:
    original_code = f.read()

thresholds = [100.0, 150.0, 200.0]

for th in thresholds:
    print(f"\n--- Probando umbral epipolar: {th} ---")
    
    # Modify threshold in the script
    mod_code = re.sub(r"if dist < 100\.0:", f"if dist < {th}:", original_code)
    mod_code = re.sub(r"if best_lat_idx != -1 and best_lat_dist < 100\.0:", f"if best_lat_idx != -1 and best_lat_dist < {th}:", mod_code)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(mod_code)
    
    # Run inference
    subprocess.run([
        ".venv/bin/python", script_path, 
        "--data_dir", "camara_domo/data/data10", 
        "--output", "camara_domo/data/data10/inferencia_consolidada_test.json"
    ], check=True)
    
    # Analyze
    with open('camara_domo/data/data10/inferencia_consolidada_test.json', 'r') as f:
        data = json.load(f)

    total_obs = 0
    lat_fails = 0

    for tid, track in data.items():
        history = track.get("history", [])
        for h in history:
            total_obs += 1
            lat = h.get("bbox_lat", [])
            if lat == [0.0, 0.0, 1.0, 1.0]:
                lat_fails += 1

    print(f"Total Observaciones Cenitales: {total_obs}")
    print(f"Fallos de Asociación Lateral: {lat_fails} ({lat_fails/total_obs*100:.2f}%)")

# Restaurar original
with open(script_path, "w", encoding="utf-8") as f:
    f.write(original_code)
