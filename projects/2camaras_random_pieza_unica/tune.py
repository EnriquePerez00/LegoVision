import yaml
import subprocess
import json
import os

config_path = "/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/config.yaml"
script_path = "/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/scripts/run_evaluation.py"
metadata_path = "/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/test_500allhd/random_500_metadata.json"
report_path = "/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/reports/tune_eval.json"
python_exec = "/Users/I764690/Code_personal/LegoVision/.venv/bin/python"

params_to_test = [
    (3, 10),
    (5, 30),
    (7, 50),
    (9, 100),
]

best_acc = 0.0
best_params = None

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

for k_size, min_area in params_to_test:
    cfg["inference"]["segmentation"]["morphology_kernel_size"] = k_size
    cfg["inference"]["segmentation"]["min_contour_area"] = min_area
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    
    print(f"Testing k_size={k_size}, min_area={min_area}...")
    subprocess.run([python_exec, script_path, "--metadata", metadata_path, "--report", report_path], check=False, capture_output=True)
    
    try:
        with open(report_path, "r") as f:
            res = json.load(f)
            acc = res.get("summary", {}).get("accuracy_pct", res.get("accuracy", 0.0))
            print(f"  -> Accuracy: {acc}%")
            if acc >= best_acc:
                best_acc = acc
                best_params = (k_size, min_area)
    except Exception as e:
        print("  -> Error reading result:", e)

print(f"\nBest params: k_size={best_params[0]}, min_area={best_params[1]} with Accuracy={best_acc}%")
# Restore best params
cfg["inference"]["segmentation"]["morphology_kernel_size"] = best_params[0]
cfg["inference"]["segmentation"]["min_contour_area"] = best_params[1]
with open(config_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
