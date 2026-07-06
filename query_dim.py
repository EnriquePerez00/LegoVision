import json
with open("/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/canonical_keypoints.json", "r") as f:
    kps_data = json.load(f)
for ref in ["54930c02", "4085d"]:
    print(f"{ref}: {kps_data['pieces'][ref].get('dimensions_mm')}")
