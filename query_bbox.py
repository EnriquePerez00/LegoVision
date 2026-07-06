import json

with open("/Users/I764690/Code_personal/LegoVision/2camaras_random_pieza_unica/data/canonical_keypoints.json", "r") as f:
    kps_data = json.load(f)

for ref in ["54930c02", "4085d"]:
    bbox = kps_data["pieces"].get(ref, {}).get("bbox_bu", [])
    if bbox:
        dx = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dz = bbox[5] - bbox[2]
        print(f"Ref {ref}: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f} BU")
    else:
        print(f"Ref {ref} not found or no bbox.")
