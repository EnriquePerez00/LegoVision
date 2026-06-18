import json

KEYPOINTS_PATH = "/Users/I764690/Code_personal/LegoVision/data/keypoints/keypoints.json"

with open(KEYPOINTS_PATH, "r") as f:
    kps_data = json.load(f)

for ref, data in kps_data["pieces"].items():
    bbox = data.get("bbox_bu", [])
    if bbox:
        # bbox is usually [min_x, min_y, min_z, max_x, max_y, max_z]
        dx = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dz = bbox[5] - bbox[2]
        max_dim = max(dx, dy, dz)
        if max_dim > 10:
            print(f"Piece {ref} is huge: {max_dim:.2f} BU (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})")
