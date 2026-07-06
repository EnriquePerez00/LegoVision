import json

with open('camara_domo/data/data10/simulation_metadata.json', 'r') as f:
    meta = json.load(f)

with open('camara_domo/data/data10/inferencia_consolidada.json', 'r') as f:
    cons = json.load(f)

# 1. Mapear cada referencia GT a sus frames visibles
gt_visibility = {}
for frame in meta["frames"]:
    fid = frame["frame_id"]
    for p in frame["visible_pieces"]:
        # Only consider pieces that are actually in the cenital field of view properly
        # The tracker starts when x > 110 or something?
        # Let's just look at 'bbox_cenital_norm' presence
        if "bbox_cenital_norm" in p and p["bbox_cenital_norm"] is not None:
            ref = p["ref"]
            if ref not in gt_visibility:
                gt_visibility[ref] = set()
            gt_visibility[ref].add(fid)

total_gt_visible_frames = sum(len(fids) for fids in gt_visibility.values())

# 2. Mapear frames trackeados cenitálmente en inferencia
total_tracked_frames = 0
for tid, track in cons.items():
    total_tracked_frames += len(track.get("history", []))

print(f"Frames totales donde las piezas eran visibles cenitalmente (GT): {total_gt_visible_frames}")
print(f"Frames totales capturados por el tracking cenital (Inferencia): {total_tracked_frames}")
if total_gt_visible_frames > 0:
    miss_rate = (1 - total_tracked_frames / total_gt_visible_frames) * 100
    print(f"Tasa de pérdida Cenital aproximada: {miss_rate:.2f}%")
