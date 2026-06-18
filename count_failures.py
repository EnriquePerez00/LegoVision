import json

with open('camara_domo/data/data10/inferencia_consolidada.json', 'r') as f:
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

print(f"Total Observaciones Cenitales (Anclaje): {total_obs}")
print(f"Fallos de Asociación Lateral: {lat_fails} ({lat_fails/total_obs*100:.2f}%)")
