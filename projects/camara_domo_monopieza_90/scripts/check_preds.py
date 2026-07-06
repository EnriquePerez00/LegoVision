# -*- coding: utf-8 -*-
import json

report_path = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_monopieza_90/data/simulation_100_all/inferencia_consolidada.json"
with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total samples: {len(data['results'])}")
for idx, r in enumerate(data["results"][:15]):
    print(f"Sample {idx+1:02d} | GT Ref: {r['ref_gt']:6s} | GT Color Code: {r['color_code_gt']} | Pred Ref: {r['ref_inferred']:6s} | Cenital Color: {r['color_name_cen']} | Lateral Color: {r['color_name_lat']} | Fused Color: {r['color_name_fused']}")
