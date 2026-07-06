import os
import json
from lightning_sdk import Studio
with open("/Users/I764690/Code_personal/LegoVision/camara_domo/scripts/gpu_providers.json", "r") as f:
    accounts = json.load(f)
acc = next(a for a in accounts if a["provider"] == "lightning" and a["status"] == "active")
os.environ["LIGHTNING_USER_ID"] = acc["user_id"]
os.environ["LIGHTNING_API_KEY"] = acc["api_key"]
studio = Studio(name="legovision-t4", teamspace=acc["teamspace"], user=acc["username"])
try:
    studio.download_file("camara_domo/models/train_runs/training_job.log", "/Users/I764690/Code_personal/LegoVision/camara_domo/models/live_training_job.log")
    print("Log descargado!")
except Exception as e:
    print(f"Error: {e}")
