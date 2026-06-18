
import os
import subprocess
import shutil

print("Installing dependencies...")
subprocess.run(["pip", "install", "ultralytics", "python-dotenv"], check=True)

print("Preparing workspace...")
os.makedirs("/kaggle/working/camara_domo/models/train_runs", exist_ok=True)
os.makedirs("/kaggle/working/camara_domo/scripts", exist_ok=True)
os.makedirs("/kaggle/working/camara_domo/data", exist_ok=True)

print("Executing training script...")
subprocess.run(["python", "/kaggle/working/train_yolo_pose_remote.py"], cwd="/kaggle/working", check=True)

print("Training finished! Models are saved in /kaggle/working/camara_domo/models/")
