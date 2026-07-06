import os
import sys
import multiprocessing
import torch

from ultralytics import YOLO

def create_yaml(dataset_path):
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val

# Keypoints format
kpt_shape: [9, 3]

names:
  0: 'lego_piece'
"""
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def train_model(view_name, yaml_path, project_root, threads):
    # Set PyTorch threads for this process
    torch.set_num_threads(threads)
    
    base_model = "yolo11s-pose.pt"
    epochs = 50
    batch_size = 32  # Optimized for 48GB RAM
    workers = max(1, threads) # Dataloader workers
    imgsz = 640
    device = "cpu"  # Explicitly using CPU as requested to avoid MPS bug
    
    print(f"[{view_name.upper()}] Starting training. CPU Threads allocated: {threads}. Batch size: {batch_size}")
    
    model = YOLO(base_model)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        project=os.path.join(project_root, "models", "train_runs"),
        name=f"domo_{view_name}_pose_cpu",
        exist_ok=True,
        cache=True, 
        pose=12.0,
        kobj=1.0,
        box=7.5,
        cls=0.5,
        degrees=360.0,
        fliplr=0.5,
        flipud=0.5
    )
    
    best_weights = os.path.join(project_root, "models", "train_runs", f"domo_{view_name}_pose_cpu", "weights", "best.pt")
    target_weights = os.path.join(project_root, "models", f"yolo_{view_name}_pose_cpu.pt")
    if os.path.exists(best_weights):
        os.system(f"cp '{best_weights}' '{target_weights}'")
        print(f"[{view_name.upper()}] Training finished. Model saved to {target_weights}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_cenital = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_cenital")
    dataset_frontal = os.path.join(project_root, "camara_domo", "data", "yolo_dataset_frontal")
    
    os.makedirs(dataset_cenital, exist_ok=True)
    os.makedirs(dataset_frontal, exist_ok=True)
        
    print("=== Generating YAMLs (Class-Agnostic) ===")
    yaml_cenital = create_yaml(dataset_cenital)
    yaml_frontal = create_yaml(dataset_frontal)
    
    # Calculate CPU allocation
    total_cores = os.cpu_count() or 4
    # Leave 2 cores free for system safety (minimum of 2 cores remaining total)
    usable_cores = max(2, total_cores - 2)
    # Divide equally between 2 training processes
    threads_per_process = usable_cores // 2
    
    print("="*50)
    print(f"Total CPU Cores: {total_cores}")
    print(f"Reserved for system: {total_cores - usable_cores}")
    print(f"Threads per training process: {threads_per_process}")
    print("="*50)
    
    # Using multiprocessing to run both simultaneously
    p_cenital = multiprocessing.Process(target=train_model, args=("cenital", yaml_cenital, project_root, threads_per_process))
    p_frontal = multiprocessing.Process(target=train_model, args=("frontal", yaml_frontal, project_root, threads_per_process))
    
    p_cenital.start()
    p_frontal.start()
    
    p_cenital.join()
    p_frontal.join()
    
    print("\nAll parallel training processes completed successfully.")

if __name__ == "__main__":
    main()
