import os
import glob
import cv2
import uuid
from PIL import Image

def load_classes(classes_path):
    with open(classes_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def extract_crops(images_dir, labels_dir, output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    print(f"Encontradas {len(image_paths)} imágenes en {images_dir}")
    
    crops_extracted = 0
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_idx = int(parts[0])
            x_c, y_c, bw, bh = map(float, parts[1:5])
            
            # YOLO format to absolute pixels
            x_center = x_c * w
            y_center = y_c * h
            box_width = bw * w
            box_height = bh * h
            
            x1 = int(max(0, x_center - box_width / 2))
            y1 = int(max(0, y_center - box_height / 2))
            x2 = int(min(w, x_center + box_width / 2))
            y2 = int(min(h, y_center + box_height / 2))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img[y1:y2, x1:x2]
            
            class_name = classes[class_idx]
            class_dir = os.path.join(output_dir, str(class_idx))  # Keep class_idx for easy PyTorch training
            os.makedirs(class_dir, exist_ok=True)
            
            crop_filename = f"crop_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(os.path.join(class_dir, crop_filename), crop)
            crops_extracted += 1
            
    print(f"Extraídos {crops_extracted} crops en {output_dir}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    classes_path = os.path.join(project_root, "data", "classes.txt")
    classes = load_classes(classes_path)
    
    print("=== Extrayendo dataset Cenital ===")
    extract_crops(
        os.path.join(project_root, "data", "yolo_dataset_cenital", "images", "train"),
        os.path.join(project_root, "data", "yolo_dataset_cenital", "labels", "train"),
        os.path.join(project_root, "data", "efficientnet_train", "cenital", "train"),
        classes
    )
    
    print("=== Extrayendo dataset Lateral/Frontal ===")
    extract_crops(
        os.path.join(project_root, "data", "yolo_dataset_frontal", "images", "train"),
        os.path.join(project_root, "data", "yolo_dataset_frontal", "labels", "train"),
        os.path.join(project_root, "data", "efficientnet_train", "lateral", "train"),
        classes
    )
    
    print("[Done] Dataset de Fine-Tuning generado.")
