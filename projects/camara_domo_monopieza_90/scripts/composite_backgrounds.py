import os
import glob
from PIL import Image
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(img_path, bg_path):
    try:
        # Load the transparent 640x640 render
        fg = Image.open(img_path).convert("RGBA")
        fw, fh = fg.size
        
        # Load the real background (e.g. 1920x1080)
        bg = Image.open(bg_path).convert("RGBA")
        bw, bh = bg.size
        
        # Take a random 640x640 crop of the background to maximize data augmentation
        if bw > fw and bh > fh:
            x1 = random.randint(0, bw - fw)
            y1 = random.randint(0, bh - fh)
            bg_crop = bg.crop((x1, y1, x1 + fw, y1 + fh))
        else:
            bg_crop = bg.resize((fw, fh))
            
        # Composite the foreground over the random background crop
        bg_crop.alpha_composite(fg)
        
        # Convert to RGB to discard alpha channel, making it a standard YOLO image
        final_img = bg_crop.convert("RGB")
        
        # Overwrite the original file
        final_img.save(img_path)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def composite_dataset(dataset_dir, bg_img_path):
    print(f"Processing dataset: {dataset_dir}")
    images = glob.glob(os.path.join(dataset_dir, "images", "train", "*.png"))
    total = len(images)
    print(f"Found {total} images. Starting compositing...")
    
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_file, path, bg_img_path): path for path in images}
        completed = 0
        for future in as_completed(futures):
            if future.result():
                completed += 1
            if completed % 1000 == 0:
                print(f"Progress: {completed}/{total}")
                
    print(f"Compositing for {dataset_dir} finished.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cenital_dir = os.path.join(base_dir, "data", "yolo_dataset_cenital")
    frontal_dir = os.path.join(base_dir, "data", "yolo_dataset_frontal")
    
    bg_cen_path = os.path.join(base_dir, "data", "data100", "frame_000.png")
    bg_lat_path = os.path.join(base_dir, "data", "data100", "frame_000_frontal.png")
    
    composite_dataset(cenital_dir, bg_cen_path)
    composite_dataset(frontal_dir, bg_lat_path)
