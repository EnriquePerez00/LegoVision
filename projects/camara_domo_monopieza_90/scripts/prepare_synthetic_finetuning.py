import os
import glob
import shutil
import json
from PIL import Image
from rotation_aligner import align_image_by_moments

def preprocess_crop_grayscale(crop_img: Image.Image, canvas_size: int = 224) -> Image.Image:
    gray_img = crop_img.convert("L")
    rgb_gray = Image.merge("RGB", (gray_img, gray_img, gray_img))

    margin = 8
    max_dim = canvas_size - 2 * margin
    w, h = rgb_gray.size
    if w > 0 and h > 0:
        scale = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = rgb_gray.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        paste_x = (canvas_size - new_w) // 2
        paste_y = (canvas_size - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
    return Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

def prepare_dataset():
    project_root = "/Users/I764690/Code_personal/LegoVision"
    src_dir = os.path.join(project_root, "projects", "2camaras_random_pieza_unica", "data", "dinov2_refs_v4_canonical")
    dst_dir = os.path.join(project_root, "projects", "camara_domo_75078", "data", "efficientnet_train")

    if not os.path.exists(src_dir):
        print(f"Source dir {src_dir} not found!")
        return
        
    # Load metadata to get bounding boxes
    metadata_lookup = {}
    for meta_file in glob.glob(os.path.join(src_dir, "metadata_worker_*.json")):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        for r in meta_data.get("renders", []):
            fname = r["file_name"]
            metadata_lookup[fname] = {
                "cenital": r["cameras"]["cenital"]["bbox_norm"],
                "lateral": r["cameras"]["lateral"]["bbox_norm"]
            }

    for cam in ["cenital", "lateral"]:
        src_cam = os.path.join(src_dir, cam)
        dst_cam = os.path.join(dst_dir, cam, "train")
        
        if os.path.exists(dst_cam):
            shutil.rmtree(dst_cam)
        os.makedirs(dst_cam, exist_ok=True)
        
        images = glob.glob(os.path.join(src_cam, "*.png"))
        count = 0
        for img_path in images:
            fname = os.path.basename(img_path)
            parts = fname.split("_")
            if len(parts) >= 2 and parts[0] == "ref":
                ref = parts[1]
                
                # Crop and Grayscale
                try:
                    img_raw = Image.open(img_path)
                    # Componer sobre fondo negro si tiene canal alfa para evitar domain shift (Medida 1)
                    if img_raw.mode == "RGBA":
                        black_bg = Image.new("RGBA", img_raw.size, (0, 0, 0, 255))
                        img = Image.alpha_composite(black_bg, img_raw).convert("RGB")
                    else:
                        img = img_raw.convert("RGB")

                    if fname in metadata_lookup:
                        bbox = metadata_lookup[fname][cam]
                        iw, ih = img.size
                        cx1, cy1, cx2, cy2 = bbox
                        
                        cropped = img.crop((
                            max(0, int(cx1 * iw)), max(0, int(cy1 * ih)),
                            min(iw, int(cx2 * iw)), min(ih, int(cy2 * ih))
                        ))
                    else:
                        cropped = img
                        
                    aligned = align_image_by_moments(cropped)
                    processed = preprocess_crop_grayscale(aligned)
                    
                    class_dir = os.path.join(dst_cam, ref)
                    os.makedirs(class_dir, exist_ok=True)
                    processed.save(os.path.join(class_dir, fname))
                    count += 1
                except Exception as e:
                    print(f"Failed {fname}: {e}")
        
        print(f"[{cam}] Procesadas {count} imágenes con BBox y Grayscale en {dst_cam}")

if __name__ == "__main__":
    prepare_dataset()
