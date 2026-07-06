# -*- coding: utf-8 -*-
import os
import re
import shutil
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def get_dir_size(path):
    total = 0
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    return total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute the deletion of files.")
    args = parser.parse_args()

    data_dir = os.path.join(project_root, "data")
    reports_dir = os.path.join(data_dir, "reports")
    active_report_path = os.path.join(reports_dir, "simulation_100_comparative_report.html")

    # Find referenced images in the active HTML report
    referenced_images = set()
    if os.path.exists(active_report_path):
        with open(active_report_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Find all src="..." attributes pointing to visual_debug or crops_debug
        matches = re.findall(r'src=["\']([^"\']+)["\']', html)
        for m in matches:
            # Normalize path relative to reports_dir
            basename = os.path.basename(m)
            parent_dir = os.path.basename(os.path.dirname(m))
            referenced_images.add((parent_dir, basename))

    print("=" * 60)
    print("RESUMEN DE LIMPIEZA - PROYECTO camara_domo_75078")
    print("=" * 60)

    to_delete = []
    
    # 1. Zip de datos original
    zip_path = os.path.join(data_dir, "domo_training_data.zip")
    if os.path.exists(zip_path):
        to_delete.append((zip_path, "Archivo zip de datos original"))

    # 2. Carpetas de datasets de entrenamiento
    train_dirs = [
        "yolo_dataset_cenital",
        "yolo_dataset_frontal",
        "efficientnet_train",
        "kaggle_dataset",
        "ldraw_recovered"
    ]
    for d in train_dirs:
        p = os.path.join(data_dir, d)
        if os.path.exists(p):
            to_delete.append((p, f"Dataset/Carpeta de entrenamiento: {d}"))

    # 3. Reportes antiguos en reports
    if os.path.exists(reports_dir):
        for item in os.listdir(reports_dir):
            if item in ["simulation_100_comparative_report.html", "simulation_100_eval.json", "visual_debug", "crops_debug", "crops"]:
                continue
            item_path = os.path.join(reports_dir, item)
            to_delete.append((item_path, f"Reporte/archivo no esencial: {item}"))

    # 4. Imágenes no referenciadas en visual_debug, crops_debug, crops
    img_subdirs = ["visual_debug", "crops_debug", "crops"]
    non_ref_images = []
    for sd in img_subdirs:
        sd_path = os.path.join(reports_dir, sd)
        if os.path.exists(sd_path):
            for img in os.listdir(sd_path):
                img_path = os.path.join(sd_path, img)
                if os.path.isfile(img_path):
                    # Check if referenced
                    if (sd, img) not in referenced_images:
                        non_ref_images.append(img_path)

    total_reclaimed = 0
    
    print("\nArchivos y carpetas a eliminar:")
    for path, desc in to_delete:
        size = get_dir_size(path)
        total_reclaimed += size
        print(f" - [{size / (1024*1024):.1f} MB] {os.path.relpath(path, project_root)} ({desc})")

    print(f"\nImágenes de debug no referenciadas en reportes (en visual_debug/crops_debug/crops):")
    total_img_size = sum(get_dir_size(p) for p in non_ref_images)
    total_reclaimed += total_img_size
    print(f" - [{total_img_size / (1024*1024):.1f} MB] {len(non_ref_images)} imágenes de debug obsoletas")

    print("\n" + "=" * 60)
    print(f"ESPACIO TOTAL A RECUPERAR: {total_reclaimed / (1024*1024*1024):.2f} GB")
    print("=" * 60)

    if args.execute:
        print("\nEjecutando eliminación...")
        for path, _ in to_delete:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        for img_path in non_ref_images:
            if os.path.exists(img_path):
                os.remove(img_path)
        print("¡Limpieza completada con éxito!")
    else:
        print("\n[Modo Dry-Run] No se realizaron cambios. Ejecuta con --execute para aplicar.")

if __name__ == "__main__":
    main()
