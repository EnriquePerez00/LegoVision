#!/bin/bash
LABEL_FILE="/Users/I764690/Code_personal/LegoVision/camara_domo/data/yolo_dataset/labels_cenital/train/train_00000_d8ed73f7.txt"
CLASSES_FILE="/Users/I764690/Code_personal/LegoVision/camara_domo/data/yolo_dataset/classes.txt"

# Get unique class IDs from label file (add 1 because awk is 1-indexed for sed)
CLASS_IDS=$(cat $LABEL_FILE | awk '{print $1 + 1}' | sort | uniq)

echo "Generando reports para las piezas encontradas..."
for ID in $CLASS_IDS; do
    REF=$(sed -n "${ID}p" $CLASSES_FILE)
    echo "Generando report para la pieza: $REF"
    python3 2camaras_pieza_unica/scripts/generate_3dposes_reports.py --ref $REF --no-index
    cp 2camaras_pieza_unica/reports/3dposes/stable_poses_${REF}.html /Users/I764690/.gemini/antigravity/brain/50dde2f1-4b96-41aa-9430-a870787d671c/ 2>/dev/null || true
done
echo "Proceso completado."
