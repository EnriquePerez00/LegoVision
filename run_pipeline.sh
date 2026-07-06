#!/bin/bash
echo "Waiting for train_and_evaluate_color_mlp.py to finish..."
while pgrep -f train_and_evaluate_color_mlp.py > /dev/null; do
    sleep 5
done
echo "Training finished. Starting inference on simulation_300..."
python3 projects/camara_domo_monopieza_90/scripts/inferencia_neuronal_v2.py --data_dir projects/camara_domo_monopieza_90/data/simulation_300/frames --output projects/camara_domo_monopieza_90/data/reports/sim300_consolidada.json
echo "Inference finished. Evaluating..."
python3 projects/camara_domo_monopieza_90/scripts/evaluate_sim300.py
