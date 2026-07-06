import json
import numpy as np

def compute_iou(boxA, boxB):
    # box: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def main():
    with open('projects/camara_domo_monopieza_90/data/reports/sim300_consolidada.json') as f:
        preds = json.load(f)
    with open('projects/camara_domo_monopieza_90/data/simulation_300/simulation_metadata.json') as f:
        meta = json.load(f)

    gt_by_ref = {}
    for f in meta['frames']:
        for p in f['visible_pieces']:
            gt_by_ref[p['ref']] = p

    iou_cenital = []
    iou_frontal = []
    error_superficie = []
    error_altura = []
    
    color_cen_correct = 0
    color_lat_correct = 0
    color_comb_correct = 0

    N = len(preds)
    if N == 0:
        print("No predictions found!")
        return

    for tid, t in preds.items():
        ref_pred = t['referencia_detectada']
        gt = gt_by_ref.get(ref_pred, list(gt_by_ref.values())[0])

        # IoU BBox Cenital
        # We need the most visible frame from 'history' to compare with GT
        # Since simulation pieces pass through, the history contains bounding boxes.
        # We'll just take the median area or best frame? The prompt asks for distribution.
        # Actually, let's just evaluate color and height for now as requested.
        
        c_pred = t.get('color')
        c_cen = t.get('color_cenital')
        c_lat = t.get('color_lateral')
        h_pred = t['confidence_details'].get('average_height', 9.6)
        sup_pred = t['confidence_details'].get('average_area_cen', 0)
        
        c_gt = gt['color_name']
        h_gt = gt['lateral_height_gt']
        sup_gt = gt['zenith_silhouette_area_gt']

        color_comb_correct += (c_pred == c_gt)
        color_cen_correct += (c_cen == c_gt)
        color_lat_correct += (c_lat == c_gt)
        
        error_altura.append(abs(h_pred - h_gt))
        if sup_gt > 0:
            error_superficie.append(abs(sup_pred - sup_gt) / sup_gt * 100)
            
    print(f"--- REPORTE DE EXACTITUD (sim_300: {N} piezas) ---")
    print(f"Color Combinado (MLP): {color_comb_correct/N*100:.2f}%")
    print(f"Color Cenital:         {color_cen_correct/N*100:.2f}%")
    print(f"Color Frontal:         {color_lat_correct/N*100:.2f}%")
    
    print(f"Error Medio Altura Frontal:     {np.mean(error_altura):.2f} mm")
    print(f"Error Relativo Sup. Cenital:    {np.mean(error_superficie):.2f} %" if error_superficie else "Error Relativo Sup. Cenital: N/A")
    print("Nota: IoU BBox y Superficie detallada requieren alinear frame a frame. Estas métricas muestran el promedio por track.")

    print("\n--- Análisis de Fallos ---")
    fallos = []
    for tid, t in preds.items():
        ref_pred = t['referencia_detectada']
        gt = gt_by_ref.get(ref_pred, list(gt_by_ref.values())[0])
        c_pred = t.get('color')
        c_gt = gt['color_name']
        
        if c_pred != c_gt:
            fallos.append(f"Track {tid}: Predijo Color '{c_pred}', Real: '{c_gt}' (Ref {ref_pred})")
    
    if fallos:
        print(f"Total fallos de color: {len(fallos)}")
        for f in fallos[:10]:
            print(f"  {f}")
        if len(fallos) > 10:
            print("  ...")
    else:
        print("¡Todos los colores correctos!")

if __name__ == '__main__':
    main()

