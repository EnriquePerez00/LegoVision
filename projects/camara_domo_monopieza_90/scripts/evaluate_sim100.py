import json
import numpy as np

def main():
    try:
        with open('projects/camara_domo_monopieza_90/data/simulation_100_inferencia') as f:
            preds = json.load(f)
    except Exception:
        with open('projects/camara_domo_monopieza_90/data/simulation_100/inferencia_consolidada.json') as f:
            preds = json.load(f)

    with open('projects/camara_domo_monopieza_90/data/simulation_100/simulation_metadata.json') as f:
        meta = json.load(f)

    gt_by_ref = {}
    for f in meta['frames']:
        for p in f['visible_pieces']:
            gt_by_ref[p['ref']] = p

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
        if sup_gt is not None and sup_gt > 0:
            error_superficie.append(abs(sup_pred - sup_gt) / sup_gt * 100)
            
    print(f"--- REPORTE DE EXACTITUD (sim_100: {N} piezas) ---")
    print(f"Color Combinado (MLP): {color_comb_correct/N*100:.2f}%")
    print(f"Color Cenital:         {color_cen_correct/N*100:.2f}%")
    print(f"Color Frontal:         {color_lat_correct/N*100:.2f}%")
    
    print(f"Error Medio Altura Frontal:     {np.mean(error_altura):.2f} mm")
    print(f"Error Relativo Sup. Cenital:    {np.mean(error_superficie):.2f} %" if error_superficie else "Error Relativo Sup. Cenital: N/A")

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
