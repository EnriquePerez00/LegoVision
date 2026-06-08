# -*- coding: utf-8 -*-
"""training/eval_on_set_simulation.py - Evaluacion post-entrenamiento YOLO en simulacion fisica"""
import os, sys, json, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def calculate_iou(b1, b2):
    xc1,yc1,w1,h1=b1; xc2,yc2,w2,h2=b2
    x1a,x1b=xc1-w1/2,xc1+w1/2; y1a,y1b=yc1-h1/2,yc1+h1/2
    x2a,x2b=xc2-w2/2,xc2+w2/2; y2a,y2b=yc2-h2/2,yc2+h2/2
    ix=max(0.0,min(x1b,x2b)-max(x1a,x2a)); iy=max(0.0,min(y1b,y2b)-max(y1a,y2a))
    inter=ix*iy; union=w1*h1+w2*h2-inter
    return inter/union if union>1e-9 else 0.0

def bbox_norm_to_yolo(n):
    x1,y1,x2,y2=n; return [(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]

def _nms(dets, thr=0.45):
    if not dets: return []
    ds=sorted(dets,key=lambda d:d["conf"],reverse=True)
    kept=[]; used=[False]*len(ds)
    for i,d in enumerate(ds):
        if used[i]: continue
        kept.append(d)
        for j in range(i+1,len(ds)):
            if not used[j] and calculate_iou(d["bbox"],ds[j]["bbox"])>thr: used[j]=True
    return kept

def detect_sliding(model, image_path, conf=0.25):
    from PIL import Image as PI
    img=PI.open(image_path).convert("RGB"); W,H=img.size; win=640; stride=480
    ys=list(range(0,max(1,H-win+1),stride))
    if not ys or ys[-1]+win<H: ys.append(max(0,H-win))
    all_d=[]
    for y0 in ys:
        y1=min(y0+win,H); crop=img.crop((0,y0,W,y1))
        if crop.size[1]<64: continue
        for r in model(crop,conf=conf,verbose=False):
            for box in r.boxes:
                xc,yc,wc,hc=box.xywhn[0].tolist(); c=float(box.conf[0])
                all_d.append({"bbox":[xc,(y0+yc*(y1-y0))/H,wc,hc*(y1-y0)/H],"conf":c})
    return _nms(all_d)

def evaluate(model_path, sim_img, sim_json, conf=0.25, iou_thr=0.50):
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"error":"ultralytics no instalado"}
    for p,l in [(model_path,"modelo"),(sim_img,"imagen"),(sim_json,"JSON")]:
        if not os.path.exists(p): return {"error":f"{l} no encontrado: {p}"}
    print(f"[Eval] Cargando modelo: {model_path}")
    model=YOLO(model_path)
    with open(sim_json,"r",encoding="utf-8") as f: meta=json.load(f)
    gt_list=[]
    for d in meta.get("detections",[]):
        bq=d.get("bbox_yolo") or (bbox_norm_to_yolo(d["bbox_norm"]) if "bbox_norm" in d else None)
        if bq is None or bq[2]<0.005 or bq[3]<0.005: continue
        gt_list.append({"bbox":bq,"ref":d.get("ref","unknown")})
    print(f"[Eval] GT valido: {len(gt_list)} piezas")
    if not gt_list: return {"error":"No hay GT valido en JSON"}
    print(f"[Eval] Ejecutando YOLO (conf={conf})...")
    dets=detect_sliding(model,sim_img,conf=conf)
    print(f"[Eval] Detecciones: {len(dets)}")
    matched=set(); tp=fp=0; per={}
    for det in sorted(dets,key=lambda d:d["conf"],reverse=True):
        bi,bv=-1,0.0
        for idx,gt in enumerate(gt_list):
            if idx in matched: continue
            v=calculate_iou(det["bbox"],gt["bbox"])
            if v>bv: bv,bi=v,idx
        if bi>=0 and bv>=iou_thr:
            tp+=1; matched.add(bi); ref=gt_list[bi]["ref"]
            per.setdefault(ref,{"tp":0,"fn":0})["tp"]+=1
        else:
            fp+=1
    fn=len(gt_list)-len(matched)
    for idx,gt in enumerate(gt_list):
        if idx not in matched: per.setdefault(gt["ref"],{"tp":0,"fn":0})["fn"]+=1
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    map50=prec*rec
    pps=[]
    for ref,s in sorted(per.items()):
        tot=s["tp"]+s["fn"]
        pps.append({"ref":ref,"tp":s["tp"],"fn":s["fn"],"total_gt":tot,"recall":round(s["tp"]/tot,3) if tot>0 else 0.0})
    res={"model_path":os.path.basename(model_path),"simulation_image":os.path.basename(sim_img),
         "gt_total":len(gt_list),"detections_total":len(dets),"tp":tp,"fp":fp,"fn":fn,
         "precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),"map50":round(map50,4),
         "iou_threshold":iou_thr,"conf_threshold":conf,"per_piece_stats":pps}
    print(f"\n[Eval] TP={tp} FP={fp} FN={fn} | Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} mAP50={map50:.3f}")
    return res

if __name__=="__main__":
    pa=argparse.ArgumentParser(); pa.add_argument("--model",default=os.path.join(project_root,"models","best.pt"))
    pa.add_argument("--image",default=os.path.join(project_root,"data","synthetic_renders","set_scatter_75078-1.png"))
    pa.add_argument("--json",default=os.path.join(project_root,"data","synthetic_renders","set_scatter_75078-1.json"))
    pa.add_argument("--conf",type=float,default=0.25); pa.add_argument("--iou",type=float,default=0.50)
    pa.add_argument("--output",default=None); args=pa.parse_args()
    res=evaluate(args.model,args.image,args.json,args.conf,args.iou)
    if args.output:
        with open(args.output,"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
    else:
        print(json.dumps(res,indent=2,ensure_ascii=False))
