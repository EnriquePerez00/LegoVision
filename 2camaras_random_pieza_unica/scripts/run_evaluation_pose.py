# -*- coding: utf-8 -*-
"""run_evaluation_pose.py — Fase 5: YOLO-Pose + triangulacion DLT + DINOv2."""
from __future__ import annotations
import json, math, os, sys, time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, project_root); sys.path.insert(0, legovic_root)
from config_loader import cfg
from database.set_catalog import REAL_SETS
try:
    from logger import get_logger; log = get_logger("eval_pose")
except Exception:
    class _L:
        def info(self,m): print(m)
        def warning(self,m): print(f"[WARN] {m}")
        def error(self,m): print(f"[ERROR] {m}")
    log = _L()

SELECTED_PARTS = list(cfg.pieces.selected_parts)
PX_PER_MM = 3.2
PART_DIMS = {}
for _r in SELECTED_PARTS:
    _d = getattr(cfg.pieces.dimensions_mm, _r, None)
    PART_DIMS[_r] = tuple(float(x) for x in _d) if _d else (8.,8.,9.6)
PART_COLORS = {}
for _p in REAL_SETS["75078-1"]["parts"]:
    if _p["ref"] in SELECTED_PARTS:
        PART_COLORS[_p["ref"]] = _p.get("color_hex","#A0A5A9").replace("#","").upper()
SURFACE_TOL, HEIGHT_TOL = 0.75, 0.75

def _surfaces(r): L,W,H=sorted(PART_DIMS[r],reverse=True); return [L*W,L*H,W*H]
def _heights(r):  L,W,H=sorted(PART_DIMS[r],reverse=True); return [H+0.9,H,W,L]
def filt_surf(m,c): o=[r for r in c if any((1-SURFACE_TOL)*s<=m<=(1+SURFACE_TOL)*s for s in _surfaces(r))]; return o or c
def filt_h(m,c):   o=[r for r in c if any((1-HEIGHT_TOL)*h<=m<=(1+HEIGHT_TOL)*h for h in _heights(r))]; return o or c
def cdist(h1,h2):
    r1,g1,b1=int(h1[:2],16),int(h1[2:4],16),int(h1[4:],16)
    r2,g2,b2=int(h2[:2],16),int(h2[2:4],16),int(h2[4:],16)
    return math.sqrt((r1-r2)**2+(g1-g2)**2+(b1-b2)**2)
def filt_col(e,c): o=[r for r in c if cdist(e,PART_COLORS.get(r,"A0A5A9"))<110]; return o or c
def est_color(crop):
    import numpy as np
    arr=np.array(crop.convert("RGB")); bri=arr.mean(axis=2)
    belt=np.sqrt((arr[:,:,0].astype(float)-37)**2+(arr[:,:,1].astype(float)-65)**2+(arr[:,:,2].astype(float)-84)**2)
    mask=(bri>40)&(belt>40)
    if mask.sum()<20: return "A0A5A9"
    m=arr[mask].mean(axis=0).astype(int); return f"{m[0]:02X}{m[1]:02X}{m[2]:02X}"

def main():
    import argparse
    from PIL import Image
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_dir",default=os.path.join(project_root,"data","300"))
    ap.add_argument("--conf",type=float,default=0.25)
    ap.add_argument("--kp_conf",type=float,default=0.3)
    ap.add_argument("--pose_model_cenital",default=os.path.join(project_root,"models","yolo_cenital_pose.pt"))
    ap.add_argument("--pose_model_lateral",default=os.path.join(project_root,"models","yolo_lateral_pose.pt"))
    ap.add_argument("--det_model_cenital",default=os.path.join(project_root,"models","yolo_cenital.pt"))
    args=ap.parse_args()

    from ultralytics import YOLO
    pose_cen=YOLO(args.pose_model_cenital) if os.path.isfile(args.pose_model_cenital) else None
    pose_lat=YOLO(args.pose_model_lateral) if os.path.isfile(args.pose_model_lateral) else None
    det_cen =YOLO(args.det_model_cenital)  if os.path.isfile(args.det_model_cenital)  else None
    log.info(f"[models] pose_cen={'OK' if pose_cen else 'N/A'} pose_lat={'OK' if pose_lat else 'N/A'} det_cen={'OK' if det_cen else 'N/A'}")

    HAS_DINO=False; clf=None
    try:
        from inference.knn_classifier import get_knn_classifier
        clf=get_knn_classifier(); clf.load_projection_head(); clf.load_reference_embeddings()
        HAS_DINO=clf.is_ready(); log.info(f"[dino] ready={HAS_DINO}")
    except Exception as e: log.warning(f"[dino] {e}")

    from _kpts_observer import extract_yolo_pose_keypoints, kpts_observer
    meta_path=os.path.join(args.test_dir,"test_metadata.json")
    if not os.path.isfile(meta_path): log.error(f"No existe: {meta_path}"); sys.exit(1)
    with open(meta_path) as f: meta=json.load(f)
    samples=meta.get("renders",meta.get("samples",[]))
    log.info(f"[eval] {len(samples)} muestras | {len(SELECTED_PARTS)} partes")

    results,part_stats=[],{}; correct=total=0; t0=time.time()
    for idx,entry in enumerate(samples):
        ref_gt=entry.get("ref",entry.get("part_ref","unknown"))
        cams=entry.get("cameras",{})
        cen_info=cams.get("cenital") or cams.get("top")
        lat_info=cams.get("lateral") or cams.get("front") or cams.get("frontal")
        if not cen_info: continue
        cen_path=os.path.join(args.test_dir,cen_info["file_name"])
        lat_path=os.path.join(args.test_dir,lat_info["file_name"]) if lat_info else None
        img_cen=Image.open(cen_path).convert("RGB")
        img_lat=Image.open(lat_path).convert("RGB") if lat_path else None

        kps_cen=extract_yolo_pose_keypoints(pose_cen,cen_path,conf=args.conf) if pose_cen else None
        kps_lat=extract_yolo_pose_keypoints(pose_lat,lat_path,conf=args.conf) if (pose_lat and lat_path) else None
        obs={}; fp=None; ht=None
        if kps_cen is not None and kps_lat is not None:
            obs=kpts_observer(kps_cen,kps_lat,conf_min=args.kp_conf)
            fp=obs.get("footprint_area_mm2"); ht=obs.get("lateral_height_mm")

        if fp is None and det_cen is not None:
            dr=det_cen(cen_path,verbose=False,conf=args.conf)
            if dr and dr[0].boxes is not None and len(dr[0].boxes)>0:
                iw,ih=img_cen.size; b=dr[0].boxes.xywhn[0].cpu().numpy()
                fp=float(b[2])*iw/PX_PER_MM*float(b[3])*ih/PX_PER_MM
        if ht is None and lat_info and img_lat:
            bn=lat_info.get("bbox_norm")
            if bn: _,ih_l=img_lat.size; ht=(bn[3]-bn[1])*ih_l/PX_PER_MM

        crop_cen=None
        bn_cen=cen_info.get("bbox_norm")
        if bn_cen:
            iw,ih=img_cen.size; x1,y1,x2,y2=bn_cen
            crop_cen=img_cen.crop((int(x1*iw),int(y1*ih),int(x2*iw),int(y2*ih)))

        cands=list(SELECTED_PARTS)
        if fp and fp>1.0: cands=filt_surf(fp,cands)
        if ht and ht>0.5: cands=filt_h(ht,cands)
        if crop_cen is not None: cands=filt_col(est_color(crop_cen),cands)

        pred=None
        if HAS_DINO and crop_cen is not None and cands:
            try:
                import numpy as np
                bw,bh=crop_cen.width,crop_cen.height
                canvas=Image.new("RGB",(224,224),(37,65,84))
                s=min(200/max(bw,1),200/max(bh,1),1.0)
                r=crop_cen.resize((max(1,int(bw*s)),max(1,int(bh*s))),Image.Resampling.LANCZOS)
                canvas.paste(r,((224-r.width)//2,(224-r.height)//2))
                refs=[e for e in clf._ref_embeddings if e["face"]%10==1 and e["part_ref"] in cands]
                if refs:
                    qv=clf._extract_embedding(canvas,size_info=(bw/PX_PER_MM,bh/PX_PER_MM))
                    sc=np.stack([e["embedding"] for e in refs])@qv
                    pred=refs[int(sc.argmax())]["part_ref"]
            except Exception as ex: log.warning(f"[{idx}] dino: {ex}")
        if pred is None: pred=cands[0] if cands else "unknown"

        is_ok=(pred==ref_gt); total+=1
        if is_ok: correct+=1
        part_stats.setdefault(ref_gt,{"c":0,"t":0}); part_stats[ref_gt]["t"]+=1
        if is_ok: part_stats[ref_gt]["c"]+=1
        nv=obs.get("n_valid",0); mark="V" if is_ok else "X"
        fp_s=f"{fp:.0f}" if fp else "N/A"; h_s=f"{ht:.1f}" if ht else "N/A"
        print(f"[{idx+1:03d}] GT={ref_gt:<8} P={pred:<8} {mark} cands={len(cands)} kp3d={nv} fp={fp_s}mm2 h={h_s}mm")
        results.append({"gt":ref_gt,"pred":pred,"ok":is_ok,"n_valid":nv,"footprint_mm2":fp,"height_mm":ht})

    acc=correct/max(1,total); elapsed=time.time()-t0
    print(f"\n=== RESULT: {correct}/{total} = {100*acc:.1f}% | {elapsed:.1f}s ===\n")
    print(f"{'Ref':<8}|{'Acc':>6}|C/T")
    for r,s in sorted(part_stats.items(),key=lambda x:x[1]["c"]/max(1,x[1]["t"])):
        print(f"{r:<8}|{100*s['c']/max(1,s['t']):>5.1f}%|{s['c']}/{s['t']}")
    rp=os.path.join(args.test_dir,"eval_report_pose.json")
    with open(rp,"w") as f:
        json.dump({"accuracy":acc,"correct":correct,"total":total,"per_piece":part_stats,"results":results},f,indent=2)
    log.info(f"Report: {rp}")

if __name__=="__main__":
    main()
