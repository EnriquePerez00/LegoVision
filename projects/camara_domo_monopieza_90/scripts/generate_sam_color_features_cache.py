#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_sam_color_features_cache.py
Generates mlp_features_cache.npz with REAL 12D features using SAM masks.
No belt background contamination. Features match inference pipeline quality.
Usage: python scripts/generate_sam_color_features_cache.py
"""
import json, os, sys, time
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

def _rgb_to_lab_batch(rgb):
    arr = rgb.astype(np.float32) / 255.0
    m = arr > 0.04045
    arr[m] = ((arr[m] + 0.055) / 1.055) ** 2.4
    arr[~m] /= 12.92
    x = arr[:,0]*0.4124 + arr[:,1]*0.3576 + arr[:,2]*0.1805
    y = arr[:,0]*0.2126 + arr[:,1]*0.7152 + arr[:,2]*0.0722
    z = arr[:,0]*0.0193 + arr[:,1]*0.1192 + arr[:,2]*0.9505
    x/=0.95047; z/=1.08883
    def f(t):
        r=np.zeros_like(t); m2=t>0.008856
        r[m2]=t[m2]**(1/3); r[~m2]=7.787*t[~m2]+16/116; return r
    fx,fy,fz=f(x),f(y),f(z)
    return np.column_stack([116*fy-16, 500*(fx-fy), 200*(fy-fz)])

def extract_features_sam(img_path, bbox_norm, sam_model, min_pixels=10):
    """Extract 12D color features using SAM mask (no background)."""
    if not os.path.exists(img_path): return None
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    x1n,y1n,x2n,y2n = bbox_norm
    x1=max(0,int(x1n*W)); y1=max(0,int(y1n*H))
    x2=min(W,int(x2n*W)); y2=min(H,int(y2n*H))
    if x2-x1<4 or y2-y1<4: return None
    try:
        results = sam_model(img_rgb, bboxes=[[x1,y1,x2,y2]], verbose=False)
        if not results or results[0].masks is None or len(results[0].masks.data)==0:
            raise ValueError("no mask")
        mask = results[0].masks.data[0].cpu().numpy().astype(bool)
        if mask.shape!=(H,W):
            mask=cv2.resize(mask.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
        pixels_rgb = img_rgb[mask]
    except Exception:
        margin_x=max(2,int((x2-x1)*0.1)); margin_y=max(2,int((y2-y1)*0.1))
        pixels_rgb = img_rgb[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x].reshape(-1,3)
    if len(pixels_rgb)<min_pixels: return None
    px_r = pixels_rgb.reshape(-1,1,3)
    pixels_hsv = cv2.cvtColor(px_r,cv2.COLOR_RGB2HSV).reshape(-1,3).astype(np.float32)
    v_ok=pixels_hsv[:,2]>=15
    if v_ok.sum()>=min_pixels: pixels_rgb,pixels_hsv=pixels_rgb[v_ok],pixels_hsv[v_ok]
    sp_ok=(pixels_hsv[:,1]>=20)|(pixels_hsv[:,2]<235)
    if sp_ok.sum()>=min_pixels: pixels_rgb,pixels_hsv=pixels_rgb[sp_ok],pixels_hsv[sp_ok]
    if len(pixels_rgb)<min_pixels: return None
    pixels_lab=_rgb_to_lab_batch(pixels_rgb)
    q25,q75=np.percentile(pixels_lab[:,0],[25,75]); iqr=q75-q25
    ok=(pixels_lab[:,0]>=q25-1.2*iqr)&(pixels_lab[:,0]<=q75+1.2*iqr)
    if ok.sum()>=min_pixels: pixels_lab,pixels_hsv=pixels_lab[ok],pixels_hsv[ok]
    if len(pixels_lab)<min_pixels: return None
    ml=np.median(pixels_lab,axis=0); sl=pixels_lab.std(axis=0)
    mh=np.median(pixels_hsv,axis=0); sh=pixels_hsv.std(axis=0)
    return np.array([ml[0],sl[0],ml[1],sl[1],ml[2],sl[2],
                     mh[0],sh[0],mh[1],sh[1],mh[2],sh[2]],dtype=np.float32)

def main():
    import torch
    from ultralytics import SAM

    META_PATH = os.path.join(_ROOT,"data","simulation_x5_1D_all","simulation_metadata.json")
    DATA_DIR  = os.path.join(_ROOT,"data","simulation_x5_1D_all")
    CACHE_OUT = os.path.join(_ROOT,"data","mlp_features_cache.npz")
    SAM_MODEL = os.path.join(_ROOT,"..","..","mobile_sam.pt")
    if not os.path.exists(SAM_MODEL): SAM_MODEL = "mobile_sam.pt"

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("[generate_sam_cache] Device:", dev)
    print("Loading SAM...")
    sam = SAM(SAM_MODEL).to(dev)
    print("SAM loaded OK")

    meta = json.load(open(META_PATH,encoding="utf-8"))
    seen = {}; samples = []
    for frame in meta["frames"]:
        img_cen = os.path.join(DATA_DIR, frame["file_name"])
        offset = frame["belt_offset_mm"]
        for p in frame.get("visible_pieces",[]):
            x_abs = offset - p["x_belt_local_mm"]
            y_abs = p["y_belt_local_mm"]
            key = (round(x_abs,1), round(y_abs,1))
            if key not in seen:
                seen[key] = True
                samples.append({"img_cen":img_cen,"bbox":p["bbox_cenital_norm"],
                                 "color_name":p["color_name"],"color_code":str(p["color_code"])})
    total = len(samples)
    print("Procesando", total, "piezas unicas...")

    X_cen=[]; y_cen=[]; y_code=[]; n_ok=0; n_fail=0; t0=time.time()
    for i,s in enumerate(samples):
        if (i+1)%50==0:
            print("  ", i+1, "/", total, "(", round(time.time()-t0,1), "s) ok=",n_ok,"fail=",n_fail)
        feat = extract_features_sam(s["img_cen"], s["bbox"], sam)
        if feat is None: n_fail+=1; continue
        X_cen.append(feat); y_cen.append(s["color_name"]); y_code.append(s["color_code"]); n_ok+=1

    elapsed=time.time()-t0
    print("Extraccion completada:", n_ok, "OK,", n_fail, "FAIL |", round(elapsed,1), "s")
    X_cen_arr=np.array(X_cen,dtype=np.float32); y_cen_arr=np.array(y_cen); y_code_arr=np.array(y_code)
    np.savez(CACHE_OUT, X_cen=X_cen_arr, y_cen=y_cen_arr, y_code=y_code_arr,
             X_lat=X_cen_arr.copy(), y_lat=y_cen_arr.copy(), y_code_lat=y_code_arr.copy())
    print("Cache guardado:", CACHE_OUT)
    print("  X_cen shape:", X_cen_arr.shape)
    print("  Unique colors:", len(set(y_cen)))
    print("  L media:", round(float(X_cen_arr[:,0].mean()),1), "std:", round(float(X_cen_arr[:,0].std()),1))
    print("  (Should be ~50-75, NOT ~85+ like palette hex — SAM extracts real pixels)")

if __name__ == "__main__":
    main()
