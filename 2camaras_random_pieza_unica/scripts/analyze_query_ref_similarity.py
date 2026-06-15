#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_query_ref_similarity.py
Similitud coseno DINOv2+SAM entre query (test_500allhd) y refs (dinov2_refs_v4_canonical).
100 muestras aleatorias.
"""
import os, sys, json, random, time, collections, statistics
import numpy as np
from PIL import Image
import torch

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
repo_root    = os.path.dirname(project_root)
sys.path.insert(0, repo_root)
sys.path.insert(0, project_root)

N_SAMPLES      = 100
RANDOM_SEED    = 42
CANVAS_SIZE    = 224
MARGIN_PX      = 8
CINTA_BG       = (37, 65, 84)
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
METADATA_PATH  = os.path.join(project_root, "data", "test_500allhd", "random_500_metadata.json")
REFS_DIR_CEN   = os.path.join(project_root, "data", "dinov2_refs_v4_canonical", "cenital")
REFS_DIR_LAT   = os.path.join(project_root, "data", "dinov2_refs_v4_canonical", "lateral")
SAM_PATH       = os.path.join(project_root, "mobile_sam.pt")
TEST_DIR       = os.path.join(project_root, "data", "test_500allhd")
REF_META_DIR   = os.path.join(project_root, "data", "dinov2_refs_v4_canonical")

SEP = "=" * 72

with open(METADATA_PATH) as f:
    data = json.load(f)
renders = data["renders"]
random.seed(RANDOM_SEED)
samples = [renders[i] for i in random.sample(range(len(renders)), min(N_SAMPLES, len(renders)))]

print(SEP)
print("  SIMILITUD QUERY<->REF — DINOv2 + SAM  (100 muestras aleatorias)")
print(SEP)
print(f"  Device: {DEVICE}  |  SAM: {SAM_PATH}")
print()

# ── Cargar modelos ──
print("[1/4] Cargando SAM...")
t0 = time.perf_counter()
from ultralytics import SAM as USAM
sam_model = USAM(SAM_PATH)
print(f"      OK {time.perf_counter()-t0:.1f}s")

print("[2/4] Cargando DINOv2 vits14...")
t0 = time.perf_counter()
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                       source="local", pretrained=True, verbose=False)
dino.eval().to(DEVICE)
print(f"      OK {time.perf_counter()-t0:.1f}s")

from torchvision import transforms
dino_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Helpers ──
def canvas(img, sz=224, mg=8, bg=CINTA_BG):
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (sz, sz), bg)
    md = sz - 2*mg
    sc = min(md/w, md/h)
    nw, nh = max(1,int(round(w*sc))), max(1,int(round(h*sc)))
    r = img.convert("RGB").resize((nw,nh), Image.Resampling.LANCZOS)
    c = Image.new("RGB", (sz,sz), bg)
    c.paste(r, ((sz-nw)//2, (sz-nh)//2))
    return c

def sam_crop(img_pil, bbox_norm):
    """Aplica SAM y devuelve crop enmascarado. Fallback: crop directo."""
    try:
        img = img_pil.convert("RGB")
        W, H = img.size
        x1,y1,x2,y2 = (max(0,int(bbox_norm[0]*W)), max(0,int(bbox_norm[1]*H)),
                        min(W,int(bbox_norm[2]*W)), min(H,int(bbox_norm[3]*H)))
        if x2<=x1 or y2<=y1:
            return img.crop((x1,y1,x2,y2)), False
        arr = np.array(img)
        res = sam_model(arr, bboxes=[[x1,y1,x2,y2]], verbose=False)
        if res and res[0].masks is not None:
            mfull = res[0].masks.data[0].cpu().numpy().astype(np.uint8)*255
            cmask = mfull[y1:y2, x1:x2].copy()
            carr  = arr[y1:y2, x1:x2].copy()
            hc,wc = carr.shape[:2]
            if hc>2 and wc>2:
                edges = np.vstack([carr[0,:],carr[-1,:],carr[:,0],carr[:,-1]])
                bg_col = edges.mean(axis=0)
                dist = np.linalg.norm(carr.astype(np.float32)-bg_col, axis=-1)
                cmask[dist<25] = 0
            carr[cmask==0] = CINTA_BG
            return Image.fromarray(carr), True
        return img.crop((x1,y1,x2,y2)), False
    except Exception:
        try:
            img = img_pil.convert("RGB")
            W,H = img.size
            x1,y1,x2,y2 = (max(0,int(bbox_norm[0]*W)), max(0,int(bbox_norm[1]*H)),
                            min(W,int(bbox_norm[2]*W)), min(H,int(bbox_norm[3]*H)))
            return img.crop((x1,y1,x2,y2)), False
        except Exception:
            return Image.new("RGB",(64,64),CINTA_BG), False

def embed(img_pil):
    c = canvas(img_pil)
    t = dino_tfm(c).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e = dino(t)[0].cpu().numpy().astype(np.float32)
    n = np.linalg.norm(e)
    return e/n if n>1e-8 else e

def find_refs(ref, pose_idx, color_hex, cam_dir):
    """Busca refs para (ref, pose_idx). Primero por color exacto, luego cualquier color."""
    hx = (color_hex or "").lstrip("#").upper()
    ps = f"pose{pose_idx:02d}"
    try:
        all_f = os.listdir(cam_dir)
    except Exception:
        return []
    # exacto
    pref = f"ref_{ref}_{hx}_{ps}_"
    ex = sorted([f for f in all_f if f.startswith(pref) and f.endswith(".png")])
    if ex:
        return [os.path.join(cam_dir, f) for f in ex]
    # cualquier color, misma pose
    ac = sorted([f for f in all_f if f.startswith(f"ref_{ref}_") and f"_{ps}_" in f and f.endswith(".png")])
    if ac:
        return [os.path.join(cam_dir, f) for f in ac]
    # cualquier pose
    ap = sorted([f for f in all_f if f.startswith(f"ref_{ref}_") and f.endswith(".png")])
    return [os.path.join(cam_dir, f) for f in ap[:12]]

def best_sim(q_emb, ref_paths, bbox_dict):
    """Calcula similitud coseno query vs cada ref. Devuelve stats."""
    sims, best, bfile = [], -2.0, None
    for rp in ref_paths:
        fn = os.path.basename(rp)
        bb = bbox_dict.get(fn, [0.35,0.35,0.65,0.65])
        try:
            rimg = Image.open(rp).convert("RGB")
            rcrop, _ = sam_crop(rimg, bb)
            re = embed(rcrop)
            s = float(np.dot(q_emb, re))
            sims.append(s)
            if s > best:
                best, bfile = s, fn
        except Exception:
            pass
    if not sims:
        return None, None, None, 0, None
    return max(sims), statistics.mean(sims), min(sims), len(sims), bfile

# ── Precargar bbox de refs ──
print("[3/4] Cargando bbox de refs...")
rbcen, rblat = {}, {}
for fn in os.listdir(REF_META_DIR):
    if not fn.startswith("metadata_worker") or not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(REF_META_DIR, fn)) as f:
            rm = json.load(f)
        for rr in rm.get("renders", []):
            rfn = rr.get("file_name","")
            cams = rr.get("cameras",{})
            if cams.get("cenital"):
                rbcen[rfn] = cams["cenital"].get("bbox_norm",[0.35,0.35,0.65,0.65])
            if cams.get("lateral"):
                rblat[rfn] = cams["lateral"].get("bbox_norm",[0.35,0.35,0.65,0.65])
    except Exception:
        pass
print(f"      cen={len(rbcen)} lat={len(rblat)} refs bbox")

# ── BUCLE ──
print(f"[4/4] Procesando {len(samples)} muestras...")
print()

rows = []
t_start = time.perf_counter()
sam_ok_cen = 0
sam_ok_lat = 0

for i, r in enumerate(samples):
    ref_gt    = r["ref"]
    pose_idx  = r["pose_index"]
    color_hex = r.get("color_hex","")
    fc        = r.get("face_class","?")
    color_nm  = r.get("color_name","?")
    cams      = r.get("cameras",{})
    cen_meta  = cams.get("cenital",{})
    lat_meta  = cams.get("lateral",{})

    cen_file = os.path.join(TEST_DIR, cen_meta.get("file_name",""))
    lat_file = os.path.join(TEST_DIR, lat_meta.get("file_name",""))
    cen_bbox = cen_meta.get("bbox_norm",[0.4,0.4,0.6,0.6])
    lat_bbox = lat_meta.get("bbox_norm",[0.4,0.4,0.6,0.6])

    # Embedding query cenital
    cen_max, cen_mean, cen_min, cen_nrefs, cen_bfile = None, None, None, 0, None
    lat_max, lat_mean, lat_min, lat_nrefs, lat_bfile = None, None, None, 0, None
    q_sam_cen, q_sam_lat = False, False

    if os.path.exists(cen_file):
        try:
            img_cen = Image.open(cen_file).convert("RGB")
            q_crop_cen, q_sam_cen = sam_crop(img_cen, cen_bbox)
            q_emb_cen = embed(q_crop_cen)
            ref_paths_cen = find_refs(ref_gt, pose_idx, color_hex, REFS_DIR_CEN)
            cen_max, cen_mean, cen_min, cen_nrefs, cen_bfile = best_sim(q_emb_cen, ref_paths_cen, rbcen)
            if q_sam_cen:
                sam_ok_cen += 1
        except Exception as e:
            pass

    if os.path.exists(lat_file):
        try:
            img_lat = Image.open(lat_file).convert("RGB")
            q_crop_lat, q_sam_lat = sam_crop(img_lat, lat_bbox)
            q_emb_lat = embed(q_crop_lat)
            ref_paths_lat = find_refs(ref_gt, pose_idx, color_hex, REFS_DIR_LAT)
            lat_max, lat_mean, lat_min, lat_nrefs, lat_bfile = best_sim(q_emb_lat, ref_paths_lat, rblat)
            if q_sam_lat:
                sam_ok_lat += 1
        except Exception as e:
            pass

    row = {
        "idx": r["index"], "ref": ref_gt, "pose": pose_idx,
        "fc": fc, "color": color_nm, "color_hex": color_hex,
        "cen_max": cen_max, "cen_mean": cen_mean, "cen_min": cen_min,
        "cen_nrefs": cen_nrefs, "cen_sam": q_sam_cen,
        "lat_max": lat_max, "lat_mean": lat_mean, "lat_min": lat_min,
        "lat_nrefs": lat_nrefs, "lat_sam": q_sam_lat,
    }
    rows.append(row)

    if (i+1) % 10 == 0 or i == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i+1) * (len(samples)-i-1)
        cstr = f"cen_max={cen_max:.3f}" if cen_max else "cen=N/A"
        lstr = f"lat_max={lat_max:.3f}" if lat_max else "lat=N/A"
        print(f"  [{i+1:3d}/{len(samples)}] {ref_gt:8s} p{pose_idx} {fc:8s} | {cstr} {lstr} | ETA {eta:.0f}s")

elapsed_total = time.perf_counter() - t_start
print()
print(f"  Procesado en {elapsed_total:.1f}s  ({elapsed_total/len(rows):.1f}s/muestra)")
print(f"  SAM exitoso: cen={sam_ok_cen}/{len(rows)} lat={sam_ok_lat}/{len(rows)}")

# ── REPORTE ESTADISTICO ──
print()
print(SEP)
print("  RESULTADOS GLOBALES")
print(SEP)

valid_cen = [r for r in