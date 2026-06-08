"""run_evaluation_v6.py - New inference criteria."""
import os, sys, json, math, time
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, legovic_root)

from config_loader import cfg
from inference.knn_classifier import get_knn_classifier
from database.set_catalog import REAL_SETS

SELECTED_PARTS = cfg.pieces.selected_parts
PX_PER_MM = 3.2

PART_DIMS = {}
for ref in SELECTED_PARTS:
    dims = cfg.pieces.dimensions_mm.get(ref)
    PART_DIMS[ref] = tuple(dims) if dims else (8.0, 8.0, 9.6)

PART_COLORS = {}
for p in REAL_SETS["75078-1"]["parts"]:
    if p["ref"] in SELECTED_PARTS:
        PART_COLORS[p["ref"]] = p["color_hex"].replace("#", "").upper()

SURFACE_TOL = 0.25
HEIGHT_TOL = 0.25

def estimate_surface_mm2(w_px, h_px):
    return (w_px / PX_PER_MM) * (h_px / PX_PER_MM)

def estimate_height_mm(h_px):
    return h_px / PX_PER_MM

def estimate_color(crop_img):
    arr = np.array(crop_img.convert("RGB"))
    brightness = arr.mean(axis=2)
    belt_dist = np.sqrt((arr[:,:,0].astype(float)-37)**2 + (arr[:,:,1].astype(float)-65)**2 + (arr[:,:,2].astype(float)-84)**2)
    mask = (brightness > 40) & (belt_dist > 40)
    if mask.sum() < 20: return "A0A5A9"
    pixels = arr[mask]
    m = pixels.mean(axis=0).astype(int)
    return f"{m[0]:02X}{m[1]:02X}{m[2]:02X}"

def color_distance(h1, h2):
    r1,g1,b1 = int(h1[0:2],16), int(h1[2:4],16), int(h1[4:6],16)
    r2,g2,b2 = int(h2[0:2],16), int(h2[2:4],16), int(h2[4:6],16)
    return math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)

def get_possible_surfaces(ref):
    L, W, H = sorted(PART_DIMS[ref], reverse=True)
    return [L*W, L*H, W*H]

def get_possible_heights(ref):
    L, W, H = sorted(PART_DIMS[ref], reverse=True)
    return [H+0.9, H, W, L]

def filter_by_surface(meas, cands):
    p = [r for r in cands if any((1-SURFACE_TOL)*n <= meas <= (1+SURFACE_TOL)*n for n in get_possible_surfaces(r))]
    return p if p else cands

def filter_by_height(meas, cands):
    p = [r for r in cands if any((1-HEIGHT_TOL)*n <= meas <= (1+HEIGHT_TOL)*n for n in get_possible_heights(r))]
    return p if p else cands

def filter_by_color(est_color, cands):
    p = [r for r in cands if color_distance(est_color, PART_COLORS.get(r, "A0A5A9")) < 100]
    return p if p else cands

def main():
    test_dir = os.path.join(project_root, "data", "test_dual")
    with open(os.path.join(test_dir, "test_metadata.json")) as f:
        meta = json.load(f)

    print("=== EVALUATION v6 ===")
    print(f"Samples: {len(meta['renders'])}, Parts: {len(SELECTED_PARTS)}")

    clf = get_knn_classifier()
    clf.load_projection_head()
    clf.load_reference_embeddings()
    if not clf.is_ready():
        print("ERROR: classifier not ready"); sys.exit(1)

    results = []
    correct = total = 0
    part_stats = {}

    for idx, entry in enumerate(meta["renders"]):
        ref_gt = entry["ref"]
        cams = entry["cameras"]
        cen = cams.get("cenital")
        frt = cams.get("frontal")
        if not cen or not frt: continue

        # Load + crop cenital
        ci = Image.open(os.path.join(test_dir, cen["file_name"])).convert("RGB")
        iw, ih = ci.size
        cx1,cy1,cx2,cy2 = cen["bbox_norm"]
        crop_cen = ci.crop((int(cx1*iw), int(cy1*ih), int(cx2*iw), int(cy2*ih)))
        bw_c, bh_c = crop_cen.size

        # Load + crop frontal
        fi = Image.open(os.path.join(test_dir, frt["file_name"])).convert("RGB")
        fw, fh = fi.size
        fx1,fy1,fx2,fy2 = frt["bbox_norm"]
        crop_frt = fi.crop((int(fx1*fw), int(fy1*fh), int(fx2*fw), int(fy2*fh)))
        bw_f, bh_f = crop_frt.size

        # A) Surface, B) Height, C) Color
        surf = estimate_surface_mm2(bw_c, bh_c)
        height = estimate_height_mm(bh_f)
        col_cen = estimate_color(crop_cen)
        col_frt = estimate_color(crop_frt)
        col_match = color_distance(col_cen, col_frt) < 80

        # 1. Filter candidates
        cands = list(SELECTED_PARTS)
        cands = filter_by_surface(surf, cands)
        cands = filter_by_height(height, cands)
        if col_match:
            cands = filter_by_color(col_cen, cands)

        # 2. DINOv2 cenital
        cen_scores = {}
        canvas = Image.new("RGB", (224,224), (37,65,84))
        s = min(200/max(bw_c,1), 200/max(bh_c,1), 1.0)
        r = crop_cen.resize((max(1,int(bw_c*s)), max(1,int(bh_c*s))), Image.Resampling.LANCZOS)
        canvas.paste(r, ((224-r.width)//2, (224-r.height)//2))
        cen_refs = [e for e in clf._ref_embeddings if e["face"]%10==1 and e["part_ref"] in cands]
        if cen_refs:
            qv = clf._extract_embedding(canvas, size_info=(bw_c/PX_PER_MM, bh_c/PX_PER_MM))
            rm = np.stack([e["embedding"] for e in cen_refs])
            sc = rm @ qv
            for i,e in enumerate(cen_refs):
                if e["part_ref"] not in cen_scores or sc[i] > cen_scores[e["part_ref"]]:
                    cen_scores[e["part_ref"]] = float(sc[i])

        # 3. DINOv2 frontal
        frt_scores = {}
        canvas = Image.new("RGB", (224,224), (37,65,84))
        s = min(200/max(bw_f,1), 200/max(bh_f,1), 1.0)
        r = crop_frt.resize((max(1,int(bw_f*s)), max(1,int(bh_f*s))), Image.Resampling.LANCZOS)
        canvas.paste(r, ((224-r.width)//2, (224-r.height)//2))
        frt_refs = [e for e in clf._ref_embeddings if e["face"]%10==2 and e["part_ref"] in cands]
        if frt_refs:
            qv = clf._extract_embedding(canvas, size_info=(bw_f/PX_PER_MM, bh_f/PX_PER_MM))
            rm = np.stack([e["embedding"] for e in frt_refs])
            sc = rm @ qv
            for i,e in enumerate(frt_refs):
                if e["part_ref"] not in frt_scores or sc[i] > frt_scores[e["part_ref"]]:
                    frt_scores[e["part_ref"]] = float(sc[i])

        # 4. Decision
        best_c = max(cen_scores, key=cen_scores.get) if cen_scores else None
        best_f = max(frt_scores, key=frt_scores.get) if frt_scores else None

        if col_match and best_c and best_f and best_c == best_f:
            pred, status = best_c, "AGREE"
        elif not col_match:
            pred, status = best_c or best_f or "unknown", "COLOR_ERR"
        else:
            combined = {r: 0.7*cen_scores.get(r,0) + 0.3*frt_scores.get(r,0) for r in cands}
            pred = max(combined, key=combined.get) if combined else "unknown"
            status = "DISAGREE"

        is_ok = (pred == ref_gt)
        total += 1
        if is_ok: correct += 1
        part_stats.setdefault(ref_gt, {"c":0,"t":0})
        part_stats[ref_gt]["t"] += 1
        if is_ok: part_stats[ref_gt]["c"] += 1

        mark = "V" if is_ok else "X"
        print(f"[{idx+1:03d}] GT={ref_gt} P={pred} {mark} st={status} cands={len(cands)} s={surf:.0f} h={height:.1f}")
        results.append({"gt":ref_gt,"pred":pred,"ok":is_ok,"status":status})

    acc = correct/max(1,total)
    print(f"\n=== RESULT: {correct}/{total} = {100*acc:.1f}% ===")
    print(f"\n{'Ref':<8}|{'Acc':<8}|{'C/T'}")
    for ref,s in sorted(part_stats.items(), key=lambda x: x[1]["c"]/max(1,x[1]["t"])):
        print(f"{ref:<8}|{100*s['c']/max(1,s['t']):>5.1f}%|{s['c']}/{s['t']}")

    report = {"accuracy":acc,"correct":correct,"total":total,"per_piece":part_stats,"results":results}
    rp = os.path.join(test_dir, "eval_report_v6.json")
    with open(rp,"w") as f: json.dump(report, f, indent=2)
    print(f"\nReport: {rp}")

if __name__ == "__main__":
    main()
