# -*- coding: utf-8 -*-
"""color_classifier_v2.py v2.1 — M1+M2+M3 improvements."""
from __future__ import annotations
import json, math, os
from typing import Dict, List, Optional, Tuple
import numpy as np, torch, torch.nn as nn

MAT_SOLID="SOLID"; MAT_TRANS="TRANSPARENT"; MAT_METALLIC="METALLIC"
MAT_PEARL="PEARL"; MAT_SPECIAL="SPECIAL"; MAT_WHITE="WHITE_BUCKET"; MAT_BLACK="BLACK_BUCKET"
_TRANS_KW=("trans-","glitter trans","satin trans","glow in dark trans")
_METALLIC_KW=("chrome ","metallic ","flat silver","flat dark gold",
    "bionicle silver","bionicle gold","bionicle copper","reddish gold")
_PEARL_KW=("pearl ","speckle ")
_SPECIAL_KW=("electric_contact","magnet","umber","sienna")

def _palette_material(n):
    if any(k in n for k in _TRANS_KW): return MAT_TRANS
    if any(k in n for k in _METALLIC_KW): return MAT_METALLIC
    if any(k in n for k in _PEARL_KW): return MAT_PEARL
    if any(k in n for k in _SPECIAL_KW): return MAT_SPECIAL
    return MAT_SOLID

FALSE_ATTRACTORS = {
    "magnet", "electric_contact_alloy", "electric_contact_copper",
    "various", "none",
    "pearl black",    # M4: dark attractor absorbs many dark pieces
    "chrome silver",  # M4b: bright metallic absorbs white/light pieces
    "chrome gold",    # M4c: yellow metallic absorbs yellow/gold pieces
}
L_BIAS=18.0
CANONICAL_CODE:Dict[str,int]={"pearl dark gray":77,"flat silver":95,"dark blue":63,
    "dark green":80,"dark red":59,"medium azure":156,"dark bluish gray":85,
    "yellow":3,"blue":7,"trans-green":20}
_MAT_CLASSES=[MAT_SOLID,MAT_TRANS,MAT_METALLIC,MAT_PEARL,MAT_SPECIAL]

class ColorMLPV3(nn.Module):
    """M3: Deeper MLP 18->256->128->64->N with GELU+BN. Hard Neg Mining via embedder."""
    def __init__(self, input_dim=19, num_classes=174, embed_dim=32):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.GELU())
        self.classifier=nn.Linear(64,num_classes)
        self.embedder=nn.Sequential(nn.Linear(64,embed_dim),nn.BatchNorm1d(embed_dim))
    def forward(self, x, return_embedding=False):
        f=self.encoder(x); logits=self.classifier(f)
        return (logits,self.embedder(f)) if return_embedding else logits

class _MatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,len(_MAT_CLASSES)))
    def forward(self,x): return self.net(x)

def _rgb_to_lab(rgb):
    r,g,b=rgb[0]/255.,rgb[1]/255.,rgb[2]/255.
    r=((r+0.055)/1.055)**2.4 if r>0.04045 else r/12.92
    g=((g+0.055)/1.055)**2.4 if g>0.04045 else g/12.92
    b=((b+0.055)/1.055)**2.4 if b>0.04045 else b/12.92
    x=r*0.4124+g*0.3576+b*0.1805; y=r*0.2126+g*0.7152+b*0.0722; z=r*0.0193+g*0.1192+b*0.9505
    x/=0.95047; z/=1.08883
    def f(t): return t**(1/3) if t>0.008856 else 7.787*t+16/116
    return np.array([116*f(y)-16,500*(f(x)-f(y)),200*(f(y)-f(z))],dtype=float)

def _ciede2000(lab1,lab2,wL=1.,wC=1.,wH=1.):
    L1,a1,b1=lab1; L2,a2,b2=lab2
    C1=math.sqrt(a1**2+b1**2); C2=math.sqrt(a2**2+b2**2); Cb=(C1+C2)/2
    G=0.5*(1-math.sqrt(Cb**7/(Cb**7+25.**7)))
    a1p,a2p=a1*(1+G),a2*(1+G)
    C1p=math.sqrt(a1p**2+b1**2); C2p=math.sqrt(a2p**2+b2**2); Cbp=(C1p+C2p)/2
    h1p=math.atan2(b1,a1p)%(2*math.pi); h2p=math.atan2(b2,a2p)%(2*math.pi)
    Hbp=(h1p+h2p)/2
    if abs(h1p-h2p)>math.pi: Hbp=(h1p+h2p+2*math.pi)/2
    T=(1-0.17*math.cos(Hbp-math.pi/6)+0.24*math.cos(2*Hbp)
       +0.32*math.cos(3*Hbp+math.pi/30)-0.20*math.cos(4*Hbp-63*math.pi/180))
    dhp=h2p-h1p
    if abs(dhp)>math.pi: dhp+=2*math.pi if h2p<=h1p else -2*math.pi
    dLp=L2-L1; dCp=C2p-C1p; dHp=2*math.sqrt(C1p*C2p)*math.sin(dhp/2)
    Lm=(L1+L2)/2
    SL=1+0.015*(Lm-50)**2/math.sqrt(20+(Lm-50)**2)
    SC=1+0.045*Cbp; SH=1+0.015*Cbp*T
    RC=2*math.sqrt(Cbp**7/(Cbp**7+25.**7))
    dTh=30*math.pi/180*math.exp(-((Hbp-275*math.pi/180)/(25*math.pi/180))**2)
    RT=-math.sin(2*dTh)*RC
    return math.sqrt((dLp/(wL*SL))**2+(dCp/(wC*SC))**2+(dHp/(wH*SH))**2
                     +RT*(dCp/(wC*SC))*(dHp/(wH*SH)))

class ColorClassifierV2:
    def __init__(self, device=None, use_l_bias=True):
        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device,str) else device
        self.cen_ready = True; self.lat_ready = True
        self.last_prediction = ("Unknown","0",0.0)
        self._use_l_bias = use_l_bias
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.dirname(_here)
        self.catalog = []
        pp = os.path.join(_root,"data","color_calibration_palette.json")
        if os.path.exists(pp):
            for item in json.load(open(pp,encoding="utf-8")):
                name = item.get("color_name","Unknown"); nl = name.strip().lower()
                rc = np.array(item.get("rgb_cenital",[128,128,128]),dtype=float)
                rl = np.array(item.get("rgb_lateral",[128,128,128]),dtype=float)
                self.catalog.append({
                    "color_code": str(item.get("color_code","0")),
                    "color_name": name, "name_lower": nl,
                    "rgb_cen": rc, "rgb_lat": rl,
                    "lab_cen": _rgb_to_lab(rc), "lab_lat": _rgb_to_lab(rl),
                    "material": _palette_material(nl),
                    "is_attractor": nl in FALSE_ATTRACTORS})
        self.classes = sorted({c["color_name"] for c in self.catalog})
        self._idx = {n:i for i,n in enumerate(self.classes)}
        self._hom = {}
        for c in self.catalog: self._hom.setdefault(c["name_lower"],[]).append(c)
        # ColorMLPV3 (M3)
        self._v3 = None; self._v3_mean = None; self._v3_std = None; self._v3_classes = None
        v3p = os.path.join(_root,"models","color_mlp_v3.pt")
        v3m = os.path.join(_root,"models","color_mlp_v3_metadata.json")
        if os.path.exists(v3p) and os.path.exists(v3m):
            try:
                meta = json.load(open(v3m,encoding="utf-8"))
                self._v3_mean = np.array(meta["mean"],dtype=np.float32)
                self._v3_std = np.array(meta["std"],dtype=np.float32)
                self._v3_classes = meta["classes"]; n_cls = len(self._v3_classes)
                m = ColorMLPV3(num_classes=n_cls)
                m.load_state_dict(torch.load(v3p,map_location=self.device))
                m.to(self.device).eval(); self._v3 = m
                print(f"[ColorClassifierV2] ColorMLPV3 cargado ({n_cls} clases).")
            except Exception as e: print(f"[ColorClassifierV2] V3 no disp: {e}")
        # Material MLP fallback
        self._mat = None; self._mat_mean = None; self._mat_std = None
        mp = os.path.join(_root,"models","material_type_classifier.pt")
        mm = os.path.join(_root,"models","material_type_classifier_metadata.json")
        if os.path.exists(mp) and os.path.exists(mm):
            try:
                meta2 = json.load(open(mm,encoding="utf-8"))
                self._mat_mean = np.array(meta2["mean"],dtype=np.float32)
                self._mat_std = np.array(meta2["std"],dtype=np.float32)
                m2 = _MatMLP(); m2.load_state_dict(torch.load(mp,map_location=self.device))
                m2.to(self.device).eval(); self._mat = m2
                print("[ColorClassifierV2] Material type MLP cargado.")
            except Exception as e: print(f"[ColorClassifierV2] mat no disp: {e}")

    def _prob_vec(self, pd):
        v = np.zeros(len(self.classes),dtype=float); tot = sum(pd.values())
        if tot <= 0: return v
        for n,s in pd.items():
            if n in self._idx: v[self._idx[n]] = s/tot
        return v
    def _make_single(self, name):
        v = np.zeros(len(self.classes),dtype=float)
        if name in self._idx: v[self._idx[name]] = 1.
        return v
    def _stage0(self, lab, hsv_est, hsv_std, lab_std):
        ch = math.sqrt(lab[1]**2+lab[2]**2)
        sV,sS,sH = float(hsv_std[2]),float(hsv_std[1]),float(hsv_std[0])
        H = float(hsv_est[0])
        if ch < 5.:
            if lab[0] > 90.: return MAT_WHITE
            if lab[0] < 20.:
                # Hue gating: if not extremely dark, check for Blue (~120) or Brown (~15)
                if lab[0] > 12.0:
                    if (10 <= H <= 30) or (110 <= H <= 130):
                        return MAT_SOLID
                return MAT_BLACK
        if sV > 35. and sS > 30. and lab[0] > 55.: return MAT_TRANS
        if sH < 8. and sS < 12. and ch < 25.: return MAT_METALLIC
        if ch < 20. and sH < 12. and lab[0] > 40.: return MAT_PEARL
        return MAT_SOLID
    def _stage1(self, lab_est, mat, allowed, cam, hsv_std):
        wL,wC,wH = ((0.5,1.0,1.2) if mat==MAT_METALLIC else
                    (1.5,0.8,2.0) if mat==MAT_TRANS else
                    (1.0,3.0,3.0) if mat in (MAT_WHITE,MAT_BLACK) else (1.0,1.0,1.2))
        lk = "lab_cen" if cam=="cenital" else "lab_lat"
        # M5: Adaptive L_bias — dark colors (L<40) have less background contamination
        # because the dark piece dominates the bbox regardless of blue belt pixels.
        # Light colors (L>70) have MORE contamination → use full L_BIAS.
        def _adaptive_lb(entry_lab_l):
            if not self._use_l_bias: return 0.
            if entry_lab_l < 30.: return 6.   # very dark: belt adds little
            if entry_lab_l < 50.: return 10.  # dark: moderate correction
            if entry_lab_l < 70.: return 14.  # medium: standard
            return L_BIAS                       # light/bright: full correction
        res = []
        for e in self.catalog:
            if e["is_attractor"] and not (allowed and e["name_lower"] in allowed): continue
            if allowed and e["name_lower"] not in allowed: continue
            ir = e["material"]==MAT_TRANS; ie = mat==MAT_TRANS
            tf = 5. if ie and not ir else 4. if not ie and ir else 1.
            ref = e[lk].copy(); ref[0] = max(0., ref[0]-_adaptive_lb(ref[0]))
            try: de = _ciede2000(lab_est, ref, wL, wC, wH)*tf
            except: de = 999.
            res.append((de,e))
        res.sort(key=lambda x:x[0]); return res
    def _stage2_v3(self, feat_arr):
        if self._v3 is None or self._v3_classes is None: return None, 0.0
        fn = (feat_arr - self._v3_mean)/(self._v3_std+1e-8)
        x = torch.tensor(fn,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._v3(x),dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return self._v3_classes[idx], float(probs[idx])
    def _stage2_mat(self, lab_std, hsv_std):
        if self._mat is None: return None
        feat = np.array([float(lab_std[0]),float(hsv_std[0]),float(hsv_std[2]),
                         float(hsv_std[1]),0.5,float(lab_std[0])],dtype=np.float32)
        fn = (feat-self._mat_mean)/(self._mat_std+1e-8)
        x = torch.tensor(fn,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._mat(x),dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return _MAT_CLASSES[idx] if probs[idx] >= 0.5 else None
    def _stage3(self, color_name, lab_est, cam):
        nl = color_name.strip().lower(); cands = self._hom.get(nl,[])
        if len(cands) <= 1: return cands[0]["color_code"] if cands else "0"
        if nl in CANONICAL_CODE:
            canon = str(CANONICAL_CODE[nl])
            for c in cands:
                if c["color_code"]==canon: return canon
            return canon
        lk = "lab_cen" if cam=="cenital" else "lab_lat"
        best_code = cands[0]["color_code"]; best_de = 999.
        for c in cands:
            try: de = _ciede2000(lab_est, c[lk])
            except: de = 999.
            if de < best_de: best_de=de; best_code=c["color_code"]
        return best_code

    def predict_cenital_probs(self, fv):
        return self.predict_gated_probs_cielab(fv, None, "cenital")
    def predict_lateral_probs(self, fv):
        return self.predict_gated_probs_cielab(fv, None, "lateral")
    def predict_gated_probs_cielab(self, feature_vector, allowed_color_names,
                                    camera_type="cenital", is_simulation=False):
        if feature_vector is None:
            return np.zeros(len(self.classes))
        fa = np.array(feature_vector, dtype=float)[:19] # Support up to 19D
        lab_est = np.array([fa[0],fa[2],fa[4]])
        lab_std = np.array([fa[1],fa[3],fa[5]])
        hsv_est = np.array([fa[6],fa[8],fa[10]])
        hsv_std = np.array([fa[7],fa[9],fa[11]])
        allowed = {n.strip().lower() for n in allowed_color_names} if allowed_color_names else None
        # Check if this is a Fast Path call (without SAM features, padded with zeros)
        is_fast_path = (fa[13] == 0.0 and fa[15] == 0.0)

        # 1. Determine material class physically first
        mat = self._stage0(lab_est, hsv_est, hsv_std, lab_std)
        
        # 2. MLP Physical Sanity Gate (ONLY for Deep Path calls with SAM)
        v3_pred, v3_prob = None, 0.0
        if not is_fast_path:
            v3_pred, v3_prob = self._stage2_v3(fa.astype(np.float32))
            if v3_pred is not None and v3_prob >= 0.75:
                bev3 = next((e for e in self.catalog if e["color_name"].lower() == v3_pred.lower()), None)
                if bev3 is not None:
                    pred_L = bev3["lab_cen"][0]
                    pred_a = bev3["lab_cen"][1]
                    pred_b = bev3["lab_cen"][2]
                    pred_ch = math.sqrt(pred_a**2 + pred_b**2)
                    pred_mat = bev3["material"]
                    
                    obs_L = lab_est[0]
                    obs_ch = math.sqrt(lab_est[1]**2 + lab_est[2]**2)
                    
                    # Sanity Rules:
                    # 1. Asymmetric Lightness: shadows can darken a piece, but can't make it much lighter
                    L_ok = (pred_L - obs_L < 35.0) and (obs_L - pred_L < 15.0)
                    
                    # 2. Chroma compatibility: prevents neutral-chromatic hallucinations
                    ch_ok = True
                    if obs_ch > 18.0 and pred_ch < 7.0:
                        ch_ok = False
                    elif obs_ch < 6.0 and pred_ch > 22.0:
                        ch_ok = False
                        
                    # 3. Transparent sanity: if predicting transparent, there must be some minimum variance
                    mat_ok = True
                    sV, sS = float(hsv_std[2]), float(hsv_std[1])
                    if pred_mat == MAT_TRANS and sV < 12.0 and sS < 12.0:
                        mat_ok = False
                        
                    if L_ok and ch_ok and mat_ok:
                        code = self._stage3(bev3["color_name"], lab_est, camera_type)
                        self.last_prediction = (bev3["color_name"], code, v3_prob)
                        return self._make_single(bev3["color_name"])
                        
        # 3. If MLP bypass was rejected or not triggered, proceed with standard fast paths
        lk = "lab_cen" if camera_type=="cenital" else "lab_lat"
        
        if mat == MAT_WHITE:
            cands = [c for c in self.catalog if c[lk][0]>85
                     and math.sqrt(c[lk][1]**2+c[lk][2]**2)<10
                     and c["material"] in (MAT_SOLID,MAT_TRANS,MAT_PEARL)
                     and not c["is_attractor"]]
            if cands:
                best = min(cands, key=lambda c:_ciede2000(lab_est,c[lk]))
                code = self._stage3(best["color_name"],lab_est,camera_type)
                
                # Fast Path: only trust 0.99 if strictly neutral white. Deep Path: trust MLP white
                if is_fast_path:
                    is_pure_white = (lab_est[0] > 80.0 and math.sqrt(lab_est[1]**2 + lab_est[2]**2) < 6.0)
                else:
                    is_pure_white = (v3_pred is not None and v3_pred.lower() == "white" and v3_prob >= 0.90)
                    
                conf = 0.99 if is_pure_white else 0.70
                self.last_prediction = (best["color_name"],code,conf)
                return self._make_single(best["color_name"])
            self.last_prediction = ("White","1",0.70)
            return self._make_single("White")
            
        if mat == MAT_BLACK:
            cands = [c for c in self.catalog if c[lk][0]<25
                     and c["material"] in (MAT_SOLID,MAT_SPECIAL)
                     and not c["is_attractor"]]
            if cands:
                best = min(cands, key=lambda c:_ciede2000(lab_est,c[lk]))
                code = self._stage3(best["color_name"],lab_est,camera_type)
                
                # Fast Path: only trust 0.99 if strictly neutral black. Deep Path: trust MLP black
                if is_fast_path:
                    is_pure_black = (lab_est[0] < 15.0 and math.sqrt(lab_est[1]**2 + lab_est[2]**2) < 6.0)
                else:
                    is_pure_black = (v3_pred is not None and v3_pred.lower() == "black" and v3_prob >= 0.90)
                    
                conf = 0.99 if is_pure_black else 0.70
                self.last_prediction = (best["color_name"],code,conf)
                return self._make_single(best["color_name"])
            self.last_prediction = ("Black","11",0.70)
            return self._make_single("Black")

        # 4. Stage 1: CIELAB match with M1+M2
        ranked = self._stage1(lab_est, mat, allowed, camera_type, hsv_std)
        if not ranked:
            return np.zeros(len(self.classes))
        top1_de, top1_entry = ranked[0]

        # 5. Stage 2a: ColorMLPV3 (M3) Gated — for medium confidence predictions
        if v3_pred is not None and v3_prob >= 0.65:
            gate_limit = len(ranked) if not self._use_l_bias else 5
            top_cand = {e["color_name"].lower() for _,e in ranked[:gate_limit]}
            if v3_pred.lower() in top_cand:
                bev3 = next((e for _,e in ranked if e["color_name"].lower()==v3_pred.lower()),None)
                if bev3 is not None:
                    code = self._stage3(bev3["color_name"],lab_est,camera_type)
                    self.last_prediction = (bev3["color_name"],code,v3_prob)
                    return self._make_single(bev3["color_name"])
        # Stage 2b: material MLP fallback if high ambiguity
        if top1_de > 8.0:
            refined_mat = self._stage2_mat(lab_std, hsv_std)
            if refined_mat is not None and refined_mat != mat:
                ranked2 = self._stage1(lab_est, refined_mat, allowed, camera_type, hsv_std)
                if ranked2 and ranked2[0][0] < top1_de:
                    ranked = ranked2; top1_de,top1_entry = ranked[0]
        # Stage 3: resolve homonyms
        best_name = top1_entry["color_name"]
        best_code = self._stage3(best_name, lab_est, camera_type)
        top5 = ranked[:min(5,len(ranked))]
        des = np.array([d for d,_ in top5], dtype=float)
        sims = np.exp(-des/8.)
        conf = float(sims[0]/(sims.sum()+1e-8))
        self.last_prediction = (best_name, best_code, conf)
        pd = {}
        for sim,(_,e) in zip(sims,top5):
            pd[e["color_name"]] = pd.get(e["color_name"],0.) + float(sim)
        return self._prob_vec(pd)
    def predict_fused_colors_flexible(self, feat_cen, feat_lat, threshold=0.25):
        p_cen = self.predict_cenital_probs(feat_cen)
        si = np.argsort(p_cen)[::-1]
        t1,t2 = si[0],si[1]
        tot = p_cen.sum()
        p1 = p_cen[t1]/(tot+1e-8); p2 = p_cen[t2]/(tot+1e-8)
        colors = [self.classes[t1]]
        if (p1-p2) < threshold: colors.append(self.classes[t2])
        return colors
    @property
    def all_classes(self): return self.classes
