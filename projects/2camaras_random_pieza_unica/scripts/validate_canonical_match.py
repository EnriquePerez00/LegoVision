# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/scripts/validate_canonical_match.py
====================================================================
Compara los embeddings DINOv2 de muestras de INFERENCIA (renders generados
con `generate_15_random_focus.py`) contra las REFS canonicas indexadas
en la BD (`piece_embeddings`). Asume que las refs ya estan indexadas con
escena canonica + pieza en (0,0,0).

Pipeline de la query (replica el del pipeline real EXCEPTO:
  - NO aplica mascara SAM (consigna: dejar fondo cinta natural).
  - El canvas 224 se rellena con color CINTA azul petroleo (37,65,84) en
    lugar de negro, para coincidir con el fondo de las refs canonicas.
  - YOLO se usa solo para detectar la bbox; si no hay deteccion se cae
    al `bbox_norm` que el generador escribio en el metadata).

Comparacion:
  - cenital -> filtra refs con `stable_face = 1`.
  - lateral -> filtra refs con `stable_face = 2`.
  - cosine similarity (todos los vectores estan L2-normalizados).
  - reporta top-3 + (mean, std, p10, max).

Output:
  data/random_focus_<ref>/canonical_match_report.json
  data/random_focus_<ref>/canonical_match_report.html

Uso:
  .venv/bin/python 2camaras_random_pieza_unica/scripts/validate_canonical_match.py \\
      --input_dir 2camaras_random_pieza_unica/data/random_focus_3023
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(os.path.dirname(project_root))
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)

from core.db import supabase_client


CINTA_RGB = (37, 65, 84)
CANVAS_SIZE = 224
SCALE_FACTOR = 208.0 / 640.0  # mismo que pipeline real
TOP_K = 3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device).eval()
    return model


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_query_canvas(crop_img: Image.Image, bg_color=CINTA_RGB) -> Image.Image:
    """Replica `_build_clean_canvas` pero con fondo en `bg_color` (cinta
    en lugar de negro). NO aplica mascara SAM."""
    w_p, h_p = crop_img.size
    if w_p <= 0 or h_p <= 0:
        return crop_img.convert("RGB")
    new_w = max(1, int(w_p * SCALE_FACTOR))
    new_h = max(1, int(h_p * SCALE_FACTOR))
    resized = crop_img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), bg_color)
    canvas.paste(resized, ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2))
    return canvas


def extract_embedding(img_canvas: Image.Image, model, transform, device) -> np.ndarray:
    tensor = transform(img_canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(tensor)
        if hasattr(feats, "last_hidden_state"):
            vec = feats.last_hidden_state[:, 0, :]
        else:
            vec = feats
        vec = vec[0]
        vec = vec / (vec.norm() + 1e-12)
    return vec.cpu().numpy().astype(np.float32)


def crop_by_bbox(img_full: Image.Image, bbox_norm) -> Image.Image:
    w, h = img_full.size
    x1 = max(0, int(bbox_norm[0] * w))
    y1 = max(0, int(bbox_norm[1] * h))
    x2 = min(w, int(bbox_norm[2] * w))
    y2 = min(h, int(bbox_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return img_full.convert("RGB")
    return img_full.crop((x1, y1, x2, y2)).convert("RGB")


def to_base64_thumb(pil_img: Image.Image, max_side=160) -> str:
    img = pil_img.copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def find_ref_image(ref_dir, cam_name, part_ref, color_hex, pose, angle):
    """Localiza el PNG de la ref para componer el HTML lado a lado."""
    fname = f"ref_{part_ref}_{color_hex.upper()}_pose{pose:02d}_rot{angle:03d}.png"
    p = os.path.join(ref_dir, cam_name, fname)
    if os.path.isfile(p):
        return p
    # fallback: busca el primer match con part_ref y angle
    cam_dir = os.path.join(ref_dir, cam_name)
    if os.path.isdir(cam_dir):
        for f in sorted(os.listdir(cam_dir)):
            if f.startswith(f"ref_{part_ref}_") and f"_rot{angle:03d}" in f:
                return os.path.join(cam_dir, f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directorio random_focus_<ref> con metadata.json y renders.")
    parser.add_argument("--ref_dir", type=str,
                        default=os.path.join(project_root, "data", "dinov2_refs_v3_canonical"),
                        help="Directorio con las refs canonicas para mostrar en el HTML.")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--report_name", type=str, default="canonical_match_report")
    pa = parser.parse_args()

    metadata_path = os.path.join(pa.input_dir, "random_focus_metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"[ERROR] No existe metadata: {metadata_path}")
        sys.exit(1)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    renders = meta.get("renders", [])
    ref_gt = meta.get("ref")
    color_hex_gt = (meta.get("color_hex") or "").lstrip("#").upper()
    print(f"[VALIDATION] {len(renders)} samples de {ref_gt} ({color_hex_gt})")

    # ── Cargar embeddings de la BD ──
    rows = supabase_client.get_all_embeddings()
    if not rows:
        print("[ERROR] BD vacia (no hay embeddings).")
        sys.exit(1)
    refs_by_cam = {1: [], 2: []}
    for r in rows:
        face = r["stable_face"]
        if face not in (1, 2):
            continue
        emb = np.asarray(r["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(emb))
        if n > 0:
            emb = emb / n
        refs_by_cam[face].append({
            "part_ref": r["part_ref"],
            "stable_face": face,
            "rotation_angle": r["rotation_angle"],
            "pose_index": r.get("pose_index"),
            "color_hex": r.get("color_hex"),
            "color_code": r.get("color_code"),
            "embedding": emb,
        })
    print(
        f"[VALIDATION] BD: {len(refs_by_cam[1])} refs cenital, "
        f"{len(refs_by_cam[2])} refs lateral"
    )

    # ── DINOv2 ──
    device = get_device()
    print(f"[VALIDATION] Cargando DINOv2 en {device}...")
    model = load_dinov2(device)
    transform = get_transform()

    # ── Inferencia + comparacion ──
    results = []
    for s in renders:
        sample_idx = s.get("index")
        cam_data = s.get("cameras", {})
        sample_out = {
            "index": sample_idx,
            "ref_gt": ref_gt,
            "pose_index_gt": s.get("pose_index"),
            "color_hex_gt": color_hex_gt,
            "position_bu": s.get("position_bu"),
            "cameras": {},
        }
        for cam_name, cam_id in (("cenital", 1), ("lateral", 2)):
            cm = cam_data.get(cam_name)
            if not cm:
                sample_out["cameras"][cam_name] = {"error": "missing"}
                continue
            img_path = os.path.join(pa.input_dir, cm["file_name"])
            if not os.path.isfile(img_path):
                sample_out["cameras"][cam_name] = {"error": f"no_image: {img_path}"}
                continue
            img_full = Image.open(img_path).convert("RGB")
            bbox_norm = cm.get("bbox_norm")
            if not bbox_norm:
                bbox_norm = [0.0, 0.0, 1.0, 1.0]
            crop_img = crop_by_bbox(img_full, bbox_norm)
            canvas = build_query_canvas(crop_img, bg_color=CINTA_RGB)
            q_emb = extract_embedding(canvas, model, transform, device)

            refs = refs_by_cam.get(cam_id, [])
            if not refs:
                sample_out["cameras"][cam_name] = {"error": "no_refs_for_cam"}
                continue
            ref_matrix = np.stack([r["embedding"] for r in refs])
            sims = (ref_matrix @ q_emb).astype(float)
            order = np.argsort(-sims)
            top_k = order[: pa.top_k]
            top_results = []
            for rank, idx in enumerate(top_k):
                r = refs[int(idx)]
                top_results.append({
                    "rank": rank + 1,
                    "part_ref": r["part_ref"],
                    "pose_index": r["pose_index"],
                    "rotation_angle": r["rotation_angle"],
                    "color_hex": r["color_hex"],
                    "similarity": round(float(sims[int(idx)]), 4),
                    "ref_image": find_ref_image(
                        pa.ref_dir, cam_name, r["part_ref"],
                        (r["color_hex"] or "").lstrip("#"),
                        int(r["pose_index"] or 0),
                        int(r["rotation_angle"] or 0),
                    ),
                })

            stats = {
                "max": round(float(np.max(sims)), 4),
                "mean": round(float(np.mean(sims)), 4),
                "std": round(float(np.std(sims)), 4),
                "p10": round(float(np.percentile(sims, 90)), 4),
                "min": round(float(np.min(sims)), 4),
                "n_refs": int(len(refs)),
            }
            sample_out["cameras"][cam_name] = {
                "image_path": img_path,
                "bbox_norm": bbox_norm,
                "top_k": top_results,
                "stats": stats,
            }
            print(
                f"  [{cam_name}] sample {sample_idx}: top-1 = "
                f"{top_results[0]['part_ref']} pose{top_results[0]['pose_index']} "
                f"rot{top_results[0]['rotation_angle']:03d} "
                f"({top_results[0]['similarity']:.4f}) | mean={stats['mean']:.3f} "
                f"max={stats['max']:.3f}"
            )
        results.append(sample_out)

    # ── Resumen agregado ──
    aggregate = {"cenital": {"top1": [], "max": []},
                 "lateral": {"top1": [], "max": []}}
    correct_top1 = {"cenital": 0, "lateral": 0}
    for s in results:
        for cam_name in ("cenital", "lateral"):
            cd = s["cameras"].get(cam_name) or {}
            top_k = cd.get("top_k") or []
            stats = cd.get("stats") or {}
            if top_k:
                aggregate[cam_name]["top1"].append(top_k[0]["similarity"])
                if top_k[0]["part_ref"] == ref_gt:
                    correct_top1[cam_name] += 1
            if "max" in stats:
                aggregate[cam_name]["max"].append(stats["max"])

    summary = {"ref_gt": ref_gt, "color_hex_gt": color_hex_gt,
               "n_samples": len(results), "per_cam": {}}
    for cam_name in ("cenital", "lateral"):
        t1 = aggregate[cam_name]["top1"]
        mx = aggregate[cam_name]["max"]
        if t1:
            summary["per_cam"][cam_name] = {
                "top1_similarity_mean": round(float(np.mean(t1)), 4),
                "top1_similarity_min": round(float(np.min(t1)), 4),
                "top1_similarity_max": round(float(np.max(t1)), 4),
                "top1_correct_count": correct_top1[cam_name],
                "top1_correct_pct": round(100.0 * correct_top1[cam_name] / len(t1), 2),
                "max_sim_mean": round(float(np.mean(mx)), 4) if mx else None,
            }
    summary["results"] = results

    out_json = os.path.join(pa.input_dir, f"{pa.report_name}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[VALIDATION] JSON => {out_json}")

    # ── HTML ──
    html_path = os.path.join(pa.input_dir, f"{pa.report_name}.html")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Canonical Match Report</title>",
        "<style>",
        "body{font-family:sans-serif;background:#f4f4f4;margin:20px}",
        "h1,h2{color:#222}",
        ".sample{background:#fff;border-radius:6px;padding:12px;margin-bottom:18px;",
        "  box-shadow:0 1px 3px rgba(0,0,0,.1)}",
        ".cam{display:flex;gap:18px;align-items:flex-start;margin-top:8px}",
        ".panel{border:1px solid #ddd;padding:8px;border-radius:4px;background:#fafafa}",
        ".query img,.refimg img{height:160px;border:1px solid #aaa}",
        ".topk{display:flex;gap:8px}",
        ".topk .item{text-align:center;font-size:12px}",
        ".topk .item .sim{font-weight:bold}",
        "table{border-collapse:collapse;margin-top:6px}",
        "td,th{border:1px solid #bbb;padding:3px 7px;font-size:12px}",
        "th{background:#eef}",
        ".ok{color:#0a7}.bad{color:#c33}",
        "</style></head><body>",
        f"<h1>Canonical Match Report — {ref_gt} ({color_hex_gt})</h1>",
        "<h2>Summary</h2>",
        "<table><thead><tr><th>cam</th><th>n</th><th>top1 mean</th>"
        "<th>top1 min</th><th>top1 max</th><th>top1 correct</th></tr></thead><tbody>",
    ]
    for cam_name in ("cenital", "lateral"):
        sc = summary["per_cam"].get(cam_name) or {}
        if not sc:
            continue
        cls = "ok" if sc.get("top1_similarity_mean", 0) >= 0.85 else "bad"
        parts.append(
            f"<tr><td>{cam_name}</td><td>{len(aggregate[cam_name]['top1'])}</td>"
            f"<td class='{cls}'>{sc.get('top1_similarity_mean')}</td>"
            f"<td>{sc.get('top1_similarity_min')}</td>"
            f"<td>{sc.get('top1_similarity_max')}</td>"
            f"<td>{sc.get('top1_correct_count')}/{len(aggregate[cam_name]['top1'])} "
            f"({sc.get('top1_correct_pct')}%)</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Per-sample details</h2>")
    for s in results:
        parts.append(f"<div class='sample'><h3>Sample {s['index']} — pose_gt="
                     f"{s.get('pose_index_gt')}, pos_bu={s.get('position_bu')}</h3>")
        for cam_name in ("cenital", "lateral"):
            cd = s["cameras"].get(cam_name) or {}
            if "error" in cd:
                parts.append(f"<p><b>{cam_name}</b>: {cd['error']}</p>")
                continue
            top_k = cd.get("top_k") or []
            stats = cd.get("stats") or {}

            try:
                q_img = Image.open(cd["image_path"]).convert("RGB")
                bbox_norm = cd.get("bbox_norm") or [0.0, 0.0, 1.0, 1.0]
                q_crop = crop_by_bbox(q_img, bbox_norm)
                q_b64 = to_base64_thumb(q_crop)
            except Exception:
                q_b64 = ""

            parts.append(f"<p><b>{cam_name.upper()}</b> — stats {stats}</p>")
            parts.append("<div class='cam'>")
            parts.append(f"<div class='panel query'><div>QUERY (crop)</div>"
                         f"<img src='{q_b64}'/></div>")
            parts.append("<div class='panel'><div>TOP-K</div><div class='topk'>")
            for tk in top_k:
                ref_path = tk.get("ref_image")
                ref_b64 = ""
                if ref_path and os.path.isfile(ref_path):
                    try:
                        ref_b64 = to_base64_thumb(Image.open(ref_path))
                    except Exception:
                        ref_b64 = ""
                ok_class = "ok" if tk["part_ref"] == ref_gt else "bad"
                parts.append(
                    f"<div class='item'>"
                    f"<img class='refimg' src='{ref_b64}'/><br>"
                    f"<span class='{ok_class}'>#{tk['rank']} {tk['part_ref']}</span><br>"
                    f"<span>pose{tk['pose_index']} rot{tk['rotation_angle']:03d}</span><br>"
                    f"<span>{tk.get('color_hex')}</span><br>"
                    f"<span class='sim'>{tk['similarity']:.4f}</span>"
                    f"</div>"
                )
            parts.append("</div></div></div>")
        parts.append("</div>")

    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[VALIDATION] HTML => {html_path}")

    # ── Conclusion ──
    print("\n" + "=" * 60)
    print("  CONCLUSION")
    print("=" * 60)
    for cam_name in ("cenital", "lateral"):
        sc = summary["per_cam"].get(cam_name) or {}
        if not sc:
            continue
        ok = "✓ HIPOTESIS OK" if sc.get("top1_similarity_mean", 0) >= 0.85 else "✗ HIPOTESIS DEBIL"
        print(f"  {cam_name}: top1_mean={sc.get('top1_similarity_mean')} "
              f"correct={sc.get('top1_correct_count')}/"
              f"{len(aggregate[cam_name]['top1'])} {ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
