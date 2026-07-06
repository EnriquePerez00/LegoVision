# -*- coding: utf-8 -*-
"""2camaras_random_pieza_unica/test/run_sam_pipeline_e2e.py
=============================================================
Test E2E que aplica el MISMO pipeline (bbox + SAM crop + fondo cinta
+ canvas 224 fit-to-canvas + DINOv2) a:
  (a) las refs canonicas en `data/dinov2_refs_v3_canonical/`
  (b) las queries de inferencia en `data/random_focus_3023/`

Y compara los embeddings resultantes via cosine similarity por camara.
Output 100% en `2camaras_random_pieza_unica/test/sam_pipeline_e2e/`.

NO escribe nada en la BD ni en `data/`.

Uso (usa venv del repo para tener torch, ultralytics, psycopg2 no se usa):
  .venv/bin/python 2camaras_random_pieza_unica/test/run_sam_pipeline_e2e.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # 2camaras_random_pieza_unica
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, legovic_root)
sys.path.insert(0, project_root)


# ─────────────────────────────────────────────────────────────────
# Constantes (alineadas con el pipeline real)
# ─────────────────────────────────────────────────────────────────
CINTA_RGB = (37, 65, 84)      # azul petroleo (linear color cinta)
CANVAS_SIZE = 224
CANVAS_MARGIN_PX = 8
DINOV2_NAME = "dinov2_vits14"

REFS_DIR = os.path.join(project_root, "data", "dinov2_refs_v3_canonical")
QUERIES_DIR = os.path.join(project_root, "data", "random_focus_3023")
OUTPUT_DIR = os.path.join(project_root, "test", "sam_pipeline_e2e")


# ─────────────────────────────────────────────────────────────────
# DINOv2 + SAM
# ─────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_NAME)
    model.to(device).eval()
    return model


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


_sam_model = None
def get_sam():
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM("mobile_sam.pt")
    return _sam_model


# ─────────────────────────────────────────────────────────────────
# Pipeline imagen -> embedding
# ─────────────────────────────────────────────────────────────────
def expand_bbox_norm(bbox_norm, pad_px, img_w, img_h):
    """Expande bbox normalizado en pad_px en cada lado (clamp 0..size)."""
    x1 = max(0, int(bbox_norm[0] * img_w) - pad_px)
    y1 = max(0, int(bbox_norm[1] * img_h) - pad_px)
    x2 = min(img_w, int(bbox_norm[2] * img_w) + pad_px)
    y2 = min(img_h, int(bbox_norm[3] * img_h) + pad_px)
    return [x1, y1, x2, y2]


def sam_segment(img_full_pil: Image.Image, bbox_px) -> np.ndarray:
    """Segmenta con MobileSAM dado bbox en pixeles. Devuelve mask uint8
    HxW (mismo tamaño que img_full)."""
    x1, y1, x2, y2 = bbox_px
    img_np = np.array(img_full_pil.convert("RGB"))
    sam = get_sam()
    try:
        results = sam(img_np, bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None:
            full_mask = (
                results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
            )
            return full_mask
    except Exception as e:
        print(f"  [WARN] SAM fallo: {e}")
    # Fallback: bbox = mascara llena
    fallback = np.zeros(img_np.shape[:2], dtype=np.uint8)
    fallback[y1:y2, x1:x2] = 255
    return fallback


def build_sam_canvas(
    img_full_pil: Image.Image, mask: np.ndarray, bbox_px,
    canvas_size=CANVAS_SIZE, margin_px=CANVAS_MARGIN_PX,
    bg_color=CINTA_RGB,
) -> Image.Image:
    """Construye un canvas canvas_size x canvas_size con fondo `bg_color`
    en el que se pega la pieza recortada por bbox y enmascarada por SAM:
      1. Recorta img_full por bbox_px -> crop RGB.
      2. Recorta mask por bbox_px -> crop_mask (uint8).
      3. En el crop, los pixeles con crop_mask == 0 se reemplazan por
         bg_color (CINTA).
      4. Hace fit-to-canvas con margen 8px (preserva aspect ratio,
         maximiza tamaño en el canvas).
      5. Pega el crop escalado en canvas con bg_color.
    """
    x1, y1, x2, y2 = bbox_px
    img_np = np.array(img_full_pil.convert("RGB"))
    crop_rgb = img_np[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return Image.new("RGB", (canvas_size, canvas_size), bg_color)
    if crop_mask.shape != crop_rgb.shape[:2]:
        # safety resize
        import cv2
        crop_mask = cv2.resize(
            crop_mask,
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    fg = crop_mask > 0
    if fg.any():
        crop_rgb[~fg] = bg_color
    else:
        # SAM produjo mask vacia: dejamos el crop tal cual
        pass

    crop_pil = Image.fromarray(crop_rgb)
    cw, ch = crop_pil.size
    max_dim = canvas_size - 2 * margin_px
    if cw <= 0 or ch <= 0:
        return Image.new("RGB", (canvas_size, canvas_size), bg_color)
    scale = min(max_dim / cw, max_dim / ch)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    crop_resized = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    canvas.paste(crop_resized, (paste_x, paste_y))
    return canvas


def extract_dinov2(canvas_pil: Image.Image, model, transform, device) -> np.ndarray:
    tensor = transform(canvas_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(tensor)
        if hasattr(feats, "last_hidden_state"):
            vec = feats.last_hidden_state[:, 0, :]
        else:
            vec = feats
        vec = vec[0]
        vec = vec / (vec.norm() + 1e-12)
    return vec.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Helpers de IO + reporte
# ─────────────────────────────────────────────────────────────────
def to_base64_thumb(pil_img: Image.Image, max_side=180) -> str:
    img = pil_img.copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_ref_meta(meta_entry):
    """Extrae info compacta de un entry de metadata.json de refs."""
    return {
        "file_name": meta_entry["file_name"],
        "ref": meta_entry["ref"],
        "color_hex": meta_entry.get("color_hex"),
        "pose_index": meta_entry.get("pose_index"),
        "rotation_deg": meta_entry.get("rotation_deg"),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs_dir", default=REFS_DIR)
    parser.add_argument("--queries_dir", default=QUERIES_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--bbox_pad_px", type=int, default=4,
                        help="Padding del bbox antes de SAM (px).")
    parser.add_argument("--save_canvases", action="store_true", default=True,
                        help="Guarda los canvases procesados a disco.")
    pa = parser.parse_args()

    out_dir = pa.output_dir
    os.makedirs(out_dir, exist_ok=True)
    canvas_refs_dir = os.path.join(out_dir, "refs_sam")
    canvas_q_dir = os.path.join(out_dir, "queries_sam")
    os.makedirs(os.path.join(canvas_refs_dir, "cenital"), exist_ok=True)
    os.makedirs(os.path.join(canvas_refs_dir, "lateral"), exist_ok=True)
    os.makedirs(canvas_q_dir, exist_ok=True)

    # ── Cargar metadatas ──
    refs_meta_path = os.path.join(pa.refs_dir, "metadata.json")
    if not os.path.isfile(refs_meta_path):
        print(f"[ERROR] No existe {refs_meta_path}. Regenera refs con la version actualizada.")
        sys.exit(1)
    with open(refs_meta_path) as f:
        refs_meta = json.load(f)

    q_meta_path = os.path.join(pa.queries_dir, "random_focus_metadata.json")
    if not os.path.isfile(q_meta_path):
        print(f"[ERROR] No existe {q_meta_path}.")
        sys.exit(1)
    with open(q_meta_path) as f:
        q_meta = json.load(f)

    ref_gt = q_meta.get("ref")
    print(f"[E2E] ref_gt={ref_gt} | "
          f"refs={len(refs_meta['renders'])} | queries={len(q_meta['renders'])}")

    # ── Modelos ──
    device = get_device()
    print(f"[E2E] Cargando DINOv2 en {device}...")
    model = load_dinov2(device)
    transform = get_transform()
    # SAM se carga lazy en la primera llamada

    # ─────────────────────────────────────────────────────────────
    # FASE 1: procesar REFS (96 imagenes = 48 cenital + 48 lateral)
    # ─────────────────────────────────────────────────────────────
    print(f"\n[E2E] Procesando refs ({len(refs_meta['renders'])} entries x 2 cams)...")
    t0 = time.perf_counter()
    refs_data = {"cenital": [], "lateral": []}
    for entry in refs_meta["renders"]:
        fname = entry["file_name"]
        for cam_name in ("cenital", "lateral"):
            cm = entry["cameras"][cam_name]
            img_path = os.path.join(pa.refs_dir, cam_name, fname)
            if not os.path.isfile(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            bbox_px = expand_bbox_norm(cm["bbox_norm"], pa.bbox_pad_px, iw, ih)
            mask = sam_segment(img, bbox_px)
            canvas = build_sam_canvas(img, mask, bbox_px,
                                      bg_color=CINTA_RGB)
            emb = extract_dinov2(canvas, model, transform, device)

            ref_info = parse_ref_meta(entry)
            ref_info["bbox_norm"] = cm["bbox_norm"]
            ref_info["bbox_px"] = bbox_px
            ref_info["mask_pixels"] = int((mask > 0).sum())
            ref_info["embedding"] = emb
            refs_data[cam_name].append(ref_info)

            if pa.save_canvases:
                canvas.save(
                    os.path.join(canvas_refs_dir, cam_name, fname),
                    optimize=True,
                )
    t_refs = time.perf_counter() - t0
    print(f"[E2E] Refs OK: cen={len(refs_data['cenital'])} | "
          f"lat={len(refs_data['lateral'])} | {t_refs:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # FASE 2: procesar QUERIES + comparacion top-K
    # ─────────────────────────────────────────────────────────────
    print(f"\n[E2E] Procesando {len(q_meta['renders'])} queries...")
    results = []
    for s in q_meta["renders"]:
        sample_idx = s.get("index")
        sample_out = {
            "index": sample_idx,
            "ref_gt": ref_gt,
            "pose_index_gt": s.get("pose_index"),
            "color_hex_gt": q_meta.get("color_hex"),
            "position_bu": s.get("position_bu"),
            "cameras": {},
        }
        for cam_name, _cam_id in (("cenital", 1), ("lateral", 2)):
            cm = s["cameras"].get(cam_name)
            if not cm:
                sample_out["cameras"][cam_name] = {"error": "missing"}
                continue
            img_path = os.path.join(pa.queries_dir, cm["file_name"])
            if not os.path.isfile(img_path):
                sample_out["cameras"][cam_name] = {"error": f"no image: {img_path}"}
                continue
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            bbox_px = expand_bbox_norm(cm["bbox_norm"], pa.bbox_pad_px, iw, ih)
            mask = sam_segment(img, bbox_px)
            canvas = build_sam_canvas(img, mask, bbox_px, bg_color=CINTA_RGB)
            q_emb = extract_dinov2(canvas, model, transform, device)

            if pa.save_canvases:
                canvas_fname = f"sample_{sample_idx:02d}_{cam_name}.png"
                canvas.save(os.path.join(canvas_q_dir, canvas_fname),
                            optimize=True)

            refs = refs_data[cam_name]
            if not refs:
                sample_out["cameras"][cam_name] = {"error": "no_refs"}
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
                    "part_ref": r["ref"],
                    "pose_index": r["pose_index"],
                    "rotation_deg": r["rotation_deg"],
                    "color_hex": r["color_hex"],
                    "ref_file": r["file_name"],
                    "similarity": round(float(sims[int(idx)]), 4),
                    "ref_canvas_path": os.path.join(canvas_refs_dir, cam_name,
                                                     r["file_name"]),
                    "ref_image_path": os.path.join(pa.refs_dir, cam_name,
                                                    r["file_name"]),
                })

            stats = {
                "max": round(float(np.max(sims)), 4),
                "mean": round(float(np.mean(sims)), 4),
                "std": round(float(np.std(sims)), 4),
                "p90": round(float(np.percentile(sims, 90)), 4),
                "min": round(float(np.min(sims)), 4),
                "n_refs": int(len(refs)),
            }
            sample_out["cameras"][cam_name] = {
                "image_path": img_path,
                "bbox_norm": cm["bbox_norm"],
                "bbox_px": bbox_px,
                "mask_pixels": int((mask > 0).sum()),
                "canvas_path": os.path.join(
                    canvas_q_dir, f"sample_{sample_idx:02d}_{cam_name}.png"
                ),
                "top_k": top_results,
                "stats": stats,
            }
            print(
                f"  [{cam_name}] sample {sample_idx}: top-1 = "
                f"{top_results[0]['part_ref']} pose{top_results[0]['pose_index']} "
                f"rot{top_results[0]['rotation_deg']:03d} "
                f"({top_results[0]['similarity']:.4f}) | "
                f"mean={stats['mean']:.3f} max={stats['max']:.3f}"
            )
        results.append(sample_out)

    # ─────────────────────────────────────────────────────────────
    # Resumen + report
    # ─────────────────────────────────────────────────────────────
    aggregate = {"cenital": {"top1": [], "max": []},
                 "lateral": {"top1": [], "max": []}}
    correct_top1 = {"cenital": 0, "lateral": 0}
    for s in results:
        for cam_name in ("cenital", "lateral"):
            cd = s["cameras"].get(cam_name) or {}
            tk = cd.get("top_k") or []
            stats = cd.get("stats") or {}
            if tk:
                aggregate[cam_name]["top1"].append(tk[0]["similarity"])
                if tk[0]["part_ref"] == ref_gt:
                    correct_top1[cam_name] += 1
            if "max" in stats:
                aggregate[cam_name]["max"].append(stats["max"])

    summary = {
        "ref_gt": ref_gt,
        "color_hex_gt": q_meta.get("color_hex"),
        "n_samples": len(results),
        "pipeline": "bbox(meta) + SAM crop + cinta_bg + canvas224 fit-to-canvas + DINOv2",
        "bbox_pad_px": pa.bbox_pad_px,
        "canvas_size": CANVAS_SIZE,
        "canvas_margin_px": CANVAS_MARGIN_PX,
        "bg_rgb": list(CINTA_RGB),
        "per_cam": {},
        "results": results,
    }
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

    out_json = os.path.join(out_dir, "sam_pipeline_report.json")
    # Eliminamos los embeddings antes de serializar el JSON principal
    # (los guardamos aparte en un .npz si interesa).
    json_safe = json.loads(json.dumps(summary, default=str))
    with open(out_json, "w") as f:
        json.dump(json_safe, f, indent=2, ensure_ascii=False)
    print(f"\n[E2E] JSON => {out_json}")

    # Guardar embeddings sueltos (ndarray) por si se quieren reanalizar
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        ref_cen=np.stack([r["embedding"] for r in refs_data["cenital"]]) if refs_data["cenital"] else np.zeros((0, 384)),
        ref_lat=np.stack([r["embedding"] for r in refs_data["lateral"]]) if refs_data["lateral"] else np.zeros((0, 384)),
        ref_cen_files=np.array([r["file_name"] for r in refs_data["cenital"]]),
        ref_lat_files=np.array([r["file_name"] for r in refs_data["lateral"]]),
    )

    # ── HTML ──
    html_path = os.path.join(out_dir, "sam_pipeline_report.html")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>SAM Pipeline E2E</title>",
        "<style>",
        "body{font-family:sans-serif;background:#f4f4f4;margin:20px}",
        ".sample{background:#fff;border-radius:6px;padding:12px;margin-bottom:18px;",
        "box-shadow:0 1px 3px rgba(0,0,0,.1)}",
        ".cam{display:flex;gap:18px;align-items:flex-start;margin-top:8px;flex-wrap:wrap}",
        ".panel{border:1px solid #ddd;padding:8px;border-radius:4px;background:#fafafa}",
        ".panel img{display:block;border:1px solid #aaa}",
        ".topk{display:flex;gap:8px}",
        ".topk .item{text-align:center;font-size:12px}",
        ".topk .item .sim{font-weight:bold}",
        "table{border-collapse:collapse;margin-top:6px}",
        "td,th{border:1px solid #bbb;padding:3px 7px;font-size:12px}",
        "th{background:#eef}",
        ".ok{color:#0a7}.bad{color:#c33}",
        "</style></head><body>",
        f"<h1>SAM Pipeline E2E Report — {ref_gt} ({q_meta.get('color_hex')})</h1>",
        f"<p>Pipeline: {summary['pipeline']}</p>",
        "<h2>Summary</h2>",
        "<table><thead><tr><th>cam</th><th>n</th><th>top1 mean</th>"
        "<th>top1 min</th><th>top1 max</th><th>top1 correct</th>"
        "<th>max_sim_mean</th></tr></thead><tbody>",
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
            f"({sc.get('top1_correct_pct')}%)</td>"
            f"<td>{sc.get('max_sim_mean')}</td></tr>"
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

            # Composite del query: imagen original + canvas SAM
            try:
                q_orig = Image.open(cd["image_path"]).convert("RGB")
                q_orig_b64 = to_base64_thumb(q_orig)
            except Exception:
                q_orig_b64 = ""
            try:
                q_canvas = Image.open(cd["canvas_path"]).convert("RGB")
                q_canvas_b64 = to_base64_thumb(q_canvas)
            except Exception:
                q_canvas_b64 = ""

            parts.append(
                f"<p><b>{cam_name.upper()}</b> — bbox_px={cd.get('bbox_px')} "
                f"| mask_pixels={cd.get('mask_pixels')} | stats {stats}</p>"
            )
            parts.append("<div class='cam'>")
            parts.append(
                f"<div class='panel'><div>QUERY full</div>"
                f"<img src='{q_orig_b64}'/></div>"
            )
            parts.append(
                f"<div class='panel'><div>QUERY canvas SAM</div>"
                f"<img src='{q_canvas_b64}'/></div>"
            )
            parts.append("<div class='panel'><div>TOP-K refs</div><div class='topk'>")
            for tk in top_k:
                ref_canvas_b64 = ""
                ref_orig_b64 = ""
                cp = tk.get("ref_canvas_path")
                ip = tk.get("ref_image_path")
                if cp and os.path.isfile(cp):
                    try:
                        ref_canvas_b64 = to_base64_thumb(Image.open(cp))
                    except Exception:
                        ref_canvas_b64 = ""
                if ip and os.path.isfile(ip):
                    try:
                        ref_orig_b64 = to_base64_thumb(Image.open(ip))
                    except Exception:
                        ref_orig_b64 = ""
                ok_cls = "ok" if tk["part_ref"] == ref_gt else "bad"
                parts.append(
                    f"<div class='item'>"
                    f"<img src='{ref_canvas_b64}'/><br>"
                    f"<small><img src='{ref_orig_b64}' height='80'/></small><br>"
                    f"<span class='{ok_cls}'>#{tk['rank']} {tk['part_ref']}</span><br>"
                    f"<span>pose{tk['pose_index']} rot{tk['rotation_deg']:03d}</span><br>"
                    f"<span>{tk.get('color_hex')}</span><br>"
                    f"<span class='sim'>{tk['similarity']:.4f}</span>"
                    f"</div>"
                )
            parts.append("</div></div></div>")
        parts.append("</div>")

    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"[E2E] HTML => {html_path}")

    # ── Conclusion ──
    print("\n" + "=" * 60)
    print("  CONCLUSION SAM PIPELINE E2E")
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
