# -*- coding: utf-8 -*-
"""
2camaras_pieza_unica/scripts/sync_stable_poses_cache.py
========================================================
Reconstruye `2camaras_pieza_unica/data/stable_poses_cache.json` a partir
de la tabla `stable_poses` de Postgres.

A partir de la migración 009, el cache **NO** pre-filtra las poses (la
versión anterior aplicaba `stability_ratio>0.05 + contact_w>=4mm`). El
filtrado ahora lo hace el consumidor según la regla TARPS documentada en
`docs/stable_pose_selection_rule.md` (campo `tipping_energy_ratio`).

El cache expone para cada pose, además de los campos clásicos, los tres
campos nuevos generados por la migración 009:
    - stability_ratio_normalized
    - support_polygon_margin_mm
    - tipping_energy_ratio

Estructura JSON por part_ref:
    {
        "<ref>": [
            {
                "pose_index":             int (0..N-1, renumerado),
                "original_pose_index":    int (el original en BD),
                "contact_normal":         [3],
                "face_class":             "Top|Side|Bottom",
                "contact_area":           float (LDU²),
                "orientation_quat":       [4],
                "orientation_euler":      [3],
                "stability_ratio":        float,
                "stability_ratio_normalized": float,
                "tipping_energy_ratio":   float | null,
                "support_polygon_margin_mm": float | null,
                "zenith_observable_area": float (mm²),
                "zenith_bbox_area":       float (mm²) | null,
                "lateral_height":         float (mm),
                "contact_stable_length":  float (mm) | null,
                "contact_stable_width":   float (mm) | null,
                "is_stable":              bool,
                "set_id":                 str
            },
            ...
        ],
        ...
    }

Uso:
    .venv/bin/python 2camaras_pieza_unica/scripts/sync_stable_poses_cache.py
        [--set_id 75078-1]
        [--include_minifig]
        [--require_stable]      (default: TRUE; sólo guarda poses con is_stable=TRUE)
        [--output 2camaras_pieza_unica/data/stable_poses_cache.json]

Filtros legacy (sólo si se especifican explícitamente; por defecto NO filtran):
    [--legacy_min_stability 0.0]
    [--legacy_min_contact_dim_mm 0.0]
"""
import os
import sys
import json
import argparse

project_root_subproj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root_subproj)
sys.path.insert(0, legovic_root)
sys.path.insert(0, os.path.join(legovic_root, "database"))

from supabase_client import get_connection  # noqa: E402


def fetch_poses_from_db(require_stable: bool = True):
    where_stable = "WHERE is_stable = TRUE" if require_stable else ""
    sql = f"""
        SELECT part_ref, pose_index, contact_normal, face_class, contact_area,
               orientation_quat, orientation_euler,
               stability_ratio, stability_ratio_normalized,
               tipping_energy_ratio, support_polygon_margin_mm,
               zenith_observable_area, zenith_silhouette_area,
               zenith_bbox_area, lateral_height,
               effective_height, efective_height,
               contact_stable_length, contact_stable_width,
               is_stable
        FROM stable_poses
        {where_stable}
        ORDER BY part_ref, pose_index
    """
    out = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            ref = row["part_ref"]
            out.setdefault(ref, []).append(dict(row))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set_id", default="75078-1")
    parser.add_argument(
        "--output",
        default=os.path.join(project_root_subproj, "data", "stable_poses_cache.json"),
    )
    parser.add_argument("--include_minifig", action="store_true",
                        help="Include 'sw*' minifig refs (excluded by default).")
    parser.add_argument("--require_stable", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Only export poses with is_stable=TRUE (default: True).")
    parser.add_argument("--legacy_min_stability", type=float, default=0.0,
                        help="Legacy: descartar poses con stability_ratio <= este valor. "
                             "Por defecto 0.0 = no filtra. La nueva regla TARPS lo "
                             "hace en el consumidor (generate_yolo_training_dataset).")
    parser.add_argument("--legacy_min_contact_dim_mm", type=float, default=0.0,
                        help="Legacy: descartar poses con contact_stable_width < este valor "
                             "(NULL pasa). Por defecto 0.0 = no filtra.")
    args = parser.parse_args()

    print("=" * 70)
    print("SYNC stable_poses (BD) → stable_poses_cache.json")
    print(f"  set_id              = {args.set_id}")
    print(f"  require_stable      = {args.require_stable}")
    print(f"  include_minifig     = {args.include_minifig}")
    print(f"  legacy_min_stability     = {args.legacy_min_stability}  (0=off)")
    print(f"  legacy_min_contact_dim_mm= {args.legacy_min_contact_dim_mm}  (0=off)")
    print("=" * 70)

    poses_by_ref = fetch_poses_from_db(require_stable=args.require_stable)
    if not poses_by_ref:
        print("[ERROR] ninguna pose obtenida de la BD.")
        sys.exit(1)
    print(f"[OK] {len(poses_by_ref)} part_refs en BD.")

    output = {}
    stats = {
        "total_db": 0,
        "kept": 0,
        "drop_legacy_stability": 0,
        "drop_legacy_contact": 0,
        "drop_minifig": 0,
        "null_contact_kept": 0,
        "null_tipping_kept": 0,
    }
    drop_breakdown = {}

    for ref in sorted(poses_by_ref.keys()):
        if (not args.include_minifig) and ref.lower().startswith("sw"):
            stats["drop_minifig"] += len(poses_by_ref[ref])
            continue

        kept = []
        for p in poses_by_ref[ref]:
            stats["total_db"] += 1
            sr = p.get("stability_ratio") or 0.0
            cw = p.get("contact_stable_width")  # may be None
            tip = p.get("tipping_energy_ratio")  # may be None

            # Legacy filters (off por defecto)
            if args.legacy_min_stability > 0 and sr <= args.legacy_min_stability:
                stats["drop_legacy_stability"] += 1
                drop_breakdown.setdefault(ref, []).append(
                    f"pose{p['pose_index']} stability_ratio={sr:.3f}")
                continue
            if args.legacy_min_contact_dim_mm > 0 and cw is not None \
               and cw < args.legacy_min_contact_dim_mm:
                stats["drop_legacy_contact"] += 1
                drop_breakdown.setdefault(ref, []).append(
                    f"pose{p['pose_index']} contact_w={cw:.2f}mm")
                continue
            if cw is None:
                stats["null_contact_kept"] += 1
            if tip is None:
                stats["null_tipping_kept"] += 1

            kept.append({
                "original_pose_index": int(p["pose_index"]),
                "contact_normal":      list(p["contact_normal"]) if p.get("contact_normal") else None,
                "face_class":          p["face_class"],
                "contact_area":        float(p["contact_area"]) if p.get("contact_area") is not None else None,
                "orientation_quat":    list(p["orientation_quat"]) if p.get("orientation_quat") else None,
                "orientation_euler":   list(p["orientation_euler"]) if p.get("orientation_euler") else None,
                "stability_ratio":     float(sr),
                "stability_ratio_normalized": float(p["stability_ratio_normalized"])
                                              if p.get("stability_ratio_normalized") is not None else None,
                "tipping_energy_ratio": float(tip) if tip is not None else None,
                "support_polygon_margin_mm": float(p["support_polygon_margin_mm"])
                                              if p.get("support_polygon_margin_mm") is not None else None,
                "zenith_observable_area": float(p["zenith_observable_area"]) if p.get("zenith_observable_area") is not None else None,
                "zenith_silhouette_area": float(p["zenith_silhouette_area"]) if p.get("zenith_silhouette_area") is not None else None,
                "zenith_bbox_area":       float(p["zenith_bbox_area"])       if p.get("zenith_bbox_area")       is not None else None,
                "lateral_height":         float(p["lateral_height"])         if p.get("lateral_height")         is not None else None,
                "effective_height":       float(p["effective_height"])       if p.get("effective_height")       is not None else None,
                "efective_height":        float(p["efective_height"])        if p.get("efective_height")        is not None else None,
                "contact_stable_length":  float(p["contact_stable_length"])  if p.get("contact_stable_length")  is not None else None,
                "contact_stable_width":   float(cw)                          if cw is not None else None,
                "is_stable":              bool(p.get("is_stable", True)),
            })

        # Renumerar pose_index 0..N-1 tras filtrado (si lo hubo)
        for i, item in enumerate(kept):
            item["pose_index"] = i

        if kept:
            output[ref] = kept
            stats["kept"] += len(kept)

    # Guardar
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    # Resumen
    print()
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"  poses BD totales         : {stats['total_db']}")
    print(f"  poses guardadas en cache : {stats['kept']}")
    print(f"  ▸ descartadas legacy_stab: {stats['drop_legacy_stability']}")
    print(f"  ▸ descartadas legacy_cw  : {stats['drop_legacy_contact']}")
    print(f"  ▸ excluidas minifig (sw*): {stats['drop_minifig']}")
    print(f"  ▸ contact_width NULL kept: {stats['null_contact_kept']}")
    print(f"  ▸ tipping NULL kept      : {stats['null_tipping_kept']}")
    print(f"  refs en cache            : {len(output)}")
    print(f"  output                   : {args.output}")
    print("=" * 70)
    print("Recordatorio: el cache NO aplica TARPS. La regla")
    print("    tipping_energy_ratio >= 0.04   (fallback argmax)")
    print("se aplica en generate_yolo_training_dataset.py · select_pose_tarps()")
    print("Ver docs/stable_pose_selection_rule.md")
    print("=" * 70)

    refs_sin_pose = [r for r in poses_by_ref if r not in output and not r.lower().startswith("sw")]
    if refs_sin_pose:
        print("\n[WARN] refs sin ninguna pose superviviente:")
        for r in refs_sin_pose:
            reasons = drop_breakdown.get(r, [])
            print(f"   - {r}  ({len(reasons)} descartadas)")
            for r_str in reasons[:5]:
                print(f"        · {r_str}")


if __name__ == "__main__":
    main()
