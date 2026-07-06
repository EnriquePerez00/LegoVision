# -*- coding: utf-8 -*-
# scripts/generate_validation_html.py

import os
import sys
import json
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from core.db.supabase_client import get_connection
from core.db.set_catalog import REAL_SETS

PART_TYPE_MAP = {
    '3001': 'Brick 2x4',
    '3002': 'Brick 2x3',
    '3003': 'Brick 2x2',
    '3004': 'Brick 1x2',
    '3005': 'Brick 1x1',
    '3010': 'Brick 1x4',
    '3020': 'Plate 2x4',
    '3021': 'Plate 2x3',
    '3022': 'Plate 2x2',
    '3023': 'Plate 1x2',
    '3024': 'Plate 1x1',
    '3037': 'Slope 45 2x4',
    '3039': 'Slope 45 2x2',
    '3062': 'Brick Round 1x1',
    '3068': 'Tile 2x2',
    '3069': 'Tile 1x2',
    '3298': 'Slope 33 3x2',
    '3622': 'Brick 1x3',
    '3665': 'Slope Inverted 45 2x1',
    '3700': 'Technic Brick 1x2',
    '3701': 'Technic Brick 1x4',
    '3710': 'Plate 1x4',
    '2412': 'Tile Grille 1x2',
    '2420': 'Plate Corner 2x2',
    '2431': 'Tile 1x4',
    '2432': 'Tile 1x2 with Handle',
    '2877': 'Brick Profile 1x2',
    '4032': 'Plate Round 2x2',
    '4070': 'Brick 1x1 Headlight',
    '6141': 'Plate Round 1x1',
    '6636': 'Tile 1x6',
    '11477': 'Slope Curved 2x1',
    '15068': 'Slope Curved 2x2',
    '15573': 'Plate Modified 1x2',
    '32000': 'Technic Brick 1x2 Holes',
    '48336': 'Plate Modified 1x2 Handle',
    '54200': 'Slope 30 1x1',
    '59900': 'Cone 1x1',
    '60478': 'Plate Modified 1x2 Handle',
    '85984': 'Slope 31 1x2',
    '98138': 'Tile Round 1x1',
    '99206': 'Plate Modified 2x2',
}

def get_piece_type(ref, name_from_json=""):
    if ref in PART_TYPE_MAP:
        return PART_TYPE_MAP[ref]
    if name_from_json and name_from_json not in ("Pieza Lego", ""):
        return name_from_json
    return "Part " + ref

def get_db_poses_for_set_parts(parts):
    conn = get_connection()
    poses_by_part = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    part_ref, 
                    pose_index, 
                    face_class, 
                    contact_area, 
                    stability_ratio, 
                    is_stable, 
                    zenith_observable_area, 
                    lateral_height,
                    contact_normal
                FROM stable_poses
                WHERE part_ref = ANY(%s)
                ORDER BY part_ref, pose_index;
            """, (parts,))
            rows = cur.fetchall()
            for r in rows:
                poses_by_part.setdefault(r["part_ref"], []).append(dict(r))
    except Exception as e:
        print(f"[ERROR] Failed to query db poses: {e}")
    finally:
        conn.close()
    return poses_by_part

def generate_html_report(set_id, validation_json_path, output_path, renders_dir_rel):
    if not os.path.exists(validation_json_path):
        print(f"[ERROR] JSON not found: {validation_json_path}")
        return False

    with open(validation_json_path, "r", encoding="utf-8") as fh:
        val_data = json.load(fh)
    
    report_items = val_data.get("report", [])
    part_refs = [item["part_ref"] for item in report_items]
    
    # Get all poses from DB (both stable and unstable)
    poses_by_part = get_db_poses_for_set_parts(part_refs)
    
    # Get set colors catalog
    set_info = REAL_SETS.get(set_id, {})
    color_map = {p["ref"]: p.get("color_hex", "#A0A5A9") for p in set_info.get("parts", [])}
    color_code_map = {p["ref"]: str(p.get("color_code", "0")) for p in set_info.get("parts", [])}
    
    # Build HTML Content
    html = []
    html.append("""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Validación de Poses Estables - Set """ + set_id + """</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0f172a;
            --bg-surface: #1e293b;
            --bg-card: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --success: #10b981;
            --danger: #ef4444;
            --border: #475569;
            --glass: rgba(30, 41, 59, 0.7);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            line-height: 1.6;
            padding-bottom: 80px;
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
        }

        header {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            padding: 40px 24px;
            border-bottom: 1px solid var(--border);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
            pointer-events: none;
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }

        .header-tag {
            background-color: var(--primary);
            color: white;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 12px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .header-title {
            font-size: 2.5rem;
            margin-bottom: 8px;
            background: linear-gradient(to right, #a5b4fc, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-desc {
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .container {
            max-width: 1400px;
            margin: 40px auto 0 auto;
            padding: 0 24px;
        }

        .section-title {
            font-size: 1.75rem;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-left: 5px solid var(--primary);
            padding-left: 16px;
        }

        /* Grid de Indice */
        .index-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 48px;
        }

        .index-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            text-decoration: none;
            color: var(--text-main);
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .index-card:hover {
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.3);
        }

        .index-card img {
            width: 72px;
            height: 72px;
            object-fit: contain;
            margin-bottom: 12px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 4px;
        }

        .index-ref {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 4px;
        }

        .index-name {
            font-size: 0.75rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            width: 100%;
        }

        .index-badge {
            background-color: rgba(255, 255, 255, 0.1);
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 4px;
            margin-top: 8px;
        }

        /* Detalle de Piezas */
        .piece-section {
            background-color: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            margin-bottom: 40px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .piece-header {
            background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
            padding: 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }

        .piece-meta {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .piece-meta img {
            width: 60px;
            height: 60px;
            object-fit: contain;
            background-color: white;
            border-radius: 10px;
            padding: 2px;
        }

        .piece-title h2 {
            font-size: 1.5rem;
            margin-bottom: 4px;
        }

        .piece-title p {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .discrepancy-badge {
            padding: 6px 16px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
        }

        .discrepancy-badge.ok {
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }

        .discrepancy-badge.warn {
            background-color: rgba(239, 68, 68, 0.2);
            color: var(--danger);
            border: 1px solid var(--danger);
        }

        /* Grid de Poses */
        .poses-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            padding: 24px;
        }

        .pose-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: border-color 0.2s ease;
        }

        .pose-card.stable {
            border-color: rgba(16, 185, 129, 0.4);
        }

        .pose-card.unstable {
            border-color: rgba(239, 68, 68, 0.3);
        }

        .pose-img-container {
            width: 100%;
            height: 180px;
            background-color: #0f172a;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .pose-img-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .pose-status-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pose-status-badge.stable {
            background-color: var(--success);
            color: white;
        }

        .pose-status-badge.unstable {
            background-color: #64748b;
            color: #f1f5f9;
        }

        .pose-info {
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .pose-header-line {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .pose-number {
            font-weight: 700;
            font-size: 1.1rem;
        }

        .pose-class {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .pose-metrics {
            font-size: 0.8rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .metric-item {
            display: flex;
            flex-direction: column;
        }

        .metric-item-full {
            display: flex;
            flex-direction: column;
            grid-column: span 2;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .metric-val {
            font-weight: 600;
            color: #e2e8f0;
        }

        .stability-progress {
            margin-top: 12px;
            width: 100%;
        }

        .progress-bar-container {
            width: 100%;
            height: 6px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 4px;
        }

        .progress-bar {
            height: 100%;
            border-radius: 3px;
        }

        .progress-bar.stable {
            background-color: var(--success);
        }

        .progress-bar.unstable {
            background-color: #f59e0b;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <span class="header-tag">Set """ + set_id + """</span>
            <h1 class="header-title">Reporte de Posiciones Físicas Estables</h1>
            <p class="header-desc">Análisis comparativo de estabilidad física simulada y calibración de gating para clasificación.</p>
        </div>
    </header>

    <div class="container">
        <h2 class="section-title">Índice de Piezas</h2>
        <div class="index-grid">
""")
    
    # Render index cards
    for item in report_items:
        ref = item["part_ref"]
        ptype = get_piece_type(ref, item.get("name", ""))
        cc = color_code_map.get(ref, "0")
        poses = poses_by_part.get(ref, [])
        
        # Link to BrickLink image
        bl_img_url = f"https://img.bricklink.com/ItemImage/PN/{cc}/{ref}.png"
        
        html.append(f"""
            <a href="#piece-{ref}" class="index-card">
                <img src="{bl_img_url}" onerror="this.src='https://img.bricklink.com/ItemImage/PL/{ref}.png'; this.onerror=null;" alt="{ptype}">
                <div class="index-ref">{ref}</div>
                <div class="index-name" title="{ptype}">{ptype}</div>
                <div class="index-badge">{len(poses)} poses</div>
            </a>
        """)
        
    html.append("""
        </div>

        <h2 class="section-title">Detalle de Poses por Pieza</h2>
    """)

    # Render piece sections
    for item in report_items:
        ref = item["part_ref"]
        ptype = get_piece_type(ref, item.get("name", ""))
        poses = poses_by_part.get(ref, [])
        disc = item.get("discrepancy", False)
        disc_text = "DISCREPANCIA" if disc else "OK"
        disc_class = "warn" if disc else "ok"
        cc = color_code_map.get(ref, "0")
        bl_img_url = f"https://img.bricklink.com/ItemImage/PN/{cc}/{ref}.png"
        
        html.append(f"""
        <div class="piece-section" id="piece-{ref}">
            <div class="piece-header">
                <div class="piece-meta">
                    <img src="{bl_img_url}" onerror="this.src='https://img.bricklink.com/ItemImage/PL/{ref}.png'; this.onerror=null;" alt="{ptype}">
                    <div class="piece-title">
                        <h2>{ref} - {ptype}</h2>
                        <p>{len(poses)} poses simuladas en total</p>
                    </div>
                </div>
                <span class="discrepancy-badge {disc_class}">{disc_text}</span>
            </div>
            <div class="poses-grid">
        """)

        # Get the validation_renders dir path to locate files
        renders_abs_dir = os.path.join(project_root, "data", "validation_renders")

        for p in poses:
            pidx = p["pose_index"]
            fc = p["face_class"]
            area = p["contact_area"]
            height = p["lateral_height"]
            zenith_area = p["zenith_observable_area"] or 0.0
            sr = p["stability_ratio"]
            is_stable = p["is_stable"]
            
            status_text = "Estable" if is_stable else "Inestable"
            status_class = "stable" if is_stable else "unstable"
            
            # Formatted normal
            normal_str = ", ".join(f"{v:.2f}" for v in p["contact_normal"])
            
            # Find the actual render file dynamically inside validation_renders
            render_filepath_rel = ""
            if os.path.exists(renders_abs_dir):
                for f in os.listdir(renders_abs_dir):
                    if f.startswith(f"validation_{ref}_pose{pidx}_face"):
                        render_filepath_rel = os.path.join(renders_dir_rel, f)
                        break
            
            if not render_filepath_rel:
                render_filepath_rel = "https://placehold.co/300x180/1e293b/94a3b8?text=Sin+Render"

            html.append(f"""
                <div class="pose-card {status_class}">
                    <div class="pose-img-container">
                        <img src="{render_filepath_rel}" alt="Pose {pidx}" onerror="this.src='https://placehold.co/300x180/1e293b/94a3b8?text=Sin+Render'">
                        <span class="pose-status-badge {status_class}">{status_text}</span>
                    </div>
                    <div class="pose-info">
                        <div>
                            <div class="pose-header-line">
                                <span class="pose-number">Pose #{pidx}</span>
                                <span class="pose-class">{fc}</span>
                            </div>
                            <div class="pose-metrics">
                                <div class="metric-item">
                                    <span class="metric-label">Altura Lateral</span>
                                    <span class="metric-val">{height:.2f} mm</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">Área Cenital</span>
                                    <span class="metric-val">{zenith_area:.1f} mm²</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">Área de Contacto</span>
                                    <span class="metric-val">{area:.1f} mm²</span>
                                </div>
                                <div class="metric-item-full">
                                    <span class="metric-label">Normal</span>
                                    <span class="metric-val" title="[{normal_str}]">[{normal_str}]</span>
                                </div>
                            </div>
                        </div>
                        <div class="stability-progress">
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                                <span style="color: var(--text-muted);">Ratio Estabilidad</span>
                                <span style="font-weight: 700; color: { 'var(--success)' if is_stable else '#f59e0b' };">{sr * 100:.1f}%</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar {status_class}" style="width: {sr * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """)

        html.append("""
            </div>
        </div>
        """)

    html.append("""
    </div>
</body>
</html>
""")

    # Write HTML output
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))
    print(f"[HTML] Report generated successfully at: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Genera reporte HTML de posiciones estables")
    parser.add_argument("--set_id", type=str, default="75078-1")
    parser.add_argument("--validation_json", type=str,
                        default=os.path.join(project_root, "data", "tmp", "stability_validation_results.json"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(project_root, "data", "validation_report_75078.html"))
    parser.add_argument("--renders_dir_rel", type=str, default="validation_renders")
    args = parser.parse_args()
    
    ok = generate_html_report(args.set_id, args.validation_json, args.output, args.renders_dir_rel)
    if ok:
        print("[HTML] Completado.")
    else:
        print("[HTML] Fallo al generar el reporte.")

if __name__ == "__main__":
    main()
