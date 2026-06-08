# -*- coding: utf-8 -*-
"""2camaras_pieza_unica/scripts/reindex_dinov2_eevee.py
=======================================================
Indexa embeddings DINOv2 para las referencias renderizadas (cenital + lateral).
"""
import os, sys, re, glob, argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
legovic_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, legovic_root)

from config_loader import cfg
from logger import get_logger, log_execution_header, log_execution_footer
log = get_logger("dinov2")


def main():
    import time as _time
    _t_start = _time.perf_counter()

    parser = argparse.ArgumentParser(description="Reindexar embeddings DINOv2.")
    parser.add_argument("--ref_dir", type=str, default=None)
    parser.add_argument("--clear", action="store_true", default=True)
    args = parser.parse_args()

    ref_dir = args.ref_dir or os.path.join(project_root, "data", "dinov2_refs")

    log_execution_header(log, "reindex_dinov2_eevee.py", ref_dir=ref_dir, clear=args.clear)

    # Import from LegoVision root
    from training.index_synthetic_renders import (
        get_device, load_dinov2, get_transform, index_directory
    )
    from database import supabase_client

    device = get_device()
    model = load_dinov2(device)
    transform = get_transform()

    if args.clear:
        log.info("Limpiando embeddings existentes...")
        supabase_client.clear_embeddings()

    total_indexed = 0
    total_failed = 0

    # Captura: 1=part_ref, 2=color_hex, 3=pose_index (opcional), 4=rotation_angle
    regex = re.compile(r"ref_([a-zA-Z0-9_]+)_([A-F0-9]{6})(?:_pose(\d+))?_rot(\d+)\.png", re.IGNORECASE)

    # Index cenital (cam_id=1)
    cenital_dir = os.path.join(ref_dir, "cenital")
    if os.path.isdir(cenital_dir):
        paths = sorted(glob.glob(os.path.join(cenital_dir, "ref_*.png")))
        log.info(f"Indexando {len(paths)} renders cenital (cam_id=1)...")
        n, f = index_directory(paths, regex, model, transform, device, face_id=1)
        total_indexed += n
        total_failed += f
    else:
        log.warning(f"Directorio cenital no encontrado: {cenital_dir}")

    # Index lateral (cam_id=2)
    lateral_dir = os.path.join(ref_dir, "lateral")
    if os.path.isdir(lateral_dir):
        paths = sorted(glob.glob(os.path.join(lateral_dir, "ref_*.png")))
        log.info(f"Indexando {len(paths)} renders lateral (cam_id=2)...")
        n, f = index_directory(paths, regex, model, transform, device, face_id=2)
        total_indexed += n
        total_failed += f
    else:
        log.warning(f"Directorio lateral no encontrado: {lateral_dir}")

    _duration = _time.perf_counter() - _t_start
    log_execution_footer(log, "reindex_dinov2_eevee.py",
                         duration_s=_duration,
                         total_indexed=total_indexed,
                         total_failed=total_failed)
    log.info(f"✅ {total_indexed} embeddings guardados | ❌ {total_failed} errores")


if __name__ == "__main__":
    main()