import sys
import os

project_root = os.path.abspath("camara_domo")
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

from core.db.supabase_client import get_connection

refs = ["11211", "2412b", "3021", "32449", "3713", "4073", "43711", "4589b", "6536", "96874"]

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT part_ref, pose_index, face_class, zenith_observable_area, lateral_height
            FROM stable_poses
            WHERE is_stable = TRUE AND part_ref = ANY(%s)
            ORDER BY part_ref, pose_index
        """, (refs,))
        rows = cur.fetchall()
        for r in rows:
            area = f"{r['zenith_observable_area']:.2f}" if r['zenith_observable_area'] is not None else "N/A"
            height = f"{r['lateral_height']:.2f}" if r['lateral_height'] is not None else "N/A"
            print(f"Ref: {r['part_ref']}, Pose: {r['pose_index']}, Face: {r['face_class']}, Area: {area}, Height: {height}")
