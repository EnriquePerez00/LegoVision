import sys
import os

project_root = os.path.abspath("camara_domo")
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

from database.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT part_ref, pose_index, face_class, is_stable, zenith_observable_area, lateral_height
            FROM stable_poses
            WHERE part_ref = '32449'
            ORDER BY pose_index
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"Pose: {r['pose_index']}, Face: {r['face_class']}, is_stable: {r['is_stable']}, Area: {r['zenith_observable_area']}, Height: {r['lateral_height']}")
