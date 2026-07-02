import sys
import os

project_root = os.path.abspath("camara_domo")
legovic_root = os.path.dirname(project_root)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scripts"))

from core.db.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT part_ref, pose_index, stability_ratio, contact_stable_width
            FROM stable_poses
            WHERE part_ref = '32449'
            ORDER BY pose_index
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"Pose: {r['pose_index']}, SR: {r['stability_ratio']}, CW: {r['contact_stable_width']}")
