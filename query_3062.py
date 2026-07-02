import sys, os
from core.db.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT pose_index, face_class, is_stable, stability_ratio FROM stable_poses WHERE part_ref = '3062' ORDER BY pose_index")
        for r in cur.fetchall():
            print(f"Pose: {r['pose_index']}, Face: {r['face_class']}, is_stable: {r['is_stable']}, SR: {r['stability_ratio']}")
