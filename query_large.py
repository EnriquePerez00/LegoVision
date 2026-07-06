from core.db.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT part_ref, MAX(zenith_observable_area) as max_area,
                   MAX(lateral_height) as max_height
            FROM stable_poses
            GROUP BY part_ref
            ORDER BY max_area DESC NULLS LAST
            LIMIT 10;
        """)
        for r in cur.fetchall():
            print(f"Ref: {r['part_ref']}, Area: {r['max_area']}, Height: {r['max_height']}")
