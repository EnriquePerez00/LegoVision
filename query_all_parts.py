import sys, os
from core.db.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT part_ref, color_hex FROM lego_set_parts")
        rows = cur.fetchall()
        print(f"Total parts in lego_set_parts: {len(rows)}")
        for r in rows[:10]:
            print(f" - {r['part_ref']}")
