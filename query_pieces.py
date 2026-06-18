import sys, os
from database.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT part_ref FROM lego_set_parts WHERE part_ref IN ('3713', '32449')")
        print(f"Found: {cur.fetchall()}")
