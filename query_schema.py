import sys, os
from database.supabase_client import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'stable_poses';
        """)
        for row in cur.fetchall():
            print(f"{row['column_name']} : {row['data_type']}")
