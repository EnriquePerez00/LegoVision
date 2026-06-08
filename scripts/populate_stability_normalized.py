# -*- coding: utf-8 -*-
"""
scripts/populate_stability_normalized.py
=========================================
Rellena el campo `stability_ratio_normalized` de la tabla `stable_poses`.

Para cada (part_ref, set_id) se calcula:
    stability_ratio_normalized = stability_ratio / max(stability_ratio)

Esto hace que el indicador sea independiente del numero de caras candidatas
y comparable entre piezas. Si todas las poses de la pieza tienen el mismo
sr, todas obtendran 1.0.

Uso:
    python3 scripts/populate_stability_normalized.py [--part_ref REF] [--set_id ID]

Sin argumentos, recalcula para TODAS las poses de la BD.
"""
import os
import sys
import argparse

# psycopg2 may not be installed in some venvs; usamos psql como fallback.
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSY = True
except ImportError:
    HAS_PSY = False

DB_CFG = {
    "host":     os.getenv("SUPABASE_DB_HOST", "localhost"),
    "port":     int(os.getenv("SUPABASE_DB_PORT", "5434")),
    "dbname":   os.getenv("SUPABASE_DB_NAME", "legvision"),
    "user":     os.getenv("SUPABASE_DB_USER", "postgres"),
    "password": os.getenv("SUPABASE_DB_PASSWORD", "legvision_pass_2024"),
}


def run_with_psycopg2(part_ref=None, set_id=None):
    where = []
    params = []
    if part_ref:
        where.append("sp.part_ref = %s")
        params.append(part_ref)
    if set_id:
        where.append("COALESCE(sp.set_id,'') = %s")
        params.append(set_id)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
        WITH max_sr AS (
            SELECT part_ref, COALESCE(set_id,'') AS set_id_norm,
                   GREATEST(MAX(stability_ratio), 1e-6) AS m
            FROM stable_poses
            GROUP BY part_ref, COALESCE(set_id,'')
        )
        UPDATE stable_poses sp
        SET stability_ratio_normalized =
            CASE WHEN sp.stability_ratio IS NULL THEN NULL
                 ELSE sp.stability_ratio / max_sr.m
            END
        FROM max_sr
        WHERE sp.part_ref = max_sr.part_ref
          AND COALESCE(sp.set_id,'') = max_sr.set_id_norm
          {('AND ' + ' AND '.join(where)) if where else ''}
    """
    with psycopg2.connect(**DB_CFG) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        affected = cur.rowcount
    return affected


def run_with_psql(part_ref=None, set_id=None):
    import subprocess
    where_extra = ""
    if part_ref:
        where_extra += f" AND sp.part_ref = '{part_ref}'"
    if set_id:
        where_extra += f" AND COALESCE(sp.set_id,'') = '{set_id}'"
    sql = f"""
        WITH max_sr AS (
            SELECT part_ref, COALESCE(set_id,'') AS set_id_norm,
                   GREATEST(MAX(stability_ratio), 1e-6) AS m
            FROM stable_poses
            GROUP BY part_ref, COALESCE(set_id,'')
        )
        UPDATE stable_poses sp
        SET stability_ratio_normalized =
            CASE WHEN sp.stability_ratio IS NULL THEN NULL
                 ELSE sp.stability_ratio / max_sr.m
            END
        FROM max_sr
        WHERE sp.part_ref = max_sr.part_ref
          AND COALESCE(sp.set_id,'') = max_sr.set_id_norm
          {where_extra};
    """
    env = dict(os.environ, PGPASSWORD=DB_CFG["password"])
    cmd = ["psql", "-h", DB_CFG["host"], "-p", str(DB_CFG["port"]),
           "-U", DB_CFG["user"], "-d", DB_CFG["dbname"], "-c", sql]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return res.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_ref", type=str, default=None)
    parser.add_argument("--set_id",   type=str, default=None)
    args = parser.parse_args()
    if HAS_PSY:
        n = run_with_psycopg2(args.part_ref, args.set_id)
        print(f"[populate_stability_normalized] {n} filas actualizadas")
    else:
        out = run_with_psql(args.part_ref, args.set_id)
        print(f"[populate_stability_normalized] {out}")


if __name__ == "__main__":
    main()