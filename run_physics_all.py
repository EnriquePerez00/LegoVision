#!/usr/bin/env python3
"""
run_physics_all.py — Orquestador de simulación de poses estables (dedupe-aware)

Recorre piezas LDraw y calcula sus poses estables, EVITANDO recomputar
geometrías idénticas gracias a la firma geométrica (part_geometry_signature):

  1. Si la pieza es un ALIAS puro de otra -> resuelve al destino.
  2. Si su geometry_hash ya tiene poses en stable_poses (en otra pieza) ->
     COPIA esas poses cambiando solo part_ref (coste ~ms, sin Blender).
  3. Si es geometría nueva -> SIMULA una sola vez (representante del grupo).

REANUDABLE ante cortes (luz/batería/Ctrl+C):
  - El progreso vive en la BD: una pieza "hecha" es la que ya tiene filas en
    stable_poses. Al rearrancar, esas piezas se SALTAN automáticamente.
  - La escritura de poses de cada pieza es transaccional (DELETE+INSERT+commit):
    o quedan todas las poses de la pieza, o ninguna. Nunca a medias.
  - Si Blender muere a mitad, esa pieza simplemente no tendrá poses y se
    reintentará en el próximo arranque.
  - Ctrl+C / SIGTERM hacen shutdown ordenado: no se lanzan nuevas piezas y se
    espera a que terminen las en curso (que commitean o se descartan enteras).

Optimizado para Apple Silicon M4 (12 CPU · 48 GB):
  - Auto-workers = nº núcleos - 2 (via core.utils.hw).
  - Cada Blender se fuerza a 1 hilo interno para evitar oversubscription
    (N workers = N núcleos limpios).

Modos (--mode):
  - solo        : usa todos los núcleos disponibles (workers auto, nice 0). DEFAULT.
  - con-render  : convive con un render pesado (workers 2, nice 10).

Requisitos previos:
  - Migración 013 aplicada (tabla part_geometry_signature).
  - Firmas calculadas: python scripts/compute_geometry_hash.py

Uso:
    python run_physics_all.py [--mode solo|con-render] [--refs-file F]
        [--set SET_ID] [--workers N] [--nice N] [--limit N] [--force]
        [--n-dirs 8]
"""

import os
import sys
import signal
import argparse
import subprocess
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from core.utils.logger import get_logger
from core.db.supabase_client import get_connection

logger = get_logger("run_physics_all")

BLENDER_EXEC = os.getenv("BLENDER_EXEC", shutil.which("blender") or "/opt/homebrew/bin/blender")
SIM_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "simulate_stable_poses.py")

# Evento global de parada para shutdown ordenado ante Ctrl+C / SIGTERM
_STOP = threading.Event()


def _auto_workers() -> int:
    """Nº de procesos Blender recomendado para esta máquina (núcleos - 2)."""
    try:
        from core.utils.hw import suggested_blender_workers
        return suggested_blender_workers()
    except Exception:
        return max(1, (os.cpu_count() or 4) - 2)


def _single_thread_env() -> dict:
    """Entorno que fuerza a Blender/Bullet a 1 hilo interno, para que N workers
    ocupen N núcleos limpios sin oversubscription (varios procesos peleando
    por los mismos hilos matemáticos)."""
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = "1"
    return env


# ----------------------------------------------------------------------------
# Consultas a BD
# ----------------------------------------------------------------------------

def _fetch_all(query, params=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur.fetchall()


def part_has_poses(part_ref: str) -> bool:
    rows = _fetch_all(
        "SELECT 1 FROM stable_poses WHERE part_ref = %s LIMIT 1", (part_ref,)
    )
    return len(rows) > 0


def get_signature(part_ref: str):
    rows = _fetch_all(
        "SELECT part_ref, geometry_hash, is_alias_of, status "
        "FROM part_geometry_signature WHERE part_ref = %s", (part_ref,)
    )
    if not rows:
        return None
    r = rows[0]
    return r if isinstance(r, dict) else {
        "part_ref": r[0], "geometry_hash": r[1], "is_alias_of": r[2], "status": r[3]
    }


def resolve_alias(part_ref: str, _depth=0) -> str:
    """Sigue la cadena de alias hasta la pieza con geometría real."""
    if _depth > 10:
        return part_ref
    sig = get_signature(part_ref)
    if sig and sig.get("is_alias_of"):
        return resolve_alias(sig["is_alias_of"], _depth + 1)
    return part_ref


def find_twin_with_poses(geometry_hash: str, exclude_ref: str):
    """Busca otra pieza con el mismo geometry_hash que YA tenga poses."""
    if not geometry_hash:
        return None
    rows = _fetch_all(
        """
        SELECT s.part_ref
        FROM part_geometry_signature s
        WHERE s.geometry_hash = %s
          AND s.part_ref <> %s
          AND EXISTS (SELECT 1 FROM stable_poses sp WHERE sp.part_ref = s.part_ref)
        LIMIT 1
        """,
        (geometry_hash, exclude_ref),
    )
    if not rows:
        return None
    r = rows[0]
    return r["part_ref"] if isinstance(r, dict) else r[0]


def copy_poses(src_ref: str, dst_ref: str) -> int:
    """Copia las poses estables de src_ref a dst_ref (mismo geometry_hash).
    Transaccional: o se copian todas o ninguna. Devuelve el nº de poses copiadas."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM stable_poses WHERE part_ref = %s", (dst_ref,))
            cur.execute(
                """
                INSERT INTO stable_poses (
                    part_ref, pose_index, contact_normal, face_class, contact_area,
                    orientation_quat, orientation_euler, simulation_passes,
                    simulation_total, stability_ratio, is_stable, set_id,
                    zenith_observable_area
                )
                SELECT %s, pose_index, contact_normal, face_class, contact_area,
                    orientation_quat, orientation_euler, simulation_passes,
                    simulation_total, stability_ratio, is_stable, set_id,
                    zenith_observable_area
                FROM stable_poses WHERE part_ref = %s
                """,
                (dst_ref, src_ref),
            )
            n = cur.rowcount
        conn.commit()
    return n


# ----------------------------------------------------------------------------
# Simulación (Blender headless, 1 hilo interno, prioridad configurable)
# ----------------------------------------------------------------------------

def simulate_part(part_ref: str, n_dirs: int, nice: int, timeout: int = 600) -> bool:
    cmd = []
    if nice and nice > 0 and shutil.which("nice"):
        cmd += ["nice", "-n", str(nice)]
    cmd += [
        BLENDER_EXEC, "-b", "-P", SIM_SCRIPT, "--",
        "--part", part_ref, "--n_dirs", str(n_dirs), "--save_db",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_ROOT, env=_single_thread_env(),
        )
        if result.returncode != 0:
            logger.error(f"{part_ref}: Blender rc={result.returncode}\n{result.stderr[-800:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{part_ref}: timeout ({timeout}s) en simulación")
        return False
    except Exception as e:
        logger.error(f"{part_ref}: excepción lanzando Blender: {e}")
        return False


# ----------------------------------------------------------------------------
# Resolución de una pieza (dedupe en cascada)
# ----------------------------------------------------------------------------

def resolve_part(part_ref: str, n_dirs: int, nice: int, force: bool) -> str:
    """Devuelve un string de estado: 'skip' | 'copied' | 'simulated' | 'failed' | 'stopped'."""
    if _STOP.is_set():
        return "stopped"

    # Ya tiene poses y no forzamos -> nada que hacer (reanudabilidad)
    if not force and part_has_poses(part_ref):
        return "skip"

    # Nivel 1: alias -> resolver a la pieza con geometría real
    real_ref = resolve_alias(part_ref)

    sig = get_signature(real_ref)
    ghash = sig.get("geometry_hash") if sig else None

    # Nivel 3: ¿hay un gemelo (mismo hash) que ya tenga poses? -> copiar
    if ghash:
        twin = find_twin_with_poses(ghash, part_ref)
        if twin:
            n = copy_poses(twin, part_ref)
            logger.info(f"{part_ref}: copiadas {n} poses del gemelo {twin} (hash={ghash[:10]})")
            return "copied"

    if _STOP.is_set():
        return "stopped"

    # Geometría nueva -> simular la pieza real
    ok = simulate_part(real_ref, n_dirs, nice)
    if not ok:
        return "failed"

    # Si part_ref != real_ref (era alias), copiar el resultado al ref original
    if real_ref != part_ref and part_has_poses(real_ref):
        copy_poses(real_ref, part_ref)
    return "simulated"


# ----------------------------------------------------------------------------
# Enumeración de piezas objetivo
# ----------------------------------------------------------------------------

def get_refs_from_set(set_id: str) -> list:
    clean = set_id if "-" in set_id else f"{set_id}-1"
    rows = _fetch_all(
        "SELECT DISTINCT part_ref FROM lego_set_parts WHERE set_code = %s", (clean,)
    )
    out = []
    for r in rows:
        out.append(r["part_ref"] if isinstance(r, dict) else r[0])
    return sorted(out)


def get_all_signed_refs() -> list:
    """Todas las piezas con firma calculada (excluye alias puros: se resuelven solos)."""
    rows = _fetch_all(
        "SELECT part_ref FROM part_geometry_signature WHERE status = 'ok'"
    )
    out = []
    for r in rows:
        out.append(r["part_ref"] if isinstance(r, dict) else r[0])
    return sorted(out)


def get_pending_refs(refs: list, force: bool) -> list:
    """Filtra en BLOQUE las piezas que YA tienen poses (reanudabilidad rápida).
    Una sola consulta en vez de N: al reanudar tras un corte, saltamos de golpe
    todo lo ya hecho sin lanzar Blender ni consultar pieza a pieza."""
    if force or not refs:
        return refs
    done = set()
    # Trocear para no pasar listas gigantes en un solo IN (...)
    CHUNK = 5000
    for i in range(0, len(refs), CHUNK):
        chunk = refs[i:i + CHUNK]
        rows = _fetch_all(
            "SELECT DISTINCT part_ref FROM stable_poses WHERE part_ref = ANY(%s)",
            (chunk,),
        )
        for r in rows:
            done.add(r["part_ref"] if isinstance(r, dict) else r[0])
    return [r for r in refs if r not in done]


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _install_signal_handlers():
    """Ctrl+C / SIGTERM -> parada ordenada (no se lanzan nuevas piezas)."""
    def _handler(signum, frame):
        if not _STOP.is_set():
            logger.warning(
                "Señal recibida: parada ordenada. No se lanzan nuevas piezas; "
                "esperando a las en curso. (El progreso está en la BD, puedes "
                "reanudar relanzando el mismo comando.)"
            )
            _STOP.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Orquesta simulación de poses estables (dedupe-aware, reanudable)")
    parser.add_argument("--mode", choices=["solo", "con-render"], default="solo",
                        help="'solo' = usa toda la CPU (default). 'con-render' = convive con render (2 workers, nice 10).")
    parser.add_argument("--refs-file", default=None, help="Fichero con un part_ref por línea.")
    parser.add_argument("--set", dest="set_id", default=None, help="Procesar solo las piezas de este set.")
    parser.add_argument("--workers", type=int, default=None, help="Override manual de workers (Blender en paralelo).")
    parser.add_argument("--nice", type=int, default=None, help="Override manual de prioridad (nice).")
    parser.add_argument("--n-dirs", type=int, default=8, help="Direcciones de impulso por cara.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recalcula aunque existan poses.")
    args = parser.parse_args()

    # Resolver workers/nice según modo (con override manual opcional)
    if args.mode == "con-render":
        workers = args.workers if args.workers is not None else 2
        nice = args.nice if args.nice is not None else 10
    else:  # solo
        workers = args.workers if args.workers is not None else _auto_workers()
        nice = args.nice if args.nice is not None else 0
    workers = max(1, workers)

    _install_signal_handlers()

    # 1. Determinar refs
    if args.refs_file:
        with open(args.refs_file) as fh:
            refs = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    elif args.set_id:
        refs = get_refs_from_set(args.set_id)
    else:
        refs = get_all_signed_refs()

    if args.limit:
        refs = refs[: args.limit]

    total_all = len(refs)

    # 2. Reanudabilidad: saltar en bloque lo ya hecho (poses en BD)
    refs = get_pending_refs(refs, args.force)
    skipped_resume = total_all - len(refs)
    logger.info(
        f"[HW] cpu={os.cpu_count()} | modo={args.mode} workers={workers} nice={nice} | "
        f"1 hilo interno por Blender"
    )
    logger.info(
        f"Piezas objetivo: {total_all} | ya hechas (saltadas al reanudar): "
        f"{skipped_resume} | pendientes: {len(refs)}"
    )

    if not refs:
        logger.info("Nada pendiente. Todo resuelto.")
        return

    counts = {"skip": 0, "copied": 0, "simulated": 0, "failed": 0, "stopped": 0}
    total = len(refs)

    # 3. Procesar en paralelo. Cada pieza es transaccional en BD (reanudable).
    def _task(ref):
        if _STOP.is_set():
            return ref, "stopped"
        try:
            return ref, resolve_part(ref, args.n_dirs, nice, args.force)
        except Exception as e:
            logger.error(f"{ref}: excepción {e}")
            return ref, "failed"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_task, r): r for r in refs}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                ref, status = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if status != "stopped":
                    logger.info(
                        f"[{i}/{total}] {ref}: {status}  "
                        f"(sim={counts['simulated']} copy={counts['copied']} "
                        f"skip={counts['skip']} fail={counts['failed']})"
                    )
        except KeyboardInterrupt:
            _STOP.set()

    if _STOP.is_set():
        logger.warning(
            "PARADA. Progreso guardado en BD. Para reanudar donde lo dejaste, "
            "relanza EXACTAMENTE el mismo comando: saltará lo ya hecho."
        )
    logger.info(
        "Resumen: "
        f"{counts['simulated']} simuladas, {counts['copied']} copiadas, "
        f"{counts['skip']} omitidas, {counts['failed']} fallidas, "
        f"{counts['stopped']} sin empezar (por parada)"
    )


if __name__ == "__main__":
    main()
