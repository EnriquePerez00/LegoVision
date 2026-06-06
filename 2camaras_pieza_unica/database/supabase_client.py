"""
LegoVision — Cliente Supabase/PostgreSQL
Gestiona conexión a la BD local (Docker puerto 5434)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configuración de conexión ---
DB_CONFIG = {
    "host":     os.getenv("SUPABASE_DB_HOST", "localhost"),
    "port":     int(os.getenv("SUPABASE_DB_PORT", "5434")),
    "dbname":   os.getenv("SUPABASE_DB_NAME", "legvision"),
    "user":     os.getenv("SUPABASE_DB_USER", "postgres"),
    "password": os.getenv("SUPABASE_DB_PASSWORD", "legvision_pass_2024"),
    "connect_timeout": 3,
}


def get_connection():
    """Obtiene una conexión a la base de datos."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


# ----------------------------------------------------------------
# Sesiones de inferencia
# ----------------------------------------------------------------

def create_session(model_version: str, belt_speed_mm_s: float = 83.3) -> str:
    """Crea una nueva sesión de inferencia y devuelve su UUID."""
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO inference_sessions (id, model_version, belt_speed_mm_s)
                VALUES (%s, %s, %s)
            """, (session_id, model_version, belt_speed_mm_s))
    return session_id


def close_session(session_id: str, avg_fps: float = None, avg_confidence: float = None):
    """Cierra una sesión de inferencia."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE inference_sessions
                SET ended_at = NOW(),
                    avg_fps = %s,
                    avg_confidence = %s,
                    total_detections = (
                        SELECT COUNT(*) FROM detections WHERE session_id = %s
                    )
                WHERE id = %s
            """, (avg_fps, avg_confidence, session_id, session_id))


# ----------------------------------------------------------------
# Detecciones
# ----------------------------------------------------------------

def save_detection(
    session_id: str,
    piece_class: str,
    confidence: float,
    bbox: tuple,          # (x_center, y_center, width, height) normalizado
    piece_name: str = None,
    image_path: str = None,
    inference_ms: float = None,
) -> str:
    """Guarda una detección individual en la BD."""
    detection_id = str(uuid.uuid4())
    x, y, w, h = bbox
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO detections
                    (id, session_id, piece_class, piece_name, confidence,
                     bbox_x_center, bbox_y_center, bbox_width, bbox_height,
                     image_path, inference_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (detection_id, session_id, piece_class, piece_name, confidence,
                  x, y, w, h, image_path, inference_ms))
    return detection_id


def save_detections_batch(session_id: str, detections: list[dict]):
    """Guarda múltiples detecciones de un frame en un solo INSERT."""
    if not detections:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (
                    str(uuid.uuid4()), session_id,
                    d["class"], d.get("name"), d["confidence"],
                    d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                    d.get("image_path"), d.get("inference_ms")
                )
                for d in detections
            ]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO detections
                    (id, session_id, piece_class, piece_name, confidence,
                     bbox_x_center, bbox_y_center, bbox_width, bbox_height,
                     image_path, inference_ms)
                VALUES %s
            """, values)


# ----------------------------------------------------------------
# Consultas para la GUI
# ----------------------------------------------------------------

def get_session_stats(session_id: str) -> dict:
    """Estadísticas de la sesión actual."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    MAX(detected_at) as last_detection,
                    COUNT(DISTINCT piece_class) as unique_classes
                FROM detections WHERE session_id = %s
            """, (session_id,))
            return dict(cur.fetchone() or {})


def get_top_classes(limit: int = 10) -> list[dict]:
    """Top clases más detectadas (histórico)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT piece_class, piece_name, COUNT(*) as count,
                       AVG(confidence) as avg_confidence
                FROM detections
                GROUP BY piece_class, piece_name
                ORDER BY count DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]


def get_recent_detections(limit: int = 50) -> list[dict]:
    """Últimas N detecciones."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, detected_at, piece_class, piece_name,
                       confidence, inference_ms
                FROM detections
                ORDER BY detected_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]



def test_connection() -> bool:
    """Verifica que la conexión a la BD funciona."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()
                print(f"[LegoVision DB] Conectado: {version['version']}")
        return True
    except Exception as e:
        print(f"[LegoVision DB] Error de conexión: {e}")
        return False

# ----------------------------------------------------------------
# Métodos para Entrenamiento y Embeddings
# ----------------------------------------------------------------

def create_training_run(epochs: int, config_used: dict = None) -> str:
    """Crea una nueva corrida de entrenamiento YOLO en la BD y devuelve su ID."""
    run_id = str(uuid.uuid4())
    config_json = json.dumps(config_used or {})
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_runs (id, status, epochs, config_used)
                VALUES (%s, 'running', %s, %s)
            """, (run_id, epochs, config_json))
    return run_id

def update_training_progress(run_id: str, current_epoch: int, loss: float, val_loss: float, map50: float, log_text: str):
    """Actualiza las métricas y los logs del entrenamiento por época."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE training_runs
                SET current_epoch = %s,
                    loss = %s,
                    val_loss = %s,
                    map50 = %s,
                    logs = COALESCE(logs, '') || %s
                WHERE id = %s
            """, (current_epoch, loss, val_loss, map50, log_text, run_id))

def append_training_log(run_id: str, log_text: str):
    """Añade texto a los logs de una corrida de entrenamiento sin alterar las métricas."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE training_runs
                SET logs = COALESCE(logs, '') || %s
                WHERE id = %s
            """, (log_text, run_id))


def complete_training_run(run_id: str, status: str, log_text: str = None):
    """Marca la corrida de entrenamiento como completada o fallida."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE training_runs
                SET status = %s,
                    ended_at = NOW(),
                    logs = COALESCE(logs, '') || %s
                WHERE id = %s
            """, (status, log_text or "", run_id))

def get_active_training_run() -> Optional[dict]:
    """Obtiene la corrida de entrenamiento activa si existe."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, status, epochs, current_epoch, loss, val_loss, map50, logs, config_used
                    FROM training_runs
                    WHERE status = 'running'
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
                res = cur.fetchone()
                return dict(res) if res else None
    except Exception as e:
        print(f"[LegoVision DB Error] Error leyendo entrenamiento activo: {e}")
        return None

def save_piece_embedding(
    part_ref: str,
    stable_face: int,
    rotation_angle: int,
    embedding: list[float],
    color_code: str = None,
    color_hex: str = None,
    pose_index: int = None,
    embedding_projected: list[float] = None,
):
    """Guarda o actualiza el embedding de características de una pieza de Lego."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            p_idx = pose_index if pose_index is not None else stable_face
            cur.execute("""
                INSERT INTO piece_embeddings (part_ref, stable_face, rotation_angle, embedding, color_code, color_hex, pose_index, embedding_projected)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (part_ref, stable_face, rotation_angle, color_hex)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    color_code = EXCLUDED.color_code,
                    color_hex = EXCLUDED.color_hex,
                    pose_index = EXCLUDED.pose_index,
                    embedding_projected = EXCLUDED.embedding_projected,
                    created_at = NOW()
            """, (part_ref, stable_face, rotation_angle, embedding, color_code, color_hex, p_idx, embedding_projected))

def save_piece_embeddings_batch(embeddings: list[dict]):
    """Guarda múltiples embeddings de piezas en un solo INSERT con ON CONFLICT para mayor velocidad."""
    if not embeddings:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            values = []
            for emb in embeddings:
                part_ref = emb["part_ref"]
                stable_face = emb["stable_face"]
                rotation_angle = emb["rotation_angle"]
                embedding_data = emb["embedding"]
                color_code = emb.get("color_code")
                color_hex = emb.get("color_hex")
                pose_index = emb.get("pose_index")
                if pose_index is None:
                    pose_index = stable_face
                embedding_projected = emb.get("embedding_projected")
                values.append((
                    part_ref, stable_face, rotation_angle, embedding_data,
                    color_code, color_hex, pose_index, embedding_projected
                ))
            
            psycopg2.extras.execute_values(cur, """
                INSERT INTO piece_embeddings (part_ref, stable_face, rotation_angle, embedding, color_code, color_hex, pose_index, embedding_projected)
                VALUES %s
                ON CONFLICT (part_ref, stable_face, rotation_angle, color_hex)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    color_code = EXCLUDED.color_code,
                    color_hex = EXCLUDED.color_hex,
                    pose_index = EXCLUDED.pose_index,
                    embedding_projected = EXCLUDED.embedding_projected,
                    created_at = NOW()
            """, values)

def clear_embeddings():
    """Borra TODOS los embeddings de la tabla (usar con cuidado)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM piece_embeddings")
            print("[LegoVision DB] Todos los embeddings borrados.")

def clear_piece_embeddings(part_ref: str):
    """Borra todos los embeddings de una pieza específica."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM piece_embeddings WHERE part_ref = %s", (part_ref,))
            print(f"[LegoVision DB] Embeddings borrados para la pieza: {part_ref}")

def get_stable_poses(part_ref: str) -> list[dict]:
    """Obtiene las posiciones estables de una pieza desde la BD."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pose_index, contact_normal, face_class, contact_area, 
                           orientation_quat, orientation_euler, stability_ratio
                    FROM stable_poses
                    WHERE part_ref = %s AND is_stable = TRUE
                    ORDER BY pose_index
                """, (part_ref,))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[LegoVision DB Error] get_stable_poses: {e}")
        return []

def get_all_embeddings() -> list[dict]:
    """Obtiene todos los embeddings almacenados para comparación por similaridad."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT part_ref, stable_face, rotation_angle, embedding, embedding_projected, color_code, color_hex, pose_index
                    FROM piece_embeddings
                """)
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[LegoVision DB Error] Error leyendo embeddings: {e}")
        return []


def get_latest_training_run() -> Optional[dict]:
    """Obtiene la corrida de entrenamiento más reciente (cualquier estado)."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, status, epochs, current_epoch, loss, val_loss,
                           map50, logs, config_used, started_at, ended_at
                    FROM training_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
                res = cur.fetchone()
                return dict(res) if res else None
    except Exception as e:
        print(f"[LegoVision DB Error] Error leyendo última corrida: {e}")
        return None


def count_embeddings() -> int:
    """Cuenta el número total de embeddings de piezas almacenados."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM piece_embeddings")
                res = cur.fetchone()
                return int(res["cnt"]) if res else 0
    except Exception as e:
        print(f"[LegoVision DB Error] Error contando embeddings: {e}")
        return 0


def init_database_tables():
    """Crea las tablas lego_sets, lego_set_parts y lego_set_minifigures si no existen."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lego_sets (
                    code            TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lego_set_parts (
                    set_code        TEXT REFERENCES lego_sets(code) ON DELETE CASCADE,
                    part_ref        TEXT NOT NULL,
                    color_code      TEXT NOT NULL,
                    color_hex       TEXT,
                    color_name      TEXT,
                    qty             INTEGER NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (set_code, part_ref, color_code)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lego_set_minifigures (
                    set_code        TEXT REFERENCES lego_sets(code) ON DELETE CASCADE,
                    minifig_ref     TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    qty             INTEGER NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (set_code, minifig_ref)
                );
            """)
    print("[LegoVision DB] Tablas de sets inicializadas en la base de datos.")


def get_set_from_db(set_id: str) -> Optional[dict]:
    """Obtiene el inventario de un set desde la base de datos."""
    clean_id = set_id.strip()
    if "-" not in clean_id:
        clean_id = f"{clean_id}-1"
        
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Obtener datos del set
                cur.execute("SELECT name FROM lego_sets WHERE code = %s", (clean_id,))
                set_row = cur.fetchone()
                if not set_row:
                    return None
                
                # 2. Obtener partes
                cur.execute("""
                    SELECT part_ref as ref, color_code, color_hex, color_name, qty
                    FROM lego_set_parts
                    WHERE set_code = %s
                """, (clean_id,))
                parts = [dict(r) for r in cur.fetchall()]
                
                # 3. Obtener minifiguras
                cur.execute("""
                    SELECT minifig_ref as ref, name, qty
                    FROM lego_set_minifigures
                    WHERE set_code = %s
                """, (clean_id,))
                minifigs = [dict(r) for r in cur.fetchall()]
                
                return {
                    "name": set_row["name"],
                    "minifigures": minifigs,
                    "parts": parts
                }
    except Exception as e:
        print(f"[LegoVision DB Error] get_set_from_db: {e}")
        return None


def save_set_to_db(set_id: str, set_data: dict):
    """Guarda o actualiza el catálogo e inventario de un set en la base de datos."""
    clean_id = set_id.strip()
    if "-" not in clean_id:
        clean_id = f"{clean_id}-1"
        
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Insertar o actualizar el set
                cur.execute("""
                    INSERT INTO lego_sets (code, name)
                    VALUES (%s, %s)
                    ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
                """, (clean_id, set_data["name"]))
                
                # 2. Insertar partes
                for part in set_data.get("parts", []):
                    cur.execute("""
                        INSERT INTO lego_set_parts (set_code, part_ref, color_code, color_hex, color_name, qty)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (set_code, part_ref, color_code) 
                        DO UPDATE SET 
                            color_hex = EXCLUDED.color_hex,
                            color_name = EXCLUDED.color_name,
                            qty = EXCLUDED.qty
                    """, (
                        clean_id,
                        part["ref"],
                        str(part["color_code"]),
                        part.get("color_hex"),
                        part.get("color_name"),
                        part["qty"]
                    ))
                
                # 3. Insertar minifiguras
                for mfg in set_data.get("minifigures", []):
                    cur.execute("""
                        INSERT INTO lego_set_minifigures (set_code, minifig_ref, name, qty)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (set_code, minifig_ref)
                        DO UPDATE SET name = EXCLUDED.name, qty = EXCLUDED.qty
                    """, (
                        clean_id,
                        mfg["ref"],
                        mfg["name"],
                        mfg["qty"]
                    ))
        print(f"[LegoVision DB] Set {clean_id} guardado correctamente en la BD.")
    except Exception as e:
        print(f"[LegoVision DB Error] save_set_to_db: {e}")


def get_set_parts_by_color(set_id: str, color_code: str) -> list[str]:
    """Retorna las referencias de piezas de un set específico que tienen un color determinado."""
    clean_id = set_id.strip()
    if "-" not in clean_id:
        clean_id = f"{clean_id}-1"
        
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT part_ref
                    FROM lego_set_parts
                    WHERE set_code = %s AND color_code = %s
                """, (clean_id, str(color_code)))
                rows = cur.fetchall()
                return [r["part_ref"] for r in rows]
    except Exception as e:
        print(f"[LegoVision DB Error] get_set_parts_by_color: {e}")
        return []


if __name__ == "__main__":
    test_connection()


# ----------------------------------------------------------------
# Minifig Assemblies - Mallas 3D ensambladas con colores por parte
# ----------------------------------------------------------------

def save_minifig_assembly(minifig_ref, name, glb_path, components, glb_data=None):
    """Guarda la malla 3D ensamblada de una minifigura con sus colores por parte."""
    import json as _json
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO minifig_assemblies (minifig_ref, name, glb_path, glb_data, components, updated_at)
                    VALUES (%s,%s,%s,%s,%s::jsonb,NOW())
                    ON CONFLICT (minifig_ref) DO UPDATE SET
                        name=EXCLUDED.name, glb_path=EXCLUDED.glb_path,
                        glb_data=EXCLUDED.glb_data, components=EXCLUDED.components, updated_at=NOW()
                """, (minifig_ref, name, glb_path, glb_data, _json.dumps(components)))
        print(f"[LegoVision DB] Minifig assembly {minifig_ref} guardada.")
        return True
    except Exception as e:
        print(f"[LegoVision DB Error] save_minifig_assembly: {e}")
        return False


def get_minifig_assembly(minifig_ref):
    """Obtiene la malla 3D ensamblada (sin datos binarios) desde la BD."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT minifig_ref, name, glb_path, components, assembled_at, updated_at "
                    "FROM minifig_assemblies WHERE minifig_ref=%s", (minifig_ref,)
                )
                res = cur.fetchone()
                if not res:
                    return None
                row = dict(res)
                for k in ("assembled_at", "updated_at"):
                    if row.get(k):
                        row[k] = row[k].isoformat()
                return row
    except Exception as e:
        print(f"[LegoVision DB Error] get_minifig_assembly: {e}")
        return None


def list_minifig_assemblies():
    """Lista todas las minifiguras que tienen malla 3D ensamblada en la BD."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT minifig_ref, name, glb_path, assembled_at, updated_at "
                    "FROM minifig_assemblies ORDER BY assembled_at DESC"
                )
                rows = cur.fetchall()
                result = []
                for r in rows:
                    row = dict(r)
                    for k in ("assembled_at", "updated_at"):
                        if row.get(k):
                            row[k] = row[k].isoformat()
                    result.append(row)
                return result
    except Exception as e:
        print(f"[LegoVision DB Error] list_minifig_assemblies: {e}")
        return []


def minifig_assembly_exists(minifig_ref):
    """Verifica si ya existe una malla 3D ensamblada para esta minifigura."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM minifig_assemblies WHERE minifig_ref=%s", (minifig_ref,))
                return cur.fetchone() is not None
    except Exception:
        return False
