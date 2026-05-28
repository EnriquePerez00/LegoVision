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


def save_part_render(part_ref: str, color_code: str, image_base64: str):
    """Guarda o actualiza la imagen renderizada (Base64) de una pieza en la BD."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO part_renders (part_ref, color_code, image_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (part_ref, color_code) 
                DO UPDATE SET image_data = EXCLUDED.image_data, created_at = NOW()
            """, (part_ref, color_code, image_base64))

def get_part_render(part_ref: str, color_code: str) -> Optional[str]:
    """Obtiene la imagen renderizada en Base64 de una pieza si existe en la BD."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT image_data FROM part_renders
                    WHERE part_ref = %s AND color_code = %s
                """, (part_ref, color_code))
                res = cur.fetchone()
                return res["image_data"] if res else None
    except Exception as e:
        print(f"[LegoVision DB Error] Error leyendo render de BD: {e}")
        return None

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

def save_piece_embedding(part_ref: str, stable_face: int, rotation_angle: int, embedding: list[float]):
    """Guarda o actualiza el embedding de características de una pieza de Lego."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO piece_embeddings (part_ref, stable_face, rotation_angle, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (part_ref, stable_face, rotation_angle)
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = NOW()
            """, (part_ref, stable_face, rotation_angle, embedding))

def get_all_embeddings() -> list[dict]:
    """Obtiene todos los embeddings almacenados para comparación por similaridad."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT part_ref, stable_face, rotation_angle, embedding
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


if __name__ == "__main__":
    test_connection()
