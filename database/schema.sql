-- ================================================================
-- LegoVision — Schema de Base de Datos
-- PostgreSQL 16 | Supabase-compatible
-- ================================================================

-- Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ----------------------------------------------------------------
-- TABLA: models
-- Registro de modelos YOLOv8 entrenados
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version         TEXT UNIQUE NOT NULL,          -- e.g. "v1.0.0-20240527"
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    model_path      TEXT NOT NULL,                 -- ruta local al .pt
    dataset_size    INTEGER,                       -- número de imágenes de training
    num_classes     INTEGER,                       -- número de clases LEGO
    map50           FLOAT,                         -- mAP@0.5
    map50_95        FLOAT,                         -- mAP@0.5:0.95
    precision       FLOAT,
    recall          FLOAT,
    epochs_trained  INTEGER,
    training_device TEXT DEFAULT 'cuda',           -- cuda | mps | cpu
    notes           TEXT,
    is_active       BOOLEAN DEFAULT FALSE,         -- modelo en producción activo
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- TABLA: inference_sessions
-- Sesiones de inferencia (cada vez que se arranca el sistema)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    model_id            UUID REFERENCES models(id) ON DELETE SET NULL,
    model_version       TEXT,
    total_detections    INTEGER DEFAULT 0,
    avg_confidence      FLOAT,
    avg_fps             FLOAT,
    belt_speed_mm_s     FLOAT DEFAULT 83.3,        -- velocidad de cinta (mm/s)
    notes               TEXT
);

-- ----------------------------------------------------------------
-- TABLA: detections
-- Cada detección individual de pieza LEGO
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES inference_sessions(id) ON DELETE CASCADE,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    -- Clase detectada
    piece_class     TEXT NOT NULL,                 -- ID LDraw, e.g. "3001"
    piece_name      TEXT,                          -- Nombre descriptivo, e.g. "Brick 2x4"
    confidence      FLOAT NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    -- Bounding box normalizada (0-1)
    bbox_x_center   FLOAT NOT NULL,
    bbox_y_center   FLOAT NOT NULL,
    bbox_width      FLOAT NOT NULL,
    bbox_height     FLOAT NOT NULL,
    -- Imagen fuente
    image_path      TEXT,                          -- ruta al frame capturado
    image_width     INTEGER DEFAULT 2448,
    image_height    INTEGER DEFAULT 2048,
    -- Metadatos de inferencia
    inference_ms    FLOAT,                         -- latencia de esta inferencia (ms)
    model_id        UUID REFERENCES models(id) ON DELETE SET NULL
);

-- ----------------------------------------------------------------
-- TABLA: lego_classes
-- Catálogo de clases LEGO reconocibles
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lego_classes (
    id              SERIAL PRIMARY KEY,
    ldraw_id        TEXT UNIQUE NOT NULL,           -- ID LDraw, e.g. "3001"
    name            TEXT NOT NULL,                  -- "Brick 2x4"
    category        TEXT,                           -- "Brick", "Plate", "Tile", etc.
    color_hex       TEXT,                           -- color representativo para UI
    yolo_class_idx  INTEGER UNIQUE,                 -- índice en el modelo YOLO
    detection_count BIGINT DEFAULT 0,               -- conteo acumulado de detecciones
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- TABLA: dataset_runs
-- Registro de generaciones de dataset con Blender
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    generated_at    TIMESTAMPTZ DEFAULT NOW(),
    num_images      INTEGER NOT NULL,
    pieces_per_image INTEGER NOT NULL,
    num_classes     INTEGER,
    blender_version TEXT,
    output_path     TEXT,
    notes           TEXT
);

-- ----------------------------------------------------------------
-- ÍNDICES para consultas frecuentes
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_detections_session   ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_class      ON detections(piece_class);
CREATE INDEX IF NOT EXISTS idx_detections_time       ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_started      ON inference_sessions(started_at DESC);

-- ----------------------------------------------------------------
-- VISTAS útiles para la GUI
-- ----------------------------------------------------------------

-- Vista: resumen de detecciones por sesión
CREATE OR REPLACE VIEW session_summary AS
SELECT
    s.id,
    s.started_at,
    s.ended_at,
    s.model_version,
    s.total_detections,
    s.avg_confidence,
    s.avg_fps,
    EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at)) AS duration_seconds,
    COUNT(d.id) AS actual_detection_count
FROM inference_sessions s
LEFT JOIN detections d ON d.session_id = s.id
GROUP BY s.id;

-- Vista: top clases detectadas
CREATE OR REPLACE VIEW top_detected_classes AS
SELECT
    piece_class,
    piece_name,
    COUNT(*) AS total_detections,
    AVG(confidence) AS avg_confidence,
    MAX(detected_at) AS last_detected
FROM detections
GROUP BY piece_class, piece_name
ORDER BY total_detections DESC;

-- ----------------------------------------------------------------
-- Datos iniciales
-- ----------------------------------------------------------------
INSERT INTO models (version, model_path, notes, is_active)
VALUES ('placeholder-v0.0', './runs/train/best.pt', 'Placeholder hasta completar training', FALSE)
ON CONFLICT (version) DO NOTHING;


-- ----------------------------------------------------------------
-- TABLA: training_runs
-- Registro de sesiones de entrenamiento YOLO
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    status          TEXT NOT NULL, -- 'running', 'completed', 'failed'
    epochs          INTEGER NOT NULL,
    current_epoch   INTEGER DEFAULT 0,
    loss            FLOAT,
    val_loss        FLOAT,
    map50           FLOAT,
    logs            TEXT,
    config_used     JSONB
);

-- ----------------------------------------------------------------
-- TABLA: piece_embeddings
-- Embeddings DINOv2 por pieza y orientación para comparación de similitud
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS piece_embeddings (
    part_ref        TEXT NOT NULL,
    stable_face     INTEGER NOT NULL,     -- 0 (Top), 1 (Side), 2 (Bottom)
    rotation_angle  INTEGER NOT NULL,     -- e.g. 0, 30, 60... 330
    embedding       DOUBLE PRECISION[] NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (part_ref, stable_face, rotation_angle)
);


-- ----------------------------------------------------------------
-- TABLA: lego_sets
-- Sets de LEGO cargados en el sistema
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lego_sets (
    code            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- TABLA: lego_set_parts
-- Inventario de piezas reales para cada set
-- ----------------------------------------------------------------
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

-- ----------------------------------------------------------------
-- TABLA: lego_set_minifigures
-- Inventario de minifiguras para cada set
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lego_set_minifigures (
    set_code        TEXT REFERENCES lego_sets(code) ON DELETE CASCADE,
    minifig_ref     TEXT NOT NULL,
    name            TEXT NOT NULL,
    qty             INTEGER NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (set_code, minifig_ref)
);



-- ----------------------------------------------------------------
-- TABLA: minifig_assemblies
-- Almacena mallas 3D ensambladas de minifiguras con colores por parte
-- (Contenido de migrations/003_add_minifig_assemblies.sql integrado aquí)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS minifig_assemblies (
    minifig_ref     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    glb_path        TEXT,
    glb_data        BYTEA,
    components      JSONB NOT NULL,
    assembled_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_minifig_assemblies_ref ON minifig_assemblies(minifig_ref);

-- Migración inline: añadir columnas a piece_embeddings si ya existía sin color_code/color_hex
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'piece_embeddings' AND column_name = 'color_code'
    ) THEN
        ALTER TABLE piece_embeddings ADD COLUMN color_code TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'piece_embeddings' AND column_name = 'color_hex'
    ) THEN
        ALTER TABLE piece_embeddings ADD COLUMN color_hex TEXT;
    END IF;
END $$;
