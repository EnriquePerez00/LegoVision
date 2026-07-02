-- Migration 008: Catálogo de piezas que componen cada minifig
-- LegoVision — Permite distinguir piezas regulares de un set vs piezas que
-- pertenecen a una minifigura (cuerpo, torso, cabeza, etc.)

-- ----------------------------------------------------------------
-- TABLA: minifig_parts
-- Inventario de piezas (cuerpo/torso/cabeza/etc.) de cada minifig.
-- Una fila por (minifig_ref, part_ref, color_code).
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS minifig_parts (
    minifig_ref   TEXT NOT NULL,
    part_ref      TEXT NOT NULL,
    color_code    TEXT NOT NULL,
    color_hex     TEXT,
    color_name    TEXT,
    qty           INTEGER NOT NULL DEFAULT 1,
    name          TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (minifig_ref, part_ref, color_code)
);

CREATE INDEX IF NOT EXISTS idx_minifig_parts_part_ref
    ON minifig_parts(part_ref);

-- ----------------------------------------------------------------
-- Añadir flag y referencia a lego_set_parts para distinguir las piezas
-- del inventario que provienen de una minifig.
-- ----------------------------------------------------------------
ALTER TABLE lego_set_parts
    ADD COLUMN IF NOT EXISTS is_minifig_part BOOLEAN DEFAULT FALSE;

ALTER TABLE lego_set_parts
    ADD COLUMN IF NOT EXISTS minifig_ref TEXT;