-- Migration 003: Add minifig_assemblies table
-- LegoVision — Almacena mallas 3D ensambladas de minifiguras con colores por parte
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
-- Add color_code/color_hex columns to piece_embeddings if missing (for render pipeline)
ALTER TABLE piece_embeddings ADD COLUMN IF NOT EXISTS color_code TEXT;
ALTER TABLE piece_embeddings ADD COLUMN IF NOT EXISTS color_hex TEXT;
