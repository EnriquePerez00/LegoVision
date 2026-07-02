-- ================================================================
-- LegoVision — Migración 004
-- Tabla subpart_color_overrides
--
-- Cachea decisiones del resolver de colores LDraw → BrickLink
-- para piezas compuestas (shortcuts cXX) y subfiles con color
-- hardcoded (ldraw_color != 16) que necesitan ser pintados con
-- un color BrickLink específico al renderizar.
--
-- Uso:
--   - Lookup determinístico por (parent_ref, parent_color, sub_ref)
--   - Persistencia de hallazgos del scrape on-demand de BrickLink
--   - Auditoría del origen de cada decisión (bricklink / ldraw_map / manual)
-- ================================================================

CREATE TABLE IF NOT EXISTS subpart_color_overrides (
    parent_ref    TEXT NOT NULL,                      -- e.g. "15391"
    parent_color  TEXT NOT NULL,                      -- e.g. "11" (BL color del padre)
    sub_ref       TEXT NOT NULL,                      -- e.g. "15392"
    sub_color     TEXT NOT NULL,                      -- e.g. "85" (BL color resuelto)
    sub_color_hex TEXT,                               -- redundante para inspección rápida
    sub_color_name TEXT,
    source        TEXT NOT NULL CHECK (source IN ('bricklink','ldraw_map','manual','inventory')),
    set_code      TEXT,                               -- set BL desde el que se aprendió
    minifig_ref   TEXT,                               -- minifig BL si aplica
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (parent_ref, parent_color, sub_ref)
);

CREATE INDEX IF NOT EXISTS idx_subpart_overrides_parent
    ON subpart_color_overrides(parent_ref);
CREATE INDEX IF NOT EXISTS idx_subpart_overrides_sub
    ON subpart_color_overrides(sub_ref);
CREATE INDEX IF NOT EXISTS idx_subpart_overrides_set
    ON subpart_color_overrides(set_code);

COMMENT ON TABLE subpart_color_overrides IS
'Resoluciones cacheadas de la jerarquía LDraw→BrickLink para piezas compuestas (shortcuts).';
COMMENT ON COLUMN subpart_color_overrides.source IS
'bricklink = re-scrape del inventario; inventory = ya estaba en lego_set_parts; ldraw_map = mapeo del color_catalog; manual = override humano.';