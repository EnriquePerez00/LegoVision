-- =============================================================================
-- Migration 011: añadir columna `effective_height` y `efective_height` a stable_poses
-- =============================================================================
-- Motivación
-- ----------
-- Para evitar la sobrecorrección por perspectiva de piezas con rampa (slopes)
-- apoyadas de forma inclinada, se introduce el concepto de "altura efectiva"
-- (altura media de la superficie proyectada) en lugar de usar la altura máxima.
--
-- Columnas añadidas
-- -----------------
--   effective_height DOUBLE PRECISION (altura efectiva real calculada en mm)
--   efective_height  DOUBLE PRECISION (alias con una sola f para compatibilidad)
-- =============================================================================

ALTER TABLE stable_poses
    ADD COLUMN IF NOT EXISTS effective_height DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS efective_height DOUBLE PRECISION;

COMMENT ON COLUMN stable_poses.effective_height IS
    'Altura efectiva de la pose (mm), calculada como la altura media ponderada '
    'por área de la proyección cenital de la pieza. Evita la sobrecorrección '
    'por perspectiva en piezas inclinadas/rampas (slopes).';

COMMENT ON COLUMN stable_poses.efective_height IS
    'Alias de effective_height para compatibilidad.';
