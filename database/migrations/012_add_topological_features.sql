-- =============================================================================
-- Migration 012: añadir columna `topological_features` a lego_classes
-- =============================================================================
-- Motivación
-- ----------
-- Almacena características topológicas deterministas extraídas del archivo
-- LDraw (.dat) correspondiente a cada pieza para enriquecer los metadatos de búsqueda
-- e inferencia.
--
-- Columnas añadidas
-- -----------------
--   topological_features JSONB (objeto JSON que cuenta la presencia de espigas,
--                              agujeros, clips, tubos, pines, etc.)
-- =============================================================================

ALTER TABLE lego_classes
    ADD COLUMN IF NOT EXISTS topological_features JSONB DEFAULT '{}'::jsonb NOT NULL;

COMMENT ON COLUMN lego_classes.topological_features IS
    'Metadatos topológicos 3D deterministas extraídos de archivos LDraw, '
    'como conteo de studs (sólidos/huecos), agujeros Technic, clips, etc.';
