-- =============================================================================
-- Migration 007: extend piece_embeddings PK to include pose_index
-- =============================================================================
-- Antes el PK era (part_ref, stable_face, rotation_angle, color_hex) — un
-- diseño que asumía UNA sola pose estable canónica por pieza. Con el cache
-- multi-pose (sync_stable_poses_cache.py) ahora generamos N renders por
-- pieza × por rotación, uno por cada pose estable, y todos comparten
-- (part_ref, stable_face, rotation_angle, color_hex). El PK colisiona y
-- solo se persiste la primera fila del batch.
--
-- Solución: añadir `pose_index` al PK. Se trata como NOT NULL con default 0
-- para mantener compatibilidad con embeddings legacy.
-- =============================================================================

ALTER TABLE piece_embeddings ALTER COLUMN pose_index SET DEFAULT 0;
UPDATE piece_embeddings SET pose_index = 0 WHERE pose_index IS NULL;
ALTER TABLE piece_embeddings ALTER COLUMN pose_index SET NOT NULL;

ALTER TABLE piece_embeddings DROP CONSTRAINT IF EXISTS piece_embeddings_pkey;
ALTER TABLE piece_embeddings
    ADD CONSTRAINT piece_embeddings_pkey
    PRIMARY KEY (part_ref, stable_face, rotation_angle, color_hex, pose_index);

COMMENT ON COLUMN piece_embeddings.pose_index
    IS 'Índice de la pose estable usada para renderizar este embedding. Forma parte del PK porque la misma pieza/cámara/rotación/color tiene N embeddings, uno por cada pose física estable.';