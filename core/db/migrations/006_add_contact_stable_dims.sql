-- =============================================================================
-- Migration 006: Add contact_stable_length / contact_stable_width to stable_poses
-- =============================================================================
-- Estos campos persisten las dimensiones reales (en mm) del rectángulo
-- envolvente mínimo (cv2.minAreaRect) de la cara que toca la cinta cuando la
-- pieza está en cada pose estable.
--
--   - contact_stable_length : DOUBLE PRECISION  (mm)
--       Lado MAYOR del minimum-area-rectangle de la cara de contacto.
--   - contact_stable_width  : DOUBLE PRECISION  (mm)
--       Lado MENOR del minimum-area-rectangle de la cara de contacto.
--
-- Se usan junto con `lateral_height` y `zenith_observable_area` (migration 005)
-- para filtrar y validar poses en el pipeline de inferencia y de generación de
-- renders sintéticos. Concretamente, el cache JSON consumido por Blender
-- (data/stable_poses_cache.json) excluye toda pose con
--      contact_stable_width  < 4.0 mm
--   O  stability_ratio       <= 0.05
-- por considerarse físicamente insostenible en una cinta transportadora real
-- (4 mm = ancho de un stud 2x2; 0.05 = 5 % de éxitos en la simulación física).
--
-- Cómo se calculan:
--   1. Cargar mesh LDraw de la pieza (`ldraw_mesh_parser.get_triangles`).
--   2. Proyectar todos los vértices sobre el plano perpendicular a
--      `contact_normal` y filtrar los que tienen proyección mínima a lo largo
--      de la normal (los más cercanos al suelo, dentro de tolerancia 0.5 LDU).
--   3. Aplicar `cv2.minAreaRect` sobre esos puntos 2D → (length_ldu, width_ldu).
--   4. Multiplicar por 0.4 (1 LDU = 0.4 mm).
--
-- Consulta de ejemplo:
--   SELECT part_ref, pose_index, face_class, stability_ratio,
--          ROUND(contact_stable_length::numeric, 2) AS L_mm,
--          ROUND(contact_stable_width::numeric, 2)  AS W_mm,
--          lateral_height, zenith_observable_area
--   FROM stable_poses
--   WHERE set_id = '75078-1'
--   ORDER BY part_ref, pose_index;
-- =============================================================================

ALTER TABLE stable_poses
    ADD COLUMN IF NOT EXISTS contact_stable_length DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS contact_stable_width  DOUBLE PRECISION;

COMMENT ON COLUMN stable_poses.contact_stable_length
    IS 'Lado mayor (mm) del minimum-area-rectangle de la cara que toca la cinta en la pose estable.';
COMMENT ON COLUMN stable_poses.contact_stable_width
    IS 'Lado menor (mm) del minimum-area-rectangle de la cara que toca la cinta en la pose estable. Si < 4 mm la pose se considera no sostenible y se excluye del cache.';