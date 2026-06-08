-- =============================================================================
-- Migration 010: añadir columna `zenith_silhouette_area` a stable_poses
-- =============================================================================
-- Motivación
-- ----------
-- Hasta ahora `zenith_observable_area` se calcula como Convex Hull 2D del mesh
-- LDraw proyectado sobre el plano perpendicular a `contact_normal`. Esto
-- sobreestima el área visible cuando la pieza tiene:
--   - Concavidades externas (escotaduras del contorno)
--   - Agujeros pasantes que el mesh modela explícitamente
--
-- Esta migración añade `zenith_silhouette_area` (DOUBLE PRECISION, mm²) que se
-- poblará mediante shapely.unary_union de los triángulos LDraw proyectados.
-- Se mantiene la columna `zenith_observable_area` (Convex Hull) sin tocar
-- para no romper análisis históricos ni reports ya generados.
--
-- Columna añadida
-- ---------------
--   zenith_silhouette_area DOUBLE PRECISION
--     · Área de la silueta REAL (no convexa) del mesh proyectado, en mm².
--     · Se calcula con `populate_silhouette_areas.py`.
--     · Detecta concavidades externas. NO siempre detecta agujeros pasantes
--       (depende de cómo el .dat de LDraw modele las paredes interiores).
--
-- Invariante esperado
-- -------------------
--   zenith_silhouette_area  <=  zenith_observable_area  <=  zenith_bbox_area
-- =============================================================================

ALTER TABLE stable_poses
    ADD COLUMN IF NOT EXISTS zenith_silhouette_area DOUBLE PRECISION;

COMMENT ON COLUMN stable_poses.zenith_silhouette_area IS
    'Área de la silueta real (mm²) del mesh LDraw proyectado sobre el plano '
    'perpendicular a contact_normal. Calculada con shapely.unary_union de los '
    'triángulos. Detecta concavidades externas; agujeros pasantes sólo si la '
    'malla los modela explícitamente. Se mantiene zenith_observable_area como '
    'Convex Hull para comparación histórica.';