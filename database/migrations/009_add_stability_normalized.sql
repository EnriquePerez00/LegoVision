-- Migration 009: campos de estabilidad normalizada y métricas deterministas
-- Adds:
--   - stability_ratio_normalized: sr / max(sr) per (part_ref, set_id)
--   - support_polygon_margin_mm:  distancia mínima 2D del CdM proyectado al
--                                 borde del polígono de soporte (mm)
--   - tipping_energy_ratio:       (sqrt(margin^2 + h_com^2) - h_com) / h_com,
--                                 ratio adimensional de incremento de altura
--                                 del CdM para volcar.

ALTER TABLE stable_poses
    ADD COLUMN IF NOT EXISTS stability_ratio_normalized DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS support_polygon_margin_mm  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tipping_energy_ratio       DOUBLE PRECISION;

COMMENT ON COLUMN stable_poses.stability_ratio_normalized IS
    'stability_ratio / max(stability_ratio) dentro del mismo (part_ref, set_id). '
    'Independiente del numero de caras candidatas y comparable entre piezas.';

COMMENT ON COLUMN stable_poses.support_polygon_margin_mm IS
    'Distancia minima (mm) desde la proyeccion del CdM hasta la arista mas '
    'cercana del poligono de soporte. >0 = estable; valores grandes = muy '
    'estable. Determinista, depende solo de la geometria LDraw + orientation_quat.';

COMMENT ON COLUMN stable_poses.tipping_energy_ratio IS
    'Ratio adimensional (sqrt(margin^2+h_com^2)-h_com)/h_com: incremento '
    'relativo de altura del CdM necesario para volcar pivotando sobre la '
    'arista mas cercana. Determinista, depende solo de la geometria.';