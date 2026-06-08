-- Migration 005: Add zenith_observable_area to stable_poses

ALTER TABLE stable_poses ADD COLUMN IF NOT EXISTS zenith_observable_area DOUBLE PRECISION;
