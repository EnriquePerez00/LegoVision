-- Migration 004: Tabla stable_poses
-- Sustituye el concepto de stable_face en piece_embeddings.

CREATE TABLE IF NOT EXISTS stable_poses (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    part_ref         TEXT NOT NULL,
    pose_index       INTEGER NOT NULL,
    contact_normal   DOUBLE PRECISION[] NOT NULL,
    face_class       TEXT NOT NULL,
    contact_area     DOUBLE PRECISION,
    orientation_quat  DOUBLE PRECISION[],
    orientation_euler DOUBLE PRECISION[],
    simulation_passes INTEGER DEFAULT 0,
    simulation_total  INTEGER DEFAULT 0,
    stability_ratio   DOUBLE PRECISION,
    is_stable        BOOLEAN DEFAULT TRUE,
    set_id           TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (part_ref, pose_index)
);

CREATE INDEX IF NOT EXISTS idx_stable_poses_part_ref   ON stable_poses(part_ref);
CREATE INDEX IF NOT EXISTS idx_stable_poses_face_class ON stable_poses(face_class);
CREATE INDEX IF NOT EXISTS idx_stable_poses_set_id     ON stable_poses(set_id);

CREATE OR REPLACE VIEW stable_pose_summary AS
SELECT part_ref,
    COUNT(*) AS n_stable_poses,
    array_agg(face_class ORDER BY pose_index) AS face_classes,
    array_agg(pose_index ORDER BY pose_index) AS pose_indices,
    MIN(stability_ratio) AS min_stability_ratio
FROM stable_poses WHERE is_stable = TRUE
GROUP BY part_ref ORDER BY part_ref;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'piece_embeddings' AND column_name = 'pose_index'
    ) THEN
        ALTER TABLE piece_embeddings ADD COLUMN pose_index INTEGER;
    END IF;
END $$;
