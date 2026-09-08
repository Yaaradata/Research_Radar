BEGIN;

-- Canonical relevance_version column + general index.
-- sql/018_relevance_version.sql depends on this column and only runs the
-- REJECTED backfill + partial index (no second ADD COLUMN).

ALTER TABLE research_radar.content_items
    ADD COLUMN IF NOT EXISTS relevance_version TEXT;

CREATE INDEX IF NOT EXISTS ix_content_items_relevance_version
    ON research_radar.content_items(relevance_version)
    WHERE relevance_version IS NOT NULL;

COMMIT;
