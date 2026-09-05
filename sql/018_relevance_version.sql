BEGIN;

ALTER TABLE research_radar.content_items
    ADD COLUMN IF NOT EXISTS relevance_version TEXT;

CREATE INDEX IF NOT EXISTS ix_content_relevance_version
    ON research_radar.content_items(relevance_version)
    WHERE status = 'REJECTED';

UPDATE research_radar.content_items
SET relevance_version = 'relevance-v0'
WHERE status = 'REJECTED' AND relevance_version IS NULL;

COMMIT;
