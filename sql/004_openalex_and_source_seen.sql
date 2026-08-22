BEGIN;

-- When Inoreader first saw the item (distinct from publisher published_at / radar ingested_at).
ALTER TABLE research_radar.content_items
    ADD COLUMN IF NOT EXISTS source_seen_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS ix_content_items_source_seen
    ON research_radar.content_items(source_seen_at DESC NULLS LAST);

-- OpenAlex resolution is tracked separately so 429/budget failures remain retryable
-- without resetting ENTITY_RESOLVED / SCORED for the whole corpus.
ALTER TABLE research_radar.paper_metadata
    ADD COLUMN IF NOT EXISTS openalex_status TEXT,
    ADD COLUMN IF NOT EXISTS openalex_work_id TEXT,
    ADD COLUMN IF NOT EXISTS openalex_checked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS openalex_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS openalex_last_error TEXT;

ALTER TABLE research_radar.paper_metadata
    DROP CONSTRAINT IF EXISTS ck_paper_metadata_openalex_status;

ALTER TABLE research_radar.paper_metadata
    ADD CONSTRAINT ck_paper_metadata_openalex_status
    CHECK (
        openalex_status IS NULL
        OR openalex_status IN (
            'NOT_NEEDED',
            'PENDING',
            'MATCHED',
            'NO_MATCH',
            'RATE_LIMITED',
            'ERROR'
        )
    );

CREATE INDEX IF NOT EXISTS ix_paper_metadata_openalex_status
    ON research_radar.paper_metadata(openalex_status)
    WHERE openalex_status IS NOT NULL;

-- Papers that already have local org evidence do not need OpenAlex.
UPDATE research_radar.paper_metadata pm
SET openalex_status = 'NOT_NEEDED',
    openalex_checked_at = COALESCE(openalex_checked_at, NOW()),
    openalex_last_error = NULL
WHERE COALESCE(openalex_status, '') NOT IN ('MATCHED', 'PENDING', 'RATE_LIMITED', 'ERROR')
  AND EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
  );

-- Papers with no org evidence yet: mark for OpenAlex retry (budget-aware stage).
UPDATE research_radar.paper_metadata pm
SET openalex_status = 'PENDING',
    openalex_last_error = COALESCE(openalex_last_error, 'backfilled_pending_after_prior_run')
WHERE COALESCE(openalex_status, '') IN ('', 'NOT_NEEDED')
  AND NOT EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
  )
  AND EXISTS (
        SELECT 1
        FROM research_radar.content_items ci
        WHERE ci.id = pm.content_id
          AND ci.status IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE', 'ENRICHED', 'RELEVANT')
  );

-- Ensure NOT_NEEDED sticks when orgs exist (override accidental PENDING from empty status).
UPDATE research_radar.paper_metadata pm
SET openalex_status = 'NOT_NEEDED',
    openalex_last_error = NULL
WHERE EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
  )
  AND COALESCE(openalex_status, '') <> 'MATCHED';

COMMIT;
