BEGIN;

-- Generic affiliation-resolution status (replaces OpenAlex as the active queue).
-- Historical openalex_* columns are retained untouched for provenance.

ALTER TABLE research_radar.paper_metadata
    ADD COLUMN IF NOT EXISTS affiliation_status TEXT,
    ADD COLUMN IF NOT EXISTS affiliation_checked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS affiliation_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS affiliation_last_error TEXT;

ALTER TABLE research_radar.paper_metadata
    DROP CONSTRAINT IF EXISTS ck_paper_metadata_affiliation_status;

ALTER TABLE research_radar.paper_metadata
    ADD CONSTRAINT ck_paper_metadata_affiliation_status
    CHECK (
        affiliation_status IS NULL
        OR affiliation_status IN (
            'NOT_NEEDED',
            'PENDING',
            'MATCHED',
            'NO_MATCH',
            'REVIEW_REQUIRED',
            'ERROR'
        )
    );

CREATE INDEX IF NOT EXISTS ix_paper_metadata_affiliation_status
    ON research_radar.paper_metadata(affiliation_status)
    WHERE affiliation_status IS NOT NULL;

-- GPT affiliation resolver audit / idempotence table.
CREATE TABLE IF NOT EXISTS research_radar.affiliation_assessments (
    id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    resolver TEXT NOT NULL DEFAULT 'gpt_affiliation',
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    decision TEXT NOT NULL
        CHECK (decision IN ('MATCHED', 'NO_MATCH', 'REVIEW_REQUIRED', 'ERROR')),
    status TEXT NOT NULL
        CHECK (status IN ('COMPLETED', 'ERROR')),
    confidence NUMERIC(4,3),
    evidence_text TEXT,
    evidence_source TEXT,
    reason TEXT,
    organisations JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_fingerprint TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd NUMERIC(12,6),
    response_id TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (content_id, provider, model_name, prompt_version)
);

CREATE INDEX IF NOT EXISTS ix_affiliation_assessments_status
    ON research_radar.affiliation_assessments(status, decision);

-- Backfill: local org evidence already present → NOT_NEEDED.
UPDATE research_radar.paper_metadata pm
SET affiliation_status = 'NOT_NEEDED',
    affiliation_checked_at = COALESCE(affiliation_checked_at, NOW()),
    affiliation_last_error = NULL
WHERE COALESCE(affiliation_status, '') NOT IN ('MATCHED', 'PENDING', 'REVIEW_REQUIRED', 'ERROR', 'NO_MATCH')
  AND EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
          AND co.evidence_type IN ('email_domain', 'explicit_affiliation_text')
  );

-- Unresolved papers → PENDING for GPT affiliation resolver.
UPDATE research_radar.paper_metadata pm
SET affiliation_status = 'PENDING',
    affiliation_last_error = COALESCE(affiliation_last_error, 'awaiting_affiliation_gpt')
WHERE COALESCE(affiliation_status, '') IN ('', 'NOT_NEEDED')
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

-- Ensure NOT_NEEDED when any local evidence exists (override accidental PENDING).
UPDATE research_radar.paper_metadata pm
SET affiliation_status = 'NOT_NEEDED',
    affiliation_last_error = NULL
WHERE EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
          AND co.evidence_type IN ('email_domain', 'explicit_affiliation_text')
  )
  AND COALESCE(affiliation_status, '') = 'PENDING';

COMMIT;
