-- Hierarchical topic tags + extracted key claims (topics stage).
-- Extends the existing topics / content_topics tables rather than creating
-- parallel ones — query.py and the deterministic scorer already read
-- content_topics.

BEGIN;

ALTER TABLE research_radar.topics
    ADD COLUMN IF NOT EXISTS level           TEXT NOT NULL DEFAULT 'topic'
        CHECK (level IN ('domain','subdomain','topic','application')),
    ADD COLUMN IF NOT EXISTS parent_topic_id BIGINT
        REFERENCES research_radar.topics(topic_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS origin          TEXT NOT NULL DEFAULT 'seed'
        CHECK (origin IN ('seed','llm')),
    ADD COLUMN IF NOT EXISTS usage_count     INT NOT NULL DEFAULT 0;

-- The ALTER above backfills level='subdomain'/origin='seed' on every existing
-- row (including the 13 topics seeded by 002_seed_watchlists.sql) via the
-- column DEFAULT, so nothing already tagged is orphaned.

CREATE INDEX IF NOT EXISTS ix_topics_level  ON research_radar.topics(level);
CREATE INDEX IF NOT EXISTS ix_topics_parent ON research_radar.topics(parent_topic_id);

CREATE TABLE IF NOT EXISTS research_radar.content_claims (
    claim_id     BIGSERIAL PRIMARY KEY,
    content_id   BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    metric       TEXT NOT NULL,          -- 'true negative rate'
    value_text   TEXT NOT NULL,          -- '99%'  (verbatim from abstract)
    value_num    NUMERIC,                -- 99.0   (parsed where possible, else NULL)
    unit         TEXT,                   -- '%', 'BLEU', 'ms'
    task         TEXT,                   -- 'AI-generated text detection'
    dataset      TEXT,                   -- named dataset, NULL if unnamed
    qualifier    TEXT NOT NULL DEFAULT 'self_reported'
        CHECK (qualifier IN ('self_reported','independent','unclear')),
    evidence     TEXT NOT NULL,          -- the abstract sentence it came from
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_claims_content ON research_radar.content_claims(content_id);
CREATE INDEX IF NOT EXISTS ix_claims_metric  ON research_radar.content_claims(lower(metric));

CREATE TABLE IF NOT EXISTS research_radar.content_topic_assessments (
    content_id     BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    provider       TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status         TEXT NOT NULL,
    tokens_in      INT,
    tokens_out     INT,
    cost_usd       NUMERIC(10,6),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, provider, model_name, prompt_version)
);

COMMIT;
