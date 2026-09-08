BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_runs (
    run_id         UUID PRIMARY KEY,
    sample_seed    INT NOT NULL,
    sample_size    INT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_results (
    result_id          BIGSERIAL PRIMARY KEY,
    run_id             UUID NOT NULL REFERENCES research_radar.bakeoff_runs(run_id) ON DELETE CASCADE,
    candidate_id       TEXT NOT NULL,
    model              TEXT NOT NULL,
    content_id         BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    pass_index         INT NOT NULL DEFAULT 1,
    application_domain TEXT[],
    audience_relevance TEXT[],
    paper_kind         TEXT,
    geography_focus    TEXT,
    domain_confidence  NUMERIC(4,1),
    raw_response       TEXT NOT NULL,
    json_valid         BOOLEAN NOT NULL,
    schema_valid       BOOLEAN NOT NULL,
    dropped_values     TEXT[],
    retries            INT NOT NULL DEFAULT 0,
    tokens_in          INT,
    tokens_out         INT,
    cost_usd           NUMERIC(10,6),
    latency_ms         INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, candidate_id, content_id, pass_index)
);

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_labels (
    content_id         BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    run_id             UUID NOT NULL REFERENCES research_radar.bakeoff_runs(run_id) ON DELETE CASCADE,
    labeller           TEXT NOT NULL,
    application_domain TEXT[],
    audience_relevance TEXT[],
    paper_kind         TEXT,
    reasoning          TEXT,
    labelled_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, run_id, labeller)
);

CREATE INDEX IF NOT EXISTS idx_bakeoff_results_run_candidate
    ON research_radar.bakeoff_results (run_id, candidate_id);

CREATE INDEX IF NOT EXISTS idx_bakeoff_labels_run
    ON research_radar.bakeoff_labels (run_id);

COMMIT;
