BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_runs (
    run_id        UUID PRIMARY KEY,
    sample_seed   INT NOT NULL,
    sample_size   INT NOT NULL,
    prompt_kind   TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_results (
    result_id       BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES research_radar.bakeoff_runs(run_id) ON DELETE CASCADE,
    candidate_id    TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    content_id      BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    pass_index      INT NOT NULL DEFAULT 1,
    batch_arrangement TEXT,
    domain          TEXT,
    subdomains      TEXT[],
    application_domains JSONB,
    primary_audience TEXT,
    ai_relevance    NUMERIC(4,1),
    json_valid_first_try BOOLEAN NOT NULL,
    retries         INT NOT NULL DEFAULT 0,
    tokens_in       INT,
    tokens_out      INT,
    cost_usd        NUMERIC(10,6),
    latency_ms      INT,
    raw_response    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, candidate_id, content_id, pass_index, batch_arrangement)
);

CREATE TABLE IF NOT EXISTS research_radar.bakeoff_labels (
    content_id   BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    run_id       UUID NOT NULL REFERENCES research_radar.bakeoff_runs(run_id) ON DELETE CASCADE,
    labeller     TEXT NOT NULL,
    domain       TEXT,
    subdomains   TEXT[],
    application_domains TEXT[],
    is_general_method BOOLEAN,
    reasoning    TEXT,
    labelled_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, run_id, labeller)
);

CREATE INDEX IF NOT EXISTS idx_bakeoff_results_run_candidate
    ON research_radar.bakeoff_results (run_id, candidate_id);

CREATE INDEX IF NOT EXISTS idx_bakeoff_labels_run
    ON research_radar.bakeoff_labels (run_id);

COMMIT;
