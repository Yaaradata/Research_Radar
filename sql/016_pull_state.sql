BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.pull_state (
    source                TEXT PRIMARY KEY,        -- 'inoreader' | 'arxiv_oai'
    last_run_started_at   TIMESTAMPTZ,
    last_run_completed_at TIMESTAMPTZ,
    last_run_id           UUID,
    last_published_at     TIMESTAMPTZ,             -- newest published_at successfully stored
    last_external_id      TEXT,
    items_last_run        INT,
    items_total           INT NOT NULL DEFAULT 0,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
