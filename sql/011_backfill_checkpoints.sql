BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.backfill_checkpoints (
    checkpoint_id   BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    set_spec        TEXT NOT NULL,
    window_from     DATE NOT NULL,
    window_until    DATE NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETE','FAILED')),
    records_seen    INT  NOT NULL DEFAULT 0,
    records_kept    INT  NOT NULL DEFAULT 0,
    records_new     INT  NOT NULL DEFAULT 0,
    records_dupe    INT  NOT NULL DEFAULT 0,
    error           TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    UNIQUE (source, set_spec, window_from, window_until)
);

COMMIT;
