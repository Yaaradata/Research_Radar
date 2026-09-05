BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.s3_archives (
    archive_id    BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('raw','rejected')),
    stage         TEXT NOT NULL,
    s3_bucket     TEXT NOT NULL,
    s3_key        TEXT NOT NULL,
    record_count  INT NOT NULL,
    bytes_written BIGINT,
    window_from   DATE,
    window_until  DATE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (s3_bucket, s3_key)
);

CREATE INDEX IF NOT EXISTS ix_s3_archives_run ON research_radar.s3_archives(run_id);

COMMIT;
