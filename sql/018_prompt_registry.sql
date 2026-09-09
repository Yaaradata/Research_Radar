BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.prompt_versions (
    prompt_id   BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN
                  ('paper_scoring','screen','classify','independence','topics',
                   'policy','embedding_policy','clustering_policy')),
    version     TEXT NOT NULL,
    model_name  TEXT,
    body_sha256 TEXT NOT NULL,
    body        TEXT NOT NULL,
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (kind, version)
);

COMMIT;
