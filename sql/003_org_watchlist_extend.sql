-- Extend organisations for Research Radar watchlist v2 (org_key + metadata).
BEGIN;

ALTER TABLE research_radar.organisations
    ADD COLUMN IF NOT EXISTS org_key TEXT,
    ADD COLUMN IF NOT EXISTS watchlist_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS evidence_sources TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS rationale TEXT,
    ADD COLUMN IF NOT EXISTS recent_highlight TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_organisations_org_key
    ON research_radar.organisations(org_key)
    WHERE org_key IS NOT NULL;

COMMIT;
