-- Add general-method to the topics-stage application vocabulary so an empty
-- application list is no longer the only way to express "no sector" — that
-- made force-fitting unmeasurable (empty ≡ never tried).

BEGIN;

INSERT INTO research_radar.topics (canonical_name, level, origin)
VALUES ('general-method', 'application', 'seed')
ON CONFLICT (canonical_name) DO UPDATE SET
    level = EXCLUDED.level,
    origin = EXCLUDED.origin,
    active = TRUE;

COMMIT;
