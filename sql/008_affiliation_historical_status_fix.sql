BEGIN;

-- Idempotent fix for DBs that already applied an earlier 007 without historical
-- Crossref/OpenAlex terminal status backfill.

UPDATE research_radar.paper_metadata pm
SET affiliation_status = 'MATCHED',
    affiliation_checked_at = COALESCE(affiliation_checked_at, NOW()),
    affiliation_last_error = COALESCE(
        affiliation_last_error,
        'historical_external_affiliation_preserved'
    )
WHERE (
        affiliation_status IS NULL
        OR COALESCE(affiliation_status, '') IN ('', 'PENDING')
      )
  AND EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
          AND co.evidence_type IN (
              'paper_specific_crossref',
              'paper_specific_openalex',
              'paper_specific_openalex_doi'
          )
  )
  AND NOT EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co2
        WHERE co2.content_id = pm.content_id
          AND co2.relationship_type = 'paper_author_affiliation'
          AND co2.current_affiliation = FALSE
          AND co2.evidence_type IN ('email_domain', 'explicit_affiliation_text')
  );

UPDATE research_radar.paper_metadata pm
SET affiliation_status = 'MATCHED',
    affiliation_checked_at = COALESCE(affiliation_checked_at, NOW()),
    affiliation_last_error = COALESCE(
        affiliation_last_error,
        'historical_external_affiliation_preserved'
    )
WHERE affiliation_status IS NULL
  AND EXISTS (
        SELECT 1
        FROM research_radar.content_organisations co
        WHERE co.content_id = pm.content_id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
          AND co.evidence_type NOT IN ('email_domain', 'explicit_affiliation_text')
  );

COMMIT;
