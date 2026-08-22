BEGIN;

CREATE OR REPLACE VIEW research_radar.v_content_analysis AS
SELECT
    ci.id AS content_id,
    ci.title,
    ci.canonical_url,
    ci.source,
    ci.source_type,
    ci.published_at,
    ci.published_at::date AS published_date,
    ci.source_seen_at,
    ci.source_seen_at::date AS source_seen_date,
    ci.ingested_at,
    ci.status,
    cs.ai_relevance,
    cs.technical_significance,
    cs.novelty,
    cs.practical_applicability,
    cs.professional_value,
    cs.student_learning_value,
    cs.explainability,
    cs.notable_org_signal,
    cs.notable_person_signal,
    cs.intrinsic_candidate_score,
    cs.score_version,
    cs.scoring_reason
FROM research_radar.content_items ci
LEFT JOIN research_radar.content_scores cs ON cs.content_id = ci.id;

COMMIT;
