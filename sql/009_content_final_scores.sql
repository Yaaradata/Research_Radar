BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.content_final_scores (
    content_id     BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    score_version  TEXT   NOT NULL,
    semantic_core  NUMERIC(5,2) NOT NULL CHECK (semantic_core BETWEEN 0 AND 10),
    org_boost      NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (org_boost >= 0),
    person_boost   NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (person_boost >= 0),
    final_score    NUMERIC(5,2) NOT NULL CHECK (final_score BETWEEN 0 AND 10),
    assessment_id  BIGINT REFERENCES research_radar.content_score_assessments(assessment_id) ON DELETE SET NULL,
    components     JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance     JSONB NOT NULL DEFAULT '{}'::jsonb,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, score_version)
);

CREATE INDEX IF NOT EXISTS ix_final_scores_rank
    ON research_radar.content_final_scores(score_version, final_score DESC);

CREATE OR REPLACE VIEW research_radar.v_research_radar_top AS
SELECT
    fs.score_version,
    fs.final_score,
    fs.semantic_core,
    fs.org_boost,
    fs.person_boost,
    fs.components,
    fs.provenance,
    fs.computed_at,
    ci.id AS content_id,
    ci.title,
    ci.canonical_url,
    ci.published_at,
    ci.status,
    pm.arxiv_id,
    pm.abstract,
    a.reasons,
    a.industry_labels,
    a.model_name,
    a.prompt_version,
    a.technical_significance,
    a.practical_applicability,
    a.professional_value,
    a.student_learning_value,
    a.apparent_novelty,
    a.explainability,
    a.industry_relevance,
    a.ai_relevance,
    a.semantic_score AS assessment_semantic_score,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'organisation', o.canonical_name,
            'priority',     o.priority,
            'evidence_type',co.evidence_type,
            'evidence_text',co.evidence_text,
            'confidence',   co.confidence
        ) ORDER BY o.priority DESC, o.canonical_name)
        FROM research_radar.content_organisations co
        JOIN research_radar.organisations o ON o.organisation_id = co.organisation_id
        WHERE co.content_id = ci.id
          AND co.relationship_type = 'paper_author_affiliation'
          AND co.current_affiliation = FALSE
    ), '[]'::jsonb) AS organisations
FROM research_radar.content_final_scores fs
JOIN research_radar.content_items ci ON ci.id = fs.content_id
LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
LEFT JOIN research_radar.content_score_assessments a ON a.assessment_id = fs.assessment_id;

COMMIT;
