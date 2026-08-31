BEGIN;

ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS learning_value NUMERIC(4,1);
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS evidence_strength NUMERIC(4,1);
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS newsletter_fit NUMERIC(4,1);
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS so_what TEXT;
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS reason_not_higher TEXT;
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS confidence NUMERIC(4,1);
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS scoring_tier TEXT;
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS batch_id UUID;
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS batch_size INT;
ALTER TABLE research_radar.content_score_assessments ADD COLUMN IF NOT EXISTS batch_position INT;

CREATE TABLE IF NOT EXISTS research_radar.content_independence_assessments (
    assessment_id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('independent','self_evaluation','unclear','not_applicable','ERROR')),
    reason TEXT,
    evidence_used TEXT,
    tokens_in INT,
    tokens_out INT,
    cost_usd NUMERIC(10,6),
    response_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (content_id, provider, model_name, prompt_version)
);
CREATE INDEX IF NOT EXISTS ix_content_independence_assessments_content
    ON research_radar.content_independence_assessments(content_id);

ALTER TABLE research_radar.content_final_scores ADD COLUMN IF NOT EXISTS newsletter_score NUMERIC(5,2);
ALTER TABLE research_radar.content_final_scores ADD COLUMN IF NOT EXISTS evidence_factor NUMERIC(4,3);
ALTER TABLE research_radar.content_final_scores ADD COLUMN IF NOT EXISTS independence_factor NUMERIC(4,3);
ALTER TABLE research_radar.content_final_scores ADD COLUMN IF NOT EXISTS independence_status TEXT;

CREATE TABLE IF NOT EXISTS research_radar.editorial_reviews (
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    score_version TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('STRONG','BORDERLINE','WEAK')),
    note TEXT,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, score_version, reviewer)
);

-- CREATE OR REPLACE cannot rename/reorder view columns; drop first.
DROP VIEW IF EXISTS research_radar.v_research_radar_top;
CREATE VIEW research_radar.v_research_radar_top AS
SELECT
    fs.score_version,
    fs.final_score,
    fs.semantic_core,
    fs.org_boost,
    fs.person_boost,
    fs.newsletter_score,
    fs.evidence_factor,
    fs.independence_factor,
    fs.independence_status,
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
    a.scoring_tier,
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
    a.learning_value,
    a.evidence_strength,
    a.newsletter_fit,
    a.so_what,
    a.reason_not_higher,
    a.confidence,
    a.ai_relevance,
    a.semantic_score AS assessment_semantic_score,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'organisation', o.canonical_name,
            'priority', o.priority,
            'evidence_type', co.evidence_type,
            'evidence_text', co.evidence_text,
            'confidence', co.confidence
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
