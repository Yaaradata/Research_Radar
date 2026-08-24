BEGIN;

CREATE TABLE IF NOT EXISTS research_radar.content_score_assessments (
    assessment_id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    assessment_type TEXT NOT NULL DEFAULT 'llm_semantic',
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    sample_group TEXT CHECK (
        sample_group IS NULL
        OR sample_group IN ('top', 'threshold', 'low', 'random', 'full')
    ),
    ai_relevance NUMERIC(4,2),
    technical_significance NUMERIC(4,2),
    practical_applicability NUMERIC(4,2),
    professional_value NUMERIC(4,2),
    student_learning_value NUMERIC(4,2),
    apparent_novelty NUMERIC(4,2),
    explainability NUMERIC(4,2),
    industry_relevance NUMERIC(4,2),
    semantic_score NUMERIC(5,2),
    reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
    industry_labels JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd NUMERIC(12,6),
    response_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('COMPLETED', 'ERROR', 'RATE_LIMITED', 'REFUSED')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ux_content_score_assessment_version
        UNIQUE (content_id, provider, model_name, prompt_version)
);

CREATE INDEX IF NOT EXISTS ix_content_score_assessments_content
    ON research_radar.content_score_assessments(content_id);

CREATE INDEX IF NOT EXISTS ix_content_score_assessments_sample
    ON research_radar.content_score_assessments(sample_group, status);

CREATE INDEX IF NOT EXISTS ix_content_score_assessments_semantic
    ON research_radar.content_score_assessments(semantic_score DESC NULLS LAST);

COMMIT;
