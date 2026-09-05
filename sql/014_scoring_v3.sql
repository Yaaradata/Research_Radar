BEGIN;

-- Scoring v3: content_classifications (classify pass) + independence paper_kind

CREATE TABLE IF NOT EXISTS research_radar.content_classifications (
    content_id              BIGINT NOT NULL
        REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    prompt_version          TEXT NOT NULL,
    model                   TEXT NOT NULL,
    classify_input_kind     TEXT NOT NULL,
    application_domain      TEXT[] NOT NULL,
    audience_relevance      TEXT[] NOT NULL,
    paper_kind              TEXT NOT NULL,
    geography_focus         TEXT NOT NULL,
    domain_confidence       NUMERIC(4,1) NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_id, prompt_version),
    CONSTRAINT content_classifications_input_kind CHECK (
        classify_input_kind = 'title_categories_abstract'
    ),
    CONSTRAINT content_classifications_application_domain_len CHECK (
        cardinality(application_domain) >= 0 AND cardinality(application_domain) <= 3
    ),
    CONSTRAINT content_classifications_application_domain_values CHECK (
        application_domain <@ ARRAY[
            'general_method', 'healthcare_life_sciences', 'financial_services',
            'manufacturing_industrial', 'energy_utilities', 'mining_resources',
            'retail_ecommerce', 'transport_logistics', 'agriculture',
            'telecom_networks', 'public_sector_govtech', 'legal_compliance',
            'education', 'media_creative', 'scientific_research',
            'security_defence', 'other'
        ]::text[]
    ),
    CONSTRAINT content_classifications_audience_len CHECK (
        cardinality(audience_relevance) >= 1 AND cardinality(audience_relevance) <= 4
    ),
    CONSTRAINT content_classifications_audience_values CHECK (
        audience_relevance <@ ARRAY[
            'practitioner', 'technical_leadership', 'enterprise_adoption', 'student'
        ]::text[]
    ),
    CONSTRAINT content_classifications_paper_kind CHECK (
        paper_kind IN (
            'method', 'empirical_study', 'benchmark_dataset', 'survey_review',
            'theory', 'position', 'negative_result', 'system_infrastructure'
        )
    ),
    CONSTRAINT content_classifications_geography_focus CHECK (
        geography_focus IN ('none', 'us', 'china', 'eu', 'india', 'other')
    ),
    CONSTRAINT content_classifications_domain_confidence CHECK (
        domain_confidence >= 0.0 AND domain_confidence <= 10.0
        AND (domain_confidence * 2) = trunc(domain_confidence * 2)
    )
);

CREATE INDEX IF NOT EXISTS ix_content_classifications_prompt
    ON research_radar.content_classifications (prompt_version);

ALTER TABLE research_radar.content_independence_assessments
    ADD COLUMN IF NOT EXISTS paper_kind TEXT;

ALTER TABLE research_radar.content_independence_assessments
    DROP CONSTRAINT IF EXISTS content_independence_assessments_paper_kind;

ALTER TABLE research_radar.content_independence_assessments
    ADD CONSTRAINT content_independence_assessments_paper_kind CHECK (
        paper_kind IS NULL OR paper_kind IN (
            'method', 'empirical_study', 'benchmark_dataset', 'survey_review',
            'theory', 'position', 'negative_result', 'system_infrastructure'
        )
    );

DO $$
BEGIN
    IF (SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'research_radar' AND table_name = 'content_classifications') <> 1 THEN
        RAISE EXCEPTION 'assertion failed: content_classifications missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'research_radar'
          AND table_name = 'content_independence_assessments'
          AND column_name = 'paper_kind'
    ) THEN
        RAISE EXCEPTION 'assertion failed: content_independence_assessments.paper_kind missing';
    END IF;
END $$;

COMMIT;
