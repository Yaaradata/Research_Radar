BEGIN;

CREATE SCHEMA IF NOT EXISTS research_radar;

CREATE TABLE IF NOT EXISTS research_radar.content_items (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('arxiv','research_paper','technical_blog','company_research','news','other')),
    source TEXT NOT NULL,
    source_external_id TEXT,
    source_feed TEXT,
    canonical_url TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    authors_raw JSONB NOT NULL DEFAULT '[]'::jsonb,
    categories_raw JSONB NOT NULL DEFAULT '[]'::jsonb,
    inoreader_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    published_at TIMESTAMPTZ,
    source_seen_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'INGESTED' CHECK (status IN ('INGESTED','RELEVANCE_CHECKED','RELEVANT','ENRICHED','ENTITY_RESOLVED','SCORED','CANDIDATE','REVIEW_REQUIRED','REJECTED','ERROR')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_canonical_url ON research_radar.content_items(canonical_url);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_source_external_id ON research_radar.content_items(source, source_external_id) WHERE source_external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_content_items_status ON research_radar.content_items(status);
CREATE INDEX IF NOT EXISTS ix_content_items_published ON research_radar.content_items(published_at DESC);

CREATE TABLE IF NOT EXISTS research_radar.paper_metadata (
    content_id BIGINT PRIMARY KEY REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    arxiv_id TEXT,
    arxiv_version INTEGER,
    doi TEXT,
    abstract TEXT,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    authors_raw JSONB NOT NULL DEFAULT '[]'::jsonb,
    submission_date TIMESTAMPTZ,
    latest_revision_date TIMESTAMPTZ,
    journal_reference TEXT,
    paper_url TEXT,
    html_url TEXT,
    pdf_url TEXT,
    affiliation_text JSONB NOT NULL DEFAULT '[]'::jsonb,
    extracted_emails JSONB NOT NULL DEFAULT '[]'::jsonb,
    enrichment_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    openalex_status TEXT CHECK (
        openalex_status IS NULL
        OR openalex_status IN ('NOT_NEEDED','PENDING','MATCHED','NO_MATCH','RATE_LIMITED','ERROR')
    ),
    openalex_work_id TEXT,
    openalex_checked_at TIMESTAMPTZ,
    openalex_attempts INTEGER NOT NULL DEFAULT 0,
    openalex_last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_metadata_arxiv ON research_radar.paper_metadata(arxiv_id) WHERE arxiv_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS research_radar.organisations (
    organisation_id BIGSERIAL PRIMARY KEY,
    org_key TEXT,
    canonical_name TEXT NOT NULL UNIQUE,
    organisation_type TEXT NOT NULL DEFAULT 'enterprise',
    aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    domains TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    priority INTEGER NOT NULL DEFAULT 3 CHECK (priority BETWEEN 0 AND 10),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    watchlist_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    evidence_sources TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    rationale TEXT,
    recent_highlight TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_org_aliases ON research_radar.organisations USING GIN(aliases);
CREATE INDEX IF NOT EXISTS ix_org_domains ON research_radar.organisations USING GIN(domains);
CREATE UNIQUE INDEX IF NOT EXISTS ux_organisations_org_key ON research_radar.organisations(org_key) WHERE org_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS research_radar.people (
    person_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    openalex_id TEXT,
    orcid TEXT,
    known_organisations TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    priority INTEGER NOT NULL DEFAULT 3 CHECK (priority BETWEEN 0 AND 10),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_people_name_openalex ON research_radar.people(canonical_name, COALESCE(openalex_id,''));

CREATE TABLE IF NOT EXISTS research_radar.content_organisations (
    id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    organisation_id BIGINT NOT NULL REFERENCES research_radar.organisations(organisation_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    evidence_text TEXT,
    evidence_url TEXT,
    confidence NUMERIC(4,3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    current_affiliation BOOLEAN NOT NULL DEFAULT FALSE,
    raw_evidence JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_org_evidence ON research_radar.content_organisations(content_id, organisation_id, relationship_type, evidence_type, COALESCE(evidence_text,''));

CREATE TABLE IF NOT EXISTS research_radar.content_people (
    id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    person_id BIGINT NOT NULL REFERENCES research_radar.people(person_id) ON DELETE CASCADE,
    author_position INTEGER,
    is_notable BOOLEAN NOT NULL DEFAULT TRUE,
    match_confidence NUMERIC(4,3) NOT NULL CHECK (match_confidence BETWEEN 0 AND 1),
    evidence_type TEXT,
    evidence_text TEXT,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_people ON research_radar.content_people(content_id, person_id);

CREATE TABLE IF NOT EXISTS research_radar.topics (
    topic_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS research_radar.content_topics (
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    topic_id BIGINT NOT NULL REFERENCES research_radar.topics(topic_id) ON DELETE CASCADE,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    confidence NUMERIC(4,3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reason TEXT,
    PRIMARY KEY (content_id, topic_id)
);

CREATE TABLE IF NOT EXISTS research_radar.content_scores (
    content_id BIGINT PRIMARY KEY REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    ai_relevance NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (ai_relevance BETWEEN 0 AND 10),
    technical_significance NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (technical_significance BETWEEN 0 AND 10),
    novelty NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (novelty BETWEEN 0 AND 10),
    notable_person_signal NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (notable_person_signal BETWEEN 0 AND 10),
    notable_org_signal NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (notable_org_signal BETWEEN 0 AND 10),
    professional_value NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (professional_value BETWEEN 0 AND 10),
    student_learning_value NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (student_learning_value BETWEEN 0 AND 10),
    practical_applicability NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (practical_applicability BETWEEN 0 AND 10),
    explainability NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (explainability BETWEEN 0 AND 10),
    industry_relevance JSONB NOT NULL DEFAULT '{}'::jsonb,
    intrinsic_candidate_score NUMERIC(4,2) NOT NULL DEFAULT 0 CHECK (intrinsic_candidate_score BETWEEN 0 AND 10),
    freshness_score NUMERIC(4,2),
    current_interest_score NUMERIC(4,2),
    channel_fit JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_version TEXT NOT NULL DEFAULT 'deterministic-v0.1',
    scoring_reason JSONB NOT NULL DEFAULT '{}'::jsonb,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_radar.content_opportunities (
    id BIGSERIAL PRIMARY KEY,
    content_id BIGINT NOT NULL REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    opportunity_type TEXT NOT NULL CHECK (opportunity_type IN ('raw_paper','notable_research','weekend_read','explainer_candidate','professional_learning','student_learning','technical_deep_dive','industry_brief','ai_fluency_material','notable_news','social_content_source')),
    audience TEXT,
    confidence NUMERIC(4,3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_opportunity ON research_radar.content_opportunities(content_id, opportunity_type, COALESCE(audience,''));

CREATE TABLE IF NOT EXISTS research_radar.pipeline_runs (
    run_id UUID PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    items_received INTEGER NOT NULL DEFAULT 0,
    items_new INTEGER NOT NULL DEFAULT 0,
    items_duplicate INTEGER NOT NULL DEFAULT 0,
    items_relevant INTEGER NOT NULL DEFAULT 0,
    items_enriched INTEGER NOT NULL DEFAULT 0,
    orgs_resolved INTEGER NOT NULL DEFAULT 0,
    people_resolved INTEGER NOT NULL DEFAULT 0,
    items_scored INTEGER NOT NULL DEFAULT 0,
    candidates_created INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    http_calls INTEGER NOT NULL DEFAULT 0,
    llm_calls INTEGER NOT NULL DEFAULT 0,
    estimated_llm_cost NUMERIC(12,6) NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    notes JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS research_radar.processing_events (
    event_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES research_radar.pipeline_runs(run_id) ON DELETE SET NULL,
    content_id BIGINT REFERENCES research_radar.content_items(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_events_run ON research_radar.processing_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS ix_events_content ON research_radar.processing_events(content_id, created_at);

CREATE OR REPLACE VIEW research_radar.v_candidates AS
SELECT
    ci.id AS content_id,
    ci.title,
    ci.canonical_url,
    ci.source_type,
    ci.published_at,
    ci.status,
    pm.arxiv_id,
    cs.ai_relevance,
    cs.technical_significance,
    cs.practical_applicability,
    cs.professional_value,
    cs.student_learning_value,
    cs.notable_org_signal,
    cs.notable_person_signal,
    cs.intrinsic_candidate_score,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'organisation',o.canonical_name,
            'relationship_type',co.relationship_type,
            'evidence_type',co.evidence_type,
            'evidence_text',co.evidence_text,
            'confidence',co.confidence
        ) ORDER BY o.priority DESC,o.canonical_name)
        FROM research_radar.content_organisations co
        JOIN research_radar.organisations o ON o.organisation_id=co.organisation_id
        WHERE co.content_id=ci.id
    ),'[]'::jsonb) AS organisations,
    COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'opportunity_type',opp.opportunity_type,
            'audience',opp.audience,
            'confidence',opp.confidence,
            'reason',opp.reason
        ))
        FROM research_radar.content_opportunities opp
        WHERE opp.content_id=ci.id
    ),'[]'::jsonb) AS opportunities
FROM research_radar.content_items ci
LEFT JOIN research_radar.paper_metadata pm ON pm.content_id=ci.id
LEFT JOIN research_radar.content_scores cs ON cs.content_id=ci.id
WHERE ci.status='CANDIDATE';

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

COMMIT;
