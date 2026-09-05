# TheNeural AI Research Radar v0.1

This package implements the MVP architecture from the supplied project brief using your existing **EC2 + Amazon RDS/Postgres** and existing **Inoreader** integration.

## Included

- `sql/001_schema.sql` — creates the `research_radar` schema, canonical corpus, people/org/topic/score/opportunity/provenance tables, pipeline observability, candidate view, and analysis view.
- `sql/002_seed_watchlists.sql` — seeds **30** editable organisations plus initial AI topics.
- `sql/005_content_analysis_view.sql` — `v_content_analysis` for corpus validation.
- `src/research_radar/pipeline.py` — staged pipeline: Inoreader ingestion → relevance → arXiv enrichment → local entity resolution (`all` path is free/unattended and stops there; independence + semantic-score + final-score are paid/manual, see below).
- `src/research_radar/affiliation_gpt.py` — evidence-only GPT affiliation resolver (paid; requires `--allow-paid`).
- `src/research_radar/independence.py` — scoring v2 Call B: categorical independence classifier (`independent` / `self_evaluation` / `unclear` / `not_applicable`), title+abstract+affiliation_text only (paid; requires `--allow-paid`).
- `src/research_radar/semantic_scoring.py` — scoring v2 Call A: OpenRouter LLM research-quality assessments, title+abstract+categories ONLY, never affiliation (paid; requires `--allow-paid`; assessment-only). v1 (`research-semantic-v1`) code and data are kept, unwired, for comparison.
- `src/research_radar/affiliation_external.py` — legacy Crossref DOI + OpenAlex DOI singleton (inactive by default).
- `src/research_radar/query.py` — query candidates by score/org/topic.
- `scripts/install_ec2.sh` — installs Python/Postgres client and project venv.
- `scripts/setup_db.sh` — creates tables + watchlists in the **existing RDS**, not a new instance.
- `scripts/run_stage.sh` — run one pipeline stage with logging.
- `scripts/run_pipeline.sh` — end-to-end run (`all` stage).
- `scripts/install_cron.sh` — optional EC2 cron schedule (default every 6 hours).
- `scripts/golden_test.py` — acceptance checks including `arXiv:2608.02345`.

## Pipeline stages

Run in this order:

```text
ingest → relevance → enrich → entities → classify → screen → semantic-score → independence → final-score

`all` (free, cron path) runs only: ingest → relevance → enrich → entities
Paid stages, run explicitly and never on cron:
  classify        --allow-paid   (scoring v3 — domain/audience/paper_kind on full eligible pool)
  affiliation-gpt --allow-paid
  screen          --allow-paid   (pass 1 — cheap, non-reasoning, gates who reaches pass 2)
  semantic-score  --allow-paid   (pass 2 — full rubric, gated subset only)
  independence    --allow-paid   (pass-2 papers only)
```

| Stage | Purpose |
|-------|---------|
| `ingest` | Inoreader → `content_items` (duplicate ingest preserves workflow status) |
| `relevance` | Deterministic AI relevance filter (`RELEVANT` / `REJECTED`) |
| `enrich` | arXiv Atom + HTML (metadata, emails, affiliations) |
| `entities` | Local org/people resolution from paper evidence |
| `affiliation-gpt` | Evidence-only GPT affiliation resolver. **Paid** — requires `--allow-paid`. GPT is the resolver, never the evidence source: organisation names must ground back to original paper/email evidence, and watchlist matching is deterministic. |
| `classify` | Scoring v3 — `research-classify-v1`, Haiku-tier, batched 15, title/abstract/categories only. Runs on **every** eligible paper before the gate. **Paid** — requires `--allow-paid`. |
| `screen` | Scoring v2 pass 1 — cheap, non-reasoning `SCREEN_MODEL`, batched 15, 4 dimensions, no prose. Every eligible paper is screened. **Paid** — requires `--allow-paid`. |
| `semantic-score` | Scoring v2 pass 2 — full rubric, batched 5 randomly, title/abstract/categories **only** (never affiliation), on ONLY the top `GATE_PERCENTILE`% of screen scores (`ai_relevance <= 3` excluded outright). **Paid** — requires `--allow-paid`. Assessment-only; does not change `status`. |
| `independence` | Scoring v2 Call B — categorical independence classification, batched 20, title+abstract+affiliation_text, runs ONLY on pass-2 (`scoring_tier='full'`) papers. **Paid** — requires `--allow-paid`. |
| `scoring-cost` | Free, zero API calls — prints the Pass 1 / Pass 2 / Independence / TOTAL cost projection for a full run over the currently eligible pool. Run this before `--allow-paid`. |
| `openalex` | **DEPRECATED / inactive.** Legacy Crossref + OpenAlex DOI path, retained for historical provenance. Off unless `OPENALEX_ENABLED=true`. |
| `score` | Deterministic component scores + `CANDIDATE` label. Kept in the codebase but unwired from `all` (scoring v2 replaces it) — still runnable manually via `--stage score`. |
| `final-score` | Combines stored quality + independence + verified org/person signals into `content_final_scores` (`radar-v1` or `radar-v2` profile). Only `scoring_tier='full'` (or untiered v1) rows are scored — screen-tier rows never get a `final_score` row. Free — reads only stored data. |
| `report` | Markdown Top-N report; `--rank-by research\|newsletter` (radar-v2 only); prints how many papers were screened vs fully scored so the reader knows the pool the Top N was drawn from. |
| `show` | Print top N from the **candidate pool** (legacy deterministic-score pool) |

`CANDIDATE` is a candidate pool, not automatically the final Top N list. `show --top 20` ranks the top 20 from that pool by intrinsic score; the LLM-scored ranking is `--stage report`.

The `all` stage runs ingest → relevance → enrich → entities only — it no longer runs deterministic `score`, and never runs a paid stage.

**Tiering config** (env-driven, never hardcoded):
- `GATE_PERCENTILE` (default `15`) — set from a measured recall test (`scripts/validate_scoring.py --test tier-recall`), never from a cost target.
- `SCREEN_MODEL` (default `anthropic/claude-haiku-4.5`) — the cheap, non-reasoning pass-1 model.
- `SCORING_CONCURRENCY` (default `4`) — shared worker-pool size for screen/pass-2/independence; each worker uses its own DB connection, and a 429 permanently shrinks the pool's effective concurrency for the rest of that run. Does not apply to `enrich`, which keeps its own hardcoded ~3s arXiv throttle.

## Forward vs historical ingestion

**FORWARD MODE** (current 7-day pipeline)

- Source: Inoreader folder with overlapping recent window
- Schedule: small number of runs per day (cron default: every 6 hours)
- Idempotent: duplicate ingest does **not** reset `CANDIDATE` / `SCORED` / etc.

**HISTORICAL BACKFILL MODE** (not implemented in this release)

- Direct arXiv date-range ingestion
- Extend corpus to Aug 1 first, then earlier months
- Inoreader lookback is only ~one month — backfill will not rely on it

## Scoring (current — deterministic, not LLM)

Scoring is a **hand-written weighted heuristic**, not an LLM. Component dimensions include technical significance, practical applicability, professional value, student learning value, explainability, and watchlist org/person boosts.

- **Novelty** is a fixed proxy (`novelty_method = fixed_proxy`), not semantic measurement.
- **Industry relevance** is **not yet semantically scored** — the column exists but deterministic scoring does not populate it.
- Full provenance is stored in `content_scores.scoring_reason` (`method`, `version`, `matched_rules`, weights).

**LLM semantic scoring v2** is the current design: three separate OpenRouter calls per paper,
never merged. Quality is now two tiers — pass 1 (`screen`, `SCREEN_MODEL`, no reasoning, batched
15, 4 dimensions, no prose, prompt version `research-screen-v2`) gates who reaches pass 2
(`semantic-score`, `openai/gpt-5.6-sol` with reasoning, full rubric, batched 5, prompt version
`research-semantic-v2`) — only the top `GATE_PERCENTILE`% of screen scores proceed. Independence
(`independence`, title+abstract+affiliation_text, prompt version `independence-v1`, batched 20)
then runs only on that same pass-2 pool — classifying a paper that screened out is wasted spend.
`final-score` combines quality × evidence_factor × independence_factor with verified
organisation/person boosts into `research_score` and `newsletter_score`, and computes ONLY for
`scoring_tier='full'` papers — two models have two scales and are never mixed into one ranking.
v1 assessments (`research-semantic-v1`) are left in place under their own prompt_version for
comparison via `scripts/validate_scoring.py`; they are not deleted or migrated. See
`sql/010_scoring_v2.sql` and `scripts/validate_scoring.py` for the full design and validation
harness (including `--test tier-recall`, which is required before trusting `GATE_PERCENTILE`).

## OpenAlex external affiliation

OpenAlex is an **affiliation fallback only**, not the scoring engine. Order:

1. Explicit affiliation text / email domain (entities stage)
2. Crossref DOI lookup (free)
3. OpenAlex singleton DOI lookup (free; budget-aware 429 handling)

`OPENALEX_TITLE_SEARCH_ENABLED=false` by default. Papers without a DOI stay deferred (`PENDING`) and are excluded from the automatic openalex stage until title search or enrich improves DOI coverage.

OpenAlex/API failure never blocks scoring — organisation is a boost, not a gate.

## 1. AWS prerequisites

Your RDS security group should allow PostgreSQL/5432 **from the EC2 security group**, not from the public internet.

The RDS user in `DATABASE_URL` must be able to create/use the `research_radar` schema.

Example:

```env
DATABASE_URL=postgresql://research_radar_user:password@your-rds-endpoint:5432/neural?sslmode=require
```

## 2. Copy to EC2 and install

```bash
unzip research-radar-v0.1.zip
cd research-radar
chmod +x scripts/*.sh
./scripts/install_ec2.sh
```

Default location is `/opt/research-radar`.

## 3. Configure

```bash
cd /opt/research-radar
nano .env
```

At minimum:

```env
DATABASE_URL=postgresql://...
INOREADER_ACCESS_TOKEN=...
INOREADER_CLIENT_ID=...
INOREADER_CLIENT_SECRET=...
INOREADER_REFRESH_TOKEN=...
INOREADER_STREAM=user/-/label/10 RESEARCH - T3 Firehose
INOREADER_LOOKBACK_DAYS=7
CROSSREF_ENABLED=true
OPENALEX_ENABLED=true
OPENALEX_API_KEY=...
OPENALEX_TITLE_SEARCH_ENABLED=false
MIN_INTRINSIC_CANDIDATE_SCORE=5.5
```

For a smoke test without calling Inoreader:

```env
INOREADER_FIXTURE=/opt/research-radar/config/inoreader_sample.json
```

## 4. Create all required RDS tables

```bash
cd /opt/research-radar
./scripts/setup_db.sh
```

Key tables/views:

```text
research_radar.content_items
research_radar.paper_metadata
research_radar.content_scores
research_radar.v_candidates
research_radar.v_content_analysis
```

## 5. Run step-by-step (recommended for first runs)

Ingest defaults to the last **7 days** (`INOREADER_LOOKBACK_DAYS=7`). Items store `published_at` (publisher date) and `source_seen_at` (Inoreader first-seen) independently.

```bash
cd /home/ubuntu/Research_Radar1
source .venv/bin/activate
mkdir -p logs

./scripts/setup_db.sh   # one-time / after migrations

./scripts/run_stage.sh ingest
./scripts/run_stage.sh relevance
./scripts/run_stage.sh enrich
./scripts/run_stage.sh entities
./scripts/run_stage.sh openalex
./scripts/run_stage.sh score
./scripts/run_stage.sh show --top 20
```

Each stage writes terminal output and a file under `logs/`.

Useful SQL checks:

```bash
source .env
psql "$DATABASE_URL" -c "SELECT status, COUNT(*) FROM research_radar.content_items GROUP BY status ORDER BY status;"
psql "$DATABASE_URL" -c "SELECT * FROM research_radar.v_content_analysis ORDER BY intrinsic_candidate_score DESC NULLS LAST LIMIT 20;"
```

## 5b. Or run end-to-end in one command

```bash
./scripts/run_pipeline.sh --top 10
# same as:
./scripts/run_stage.sh all --top 10
```

Per-item failures are logged in `research_radar.processing_events` and do not terminate the batch.

## 6. Query candidates

```bash
export PYTHONPATH=/opt/research-radar/src
.venv/bin/python -m research_radar.query --min-score 7
.venv/bin/python -m research_radar.query --org Amazon
.venv/bin/python -m research_radar.query --topic agents
```

## 7. Re-run safely

arXiv versions canonicalise to one identity:

```text
2608.02345v1 / 2608.02345v2 → https://arxiv.org/abs/2608.02345
```

Duplicate ingest in overlapping lookback windows preserves pipeline status (`CANDIDATE` stays `CANDIDATE`).

## 8. Golden acceptance test

```bash
export PYTHONPATH=/opt/research-radar/src
.venv/bin/python scripts/golden_test.py
```

## 9. Optional cron

```bash
./scripts/install_cron.sh
```

Default: **every 6 hours** (`0 */6 * * *`). Override with:

```bash
SCHEDULE='0 */4 * * *' ./scripts/install_cron.sh
```

## Important evidence rule

Paper-specific relationships only:

```text
relationship_type = paper_author_affiliation
current_affiliation = false
```

Current employer must be a separate relationship with `current_affiliation = true` and its own evidence.

## Explicitly deferred

No UI, agents, embeddings/vector search, LinkedIn scraping, Kafka, Airflow, historical backfill, or LLM scoring in this release.
