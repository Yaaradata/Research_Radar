# TheNeural AI Research Radar v0.1

This package implements the MVP architecture from the supplied project brief using your existing **EC2 + Amazon RDS/Postgres** and existing **Inoreader** integration.

## Included

- `sql/001_schema.sql` — creates the `research_radar` schema, canonical corpus, people/org/topic/score/opportunity/provenance tables, pipeline observability, and candidate view.
- `sql/002_seed_watchlists.sql` — seeds 20 editable organisations plus initial AI topics.
- `src/research_radar/pipeline.py` — Inoreader ingestion → normalisation → deterministic relevance → arXiv enrichment → evidence-backed org/person resolution → deterministic scoring → candidate corpus.
- `src/research_radar/query.py` — query candidates by score/org/topic.
- `scripts/install_ec2.sh` — installs Python/Postgres client and project venv.
- `scripts/setup_db.sh` — creates tables + watchlists in the **existing RDS**, not a new instance.
- `scripts/run_pipeline.sh` — one pipeline run.
- `scripts/install_cron.sh` — optional 15-minute EC2 cron schedule.
- `scripts/golden_test.py` — acceptance checks including `arXiv:2608.02345`.

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

At minimum, put keys in this project's own `.env` (already created under `Research_Radar1/.env`):

```env
DATABASE_URL=postgresql://...
INOREADER_ACCESS_TOKEN=...
INOREADER_CLIENT_ID=...
INOREADER_CLIENT_SECRET=...
INOREADER_REFRESH_TOKEN=...
```

The adapter defaults to Inoreader's Google Reader-compatible reading-list stream. If your existing integration uses another stream, set `INOREADER_STREAM`.

For a smoke test without calling Inoreader:

```env
INOREADER_FIXTURE=/opt/research-radar/config/inoreader_sample.json
```

## 4. Create all required RDS tables

```bash
cd /opt/research-radar
./scripts/setup_db.sh
```

Tables created:

```text
research_radar.content_items
research_radar.paper_metadata
research_radar.people
research_radar.organisations
research_radar.content_people
research_radar.content_organisations
research_radar.topics
research_radar.content_topics
research_radar.content_scores
research_radar.content_opportunities
research_radar.pipeline_runs
research_radar.processing_events
research_radar.v_candidates
```

## 5. Run step-by-step (recommended for first runs)

Ingest defaults to the last **7 days** from folder `10 RESEARCH - T3 Firehose` (`INOREADER_LOOKBACK_DAYS=7` in `.env`). Each item stores `published_at` for date-based analysis.

Use separate stages so you can inspect progress and logs after each step:

```bash
cd /home/ubuntu/Research_Radar1
source .venv/bin/activate   # if not already active
mkdir -p logs

# 0) one-time schema (only once)
./scripts/setup_db.sh

# 1) Inoreader → content_items
./scripts/run_stage.sh ingest --limit 50

# 2) AI relevance filter
./scripts/run_stage.sh relevance

# 3) arXiv enrichment (metadata + emails/affiliations)
./scripts/run_stage.sh enrich --limit 20

# 4) organisation / people resolution
./scripts/run_stage.sh entities

# 5) scoring + candidate labels
./scripts/run_stage.sh score

# 6) print top candidates
./scripts/run_stage.sh show --top 10
```

Each stage writes terminal output **and** a file under `logs/`.

Useful SQL checks between stages:

```bash
source .env
psql "$DATABASE_URL" -c "SELECT status, COUNT(*) FROM research_radar.content_items GROUP BY status ORDER BY status;"
psql "$DATABASE_URL" -c "SELECT run_id, started_at, status, notes, items_received, items_relevant, items_enriched, candidates_created, errors FROM research_radar.pipeline_runs ORDER BY started_at DESC LIMIT 10;"
```

## 5b. Or run end-to-end in one command

```bash
./scripts/run_pipeline.sh --top 10
# same as:
./scripts/run_stage.sh all --top 10
```

Flow:

```text
Inoreader
  → canonical ingestion
  → URL normalisation / arXiv canonical ID
  → deterministic high-recall AI relevance
  → arXiv Atom + HTML enrichment
  → email / affiliation evidence extraction
  → database watchlist resolution
  → component scoring
  → opportunity labels
  → CANDIDATE corpus
```

Per-item failures are logged in `research_radar.processing_events` and do not terminate the batch.

## 6. Query candidates

```bash
export PYTHONPATH=/opt/research-radar/src
.venv/bin/python -m research_radar.query --min-score 7
.venv/bin/python -m research_radar.query --org Amazon
.venv/bin/python -m research_radar.query --topic agents
```

Or SQL:

```sql
SELECT *
FROM research_radar.v_candidates
ORDER BY intrinsic_candidate_score DESC
LIMIT 20;
```

## 7. Re-run safely

The pipeline canonicalises:

```text
2608.02345v1
2608.02345v2
```

as one content identity:

```text
https://arxiv.org/abs/2608.02345
```

The version is retained separately in `paper_metadata.arxiv_version`, so reruns update rather than duplicate.

## 8. Golden acceptance test

```bash
export PYTHONPATH=/opt/research-radar/src
.venv/bin/python scripts/golden_test.py
```

It checks that Amazon is found from **paper evidence** such as an `@amazon.com` author email, and that unknown organisations do not prevent a high-value paper from surviving.

## 9. Optional cron

```bash
./scripts/install_cron.sh
```

Default: every 15 minutes. Override with:

```bash
SCHEDULE='*/30 * * * *' ./scripts/install_cron.sh
```

## Important evidence rule

The current code creates paper-specific relationships only:

```text
relationship_type = paper_author_affiliation
current_affiliation = false
```

If a later resolver finds an author's present employer, store that as a separate relationship with `current_affiliation = true`. Never turn current employment into paper affiliation without paper-specific evidence.

## Explicitly deferred for v0.1

No UI, agents, sophisticated clustering, embeddings/vector search, LinkedIn scraping, automatic publishing, separate RDS, Kubernetes, Kafka, Airflow, or prompt orchestration.
