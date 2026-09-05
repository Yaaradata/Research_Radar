#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }
DATABASE_URL="${DATABASE_URL:-${PG_DSN:-}}"
: "${DATABASE_URL:?DATABASE_URL (or PG_DSN) is not set}"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/001_schema.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/003_org_watchlist_extend.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/002_seed_watchlists.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/004_openalex_and_source_seen.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/005_content_analysis_view.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/006_content_score_assessments.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/007_affiliation_gpt.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/008_affiliation_historical_status_fix.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/009_content_final_scores.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/010_scoring_v2.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/011_backfill_checkpoints.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/012_topic_hierarchy.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/013_seed_topic_hierarchy.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/014_scoring_v3.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/017_s3_manifest.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/018_relevance_version.sql
echo "research_radar schema + watchlists installed."
