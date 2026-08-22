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
echo "research_radar schema + watchlists installed."
