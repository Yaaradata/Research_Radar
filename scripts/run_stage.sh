#!/usr/bin/env bash
# Step-by-step Research Radar runner with logs.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="logs/${STAMP}-stage.log"
PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

stage="${1:-}"
shift || true

if [[ -z "$stage" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run_stage.sh <stage> [extra args]

Stages (run in this order):
  1) ingest      Pull Inoreader → content_items (status=INGESTED)
  2) relevance   AI relevance filter (RELEVANT / REJECTED)
  3) enrich      arXiv metadata + emails/affiliations (ENRICHED)
  4) entities    Local org/people resolution (ENTITY_RESOLVED); OpenAlex deferred
  5) openalex    Crossref DOI + OpenAlex DOI (budget-aware; no auto title search)
  6) score       Component scores + opportunities (SCORED / CANDIDATE)
  7) show        Print top candidates

Examples:
  ./scripts/run_stage.sh ingest --limit 50
  ./scripts/run_stage.sh relevance
  ./scripts/run_stage.sh enrich --limit 20
  ./scripts/run_stage.sh entities
  ./scripts/run_stage.sh openalex --limit 100
  ./scripts/run_stage.sh score
  ./scripts/run_stage.sh show --top 10

Logs are written to logs/ and also printed to the terminal.
EOF
  exit 1
fi

echo "Logging to $LOG"
set -o pipefail
"$PY" -m research_radar.pipeline --stage "$stage" "$@" 2>&1 | tee -a "$LOG"
echo "Done. Log file: $LOG"
