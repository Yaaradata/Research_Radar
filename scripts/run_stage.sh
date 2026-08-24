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
  4) entities    Local org/people resolution (ENTITY_RESOLVED)
  4b) entities-reprocess  Re-run local org matching from stored enrichment (no API, preserves status)
  4c) repair-timestamps   Fix published_at/source_seen_at from stored raw_metadata (no Inoreader)
  5) affiliation-gpt   GPT evidence-only affiliation resolver (replaces OpenAlex/Crossref)
  5b) openalex   DEPRECATED legacy Crossref/OpenAlex (inactive unless explicitly enabled)
  6) score       Deterministic component scores + opportunities (SCORED / CANDIDATE)
                 NOTE: `all` ends here. Semantic scoring is a separate paid stage.
  7) semantic-score   GPT assessment-only experiment (explicit; NOT part of `all`)
  8) semantic-compare Compare deterministic vs GPT semantic ranks (explicit)
  9) show        Print top candidates

`all` pipeline order (intentional):
  ingest → relevance → enrich → entities → affiliation-gpt → score
  OpenAlex/Crossref are NOT called. semantic-score is NOT auto-run.

Examples:
  ./scripts/run_stage.sh ingest --limit 50
  ./scripts/run_stage.sh relevance
  ./scripts/run_stage.sh enrich --limit 20
  ./scripts/run_stage.sh entities
  ./scripts/run_stage.sh entities-reprocess
  ./scripts/run_stage.sh repair-timestamps
  ./scripts/run_stage.sh affiliation-gpt --dry-run
  AFFILIATION_GPT_ENABLED=true ./scripts/run_stage.sh affiliation-gpt --limit 50
  ./scripts/run_stage.sh score
  ./scripts/run_stage.sh semantic-score --sample 100 --dry-run
  SEMANTIC_SCORING_ENABLED=true ./scripts/run_stage.sh semantic-score --sample 100
  ./scripts/run_stage.sh semantic-compare
  ./scripts/run_stage.sh show --top 10

Logs are written to logs/ and also printed to the terminal.
EOF
  exit 1
fi

echo "Logging to $LOG"
set -o pipefail
"$PY" -m research_radar.pipeline --stage "$stage" "$@" 2>&1 | tee -a "$LOG"
echo "Done. Log file: $LOG"
