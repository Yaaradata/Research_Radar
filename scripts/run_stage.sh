#!/usr/bin/env bash
# Step-by-step Research Radar runner with logs.
set -euo pipefail
cd "$(dirname "$0")/.."
# Load .env WITHOUT clobbering variables already exported in the calling shell.
# This makes inline overrides work, e.g.:
#   AFFILIATION_GPT_ENABLED=true ./scripts/run_stage.sh affiliation-gpt --limit 20
if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key#export }"
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # Inline shell value wins over .env
    [[ -n "${!key+x}" ]] && continue
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
    export "$key=$val"
  done < .env
fi
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
  5) affiliation-gpt   GPT evidence-only affiliation resolver (PAID — needs --allow-paid)
  5b) openalex   DEPRECATED legacy Crossref/OpenAlex (inactive unless explicitly enabled)
  6) classify    Scoring v3 — domain/audience/paper_kind on full eligible pool (PAID — needs --allow-paid)
                 Optional --since/--until (ISO YYYY-MM-DD) bound ci.published_at; --until is inclusive.
  7) screen      Scoring v2 pass 1 — cheap screen gate (PAID — needs --allow-paid)
                 Optional --since/--until (ISO YYYY-MM-DD) bound ci.published_at; --until is inclusive.
  8) semantic-score   GPT full rubric pass 2 (PAID — needs --allow-paid)
                 Optional --since/--until (ISO YYYY-MM-DD) bound ci.published_at; --until is inclusive.
  9) independence     Categorical independence + paper_kind (PAID — needs --allow-paid)
                 Optional --since/--until (ISO YYYY-MM-DD) bound ci.published_at; --until is inclusive.
 10) semantic-compare Compare deterministic vs GPT semantic ranks (explicit)
 11) final-score      Recompute ranking from stored assessments + org/person boosts (FREE)
 12) report           Markdown Top N from content_final_scores (FREE)
 13) show             Print top candidates (deterministic CANDIDATE pool)
 12) arxiv-backfill   arXiv OAI-PMH metadata backfill (FREE, explicit, not in `all`)
                 --from/--until (ISO dates) required. --dry-run projects volume with
                 zero writes. --force re-harvests COMPLETE checkpoint windows.
 13) topics           Hierarchical topic tags + key-claim extraction (PAID — needs
                 --allow-paid). Annotation only, NOT scoring: runs on every
                 RELEVANT-or-later paper, never changes content_items.status,
                 not in `all`. --dry-run projects cost with zero calls/writes.
                 Optional --since/--until (ISO YYYY-MM-DD) bound ci.published_at;
                 --until is inclusive.
 14) corpus-search     Free, pure-SQL query over topics/claims (FREE, explicit,
                 not in `all`). --tag/--subdomain/--application/--domain/--since
                 filter (AND); --list-topics audits the tag vocabulary;
                 --claims-for lists claims for a metric; --json/--out for output.

`all` pipeline order (intentional, free/unattended — safe for cron):
  ingest → relevance → enrich → entities → score
  No paid stage is ever run by `all`. OpenAlex/Crossref are NOT called.
  affiliation-gpt and semantic-score are explicit and require --allow-paid.
  final-score / report are free but explicit (not yet in `all`).

Examples:
  ./scripts/run_stage.sh ingest --limit 50
  ./scripts/run_stage.sh relevance
  ./scripts/run_stage.sh enrich --limit 20
  ./scripts/run_stage.sh entities
  ./scripts/run_stage.sh entities-reprocess
  ./scripts/run_stage.sh repair-timestamps
  ./scripts/run_stage.sh affiliation-gpt --dry-run
  AFFILIATION_GPT_ENABLED=true ./scripts/run_stage.sh affiliation-gpt --limit 50 --allow-paid
  ./scripts/run_stage.sh score
  ./scripts/run_stage.sh semantic-score --sample 100 --dry-run
  ./scripts/run_stage.sh screen --since 2026-09-04 --until 2026-09-04 --dry-run
  ./scripts/run_stage.sh classify --since 2026-09-04 --until 2026-09-05 --dry-run
  SEMANTIC_SCORING_ENABLED=true ./scripts/run_stage.sh semantic-score --sample 100 --allow-paid
  ./scripts/run_stage.sh semantic-compare
  ./scripts/run_stage.sh final-score --diagnose
  ./scripts/run_stage.sh final-score --profile radar-v1
  ./scripts/run_stage.sh report --top 20 --out reports/research-radar-top20.md
  ./scripts/run_stage.sh report --top 20 --since-days 30 --out reports/top20-30d.md
  ./scripts/run_stage.sh show --top 10
  ./scripts/run_stage.sh arxiv-backfill --from 2026-01-01 --until 2026-01-07 --dry-run
  ./scripts/run_stage.sh arxiv-backfill --from 2026-01-01 --until 2026-08-31
  ./scripts/run_stage.sh topics --dry-run
  ./scripts/run_stage.sh topics --limit 100 --allow-paid
  ./scripts/run_stage.sh topics --full --allow-paid
  ./scripts/run_stage.sh corpus-search --tag "ai text detection" --top 20
  ./scripts/run_stage.sh corpus-search --subdomain "Text Classification" --application education
  ./scripts/run_stage.sh corpus-search --domain "Natural Language Processing" --with-claims --since 2026-01-01
  ./scripts/run_stage.sh corpus-search --list-topics --min-usage 5
  ./scripts/run_stage.sh corpus-search --claims-for "true negative rate"

Logs are written to logs/ and also printed to the terminal.
EOF
  exit 1
fi

echo "Logging to $LOG"
set -o pipefail
"$PY" -m research_radar.pipeline --stage "$stage" "$@" 2>&1 | tee -a "$LOG"
echo "Done. Log file: $LOG"
