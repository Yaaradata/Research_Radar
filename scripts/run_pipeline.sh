#!/usr/bin/env bash
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
exec .venv/bin/python -m research_radar.pipeline "$@"
