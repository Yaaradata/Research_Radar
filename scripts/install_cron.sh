#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/opt/research-radar}"
SCHEDULE="${SCHEDULE:-0 */6 * * *}"
LOG_FILE="${LOG_FILE:-/var/log/research-radar.log}"
LINE="$SCHEDULE cd $PROJECT_DIR && $PROJECT_DIR/scripts/run_pipeline.sh --top 10 >> $LOG_FILE 2>&1"
( crontab -l 2>/dev/null | grep -v 'research_radar.pipeline' | grep -v 'run_pipeline.sh' || true; echo "$LINE" ) | crontab -
echo "$LINE"
