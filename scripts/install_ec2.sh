#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/opt/research-radar}"
SOURCE_DIR="$(pwd)"
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y python3 python3-pip postgresql15
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip postgresql-client
else
  echo "Install Python 3.11+, pip/venv and psql manually."; exit 1
fi
sudo mkdir -p "$PROJECT_DIR"; sudo chown "$USER":"$USER" "$PROJECT_DIR"
if [[ "$SOURCE_DIR" != "$PROJECT_DIR" ]]; then cp -R . "$PROJECT_DIR/"; fi
cd "$PROJECT_DIR"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[[ -f .env ]] || cp .env.example .env
chmod +x scripts/*.sh
printf '\nInstalled in %s\nEdit .env, then run:\n  cd %s\n  ./scripts/setup_db.sh\n  ./scripts/run_pipeline.sh --top 10\n' "$PROJECT_DIR" "$PROJECT_DIR"
