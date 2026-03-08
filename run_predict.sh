#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env.local"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "Copy .env.example to .env.local and fill in values first."
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -d "$ROOT_DIR/.venv" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

python "$ROOT_DIR/predict.py" "$@"
