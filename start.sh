#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/app"
VENV_ACTIVATE="$SCRIPT_DIR/stockmarketanalysys/bin/activate"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3:8b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed or not on PATH"
  exit 1
fi

if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  nohup ollama serve >"$SCRIPT_DIR/ollama.log" 2>&1 &

  for _ in {1..30}; do
    if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$OLLAMA_MODEL"; then
  echo "Pulling Ollama model: $OLLAMA_MODEL"
  ollama pull "$OLLAMA_MODEL"
fi

export OLLAMA_BASE_URL
export OLLAMA_MODEL

echo "Starting Stock Market Analysis app..."
cd "$APP_DIR"
exec uvicorn api:app --reload --host 0.0.0.0 --port 8000