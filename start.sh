#!/bin/bash
# LLM Wiki — local mode launcher
# Usage: ./start.sh [workspace-folder]   (default: ./workspace)
set -e
cd "$(dirname "$0")"

export NODE_ENV=development   # shell default of 'production' breaks `next dev`
export API_PORT="${API_PORT:-8010}"   # API port (8000 is reserved for other apps)
WORKSPACE="${1:-$PWD/workspace}"

# Free our ports if a previous run is still holding them. Only kill processes
# that look like our own servers (uvicorn / next); anything else on the port
# is another app — bail with a message instead of killing it.
if command -v lsof >/dev/null 2>&1; then
  for port in 3000 "$API_PORT"; do
    for pid in $(lsof -ti :"$port" 2>/dev/null); do
      if ps -o command= -p "$pid" 2>/dev/null | grep -qE 'uvicorn|next'; then
        kill "$pid" 2>/dev/null || true
      else
        echo "Port $port is held by PID $pid ($(ps -o comm= -p "$pid" 2>/dev/null)) — not an LLM Wiki server." >&2
        echo "Free the port or set API_PORT to a different one." >&2
        exit 1
      fi
    done
  done
  sleep 1
fi

# Prefer the repo venv (per README setup); fall back to system python3.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x .venv/bin/python3 ]; then
    PYTHON=.venv/bin/python3
  else
    PYTHON="$(command -v python3)"
  fi
fi

exec "$PYTHON" ./llmwiki open "$WORKSPACE"
