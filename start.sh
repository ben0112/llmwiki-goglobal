#!/bin/bash
# LLM Wiki — local mode launcher
# Usage: ./start.sh [workspace-folder]   (default: ./workspace)
set -e
cd "$(dirname "$0")"

export NODE_ENV=development   # shell default of 'production' breaks `next dev`
export API_PORT="${API_PORT:-8010}"   # API port (8000 is reserved for other apps)
WORKSPACE="${1:-$PWD/workspace}"

# Free our ports if a previous run is still holding them
lsof -ti:3000,"$API_PORT" 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

exec .venv/bin/python3 ./llmwiki open "$WORKSPACE"
