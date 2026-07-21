#!/bin/bash
# Entrypoint for the local-mode all-in-one image (Dockerfile.local).
# Starts the API (uvicorn) and the web server (Next.js standalone) together;
# the API's local lifespan initializes the workspace, reconciles the index in
# the background, and runs the file watcher — no separate init step needed.
set -e

: "${WORKSPACE_PATH:=/workspace}"
: "${API_PORT:=8000}"
: "${WEB_PORT:=3000}"
export WORKSPACE_PATH MODE=local DATABASE_URL=""
mkdir -p "$WORKSPACE_PATH"

(cd /app/api && exec uvicorn main:app --host 0.0.0.0 --port "$API_PORT") &
api_pid=$!
(cd /app/web && exec env HOSTNAME=0.0.0.0 PORT="$WEB_PORT" node server.js) &
web_pid=$!

term() { kill -TERM "$api_pid" "$web_pid" 2>/dev/null || true; }
trap term TERM INT

# Exit when either server dies, then stop the other.
wait -n "$api_pid" "$web_pid"
code=$?
term
wait || true
exit "$code"
