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

# First boot: scaffold the workspace (index.db + wiki/overview.md + log.md),
# same as the CLI's `llmwiki init`, so the wiki isn't empty on first open.
[ -f "$WORKSPACE_PATH/.llmwiki/index.db" ] || python3 /app/llmwiki init "$WORKSPACE_PATH"

# Browser-facing API origin. NEXT_PUBLIC_API_URL is baked into the web bundle
# at build time, so remapped host ports (e.g. -p 9000:8000) need a runtime
# override: PUBLIC_API_URL is served to the browser via /__llmwiki_env.js and
# also used by the API for asset URLs it hands out.
PUBLIC_API_URL="${PUBLIC_API_URL:-http://localhost:8000}"
export API_URL="$PUBLIC_API_URL"
printf 'window.__LLMWIKI_ENV__={API_URL:%s};\n' "\"$PUBLIC_API_URL\"" \
  > /app/web/public/__llmwiki_env.js

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
