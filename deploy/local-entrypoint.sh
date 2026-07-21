#!/bin/bash
# Entrypoint for the local-mode all-in-one image (Dockerfile.local).
# Starts the API (uvicorn), the web server (Next.js standalone), and the MCP
# server (streamable HTTP) together; the API's local lifespan initializes the
# workspace, reconciles the index in the background, and runs the file
# watcher — no separate init step needed.
set -e

: "${WORKSPACE_PATH:=/workspace}"
: "${API_PORT:=8000}"
: "${WEB_PORT:=3000}"
: "${MCP_PORT:=8080}"
export WORKSPACE_PATH MODE=local DATABASE_URL=""
mkdir -p "$WORKSPACE_PATH"

# First boot: scaffold the workspace (index.db + wiki/overview.md + log.md),
# same as the CLI's `llmwiki init`, so the wiki isn't empty on first open.
[ -f "$WORKSPACE_PATH/.llmwiki/index.db" ] || python3 /app/llmwiki init "$WORKSPACE_PATH"

# Browser/客户端-facing origins. NEXT_PUBLIC_* is baked into the web bundle at
# build time, so remapped host ports (e.g. -p 9100:8000) need runtime
# overrides: PUBLIC_API_URL reaches the browser via /__llmwiki_env.js and is
# also used by the API for asset URLs; PUBLIC_MCP_URL is what MCP clients on
# the host should connect to.
# Fallback matches the documented bare `docker run -p 9000:8000` (compose
# always passes PUBLIC_API_URL explicitly); the in-container port stays 8000.
PUBLIC_API_URL="${PUBLIC_API_URL:-http://localhost:9000}"
PUBLIC_MCP_URL="${PUBLIC_MCP_URL:-http://localhost:8080/mcp}"
export API_URL="$PUBLIC_API_URL"
printf 'window.__LLMWIKI_ENV__={API_URL:%s,MCP_URL:%s};\n' \
  "\"$PUBLIC_API_URL\"" "\"$PUBLIC_MCP_URL\"" \
  > /app/web/public/__llmwiki_env.js

(cd /app/api && exec uvicorn main:app --host 0.0.0.0 --port "$API_PORT") &
api_pid=$!
(cd /app/web && exec env HOSTNAME=0.0.0.0 PORT="$WEB_PORT" node server.js) &
web_pid=$!
(cd /app/mcp && exec python3 -m local_server --workspace "$WORKSPACE_PATH" --http --port "$MCP_PORT") &
mcp_pid=$!

term() { kill -TERM "$api_pid" "$web_pid" "$mcp_pid" 2>/dev/null || true; }
trap term TERM INT

# Once everything answers, print copy-paste connection info into the logs
# (docker logs llmwiki / docker compose logs llmwiki).
(
  for _ in $(seq 1 60); do
    curl -fsS "http://localhost:$API_PORT/health" >/dev/null 2>&1 \
      && curl -fsS "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  web_url="${PUBLIC_WEB_URL:-${PUBLIC_API_URL%:*}:${WEB_PORT}}"
  pipe_line=$(curl -fsS "http://localhost:$API_PORT/v1/corpus/pipeline/status" \
      -H "Authorization: Bearer local" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    c, a = d["counts"], d["auto"]
    auto = "开" if a["enabled"] else "关(设置页可开)"
    print("待分类 %d · 今日入库 %d · 自动分类%s" % (c["pending"], c["imported_today"], auto))
except Exception:
    pass' 2>/dev/null)
  cat <<BANNER

============================= LLM Wiki 已就绪 =============================
  Web 应用:  ${web_url}
  API:       ${PUBLIC_API_URL}
  MCP(HTTP): ${PUBLIC_MCP_URL}
  语料流水线: ${pipe_line:-未就绪}

  连接 Claude —— 任选其一:

  · Claude Code(一条命令):
      claude mcp add --transport http llmwiki ${PUBLIC_MCP_URL}

  · 任意支持 Streamable HTTP 的 MCP 客户端(配置 JSON):
      {"mcpServers": {"llmwiki": {"url": "${PUBLIC_MCP_URL}"}}}

  · Claude Desktop(stdio,经容器;无需额外依赖):
      {"mcpServers": {"llmwiki": {"command": "docker",
        "args": ["exec", "-i", "${LLMWIKI_CONTAINER_NAME:-llmwiki}", "/app/llmwiki", "mcp", "/workspace"]}}}
===========================================================================

BANNER
) &

# Exit when any server dies, then stop the others.
wait -n "$api_pid" "$web_pid" "$mcp_pid"
code=$?
term
wait || true
exit "$code"
