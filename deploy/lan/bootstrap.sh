#!/usr/bin/env bash
# 局域网一键部署(多用户托管模式)。在目标机、仓库根目录运行:
#
#   bash deploy/lan/bootstrap.sh <本机局域网IP>
#   例:bash deploy/lan/bootstrap.sh 192.168.171.9
#
# 职责:
#   1. 检查 docker + compose v2;
#   2. 首次运行时在一次性 python 容器里生成全部密钥,写 deploy/lan/.env.lan
#      (已存在则原样沿用 —— 重复运行只做 up -d --build,幂等);
#   3. 拉起整套栈并做健康验证,最后打印各服务地址。
#
# 端口:Web 3000 · API 8000 · MCP 8080 · Auth 8001 · MinIO 9000/9001 ·
#   Postgres 仅宿主机本机 5432。
set -euo pipefail

LAN_IP="${1:-}"
if [ -z "$LAN_IP" ]; then
  echo "用法: bash deploy/lan/bootstrap.sh <本机局域网IP>" >&2
  echo "例:  bash deploy/lan/bootstrap.sh 192.168.171.9" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.lan"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.lan.yml"

command -v docker >/dev/null 2>&1 || { echo "未找到 docker,请先安装 Docker Engine" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "未找到 docker compose v2 插件(docker compose 子命令)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker 守护进程不可用(需要 root 或 docker 组权限?试试 sudo)" >&2; exit 1; }

# ── 生成 .env.lan(仅首次)──
if [ -f "$ENV_FILE" ]; then
  echo "[bootstrap] 沿用已有 $ENV_FILE(如需换密钥请先删除该文件 —— 会作废所有已发 token)"
else
  echo "[bootstrap] 生成密钥(一次性 python 容器)..."
  SECRETS=$(docker run --rm -v "$SCRIPT_DIR/gen_secrets.py:/gen.py:ro" python:3.11-alpine \
    sh -c 'pip install -q "PyJWT[crypto]" >/dev/null 2>&1 && python /gen.py')
  if ! grep -q '^GOTRUE_JWT_KEYS=' <<<"$SECRETS"; then
    echo "[bootstrap] 密钥生成失败,输出如下:" >&2
    echo "$SECRETS" >&2
    exit 1
  fi

  umask 077
  {
    echo "# llmwiki 局域网自部署配置 — 由 bootstrap.sh 生成,含密钥,勿提交/外传"
    echo "# LAN_IP=$LAN_IP  生成时间:$(date -Iseconds)"
    echo
    echo "APP_URL=http://$LAN_IP:3000"
    echo "API_URL=http://$LAN_IP:8000"
    echo "MCP_URL=http://$LAN_IP:8080/mcp"
    echo "SUPABASE_URL=http://$LAN_IP:8001"
    echo "S3_ENDPOINT_URL=http://$LAN_IP:9000"
    echo
    echo "S3_BUCKET=llmwiki-documents"
    echo "AWS_ACCESS_KEY_ID=llmwiki-app"
    echo "MINIO_ROOT_USER=minio-admin"
    echo
    echo "$SECRETS"
  } > "$ENV_FILE"
  # DATABASE_URL 依赖上面生成的 POSTGRES_PASSWORD,读回再拼
  PG_PASS=$(grep '^POSTGRES_PASSWORD=' "$ENV_FILE" | cut -d= -f2-)
  echo "DATABASE_URL=postgresql://postgres:$PG_PASS@db:5432/postgres" >> "$ENV_FILE"
  echo "[bootstrap] 已写入 $ENV_FILE(权限 600)"
fi

# ── 构建并启动 ──
echo "[bootstrap] docker compose up -d --build(首次构建约需数分钟)..."
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build

# ── 健康验证 ──
check() {  # check <名称> <URL> [重试次数]
  local name="$1" url="$2" tries="${3:-30}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS -o /dev/null --max-time 3 "$url" 2>/dev/null; then
      echo "[bootstrap] ✔ $name 就绪 ($url)"
      return 0
    fi
    sleep 2
  done
  echo "[bootstrap] ✘ $name 未就绪 ($url) — 查看日志:docker compose -f $COMPOSE_FILE --env-file $ENV_FILE logs" >&2
  return 1
}

ok=0
check "Auth(GoTrue)" "http://$LAN_IP:8001/auth/v1/health" 45 || ok=1
check "API"          "http://$LAN_IP:8000/health"          60 || ok=1
check "MinIO S3"     "http://$LAN_IP:9000/minio/health/live" 30 || ok=1
check "Web"          "http://$LAN_IP:3000"                 60 || ok=1

echo
if [ "$ok" -eq 0 ]; then
  echo "════════════════════════════════════════════════════"
  echo " 部署完成 ✔"
else
  echo "════════════════════════════════════════════════════"
  echo " 部分服务未通过健康检查,请按上方提示查看日志"
fi
echo "  Web 前端     http://$LAN_IP:3000   (注册即用,邮箱免验证)"
echo "  API          http://$LAN_IP:8000"
echo "  MCP          http://$LAN_IP:8080/mcp"
echo "  Auth         http://$LAN_IP:8001/auth/v1/health"
echo "  MinIO 控制台 http://$LAN_IP:9001   (账号 minio-admin,密码见 .env.lan)"
echo "  Postgres     127.0.0.1:5432(仅本机;密码见 .env.lan)"
echo "════════════════════════════════════════════════════"
exit "$ok"
