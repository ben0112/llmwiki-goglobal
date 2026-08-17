#!/usr/bin/env bash
# 幂等数据库初始化(在 supabase/postgres 一次性容器里运行)。
#
# 职责:
#   1. 等 Postgres 可用、等 GoTrue 完成自身迁移(auth.users 出现);
#   2. 兜底补齐 auth schema / auth.uid()(GoTrue 正常时已存在,IF NOT EXISTS 幂等);
#   3. 按文件名顺序应用 /migrations/*.sql,用 app_migrations 账本跳过已应用项。
#
# compose 里以 restart:"no" 运行;api/mcp 依赖其 service_completed_successfully。
set -euo pipefail

PSQL=(psql -h db -U postgres -d postgres -v ON_ERROR_STOP=1 -q)

echo "[db-init] waiting for postgres ..."
for i in $(seq 1 60); do
  if pg_isready -h db -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
pg_isready -h db -U postgres >/dev/null 2>&1 || { echo "[db-init] postgres not ready after 120s" >&2; exit 1; }

# GoTrue 启动时自建 auth schema 并跑完自己的迁移;auth.users 出现即完成。
# compose 已让本容器 depends_on auth(healthy),这里再兜一层轮询防竞态。
echo "[db-init] waiting for gotrue migrations (auth.users) ..."
for i in $(seq 1 60); do
  ready=$("${PSQL[@]}" -tAc "SELECT to_regclass('auth.users') IS NOT NULL") || ready=f
  [ "$ready" = "t" ] && break
  sleep 2
done
if [ "${ready:-f}" != "t" ]; then
  echo "[db-init] auth.users not found after 120s — check the auth container logs" >&2
  exit 1
fi

# 兜底:仓库迁移引用 auth.uid()。Supabase 云端由平台注入;自部署时 GoTrue
# 不负责创建该函数,这里补上(与 Supabase 定义一致,幂等)。
echo "[db-init] ensuring auth.uid() helper ..."
"${PSQL[@]}" <<'SQL'
CREATE SCHEMA IF NOT EXISTS auth;
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid
LANGUAGE sql STABLE
AS $$
  SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
$$;
SQL

# 迁移账本 + 逐个应用
"${PSQL[@]}" -c "CREATE TABLE IF NOT EXISTS app_migrations (
  name text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
)"

shopt -s nullglob
applied=0
skipped=0
for f in /migrations/*.sql; do
  name=$(basename "$f")
  done_flag=$("${PSQL[@]}" -tAc "SELECT EXISTS(SELECT 1 FROM app_migrations WHERE name = '$name')")
  if [ "$done_flag" = "t" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  echo "[db-init] applying $name ..."
  "${PSQL[@]}" -f "$f"
  "${PSQL[@]}" -c "INSERT INTO app_migrations (name) VALUES ('$name')"
  applied=$((applied + 1))
done

echo "[db-init] done — applied $applied, skipped $skipped (already applied)"
