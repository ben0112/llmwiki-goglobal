"""语料分类流水线编排器(L3-P1,本地模式)。

对工作区里尚未分类的源文档,逐条执行 审核 → 八维标注 → 业务派生 → 入库,
复用 import_annotations 的按条导入原语(文件即真相源 + 幂等 upsert)。

状态机(存 corpus_pipeline 表,和 API 主连接同库、WAL 并存):
    (无记录) → imported | excluded | failed(attempts<3 时下轮重试)
C1/R1/X9/低置信条目由 schema.parse_row 按公理一置为「待复核」,与 CLI 导入一致。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from .annotate import audit, classify, extract_title
from .codetable import DEFAULT_VERSION, load
from .derive import apply_business_view
from .import_annotations import entry_relative_path, render_entry_markdown, upsert_document
from .llm import LLMConfig, LLMError
from .schema import ERROR, parse_row

MAX_ATTEMPTS = 3


@dataclass
class RunResult:
    picked: int = 0
    imported: int = 0
    review: int = 0        # imported 中进待复核队列的
    excluded: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)   # [(relpath, message)] 最近若干条
    started_at: str = ""
    finished_at: str = ""

    def summary(self) -> dict:
        return {
            "picked": self.picked, "imported": self.imported, "review": self.review,
            "excluded": self.excluded, "failed": self.failed,
            "errors": self.errors[-10:],
            "started_at": self.started_at, "finished_at": self.finished_at,
        }


def _connect(db_path: str) -> sqlite3.Connection:
    # check_same_thread=False: 导入步骤在 to_thread 里落盘+写库,访问是顺序的
    # (同一时刻只有一个使用者),放开 SQLite 的线程绑定检查是安全的。
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_state(conn: sqlite3.Connection, doc_id: str, state: str,
               error: str = "", entry_id: str = "") -> None:
    conn.execute(
        "INSERT INTO corpus_pipeline (doc_id, state, attempts, error, entry_id, updated_at) "
        "VALUES (?, ?, 1, ?, ?, ?) "
        "ON CONFLICT(doc_id) DO UPDATE SET state=excluded.state, "
        "attempts=corpus_pipeline.attempts+1, error=excluded.error, "
        "entry_id=excluded.entry_id, updated_at=excluded.updated_at",
        (doc_id, state, error, entry_id, _now()),
    )
    conn.commit()


def select_candidates(conn: sqlite3.Connection, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT d.id, d.title, d.filename, d.relative_path, COALESCE(d.content, '') "
        "FROM documents d LEFT JOIN corpus_pipeline p ON p.doc_id = d.id "
        "WHERE d.source_kind = 'source' AND d.status = 'ready' "
        "AND d.relative_path NOT LIKE 'corpus/%' "
        "AND (p.doc_id IS NULL OR (p.state = 'failed' AND p.attempts < ?)) "
        "ORDER BY d.created_at LIMIT ?",
        (MAX_ATTEMPTS, limit),
    ).fetchall()
    return [
        {"id": r[0], "title": r[1] or r[2], "relative_path": r[3], "content": r[4]}
        for r in rows
    ]


def status_counts(conn: sqlite3.Connection) -> dict:
    # 联 documents:全量 reindex 会更换文档 id,孤儿状态行不计入
    counts = dict(conn.execute(
        "SELECT p.state, COUNT(*) FROM corpus_pipeline p "
        "JOIN documents d ON d.id = p.doc_id GROUP BY p.state").fetchall())
    pending = conn.execute(
        "SELECT COUNT(*) FROM documents d LEFT JOIN corpus_pipeline p ON p.doc_id = d.id "
        "WHERE d.source_kind = 'source' AND d.status = 'ready' "
        "AND d.relative_path NOT LIKE 'corpus/%' AND p.doc_id IS NULL").fetchone()[0]
    retryable = conn.execute(
        "SELECT COUNT(*) FROM corpus_pipeline WHERE state='failed' AND attempts < ?",
        (MAX_ATTEMPTS,)).fetchone()[0]
    imported_today = conn.execute(
        "SELECT COUNT(*) FROM corpus_pipeline p JOIN documents d ON d.id = p.doc_id "
        "WHERE p.state='imported' AND p.updated_at >= date('now')").fetchone()[0]
    return {"pending": pending + retryable, "imported": counts.get("imported", 0),
            "imported_today": imported_today,
            "excluded": counts.get("excluded", 0), "failed": counts.get("failed", 0)}


async def run_batch(workspace: Path, config: LLMConfig, limit: int | None = None,
                    version: str = DEFAULT_VERSION,
                    on_progress=None) -> RunResult:
    """跑一轮:最多处理 limit(缺省=端点感知默认)条候选。逐条隔离失败。"""
    result = RunResult(started_at=_now())
    limit = limit or config.effective_batch_limit
    db_path = str(workspace / ".llmwiki" / "index.db")
    table = load(version)
    today = date.today()
    imported_on = today.isoformat()

    conn = _connect(db_path)
    try:
        user_row = conn.execute("SELECT user_id FROM workspace LIMIT 1").fetchone()
        if user_row is None:
            raise RuntimeError("工作区未初始化(缺 workspace 记录)")
        user_id = user_row[0]

        # 顺手清理孤儿状态行(文档已被删除/重建)
        conn.execute("DELETE FROM corpus_pipeline WHERE doc_id NOT IN (SELECT id FROM documents)")
        conn.commit()
        candidates = select_candidates(conn, limit)
        result.picked = len(candidates)

        done = 0
        for doc in candidates:
            relpath = doc["relative_path"]
            try:
                title = extract_title(doc["content"]) or doc["title"]
                include, reason = await audit(config, title, doc["content"])
                if not include:
                    _set_state(conn, doc["id"], "excluded", error=reason)
                    result.excluded += 1
                    continue

                row = await classify(config, title, doc["content"], relpath)
                apply_business_view(row)
                rec = parse_row(row, table, today=today)
                if any(i.level == ERROR for i in rec.issues):
                    # 校验错误仍入库(公理一,parse_row 已兜底并标待复核)
                    pass

                body = doc["content"] or None
                content = render_entry_markdown(rec, body, imported_on)
                rel_path = entry_relative_path(rec)

                def _import() -> None:
                    full = workspace / rel_path
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_text(content, encoding="utf-8")
                    upsert_document(conn, user_id, rec, rel_path, content,
                                    full.stat().st_mtime_ns)
                    conn.commit()

                await asyncio.to_thread(_import)
                _set_state(conn, doc["id"], "imported", entry_id=rec.entry_id)
                result.imported += 1
                if rec.needs_review:
                    result.review += 1
            except (LLMError, RuntimeError, OSError, sqlite3.Error) as e:
                msg = f"{type(e).__name__}: {e}"
                _set_state(conn, doc["id"], "failed", error=msg[:500])
                result.failed += 1
                result.errors.append((relpath, msg[:200]))
            finally:
                done += 1
                if on_progress:
                    try:
                        on_progress(done, result.picked)
                    except Exception:
                        pass
    finally:
        conn.close()

    result.finished_at = _now()
    return result


def load_llm_settings(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT value FROM settings WHERE key='corpus_llm'").fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except ValueError:
        return {}


def save_llm_settings(conn: sqlite3.Connection, data: dict) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('corpus_llm', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(data, ensure_ascii=False),),
    )
    conn.commit()


def resolve_config(stored: dict, env) -> LLMConfig:
    """优先级:设置存储(前端) > 环境变量 > 内置默认。"""
    from .llm import DEFAULT_BASE_URL
    return LLMConfig(
        base_url=stored.get("base_url") or getattr(env, "CORPUS_LLM_BASE_URL", "") or DEFAULT_BASE_URL,
        model=stored.get("model") or getattr(env, "CORPUS_LLM_MODEL", ""),
        api_key=stored.get("api_key") or getattr(env, "CORPUS_LLM_API_KEY", ""),
        timeout=float(stored.get("timeout") or getattr(env, "CORPUS_LLM_TIMEOUT", 120) or 120),
        batch_limit=int(stored.get("batch_limit") or getattr(env, "CORPUS_BATCH_LIMIT", 0) or 0),
    )


def resolve_auto(stored: dict, env) -> dict:
    """自动分类开关与轮询间隔;设置页显式值(含显式关闭)优先于环境变量。"""
    if "auto_enabled" in stored:
        enabled = bool(stored["auto_enabled"])
    else:
        enabled = bool(getattr(env, "CORPUS_AUTOCLASSIFY", False))
    interval = int(stored.get("auto_interval")
                   or getattr(env, "CORPUS_AUTO_INTERVAL", 120) or 120)
    return {"enabled": enabled, "interval": max(30, interval)}


async def run_batch_hosted(database_url: str, user_email: str, kb_slug: str,
                           config: LLMConfig, limit: int | None = None,
                           version: str = DEFAULT_VERSION,
                           on_progress=None) -> RunResult:
    """托管(Postgres)模式跑一轮:候选=目标知识库中未分类的文本源文档。

    条目经 hosted_import 的事务化 upsert 落库(文档+检索分块一步到位);
    状态存 Postgres 版 corpus_pipeline 表(migration 010)。
    """
    import asyncpg

    from .hosted_import import _load_chunker, _resolve_target, _upsert_entry

    result = RunResult(started_at=_now())
    limit = limit or config.effective_batch_limit
    table = load(version)
    today = date.today()
    imported_on = today.isoformat()
    chunker = _load_chunker()

    conn = await asyncpg.connect(database_url)
    try:
        target = await _resolve_target(conn, user_email, kb_slug)
        rows = await conn.fetch(
            "SELECT d.id, d.title, d.filename, d.path, COALESCE(d.content, '') AS content "
            "FROM documents d LEFT JOIN corpus_pipeline p ON p.doc_id = d.id "
            "WHERE d.knowledge_base_id = $1 AND d.user_id = $2 AND NOT d.archived "
            "AND d.status = 'ready' AND COALESCE(d.content, '') <> '' "
            "AND (d.metadata IS NULL OR d.metadata->>'entry_id' IS NULL) "
            "AND (p.doc_id IS NULL OR (p.state = 'failed' AND p.attempts < $3)) "
            "ORDER BY d.created_at LIMIT $4",
            target.kb_id, target.user_id, MAX_ATTEMPTS, limit)
        result.picked = len(rows)

        async def set_state(doc_id, state, error="", entry_id=""):
            await conn.execute(
                "INSERT INTO corpus_pipeline (doc_id, state, attempts, error, entry_id, updated_at) "
                "VALUES ($1, $2, 1, $3, $4, now()) "
                "ON CONFLICT (doc_id) DO UPDATE SET state = EXCLUDED.state, "
                "attempts = corpus_pipeline.attempts + 1, error = EXCLUDED.error, "
                "entry_id = EXCLUDED.entry_id, updated_at = now()",
                doc_id, state, error, entry_id)

        done = 0
        for doc in rows:
            relpath = f"{doc['path'].strip('/')}/{doc['filename']}".strip("/")
            content_src = doc["content"]
            try:
                title = extract_title(content_src) or doc["title"] or doc["filename"]
                include, reason = await audit(config, title, content_src)
                if not include:
                    await set_state(doc["id"], "excluded", error=reason)
                    result.excluded += 1
                    continue

                row = await classify(config, title, content_src, relpath)
                apply_business_view(row)
                rec = parse_row(row, table, today=today)
                entry_content = render_entry_markdown(rec, content_src, imported_on)
                rel_path = entry_relative_path(rec)
                async with conn.transaction():
                    await _upsert_entry(conn, target, rec, rel_path, entry_content, chunker)
                await set_state(doc["id"], "imported", entry_id=rec.entry_id)
                result.imported += 1
                if rec.needs_review:
                    result.review += 1
            except Exception as e:  # noqa: BLE001 — 逐条隔离,任何失败不拖累整轮
                msg = f"{type(e).__name__}: {e}"
                await set_state(doc["id"], "failed", error=msg[:500])
                result.failed += 1
                result.errors.append((relpath, msg[:200]))
            finally:
                done += 1
                if on_progress:
                    try:
                        on_progress(done, result.picked)
                    except Exception:
                        pass
    finally:
        await conn.close()

    result.finished_at = _now()
    return result


class _EnvShim:
    """CLI 场景下从 os.environ 读 CORPUS_* 配置(与 api settings 同名)。"""

    def __getattr__(self, name):
        import os
        v = os.environ.get(name, "")
        if name in ("CORPUS_BATCH_LIMIT", "CORPUS_AUTO_INTERVAL"):
            return int(v) if v else 0
        if name == "CORPUS_LLM_TIMEOUT":
            return float(v) if v else 120.0
        if name == "CORPUS_AUTOCLASSIFY":
            return v.lower() in ("1", "true", "yes")
        return v


def main() -> None:
    """CLI:本地(--workspace)/托管(--database-url)双模式,配 cron 即定时分类。"""
    import argparse
    import asyncio as aio

    ap = argparse.ArgumentParser(description="语料分类流水线(审核→八维标注→业务派生→入库)")
    ap.add_argument("--workspace", default=None, help="本地模式: LLM Wiki 工作区目录")
    ap.add_argument("--database-url", default=None, help="hosted 模式: Postgres 连接串")
    ap.add_argument("--user-email", default=None, help="hosted 模式: 语料归属账号邮箱")
    ap.add_argument("--kb", default="goglobal-corpus", help="hosted 模式: 知识库 slug")
    ap.add_argument("--limit", type=int, default=None, help="本轮批量上限(缺省=端点感知默认)")
    ap.add_argument("--mock", action="store_true", help="规则桩自测,无需 LLM 端点")
    ap.add_argument("--base-url", default=None, help="覆盖 LLM 端点")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--codetable", default=DEFAULT_VERSION, help=f"码表版本(默认 {DEFAULT_VERSION})")
    args = ap.parse_args()

    if bool(args.workspace) == bool(args.database_url):
        ap.error("二选一:--workspace(本地)或 --database-url(hosted)")

    env = _EnvShim()
    stored: dict = {}
    if args.workspace:
        db = Path(args.workspace) / ".llmwiki" / "index.db"
        if db.exists():
            conn = _connect(str(db))
            try:
                stored = load_llm_settings(conn)
            except sqlite3.Error:
                stored = {}
            finally:
                conn.close()
    config = resolve_config(stored, env)
    if args.mock:
        config.base_url = "mock"
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key

    if args.workspace:
        result = aio.run(run_batch(Path(args.workspace).resolve(), config,
                                   limit=args.limit, version=args.codetable))
    else:
        if not args.user_email:
            ap.error("hosted 模式需要 --user-email")
        result = aio.run(run_batch_hosted(args.database_url, args.user_email, args.kb,
                                          config, limit=args.limit, version=args.codetable))

    s = result.summary()
    print(f"[✓] 本轮 {s['picked']} 条: 入库 {s['imported']}(待复核 {s['review']}) "
          f"排除 {s['excluded']} 失败 {s['failed']}")
    for relpath, msg in s["errors"]:
        print(f"    ✗ {relpath}: {msg}")
    raise SystemExit(1 if s["failed"] and not s["imported"] else 0)


if __name__ == "__main__":
    main()
