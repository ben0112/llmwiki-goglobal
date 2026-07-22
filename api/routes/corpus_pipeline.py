"""语料分类流水线端点(L3-P1,仅本地模式)。

- GET/PUT  /v1/corpus/llm-config       分类 LLM 端点配置(前端可配;密钥只写不读)
- POST     /v1/corpus/llm-config/test  测试连接
- POST     /v1/corpus/pipeline/run     手动触发一轮(单飞;处理全部待分类,limit 可截断)
- POST     /v1/corpus/pipeline/stop    停止队列(取消在飞轮次 + 关闭自动分类)
- GET      /v1/corpus/pipeline/status  各状态计数 + 当前/上次运行摘要
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/corpus", tags=["corpus-pipeline"])

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _corpus():
    """corpus 包按仓库布局导入(与 llmwiki CLI 同源)。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from corpus import llm as corpus_llm
    from corpus import pipeline as corpus_pipeline
    return corpus_llm, corpus_pipeline


def _require_local(request: Request) -> Path:
    if getattr(request.app.state, "mode", "") != "local":
        raise HTTPException(status_code=501, detail="语料流水线目前仅本地模式可用(P4 规划托管模式)")
    return Path(request.app.state.workspace_path)


def _db(workspace: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(workspace / ".llmwiki" / "index.db"))
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


class LLMConfigIn(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str | None = None   # None/缺省 = 沿用现值;空串 = 清除
    timeout: float | None = None
    concurrency: int | None = Field(default=None, ge=0)  # 0 = 恢复端点感知默认
    auto_enabled: bool | None = None
    auto_interval: int | None = Field(default=None, ge=30)


def _masked(key: str) -> str:
    if not key:
        return ""
    return "•" * 8 + key[-4:] if len(key) > 4 else "•" * 8


def _config_out(stored: dict, request: Request) -> dict:
    corpus_llm, corpus_pipeline = _corpus()
    from config import settings as env
    cfg = corpus_pipeline.resolve_config(stored, env)
    return {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_key_masked": _masked(cfg.api_key),
        "timeout": cfg.timeout,
        "concurrency": cfg.concurrency,
        "effective_concurrency": cfg.effective_concurrency,
        "is_local_endpoint": corpus_llm.is_local_endpoint(cfg.base_url),
        "auto": corpus_pipeline.resolve_auto(stored, env),
        "source": "settings" if stored.get("base_url") else ("env" if getattr(env, "CORPUS_LLM_BASE_URL", "") else "default"),
    }


@router.get("/llm-config")
async def get_llm_config(request: Request):
    workspace = _require_local(request)
    _, corpus_pipeline = _corpus()
    conn = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
    finally:
        conn.close()
    return _config_out(stored, request)


@router.put("/llm-config")
async def put_llm_config(request: Request, body: LLMConfigIn):
    workspace = _require_local(request)
    _, corpus_pipeline = _corpus()
    conn = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
        if body.base_url:
            stored["base_url"] = body.base_url.strip()
        if body.model:
            stored["model"] = body.model.strip()
        if body.api_key is not None:          # 缺省沿用;显式空串清除
            stored["api_key"] = body.api_key.strip()
        if body.timeout is not None:
            stored["timeout"] = body.timeout
        if body.concurrency is not None:      # 0 = 恢复端点感知默认
            stored["concurrency"] = body.concurrency
        if body.auto_enabled is not None:
            stored["auto_enabled"] = body.auto_enabled
        if body.auto_interval is not None:
            stored["auto_interval"] = body.auto_interval
        corpus_pipeline.save_llm_settings(conn, stored)
    finally:
        conn.close()
    return _config_out(stored, request)


@router.post("/llm-config/test")
async def test_llm_config(request: Request):
    workspace = _require_local(request)
    corpus_llm, corpus_pipeline = _corpus()
    from config import settings as env
    conn = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
    finally:
        conn.close()
    cfg = corpus_pipeline.resolve_config(stored, env)
    return await corpus_llm.probe(cfg)


class RunIn(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=10000)


def start_run(state, workspace: Path, limit: int | None = None) -> dict:
    """单飞启动一轮(手动端点与自动循环共用)。"""
    _, corpus_pipeline = _corpus()
    from config import settings as env

    task = getattr(state, "corpus_pipeline_task", None)
    if task is not None and not task.done():
        return {"started": False, "running": True, "detail": "已有一轮在运行"}

    conn = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
    finally:
        conn.close()
    cfg = corpus_pipeline.resolve_config(stored, env)

    def _progress(done: int, total: int) -> None:
        state.corpus_pipeline_progress = {"done": done, "total": total}

    async def _run():
        state.corpus_pipeline_progress = {"done": 0, "total": 0}
        try:
            from domain.watcher import mark_written
            result = await corpus_pipeline.run_batch(
                workspace, cfg, limit=limit, on_progress=_progress,
                on_file_written=mark_written)
            state.corpus_pipeline_last_run = result.summary()
            return result
        except asyncio.CancelledError:
            prog = getattr(state, "corpus_pipeline_progress", None) or {}
            state.corpus_pipeline_last_run = {"stopped": True, **prog}
            raise
        except Exception as e:
            # 记进 last_run 让状态条可见,再抛给 spawn_logged 落日志
            state.corpus_pipeline_last_run = {"error": f"{type(e).__name__}: {e}"[:300]}
            raise
        finally:
            state.corpus_pipeline_progress = None

    from infra.tasks import spawn_logged
    state.corpus_pipeline_task = spawn_logged(_run(), "corpus-pipeline")
    return {"started": True, "running": True,
            "concurrency": cfg.effective_concurrency}


@router.post("/pipeline/run", status_code=202)
async def run_pipeline(request: Request, body: RunIn | None = None):
    workspace = _require_local(request)
    limit = body.limit if body and body.limit else None
    return start_run(request.app.state, workspace, limit)


@router.post("/pipeline/stop")
async def stop_pipeline(request: Request):
    """停止分类队列:取消在飞轮次,并关闭自动分类(否则 30s 内会自动重启)。

    被取消的条目不落任何状态,留在待分类队列,重新开始时继续处理。
    """
    workspace = _require_local(request)
    _, corpus_pipeline = _corpus()
    state = request.app.state

    task = getattr(state, "corpus_pipeline_task", None)
    was_running = bool(task is not None and not task.done())
    if was_running:
        task.cancel()

    conn = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
        stored["auto_enabled"] = False
        corpus_pipeline.save_llm_settings(conn, stored)
    finally:
        conn.close()
    return {"stopped": was_running, "auto_disabled": True}


async def auto_loop(app) -> None:
    """自动分类轮询:开着且有待分类文档、无在飞轮次时,静默起一轮。"""
    _, corpus_pipeline = _corpus()
    from config import settings as env
    workspace = Path(app.state.workspace_path)
    interval = 30
    while True:
        try:
            conn = _db(workspace)
            try:
                stored = corpus_pipeline.load_llm_settings(conn)
                auto = corpus_pipeline.resolve_auto(stored, env)
                pending = corpus_pipeline.status_counts(conn)["pending"] if auto["enabled"] else 0
            finally:
                conn.close()
            interval = auto["interval"]
            if auto["enabled"] and pending > 0:
                start_run(app.state, workspace)
        except Exception:   # 自动循环绝不因单次异常退出
            pass
        await asyncio.sleep(interval)


@router.get("/pipeline/status")
async def pipeline_status(request: Request):
    workspace = _require_local(request)
    _, corpus_pipeline = _corpus()
    state = request.app.state
    task = getattr(state, "corpus_pipeline_task", None)
    conn = _db(workspace)
    try:
        counts = corpus_pipeline.status_counts(conn)
    finally:
        conn.close()
    from config import settings as env
    conn2 = _db(workspace)
    try:
        stored = corpus_pipeline.load_llm_settings(conn2)
    finally:
        conn2.close()
    return {
        "running": bool(task is not None and not task.done()),
        "progress": getattr(state, "corpus_pipeline_progress", None),
        "auto": corpus_pipeline.resolve_auto(stored, env),
        "counts": counts,
        "last_run": getattr(state, "corpus_pipeline_last_run", None),
    }


# ── 复核队列(L3-P3) ─────────────────────────────────────────────

class ReviewIn(BaseModel):
    action: str  # approve | update | exclude
    labels: dict | None = None
    note: str = ""


@router.get("/codetable")
async def get_codetable(request: Request):
    _require_local(request)
    _corpus()
    from corpus.review import codetable_options
    return codetable_options()


@router.post("/entries/{doc_id}/review")
async def review_entry(request: Request, doc_id: str, body: ReviewIn):
    workspace = _require_local(request)
    _corpus()
    from corpus.review import ReviewError, apply_review

    try:
        result = await asyncio.to_thread(
            apply_review, workspace, doc_id, body.action,
            body.labels, body.note)
    except ReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
