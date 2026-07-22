"""删除源文件后自动重生成引用它的维基页面(仅本地模式)。

删除入口(services/local.py 的 delete/bulk_delete)在归档前通过
find_citing_wiki_pages 反查 document_references 中 cites 指向被删文档的
维基页面,归档成功后调度本模块在后台重写这些页面:配置了分类 LLM 端点
时整页重写(移除依据已删源文件的句段),mock/未配置/调用失败时退化为
确定性剥离——删掉指向已删文件的脚注定义行与正文内联标记,其余内容原样
保留。批次完成后全量重建引用图,前端通过 /v1/documents/regen-status 轮询。
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from pathlib import Path

from services.references import parse_citation_filename

logger = logging.getLogger(__name__)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]\s]+)\]:\s*(.+)$", re.MULTILINE)
_EXT_RE = re.compile(r"\.(pdf|docx?|pptx?|xlsx?|csv|html?|md|txt)$")

# 单页超过此长度不走 LLM 整页重写(输出易被截断),直接确定性剥离
_MAX_LLM_CHARS = 30_000

# 后台重生成状态(本地模式单进程,模块级即全局)
_status: dict = {
    "running": False, "total": 0, "done": 0, "failed": 0,
    "pages": [], "mode": "", "finished_at": None,
}
_lock = asyncio.Lock()

_REWRITE_SYSTEM = (
    "你是维基维护助手。用户删除了部分源文件,请更新一篇引用了这些源文件的维基页面。\n"
    "要求:\n"
    "1. 删除或改写正文中依据已删除源文件的句段(即内联脚注标记 [^N] 指向这些文件的内容);\n"
    "2. 移除对应的脚注定义行([^N]: 文件名 …);\n"
    "3. 其余内容逐字保留,不要改写无关段落,不要新增内容;\n"
    "4. 只输出更新后的完整 Markdown,不要任何解释。"
)


def regen_status() -> dict:
    return dict(_status, pages=list(_status["pages"]))


async def find_citing_wiki_pages(db, doc_ids: list[str]) -> list[dict]:
    """反查通过 cites 边引用了给定文档的维基页面(去重)。"""
    if not doc_ids:
        return []
    placeholders = ",".join("?" for _ in doc_ids)
    cursor = await db.execute(
        "SELECT DISTINCT d.id, d.title, d.path, d.filename "
        "FROM document_references r JOIN documents d ON d.id = r.source_document_id "
        f"WHERE r.reference_type = 'cites' AND r.target_document_id IN ({placeholders}) "
        "AND d.source_kind = 'wiki'",
        doc_ids,
    )
    rows = await cursor.fetchall()
    return [{"id": r[0], "title": r[1], "path": r[2], "filename": r[3]} for r in rows]


def _name_variants(name: str) -> set[str]:
    n = name.strip().lower()
    return {n, _EXT_RE.sub("", n)}


def strip_deleted_citations(content: str, deleted_names: list[str]) -> tuple[str, bool]:
    """确定性剥离:删除指向已删源文件的脚注定义行与正文内联标记。"""
    targets: set[str] = set()
    for name in deleted_names:
        targets |= _name_variants(name)

    labels: list[str] = []

    def _drop(match: re.Match) -> str:
        fname, _page = parse_citation_filename(match.group(2))
        if _name_variants(fname) & targets:
            labels.append(match.group(1))
            return ""
        return match.group(0)

    new = _FOOTNOTE_DEF_RE.sub(_drop, content)
    if not labels:
        return content, False
    for label in labels:
        new = new.replace(f"[^{label}]", "")
    new = re.sub(r"\n{3,}", "\n\n", new)
    return new, True


def _resolve_llm():
    """按 corpus 流水线同一优先级(设置 > 环境变量 > 默认)解析 LLM 配置。"""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    import sqlite3

    from config import settings as env
    from corpus import llm as corpus_llm
    from corpus import pipeline as corpus_pipeline

    workspace = Path(env.WORKSPACE_PATH).resolve()
    conn = sqlite3.connect(str(workspace / ".llmwiki" / "index.db"))
    try:
        stored = corpus_pipeline.load_llm_settings(conn)
    finally:
        conn.close()
    return corpus_llm, corpus_pipeline.resolve_config(stored, env)


async def _rewrite_page(content: str, deleted_names: list[str]) -> tuple[str, str]:
    """返回 (新内容, 方式);方式 ∈ {"llm", "strip", "unchanged"}。"""
    stripped, changed = strip_deleted_citations(content, deleted_names)
    if not changed:
        return content, "unchanged"
    if len(content) <= _MAX_LLM_CHARS:
        try:
            corpus_llm, config = _resolve_llm()
            if not config.is_mock:
                names = "\n".join(f"- {n}" for n in deleted_names)
                out = await corpus_llm.chat_text(
                    config, _REWRITE_SYSTEM,
                    f"已删除的源文件:\n{names}\n\n维基页面内容:\n\n{content}",
                    max_tokens=max(2048, min(16384, len(content) + 1024)),
                )
                # 输出骤缩视为截断/失败,退回确定性剥离
                if len(out) >= len(stripped) * 0.3:
                    return out, "llm"
                logger.warning("LLM 重写输出过短(%d/%d),退化为剥离引用", len(out), len(stripped))
        except Exception as e:
            logger.warning("LLM 重写失败,退化为剥离引用: %s", e)
    return stripped, "strip"


async def regenerate_pages(db, user_id: str, pages: list[dict], deleted_names: list[str]) -> None:
    from services.graph import rebuild_local
    from services.local import LocalDocumentService

    async with _lock:
        _status.update(
            running=True, total=len(pages), done=0, failed=0,
            pages=[p.get("title") or p.get("filename") or "" for p in pages],
            mode="", finished_at=None,
        )
        service = LocalDocumentService(db, user_id)
        modes: list[str] = []
        for page in pages:
            try:
                row = await service.get_content(page["id"])
                content = (row or {}).get("content") or ""
                if content.strip():
                    new_content, mode = await _rewrite_page(content, deleted_names)
                    modes.append(mode)
                    if mode != "unchanged" and new_content != content:
                        await service.update_content(page["id"], new_content)
            except Exception:
                logger.exception("重生成维基页面失败: %s", page.get("filename"))
                _status["failed"] += 1
            _status["done"] += 1
        try:
            await rebuild_local(db, user_id)
        except Exception:
            logger.exception("删除后重建引用图失败")
        _status["mode"] = "llm" if "llm" in modes else ("strip" if "strip" in modes else "unchanged")
        _status["running"] = False
        _status["finished_at"] = time.time()
        logger.info(
            "已重生成 %d/%d 个维基页面(方式=%s,失败 %d)",
            _status["done"] - _status["failed"], _status["total"], _status["mode"], _status["failed"],
        )


def schedule_regeneration(db, user_id: str, pages: list[dict], deleted_names: list[str]) -> None:
    if not pages:
        return
    from infra.tasks import spawn_logged
    spawn_logged(regenerate_pages(db, user_id, pages, deleted_names),
                 f"wiki-regen:{len(pages)}页")
