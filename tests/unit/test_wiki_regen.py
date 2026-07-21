"""删除源文件后维基自动重生成(services/wiki_regen)的单元测试。

覆盖:确定性剥离(脚注定义行 + 正文内联标记,含中文定位后缀与基名匹配)、
cites 反查引用维基页面、删除影响预估(排除同批删除的页面)、mock LLM 配置下
regenerate_pages 端到端(strip 路径:改库、落盘、重建引用图、状态汇报)。
"""

from pathlib import Path

import aiosqlite
import pytest

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"

WIKI_CONTENT = """# 市场准入

东南亚市场需要本地代理[^1],欧盟需要 CE 认证[^2]。

日本市场参见许可证要求[^3]。

[^1]: sea-agents.pdf, p.3
[^2]: eu-ce.docx
[^3]: 日本许可证指引.md,第3条
"""


# ── 确定性剥离 ──────────────────────────────────────────────


def test_strip_removes_deleted_footnotes():
    from services.wiki_regen import strip_deleted_citations

    new, changed = strip_deleted_citations(WIKI_CONTENT, ["eu-ce.docx"])
    assert changed
    assert "[^2]" not in new
    assert "eu-ce.docx" not in new
    # 其他脚注与正文原样保留(剥离只摘引用标记,不动叙述)
    assert "[^1]" in new and "sea-agents.pdf" in new
    assert "CE 认证" in new


def test_strip_matches_chinese_suffix_and_base_name():
    from services.wiki_regen import strip_deleted_citations

    # 中文定位后缀(第3条)不妨碍文件名解析
    new, changed = strip_deleted_citations(WIKI_CONTENT, ["日本许可证指引.md"])
    assert changed and "[^3]" not in new
    # 基名匹配:删除的文件名与引用只差扩展名
    new2, changed2 = strip_deleted_citations(WIKI_CONTENT, ["sea-agents"])
    assert changed2 and "[^1]" not in new2


def test_strip_no_match_returns_unchanged():
    from services.wiki_regen import strip_deleted_citations

    new, changed = strip_deleted_citations(WIKI_CONTENT, ["unrelated.pdf"])
    assert not changed
    assert new == WIKI_CONTENT


# ── 反查与影响预估 ──────────────────────────────────────────


async def _init_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) "
        "VALUES ('ws1', 'test', '', 'u1')"
    )
    await db.commit()
    return db


async def _insert_doc(db, doc_id: str, filename: str, path: str,
                      source_kind: str, content: str | None = None) -> None:
    relative = (path.rstrip("/") + "/" + filename).lstrip("/")
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, content) "
        "VALUES (?, 'u1', ?, ?, ?, ?, ?, 'md', 'ready', ?)",
        (doc_id, filename, filename, path, relative, source_kind, content),
    )
    await db.commit()


async def _insert_ref(db, source_id: str, target_id: str, ref_type: str = "cites") -> None:
    await db.execute(
        "INSERT INTO document_references (source_document_id, target_document_id, reference_type) "
        "VALUES (?, ?, ?)",
        (source_id, target_id, ref_type),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_find_citing_wiki_pages_filters_cites_and_kind():
    from services.wiki_regen import find_citing_wiki_pages

    db = await _init_db()
    try:
        await _insert_doc(db, "w1", "page.md", "/wiki/", "wiki", WIKI_CONTENT)
        await _insert_doc(db, "w2", "other.md", "/wiki/", "wiki")
        await _insert_doc(db, "s1", "eu-ce.docx", "/", "source")
        await _insert_doc(db, "n1", "note.md", "/", "source")
        await _insert_ref(db, "w1", "s1", "cites")
        await _insert_ref(db, "w2", "s1", "links_to")   # 非 cites 边不算引用
        await _insert_ref(db, "n1", "s1", "cites")      # 源文件引用源文件不算维基页面

        pages = await find_citing_wiki_pages(db, ["s1"])
        assert [p["id"] for p in pages] == ["w1"]
        assert pages[0]["filename"] == "page.md"

        assert await find_citing_wiki_pages(db, []) == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_delete_impact_excludes_pages_deleted_in_same_batch():
    db = await _init_db()
    try:
        from services.local import LocalDocumentService

        await _insert_doc(db, "w1", "page.md", "/wiki/", "wiki", WIKI_CONTENT)
        await _insert_doc(db, "s1", "eu-ce.docx", "/", "source")
        await _insert_ref(db, "w1", "s1", "cites")

        service = LocalDocumentService(db, "u1")
        impact = await service.delete_impact(["s1"])
        assert impact["count"] == 1 and impact["pages"][0]["id"] == "w1"

        # 引用页面自身也在删除名单里时不再计入(删除后无需重生成)
        impact2 = await service.delete_impact(["s1", "w1"])
        assert impact2["count"] == 0
    finally:
        await db.close()


# ── mock 配置下端到端重生成 ─────────────────────────────────


@pytest.mark.asyncio
async def test_regenerate_pages_strip_path(tmp_path, monkeypatch):
    import services.local as local_module
    import services.wiki_regen as wiki_regen
    from corpus.llm import LLMConfig

    ws = tmp_path / "ws"
    (ws / "wiki").mkdir(parents=True)
    (ws / "wiki" / "page.md").write_text(WIKI_CONTENT, encoding="utf-8")
    monkeypatch.setattr(local_module.settings, "WORKSPACE_PATH", str(ws))
    # mock 配置 → 跳过 LLM,走确定性剥离
    monkeypatch.setattr(wiki_regen, "_resolve_llm", lambda: (None, LLMConfig(base_url="mock")))

    db = await _init_db()
    try:
        await _insert_doc(db, "w1", "page.md", "/wiki/", "wiki", WIKI_CONTENT)
        await _insert_doc(db, "s2", "sea-agents.pdf", "/", "source")
        await _insert_ref(db, "w1", "s2", "cites")
        # s1(eu-ce.docx)已被删除:文档行不存在,只重生成引用页面

        await wiki_regen.regenerate_pages(
            db, "u1",
            [{"id": "w1", "title": "市场准入", "path": "/wiki/", "filename": "page.md"}],
            ["eu-ce.docx"],
        )

        cursor = await db.execute("SELECT content FROM documents WHERE id = 'w1'")
        content = (await cursor.fetchone())[0]
        assert "[^2]" not in content and "eu-ce.docx" not in content
        assert "sea-agents.pdf" in content

        on_disk = (ws / "wiki" / "page.md").read_text(encoding="utf-8")
        assert on_disk == content

        # 引用图已重建:仍引用现存源文件,不再有指向已删文件的悬挂边
        cursor = await db.execute(
            "SELECT target_document_id FROM document_references "
            "WHERE source_document_id = 'w1' AND reference_type = 'cites'"
        )
        targets = [r[0] for r in await cursor.fetchall()]
        assert targets == ["s2"]

        status = wiki_regen.regen_status()
        assert status["running"] is False
        assert status["total"] == 1 and status["failed"] == 0
        assert status["mode"] == "strip"
        assert status["finished_at"] is not None
    finally:
        await db.close()
