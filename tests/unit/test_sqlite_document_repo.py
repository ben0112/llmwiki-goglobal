"""Regression tests for local-mode document queries.

find_by_path and get_by_source_url once referenced knowledge_base_id and
archived — columns that don't exist in the local SQLite schema — so note
creation and web-clip lookups failed with 500 in local mode (PR #1). These
tests run the repository methods against the real shared/sqlite_schema.sql
so a query referencing a nonexistent column fails here, not at runtime.
"""

import json
from pathlib import Path

import aiosqlite
import pytest

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"


async def _init_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) "
        "VALUES ('ws1', 'test', '', 'u1')"
    )
    await db.commit()
    return db


async def _insert_doc(db: aiosqlite.Connection, **overrides) -> None:
    row = {
        "id": "d1",
        "user_id": "u1",
        "filename": "clip.md",
        "path": "/webclipper/",
        "relative_path": "webclipper/clip.md",
        "source_kind": "source",
        "file_type": "md",
        "status": "ready",
        "metadata": json.dumps({"source_url": "https://example.com/article"}),
    }
    row.update(overrides)
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    await db.execute(f"INSERT INTO documents ({cols}) VALUES ({marks})", tuple(row.values()))
    await db.commit()


@pytest.mark.asyncio
async def test_find_by_path_matches_without_kb_columns():
    db = await _init_db()
    try:
        from infra.db.sqlite import SQLiteDocumentRepository

        repo = SQLiteDocumentRepository(db)
        await _insert_doc(db)

        hit = await repo.find_by_path("ws1", "u1", "clip.md", "/webclipper/")
        assert hit is not None and hit["id"] == "d1"

        miss = await repo.find_by_path("ws1", "u1", "other.md", "/webclipper/")
        assert miss is None
    finally:
        await db.close()




@pytest.mark.asyncio
async def test_list_survives_malformed_tags_row():
    """一条 tags 非法 JSON 的坏行不应让整个文档列表 500(线上曾整页瘫痪)。"""
    db = await _init_db()
    try:
        from infra.db.sqlite import SQLiteDocumentRepository

        repo = SQLiteDocumentRepository(db)
        await _insert_doc(db, id="ok1", filename="好行.md", tags="[\"a\"]")
        await _insert_doc(db, id="bad", filename="坏行.md", relative_path="坏行.md",
                          tags="not-json![", path="/")

        docs = await repo.list_by_kb("ws1")
        by_name = {d["filename"]: d for d in docs}
        assert "好行.md" in by_name and by_name["好行.md"]["tags"] == ["a"]
        # 坏行按容错解码保留,tags 退回空列表
        assert "坏行.md" in by_name and by_name["坏行.md"]["tags"] == []
    finally:
        await db.close()
