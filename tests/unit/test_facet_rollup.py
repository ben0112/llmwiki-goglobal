"""维基页面分面聚合(facet_rollup)的单元测试。

纯聚合函数(并集/最紧时效/非条目过滤/空聚合)+ rebuild_local 端到端
(引用条目 → 页面 metadata 写入 rollup;取消引用 → rollup 移除)。
"""

import json
from pathlib import Path

import aiosqlite
import pytest

from services.facet_rollup import apply_rollup, refresh_rollups_local, rollup_from_metas

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"


def _entry_meta(**kw) -> dict:
    base = {"entry_id": "E-1", "stage": "S2", "domain": "G1",
            "geo_country": ["IDN"], "timeliness": "M2",
            "business": {"code": "B1.2"}}
    base.update(kw)
    return base


def test_rollup_unions_and_worst_timeliness():
    rollup = rollup_from_metas([
        _entry_meta(),
        _entry_meta(entry_id="E-2", stage="S3", stage_ext=["S4"], domain_ext=["Z1"],
                    geo_country=["VNM"], timeliness="M1", business={"code": "B2.1"}),
    ], "2026-07-22")
    assert rollup["stage"] == ["S2", "S3", "S4"]
    assert rollup["domain"] == ["G1", "Z1"]
    assert rollup["country"] == ["IDN", "VNM"]
    assert rollup["business"] == ["B1.2", "B2.1"]
    assert rollup["timeliness_worst"] == "M1"
    assert rollup["entry_count"] == 2


def test_rollup_skips_non_entries_and_empty():
    # 非条目(无 entry_id)不计入;全非条目 → None
    assert rollup_from_metas([{"stage": "S2"}], "2026-07-22") is None
    rollup = rollup_from_metas([{"stage": "S1"}, _entry_meta()], "2026-07-22")
    assert rollup["entry_count"] == 1 and rollup["stage"] == ["S2"]


def test_apply_rollup_change_detection():
    meta = {}
    r1 = rollup_from_metas([_entry_meta()], "2026-07-01")
    assert apply_rollup(meta, r1) is True
    # 仅 computed_at 不同 → 视为无变化
    r2 = rollup_from_metas([_entry_meta()], "2026-07-22")
    assert apply_rollup(meta, r2) is False
    # 移除
    assert apply_rollup(meta, None) is True and "facet_rollup" not in meta
    assert apply_rollup(meta, None) is False


async def _init_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES ('ws1','t','','u1')")
    await db.commit()
    return db


async def _insert(db, doc_id, filename, path, kind, relative, meta=None):
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, tags, metadata) "
        "VALUES (?, 'u1', ?, ?, ?, ?, ?, 'md', 'ready', '[]', ?)",
        (doc_id, filename, filename, path, relative, kind,
         json.dumps(meta, ensure_ascii=False) if meta else None))
    await db.commit()


@pytest.mark.asyncio
async def test_refresh_rollups_local_end_to_end():
    db = await _init_db()
    try:
        await _insert(db, "w1", "page.md", "/wiki/", "wiki", "wiki/page.md")
        await _insert(db, "e1", "条目.md", "/corpus/S2-G1/", "source",
                      "corpus/S2-G1/条目.md", _entry_meta())
        await _insert(db, "s1", "raw.pdf", "/", "source", "raw.pdf")
        await db.execute(
            "INSERT INTO document_references (source_document_id, target_document_id, reference_type) "
            "VALUES ('w1','e1','cites'), ('w1','s1','cites')")
        await db.commit()

        assert await refresh_rollups_local(db) == 1
        cursor = await db.execute("SELECT metadata FROM documents WHERE id='w1'")
        meta = json.loads((await cursor.fetchone())[0])
        assert meta["facet_rollup"]["stage"] == ["S2"]
        assert meta["facet_rollup"]["entry_count"] == 1  # 原始文件不计入

        # 幂等:无变化不再计数
        assert await refresh_rollups_local(db) == 0

        # 取消条目引用 → rollup 移除
        await db.execute("DELETE FROM document_references WHERE target_document_id='e1'")
        await db.commit()
        assert await refresh_rollups_local(db) == 1
        cursor = await db.execute("SELECT metadata FROM documents WHERE id='w1'")
        meta = json.loads((await cursor.fetchone())[0])
        assert "facet_rollup" not in meta
    finally:
        await db.close()
