"""metadata 坏行容错:一条非法 JSON/BLOB 行不得让 MCP 列表/分面查询整体报错。

线上实测:文件夹批量上传后出现 metadata 非法 JSON 的文档行,
json_extract 使 search(mode=list) 直接 "malformed JSON" 报错,
Claude 因此拿不到任何源文件。
"""

import sqlite3
from pathlib import Path

import pytest

SCHEMA = (Path(__file__).parents[3] / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")


def _make_db() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    c.execute("INSERT INTO workspace (id, name, description, user_id) VALUES ('w1','t','','u1')")
    rows = [
        ("ok", "好行.md", '{"stage":"S2"}'),
        ("badjson", "坏行1.pdf", "not-json{broken"),
    ]
    for doc_id, fn, meta in rows:
        c.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, tags, metadata) "
            "VALUES (?, 'u1', ?, ?, '/', ?, 'source', 'md', 'ready', '[]', ?)",
            (doc_id, fn, fn, fn, meta),
        )
    # BLOB 变体
    c.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, tags, metadata) "
        "VALUES ('blob', 'u1', '坏行2.pdf', 'b', '/', '坏行2.pdf', 'source', 'md', 'ready', '[]', ?)",
        (b"\x80\x81binary",),
    )
    c.commit()
    return c


def test_asset_filter_survives_malformed_metadata():
    """list_documents 的 asset 过滤条件对坏行不报错、且不误过滤。"""
    c = _make_db()
    sql = (
        "SELECT id FROM documents d WHERE status != 'failed' "
        "AND COALESCE(CASE WHEN typeof(metadata)='text' AND json_valid(metadata) "
        "THEN json_extract(metadata, '$.asset') END, 0) != 1 "
        "ORDER BY id"
    )
    assert [r[0] for r in c.execute(sql)] == ["badjson", "blob", "ok"]


def test_facet_conditions_survive_malformed_metadata():
    from vaultfs.facets import sqlite_facet_conditions

    c = _make_db()
    conds, params = sqlite_facet_conditions({"stage": "S2"})
    sql = "SELECT id FROM documents d WHERE " + " AND ".join(conds)
    # 坏行不命中分面(而不是让整条查询报错)
    assert [r[0] for r in c.execute(sql, params)] == ["ok"]


def test_to_dir_path_always_leading_slash():
    from tools.write import WriteHandler

    h = WriteHandler.__new__(WriteHandler)
    assert h._to_dir_path("wiki/测试页.md") == "/wiki/"
    assert h._to_dir_path("/wiki/x.md") == "/wiki/"
    assert h._to_dir_path("wiki") == "/wiki/"
    assert h._to_dir_path("/wiki/") == "/wiki/"
    assert h._to_dir_path("笔记.md") == "/"


def test_facet_rollup_helper_copies_in_sync():
    """mcp 与 api 两份 facet_rollup 拷贝必须一致(同 parse_citation 模式)。"""
    from pathlib import Path
    mcp_copy = Path(__file__).parents[3] / "mcp" / "vaultfs" / "facet_rollup.py"
    api_copy = Path(__file__).parents[3] / "api" / "services" / "facet_rollup.py"
    mcp_text = mcp_copy.read_text(encoding="utf-8")
    api_text = api_copy.read_text(encoding="utf-8")
    # api 侧在共同部分之后追加批量刷新;共同前缀必须逐字一致
    assert api_text.startswith(mcp_text.rstrip() + "\n")
