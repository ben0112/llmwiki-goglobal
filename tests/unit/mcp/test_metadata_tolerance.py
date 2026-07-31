"""metadata 坏行容错:一条非法 JSON/BLOB 行不得让 MCP 列表/分面查询整体报错。

线上实测:文件夹批量上传后出现 metadata 非法 JSON 的文档行,
json_extract 使 search(mode=list) 直接 "malformed JSON" 报错,
MCP 客户端因此拿不到任何源文件。
"""

import sqlite3
from pathlib import Path

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


def test_facet_rollup_adapters_share_core_functions():
    """API 与 MCP 适配器必须复用共享内核语义。"""
    import importlib.util

    from llmwiki_core.facets import apply_rollup, rollup_from_metas
    from vaultfs import facet_rollup as mcp_rollup

    api_path = Path(__file__).parents[3] / "api" / "services" / "facet_rollup.py"
    spec = importlib.util.spec_from_file_location("api_facet_rollup_adapter", api_path)
    assert spec is not None and spec.loader is not None
    api_rollup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_rollup)

    assert api_rollup.apply_rollup is apply_rollup
    assert api_rollup.rollup_from_metas is rollup_from_metas
    assert mcp_rollup.apply_rollup is apply_rollup
    assert mcp_rollup.rollup_from_metas is rollup_from_metas


def test_wiki_coverage_summary():
    """P2A:有语料的货架格/业务场景 × 维基页 rollup 的覆盖与缺口。"""
    from tools.lint import wiki_coverage_summary

    entries = [
        {"stage": "S2", "domain": "G1", "business": {"code": "B1.2"}},
        {"stage": "S2", "domain": "G1"},
        {"stage": "S3", "domain": "C2", "business": {"code": "B2.1"}},
    ]
    rollups = [{"stage": ["S2"], "domain": ["G1"], "business": ["B1.2"]}]
    s = wiki_coverage_summary(entries, rollups)
    assert s["cells_total"] == 2 and s["cells_covered"] == 1
    assert s["cell_gaps"] == ["S3×C(1条)"]
    assert s["scenes_total"] == 2 and s["scenes_covered"] == 1
    assert s["scene_gaps"] == ["B2.1(1条)"]

    # 无维基页 → 全缺口;无语料 → 全零
    s2 = wiki_coverage_summary(entries, [])
    assert s2["cells_covered"] == 0 and len(s2["cell_gaps"]) == 2
    s3 = wiki_coverage_summary([], rollups)
    assert s3["cells_total"] == 0 and s3["scene_gaps"] == []
