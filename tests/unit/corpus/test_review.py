"""复核操作(L3-P3):通过/改标签(含货架迁移)/不收录、审计、取值校验。"""

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from .test_pipeline import POLICY_BODY, _init_ws, _insert_source


def _import_one(ws: Path) -> tuple[str, dict]:
    """跑一轮 mock 流水线,返回 (条目doc_id, metadata)。"""
    from corpus.llm import LLMConfig
    from corpus.pipeline import run_batch

    asyncio.run(run_batch(ws, LLMConfig(base_url="mock")))
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    row = conn.execute(
        "SELECT id, metadata FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()
    conn.close()
    assert row
    return row[0], json.loads(row[1])


def _entry_meta(ws: Path) -> dict:
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    row = conn.execute(
        "SELECT metadata FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()
    conn.close()
    return json.loads(row[0]) if row else {}


def test_approve_sets_state_and_audit(tmp_path):
    from corpus.review import apply_review

    ws = _init_ws(tmp_path)
    _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    doc_id, _ = _import_one(ws)

    out = apply_review(ws, doc_id, "approve", note="人工核对无误")
    assert out["moved"] is False
    meta = _entry_meta(ws)
    assert meta["lifecycle_state"] == "已入库"
    assert meta["review"][-1]["action"] == "approve"
    assert "人工核对无误" in meta["review"][-1]["note"]
    # 文件 frontmatter 同步
    text = (ws / out["relative_path"]).read_text(encoding="utf-8")
    assert "已入库" in text


def test_update_moves_shelf_and_rederives_business(tmp_path):
    from corpus.review import apply_review

    ws = _init_ws(tmp_path)
    _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    doc_id, meta = _import_one(ws)
    old_rel = [p for p in (ws / "corpus").rglob("*.md")][0]
    assert meta["stage"] == "S2" and meta["domain"] == "G1"

    out = apply_review(ws, doc_id, "update",
                       labels={"stage": "S1", "domain": "G2", "timeliness": "M1"},
                       note="人工改判")
    assert out["moved"] is True
    assert out["relative_path"].startswith("corpus/S1-G2/")
    assert not old_rel.exists()                      # 旧货架文件已迁移
    assert (ws / out["relative_path"]).exists()

    meta2 = _entry_meta(ws)
    assert meta2["stage"] == "S1" and meta2["domain"] == "G2"
    assert meta2["timeliness"] == "M1"
    assert meta2["business"]["code"] == "B6.25"      # M1 → 风险预警场景重推导
    assert meta2["entry_id"].startswith("S1-G2-")
    # 索引无重复条目
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    n = conn.execute("SELECT COUNT(*) FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()[0]
    conn.close()
    assert n == 1


def test_exclude_removes_entry_and_marks_source(tmp_path):
    from corpus.review import apply_review

    ws = _init_ws(tmp_path)
    src_id = _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    doc_id, _ = _import_one(ws)

    apply_review(ws, doc_id, "exclude", note="重复转载")
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    n = conn.execute("SELECT COUNT(*) FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()[0]
    state = conn.execute("SELECT state, error FROM corpus_pipeline WHERE doc_id = ?", (src_id,)).fetchone()
    conn.close()
    assert n == 0
    assert list((ws / "corpus").rglob("*.md")) == []
    assert state[0] == "excluded" and "重复转载" in state[1]


def test_update_rejects_off_codetable_values(tmp_path):
    from corpus.review import ReviewError, apply_review

    ws = _init_ws(tmp_path)
    _insert_source(ws, "0001_境外投资备案通知.txt", "境外投资备案(核准)无纸化通知", POLICY_BODY)
    doc_id, _ = _import_one(ws)

    with pytest.raises(ReviewError):
        apply_review(ws, doc_id, "update", labels={"stage": "S9"})
    with pytest.raises(ReviewError):
        apply_review(ws, doc_id, "update", labels={"entry_id": "X"})


def test_codetable_options_shape():
    from corpus.review import codetable_options

    opts = codetable_options()
    assert {o["code"] for o in opts["stages"]} == {"S0", "S1", "S2", "S3", "S4"}
    assert any(o["code"] == "R1" for o in opts["rules"])
    assert len(opts["domains"]) == 20 and len(opts["timeliness"]) == 3
