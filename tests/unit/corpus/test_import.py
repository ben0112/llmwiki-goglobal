"""End-to-end importer: CSV → markdown entries + SQLite index rows + reports."""

import csv
import json
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import pytest

from corpus.import_annotations import SUBDIR, run

TODAY = date(2026, 7, 20)
REPO_ROOT = Path(__file__).resolve().parents[3]

COLUMNS = ["entry_id", "relpath", "source", "title", "url", "阶段", "服务大类", "体裁",
           "隐性规则", "证据", "来源", "归口", "国别区域", "行业形态", "时效", "置信度",
           "理由", "建议消费", "业务码", "业务场景", "业务需求类", "业务优先级", "业务待定"]

ROWS = [
    {"relpath": "12_山东/odi_notice.txt", "source": "12_山东",
     "title": "境外投资备案无纸化管理通知", "url": "https://example.gov.cn/odi",
     "阶段": "S2", "服务大类": "G1(副C1)", "体裁": "政策法规", "隐性规则": "R0",
     "证据": "E1", "来源": "国内", "归口": "商务委", "国别区域": "通用",
     "行业形态": "通用/通用", "时效": "M2", "置信度": "高",
     "理由": "国家政策文件", "建议消费": "出海导办·ODI预审", "业务码": "B1.2"},
    {"relpath": "01_丝路基金/colombia_road.txt", "source": "01_丝路基金",
     "title": "哥伦比亚公路项目", "url": "",
     "阶段": "S2④(副S3⑤)", "服务大类": "O1金融(副Z2)", "体裁": "案例经验",
     "隐性规则": "R0", "证据": "E2", "来源": "国内", "归口": "其他",
     "国别区域": "拉美·哥伦比亚", "行业形态": "工程承包/产能", "时效": "M3/已发布",
     "置信度": "0.8", "业务码": "B7.27"},
    {"relpath": "09_未知站点/junk.txt", "source": "09_未知站点",
     "title": "某未分类资讯", "url": "",
     "阶段": "S0", "服务大类": "X9", "体裁": "普通资讯", "隐性规则": "R0",
     "证据": "E4", "来源": "混合", "归口": "", "国别区域": "通用",
     "行业形态": "通用/通用", "时效": "M3", "置信度": "低", "业务码": "待定",
     "业务待定": "是"},
]

SOURCE_TEXT = """标题: 境外投资备案无纸化管理通知
URL: https://example.gov.cn/odi
命中关键词: 境外投资,备案

为进一步优化境外投资管理服务,现就备案(核准)无纸化管理有关事项通知如下。
一、全面推行无纸化申报。
"""


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".llmwiki").mkdir(parents=True)
    schema_sql = (REPO_ROOT / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    conn.executescript(schema_sql)
    conn.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'ws', '', ?)",
        (str(uuid.uuid4()), str(uuid.uuid5(uuid.NAMESPACE_DNS, "local"))),
    )
    conn.commit()
    conn.close()
    return ws


@pytest.fixture()
def raw_root(tmp_path):
    raw = tmp_path / "收录"
    (raw / "12_山东").mkdir(parents=True)
    (raw / "12_山东" / "odi_notice.txt").write_text(SOURCE_TEXT, encoding="utf-8")
    return raw


@pytest.fixture()
def csv_path(tmp_path):
    path = tmp_path / "标注明细_业务视图.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in ROWS:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    return path


def db(ws):
    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    conn.row_factory = sqlite3.Row
    return conn


def test_import_creates_files_and_rows(workspace, raw_root, csv_path):
    stats = run(csv_path, workspace, raw_root=raw_root, today=TODAY)
    assert stats.total == 3 and stats.imported == 3
    assert stats.review == 1  # the X9/低置信 row

    entry_files = sorted((workspace / SUBDIR).rglob("*.md"))
    assert len(entry_files) == 3
    shelves = {p.parent.name for p in entry_files}
    assert shelves == {"S2-G1", "S2-O1", "S0-X9"}

    # Entry with raw source: body embedded, header block stripped, frontmatter present
    odi = next(p for p in entry_files if p.parent.name == "S2-G1")
    text = odi.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert '"S2"' in text and "entry_id" in text
    assert "全面推行无纸化申报" in text
    assert "标题: 境外投资备案" not in text  # 采集头部已剥离
    assert "[12_山东](https://example.gov.cn/odi)" in text

    conn = db(workspace)
    rows = conn.execute("SELECT * FROM documents WHERE relative_path LIKE 'corpus/%'").fetchall()
    assert len(rows) == 3
    by_shelf = {r["relative_path"].split("/")[1]: r for r in rows}
    meta = json.loads(by_shelf["S2-G1"]["metadata"])
    assert meta["stage"] == "S2" and meta["domain"] == "G1" and meta["domain_ext"] == ["C1"]
    assert meta["review_due"] == "2026-10-20"
    assert meta["business"]["code"] == "B1.2" and meta["business"]["priority"] == "P2"
    tags = json.loads(by_shelf["S2-G1"]["tags"])
    assert "G1" in tags and "B1.2" in tags
    assert by_shelf["S2-G1"]["source_kind"] == "source"
    assert by_shelf["S2-G1"]["parser"] is None  # reconcile will chunk it
    assert by_shelf["S0-X9"]["metadata"] and json.loads(by_shelf["S0-X9"]["metadata"])["lifecycle_state"] == "待复核"
    numbers = [r["document_number"] for r in rows]
    assert len(set(numbers)) == 3
    conn.close()

    report_dir = workspace / ".llmwiki" / "corpus_import"
    report = (report_dir / "导入报告.md").read_text(encoding="utf-8")
    assert "覆盖率账本" in report and "| S2 |" in report
    with open(report_dir / "复核队列.csv", encoding="utf-8-sig") as f:
        review_rows = list(csv.DictReader(f))
    assert len(review_rows) == 1 and review_rows[0]["服务大类"] == "X9"


def test_reimport_is_idempotent(workspace, raw_root, csv_path):
    run(csv_path, workspace, raw_root=raw_root, today=TODAY)
    conn = db(workspace)
    before = {r["relative_path"]: (r["content_hash"], r["version"])
              for r in conn.execute("SELECT * FROM documents").fetchall()}
    conn.close()

    stats = run(csv_path, workspace, raw_root=raw_root, today=TODAY)
    assert stats.imported == 3
    conn = db(workspace)
    after = {r["relative_path"]: (r["content_hash"], r["version"])
             for r in conn.execute("SELECT * FROM documents").fetchall()}
    conn.close()
    assert before == after  # same content → same hash, version untouched


def test_reimport_updated_row_bumps_version(workspace, raw_root, csv_path, tmp_path):
    run(csv_path, workspace, raw_root=raw_root, today=TODAY)

    updated = [dict(r) for r in ROWS]
    updated[0]["title"] = "境外投资备案无纸化管理通知(修订版)"
    path2 = tmp_path / "v2.csv"
    with open(path2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in updated:
            w.writerow({c: row.get(c, "") for c in COLUMNS})

    run(path2, workspace, raw_root=raw_root, today=TODAY)
    conn = db(workspace)
    row = conn.execute(
        "SELECT title, version FROM documents WHERE relative_path LIKE 'corpus/S2-G1/%'"
    ).fetchone()
    assert row["title"] == "境外投资备案无纸化管理通知(修订版)"
    assert row["version"] == 1
    assert conn.execute("SELECT COUNT(*) FROM documents WHERE relative_path LIKE 'corpus/%'").fetchone()[0] == 3
    conn.close()


def test_dry_run_writes_nothing(workspace, csv_path):
    stats = run(csv_path, workspace, dry_run=True, today=TODAY)
    assert stats.total == 3
    assert not (workspace / SUBDIR).exists()
    conn = db(workspace)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    conn.close()
    # 校验报告仍然产出(在 CSV 同目录)
    assert (csv_path.parent / "corpus_import_dryrun" / "导入报告.md").exists()


def test_missing_workspace_errors(tmp_path, csv_path):
    with pytest.raises(SystemExit, match="init"):
        run(csv_path, tmp_path / "nowhere", today=TODAY)
