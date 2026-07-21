"""Hosted (Postgres) corpus import: documents + chunks + facet SQL + idempotency."""

import csv
import importlib.util
import json
import os
import uuid
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_URL = os.environ["DATABASE_URL"]
TODAY = date(2026, 7, 21)

USER_ID = str(uuid.uuid4())
USER_EMAIL = "corpus-admin@example.com"
KB_SLUG = "goglobal-corpus"

# Long enough to clear the chunker's MIN_CHUNK_TOKENS threshold.
CN_POLICY = (
    "为进一步优化境外投资管理服务,现就境外投资备案(核准)无纸化管理有关事项通知如下。"
    "企业应当通过商务部业务系统统一平台在线提交备案申请材料,实行全程无纸化办理,"
    "不再要求企业报送纸质材料。各级商务主管部门应当及时受理并按规定时限办结,"
    "同时加强事中事后监管,确保境外投资备案管理工作平稳有序开展。"
)
CN_DATA = (
    "印尼监管机构近日发布数据本地化新规,要求跨境电商平台在印尼境内存储用户个人数据,"
    "并对数据出境活动实施更严格的审查。中资跨境电商企业需要评估数据出境合规风险,"
    "调整数据架构与存储方案,以满足当地监管执法要求。新规同时规定了过渡期安排,"
    "未按期完成整改的企业可能面临罚款、业务限制等处罚措施。"
)

COLUMNS = ["entry_id", "relpath", "source", "title", "url", "阶段", "服务大类", "体裁",
           "隐性规则", "证据", "来源", "归口", "国别区域", "行业形态", "时效", "置信度",
           "理由", "建议消费", "业务码", "业务场景", "业务需求类", "业务优先级", "业务待定"]

ROWS = [
    {"relpath": "12_山东/odi_notice.txt", "source": "12_山东",
     "title": "境外投资备案无纸化管理通知", "url": "https://example.gov.cn/odi",
     "阶段": "S2", "服务大类": "G1(副C1)", "体裁": "政策法规", "隐性规则": "R0",
     "证据": "E1", "来源": "国内", "归口": "商务委", "国别区域": "通用",
     "行业形态": "通用/通用", "时效": "M2", "置信度": "高", "业务码": "B1.2"},
    {"relpath": "03_贸法通/idn_data.txt", "source": "03_贸法通",
     "title": "印尼数据本地化新规", "url": "https://example.com/idn",
     "阶段": "S3", "服务大类": "Z1(副C1)", "体裁": "国别研究", "隐性规则": "R1",
     "证据": "E2", "来源": "目的地国", "归口": "网信办", "国别区域": "东盟·印尼",
     "行业形态": "跨境电商/产品", "时效": "M2", "置信度": "0.85", "业务码": "B4.14"},
]


@pytest.fixture
async def seeded_user(pool):
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Corpus Admin') "
        "ON CONFLICT (id) DO NOTHING",
        USER_ID, USER_EMAIL,
    )
    yield
    await pool.execute("DELETE FROM knowledge_bases WHERE user_id = $1", USER_ID)
    await pool.execute("DELETE FROM users WHERE id = $1", USER_ID)


@pytest.fixture
def csv_path(tmp_path):
    raw = tmp_path / "收录"
    (raw / "12_山东").mkdir(parents=True)
    (raw / "12_山东" / "odi_notice.txt").write_text(
        f"标题: 境外投资备案无纸化管理通知\nURL: https://example.gov.cn/odi\n\n{CN_POLICY}\n",
        encoding="utf-8",
    )
    (raw / "03_贸法通").mkdir(parents=True)
    (raw / "03_贸法通" / "idn_data.txt").write_text(
        f"标题: 印尼数据本地化新规\nURL: https://example.com/idn\n\n{CN_DATA}\n",
        encoding="utf-8",
    )
    path = tmp_path / "标注明细_业务视图.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in ROWS:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    return path


async def _run(csv_path):
    from corpus.hosted_import import run_hosted
    return await run_hosted(
        csv_path=csv_path, database_url=DB_URL, user_email=USER_EMAIL,
        kb_slug=KB_SLUG, raw_root=csv_path.parent / "收录", today=TODAY,
    )


async def test_hosted_import_creates_kb_docs_chunks(pool, seeded_user, csv_path):
    stats = await _run(csv_path)
    assert stats.total == 2 and stats.imported == 2

    kb = await pool.fetchrow(
        "SELECT id FROM knowledge_bases WHERE user_id = $1 AND slug = $2", USER_ID, KB_SLUG,
    )
    assert kb is not None

    docs = await pool.fetch(
        "SELECT * FROM documents WHERE knowledge_base_id = $1 ORDER BY path", kb["id"],
    )
    assert len(docs) == 2
    by_path = {d["path"]: d for d in docs}
    idn = by_path["/corpus/S3-Z1/"]
    meta = json.loads(idn["metadata"])
    assert meta["stage"] == "S3" and meta["domain"] == "Z1"
    assert meta["geo_country"] == ["IDN"]
    assert meta["business"]["code"] == "B4.14"
    assert "Z1" in idn["tags"] and "印尼" in idn["tags"]  # TEXT[] facet tags
    assert idn["status"] == "ready"
    numbers = sorted(d["document_number"] for d in docs)
    assert numbers == [1, 2]  # set_document_number trigger fired

    for d in docs:
        assert "全程无纸化办理" in d["content"] or "数据本地化" in d["content"]
        n_chunks = await pool.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = $1", d["id"],
        )
        assert n_chunks >= 1
        assert await pool.fetchval(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = $1 "
            "AND (user_id != $2 OR knowledge_base_id != $3)",
            d["id"], USER_ID, kb["id"],
        ) == 0


async def test_hosted_import_idempotent_and_versioning(pool, seeded_user, csv_path, tmp_path):
    await _run(csv_path)
    kb_id = await pool.fetchval(
        "SELECT id FROM knowledge_bases WHERE user_id = $1 AND slug = $2", USER_ID, KB_SLUG,
    )
    before = {r["path"]: (r["content"], r["version"]) for r in await pool.fetch(
        "SELECT path, content, version FROM documents WHERE knowledge_base_id = $1", kb_id,
    )}

    stats = await _run(csv_path)  # identical re-run
    assert stats.imported == 2
    after = {r["path"]: (r["content"], r["version"]) for r in await pool.fetch(
        "SELECT path, content, version FROM documents WHERE knowledge_base_id = $1", kb_id,
    )}
    assert before == after
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE knowledge_base_id = $1", kb_id,
    ) == 2

    # Change one row's title -> version bump + chunk rebuild for that doc only.
    updated = [dict(r) for r in ROWS]
    updated[1]["title"] = "印尼数据本地化新规(修订)"
    path2 = tmp_path / "v2.csv"
    with open(path2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in updated:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    from corpus.hosted_import import run_hosted
    await run_hosted(csv_path=path2, database_url=DB_URL, user_email=USER_EMAIL,
                     kb_slug=KB_SLUG, raw_root=csv_path.parent / "收录", today=TODAY)
    row = await pool.fetchrow(
        "SELECT title, version FROM documents WHERE knowledge_base_id = $1 AND path = '/corpus/S3-Z1/'",
        kb_id,
    )
    assert row["title"] == "印尼数据本地化新规(修订)"
    assert row["version"] == 1


async def test_hosted_facet_sql_matches_entries(pool, seeded_user, csv_path):
    """The phase-2 Postgres facet SQL works against imported metadata."""
    await _run(csv_path)
    spec = importlib.util.spec_from_file_location(
        "facets_mod", REPO_ROOT / "mcp" / "vaultfs" / "facets.py",
    )
    facets_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(facets_mod)

    kb_id = await pool.fetchval(
        "SELECT id FROM knowledge_bases WHERE user_id = $1 AND slug = $2", USER_ID, KB_SLUG,
    )
    for facets, expected in [
        ({"domain": "Z1"}, {"/corpus/S3-Z1/"}),
        ({"domain": "C1"}, {"/corpus/S2-G1/", "/corpus/S3-Z1/"}),  # secondary label
        ({"country": "印尼"}, {"/corpus/S3-Z1/"}),
        ({"business": "B4"}, {"/corpus/S3-Z1/"}),
        ({"timeliness": "M2"}, {"/corpus/S2-G1/", "/corpus/S3-Z1/"}),
        ({"rule": "R5"}, set()),
    ]:
        conds, params = facets_mod.postgres_facet_conditions(
            facets_mod.validate_facets(facets), start_index=2, doc_alias="d",
        )
        sql = "SELECT d.path FROM documents d WHERE d.knowledge_base_id = $1"
        for c in conds:
            sql += f" AND {c}"
        rows = await pool.fetch(sql, kb_id, *params)
        assert {r["path"] for r in rows} == expected, facets


async def test_missing_user_fails_cleanly(pool, csv_path):
    from corpus.hosted_import import run_hosted
    with pytest.raises(SystemExit, match="不存在"):
        await run_hosted(csv_path=csv_path, database_url=DB_URL,
                         user_email="ghost@example.com", kb_slug=KB_SLUG, today=TODAY)
