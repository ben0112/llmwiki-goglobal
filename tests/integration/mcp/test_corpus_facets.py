"""Phase-2 corpus features on SqliteVaultFS: CJK search, facet filters,
FTS tokenizer migration, and 八维 lint checks + coverage ledger."""

import sqlite3
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Chunker drops content under MIN_CHUNK_TOKENS (~128 chars) — keep these long.
CN_POLICY = (
    "为进一步优化境外投资管理服务,现就境外投资备案(核准)无纸化管理有关事项通知如下。"
    "企业应当通过商务部业务系统统一平台在线提交备案申请材料,实行全程无纸化办理,"
    "不再要求企业报送纸质材料。各级商务主管部门应当及时受理并按规定时限办结,"
    "同时加强事中事后监管,确保境外投资备案管理工作平稳有序开展。"
    "本通知自发布之日起施行,此前规定与本通知不一致的,以本通知为准。"
)
CN_DATA = (
    "印尼监管机构近日发布数据本地化新规,要求跨境电商平台在印尼境内存储用户个人数据,"
    "并对数据出境活动实施更严格的审查。中资跨境电商企业需要评估数据出境合规风险,"
    "调整数据架构与存储方案,以满足当地监管执法要求。新规同时规定了过渡期安排,"
    "未按期完成整改的企业可能面临罚款、业务限制等处罚措施,建议企业尽快开展合规自查。"
)
EN_NOTES = (
    "Some plain research notes about data regulation without classification metadata. "
    "These notes discuss how different jurisdictions approach data residency, "
    "cross-border transfer mechanisms, and the compliance obligations that follow "
    "for companies operating internationally across multiple markets."
)


def corpus_meta(**overrides) -> dict:
    meta = {
        "spec_version": "v2026.06",
        "entry_id": "S2-G1-政策-GEN-AAAAA",
        "stage": "S2",
        "stage_ext": [],
        "domain": "G1",
        "domain_ext": [],
        "genre": "政策法规",
        "rule_type": ["R0"],
        "evidence": "E1",
        "origin": "国内",
        "gov_dept": ["商务委"],
        "geo_scope": "通用",
        "geo_region": [],
        "geo_country": [],
        "geo_country_names": [],
        "industry": ["通用"],
        "mode": [],
        "timeliness": "M2",
        "lifecycle_state": "已入库",
        "review_due": (date.today() + timedelta(days=90)).isoformat(),
        "confidence": 0.9,
        "business": {"code": "B1.2", "scene": "市场主体设立流程", "class": "市场准入与主体设立类", "priority": "P2"},
    }
    meta.update(overrides)
    return meta


@pytest.fixture
async def corpus_docs(fs):
    """Two classified corpus entries + one unclassified source, with content chunks."""
    instance, kb_id = fs
    odi = await instance.create_document(
        kb_id, "odi.md", "境外投资备案无纸化管理通知", "/corpus/S2-G1/", "md",
        CN_POLICY, ["S2", "G1"], metadata=corpus_meta(),
    )
    idn = await instance.create_document(
        kb_id, "idn.md", "印尼数据本地化新规", "/corpus/S3-Z1/", "md",
        CN_DATA, ["S3", "Z1"],
        metadata=corpus_meta(
            entry_id="S3-Z1-国别-IDN-BBBBB",
            stage="S3", stage_ext=["S2"],
            domain="Z1", domain_ext=["C1"],
            genre="国别研究", rule_type=["R1"],
            origin="目的地国", gov_dept=["网信办", "数据局"],
            geo_scope="单国", geo_region=["东盟"],
            geo_country=["IDN"], geo_country_names=["印尼"],
            industry=["跨境电商"],
            business={"code": "B4.14", "scene": "数据安全与隐私合规", "class": "本地化运营与合规管理类", "priority": "P1"},
        ),
    )
    plain = await instance.create_document(
        kb_id, "notes.md", "Plain notes", "/", "md", EN_NOTES, [],
    )
    return {"odi": odi, "idn": idn, "plain": plain}


# ---------------------------------------------------------------------------
# CJK full-text search (trigram + LIKE fallback)
# ---------------------------------------------------------------------------

async def test_cjk_search_three_char_query(fs, corpus_docs):
    instance, kb_id = fs
    rows = await instance.search_chunks(kb_id, "境外投资", 10)
    assert rows and any(r["filename"] == "odi.md" for r in rows)


async def test_cjk_search_two_char_query_falls_back_to_like(fs, corpus_docs):
    instance, kb_id = fs
    rows = await instance.search_chunks(kb_id, "备案", 10)  # 2 chars: no trigram
    assert rows and any(r["filename"] == "odi.md" for r in rows)


async def test_fts_operators_do_not_raise(fs, corpus_docs):
    instance, kb_id = fs
    # Previously raised sqlite OperationalError (raw MATCH string).
    for query in ('don"t', "NEAR(", "境外投资 AND", '"unbalanced'):
        rows = await instance.search_chunks(kb_id, query, 10)
        assert isinstance(rows, list)


async def test_english_search_still_works(fs, corpus_docs):
    instance, kb_id = fs
    rows = await instance.search_chunks(kb_id, "regulation", 10)
    assert rows and rows[0]["filename"] == "notes.md"


# ---------------------------------------------------------------------------
# Facet filtering
# ---------------------------------------------------------------------------

async def test_search_facet_domain(fs, corpus_docs):
    instance, kb_id = fs
    rows = await instance.search_chunks(kb_id, "数据", 10, facets={"domain": "Z1"})
    assert rows and all(r["filename"] == "idn.md" for r in rows)
    # Secondary domain label also matches.
    rows = await instance.search_chunks(kb_id, "数据", 10, facets={"domain": "C1"})
    assert rows and all(r["filename"] == "idn.md" for r in rows)


async def test_search_facet_excludes_unclassified(fs, corpus_docs):
    instance, kb_id = fs
    # "data" appears in the unclassified doc; facets restrict to corpus entries.
    rows = await instance.search_chunks(kb_id, "data", 10, facets={"origin": "目的地国"})
    assert all(r["filename"] != "notes.md" for r in rows)


async def test_search_facet_country_and_rule(fs, corpus_docs):
    instance, kb_id = fs
    by_iso = await instance.search_chunks(kb_id, "新规", 10, facets={"country": "IDN"})
    by_name = await instance.search_chunks(kb_id, "新规", 10, facets={"country": "印尼"})
    assert by_iso and by_name and by_iso[0]["filename"] == by_name[0]["filename"] == "idn.md"
    rows = await instance.search_chunks(kb_id, "新规", 10, facets={"rule": "R1"})
    assert rows and rows[0]["filename"] == "idn.md"
    assert await instance.search_chunks(kb_id, "新规", 10, facets={"rule": "R5"}) == []


async def test_search_facet_business_prefix(fs, corpus_docs):
    instance, kb_id = fs
    exact = await instance.search_chunks(kb_id, "数据", 10, facets={"business": "B4.14"})
    prefix = await instance.search_chunks(kb_id, "数据", 10, facets={"business": "B4"})
    assert exact and prefix and {r["filename"] for r in exact} == {r["filename"] for r in prefix} == {"idn.md"}


async def test_list_documents_facets(fs, corpus_docs):
    instance, kb_id = fs
    all_docs = await instance.list_documents(kb_id)
    m1 = await instance.list_documents(kb_id, facets={"timeliness": "M2"})
    stage3 = await instance.list_documents(kb_id, facets={"stage": "S3"})
    dept = await instance.list_documents(kb_id, facets={"dept": "数据局"})
    assert len(all_docs) >= 3
    assert {d["filename"] for d in m1} == {"odi.md", "idn.md"}
    assert {d["filename"] for d in stage3} == {"idn.md"}
    assert {d["filename"] for d in dept} == {"idn.md"}


async def test_unknown_facet_raises(fs, corpus_docs):
    from vaultfs.facets import UnknownFacetError
    instance, kb_id = fs
    with pytest.raises(UnknownFacetError, match="galaxy"):
        await instance.search_chunks(kb_id, "数据", 10, facets={"galaxy": "M31"})


# ---------------------------------------------------------------------------
# FTS tokenizer migration (porter unicode61 -> trigram)
# ---------------------------------------------------------------------------

async def test_legacy_porter_db_is_migrated(tmp_path):
    from vaultfs.sqlite import SqliteVaultFS

    ws = tmp_path / "legacy-ws"
    (ws / ".llmwiki").mkdir(parents=True)

    schema = (REPO_ROOT / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")
    legacy_schema = schema.replace("tokenize='trigram'", "tokenize='porter unicode61'")
    assert "porter unicode61" in legacy_schema

    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    conn.executescript(legacy_schema)
    doc_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, source_kind, "
        "file_type, status, content) VALUES (?, 'u', 'a.md', 'a', '/', 'a.md', 'source', 'md', 'ready', ?)",
        (doc_id, CN_DATA),
    )
    conn.execute(
        "INSERT INTO document_chunks (id, document_id, chunk_index, content, token_count) "
        "VALUES (?, ?, 0, ?, 10)",
        (str(uuid.uuid4()), doc_id, CN_DATA),
    )
    conn.commit()
    conn.close()

    await SqliteVaultFS.init(str(ws))
    try:
        db = SqliteVaultFS._db_or_raise()
        cur = await db.execute("SELECT sql FROM sqlite_master WHERE name='chunks_fts'")
        row = await cur.fetchone()
        assert "trigram" in row[0]

        instance = SqliteVaultFS("u")
        await instance.ensure_workspace("legacy")
        rows = await instance.search_chunks("any-kb", "数据本地化", 10)
        assert rows and rows[0]["filename"] == "a.md"
    finally:
        await SqliteVaultFS.close()


# ---------------------------------------------------------------------------
# Lint: 八维 completeness + code membership + coverage ledger
# ---------------------------------------------------------------------------

async def test_lint_clean_corpus_entries_and_coverage(fs, corpus_docs):
    from tools.lint import LintHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    report = await LintHandler(instance, kb).run(path="/corpus/**", scope="sources")
    assert "corpus-missing-dimension" not in report
    assert "corpus-unknown-code" not in report
    assert "覆盖率账本" in report and "| S2 |" in report
    assert "空格" in report  # sparsely-populated grid lists empty cells


async def test_lint_flags_bad_corpus_entries(fs):
    from tools.lint import LintHandler
    instance, kb_id = fs
    await instance.create_document(
        kb_id, "bad.md", "问题条目", "/corpus/S1-C2/", "md", "内容", [],
        metadata=corpus_meta(
            entry_id="S1-C2-政策-GEN-CCCCC",
            stage="S9",              # unknown code
            domain="C2",
            gov_dept=[],             # missing required dimension
            timeliness="M2",
            lifecycle_state="待复核",  # pending manual review
            review_due=(date.today() - timedelta(days=5)).isoformat(),  # overdue
        ),
    )
    kb = await instance.resolve_kb("test-workspace")
    report = await LintHandler(instance, kb).run(path="/corpus/**", scope="sources")
    assert "corpus-unknown-code" in report and "S9" in report
    assert "corpus-missing-dimension" in report and "gov_dept" in report
    assert "corpus-review-overdue" in report
    assert "corpus-pending-review" in report
    assert "待复核 1" in report


async def test_lint_ignores_unclassified_sources(fs, corpus_docs):
    from tools.lint import LintHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    report = await LintHandler(instance, kb).run(path="/notes.md", scope="sources")
    assert "corpus-" not in report and "覆盖率账本" not in report
