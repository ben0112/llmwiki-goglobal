"""Phase-4 relation layer + review worklist + lint KPIs on SqliteVaultFS."""

import sqlite3
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest

from .test_corpus_facets import CN_DATA, corpus_meta

REPO_ROOT = Path(__file__).resolve().parents[3]

WIKI_BODY = (
    "---\ntitle: 数据合规\ndescription: 数据出境合规要点\ndate: 2026-07-20\n"
    "tags: [数据, 合规]\n---\n\n# 数据合规\n\n"
    "跨境电商企业的数据出境合规要求正在收紧,须重点关注目的地国本地化存储义务。"
    "详见印尼监管案例的具体分析,并结合企业自身数据架构评估合规缺口。[^1]\n\n"
    "[^1]: idn.md\n"
)


@pytest.fixture
async def linked_docs(fs):
    """A corpus source, a governing-department page, and a wiki page citing the source."""
    instance, kb_id = fs
    idn = await instance.create_document(
        kb_id, "idn.md", "印尼数据本地化新规", "/corpus/S3-Z1/", "md",
        CN_DATA, ["S3", "Z1"],
        metadata=corpus_meta(
            entry_id="S3-Z1-国别-IDN-BBBBB", stage="S3", domain="Z1",
            genre="国别研究", rule_type=["R1"], origin="目的地国",
            gov_dept=["网信办"], geo_country=["IDN"], geo_country_names=["印尼"],
        ),
    )
    dept = await instance.create_document(
        kb_id, "cyberspace-admin.md", "网信办 (数据跨境专班)", "/wiki/depts/", "md",
        "---\ntitle: 网信办\ndescription: 数据出境评估归口\ndate: 2026-07-20\ntags: [部门, 数据]\n---\n\n"
        "# 网信办\n\n数据出境安全评估与个人信息出境标准合同的归口部门。",
        ["部门"],
    )
    wiki = await instance.create_document(
        kb_id, "data-compliance.md", "数据合规", "/wiki/concepts/", "md",
        WIKI_BODY, ["数据", "合规"],
    )
    # Build the derived citation edges the way the write tool does.
    from tools.references import update_references
    await update_references(instance, kb_id, str(wiki["id"]), WIKI_BODY, "/wiki/concepts/")
    return {"idn": idn, "dept": dept, "wiki": wiki}


# ---------------------------------------------------------------------------
# relate tool: add / remove / validation
# ---------------------------------------------------------------------------

async def test_relate_add_and_list(fs, linked_docs):
    from tools.relate import RelateHandler
    from tools.search import SearchHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")

    out = await RelateHandler(instance, kb).run(
        "/corpus/S3-Z1/idn.md", "/wiki/depts/cyberspace-admin.md", "governed_by", "add",
    )
    assert "governed_by (归口映射)" in out

    refs = await SearchHandler(instance, kb).query_references("/corpus/S3-Z1/idn.md", "")
    assert "Relations (1, curated)" in refs
    assert "归口映射" in refs and "cyberspace-admin.md" in refs

    # Backlink side shows the typed label too.
    back = await SearchHandler(instance, kb).query_references("/wiki/depts/cyberspace-admin.md", "")
    assert "governed_by (归口映射)" in back


async def test_relate_remove(fs, linked_docs):
    from tools.relate import RelateHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    handler = RelateHandler(instance, kb)

    await handler.run("idn.md", "cyberspace-admin.md", "governed_by", "add")
    out = await handler.run("idn.md", "cyberspace-admin.md", "governed_by", "remove")
    assert out.startswith("Removed:")
    out = await handler.run("idn.md", "cyberspace-admin.md", "governed_by", "remove")
    assert "No governed_by" in out


async def test_relate_rejects_self_and_missing(fs, linked_docs):
    from tools.relate import RelateHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    handler = RelateHandler(instance, kb)

    out = await handler.run("idn.md", "idn.md", "next", "add")
    assert "same document" in out
    out = await handler.run("idn.md", "nonexistent.md", "next", "add")
    assert "not found" in out


# ---------------------------------------------------------------------------
# Curated edges survive content-driven rebuilds
# ---------------------------------------------------------------------------

async def test_relations_survive_reference_rebuild(fs, linked_docs):
    from tools.references import update_references
    from tools.relate import RelateHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    wiki_id = str(linked_docs["wiki"]["id"])

    await RelateHandler(instance, kb).run(
        "/wiki/concepts/data-compliance.md", "/wiki/depts/cyberspace-admin.md", "governed_by", "add",
    )

    before = await instance.get_forward_references(wiki_id)
    assert {r["reference_type"] for r in before} == {"cites", "governed_by"}

    # Re-derive content edges (what every create/edit/append does).
    await update_references(instance, kb_id, wiki_id, WIKI_BODY, "/wiki/concepts/")

    after = await instance.get_forward_references(wiki_id)
    assert {r["reference_type"] for r in after} == {"cites", "governed_by"}, \
        "curated relation edge must survive the citation rebuild"


# ---------------------------------------------------------------------------
# Legacy DB migration: old CHECK constraint rebuilt in place
# ---------------------------------------------------------------------------

async def test_legacy_reference_check_is_migrated(tmp_path):
    from vaultfs.sqlite import SqliteVaultFS

    ws = tmp_path / "legacy-refs"
    (ws / ".llmwiki").mkdir(parents=True)
    schema = (REPO_ROOT / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")
    legacy = schema.replace(
        "CHECK (reference_type IN (\n        'cites', 'links_to',\n"
        "        'is_a', 'next', 'routes_to', 'governed_by', 'serves'\n    ))",
        "CHECK (reference_type IN ('cites', 'links_to'))",
    )
    assert "'governed_by'" not in legacy  # quoted form only lives in the CHECK

    conn = sqlite3.connect(str(ws / ".llmwiki" / "index.db"))
    conn.executescript(legacy)
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    for doc_id, name in ((a, "a.md"), (b, "b.md")):
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, source_kind, "
            "file_type, status) VALUES (?, 'u', ?, ?, '/', ?, 'source', 'md', 'ready')",
            (doc_id, name, name, name),
        )
    conn.execute(
        "INSERT INTO document_references (source_document_id, target_document_id, reference_type) "
        "VALUES (?, ?, 'cites')", (a, b),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO document_references (source_document_id, target_document_id, reference_type) "
            "VALUES (?, ?, 'governed_by')", (a, b),
        )
    conn.commit()
    conn.close()

    await SqliteVaultFS.init(str(ws))
    try:
        instance = SqliteVaultFS("u")
        await instance.ensure_workspace("legacy")
        # Old row preserved, new type now accepted.
        await instance.upsert_reference(a, b, "kb", "governed_by", None)
        refs = await instance.get_forward_references(a)
        assert {r["reference_type"] for r in refs} == {"cites", "governed_by"}
    finally:
        await SqliteVaultFS.close()


# ---------------------------------------------------------------------------
# Review worklist (query="due") and lint KPIs
# ---------------------------------------------------------------------------

async def test_due_worklist(fs, linked_docs):
    from tools.search import SearchHandler
    instance, kb_id = fs
    await instance.create_document(
        kb_id, "overdue.md", "过期条目", "/corpus/S2-C2/", "md", CN_DATA, [],
        metadata=corpus_meta(
            entry_id="S2-C2-政策-GEN-DDDDD", stage="S2", domain="C2",
            review_due=(date.today() - timedelta(days=3)).isoformat(),
        ),
    )
    await instance.create_document(
        kb_id, "pending.md", "待复核条目", "/corpus/S2-C2/", "md", CN_DATA, [],
        metadata=corpus_meta(
            entry_id="S2-C2-政策-GEN-EEEEE", stage="S2", domain="C2",
            lifecycle_state="待复核",
        ),
    )
    kb = await instance.resolve_kb("test-workspace")
    out = await SearchHandler(instance, kb).query_references("*", "due")
    assert "复审工作清单 (2 条)" in out
    assert "overdue.md" in out and "复审到期" in out
    assert "pending.md" in out and "待复核" in out
    # The healthy entry from linked_docs is not on the worklist.
    assert "idn.md" not in out


async def test_due_worklist_empty(fs, linked_docs):
    from tools.search import SearchHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    out = await SearchHandler(instance, kb).query_references("*", "due")
    assert "复审工作清单为空" in out


async def test_lint_kpis(fs, linked_docs):
    from tools.lint import LintHandler
    instance, kb_id = fs
    kb = await instance.resolve_kb("test-workspace")
    report = await LintHandler(instance, kb).run(path="*", scope="all")
    assert "KPI:" in report
    # One corpus entry, complete, on time, and cited by the wiki page.
    assert "分面完备率 100%" in report
    assert "时效达标率 100%" in report
    assert "引用溯源率 100%" in report
    assert "货架覆盖率 1/20格" in report
