import json
from datetime import date, timedelta

import pytest
from infra.db.sqlite import create_pool
from services.read_local import LocalReadService


def _meta(entry_id, stage, domain, state="已入库", **extra):
    value = {
        "spec_version": "v2026.06",
        "entry_id": entry_id,
        "stage": stage,
        "stage_ext": [],
        "domain": domain,
        "domain_ext": [],
        "genre": "政策法规",
        "rule_type": ["R0"],
        "evidence": "E1",
        "origin": "国内",
        "gov_dept": ["商务委"],
        "geo_region": [],
        "geo_country_names": [],
        "industry": ["通用"],
        "timeliness": "M2",
        "lifecycle_state": state,
        "review_due": (date.today() + timedelta(days=10)).isoformat(),
        "business": {"code": "B1.2", "scene": "设立", "class": "准入", "priority": "P2"},
    }
    value.update(extra)
    return json.dumps(value, ensure_ascii=False)


@pytest.fixture
async def corpus_service(tmp_path):
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws1','test','u1')")
    rows = [
        ("d1", "one.md", _meta("e1", "S2", "G1", "待复核")),
        ("d2", "two.md", _meta("e2", "S2", "C1", "待复核", stage_ext=["S1"])),
        ("d3", "three.md", _meta("e3", "S1", "G1")),
        (
            "weird",
            "weird.md",
            _meta("e4", "S4", "O1", stage_ext=42, domain_ext="not-an-array"),
        ),
        ("bad", "bad.md", "{not-json"),
    ]
    await db.executemany(
        "INSERT INTO documents(id,user_id,filename,path,relative_path,source_kind,file_type,status,metadata) "
        "VALUES(?,'u1',?,'/corpus/',?,'source','md','ready',?)",
        [(doc_id, filename, f"corpus/{filename}", metadata) for doc_id, filename, metadata in rows],
    )
    await db.execute(
        "INSERT INTO documents(id,user_id,filename,path,relative_path,source_kind,file_type,status,metadata) "
        "VALUES('w1','u1','guide.md','/wiki/','wiki/guide.md','wiki','md','ready',?)",
        (
            json.dumps(
                {"facet_rollup": {"stage": ["S2"], "domain": ["G1"]}},
                ensure_ascii=False,
            ),
        ),
    )
    await db.execute(
        "INSERT INTO document_references(id,source_document_id,target_document_id,reference_type) "
        "VALUES('r1','w1','d1','cites')"
    )
    await db.commit()
    yield LocalReadService(db, "u1"), db
    await db.close()


async def test_corpus_entries_summary_graph_and_cache_invalidation(corpus_service):
    service, db = corpus_service
    page = await service.corpus_entries("ws1", {"stage": "S2"}, limit=1, cursor=None)
    assert len(page.items) == 1 and page.next_cursor
    second = await service.corpus_entries("ws1", {"stage": "S2"}, limit=1, cursor=page.next_cursor)
    assert {page.items[0]["id"], second.items[0]["id"]} == {"d1", "d2"}
    queried = await service.corpus_entries(
        "ws1",
        {},
        query="two",
        sort="domain",
        direction="desc",
        limit=20,
        cursor=None,
    )
    assert queried.total_count == 1
    assert [item["id"] for item in queried.items] == ["d2"]

    summary = await service.corpus_summary("ws1", {"stage": "S2", "state": "待复核"})
    assert summary.total_count == 4
    assert summary.filtered_count == 2
    assert summary.facets["stage"]["S1"] == 1
    assert summary.facets["stage"]["S2"] == 2
    assert summary.kpis["pending_review"] == 2
    assert summary.kpis["cited"] == 1
    assert summary.kpis["wiki_covered"] == 1
    assert summary.kpis["wiki_cells_with_entries"] == 2
    assert summary.coverage["counts"]["S2"] == {"G": 1, "C": 1, "O": 0, "Z": 0, "X": 0}
    graph = await service.graph_summary("ws1")
    assert graph.edge_count == 1 and graph.cited_document_ids == ["d1"]

    await db.execute("UPDATE documents SET title='changed' WHERE id='d1'")
    await db.commit()
    refreshed = await service.corpus_summary("ws1", {"stage": "S2", "state": "待复核"})
    assert refreshed.revision > summary.revision
