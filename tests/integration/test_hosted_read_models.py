import json
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from services.read_hosted import HostedReadService
from services.read_local import StaleReadCursor


def _corpus_meta(entry_id: str, stage: str, domain: str, state: str = "已入库") -> str:
    return json.dumps(
        {
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
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_hosted_read_adapter_is_stable_bounded_and_tenant_scoped(pool):
    user_id, other_id, kb_id = uuid4(), uuid4(), uuid4()
    await pool.executemany(
        "INSERT INTO users(id,email) VALUES($1,$2)",
        [(user_id, f"{user_id}@read.test"), (other_id, f"{other_id}@read.test")],
    )
    await pool.execute(
        "INSERT INTO knowledge_bases(id,user_id,name,slug) VALUES($1,$2,'Read','read-$1')",
        kb_id,
        user_id,
    )
    document_ids = [uuid4() for _ in range(4)]
    for index, (document_id, filename) in enumerate(
        zip(document_ids, ["Alpha.md", "alpha.pdf", "Beta.md", "Zeta.md"], strict=True),
        1,
    ):
        await pool.execute(
            "INSERT INTO documents"
            "(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,"
            "document_number,updated_at) VALUES($1,$2,$3,$4,'/','source','md','ready',$5,$6)",
            document_id,
            kb_id,
            user_id,
            filename,
            index,
            datetime(2026, 1, 1 if index < 3 else 2, tzinfo=UTC),
        )

    service = HostedReadService(pool, str(user_id))
    first = await service.browse(str(kb_id), path="/", query=None, sort="name", direction="asc", limit=2, cursor=None)
    second = await service.browse(
        str(kb_id),
        path="/",
        query=None,
        sort="name",
        direction="asc",
        limit=2,
        cursor=first.next_cursor,
    )
    assert [item.id for item in [*first.items, *second.items]] == [str(value) for value in document_ids]
    assert (await service.resolve(str(kb_id), document_number=2)).id == str(document_ids[1])
    statuses = await service.statuses(str(kb_id), document_numbers=[4, 1])
    assert [item.document_number for item in statuses.items] == [4, 1]
    preflight = await service.upload_preflight(
        str(kb_id),
        [
            {"path": "/", "filename": "Alpha.md", "size": 1},
            {"path": "/", "filename": "new.md", "size": 1, "sha256": "a" * 64},
        ],
    )
    assert [item.code for item in preflight.items] == ["duplicate_name", "accepted"]

    await pool.execute("UPDATE documents SET title='changed' WHERE id=$1", document_ids[0])
    with pytest.raises(StaleReadCursor):
        await service.browse(
            str(kb_id),
            path="/",
            query=None,
            sort="name",
            direction="asc",
            limit=2,
            cursor=first.next_cursor,
        )
    with pytest.raises(LookupError):
        await HostedReadService(pool, str(other_id)).revision(str(kb_id))


@pytest.mark.asyncio
async def test_hosted_corpus_entries_summary_and_graph_are_db_filtered_and_kb_scoped(pool):
    user_id, other_id, kb_id, other_kb_id = uuid4(), uuid4(), uuid4(), uuid4()
    await pool.executemany(
        "INSERT INTO users(id,email) VALUES($1,$2)",
        [(user_id, f"{user_id}@corpus.test"), (other_id, f"{other_id}@corpus.test")],
    )
    await pool.executemany(
        "INSERT INTO knowledge_bases(id,user_id,name,slug) VALUES($1,$2,$3,$4)",
        [
            (kb_id, user_id, "Corpus", f"corpus-{kb_id}"),
            (other_kb_id, other_id, "Other corpus", f"other-{other_kb_id}"),
        ],
    )
    doc_ids = [uuid4() for _ in range(3)]
    wiki_doc_id = uuid4()
    other_doc_id = uuid4()
    await pool.executemany(
        "INSERT INTO documents(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,metadata) "
        "VALUES($1,$2,$3,$4,'/corpus/','source','md','ready',$5::jsonb)",
        [
            (doc_ids[0], kb_id, user_id, "one.md", _corpus_meta("e1", "S2", "G1", "待复核")),
            (doc_ids[1], kb_id, user_id, "two.md", _corpus_meta("e2", "S2", "C1", "待复核")),
            (doc_ids[2], kb_id, user_id, "three.md", _corpus_meta("e3", "S1", "G1")),
            (
                wiki_doc_id,
                kb_id,
                user_id,
                "guide.md",
                json.dumps({"facet_rollup": {"stage": ["S2"], "domain": ["G1"]}}),
            ),
            (other_doc_id, other_kb_id, other_id, "other.md", _corpus_meta("e4", "S2", "G1")),
        ],
    )
    await pool.execute("UPDATE documents SET source_kind='wiki',path='/wiki/' WHERE id=$1", wiki_doc_id)
    await pool.executemany(
        "INSERT INTO document_references(id,source_document_id,target_document_id,knowledge_base_id,reference_type) "
        "VALUES($1,$2,$3,$4,'cites')",
        [
            (uuid4(), wiki_doc_id, doc_ids[0], kb_id),
            (uuid4(), wiki_doc_id, other_doc_id, kb_id),
        ],
    )

    service = HostedReadService(pool, str(user_id))
    page = await service.corpus_entries(
        str(kb_id),
        {"stage": "S2"},
        query="two",
        sort="domain",
        direction="desc",
        limit=1,
        cursor=None,
    )
    assert page.total_count == 1
    assert [item["id"] for item in page.items] == [str(doc_ids[1])]
    assert "document_number" in page.items[0]
    summary = await service.corpus_summary(str(kb_id), {"stage": "S2", "state": "待复核"})
    assert summary.filtered_count == 2
    assert summary.kpis["cited"] == 1
    assert summary.kpis["wiki_covered"] == 1
    assert summary.kpis["wiki_cells_with_entries"] == 2
    graph = await service.graph_summary(str(kb_id))
    assert graph.node_count == 4
    assert graph.edge_count == 1
    assert graph.cited_document_ids == [str(doc_ids[0])]
