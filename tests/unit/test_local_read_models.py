import hashlib

import pytest
from infra.db.sqlite import create_pool
from services.local import LocalServiceFactory
from services.read_local import LocalReadService, StaleReadCursor


@pytest.fixture
async def read_service(tmp_path):
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws1','test','u1')")
    duplicate_digest = hashlib.sha256(b"same").hexdigest()
    rows = [
        ("d1", "Alpha.md", "/", "Alpha.md", "source", "md", "ready", 1, duplicate_digest, "2026-01-01"),
        ("d2", "alpha.pdf", "/", "alpha.pdf", "source", "pdf", "ready", 2, "other", "2026-01-01"),
        ("d3", "Beta.md", "/", "Beta.md", "source", "md", "failed", 3, None, "2026-01-02"),
        ("d4", "Zeta.md", "/", "Zeta.md", "source", "md", "ready", 4, None, "2026-01-02"),
        ("w1", "Overview.md", "/wiki/", "wiki/Overview.md", "wiki", "md", "ready", 5, None, "2026-01-03"),
        ("f1", "nested.md", "/folder/", "folder/nested.md", "source", "md", "ready", 6, None, "2026-01-04"),
        ("f2", "deep.md", "/folder/deep/", "folder/deep/deep.md", "source", "md", "ready", 7, None, "2026-01-05"),
    ]
    await db.executemany(
        "INSERT INTO documents"
        "(id,user_id,filename,path,relative_path,source_kind,file_type,status,"
        "document_number,content_hash,updated_at,content,title) "
        "VALUES(?, 'u1', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'large body', ?)",
        [(*row, row[1].rsplit(".", 1)[0]) for row in rows],
    )
    await db.commit()
    service = LocalReadService(db, "u1")
    yield service, db
    await db.close()


@pytest.mark.asyncio
async def test_browse_uses_stable_bounded_keyset_and_narrow_projection(read_service):
    service, _ = read_service
    first = await service.browse("ws1", path="/", query=None, sort="name", direction="asc", limit=2, cursor=None)
    second = await service.browse(
        "ws1",
        path="/",
        query=None,
        sort="name",
        direction="asc",
        limit=2,
        cursor=first.next_cursor,
    )
    items = [*first.items, *second.items]
    assert [item.id for item in items] == ["d1", "d2", "d3", "d4"]
    assert len({item.id for item in items}) == 4
    assert all("content" not in item.model_dump() for item in items)
    assert [folder.path for folder in first.folders] == ["/folder/", "/wiki/"]
    assert (first.source_count, first.failed_count, first.corpus_count) == (6, 1, 0)


@pytest.mark.asyncio
async def test_browse_rejects_empty_workspace_wide_search_and_stale_cursor(read_service):
    service, db = read_service
    with pytest.raises(ValueError, match="query"):
        await service.browse("ws1", path=None, query=" ", sort="name", direction="asc", limit=10, cursor=None)
    first = await service.browse("ws1", path="/", query=None, sort="name", direction="asc", limit=2, cursor=None)
    await db.execute("UPDATE documents SET title='changed' WHERE id='d1'")
    await db.commit()
    with pytest.raises(StaleReadCursor):
        await service.browse(
            "ws1",
            path="/",
            query=None,
            sort="name",
            direction="asc",
            limit=2,
            cursor=first.next_cursor,
        )


@pytest.mark.asyncio
async def test_wiki_resolver_status_and_preflight_are_bounded_and_ordered(read_service):
    service, _ = read_service
    wiki = await service.wiki_pages("ws1", limit=20, cursor=None)
    assert [item.id for item in wiki.items] == ["w1"]
    assert (await service.resolve("ws1", document_number=5)).id == "w1"
    assert (await service.resolve("ws1", logical_reference="Overview.md")).id == "w1"

    statuses = await service.statuses("ws1", document_numbers=[4, 1, 999])
    assert [item.document_number for item in statuses.items] == [4, 1]

    digest = hashlib.sha256(b"different").hexdigest()
    response = await service.upload_preflight(
        "ws1",
        [
            {"path": "/", "filename": "Alpha.md", "size": 1, "sha256": None},
            {"path": "/new/", "filename": "new.md", "size": 4, "sha256": hashlib.sha256(b"same").hexdigest()},
            {"path": "/new/", "filename": "ok.md", "size": 4, "sha256": digest},
            {"path": "/new/", "filename": "bad.exe", "size": 4, "sha256": None},
            {"path": "/new/", "filename": "huge.pdf", "size": 1_073_741_825, "sha256": None},
        ],
    )
    assert [item.code for item in response.items] == [
        "duplicate_name",
        "duplicate_content",
        "accepted",
        "unsupported",
        "too_large",
    ]


def test_local_factory_exposes_user_scoped_read_service():
    marker = object()
    service = LocalServiceFactory(marker).read_service("user-1")
    assert isinstance(service, LocalReadService)
    assert service.db is marker
    assert service.user_id == "user-1"
