from datetime import UTC, datetime
from uuid import uuid4

import pytest
from services.read_hosted import HostedReadService
from services.read_local import StaleReadCursor


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
