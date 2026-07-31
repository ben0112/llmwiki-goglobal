import pytest

from infra.db.sqlite import create_pool
from services.read_local import LocalReadService


ROW_COUNT = 75_000
PAGE_LIMIT = 200


def _browse_plan_sql(*, second_page: bool) -> str:
    cursor = " AND (filename COLLATE NOCASE,id) > (?,?)" if second_page else ""
    return (
        "EXPLAIN QUERY PLAN SELECT id,filename,title,path,file_type,status,file_size,"
        "page_count,tags,date,metadata,error_message,version,document_number,stale_since,"
        "created_at,updated_at FROM documents WHERE path=?"
        f"{cursor} ORDER BY filename COLLATE NOCASE,id LIMIT ?"
    )


@pytest.mark.asyncio
async def test_large_local_browse_is_index_bounded_and_serialization_is_small(tmp_path):
    db = await create_pool(str(tmp_path / "index.db"))
    try:
        await db.execute("INSERT INTO workspace(id,name,user_id) VALUES('ws1','benchmark','u1')")
        await db.executemany(
            "INSERT INTO documents(id,user_id,filename,path,relative_path,source_kind,file_type,"
            "status,document_number,title) VALUES(?,'u1',?,'/',?,'source','md','ready',?,?)",
            (
                (
                    f"d{index:05d}",
                    f"file-{index:05d}.md",
                    f"file-{index:05d}.md",
                    index + 1,
                    f"File {index:05d}",
                )
                for index in range(ROW_COUNT)
            ),
        )
        await db.commit()

        first_plan = await db.execute_fetchall(_browse_plan_sql(second_page=False), ("/", PAGE_LIMIT + 1))
        second_plan = await db.execute_fetchall(
            _browse_plan_sql(second_page=True),
            ("/", "file-00199.md", "d00199", PAGE_LIMIT + 1),
        )
        details = [str(row[3]) for row in [*first_plan, *second_plan]]
        assert all("idx_documents_browse_name" in detail for detail in details)
        assert not any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in details)

        service = LocalReadService(db, "u1")
        page = await service.browse(
            "ws1", path="/", query=None, sort="name", direction="asc", limit=PAGE_LIMIT, cursor=None,
        )
        assert len(page.items) == PAGE_LIMIT
        assert len(page.model_dump_json().encode("utf-8")) < 1_048_576

        statements: list[str] = []
        await db.set_trace_callback(statements.append)
        await service.browse(
            "ws1", path="/", query=None, sort="name", direction="asc", limit=PAGE_LIMIT, cursor=None,
        )
        assert not any("count(*) FILTER" in statement for statement in statements)
        assert not any("GROUP BY child" in statement for statement in statements)

        with pytest.raises(ValueError, match="200"):
            await service.statuses("ws1", ids=[f"d{index:05d}" for index in range(PAGE_LIMIT + 1)])
    finally:
        await db.close()
