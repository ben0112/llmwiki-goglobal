import pytest


@pytest.mark.asyncio
async def test_find_inconsistent_ready_documents(pool, seed_pending_document):
    from infra.db.derived_documents import find_inconsistent_ready_documents

    good = await seed_pending_document(status="ready", version=2)
    bad = await seed_pending_document(status="ready", version=3)
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'good','good',1,2)",
        good["id"], good["user_id"], good["knowledge_base_id"],
    )
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'stale','stale',1,2)",
        bad["id"], bad["user_id"], bad["knowledge_base_id"],
    )

    rows = await find_inconsistent_ready_documents(pool)

    assert {row["id"] for row in rows} == {bad["id"]}


@pytest.mark.asyncio
async def test_audit_missing_chunks_excludes_short_text_and_assets(
    pool,
    seed_pending_document,
):
    from infra.db.derived_documents import find_inconsistent_ready_documents

    missing = await seed_pending_document(status="ready", version=1)
    short = await seed_pending_document(status="ready", version=1)
    asset = await seed_pending_document(status="ready", version=1)
    await pool.execute(
        "UPDATE documents SET content = $2 WHERE id = $1",
        missing["id"],
        "chunkable content " * 20,
    )
    await pool.execute(
        "UPDATE documents SET content = 'short' WHERE id = $1",
        short["id"],
    )
    await pool.execute(
        "UPDATE documents SET content = $2, source_kind = 'asset' WHERE id = $1",
        asset["id"],
        "asset content " * 20,
    )

    found = {row["id"] for row in await find_inconsistent_ready_documents(pool)}

    assert missing["id"] in found
    assert short["id"] not in found
    assert asset["id"] not in found


@pytest.mark.asyncio
async def test_reset_inconsistent_ready_documents_is_selective(
    pool,
    seed_pending_document,
):
    from infra.db.derived_documents import reset_inconsistent_ready_documents

    good = await seed_pending_document(status="ready", version=4)
    bad = await seed_pending_document(status="ready", version=4)
    for document, derived_version in ((good, 4), (bad, 3)):
        await pool.execute(
            "INSERT INTO document_pages "
            "(document_id,page,content,document_version) VALUES ($1,1,'page',$2)",
            document["id"],
            derived_version,
        )

    repaired = await reset_inconsistent_ready_documents(pool)

    repaired_ids = {row["id"] for row in repaired}
    assert bad["id"] in repaired_ids
    states = dict(
        await pool.fetch(
            "SELECT id, status::text FROM documents WHERE id = ANY($1::uuid[])",
            [good["id"], bad["id"]],
        )
    )
    assert states == {good["id"]: "ready", bad["id"]: "pending"}


@pytest.mark.asyncio
async def test_hosted_startup_repairs_before_recovery_scan(
    pool,
    seed_pending_document,
):
    from main import _repair_hosted_derived_drift

    bad = await seed_pending_document(status="ready", version=5)
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'stale','stale',1,4)",
        bad["id"], bad["user_id"], bad["knowledge_base_id"],
    )

    repaired = await _repair_hosted_derived_drift(pool)

    assert bad["id"] in {row["id"] for row in repaired}
    assert await pool.fetchval(
        "SELECT status::text FROM documents WHERE id = $1",
        bad["id"],
    ) == "pending"
