import pytest


@pytest.mark.asyncio
async def test_replace_derived_content_commits_ready_with_matching_versions(
    seed_pending_document,
    pool,
):
    from infra.db.derived_documents import replace_derived_content
    from llmwiki_core.chunking import chunk_pages

    doc = await seed_pending_document(status="processing", version=3)
    pages = [(1, "Indonesia data localization requirements. " * 8)]

    version = await replace_derived_content(
        pool,
        document_id=doc["id"],
        user_id=doc["user_id"],
        knowledge_base_id=doc["knowledge_base_id"],
        pages=pages,
        chunks=chunk_pages(pages),
        parser="test",
    )

    row = await pool.fetchrow(
        "SELECT status, version, content, parser FROM documents WHERE id=$1",
        doc["id"],
    )
    page_versions = await pool.fetch(
        "SELECT DISTINCT document_version FROM document_pages WHERE document_id=$1",
        doc["id"],
    )
    chunk_versions = await pool.fetch(
        "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=$1",
        doc["id"],
    )
    assert version == 4
    assert dict(row) == {
        "status": "ready",
        "version": 4,
        "content": pages[0][1],
        "parser": "test",
    }
    assert {item["document_version"] for item in page_versions} == {row["version"]}
    assert {item["document_version"] for item in chunk_versions} == {row["version"]}


@pytest.mark.asyncio
async def test_replace_derived_content_rolls_back_before_ready(
    seed_pending_document,
    pool,
    monkeypatch,
):
    from infra.db import derived_documents

    doc = await seed_pending_document(status="processing", version=1)

    async def fail_chunks(*args, **kwargs):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(derived_documents, "_insert_chunks", fail_chunks)
    with pytest.raises(RuntimeError, match="chunk write failed"):
        await derived_documents.replace_derived_content(
            pool,
            document_id=doc["id"],
            user_id=doc["user_id"],
            knowledge_base_id=doc["knowledge_base_id"],
            pages=[(1, "content")],
            chunks=[],
            parser="test",
        )

    row = await pool.fetchrow(
        "SELECT status, version, content FROM documents WHERE id=$1",
        doc["id"],
    )
    assert dict(row) == {"status": "processing", "version": 1, "content": None}
    assert await pool.fetchval(
        "SELECT count(*) FROM document_pages WHERE document_id=$1",
        doc["id"],
    ) == 0


@pytest.mark.asyncio
async def test_replace_derived_content_marks_extracted_documents_as_assets(
    seed_pending_document,
    pool,
):
    from infra.db.derived_documents import replace_derived_content
    from services.extracted_assets import ExtractedAsset

    doc = await seed_pending_document()
    asset = ExtractedAsset(
        document_id="00000000-0000-0000-0000-000000000123",
        filename="page-001-image-01.png",
        path="/source.assets/",
        src="source.assets/page-001-image-01.png",
        data=b"image-data",
        content_type="image/png",
        file_type="png",
        parent_document_id=str(doc["id"]),
        page=1,
        index=1,
        kind="pdf_image",
    )

    await replace_derived_content(
        pool,
        document_id=doc["id"],
        user_id=doc["user_id"],
        knowledge_base_id=doc["knowledge_base_id"],
        pages=[(1, "page with an image")],
        chunks=[],
        parser="test",
        assets=[asset],
        metadata_patch={"assets": [asset.metadata()]},
    )

    row = await pool.fetchrow(
        "SELECT source_kind, status, version FROM documents WHERE id=$1::uuid",
        asset.document_id,
    )
    assert dict(row) == {"source_kind": "asset", "status": "ready", "version": 1}
