import hashlib
import uuid

import pytest


async def _init_db(tmp_path, monkeypatch):
    from config import settings
    from infra.db.sqlite import create_pool

    monkeypatch.setattr(settings, "WORKSPACE_PATH", str(tmp_path))
    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) "
        "VALUES ('ws', 'ws', '', 'user')"
    )
    await db.commit()
    return db


@pytest.mark.asyncio
async def test_text_upload_ready_implies_current_chunks(tmp_path, monkeypatch):
    from routes.local_upload import _index_file_on_disk

    db = await _init_db(tmp_path, monkeypatch)
    content = "Policy sentence. " * 100
    source = tmp_path / "policy.md"
    source.write_text(content, encoding="utf-8")

    doc = await _index_file_on_disk(
        db,
        "policy.md",
        source,
        hashlib.sha256(content.encode()).hexdigest(),
    )

    row = await (
        await db.execute("SELECT status, version FROM documents WHERE id=?", (doc["id"],))
    ).fetchone()
    versions = await (
        await db.execute(
            "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=?",
            (doc["id"],),
        )
    ).fetchall()
    assert row == ("ready", 1)
    assert {version[0] for version in versions} == {row[1]}
    await db.close()


@pytest.mark.asyncio
async def test_text_upload_rolls_back_document_when_chunk_write_fails(tmp_path, monkeypatch):
    import routes.local_upload as local_upload

    db = await _init_db(tmp_path, monkeypatch)
    source = tmp_path / "policy.md"
    source.write_text("Policy sentence. " * 100, encoding="utf-8")

    async def fail_chunk_write(*args, **kwargs):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(
        local_upload,
        "_store_chunks_for_upload",
        fail_chunk_write,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="chunk write failed"):
        await local_upload._index_file_on_disk(db, "policy.md", source, "digest")
    assert await db.execute_fetchall("SELECT id FROM documents") == []
    await db.close()


@pytest.mark.asyncio
async def test_extracted_pages_and_chunks_use_next_document_version(tmp_path, monkeypatch):
    from domain.local_processor import _store_page_contents

    db = await _init_db(tmp_path, monkeypatch)
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, tags, version, document_number) "
        "VALUES (?, 'user', 'report.pdf', 'Report', '/', 'report.pdf', 'source', 'pdf', "
        "'processing', '[]', 2, 1)",
        (doc_id,),
    )
    await db.commit()
    pages = [(1, "Extracted policy requirements. " * 20)]

    await _store_page_contents(db, doc_id, pages, "test")

    row = await (
        await db.execute("SELECT status, version FROM documents WHERE id=?", (doc_id,))
    ).fetchone()
    page_versions = await db.execute_fetchall(
        "SELECT DISTINCT document_version FROM document_pages WHERE document_id=?",
        (doc_id,),
    )
    chunk_versions = await db.execute_fetchall(
        "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=?",
        (doc_id,),
    )
    assert row == ("ready", 3)
    assert page_versions == [(3,)]
    assert chunk_versions == [(3,)]
    await db.close()


@pytest.mark.asyncio
async def test_sqlite_chunk_repository_requires_explicit_document_version(
    tmp_path,
    monkeypatch,
):
    from infra.db.sqlite import SQLiteChunkRepository
    from services.chunker import chunk_text

    db = await _init_db(tmp_path, monkeypatch)
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, content, tags, version, document_number) "
        "VALUES (?, 'user', 'note.md', 'Note', '/', 'note.md', 'source', 'md', "
        "'ready', 'body', '[]', 7, 1)",
        (doc_id,),
    )
    await db.commit()

    chunks = chunk_text("Versioned local note content. " * 20)
    await SQLiteChunkRepository(db).store(
        doc_id,
        "user",
        "ws",
        chunks,
        document_version=7,
    )

    assert await db.execute_fetchall(
        "SELECT DISTINCT document_version FROM document_chunks WHERE document_id=?",
        (doc_id,),
    ) == [(7,)]
    await db.close()
