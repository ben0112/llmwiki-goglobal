"""Reconcile backfills search chunks for files that `llmwiki init` only listed."""

import asyncio
import uuid
from pathlib import Path

import aiosqlite

SCHEMA_PATH = Path(__file__).parents[2] / "shared" / "sqlite_schema.sql"
USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


async def _init_db(workspace: Path) -> aiosqlite.Connection:
    (workspace / ".llmwiki").mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(workspace / ".llmwiki" / "index.db"))
    await db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    await db.execute(
        "INSERT INTO workspace (id, name, description, user_id) VALUES (?, 'ws', '', ?)",
        (str(uuid.uuid4()), USER_ID),
    )
    await db.commit()
    return db


async def _insert_indexed_text(
    db: aiosqlite.Connection,
    workspace: Path,
    content: str,
) -> str:
    """Mimic `init`: a text source listed in the index, ready, but never chunked."""
    (workspace / "notes.md").write_text(content, encoding="utf-8")
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, source_kind, "
        "file_type, status, content, tags, version, document_number) "
        "VALUES (?, ?, 'notes.md', 'Notes', '/', 'notes.md', 'source', 'md', 'ready', ?, '[]', 0, 1)",
        (doc_id, USER_ID, content),
    )
    await db.commit()
    return doc_id


async def _chunk_count(db: aiosqlite.Connection, doc_id: str) -> int:
    cursor = await db.execute(
        "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?", (doc_id,),
    )
    return (await cursor.fetchone())[0]


async def _fts_hits(db: aiosqlite.Connection, term: str) -> int:
    cursor = await db.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (term,),
    )
    return (await cursor.fetchone())[0]


async def _insert_indexed_html(db: aiosqlite.Connection, workspace: Path, html: str) -> str:
    """Mimic `init`: an HTML source listed in the index, on disk, never processed."""
    (workspace / "page.html").write_text(html, encoding="utf-8")
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, source_kind, "
        "file_type, status, content, tags, version, document_number) "
        "VALUES (?, ?, 'page.html', 'Page', '/', 'page.html', 'source', 'html', 'ready', NULL, '[]', 0, 2)",
        (doc_id, USER_ID),
    )
    await db.commit()
    return doc_id


async def test_reconcile_routes_html_through_webmd(tmp_path):
    from domain.local_processor import reconcile_workspace

    workspace = tmp_path / "research"
    workspace.mkdir()
    db = await _init_db(workspace)
    html = (
        "<html><body><h1>Quantum Annealing</h1><p>"
        + ("Tunneling explores the energy landscape. " * 20)
        + "</p></body></html>"
    )
    doc_id = await _insert_indexed_html(db, workspace, html)

    await reconcile_workspace(db, workspace)

    cursor = await db.execute("SELECT parser, status FROM documents WHERE id = ?", (doc_id,))
    parser, status = await cursor.fetchone()
    assert parser == "webmd"  # routed through the HTML parser, not chunked as raw text ('text')
    assert status == "ready"
    assert await _chunk_count(db, doc_id) > 0
    assert await _fts_hits(db, "tunneling") > 0

    await db.close()


async def test_reconcile_backfills_chunks_for_indexed_text(tmp_path):
    from domain.local_processor import reconcile_workspace

    workspace = tmp_path / "research"
    workspace.mkdir()
    db = await _init_db(workspace)
    content = "# Reinforcement Learning\n\n" + (
        "Policy gradient methods optimize the expected return directly. " * 40
    )
    doc_id = await _insert_indexed_text(db, workspace, content)

    # Without the fix the file is listed but unsearchable: no chunks, no FTS rows.
    assert await _chunk_count(db, doc_id) == 0
    assert await _fts_hits(db, "policy") == 0

    await reconcile_workspace(db, workspace)

    chunks_after_first = await _chunk_count(db, doc_id)
    assert chunks_after_first > 0
    assert await _fts_hits(db, "policy") > 0

    # Idempotent: the `parser` marker keeps a second pass from re-chunking.
    from domain.local_processor import _unchunked_text_docs
    assert await _unchunked_text_docs(db) == []
    await reconcile_workspace(db, workspace)
    assert await _chunk_count(db, doc_id) == chunks_after_first

    await db.close()


async def test_inconsistent_ready_scan_finds_stale_chunk_version(tmp_path):
    from domain.local_processor import _inconsistent_ready_document_ids

    workspace = tmp_path / "research"
    workspace.mkdir()
    db = await _init_db(workspace)
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, content, tags, parser, version, document_number) "
        "VALUES (?, ?, 'policy.md', 'Policy', '/', 'policy.md', 'source', 'md', "
        "'ready', 'current body', '[]', 'text', 2, 3)",
        (doc_id, USER_ID),
    )
    await db.execute(
        "INSERT INTO document_chunks "
        "(id, document_id, chunk_index, content, source_content, token_count, document_version) "
        "VALUES (?, ?, 0, 'stale body', 'stale body', 2, 1)",
        (str(uuid.uuid4()), doc_id),
    )
    await db.commit()

    assert await _inconsistent_ready_document_ids(db) == [doc_id]
    await db.close()


async def test_inconsistent_ready_scan_ignores_legitimate_short_text(tmp_path):
    from domain.local_processor import _inconsistent_ready_document_ids

    workspace = tmp_path / "research"
    workspace.mkdir()
    db = await _init_db(workspace)
    doc_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, content, tags, parser, version, document_number) "
        "VALUES (?, ?, 'short.md', 'Short', '/', 'short.md', 'source', 'md', "
        "'ready', 'brief note', '[]', 'text', 1, 4)",
        (doc_id, USER_ID),
    )
    await db.commit()

    assert await _inconsistent_ready_document_ids(db) == []
    await db.close()


async def test_reconcile_rebuilds_stale_text_from_disk(tmp_path, monkeypatch):
    import domain.local_processor as local_processor

    workspace = tmp_path / "research"
    workspace.mkdir()
    disk_content = "Current disk policy. " * 40
    (workspace / "policy.md").write_text(disk_content, encoding="utf-8")
    db = await _init_db(workspace)
    doc_id = str(uuid.uuid4())
    stale_content = "Stale database body. " * 40
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, title, path, relative_path, source_kind, file_type, "
        "status, content, tags, version, document_number) "
        "VALUES (?, ?, 'policy.md', 'Policy', '/', 'policy.md', 'source', 'md', "
        "'ready', ?, '[]', 2, 3)",
        (doc_id, USER_ID, stale_content),
    )
    await db.commit()

    real_spawn = local_processor.spawn_logged

    def delayed_spawn(coro, name):
        async def delayed():
            await asyncio.sleep(0.05)
            return await coro

        return real_spawn(delayed(), name)

    monkeypatch.setattr(local_processor, "spawn_logged", delayed_spawn)
    await local_processor.reconcile_workspace(db, workspace)

    row = await (
        await db.execute("SELECT status, content, version FROM documents WHERE id=?", (doc_id,))
    ).fetchone()
    chunk_rows = await db.execute_fetchall(
        "SELECT content, document_version FROM document_chunks WHERE document_id=?",
        (doc_id,),
    )
    assert row == ("ready", disk_content, 3)
    assert chunk_rows
    assert {version for _, version in chunk_rows} == {3}
    assert all("Current disk policy" in content for content, _ in chunk_rows)
    await db.close()
