"""Shared atomic wiki-write contract for SQLite and Postgres VaultFS."""

import json
import os
import uuid

import pytest

from llmwiki_core.references import ReferenceEdge
from llmwiki_core.wiki import VersionConflict, WikiWriteBundle
from vaultfs.base import DuplicateDocumentError


def _bundle(
    document_id: str,
    expected_version: int | None,
    content: str,
    target_id: str,
) -> WikiWriteBundle:
    return WikiWriteBundle.build(
        document_id=document_id,
        expected_version=expected_version,
        filename="atomic-page.md",
        path="/wiki/",
        file_type="md",
        content=content,
        title="Atomic Page",
        tags=["atomic"],
        date="2026-07-24",
        metadata={"description": "Atomic page"},
        edges=[ReferenceEdge(target_id, "cites", 3)],
    )


async def _snapshot(fs, document_id: str) -> dict:
    if fs.__class__.__name__ == "SqliteVaultFS":
        db = fs._db_or_raise()
        document = await (
            await db.execute(
                "SELECT content, version, tags, metadata FROM documents WHERE id = ?",
                (document_id,),
            )
        ).fetchone()
        chunks = await (
            await db.execute(
                "SELECT document_version, chunk_index, content FROM document_chunks "
                "WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            )
        ).fetchall()
        edges = await (
            await db.execute(
                "SELECT target_document_id, reference_type, page FROM document_references "
                "WHERE source_document_id = ? ORDER BY target_document_id, reference_type",
                (document_id,),
            )
        ).fetchall()
        metadata = json.loads(document[3] or "{}")
        tags = json.loads(document[2] or "[]")
    else:
        import db as mcp_db

        async with mcp_db._pool.acquire() as conn:
            document = await conn.fetchrow(
                "SELECT content, version, tags, metadata FROM documents WHERE id = $1",
                document_id,
            )
            chunks = await conn.fetch(
                "SELECT document_version, chunk_index, content FROM document_chunks "
                "WHERE document_id = $1 ORDER BY chunk_index",
                document_id,
            )
            edges = await conn.fetch(
                "SELECT target_document_id, reference_type, page FROM document_references "
                "WHERE source_document_id = $1 ORDER BY target_document_id, reference_type",
                document_id,
            )
        raw_metadata = document["metadata"]
        metadata = (
            json.loads(raw_metadata)
            if isinstance(raw_metadata, str)
            else dict(raw_metadata or {})
        )
        tags = list(document["tags"] or [])
        document = tuple(document)
        chunks = [tuple(row) for row in chunks]
        edges = [(str(row[0]), row[1], row[2]) for row in edges]
    return {
        "content": document[0],
        "version": document[1],
        "tags": tags,
        "metadata": metadata,
        "chunks": [tuple(row) for row in chunks],
        "edges": [tuple(row) for row in edges],
    }


async def assert_atomic_wiki_write(fs, kb_id: str) -> None:
    source_one = await fs.create_document(
        kb_id,
        "entry-one.txt",
        "Entry One",
        "/corpus/",
        "txt",
        "classified source one",
        ["corpus"],
        metadata={"entry_id": "E-1", "stage": "S2", "timeliness": "M2"},
    )
    source_two = await fs.create_document(
        kb_id,
        "entry-two.txt",
        "Entry Two",
        "/corpus/",
        "txt",
        "classified source two",
        ["corpus"],
        metadata={"entry_id": "E-2", "stage": "S3", "timeliness": "M1"},
    )
    document_id = str(uuid.uuid4())
    first_content = "first atomic revision " * 80
    second_content = "second atomic revision " * 80

    created = await fs.write_wiki_bundle(
        kb_id,
        _bundle(document_id, None, first_content, str(source_one["id"])),
    )
    assert created["version"] == 1
    first = await _snapshot(fs, document_id)
    assert first["version"] == 1
    assert {row[0] for row in first["chunks"]} == {1}
    assert first["edges"] == [(str(source_one["id"]), "cites", 3)]
    assert first["metadata"]["facet_rollup"]["stage"] == ["S2"]

    with pytest.raises(DuplicateDocumentError):
        await fs.write_wiki_bundle(
            kb_id,
            _bundle(str(uuid.uuid4()), None, "duplicate path " * 80, str(source_one["id"])),
        )
    assert await _snapshot(fs, document_id) == first

    referrer = await fs.create_document(
        kb_id,
        "referrer.md",
        "Referrer",
        "/wiki/",
        "md",
        "referrer content " * 80,
        ["wiki"],
    )
    await fs.upsert_reference(
        str(referrer["id"]), document_id, kb_id, "links_to", None
    )

    updated = await fs.write_wiki_bundle(
        kb_id,
        _bundle(document_id, 1, second_content, str(source_two["id"])),
    )
    assert updated["version"] == 2
    second = await _snapshot(fs, document_id)
    assert second["content"] == second_content
    assert second["version"] == 2
    assert {row[0] for row in second["chunks"]} == {2}
    assert second["edges"] == [(str(source_two["id"]), "cites", 3)]
    assert second["metadata"]["facet_rollup"]["stage"] == ["S3"]
    stale_names = {row["filename"] for row in await fs.find_stale_pages(kb_id)}
    assert "referrer.md" in stale_names

    with pytest.raises(VersionConflict):
        await fs.write_wiki_bundle(
            kb_id,
            _bundle(document_id, 1, "stale revision", str(source_one["id"])),
        )
    assert await _snapshot(fs, document_id) == second


async def test_sqlite_atomic_wiki_write(fs):
    instance, kb_id = fs
    await assert_atomic_wiki_write(instance, kb_id)


async def test_sqlite_startup_reconciles_disk_after_database_rollback(fs):
    instance, kb_id = fs
    created = await instance.create_document(
        kb_id,
        "recovery.md",
        "Recovery",
        "/wiki/",
        "md",
        "database revision " * 80,
        ["recovery"],
    )
    disk_content = "disk replacement survived rollback " * 80
    assert instance.write_to_disk("/wiki/", "recovery.md", disk_content)

    await instance.ensure_workspace("test-workspace")

    recovered = await instance.get_document(kb_id, "recovery.md", "/wiki/")
    assert recovered["content"] == disk_content
    assert recovered["version"] == 2
    snapshot = await _snapshot(instance, str(created["id"]))
    assert {row[0] for row in snapshot["chunks"]} == {2}


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="requires Postgres")
async def test_postgres_atomic_wiki_write(pg_atomic_fs):
    instance, kb_id = pg_atomic_fs
    await assert_atomic_wiki_write(instance, kb_id)


if "DATABASE_URL" in os.environ:
    from tests.integration.mcp.test_mcp_isolation import KB_A_ID

    pytest_plugins = ("tests.integration.mcp.test_mcp_isolation",)

    @pytest.fixture
    async def pg_atomic_fs(fs_alice, seed_and_bind_pool):
        return fs_alice, KB_A_ID
