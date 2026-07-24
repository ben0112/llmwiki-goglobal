"""Atomic Postgres persistence for hosted document-derived content."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import asyncpg

from llmwiki_core.chunking import Chunk, MIN_CHUNK_TOKENS
from services.chunker import _store_chunks_on_conn


class DerivedAsset(Protocol):
    document_id: str
    filename: str
    path: str
    file_type: str
    data: bytes

    def metadata(self) -> dict: ...


BeforeWrite = Callable[[asyncpg.Connection], Awaitable[None]]


async def find_inconsistent_ready_documents(
    executor: asyncpg.Pool | asyncpg.Connection,
) -> list[dict]:
    """Return ready documents whose derived rows do not match their version."""
    rows = await executor.fetch(
        "SELECT d.id, d.user_id FROM documents d "
        "WHERE d.status = 'ready' AND NOT d.archived AND d.source_kind != 'asset' "
        "AND ("
        "  EXISTS (SELECT 1 FROM document_pages p WHERE p.document_id = d.id "
        "          AND p.document_version != d.version) "
        "  OR EXISTS (SELECT 1 FROM document_chunks c WHERE c.document_id = d.id "
        "             AND c.document_version != d.version) "
        "  OR (char_length(btrim(COALESCE(d.content, ''))) >= $1 "
        "      AND NOT EXISTS (SELECT 1 FROM document_chunks c "
        "                      WHERE c.document_id = d.id))"
        ") ORDER BY d.id FOR UPDATE OF d",
        MIN_CHUNK_TOKENS * 4,
    )
    return [dict(row) for row in rows]


async def reset_inconsistent_ready_documents(pool: asyncpg.Pool) -> list[dict]:
    """Atomically move inconsistent ready documents back to the recovery queue."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await find_inconsistent_ready_documents(conn)
            if rows:
                await conn.execute(
                    "UPDATE documents SET status = 'pending', error_message = NULL, "
                    "updated_at = now() WHERE id = ANY($1::uuid[])",
                    [row["id"] for row in rows],
                )
            return rows


async def _insert_chunks(
    conn: asyncpg.Connection,
    document_id,
    user_id,
    knowledge_base_id,
    chunks: list[Chunk],
    document_version: int,
) -> None:
    await _store_chunks_on_conn(
        conn,
        document_id,
        user_id,
        knowledge_base_id,
        chunks,
        document_version,
    )


async def _replace_assets(
    conn: asyncpg.Connection,
    *,
    parent_document_id,
    user_id,
    knowledge_base_id,
    assets: Sequence[DerivedAsset],
) -> None:
    await conn.execute(
        "DELETE FROM documents WHERE user_id = $1 AND source_kind = 'asset' "
        "AND metadata->>'parent_document_id' = $2",
        user_id,
        str(parent_document_id),
    )
    if not assets:
        return
    await conn.executemany(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, title, source_kind, "
        "file_type, file_size, status, content, metadata, version) "
        "VALUES ($1, $2, $3, $4, $5, $4, 'asset', $6, $7, 'ready', NULL, $8::jsonb, 1)",
        [
            (
                asset.document_id,
                knowledge_base_id,
                user_id,
                asset.filename,
                asset.path,
                asset.file_type,
                len(asset.data),
                json.dumps(asset.metadata()),
            )
            for asset in assets
        ],
    )


async def replace_derived_content(
    pool: asyncpg.Pool,
    *,
    document_id,
    user_id,
    knowledge_base_id,
    pages: list[tuple[int, str]],
    chunks: list[Chunk],
    parser: str,
    content: str | None = None,
    page_elements: dict[int, dict] | None = None,
    metadata_patch: dict | None = None,
    assets: Sequence[DerivedAsset] | None = None,
    before_write: BeforeWrite | None = None,
) -> int:
    """Replace one hosted document's derived rows and publish readiness atomically."""
    conn = await pool.acquire()
    try:
        async with conn.transaction():
            document = await conn.fetchrow(
                "SELECT version FROM documents "
                "WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3 FOR UPDATE",
                document_id,
                user_id,
                knowledge_base_id,
            )
            if document is None:
                raise LookupError(f"document {document_id} not found for derived-content write")

            if before_write is not None:
                await before_write(conn)

            document_version = document["version"] + 1
            await conn.execute("DELETE FROM document_pages WHERE document_id = $1", document_id)
            if pages:
                await conn.executemany(
                    "INSERT INTO document_pages "
                    "(document_id, page, content, elements, document_version) "
                    "VALUES ($1, $2, $3, $4::jsonb, $5)",
                    [
                        (
                            document_id,
                            page,
                            page_content,
                            json.dumps((page_elements or {}).get(page))
                            if (page_elements or {}).get(page)
                            else None,
                            document_version,
                        )
                        for page, page_content in pages
                    ],
                )

            if assets is not None:
                await _replace_assets(
                    conn,
                    parent_document_id=document_id,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    assets=assets,
                )

            await _insert_chunks(
                conn,
                document_id,
                user_id,
                knowledge_base_id,
                chunks,
                document_version,
            )

            full_content = (
                content
                if content is not None
                else "\n\n---\n\n".join(page_content for _, page_content in pages)
            )
            await conn.execute(
                "UPDATE documents "
                "SET status = 'ready', content = $2, page_count = $3, parser = $4, "
                "version = $5, metadata = COALESCE(metadata, '{}'::jsonb) || $6::jsonb, "
                "error_message = NULL, updated_at = now() "
                "WHERE id = $1 AND user_id = $7",
                document_id,
                full_content,
                len(pages),
                parser,
                document_version,
                json.dumps(metadata_patch or {}),
                user_id,
            )
            return document_version
    finally:
        await pool.release(conn)


__all__ = [
    "find_inconsistent_ready_documents",
    "replace_derived_content",
    "reset_inconsistent_ready_documents",
]
