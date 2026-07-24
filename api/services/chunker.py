"""Postgres persistence adapter for the shared text chunker."""

import logging

import asyncpg

from llmwiki_core.chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_TOKENS,
    Chunk,
    _enforce_max_chars,
    _estimate_tokens,
    _get_overlap,
    _split_oversized,
    _split_paragraphs,
    chunk_pages,
    chunk_text,
)

logger = logging.getLogger(__name__)


async def store_chunks(
    pool_or_conn,
    document_id: str,
    user_id: str,
    knowledge_base_id: str,
    chunks: list[Chunk],
):
    if isinstance(pool_or_conn, asyncpg.Connection):
        await _store_chunks_on_conn(pool_or_conn, document_id, user_id, knowledge_base_id, chunks)
    else:
        conn = await pool_or_conn.acquire()
        try:
            await _store_chunks_on_conn(conn, document_id, user_id, knowledge_base_id, chunks)
        finally:
            await pool_or_conn.release(conn)


async def _store_chunks_on_conn(
    conn: asyncpg.Connection,
    document_id: str,
    user_id: str,
    knowledge_base_id: str,
    chunks: list[Chunk],
):
    await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", document_id)

    if not chunks:
        return

    await conn.executemany(
        "INSERT INTO document_chunks "
        "(document_id, user_id, knowledge_base_id, chunk_index, content, source_content, page, start_char, token_count, header_breadcrumb) "
        "VALUES ($1, $2, $3, $4, $5, $5, $6, $7, $8, $9)",
        [
            (
                document_id,
                user_id,
                knowledge_base_id,
                chunk.index,
                chunk.content,
                chunk.page,
                chunk.start_char,
                chunk.token_count,
                chunk.header_breadcrumb,
            )
            for chunk in chunks
        ],
    )
    logger.info("Stored %d chunks for doc %s", len(chunks), document_id[:8])


__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "MAX_CHUNK_CHARS",
    "MIN_CHUNK_TOKENS",
    "Chunk",
    "_enforce_max_chars",
    "_estimate_tokens",
    "_get_overlap",
    "_split_oversized",
    "_split_paragraphs",
    "_store_chunks_on_conn",
    "chunk_pages",
    "chunk_text",
    "store_chunks",
]
