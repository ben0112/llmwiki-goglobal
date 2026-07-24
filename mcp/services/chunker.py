"""Database persistence adapters for the shared text chunker."""

import aiosqlite
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


async def store_chunks_pg(
    conn: asyncpg.Connection,
    document_id: str,
    user_id: str,
    knowledge_base_id: str,
    document_version: int,
    chunks: list[Chunk],
) -> None:
    await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", document_id)
    if not chunks:
        return
    await conn.executemany(
        "INSERT INTO document_chunks "
        "(document_id, user_id, knowledge_base_id, document_version, chunk_index, content, source_content, page, start_char, "
        "token_count, header_breadcrumb) "
        "VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $9, $10)",
        [
            (
                document_id,
                user_id,
                knowledge_base_id,
                document_version,
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


async def store_chunks_sqlite(
    db: aiosqlite.Connection,
    document_id: str,
    document_version: int,
    chunks: list[Chunk],
) -> None:
    """Replace SQLite chunks; FTS triggers keep the search index synchronized."""
    await db.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
    if chunks:
        await db.executemany(
            "INSERT INTO document_chunks "
            "(document_id, document_version, chunk_index, content, source_content, page, start_char, token_count, header_breadcrumb) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    document_id,
                    document_version,
                    chunk.index,
                    chunk.content,
                    chunk.content,
                    chunk.page,
                    chunk.start_char,
                    chunk.token_count,
                    chunk.header_breadcrumb,
                )
                for chunk in chunks
            ],
        )


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
    "chunk_pages",
    "chunk_text",
    "store_chunks_pg",
    "store_chunks_sqlite",
]
