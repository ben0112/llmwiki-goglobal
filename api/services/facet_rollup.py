"""Persistence adapters for shared wiki-page facet rollups."""

import json
import logging
from datetime import date

from llmwiki_core.facets import apply_rollup, rollup_from_metas

logger = logging.getLogger(__name__)


def _parse_meta(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def refresh_rollups_local(db) -> int:
    """Recompute facet rollups for every local wiki page."""
    from infra.db.sqlite import serialized_write

    cursor = await db.execute(
        "SELECT r.source_document_id, d.metadata FROM document_references r "
        "JOIN documents d ON d.id = r.target_document_id "
        "JOIN documents w ON w.id = r.source_document_id "
        "WHERE r.reference_type = 'cites' AND w.source_kind = 'wiki' "
        "AND d.relative_path LIKE 'corpus/%'"
    )
    by_page: dict[str, list[dict]] = {}
    for page_id, meta_raw in await cursor.fetchall():
        by_page.setdefault(page_id, []).append(_parse_meta(meta_raw))

    cursor = await db.execute("SELECT id, metadata FROM documents WHERE source_kind = 'wiki'")
    pages = await cursor.fetchall()

    today = date.today().isoformat()
    updated = 0
    async with serialized_write():
        for page_id, meta_raw in pages:
            metadata = _parse_meta(meta_raw)
            rollup = rollup_from_metas(by_page.get(page_id, []), today)
            if apply_rollup(metadata, rollup):
                await db.execute(
                    "UPDATE documents SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), page_id),
                )
                updated += 1
        await db.commit()
    return updated


async def refresh_rollups_hosted(conn, kb_id, user_id: str) -> int:
    """Recompute facet rollups for every hosted wiki page in a tenant KB."""
    rows = await conn.fetch(
        "SELECT r.source_document_id::text AS page_id, d.metadata "
        "FROM document_references r "
        "JOIN documents d ON d.id = r.target_document_id "
        "JOIN documents w ON w.id = r.source_document_id "
        "WHERE r.reference_type = 'cites' AND w.path LIKE '/wiki/%' "
        "AND d.path LIKE '/corpus/%' AND w.knowledge_base_id = $1 AND w.user_id = $2",
        kb_id,
        user_id,
    )
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["page_id"], []).append(_parse_meta(row["metadata"]))

    pages = await conn.fetch(
        "SELECT id::text, metadata FROM documents "
        "WHERE knowledge_base_id = $1 AND user_id = $2 AND path LIKE '/wiki/%' AND NOT archived",
        kb_id,
        user_id,
    )
    today = date.today().isoformat()
    updated = 0
    for row in pages:
        metadata = _parse_meta(row["metadata"])
        rollup = rollup_from_metas(by_page.get(row["id"], []), today)
        if apply_rollup(metadata, rollup):
            await conn.execute(
                "UPDATE documents SET metadata = $1::jsonb WHERE id = $2::uuid AND user_id = $3",
                json.dumps(metadata, ensure_ascii=False),
                row["id"],
                user_id,
            )
            updated += 1
    return updated


__all__ = [
    "apply_rollup",
    "refresh_rollups_hosted",
    "refresh_rollups_local",
    "rollup_from_metas",
]
