"""Persist citation and wiki-link edges parsed by the shared kernel."""

import logging

from vaultfs import VaultFS
from vaultfs.base import CITATION_TYPES

from llmwiki_core.references import (
    build_lookup_maps,
    extract_references,
    parse_citation_filename,
    parse_wiki_links,
)

logger = logging.getLogger(__name__)

# Compatibility names used by the MCP linter and existing tests.
_parse_citation_filename = parse_citation_filename
_parse_wiki_links = parse_wiki_links


async def update_references(
    fs: VaultFS,
    kb_id: str,
    document_id: str,
    content: str,
    doc_path: str,
) -> None:
    """Rebuild content-derived edges for one document."""
    wiki_relative_dir = doc_path.replace("/wiki/", "", 1) if doc_path.startswith("/wiki/") else ""
    all_docs = await fs.list_documents(kb_id)
    filename_to_doc, base_to_doc, wiki_path_to_doc = build_lookup_maps(all_docs)
    edges = extract_references(
        content,
        document_id,
        wiki_relative_dir,
        filename_to_doc,
        base_to_doc,
        wiki_path_to_doc,
    )

    await fs.delete_references(document_id, ref_types=CITATION_TYPES)
    for edge in edges:
        await fs.upsert_reference(
            document_id,
            edge["target_id"],
            kb_id,
            edge["type"],
            edge["page"],
        )

    logger.info(
        "Updated references for doc=%s: %d citations, %d links",
        document_id[:8],
        sum(1 for edge in edges if edge["type"] == "cites"),
        sum(1 for edge in edges if edge["type"] == "links_to"),
    )


async def get_backlinks_summary(fs: VaultFS, doc_id: str) -> str:
    """Return backlinks for display when reading a page."""
    rows = await fs.get_backlinks(doc_id)
    if not rows:
        return ""
    lines = [f"\n---\n**Referenced by ({len(rows)}):**"]
    for row in rows:
        title = row["title"] or row["filename"]
        lines.append(f"  - {title} ({row['reference_type']})")
    return "\n".join(lines)


__all__ = [
    "_parse_citation_filename",
    "_parse_wiki_links",
    "get_backlinks_summary",
    "update_references",
]
