"""Compatibility exports for shared citation and wiki-link parsing."""

from llmwiki_core.references import (
    build_lookup_maps,
    extract_references,
    parse_citation_filename,
    parse_wiki_links,
)

__all__ = [
    "build_lookup_maps",
    "extract_references",
    "parse_citation_filename",
    "parse_wiki_links",
]
