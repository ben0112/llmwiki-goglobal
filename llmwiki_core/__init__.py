"""Shared domain contracts and pure logic for all LLM Wiki runtimes."""

from .chunking import Chunk, chunk_pages, chunk_text
from .documents import (
    DocumentIdentity,
    DocumentKind,
    DocumentStatus,
    InvalidStatusTransition,
    assert_status_transition,
    join_logical_path,
    normalize_directory_path,
)
from .facets import FACET_KEYS, UnknownFacetError, apply_rollup, rollup_from_metas, validate_facets
from .references import build_lookup_maps, extract_references, parse_citation_filename, parse_wiki_links
from .search import SearchArea, SearchHit, SearchQuery, SearchScope

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "DocumentIdentity",
    "DocumentKind",
    "DocumentStatus",
    "FACET_KEYS",
    "InvalidStatusTransition",
    "assert_status_transition",
    "chunk_pages",
    "chunk_text",
    "join_logical_path",
    "normalize_directory_path",
    "SearchArea",
    "SearchHit",
    "SearchQuery",
    "SearchScope",
    "UnknownFacetError",
    "apply_rollup",
    "build_lookup_maps",
    "extract_references",
    "parse_citation_filename",
    "parse_wiki_links",
    "rollup_from_metas",
    "validate_facets",
]
