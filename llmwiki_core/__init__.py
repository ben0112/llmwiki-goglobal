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
from .search import SearchArea, SearchHit, SearchQuery, SearchScope

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "DocumentIdentity",
    "DocumentKind",
    "DocumentStatus",
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
]
