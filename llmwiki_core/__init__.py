"""Shared domain contracts and pure logic for all LLM Wiki runtimes."""

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
    "DocumentIdentity",
    "DocumentKind",
    "DocumentStatus",
    "InvalidStatusTransition",
    "assert_status_transition",
    "join_logical_path",
    "normalize_directory_path",
    "SearchArea",
    "SearchHit",
    "SearchQuery",
    "SearchScope",
]
