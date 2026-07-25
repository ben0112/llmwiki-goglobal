"""Backend-neutral search request, result, and retrieval port contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from .documents import DocumentKind


class SearchArea(StrEnum):
    ALL = "all"
    WIKI = "wiki"
    SOURCES = "sources"


class SearchScope(StrEnum):
    ALL = "all"
    ANNOTATIONS = "annotations"
    SOURCE = "source"


@dataclass(frozen=True)
class SearchQuery:
    text: str
    limit: int = 20
    area: SearchArea = SearchArea.ALL
    scope: SearchScope = SearchScope.ALL
    facets: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    candidate_limit: int = 20
    path_glob: str | None = None
    tags: tuple[str, ...] = ()
    document_kinds: tuple[DocumentKind, ...] = ()
    annotated_only: bool = False

    @classmethod
    def build(
        cls,
        *,
        text: str,
        limit: int = 20,
        area: str | SearchArea | None = None,
        scope: str | SearchScope = SearchScope.ALL,
        facets: Mapping[str, Any] | None = None,
        candidate_limit: int | None = None,
        path_glob: str | None = None,
        tags: Sequence[str] | None = None,
        document_kinds: Sequence[str | DocumentKind] | None = None,
        annotated_only: bool = False,
    ) -> "SearchQuery":
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("search text must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("search limit must be between 1 and 100")

        normalized_candidate_limit = limit if candidate_limit is None else candidate_limit
        if normalized_candidate_limit < limit:
            raise ValueError("candidate limit must be at least the search limit")
        if normalized_candidate_limit > 500:
            raise ValueError("candidate limit must be at most 500")

        normalized_path_glob = _normalize_path_glob(path_glob)
        normalized_tags = _normalize_tags(tags)
        normalized_kinds = _normalize_document_kinds(document_kinds)
        return cls(
            text=normalized_text,
            limit=limit,
            area=SearchArea(area or SearchArea.ALL),
            scope=SearchScope(scope),
            facets=MappingProxyType(dict(facets or {})),
            candidate_limit=normalized_candidate_limit,
            path_glob=normalized_path_glob,
            tags=normalized_tags,
            document_kinds=normalized_kinds,
            annotated_only=annotated_only,
        )


def _normalize_path_glob(path_glob: str | None) -> str | None:
    if path_glob is None:
        return None
    if "\x00" in path_glob:
        raise ValueError("path glob contains NUL")
    normalized = path_glob.strip().replace("\\", "/")
    if not normalized:
        return None
    return "/" + normalized.lstrip("/")


def _normalize_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    if tags is None:
        return ()
    normalized: set[str] = set()
    for raw_tag in tags:
        if not isinstance(raw_tag, str) or not (tag := raw_tag.strip().lower()):
            raise ValueError("tags must not be empty")
        normalized.add(tag)
    return tuple(sorted(normalized))


def _normalize_document_kinds(
    document_kinds: Sequence[str | DocumentKind] | None,
) -> tuple[DocumentKind, ...]:
    if document_kinds is None:
        return ()
    try:
        normalized = {DocumentKind(kind) for kind in document_kinds}
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported document kind") from exc
    return tuple(sorted(normalized, key=lambda kind: kind.value))


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    document_version: int
    chunk_index: int
    content: str
    score: float
    path: str
    title: str | None = None
    page: int | None = None
    header_breadcrumb: str | None = None
    tags: tuple[str, ...] = ()
    document_kind: DocumentKind | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.document_id, self.document_version, self.chunk_index)


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    candidate_count: int
    latency_ms: float = 0.0
    profile: str = "lexical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))

    @property
    def returned_count(self) -> int:
        return len(self.hits)


class Retriever(Protocol):
    async def retrieve(self, query: SearchQuery) -> SearchResult: ...


class Reranker(Protocol):
    async def rerank(
        self,
        query: SearchQuery,
        hits: Sequence[SearchHit],
    ) -> Sequence[SearchHit]: ...


class ContextExpander(Protocol):
    async def expand(
        self,
        query: SearchQuery,
        hits: Sequence[SearchHit],
    ) -> Sequence[SearchHit]: ...
