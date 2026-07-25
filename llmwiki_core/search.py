"""Backend-neutral search request, result, and retrieval port contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from numbers import Real
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
    candidate_limit: int | None = None
    path_glob: str | None = None
    tags: tuple[str, ...] = ()
    document_kinds: tuple[DocumentKind, ...] = ()
    annotated_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not (normalized_text := self.text.strip()):
            raise ValueError("search text must not be empty")
        normalized_limit = _validated_int("search limit", self.limit)
        if not 1 <= normalized_limit <= 100:
            raise ValueError("search limit must be between 1 and 100")

        candidate_limit = self.limit if self.candidate_limit is None else self.candidate_limit
        normalized_candidate_limit = _validated_int("candidate limit", candidate_limit)
        if normalized_candidate_limit < normalized_limit:
            raise ValueError("candidate limit must be at least the search limit")
        if normalized_candidate_limit > 500:
            raise ValueError("candidate limit must be at most 500")
        if not isinstance(self.facets, Mapping):
            raise ValueError("facets must be a mapping")
        if not isinstance(self.annotated_only, bool):
            raise ValueError("annotated_only must be a boolean")

        try:
            area = SearchArea(self.area or SearchArea.ALL)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported search area") from exc
        try:
            scope = SearchScope(self.scope)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported search scope") from exc

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "limit", normalized_limit)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "facets", MappingProxyType(dict(self.facets)))
        object.__setattr__(self, "candidate_limit", normalized_candidate_limit)
        object.__setattr__(self, "path_glob", _normalize_path_glob(self.path_glob))
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        object.__setattr__(
            self,
            "document_kinds",
            _normalize_document_kinds(self.document_kinds),
        )

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
        return cls(
            text=text,
            limit=limit,
            area=area or SearchArea.ALL,
            scope=scope,
            facets={} if facets is None else facets,
            candidate_limit=candidate_limit,
            path_glob=path_glob,
            tags=() if tags is None else tags,
            document_kinds=() if document_kinds is None else document_kinds,
            annotated_only=annotated_only,
        )


def _validated_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


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
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise ValueError("tags must be a sequence of strings")
    normalized: set[str] = set()
    for raw_tag in tags:
        if not isinstance(raw_tag, str):
            raise ValueError("tags must be a sequence of strings")
        if not (tag := raw_tag.strip().lower()):
            raise ValueError("tags must not be empty")
        normalized.add(tag)
    return tuple(sorted(normalized))


def _normalize_document_kinds(
    document_kinds: Sequence[str | DocumentKind] | None,
) -> tuple[DocumentKind, ...]:
    if document_kinds is None:
        return ()
    if isinstance(document_kinds, (str, bytes)) or not isinstance(document_kinds, Sequence):
        raise ValueError("document_kinds must be a sequence of strings or DocumentKind values")
    if any(not isinstance(kind, (str, DocumentKind)) for kind in document_kinds):
        raise ValueError("document_kinds must be a sequence of strings or DocumentKind values")
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
        document_version = _validated_int("document_version", self.document_version)
        chunk_index = _validated_int("chunk_index", self.chunk_index)
        if document_version < 0:
            raise ValueError("document_version must not be negative")
        if chunk_index < 0:
            raise ValueError("chunk_index must not be negative")
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise ValueError("score must be a real number")
        normalized_score = float(self.score)
        if not isfinite(normalized_score):
            raise ValueError("score must be finite")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        document_kind = self.document_kind
        if document_kind is not None:
            try:
                document_kind = DocumentKind(document_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported document kind") from exc

        object.__setattr__(self, "document_version", document_version)
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "score", normalized_score)
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        object.__setattr__(self, "document_kind", document_kind)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

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
        try:
            hits = tuple(self.hits)
        except TypeError as exc:
            raise ValueError("hits must be a sequence of SearchHit values") from exc
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise ValueError("hits must contain only SearchHit values")

        candidate_count = _validated_int("candidate_count", self.candidate_count)
        if candidate_count < 0:
            raise ValueError("candidate_count must not be negative")
        if candidate_count < len(hits):
            raise ValueError("candidate_count must be at least the returned hit count")

        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, Real):
            raise ValueError("latency_ms must be a real number")
        latency_ms = float(self.latency_ms)
        if not isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not isinstance(self.profile, str) or not (profile := self.profile.strip()):
            raise ValueError("profile must be a nonblank string")

        object.__setattr__(self, "hits", hits)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "latency_ms", latency_ms)
        object.__setattr__(self, "profile", profile)

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


def _freeze_metadata(value: object) -> object:
    """Recursively freeze JSON-like metadata into immutable equivalents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not isfinite(value):
            raise TypeError("metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata mapping keys must be strings")
            frozen[key] = _freeze_metadata(nested)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        try:
            return frozenset(_freeze_metadata(item) for item in value)
        except TypeError as exc:
            raise TypeError("metadata set items must freeze to hashable values") from exc
    raise TypeError(f"metadata contains unsupported value type: {type(value).__name__}")
