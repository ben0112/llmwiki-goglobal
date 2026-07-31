"""Backend-neutral search request, result, and retrieval port contracts."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import islice
from math import isfinite
from numbers import Real
from time import perf_counter
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


@dataclass(frozen=True, init=False)
class SearchQuery:
    text: str
    limit: int = 20
    area: SearchArea = SearchArea.ALL
    scope: SearchScope = SearchScope.ALL
    facets: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    candidate_limit: int
    path_glob: str | None = None
    tags: tuple[str, ...] = ()
    document_kinds: tuple[DocumentKind, ...] = ()
    annotated_only: bool = False

    def __init__(
        self,
        text: str,
        limit: int = 20,
        area: str | SearchArea | None = SearchArea.ALL,
        scope: str | SearchScope = SearchScope.ALL,
        facets: Mapping[str, Any] | None = None,
        candidate_limit: int | None = None,
        path_glob: str | None = None,
        tags: Sequence[str] | None = None,
        document_kinds: Sequence[str | DocumentKind] | None = None,
        annotated_only: bool = False,
    ) -> None:
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "facets", {} if facets is None else facets)
        object.__setattr__(
            self,
            "candidate_limit",
            limit if candidate_limit is None else candidate_limit,
        )
        object.__setattr__(self, "path_glob", path_glob)
        object.__setattr__(self, "tags", () if tags is None else tags)
        object.__setattr__(
            self,
            "document_kinds",
            () if document_kinds is None else document_kinds,
        )
        object.__setattr__(self, "annotated_only", annotated_only)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not (normalized_text := self.text.strip()):
            raise ValueError("search text must not be empty")
        normalized_limit = _validated_int("search limit", self.limit)
        if not 1 <= normalized_limit <= 100:
            raise ValueError("search limit must be between 1 and 100")

        normalized_candidate_limit = _validated_int("candidate limit", self.candidate_limit)
        if normalized_candidate_limit < normalized_limit:
            raise ValueError("candidate limit must be at least the search limit")
        if normalized_candidate_limit > 500:
            raise ValueError("candidate limit must be at most 500")
        if not isinstance(self.facets, Mapping):
            raise ValueError("facets must be a mapping")
        if not isinstance(self.annotated_only, bool):
            raise ValueError("annotated_only must be a boolean")

        try:
            area = SearchArea(SearchArea.ALL if self.area is None else self.area)
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
        object.__setattr__(self, "facets", _freeze_json_like(self.facets, label="facets"))
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
            area=SearchArea.ALL if area is None else area,
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


def _validated_nonblank_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{name} must be a nonblank string")
    return normalized


def _validated_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _validated_optional_string(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
    return value


def _validated_optional_page(value: object) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError("page must be a non-negative integer or None")
    return value


def _normalize_path_glob(path_glob: str | None) -> str | None:
    if path_glob is None:
        return None
    if not isinstance(path_glob, str):
        raise ValueError("path_glob must be a string or None")
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
        document_id = _validated_nonblank_string("document_id", self.document_id)
        path = _validated_nonblank_string("path", self.path)
        _validated_string("content", self.content)
        _validated_optional_string("title", self.title)
        _validated_optional_string("header_breadcrumb", self.header_breadcrumb)
        page = _validated_optional_page(self.page)

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

        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "document_version", document_version)
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "score", normalized_score)
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        object.__setattr__(self, "document_kind", document_kind)
        object.__setattr__(self, "metadata", _freeze_json_like(self.metadata, label="metadata"))

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


class RetrieverUnavailable(RuntimeError):
    """A retriever cannot currently serve the request."""


_VECTOR_UNAVAILABLE = object()
_MAX_RRF_K = 1_000_000
_MIN_POSTPROCESS_SCAN_LIMIT = 16
_MAX_POSTPROCESS_SCAN_LIMIT = 600


class HybridRetrievalService:
    """Compose retrieval with bounded post-processing and RRF k up to 1,000,000."""

    def __init__(
        self,
        *,
        lexical: Retriever,
        vector: Retriever | None = None,
        reranker: Reranker | None = None,
        expander: ContextExpander | None = None,
        rrf_k: int = 60,
    ) -> None:
        _require_callable_port(lexical, method="retrieve", label="lexical")
        if vector is not None:
            _require_callable_port(vector, method="retrieve", label="vector")
        if reranker is not None:
            _require_callable_port(reranker, method="rerank", label="reranker")
        if expander is not None:
            _require_callable_port(expander, method="expand", label="expander")
        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
            or not 1 <= rrf_k <= _MAX_RRF_K
        ):
            raise ValueError(f"rrf_k must be between 1 and {_MAX_RRF_K}")
        self._lexical = lexical
        self._vector = vector
        self._reranker = reranker
        self._expander = expander
        self._rrf_k = rrf_k

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        started_at = perf_counter()
        if self._vector is None:
            lexical = await self._retrieve_lexical(query)
            return await self._finish(
                query,
                lexical.hits,
                candidate_count=lexical.candidate_count,
                latency_ms=lexical.latency_ms,
                profile="lexical",
                started_at=started_at,
            )

        lexical, vector = await _gather_retrievers(
            self._retrieve_lexical(query),
            self._retrieve_vector(query),
        )
        if vector is _VECTOR_UNAVAILABLE:
            return await self._finish(
                query,
                lexical.hits,
                candidate_count=lexical.candidate_count,
                latency_ms=lexical.latency_ms,
                profile="lexical_fallback",
                started_at=started_at,
            )

        fused = _reciprocal_rank_fusion(lexical.hits, vector.hits, rrf_k=self._rrf_k)
        return await self._finish(
            query,
            fused,
            candidate_count=lexical.candidate_count + vector.candidate_count,
            latency_ms=max(lexical.latency_ms, vector.latency_ms),
            profile="hybrid",
            started_at=started_at,
        )

    async def _retrieve_lexical(self, query: SearchQuery) -> SearchResult:
        return _validate_backend_result(
            await self._lexical.retrieve(query),
            query=query,
            label="lexical",
        )

    async def _retrieve_vector(self, query: SearchQuery) -> SearchResult | object:
        if self._vector is None:  # pragma: no cover - guarded by retrieve().
            raise RuntimeError("vector retriever is not configured")
        try:
            result = await self._vector.retrieve(query)
        except RetrieverUnavailable:
            return _VECTOR_UNAVAILABLE
        return _validate_backend_result(result, query=query, label="vector")

    async def _finish(
        self,
        query: SearchQuery,
        hits: Sequence[SearchHit],
        *,
        candidate_count: int,
        latency_ms: float,
        profile: str,
        started_at: float,
    ) -> SearchResult:
        candidates = list(_unique_hits(hits))[: query.candidate_limit]
        scan_limit = min(
            _MAX_POSTPROCESS_SCAN_LIMIT,
            max(_MIN_POSTPROCESS_SCAN_LIMIT, query.candidate_limit + query.limit),
        )
        if self._reranker is not None and candidates:
            reranked = await self._reranker.rerank(query, tuple(candidates))
            candidates = _sanitize_reranked(
                candidates,
                reranked,
                scan_limit=scan_limit,
            )
        direct = candidates[: query.limit]

        final = direct
        if self._expander is not None and final and len(final) < query.limit:
            expanded = await self._expander.expand(query, tuple(final))
            final = _append_expanded(
                final,
                expanded,
                limit=query.limit,
                scan_limit=scan_limit,
            )

        return SearchResult(
            hits=tuple(final),
            candidate_count=candidate_count,
            latency_ms=max(latency_ms, (perf_counter() - started_at) * 1000),
            profile=profile,
        )


def _require_callable_port(port: object, *, method: str, label: str) -> None:
    if not callable(getattr(port, method, None)):
        raise TypeError(f"{label}.{method} must be callable")


async def _gather_retrievers(*coroutines) -> tuple[object, ...]:
    tasks = tuple(asyncio.create_task(coroutine) for coroutine in coroutines)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _validate_backend_result(
    result: object,
    *,
    query: SearchQuery,
    label: str,
) -> SearchResult:
    if type(result) is not SearchResult:
        raise TypeError(f"{label} retriever must return exact SearchResult")
    if len(result.hits) > query.candidate_limit:
        raise ValueError(f"{label} retriever returned more hits than candidate_limit")
    if result.candidate_count < len(result.hits):
        raise ValueError(f"{label} retriever candidate_count is less than returned hits")
    return result


def _unique_hits(hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
    unique: dict[tuple[str, int, int], SearchHit] = {}
    for hit in hits:
        if not isinstance(hit, SearchHit):
            raise TypeError("retriever hits must contain only SearchHit values")
        unique.setdefault(hit.identity, hit)
    return tuple(unique.values())


def _reciprocal_rank_fusion(
    lexical_hits: Sequence[SearchHit],
    vector_hits: Sequence[SearchHit],
    *,
    rrf_k: int,
) -> tuple[SearchHit, ...]:
    representatives: dict[tuple[str, int, int], SearchHit] = {}
    scores: dict[tuple[str, int, int], float] = {}
    for ranked_hits in (_unique_hits(lexical_hits), _unique_hits(vector_hits)):
        for rank, hit in enumerate(ranked_hits, start=1):
            representatives.setdefault(hit.identity, hit)
            scores[hit.identity] = scores.get(hit.identity, 0.0) + 1 / (rrf_k + rank)

    identities = sorted(scores, key=lambda identity: (-scores[identity], identity))
    return tuple(
        replace(representatives[identity], score=scores[identity]) for identity in identities
    )


def _sanitize_reranked(
    direct: Sequence[SearchHit],
    reranked: Sequence[SearchHit],
    *,
    scan_limit: int,
) -> list[SearchHit]:
    known = {hit.identity: hit for hit in direct}
    ordered: list[SearchHit] = []
    seen: set[tuple[str, int, int]] = set()
    for hit in _bounded_output(reranked, label="reranker", scan_limit=scan_limit):
        if not isinstance(hit, SearchHit) or hit.identity not in known or hit.identity in seen:
            continue
        ordered.append(known[hit.identity])
        seen.add(hit.identity)
    ordered.extend(hit for hit in direct if hit.identity not in seen)
    return ordered


def _append_expanded(
    direct: Sequence[SearchHit],
    expanded: Sequence[SearchHit],
    *,
    limit: int,
    scan_limit: int,
) -> list[SearchHit]:
    result = list(direct)
    seen = {hit.identity for hit in result}
    for hit in _bounded_output(expanded, label="expander", scan_limit=scan_limit):
        if not isinstance(hit, SearchHit) or hit.identity in seen:
            continue
        seen.add(hit.identity)
        if len(result) < limit:
            result.append(hit)
    return result


def _bounded_output(output: object, *, label: str, scan_limit: int):
    for index, item in enumerate(islice(output, scan_limit + 1), start=1):
        if index > scan_limit:
            raise ValueError(f"{label} output exceeds scan limit")
        yield item


def _freeze_json_like(value: object, *, label: str) -> object:
    """Recursively freeze JSON-like values into immutable equivalents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not isfinite(value):
            raise TypeError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} mapping keys must be strings")
            frozen[key] = _freeze_json_like(nested, label=label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_like(item, label=label) for item in value)
    if isinstance(value, (set, frozenset)):
        try:
            return frozenset(_freeze_json_like(item, label=label) for item in value)
        except TypeError as exc:
            raise TypeError(f"{label} set items must freeze to hashable values") from exc
    raise TypeError(f"{label} contains unsupported value type: {type(value).__name__}")
