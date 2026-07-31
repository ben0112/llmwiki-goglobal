"""Hosted Postgres retrieval and immutable RAG read projections."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from math import isfinite
from numbers import Real
from pathlib import PurePosixPath
from time import perf_counter
from types import MappingProxyType
from typing import Any, NoReturn
from uuid import UUID

from llmwiki_core.documents import (
    DocumentKind,
    DocumentStatus,
    join_logical_path,
    normalize_directory_path,
)
from llmwiki_core.models import (
    EmbeddingError,
    EmbeddingProfile,
    EmbeddingUnavailable,
    InvalidEmbeddingResponse,
)
from llmwiki_core.postgres_retrieval import compile_postgres_lexical_query
from llmwiki_core.rag import MAX_CONTEXT_CHARS, MAX_PAGE_CHARS
from llmwiki_core.search import (
    HybridRetrievalService,
    RetrieverUnavailable,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchScope,
)
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

_MAX_SELECTED_HITS = 500
_MAX_JSON_BYTES = 64 * 1024
_MAX_JSON_DEPTH = 32
_MAX_PATH_CHARS = 4_096
_MAX_TAGS = 100
_MAX_TAG_CHARS = 128
_POSTGRES_INTEGER_MAX = 2_147_483_647
_EMBEDDING_CLEANUP_FAILED = object()
_HYBRID_SETTINGS_ERROR = "hybrid retrieval is unavailable"

# Retain the public monkeypatch seams without importing the top-level
# ``services`` package merely to use immutable retrieval projections.
PostgresVectorStore = None
OpenAIEmbeddingClient = None


class _SettingsValidationRetriever:
    async def retrieve(self, query: SearchQuery):  # pragma: no cover - validation-only port.
        raise RuntimeError("settings validation retriever must not execute")


@dataclass(frozen=True, slots=True)
class HostedRagRetrievalSettings:
    """One validated immutable view of stateful hosted hybrid settings."""

    embedding_profile: EmbeddingProfile
    lexical_limit: int
    vector_limit: int
    rrf_k: int


def validate_hosted_rag_retrieval_settings(
    settings: object,
    *,
    request_limit: int = 1,
) -> HostedRagRetrievalSettings:
    """Validate the pure hosted hybrid settings used by retrieval and RAG."""
    failure: BaseException | None = None
    try:
        if type(request_limit) is not int or not 1 <= request_limit <= 100:
            raise ValueError
        profile = settings.embedding_profile  # type: ignore[attr-defined]
        mode = settings.MODE  # type: ignore[attr-defined]
        enabled = settings.HYBRID_SEARCH_ENABLED  # type: ignore[attr-defined]
        helper = getattr(settings, "hybrid_candidate_limits", None)
        if callable(helper):
            limits = helper(request_limit)
            if type(limits) is not tuple or len(limits) != 2:
                raise ValueError
            lexical_limit, vector_limit = limits
        else:
            lexical_limit = settings.HYBRID_LEXICAL_CANDIDATES  # type: ignore[attr-defined]
            vector_limit = settings.HYBRID_VECTOR_CANDIDATES  # type: ignore[attr-defined]
        rrf_k = settings.HYBRID_RRF_K  # type: ignore[attr-defined]
        if (
            mode != "hosted"
            or enabled is not True
            or type(profile) is not EmbeddingProfile
            or type(lexical_limit) is not int
            or type(vector_limit) is not int
        ):
            raise ValueError
        SearchQuery("hybrid settings validation", limit=request_limit, candidate_limit=lexical_limit)
        SearchQuery("hybrid settings validation", limit=request_limit, candidate_limit=vector_limit)
        validator = _SettingsValidationRetriever()
        HybridRetrievalService(
            lexical=validator,
            vector=validator,
            rrf_k=rrf_k,
        )
        snapshot = HostedRagRetrievalSettings(
            embedding_profile=profile,
            lexical_limit=lexical_limit,
            vector_limit=vector_limit,
            rrf_k=rrf_k,
        )
    except BaseException as caught:  # noqa: BLE001 - hostile settings boundary.
        failure = caught
    if failure is not None:
        if (signal := sanitized_boundary_signal_or_unknown(failure)) is not None and type(signal) is not BaseException:
            raise signal from None
        raise ValueError(_HYBRID_SETTINGS_ERROR) from None
    return snapshot


@dataclass(frozen=True, slots=True, repr=False)
class RagEvidence:
    """Immutable, current-version evidence selected for a RAG prompt."""

    document_id: UUID
    document_version: int
    chunk_index: int
    page: int | None
    filename: str
    path: str
    title: str | None
    content: str
    status: DocumentStatus
    archived: bool
    score: float
    tags: tuple[str, ...] = ()
    document_kind: DocumentKind = DocumentKind.SOURCE
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, UUID):
            raise ValueError("RAG evidence is invalid")
        _bounded_int(self.document_version, minimum=1, label="document version")
        _bounded_int(self.chunk_index, minimum=0, label="chunk index")
        if self.page is not None:
            _bounded_int(self.page, minimum=1, label="page")
        filename = _filename(self.filename)
        path = _directory_path(self.path)
        title = _optional_text(self.title, maximum=_MAX_PATH_CHARS)
        content = _text(self.content, maximum=MAX_CONTEXT_CHARS, allow_empty=False)
        try:
            status = DocumentStatus(self.status)
            document_kind = DocumentKind(self.document_kind)
        except (TypeError, ValueError):
            raise ValueError("RAG evidence is invalid") from None
        if status is DocumentStatus.FAILED or type(self.archived) is not bool or self.archived:
            raise ValueError("RAG evidence is invalid")
        score = _finite_score(self.score)
        tags = _tags(self.tags)
        metadata = _strict_json_mapping(self.metadata)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "document_kind", document_kind)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True, repr=False)
class RagWikiPage:
    """Immutable projection of one exact current wiki page."""

    document_id: UUID
    version: int
    path: str
    filename: str
    content: str
    title: str | None
    tags: tuple[str, ...]
    date: date | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, UUID):
            raise ValueError("wiki page is invalid")
        _bounded_int(self.version, minimum=1, label="document version")
        filename = _filename(self.filename)
        directory, expected_filename, logical_path = _split_logical_path(self.path)
        if filename != expected_filename or not directory.startswith("/wiki/"):
            raise ValueError("wiki page is invalid")
        content = _text(self.content, maximum=MAX_PAGE_CHARS, allow_empty=True)
        title = _optional_text(self.title, maximum=_MAX_PATH_CHARS)
        tags = _tags(self.tags)
        page_date = _optional_date(self.date)
        metadata = _strict_json_mapping(self.metadata)
        object.__setattr__(self, "path", logical_path)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "date", page_date)
        object.__setattr__(self, "metadata", metadata)


class PostgresLexicalRetriever:
    """Execute the shared Postgres lexical query on an injected database handle."""

    def __init__(
        self,
        database,
        *,
        user_id: object,
        knowledge_base_id: object,
        candidate_limit: int | None = None,
    ) -> None:
        self._database = database
        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._candidate_limit = candidate_limit

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        started_at = perf_counter()
        effective_query = (
            query if self._candidate_limit is None else replace(query, candidate_limit=self._candidate_limit)
        )
        compiled = compile_postgres_lexical_query(
            self._user_id,
            self._knowledge_base_id,
            effective_query,
        )
        rows = await _safe_fetch(
            self._database,
            compiled.sql,
            *compiled.params,
            unavailable_message="lexical retrieval is unavailable",
        )
        validation_failure: BaseException | None = None
        dictionaries: list[dict[str, object]] = []
        hits: tuple[SearchHit, ...] = ()
        candidate_count = 0
        try:
            dictionaries = _validated_lexical_rows(rows, query=effective_query)
            hits = tuple(_search_hit(row) for row in dictionaries)
            candidate_count = dictionaries[0]["candidate_count"] if dictionaries else 0
        except BaseException as failure:  # noqa: BLE001 - sanitize malformed database rows.
            validation_failure = failure
        if validation_failure is not None:
            if signal := sanitized_boundary_signal_or_unknown(validation_failure):
                raise signal from None
            raise RetrieverUnavailable("lexical retrieval is unavailable") from None
        return SearchResult(
            hits=hits,
            candidate_count=candidate_count,
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="lexical",
        )


class PostgresVectorRetriever:
    """Create one query embedding and delegate vector search to the shared store."""

    def __init__(
        self,
        database,
        *,
        user_id: object,
        knowledge_base_id: object,
        profile: EmbeddingProfile,
        embedding_client_factory,
        candidate_limit: int | None = None,
    ) -> None:
        store_type = PostgresVectorStore
        if store_type is None:
            from services.vector_store import PostgresVectorStore as store_type

        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._profile = profile
        self._embedding_client_factory = embedding_client_factory
        self._candidate_limit = candidate_limit
        self._store = store_type(database, profile=profile)

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        if query.scope is not SearchScope.ALL:
            raise RetrieverUnavailable("vector retrieval does not support scoped content")
        client = None
        result = None
        main_failure: BaseException | None = None
        try:
            client = self._embedding_client_factory()
            if getattr(client, "profile", None) != self._profile:
                raise EmbeddingUnavailable("query embedding is unavailable")
            embedding = _validated_query_embedding(
                await client.embed((query.text,)),
                profile=self._profile,
            )
            effective_query = (
                query if self._candidate_limit is None else replace(query, candidate_limit=self._candidate_limit)
            )
            result = _validated_vector_result(
                await self._store.search(
                    user_id=self._user_id,
                    knowledge_base_id=self._knowledge_base_id,
                    query=effective_query,
                    embedding=embedding,
                ),
                query=effective_query,
            )
        except BaseException as failure:  # noqa: BLE001 - classify adapter controls below.
            main_failure = failure
        close_outcome = await _embedding_close_outcome(client)
        outcome = _resolved_vector_outcome(result, main_failure, close_outcome)
        main_failure = None
        result = None
        if isinstance(outcome, BaseException):
            raise outcome from None
        return outcome


class HostedRagRetrieval:
    """Compose hosted lexical and optional vector retrieval through the core service."""

    def __init__(
        self,
        database,
        *,
        user_id: object,
        knowledge_base_id: object,
        settings: object,
        embedding_client_factory=None,
    ) -> None:
        self._database = database
        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._settings = settings
        self._embedding_client_factory = embedding_client_factory

    async def retrieve(
        self,
        query: SearchQuery,
        *,
        profile: str = "lexical",
    ) -> SearchResult:
        if profile == "lexical":
            return await HybridRetrievalService(
                lexical=PostgresLexicalRetriever(
                    self._database,
                    user_id=self._user_id,
                    knowledge_base_id=self._knowledge_base_id,
                )
            ).retrieve(query)
        if profile != "hybrid":
            raise ValueError("unsupported retrieval profile")
        settings = validate_hosted_rag_retrieval_settings(
            self._settings,
            request_limit=query.limit,
        )
        service_query = replace(
            query,
            candidate_limit=max(settings.lexical_limit, settings.vector_limit),
        )
        service = HybridRetrievalService(
            lexical=PostgresLexicalRetriever(
                self._database,
                user_id=self._user_id,
                knowledge_base_id=self._knowledge_base_id,
                candidate_limit=settings.lexical_limit,
            ),
            vector=PostgresVectorRetriever(
                self._database,
                user_id=self._user_id,
                knowledge_base_id=self._knowledge_base_id,
                candidate_limit=settings.vector_limit,
                profile=settings.embedding_profile,
                embedding_client_factory=lambda: self._new_embedding_client(settings.embedding_profile),
            ),
            rrf_k=settings.rrf_k,
        )
        return await service.retrieve(service_query)

    def _new_embedding_client(self, profile: EmbeddingProfile):
        if self._embedding_client_factory is not None:
            return self._embedding_client_factory()
        failure: BaseException | None = None
        client = None
        try:
            client_type = OpenAIEmbeddingClient
            if client_type is None:
                from services.embeddings import OpenAIEmbeddingClient as client_type

            secret = self._settings.EMBEDDING_API_KEY
            client = client_type(
                profile=profile,
                base_url=self._settings.EMBEDDING_BASE_URL,
                api_key=secret.get_secret_value(),
                batch_size=self._settings.EMBEDDING_BATCH_SIZE,
                timeout_seconds=self._settings.EMBEDDING_TIMEOUT_SECONDS,
            )
        except BaseException as caught:  # noqa: BLE001 - sanitize provider construction.
            failure = caught
        if failure is not None:
            if signal := sanitized_boundary_signal_or_unknown(failure):
                raise signal from None
            raise EmbeddingUnavailable("embedding provider unavailable") from None
        if client is None:
            raise EmbeddingUnavailable("embedding provider unavailable")
        return client


class PostgresEvidenceReader:
    """Read selected current chunks once, preserving retrieval order and budget."""

    def __init__(self, database) -> None:
        self._database = database

    async def read(
        self,
        user_id: str | UUID,
        knowledge_base_id: str | UUID,
        hits: Sequence[SearchHit],
        max_chars: int,
    ) -> tuple[RagEvidence, ...]:
        user_uuid = _uuid(user_id, label="user_id")
        knowledge_base_uuid = _uuid(knowledge_base_id, label="knowledge_base_id")
        _bounded_int(max_chars, minimum=1, maximum=MAX_CONTEXT_CHARS, label="max_chars")
        identities, scores = _selected_evidence_inputs(hits)
        if not identities:
            return ()

        sql = (
            "WITH selected AS ("
            "SELECT input.document_id, input.document_version, input.chunk_index, "
            "input.ordinal FROM unnest($3::uuid[], $4::integer[], $5::integer[]) "
            "WITH ORDINALITY AS input(document_id, document_version, chunk_index, ordinal)"
            ") SELECT selected.ordinal, d.id AS document_id, d.version AS document_version, "
            "dc.chunk_index, dc.page, d.filename, d.path, d.title, dc.content, "
            "d.status::text AS status, d.archived, d.tags, d.source_kind, "
            "COALESCE(d.metadata, '{}'::jsonb) AS metadata "
            "FROM selected JOIN documents d ON d.id = selected.document_id "
            "AND d.version = selected.document_version "
            "JOIN document_chunks dc ON dc.document_id = selected.document_id "
            "AND dc.document_version = selected.document_version "
            "AND dc.chunk_index = selected.chunk_index "
            "WHERE d.user_id = $1 AND d.knowledge_base_id = $2 "
            "AND dc.user_id = $1 AND dc.knowledge_base_id = $2 "
            "AND dc.document_version = d.version "
            "AND d.status IN ('pending', 'processing', 'ready') "
            "AND NOT d.archived ORDER BY selected.ordinal"
        )
        rows = await _safe_fetch(
            self._database,
            sql,
            user_uuid,
            knowledge_base_uuid,
            [identity[0] for identity in identities],
            [identity[1] for identity in identities],
            [identity[2] for identity in identities],
            unavailable_message="RAG evidence is unavailable",
        )
        return _evidence_from_database_rows(
            rows,
            identities=identities,
            scores=scores,
            max_chars=max_chars,
        )


class PostgresWikiPageReader:
    """Read one exact, active, current-version wiki page by logical path."""

    def __init__(self, database) -> None:
        self._database = database

    async def get_by_path(
        self,
        user_id: str | UUID,
        knowledge_base_id: str | UUID,
        path: str,
    ) -> RagWikiPage | None:
        user_uuid = _uuid(user_id, label="user_id")
        knowledge_base_uuid = _uuid(knowledge_base_id, label="knowledge_base_id")
        directory, filename, logical_path = _split_logical_path(path)
        if not directory.startswith("/wiki/") or not filename.endswith(".md"):
            raise ValueError("wiki path is invalid")
        sql = (
            "SELECT d.id AS document_id, d.version, d.path, d.filename, d.content, "
            "d.title, d.tags, d.date, COALESCE(d.metadata, '{}'::jsonb) AS metadata "
            "FROM documents d WHERE d.user_id = $1 AND d.knowledge_base_id = $2 "
            "AND d.path = $3 AND d.filename = $4 AND d.source_kind = 'wiki' "
            "AND d.status IN ('pending', 'processing', 'ready') AND NOT d.archived "
            "AND d.content IS NOT NULL AND EXISTS (SELECT 1 FROM document_chunks dc "
            "WHERE dc.document_id = d.id AND dc.document_version = d.version "
            "AND dc.user_id = $1 AND dc.knowledge_base_id = $2)"
        )
        rows = await _safe_fetch(
            self._database,
            sql,
            user_uuid,
            knowledge_base_uuid,
            directory,
            filename,
            unavailable_message="wiki page is unavailable",
        )
        dictionaries = _bounded_database_rows(rows, maximum=1)
        if not dictionaries:
            return None
        row = dictionaries[0]
        page: RagWikiPage | None = None
        page_failure: BaseException | None = None
        try:
            document_id = row.get("document_id")
            if not isinstance(document_id, UUID):
                raise ValueError
            row_directory = _directory_path(row.get("path"))
            row_filename = _filename(row.get("filename"))
            if row_directory != directory or row_filename != filename:
                raise ValueError
            page = RagWikiPage(
                document_id=document_id,
                version=row.get("version"),
                path=logical_path,
                filename=row_filename,
                content=row.get("content"),
                title=row.get("title"),
                tags=row.get("tags"),
                date=row.get("date"),
                metadata=row.get("metadata"),
            )
        except BaseException as failure:  # noqa: BLE001 - sanitize hostile row values.
            page_failure = failure
        if page_failure is not None:
            signal = sanitized_boundary_signal_or_unknown(page_failure)
            page_failure = None
            if signal is not None and type(signal) is not BaseException:
                raise signal from None
            raise RetrieverUnavailable("wiki page is unavailable") from None
        if page is None:
            raise RetrieverUnavailable("wiki page is unavailable")
        return page


def _selected_evidence_inputs(
    hits: object,
) -> tuple[list[tuple[UUID, int, int]], dict[tuple[UUID, int, int], float]]:
    failure: BaseException | None = None
    selected: tuple[list[tuple[UUID, int, int]], dict[tuple[UUID, int, int], float]] | None = None
    try:
        selected = _selected_evidence_inputs_impl(hits)
    except BaseException as caught:  # noqa: BLE001 - sanitize hostile caller values.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        hits = None
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise ValueError("selected evidence hits are invalid") from None
    if selected is None:
        raise ValueError("selected evidence hits are invalid")
    return selected


def _selected_evidence_inputs_impl(
    hits: object,
) -> tuple[list[tuple[UUID, int, int]], dict[tuple[UUID, int, int], float]]:
    if (
        isinstance(hits, (str, bytes))
        or not isinstance(hits, Sequence)
        or len(hits) > _MAX_SELECTED_HITS
        or any(type(hit) is not SearchHit for hit in hits)
    ):
        raise ValueError("selected evidence hits are invalid")
    identities: list[tuple[UUID, int, int]] = []
    scores: dict[tuple[UUID, int, int], float] = {}
    for hit in hits:
        document_id = _uuid(hit.document_id, label="document_id")
        _bounded_int(hit.document_version, minimum=1, label="document version")
        _bounded_int(hit.chunk_index, minimum=0, label="chunk index")
        identity = (document_id, hit.document_version, hit.chunk_index)
        if identity in scores:
            raise ValueError("duplicate evidence identity")
        identities.append(identity)
        scores[identity] = _finite_score(hit.score)
    return identities, scores


def _evidence_from_database_rows(
    rows: object,
    *,
    identities: Sequence[tuple[UUID, int, int]],
    scores: Mapping[tuple[UUID, int, int], float],
    max_chars: int,
) -> tuple[RagEvidence, ...]:
    dictionaries = _bounded_database_rows(rows, maximum=len(identities))
    dictionaries.sort(key=lambda row: _ordinal(row, maximum=len(identities)))
    evidence: list[RagEvidence] = []
    seen_ordinals: set[int] = set()
    used_chars = 0
    for row in dictionaries:
        ordinal = _ordinal(row, maximum=len(identities))
        if ordinal in seen_ordinals:
            raise RetrieverUnavailable("RAG evidence is unavailable")
        seen_ordinals.add(ordinal)
        actual = _row_identity(row)
        if actual != identities[ordinal - 1]:
            raise RetrieverUnavailable("RAG evidence is unavailable")
        content = _evidence_content(row.get("content"))
        if used_chars + len(content) > max_chars:
            break
        evidence.append(_evidence_from_row(row, identity=actual, score=scores[actual]))
        used_chars += len(content)
    return tuple(evidence)


def _evidence_content(value: object) -> str:
    if type(value) is not str:
        raise RetrieverUnavailable("RAG evidence is unavailable")
    failure: BaseException | None = None
    try:
        value.encode("utf-8")
    except BaseException as caught:  # noqa: BLE001 - sanitize invalid backend text.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        value = None
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise RetrieverUnavailable("RAG evidence is unavailable") from None
    return value


def _evidence_from_row(
    row: Mapping[str, object],
    *,
    identity: tuple[UUID, int, int],
    score: float,
) -> RagEvidence:
    item: RagEvidence | None = None
    failure: BaseException | None = None
    try:
        item = RagEvidence(
            document_id=identity[0],
            document_version=identity[1],
            chunk_index=identity[2],
            page=row.get("page"),
            filename=row.get("filename"),
            path=row.get("path"),
            title=row.get("title"),
            content=row.get("content"),
            status=row.get("status"),
            archived=row.get("archived"),
            score=score,
            tags=row.get("tags"),
            document_kind=row.get("source_kind"),
            metadata=row.get("metadata"),
        )
    except BaseException as caught:  # noqa: BLE001 - sanitize hostile row values.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        row = {}
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise RetrieverUnavailable("RAG evidence is unavailable") from None
    if item is None:
        raise RetrieverUnavailable("RAG evidence is unavailable")
    return item


def _validated_vector_result(result: object, *, query: SearchQuery) -> SearchResult:
    if type(result) is not SearchResult:
        raise TypeError("vector store must return exact SearchResult")
    if result.profile != "vector":
        raise ValueError("vector store profile is invalid")
    if len(result.hits) > query.candidate_limit:
        raise ValueError("vector store returned too many hits")
    if result.candidate_count < len(result.hits) or result.candidate_count > _POSTGRES_INTEGER_MAX:
        raise ValueError("vector store candidate count is invalid")
    identities: set[tuple[UUID, int, int]] = set()
    for hit in result.hits:
        if type(hit.document_kind) is not DocumentKind:
            raise ValueError("vector store document kind is invalid")
        document_id = _uuid(hit.document_id, label="document_id")
        version = _bounded_int(hit.document_version, minimum=0, label="document version")
        chunk_index = _bounded_int(
            hit.chunk_index,
            minimum=0,
            maximum=9_999,
            label="chunk index",
        )
        identity = (document_id, version, chunk_index)
        if identity in identities:
            raise ValueError("vector store identities are invalid")
        identities.add(identity)
        _text(hit.content, maximum=1_000_000, allow_empty=True)
        _canonical_logical_path(hit.path)
        _optional_text(hit.title, maximum=_MAX_PATH_CHARS)
        _optional_text(hit.header_breadcrumb, maximum=_MAX_PATH_CHARS)
        if not -1.0 <= _finite_score(hit.score) <= 1.0:
            raise ValueError("vector store score is invalid")
        if hit.page is not None:
            _bounded_int(hit.page, minimum=0, label="page")
        _tags(hit.tags)
        _strict_json_mapping(hit.metadata)
    return result


async def _embedding_close_outcome(client: object | None) -> BaseException | object | None:
    if client is None:
        return None
    try:
        close = getattr(client, "aclose", None)
        if callable(close):
            await close()
    except BaseException as failure:  # noqa: BLE001 - return only sanitized cleanup state.
        if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
            return signal
        return _EMBEDDING_CLEANUP_FAILED
    return None


def _resolved_vector_outcome(
    result: SearchResult | None,
    main_failure: BaseException | None,
    close_outcome: BaseException | object | None,
) -> SearchResult | BaseException:
    failures = tuple(failure for failure in (main_failure, close_outcome) if isinstance(failure, BaseException))
    if (signal := sanitized_boundary_signal_or_unknown(*failures)) and type(signal) is not BaseException:
        return signal
    if close_outcome is _EMBEDDING_CLEANUP_FAILED:
        return RuntimeError("query embedding cleanup failed")
    if main_failure is not None:
        if isinstance(main_failure, (EmbeddingError, RetrieverUnavailable)):
            return RetrieverUnavailable("query embedding is unavailable")
        return RuntimeError("query embedding failed")
    if result is None:
        return RetrieverUnavailable("query embedding is unavailable")
    return result


def _validated_query_embedding(
    vectors: object,
    *,
    profile: EmbeddingProfile,
) -> tuple[float, ...]:
    if (
        isinstance(vectors, (str, bytes))
        or not isinstance(vectors, Sequence)
        or len(vectors) != 1
        or isinstance(vectors[0], (str, bytes))
        or not isinstance(vectors[0], Sequence)
        or len(vectors[0]) != profile.dimensions
    ):
        raise InvalidEmbeddingResponse("invalid embedding response")
    embedding: list[float] = []
    for coordinate in vectors[0]:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise InvalidEmbeddingResponse("invalid embedding response")
        try:
            normalized = float(coordinate)
        except (OverflowError, ValueError):
            raise InvalidEmbeddingResponse("invalid embedding response") from None
        if not isfinite(normalized):
            raise InvalidEmbeddingResponse("invalid embedding response")
        embedding.append(normalized)
    if not any(embedding):
        raise InvalidEmbeddingResponse("invalid embedding response")
    return tuple(embedding)


def _search_hit(row: Mapping[str, object]) -> SearchHit:
    raw_metadata = row["metadata"]
    if isinstance(raw_metadata, str):
        raw_metadata = json.loads(raw_metadata)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    return SearchHit(
        document_id=str(row["document_id"]),
        document_version=row["document_version"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        score=row["score"],
        path=f"{row['path']}{row['filename']}",
        title=row.get("title"),
        page=row.get("page"),
        header_breadcrumb=row.get("header_breadcrumb"),
        tags=row["tags"],
        document_kind=DocumentKind(row["source_kind"]),
        metadata=metadata,
    )


def _validated_lexical_rows(
    rows: object,
    *,
    query: SearchQuery,
) -> list[dict[str, object]]:
    dictionaries = _bounded_database_rows(rows, maximum=query.candidate_limit)
    required = {
        "candidate_count",
        "chunk_index",
        "content",
        "document_id",
        "document_version",
        "filename",
        "metadata",
        "page",
        "path",
        "score",
        "source_kind",
        "tags",
    }
    candidate_count: int | None = None
    normalized: list[dict[str, object]] = []
    for row in dictionaries:
        if not required <= row.keys():
            raise ValueError("lexical row is incomplete")
        count = row["candidate_count"]
        if (
            type(count) is not int
            or not 0 <= count <= _POSTGRES_INTEGER_MAX
            or (candidate_count is not None and candidate_count != count)
        ):
            raise ValueError("lexical row is invalid")
        document_id = row["document_id"]
        if not isinstance(document_id, UUID):
            raise ValueError("lexical row is invalid")
        _bounded_int(row["document_version"], minimum=0, label="document version")
        _bounded_int(row["chunk_index"], minimum=0, maximum=9_999, label="chunk index")
        content = _text(row["content"], maximum=1_000_000, allow_empty=True)
        directory = _directory_path(row["path"])
        filename = _filename(row["filename"])
        score = _finite_score(row["score"])
        page = row["page"]
        if page is not None:
            _bounded_int(page, minimum=0, label="page")
        title = _optional_text(row.get("title"), maximum=_MAX_PATH_CHARS)
        header = _optional_text(row.get("header_breadcrumb"), maximum=_MAX_PATH_CHARS)
        tags = _tags(row["tags"])
        try:
            source_kind = DocumentKind(row["source_kind"])
        except (TypeError, ValueError):
            raise ValueError("lexical row is invalid") from None
        metadata = _strict_json_mapping(row["metadata"])
        candidate_count = count
        normalized.append(
            {
                **row,
                "content": content,
                "path": directory,
                "filename": filename,
                "score": score,
                "title": title,
                "header_breadcrumb": header,
                "tags": tags,
                "source_kind": source_kind.value,
                "metadata": metadata,
            }
        )
    if candidate_count is not None and candidate_count < len(normalized):
        raise ValueError("lexical candidate count is invalid")
    return normalized


async def _safe_fetch(database, sql: str, *params: object, unavailable_message: str):
    failure: BaseException | None = None
    rows = None
    try:
        rows = await database.fetch(sql, *params)
    except BaseException as caught:  # noqa: BLE001 - sanitize the database boundary.
        failure = caught
    if failure is not None:
        if signal := sanitized_boundary_signal_or_unknown(failure):
            raise signal from None
        raise RetrieverUnavailable(unavailable_message) from None
    return rows


def _bounded_database_rows(rows: object, *, maximum: int) -> list[dict[str, object]]:
    failure: BaseException | None = None
    dictionaries: list[dict[str, object]] = []
    try:
        dictionaries = _bounded_database_rows_impl(rows, maximum=maximum)
    except BaseException as caught:  # noqa: BLE001 - sanitize hostile row containers.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        rows = None
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise RetrieverUnavailable("database returned invalid rows") from None
    return dictionaries


def _bounded_database_rows_impl(rows: object, *, maximum: int) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence) or len(rows) > maximum:
        raise RetrieverUnavailable("database returned invalid rows")
    return [dict(raw) for raw in rows]


def _ordinal(row: Mapping[str, object], *, maximum: int) -> int:
    value = row.get("ordinal")
    if type(value) is not int or not 1 <= value <= maximum:
        raise RetrieverUnavailable("RAG evidence is unavailable")
    return value


def _row_identity(row: Mapping[str, object]) -> tuple[UUID, int, int]:
    document_id = row.get("document_id")
    version = row.get("document_version")
    chunk_index = row.get("chunk_index")
    if not isinstance(document_id, UUID):
        raise RetrieverUnavailable("RAG evidence is unavailable")
    try:
        _bounded_int(version, minimum=1, label="document version")
        _bounded_int(chunk_index, minimum=0, label="chunk index")
    except ValueError:
        raise RetrieverUnavailable("RAG evidence is unavailable") from None
    return document_id, version, chunk_index


def _uuid(value: object, *, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if type(value) is not str:
        raise ValueError(f"{label} must be a UUID")
    try:
        parsed = UUID(value)
    except ValueError:
        raise ValueError(f"{label} must be a UUID") from None
    if str(parsed) != value:
        raise ValueError(f"{label} must be a UUID")
    return parsed


def _bounded_int(
    value: object,
    *,
    minimum: int,
    label: str,
    maximum: int = _POSTGRES_INTEGER_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside its allowed range")
    return value


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("score must be finite")
    try:
        score = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("score must be finite") from None
    if not isfinite(score):
        raise ValueError("score must be finite")
    return score


def _text(value: object, *, maximum: int, allow_empty: bool) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise ValueError("text is invalid")
    if len(value) > maximum or "\x00" in value:
        raise ValueError("text is invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("text is invalid") from None
    return value


def _optional_text(value: object, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, maximum=maximum, allow_empty=True)


def _filename(value: object) -> str:
    filename = _text(value, maximum=_MAX_PATH_CHARS, allow_empty=False)
    if filename in {".", ".."} or PurePosixPath(filename).name != filename or "\\" in filename:
        raise ValueError("filename is invalid")
    return filename


def _directory_path(value: object) -> str:
    if type(value) is not str or not value.startswith("/"):
        raise ValueError("directory path is invalid")
    _text(value, maximum=_MAX_PATH_CHARS, allow_empty=False)
    normalized = normalize_directory_path(value)
    if normalized != value:
        raise ValueError("directory path is invalid")
    return normalized


def _split_logical_path(value: object) -> tuple[str, str, str]:
    if type(value) is not str or not value.startswith("/"):
        raise ValueError("wiki path is invalid")
    _text(value, maximum=_MAX_PATH_CHARS, allow_empty=False)
    raw_parts = value.replace("\\", "/").split("/")
    if any(part == ".." for part in raw_parts):
        raise ValueError("wiki path is invalid")
    parts = [part for part in raw_parts if part and part != "."]
    if not parts:
        raise ValueError("wiki path is invalid")
    filename = _filename(parts[-1])
    directory = normalize_directory_path("/" + "/".join(parts[:-1]))
    logical_path = join_logical_path(directory, filename)
    return directory, filename, logical_path


def _canonical_logical_path(value: object) -> str:
    _directory, _filename_value, logical_path = _split_logical_path(value)
    if logical_path != value:
        raise ValueError("logical path is invalid")
    return logical_path


def _tags(value: object) -> tuple[str, ...]:
    failure: BaseException | None = None
    tags: tuple[str, ...] = ()
    try:
        tags = _tags_impl(value)
    except BaseException as caught:  # noqa: BLE001 - sanitize hostile containers.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        value = None
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise ValueError("tags are invalid") from None
    return tags


def _tags_impl(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) > _MAX_TAGS:
        raise ValueError("tags are invalid")
    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in value:
        tag = _text(raw_tag, maximum=_MAX_TAG_CHARS, allow_empty=False)
        if tag.strip() != tag or tag in seen:
            raise ValueError("tags are invalid")
        seen.add(tag)
        tags.append(tag)
    return tuple(tags)


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if type(value) is date:
        return value
    if type(value) is not str or len(value) != 10:
        raise ValueError("date is invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("date is invalid") from None
    if parsed.isoformat() != value:
        raise ValueError("date is invalid")
    return parsed


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("metadata is invalid")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("metadata is invalid")
        result[key] = value
    return result


def _strict_json_mapping(value: object) -> Mapping[str, Any]:
    failure: BaseException | None = None
    metadata: Mapping[str, Any] | None = None
    try:
        metadata = _strict_json_mapping_impl(value)
    except BaseException as caught:  # noqa: BLE001 - sanitize hostile containers.
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        failure = None
        value = None
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
        raise ValueError("metadata is invalid") from None
    if metadata is None:
        raise ValueError("metadata is invalid")
    return metadata


def _strict_json_mapping_impl(value: object) -> Mapping[str, Any]:
    if type(value) is str:
        try:
            if len(value.encode("utf-8")) > _MAX_JSON_BYTES:
                raise ValueError
            value = json.loads(
                value,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeEncodeError, json.JSONDecodeError, RecursionError, ValueError):
            raise ValueError("metadata is invalid") from None
    if not isinstance(value, Mapping):
        raise ValueError("metadata is invalid")
    plain = dict(value)
    _validate_json_tree(plain)
    try:
        encoded = json.dumps(
            plain,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise ValueError("metadata is invalid") from None
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("metadata is invalid")
    return _freeze_json(plain)


def _validate_json_tree(value: object) -> None:
    pending = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("metadata is invalid")
        if isinstance(current, Mapping):
            if any(type(key) is not str for key in current):
                raise ValueError("metadata is invalid")
            pending.extend((key, depth + 1) for key in current)
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            pending.extend((item, depth + 1) for item in current)
        elif type(current) is str:
            _text(current, maximum=_MAX_JSON_BYTES, allow_empty=True)
        elif (
            current is not None
            and type(current) not in {bool, int, float}
            or type(current) is float
            and not isfinite(current)
        ):
            raise ValueError("metadata is invalid")


def _freeze_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


__all__ = [
    "HostedRagRetrieval",
    "HostedRagRetrievalSettings",
    "PostgresEvidenceReader",
    "PostgresLexicalRetriever",
    "PostgresVectorRetriever",
    "PostgresWikiPageReader",
    "RagEvidence",
    "RagWikiPage",
    "validate_hosted_rag_retrieval_settings",
]
