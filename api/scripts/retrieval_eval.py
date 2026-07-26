"""Deterministic, content-free retrieval evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, BinaryIO, Never
from uuid import UUID

from llmwiki_core import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationReport,
    EvaluationRun,
    HybridRetrievalService,
    RankedResult,
    RetrieverUnavailable,
    SearchArea,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchScope,
    evaluate_rankings,
    evaluation_dataset_digest,
    load_cases,
    promotion_decision,
)
from llmwiki_core.documents import DocumentKind
from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

REPORT_SCHEMA_VERSION = 1
MAX_CORPUS_BYTES = 8 * 1024 * 1024
MAX_CORPUS_LINE_BYTES = 256 * 1024
MAX_CORPUS_DOCUMENTS = 10_000
MAX_CORPUS_CHUNKS = 100_000
MAX_CORPUS_TEXT_CHARS = 1_000_000
_POSTGRES_INTEGER_MAX = 2_147_483_647
_MAX_BACKEND_PATH_CHARS = 4_096
_MAX_BACKEND_METADATA_BYTES = 64 * 1_024
_MAX_HOSTED_DSN_CHARS = 8_192
_MAX_HOSTED_SECRET_CHARS = 16_384

_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "document_id", "document_kind", "path", "title", "tags", "facets", "chunks"}
)
_DOCUMENT_REQUIRED_FIELDS = frozenset(
    {"schema_version", "document_id", "document_kind", "path", "title", "chunks"}
)
_CHUNK_FIELDS = frozenset({"chunk_index", "content", "source_content", "annotations_text"})
_CHUNK_REQUIRED_FIELDS = frozenset({"chunk_index", "content"})
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

RetrieverFactory = Callable[[str, Path], object]


class HybridConfigurationUnavailable(RuntimeError):
    """The later hosted hybrid-service wiring is not configured."""


class EvaluationDatasetError(RuntimeError):
    """The evaluation cohort or its sibling synthetic corpus is invalid."""


class RetrievalContractError(RuntimeError):
    """An injected retriever violated the shared retrieval contract."""


class RetrievalExecutionError(RuntimeError):
    """A retriever failed without exposing its backend exception."""


class OutputWriteError(RuntimeError):
    """The requested report output could not be written safely."""


_SAFE_BACKEND_EXCEPTIONS = (
    EvaluationDatasetError,
    HybridConfigurationUnavailable,
    RetrievalContractError,
    RetrievalExecutionError,
)


def _sanitized_process_control(error: BaseException) -> KeyboardInterrupt | SystemExit | None:
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt()
    if isinstance(error, SystemExit):
        if isinstance(error.code, bool):
            return SystemExit(int(error.code))
        return SystemExit(error.code if type(error.code) is int else 1)
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            if process_control := _sanitized_process_control(nested):
                return process_control
    return None


def _sanitized_backend_failure(error: BaseException) -> BaseException:
    if process_control := _sanitized_process_control(error):
        return process_control
    return RetrievalExecutionError()


def _backend_call(
    operation: Callable[[], Any],
    *,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Any:
    failure: BaseException | None = None
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - the privacy boundary must classify BaseExceptionGroup.
        failure = error if isinstance(error, passthrough) else _sanitized_backend_failure(error)
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("backend boundary lost its failure")
    raise failure from None


async def _await_backend(awaitable: object) -> Any:
    failure: BaseException | None = None
    try:
        return await awaitable
    except BaseException as error:  # noqa: BLE001 - the privacy boundary must classify BaseExceptionGroup.
        failure = _sanitized_backend_failure(error)
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("backend boundary lost its failure")
    raise failure from None


class _ArgumentError(ValueError):
    pass


class _HelpRequested(RuntimeError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _ArgumentError

    def print_help(self, file: Any = None) -> None:
        del file
        raise _HelpRequested

    def exit(self, status: int = 0, _message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested
        raise _ArgumentError


@dataclass(frozen=True, slots=True)
class _CorpusChunk:
    document_id: str
    document_kind: DocumentKind
    path: str
    title: str
    tags: frozenset[str]
    facets: Mapping[str, object]
    chunk_index: int
    content: str
    source_content: str
    annotations_text: str

    @property
    def annotated(self) -> bool:
        return bool(self.annotations_text.strip())


class _SyntheticLexicalRetriever:
    """Small deterministic lexical adapter for the sibling synthetic corpus."""

    def __init__(self, chunks: Sequence[_CorpusChunk]) -> None:
        self._chunks = tuple(chunks)

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        ranked: list[tuple[int, _CorpusChunk]] = []
        query_tokens = _tokenize(query.text)
        for chunk in self._chunks:
            if not _matches_filters(chunk, query):
                continue
            searchable = _scope_content(chunk, query.scope)
            corpus_tokens = _tokenize(searchable)
            content_score = sum(corpus_tokens.count(token) for token in query_tokens)
            if content_score > 0:
                title_tokens = _tokenize(chunk.title)
                title_score = sum(title_tokens.count(token) for token in query_tokens)
                ranked.append((content_score * 100 + title_score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].document_id, item[1].chunk_index))
        candidates = ranked[: query.candidate_limit]
        hits = tuple(
            SearchHit(
                document_id=chunk.document_id,
                document_version=1,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=float(score),
                path=chunk.path,
                title=chunk.title,
                tags=tuple(sorted(chunk.tags)),
                document_kind=chunk.document_kind,
                metadata={"facets": dict(chunk.facets), "synthetic": True},
            )
            for score, chunk in candidates
        )
        return SearchResult(
            hits=hits,
            candidate_count=len(ranked),
            latency_ms=0.0,
            profile="lexical",
        )


class _PostgresEvaluationLexicalRetriever:
    """Tenant-scoped PostgreSQL lexical adapter used by evidence runs."""

    def __init__(
        self,
        pool: object,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        candidate_limit: int | None = None,
    ) -> None:
        self._pool = pool
        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._candidate_limit = candidate_limit

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        from services.vector_store import _facet_conditions, _logical_glob_to_sql_like

        effective_query = _query_with_candidate_limit(query, self._candidate_limit)
        started_at = perf_counter()
        params: list[object] = [
            self._user_id,
            self._knowledge_base_id,
            effective_query.text,
        ]

        def bind(value: object) -> str:
            params.append(value)
            return f"${len(params)}"

        where = [
            "d.user_id=$1",
            "d.knowledge_base_id=$2",
            "dc.user_id=$1",
            "dc.knowledge_base_id=$2",
            "dc.document_version=d.version",
            "d.status='ready'",
            "NOT d.archived",
        ]
        if effective_query.annotated_only:
            where.append("dc.has_highlight=true")
        if effective_query.area is SearchArea.WIKI:
            where.append("d.source_kind='wiki'")
        elif effective_query.area is SearchArea.SOURCES:
            where.append("d.source_kind!='wiki'")
        if effective_query.document_kinds:
            kinds = bind([kind.value for kind in effective_query.document_kinds])
            where.append(f"d.source_kind=ANY({kinds}::text[])")
        if effective_query.path_glob is not None:
            path_pattern = bind(_logical_glob_to_sql_like(effective_query.path_glob))
            where.append(f"(d.path || d.filename) LIKE {path_pattern} ESCAPE '\\'")
        if effective_query.tags:
            tags = bind(list(effective_query.tags))
            where.append(
                "ARRAY(SELECT lower(tag) FROM unnest(COALESCE(d.tags, ARRAY[]::text[])) tag) "
                f"@> {tags}::text[]"
            )
        where.extend(_facet_conditions(dict(effective_query.facets), bind=bind))

        searchable = {
            SearchScope.ALL: "dc.content",
            SearchScope.SOURCE: "dc.source_content",
            SearchScope.ANNOTATIONS: "COALESCE(dc.annotations_text, '')",
        }[effective_query.scope]
        where.append(
            f"to_tsvector('simple', {searchable}) @@ plainto_tsquery('simple', $3)"
        )
        limit_parameter = bind(effective_query.candidate_limit)
        sql = (
            "WITH filtered AS ("
            "SELECT dc.document_id, dc.document_version, dc.chunk_index, dc.content, "
            "dc.page, dc.header_breadcrumb, d.path, d.filename, d.title, "
            "COALESCE(d.tags, ARRAY[]::text[]) AS tags, d.source_kind, "
            "COALESCE(d.metadata, '{}'::jsonb)::text AS metadata, "
            f"ts_rank_cd(to_tsvector('simple', {searchable}), "
            "plainto_tsquery('simple', $3)) AS score "
            "FROM document_chunks dc JOIN documents d ON d.id=dc.document_id "
            f"WHERE {' AND '.join(where)}"
            "), counted AS ("
            "SELECT *, count(*) OVER () AS candidate_count FROM filtered"
            ") SELECT * FROM counted "
            "ORDER BY score DESC, document_id, document_version, chunk_index "
            f"LIMIT {limit_parameter}"
        )
        rows = await _postgres_fetch(self._pool, sql, params, label="lexical")
        dictionaries = tuple(dict(row) for row in rows)
        hits = tuple(_postgres_evaluation_hit(row) for row in dictionaries)
        return SearchResult(
            hits=hits,
            candidate_count=(dictionaries[0]["candidate_count"] if dictionaries else 0),
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="lexical",
        )


class _PostgresEvaluationVectorRetriever:
    """Query-embedding adapter over the production pgvector store."""

    def __init__(
        self,
        pool: object,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        profile: EmbeddingProfile,
        embedding_client_factory: Callable[[], object],
        candidate_limit: int,
        profile_is_available: Callable[[], Coroutine[Any, Any, bool]],
    ) -> None:
        from services.vector_store import PostgresVectorStore

        self._pool = pool
        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._profile = profile
        self._embedding_client_factory = embedding_client_factory
        self._candidate_limit = candidate_limit
        self._profile_is_available = profile_is_available
        self._store = PostgresVectorStore(pool, profile=profile)

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        if not await self._profile_is_available():
            raise RetrieverUnavailable("evaluation vectors are unavailable")
        client = self._embedding_client_factory()
        try:
            if getattr(client, "profile", None) != self._profile:
                raise RetrieverUnavailable("evaluation query embedding is unavailable")
            embed = getattr(client, "embed", None)
            if not callable(embed):
                raise RetrieverUnavailable("evaluation query embedding is unavailable")
            vectors = await embed((query.text,))
            embedding = _validated_evaluation_query_embedding(
                vectors,
                dimensions=self._profile.dimensions,
            )
        except RetrieverUnavailable:
            raise
        except BaseException as error:  # noqa: BLE001 - provider details must not cross the boundary.
            if process_control := _sanitized_process_control(error):
                raise process_control from None
            raise RetrieverUnavailable("evaluation query embedding is unavailable") from None
        finally:
            close = getattr(client, "aclose", None)
            if callable(close):
                try:
                    await close()
                except BaseException as error:  # noqa: BLE001 - cleanup shares the privacy boundary.
                    if process_control := _sanitized_process_control(error):
                        raise process_control from None

        effective_query = _query_with_candidate_limit(query, self._candidate_limit)
        return await self._store.search(
            user_id=self._user_id,
            knowledge_base_id=self._knowledge_base_id,
            query=effective_query,
            embedding=embedding,
        )


class _EvaluationLatencyRetriever:
    def __init__(
        self,
        retriever: object,
        *,
        selected_profile: str,
        latency_ms: Callable[[str, SearchQuery], Real] | None,
    ) -> None:
        self._retriever = retriever
        self._selected_profile = selected_profile
        self._latency_ms = latency_ms

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        result = await self._retriever.retrieve(query)
        if self._latency_ms is None:
            return result
        latency = self._latency_ms(self._selected_profile, query)
        return SearchResult(
            hits=result.hits,
            candidate_count=result.candidate_count,
            latency_ms=latency,
            profile=result.profile,
        )


class _CleanupStatus(Enum):
    MISSING = "missing"
    FAILED = "failed"
    COMPLETED_SUCCESSFULLY = "completed_successfully"


@dataclass
class _CleanupObligations:
    release: bool = False
    rollback: bool = False


@dataclass
class _CleanupProofs:
    released: bool = False
    rolled_back: bool = False


async def _guarded_async_cleanup(
    target: object,
    attribute: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    failures: list[BaseException],
) -> _CleanupStatus:
    """Attempt cleanup and report whether it completed, failed, or was missing."""

    try:
        operation = getattr(target, attribute, None)
    except BaseException as error:  # noqa: BLE001 - cleanup must inventory every failure.
        failures.append(error)
        return _CleanupStatus.FAILED
    if not callable(operation):
        return _CleanupStatus.MISSING
    try:
        pending = operation(*args, **kwargs)
        if inspect.isawaitable(pending):
            await pending
        else:
            raise TypeError("cleanup operation must be awaitable")
    except BaseException as error:  # noqa: BLE001 - cleanup must inventory every failure.
        failures.append(error)
        return _CleanupStatus.FAILED
    return _CleanupStatus.COMPLETED_SUCCESSFULLY


def _validate_cleanup_obligations(
    obligations: _CleanupObligations,
    proofs: _CleanupProofs,
    failures: list[BaseException],
) -> None:
    """Record one stable failure for every resource obligation without proof."""

    if obligations.rollback and not proofs.rolled_back:
        failures.append(RetrievalExecutionError())
    if obligations.release and not proofs.released:
        failures.append(RetrievalExecutionError())


async def _settle_cleanup_obligations(
    *,
    pool: object,
    lease: object | None,
    lease_entered: bool,
    connection: object | None,
    transaction: object | None,
    obligations: _CleanupObligations,
    failures: list[BaseException],
) -> _CleanupProofs:
    """Attempt every required cleanup, then validate all resource proofs once."""

    proofs = _CleanupProofs()
    if obligations.rollback and transaction is not None:
        status = await _guarded_async_cleanup(
            transaction,
            "rollback",
            (),
            {},
            failures,
        )
        proofs.rolled_back = status is _CleanupStatus.COMPLETED_SUCCESSFULLY
    if obligations.release and connection is not None:
        if lease_entered and lease is not None:
            status = await _guarded_async_cleanup(
                lease,
                "__aexit__",
                (None, None, None),
                {},
                failures,
            )
            proofs.released = status is _CleanupStatus.COMPLETED_SUCCESSFULLY
        if not proofs.released:
            status = await _guarded_async_cleanup(
                pool,
                "release",
                (connection,),
                {},
                failures,
            )
            proofs.released = status is _CleanupStatus.COMPLETED_SUCCESSFULLY
    _validate_cleanup_obligations(obligations, proofs, failures)
    return proofs


def _raise_session_failure(error: BaseException) -> Never:
    """Raise a fresh boundary failure without retaining the injected body error."""

    try:
        raise error from None
    finally:
        error.__cause__ = None
        error.__context__ = None


class _PostgresEvaluationSnapshot:
    """One strict, transaction-bound read surface for an evaluation run."""

    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._fetch_lock = asyncio.Lock()
        self._active = True

    def revoke(self) -> None:
        self._active = False

    async def fetch(self, sql: str, *params: object) -> Sequence[object]:
        if not self._active:
            raise RetrieverUnavailable("evaluation store is unavailable") from None
        fetch = getattr(self._connection, "fetch", None)
        if not callable(fetch):
            raise RetrieverUnavailable("evaluation store is unavailable")
        async with self._fetch_lock:
            if not self._active:
                raise RetrieverUnavailable("evaluation store is unavailable") from None
            rows = await fetch(sql, *params)
        if "candidate_count" not in sql:
            return rows
        profile = "vector" if "1.0 - distance AS score" in sql else "lexical"
        return _validated_evaluation_rows(rows, profile=profile)


class _PostgresEvaluationFactory:
    """Reusable configuration with one fresh snapshot per evaluation request."""

    def __init__(
        self,
        pool: object,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        embedding_profile: EmbeddingProfile,
        embedding_client_factory: Callable[[], object] | None,
        lexical_candidate_limit: int,
        vector_candidate_limit: int,
        rrf_k: int,
        latency_ms: Callable[[str, SearchQuery], Real] | None,
    ) -> None:
        self._pool = pool
        self._user_id = user_id
        self._knowledge_base_id = knowledge_base_id
        self._embedding_profile = embedding_profile
        self._embedding_client_factory = embedding_client_factory
        self._lexical_candidate_limit = lexical_candidate_limit
        self._vector_candidate_limit = vector_candidate_limit
        self._rrf_k = rrf_k
        self._latency_ms = latency_ms
        self._session_active = False
        self._snapshot: _PostgresEvaluationSnapshot | None = None
        self._coverage_cache: dict[EmbeddingProfile, bool] = {}

    def __call__(self, profile: str, _dataset_path: Path) -> object:
        snapshot = self._snapshot
        if not self._session_active or snapshot is None:
            raise RetrievalContractError from None
        if profile == "lexical":
            retriever: object = _PostgresEvaluationLexicalRetriever(
                snapshot,
                user_id=self._user_id,
                knowledge_base_id=self._knowledge_base_id,
            )
        elif profile == "hybrid":
            if self._embedding_client_factory is None:
                raise HybridConfigurationUnavailable
            retriever = HybridRetrievalService(
                lexical=_PostgresEvaluationLexicalRetriever(
                    snapshot,
                    user_id=self._user_id,
                    knowledge_base_id=self._knowledge_base_id,
                    candidate_limit=self._lexical_candidate_limit,
                ),
                vector=_PostgresEvaluationVectorRetriever(
                    snapshot,
                    user_id=self._user_id,
                    knowledge_base_id=self._knowledge_base_id,
                    profile=self._embedding_profile,
                    embedding_client_factory=self._embedding_client_factory,
                    candidate_limit=self._vector_candidate_limit,
                    profile_is_available=self._profile_is_available,
                ),
                rrf_k=self._rrf_k,
            )
        else:
            raise RetrievalContractError
        return _EvaluationLatencyRetriever(
            retriever,
            selected_profile=profile,
            latency_ms=self._latency_ms,
        )

    async def _profile_is_available(self) -> bool:
        if not self._session_active:
            raise RetrieverUnavailable("vector evaluation store is unavailable") from None
        cached = self._coverage_cache.get(self._embedding_profile)
        if cached is not None:
            return cached
        snapshot = self._snapshot
        if snapshot is None:
            raise RetrieverUnavailable("vector evaluation store is unavailable")
        rows = await _postgres_fetch(
            snapshot,
            "WITH current_chunks AS ("
            "SELECT dc.document_id, dc.document_version, dc.chunk_index "
            "FROM document_chunks dc JOIN documents d ON d.id=dc.document_id "
            "WHERE dc.user_id=$1 AND dc.knowledge_base_id=$2 "
            "AND d.user_id=$1 AND d.knowledge_base_id=$2 "
            "AND dc.document_version=d.version AND d.status='ready' AND NOT d.archived"
            ") SELECT EXISTS(SELECT 1 FROM current_chunks) "
            "AND NOT EXISTS("
            "SELECT 1 FROM current_chunks cc WHERE NOT EXISTS("
            "SELECT 1 FROM chunk_embeddings ce "
            "WHERE ce.user_id=$1 AND ce.knowledge_base_id=$2 "
            "AND ce.document_id=cc.document_id "
            "AND ce.document_version=cc.document_version "
            "AND ce.chunk_index=cc.chunk_index "
            "AND ce.provider=$3 AND ce.model=$4 AND ce.dimensions=$5"
            ")) AND NOT EXISTS("
            "SELECT 1 FROM chunk_embeddings ce JOIN documents d ON d.id=ce.document_id "
            "WHERE ce.user_id=$1 AND ce.knowledge_base_id=$2 "
            "AND ce.provider=$3 AND ce.model=$4 AND ce.dimensions=$5 "
            "AND d.user_id=$1 AND d.knowledge_base_id=$2 "
            "AND ce.document_version=d.version AND NOT d.archived "
            "AND d.status NOT IN ('ready','failed')"
            ") AS available",
            (
                self._user_id,
                self._knowledge_base_id,
                self._embedding_profile.provider,
                self._embedding_profile.model,
                self._embedding_profile.dimensions,
            ),
            label="vector",
        )
        available = bool(rows and dict(rows[0]).get("available") is True)
        self._coverage_cache[self._embedding_profile] = available
        return available

    @asynccontextmanager
    async def evaluation_session(self) -> AsyncIterator[None]:  # noqa: C901 - linear lifecycle inventory.
        if self._session_active:
            raise RetrievalContractError from None
        self._session_active = True
        self._snapshot = None
        self._coverage_cache.clear()
        lease: object | None = None
        lease_entered = False
        connection: object | None = None
        transaction: object | None = None
        obligations = _CleanupObligations()
        primary: BaseException | None = None
        cleanup_failures: list[BaseException] = []
        try:
            acquire = getattr(self._pool, "acquire", None)
            if not callable(acquire):
                raise RetrievalExecutionError
            lease = acquire()
            enter = getattr(lease, "__aenter__", None)
            if callable(enter):
                connection = await enter()
                lease_entered = True
            elif inspect.isawaitable(lease):
                connection = await lease
            else:
                connection = lease
            obligations.release = True
            begin = getattr(connection, "transaction", None)
            if not callable(begin):
                raise RetrievalExecutionError
            transaction = begin(isolation="repeatable_read", readonly=True)
            start = getattr(transaction, "start", None)
            if not callable(start):
                raise RetrievalExecutionError
            obligations.rollback = True
            await start()
            self._snapshot = _PostgresEvaluationSnapshot(connection)
            self._coverage_cache.clear()
            yield
        except BaseException as error:  # noqa: BLE001 - cleanup must run for all process controls.
            primary = error
        finally:
            snapshot = self._snapshot
            if snapshot is not None:
                snapshot.revoke()
            self._snapshot = None
            self._coverage_cache.clear()
            await _settle_cleanup_obligations(
                pool=self._pool,
                lease=lease,
                lease_entered=lease_entered,
                connection=connection,
                transaction=transaction,
                obligations=obligations,
                failures=cleanup_failures,
            )
            self._session_active = False

        if primary is not None:
            if signal := sanitized_boundary_signal_or_unknown(primary, *cleanup_failures):
                _raise_session_failure(signal)
            if isinstance(primary, _SAFE_BACKEND_EXCEPTIONS):
                _raise_session_failure(type(primary)())
            _raise_session_failure(RetrievalExecutionError())
        if cleanup_failures:
            if signal := sanitized_boundary_signal_or_unknown(*cleanup_failures):
                _raise_session_failure(signal)
            _raise_session_failure(RetrievalExecutionError())


def postgres_evaluation_retriever_factory(
    pool: object,
    *,
    user_id: str | UUID,
    knowledge_base_id: str | UUID,
    embedding_profile: EmbeddingProfile,
    embedding_client_factory: Callable[[], object] | None,
    lexical_candidate_limit: int,
    vector_candidate_limit: int,
    rrf_k: int,
    latency_ms: Callable[[str, SearchQuery], Real] | None = None,
) -> RetrieverFactory:
    """Build real PostgreSQL lexical/hybrid adapters for one evaluation tenant.

    Fake embeddings and deterministic latency are possible only through explicit
    caller injection; neither becomes a deployment default.
    """

    user_uuid = _evaluation_uuid(user_id)
    knowledge_base_uuid = _evaluation_uuid(knowledge_base_id)
    if not isinstance(embedding_profile, EmbeddingProfile):
        raise TypeError("embedding_profile must be an EmbeddingProfile")
    if embedding_client_factory is not None and not callable(embedding_client_factory):
        raise TypeError("embedding_client_factory must be callable")
    for value in (lexical_candidate_limit, vector_candidate_limit):
        if type(value) is not int or not 1 <= value <= 500:
            raise ValueError("evaluation candidate limits must be between 1 and 500")
    if latency_ms is not None and not callable(latency_ms):
        raise TypeError("latency_ms must be callable")

    return _PostgresEvaluationFactory(
        pool,
        user_id=user_uuid,
        knowledge_base_id=knowledge_base_uuid,
        embedding_profile=embedding_profile,
        embedding_client_factory=embedding_client_factory,
        lexical_candidate_limit=lexical_candidate_limit,
        vector_candidate_limit=vector_candidate_limit,
        rrf_k=rrf_k,
        latency_ms=latency_ms,
    )


def _evaluation_uuid(value: str | UUID) -> UUID:
    if isinstance(value, bool):
        raise ValueError("evaluation scope must contain UUIDs")
    try:
        return value if isinstance(value, UUID) else UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("evaluation scope must contain UUIDs") from None


def _query_with_candidate_limit(query: SearchQuery, candidate_limit: int | None) -> SearchQuery:
    if candidate_limit is None or candidate_limit == query.candidate_limit:
        return query
    return SearchQuery.build(
        text=query.text,
        limit=query.limit,
        candidate_limit=candidate_limit,
        area=query.area,
        scope=query.scope,
        facets=query.facets,
        path_glob=query.path_glob,
        tags=query.tags,
        document_kinds=query.document_kinds,
        annotated_only=query.annotated_only,
    )


async def _postgres_fetch(
    pool: object,
    sql: str,
    params: Sequence[object],
    *,
    label: str,
) -> Sequence[object]:
    fetch = getattr(pool, "fetch", None)
    if not callable(fetch):
        raise RetrieverUnavailable(f"{label} evaluation store is unavailable")
    failure: BaseException | None = None
    try:
        return await fetch(sql, *params)
    except BaseException as error:  # noqa: BLE001 - database details must not cross the boundary.
        failure = error
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("evaluation store boundary lost its failure")
    if process_control := _sanitized_process_control(failure):
        _raise_session_failure(process_control)
    _raise_session_failure(RetrieverUnavailable(f"{label} evaluation store is unavailable"))


def _validated_evaluation_query_embedding(
    vectors: object,
    *,
    dimensions: int,
) -> tuple[float, ...]:
    if (
        isinstance(vectors, (str, bytes))
        or not isinstance(vectors, Sequence)
        or len(vectors) != 1
        or isinstance(vectors[0], (str, bytes))
        or not isinstance(vectors[0], Sequence)
        or len(vectors[0]) != dimensions
    ):
        raise RetrieverUnavailable("evaluation query embedding is unavailable")
    normalized: list[float] = []
    for coordinate in vectors[0]:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise RetrieverUnavailable("evaluation query embedding is unavailable")
        try:
            value = float(coordinate)
        except (OverflowError, TypeError, ValueError):
            raise RetrieverUnavailable("evaluation query embedding is unavailable") from None
        if not isfinite(value):
            raise RetrieverUnavailable("evaluation query embedding is unavailable")
        normalized.append(value)
    if not any(normalized):
        raise RetrieverUnavailable("evaluation query embedding is unavailable")
    return tuple(normalized)


def _metadata_is_strict_json(value: object, *, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or type(value) in (bool, str):
        return True
    if type(value) is int:
        return -_POSTGRES_INTEGER_MAX <= value <= _POSTGRES_INTEGER_MAX
    if type(value) is float:
        return isfinite(value)
    if isinstance(value, Mapping):
        if len(value) > 1_000:
            return False
        return all(
            type(key) is str
            and bool(key)
            and len(key) <= 1_024
            and "\x00" not in key
            and _metadata_is_strict_json(nested, depth=depth + 1)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return len(value) <= 10_000 and all(
            _metadata_is_strict_json(nested, depth=depth + 1) for nested in value
        )
    return False


def _metadata_fits_boundary(value: Mapping[str, object]) -> bool:
    if not _metadata_is_strict_json(value):
        return False
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError):
        return False
    return len(encoded) <= _MAX_BACKEND_METADATA_BYTES


def _evaluation_metadata_text(value: object) -> dict[str, object]:
    """Parse the exact default-asyncpg JSONB text shape at the adapter boundary."""

    if type(value) is not str:
        raise ValueError("evaluation metadata must be JSON text")
    try:
        if len(value.encode("utf-8")) > _MAX_BACKEND_METADATA_BYTES:
            raise ValueError("evaluation metadata exceeds the boundary")
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, UnicodeEncodeError, json.JSONDecodeError, ValueError):
        raise ValueError("evaluation metadata is invalid") from None
    if type(decoded) is not dict or not _metadata_fits_boundary(decoded):
        raise ValueError("evaluation metadata is invalid")
    return decoded


def _validated_evaluation_rows_impl(
    rows: object,
    *,
    profile: str,
) -> tuple[Mapping[str, object], ...]:
    """Reject malformed backend rows without coercion or backend detail leakage."""

    invalid = False
    normalized: list[Mapping[str, object]] = []
    candidate_count: int | None = None
    if (
        profile not in {"lexical", "vector"}
        or isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) > 10_000
    ):
        invalid = True
    else:
        for raw_row in rows:
            is_asyncpg_record = (
                type(raw_row).__name__ == "Record"
                and type(raw_row).__module__.startswith("asyncpg.")
            )
            if not isinstance(raw_row, Mapping) and not is_asyncpg_record:
                invalid = True
                break
            row = dict(raw_row)
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
            if not required <= row.keys():
                invalid = True
                break

            raw_count = row["candidate_count"]
            document_id = row["document_id"]
            document_version = row["document_version"]
            chunk_index = row["chunk_index"]
            content = row["content"]
            path = row["path"]
            filename = row["filename"]
            score = row["score"]
            page = row["page"]
            metadata = _evaluation_metadata_text(row["metadata"])
            tags = row["tags"]
            source_kind = row["source_kind"]
            title = row.get("title")
            header = row.get("header_breadcrumb")

            if (
                type(raw_count) is not int
                or not 0 <= raw_count <= _POSTGRES_INTEGER_MAX
                or (candidate_count is not None and raw_count != candidate_count)
                or not isinstance(document_id, UUID)
                or type(document_version) is not int
                or not 1 <= document_version <= _POSTGRES_INTEGER_MAX
                or type(chunk_index) is not int
                or not 0 <= chunk_index < 10_000
                or type(content) is not str
                or len(content) > MAX_CORPUS_TEXT_CHARS
                or type(path) is not str
                or not path.startswith("/")
                or "\x00" in path
                or len(path) > _MAX_BACKEND_PATH_CHARS
                or type(filename) is not str
                or not filename
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
                or "\x00" in filename
                or len(filename) > _MAX_BACKEND_PATH_CHARS
                or type(score) is not float
                or not isfinite(score)
                or (profile == "lexical" and score < 0.0)
                or (profile == "vector" and not -1.0 <= score <= 1.0)
                or (
                    page is not None
                    and (type(page) is not int or not 1 <= page <= _POSTGRES_INTEGER_MAX)
                )
                or type(tags) is not list
                or len(tags) > 100
                or any(
                    type(tag) is not str
                    or not tag.strip()
                    or len(tag) > 128
                    or "\x00" in tag
                    for tag in tags
                )
                or type(source_kind) is not str
                or source_kind not in {kind.value for kind in DocumentKind}
                or (title is not None and (type(title) is not str or len(title) > 4_096))
                or (header is not None and (type(header) is not str or len(header) > 4_096))
            ):
                invalid = True
                break

            candidate_count = raw_count
            clean = {
                **row,
                "metadata": dict(metadata),
                "tags": tuple(tags),
            }
            normalized.append(clean)

        if not invalid and candidate_count is not None and candidate_count < len(normalized):
            invalid = True

    if invalid:
        raise ValueError("evaluation row is invalid")
    return tuple(normalized)


def _validated_evaluation_rows(
    rows: object,
    *,
    profile: str,
) -> tuple[Mapping[str, object], ...]:
    failure: BaseException | None = None
    try:
        return _validated_evaluation_rows_impl(rows, profile=profile)
    except BaseException as error:  # noqa: BLE001 - malformed adapter values stay private.
        failure = error
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("evaluation row boundary lost its failure")
    if signal := sanitized_boundary_signal_or_unknown(failure):
        raise signal from None
    raise RetrieverUnavailable("evaluation row is invalid") from None


def _postgres_evaluation_hit(row: Mapping[str, object]) -> SearchHit:
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
        metadata=row["metadata"],
    )


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(value.casefold()))


def _scope_content(chunk: _CorpusChunk, scope: SearchScope) -> str:
    if scope is SearchScope.SOURCE:
        return chunk.source_content
    if scope is SearchScope.ANNOTATIONS:
        return chunk.annotations_text
    return f"{chunk.source_content}\n{chunk.annotations_text}"


def _matches_filters(chunk: _CorpusChunk, query: SearchQuery) -> bool:
    if query.area is SearchArea.WIKI and chunk.document_kind is not DocumentKind.WIKI:
        return False
    if query.area is SearchArea.SOURCES and chunk.document_kind is DocumentKind.WIKI:
        return False
    if query.document_kinds and chunk.document_kind not in query.document_kinds:
        return False
    if query.path_glob is not None and not _logical_glob_matches(query.path_glob, chunk.path):
        return False
    if query.tags and not set(query.tags).issubset(chunk.tags):
        return False
    if query.annotated_only and not chunk.annotated:
        return False
    if query.scope is SearchScope.SOURCE and not chunk.source_content.strip():
        return False
    if query.scope is SearchScope.ANNOTATIONS and not chunk.annotations_text.strip():
        return False
    return all(key in chunk.facets and chunk.facets[key] == value for key, value in query.facets.items())


def _logical_glob_matches(path_glob: str, path: str) -> bool:
    if path_glob == "/":
        return path.startswith("/")
    directory = path_glob.endswith("/") or (
        "*" not in path_glob and "." not in path_glob.rsplit("/", 1)[-1]
    )
    pattern = path_glob.rstrip("/") + "/*" if directory else path_glob
    expression: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        if pattern[index] == "*":
            while index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 1
            expression.append(".*")
        else:
            expression.append(re.escape(pattern[index]))
        index += 1
    expression.append("$")
    return re.fullmatch("".join(expression), path) is not None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate corpus JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_object(
    value: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("corpus value must be an object")
    if set(value) - allowed or required - set(value):
        raise ValueError("corpus object fields are invalid")
    return value


def _nonblank_string(value: object) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()) or len(normalized) > 4096:
        raise ValueError("corpus string is invalid")
    return normalized


def _text(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_CORPUS_TEXT_CHARS:
        raise ValueError("corpus text is invalid")
    return value


def _tags(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("corpus tags are invalid")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not (tag := item.strip().casefold()) or len(tag) > 128:
            raise ValueError("corpus tags are invalid")
        normalized.append(tag)
    return frozenset(normalized)


def _facets(value: object) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping) or len(value) > 100:
        raise ValueError("corpus facets are invalid")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("corpus facets are invalid")
        if nested is not None and not isinstance(nested, (str, int, float, bool)):
            raise ValueError("corpus facets are invalid")
        if isinstance(nested, float) and not isfinite(nested):
            raise ValueError("corpus facets are invalid")
        result[key] = nested
    return MappingProxyType(result)


def _parse_corpus_line(raw_line: bytes) -> tuple[_CorpusChunk, ...]:
    if len(raw_line) > MAX_CORPUS_LINE_BYTES or not raw_line.strip():
        raise ValueError("corpus line is invalid")
    try:
        raw = json.loads(
            raw_line.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("corpus line is invalid") from exc
    document = _strict_object(raw, allowed=_DOCUMENT_FIELDS, required=_DOCUMENT_REQUIRED_FIELDS)
    if type(document["schema_version"]) is not int or document["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise ValueError("corpus schema is unsupported")
    document_id = _nonblank_string(document["document_id"])
    try:
        document_kind = DocumentKind(document["document_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("corpus document kind is invalid") from exc
    path = _nonblank_string(document["path"])
    if not path.startswith("/") or "\x00" in path or any(part == ".." for part in path.split("/")):
        raise ValueError("corpus path is invalid")
    title = _nonblank_string(document["title"])
    tags = _tags(document.get("tags"))
    facets = _facets(document.get("facets"))
    raw_chunks = document["chunks"]
    if not isinstance(raw_chunks, list) or not raw_chunks or len(raw_chunks) > MAX_CORPUS_CHUNKS:
        raise ValueError("corpus chunks are invalid")

    chunks: list[_CorpusChunk] = []
    seen_indices: set[int] = set()
    for raw_chunk in raw_chunks:
        chunk = _strict_object(raw_chunk, allowed=_CHUNK_FIELDS, required=_CHUNK_REQUIRED_FIELDS)
        chunk_index = chunk["chunk_index"]
        if type(chunk_index) is not int or chunk_index < 0 or chunk_index in seen_indices:
            raise ValueError("corpus chunk index is invalid")
        seen_indices.add(chunk_index)
        content = _text(chunk["content"])
        source_content = _text(chunk.get("source_content", content))
        annotations_text = _text(chunk.get("annotations_text", ""))
        if content != "\n".join(part for part in (source_content, annotations_text) if part):
            raise ValueError("corpus chunk content does not match its retrieval fields")
        chunks.append(
            _CorpusChunk(
                document_id=document_id,
                document_kind=document_kind,
                path=path,
                title=title,
                tags=tags,
                facets=facets,
                chunk_index=chunk_index,
                content=content,
                source_content=source_content,
                annotations_text=annotations_text,
            )
        )
    return tuple(chunks)


def _open_corpus(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("corpus cannot be read") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_CORPUS_BYTES:
            raise ValueError("corpus file is invalid")
        return os.fdopen(descriptor, "rb")
    except (OSError, ValueError):
        with suppress(OSError):
            os.close(descriptor)
        raise


def _register_corpus_document(
    parsed: Sequence[_CorpusChunk],
    *,
    chunks: list[_CorpusChunk],
    identities: set[tuple[str, int]],
    document_ids: set[str],
) -> None:
    document_id = parsed[0].document_id
    if document_id in document_ids:
        raise ValueError("corpus document id is duplicated")
    document_ids.add(document_id)
    if len(document_ids) > MAX_CORPUS_DOCUMENTS:
        raise ValueError("corpus has too many documents")
    for chunk in parsed:
        identity = (chunk.document_id, chunk.chunk_index)
        if identity in identities:
            raise ValueError("corpus chunk identity is duplicated")
        identities.add(identity)
        chunks.append(chunk)
        if len(chunks) > MAX_CORPUS_CHUNKS:
            raise ValueError("corpus has too many chunks")


def _load_corpus(path: Path) -> tuple[_CorpusChunk, ...]:
    chunks: list[_CorpusChunk] = []
    identities: set[tuple[str, int]] = set()
    document_ids: set[str] = set()
    total_bytes = 0
    try:
        with _open_corpus(path) as corpus:
            while raw_line := corpus.readline(MAX_CORPUS_LINE_BYTES + 1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_CORPUS_BYTES:
                    raise ValueError("corpus is too large")
                _register_corpus_document(
                    _parse_corpus_line(raw_line),
                    chunks=chunks,
                    identities=identities,
                    document_ids=document_ids,
                )
    except OSError as exc:
        raise ValueError("corpus cannot be read") from exc
    if not chunks:
        raise ValueError("corpus must not be empty")
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class _HostedEvaluationConfiguration:
    database_url: str
    user_id: UUID
    knowledge_base_id: UUID
    embedding_profile: EmbeddingProfile
    embedding_base_url: str
    embedding_api_key: str
    embedding_batch_size: int
    embedding_timeout_seconds: float
    lexical_candidate_limit: int
    vector_candidate_limit: int
    rrf_k: int


def _required_hosted_environment(name: str, *, maximum: int) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise HybridConfigurationUnavailable
    return value.strip()


def _hosted_integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        raise HybridConfigurationUnavailable
    value = int(raw)
    if not minimum <= value <= maximum:
        raise HybridConfigurationUnavailable
    return value


def _hosted_timeout() -> float:
    raw = os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "30")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise HybridConfigurationUnavailable from None
    if not isfinite(value) or not 0 < value <= 300:
        raise HybridConfigurationUnavailable
    return value


def _load_hosted_evaluation_configuration(
    cases: Sequence[object],
) -> _HostedEvaluationConfiguration:
    if os.environ.get("HYBRID_SEARCH_ENABLED", "").strip().lower() not in {"1", "true"}:
        raise HybridConfigurationUnavailable
    try:
        user_id = _evaluation_uuid(
            _required_hosted_environment("RETRIEVAL_EVAL_USER_ID", maximum=64)
        )
        knowledge_base_id = _evaluation_uuid(
            _required_hosted_environment(
                "RETRIEVAL_EVAL_KNOWLEDGE_BASE_ID",
                maximum=64,
            )
        )
        profile = EmbeddingProfile(
            provider=os.environ.get("EMBEDDING_PROVIDER", "openai_compatible"),
            model=_required_hosted_environment("EMBEDDING_MODEL", maximum=200),
            dimensions=_hosted_integer(
                "EMBEDDING_DIMENSIONS",
                0,
                minimum=1,
                maximum=4_096,
            ),
        )
    except (TypeError, ValueError):
        raise HybridConfigurationUnavailable from None
    lexical_limit = _hosted_integer(
        "HYBRID_LEXICAL_CANDIDATES",
        50,
        minimum=1,
        maximum=500,
    )
    vector_limit = _hosted_integer(
        "HYBRID_VECTOR_CANDIDATES",
        50,
        minimum=1,
        maximum=500,
    )
    request_limit = max((case.query.limit for case in cases), default=1)
    if lexical_limit < request_limit or vector_limit < request_limit:
        raise HybridConfigurationUnavailable
    api_key = os.environ.get("EMBEDDING_API_KEY", "")
    if not isinstance(api_key, str) or len(api_key) > _MAX_HOSTED_SECRET_CHARS:
        raise HybridConfigurationUnavailable
    return _HostedEvaluationConfiguration(
        database_url=_required_hosted_environment(
            "DATABASE_URL",
            maximum=_MAX_HOSTED_DSN_CHARS,
        ),
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        embedding_profile=profile,
        embedding_base_url=_required_hosted_environment(
            "EMBEDDING_BASE_URL",
            maximum=4_096,
        ),
        embedding_api_key=api_key,
        embedding_batch_size=_hosted_integer(
            "EMBEDDING_BATCH_SIZE",
            32,
            minimum=1,
            maximum=512,
        ),
        embedding_timeout_seconds=_hosted_timeout(),
        lexical_candidate_limit=lexical_limit,
        vector_candidate_limit=vector_limit,
        rrf_k=_hosted_integer("HYBRID_RRF_K", 60, minimum=1, maximum=10_000),
    )


async def _create_hosted_pool(database_url: str) -> object:
    import asyncpg

    return await asyncpg.create_pool(database_url, min_size=1, max_size=4)


def _new_hosted_embedding_client(
    configuration: _HostedEvaluationConfiguration,
) -> object:
    from services.embeddings import OpenAIEmbeddingClient

    return OpenAIEmbeddingClient(
        profile=configuration.embedding_profile,
        base_url=configuration.embedding_base_url,
        api_key=configuration.embedding_api_key,
        batch_size=configuration.embedding_batch_size,
        timeout_seconds=configuration.embedding_timeout_seconds,
    )


async def _validate_hosted_embedding_client(
    configuration: _HostedEvaluationConfiguration,
) -> None:
    client = None
    primary_failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    try:
        client = _new_hosted_embedding_client(configuration)
        if getattr(client, "profile", None) != configuration.embedding_profile:
            raise HybridConfigurationUnavailable
    except BaseException as error:  # noqa: BLE001 - configuration must remain private.
        primary_failure = error
    if client is not None:
        try:
            close = getattr(client, "aclose", None)
            if not callable(close):
                raise HybridConfigurationUnavailable
            pending = close()
            if not inspect.isawaitable(pending):
                raise HybridConfigurationUnavailable
            await pending
        except BaseException as error:  # noqa: BLE001 - cleanup is part of validation.
            cleanup_failure = error
    if primary_failure is None and cleanup_failure is None:
        return
    failures = tuple(
        failure
        for failure in (primary_failure, cleanup_failure)
        if failure is not None
    )
    if signal := sanitized_boundary_signal_or_unknown(*failures):
        raise signal from None
    raise HybridConfigurationUnavailable from None


def _default_retriever_factory(profile: str, dataset_path: Path) -> object:
    if profile == "lexical":
        try:
            chunks = _load_corpus(dataset_path.with_name("corpus.jsonl"))
        except (OSError, ValueError):
            raise EvaluationDatasetError from None
        return _SyntheticLexicalRetriever(chunks)
    if profile == "hybrid":
        raise HybridConfigurationUnavailable
    raise RetrievalContractError


def _parser() -> _SafeArgumentParser:
    parser = _SafeArgumentParser(prog="retrieval_eval")
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=("lexical", "hybrid"))
    selection.add_argument("--compare", action="store_true")
    parser.add_argument("--hosted", action="store_true")
    parser.add_argument("--require-promotion-gate", action="store_true")
    parser.add_argument("--output-json")
    return parser


async def _retrieve_case(profile: str, retriever: object, query: SearchQuery) -> SearchResult:
    retrieve = _backend_call(lambda: getattr(retriever, "retrieve", None))
    if not callable(retrieve):
        raise RetrievalContractError
    pending = _backend_call(lambda: retrieve(query))
    if not inspect.isawaitable(pending):
        raise RetrievalContractError
    result = await _await_backend(pending)
    if type(result) is not SearchResult or result.profile != profile:
        raise RetrievalContractError
    if any(type(hit) is not SearchHit for hit in result.hits):
        raise RetrievalContractError
    if result.candidate_count < len(result.hits) or len(result.hits) > query.candidate_limit:
        raise RetrievalContractError
    identities = {(hit.document_id, hit.chunk_index) for hit in result.hits}
    if len(identities) != len(result.hits):
        raise RetrievalContractError
    return result


async def _evaluate_profile(profile: str, retriever: object, cases: Sequence[object]) -> EvaluationReport:
    runs: list[EvaluationRun] = []
    for case in cases:
        result = await _retrieve_case(profile, retriever, case.query)
        runs.append(
            EvaluationRun(
                case_id=case.case_id,
                ranking=tuple(RankedResult(hit.document_id, hit.chunk_index) for hit in result.hits),
                latency_ms=result.latency_ms,
            )
        )
    return evaluate_rankings(cases, runs)


def _build_retriever(factory: RetrieverFactory, profile: str, dataset_path: Path) -> object:
    factory_call = _backend_call(
        lambda: getattr(factory, "__call__", None)  # noqa: B004 - attribute access is a guarded backend boundary.
    )
    if not callable(factory_call):
        raise RetrievalContractError
    retriever = _backend_call(
        lambda: factory_call(profile, dataset_path),
        passthrough=_SAFE_BACKEND_EXCEPTIONS,
    )
    retrieve = _backend_call(lambda: getattr(retriever, "retrieve", None))
    if not callable(retrieve):
        raise RetrievalContractError
    return retriever


@asynccontextmanager
async def _evaluation_session(factory: RetrieverFactory) -> AsyncIterator[None]:
    if not isinstance(factory, _PostgresEvaluationFactory):
        yield
        return
    async with factory.evaluation_session():
        yield


def _has_running_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_async(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    if not _has_running_event_loop():
        return _backend_call(
            lambda: asyncio.run(factory()),
            passthrough=_SAFE_BACKEND_EXCEPTIONS,
        )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-eval") as executor:
        return _backend_call(
            lambda: executor.submit(lambda: asyncio.run(factory())).result(),
            passthrough=_SAFE_BACKEND_EXCEPTIONS,
        )


def _stable_number(value: float) -> float:
    rounded = round(value, 12)
    return 0.0 if rounded == 0 else rounded


def _metrics_payload(report: EvaluationReport) -> dict[str, object]:
    return {
        "filtered_result_count": report.filtered_result_count,
        "latency_p50_ms": _stable_number(report.latency_p50_ms),
        "latency_p95_ms": _stable_number(report.latency_p95_ms),
        "mrr": _stable_number(report.mrr),
        "ndcg_at_10": _stable_number(report.ndcg_at_10),
        "recall_at_10": _stable_number(report.recall_at_10),
        "recall_at_20": _stable_number(report.recall_at_20),
        "recall_at_5": _stable_number(report.recall_at_5),
    }


def _profile_payload(profile: str, report: EvaluationReport) -> dict[str, object]:
    return {
        "case_count": report.case_count,
        "metrics": _metrics_payload(report),
        "profile": profile,
    }


def _base_payload(cases: Sequence[object], *, profile: str) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "dataset_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_dataset_digest": evaluation_dataset_digest(cases),
        "profile": profile,
        "schema_version": REPORT_SCHEMA_VERSION,
    }


def _single_report(cases: Sequence[object], profile: str, report: EvaluationReport) -> dict[str, object]:
    return {**_base_payload(cases, profile=profile), "metrics": _metrics_payload(report)}


def _compare_report(
    cases: Sequence[object],
    lexical: EvaluationReport,
    hybrid: EvaluationReport,
) -> tuple[dict[str, object], bool]:
    decision = promotion_decision(lexical, hybrid)
    payload = {
        **_base_payload(cases, profile="compare"),
        "profiles": {
            "hybrid": _profile_payload("hybrid", hybrid),
            "lexical": _profile_payload("lexical", lexical),
        },
        "promotion": {
            "eligible": decision.eligible,
            "latency_ratio": None if decision.latency_ratio is None else _stable_number(decision.latency_ratio),
            "reason": decision.reason,
            "recall_ratio": None if decision.recall_ratio is None else _stable_number(decision.recall_ratio),
        },
    }
    return payload, decision.eligible


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _raw_output_components(path: str | os.PathLike[str]) -> tuple[bool, tuple[str, ...], str]:
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path or raw_path.endswith("/"):
        raise OutputWriteError
    components = tuple(component for component in raw_path.split("/") if component)
    if not components or any(component in {".", ".."} for component in components):
        raise OutputWriteError
    return raw_path.startswith("/"), components[:-1], components[-1]


def _open_output_parent(path: str | os.PathLike[str]) -> tuple[int, int, str]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    cloexec = getattr(os, "O_CLOEXEC", None)
    if nofollow is None or directory is None or cloexec is None:
        raise OutputWriteError
    absolute, parent_components, basename = _raw_output_components(path)
    descriptor = -1
    try:
        flags = os.O_RDONLY | directory | nofollow | cloexec
        descriptor = os.open("/" if absolute else ".", flags)
        for component in parent_components:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            with suppress(OSError):
                os.close(descriptor)
            descriptor = next_descriptor
        result = (descriptor, nofollow, basename)
        descriptor = -1
        return result
    except OSError as exc:
        raise OutputWriteError from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


def _create_output_temporary(parent_descriptor: int, basename: str, nofollow: int) -> tuple[int, str]:
    create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    for _attempt in range(10):
        temporary_name = f".{basename}.{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                create_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            return descriptor, temporary_name
        except FileExistsError:
            continue
    raise OutputWriteError


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OutputWriteError
        remaining = remaining[written:]


def _write_output(path: str | os.PathLike[str], content: bytes) -> None:
    parent_descriptor = -1
    temporary_descriptor = -1
    temporary_name: str | None = None
    try:
        parent_descriptor, nofollow, basename = _open_output_parent(path)
        try:
            current = os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise OutputWriteError
        temporary_descriptor, temporary_name = _create_output_temporary(
            parent_descriptor,
            basename,
            nofollow,
        )
        _write_all(temporary_descriptor, content)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.replace(
            temporary_name,
            basename,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    except OutputWriteError:
        raise
    except OSError as exc:
        raise OutputWriteError from exc
    finally:
        if temporary_descriptor >= 0:
            with suppress(OSError):
                os.close(temporary_descriptor)
        if temporary_name is not None and parent_descriptor >= 0:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        if parent_descriptor >= 0:
            with suppress(OSError):
                os.close(parent_descriptor)


def _discard_failed_stream_buffer(buffer: Any) -> None:
    with suppress(AttributeError, OSError, TypeError, ValueError):
        buffer.seek(0)
        buffer.truncate(0)


def _silence_failed_stream(buffer: Any) -> None:
    try:
        descriptor = buffer.fileno()
    except (AttributeError, OSError, TypeError, ValueError):
        _discard_failed_stream_buffer(buffer)
        return

    null_descriptor = -1
    try:
        null_descriptor = os.open(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        if null_descriptor == descriptor:
            null_descriptor = -1
        else:
            os.dup2(null_descriptor, descriptor, inheritable=False)
    except (OSError, TypeError, ValueError):
        _discard_failed_stream_buffer(buffer)
        return
    finally:
        if null_descriptor >= 0:
            with suppress(OSError):
                os.close(null_descriptor)

    with suppress(AttributeError, OSError, TypeError, ValueError):
        buffer.flush()


def _emit_stream(stream: Any, content: bytes) -> bool:
    buffer = stream
    remaining = memoryview(content)
    try:
        buffer = getattr(stream, "buffer", stream)
        while remaining:
            written = buffer.write(remaining)
            if type(written) is not int or written <= 0 or written > len(remaining):
                raise OSError
            remaining = remaining[written:]
        buffer.flush()
    except (AttributeError, OSError, TypeError, ValueError):
        _silence_failed_stream(buffer)
        return False
    return True


def _emit_error(category: str, code: str) -> None:
    _emit_stream(sys.stderr, _json_bytes({"error": {"category": category, "code": code}}))


def _parse_cli_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace | None, int]:
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        if (args.require_promotion_gate or args.hosted) and not args.compare:
            raise _ArgumentError
    except _HelpRequested:
        emitted = _emit_stream(sys.stdout, parser.format_help().encode("utf-8"))
        return None, 0 if emitted else 4
    except (_ArgumentError, SystemExit):
        _emit_error("arguments", "invalid_arguments")
        return None, 2
    return args, 0


def _load_cli_cases(dataset_path: Path) -> tuple[Sequence[object] | None, int]:
    try:
        cases = load_cases(dataset_path)
    except (OSError, TypeError, ValueError):
        _emit_error("dataset", "dataset_invalid")
        return None, 2
    return cases, 0


async def _evaluate_request_async(
    args: argparse.Namespace,
    cases: Sequence[object],
    factory: RetrieverFactory,
    dataset_path: Path,
) -> tuple[dict[str, object], int]:
    if not args.compare:
        profile = args.profile
        async with _evaluation_session(factory):
            retriever = _build_retriever(factory, profile, dataset_path)
            report = await _evaluate_profile(profile, retriever, cases)
        return _single_report(cases, profile, report), 0

    async with _evaluation_session(factory):
        lexical_retriever = _build_retriever(factory, "lexical", dataset_path)
        hybrid_retriever = _build_retriever(factory, "hybrid", dataset_path)
        lexical_report = await _evaluate_profile("lexical", lexical_retriever, cases)
        hybrid_report = await _evaluate_profile("hybrid", hybrid_retriever, cases)
    payload, eligible = _compare_report(cases, lexical_report, hybrid_report)
    exit_code = 0 if eligible or not args.require_promotion_gate else 3
    return payload, exit_code


def _evaluate_request(
    args: argparse.Namespace,
    cases: Sequence[object],
    factory: RetrieverFactory,
    dataset_path: Path,
) -> tuple[dict[str, object], int]:
    return _run_async(
        lambda: _evaluate_request_async(args, cases, factory, dataset_path)
    )


def _raise_hosted_runtime_failure(
    primary: BaseException | None,
    cleanup: BaseException | None,
) -> Never:
    failures = tuple(error for error in (primary, cleanup) if error is not None)
    if signal := sanitized_boundary_signal_or_unknown(*failures):
        raise signal from None
    if primary is not None and isinstance(primary, _SAFE_BACKEND_EXCEPTIONS):
        raise type(primary)() from None
    raise RetrievalExecutionError from None


async def _evaluate_hosted_request_async(
    args: argparse.Namespace,
    cases: Sequence[object],
    dataset_path: Path,
) -> tuple[dict[str, object], int]:
    configuration = _load_hosted_evaluation_configuration(cases)
    await _validate_hosted_embedding_client(configuration)
    pool = None
    result: tuple[dict[str, object], int] | None = None
    primary_failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    try:
        pool = await _create_hosted_pool(configuration.database_url)
        factory = postgres_evaluation_retriever_factory(
            pool,
            user_id=configuration.user_id,
            knowledge_base_id=configuration.knowledge_base_id,
            embedding_profile=configuration.embedding_profile,
            embedding_client_factory=lambda: _new_hosted_embedding_client(configuration),
            lexical_candidate_limit=configuration.lexical_candidate_limit,
            vector_candidate_limit=configuration.vector_candidate_limit,
            rrf_k=configuration.rrf_k,
        )
        result = await _evaluate_request_async(args, cases, factory, dataset_path)
    except BaseException as error:  # noqa: BLE001 - hosted boundary owns cleanup and privacy.
        primary_failure = error
    if pool is not None:
        try:
            close = getattr(pool, "close", None)
            if not callable(close):
                raise RetrievalExecutionError
            pending = close()
            if not inspect.isawaitable(pending):
                raise RetrievalExecutionError
            await pending
        except BaseException as error:  # noqa: BLE001 - cleanup failures are sanitized below.
            cleanup_failure = error
    if primary_failure is not None or cleanup_failure is not None:
        _raise_hosted_runtime_failure(primary_failure, cleanup_failure)
    if result is None:  # pragma: no cover - success assigns before cleanup.
        raise RetrievalExecutionError
    return result


def _safe_evaluate_request(
    args: argparse.Namespace,
    cases: Sequence[object],
    factory: RetrieverFactory,
    dataset_path: Path,
) -> tuple[dict[str, object] | None, int]:
    try:
        return _evaluate_request(args, cases, factory, dataset_path)
    except EvaluationDatasetError:
        _emit_error("dataset", "dataset_invalid")
        return None, 2
    except HybridConfigurationUnavailable:
        _emit_error("configuration", "hybrid_unavailable")
        return None, 2
    except RetrievalContractError:
        _emit_error("retrieval", "retrieval_contract_invalid")
        return None, 2
    except RetrievalExecutionError:
        _emit_error("retrieval", "retrieval_failed")
        return None, 2
    except (TypeError, ValueError):
        _emit_error("retrieval", "retrieval_contract_invalid")
        return None, 2
    except Exception:  # noqa: BLE001 - final privacy boundary intentionally discards all exception details.
        _emit_error("retrieval", "retrieval_failed")
        return None, 2


def _safe_evaluate_hosted_request(
    args: argparse.Namespace,
    cases: Sequence[object],
    dataset_path: Path,
) -> tuple[dict[str, object] | None, int]:
    try:
        return _run_async(
            lambda: _evaluate_hosted_request_async(args, cases, dataset_path)
        )
    except HybridConfigurationUnavailable:
        _emit_error("configuration", "hybrid_unavailable")
        return None, 2
    except RetrievalContractError:
        _emit_error("retrieval", "retrieval_contract_invalid")
        return None, 2
    except (RetrievalExecutionError, EvaluationDatasetError, TypeError, ValueError):
        _emit_error("retrieval", "retrieval_failed")
        return None, 2
    except Exception:  # noqa: BLE001 - final privacy boundary discards backend details.
        _emit_error("retrieval", "retrieval_failed")
        return None, 2


def _emit_report(payload: Mapping[str, object], output_json: str | None, exit_code: int) -> int:
    encoded = _json_bytes(payload)
    if output_json is not None:
        try:
            _write_output(output_json, encoded)
        except OutputWriteError:
            _emit_error("output", "output_write_failed")
            return 4
    if not _emit_stream(sys.stdout, encoded):
        return 4
    return exit_code


def main(argv: Sequence[str] | None = None, retriever_factory: RetrieverFactory | None = None) -> int:
    """Run retrieval evaluation without exposing query or backend content."""

    args, early_exit = _parse_cli_args(argv)
    if args is None:
        return early_exit
    dataset_path = Path(args.dataset)
    cases, early_exit = _load_cli_cases(dataset_path)
    if cases is None:
        return early_exit
    if args.hosted:
        if retriever_factory is not None:
            _emit_error("arguments", "invalid_arguments")
            return 2
        payload, exit_code = _safe_evaluate_hosted_request(args, cases, dataset_path)
    else:
        factory = _default_retriever_factory if retriever_factory is None else retriever_factory
        payload, exit_code = _safe_evaluate_request(args, cases, factory, dataset_path)
    if payload is None:
        return exit_code
    return _emit_report(payload, args.output_json, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
