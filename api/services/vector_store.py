"""Tenant-scoped, document-version-fenced pgvector persistence."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Never
from uuid import UUID

import llmwiki_core.postgres_retrieval as postgres_retrieval
from llmwiki_core.documents import DocumentKind
from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.search import (
    RetrieverUnavailable,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchScope,
)
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

_MAX_EMBEDDINGS_PER_DOCUMENT = 10_000
_POSTGRES_INTEGER_MAX = 2_147_483_647
_MAX_SEARCH_TEXT_CHARS = 1_000_000
_MAX_SEARCH_PATH_CHARS = 4_096
_MAX_SEARCH_METADATA_BYTES = 64 * 1_024

class _VectorWriteRejected(ValueError):
    """A stable caller-visible version or chunk-set rejection."""


class PostgresVectorStore:
    """Store complete embedding sets and retrieve exact cosine candidates."""

    def __init__(self, pool, *, profile: EmbeddingProfile) -> None:
        if not isinstance(profile, EmbeddingProfile):
            raise TypeError("profile must be an EmbeddingProfile")
        if (
            not isinstance(profile.provider, str)
            or not profile.provider
            or not isinstance(profile.model, str)
            or not profile.model
            or len(profile.provider) > 100
            or len(profile.model) > 200
        ):
            raise ValueError("embedding profile model identity is too long")
        if type(profile.dimensions) is not int or not 1 <= profile.dimensions <= 4096:
            raise ValueError("embedding dimensions must be between 1 and 4096")
        self._pool = pool
        self._profile = profile

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    async def replace_document_embeddings(
        self,
        *,
        user_id: str | UUID,
        knowledge_base_id: str | UUID,
        document_id: str | UUID,
        document_version: int,
        embeddings: Sequence[tuple[int, Sequence[Real]]],
    ) -> int:
        """Atomically replace one current document/profile's complete vector set."""
        user_uuid = _uuid(user_id, label="user_id")
        kb_uuid = _uuid(knowledge_base_id, label="knowledge_base_id")
        document_uuid = _uuid(document_id, label="document_id")
        version = _bounded_int(
            document_version,
            label="document_version",
            minimum=0,
            maximum=_POSTGRES_INTEGER_MAX,
        )
        _document_vectors(embeddings, dimensions=self._profile.dimensions)
        failure = None
        count = 0
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                count = await self.replace_document_embeddings_in_transaction(
                    conn,
                    user_id=user_uuid,
                    knowledge_base_id=kb_uuid,
                    document_id=document_uuid,
                    document_version=version,
                    embeddings=embeddings,
                )
        except (_VectorWriteRejected, RetrieverUnavailable):
            raise
        except BaseException as caught:  # noqa: BLE001 - sanitize the database adapter boundary.
            failure = caught
        if failure is not None:
            _raise_sanitized_boundary(failure, "vector store is unavailable")
        return count

    async def replace_document_embeddings_in_transaction(
        self,
        conn,
        *,
        user_id: str | UUID,
        knowledge_base_id: str | UUID,
        document_id: str | UUID,
        document_version: int,
        embeddings: Sequence[tuple[int, Sequence[Real]]],
    ) -> int:
        """Replace a complete set inside the caller's fenced transaction."""
        is_in_transaction = getattr(conn, "is_in_transaction", None)
        if callable(is_in_transaction) and not is_in_transaction():
            raise RuntimeError("vector replacement requires an explicit transaction")
        user_uuid = _uuid(user_id, label="user_id")
        kb_uuid = _uuid(knowledge_base_id, label="knowledge_base_id")
        document_uuid = _uuid(document_id, label="document_id")
        version = _bounded_int(
            document_version,
            label="document_version",
            minimum=0,
            maximum=_POSTGRES_INTEGER_MAX,
        )
        vectors = _document_vectors(embeddings, dimensions=self._profile.dimensions)

        current_version = await conn.fetchval(
            "SELECT version FROM documents "
            "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 "
            "AND NOT archived FOR UPDATE",
            document_uuid,
            user_uuid,
            kb_uuid,
        )
        if current_version is None:
            raise _VectorWriteRejected("document is not available in the requested tenant scope")
        if current_version != version:
            raise _VectorWriteRejected("document_version must equal the current document version")

        chunk_rows = await conn.fetch(
            "SELECT chunk_index FROM document_chunks "
            "WHERE document_id=$1 AND document_version=$2 "
            "AND user_id=$3 AND knowledge_base_id=$4 "
            "ORDER BY chunk_index FOR SHARE",
            document_uuid,
            version,
            user_uuid,
            kb_uuid,
        )
        expected = tuple(row["chunk_index"] for row in chunk_rows)
        submitted = tuple(index for index, _vector in vectors)
        if submitted != expected:
            raise _VectorWriteRejected("embeddings must match the complete current chunk set")

        if vectors:
            await conn.executemany(
                "INSERT INTO chunk_embeddings "
                "(user_id, knowledge_base_id, document_id, document_version, "
                "chunk_index, provider, model, dimensions, embedding) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::vector) "
                "ON CONFLICT (document_id, document_version, chunk_index, "
                "provider, model, dimensions) DO UPDATE SET "
                "embedding=EXCLUDED.embedding, updated_at=now()",
                [
                    (
                        user_uuid,
                        kb_uuid,
                        document_uuid,
                        version,
                        chunk_index,
                        self._profile.provider,
                        self._profile.model,
                        self._profile.dimensions,
                        vector,
                    )
                    for chunk_index, vector in vectors
                ],
            )

        complete_count = await conn.fetchval(
            "SELECT count(*) FROM chunk_embeddings "
            "WHERE user_id=$1 AND knowledge_base_id=$2 AND document_id=$3 "
            "AND document_version=$4 AND provider=$5 AND model=$6 "
            "AND dimensions=$7",
            user_uuid,
            kb_uuid,
            document_uuid,
            version,
            self._profile.provider,
            self._profile.model,
            self._profile.dimensions,
        )
        if complete_count != len(expected):
            raise RetrieverUnavailable("vector store is unavailable")

        await conn.execute(
            "DELETE FROM chunk_embeddings "
            "WHERE user_id=$1 AND knowledge_base_id=$2 AND document_id=$3 "
            "AND provider=$4 AND model=$5 AND dimensions=$6 "
            "AND document_version <> $7",
            user_uuid,
            kb_uuid,
            document_uuid,
            self._profile.provider,
            self._profile.model,
            self._profile.dimensions,
            version,
        )
        return len(vectors)

    async def search(
        self,
        *,
        user_id: str | UUID,
        knowledge_base_id: str | UUID,
        query: SearchQuery,
        embedding: Sequence[Real],
    ) -> SearchResult:
        """Return bounded current-version exact cosine candidates."""
        user_uuid = _uuid(user_id, label="user_id")
        kb_uuid = _uuid(knowledge_base_id, label="knowledge_base_id")
        if not isinstance(query, SearchQuery):
            raise TypeError("query must be a SearchQuery")
        _candidate_limit(query)
        if query.scope is not SearchScope.ALL:
            raise RetrieverUnavailable("vector search does not support scoped content")
        vector = _vector(embedding, dimensions=self._profile.dimensions)
        started_at = perf_counter()

        params: list[object] = [
            user_uuid,
            kb_uuid,
            self._profile.provider,
            self._profile.model,
            self._profile.dimensions,
            vector,
        ]

        def bind(value: object) -> str:
            params.append(value)
            return f"${len(params)}"

        where = [
            "ce.user_id=$1",
            "ce.knowledge_base_id=$2",
            "ce.provider=$3",
            "ce.model=$4",
            "ce.dimensions=$5",
            "d.user_id=$1",
            "d.knowledge_base_id=$2",
            "dc.user_id=$1",
            "dc.knowledge_base_id=$2",
            "ce.document_version=d.version",
            "dc.document_version=d.version",
            "d.status != 'failed'",
            "NOT d.archived",
        ]
        where.extend(
            postgres_retrieval.postgres_document_filter_conditions(
                query,
                params,
                doc_alias="d",
                chunk_alias="dc",
            )
        )
        limit_parameter = bind(query.candidate_limit)

        sql = (
            "WITH filtered AS ("
            "SELECT ce.document_id, ce.document_version, ce.chunk_index, "
            "dc.content, dc.page, dc.header_breadcrumb, d.path, d.filename, "
            "d.title, COALESCE(d.tags, ARRAY[]::text[]) AS tags, d.source_kind, "
            "COALESCE(d.metadata, '{}'::jsonb) AS metadata, "
            "ce.embedding <=> $6::vector AS distance "
            "FROM chunk_embeddings ce "
            "JOIN documents d ON d.id=ce.document_id "
            "JOIN document_chunks dc ON dc.document_id=ce.document_id "
            "AND dc.document_version=ce.document_version "
            "AND dc.chunk_index=ce.chunk_index "
            f"WHERE {' AND '.join(where)}"
            "), counted AS ("
            "SELECT *, count(*) OVER () AS candidate_count FROM filtered"
            ") SELECT *, 1.0 - distance AS score FROM counted "
            "ORDER BY distance ASC, document_id, document_version, chunk_index "
            f"LIMIT {limit_parameter}"
        )
        rows = await _fetch_rows(self._pool, sql, params)
        return _result_from_rows(rows, started_at=started_at)


def _raise_sanitized_boundary(failure: BaseException, message: str) -> Never:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None:
        raise signal
    raise RetrieverUnavailable(message)


async def _fetch_rows(pool, sql: str, params: Sequence[object]):
    failure = None
    rows = ()
    try:
        rows = await pool.fetch(sql, *params)
    except BaseException as caught:  # noqa: BLE001 - sanitize the database adapter boundary.
        failure = caught
    if failure is not None:
        _raise_sanitized_boundary(failure, "vector store is unavailable")
    return rows


def _result_from_rows(rows, *, started_at: float) -> SearchResult:
    failure = None
    result = None
    try:
        dictionaries = _validated_search_rows(rows)
        hits = tuple(_search_hit(row) for row in dictionaries)
        candidate_count = dictionaries[0]["candidate_count"] if dictionaries else 0
        result = SearchResult(
            hits=hits,
            candidate_count=candidate_count,
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="vector",
        )
    except BaseException as caught:  # noqa: BLE001 - malformed backend rows are unavailable.
        failure = caught
    if failure is not None:
        _raise_sanitized_boundary(failure, "vector store is unavailable")
    if result is None:
        raise RetrieverUnavailable("vector store is unavailable")
    return result


def _strict_search_metadata(value: object) -> dict[str, object]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    metadata = dict(value)
    encoded = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_SEARCH_METADATA_BYTES:
        raise ValueError("metadata exceeds the search boundary")
    return metadata


def _validated_search_rows(rows: object) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) > 10_000:
        raise ValueError("search rows must be a bounded sequence")
    normalized: list[dict[str, object]] = []
    candidate_count: int | None = None
    for raw_row in rows:
        is_asyncpg_record = (
            type(raw_row).__name__ == "Record"
            and type(raw_row).__module__.startswith("asyncpg.")
        )
        if not isinstance(raw_row, Mapping) and not is_asyncpg_record:
            try:
                dict(raw_row)
            except BaseException as failure:  # noqa: BLE001 - preserve boundary control classification.
                if signal := sanitized_boundary_signal_or_unknown(failure):
                    raise signal from None
            raise ValueError("search row must be a database record")
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
            raise ValueError("search row is incomplete")

        count = row["candidate_count"]
        document_id = row["document_id"]
        version = row["document_version"]
        chunk_index = row["chunk_index"]
        content = row["content"]
        path = row["path"]
        filename = row["filename"]
        score = row["score"]
        page = row["page"]
        tags = row["tags"]
        title = row.get("title")
        header = row.get("header_breadcrumb")
        source_kind = row["source_kind"]
        if (
            type(count) is not int
            or not 0 <= count <= _POSTGRES_INTEGER_MAX
            or (candidate_count is not None and count != candidate_count)
            or not isinstance(document_id, UUID)
            or type(version) is not int
            or not 0 <= version <= _POSTGRES_INTEGER_MAX
            or type(chunk_index) is not int
            or not 0 <= chunk_index < _MAX_EMBEDDINGS_PER_DOCUMENT
            or type(content) is not str
            or len(content) > _MAX_SEARCH_TEXT_CHARS
            or type(path) is not str
            or not path.startswith("/")
            or "\x00" in path
            or len(path) > _MAX_SEARCH_PATH_CHARS
            or type(filename) is not str
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or len(filename) > _MAX_SEARCH_PATH_CHARS
            or type(score) is not float
            or not isfinite(score)
            or not -1.0 <= score <= 1.0
            or (
                page is not None
                and (type(page) is not int or not 0 <= page <= _POSTGRES_INTEGER_MAX)
            )
            or isinstance(tags, (str, bytes))
            or not isinstance(tags, Sequence)
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
            or (title is not None and (type(title) is not str or len(title) > _MAX_SEARCH_PATH_CHARS))
            or (header is not None and (type(header) is not str or len(header) > _MAX_SEARCH_PATH_CHARS))
        ):
            raise ValueError("search row is invalid")
        metadata = _strict_search_metadata(row["metadata"])
        candidate_count = count
        normalized.append({**row, "metadata": metadata, "tags": tuple(tags)})
    if candidate_count is not None and candidate_count < len(normalized):
        raise ValueError("candidate count is less than returned rows")
    return normalized


def _document_vectors(embeddings: object, *, dimensions: int) -> tuple[tuple[int, str], ...]:
    if isinstance(embeddings, (str, bytes)) or not isinstance(embeddings, Sequence):
        raise ValueError("embeddings must be a sequence")
    if len(embeddings) > _MAX_EMBEDDINGS_PER_DOCUMENT:
        raise ValueError("embeddings exceed the document chunk limit")
    normalized: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in embeddings:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("each embedding must contain a chunk index and vector")
        index = _bounded_int(
            item[0],
            label="chunk index",
            minimum=0,
            maximum=_MAX_EMBEDDINGS_PER_DOCUMENT - 1,
        )
        if index in seen:
            raise ValueError("duplicate chunk index in embeddings")
        seen.add(index)
        normalized.append((index, _vector(item[1], dimensions=dimensions)))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _vector(value: object, *, dimensions: int) -> str:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("embedding must be a sequence of finite numbers")
    if len(value) != dimensions:
        raise ValueError("embedding dimensions do not match the configured profile")
    normalized: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ValueError("embedding coordinate must be a finite float32 number")
        normalized.append(_float32_coordinate(coordinate))
    if not any(normalized):
        raise ValueError("embedding must contain a non-zero coordinate")
    return "[" + ",".join(format(number, ".17g") for number in normalized) + "]"


def _float32_coordinate(coordinate: Real) -> float:
    invalid = False
    number = 0.0
    try:
        number = float(coordinate)
    except (OverflowError, TypeError, ValueError):
        invalid = True
    if invalid or not isfinite(number):
        raise ValueError("embedding coordinate must be a finite float32 number")

    invalid = False
    quantized = 0.0
    try:
        quantized = struct.unpack("!f", struct.pack("!f", number))[0]
    except (OverflowError, struct.error):
        invalid = True
    if invalid or not isfinite(quantized) or (number != 0.0 and quantized == 0.0):
        raise ValueError("embedding coordinate must be a finite float32 number")
    return quantized


def _uuid(value: object, *, label: str) -> UUID:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a UUID")
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _bounded_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _candidate_limit(query: SearchQuery) -> int:
    value = query.candidate_limit
    if isinstance(value, bool) or not isinstance(value, int) or value < query.limit or value > 500:
        raise ValueError("candidate limit must be between search limit and 500")
    return value


def _search_hit(row: Mapping[str, object]) -> SearchHit:
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


__all__ = ["PostgresVectorStore"]
