"""Tenant-scoped, document-version-fenced pgvector persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from time import perf_counter
from uuid import UUID

from llmwiki_core.documents import DocumentKind
from llmwiki_core.facets import validate_facets
from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.search import (
    RetrieverUnavailable,
    SearchArea,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchScope,
)

_MAX_EMBEDDINGS_PER_DOCUMENT = 10_000

_SCALAR_FACETS = {
    "genre": "genre",
    "evidence": "evidence",
    "origin": "origin",
    "timeliness": "timeliness",
    "state": "lifecycle_state",
    "entry_id": "entry_id",
}
_ARRAY_FACETS = {
    "rule": "rule_type",
    "dept": "gov_dept",
    "region": "geo_region",
    "industry": "industry",
    "mode": "mode",
}
_PRIMARY_EXTENSION_FACETS = {
    "stage": ("stage", "stage_ext"),
    "domain": ("domain", "domain_ext"),
}


class _VectorWriteRejected(ValueError):
    """A stable caller-visible version or chunk-set rejection."""


class PostgresVectorStore:
    """Store complete embedding sets and retrieve exact cosine candidates."""

    def __init__(self, pool, *, profile: EmbeddingProfile) -> None:
        if not isinstance(profile, EmbeddingProfile):
            raise TypeError("profile must be an EmbeddingProfile")
        if len(profile.provider) > 100 or len(profile.model) > 200:
            raise ValueError("embedding profile model identity is too long")
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
        version = _bounded_int(document_version, label="document_version", minimum=0)
        vectors = _document_vectors(embeddings, dimensions=self._profile.dimensions)

        backend_failed = False
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                current_version = await conn.fetchval(
                    "SELECT version FROM documents "
                    "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 "
                    "AND NOT archived FOR UPDATE",
                    document_uuid,
                    user_uuid,
                    kb_uuid,
                )
                if current_version is None:
                    raise _VectorWriteRejected(
                        "document is not available in the requested tenant scope"
                    )
                if current_version != version:
                    raise _VectorWriteRejected(
                        "document_version must equal the current document version"
                    )

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
                    raise _VectorWriteRejected(
                        "embeddings must match the complete current chunk set"
                    )

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
        except (_VectorWriteRejected, RetrieverUnavailable):
            raise
        except Exception:  # noqa: BLE001 - sanitize the database adapter boundary.
            backend_failed = True
        if backend_failed:
            raise RetrieverUnavailable("vector store is unavailable")
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
        if query.annotated_only or query.scope is SearchScope.ANNOTATIONS:
            where.append("dc.has_highlight=true")
        if query.scope is SearchScope.SOURCE:
            where.append("dc.source_content <> ''")
        if query.area is SearchArea.WIKI:
            where.append("d.source_kind='wiki'")
        elif query.area is SearchArea.SOURCES:
            where.append("d.source_kind!='wiki'")
        if query.document_kinds:
            where.append(f"d.source_kind=ANY({bind([kind.value for kind in query.document_kinds])}::text[])")
        if query.path_glob is not None:
            where.append(f"(d.path || d.filename) LIKE {bind(_logical_glob_to_sql_like(query.path_glob))} ESCAPE '\\'")
        if query.tags:
            where.append(
                "ARRAY(SELECT lower(tag) FROM unnest(COALESCE(d.tags, ARRAY[]::text[])) tag) "
                f"@> {bind(list(query.tags))}::text[]"
            )
        where.extend(_facet_conditions(dict(query.facets), bind=bind))
        limit_parameter = bind(query.candidate_limit)

        sql = (
            "WITH filtered AS ("
            "SELECT ce.document_id, ce.document_version, ce.chunk_index, "
            "dc.content, dc.page, dc.header_breadcrumb, d.path, d.filename, "
            "d.title, d.tags, d.source_kind, d.metadata, "
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


async def _fetch_rows(pool, sql: str, params: Sequence[object]):
    failed = False
    rows = ()
    try:
        rows = await pool.fetch(sql, *params)
    except Exception:  # noqa: BLE001 - sanitize the database adapter boundary.
        failed = True
    if failed:
        raise RetrieverUnavailable("vector store is unavailable")
    return rows


def _result_from_rows(rows, *, started_at: float) -> SearchResult:
    invalid = False
    result = None
    try:
        dictionaries = [dict(row) for row in rows]
        hits = tuple(_search_hit(row) for row in dictionaries)
        candidate_count = int(dictionaries[0]["candidate_count"]) if dictionaries else 0
        result = SearchResult(
            hits=hits,
            candidate_count=candidate_count,
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="vector",
        )
    except Exception:  # noqa: BLE001 - malformed backend rows are unavailable.
        invalid = True
    if invalid or result is None:
        raise RetrieverUnavailable("vector store is unavailable")
    return result


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
        index = _bounded_int(item[0], label="chunk index", minimum=0)
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
            raise ValueError("embedding must contain only finite numbers")
        number = float(coordinate)
        if not isfinite(number):
            raise ValueError("embedding must contain only finite numbers")
        normalized.append(number)
    if not any(normalized):
        raise ValueError("embedding must contain a non-zero coordinate")
    return "[" + ",".join(format(number, ".17g") for number in normalized) + "]"


def _uuid(value: object, *, label: str) -> UUID:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a UUID")
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a UUID") from exc


def _bounded_int(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _candidate_limit(query: SearchQuery) -> int:
    value = query.candidate_limit
    if isinstance(value, bool) or not isinstance(value, int) or value < query.limit or value > 500:
        raise ValueError("candidate limit must be between search limit and 500")
    return value


def _logical_glob_to_sql_like(path_glob: str) -> str:
    directory = path_glob.endswith("/") or (
        "*" not in path_glob and "." not in path_glob.rsplit("/", 1)[-1] and path_glob != "/"
    )
    escaped: list[str] = []
    index = 0
    while index < len(path_glob):
        char = path_glob[index]
        if char == "*":
            if index + 1 < len(path_glob) and path_glob[index + 1] == "*":
                index += 1
            escaped.append("%")
        elif char in {"%", "_", "\\"}:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
        index += 1
    pattern = "".join(escaped)
    if path_glob == "/":
        return "/%"
    if directory:
        return pattern.rstrip("/") + "/%"
    return pattern


def _facet_conditions(facets: dict[str, object], *, bind) -> list[str]:
    clean = validate_facets(facets)
    conditions: list[str] = []
    for key, value in clean.items():
        parameter = bind(value)
        if key == "timeliness":
            conditions.append(
                f"(d.metadata->>'timeliness'={parameter} OR "
                f"d.metadata#>>'{{facet_rollup,timeliness_worst}}'={parameter})"
            )
        elif key in _SCALAR_FACETS:
            conditions.append(f"d.metadata->>'{_SCALAR_FACETS[key]}'={parameter}")
        elif key in _ARRAY_FACETS:
            conditions.append(f"d.metadata->'{_ARRAY_FACETS[key]}' ? {parameter}")
        elif key in _PRIMARY_EXTENSION_FACETS:
            primary, extension = _PRIMARY_EXTENSION_FACETS[key]
            conditions.append(
                f"(d.metadata->>'{primary}'={parameter} OR "
                f"d.metadata->'{extension}' ? {parameter} OR "
                f"d.metadata#>'{{facet_rollup,{key}}}' ? {parameter})"
            )
        elif key == "country":
            conditions.append(
                f"(d.metadata->'geo_country' ? {parameter} OR "
                f"d.metadata->'geo_country_names' ? {parameter} OR "
                f"d.metadata#>'{{facet_rollup,country}}' ? {parameter})"
            )
        elif key == "business":
            if "." in value:
                conditions.append(
                    f"(d.metadata#>>'{{business,code}}'={parameter} OR "
                    f"d.metadata#>'{{facet_rollup,business}}' ? {parameter})"
                )
            else:
                prefix = bind(f"{value}.%")
                conditions.append(
                    f"(d.metadata#>>'{{business,code}}'={parameter} OR "
                    f"d.metadata#>>'{{business,code}}' LIKE {prefix} OR "
                    f"d.metadata#>'{{facet_rollup,business}}' ? {parameter})"
                )
    return conditions


def _search_hit(row: Mapping[str, object]) -> SearchHit:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    tags = row.get("tags")
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        tags = ()
    return SearchHit(
        document_id=str(row["document_id"]),
        document_version=int(row["document_version"]),
        chunk_index=int(row["chunk_index"]),
        content=str(row["content"]),
        score=float(row["score"]),
        path=f"{row['path']}{row['filename']}",
        title=row.get("title"),
        page=row.get("page"),
        header_breadcrumb=row.get("header_breadcrumb"),
        tags=tuple(tags),
        document_kind=DocumentKind(str(row["source_kind"])),
        metadata=dict(metadata),
    )


__all__ = ["PostgresVectorStore"]
