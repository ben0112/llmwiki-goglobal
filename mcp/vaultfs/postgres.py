"""Postgres + S3 implementation of VaultFS."""

import json
import logging
import re
from collections.abc import Sequence
from datetime import date
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import NoReturn
from uuid import UUID

import aioboto3
import asyncpg
from config import settings
from db import get_pool, scoped_execute, scoped_query, scoped_queryrow, service_execute, service_queryrow
from services.chunker import chunk_text, store_chunks_pg

import llmwiki_adapters.postgres.wiki as postgres_wiki_adapter
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
from llmwiki_core.wiki import WikiWriteBundle

from .base import (
    DuplicateDocumentError,
    VaultFS,
    _vault_search_hit,
    is_wiki_directory,
)
from .facets import postgres_facet_conditions, validate_facets

logger = logging.getLogger(__name__)

_s3_session = None

_OVERVIEW_TEMPLATE = """\
---
title: Overview
description: Research hub for {name}.
date: {date}
tags: [overview, wiki]
---

This wiki tracks research on {name}. No sources have been ingested yet.

## Key Findings

No sources ingested yet - add your first source to get started.

## Recent Updates

No activity yet.\
"""

_LOG_TEMPLATE = """\
Chronological record of ingests, queries, and maintenance passes.

## [{date}] created | Wiki Created
- Initialized wiki: {name}\
"""


def _get_s3_session():
    global _s3_session
    if _s3_session is None and settings.AWS_ACCESS_KEY_ID:
        _s3_session = aioboto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return _s3_session


def _s3_client_kwargs() -> dict:
    """Client kwargs honoring a self-hosted S3-compatible endpoint (MinIO).

    Mirrors api/services/s3.py s3_client_kwargs.
    """
    kwargs: dict = {}
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
    if settings.S3_FORCE_PATH_STYLE:
        from botocore.config import Config as BotoConfig
        kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
    return kwargs


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug).strip("-")
    return slug or "kb"


def _postgres_search_hit(row: dict):
    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            raw_metadata = {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    raw_tags = row.get("tags")
    return _vault_search_hit(
        document_id=str(row["document_id"]),
        document_version=int(row["document_version"]),
        chunk_index=int(row["chunk_index"]),
        content=row["content"],
        score=float(row["score"]),
        path=f"{row['path']}{row['filename']}",
        title=row.get("title"),
        page=row.get("page"),
        header_breadcrumb=row.get("header_breadcrumb"),
        tags=raw_tags,
        document_kind=DocumentKind(row["source_kind"]),
        metadata=metadata,
        filename=row["filename"],
        directory=row["path"],
        file_type=row["file_type"],
        source_content=row.get("source_content") or "",
        annotations_text=row.get("annotations_text"),
        has_highlight=bool(row.get("has_highlight")),
        source_hit=bool(row.get("source_hit")),
        annotation_hit=bool(row.get("annotation_hit")),
    )


def _vector_literal(value: object, *, dimensions: int) -> str:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("embedding must be a sequence of finite numbers")
    if len(value) != dimensions:
        raise ValueError("embedding dimensions do not match the configured profile")
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ValueError("embedding must contain finite numbers")
        normalized = float(coordinate)
        if not isfinite(normalized):
            raise ValueError("embedding must contain finite numbers")
        coordinates.append(normalized)
    if not any(coordinates):
        raise ValueError("embedding must contain a non-zero coordinate")
    return "[" + ",".join(format(number, ".17g") for number in coordinates) + "]"


def _postgres_document_filters(
    query: SearchQuery,
    params: list,
    *,
    doc_alias: str,
    chunk_alias: str,
) -> list[str]:
    return postgres_retrieval.postgres_document_filter_conditions(
        query,
        params,
        doc_alias=doc_alias,
        chunk_alias=chunk_alias,
    )


def _raise_postgres_ordinary_boundary(failure: BaseException, message: str) -> NoReturn:
    if isinstance(failure, (asyncpg.PostgresError, OSError, TimeoutError)):
        raise RetrieverUnavailable(message) from None
    raise failure


class PostgresVaultFS(VaultFS):
    """Postgres + S3 vault."""

    def __init__(self, user_id: str):
        self.user_id = user_id


    async def resolve_kb(self, slug: str) -> dict | None:
        return await scoped_queryrow(
            self.user_id,
            "SELECT id, name, slug FROM knowledge_bases WHERE slug = $1 AND user_id = $2",
            slug, self.user_id,
        )

    async def list_knowledge_bases(self) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT name, slug, created_at FROM knowledge_bases WHERE user_id = $1 ORDER BY created_at DESC",
            self.user_id,
        )

    async def create_knowledge_base(self, name: str, description: str | None = None, kind: str = "wiki") -> dict:
        row = await self._insert_knowledge_base(name, description, kind)
        await self._scaffold_wiki(str(row["id"]), row["name"])
        return row

    async def update_knowledge_base(self, kb_id: str, name: str | None = None, description: str | None = None, kind: str | None = None) -> dict | None:
        # knowledge_bases has no RLS write policy; writes go through the
        # service role with the explicit user_id filter, like every other KB write.
        # Renaming regenerates the slug, matching the web API's semantics.
        if name is not None:
            slug = await self._unique_slug(name)
            return await service_queryrow(
                "UPDATE knowledge_bases SET name = $1, slug = $2, "
                "description = COALESCE($3, description), kind = COALESCE($4, kind), updated_at = now() "
                "WHERE id = $5::uuid AND user_id = $6 "
                "RETURNING id, name, slug, description, kind",
                name, slug, description, kind, kb_id, self.user_id,
            )
        return await service_queryrow(
            "UPDATE knowledge_bases SET description = COALESCE($1, description), "
            "kind = COALESCE($2, kind), updated_at = now() "
            "WHERE id = $3::uuid AND user_id = $4 "
            "RETURNING id, name, slug, description, kind",
            description, kind, kb_id, self.user_id,
        )


    async def get_document(self, kb_id: str, filename: str, dir_path: str) -> dict | None:
        return await scoped_queryrow(
            self.user_id,
            "SELECT id, user_id, filename, title, path, content, tags, version, file_type, "
            "page_count, highlights, metadata, date, created_at, updated_at "
            "FROM documents WHERE knowledge_base_id = $1 AND filename = $2 AND path = $3 AND NOT archived AND user_id = $4",
            kb_id, filename, dir_path, self.user_id,
        )

    async def find_document_by_name(self, kb_id: str, name: str) -> dict | None:
        return await scoped_queryrow(
            self.user_id,
            "SELECT id, user_id, filename, title, path, content, tags, version, file_type, "
            "page_count, highlights, metadata, date, created_at, updated_at "
            "FROM documents WHERE knowledge_base_id = $1 AND (filename = $2 OR title = $2) AND NOT archived AND user_id = $3",
            kb_id, name, self.user_id,
        )

    async def create_document(self, kb_id: str, filename: str, title: str, dir_path: str, file_type: str, content: str, tags: list[str], date: str | None = None, metadata: dict | None = None) -> dict:
        import json as _json
        pool = await get_pool()
        source_kind = "wiki" if is_wiki_directory(dir_path) else "source"
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    row = await conn.fetchrow(
                        "INSERT INTO documents (knowledge_base_id, user_id, filename, title, path, "
                        "source_kind, file_type, status, content, tags, date, metadata, version) "
                        "SELECT $1, $2, $3, $4, $5, $6, $7, 'ready', $8, $9, $10, $11::jsonb, 1 "
                        "WHERE EXISTS (SELECT 1 FROM knowledge_bases WHERE id = $1 AND user_id = $2) "
                        "RETURNING id, filename, path",
                        kb_id, self.user_id, filename, title, dir_path, source_kind, file_type,
                        content, tags, date, _json.dumps(metadata) if metadata else None,
                    )
                except asyncpg.UniqueViolationError as e:
                    # Only re-raise as DuplicateDocumentError for the path/filename index.
                    # Any other unique violation is a different bug worth surfacing.
                    if e.constraint_name == "idx_documents_unique_active":
                        raise DuplicateDocumentError(dir_path, filename)
                    raise
                if row is None:
                    raise PermissionError(f"knowledge base {kb_id} not owned by user")
                if file_type in ("md", "txt"):
                    chunks = chunk_text(content or "")
                    await store_chunks_pg(conn, str(row["id"]), self.user_id, kb_id, 1, chunks)
        return dict(row)

    async def update_document(self, doc_id: str, content: str, tags: list[str] | None = None, title: str | None = None, date: str | None = None, metadata: dict | None = None) -> dict | None:
        import json as _json
        # 内容更新即视为已复查,清除待复查标记(stale 只在这里被清除)
        sets = ["content = $1", "version = COALESCE(version, 0) + 1",
                "updated_at = now()", "stale_since = NULL"]
        args: list = [content, doc_id, self.user_id]
        idx = 4

        if title is not None:
            sets.append(f"title = ${idx}")
            args.append(title)
            idx += 1
        if tags is not None:
            sets.append(f"tags = ${idx}")
            args.append(tags)
            idx += 1
        if date is not None:
            sets.append(f"date = ${idx}")
            args.append(date)
            idx += 1
        if metadata is not None:
            sets.append(f"metadata = ${idx}::jsonb")
            args.append(_json.dumps(metadata))
            idx += 1

        sql = (
            f"UPDATE documents SET {', '.join(sets)} "
            f"WHERE id = $2 AND user_id = $3 "
            f"RETURNING id, filename, path, knowledge_base_id, file_type, version"
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(sql, *args)
                if row and row["file_type"] in ("md", "txt"):
                    chunks = chunk_text(content or "")
                    await store_chunks_pg(
                        conn, str(row["id"]), self.user_id,
                        str(row["knowledge_base_id"]), row["version"], chunks,
                    )
        return {"id": row["id"], "filename": row["filename"], "path": row["path"]} if row else None

    async def write_wiki_bundle(self, kb_id: str, bundle: WikiWriteBundle) -> dict:
        """Commit a wiki revision and every derived row in one Postgres transaction."""
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            result = await postgres_wiki_adapter.write_wiki_bundle_in_transaction(
                conn,
                user_id=UUID(self.user_id),
                knowledge_base_id=UUID(kb_id),
                bundle=bundle,
            )
        return {
            "id": str(result.document_id),
            "filename": result.filename,
            "path": result.path,
            "version": result.version,
        }

    async def archive_documents(self, doc_ids: list[str]) -> int:
        result = await service_execute(
            "UPDATE documents SET archived = true, updated_at = now() "
            "WHERE id = ANY($1::uuid[]) AND user_id = $2",
            doc_ids, self.user_id,
        )
        return int(result.split()[-1]) if result else 0


    async def list_documents(self, kb_id: str, facets: dict | None = None) -> list[dict]:
        conds, facet_params = postgres_facet_conditions(validate_facets(facets), start_index=3, doc_alias="d")
        facet_sql = "".join(f" AND {c}" for c in conds)
        return await scoped_query(
            self.user_id,
            "SELECT id, filename, title, path, file_type, tags, page_count, date, updated_at, metadata "
            "FROM documents d WHERE knowledge_base_id = $1 AND NOT archived AND user_id = $2 "
            "AND COALESCE(metadata->>'asset', 'false') <> 'true' "
            f"{facet_sql} "
            "ORDER BY path, filename",
            kb_id, self.user_id, *facet_params,
        )

    async def list_documents_with_content(self, kb_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT id, filename, title, path, content, tags, file_type, page_count, highlights, metadata, date "
            "FROM documents WHERE knowledge_base_id = $1 AND NOT archived AND user_id = $2 "
            "AND COALESCE(metadata->>'asset', 'false') <> 'true' "
            "ORDER BY path, filename",
            kb_id, self.user_id,
        )


    async def get_pages(self, doc_id: str, page_nums: list[int]) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT page, content, elements FROM document_pages "
            "WHERE document_id = $1 AND page = ANY($2) ORDER BY page",
            doc_id, page_nums,
        )

    async def get_all_pages(self, doc_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT page, content, elements FROM document_pages "
            "WHERE document_id = $1 ORDER BY page",
            doc_id,
        )


    async def retrieve(self, kb_id: str, query: SearchQuery) -> SearchResult:
        started_at = perf_counter()
        compiled = postgres_retrieval.compile_postgres_lexical_query(
            self.user_id,
            kb_id,
            query,
        )

        rows = await scoped_query(
            self.user_id,
            compiled.sql,
            *compiled.params,
        )
        row_dicts = [dict(row) for row in rows]
        hits = tuple(_postgres_search_hit(row) for row in row_dicts)
        candidate_count = int(row_dicts[0]["candidate_count"]) if row_dicts else 0
        return SearchResult(
            hits=hits,
            candidate_count=candidate_count,
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="lexical",
        )

    async def retrieve_vector(
        self,
        kb_id: str,
        query: SearchQuery,
        *,
        embedding: tuple[float, ...],
        profile: EmbeddingProfile,
    ) -> SearchResult:
        """Retrieve exact cosine candidates inside the authenticated tenant scope."""
        if not isinstance(profile, EmbeddingProfile):
            raise TypeError("profile must be an EmbeddingProfile")
        if query.scope is not SearchScope.ALL:
            raise RetrieverUnavailable("vector retrieval does not support scoped content")
        vector = _vector_literal(embedding, dimensions=profile.dimensions)
        started_at = perf_counter()

        availability_failure = None
        available = None
        try:
            available = await scoped_queryrow(
                self.user_id,
                "SELECT EXISTS(SELECT 1 FROM chunk_embeddings ce "
                "JOIN documents d ON d.id=ce.document_id "
                "WHERE ce.user_id=$1::uuid AND ce.knowledge_base_id=$2::uuid "
                "AND ce.provider=$3 AND ce.model=$4 AND ce.dimensions=$5 "
                "AND d.user_id=$1::uuid AND d.knowledge_base_id=$2::uuid "
                "AND ce.document_version=d.version AND NOT d.archived "
                "AND d.status != 'failed') AS available",
                self.user_id,
                kb_id,
                profile.provider,
                profile.model,
                profile.dimensions,
            )
        except BaseException as failure:  # noqa: BLE001 - sanitize database signals.
            availability_failure = failure
        if availability_failure is not None:
            if signal := sanitized_boundary_signal_or_unknown(availability_failure):
                raise signal from None
            _raise_postgres_ordinary_boundary(
                availability_failure,
                "vector store is unavailable",
            )
        if not available or not available["available"]:
            raise RetrieverUnavailable("current embeddings are unavailable")

        params: list = [
            self.user_id,
            kb_id,
            profile.provider,
            profile.model,
            profile.dimensions,
            vector,
        ]

        def bind(value) -> str:
            params.append(value)
            return f"${len(params)}"

        where = [
            "ce.user_id=$1::uuid",
            "ce.knowledge_base_id=$2::uuid",
            "ce.provider=$3",
            "ce.model=$4",
            "ce.dimensions=$5",
            "d.user_id=$1::uuid",
            "d.knowledge_base_id=$2::uuid",
            "dc.user_id=$1::uuid",
            "dc.knowledge_base_id=$2::uuid",
            "ce.document_version=d.version",
            "dc.document_version=d.version",
            "d.status != 'failed'",
            "NOT d.archived",
        ]
        where.extend(
            _postgres_document_filters(
                query,
                params,
                doc_alias="d",
                chunk_alias="dc",
            )
        )
        limit_param = bind(query.candidate_limit)

        search_failure = None
        rows = []
        try:
            rows = await scoped_query(
                self.user_id,
                "WITH filtered AS ("
                "SELECT ce.document_id, ce.document_version, ce.chunk_index, "
                "dc.content, dc.source_content, dc.annotations_text, dc.has_highlight, "
                "dc.page, dc.header_breadcrumb, d.path, d.filename, d.title, d.file_type, "
                "d.tags, d.source_kind, d.metadata, "
                "ce.embedding <=> $6::vector AS distance "
                "FROM chunk_embeddings ce JOIN documents d ON d.id=ce.document_id "
                "JOIN document_chunks dc ON dc.document_id=ce.document_id "
                "AND dc.document_version=ce.document_version "
                "AND dc.chunk_index=ce.chunk_index "
                f"WHERE {' AND '.join(where)}"
                "), counted AS ("
                "SELECT *, count(*) OVER () AS candidate_count FROM filtered"
                ") SELECT *, 1.0-distance AS score, false AS source_hit, "
                "false AS annotation_hit FROM counted "
                "ORDER BY distance, document_id, document_version, chunk_index "
                f"LIMIT {limit_param}",
                *params,
            )
        except BaseException as failure:  # noqa: BLE001 - sanitize database signals.
            search_failure = failure
        if search_failure is not None:
            if signal := sanitized_boundary_signal_or_unknown(search_failure):
                raise signal from None
            _raise_postgres_ordinary_boundary(
                search_failure,
                "vector store is unavailable",
            )
        hits = tuple(_postgres_search_hit(row) for row in rows)
        candidate_count = int(rows[0]["candidate_count"]) if rows else 0
        return SearchResult(
            hits=hits,
            candidate_count=candidate_count,
            latency_ms=(perf_counter() - started_at) * 1000,
            profile="vector",
        )

    async def expand_references(
        self,
        kb_id: str,
        query: SearchQuery,
        hits: tuple[SearchHit, ...],
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        """Follow one outbound edge and fetch one current chunk per related doc."""
        if type(limit) is not int or limit <= 0 or not hits:
            return ()
        direct_ids = tuple(dict.fromkeys(hit.document_id for hit in hits))[:100]
        if not direct_ids:
            return ()
        scan_limit = min(100, limit)
        params: list = [kb_id, list(direct_ids), self.user_id]
        where = [
            "ref.knowledge_base_id=$1::uuid",
            "target.knowledge_base_id=$1::uuid",
            "target.user_id=$3::uuid",
            "dc.user_id=$3::uuid",
            "dc.knowledge_base_id=$1::uuid",
            "NOT target.archived",
            "target.status != 'failed'",
            "NOT (target.id=ANY($2::uuid[]))",
        ]
        where.extend(
            _postgres_document_filters(
                query,
                params,
                doc_alias="target",
                chunk_alias="dc",
            )
        )
        params.append(scan_limit)
        limit_parameter = f"${len(params)}"
        expansion_failure = None
        rows = []
        try:
            rows = await scoped_query(
                self.user_id,
                "WITH direct AS ("
                "SELECT document_id, direct_rank FROM "
                "unnest($2::uuid[]) WITH ORDINALITY AS input(document_id, direct_rank)"
                "), related AS ("
                "SELECT DISTINCT ON (target.id) direct.direct_rank, "
                "target.id AS document_id, target.version AS document_version, "
                "dc.chunk_index, dc.content, dc.source_content, dc.annotations_text, "
                "dc.has_highlight, dc.page, dc.header_breadcrumb, target.path, "
                "target.filename, target.title, target.file_type, target.tags, "
                "target.source_kind, target.metadata "
                "FROM direct JOIN document_references ref "
                "ON ref.source_document_id=direct.document_id "
                "JOIN documents target ON target.id=ref.target_document_id "
                "JOIN document_chunks dc ON dc.document_id=target.id "
                "AND dc.document_version=target.version "
                f"WHERE {' AND '.join(where)} "
                "ORDER BY target.id, direct.direct_rank, dc.chunk_index"
                ") SELECT *, 0.0::double precision AS score, "
                "false AS source_hit, false AS annotation_hit FROM related "
                f"ORDER BY direct_rank, document_id LIMIT {limit_parameter}",
                *params,
            )
        except BaseException as failure:  # noqa: BLE001 - sanitize database signals.
            expansion_failure = failure
        if expansion_failure is not None:
            if signal := sanitized_boundary_signal_or_unknown(expansion_failure):
                raise signal from None
            _raise_postgres_ordinary_boundary(
                expansion_failure,
                "reference expansion is unavailable",
            )
        return tuple(_postgres_search_hit(row) for row in rows[:limit])

    async def search_chunks(
        self, kb_id: str, query: str, limit: int,
        path_filter: str | None = None,
        annotated_only: bool = False,
        scope: str = "all",
        facets: dict | None = None,
    ) -> list[dict]:
        return await super().search_chunks(
            kb_id,
            query,
            limit,
            path_filter,
            annotated_only,
            scope,
            facets,
        )


    async def load_source_bytes(self, doc: dict) -> bytes | None:
        file_type = doc.get("file_type", "")
        s3_key = f"{self.user_id}/{doc['id']}/source.{file_type}"
        return await self._load_s3(s3_key)

    async def load_image_bytes(self, doc_id: str, image_id: str) -> bytes | None:
        s3_key = f"{self.user_id}/{doc_id}/images/{image_id}"
        return await self._load_s3(s3_key)

    async def load_asset_bytes(self, asset_doc_id: str) -> bytes | None:
        row = await scoped_queryrow(
            self.user_id,
            "SELECT id, user_id, filename, file_type FROM documents "
            "WHERE id = $1 AND user_id = $2 AND NOT archived",
            asset_doc_id, self.user_id,
        )
        if not row:
            return None
        return await self.load_source_bytes(dict(row))

    async def _load_s3(self, key: str) -> bytes | None:
        session = _get_s3_session()
        if not session:
            return None
        try:
            async with session.client("s3", **_s3_client_kwargs()) as s3:
                resp = await s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
                return await resp["Body"].read()
        except Exception as e:
            logger.warning("Failed to load S3 key %s: %s", key, e)
            return None


    def write_to_disk(self, dir_path: str, filename: str, content: str) -> bool:
        return True

    def delete_from_disk(self, docs: list[dict]) -> None:
        pass


    async def delete_references(self, source_doc_id: str, ref_types: tuple | None = None) -> None:
        if ref_types:
            await scoped_execute(
                self.user_id,
                "DELETE FROM document_references WHERE source_document_id = $1 "
                "AND reference_type = ANY($2::text[])",
                source_doc_id, list(ref_types),
            )
        else:
            await scoped_execute(
                self.user_id,
                "DELETE FROM document_references WHERE source_document_id = $1",
                source_doc_id,
            )

    async def delete_reference(self, source_id: str, target_id: str, ref_type: str) -> bool:
        result = await scoped_execute(
            self.user_id,
            "DELETE FROM document_references WHERE source_document_id = $1 "
            "AND target_document_id = $2 AND reference_type = $3",
            source_id, target_id, ref_type,
        )
        return bool(result) and result.split()[-1] != "0"

    async def upsert_reference(self, source_id: str, target_id: str, kb_id: str, ref_type: str, page: int | None) -> None:
        try:
            await scoped_execute(
                self.user_id,
                "INSERT INTO document_references "
                "(source_document_id, target_document_id, knowledge_base_id, reference_type, page) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (source_document_id, target_document_id, reference_type) DO UPDATE "
                "SET page = EXCLUDED.page, created_at = now()",
                source_id, target_id, kb_id, ref_type, page,
            )
        except Exception as e:
            logger.warning("Failed to insert reference %s -> %s: %s", source_id[:8], target_id[:8], e)

    async def propagate_staleness(self, doc_id: str) -> None:
        await service_execute(
            "UPDATE documents SET stale_since = now() "
            "WHERE id IN ("
            "  SELECT source_document_id FROM document_references "
            "  WHERE target_document_id = $1 AND reference_type = 'links_to'"
            ") AND stale_since IS NULL AND user_id = $2",
            doc_id, self.user_id,
        )

    async def mark_cites_stale(self, target_doc_ids: list[str]) -> int:
        if not target_doc_ids:
            return 0
        result = await service_execute(
            "UPDATE documents SET stale_since = now() "
            "WHERE id IN ("
            "  SELECT source_document_id FROM document_references "
            "  WHERE reference_type = 'cites' AND target_document_id = ANY($1::uuid[])"
            ") AND source_kind = 'wiki' AND stale_since IS NULL AND user_id = $2",
            target_doc_ids, self.user_id,
        )
        try:
            return int(str(result).split()[-1])
        except (ValueError, IndexError):
            return 0

    async def refresh_facet_rollup(self, doc_id: str) -> None:
        import json as _json
        from datetime import date as _date

        from .facet_rollup import apply_rollup, rollup_from_metas

        rows = await scoped_query(
            "SELECT d.metadata FROM document_references r "
            "JOIN documents d ON d.id = r.target_document_id "
            "WHERE r.source_document_id = $1 AND r.reference_type = 'cites' "
            "AND d.path LIKE '/corpus/%'",
            doc_id,
        )
        metas = []
        for row in rows:
            raw = row["metadata"]
            try:
                parsed = _json.loads(raw) if isinstance(raw, str) else (raw or {})
            except ValueError:
                continue
            if isinstance(parsed, dict):
                metas.append(parsed)
        rollup = rollup_from_metas(metas, _date.today().isoformat())

        page = await scoped_queryrow("SELECT metadata FROM documents WHERE id = $1", doc_id)
        if page is None:
            return
        raw = page["metadata"]
        try:
            meta = _json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if apply_rollup(meta, rollup):
            await service_execute(
                "UPDATE documents SET metadata = $1::jsonb WHERE id = $2 AND user_id = $3",
                _json.dumps(meta, ensure_ascii=False), doc_id, self.user_id,
            )

    async def get_backlinks(self, doc_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT d.path, d.filename, d.title, dr.reference_type "
            "FROM document_references dr "
            "JOIN documents d ON dr.source_document_id = d.id "
            "WHERE dr.target_document_id = $1 AND NOT d.archived AND d.user_id = $2 "
            "ORDER BY d.path, d.filename",
            doc_id, self.user_id,
        )

    async def get_forward_references(self, doc_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT d.id, d.filename, d.title, d.path, dr.reference_type, dr.page "
            "FROM document_references dr "
            "JOIN documents d ON dr.target_document_id = d.id "
            "WHERE dr.source_document_id = $1 AND NOT d.archived AND d.user_id = $2 "
            "ORDER BY dr.reference_type, d.path, d.filename",
            doc_id, self.user_id,
        )

    async def find_uncited_sources(self, kb_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT d.filename, d.title, d.path, d.file_type "
            "FROM documents d "
            "WHERE d.knowledge_base_id = $1 AND NOT d.archived AND d.user_id = $2 "
            "  AND d.path NOT LIKE '/wiki/%' "
            "  AND d.id NOT IN (SELECT target_document_id FROM document_references WHERE reference_type = 'cites') "
            "ORDER BY d.filename",
            kb_id, self.user_id,
        )

    async def find_stale_pages(self, kb_id: str) -> list[dict]:
        return await scoped_query(
            self.user_id,
            "SELECT d.filename, d.title, d.path, d.stale_since "
            "FROM documents d "
            "WHERE d.knowledge_base_id = $1 AND NOT d.archived AND d.user_id = $2 "
            "  AND d.stale_since IS NOT NULL "
            "ORDER BY d.stale_since DESC",
            kb_id, self.user_id,
        )

    async def _insert_knowledge_base(self, name: str, description: str | None, kind: str = "wiki") -> dict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            current_name = name
            for attempt in range(10):
                slug = await self._unique_slug(current_name, conn)
                try:
                    row = await conn.fetchrow(
                        "INSERT INTO knowledge_bases (user_id, name, slug, description, kind) "
                        "VALUES ($1, $2, $3, $4, $5) "
                        "RETURNING id, user_id, name, slug, description, kind, created_at, updated_at",
                        self.user_id, current_name, slug, description, kind,
                    )
                    return dict(row)
                except asyncpg.UniqueViolationError:
                    current_name = f"{name} ({attempt + 2})"
        raise RuntimeError("Could not create knowledge base after too many duplicate names")

    async def _unique_slug(self, name: str, conn=None) -> str:
        base = _slugify(name)
        slug = base
        counter = 2

        if conn is not None:
            while await conn.fetchval(
                "SELECT 1 FROM knowledge_bases WHERE slug = $1 AND user_id = $2",
                slug, self.user_id,
            ):
                slug = f"{base}-{counter}"
                counter += 1
            return slug

        pool = await get_pool()
        async with pool.acquire() as acquired:
            return await self._unique_slug(name, acquired)

    async def _scaffold_wiki(self, kb_id: str, name: str) -> None:
        today = date.today().isoformat()
        await self.create_document(
            kb_id,
            "overview.md",
            "Overview",
            "/wiki/",
            "md",
            _OVERVIEW_TEMPLATE.format(name=name, date=today),
            ["overview", "wiki"],
            date=today,
            metadata={"description": f"Research hub for {name}."},
        )
        await self.create_document(
            kb_id,
            "log.md",
            "Log",
            "/wiki/",
            "md",
            _LOG_TEMPLATE.format(name=name, date=today),
            ["log"],
        )
