"""Transaction-scoped PostgreSQL persistence for one complete wiki revision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath
from uuid import UUID

import asyncpg

from llmwiki_core.chunking import chunk_text
from llmwiki_core.documents import normalize_directory_path
from llmwiki_core.facets import apply_rollup, rollup_from_metas
from llmwiki_core.references import ReferenceEdge
from llmwiki_core.wiki import (
    DuplicateDocumentError,
    VersionConflict,
    WikiWriteBundle,
)

_CONTENT_REFERENCE_TYPES = ("cites", "links_to")
_POSTGRES_INTEGER_MAX = 2_147_483_647
_MAX_JSON_DEPTH = 32
_MAX_METADATA_JSON_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class WikiWriteResult:
    """Authoritative identity and version persisted for a wiki revision."""

    document_id: UUID
    filename: str
    path: str
    version: int


def _require_uuid(value: UUID, field: str) -> None:
    if not isinstance(value, UUID):
        raise ValueError(f"{field} must be a UUID")


def _validate_postgres_text(value: str, field: str) -> None:
    if "\0" in value:
        raise ValueError(f"{field} contains NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be UTF-8 encodable") from exc


def _bundle_document_id(bundle: WikiWriteBundle) -> UUID:
    if type(bundle.document_id) is not str:
        raise ValueError("bundle document_id must be a UUID string")
    _validate_postgres_text(bundle.document_id, "bundle document_id")
    try:
        return UUID(bundle.document_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("bundle document_id must be a UUID string") from exc


def _target_id(edge: ReferenceEdge) -> UUID:
    if type(edge.target_id) is not str:
        raise ValueError("bundle edge target_id must be a UUID string")
    _validate_postgres_text(edge.target_id, "bundle edge target_id")
    try:
        return UUID(edge.target_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("bundle edge target_id must be a UUID string") from exc


def _validate_revision_fields(bundle: WikiWriteBundle) -> None:
    if bundle.expected_version is not None and (
        type(bundle.expected_version) is not int
        or not 1 <= bundle.expected_version < _POSTGRES_INTEGER_MAX
    ):
        raise ValueError("bundle expected_version must be a positive PostgreSQL integer or None")
    if (
        type(bundle.filename) is not str
        or not bundle.filename
        or bundle.filename in {".", ".."}
        or PurePosixPath(bundle.filename).name != bundle.filename
    ):
        raise ValueError("bundle filename must be a basename")
    _validate_postgres_text(bundle.filename, "bundle filename")
    if type(bundle.path) is not str:
        raise ValueError("bundle path must be a string")
    _validate_postgres_text(bundle.path, "bundle path")
    if type(bundle.file_type) is not str or not bundle.file_type:
        raise ValueError("bundle file_type must be a non-empty string")
    _validate_postgres_text(bundle.file_type, "bundle file_type")
    if type(bundle.content) is not str:
        raise ValueError("bundle content must be a string")
    _validate_postgres_text(bundle.content, "bundle content")
    if bundle.title is not None and type(bundle.title) is not str:
        raise ValueError("bundle title must be a string or None")
    if bundle.title is not None:
        _validate_postgres_text(bundle.title, "bundle title")
    if type(bundle.tags) is not tuple or not all(type(tag) is str for tag in bundle.tags):
        raise ValueError("bundle tags must be a tuple of strings")
    for tag in bundle.tags:
        _validate_postgres_text(tag, "bundle tag")
    if bundle.date is not None and type(bundle.date) is not str:
        raise ValueError("bundle date must be a string or None")
    if bundle.date is not None:
        _validate_postgres_text(bundle.date, "bundle date")


def _validate_json_strings(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("bundle metadata exceeds the JSON nesting limit")
    if isinstance(value, str):
        _validate_postgres_text(value, "bundle metadata string")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("bundle metadata must be a string-keyed object")
            _validate_postgres_text(key, "bundle metadata key")
            _validate_json_strings(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_strings(item, depth=depth + 1)


def _encode_metadata(bundle: WikiWriteBundle) -> str:
    if type(bundle.metadata) is not dict or not all(
        type(key) is str for key in bundle.metadata
    ):
        raise ValueError("bundle metadata must be a string-keyed object")
    _validate_json_strings(bundle.metadata)
    try:
        metadata_json = json.dumps(
            bundle.metadata,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("bundle metadata must contain finite JSON values") from exc
    if len(metadata_json.encode("utf-8")) > _MAX_METADATA_JSON_BYTES:
        raise ValueError("bundle metadata exceeds its byte limit")
    return metadata_json


def _validate_edges(bundle: WikiWriteBundle) -> None:
    if type(bundle.edges) is not tuple:
        raise ValueError("bundle edges must be a tuple of ReferenceEdge values")
    for edge in bundle.edges:
        if type(edge) is not ReferenceEdge:
            raise ValueError("bundle edges must be a tuple of ReferenceEdge values")
        _target_id(edge)
        if type(edge.reference_type) is not str:
            raise ValueError("bundle edge reference_type must be a string")
        _validate_postgres_text(edge.reference_type, "bundle edge reference_type")
        if edge.reference_type not in _CONTENT_REFERENCE_TYPES:
            raise ValueError("bundle edge reference_type must be cites or links_to")
        if edge.page is not None and (
            type(edge.page) is not int or not 1 <= edge.page <= _POSTGRES_INTEGER_MAX
        ):
            raise ValueError("bundle edge page must be a positive PostgreSQL integer or None")


def _validate_bundle(bundle: WikiWriteBundle) -> tuple[UUID, str, str]:
    if type(bundle) is not WikiWriteBundle:
        raise ValueError("bundle must be a WikiWriteBundle")
    document_id = _bundle_document_id(bundle)
    _validate_revision_fields(bundle)
    path = normalize_directory_path(bundle.path)
    metadata_json = _encode_metadata(bundle)
    _validate_edges(bundle)
    return document_id, path, metadata_json


async def _replace_chunks(
    conn: asyncpg.Connection,
    *,
    document_id: UUID,
    user_id: UUID,
    knowledge_base_id: UUID,
    version: int,
    content: str,
) -> None:
    await conn.execute(
        "DELETE FROM document_chunks c USING documents d "
        "WHERE c.document_id=$1 AND c.user_id=$2 AND c.knowledge_base_id=$3 "
        "AND d.id=c.document_id AND d.user_id=$2 AND d.knowledge_base_id=$3",
        document_id,
        user_id,
        knowledge_base_id,
    )
    chunks = chunk_text(content)
    if not chunks:
        return
    await conn.executemany(
        "INSERT INTO document_chunks "
        "(document_id, user_id, knowledge_base_id, document_version, chunk_index, "
        "content, source_content, page, start_char, token_count, header_breadcrumb) "
        "SELECT $1, $2, $3, $4, $5, $6, $6, $7, $8, $9, $10 "
        "WHERE EXISTS (SELECT 1 FROM documents d WHERE d.id=$1 "
        "AND d.user_id=$2 AND d.knowledge_base_id=$3)",
        [
            (
                document_id,
                user_id,
                knowledge_base_id,
                version,
                chunk.index,
                chunk.content,
                chunk.page,
                chunk.start_char,
                chunk.token_count,
                chunk.header_breadcrumb,
            )
            for chunk in chunks
        ],
    )


async def _replace_content_references(
    conn: asyncpg.Connection,
    *,
    document_id: UUID,
    user_id: UUID,
    knowledge_base_id: UUID,
    edges: tuple[ReferenceEdge, ...],
) -> None:
    await conn.execute(
        "DELETE FROM document_references r USING documents source "
        "WHERE r.source_document_id=$1 AND r.knowledge_base_id=$2 "
        "AND r.reference_type=ANY($3::text[]) "
        "AND source.id=r.source_document_id AND source.user_id=$4 "
        "AND source.knowledge_base_id=$2",
        document_id,
        knowledge_base_id,
        list(_CONTENT_REFERENCE_TYPES),
        user_id,
    )
    for edge in edges:
        target_id = _target_id(edge)
        inserted = await conn.fetchval(
            "INSERT INTO document_references "
            "(source_document_id, target_document_id, knowledge_base_id, reference_type, page) "
            "SELECT source.id, target.id, $1, $2, $3 "
            "FROM documents source JOIN documents target ON target.id=$4 "
            "WHERE source.id=$5 AND source.user_id=$6 AND source.knowledge_base_id=$1 "
            "AND target.user_id=$6 AND target.knowledge_base_id=$1 "
            "RETURNING 1",
            knowledge_base_id,
            edge.reference_type,
            edge.page,
            target_id,
            document_id,
            user_id,
        )
        if inserted is None:
            raise PermissionError("wiki references must stay within the knowledge base")


async def _propagate_incoming_link_staleness(
    conn: asyncpg.Connection,
    *,
    document_id: UUID,
    user_id: UUID,
    knowledge_base_id: UUID,
) -> None:
    await conn.execute(
        "UPDATE documents stale SET stale_since=now() "
        "FROM document_references r, documents target "
        "WHERE r.target_document_id=$1 AND r.reference_type='links_to' "
        "AND r.knowledge_base_id=$2 AND stale.id=r.source_document_id "
        "AND stale.user_id=$3 AND stale.knowledge_base_id=$2 "
        "AND target.id=r.target_document_id AND target.user_id=$3 "
        "AND target.knowledge_base_id=$2 AND stale.stale_since IS NULL",
        document_id,
        knowledge_base_id,
        user_id,
    )


async def _refresh_citation_rollup(
    conn: asyncpg.Connection,
    *,
    document_id: UUID,
    user_id: UUID,
    knowledge_base_id: UUID,
    metadata: dict,
) -> None:
    rows = await conn.fetch(
        "SELECT target.metadata FROM document_references r "
        "JOIN documents source ON source.id=r.source_document_id "
        "JOIN documents target ON target.id=r.target_document_id "
        "WHERE r.source_document_id=$1 AND r.knowledge_base_id=$2 "
        "AND r.reference_type='cites' AND source.user_id=$3 "
        "AND source.knowledge_base_id=$2 AND target.user_id=$3 "
        "AND target.knowledge_base_id=$2 AND target.path LIKE '/corpus/%'",
        document_id,
        knowledge_base_id,
        user_id,
    )
    metas = []
    for item in rows:
        raw = item["metadata"]
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            continue
        if isinstance(parsed, dict):
            metas.append(parsed)
    rollup = rollup_from_metas(metas, date.today().isoformat())
    if apply_rollup(metadata, rollup):
        await conn.execute(
            "UPDATE documents SET metadata=$1::jsonb "
            "WHERE id=$2 AND user_id=$3 AND knowledge_base_id=$4",
            json.dumps(metadata, ensure_ascii=False, allow_nan=False),
            document_id,
            user_id,
            knowledge_base_id,
        )


async def write_wiki_bundle_in_transaction(
    conn: asyncpg.Connection,
    *,
    user_id: UUID,
    knowledge_base_id: UUID,
    bundle: WikiWriteBundle,
) -> WikiWriteResult:
    """Persist one wiki revision without owning the surrounding transaction."""
    if not conn.is_in_transaction():
        raise RuntimeError("wiki writer requires an explicit transaction")
    _require_uuid(user_id, "user_id")
    _require_uuid(knowledge_base_id, "knowledge_base_id")
    document_id, path, metadata_json = _validate_bundle(bundle)
    metadata = json.loads(metadata_json)
    version = 1 if bundle.expected_version is None else bundle.expected_version + 1

    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"wiki-write:{user_id}:{knowledge_base_id}",
    )

    try:
        if bundle.expected_version is None:
            logical_path = f"{user_id}:{knowledge_base_id}:{path}:{bundle.filename}"
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                logical_path,
            )
            duplicate = await conn.fetchval(
                "SELECT 1 FROM documents d JOIN knowledge_bases kb "
                "ON kb.id=d.knowledge_base_id AND kb.user_id=$1 "
                "WHERE d.user_id=$1 AND d.knowledge_base_id=$2 "
                "AND d.path=$3 AND d.filename=$4 AND NOT d.archived",
                user_id,
                knowledge_base_id,
                path,
                bundle.filename,
            )
            if duplicate:
                raise DuplicateDocumentError(path, bundle.filename)
            row = await conn.fetchrow(
                "INSERT INTO documents "
                "(id, knowledge_base_id, user_id, filename, title, path, source_kind, "
                "file_type, status, content, tags, date, metadata, version) "
                "SELECT $1, $2, $3, $4, $5, $6, 'wiki', $7, 'ready', $8, $9, $10, "
                "$11::jsonb, 1 FROM knowledge_bases kb WHERE kb.id=$2 AND kb.user_id=$3 "
                "RETURNING id, filename, path, version",
                document_id,
                knowledge_base_id,
                user_id,
                bundle.filename,
                bundle.title,
                path,
                bundle.file_type,
                bundle.content,
                list(bundle.tags),
                bundle.date,
                metadata_json,
            )
            if row is None:
                raise PermissionError("knowledge base is not owned by user")
        else:
            row = await conn.fetchrow(
                "UPDATE documents SET content=$1, title=$2, tags=$3, date=$4, "
                "metadata=$5::jsonb, version=$6, stale_since=NULL, updated_at=now() "
                "WHERE id=$7 AND knowledge_base_id=$8 AND user_id=$9 AND version=$10 "
                "RETURNING id, filename, path, version",
                bundle.content,
                bundle.title,
                list(bundle.tags),
                bundle.date,
                metadata_json,
                version,
                document_id,
                knowledge_base_id,
                user_id,
                bundle.expected_version,
            )
            if row is None:
                raise VersionConflict(
                    f"document {bundle.document_id} is not at version "
                    f"{bundle.expected_version}"
                )
    except asyncpg.UniqueViolationError as exc:
        if exc.constraint_name == "idx_documents_unique_active":
            raise DuplicateDocumentError(path, bundle.filename) from exc
        raise

    await _replace_chunks(
        conn,
        document_id=document_id,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        version=version,
        content=bundle.content,
    )
    await _replace_content_references(
        conn,
        document_id=document_id,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        edges=bundle.edges,
    )
    await _propagate_incoming_link_staleness(
        conn,
        document_id=document_id,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )
    await _refresh_citation_rollup(
        conn,
        document_id=document_id,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        metadata=metadata,
    )
    persisted = await conn.fetchrow(
        "SELECT id, filename, path, version FROM documents "
        "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
        document_id,
        user_id,
        knowledge_base_id,
    )
    if persisted is None:
        raise RuntimeError("persisted wiki document could not be read")
    return WikiWriteResult(
        document_id=persisted["id"],
        filename=persisted["filename"],
        path=persisted["path"],
        version=persisted["version"],
    )
