"""Bounded Postgres read models for Hosted mode."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from domain.file_types import EXTRACTION_TYPES, IMAGE_TYPES, SIMPLE_TEXT_TYPES

from llmwiki_core.facets import postgres_facet_conditions, validate_facets
from llmwiki_core.read_cursor import ReadCursor, decode_cursor, encode_cursor

from .corpus_summary import build_summary
from .read_models import (
    BrowsePage,
    CorpusSummary,
    DocumentStatus,
    DocumentStatusPage,
    FolderItem,
    GraphSummary,
    ReadPage,
    ResolvedDocument,
    StaleReadCursor,
    UploadPreflightItem,
    UploadPreflightResponse,
)

_PROJECTION = (
    "id::text, filename, title, path, file_type, status::text, file_size, "
    "page_count, tags, date, metadata, error_message, version, document_number, "
    "sort_order, archived, created_at::text, updated_at::text, "
    "knowledge_base_id::text"
)
_SUPPORTED_UPLOAD_TYPES = SIMPLE_TEXT_TYPES | EXTRACTION_TYPES | IMAGE_TYPES
_MAX_UPLOAD_BYTES = 1_073_741_824
_SUMMARY_CACHE: OrderedDict[tuple, CorpusSummary] = OrderedDict()


def _document(row) -> ResolvedDocument:
    payload = dict(row)
    if isinstance(payload.get("metadata"), str):
        try:
            payload["metadata"] = json.loads(payload["metadata"])
        except json.JSONDecodeError:
            payload["metadata"] = None
    return ResolvedDocument.model_validate(payload)


class HostedReadService:
    def __init__(self, database, user_id: str):
        self.database = database
        self.user_id = user_id

    async def revision(self, kb_id: str) -> int:
        value = await self.database.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1 AND user_id=$2",
            kb_id,
            self.user_id,
        )
        if value is None:
            raise LookupError("knowledge base not found")
        return int(value)

    async def browse(
        self,
        kb_id: str,
        *,
        path: str | None,
        query: str | None,
        sort: str,
        direction: str,
        limit: int,
        cursor: str | None,
    ) -> BrowsePage:
        revision = await self.revision(kb_id)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        query = query.strip() if query is not None else None
        if path is None and not query:
            raise ValueError("query is required for workspace-wide search")
        sorts = {
            "name": "lower(filename)",
            "date": "updated_at",
            "type": "lower(file_type)",
            "path": "path",
        }
        if sort not in sorts or direction not in {"asc", "desc"}:
            raise ValueError("invalid sort")
        sort_sql = sorts[sort]
        conditions = ["knowledge_base_id=$1", "user_id=$2", "NOT archived"]
        params: list[Any] = [kb_id, self.user_id]
        if path is not None:
            params.append(path)
            conditions.append(f"path=${len(params)}")
        else:
            conditions.append("source_kind='source'")
        if query:
            params.append(f"%{query}%")
            conditions.append(f"filename ILIKE ${len(params)}")
        count_conditions = list(conditions)
        count_params = list(params)
        if cursor:
            decoded = decode_cursor(cursor, expected_scope="documents.browse")
            if decoded.revision != revision:
                raise StaleReadCursor(revision)
            if decoded.sort != sort or decoded.direction != direction:
                raise ValueError("cursor sort does not match request")
            params.extend(decoded.key)
            operator = "<" if direction == "desc" else ">"
            conditions.append(f"({sort_sql},id::text) {operator} (${len(params) - 1},${len(params)})")
        params.append(limit + 1)
        order = "DESC" if direction == "desc" else "ASC"
        rows = await self.database.fetch(
            f"SELECT {_PROJECTION}, {sort_sql}::text AS _sort_key FROM documents "
            f"WHERE {' AND '.join(conditions)} ORDER BY {sort_sql} {order}, id {order} "
            f"LIMIT ${len(params)}",
            *params,
        )
        has_next = len(rows) > limit
        rows = rows[:limit]
        items = [_document(row) for row in rows]
        next_cursor = None
        if has_next and rows:
            last = rows[-1]
            next_cursor = encode_cursor(
                ReadCursor(
                    "documents.browse",
                    revision,
                    sort,
                    direction,
                    (str(last["_sort_key"] or ""), str(last["id"])),
                )
            )
        total = await self.database.fetchval(
            f"SELECT count(*) FROM documents WHERE {' AND '.join(count_conditions)}",
            *count_params,
        )
        folders = await self._folders(kb_id, path) if path is not None and cursor is None else []
        return BrowsePage(
            revision=revision,
            items=items,
            next_cursor=next_cursor,
            total_count=int(total or 0),
            folders=folders,
        )

    async def _folders(self, kb_id: str, path: str) -> list[FolderItem]:
        prefix = path if path.endswith("/") else f"{path}/"
        rows = await self.database.fetch(
            "SELECT $3 || split_part(substring(path FROM char_length($3) + 1), '/', 1) || '/' "
            "AS child, count(*) AS document_count FROM documents "
            "WHERE knowledge_base_id=$1 AND user_id=$2 AND NOT archived "
            "AND path LIKE $3 || '%' AND path != $3 "
            "GROUP BY child ORDER BY child LIMIT 200",
            kb_id,
            self.user_id,
            prefix,
        )
        return [
            FolderItem(
                name=row["child"].rstrip("/").rsplit("/", 1)[-1],
                path=row["child"],
                document_count=row["document_count"],
            )
            for row in rows
        ]

    async def wiki_pages(self, kb_id: str, *, limit: int, cursor: str | None) -> BrowsePage:
        revision = await self.revision(kb_id)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        params: list[Any] = [kb_id, self.user_id]
        cursor_condition = ""
        if cursor:
            decoded = decode_cursor(cursor, expected_scope="wiki.pages")
            if decoded.revision != revision:
                raise StaleReadCursor(revision)
            params.extend(decoded.key)
            cursor_condition = "AND (path,filename,id::text)>($3,$4,$5)"
        params.append(limit + 1)
        rows = await self.database.fetch(
            f"SELECT {_PROJECTION} FROM documents WHERE knowledge_base_id=$1 AND user_id=$2 "
            f"AND NOT archived AND source_kind='wiki' {cursor_condition} "
            f"ORDER BY path,filename,id LIMIT ${len(params)}",
            *params,
        )
        has_next = len(rows) > limit
        rows = rows[:limit]
        next_cursor = None
        if has_next and rows:
            last = rows[-1]
            next_cursor = encode_cursor(
                ReadCursor(
                    "wiki.pages",
                    revision,
                    "path",
                    "asc",
                    (last["path"], last["filename"], str(last["id"])),
                )
            )
        total = await self.database.fetchval(
            "SELECT count(*) FROM documents WHERE knowledge_base_id=$1 AND user_id=$2 "
            "AND NOT archived AND source_kind='wiki'",
            kb_id,
            self.user_id,
        )
        return BrowsePage(
            revision=revision,
            items=[_document(row) for row in rows],
            next_cursor=next_cursor,
            total_count=int(total or 0),
            folders=[],
        )

    async def resolve(
        self,
        kb_id: str,
        *,
        document_number: int | None = None,
        logical_reference: str | None = None,
    ) -> ResolvedDocument | None:
        await self.revision(kb_id)
        if (document_number is None) == (logical_reference is None):
            raise ValueError("provide exactly one resolver key")
        if document_number is not None:
            extra, value = "document_number=$3", document_number
        else:
            reference = (logical_reference or "").strip().lstrip("/")
            extra, value = (
                "(filename ILIKE $3 OR title ILIKE $3 OR ltrim(path,'/') || filename=$3)",
                reference,
            )
        row = await self.database.fetchrow(
            f"SELECT {_PROJECTION} FROM documents WHERE knowledge_base_id=$1 AND user_id=$2 "
            f"AND NOT archived AND {extra} ORDER BY id LIMIT 1",
            kb_id,
            self.user_id,
            value,
        )
        return _document(row) if row else None

    async def statuses(
        self,
        kb_id: str,
        *,
        ids: Sequence[str] | None = None,
        document_numbers: Sequence[int] | None = None,
    ) -> DocumentStatusPage:
        revision = await self.revision(kb_id)
        values = list(ids or document_numbers or [])
        if not values or len(values) > 200 or bool(ids) == bool(document_numbers):
            raise ValueError("provide 1 to 200 ids or document numbers")
        field = "id::text" if ids else "document_number"
        rows = await self.database.fetch(
            f"SELECT id::text,document_number,status::text,error_message,version FROM documents "
            f"WHERE knowledge_base_id=$1 AND user_id=$2 AND NOT archived AND {field}=ANY($3)",
            kb_id,
            self.user_id,
            values,
        )
        found = {row["id" if ids else "document_number"]: row for row in rows}
        return DocumentStatusPage(
            revision=revision,
            items=[DocumentStatus.model_validate(dict(found[value])) for value in values if value in found],
        )

    async def upload_preflight(self, kb_id: str, descriptors: Sequence[dict[str, Any]]) -> UploadPreflightResponse:
        revision = await self.revision(kb_id)
        if not 1 <= len(descriptors) <= 200:
            raise ValueError("upload preflight accepts 1 to 200 items")
        items = [UploadPreflightItem.model_validate(item) for item in descriptors]
        rows = await self.database.fetch(
            "SELECT id::text,path,lower(filename) AS filename FROM documents "
            "WHERE knowledge_base_id=$1 AND user_id=$2 AND NOT archived "
            "AND (path,lower(filename)) IN (SELECT * FROM unnest($3::text[],$4::text[]))",
            kb_id,
            self.user_id,
            [item.path for item in items],
            [item.filename.lower() for item in items],
        )
        names = {(row["path"], row["filename"]): row["id"] for row in rows}
        decisions = []
        for item in items:
            payload = item.model_dump()
            existing = names.get((item.path, item.filename.lower()))
            extension = item.filename.rsplit(".", 1)[-1].lower() if "." in item.filename else ""
            if extension not in _SUPPORTED_UPLOAD_TYPES:
                payload.update(accepted=False, code="unsupported")
            elif item.size > _MAX_UPLOAD_BYTES:
                payload.update(accepted=False, code="too_large")
            elif existing:
                payload.update(accepted=False, code="duplicate_name", existing_document_id=existing)
            else:
                payload.update(accepted=True, code="accepted")
            decisions.append(UploadPreflightItem.model_validate(payload))
        return UploadPreflightResponse(revision=revision, items=decisions)

    async def _corpus_rows(self, kb_id: str, query: str | None = None) -> list[dict[str, Any]]:
        conditions = [
            "knowledge_base_id=$1",
            "user_id=$2",
            "NOT archived",
            "source_kind='source'",
            "status!='failed'",
            "jsonb_typeof(metadata)='object'",
            "metadata ? 'spec_version'",
        ]
        params: list[Any] = [kb_id, self.user_id]
        if query:
            params.append(f"%{query}%")
            conditions.append(
                f"(filename ILIKE ${len(params)} OR COALESCE(title,'') ILIKE ${len(params)} "
                f"OR metadata->>'entry_id' ILIKE ${len(params)})"
            )
        rows = await self.database.fetch(
            f"SELECT id::text,filename,title,path,metadata FROM documents WHERE {' AND '.join(conditions)}",
            *params,
        )
        result = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    continue
            if isinstance(metadata, dict) and isinstance(metadata.get("stage"), str):
                result.append(
                    {
                        "id": row["id"],
                        "filename": row["filename"],
                        "title": row["title"],
                        "path": row["path"],
                        "metadata": metadata,
                    }
                )
        return result

    async def corpus_entries(
        self,
        kb_id: str,
        filters: dict[str, str] | None,
        *,
        query: str | None = None,
        sort: str = "name",
        direction: str = "asc",
        limit: int,
        cursor: str | None,
    ) -> ReadPage:
        revision = await self.revision(kb_id)
        clean = validate_facets(filters)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        query = query.strip() if query else None
        sorts = {
            "name": "lower(COALESCE(NULLIF(d.title,''),d.filename))",
            "stage": "COALESCE(d.metadata->>'stage','')",
            "domain": "COALESCE(d.metadata->>'domain','')",
            "review_due": "COALESCE(d.metadata->>'review_due','')",
            "updated": "d.updated_at::text",
        }
        if sort not in sorts or direction not in {"asc", "desc"}:
            raise ValueError("invalid sort")
        sort_sql = sorts[sort]
        conditions = [
            "d.knowledge_base_id=$1",
            "d.user_id=$2",
            "NOT d.archived",
            "d.source_kind='source'",
            "d.status!='failed'",
            "jsonb_typeof(d.metadata)='object'",
            "jsonb_typeof(d.metadata->'spec_version')='string'",
            "jsonb_typeof(d.metadata->'stage')='string'",
        ]
        params: list[Any] = [kb_id, self.user_id]
        facet_conditions, facet_params = postgres_facet_conditions(clean, 3, "d")
        conditions.extend(facet_conditions)
        params.extend(facet_params)
        if query:
            params.append(f"%{query}%")
            conditions.append(
                f"(d.filename ILIKE ${len(params)} OR COALESCE(d.title,'') ILIKE ${len(params)} "
                f"OR d.metadata->>'entry_id' ILIKE ${len(params)})"
            )
        count_conditions = list(conditions)
        count_params = list(params)
        if cursor:
            decoded = decode_cursor(cursor, expected_scope="corpus.entries")
            if decoded.revision != revision:
                raise StaleReadCursor(revision)
            if decoded.sort != sort or decoded.direction != direction:
                raise ValueError("cursor sort does not match request")
            if len(decoded.key) != 2:
                raise ValueError("invalid corpus cursor key")
            params.extend(decoded.key)
            operator = "<" if direction == "desc" else ">"
            conditions.append(f"({sort_sql},d.id::text) {operator} (${len(params) - 1},${len(params)})")
        params.append(limit + 1)
        order = "DESC" if direction == "desc" else "ASC"
        rows = await self.database.fetch(
            "SELECT d.id::text,d.filename,d.title,d.path,d.metadata,"
            + sort_sql
            + " AS _sort_key FROM documents d WHERE "
            + " AND ".join(conditions)
            + f" ORDER BY {sort_sql} {order},d.id {order} LIMIT ${len(params)}",
            *params,
        )
        has_next = len(rows) > limit
        rows = rows[:limit]
        page = [
            {
                "id": row["id"],
                "filename": row["filename"],
                "title": row["title"],
                "path": row["path"],
                "metadata": row["metadata"],
            }
            for row in rows
        ]
        next_cursor = (
            encode_cursor(
                ReadCursor(
                    "corpus.entries",
                    revision,
                    sort,
                    direction,
                    (str(rows[-1]["_sort_key"] or ""), str(rows[-1]["id"])),
                )
            )
            if has_next and rows
            else None
        )
        total = await self.database.fetchval(
            "SELECT count(*) FROM documents d WHERE " + " AND ".join(count_conditions),
            *count_params,
        )
        return ReadPage(
            revision=revision,
            items=page,
            next_cursor=next_cursor,
            total_count=int(total or 0),
        )

    async def corpus_summary(
        self, kb_id: str, filters: dict[str, str] | None, query: str | None = None
    ) -> CorpusSummary:
        revision = await self.revision(kb_id)
        clean = validate_facets(filters)
        query = query.strip() if query else None
        key = (self.user_id, kb_id, tuple(sorted(clean.items())), query, revision)
        cached = _SUMMARY_CACHE.get(key)
        if cached is not None:
            _SUMMARY_CACHE.move_to_end(key)
            return cached
        summary = build_summary(await self._corpus_rows(kb_id, query), clean, revision)
        cited, wiki_covered, entry_cells = await self._summary_graph_kpis(kb_id, clean, query)
        summary.kpis.update(
            cited=cited,
            wiki_covered=wiki_covered,
            wiki_cells_with_entries=entry_cells,
        )
        _SUMMARY_CACHE[key] = summary
        _SUMMARY_CACHE.move_to_end(key)
        while len(_SUMMARY_CACHE) > 256:
            _SUMMARY_CACHE.popitem(last=False)
        return summary

    async def _summary_graph_kpis(self, kb_id: str, filters: dict[str, str], query: str | None) -> tuple[int, int, int]:
        conditions = [
            "t.knowledge_base_id=$1",
            "t.user_id=$2",
            "NOT t.archived",
            "t.source_kind='source'",
            "t.status!='failed'",
            "jsonb_typeof(t.metadata)='object'",
            "jsonb_typeof(t.metadata->'spec_version')='string'",
            "jsonb_typeof(t.metadata->'stage')='string'",
        ]
        params: list[Any] = [kb_id, self.user_id]
        facet_conditions, facet_params = postgres_facet_conditions(filters, 3, "t")
        conditions.extend(facet_conditions)
        params.extend(facet_params)
        if query:
            params.append(f"%{query}%")
            conditions.append(
                f"(t.filename ILIKE ${len(params)} OR COALESCE(t.title,'') ILIKE ${len(params)} "
                f"OR t.metadata->>'entry_id' ILIKE ${len(params)})"
            )
        where = " AND ".join(conditions)
        cited = await self.database.fetchval(
            "SELECT count(DISTINCT t.id) FROM documents t "
            "JOIN document_references r ON r.target_document_id=t.id "
            "JOIN documents s ON s.id=r.source_document_id "
            f"WHERE {where} AND r.knowledge_base_id=$1 "
            "AND s.knowledge_base_id=$1 AND s.user_id=$2 AND NOT s.archived "
            "AND s.source_kind='wiki' AND r.reference_type IN ('cites','links_to')",
            *params,
        )
        coverage = await self.database.fetchrow(
            "WITH entry_cells AS ("
            "SELECT DISTINCT t.metadata->>'stage' AS stage,left(t.metadata->>'domain',1) AS layer "
            f"FROM documents t WHERE {where} "
            "AND left(t.metadata->>'domain',1) IN ('G','C','O','Z')"
            "),wiki_cells AS ("
            "SELECT DISTINCT ws.stage,left(wd.domain,1) AS layer FROM documents w "
            "CROSS JOIN LATERAL jsonb_array_elements_text("
            "CASE WHEN jsonb_typeof(w.metadata#>'{facet_rollup,stage}')='array' "
            "THEN w.metadata#>'{facet_rollup,stage}' ELSE '[]'::jsonb END) ws(stage) "
            "CROSS JOIN LATERAL jsonb_array_elements_text("
            "CASE WHEN jsonb_typeof(w.metadata#>'{facet_rollup,domain}')='array' "
            "THEN w.metadata#>'{facet_rollup,domain}' ELSE '[]'::jsonb END) wd(domain) "
            "WHERE w.knowledge_base_id=$1 AND w.user_id=$2 AND NOT w.archived "
            "AND w.source_kind='wiki'"
            ") SELECT (SELECT count(*) FROM entry_cells) AS entry_cells,"
            "(SELECT count(*) FROM entry_cells e JOIN wiki_cells w "
            "ON w.stage=e.stage AND w.layer=e.layer) AS wiki_covered",
            *params,
        )
        return (
            int(cited or 0),
            int(coverage["wiki_covered"] or 0),
            int(coverage["entry_cells"] or 0),
        )

    async def graph_summary(self, kb_id: str) -> GraphSummary:
        revision = await self.revision(kb_id)
        node_count = await self.database.fetchval(
            "SELECT count(*) FROM documents WHERE knowledge_base_id=$1 AND user_id=$2 AND NOT archived",
            kb_id,
            self.user_id,
        )
        rows = await self.database.fetch(
            "SELECT DISTINCT r.target_document_id::text AS id FROM document_references r "
            "JOIN documents s ON s.id=r.source_document_id JOIN documents t ON t.id=r.target_document_id "
            "WHERE r.knowledge_base_id=$1 AND s.knowledge_base_id=$1 AND t.knowledge_base_id=$1 "
            "AND s.user_id=$2 AND t.user_id=$2 "
            "AND r.reference_type IN ('cites','links_to') ORDER BY id LIMIT 200",
            kb_id,
            self.user_id,
        )
        edge_count = await self.database.fetchval(
            "SELECT count(*) FROM document_references r JOIN documents s ON s.id=r.source_document_id "
            "JOIN documents t ON t.id=r.target_document_id WHERE r.knowledge_base_id=$1 "
            "AND s.knowledge_base_id=$1 AND t.knowledge_base_id=$1 AND s.user_id=$2 AND t.user_id=$2",
            kb_id,
            self.user_id,
        )
        return GraphSummary(
            revision=revision,
            node_count=int(node_count or 0),
            edge_count=int(edge_count or 0),
            cited_document_ids=[row["id"] for row in rows],
        )


__all__ = ["HostedReadService"]
