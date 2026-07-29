"""Bounded SQLite read models for Local mode."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import aiosqlite
from domain.file_types import EXTRACTION_TYPES, IMAGE_TYPES, SIMPLE_TEXT_TYPES

from llmwiki_core.facets import sqlite_facet_conditions, validate_facets
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
    "id, filename, title, path, file_type, status, file_size, page_count, tags, "
    "date, metadata, error_message, version, document_number, stale_since, "
    "created_at, updated_at"
)
_SUPPORTED_UPLOAD_TYPES = SIMPLE_TEXT_TYPES | EXTRACTION_TYPES | IMAGE_TYPES
_MAX_UPLOAD_BYTES = 1_073_741_824
_SUMMARY_CACHE: OrderedDict[tuple, CorpusSummary] = OrderedDict()
_BROWSE_SUMMARY_CACHE: OrderedDict[tuple[str, str, int], tuple[int, int, int]] = OrderedDict()
_FOLDER_CACHE: OrderedDict[tuple[str, str, int, str], tuple[FolderItem, ...]] = OrderedDict()


def _remember(cache: OrderedDict, key: tuple, value: Any) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > 256:
        cache.popitem(last=False)
    return value


def _row_dict(cursor: aiosqlite.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    result = dict(zip((column[0] for column in cursor.description), row, strict=True))
    for field, fallback in (("tags", []), ("metadata", None)):
        value = result.get(field)
        if isinstance(value, str):
            try:
                result[field] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                result[field] = fallback
    result["archived"] = False
    return result


class LocalReadService:
    def __init__(self, db: aiosqlite.Connection, user_id: str):
        self.db = db
        self.user_id = user_id

    async def revision(self, kb_id: str) -> int:
        cursor = await self.db.execute(
            "SELECT read_revision FROM workspace WHERE id=? AND user_id=?",
            (kb_id, self.user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise LookupError("knowledge base not found")
        return int(row[0])

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
        sort_sql, key_columns = self._sort(sort)
        descending = direction == "desc"
        if direction not in {"asc", "desc"}:
            raise ValueError("invalid direction")

        conditions = ["source_kind = 'source'"]
        params: list[Any] = []
        if path is not None:
            conditions.append("path = ?")
            params.append(path)
        if query:
            conditions.append("filename LIKE ? ESCAPE '\\'")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        count_conditions = list(conditions)
        count_params = list(params)

        if cursor:
            decoded = decode_cursor(cursor, expected_scope="documents.browse")
            if decoded.revision != revision:
                raise StaleReadCursor(revision)
            if decoded.sort != sort or decoded.direction != direction:
                raise ValueError("cursor sort does not match request")
            operator = "<" if descending else ">"
            conditions.append(f"({key_columns}) {operator} ({','.join('?' for _ in decoded.key)})")
            params.extend(decoded.key)

        order = "DESC" if descending else "ASC"
        sql = (
            f"SELECT {_PROJECTION}, {sort_sql} AS _sort_key FROM documents "
            f"WHERE {' AND '.join(conditions)} "
            f"ORDER BY {sort_sql} {order}, id {order} LIMIT ?"
        )
        params.append(limit + 1)
        db_cursor = await self.db.execute(sql, params)
        rows = await db_cursor.fetchall()
        has_next = len(rows) > limit
        rows = rows[:limit]
        items = [ResolvedDocument.model_validate(_row_dict(db_cursor, row)) for row in rows]
        next_cursor = None
        if has_next and rows:
            last = _row_dict(db_cursor, rows[-1])
            next_cursor = encode_cursor(
                ReadCursor(
                    "documents.browse",
                    revision,
                    sort,
                    direction,
                    (str(last["_sort_key"] or ""), str(last["id"])),
                )
            )
        count_cursor = await self.db.execute(
            f"SELECT count(*) FROM documents WHERE {' AND '.join(count_conditions)}",
            count_params,
        )
        count_row = await count_cursor.fetchone()
        summary_key = (self.user_id, kb_id, revision)
        summary = _BROWSE_SUMMARY_CACHE.get(summary_key)
        if summary is None:
            summary_cursor = await self.db.execute(
                "SELECT count(*) FILTER (WHERE source_kind='source' AND path NOT LIKE '/corpus/%'),"
                "count(*) FILTER (WHERE source_kind='source' AND path NOT LIKE '/corpus/%' AND status='failed'),"
                "count(*) FILTER (WHERE source_kind='source' AND json_extract("
                "CASE WHEN typeof(metadata)='text' AND json_valid(metadata) "
                "THEN metadata ELSE '{}' END,'$.spec_version') IS NOT NULL) "
                "FROM documents WHERE user_id=?",
                (self.user_id,),
            )
            summary_row = await summary_cursor.fetchone()
            summary = _remember(
                _BROWSE_SUMMARY_CACHE,
                summary_key,
                tuple(int(value or 0) for value in (summary_row or (0, 0, 0))),
            )
        else:
            _BROWSE_SUMMARY_CACHE.move_to_end(summary_key)
        folders: list[FolderItem] = []
        if path is not None and cursor is None:
            folder_key = (self.user_id, kb_id, revision, path)
            cached_folders = _FOLDER_CACHE.get(folder_key)
            if cached_folders is None:
                cached_folders = _remember(_FOLDER_CACHE, folder_key, tuple(await self._folders(path)))
            else:
                _FOLDER_CACHE.move_to_end(folder_key)
            folders = list(cached_folders)
        return BrowsePage(
            revision=revision,
            items=items,
            next_cursor=next_cursor,
            total_count=int(count_row[0]) if count_row else 0,
            folders=folders,
            source_count=summary[0],
            failed_count=summary[1],
            corpus_count=summary[2],
        )

    @staticmethod
    def _sort(sort: str) -> tuple[str, str]:
        values = {
            "name": ("filename COLLATE NOCASE", "filename COLLATE NOCASE,id"),
            "date": ("updated_at", "updated_at,id"),
            "type": ("file_type COLLATE NOCASE", "file_type COLLATE NOCASE,id"),
            "path": ("path", "path,id"),
        }
        try:
            return values[sort]
        except KeyError:
            raise ValueError("invalid sort") from None

    async def _folders(self, path: str) -> list[FolderItem]:
        prefix = path if path.endswith("/") else f"{path}/"
        cursor = await self.db.execute(
            "SELECT ? || substr(remainder,1,instr(remainder,'/')) AS child, count(*) "
            "FROM (SELECT substr(path, ?) AS remainder FROM documents "
            "WHERE source_kind = 'source' AND path LIKE ? AND path != ?) "
            "WHERE instr(remainder,'/') > 0 GROUP BY child ORDER BY child LIMIT 200",
            (prefix, len(prefix) + 1, f"{prefix}%", prefix),
        )
        return [
            FolderItem(name=child.rstrip("/").rsplit("/", 1)[-1], path=child, document_count=count)
            for child, count in await cursor.fetchall()
        ]

    async def wiki_pages(self, kb_id: str, *, limit: int, cursor: str | None) -> BrowsePage:
        revision = await self.revision(kb_id)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        params: list[Any] = []
        condition = "source_kind='wiki'"
        if cursor:
            decoded = decode_cursor(cursor, expected_scope="wiki.pages")
            if decoded.revision != revision:
                raise StaleReadCursor(revision)
            condition += " AND (path,filename,id) > (?,?,?)"
            params.extend(decoded.key)
        params.append(limit + 1)
        db_cursor = await self.db.execute(
            f"SELECT {_PROJECTION} FROM documents WHERE {condition} ORDER BY path,filename,id LIMIT ?",
            params,
        )
        rows = await db_cursor.fetchall()
        has_next = len(rows) > limit
        rows = rows[:limit]
        items = [ResolvedDocument.model_validate(_row_dict(db_cursor, row)) for row in rows]
        next_cursor = None
        if has_next and rows:
            last = _row_dict(db_cursor, rows[-1])
            next_cursor = encode_cursor(
                ReadCursor(
                    "wiki.pages",
                    revision,
                    "path",
                    "asc",
                    (last["path"], last["filename"], last["id"]),
                )
            )
        return BrowsePage(
            revision=revision,
            items=items,
            next_cursor=next_cursor,
            total_count=len(items),
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
            where, params = "document_number=?", [document_number]
        else:
            reference = (logical_reference or "").strip().lstrip("/")
            where = "relative_path=? OR filename=? COLLATE NOCASE OR title=? COLLATE NOCASE"
            params = [reference, reference.rsplit("/", 1)[-1], reference]
        cursor = await self.db.execute(f"SELECT {_PROJECTION} FROM documents WHERE {where} ORDER BY id LIMIT 1", params)
        row = await cursor.fetchone()
        return ResolvedDocument.model_validate(_row_dict(cursor, row)) if row else None

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
        field = "id" if ids else "document_number"
        cursor = await self.db.execute(
            f"SELECT id,document_number,status,error_message,version FROM documents "
            f"WHERE {field} IN ({','.join('?' for _ in values)})",
            values,
        )
        found = {row[0 if ids else 1]: row for row in await cursor.fetchall()}
        items = [
            DocumentStatus(
                id=row[0],
                document_number=row[1],
                status=row[2],
                error_message=row[3],
                version=row[4],
            )
            for value in values
            if (row := found.get(value)) is not None
        ]
        return DocumentStatusPage(revision=revision, items=items)

    async def upload_preflight(self, kb_id: str, descriptors: Sequence[dict[str, Any]]) -> UploadPreflightResponse:
        revision = await self.revision(kb_id)
        if not 1 <= len(descriptors) <= 200:
            raise ValueError("upload preflight accepts 1 to 200 items")
        items = [UploadPreflightItem.model_validate(item) for item in descriptors]
        name_conditions = " OR ".join("(path=? AND lower(filename)=lower(?))" for _ in items)
        name_params = [value for item in items for value in (item.path, item.filename)]
        cursor = await self.db.execute(
            f"SELECT id,path,lower(filename) FROM documents WHERE {name_conditions}", name_params
        )
        names = {(row[1], row[2]): row[0] for row in await cursor.fetchall()}
        hashes = [item.sha256 for item in items if item.sha256]
        hash_rows: dict[str, str] = {}
        if hashes:
            cursor = await self.db.execute(
                f"SELECT id,content_hash FROM documents WHERE status!='failed' AND content_hash IN "
                f"({','.join('?' for _ in hashes)})",
                hashes,
            )
            hash_rows = {row[1]: row[0] for row in await cursor.fetchall()}
        decisions: list[UploadPreflightItem] = []
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
            elif item.sha256 and item.sha256 in hash_rows:
                payload.update(
                    accepted=False,
                    code="duplicate_content",
                    existing_document_id=hash_rows[item.sha256],
                )
            else:
                payload.update(accepted=True, code="accepted")
            decisions.append(UploadPreflightItem.model_validate(payload))
        return UploadPreflightResponse(revision=revision, items=decisions)

    async def _corpus_rows(self, query: str | None = None) -> list[dict[str, Any]]:
        conditions = [
            "user_id=?",
            "source_kind='source'",
            "status!='failed'",
            "json_valid(metadata)",
            "json_type(metadata)='object'",
            "json_extract(metadata,'$.spec_version') IS NOT NULL",
        ]
        params: list[Any] = [self.user_id]
        if query:
            conditions.append(
                "(filename LIKE ? OR COALESCE(title,'') LIKE ? OR json_extract(metadata,'$.entry_id') LIKE ?)"
            )
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern])
        cursor = await self.db.execute(
            f"SELECT id,filename,title,path,metadata FROM documents WHERE {' AND '.join(conditions)}",
            params,
        )
        rows = []
        for row in await cursor.fetchall():
            try:
                metadata = json.loads(row[4])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(metadata, dict) and isinstance(metadata.get("stage"), str):
                rows.append({"id": row[0], "filename": row[1], "title": row[2], "path": row[3], "metadata": metadata})
        return rows

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
        safe_meta = "CASE WHEN typeof(d.metadata)='text' AND json_valid(d.metadata) THEN d.metadata ELSE '{}' END"
        sorts = {
            "name": "lower(COALESCE(NULLIF(d.title,''),d.filename))",
            "stage": f"COALESCE(json_extract({safe_meta},'$.stage'),'')",
            "domain": f"COALESCE(json_extract({safe_meta},'$.domain'),'')",
            "review_due": f"COALESCE(json_extract({safe_meta},'$.review_due'),'')",
            "updated": "d.updated_at",
        }
        if sort not in sorts or direction not in {"asc", "desc"}:
            raise ValueError("invalid sort")
        sort_sql = sorts[sort]
        conditions = [
            "d.user_id=?",
            "d.source_kind='source'",
            "d.status!='failed'",
            "json_type(" + safe_meta + ")='object'",
            "json_type(" + safe_meta + ",'$.spec_version')='text'",
            "json_type(" + safe_meta + ",'$.stage')='text'",
        ]
        params: list[Any] = [self.user_id]
        facet_conditions, facet_params = sqlite_facet_conditions(clean, "d")
        conditions.extend(facet_conditions)
        params.extend(facet_params)
        if query:
            pattern = f"%{query}%"
            conditions.append(
                f"(d.filename LIKE ? OR COALESCE(d.title,'') LIKE ? OR json_extract({safe_meta},'$.entry_id') LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])
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
            operator = "<" if direction == "desc" else ">"
            conditions.append(f"({sort_sql},d.id) {operator} (?,?)")
            params.extend(decoded.key)
        order = "DESC" if direction == "desc" else "ASC"
        cursor_result = await self.db.execute(
            "SELECT d.id,d.filename,d.title,d.path,d.metadata,d.document_number," + sort_sql + " AS _sort_key "
            "FROM documents d WHERE " + " AND ".join(conditions) + f" ORDER BY {sort_sql} {order},d.id {order} LIMIT ?",
            [*params, limit + 1],
        )
        rows = await cursor_result.fetchall()
        has_next = len(rows) > limit
        rows = rows[:limit]
        page: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(row[4])
            except (TypeError, json.JSONDecodeError):
                continue
            page.append({
                "id": row[0], "filename": row[1], "title": row[2], "path": row[3],
                "metadata": metadata, "document_number": row[5],
            })
        next_cursor = None
        if has_next and page:
            next_cursor = encode_cursor(
                ReadCursor(
                    "corpus.entries",
                    revision,
                    sort,
                    direction,
                    (str(rows[-1][6] or ""), str(rows[-1][0])),
                )
            )
        count_cursor = await self.db.execute(
            "SELECT count(*) FROM documents d WHERE " + " AND ".join(count_conditions),
            count_params,
        )
        count_row = await count_cursor.fetchone()
        return ReadPage(
            revision=revision,
            items=page,
            next_cursor=next_cursor,
            total_count=int(count_row[0]) if count_row else 0,
        )

    async def corpus_summary(
        self, kb_id: str, filters: dict[str, str] | None, query: str | None = None
    ) -> CorpusSummary:
        revision = await self.revision(kb_id)
        clean = validate_facets(filters)
        query = query.strip() if query else None
        cache_key = (self.user_id, kb_id, tuple(sorted(clean.items())), query, revision)
        cached = _SUMMARY_CACHE.get(cache_key)
        if cached is not None:
            _SUMMARY_CACHE.move_to_end(cache_key)
            return cached
        rows = await self._corpus_rows(query)
        summary = build_summary(rows, clean, revision)
        cited, wiki_covered, entry_cells = await self._summary_graph_kpis(clean, query)
        summary.kpis.update(
            cited=cited,
            wiki_covered=wiki_covered,
            wiki_cells_with_entries=entry_cells,
        )
        _SUMMARY_CACHE[cache_key] = summary
        _SUMMARY_CACHE.move_to_end(cache_key)
        while len(_SUMMARY_CACHE) > 256:
            _SUMMARY_CACHE.popitem(last=False)
        return summary

    async def _summary_graph_kpis(self, filters: dict[str, str], query: str | None) -> tuple[int, int, int]:
        safe_meta = "CASE WHEN typeof(t.metadata)='text' AND json_valid(t.metadata) THEN t.metadata ELSE '{}' END"
        conditions = [
            "t.user_id=?",
            "t.source_kind='source'",
            "t.status!='failed'",
            f"json_type({safe_meta})='object'",
            f"json_type({safe_meta},'$.spec_version')='text'",
            f"json_type({safe_meta},'$.stage')='text'",
        ]
        params: list[Any] = [self.user_id]
        facet_conditions, facet_params = sqlite_facet_conditions(filters, "t")
        conditions.extend(facet_conditions)
        params.extend(facet_params)
        if query:
            pattern = f"%{query}%"
            conditions.append(
                f"(t.filename LIKE ? OR COALESCE(t.title,'') LIKE ? OR json_extract({safe_meta},'$.entry_id') LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])
        where = " AND ".join(conditions)
        cited_cursor = await self.db.execute(
            "SELECT count(DISTINCT t.id) FROM documents t "
            "JOIN document_references r ON r.target_document_id=t.id "
            "JOIN documents s ON s.id=r.source_document_id "
            f"WHERE {where} AND s.user_id=? AND s.source_kind='wiki' "
            "AND r.reference_type IN ('cites','links_to')",
            [*params, self.user_id],
        )
        cited_row = await cited_cursor.fetchone()

        wiki_meta = "CASE WHEN typeof(w.metadata)='text' AND json_valid(w.metadata) THEN w.metadata ELSE '{}' END"
        coverage_cursor = await self.db.execute(
            "WITH entry_cells AS ("
            f"SELECT DISTINCT json_extract({safe_meta},'$.stage') AS stage,"
            f"substr(json_extract({safe_meta},'$.domain'),1,1) AS layer "
            f"FROM documents t WHERE {where} "
            f"AND substr(json_extract({safe_meta},'$.domain'),1,1) IN ('G','C','O','Z')"
            "),wiki_cells AS ("
            "SELECT DISTINCT ws.value AS stage,substr(wd.value,1,1) AS layer "
            f"FROM documents w,json_each({wiki_meta},'$.facet_rollup.stage') ws,"
            f"json_each({wiki_meta},'$.facet_rollup.domain') wd "
            "WHERE w.user_id=? AND w.source_kind='wiki'"
            ") SELECT (SELECT count(*) FROM entry_cells),"
            "(SELECT count(*) FROM entry_cells e JOIN wiki_cells w "
            "ON w.stage=e.stage AND w.layer=e.layer)",
            [*params, self.user_id],
        )
        coverage_row = await coverage_cursor.fetchone()
        return (
            int(cited_row[0]) if cited_row else 0,
            int(coverage_row[1]) if coverage_row else 0,
            int(coverage_row[0]) if coverage_row else 0,
        )

    async def graph_summary(self, kb_id: str) -> GraphSummary:
        revision = await self.revision(kb_id)
        node_count = await self.db.execute_fetchall("SELECT count(*) FROM documents WHERE user_id=?", (self.user_id,))
        rows = await self.db.execute_fetchall(
            "SELECT DISTINCT r.target_document_id FROM document_references r "
            "JOIN documents s ON s.id=r.source_document_id "
            "JOIN documents t ON t.id=r.target_document_id "
            "WHERE s.user_id=? AND t.user_id=? "
            "AND r.reference_type IN ('cites','links_to') "
            "ORDER BY r.target_document_id LIMIT 200",
            (self.user_id, self.user_id),
        )
        edge_count = await self.db.execute_fetchall(
            "SELECT count(*) FROM document_references r "
            "JOIN documents s ON s.id=r.source_document_id "
            "JOIN documents t ON t.id=r.target_document_id "
            "WHERE s.user_id=? AND t.user_id=?",
            (self.user_id, self.user_id),
        )
        return GraphSummary(
            revision=revision,
            node_count=int(node_count[0][0]),
            edge_count=int(edge_count[0][0]),
            cited_document_ids=[row[0] for row in rows],
        )


__all__ = ["LocalReadService", "StaleReadCursor"]
