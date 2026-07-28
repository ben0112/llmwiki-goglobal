"""Bounded Postgres read models for Hosted mode."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from domain.file_types import EXTRACTION_TYPES, IMAGE_TYPES, SIMPLE_TEXT_TYPES

from llmwiki_core.read_cursor import ReadCursor, decode_cursor, encode_cursor

from .read_models import (
    BrowsePage,
    DocumentStatus,
    DocumentStatusPage,
    FolderItem,
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


__all__ = ["HostedReadService"]
