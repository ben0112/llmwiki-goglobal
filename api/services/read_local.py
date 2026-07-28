"""Bounded SQLite read models for Local mode."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import aiosqlite
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
    "id, filename, title, path, file_type, status, file_size, page_count, tags, "
    "date, metadata, error_message, version, document_number, stale_since, "
    "created_at, updated_at"
)
_SUPPORTED_UPLOAD_TYPES = SIMPLE_TEXT_TYPES | EXTRACTION_TYPES | IMAGE_TYPES
_MAX_UPLOAD_BYTES = 1_073_741_824


def _row_dict(cursor: aiosqlite.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    result = dict(zip((column[0] for column in cursor.description), row, strict=True))
    for field, fallback in (("tags", []), ("metadata", None)):
        value = result.get(field)
        if isinstance(value, str):
            import json

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

        conditions: list[str] = []
        params: list[Any] = []
        if path is not None:
            conditions.append("path = ?")
            params.append(path)
        else:
            conditions.append("source_kind = 'source'")
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
        folders = await self._folders(path) if path is not None and cursor is None else []
        return BrowsePage(
            revision=revision,
            items=items,
            next_cursor=next_cursor,
            total_count=int(count_row[0]) if count_row else 0,
            folders=folders,
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
            "WHERE path LIKE ? AND path != ?) "
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


__all__ = ["LocalReadService", "StaleReadCursor"]
