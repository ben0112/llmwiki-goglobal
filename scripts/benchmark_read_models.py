#!/usr/bin/env python3
"""Measure bounded read-model latency and structure without exposing corpus data."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MAX_PAGE_ITEMS = 200
MAX_RESPONSE_BYTES = 1_048_576
MAX_WARM_PAGE_MS = 200.0
MAX_NOT_MODIFIED_MS = 50.0


@dataclass(frozen=True)
class HttpResult:
    status: int
    elapsed_ms: float
    body: bytes
    etag: str | None


def _request(url: str, token: str | None, etag: str | None = None) -> HttpResult:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if etag:
        headers["If-None-Match"] = etag
    request = Request(url, headers=headers)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - operator supplied endpoint
            body = response.read()
            return HttpResult(
                status=response.status,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                body=body,
                etag=response.headers.get("ETag"),
            )
    except HTTPError as error:
        body = error.read()
        return HttpResult(
            status=error.code,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            body=body,
            etag=error.headers.get("ETag"),
        )


def _json(result: HttpResult) -> dict[str, Any]:
    if result.status != 200:
        raise RuntimeError(f"read endpoint returned HTTP {result.status}")
    value = json.loads(result.body)
    if not isinstance(value, dict):
        raise RuntimeError("read endpoint returned a non-object payload")
    return value


def _query_plan(database: sqlite3.Connection, *, second_page: bool) -> list[str]:
    cursor_sql = " AND (filename COLLATE NOCASE,id) > (?,?)" if second_page else ""
    sql = (
        "EXPLAIN QUERY PLAN SELECT id,filename,title,path,file_type,status,file_size,"
        "page_count,tags,date,metadata,error_message,version,document_number,stale_since,"
        "created_at,updated_at FROM documents WHERE path=?"
        f"{cursor_sql} ORDER BY filename COLLATE NOCASE,id LIMIT ?"
    )
    params: tuple[Any, ...] = (
        ("/", "", "", MAX_PAGE_ITEMS + 1)
        if second_page
        else ("/", MAX_PAGE_ITEMS + 1)
    )
    return [str(row[3]) for row in database.execute(sql, params)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--token-env", help="Environment variable containing a bearer token")
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error("--database must point to an existing SQLite file")
    token = None
    if args.token_env:
        token = os.environ.get(args.token_env)
        if not token:
            parser.error("--token-env is set but the environment variable is empty")

    database = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    try:
        document_count = int(database.execute("SELECT count(*) FROM documents").fetchone()[0])
        revision_started = time.perf_counter()
        revision_row = database.execute(
            "SELECT read_revision FROM workspace WHERE id=?", (args.kb_id,),
        ).fetchone()
        revision_ms = (time.perf_counter() - revision_started) * 1000
        if revision_row is None:
            raise RuntimeError("knowledge base id is not present in the copied database")
        first_plan = _query_plan(database, second_page=False)
        second_plan = _query_plan(database, second_page=True)
    finally:
        database.close()

    params = urlencode({"path": "/", "sort": "name", "direction": "asc", "limit": MAX_PAGE_ITEMS})
    first_url = (
        f"{args.api_base.rstrip('/')}/v1/knowledge-bases/{args.kb_id}/documents/browse?{params}"
    )
    _request(first_url, token)  # warm route, connection, SQLite page cache, and serializers
    first = _request(first_url, token)
    first_payload = _json(first)
    next_cursor = first_payload.get("next_cursor")
    if isinstance(next_cursor, str) and next_cursor:
        second_url = f"{first_url}&{urlencode({'cursor': next_cursor})}"
        second = _request(second_url, token)
        second_payload = _json(second)
    else:
        second = HttpResult(status=200, elapsed_ms=0.0, body=b"", etag=None)
        second_payload = {"items": []}
    not_modified = _request(first_url, token, first.etag)

    first_items = first_payload.get("items")
    second_items = second_payload.get("items")
    if not isinstance(first_items, list) or not isinstance(second_items, list):
        raise RuntimeError("read endpoint omitted the items array")
    item_count = max(len(first_items), len(second_items))
    response_bytes = max(len(first.body), len(second.body))
    plan = [*first_plan, *second_plan]
    temporary_sort = any("USE TEMP B-TREE FOR ORDER BY" in detail for detail in plan)

    print(f"document_count={document_count}")
    print(f"revision_latency_ms={revision_ms:.3f}")
    print(f"first_page_latency_ms={first.elapsed_ms:.3f}")
    print(f"second_page_latency_ms={second.elapsed_ms:.3f}")
    print(f"etag_304_latency_ms={not_modified.elapsed_ms:.3f}")
    print(f"item_count={item_count}")
    print(f"response_bytes={response_bytes}")
    for detail in plan:
        print(f"query_plan={detail}")

    failures = []
    if max(first.elapsed_ms, second.elapsed_ms) > MAX_WARM_PAGE_MS:
        failures.append(f"warm page latency exceeds {MAX_WARM_PAGE_MS:.0f} ms")
    if not_modified.status != 304:
        failures.append(f"conditional request returned HTTP {not_modified.status}, expected 304")
    elif not_modified.elapsed_ms > MAX_NOT_MODIFIED_MS:
        failures.append(f"304 latency exceeds {MAX_NOT_MODIFIED_MS:.0f} ms")
    if item_count > MAX_PAGE_ITEMS:
        failures.append(f"page item count exceeds {MAX_PAGE_ITEMS}")
    if response_bytes > MAX_RESPONSE_BYTES:
        failures.append("page response exceeds 1 MiB")
    if temporary_sort:
        failures.append("query plan uses a temporary ORDER BY sort")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
