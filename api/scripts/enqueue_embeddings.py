"""Reconcile ready current document versions missing complete embedding sets."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

import asyncpg
from jobs.service import JobService

from llmwiki_core.models import EmbeddingProfile

DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500
_INVALID_ARGUMENTS = '{"error":"invalid_arguments"}\n'


def _emit_invalid_arguments() -> None:
    sys.stderr.write(_INVALID_ARGUMENTS)


class _ParserExit(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__()


class _SafeArgumentParser(argparse.ArgumentParser):
    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status:
            _emit_invalid_arguments()
        elif message:
            self._print_message(message)
        raise _ParserExit(status)

    def error(self, message: str) -> None:
        del message
        _emit_invalid_arguments()
        raise _ParserExit(2)


async def reconcile_missing_embeddings(
    pool,
    profile: EmbeddingProfile,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, int]:
    if not isinstance(profile, EmbeddingProfile):
        raise TypeError("profile must be an EmbeddingProfile")
    if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")

    scanned = 0
    enqueued = 0
    cursor_user = None
    cursor_document = None
    service = JobService(pool)
    while True:
        rows = await pool.fetch(
            "SELECT d.user_id,d.knowledge_base_id,d.id,d.version FROM documents d "
            "WHERE d.status='ready' AND NOT d.archived AND d.source_kind='source' "
            "AND EXISTS (SELECT 1 FROM document_chunks dc "
            "            WHERE dc.document_id=d.id AND dc.document_version=d.version "
            "            AND dc.user_id=d.user_id AND dc.knowledge_base_id=d.knowledge_base_id) "
            "AND (SELECT count(*) FROM document_chunks dc "
            "     WHERE dc.document_id=d.id AND dc.document_version=d.version "
            "     AND dc.user_id=d.user_id AND dc.knowledge_base_id=d.knowledge_base_id) "
            "    <> (SELECT count(*) FROM chunk_embeddings ce "
            "        WHERE ce.document_id=d.id AND ce.document_version=d.version "
            "        AND ce.user_id=d.user_id AND ce.knowledge_base_id=d.knowledge_base_id "
            "        AND ce.provider=$1 AND ce.model=$2 AND ce.dimensions=$3) "
            "AND ($4::uuid IS NULL OR (d.user_id,d.id) > ($4::uuid,$5::uuid)) "
            "ORDER BY d.user_id,d.id LIMIT $6",
            profile.provider,
            profile.model,
            profile.dimensions,
            cursor_user,
            cursor_document,
            page_size,
        )
        if not rows:
            break
        for row in rows:
            _job, created = await service.ensure_document_embedding_with_status(
                document_id=row["id"],
                document_version=row["version"],
                user_id=row["user_id"],
                knowledge_base_id=row["knowledge_base_id"],
                profile=profile,
                recover_terminal=True,
            )
            scanned += 1
            enqueued += int(created)
        cursor_user = rows[-1]["user_id"]
        cursor_document = rows[-1]["id"]
    return {"scanned": scanned, "enqueued": enqueued}


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="enqueue_embeddings",
        description="Enqueue missing durable document embeddings.",
    )
    parser.add_argument("--missing", action="store_true", help="scan ready current document versions")
    parser.add_argument("--page-size", default=str(DEFAULT_PAGE_SIZE))
    return parser


def _page_size(raw: object) -> int:
    if not isinstance(raw, str) or not 1 <= len(raw) <= 3 or not raw.isascii() or not raw.isdecimal():
        raise ValueError("invalid page size")
    value = int(raw)
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise ValueError("invalid page size")
    return value


async def _run(page_size: int) -> dict[str, int]:
    from config import settings

    profile = settings.embedding_profile
    if profile is None:
        raise ValueError("hybrid embedding profile is not configured")
    pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=1, max_size=4)
    try:
        return await reconcile_missing_embeddings(pool, profile, page_size=page_size)
    finally:
        await pool.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if not args.missing:
            parser.error("missing reconciliation mode")
        page_size = _page_size(args.page_size)
    except ValueError:
        _emit_invalid_arguments()
        return 2
    except _ParserExit as exc:
        return exc.status
    try:
        result = asyncio.run(_run(page_size))
    except ValueError:
        print(json.dumps({"error": "invalid_configuration"}, sort_keys=True))
        return 3
    except Exception:  # noqa: BLE001 - never expose connection strings or credentials.
        print(json.dumps({"error": "reconciliation_failed"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
