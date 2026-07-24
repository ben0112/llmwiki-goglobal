"""Download a public PDF by URL and feed it into the standard ingest pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import suppress
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

import asyncpg
import httpx
from config import settings
from fastapi import HTTPException
from infra.safe_fetch import build_pinned_request, parse_public_fetch_url, redirect_location, resolve_public_ip
from services.types import DownloadedPdf, IngestedPdf

if TYPE_CHECKING:
    from jobs.service import JobService
    from services.s3 import S3Service

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30
MAX_REDIRECTS = 5
USER_AGENT = "LLMWiki/1.0 (+https://llmwiki.app)"

_ARXIV_ABS_RE = re.compile(r"^(https?://(?:www\.)?arxiv\.org)/abs/(.+)$")
_DISPOSITION_FILENAME_RE = re.compile(r'filename\*?=(?:"([^"]+)"|([^;\s]+))', re.IGNORECASE)


class UrlIngestService:
    def __init__(self, pool: asyncpg.Pool, s3_service: S3Service, job_service: JobService):
        self.pool = pool
        self.s3 = s3_service
        self.jobs = job_service

    async def ingest_pdf(self, user_id: str, kb_id: str, url: str, path: str) -> IngestedPdf:
        url = _normalize_pdf_url(url)
        path = _sanitize_path(path)
        await self._require_kb_owned(user_id, kb_id)

        existing = await self._find_by_source_url(user_id, kb_id, url)
        if existing:
            return await self._return_existing(existing, user_id, kb_id)

        pdf = await self._download(url)
        return await self._create_pending_document(user_id, kb_id, url, path, pdf)

    async def _download(self, url: str) -> DownloadedPdf:
        current = url
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=False, trust_env=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                parsed = parse_public_fetch_url(current)
                if not parsed:
                    raise HTTPException(status_code=400, detail="URL must be a public http(s) address")
                ip = resolve_public_ip(parsed.hostname)
                if not ip:
                    raise HTTPException(status_code=400, detail="URL host is not publicly reachable")
                request = build_pinned_request(
                    client,
                    parsed,
                    ip,
                    {"Accept": "application/pdf,*/*", "User-Agent": USER_AGENT},
                )
                try:
                    resp = await client.send(request, stream=True)
                except httpx.HTTPError:
                    raise HTTPException(status_code=400, detail="Could not fetch URL") from None
                try:
                    redirect = redirect_location(resp, current)
                    if redirect:
                        current = redirect
                        continue
                    return self._validate_pdf_response(resp, await self._read_capped(resp), current)
                finally:
                    await resp.aclose()
        raise HTTPException(status_code=400, detail="Too many redirects")

    async def _read_capped(self, resp: httpx.Response) -> bytes:
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"URL returned HTTP {resp.status_code}")
        chunks = bytearray()
        async for chunk in resp.aiter_bytes(chunk_size=65536):
            if len(chunks) + len(chunk) > MAX_PDF_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB download limit",
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _validate_pdf_response(self, resp: httpx.Response, data: bytes, final_url: str) -> DownloadedPdf:
        if not data.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="URL did not return a PDF. For web pages, use the browser extension instead.",
            )
        return DownloadedPdf(data=data, filename=_derive_filename(resp, final_url))

    async def _create_pending_document(
        self,
        user_id: str,
        kb_id: str,
        url: str,
        path: str,
        pdf: DownloadedPdf,
    ) -> IngestedPdf:
        from jobs.models import JobCreate, JobType

        document_id = str(uuid4())
        s3_key = f"{user_id}/{document_id}/source.pdf"
        try:
            await self.s3.upload_bytes(s3_key, pdf.data, "application/pdf")
        except asyncio.CancelledError:
            await asyncio.shield(self._delete_uploaded_document_prefix(user_id, document_id))
            raise
        except Exception:  # noqa: BLE001 - storage implementations use different typed SDK errors.
            await self._delete_uploaded_document_prefix(user_id, document_id)
            raise HTTPException(
                status_code=502,
                detail="Could not store the downloaded PDF — try again",
            ) from None

        duplicate: dict | None = None
        job = None
        transaction_started = False
        commit_attempted = False
        failure: Exception | asyncio.CancelledError | None = None
        failure_traceback = None
        conn = None
        transaction = None
        try:
            conn = await self.pool.acquire()
            transaction = conn.transaction()
            await transaction.start()
            transaction_started = True
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", user_id)
            duplicate = await self._find_by_source_url(user_id, kb_id, url, conn=conn)
            if duplicate is None:
                await self._insert_within_quota(
                    conn,
                    document_id,
                    kb_id,
                    user_id,
                    pdf,
                    path,
                    url,
                )
                job = await self.jobs.create_in_transaction(
                    conn,
                    JobCreate(
                        job_type=JobType.DOCUMENT_EXTRACT,
                        user_id=UUID(user_id),
                        knowledge_base_id=UUID(kb_id),
                        document_id=UUID(document_id),
                        payload={"document_id": document_id},
                        idempotency_key=f"document.extract:{document_id}",
                    ),
                    authenticated_user_id=UUID(user_id),
                )
            else:
                job = await self._ensure_existing_job(conn, duplicate, user_id, kb_id)
            commit_attempted = True
            await self._commit_transaction(transaction)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - compensate all persistence failures.
            failure = exc
            failure_traceback = exc.__traceback__
            if transaction_started and not commit_attempted:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(transaction.rollback())
        finally:
            if conn is not None:
                await asyncio.shield(self.pool.release(conn))

        if failure is not None:
            committed: bool | None = False
            if commit_attempted and duplicate is None and job is not None:
                committed = await asyncio.shield(
                    self._confirm_document_job_committed(user_id, kb_id, document_id, job.id)
                )
            if duplicate is not None or committed is False:
                await asyncio.shield(self._delete_uploaded_document_prefix(user_id, document_id))
            raise failure.with_traceback(failure_traceback)

        if duplicate is not None:
            await self._delete_uploaded_document_prefix(user_id, document_id)
            return {**duplicate, "already_exists": True, "job_id": str(job.id)}

        return {
            "id": document_id,
            "filename": pdf.filename,
            "status": "pending",
            "already_exists": False,
            "job_id": str(job.id),
        }

    async def _commit_transaction(self, transaction) -> None:
        await transaction.commit()

    async def _confirm_document_job_committed(
        self,
        user_id: str,
        kb_id: str,
        document_id: str,
        job_id,
    ) -> bool | None:
        """Resolve a commit error without deleting storage on an unknown outcome."""
        try:
            return bool(
                await self.pool.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM documents d "
                    "JOIN background_jobs j ON j.document_id = d.id "
                    "WHERE d.id = $1::uuid AND d.user_id = $2::uuid "
                    "AND d.knowledge_base_id = $3::uuid AND j.id = $4 "
                    "AND j.user_id = d.user_id AND j.knowledge_base_id = d.knowledge_base_id)",
                    document_id,
                    user_id,
                    kb_id,
                    job_id,
                )
            )
        except Exception:  # noqa: BLE001 - unknown commit outcome must preserve the object.
            return None

    async def _require_kb_owned(self, user_id: str, kb_id: str) -> None:
        owner = await self.pool.fetchval(
            "SELECT user_id::text FROM knowledge_bases WHERE id = $1::uuid",
            kb_id,
        )
        if owner != user_id:
            raise HTTPException(status_code=403, detail="Knowledge base not found or not owned by you")

    async def _insert_within_quota(
        self,
        conn: asyncpg.Connection,
        document_id: str,
        kb_id: str,
        user_id: str,
        pdf: DownloadedPdf,
        path: str,
        url: str,
    ) -> None:
        """Quota check + pending-row insert under a per-user advisory lock, so
        concurrent ingests cannot all pass the same SUM(file_size) read."""
        title = pdf.filename.rsplit(".", 1)[0]
        await self._check_storage_quota(conn, user_id, len(pdf.data))
        await conn.execute(
            "INSERT INTO documents (id, knowledge_base_id, user_id, filename, path, title, "
            "file_type, file_size, status, metadata) "
            "VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, 'pdf', $7, 'pending', $8::jsonb)",
            document_id,
            kb_id,
            user_id,
            pdf.filename,
            path,
            title,
            len(pdf.data),
            json.dumps({"source_url": url}),
        )

    async def _check_storage_quota(self, conn: asyncpg.Connection, user_id: str, incoming_bytes: int) -> None:
        row = await conn.fetchrow(
            "SELECT storage_limit_bytes FROM users WHERE id = $1",
            user_id,
        )
        storage_limit = row["storage_limit_bytes"] if row else settings.QUOTA_MAX_STORAGE_BYTES
        current_bytes = await conn.fetchval(
            "SELECT COALESCE(SUM(file_size), 0) FROM documents WHERE user_id = $1",
            user_id,
        )
        if current_bytes + incoming_bytes > storage_limit:
            used_mb = current_bytes / (1024 * 1024)
            max_mb = storage_limit / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Storage quota exceeded. Using {used_mb:.0f} MB of {max_mb:.0f} MB.",
            )

    async def _find_by_source_url(
        self,
        user_id: str,
        kb_id: str,
        url: str,
        *,
        conn: asyncpg.Connection | None = None,
    ) -> dict | None:
        executor = conn or self.pool
        row = await executor.fetchrow(
            "SELECT id::text, knowledge_base_id::text, title, path, filename, status::text "
            "FROM documents "
            "WHERE user_id = $1 AND knowledge_base_id = $2::uuid AND file_type = 'pdf' "
            "AND NOT archived AND metadata->>'source_url' = $3 "
            "ORDER BY created_at DESC LIMIT 1",
            user_id,
            kb_id,
            url,
        )
        return dict(row) if row else None

    async def _return_existing(
        self,
        existing: dict,
        user_id: str,
        kb_id: str,
    ) -> IngestedPdf:
        async with self.pool.acquire() as conn, conn.transaction():
            job = await self._ensure_existing_job(conn, existing, user_id, kb_id)
        return {**existing, "already_exists": True, "job_id": str(job.id)}

    async def _ensure_existing_job(
        self,
        conn: asyncpg.Connection,
        existing: dict,
        user_id: str,
        kb_id: str,
    ):
        from jobs import repository
        from jobs.models import JobCreate, JobType

        document_id = UUID(existing["id"])
        latest = await conn.fetchrow(
            "SELECT id, state, idempotency_key FROM background_jobs "
            "WHERE user_id = $1 AND job_type = 'document.extract' AND document_id = $2 "
            "ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE",
            UUID(user_id),
            document_id,
        )
        if latest is not None and latest["state"] in {"queued", "running", "retry_wait", "succeeded"}:
            return await repository.get_for_user(conn, latest["id"], UUID(user_id))

        idempotency_key = f"document.extract:{document_id}"
        if latest is not None:
            idempotency_key = f"{idempotency_key}:after:{latest['id']}"
            await conn.execute(
                "UPDATE documents SET "
                "status = CASE WHEN status = 'ready' AND version > 0 THEN status ELSE 'pending' END, "
                "error_message = NULL, "
                "updated_at = now() WHERE id = $1 AND user_id = $2 AND knowledge_base_id = $3 "
                "AND NOT archived AND source_kind = 'source'",
                document_id,
                UUID(user_id),
                UUID(kb_id),
            )
        return await self.jobs.create_in_transaction(
            conn,
            JobCreate(
                job_type=JobType.DOCUMENT_EXTRACT,
                user_id=UUID(user_id),
                knowledge_base_id=UUID(kb_id),
                document_id=document_id,
                payload={"document_id": str(document_id)},
                idempotency_key=idempotency_key,
            ),
            authenticated_user_id=UUID(user_id),
        )

    async def _delete_uploaded_document_prefix(self, user_id: str, document_id: str) -> None:
        try:
            await self.s3.delete_prefix(f"{user_id}/{document_id}/")
        except Exception as exc:  # noqa: BLE001 - cleanup failure is logged without raw storage details.
            logger.error(
                "URL ingest orphan cleanup failed document_id=%s error_type=%s",
                document_id,
                type(exc).__name__,
            )


class _LegacyUrlIngestCompatibility(UrlIngestService):
    """Rollback-only Hosted URL producer using process-local OCR dispatch."""

    def __init__(self, pool: asyncpg.Pool, s3_service: S3Service, ocr_service: object):
        super().__init__(pool, s3_service, job_service=None)
        self.ocr = ocr_service

    async def _return_existing(self, existing: dict, user_id: str, kb_id: str) -> dict:
        del user_id, kb_id
        return {**existing, "already_exists": True}

    async def _create_pending_document(
        self,
        user_id: str,
        kb_id: str,
        url: str,
        path: str,
        pdf: DownloadedPdf,
    ) -> dict:
        document_id = str(uuid4())
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", user_id)
            await self._insert_within_quota(conn, document_id, kb_id, user_id, pdf, path, url)

        try:
            await self.s3.upload_bytes(
                f"{user_id}/{document_id}/source.pdf",
                pdf.data,
                "application/pdf",
            )
        except Exception:  # noqa: BLE001 - legacy storage adapters expose different errors.
            await self.pool.execute("DELETE FROM documents WHERE id = $1::uuid", document_id)
            raise HTTPException(
                status_code=502,
                detail="Could not store the downloaded PDF — try again",
            ) from None

        from infra.tasks import spawn_logged

        spawn_logged(
            self.ocr.process_document(document_id, user_id),
            f"url-ingest:{document_id[:8]}",
        )
        return {
            "id": document_id,
            "filename": pdf.filename,
            "status": "pending",
            "already_exists": False,
        }


def _normalize_pdf_url(url: str) -> str:
    """arXiv abstract pages link to a canonical PDF — fetch that directly."""
    match = _ARXIV_ABS_RE.match(url.strip())
    if match:
        return f"{match.group(1)}/pdf/{match.group(2)}"
    return url.strip()


def _derive_filename(resp: httpx.Response, final_url: str) -> str:
    disposition = resp.headers.get("content-disposition", "")
    match = _DISPOSITION_FILENAME_RE.search(disposition)
    raw = (match.group(1) or match.group(2)) if match else ""
    if not raw:
        raw = unquote(urlparse(final_url).path.rsplit("/", 1)[-1])
    return _sanitize_filename(raw)


def _sanitize_filename(raw: str) -> str:
    name = re.sub(r"[^\w.\- ]", "", raw.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    name = name[:120].rstrip(". ")
    if not name:
        name = "document"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def _sanitize_path(raw_path: str) -> str:
    path = re.sub(r"[\x00-\x1f\x7f]", "", raw_path or "/")
    path = "/" + path.replace("\\", "/").strip("/") + "/"
    path = re.sub(r"/\.\.(/|$)", "/", path)
    path = re.sub(r"/+", "/", path)
    return "/" if path == "//" else path
