from __future__ import annotations

import asyncio
import base64
import inspect
import json
import secrets
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import services.url_ingest as url_ingest_module
from botocore.exceptions import EndpointConnectionError
from config import settings
from fastapi import HTTPException
from infra.quota import QuotaExceeded, QuotaReservation
from jobs import repository
from jobs.handlers import (
    RetryableJobError,
    TerminalJobError,
    WorkerContext,
    _prepare_document_extraction,
    _set_extraction_failure_status,
    handle_document_extract,
)
from jobs.lease import JobLease
from jobs.models import JobCancelled, JobCreate, JobRecord, JobType, LeaseLost
from jobs.service import JobService
from services.ocr import OCRService
from services.types import DownloadedPdf
from services.url_ingest import UrlIngestService as _RealUrlIngestService


class RecordingS3:
    def __init__(self, *, upload_error: Exception | None = None, events: list[str] | None = None) -> None:
        self.upload_error = upload_error
        self.events = events
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.deleted_prefixes: list[str] = []
        self.presigned_gets: list[str] = []

    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
        if self.events is not None:
            self.events.append("s3_upload")
        self.uploads.append(key)
        if self.upload_error is not None:
            raise self.upload_error
        self.objects[key] = data

    async def delete_prefix(self, prefix: str) -> None:
        self.deleted_prefixes.append(prefix)
        for key in tuple(self.objects):
            if key.startswith(prefix):
                del self.objects[key]

    async def generate_presigned_get(self, key: str) -> str:
        self.presigned_gets.append(key)
        return "https://storage.invalid/signed-source"

    async def download_bytes(self, key: str) -> bytes:
        return self.objects[key]

    async def download_to_file(self, key: str, file_path: str) -> None:
        Path(file_path).write_bytes(self.objects[key])

    async def delete_key(self, key: str) -> None:
        self.objects.pop(key, None)


class RecordingQuota:
    def __init__(
        self,
        *,
        events: list[str] | None = None,
        reserve_error: Exception | None = None,
        finalize_error: Exception | None = None,
        release_error: Exception | None = None,
        renew_result: bool = True,
        renew_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.reserve_error = reserve_error
        self.finalize_error = finalize_error
        self.release_error = release_error
        self.renew_result = renew_result
        self.renew_error = renew_error
        self.reservations: list[QuotaReservation] = []
        self.finalized: list[QuotaReservation] = []
        self.released: list[QuotaReservation] = []
        self.renewed: list[QuotaReservation] = []

    async def reserve(self, user_id, upload_id, byte_count, ttl_seconds):
        del ttl_seconds
        if self.events is not None:
            self.events.append("quota_reserve")
        if self.reserve_error is not None:
            raise self.reserve_error
        reservation = QuotaReservation(user_id, upload_id, byte_count, secrets.token_urlsafe(24))
        self.reservations.append(reservation)
        return reservation

    async def finalize(self, reservation):
        if self.events is not None:
            self.events.append("quota_finalize")
        self.finalized.append(reservation)
        if self.finalize_error is not None:
            raise self.finalize_error
        return True

    async def renew(self, reservation, ttl_seconds):
        del ttl_seconds
        if self.events is not None:
            self.events.append("quota_renew")
        self.renewed.append(reservation)
        if self.renew_error is not None:
            raise self.renew_error
        return self.renew_result

    async def release(self, reservation):
        if self.events is not None:
            self.events.append("quota_release")
        self.released.append(reservation)
        if self.release_error is not None:
            raise self.release_error
        return True


def UrlIngestService(pool, s3_service, job_service, quota_service=None):
    """Keep older extraction tests explicit about using an injectable quota contract."""
    return _RealUrlIngestService(
        pool,
        s3_service,
        job_service,
        quota_service or RecordingQuota(),
    )


class ReleaseFailingPool:
    def __init__(self, pool, *, cancelled: bool = False) -> None:
        self._pool = pool
        self._cancelled = cancelled

    def __getattr__(self, name):
        return getattr(self._pool, name)

    async def acquire(self):
        return await self._pool.acquire()

    async def release(self, conn) -> None:
        await self._pool.release(conn)
        if self._cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("pool release failed")


async def _seed_tenant(pool, *, storage_limit_bytes: int = 1_000_000):
    user_id = uuid4()
    kb_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, storage_limit_bytes) VALUES ($1, $2, $3)",
        user_id,
        f"{user_id}@durable-extraction.test",
        storage_limit_bytes,
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        kb_id,
        user_id,
        f"KB {kb_id}",
        f"kb-{kb_id}",
    )
    return user_id, kb_id


async def _seed_document_job(
    pool,
    *,
    filename: str = "paper.pdf",
    status: str = "pending",
    version: int = 0,
    max_attempts: int = 3,
):
    user_id, kb_id = await _seed_tenant(pool)
    doc_id = uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, file_type, status, version, source_kind) "
        "VALUES ($1, $2, $3, $4, '/', $5, $6, $7, 'source')",
        doc_id,
        kb_id,
        user_id,
        filename,
        filename.rsplit(".", 1)[-1],
        status,
        version,
    )
    command = JobCreate(
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        payload={"document_id": str(doc_id)},
        idempotency_key=f"document.extract:{doc_id}",
        max_attempts=max_attempts,
    )
    async with pool.acquire() as conn, conn.transaction():
        created = await repository.create(conn, command)
    claimed = await repository.claim(pool, created.id, "worker-extraction", 120)
    assert claimed is not None
    return claimed, user_id, kb_id, doc_id


def _context(pool, s3) -> WorkerContext:
    return WorkerContext(
        pool=pool,
        s3=s3,
        converter_url="http://converter.internal",
        converter_secret="converter-secret",
    )


def _lease(pool, job: JobRecord) -> JobLease:
    return JobLease(pool, job.id, "worker-extraction", 120, 30)


class ScriptedLease:
    def __init__(
        self,
        failure: Exception | None = None,
        *,
        fail_on_connection_call: int = 2,
    ) -> None:
        self.failure = failure
        self.calls = 0
        self.connection_calls = 0
        self.fail_on_connection_call = fail_on_connection_call

    async def checkpoint(self, conn=None):
        self.calls += 1
        if conn is not None:
            assert conn.is_in_transaction()
            self.connection_calls += 1
            if self.failure is not None and self.connection_calls == self.fail_on_connection_call:
                raise self.failure
        return


class GatedDatabaseLease:
    def __init__(self, job: JobRecord) -> None:
        self.job = job
        self.job_locked = asyncio.Event()
        self.release_job = asyncio.Event()
        self.backend_pid: int | None = None

    async def checkpoint(self, conn=None):
        assert conn is not None and conn.is_in_transaction()
        await conn.execute("SET LOCAL lock_timeout = '2s'")
        await repository.assert_active(conn, self.job.id, "worker-extraction")
        self.backend_pid = await conn.fetchval("SELECT pg_backend_pid()")
        self.job_locked.set()
        await self.release_job.wait()


async def _wait_for_blocked_backend(observer, backend_pid: int) -> None:
    async with asyncio.timeout(1):
        while not await observer.fetchval(
            "SELECT COALESCE(cardinality(pg_blocking_pids($1)) > 0, false)",
            backend_pid,
        ):
            pass


@pytest.mark.asyncio
async def test_url_ingest_commits_document_and_job_atomically_after_upload(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    events: list[str] = []
    s3 = RecordingS3(events=events)
    quota = RecordingQuota(events=events)
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    original_commit = service._commit_transaction

    async def recording_commit(transaction):
        await original_commit(transaction)
        events.append("pg_commit")

    monkeypatch.setattr(service, "_commit_transaction", recording_commit)

    result = await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/paper.pdf", "/")

    row = await pool.fetchrow(
        "SELECT d.id::text AS document_id, d.status::text, d.user_id, d.knowledge_base_id, "
        "j.id::text AS job_id, j.job_type, j.user_id AS job_user_id, "
        "j.knowledge_base_id AS job_kb_id, j.document_id AS job_document_id, j.payload, j.idempotency_key "
        "FROM documents d JOIN background_jobs j ON j.document_id = d.id WHERE d.id = $1::uuid",
        result["id"],
    )
    assert row is not None
    assert result == {
        "id": row["document_id"],
        "filename": "paper.pdf",
        "status": "pending",
        "already_exists": False,
        "job_id": row["job_id"],
    }
    assert row["job_type"] == "document.extract"
    assert row["job_user_id"] == row["user_id"] == user_id
    assert row["job_kb_id"] == row["knowledge_base_id"] == kb_id
    assert row["job_document_id"] == user_id.__class__(result["id"])
    assert json.loads(row["payload"]) == {"document_id": result["id"]}
    assert row["idempotency_key"] == f"document.extract:{result['id']}"
    assert s3.uploads == [f"{user_id}/{result['id']}/source.pdf"]
    assert events == ["quota_reserve", "s3_upload", "quota_renew", "pg_commit", "quota_finalize"]
    assert quota.finalized == quota.reservations
    assert quota.released == []


@pytest.mark.asyncio
async def test_url_ingest_pool_wait_then_expired_renew_never_starts_transaction(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    acquire_started = asyncio.Event()
    allow_acquire = asyncio.Event()

    class GatedAcquirePool:
        def __getattr__(self, name):
            return getattr(pool, name)

        async def acquire(self):
            acquire_started.set()
            await allow_acquire.wait()
            return await pool.acquire()

    s3 = RecordingS3()
    quota = RecordingQuota()
    service = UrlIngestService(GatedAcquirePool(), s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))
    ingest = asyncio.create_task(
        service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/acquire-expired.pdf", "/")
    )
    await acquire_started.wait()
    assert len(s3.objects) == 1
    quota.renew_result = False
    allow_acquire.set()

    with pytest.raises(HTTPException) as raised:
        await ingest

    assert raised.value.status_code == 503
    assert quota.renewed == quota.reservations
    assert quota.released == quota.reservations
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0


@pytest.mark.asyncio
async def test_url_ingest_transaction_timeout_rolls_back_and_releases_reservation(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))
    monkeypatch.setattr(url_ingest_module, "URL_TRANSACTION_TIMEOUT_SECONDS", 0.01)

    async def block_inside_transaction(_user_id, _kb_id, _url, *, conn=None):
        if conn is None:
            return
        await asyncio.Event().wait()

    monkeypatch.setattr(service, "_find_by_source_url", block_inside_transaction)

    with pytest.raises(TimeoutError):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/transaction-timeout.pdf", "/")

    assert quota.renewed == quota.reservations
    assert quota.released == quota.reservations
    assert quota.finalized == []
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0


@pytest.mark.asyncio
async def test_url_ingest_transaction_failure_deletes_exact_uploaded_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool, storage_limit_bytes=1)
    s3 = RecordingS3()
    quota = RecordingQuota(reserve_error=QuotaExceeded(15, 1))
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(HTTPException) as raised:
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/quota.pdf", "/")

    assert raised.value.status_code == 413
    assert s3.uploads == []
    assert s3.deleted_prefixes == []
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0


@pytest.mark.asyncio
async def test_url_ingest_job_failure_rolls_back_insert_and_deletes_uploaded_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()

    class FailingJobService(JobService):
        async def create_in_transaction(self, conn, command, *, authenticated_user_id):
            assert conn.is_in_transaction()
            assert await conn.fetchval("SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1)", command.document_id)
            raise RuntimeError("deterministic job insert failure")

    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, FailingJobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(RuntimeError, match="deterministic job insert failure"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/job-fail.pdf", "/")

    assert len(s3.uploads) == 1
    document_prefix = s3.uploads[0].rsplit("source.pdf", 1)[0]
    assert s3.deleted_prefixes == [document_prefix]
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0
    assert quota.released == quota.reservations
    assert quota.finalized == []


@pytest.mark.asyncio
async def test_url_ingest_upload_failure_leaves_no_database_rows(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3(upload_error=RuntimeError("storage offline secret=s3-token"))
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(HTTPException) as raised:
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/upload.pdf", "/")

    assert raised.value.status_code == 502
    assert "secret" not in raised.value.detail
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0
    assert quota.released == quota.reservations


@pytest.mark.asyncio
async def test_url_ingest_cancellation_after_upload_compensates_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)

    class CancelledAfterUpload(RecordingS3):
        async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
            await super().upload_bytes(key, data, content_type)
            raise asyncio.CancelledError

    s3 = CancelledAfterUpload()
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(asyncio.CancelledError):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/cancel.pdf", "/")

    assert len(s3.uploads) == 1
    assert s3.deleted_prefixes == [s3.uploads[0].rsplit("source.pdf", 1)[0]]
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert quota.released == quota.reservations


@pytest.mark.asyncio
async def test_url_ingest_acquire_failure_after_upload_compensates_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()

    class AcquireFailingPool:
        def __getattr__(self, name):
            return getattr(pool, name)

        async def acquire(self):
            raise RuntimeError("pool unavailable")

    service = UrlIngestService(AcquireFailingPool(), s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(RuntimeError, match="pool unavailable"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/acquire.pdf", "/")

    assert s3.objects == {}
    assert s3.deleted_prefixes == [s3.uploads[0].rsplit("source.pdf", 1)[0]]


@pytest.mark.asyncio
async def test_url_ingest_commit_error_preserves_artifact_when_document_and_job_committed(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def commit_then_fail(transaction):
        await transaction.commit()
        raise RuntimeError("connection lost after commit")

    monkeypatch.setattr(service, "_commit_transaction", commit_then_fail)

    with pytest.raises(RuntimeError, match="connection lost after commit"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/committed.pdf", "/")

    assert len(s3.objects) == 1
    assert s3.deleted_prefixes == []
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 1
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 1
    assert quota.finalized == quota.reservations
    assert quota.released == []


@pytest.mark.asyncio
async def test_url_ingest_commit_error_deletes_artifact_only_when_rollback_is_confirmed(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def fail_before_commit(_transaction):
        raise RuntimeError("commit rejected")

    monkeypatch.setattr(service, "_commit_transaction", fail_before_commit)

    with pytest.raises(RuntimeError, match="commit rejected"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/rolled-back.pdf", "/")

    assert s3.objects == {}
    assert s3.deleted_prefixes == [s3.uploads[0].rsplit("source.pdf", 1)[0]]
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert quota.released == quota.reservations
    assert quota.finalized == []


@pytest.mark.asyncio
async def test_url_ingest_unknown_commit_outcome_preserves_artifact(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def fail_before_commit(_transaction):
        raise RuntimeError("commit uncertain")

    async def unknown_outcome(*_args):
        return None

    monkeypatch.setattr(service, "_commit_transaction", fail_before_commit)
    monkeypatch.setattr(service, "_confirm_document_job_committed", unknown_outcome)

    with pytest.raises(RuntimeError, match="commit uncertain"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/unknown.pdf", "/")

    assert len(s3.objects) == 1
    assert s3.deleted_prefixes == []
    assert quota.finalized == []
    assert quota.released == []


@pytest.mark.asyncio
async def test_url_ingest_finalize_failure_never_rolls_back_committed_document(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    quota = RecordingQuota(finalize_error=ConnectionError("redis secret owner-token"))
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    result = await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/finalize-fail.pdf", "/")

    assert await pool.fetchval("SELECT count(*) FROM documents WHERE id = $1::uuid", result["id"]) == 1
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE document_id = $1::uuid", result["id"]) == 1
    assert len(s3.objects) == 1
    assert quota.finalized == quota.reservations
    assert quota.released == []


@pytest.mark.asyncio
@pytest.mark.parametrize("release_cancelled", [False, True])
async def test_url_ingest_precommit_failure_with_release_failure_preserves_original_and_cleans_source(
    pool,
    monkeypatch,
    release_cancelled,
):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()

    class BodyFailingJobs(JobService):
        async def create_in_transaction(self, conn, command, *, authenticated_user_id):
            del conn, command, authenticated_user_id
            raise RuntimeError("body insert failed")

    service = UrlIngestService(
        ReleaseFailingPool(pool, cancelled=release_cancelled),
        s3,
        BodyFailingJobs(pool),
    )
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(RuntimeError, match="body insert failed"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/release-body.pdf", "/")

    assert s3.objects == {}
    assert s3.deleted_prefixes == [s3.uploads[0].rsplit("source.pdf", 1)[0]]
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_outcome", ["committed", "rolled_back", "unknown"])
async def test_url_ingest_commit_failure_with_release_failure_still_resolves_original_outcome(
    pool,
    monkeypatch,
    commit_outcome,
):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(ReleaseFailingPool(pool), s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def fail_commit(transaction):
        if commit_outcome == "committed":
            await transaction.commit()
        raise TimeoutError("commit result lost")

    monkeypatch.setattr(service, "_commit_transaction", fail_commit)
    if commit_outcome == "unknown":

        async def unknown_result(*_args):
            return None

        monkeypatch.setattr(service, "_confirm_document_job_committed", unknown_result)

    with pytest.raises(TimeoutError, match="commit result lost"):
        await service.ingest_pdf(
            str(user_id),
            str(kb_id),
            f"https://example.test/release-{commit_outcome}.pdf",
            "/",
        )

    if commit_outcome == "rolled_back":
        assert s3.objects == {}
        assert len(s3.deleted_prefixes) == 1
    else:
        assert len(s3.objects) == 1
        assert s3.deleted_prefixes == []


@pytest.mark.asyncio
async def test_url_ingest_duplicate_commit_with_release_failure_deletes_only_temporary_source(pool, monkeypatch):
    _job, user_id, kb_id, existing_doc_id = await _seed_document_job(pool)
    source_url = "https://example.test/release-duplicate.pdf"
    await pool.execute(
        "UPDATE documents SET metadata = $2::jsonb WHERE id = $1",
        existing_doc_id,
        json.dumps({"source_url": source_url}),
    )
    existing = {
        "id": str(existing_doc_id),
        "knowledge_base_id": str(kb_id),
        "title": None,
        "path": "/",
        "filename": "paper.pdf",
        "status": "pending",
    }
    s3 = RecordingS3()
    existing_key = f"{user_id}/{existing_doc_id}/source.pdf"
    s3.objects[existing_key] = b"existing-source"
    quota = RecordingQuota()
    service = UrlIngestService(ReleaseFailingPool(pool), s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ntemporary", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def race_duplicate(_user_id, _kb_id, _url, *, conn=None):
        return existing if conn is not None else None

    monkeypatch.setattr(service, "_find_by_source_url", race_duplicate)

    with pytest.raises(RuntimeError, match="pool release failed"):
        await service.ingest_pdf(str(user_id), str(kb_id), source_url, "/")

    temporary_prefix = s3.uploads[0].rsplit("source.pdf", 1)[0]
    assert s3.deleted_prefixes == [temporary_prefix]
    assert s3.objects == {existing_key: b"existing-source"}
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE document_id = $1", existing_doc_id) == 1
    assert quota.released == quota.reservations
    assert quota.finalized == []


@pytest.mark.asyncio
async def test_url_ingest_duplicate_cleanup_finishes_after_caller_cancellation(pool, monkeypatch):
    _job, user_id, kb_id, existing_doc_id = await _seed_document_job(pool)
    source_url = "https://example.test/cancel-duplicate-cleanup.pdf"
    existing = {
        "id": str(existing_doc_id),
        "knowledge_base_id": str(kb_id),
        "title": None,
        "path": "/",
        "filename": "paper.pdf",
        "status": "pending",
    }
    s3 = RecordingS3()
    existing_key = f"{user_id}/{existing_doc_id}/source.pdf"
    s3.objects[existing_key] = b"existing-source"
    quota = RecordingQuota()
    service = UrlIngestService(pool, s3, JobService(pool), quota)
    pdf = DownloadedPdf(data=b"%PDF-1.7\ntemporary", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    async def race_duplicate(_user_id, _kb_id, _url, *, conn=None):
        return existing if conn is not None else None

    monkeypatch.setattr(service, "_find_by_source_url", race_duplicate)
    cleanup_entered = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original_delete = service._delete_uploaded_document_prefix

    async def controlled_delete(cleanup_user_id, cleanup_document_id):
        cleanup_entered.set()
        await allow_cleanup.wait()
        await original_delete(cleanup_user_id, cleanup_document_id)

    monkeypatch.setattr(service, "_delete_uploaded_document_prefix", controlled_delete)
    ingest = asyncio.create_task(service.ingest_pdf(str(user_id), str(kb_id), source_url, "/"))
    await cleanup_entered.wait()

    ingest.cancel()
    await asyncio.sleep(0)
    assert not ingest.done()
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await ingest

    temporary_prefix = s3.uploads[0].rsplit("source.pdf", 1)[0]
    assert s3.deleted_prefixes == [temporary_prefix]
    assert s3.objects == {existing_key: b"existing-source"}
    assert quota.released == quota.reservations
    assert quota.finalized == []


@pytest.mark.asyncio
async def test_url_ingest_new_document_commit_with_release_failure_preserves_committed_source(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(ReleaseFailingPool(pool), s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncommitted", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(RuntimeError, match="pool release failed"):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/release-new.pdf", "/")

    assert len(s3.objects) == 1
    assert s3.deleted_prefixes == []
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 1
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 1


@pytest.mark.asyncio
async def test_url_ingest_repeated_normalized_source_returns_same_document_and_job(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(pool, s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    downloads = 0

    async def download(_url):
        nonlocal downloads
        downloads += 1
        return pdf

    monkeypatch.setattr(service, "_download", download)
    first = await service.ingest_pdf(str(user_id), str(kb_id), " https://arxiv.org/abs/2506.06266 ", "/")
    second = await service.ingest_pdf(str(user_id), str(kb_id), "https://arxiv.org/pdf/2506.06266", "/")

    assert second["already_exists"] is True
    assert second["id"] == first["id"]
    assert second["job_id"] == first["job_id"]
    assert downloads == 1
    assert len(s3.uploads) == 1
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 1
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 1


@pytest.mark.asyncio
async def test_document_extract_handler_publishes_one_current_derived_set_on_duplicate_execution(
    pool,
    monkeypatch,
):
    job, user_id, _kb_id, doc_id = await _seed_document_job(pool)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def extracted_pages(self, source_url, ext):
        assert "signed-source" in source_url
        assert ext == "pdf"
        return [(1, "first page"), (2, "second page")]

    monkeypatch.setattr(OCRService, "_call_converter_extract", extracted_pages)

    first = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))
    second = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert first == {"document_id": str(doc_id), "derived_version": 1}
    assert second == {"document_id": str(doc_id), "derived_version": 2}
    row = await pool.fetchrow(
        "SELECT status::text, version, page_count FROM documents WHERE id = $1 AND user_id = $2",
        doc_id,
        user_id,
    )
    assert dict(row) == {"status": "ready", "version": 2, "page_count": 2}
    assert await pool.fetchval("SELECT count(*) FROM document_pages WHERE document_id = $1", doc_id) == 2
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM document_pages WHERE document_id = $1 AND document_version = 2",
            doc_id,
        )
        == 2
    )
    assert (
        await pool.fetchval(
            "SELECT count(DISTINCT document_version) FROM document_pages WHERE document_id = $1",
            doc_id,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_document_extract_rejects_payload_mismatch_before_storage_side_effect(pool):
    job, _user_id, _kb_id, _doc_id = await _seed_document_job(pool)
    s3 = RecordingS3()
    mismatched = replace(job, payload={"document_id": str(uuid4())})

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_extract(mismatched, _lease(pool, job), _context(pool, s3))

    assert raised.value.error_code == "invalid_document_job"
    assert s3.presigned_gets == []


@pytest.mark.asyncio
async def test_document_extract_rejects_cross_tenant_document_before_storage_side_effect(pool):
    _original_job, _owner_id, _owner_kb_id, doc_id = await _seed_document_job(pool)
    attacker_id, attacker_kb_id = await _seed_tenant(pool)
    command = JobCreate(
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=attacker_id,
        knowledge_base_id=attacker_kb_id,
        document_id=doc_id,
        payload={"document_id": str(doc_id)},
        idempotency_key=f"cross-tenant:{doc_id}",
    )
    async with pool.acquire() as conn, conn.transaction():
        created = await repository.create(conn, command)
    attacker_job = await repository.claim(pool, created.id, "worker-attacker", 120)
    assert attacker_job is not None
    attacker_lease = JobLease(pool, attacker_job.id, "worker-attacker", 120, 30)
    s3 = RecordingS3()

    async with pool.acquire() as blocker, blocker.transaction():
        await blocker.fetchrow("SELECT id FROM documents WHERE id = $1 FOR UPDATE", doc_id)
        with pytest.raises(TerminalJobError) as raised:
            await asyncio.wait_for(
                handle_document_extract(attacker_job, attacker_lease, _context(pool, s3)),
                timeout=0.5,
            )

    assert raised.value.error_code == "document_not_found"
    assert s3.presigned_gets == []


@pytest.mark.asyncio
async def test_document_extract_rejects_missing_document_terminally_before_storage(pool):
    user_id, kb_id = await _seed_tenant(pool)
    doc_id = uuid4()
    missing = JobRecord(
        id=uuid4(),
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        payload={"document_id": str(doc_id)},
    )
    s3 = RecordingS3()

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_extract(missing, ScriptedLease(), _context(pool, s3))

    assert raised.value.error_code == "document_not_found"
    assert s3.presigned_gets == []


@pytest.mark.asyncio
async def test_document_extract_rejects_unsupported_type_terminally_and_marks_failed(pool):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool, filename="payload.exe")
    s3 = RecordingS3()

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert raised.value.error_code == "unsupported_document_type"
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "failed"
    assert s3.presigned_gets == []


@pytest.mark.asyncio
async def test_document_extract_transient_converter_error_retries_without_secret_and_restores_pending(
    pool,
    monkeypatch,
    caplog,
):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def converter_offline(self, source_url, ext):
        del self, source_url, ext
        raise httpx.ConnectError("credential=converter-secret url=https://private.invalid")

    monkeypatch.setattr(OCRService, "_call_converter_extract", converter_offline)

    with pytest.raises(RetryableJobError) as raised:
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert raised.value.error_code == "extraction_transient"
    assert raised.value.error_message == "Document extraction will be retried."
    assert "converter-secret" not in caplog.text
    assert "private.invalid" not in caplog.text
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "processing"


@pytest.mark.asyncio
async def test_document_extract_transient_s3_error_retries_without_secret(pool, monkeypatch, caplog):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    class FailingS3(RecordingS3):
        async def generate_presigned_get(self, key: str) -> str:
            del key
            raise EndpointConnectionError(endpoint_url="https://credential@s3.private.invalid")

    with pytest.raises(RetryableJobError) as raised:
        await handle_document_extract(job, _lease(pool, job), _context(pool, FailingS3()))

    assert raised.value.error_code == "extraction_transient"
    assert "credential" not in raised.value.error_message
    assert "s3.private.invalid" not in caplog.text
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "processing"


@pytest.mark.asyncio
async def test_document_extract_last_retry_marks_document_failed(pool, monkeypatch):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool, max_attempts=1)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def converter_offline(self, source_url, ext):
        del self, source_url, ext
        raise httpx.ReadTimeout("offline")

    monkeypatch.setattr(OCRService, "_call_converter_extract", converter_offline)

    with pytest.raises(RetryableJobError):
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert job.attempt_count == job.max_attempts == 1
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["retryable", "terminal", "unsupported"])
async def test_reextract_failure_preserves_ready_current_version(pool, monkeypatch, failure_kind):
    filename = "payload.exe" if failure_kind == "unsupported" else "paper.pdf"
    job, user_id, kb_id, doc_id = await _seed_document_job(
        pool,
        filename=filename,
        status="ready",
        version=3,
    )
    await pool.execute(
        "INSERT INTO document_pages (document_id,page,content,document_version) VALUES ($1,1,'stable',3)",
        doc_id,
    )
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'stable','stable',1,3)",
        doc_id,
        user_id,
        kb_id,
    )
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    if failure_kind == "retryable":

        async def fail(self, source_url, ext):
            del self, source_url, ext
            raise httpx.ConnectError("offline")
    else:

        async def fail(self, source_url, ext):
            del self, source_url, ext
            return [(1, "over quota")]

        if failure_kind == "terminal":
            await pool.execute("UPDATE users SET page_limit = 0 WHERE id = $1", user_id)
    monkeypatch.setattr(OCRService, "_call_converter_extract", fail)

    error_type = RetryableJobError if failure_kind == "retryable" else TerminalJobError
    with pytest.raises(error_type):
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    row = await pool.fetchrow("SELECT status::text, version FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"status": "ready", "version": 3}
    assert await pool.fetchval("SELECT content FROM document_pages WHERE document_id = $1", doc_id) == "stable"
    assert await pool.fetchval("SELECT content FROM document_chunks WHERE document_id = $1", doc_id) == "stable"


@pytest.mark.asyncio
async def test_document_extract_quota_error_is_terminal_and_marks_failed(pool, monkeypatch):
    job, user_id, _kb_id, doc_id = await _seed_document_job(pool)
    await pool.execute("UPDATE users SET page_limit = 0 WHERE id = $1", user_id)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def extracted_pages(self, source_url, ext):
        del self, source_url, ext
        return [(1, "over quota")]

    monkeypatch.setattr(OCRService, "_call_converter_extract", extracted_pages)

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert raised.value.error_code == "quota_exceeded"
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "failed"
    assert await pool.fetchval("SELECT count(*) FROM document_pages WHERE document_id = $1", doc_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [LeaseLost("lost"), JobCancelled("cancelled")])
async def test_document_extract_final_checkpoint_prevents_publish_and_preserves_previous_version(
    pool,
    monkeypatch,
    failure,
):
    job, user_id, kb_id, doc_id = await _seed_document_job(pool, status="ready", version=1)
    await pool.execute(
        "INSERT INTO document_pages (document_id, page, content, document_version) VALUES ($1, 1, 'old', 1)",
        doc_id,
    )
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,user_id,knowledge_base_id,chunk_index,content,source_content,token_count,document_version) "
        "VALUES ($1,$2,$3,0,'old','old',1,1)",
        doc_id,
        user_id,
        kb_id,
    )
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def extracted_pages(self, source_url, ext):
        del self, source_url, ext
        return [(1, "replacement")]

    monkeypatch.setattr(OCRService, "_call_converter_extract", extracted_pages)
    lease = ScriptedLease(failure)

    with pytest.raises(type(failure)):
        await handle_document_extract(job, lease, _context(pool, s3))

    row = await pool.fetchrow("SELECT status::text, version, content FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"status": "ready", "version": 1, "content": None}
    assert await pool.fetchval("SELECT content FROM document_pages WHERE document_id = $1", doc_id) == "old"
    assert await pool.fetchval("SELECT content FROM document_chunks WHERE document_id = $1", doc_id) == "old"


@pytest.mark.asyncio
async def test_document_extract_failure_status_write_is_lease_fenced(pool, monkeypatch):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def converter_offline(self, source_url, ext):
        del self, source_url, ext
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(OCRService, "_call_converter_extract", converter_offline)
    lease = ScriptedLease(LeaseLost("lost before status transition"))

    with pytest.raises(LeaseLost):
        await handle_document_extract(job, lease, _context(pool, s3))

    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "processing"


@pytest.mark.asyncio
async def test_image_extraction_final_update_is_fenced_in_transaction(pool):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(
        pool,
        filename="image.png",
        status="ready",
        version=4,
    )
    lease = ScriptedLease(LeaseLost("lost before image publish"))

    with pytest.raises(LeaseLost):
        await handle_document_extract(job, lease, _context(pool, RecordingS3()))

    row = await pool.fetchrow("SELECT status::text, version, page_count FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"status": "ready", "version": 4, "page_count": None}


@pytest.mark.asyncio
async def test_durable_spreadsheet_handler_accepts_artifact_namespace(pool):
    job, user_id, _kb_id, doc_id = await _seed_document_job(pool, filename="sheet.csv")
    s3 = RecordingS3()
    s3.objects[f"{user_id}/{doc_id}/source.csv"] = b"name,value\nalpha,1\n"

    version = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    row = await pool.fetchrow("SELECT status::text, version, parser FROM documents WHERE id = $1", doc_id)
    assert version == {"document_id": str(doc_id), "derived_version": 1}
    assert dict(row) == {"status": "ready", "version": 1, "parser": "openpyxl"}


@pytest.mark.asyncio
async def test_html_cancel_before_final_publish_does_not_overwrite_stable_artifact(pool, monkeypatch):
    job, user_id, _kb_id, doc_id = await _seed_document_job(pool, filename="page.html")
    s3 = RecordingS3()
    stable_key = f"{user_id}/{doc_id}/tagged.html"
    source_key = f"{user_id}/{doc_id}/source.html"
    s3.objects[stable_key] = b"stable-old"
    s3.objects[source_key] = b"<main>new</main>"

    html_parser = ModuleType("html_parser")

    class Parser:
        def __init__(self, raw_html, content_only=True):
            del raw_html, content_only

        def parse(self):
            return SimpleNamespace(content="new markdown")

        async def embed_images(self):
            return None

        def html(self, sanitize=True):
            assert sanitize
            return "<main>stable-new</main>"

    html_parser.Parser = Parser
    monkeypatch.setitem(sys.modules, "html_parser", html_parser)
    lease = ScriptedLease(JobCancelled("cancel before publish"))

    with pytest.raises(JobCancelled):
        await handle_document_extract(job, lease, _context(pool, s3))

    assert s3.objects[stable_key] == b"stable-old"
    assert not [key for key in s3.objects if "/derived/" in key]
    assert await pool.fetchval("SELECT version FROM documents WHERE id = $1", doc_id) == 0


@pytest.mark.asyncio
async def test_mistral_asset_upload_failure_cannot_publish_ready_or_leave_attempt_artifacts(
    pool,
    monkeypatch,
):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)

    class AssetFailingS3(RecordingS3):
        async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
            await super().upload_bytes(key, data, content_type)
            if key.endswith(".jpg"):
                raise EndpointConnectionError(endpoint_url="https://s3.private.invalid")

    s3 = AssetFailingS3()
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        encoded = base64.b64encode(b"jpeg-bytes").decode()
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": "page with image",
                    "images": [{"id": "img-1", "image_base64": encoded}],
                }
            ]
        }

    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)

    with pytest.raises(RetryableJobError):
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    row = await pool.fetchrow("SELECT status::text, version FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"status": "processing", "version": 0}
    assert await pool.fetchval("SELECT count(*) FROM document_pages WHERE document_id = $1", doc_id) == 0
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE metadata->>'parent_document_id' = $1",
            str(doc_id),
        )
        == 0
    )
    assert not [key for key in s3.objects if "/derived/" in key]


@pytest.mark.asyncio
async def test_versioned_artifact_staging_does_not_hold_final_job_or_document_locks(pool, monkeypatch):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)

    class GatedArtifactS3(RecordingS3):
        def __init__(self):
            super().__init__()
            self.upload_started = asyncio.Event()
            self.release_upload = asyncio.Event()

        async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
            if "/derived/" in key and key.endswith("/ocr.json"):
                self.upload_started.set()
                await self.release_upload.wait()
            await super().upload_bytes(key, data, content_type)

    s3 = GatedArtifactS3()
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {"pages": [{"index": 0, "markdown": "slow staged artifact"}]}

    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)
    extraction = asyncio.create_task(handle_document_extract(job, _lease(pool, job), _context(pool, s3)))
    await asyncio.wait_for(s3.upload_started.wait(), timeout=1)

    async def heartbeat_once():
        async with pool.acquire() as conn, conn.transaction():
            return await repository.heartbeat(conn, job.id, "worker-extraction", 120)

    heartbeat = asyncio.create_task(heartbeat_once())
    heartbeat_completed_during_upload = True
    try:
        await asyncio.wait_for(asyncio.shield(heartbeat), timeout=1)
    except TimeoutError:
        heartbeat_completed_during_upload = False
    finally:
        s3.release_upload.set()

    await extraction
    await heartbeat

    assert heartbeat_completed_during_upload
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "ready"


@pytest.mark.asyncio
async def test_successful_reextraction_cleans_previous_published_asset_after_commit(pool, monkeypatch):
    job, user_id, kb_id, doc_id = await _seed_document_job(pool, status="ready", version=1)
    old_asset_id = uuid4()
    old_key = f"{user_id}/{old_asset_id}/source.jpg"
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, title, source_kind, file_type, "
        "file_size, status, metadata, version) "
        "VALUES ($1, $2, $3, 'old.jpg', '/paper.assets/', 'old.jpg', 'asset', 'jpg', "
        "3, 'ready', $4::jsonb, 1)",
        old_asset_id,
        kb_id,
        user_id,
        json.dumps({"parent_document_id": str(doc_id), "asset": True}),
    )
    s3 = RecordingS3()
    s3.objects[old_key] = b"old"
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": "new page",
                    "images": [
                        {
                            "id": "new-image",
                            "image_base64": base64.b64encode(b"new-jpeg").decode(),
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)

    assert await handle_document_extract(job, _lease(pool, job), _context(pool, s3)) == {
        "document_id": str(doc_id),
        "derived_version": 2,
    }

    assert old_key not in s3.objects
    new_asset_keys = [key for key in s3.objects if key.endswith(".jpg")]
    assert len(new_asset_keys) == 1
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE id = $1", old_asset_id) == 0


@pytest.mark.asyncio
async def test_legacy_html_publish_clears_durable_pointer_before_deleting_old_artifact(pool, monkeypatch):
    _job, user_id, _kb_id, doc_id = await _seed_document_job(pool, filename="page.html")
    old_key = f"{user_id}/{doc_id}/derived/old-job/attempt-1/tagged.html"
    await pool.execute(
        "UPDATE documents SET metadata = $2::jsonb WHERE id = $1",
        doc_id,
        json.dumps({"tagged_s3_key": old_key}),
    )
    s3 = RecordingS3()
    s3.objects[old_key] = b"old durable"
    s3.objects[f"{user_id}/{doc_id}/source.html"] = b"<main>legacy</main>"
    html_parser = ModuleType("html_parser")

    class Parser:
        def __init__(self, raw_html, content_only=True):
            del raw_html, content_only

        def parse(self):
            return SimpleNamespace(content="legacy markdown")

        async def embed_images(self):
            return None

        def html(self, sanitize=True):
            assert sanitize
            return "<main>legacy tagged</main>"

    html_parser.Parser = Parser
    monkeypatch.setitem(sys.modules, "html_parser", html_parser)

    version = await OCRService(s3, pool)._do_process(str(doc_id), str(user_id))

    metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert version == 1
    assert metadata["tagged_s3_key"] is None
    assert old_key not in s3.objects
    assert s3.objects[f"{user_id}/{doc_id}/tagged.html"] == b"<main>legacy tagged</main>"


@pytest.mark.asyncio
async def test_durable_office_cancel_after_conversion_preserves_current_pointer_and_cleans_attempt(
    pool,
    monkeypatch,
):
    job, user_id, _kb_id, doc_id = await _seed_document_job(
        pool,
        filename="paper.docx",
        status="ready",
        version=3,
    )
    old_key = f"{user_id}/{doc_id}/derived/old-job/attempt-1/converted.pdf"
    new_key = f"{user_id}/{doc_id}/derived/{job.id}/attempt-{job.attempt_count}/converted.pdf"
    await pool.execute(
        "UPDATE documents SET metadata = $2::jsonb WHERE id = $1",
        doc_id,
        json.dumps({"converted_s3_key": old_key}),
    )
    s3 = RecordingS3()
    s3.objects[old_key] = b"current-pdf"
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def converted(self, document_id, user_id, source_key, ext, *, artifact_namespace=None):
        del self, document_id, user_id, source_key, ext
        assert artifact_namespace == f"{job.id}/attempt-{job.attempt_count}"
        s3.objects[new_key] = b"attempt-pdf"
        return new_key

    async def cancel_after_conversion(self, url, url_type="document_url"):
        del self, url, url_type
        raise JobCancelled("cancelled after conversion")

    monkeypatch.setattr(OCRService, "_convert_to_pdf_s3", converted)
    monkeypatch.setattr(OCRService, "_call_mistral_ocr", cancel_after_conversion)

    with pytest.raises(JobCancelled):
        await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata["converted_s3_key"] == old_key
    assert s3.objects[old_key] == b"current-pdf"
    assert new_key not in s3.objects


@pytest.mark.asyncio
async def test_durable_office_success_switches_converted_pointer_and_cleans_previous(pool, monkeypatch):
    job, user_id, kb_id, doc_id = await _seed_document_job(
        pool,
        filename="paper.docx",
        status="ready",
        version=2,
    )
    old_key = f"{user_id}/{doc_id}/derived/old-job/attempt-1/converted.pdf"
    new_key = f"{user_id}/{doc_id}/derived/{job.id}/attempt-{job.attempt_count}/converted.pdf"
    await pool.execute(
        "UPDATE documents SET metadata = $2::jsonb WHERE id = $1",
        doc_id,
        json.dumps({"converted_s3_key": old_key}),
    )
    s3 = RecordingS3()
    s3.objects[old_key] = b"old-pdf"
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def converted(self, document_id, user_id, source_key, ext, *, artifact_namespace=None):
        del self, document_id, user_id, source_key, ext
        assert artifact_namespace == f"{job.id}/attempt-{job.attempt_count}"
        s3.objects[new_key] = b"new-pdf"
        return new_key

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {"pages": [{"index": 0, "markdown": "converted office"}]}

    monkeypatch.setattr(OCRService, "_convert_to_pdf_s3", converted)
    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)

    result = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert result["derived_version"] == 3
    assert metadata["converted_s3_key"] == new_key
    assert old_key not in s3.objects
    assert s3.objects[new_key] == b"new-pdf"

    from services.hosted import HostedDocumentService, HostedPublicWikiService

    await HostedDocumentService(pool, str(user_id), s3).get_url(str(doc_id))
    assert s3.presigned_gets[-1] == new_key
    await pool.execute(
        "UPDATE knowledge_bases SET visibility = 'public', public_slug = 'converted-office' WHERE id = $1",
        kb_id,
    )
    await pool.execute(
        "UPDATE documents SET path = '/wiki/', document_number = 42 WHERE id = $1",
        doc_id,
    )
    assert await HostedPublicWikiService(pool).get_asset_key("converted-office", 42) == new_key


@pytest.mark.asyncio
async def test_office_backend_switch_preserves_unreplaced_mistral_artifact_pointers(pool, monkeypatch):
    job, user_id, _kb_id, doc_id = await _seed_document_job(pool, filename="paper.docx")
    converted_key = f"{user_id}/{doc_id}/derived/{job.id}/attempt-{job.attempt_count}/converted.pdf"
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def converted(self, document_id, user_id, source_key, ext, *, artifact_namespace=None):
        del self, document_id, user_id, source_key, ext
        assert artifact_namespace == f"{job.id}/attempt-{job.attempt_count}"
        s3.objects[converted_key] = b"mistral-converted"
        return converted_key

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {"pages": [{"index": 0, "markdown": "mistral office"}]}

    monkeypatch.setattr(OCRService, "_convert_to_pdf_s3", converted)
    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)
    await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    first_metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(first_metadata, str):
        first_metadata = json.loads(first_metadata)
    ocr_key = first_metadata["ocr_s3_key"]
    assert s3.objects[converted_key] == b"mistral-converted"
    assert ocr_key in s3.objects

    monkeypatch.setattr(settings, "PDF_BACKEND", "opendataloader")
    monkeypatch.setattr(settings, "CONVERTER_URL", "https://converter.invalid")

    async def extracted_pages(self, source_url, ext):
        del self, source_url, ext
        return [(1, "opendataloader office")]

    monkeypatch.setattr(OCRService, "_call_converter_extract", extracted_pages)
    await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    second_metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(second_metadata, str):
        second_metadata = json.loads(second_metadata)
    assert second_metadata["converted_s3_key"] == converted_key
    assert second_metadata["ocr_s3_key"] == ocr_key
    assert s3.objects[converted_key] == b"mistral-converted"
    assert ocr_key in s3.objects


@pytest.mark.asyncio
async def test_same_office_job_attempt_reexecution_preserves_current_converted_artifact(pool, monkeypatch):
    job, user_id, kb_id, doc_id = await _seed_document_job(pool, filename="paper.docx")
    current_key = f"{user_id}/{doc_id}/derived/{job.id}/attempt-{job.attempt_count}/converted.pdf"
    s3 = RecordingS3()
    conversions = 0
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def converted(self, document_id, user_id, source_key, ext, *, artifact_namespace=None):
        nonlocal conversions
        del self, document_id, user_id, source_key, ext
        assert artifact_namespace == f"{job.id}/attempt-{job.attempt_count}"
        conversions += 1
        s3.objects[current_key] = f"converted-{conversions}".encode()
        return current_key

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {"pages": [{"index": 0, "markdown": "idempotent office"}]}

    monkeypatch.setattr(OCRService, "_convert_to_pdf_s3", converted)
    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)

    first = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))
    second = await handle_document_extract(job, _lease(pool, job), _context(pool, s3))

    assert first["derived_version"] == 1
    assert second["derived_version"] == 2
    assert conversions == 2
    assert s3.objects[current_key] == b"converted-2"

    from services.hosted import HostedDocumentService, HostedPublicWikiService

    await HostedDocumentService(pool, str(user_id), s3).get_url(str(doc_id))
    assert s3.presigned_gets[-1] == current_key
    await pool.execute(
        "UPDATE knowledge_bases SET visibility = 'public', public_slug = 'repeated-office' WHERE id = $1",
        kb_id,
    )
    await pool.execute(
        "UPDATE documents SET path = '/wiki/', document_number = 43 WHERE id = $1",
        doc_id,
    )
    assert await HostedPublicWikiService(pool).get_asset_key("repeated-office", 43) == current_key


@pytest.mark.asyncio
async def test_legacy_office_publish_clears_durable_converted_pointer_and_uses_fixed_key(pool, monkeypatch):
    _job, user_id, _kb_id, doc_id = await _seed_document_job(
        pool,
        filename="paper.docx",
        status="ready",
        version=1,
    )
    old_key = f"{user_id}/{doc_id}/derived/old-job/attempt-1/converted.pdf"
    fixed_key = f"{user_id}/{doc_id}/converted.pdf"
    await pool.execute(
        "UPDATE documents SET metadata = $2::jsonb WHERE id = $1",
        doc_id,
        json.dumps({"converted_s3_key": old_key}),
    )
    s3 = RecordingS3()
    s3.objects[old_key] = b"durable-pdf"
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")

    async def converted(self, document_id, user_id, source_key, ext, *, artifact_namespace=None):
        del self, document_id, user_id, source_key, ext
        assert artifact_namespace is None
        s3.objects[fixed_key] = b"legacy-pdf"
        return fixed_key

    async def mistral_result(self, url, url_type="document_url"):
        del self, url, url_type
        return {"pages": [{"index": 0, "markdown": "legacy office"}]}

    monkeypatch.setattr(OCRService, "_convert_to_pdf_s3", converted)
    monkeypatch.setattr(OCRService, "_call_mistral_ocr", mistral_result)

    assert await OCRService(s3, pool)._do_process(str(doc_id), str(user_id)) == 2

    metadata = await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata["converted_s3_key"] is None
    assert old_key not in s3.objects
    assert s3.objects[fixed_key] == b"legacy-pdf"

    from services.hosted import HostedDocumentService

    await HostedDocumentService(pool, str(user_id), s3).get_url(str(doc_id))
    assert s3.presigned_gets[-1] == fixed_key


@pytest.mark.asyncio
async def test_durable_converter_failure_after_put_cleans_versioned_converted_object(monkeypatch):
    namespace = "job-id/attempt-1"
    expected_key = f"user-id/doc-id/derived/{namespace}/converted.pdf"

    class PutThenFailS3(RecordingS3):
        async def generate_presigned_put(self, key: str, content_type: str = "application/pdf") -> str:
            del content_type
            self.objects[key] = b"converter-put"
            return "https://storage.invalid/signed-put"

    class FailedResponse:
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            request = httpx.Request("POST", "https://converter.invalid/convert")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("converter failed", request=request, response=response)

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        async def post(self, url, json, headers):
            del url, json, headers
            return FailedResponse()

    s3 = PutThenFailS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "https://converter.invalid")
    monkeypatch.setattr(settings, "PDF_BACKEND", "mistral")
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "secret")
    monkeypatch.setattr("services.ocr.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())

    with pytest.raises(httpx.HTTPStatusError, match="converter failed"):
        await OCRService(s3, pool=None)._process_office(
            "doc-id",
            "user-id",
            "kb-id",
            "user-id/doc-id/source.docx",
            "docx",
            artifact_namespace=namespace,
        )

    assert expected_key not in s3.objects


@pytest.mark.asyncio
async def test_archived_immediately_before_final_write_cannot_publish_derived_rows(pool, monkeypatch):
    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)
    s3 = RecordingS3()
    monkeypatch.setattr(settings, "CONVERTER_URL", "http://converter.internal")

    async def extracted_pages(self, source_url, ext):
        del self, source_url, ext
        return [(1, "must not publish")]

    monkeypatch.setattr(OCRService, "_call_converter_extract", extracted_pages)

    class ArchiveBeforeFinal(ScriptedLease):
        async def checkpoint(self, conn=None):
            await super().checkpoint(conn)
            if conn is not None and self.connection_calls == 2:
                await conn.execute("UPDATE documents SET archived = true WHERE id = $1", doc_id)

    with pytest.raises(TerminalJobError):
        await handle_document_extract(job, ArchiveBeforeFinal(), _context(pool, s3))

    row = await pool.fetchrow("SELECT archived, version FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"archived": False, "version": 0}
    assert await pool.fetchval("SELECT count(*) FROM document_pages WHERE document_id = $1", doc_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM document_chunks WHERE document_id = $1", doc_id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_phase", ["prepare", "failure"])
async def test_ensure_and_worker_document_transition_share_job_then_document_lock_order(pool, worker_phase):
    job, user_id, kb_id, doc_id = await _seed_document_job(pool)
    lease = GatedDatabaseLease(job)
    if worker_phase == "prepare":
        worker = asyncio.create_task(_prepare_document_extraction(job, lease, _context(pool, None), {"pdf"}))
    else:
        worker = asyncio.create_task(_set_extraction_failure_status(job, lease, _context(pool, None), retryable=True))

    ensure_conn = await pool.acquire()
    observer = await pool.acquire()
    transaction = ensure_conn.transaction()
    ensure = None
    committed = False
    try:
        await asyncio.wait_for(lease.job_locked.wait(), timeout=1)
        await transaction.start()
        await ensure_conn.execute("SET LOCAL lock_timeout = '2s'")
        ensure_pid = await ensure_conn.fetchval("SELECT pg_backend_pid()")
        ensure = asyncio.create_task(
            JobService(pool).ensure_document_extraction_in_transaction(
                ensure_conn,
                document_id=doc_id,
                user_id=user_id,
                knowledge_base_id=kb_id,
                restart_terminal=True,
            )
        )
        await _wait_for_blocked_backend(observer, ensure_pid)
        lease.release_job.set()
        async with asyncio.timeout(3):
            worker_result, (ensured, created) = await asyncio.gather(worker, ensure)
        assert worker_result is None
        assert ensured.id == job.id
        assert not created
        assert await ensure_conn.fetchval("SELECT 1") == 1
        await transaction.commit()
        committed = True
    finally:
        lease.release_job.set()
        for task in (worker, ensure):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (worker, ensure) if task is not None), return_exceptions=True)
        if not committed:
            with suppress(Exception):
                await transaction.rollback()
        await pool.release(observer)
        await pool.release(ensure_conn)

    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "processing"


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_kind", ["derived", "image"])
async def test_url_and_publish_share_advisory_job_document_lock_order(pool, publish_kind):
    filename = "image.png" if publish_kind == "image" else "paper.pdf"
    job, user_id, kb_id, doc_id = await _seed_document_job(pool, filename=filename)
    url_conn = await pool.acquire()
    observer = await pool.acquire()
    transaction = url_conn.transaction()
    publish = None
    committed = False

    async def final_checkpoint(conn):
        await conn.execute("SET LOCAL lock_timeout = '2s'")
        await repository.assert_active(conn, job.id, "worker-extraction")

    try:
        await transaction.start()
        await url_conn.execute("SET LOCAL lock_timeout = '2s'")
        url_pid = await url_conn.fetchval("SELECT pg_backend_pid()")
        await url_conn.execute("SELECT pg_advisory_xact_lock(hashtext($1::text))", str(user_id))
        service = OCRService(RecordingS3(), pool)
        if publish_kind == "image":
            operation = service._process_image(
                str(doc_id),
                str(user_id),
                str(kb_id),
                f"{user_id}/{doc_id}/source.png",
                "png",
                before_write=final_checkpoint,
            )
        else:
            operation = service._commit_derived_content(
                str(doc_id),
                str(user_id),
                str(kb_id),
                pages=[(1, "published")],
                chunks=[],
                parser="lock-order-test",
                before_write=final_checkpoint,
            )
        publish = asyncio.create_task(operation)
        async with asyncio.timeout(1):
            while True:
                blocked_pids = await observer.fetchval(
                    "SELECT COALESCE(array_agg(pid), '{}') FROM pg_stat_activity "
                    "WHERE pid <> $1 AND wait_event_type = 'Lock' AND $1 = ANY(pg_blocking_pids(pid))",
                    url_pid,
                )
                if blocked_pids:
                    break

        ensured, created = await JobService(pool).ensure_document_extraction_in_transaction(
            url_conn,
            document_id=doc_id,
            user_id=user_id,
            knowledge_base_id=kb_id,
            restart_terminal=True,
        )
        assert ensured.id == job.id
        assert not created
        assert await url_conn.fetchval("SELECT 1") == 1
        await transaction.commit()
        committed = True
        async with asyncio.timeout(3):
            assert await publish == 1
    finally:
        if publish is not None and not publish.done():
            publish.cancel()
        if publish is not None:
            await asyncio.gather(publish, return_exceptions=True)
        if not committed:
            with suppress(Exception):
                await transaction.rollback()
        await pool.release(observer)
        await pool.release(url_conn)

    row = await pool.fetchrow("SELECT status::text, version, page_count FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {"status": "ready", "version": 1, "page_count": 1}


@pytest.mark.asyncio
async def test_two_startup_replicas_do_not_prelock_document_before_authoritative_job(pool):
    from main import _recover_durable_extraction_jobs

    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool)
    lease = GatedDatabaseLease(job)
    worker = asyncio.create_task(_prepare_document_extraction(job, lease, _context(pool, None), {"pdf"}))
    entered = [asyncio.Event(), asyncio.Event()]
    pids: list[int] = []

    class ObservedJobService(JobService):
        def __init__(self, pool, index):
            super().__init__(pool)
            self.index = index

        async def ensure_document_extraction_in_transaction(self, conn, **kwargs):
            await conn.execute("SET LOCAL lock_timeout = '2s'")
            pids.append(await conn.fetchval("SELECT pg_backend_pid()"))
            entered[self.index].set()
            return await super().ensure_document_extraction_in_transaction(conn, **kwargs)

    recoveries = []
    observer = await pool.acquire()
    try:
        await asyncio.wait_for(lease.job_locked.wait(), timeout=1)
        recoveries = [
            asyncio.create_task(_recover_durable_extraction_jobs(pool, ObservedJobService(pool, index)))
            for index in range(2)
        ]
        async with asyncio.timeout(1):
            await asyncio.gather(*(event.wait() for event in entered))
        for pid in pids:
            await _wait_for_blocked_backend(observer, pid)
        lease.release_job.set()
        async with asyncio.timeout(3):
            worker_result, first, second = await asyncio.gather(worker, *recoveries)
        assert worker_result is None
        assert [record for record in (*first, *second) if record.document_id == doc_id] == []
    finally:
        lease.release_job.set()
        for task in (worker, *recoveries):
            if not task.done():
                task.cancel()
        await asyncio.gather(worker, *recoveries, return_exceptions=True)
        await pool.release(observer)


@pytest.mark.asyncio
async def test_startup_recovery_creates_only_missing_source_extraction_jobs_idempotently(pool):
    from main import _recover_durable_extraction_jobs

    user_id, kb_id = await _seed_tenant(pool)
    documents = []
    for filename, status, source_kind, archived in (
        ("pending.pdf", "pending", "source", False),
        ("processing.pdf", "processing", "source", False),
        ("ready.pdf", "ready", "source", False),
        ("asset.png", "pending", "asset", False),
        ("archived.pdf", "pending", "source", True),
    ):
        doc_id = uuid4()
        documents.append((doc_id, filename, status, source_kind, archived))
        await pool.execute(
            "INSERT INTO documents "
            "(id, knowledge_base_id, user_id, filename, path, file_type, status, source_kind, archived) "
            "VALUES ($1, $2, $3, $4, '/', $5, $6, $7, $8)",
            doc_id,
            kb_id,
            user_id,
            filename,
            filename.rsplit(".", 1)[-1],
            status,
            source_kind,
            archived,
        )

    recovered_first = await _recover_durable_extraction_jobs(pool, JobService(pool))
    recovered_second = await _recover_durable_extraction_jobs(pool, JobService(pool))

    expected_ids = {documents[0][0], documents[1][0]}
    assert {record.document_id for record in recovered_first if record.user_id == user_id} == expected_ids
    assert recovered_second == []
    rows = await pool.fetch(
        "SELECT document_id, user_id, knowledge_base_id, payload, idempotency_key "
        "FROM background_jobs WHERE user_id = $1 AND document_id = ANY($2::uuid[]) ORDER BY document_id",
        user_id,
        list(expected_ids),
    )
    assert len(rows) == 2
    for row in rows:
        assert row["user_id"] == user_id
        assert row["knowledge_base_id"] == kb_id
        assert json.loads(row["payload"]) == {"document_id": str(row["document_id"])}
        assert row["idempotency_key"] == f"document.extract:{row['document_id']}"


@pytest.mark.asyncio
async def test_drifted_ready_document_with_succeeded_job_gets_one_successor_across_replicas(pool):
    from main import _recover_durable_extraction_jobs, _repair_hosted_derived_drift

    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool, status="ready", version=2)
    await pool.execute(
        "UPDATE background_jobs SET state = 'succeeded', lease_owner = NULL, lease_expires_at = NULL WHERE id = $1",
        job.id,
    )
    await pool.execute(
        "INSERT INTO document_pages (document_id, page, content, document_version) VALUES ($1, 1, 'stale', 1)",
        doc_id,
    )

    repaired = await _repair_hosted_derived_drift(pool)
    await pool.execute(
        "UPDATE documents SET status = 'failed' WHERE id <> $1 "
        "AND status IN ('pending', 'processing') AND NOT archived AND source_kind = 'source'",
        doc_id,
    )
    first_locked_old_job = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    second_pid = None

    class FirstReplicaJobService(JobService):
        async def ensure_document_extraction_in_transaction(self, conn, **kwargs):
            if kwargs["document_id"] == doc_id:
                await conn.execute("SET LOCAL lock_timeout = '2s'")
                await conn.fetchval("SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE", job.id)
                first_locked_old_job.set()
                await release_first.wait()
            return await super().ensure_document_extraction_in_transaction(conn, **kwargs)

    class SecondReplicaJobService(JobService):
        async def ensure_document_extraction_in_transaction(self, conn, **kwargs):
            nonlocal second_pid
            if kwargs["document_id"] == doc_id:
                await conn.execute("SET LOCAL lock_timeout = '2s'")
                second_pid = await conn.fetchval("SELECT pg_backend_pid()")
                second_entered.set()
            return await super().ensure_document_extraction_in_transaction(conn, **kwargs)

    observer = await pool.acquire()
    first_recovery = asyncio.create_task(_recover_durable_extraction_jobs(pool, FirstReplicaJobService(pool)))
    second_recovery = None
    try:
        await asyncio.wait_for(first_locked_old_job.wait(), timeout=1)
        second_recovery = asyncio.create_task(_recover_durable_extraction_jobs(pool, SecondReplicaJobService(pool)))
        await asyncio.wait_for(second_entered.wait(), timeout=1)
        assert second_pid is not None
        await _wait_for_blocked_backend(observer, second_pid)
        release_first.set()
        async with asyncio.timeout(3):
            recovered = await asyncio.gather(first_recovery, second_recovery)
    finally:
        release_first.set()
        for task in (first_recovery, second_recovery):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first_recovery, second_recovery) if task is not None),
            return_exceptions=True,
        )
        await pool.release(observer)

    assert [row["id"] for row in repaired] == [doc_id]
    assert sum(record.document_id == doc_id for records in recovered for record in records) == 1
    jobs = await pool.fetch(
        "SELECT id, state::text, idempotency_key FROM background_jobs WHERE document_id = $1 ORDER BY created_at, id",
        doc_id,
    )
    assert len(jobs) == 2
    assert jobs[0]["state"] == "succeeded"
    assert jobs[1]["state"] == "queued"
    assert jobs[1]["idempotency_key"] == f"document.extract:{doc_id}:after:{job.id}"


@pytest.mark.asyncio
async def test_startup_recovery_marks_cancelled_first_extraction_failed_without_successor(pool):
    from main import _recover_durable_extraction_jobs

    job, _user_id, _kb_id, doc_id = await _seed_document_job(pool, status="processing")
    await pool.execute(
        "UPDATE background_jobs SET state = 'cancelled', updated_at = now() WHERE id = $1",
        job.id,
    )

    assert await _recover_durable_extraction_jobs(pool, JobService(pool)) == []

    row = await pool.fetchrow("SELECT status::text, error_message FROM documents WHERE id = $1", doc_id)
    assert dict(row) == {
        "status": "failed",
        "error_message": "Document extraction was cancelled.",
    }
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE document_id = $1", doc_id) == 1


@pytest.mark.asyncio
async def test_hosted_startup_durable_branch_backfills_without_legacy_spawn(monkeypatch):
    import main

    recovered = [object()]
    calls = []

    async def durable_recovery(pool, job_service):
        calls.append((pool, job_service))
        return recovered

    monkeypatch.setattr(main, "_recover_durable_extraction_jobs", durable_recovery)
    result = await main._recover_hosted_extractions(
        object(),
        job_service="jobs",
    )

    assert result == recovered
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_hosted_startup_has_no_legacy_scan_or_spawn():
    from main import _recover_hosted_extractions

    # The helper delegates only to the durable ledger path; no OCR/spawn inputs exist.
    assert "ocr_service" not in inspect.signature(_recover_hosted_extractions).parameters
    assert "spawn" not in inspect.signature(_recover_hosted_extractions).parameters


@pytest.mark.asyncio
async def test_job_service_preserves_managed_create_and_guards_caller_transaction(pool):
    user_id, kb_id = await _seed_tenant(pool)
    doc_id = uuid4()
    await pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, path, file_type) "
        "VALUES ($1, $2, $3, 'source.pdf', '/', 'pdf')",
        doc_id,
        kb_id,
        user_id,
    )
    service = JobService(pool)
    command = JobCreate(
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=doc_id,
        payload={"document_id": str(doc_id)},
        idempotency_key=f"document.extract:{doc_id}",
    )

    managed = await service.create(command, authenticated_user_id=user_id)
    assert managed.document_id == doc_id

    async with pool.acquire() as conn:
        with pytest.raises(RuntimeError, match="explicit transaction"):
            await service.create_in_transaction(conn, command, authenticated_user_id=user_id)

    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE id = $1", managed.id) == 1


@pytest.mark.asyncio
async def test_cancelled_first_extraction_can_be_explicitly_recreated_by_url_producer(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    doc_id = uuid4()
    source_url = "https://example.test/cancelled.pdf"
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,path,file_type,status,metadata,version) "
        "VALUES ($1,$2,$3,'cancelled.pdf','/','pdf','processing',$4::jsonb,0)",
        doc_id,
        kb_id,
        user_id,
        json.dumps({"source_url": source_url}),
    )
    service = JobService(pool)
    original = await service.create(
        JobCreate(
            job_type=JobType.DOCUMENT_EXTRACT,
            user_id=user_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            payload={"document_id": str(doc_id)},
            idempotency_key=f"document.extract:{doc_id}",
        ),
        authenticated_user_id=user_id,
    )
    cancelled = await service.cancel(original.id, authenticated_user_id=user_id)
    assert cancelled is not None
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "failed"

    producer = UrlIngestService(pool, RecordingS3(), service)
    monkeypatch.setattr(producer, "_download", lambda _url: pytest.fail("existing source must not redownload"))
    recreated = await producer.ingest_pdf(str(user_id), str(kb_id), source_url, "/")

    assert recreated["id"] == str(doc_id)
    assert recreated["job_id"] != str(original.id)
    successor = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1::uuid", recreated["job_id"])
    assert successor["state"] == "queued"
    assert successor["idempotency_key"] == f"document.extract:{doc_id}:after:{original.id}"
    assert await pool.fetchval("SELECT status::text FROM documents WHERE id = $1", doc_id) == "pending"


@pytest.mark.asyncio
async def test_url_producer_creates_successor_for_pending_document_with_succeeded_latest_job(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    doc_id = uuid4()
    source_url = "https://example.test/drifted.pdf"
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,path,file_type,status,metadata,version) "
        "VALUES ($1,$2,$3,'drifted.pdf','/','pdf','pending',$4::jsonb,1)",
        doc_id,
        kb_id,
        user_id,
        json.dumps({"source_url": source_url}),
    )
    jobs = JobService(pool)
    original = await jobs.create(
        JobCreate(
            job_type=JobType.DOCUMENT_EXTRACT,
            user_id=user_id,
            knowledge_base_id=kb_id,
            document_id=doc_id,
            payload={"document_id": str(doc_id)},
            idempotency_key=f"document.extract:{doc_id}",
        ),
        authenticated_user_id=user_id,
    )
    await pool.execute("UPDATE background_jobs SET state = 'succeeded' WHERE id = $1", original.id)
    producer = UrlIngestService(pool, RecordingS3(), jobs)
    monkeypatch.setattr(producer, "_download", lambda _url: pytest.fail("existing source must not redownload"))

    recreated = await producer.ingest_pdf(str(user_id), str(kb_id), source_url, "/")

    assert recreated["job_id"] != str(original.id)
    successor = await pool.fetchrow(
        "SELECT state::text, idempotency_key FROM background_jobs WHERE id = $1::uuid", recreated["job_id"]
    )
    assert dict(successor) == {
        "state": "queued",
        "idempotency_key": f"document.extract:{doc_id}:after:{original.id}",
    }


async def _async_value(value):
    return value
