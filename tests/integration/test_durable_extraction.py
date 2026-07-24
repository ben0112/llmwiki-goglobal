from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from botocore.exceptions import EndpointConnectionError
from config import settings
from fastapi import HTTPException
from jobs import repository
from jobs.handlers import RetryableJobError, TerminalJobError, WorkerContext, handle_document_extract
from jobs.lease import JobLease
from jobs.models import JobCancelled, JobCreate, JobRecord, JobType, LeaseLost
from jobs.service import JobService
from services.ocr import OCRService
from services.types import DownloadedPdf
from services.url_ingest import UrlIngestService


class RecordingS3:
    def __init__(self, *, upload_error: Exception | None = None) -> None:
        self.upload_error = upload_error
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.deleted_prefixes: list[str] = []
        self.presigned_gets: list[str] = []

    async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
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


@pytest.mark.asyncio
async def test_url_ingest_commits_document_and_job_atomically_after_upload(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(pool, s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

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


@pytest.mark.asyncio
async def test_url_ingest_transaction_failure_deletes_exact_uploaded_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool, storage_limit_bytes=1)
    s3 = RecordingS3()
    service = UrlIngestService(pool, s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(HTTPException) as raised:
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/quota.pdf", "/")

    assert raised.value.status_code == 413
    assert len(s3.uploads) == 1
    document_prefix = s3.uploads[0].rsplit("source.pdf", 1)[0]
    assert s3.deleted_prefixes == [document_prefix]
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

    service = UrlIngestService(pool, s3, FailingJobService(pool))
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


@pytest.mark.asyncio
async def test_url_ingest_upload_failure_leaves_no_database_rows(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3(upload_error=RuntimeError("storage offline secret=s3-token"))
    service = UrlIngestService(pool, s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(HTTPException) as raised:
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/upload.pdf", "/")

    assert raised.value.status_code == 502
    assert "secret" not in raised.value.detail
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 0


@pytest.mark.asyncio
async def test_url_ingest_cancellation_after_upload_compensates_orphan(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)

    class CancelledAfterUpload(RecordingS3):
        async def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
            await super().upload_bytes(key, data, content_type)
            raise asyncio.CancelledError

    s3 = CancelledAfterUpload()
    service = UrlIngestService(pool, s3, JobService(pool))
    pdf = DownloadedPdf(data=b"%PDF-1.7\ncontent", filename="paper.pdf")
    monkeypatch.setattr(service, "_download", lambda _url: _async_value(pdf))

    with pytest.raises(asyncio.CancelledError):
        await service.ingest_pdf(str(user_id), str(kb_id), "https://example.test/cancel.pdf", "/")

    assert len(s3.uploads) == 1
    assert s3.deleted_prefixes == [s3.uploads[0].rsplit("source.pdf", 1)[0]]
    assert s3.objects == {}
    assert await pool.fetchval("SELECT count(*) FROM documents WHERE user_id = $1", user_id) == 0


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
    service = UrlIngestService(pool, s3, JobService(pool))
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


@pytest.mark.asyncio
async def test_url_ingest_commit_error_deletes_artifact_only_when_rollback_is_confirmed(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(pool, s3, JobService(pool))
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


@pytest.mark.asyncio
async def test_url_ingest_unknown_commit_outcome_preserves_artifact(pool, monkeypatch):
    user_id, kb_id = await _seed_tenant(pool)
    s3 = RecordingS3()
    service = UrlIngestService(pool, s3, JobService(pool))
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
    assert {record.document_id for record in recovered_first} == expected_ids
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

    def forbidden_spawn(*_args):
        raise AssertionError("durable startup must not spawn process-local extraction")

    monkeypatch.setattr(main, "_recover_durable_extraction_jobs", durable_recovery)
    result = await main._recover_hosted_extractions(
        object(),
        durable_jobs_enabled=True,
        job_service="jobs",
        ocr_service=object(),
        spawn=forbidden_spawn,
    )

    assert result == recovered
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_hosted_startup_rollback_branch_keeps_legacy_scan_and_spawn():
    from main import _recover_hosted_extractions

    class FakePool:
        async def fetch(self, query):
            assert "pending" in query and "processing" in query
            return [{"id": "document-1", "user_id": "user-1"}]

    class FakeOCR:
        async def process_document(self, document_id, user_id):
            return document_id, user_id

    spawned = []

    def capture_spawn(coroutine, label):
        spawned.append((coroutine, label))
        coroutine.close()

    result = await _recover_hosted_extractions(
        FakePool(),
        durable_jobs_enabled=False,
        job_service=None,
        ocr_service=FakeOCR(),
        spawn=capture_spawn,
    )

    assert result == [{"id": "document-1", "user_id": "user-1"}]
    assert [label for _, label in spawned] == ["recover:document"]


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


async def _async_value(value):
    return value
