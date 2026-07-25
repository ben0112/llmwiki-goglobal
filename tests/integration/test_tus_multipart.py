"""Real cross-replica Hosted TUS tests using Postgres, Redis, and MinIO."""

from __future__ import annotations

import asyncio
import base64
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from fastapi import FastAPI
from infra.quota import HostedQuotaService, quota_keys
from infra.redis import create_redis
from infra.tus_sessions import TusSessionStore, lock_key
from jobs.service import JobService
from services import s3 as s3_module

pytestmark = pytest.mark.asyncio

MIB = 1024 * 1024


def _metadata(filename: str, kb_id: UUID, path: str = "/") -> str:
    values = {"filename": filename, "knowledge_base_id": str(kb_id), "path": path}
    return ",".join(f"{key} {base64.b64encode(value.encode('utf-8')).decode('ascii')}" for key, value in values.items())


def _headers(user_id: UUID, **extra: str) -> dict[str, str]:
    return {"X-Test-User": str(user_id), "Tus-Resumable": "1.0.0", **extra}


async def _ensure_bucket(service: s3_module.S3Service, bucket: str) -> None:
    last_error: Exception | None = None
    for _ in range(40):
        try:
            async with service._session.client("s3", **s3_module.s3_client_kwargs()) as client:
                try:
                    await client.head_bucket(Bucket=bucket)
                except ClientError:
                    await client.create_bucket(Bucket=bucket)
            return
        except EndpointConnectionError as exc:
            last_error = exc
            await asyncio.sleep(0.1)
    pytest.fail(f"MinIO did not become ready: {type(last_error).__name__}")


@pytest.fixture
async def multipart_runtime(pool, monkeypatch):
    from infra import tus

    redis_url = os.getenv("TUS_MULTIPART_REDIS_URL")
    endpoint = os.getenv("TUS_MULTIPART_S3_ENDPOINT")
    if not redis_url or not endpoint:
        pytest.skip("real Hosted TUS dependencies are not configured")

    bucket = os.getenv("TUS_MULTIPART_S3_BUCKET", "llmwiki-tus-test")
    monkeypatch.setattr(s3_module.settings, "AWS_ACCESS_KEY_ID", os.environ["TUS_MULTIPART_S3_ACCESS_KEY"])
    monkeypatch.setattr(s3_module.settings, "AWS_SECRET_ACCESS_KEY", os.environ["TUS_MULTIPART_S3_SECRET_KEY"])
    monkeypatch.setattr(s3_module.settings, "AWS_REGION", "us-east-1")
    monkeypatch.setattr(s3_module.settings, "S3_BUCKET", bucket)
    monkeypatch.setattr(s3_module.settings, "S3_ENDPOINT_URL", endpoint)
    monkeypatch.setattr(s3_module.settings, "S3_FORCE_PATH_STYLE", True)
    monkeypatch.setattr(tus.settings, "TUS_MULTIPART_ENABLED", True)

    redis = create_redis(redis_url)
    await redis.ping()
    s3 = s3_module.S3Service()
    await _ensure_bucket(s3, bucket)

    user_id = uuid4()
    other_user_id = uuid4()
    kb_id = uuid4()
    await pool.executemany(
        "INSERT INTO users (id, email, storage_limit_bytes) VALUES ($1, $2, $3)",
        [
            (user_id, f"{user_id}@tus.test", 50 * MIB),
            (other_user_id, f"{other_user_id}@tus.test", 50 * MIB),
        ],
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, 'TUS KB', $3)",
        kb_id,
        user_id,
        f"tus-{kb_id}",
    )

    async def authenticated(request):
        return request.headers["X-Test-User"]

    monkeypatch.setattr(tus, "_get_user_id", authenticated)

    def make_app() -> tuple[FastAPI, tus.HostedTusMultipartService]:
        app = FastAPI()
        store = TusSessionStore(redis)
        service = tus.HostedTusMultipartService(
            pool,
            s3,
            JobService(pool),
            HostedQuotaService(pool, redis),
            store,
            session_ttl_seconds=300,
            stale_seconds=180,
            lock_seconds=10,
            max_patch_bytes=8 * MIB,
        )
        app.state.tus_service = service
        app.include_router(tus.router)
        return app, service

    app_a, service_a = make_app()
    app_b, service_b = make_app()
    clients = (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app_a), base_url="http://replica-a"),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app_b), base_url="http://replica-b"),
    )
    await clients[0].__aenter__()
    await clients[1].__aenter__()
    try:
        yield {
            "pool": pool,
            "redis": redis,
            "s3": s3,
            "store": TusSessionStore(redis),
            "user_id": user_id,
            "other_user_id": other_user_id,
            "kb_id": kb_id,
            "a": clients[0],
            "b": clients[1],
            "service_a": service_a,
            "service_b": service_b,
        }
    finally:
        await clients[1].__aexit__(None, None, None)
        await clients[0].__aexit__(None, None, None)
        await s3.delete_prefix(f"{user_id}/")
        keys = []
        async for key in redis.scan_iter(match="tus:*"):
            keys.append(key)
        keys.extend(quota_keys(user_id))
        keys.extend(quota_keys(other_user_id))
        if keys:
            await redis.delete(*keys)
        await redis.aclose()
        await pool.execute("DELETE FROM background_jobs WHERE user_id = ANY($1::uuid[])", [user_id, other_user_id])
        await pool.execute("DELETE FROM documents WHERE user_id = ANY($1::uuid[])", [user_id, other_user_id])
        await pool.execute("DELETE FROM knowledge_bases WHERE user_id = ANY($1::uuid[])", [user_id, other_user_id])
        await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", [user_id, other_user_id])


async def _create_upload(runtime, body_length: int) -> str:
    response = await runtime["a"].post(
        "/v1/uploads",
        headers=_headers(
            runtime["user_id"],
            **{
                "Upload-Length": str(body_length),
                "Upload-Metadata": _metadata("cross-replica.pdf", runtime["kb_id"], "/reports/"),
            },
        ),
    )
    assert response.status_code == 201, response.text
    return response.headers["location"]


async def test_cross_replica_resume_finalization_and_duplicate_final_patch(multipart_runtime):
    runtime = multipart_runtime
    first = b"%PDF-" + b"a" * (5 * MIB - 5)
    final = b"tail"
    total = len(first) + len(final)
    temp_root = Path("/tmp/supavault_tus_uploads")
    before = set(temp_root.iterdir()) if temp_root.exists() else set()
    location = await _create_upload(runtime, total)

    head = await runtime["b"].head(location, headers=_headers(runtime["user_id"]))
    assert head.status_code == 200
    assert head.headers["upload-offset"] == "0"
    assert head.headers["upload-length"] == str(total)

    mismatch = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "1", "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"ignored",
    )
    assert mismatch.status_code == 409
    assert mismatch.headers["upload-offset"] == "0"

    too_small = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"%PDF-small",
    )
    assert too_small.status_code in {413, 422}
    assert (await runtime["a"].head(location, headers=_headers(runtime["user_id"]))).headers["upload-offset"] == "0"

    first_response = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=first,
    )
    assert first_response.status_code == 204
    assert first_response.headers["upload-offset"] == str(len(first))

    completed = await runtime["a"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": str(len(first)), "Content-Type": "application/offset+octet-stream"},
        ),
        content=final,
    )
    assert completed.status_code == 204, completed.text
    assert completed.headers["upload-offset"] == str(total)
    document_id = UUID(completed.headers["x-document-id"])
    job_id = UUID(completed.headers["x-job-id"])

    duplicate = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": str(total), "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"",
    )
    assert duplicate.status_code == 204
    assert UUID(duplicate.headers["x-document-id"]) == document_id
    assert UUID(duplicate.headers["x-job-id"]) == job_id
    assert (
        await runtime["pool"].fetchval(
            "SELECT COUNT(*) FROM documents WHERE id = $1 AND user_id = $2", document_id, runtime["user_id"]
        )
        == 1
    )
    assert (
        await runtime["pool"].fetchval(
            "SELECT COUNT(*) FROM background_jobs WHERE id = $1 AND user_id = $2", job_id, runtime["user_id"]
        )
        == 1
    )
    after = set(temp_root.iterdir()) if temp_root.exists() else set()
    assert after == before


async def test_owner_checks_lock_contention_and_actual_stream_cap(multipart_runtime):
    runtime = multipart_runtime
    location = await _create_upload(runtime, 9 * MIB)
    upload_id = UUID(location.rsplit("/", 1)[-1])

    assert (await runtime["b"].head(location, headers=_headers(runtime["other_user_id"]))).status_code == 404
    forbidden_patch = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["other_user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"x",
    )
    assert forbidden_patch.status_code == 404

    assert await runtime["redis"].set(lock_key(upload_id), "contender", nx=True, ex=10)
    contended = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"%PDF-" + b"a" * (5 * MIB),
    )
    assert contended.status_code in {409, 423}
    assert "retry-after" in contended.headers
    await runtime["redis"].delete(lock_key(upload_id))

    oversized = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=b"%PDF-" + b"a" * (8 * MIB),
    )
    assert oversized.status_code == 413
    assert (await runtime["a"].head(location, headers=_headers(runtime["user_id"]))).headers["upload-offset"] == "0"


async def test_real_s3_etag_interruption_leaves_zero_offset_and_worker_recovers(multipart_runtime):
    from infra.tus import HostedTusCleanupService
    from infra.tus_sessions import TusReservationState

    runtime = multipart_runtime
    body = b"%PDF-" + b"a" * (5 * MIB - 5)
    location = await _create_upload(runtime, len(body))
    upload_id = UUID(location.rsplit("/", 1)[-1])
    etag_ready = asyncio.Event()
    never_return = asyncio.Event()
    original_upload_part = runtime["s3"].upload_part

    async def gated_upload_part(key, multipart_id, part_number, payload):
        etag = await original_upload_part(key, multipart_id, part_number, payload)
        etag_ready.set()
        await never_return.wait()
        return etag

    runtime["s3"].upload_part = gated_upload_part
    patch_task = asyncio.create_task(
        runtime["a"].patch(
            location,
            headers=_headers(
                runtime["user_id"],
                **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
            ),
            content=body,
        )
    )
    await etag_ready.wait()
    patch_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await patch_task
    runtime["s3"].upload_part = original_upload_part

    interrupted = await runtime["store"].get(upload_id)
    assert interrupted is not None
    assert interrupted.offset == 0
    assert interrupted.parts == ()
    cleanup = HostedTusCleanupService(
        runtime["pool"],
        runtime["s3"],
        JobService(runtime["pool"]),
        HostedQuotaService(runtime["pool"], runtime["redis"]),
        runtime["store"],
        session_ttl_seconds=300,
        stale_seconds=-1,
        lock_seconds=10,
    )
    assert (await cleanup.cleanup(upload_id, runtime["user_id"]))["status"] == "cleaned"
    assert await runtime["store"].get(upload_id) is None
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RELEASED
    assert await runtime["redis"].zscore(quota_keys(runtime["user_id"])[1], str(upload_id)) is None


async def test_real_signature_rejection_removes_object_session_and_quota(multipart_runtime):
    from infra.tus_sessions import TusReservationState

    runtime = multipart_runtime
    body = b"not-a-pdf"
    location = await _create_upload(runtime, len(body))
    upload_id = UUID(location.rsplit("/", 1)[-1])
    session = await runtime["store"].get(upload_id)

    response = await runtime["b"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=body,
    )

    assert response.status_code == 400
    assert await runtime["store"].get(upload_id) is None
    assert await runtime["s3"].head_object(session.s3_key) is None
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RELEASED
    assert await runtime["redis"].zscore(quota_keys(runtime["user_id"])[1], str(upload_id)) is None


async def test_real_postgres_rollback_orphan_is_recovered_by_durable_cleanup(multipart_runtime):
    from infra.tus import HostedTusCleanupService
    from infra.tus_sessions import TusReservationState, TusSessionState

    runtime = multipart_runtime
    body = b"%PDF-valid"
    location = await _create_upload(runtime, len(body))
    upload_id = UUID(location.rsplit("/", 1)[-1])
    session = await runtime["store"].get(upload_id)
    original_delete = runtime["s3"].delete_object
    delete_calls = 0

    async def rollback_commit(_transaction):
        raise RuntimeError("forced pre-commit rollback")

    async def fail_first_delete(key):
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise RuntimeError("temporary object-store failure")
        return await original_delete(key)

    runtime["service_a"]._commit_transaction = rollback_commit
    runtime["s3"].delete_object = fail_first_delete
    response = await runtime["a"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=body,
    )
    runtime["s3"].delete_object = original_delete

    assert response.status_code == 503
    assert await runtime["pool"].fetchval("SELECT COUNT(*) FROM documents WHERE id = $1", upload_id) == 0
    orphan = await runtime["store"].get(upload_id)
    assert orphan is not None and orphan.state is TusSessionState.CLEANUP_REQUIRED
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RESERVED

    cleanup = HostedTusCleanupService(
        runtime["pool"],
        runtime["s3"],
        JobService(runtime["pool"]),
        HostedQuotaService(runtime["pool"], runtime["redis"]),
        runtime["store"],
        session_ttl_seconds=300,
        stale_seconds=180,
        lock_seconds=10,
    )
    assert (await cleanup.cleanup(upload_id, runtime["user_id"]))["status"] == "cleaned"
    assert await runtime["store"].get(upload_id) is None
    assert await runtime["s3"].head_object(session.s3_key) is None
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RELEASED
    assert await runtime["redis"].zscore(quota_keys(runtime["user_id"])[1], str(upload_id)) is None


async def test_real_unknown_commit_is_reconciled_without_deleting_committed_object(multipart_runtime):
    from infra.tus import HostedTusCleanupService, _CommitOutcomeError
    from infra.tus_sessions import TusReservationState, TusSessionState

    runtime = multipart_runtime
    body = b"%PDF-committed"
    location = await _create_upload(runtime, len(body))
    upload_id = UUID(location.rsplit("/", 1)[-1])
    session = await runtime["store"].get(upload_id)
    original_persist = runtime["service_a"]._persist_document_job

    async def commit_then_lose_outcome(actual_session):
        document_id, job_id = await original_persist(actual_session)
        raise _CommitOutcomeError(None, document_id, job_id)

    runtime["service_a"]._persist_document_job = commit_then_lose_outcome
    response = await runtime["a"].patch(
        location,
        headers=_headers(
            runtime["user_id"],
            **{"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        ),
        content=body,
    )

    assert response.status_code == 503
    unknown = await runtime["store"].get(upload_id)
    assert unknown is not None
    assert unknown.state is TusSessionState.UPLOADING
    assert unknown.object_completed is True
    assert await runtime["pool"].fetchval("SELECT COUNT(*) FROM documents WHERE id = $1", upload_id) == 1
    assert await runtime["s3"].head_object(session.s3_key) is not None

    cleanup = HostedTusCleanupService(
        runtime["pool"],
        runtime["s3"],
        JobService(runtime["pool"]),
        HostedQuotaService(runtime["pool"], runtime["redis"]),
        runtime["store"],
        session_ttl_seconds=300,
        stale_seconds=86_400,
        lock_seconds=10,
    )
    assert (await cleanup.cleanup(upload_id, runtime["user_id"]))["status"] == "committed"
    recovered = await runtime["store"].get(upload_id)
    assert recovered is not None and recovered.state is TusSessionState.COMPLETED
    assert recovered.document_id == upload_id
    assert await runtime["s3"].head_object(session.s3_key) is not None
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RELEASED
    assert await runtime["redis"].zscore(quota_keys(runtime["user_id"])[1], str(upload_id)) is None


async def test_real_stale_cleanup_treats_nosuch_as_success_replays_and_releases_quota_once(
    multipart_runtime,
):
    from infra.tus import HostedTusCleanupService
    from infra.tus_sessions import TusReservationState

    runtime = multipart_runtime
    location = await _create_upload(runtime, 5 * MIB)
    upload_id = UUID(location.rsplit("/", 1)[-1])
    session = await runtime["store"].get(upload_id)
    await runtime["s3"].abort_multipart(session.s3_key, session.multipart_upload_id)
    await runtime["s3"].delete_object(session.s3_key)
    delegate = HostedQuotaService(runtime["pool"], runtime["redis"])

    class CountingQuota:
        def __init__(self):
            self.release_calls = 0

        async def release(self, reservation):
            self.release_calls += 1
            return await delegate.release(reservation)

    quota = CountingQuota()
    cleanup = HostedTusCleanupService(
        runtime["pool"],
        runtime["s3"],
        JobService(runtime["pool"]),
        quota,
        runtime["store"],
        session_ttl_seconds=300,
        stale_seconds=-1,
        lock_seconds=10,
    )

    assert (await cleanup.cleanup(upload_id, runtime["user_id"]))["status"] == "cleaned"
    assert (await cleanup.cleanup(upload_id, runtime["user_id"]))["status"] == "already_clean"
    assert quota.release_calls == 1
    assert await runtime["store"].get(upload_id) is None
    assert await runtime["s3"].head_object(session.s3_key) is None
    marker = await runtime["store"].get_reservation(runtime["user_id"], upload_id)
    assert marker is not None and marker.state is TusReservationState.RELEASED
    assert await runtime["redis"].zscore(quota_keys(runtime["user_id"])[1], str(upload_id)) is None


async def test_real_cleanup_scan_creates_claimable_successor_after_exhausted_job(multipart_runtime):
    from infra.tus import HostedTusCleanupService
    from jobs import repository

    runtime = multipart_runtime
    location = await _create_upload(runtime, 5 * MIB)
    upload_id = UUID(location.rsplit("/", 1)[-1])
    cleanup = HostedTusCleanupService(
        runtime["pool"],
        runtime["s3"],
        JobService(runtime["pool"]),
        HostedQuotaService(runtime["pool"], runtime["redis"]),
        runtime["store"],
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )
    first_scan = (datetime.now(UTC) + timedelta(minutes=5)).replace(second=15, microsecond=0)

    await cleanup.enqueue_stale_jobs(now=first_scan)
    await cleanup.enqueue_stale_jobs(now=first_scan.replace(second=45))
    first_rows = await runtime["pool"].fetch(
        "SELECT id, idempotency_key FROM background_jobs "
        "WHERE user_id = $1 AND job_type = 'upload.cleanup' AND payload->>'upload_id' = $2",
        runtime["user_id"],
        str(upload_id),
    )
    assert len(first_rows) == 1
    await runtime["pool"].execute(
        "UPDATE background_jobs SET state = 'failed', attempt_count = max_attempts, updated_at = now() WHERE id = $1",
        first_rows[0]["id"],
    )

    await cleanup.enqueue_stale_jobs(now=first_scan + timedelta(minutes=1))
    rows = await runtime["pool"].fetch(
        "SELECT id, state::text, idempotency_key FROM background_jobs "
        "WHERE user_id = $1 AND job_type = 'upload.cleanup' AND payload->>'upload_id' = $2 "
        "ORDER BY created_at, id",
        runtime["user_id"],
        str(upload_id),
    )
    assert len(rows) == 2
    assert rows[0]["state"] == "failed"
    assert rows[0]["idempotency_key"] != rows[1]["idempotency_key"]
    async with runtime["pool"].acquire() as conn, conn.transaction():
        claimed = await repository.claim(conn, rows[1]["id"], "cleanup-test-worker", lease_seconds=30)
    assert claimed is not None
    assert claimed.id == rows[1]["id"]
