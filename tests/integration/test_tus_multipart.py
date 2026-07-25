"""Real cross-replica Hosted TUS tests using Postgres, Redis, and MinIO."""

from __future__ import annotations

import asyncio
import base64
import os
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
    return ",".join(
        f"{key} {base64.b64encode(value.encode('utf-8')).decode('ascii')}" for key, value in values.items()
    )


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

    def make_app() -> FastAPI:
        app = FastAPI()
        store = TusSessionStore(redis)
        app.state.tus_service = tus.HostedTusMultipartService(
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
        app.include_router(tus.router)
        return app

    app_a = make_app()
    app_b = make_app()
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
    assert await runtime["pool"].fetchval(
        "SELECT COUNT(*) FROM documents WHERE id = $1 AND user_id = $2", document_id, runtime["user_id"]
    ) == 1
    assert await runtime["pool"].fetchval(
        "SELECT COUNT(*) FROM background_jobs WHERE id = $1 AND user_id = $2", job_id, runtime["user_id"]
    ) == 1
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
