"""Deterministic failure matrix for the durable Hosted execution boundary."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from infra.tus_sessions import (
    TusPart,
    TusSession,
    TusSessionState,
)
from jobs import repository, worker
from jobs.dispatcher import dispatch_due_jobs
from jobs.handlers import WorkerContext
from jobs.models import JobType

from tests.unit.test_hosted_tus_multipart import (
    MIB,
    S3,
    USER_ID,
    Quota,
    Store,
    _app,
    _client,
    _session,
)

OLD = datetime(2020, 1, 1, tzinfo=UTC)


async def _seed_user(pool):
    user_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@durable-failure-matrix.test",
    )
    return user_id


async def _insert_job(pool, user_id, *, state="queued", **values):
    job_type = values.pop("job_type", "document.extract")
    columns = ["user_id", "job_type", "state", *values]
    parameters = [user_id, job_type, state, *values.values()]
    placeholders = [f"${index}" for index in range(1, len(parameters) + 1)]
    casts = ["" if key not in {"payload", "progress", "result"} else "::jsonb" for key in columns]
    return await pool.fetchrow(
        "INSERT INTO background_jobs ("
        + ", ".join(columns)
        + ") VALUES ("
        + ", ".join(value + cast for value, cast in zip(placeholders, casts, strict=True))
        + ") RETURNING *",
        *parameters,
    )


class FailOnceRedis:
    def __init__(self) -> None:
        self.calls = 0

    async def enqueue_job(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("fault injection: Redis unavailable")
        return object()


def _worker_context(pool) -> WorkerContext:
    return WorkerContext(pool=pool, s3=None, converter_url="", converter_secret="")


def _worker_ctx(pool, handler, *, worker_id="matrix-worker") -> dict:
    return {
        "pool": pool,
        "worker_id": worker_id,
        "lease_seconds": 30,
        "heartbeat_seconds": 10,
        "worker_context": _worker_context(pool),
        "handlers": {JobType.DOCUMENT_EXTRACT: handler},
    }


def _json_events(caplog) -> list[dict]:
    events = []
    for record in caplog.records:
        try:
            message = json.loads(record.getMessage())
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and isinstance(message.get("event"), str):
            events.append(message)
    return events


@pytest.mark.asyncio
async def test_redis_outage_after_acceptance_eventually_dispatches_with_json_telemetry(pool, caplog):
    """A committed ledger row remains due until transport recovers."""
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    redis = FailOnceRedis()

    with caplog.at_level(logging.INFO):
        failed = await dispatch_due_jobs(pool, redis, batch_size=10, redeliver_seconds=30)
        recovered = await dispatch_due_jobs(pool, redis, batch_size=10, redeliver_seconds=30)

    assert failed.enqueue_failed == 1
    assert failed.marked == 0
    assert recovered.enqueued == 1
    assert recovered.marked == 1
    row = await pool.fetchrow(
        "SELECT state, dispatch_attempts, last_dispatched_at FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert dict(row) == {
        "state": "queued",
        "dispatch_attempts": 1,
        "last_dispatched_at": row["last_dispatched_at"],
    }
    assert row["last_dispatched_at"] is not None

    dispatched = [event for event in _json_events(caplog) if event["event"] == "durable_job_dispatched"]
    assert len(dispatched) == 1
    assert dispatched[0] == {
        "dispatch_attempts": 1,
        "dispatch_lag_ms": dispatched[0]["dispatch_lag_ms"],
        "event": "durable_job_dispatched",
        "job_id": str(job["id"]),
        "job_type": "document.extract",
        "replica_role": "worker",
        "state": "queued",
    }
    assert dispatched[0]["dispatch_lag_ms"] >= 0


@pytest.mark.asyncio
async def test_api_termination_after_acceptance_loses_no_document_or_job(pool):
    """Acceptance is the document/job transaction, not an API-owned coroutine."""
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    knowledge_base_id = uuid4()
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, 'Failure Matrix', $3)",
        knowledge_base_id,
        user_id,
        f"failure-matrix-{knowledge_base_id}",
    )
    async with pool.acquire() as api_conn, api_conn.transaction():
        await api_conn.execute(
            "INSERT INTO documents "
            "(id, knowledge_base_id, user_id, filename, path, file_type, status, source_kind) "
            "VALUES ($1, $2, $3, 'accepted.pdf', '/', 'pdf', 'pending', 'source')",
            document_id,
            knowledge_base_id,
            user_id,
        )
        job = await api_conn.fetchrow(
            "INSERT INTO background_jobs "
            "(user_id, knowledge_base_id, document_id, job_type, payload, idempotency_key) "
            "VALUES ($1, $2, $3, 'document.extract', $4::jsonb, $5) RETURNING id",
            user_id,
            knowledge_base_id,
            document_id,
            json.dumps({"document_id": str(document_id)}),
            f"document.extract:{document_id}",
        )

    # The accepting request and all of its process-local state are now gone.
    accepted = await pool.fetchrow(
        "SELECT d.id AS document_id, j.id AS job_id, j.state "
        "FROM documents d JOIN background_jobs j ON j.document_id = d.id WHERE d.id = $1",
        document_id,
    )
    assert dict(accepted) == {"document_id": document_id, "job_id": job["id"], "state": "queued"}


@pytest.mark.asyncio
async def test_worker_termination_before_final_write_recovers_at_lease_boundary(pool):
    """A killed owner has no materialized effect; the next lease generation commits once."""
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    async with pool.acquire() as conn, conn.transaction():
        claimed = await repository.claim(conn, job["id"], "killed-worker", 30)
        assert claimed is not None
        await conn.execute(
            "UPDATE background_jobs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = $1",
            job["id"],
        )
        assert await repository.reap_expired(conn, limit=1) == [job["id"]]

    materialized = []

    async def handler(record, lease, _context):
        await lease.checkpoint()
        materialized.append(record.attempt_count)
        return {"materialized": True}

    outcome = await worker.run_job(_worker_ctx(pool, handler, worker_id="replacement-worker"), str(job["id"]))

    assert outcome == {"status": "succeeded", "job_id": str(job["id"])}
    assert materialized == [2]
    row = await pool.fetchrow("SELECT state, attempt_count, result FROM background_jobs WHERE id = $1", job["id"])
    assert row["state"] == "succeeded"
    assert row["attempt_count"] == 2
    assert json.loads(row["result"]) == {"materialized": True}


@pytest.mark.asyncio
async def test_duplicate_arq_delivery_materializes_once(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    handled = []

    async def handler(record, _lease, _context):
        handled.append(record.id)
        return {"ok": True}

    ctx = _worker_ctx(pool, handler)
    first = await worker.run_job(ctx, str(job["id"]))
    duplicate = await worker.run_job(ctx, str(job["id"]))

    assert first["status"] == "succeeded"
    assert duplicate["status"] == "duplicate"
    assert handled == [job["id"]]


@pytest.mark.asyncio
async def test_redis_offset_failure_after_s3_part_never_advances_session(monkeypatch):
    events = []
    session = _session()

    class OffsetWriteFailure(Store):
        async def append_part(self, *_args, **_kwargs):
            self.events.append("redis.append")
            raise ConnectionError("fault injection: Redis offset write failed")

    store = OffsetWriteFailure(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)
    body = b"%PDF-" + b"x" * (5 * MIB - 5)

    async with await _client(app) as client:
        with pytest.raises(ConnectionError, match="offset write failed"):
            await client.patch(
                f"/v1/uploads/{session.upload_id}",
                headers={
                    "X-Test-User": str(USER_ID),
                    "Tus-Resumable": "1.0.0",
                    "Upload-Offset": "0",
                    "Content-Type": "application/offset+octet-stream",
                },
                content=body,
            )

    assert store.session.offset == 0
    assert store.session.parts == ()
    assert events.index("s3.upload.etag") < events.index("redis.append")


@pytest.mark.asyncio
async def test_multipart_completion_then_postgres_rollback_deletes_orphan(monkeypatch):
    from infra.tus import _CommitOutcomeError

    events = []
    session = _session(
        total=5,
        offset=5,
        parts=(TusPart(1, "etag-1"),),
        object_completed=True,
    )
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)

    async def rolled_back(_session):
        raise _CommitOutcomeError(False, session.upload_id, uuid4())

    monkeypatch.setattr(app.state.tus_service, "_persist_document_job", rolled_back)

    with pytest.raises(Exception):
        await app.state.tus_service._finalize(session, "L" * 32)

    assert "s3.delete" in events
    assert events.count("quota.release") == 1
    assert events.count("session.delete") == 1


def test_redis_aof_restart_restores_resumable_session():
    """Model a Redis AOF restart at the serialized session durability boundary."""
    before = replace(
        _session(offset=5 * MIB, parts=(TusPart(1, "etag-1"),)),
        updated_at=datetime.now(UTC),
    )
    append_only_record = before.to_json()

    after_restart = TusSession.from_json(append_only_record)

    assert after_restart.upload_id == before.upload_id
    assert after_restart.offset == 5 * MIB
    assert after_restart.parts == (TusPart(1, "etag-1"),)
    compose = Path("deploy/docker-compose.selfhost.yml").read_text()
    assert "--appendonly" in compose and "yes" in compose
    assert "--appendfsync" in compose and "everysec" in compose


@pytest.mark.asyncio
async def test_repeated_cleanup_releases_quota_once(monkeypatch):
    from infra.tus import HostedTusCleanupService

    events = []
    session = replace(_session(total=5), state=TusSessionState.CLEANUP_REQUIRED)
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)
    service = app.state.tus_service
    cleanup = HostedTusCleanupService(
        service.pool,
        service.s3,
        SimpleNamespace(),
        service.quota,
        service.sessions,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    first = await cleanup.cleanup(session.upload_id, USER_ID)
    second = await cleanup.cleanup(session.upload_id, USER_ID)

    assert first["status"] == "cleaned"
    assert second["status"] == "already_clean"
    assert events.count("quota.release") == 1
    assert events.count("marker.release") == 1


@pytest.mark.asyncio
async def test_graceful_shutdown_relinquishes_current_lease_and_closes_owned_resources(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    started = asyncio.Event()

    async def blocked_handler(*_args):
        started.set()
        await asyncio.Event().wait()

    ctx = _worker_ctx(pool, blocked_handler, worker_id="terminating-worker")
    running = asyncio.create_task(worker.run_job(ctx, str(job["id"])))
    await started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    row = await pool.fetchrow(
        "SELECT state, lease_owner, lease_expires_at, error_code FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert dict(row) == {
        "state": "retry_wait",
        "lease_owner": None,
        "lease_expires_at": None,
        "error_code": "worker_shutdown",
    }

    class Closable:
        def __init__(self):
            self.closed = 0

        async def close(self):
            self.closed += 1

    owned_pool = Closable()
    owned_s3 = Closable()
    arq_redis = Closable()
    shutdown_ctx = {
        "pool": owned_pool,
        "s3": owned_s3,
        "redis": arq_redis,
        worker._OWNED_RESOURCES_CTX_KEY: {"pool": owned_pool, "s3": owned_s3},
        "worker_context": SimpleNamespace(),
    }
    await worker.shutdown(shutdown_ctx)

    assert owned_pool.closed == 1
    assert owned_s3.closed == 1
    assert arq_redis.closed == 0
