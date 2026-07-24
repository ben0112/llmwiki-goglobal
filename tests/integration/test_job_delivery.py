from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from jobs import repository
from jobs.dispatcher import dispatch_due_jobs, mark_dispatched, select_due_jobs
from jobs.handlers import TerminalJobError, WorkerContext
from jobs.models import JobState, JobType

OLD = datetime(2020, 1, 1, tzinfo=UTC)


async def _seed_user(pool):
    user_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@job-delivery.test",
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


async def _select(pool, *, batch_size=100, redeliver_seconds=30):
    async with pool.acquire() as conn, conn.transaction():
        selected = await select_due_jobs(
            conn,
            batch_size=batch_size,
            redeliver_seconds=redeliver_seconds,
        )
    return [candidate.job_id for candidate in selected]


@pytest.mark.asyncio
async def test_due_selection_filters_orders_batches_and_redelivers_by_database_time(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    now = await pool.fetchval("SELECT clock_timestamp()")
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    first = await _insert_job(
        pool,
        user_id,
        id=first_id,
        run_after=OLD,
        created_at=OLD,
    )
    second = await _insert_job(
        pool,
        user_id,
        state="retry_wait",
        id=second_id,
        run_after=OLD,
        created_at=OLD,
        attempt_count=1,
    )
    future = await _insert_job(pool, user_id, run_after=now + timedelta(hours=1))
    recent = await _insert_job(pool, user_id, run_after=OLD, last_dispatched_at=now)
    cancelled = await _insert_job(pool, user_id, run_after=OLD, cancel_requested_at=now)
    exhausted = await _insert_job(pool, user_id, run_after=OLD, attempt_count=3, max_attempts=3)
    terminal = [
        await _insert_job(pool, user_id, state=state, run_after=OLD) for state in ("succeeded", "failed", "cancelled")
    ]

    assert await _select(pool, batch_size=1) == [first["id"]]
    assert await _select(pool, batch_size=2) == [first["id"], second["id"]]
    selected = set(await _select(pool, redeliver_seconds=30))
    assert future["id"] not in selected
    assert recent["id"] not in selected
    assert cancelled["id"] not in selected
    assert exhausted["id"] not in selected
    assert not ({row["id"] for row in terminal} & selected)

    await pool.execute(
        "UPDATE background_jobs SET last_dispatched_at = clock_timestamp() - interval '31 seconds' WHERE id = $1",
        recent["id"],
    )
    assert recent["id"] in await _select(pool, redeliver_seconds=30)


@pytest.mark.asyncio
async def test_new_retry_generation_dispatches_when_due_without_waiting_for_redelivery(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    now = await pool.fetchval("SELECT clock_timestamp()")
    new_generation = await _insert_job(
        pool,
        user_id,
        state="retry_wait",
        attempt_count=1,
        run_after=now - timedelta(seconds=1),
        last_dispatched_at=now - timedelta(seconds=5),
    )
    same_generation = await _insert_job(
        pool,
        user_id,
        state="retry_wait",
        attempt_count=1,
        run_after=now - timedelta(seconds=5),
        last_dispatched_at=now - timedelta(seconds=1),
    )

    selected = await _select(pool, redeliver_seconds=30)

    assert new_generation["id"] in selected
    assert same_generation["id"] not in selected


@pytest.mark.asyncio
async def test_select_due_jobs_uses_skip_locked_with_deterministic_row_lock_evidence(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    first = await _insert_job(pool, user_id, id=UUID(int=101), run_after=OLD, created_at=OLD)
    second = await _insert_job(pool, user_id, id=UUID(int=102), run_after=OLD, created_at=OLD)
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    tx_a = conn_a.transaction()
    try:
        await tx_a.start()
        assert [
            candidate.job_id for candidate in await select_due_jobs(conn_a, batch_size=1, redeliver_seconds=30)
        ] == [first["id"]]
        async with conn_b.transaction(), asyncio.timeout(1):
            skipped = await select_due_jobs(conn_b, batch_size=1, redeliver_seconds=30)
        assert [candidate.job_id for candidate in skipped] == [second["id"]]
    finally:
        await tx_a.rollback()
        await pool.release(conn_b)
        await pool.release(conn_a)


@pytest.mark.asyncio
async def test_mark_dispatched_changes_only_dispatch_metadata_and_rejects_terminal(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    queued = await _insert_job(
        pool,
        user_id,
        run_after=OLD,
        payload='{"secret": "kept"}',
        attempt_count=1,
        max_attempts=4,
        lease_owner="old-owner",
        lease_expires_at=OLD,
        heartbeat_at=OLD,
    )
    running = await _insert_job(pool, user_id, state="running", run_after=OLD)
    terminal = await _insert_job(pool, user_id, state="succeeded", run_after=OLD)
    before = await pool.fetchval("SELECT clock_timestamp()")
    async with pool.acquire() as conn, conn.transaction():
        assert await mark_dispatched(conn, queued["id"], queued["run_after"]) is True
        assert await mark_dispatched(conn, running["id"], running["run_after"]) is True
        assert await mark_dispatched(conn, terminal["id"], terminal["run_after"]) is False
    after = await pool.fetchval("SELECT clock_timestamp()")

    changed = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", queued["id"])
    unchanged_fields = {
        "state",
        "payload",
        "attempt_count",
        "max_attempts",
        "run_after",
        "lease_owner",
        "lease_expires_at",
        "heartbeat_at",
        "cancel_requested_at",
    }
    for field in unchanged_fields:
        assert changed[field] == queued[field]
    assert changed["dispatch_attempts"] == queued["dispatch_attempts"] + 1
    assert before <= changed["last_dispatched_at"] <= after
    assert (
        await pool.fetchval(
            "SELECT dispatch_attempts FROM background_jobs WHERE id = $1",
            running["id"],
        )
        == 1
    )
    assert dict(await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", terminal["id"])) == dict(terminal)


class AlwaysFailRedis:
    async def enqueue_job(self, *_args, **_kwargs):
        raise ConnectionError("redis unavailable")


@pytest.mark.asyncio
async def test_redis_failure_leaves_row_immediately_due_and_dispatch_metadata_unchanged(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)

    summary = await dispatch_due_jobs(
        pool,
        AlwaysFailRedis(),
        batch_size=10,
        redeliver_seconds=30,
    )

    assert summary.enqueue_failed == 1
    row = await pool.fetchrow(
        "SELECT last_dispatched_at, dispatch_attempts FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert dict(row) == {"last_dispatched_at": None, "dispatch_attempts": 0}
    assert await _select(pool) == [job["id"]]


class CollapsingRedis:
    def __init__(self, expected_calls):
        self.expected_calls = expected_calls
        self.calls = []
        self.first_arrived = asyncio.Event()
        self._arrived = asyncio.Event()
        self._seen = set()
        self._lock = asyncio.Lock()

    async def enqueue_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.first_arrived.set()
        if len(self.calls) == self.expected_calls:
            self._arrived.set()
        await asyncio.wait_for(self._arrived.wait(), timeout=1)
        async with self._lock:
            job_id = kwargs["_job_id"]
            if job_id in self._seen:
                return None
            self._seen.add(job_id)
            return object()


@pytest.mark.asyncio
async def test_concurrent_dispatchers_release_database_then_arq_collapses_duplicate(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    redis = CollapsingRedis(expected_calls=2)

    async with asyncio.timeout(2):
        first_dispatcher = asyncio.create_task(dispatch_due_jobs(pool, redis, batch_size=1, redeliver_seconds=30))
        await redis.first_arrived.wait()
        second_dispatcher = asyncio.create_task(dispatch_due_jobs(pool, redis, batch_size=1, redeliver_seconds=30))
        summaries = await asyncio.gather(first_dispatcher, second_dispatcher)

    assert sum(summary.enqueued for summary in summaries) == 1
    assert sum(summary.already_present for summary in summaries) == 1
    assert sum(summary.marked for summary in summaries) == 2
    expected = (("run_job", str(job["id"])), {"_job_id": str(job["id"])})
    assert redis.calls == [expected, expected]
    assert len(redis._seen) == 1
    assert await pool.fetchval("SELECT dispatch_attempts FROM background_jobs WHERE id = $1", job["id"]) == 2


@pytest.mark.asyncio
async def test_stale_delivery_mark_cannot_throttle_a_due_retry_generation(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)

    class RetryBeforeDeliveryReturnsRedis:
        def __init__(self):
            self.retried = None
            self.due_run_after = None

        async def enqueue_job(self, function, job_id_text, **kwargs):
            assert (function, job_id_text) == ("run_job", str(job["id"]))
            assert kwargs == {"_job_id": str(job["id"])}
            async with pool.acquire() as conn, conn.transaction():
                claimed = await repository.claim(conn, job["id"], "race-worker", 30)
                assert claimed is not None
                self.retried = await repository.fail_or_retry(
                    conn,
                    job["id"],
                    "race-worker",
                    error_code="transient",
                    error_message="The job could not be completed.",
                    retryable=True,
                )
                self.due_run_after = await conn.fetchval(
                    "UPDATE background_jobs "
                    "SET run_after = clock_timestamp() - interval '1 second' "
                    "WHERE id = $1 RETURNING run_after",
                    job["id"],
                )
            return object()

    redis = RetryBeforeDeliveryReturnsRedis()

    async with asyncio.timeout(1):
        summary = await dispatch_due_jobs(
            pool,
            redis,
            batch_size=1,
            redeliver_seconds=30,
        )

    assert summary.selected == 1
    assert summary.enqueued == 1
    assert summary.marked == 0
    assert summary.mark_failed == 1
    assert redis.retried is not None
    assert redis.due_run_after is not None
    assert redis.retried.run_after > redis.due_run_after
    row = await pool.fetchrow(
        "SELECT state, run_after, last_dispatched_at, dispatch_attempts FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert row["state"] == "retry_wait"
    assert row["run_after"] == redis.due_run_after
    assert row["last_dispatched_at"] is None
    assert row["dispatch_attempts"] == 0
    assert job["id"] in await _select(pool, redeliver_seconds=30)


@pytest.mark.asyncio
async def test_worker_claims_once_uses_persisted_type_and_records_success_with_real_ledger(pool):
    from jobs import worker

    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(
        pool,
        user_id,
        job_type=JobType.GRAPH_REBUILD.value,
        run_after=OLD,
        payload='{"source": "postgres"}',
    )
    handled = []

    async def handler(record, lease, context):
        handled.append((record.job_type, dict(record.payload), context))
        assert (await lease.checkpoint()).state is JobState.RUNNING
        return {"completed": True}

    context = WorkerContext(pool=pool, s3=None, converter_url="", converter_secret="")
    ctx = {
        "pool": pool,
        "worker_id": "integration-worker",
        "lease_seconds": 30,
        "heartbeat_seconds": 10,
        "worker_context": context,
        "handlers": {JobType.GRAPH_REBUILD: handler},
    }

    first = await worker.run_job(ctx, str(job["id"]))
    duplicate = await worker.run_job(ctx, str(job["id"]))

    assert first == {"status": "succeeded", "job_id": str(job["id"])}
    assert duplicate == {"status": "duplicate", "job_id": str(job["id"])}
    assert [(job_type, payload) for job_type, payload, _ in handled] == [
        (JobType.GRAPH_REBUILD, {"source": "postgres"})
    ]
    row = await pool.fetchrow(
        "SELECT state, attempt_count, result, lease_owner, lease_expires_at FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert row["state"] == "succeeded"
    assert row["attempt_count"] == 1
    assert row["result"] == '{"completed": true}'
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


async def _run_result_job(pool, monkeypatch, result):
    from jobs import worker

    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)
    succeed_calls = []
    real_succeed = worker.repository.succeed

    async def handler(*_args):
        return result

    async def succeed(conn, job_id, owner, persisted_result):
        succeed_calls.append((conn, job_id, owner, persisted_result))
        return await real_succeed(conn, job_id, owner, persisted_result)

    monkeypatch.setattr(worker.repository, "succeed", succeed)
    context = WorkerContext(pool=pool, s3=None, converter_url="", converter_secret="")
    ctx = {
        "pool": pool,
        "worker_id": "result-validation-worker",
        "lease_seconds": 30,
        "heartbeat_seconds": 10,
        "worker_context": context,
        "handlers": {JobType.DOCUMENT_EXTRACT: handler},
    }
    outcome = await worker.run_job(ctx, str(job["id"]))
    row = await pool.fetchrow(
        "SELECT state, error_code, error_message, result, attempt_count FROM background_jobs WHERE id = $1",
        job["id"],
    )
    return job, outcome, row, succeed_calls


@pytest.mark.asyncio
async def test_postgres_canonical_exponent_expansion_is_terminal_before_succeed(
    pool,
    monkeypatch,
):
    result = {"x": [1e300] * 55}
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert len(serialized.encode("utf-8")) < 16_384
    assert await pool.fetchval("SELECT octet_length($1::jsonb::text)", serialized) > 16_384

    job, outcome, row, succeed_calls = await _run_result_job(pool, monkeypatch, result)

    assert outcome == {"status": "failed", "job_id": str(job["id"])}
    assert succeed_calls == []
    assert dict(row) == {
        "state": "failed",
        "error_code": "invalid_job_result",
        "error_message": "The job produced an invalid result.",
        "result": None,
        "attempt_count": 1,
    }


@pytest.mark.asyncio
async def test_postgres_rejects_nul_result_without_leaking_or_calling_succeed(
    pool,
    monkeypatch,
    caplog,
):
    result = {"x": "TOP_SECRET_TOKEN\x00RAW_RESULT_TOKEN"}
    caplog.set_level(logging.ERROR, logger="jobs.worker")

    job, outcome, row, succeed_calls = await _run_result_job(pool, monkeypatch, result)

    assert outcome == {"status": "failed", "job_id": str(job["id"])}
    assert succeed_calls == []
    assert dict(row) == {
        "state": "failed",
        "error_code": "invalid_job_result",
        "error_message": "The job produced an invalid result.",
        "result": None,
        "attempt_count": 1,
    }
    assert "TOP_SECRET_TOKEN" not in caplog.text
    assert "RAW_RESULT_TOKEN" not in caplog.text
    assert "TOP_SECRET_TOKEN" not in repr(dict(row))
    assert "RAW_RESULT_TOKEN" not in repr(dict(row))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_message",
    [
        "TOP_SECRET_HANDLER_TOKEN\x00postgres unsafe",
        "TOP_SECRET_HANDLER_TOKEN\ud800",
    ],
)
async def test_handler_failure_messages_are_postgres_safe_and_do_not_leak(
    pool,
    caplog,
    unsafe_message,
):
    from jobs import worker

    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD)

    async def handler(*_args):
        raise TerminalJobError("unsafe_handler_failure", unsafe_message)

    context = WorkerContext(pool=pool, s3=None, converter_url="", converter_secret="")
    ctx = {
        "pool": pool,
        "worker_id": "safe-message-worker",
        "lease_seconds": 30,
        "heartbeat_seconds": 10,
        "worker_context": context,
        "handlers": {JobType.DOCUMENT_EXTRACT: handler},
    }
    caplog.set_level(logging.ERROR, logger="jobs.worker")

    outcome = await worker.run_job(ctx, str(job["id"]))

    row = await pool.fetchrow(
        "SELECT state, error_code, error_message FROM background_jobs WHERE id = $1",
        job["id"],
    )
    assert outcome == {"status": "failed", "job_id": str(job["id"])}
    assert dict(row) == {
        "state": "failed",
        "error_code": "unsafe_handler_failure",
        "error_message": "The job could not be completed.",
    }
    assert "TOP_SECRET_HANDLER_TOKEN" not in caplog.text
    assert "TOP_SECRET_HANDLER_TOKEN" not in repr(dict(row))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,expected_bytes,expected_state,expected_succeed_calls",
    [
        ({"x": "a" * 16_375}, 16_384, "succeeded", 1),
        ({"x": "a" * 16_376}, 16_385, "failed", 0),
        ({"x": "你" * 5_458 + "a"}, 16_384, "succeeded", 1),
        ({"x": "你" * 5_458 + "aa"}, 16_385, "failed", 0),
    ],
)
async def test_postgres_canonical_result_boundary_controls_succeed(
    pool,
    monkeypatch,
    result,
    expected_bytes,
    expected_state,
    expected_succeed_calls,
):
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
    canonical_bytes = await pool.fetchval(
        "SELECT octet_length($1::jsonb::text)",
        serialized,
    )
    assert canonical_bytes == expected_bytes

    job, outcome, row, succeed_calls = await _run_result_job(pool, monkeypatch, result)

    assert row["state"] == expected_state
    assert len(succeed_calls) == expected_succeed_calls
    if expected_state == "succeeded":
        assert outcome == {"status": "succeeded", "job_id": str(job["id"])}
        assert row["error_code"] is None
        assert row["result"] is not None
    else:
        assert outcome == {"status": "failed", "job_id": str(job["id"])}
        assert row["error_code"] == "invalid_job_result"
        assert row["error_message"] == "The job produced an invalid result."
        assert row["result"] is None
