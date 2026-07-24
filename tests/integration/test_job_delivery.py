from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from jobs.dispatcher import dispatch_due_jobs, mark_dispatched, select_due_job_ids
from jobs.handlers import WorkerContext
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
        return await select_due_job_ids(
            conn,
            batch_size=batch_size,
            redeliver_seconds=redeliver_seconds,
        )


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
        assert await select_due_job_ids(conn_a, batch_size=1, redeliver_seconds=30) == [first["id"]]
        async with conn_b.transaction(), asyncio.timeout(1):
            skipped = await select_due_job_ids(conn_b, batch_size=1, redeliver_seconds=30)
        assert skipped == [second["id"]]
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
    terminal = await _insert_job(pool, user_id, state="succeeded", run_after=OLD)
    before = await pool.fetchval("SELECT clock_timestamp()")
    async with pool.acquire() as conn, conn.transaction():
        assert await mark_dispatched(conn, queued["id"]) is True
        assert await mark_dispatched(conn, terminal["id"]) is False
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
