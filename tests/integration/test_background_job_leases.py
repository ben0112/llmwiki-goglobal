from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from jobs import repository
from jobs.models import (
    ERROR_CODE_MAX_CHARS,
    ERROR_MESSAGE_MAX_CHARS,
    JobCancelled,
    JobState,
    LeaseLost,
)

NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
OLD_UPDATED_AT = datetime(2020, 1, 1, tzinfo=UTC)


async def _seed_user(pool):
    user_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@job-leases.test",
    )
    return user_id


async def _insert_job(pool, user_id, *, state="queued", **values):
    values.setdefault("updated_at", OLD_UPDATED_AT)
    columns = ["user_id", "job_type", "state", *values]
    parameters = [user_id, "document.extract", state, *values.values()]
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


async def _wait_for_lock_wait(observer_conn, backend_pid, timeout_seconds=1) -> None:
    """Wait until ``backend_pid`` is blocked by another PostgreSQL lock holder."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_wait_event = None
    while True:
        row = await observer_conn.fetchrow(
            "SELECT a.wait_event_type, EXISTS("
            "SELECT 1 FROM pg_locks AS waiting "
            "JOIN pg_locks AS holding "
            "ON holding.locktype = waiting.locktype "
            "AND holding.database IS NOT DISTINCT FROM waiting.database "
            "AND holding.relation IS NOT DISTINCT FROM waiting.relation "
            "AND holding.page IS NOT DISTINCT FROM waiting.page "
            "AND holding.tuple IS NOT DISTINCT FROM waiting.tuple "
            "AND holding.virtualxid IS NOT DISTINCT FROM waiting.virtualxid "
            "AND holding.transactionid IS NOT DISTINCT FROM waiting.transactionid "
            "AND holding.classid IS NOT DISTINCT FROM waiting.classid "
            "AND holding.objid IS NOT DISTINCT FROM waiting.objid "
            "AND holding.objsubid IS NOT DISTINCT FROM waiting.objsubid "
            "WHERE waiting.pid = $1 AND NOT waiting.granted "
            "AND holding.granted AND holding.pid <> waiting.pid"
            ") AS blocked_by_other "
            "FROM pg_stat_activity AS a WHERE a.pid = $1",
            backend_pid,
        )
        if row is not None:
            last_wait_event = row["wait_event_type"]
            if row["wait_event_type"] == "Lock" and row["blocked_by_other"]:
                return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"backend {backend_pid} did not enter a PostgreSQL lock wait; last wait event was {last_wait_event!r}"
            )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_claims_queued_and_due_retry_but_not_future_retry(pool):
    user_id = await _seed_user(pool)
    queued = await _insert_job(pool, user_id, run_after=NOW + timedelta(days=1))
    due_retry = await _insert_job(pool, user_id, state="retry_wait", run_after=NOW, attempt_count=1, max_attempts=3)
    future_retry = await _insert_job(
        pool,
        user_id,
        state="retry_wait",
        run_after=NOW + timedelta(microseconds=1),
    )

    async with pool.acquire() as conn:
        claimed_queued = await repository.claim(conn, queued["id"], "worker-a", 30, now=NOW)
        claimed_retry = await repository.claim(conn, due_retry["id"], "worker-a", 30, now=NOW)
        not_due = await repository.claim(conn, future_retry["id"], "worker-a", 30, now=NOW)

    assert claimed_queued.state is JobState.RUNNING
    assert claimed_queued.attempt_count == 1
    assert claimed_queued.heartbeat_at == NOW
    assert claimed_queued.lease_expires_at == NOW + timedelta(seconds=30)
    assert claimed_queued.updated_at > OLD_UPDATED_AT
    assert claimed_retry.state is JobState.RUNNING
    assert claimed_retry.attempt_count == 2
    assert not_due is None
    unchanged = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", future_retry["id"])
    assert unchanged["attempt_count"] == 0
    assert unchanged["updated_at"] == OLD_UPDATED_AT


@pytest.mark.asyncio
async def test_two_claimers_have_one_winner_and_one_attempt_increment(pool):
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id)
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    observer = await pool.acquire()
    transaction_a = conn_a.transaction()
    pending = None
    committed = False
    try:
        await transaction_a.start()
        first = await repository.claim(conn_a, job["id"], "worker-a", 30, now=NOW)
        backend_b = await conn_b.fetchval("SELECT pg_backend_pid()")
        pending = asyncio.create_task(repository.claim(conn_b, job["id"], "worker-b", 30, now=NOW))
        await _wait_for_lock_wait(observer, backend_b)
        await transaction_a.commit()
        committed = True
        second = await pending
    finally:
        if not committed:
            await transaction_a.rollback()
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        await pool.release(observer)
        await pool.release(conn_b)
        await pool.release(conn_a)

    assert [first is not None, second is not None].count(True) == 1
    row = await pool.fetchrow("SELECT state, lease_owner, attempt_count FROM background_jobs WHERE id = $1", job["id"])
    assert dict(row) == {"state": "running", "lease_owner": "worker-a", "attempt_count": 1}


@pytest.mark.asyncio
async def test_claim_rejects_cancelled_terminal_running_exhausted_missing_and_bad_inputs(pool):
    user_id = await _seed_user(pool)
    rows = [
        await _insert_job(pool, user_id, cancel_requested_at=NOW),
        await _insert_job(pool, user_id, state="succeeded"),
        await _insert_job(pool, user_id, state="running"),
        await _insert_job(pool, user_id, attempt_count=3, max_attempts=3),
    ]
    async with pool.acquire() as conn:
        for row in rows:
            assert await repository.claim(conn, row["id"], "worker", 10, now=NOW) is None
        assert await repository.claim(conn, uuid4(), "worker", 10, now=NOW) is None
        with pytest.raises(ValueError, match="owner"):
            await repository.claim(conn, rows[0]["id"], " ", 10, now=NOW)
        with pytest.raises(ValueError, match="lease_seconds"):
            await repository.claim(conn, rows[0]["id"], "worker", 0, now=NOW)
        for invalid_lease_seconds in (True, 1.5):
            with pytest.raises(ValueError, match="lease_seconds"):
                await repository.claim(
                    conn,
                    rows[0]["id"],
                    "worker",
                    invalid_lease_seconds,
                    now=NOW,
                )


@pytest.mark.asyncio
async def test_heartbeat_extends_only_a_live_owner_lease(pool):
    user_id = await _seed_user(pool)
    live = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW + timedelta(seconds=1),
        heartbeat_at=NOW - timedelta(seconds=2),
    )
    expired = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW,
        heartbeat_at=NOW - timedelta(seconds=2),
    )
    cancelled = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW + timedelta(seconds=1),
        cancel_requested_at=NOW,
    )
    async with pool.acquire() as conn:
        beat = await repository.heartbeat(conn, live["id"], "worker-a", 30, now=NOW)
        cancelled_beat = await repository.heartbeat(conn, cancelled["id"], "worker-a", 30, now=NOW)
        with pytest.raises(LeaseLost):
            await repository.heartbeat(conn, live["id"], "worker-b", 30, now=NOW)
        with pytest.raises(LeaseLost):
            await repository.heartbeat(conn, expired["id"], "worker-a", 30, now=NOW)
        for invalid_lease_seconds in (True, 1.5):
            with pytest.raises(ValueError, match="lease_seconds"):
                await repository.heartbeat(
                    conn,
                    live["id"],
                    "worker-a",
                    invalid_lease_seconds,
                    now=NOW,
                )

    assert beat.heartbeat_at == NOW
    assert beat.lease_expires_at == NOW + timedelta(seconds=30)
    assert beat.updated_at > OLD_UPDATED_AT
    assert cancelled_beat.cancel_requested_at == NOW
    assert cancelled_beat.lease_expires_at == NOW + timedelta(seconds=30)


@pytest.mark.asyncio
async def test_assert_active_distinguishes_cancellation_from_lost_lease(pool):
    user_id = await _seed_user(pool)
    live = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW + timedelta(seconds=1),
    )
    cancelled = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW + timedelta(seconds=1),
        cancel_requested_at=NOW,
    )
    expired_equal = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW,
    )
    expired_earlier = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=NOW - timedelta(microseconds=1),
        cancel_requested_at=NOW,
    )
    async with pool.acquire() as conn:
        assert (await repository.assert_active(conn, live["id"], "worker-a", now=NOW)).id == live["id"]
        with pytest.raises(JobCancelled):
            await repository.assert_active(conn, cancelled["id"], "worker-a", now=NOW)
        for expired in (expired_equal, expired_earlier):
            with pytest.raises(LeaseLost):
                await repository.assert_active(conn, expired["id"], "worker-a", now=NOW)
        for job_id, owner in ((live["id"], "worker-b"), (uuid4(), "worker-a")):
            with pytest.raises(LeaseLost):
                await repository.assert_active(conn, job_id, owner, now=NOW)
        await conn.execute("UPDATE background_jobs SET state = 'failed' WHERE id = $1", live["id"])
        with pytest.raises(LeaseLost):
            await repository.assert_active(conn, live["id"], "worker-a", now=NOW)


@pytest.mark.asyncio
async def test_succeed_persists_strict_json_and_rejects_stale_or_cancelled_workers(pool):
    user_id = await _seed_user(pool)

    async def running(**changes):
        values = {
            "lease_owner": "worker-a",
            "lease_expires_at": NOW + timedelta(seconds=1),
            "heartbeat_at": NOW,
            "progress": '{"percent": 50}',
            "error_code": "old",
            "error_message": "old",
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    live = await running()
    expired = await running(lease_expires_at=NOW)
    cancelled = await running(cancel_requested_at=NOW)
    async with pool.acquire() as conn:
        succeeded = await repository.succeed(conn, live["id"], "worker-a", {"nested": [{"ok": True}]}, now=NOW)
        for job_id, owner, error in (
            (live["id"], "worker-a", LeaseLost),
            (expired["id"], "worker-a", LeaseLost),
            (cancelled["id"], "worker-a", JobCancelled),
            (cancelled["id"], "worker-b", LeaseLost),
        ):
            with pytest.raises(error):
                await repository.succeed(conn, job_id, owner, {"ok": True}, now=NOW)
        with pytest.raises(TypeError):
            await repository.succeed(conn, uuid4(), "worker-a", {"bad": object()}, now=NOW)

    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.result == {"nested": ({"ok": True},)}
    assert succeeded.progress is None
    assert succeeded.error_code is None
    assert succeeded.error_message is None
    assert succeeded.lease_owner is None
    assert succeeded.lease_expires_at is None
    assert succeeded.heartbeat_at is None


@pytest.mark.asyncio
async def test_fail_or_retry_handles_cancel_retry_failure_and_exhaustion(pool):
    user_id = await _seed_user(pool)

    async def running(*, attempt_count=1, max_attempts=3, **changes):
        values = {
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "lease_owner": "worker-a",
            "lease_expires_at": NOW + timedelta(seconds=1),
            "heartbeat_at": NOW,
            "progress": '{"percent": 50}',
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    cancelled = await running(cancel_requested_at=NOW)
    retry = await running(attempt_count=2)
    nonretry = await running()
    exhausted = await running(attempt_count=3, max_attempts=3)
    async with pool.acquire() as conn:
        cancelled_result = await repository.fail_or_retry(
            conn,
            cancelled["id"],
            "worker-a",
            error_code="internal",
            error_message="hidden",
            retryable=True,
            now=NOW,
        )
        retry_result = await repository.fail_or_retry(
            conn,
            retry["id"],
            "worker-a",
            error_code="temporary",
            error_message="try again",
            retryable=True,
            now=NOW,
        )
        failed_result = await repository.fail_or_retry(
            conn,
            nonretry["id"],
            "worker-a",
            error_code="invalid",
            error_message="cannot retry",
            retryable=False,
            now=NOW,
        )
        exhausted_result = await repository.fail_or_retry(
            conn,
            exhausted["id"],
            "worker-a",
            error_code="temporary",
            error_message="still failing",
            retryable=True,
            now=NOW,
        )
        with pytest.raises(LeaseLost):
            await repository.fail_or_retry(
                conn,
                retry["id"],
                "worker-a",
                error_code="temporary",
                error_message="again",
                retryable=True,
                now=NOW,
            )
        for field, value in (
            ("error_code", "x" * (ERROR_CODE_MAX_CHARS + 1)),
            ("error_message", "x" * (ERROR_MESSAGE_MAX_CHARS + 1)),
        ):
            arguments = {"error_code": "code", "error_message": "message", field: value}
            with pytest.raises(ValueError, match=field):
                await repository.fail_or_retry(
                    conn,
                    uuid4(),
                    "worker-a",
                    retryable=False,
                    now=NOW,
                    **arguments,
                )

    assert cancelled_result.state is JobState.CANCELLED
    assert cancelled_result.cancel_requested_at == NOW
    assert cancelled_result.error_code is None
    assert cancelled_result.error_message is None
    assert retry_result.state is JobState.RETRY_WAIT
    assert retry_result.run_after == NOW + timedelta(seconds=4)
    assert retry_result.error_code == "temporary"
    assert retry_result.error_message == "try again"
    assert failed_result.state is JobState.FAILED
    assert failed_result.error_code == "invalid"
    assert exhausted_result.state is JobState.FAILED
    assert exhausted_result.error_code == "attempts_exhausted"
    assert exhausted_result.error_message == "The job could not be completed after retrying."
    for record in (cancelled_result, retry_result, failed_result, exhausted_result):
        assert record.lease_owner is None
        assert record.lease_expires_at is None
        assert record.heartbeat_at is None
        assert record.progress is None
        assert record.updated_at > OLD_UPDATED_AT


@pytest.mark.asyncio
async def test_reaper_handles_all_branches_limit_repeat_and_renewed_lease(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)

    async def expired(**changes):
        values = {
            "attempt_count": 1,
            "max_attempts": 3,
            "lease_owner": "dead-worker",
            "lease_expires_at": NOW,
            "heartbeat_at": NOW - timedelta(seconds=30),
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    retry = await expired()
    cancelled = await expired(cancel_requested_at=NOW)
    exhausted = await expired(attempt_count=3, max_attempts=3)
    renewed = await expired(lease_expires_at=NOW + timedelta(microseconds=1))
    async with pool.acquire() as conn:
        first = await repository.reap_expired(conn, now=NOW, limit=2)
        second = await repository.reap_expired(conn, now=NOW, limit=2)
        repeated = await repository.reap_expired(conn, now=NOW, limit=2)
        for invalid_limit in (0, True, 1.5):
            with pytest.raises(ValueError, match="limit"):
                await repository.reap_expired(conn, now=NOW, limit=invalid_limit)

    assert len(first) == 2
    assert len(second) == 1
    assert repeated == []
    assert set(first + second) == {retry["id"], cancelled["id"], exhausted["id"]}
    rows = {
        row["id"]: row
        for row in await pool.fetch(
            "SELECT id, state, run_after, lease_owner, lease_expires_at, heartbeat_at, "
            "error_code, error_message FROM background_jobs WHERE id = ANY($1::uuid[])",
            [retry["id"], cancelled["id"], exhausted["id"], renewed["id"]],
        )
    }
    assert rows[retry["id"]]["state"] == "retry_wait"
    assert rows[retry["id"]]["run_after"] == NOW
    assert rows[retry["id"]]["error_code"] == "lease_expired"
    assert rows[cancelled["id"]]["state"] == "cancelled"
    assert rows[cancelled["id"]]["error_code"] is None
    assert rows[exhausted["id"]]["state"] == "failed"
    assert rows[exhausted["id"]]["error_code"] == "attempts_exhausted"
    assert rows[renewed["id"]]["state"] == "running"
    for job_id in (retry["id"], cancelled["id"], exhausted["id"]):
        assert rows[job_id]["lease_owner"] is None
        assert rows[job_id]["lease_expires_at"] is None
        assert rows[job_id]["heartbeat_at"] is None


@pytest.mark.asyncio
async def test_concurrent_reapers_return_disjoint_ids_with_skip_locked(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    jobs = [
        await _insert_job(
            pool,
            user_id,
            state="running",
            attempt_count=1,
            max_attempts=3,
            lease_owner="dead-worker",
            lease_expires_at=NOW,
        )
        for _ in range(2)
    ]
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    transaction_a = conn_a.transaction()
    committed = False
    try:
        await transaction_a.start()
        first = await repository.reap_expired(conn_a, now=NOW, limit=1)
        second = await repository.reap_expired(conn_b, now=NOW, limit=1)
        await transaction_a.commit()
        committed = True
    finally:
        if not committed:
            await transaction_a.rollback()
        await pool.release(conn_b)
        await pool.release(conn_a)

    assert len(first) == len(second) == 1
    assert set(first).isdisjoint(second)
    assert set(first + second) == {job["id"] for job in jobs}
