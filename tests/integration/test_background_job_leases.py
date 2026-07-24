from __future__ import annotations

import asyncio
import inspect
import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest
from jobs import repository
from jobs.models import (
    ERROR_CODE_MAX_CHARS,
    ERROR_MESSAGE_MAX_CHARS,
    JobCancelled,
    JobState,
    LeaseLost,
)

OLD_UPDATED_AT = datetime(2020, 1, 1, tzinfo=UTC)


async def _db_now(pool):
    return await pool.fetchval("SELECT clock_timestamp()")


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
async def test_claims_only_database_due_queued_and_retry_jobs(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    due = database_now - timedelta(minutes=1)
    future = database_now + timedelta(hours=1)
    queued = await _insert_job(pool, user_id, run_after=due)
    queued_future = await _insert_job(pool, user_id, run_after=future)
    due_retry = await _insert_job(
        pool,
        user_id,
        state="retry_wait",
        run_after=due,
        attempt_count=1,
        max_attempts=3,
    )
    future_retry = await _insert_job(pool, user_id, state="retry_wait", run_after=future)

    before = await _db_now(pool)
    async with pool.acquire() as conn:
        claimed_queued = await repository.claim(conn, queued["id"], "worker-a", 30)
        claimed_retry = await repository.claim(conn, due_retry["id"], "worker-a", 30)
        queued_not_due = await repository.claim(conn, queued_future["id"], "worker-a", 30)
        retry_not_due = await repository.claim(conn, future_retry["id"], "worker-a", 30)
    after = await _db_now(pool)

    for claimed, attempts in ((claimed_queued, 1), (claimed_retry, 2)):
        assert claimed.state is JobState.RUNNING
        assert claimed.attempt_count == attempts
        assert before <= claimed.heartbeat_at <= after
        assert claimed.lease_expires_at - claimed.heartbeat_at == timedelta(seconds=30)
        assert claimed.updated_at > OLD_UPDATED_AT
    assert queued_not_due is None
    assert retry_not_due is None
    unchanged = await pool.fetch(
        "SELECT id, attempt_count, updated_at FROM background_jobs WHERE id = ANY($1::uuid[])",
        [queued_future["id"], future_retry["id"]],
    )
    assert {row["attempt_count"] for row in unchanged} == {0}
    assert {row["updated_at"] for row in unchanged} == {OLD_UPDATED_AT}


@pytest.mark.asyncio
async def test_two_claimers_have_one_winner_and_one_attempt_increment(pool):
    user_id = await _seed_user(pool)
    job = await _insert_job(pool, user_id, run_after=OLD_UPDATED_AT)
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    observer = await pool.acquire()
    transaction_a = conn_a.transaction()
    pending = None
    committed = False
    try:
        await transaction_a.start()
        first = await repository.claim(conn_a, job["id"], "worker-a", 30)
        backend_b = await conn_b.fetchval("SELECT pg_backend_pid()")
        pending = asyncio.create_task(repository.claim(conn_b, job["id"], "worker-b", 30))
        await _wait_for_lock_wait(observer, backend_b)
        await transaction_a.commit()
        committed = True
        async with asyncio.timeout(1):
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
async def test_claim_rejects_ineligible_rows_and_non_integral_lease_durations(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    rows = [
        await _insert_job(pool, user_id, run_after=OLD_UPDATED_AT, cancel_requested_at=database_now),
        await _insert_job(pool, user_id, state="succeeded"),
        await _insert_job(pool, user_id, state="running"),
        await _insert_job(pool, user_id, run_after=OLD_UPDATED_AT, attempt_count=3, max_attempts=3),
    ]
    async with pool.acquire() as conn:
        for row in rows:
            assert await repository.claim(conn, row["id"], "worker", 10) is None
        assert await repository.claim(conn, uuid4(), "worker", 10) is None
        with pytest.raises(ValueError, match="owner"):
            await repository.claim(conn, rows[0]["id"], " ", 10)
        for invalid_lease_seconds in (0, True, 1.5):
            with pytest.raises(ValueError, match="lease_seconds"):
                await repository.claim(conn, rows[0]["id"], "worker", invalid_lease_seconds)


@pytest.mark.asyncio
async def test_heartbeat_extends_only_a_live_owner_lease_using_database_time(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    live_expiry = database_now + timedelta(minutes=1)
    expired_at = database_now - timedelta(minutes=1)
    live = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=live_expiry,
        heartbeat_at=database_now,
    )
    expired = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=expired_at,
        heartbeat_at=expired_at,
    )
    cancelled = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=live_expiry,
        cancel_requested_at=database_now,
    )
    before = await _db_now(pool)
    async with pool.acquire() as conn:
        beat = await repository.heartbeat(conn, live["id"], "worker-a", 30)
        cancelled_beat = await repository.heartbeat(conn, cancelled["id"], "worker-a", 30)
        with pytest.raises(LeaseLost):
            await repository.heartbeat(conn, live["id"], "worker-b", 30)
        with pytest.raises(LeaseLost):
            await repository.heartbeat(conn, expired["id"], "worker-a", 30)
        for invalid_lease_seconds in (True, 1.5):
            with pytest.raises(ValueError, match="lease_seconds"):
                await repository.heartbeat(conn, live["id"], "worker-a", invalid_lease_seconds)
    after = await _db_now(pool)

    assert before <= beat.heartbeat_at <= after
    assert beat.lease_expires_at - beat.heartbeat_at == timedelta(seconds=30)
    assert beat.updated_at > OLD_UPDATED_AT
    assert cancelled_beat.cancel_requested_at == database_now
    assert cancelled_beat.lease_expires_at - cancelled_beat.heartbeat_at == timedelta(seconds=30)


@pytest.mark.asyncio
async def test_assert_active_locks_and_distinguishes_live_cancellation_from_expiry(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    live_expiry = database_now + timedelta(minutes=1)
    expired_at = database_now - timedelta(minutes=1)
    live = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=live_expiry,
    )
    cancelled = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=live_expiry,
        cancel_requested_at=database_now,
    )
    expired_equal_or_earlier = [
        await _insert_job(
            pool,
            user_id,
            state="running",
            lease_owner="worker-a",
            lease_expires_at=expired_at,
            cancel_requested_at=database_now if index else None,
        )
        for index in range(2)
    ]
    async with pool.acquire() as conn:
        async with conn.transaction():
            assert (await repository.assert_active(conn, live["id"], "worker-a")).id == live["id"]
        with pytest.raises(JobCancelled):
            await repository.assert_active(conn, cancelled["id"], "worker-a")
        for expired in expired_equal_or_earlier:
            with pytest.raises(LeaseLost):
                await repository.assert_active(conn, expired["id"], "worker-a")
        for job_id, owner in ((live["id"], "worker-b"), (uuid4(), "worker-a")):
            with pytest.raises(LeaseLost):
                await repository.assert_active(conn, job_id, owner)
        await conn.execute("UPDATE background_jobs SET state = 'failed' WHERE id = $1", live["id"])
        with pytest.raises(LeaseLost):
            await repository.assert_active(conn, live["id"], "worker-a")


@pytest.mark.asyncio
async def test_succeed_persists_strict_json_and_rejects_stale_or_cancelled_workers(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)

    async def running(**changes):
        values = {
            "lease_owner": "worker-a",
            "lease_expires_at": database_now + timedelta(minutes=1),
            "heartbeat_at": database_now,
            "progress": '{"percent": 50}',
            "error_code": "old",
            "error_message": "old",
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    live = await running()
    expired = await running(lease_expires_at=database_now - timedelta(minutes=1))
    cancelled = await running(cancel_requested_at=database_now)
    async with pool.acquire() as conn:
        succeeded = await repository.succeed(conn, live["id"], "worker-a", {"nested": [{"ok": True}]})
        for job_id, owner, error in (
            (live["id"], "worker-a", LeaseLost),
            (expired["id"], "worker-a", LeaseLost),
            (cancelled["id"], "worker-a", JobCancelled),
            (cancelled["id"], "worker-b", LeaseLost),
        ):
            with pytest.raises(error):
                await repository.succeed(conn, job_id, owner, {"ok": True})
        with pytest.raises(TypeError):
            await repository.succeed(conn, uuid4(), "worker-a", {"bad": object()})

    assert succeeded.state is JobState.SUCCEEDED
    assert succeeded.result == {"nested": ({"ok": True},)}
    assert succeeded.progress is None
    assert succeeded.error_code is None
    assert succeeded.error_message is None
    assert succeeded.lease_owner is None
    assert succeeded.lease_expires_at is None
    assert succeeded.heartbeat_at is None


@pytest.mark.asyncio
async def test_succeed_classifies_cancellation_from_its_locked_statement(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    cancelled = await _insert_job(
        pool,
        user_id,
        state="running",
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker-a",
        lease_expires_at=database_now + timedelta(minutes=1),
        cancel_requested_at=database_now,
    )
    conn_succeed = await pool.acquire()
    conn_transition = await pool.acquire()
    statement_done = asyncio.Event()
    release_result = asyncio.Event()

    class InterleavingConnection:
        def __init__(self, conn):
            self.conn = conn
            self.fetchval_calls = 0

        async def fetchrow(self, query, *args):
            row = await self.conn.fetchrow(query, *args)
            statement_done.set()
            await release_result.wait()
            return row

        async def fetchval(self, query, *args):
            self.fetchval_calls += 1
            return await self.conn.fetchval(query, *args)

    interleaving = InterleavingConnection(conn_succeed)
    pending = asyncio.create_task(repository.succeed(interleaving, cancelled["id"], "worker-a", {"ok": True}))
    try:
        async with asyncio.timeout(1):
            await statement_done.wait()
        transitioned = await repository.fail_or_retry(
            conn_transition,
            cancelled["id"],
            "worker-a",
            error_code="internal",
            error_message="hidden",
            retryable=False,
        )
        release_result.set()
        with pytest.raises(JobCancelled):
            async with asyncio.timeout(1):
                await pending
    finally:
        release_result.set()
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
        await pool.release(conn_transition)
        await pool.release(conn_succeed)

    assert transitioned.state is JobState.CANCELLED
    assert interleaving.fetchval_calls == 0


@pytest.mark.asyncio
async def test_application_clock_cannot_bypass_database_expiry(pool):
    for operation in (
        repository.claim,
        repository.heartbeat,
        repository.assert_active,
        repository.succeed,
        repository.fail_or_retry,
        repository.reap_expired,
    ):
        assert "now" not in inspect.signature(operation).parameters

    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    expired = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=database_now - timedelta(minutes=1),
    )
    application_old_time = database_now - timedelta(days=365)
    async with pool.acquire() as conn:
        with pytest.raises(LeaseLost):
            await repository.heartbeat(conn, expired["id"], "worker-a", 30)
        with pytest.raises(LeaseLost):
            await repository.succeed(conn, expired["id"], "worker-a", {"ok": True})
        with pytest.raises(TypeError, match="now"):
            await repository.heartbeat(
                conn,
                expired["id"],
                "worker-a",
                30,
                now=application_old_time,
            )


@pytest.mark.asyncio
async def test_fail_or_retry_uses_database_time_and_persisted_attempt_count(pool):
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)

    async def running(*, attempt_count=1, max_attempts=3, **changes):
        values = {
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "lease_owner": "worker-a",
            "lease_expires_at": database_now + timedelta(minutes=1),
            "heartbeat_at": database_now,
            "progress": '{"percent": 50}',
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    cancelled = await running(cancel_requested_at=database_now)
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
        )
        before_retry = await conn.fetchval("SELECT clock_timestamp()")
        retry_result = await repository.fail_or_retry(
            conn,
            retry["id"],
            "worker-a",
            error_code="temporary",
            error_message="try again",
            retryable=True,
        )
        after_retry = await conn.fetchval("SELECT clock_timestamp()")
        failed_result = await repository.fail_or_retry(
            conn,
            nonretry["id"],
            "worker-a",
            error_code="invalid",
            error_message="cannot retry",
            retryable=False,
        )
        exhausted_result = await repository.fail_or_retry(
            conn,
            exhausted["id"],
            "worker-a",
            error_code="temporary",
            error_message="still failing",
            retryable=True,
        )
        with pytest.raises(LeaseLost):
            await repository.fail_or_retry(
                conn,
                retry["id"],
                "worker-a",
                error_code="temporary",
                error_message="again",
                retryable=True,
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
                    **arguments,
                )

    assert cancelled_result.state is JobState.CANCELLED
    assert cancelled_result.cancel_requested_at == database_now
    assert cancelled_result.error_code is None
    assert cancelled_result.error_message is None
    assert retry_result.state is JobState.RETRY_WAIT
    assert before_retry <= retry_result.run_after - timedelta(seconds=4) <= after_retry
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
async def test_reaper_uses_database_time_for_all_branches_limit_and_repeat(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    expired_at = database_now - timedelta(minutes=1)

    async def expired(**changes):
        values = {
            "attempt_count": 1,
            "max_attempts": 3,
            "lease_owner": "dead-worker",
            "lease_expires_at": expired_at,
            "heartbeat_at": expired_at,
        }
        values.update(changes)
        return await _insert_job(pool, user_id, state="running", **values)

    retry = await expired()
    cancelled = await expired(cancel_requested_at=database_now)
    exhausted = await expired(attempt_count=3, max_attempts=3)
    live = await expired(lease_expires_at=database_now + timedelta(hours=1))
    before = await _db_now(pool)
    async with pool.acquire() as conn:
        first = await repository.reap_expired(conn, limit=2)
        second = await repository.reap_expired(conn, limit=2)
        repeated = await repository.reap_expired(conn, limit=2)
        for invalid_limit in (0, True, 1.5):
            with pytest.raises(ValueError, match="limit"):
                await repository.reap_expired(conn, limit=invalid_limit)
    after = await _db_now(pool)

    assert len(first) == 2
    assert len(second) == 1
    assert repeated == []
    assert set(first + second) == {retry["id"], cancelled["id"], exhausted["id"]}
    rows = {
        row["id"]: row
        for row in await pool.fetch(
            "SELECT id, state, run_after, lease_owner, lease_expires_at, heartbeat_at, "
            "error_code, error_message FROM background_jobs WHERE id = ANY($1::uuid[])",
            [retry["id"], cancelled["id"], exhausted["id"], live["id"]],
        )
    }
    assert rows[retry["id"]]["state"] == "retry_wait"
    assert before <= rows[retry["id"]]["run_after"] <= after
    assert rows[retry["id"]]["error_code"] == "lease_expired"
    assert rows[cancelled["id"]]["state"] == "cancelled"
    assert rows[cancelled["id"]]["error_code"] is None
    assert rows[exhausted["id"]]["state"] == "failed"
    assert rows[exhausted["id"]]["error_code"] == "attempts_exhausted"
    assert rows[exhausted["id"]]["error_message"] == "The job could not be completed after retrying."
    assert rows[live["id"]]["state"] == "running"
    for job_id in (retry["id"], cancelled["id"], exhausted["id"]):
        assert rows[job_id]["lease_owner"] is None
        assert rows[job_id]["lease_expires_at"] is None
        assert rows[job_id]["heartbeat_at"] is None


@pytest.mark.asyncio
async def test_reaper_locks_candidates_before_entering_clock_function(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    job = await _insert_job(
        pool,
        user_id,
        state="running",
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker-a",
        lease_expires_at=database_now - timedelta(minutes=1),
    )
    function_name = f"test_reaper_clock_{uuid4().hex}"
    if re.fullmatch(r"[a-z_][a-z0-9_]*", function_name) is None:
        raise AssertionError("generated PostgreSQL identifier was not safe")
    advisory_key = uuid4().int % 2_000_000_000 + 1
    await pool.execute(
        f"""
        CREATE FUNCTION public.{function_name}() RETURNS timestamptz
        LANGUAGE plpgsql VOLATILE AS $function$
        BEGIN
            PERFORM pg_advisory_xact_lock({advisory_key});
            RETURN clock_timestamp();
        END
        $function$
        """
    )
    conn_gate = await pool.acquire()
    conn_reaper = await pool.acquire()
    conn_probe = await pool.acquire()
    observer = await pool.acquire()
    probe_transaction = conn_probe.transaction()
    pending_reaper = None
    gate_locked = False
    probe_started = False
    try:
        await conn_gate.fetchval("SELECT pg_advisory_lock($1)", advisory_key)
        gate_locked = True
        reaper_pid = await conn_reaper.fetchval("SELECT pg_backend_pid()")
        assert repository._REAP_EXPIRED.count("clock_timestamp()") == 1
        probe_query = repository._REAP_EXPIRED.replace(
            "clock_timestamp()",
            f"public.{function_name}()",
        )
        pending_reaper = asyncio.create_task(
            conn_reaper.fetch(
                probe_query,
                1,
                repository._ATTEMPTS_EXHAUSTED_MESSAGE,
            )
        )
        await _wait_for_lock_wait(observer, reaper_pid)
        await probe_transaction.start()
        probe_started = True
        with pytest.raises(asyncpg.LockNotAvailableError):
            await conn_probe.fetchval(
                "SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE NOWAIT",
                job["id"],
            )
        await probe_transaction.rollback()
        probe_started = False
        assert await conn_gate.fetchval("SELECT pg_advisory_unlock($1)", advisory_key)
        gate_locked = False
        async with asyncio.timeout(1):
            rows = await pending_reaper
    finally:
        if probe_started:
            await probe_transaction.rollback()
        if gate_locked:
            await conn_gate.fetchval("SELECT pg_advisory_unlock($1)", advisory_key)
        if pending_reaper is not None and not pending_reaper.done():
            pending_reaper.cancel()
        if pending_reaper is not None:
            await asyncio.gather(pending_reaper, return_exceptions=True)
        await pool.release(observer)
        await pool.release(conn_probe)
        await pool.release(conn_reaper)
        await pool.release(conn_gate)
        await pool.execute(f"DROP FUNCTION IF EXISTS public.{function_name}()")

    assert [row["id"] for row in rows] == [job["id"]]


@pytest.mark.asyncio
async def test_concurrent_reapers_return_disjoint_ids_with_skip_locked(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    expired_at = (await _db_now(pool)) - timedelta(minutes=1)
    jobs = [
        await _insert_job(
            pool,
            user_id,
            state="running",
            attempt_count=1,
            max_attempts=3,
            lease_owner="dead-worker",
            lease_expires_at=expired_at,
        )
        for _ in range(2)
    ]
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    transaction_a = conn_a.transaction()
    committed = False
    try:
        await transaction_a.start()
        async with asyncio.timeout(1):
            first = await repository.reap_expired(conn_a, limit=1)
            second = await repository.reap_expired(conn_b, limit=1)
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


@pytest.mark.asyncio
async def test_heartbeat_transaction_fences_concurrent_reaper(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    job = await _insert_job(
        pool,
        user_id,
        state="running",
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker-a",
        lease_expires_at=database_now + timedelta(minutes=1),
    )
    conn_heartbeat = await pool.acquire()
    conn_reaper = await pool.acquire()
    conn_probe = await pool.acquire()
    observer = await pool.acquire()
    transaction = conn_heartbeat.transaction()
    probe = None
    committed = False
    try:
        await transaction.start()
        renewed = await repository.heartbeat(conn_heartbeat, job["id"], "worker-a", 300)
        probe_pid = await conn_probe.fetchval("SELECT pg_backend_pid()")
        probe = asyncio.create_task(
            conn_probe.fetchval("SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE", job["id"])
        )
        await _wait_for_lock_wait(observer, probe_pid)
        async with asyncio.timeout(1):
            assert await repository.reap_expired(conn_reaper) == []
        probe.cancel()
        with suppress(asyncio.CancelledError):
            await probe
        await transaction.commit()
        committed = True
        assert await repository.reap_expired(conn_reaper) == []
    finally:
        if probe is not None and not probe.done():
            probe.cancel()
            with suppress(asyncio.CancelledError):
                await probe
        if not committed:
            await transaction.rollback()
        await pool.release(observer)
        await pool.release(conn_probe)
        await pool.release(conn_reaper)
        await pool.release(conn_heartbeat)

    assert renewed.lease_expires_at - renewed.heartbeat_at == timedelta(seconds=300)
    assert await pool.fetchval("SELECT state FROM background_jobs WHERE id = $1", job["id"]) == "running"


@pytest.mark.asyncio
async def test_heartbeat_samples_database_time_after_waiting_for_row_lock(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    job = await _insert_job(
        pool,
        user_id,
        state="running",
        lease_owner="worker-a",
        lease_expires_at=database_now + timedelta(milliseconds=200),
    )
    conn_lock = await pool.acquire()
    conn_heartbeat = await pool.acquire()
    observer = await pool.acquire()
    transaction = conn_lock.transaction()
    pending_heartbeat = None
    committed = False
    try:
        await transaction.start()
        await conn_lock.fetchval("SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE", job["id"])
        heartbeat_pid = await conn_heartbeat.fetchval("SELECT pg_backend_pid()")
        pending_heartbeat = asyncio.create_task(repository.heartbeat(conn_heartbeat, job["id"], "worker-a", 30))
        await _wait_for_lock_wait(observer, heartbeat_pid)
        async with asyncio.timeout(1):
            while not await observer.fetchval(
                "SELECT clock_timestamp() >= lease_expires_at FROM background_jobs WHERE id = $1",
                job["id"],
            ):
                await asyncio.sleep(0)
        await transaction.commit()
        committed = True
        with pytest.raises(LeaseLost):
            async with asyncio.timeout(1):
                await pending_heartbeat
    finally:
        if pending_heartbeat is not None and not pending_heartbeat.done():
            pending_heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await pending_heartbeat
        if not committed:
            await transaction.rollback()
        await pool.release(observer)
        await pool.release(conn_heartbeat)
        await pool.release(conn_lock)


@pytest.mark.asyncio
async def test_final_transaction_checkpoint_fences_cancel_and_recovery(pool):
    await pool.execute("DELETE FROM background_jobs")
    user_id = await _seed_user(pool)
    database_now = await _db_now(pool)
    job = await _insert_job(
        pool,
        user_id,
        state="running",
        attempt_count=1,
        max_attempts=3,
        lease_owner="worker-a",
        lease_expires_at=database_now + timedelta(minutes=1),
    )
    conn_worker = await pool.acquire()
    conn_cancel = await pool.acquire()
    conn_reaper = await pool.acquire()
    observer = await pool.acquire()
    transaction = conn_worker.transaction()
    pending_cancel = None
    committed = False
    try:
        await transaction.start()
        await repository.assert_active(conn_worker, job["id"], "worker-a")
        cancel_pid = await conn_cancel.fetchval("SELECT pg_backend_pid()")
        pending_cancel = asyncio.create_task(repository.request_cancel(conn_cancel, job["id"], user_id))
        await _wait_for_lock_wait(observer, cancel_pid)
        await conn_worker.execute(
            "UPDATE background_jobs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = $1",
            job["id"],
        )
        await conn_worker.execute("UPDATE users SET display_name = 'fenced-write' WHERE id = $1", user_id)
        async with asyncio.timeout(1):
            assert await repository.reap_expired(conn_reaper) == []
        assert await observer.fetchval("SELECT display_name FROM users WHERE id = $1", user_id) is None
        await transaction.commit()
        committed = True
        async with asyncio.timeout(1):
            cancellation = await pending_cancel
        recovered = await repository.reap_expired(conn_reaper)
    finally:
        if pending_cancel is not None and not pending_cancel.done():
            pending_cancel.cancel()
            with suppress(asyncio.CancelledError):
                await pending_cancel
        if not committed:
            await transaction.rollback()
        await pool.release(observer)
        await pool.release(conn_reaper)
        await pool.release(conn_cancel)
        await pool.release(conn_worker)

    assert cancellation.cancel_requested_at is not None
    assert recovered == [job["id"]]
    row = await pool.fetchrow("SELECT state FROM background_jobs WHERE id = $1", job["id"])
    assert row["state"] == "cancelled"
    assert await pool.fetchval("SELECT display_name FROM users WHERE id = $1", user_id) == "fenced-write"
