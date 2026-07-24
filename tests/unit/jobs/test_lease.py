from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from math import inf, nan
from uuid import uuid4

import pytest
from jobs.lease import JobLease
from jobs.models import JobCancelled, LeaseLost


class FakeConnection:
    def __init__(self, calls):
        self.calls = calls

    @asynccontextmanager
    async def transaction(self):
        self.calls.append("transaction-enter")
        try:
            yield
        finally:
            self.calls.append("transaction-exit")


class FakePool:
    def __init__(self):
        self.calls = []
        self.connection = FakeConnection(self.calls)

    @asynccontextmanager
    async def acquire(self):
        self.calls.append("acquire-enter")
        try:
            yield self.connection
        finally:
            self.calls.append("acquire-exit")


class ControlledWaiter:
    def __init__(self):
        self.pulses = asyncio.Queue()
        self.waiting = asyncio.Event()

    async def __call__(self, stop, _seconds):
        self.waiting.set()
        stop_task = asyncio.create_task(stop.wait())
        pulse_task = asyncio.create_task(self.pulses.get())
        done, pending = await asyncio.wait({stop_task, pulse_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return stop_task in done

    async def pulse(self):
        self.waiting.clear()
        await self.pulses.put(None)


def make_lease(**changes):
    values = {
        "pool": FakePool(),
        "job_id": uuid4(),
        "owner": "worker-a",
        "lease_seconds": 30,
        "heartbeat_seconds": 10,
    }
    values.update(changes)
    return JobLease(**values)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"owner": ""}, "owner"),
        ({"owner": "   "}, "owner"),
        ({"heartbeat_seconds": 30}, "less than lease_seconds"),
        ({"heartbeat_seconds": 31}, "less than lease_seconds"),
    ],
)
def test_job_lease_validates_configuration(changes, message):
    with pytest.raises(ValueError, match=message):
        make_lease(**changes)


@pytest.mark.parametrize("value", [True, nan, inf, -inf, 0, -1, 1.5, "30", None])
def test_job_lease_rejects_non_positive_non_finite_or_non_integer_lease_seconds(value):
    with pytest.raises(ValueError, match="lease_seconds"):
        make_lease(lease_seconds=value, heartbeat_seconds=1)


@pytest.mark.parametrize("value", [True, nan, inf, -inf, 0, -1, "10", None])
def test_job_lease_rejects_non_real_non_finite_or_non_positive_heartbeat_seconds(value):
    with pytest.raises(ValueError, match="heartbeat_seconds"):
        make_lease(heartbeat_seconds=value)


@pytest.mark.asyncio
async def test_context_starts_one_heartbeat_and_leaves_no_pending_task():
    waiter = ControlledWaiter()
    heartbeats = []

    async def heartbeat(conn, job_id, owner, lease_seconds):
        heartbeats.append((conn, job_id, owner, lease_seconds))

    lease = make_lease(waiter=waiter, heartbeat_fn=heartbeat)
    async with lease:
        task = lease.heartbeat_task
        assert task is not None
        assert not task.done()
        with pytest.raises(RuntimeError, match="already active"):
            await lease.__aenter__()

    assert task.done()
    assert heartbeats == []
    assert lease.heartbeat_task is None


@pytest.mark.asyncio
async def test_heartbeat_repeats_using_short_acquired_transactions():
    waiter = ControlledWaiter()
    calls = []

    async def heartbeat(conn, job_id, owner, lease_seconds):
        calls.append((conn, job_id, owner, lease_seconds))

    pool = FakePool()
    lease = make_lease(pool=pool, waiter=waiter, heartbeat_fn=heartbeat)
    async with lease:
        await waiter.waiting.wait()
        await waiter.pulse()
        while len(calls) < 1:
            await asyncio.sleep(0)
        await waiter.waiting.wait()
        await waiter.pulse()
        while len(calls) < 2:
            await asyncio.sleep(0)

    assert [call[1:] for call in calls] == [
        (lease.job_id, "worker-a", 30),
        (lease.job_id, "worker-a", 30),
    ]
    assert pool.calls == [
        "acquire-enter",
        "transaction-enter",
        "transaction-exit",
        "acquire-exit",
        "acquire-enter",
        "transaction-enter",
        "transaction-exit",
        "acquire-exit",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [JobCancelled("cancelled"), LeaseLost("lost")])
async def test_checkpoint_delegates_and_propagates_domain_errors(error):
    calls = []

    async def assert_active(conn, job_id, owner):
        calls.append((conn, job_id, owner))
        raise error

    lease = make_lease(assert_active_fn=assert_active)
    with pytest.raises(type(error), match=str(error)):
        await lease.checkpoint()

    assert len(calls) == 1
    assert calls[0][1:] == (lease.job_id, "worker-a")
    assert lease.pool.calls == [
        "acquire-enter",
        "transaction-enter",
        "transaction-exit",
        "acquire-exit",
    ]


@pytest.mark.asyncio
async def test_checkpoint_returns_active_record():
    expected = object()

    async def assert_active(_conn, _job_id, _owner):
        return expected

    assert await make_lease(assert_active_fn=assert_active).checkpoint() is expected


@pytest.mark.asyncio
async def test_checkpoint_raises_stored_heartbeat_failure_before_database_access():
    waiter = ControlledWaiter()
    failure = LeaseLost("heartbeat lease lost")

    async def heartbeat(_conn, _job_id, _owner, _lease_seconds):
        raise failure

    lease = make_lease(waiter=waiter, heartbeat_fn=heartbeat)
    async with lease:
        await waiter.waiting.wait()
        await waiter.pulse()
        while lease.heartbeat_failure is None:
            await asyncio.sleep(0)
        lease.pool.calls.clear()
        with pytest.raises(LeaseLost, match="heartbeat lease lost") as raised:
            await lease.checkpoint()

    assert raised.value is failure
    assert lease.pool.calls == []


@pytest.mark.asyncio
async def test_cleanup_does_not_mask_caller_exception():
    lease = make_lease()

    with pytest.raises(ValueError, match="caller failed"):
        async with lease:
            raise ValueError("caller failed")

    assert lease.heartbeat_task is None


@pytest.mark.asyncio
async def test_cancelled_heartbeat_task_does_not_mask_caller_exception():
    lease = make_lease()

    with pytest.raises(ValueError, match="caller failed"):
        async with lease:
            task = lease.heartbeat_task
            assert task is not None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise ValueError("caller failed")

    assert lease.heartbeat_task is None


@pytest.mark.asyncio
async def test_caller_cancellation_stops_and_awaits_heartbeat_task():
    lease = make_lease()
    entered = asyncio.Event()

    async def run_with_lease():
        async with lease:
            entered.set()
            await asyncio.Event().wait()

    caller = asyncio.create_task(run_with_lease())
    await entered.wait()
    heartbeat_task = lease.heartbeat_task
    assert heartbeat_task is not None
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert heartbeat_task.done()
    assert lease.heartbeat_task is None


@pytest.mark.asyncio
async def test_job_lease_never_calls_terminal_transitions():
    called = []

    async def assert_active(_conn, _job_id, _owner):
        called.append("assert_active")
        return object()

    lease = make_lease(assert_active_fn=assert_active)
    async with lease:
        await lease.checkpoint()

    assert called == ["assert_active"]
