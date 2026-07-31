"""Cooperative heartbeat and cancellation checkpoints for durable job workers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from math import isfinite
from numbers import Real
from types import TracebackType
from uuid import UUID

import asyncpg

from jobs import repository
from jobs.models import JobRecord

Heartbeat = Callable[[asyncpg.Connection, UUID, str, int], Awaitable[JobRecord]]
AssertActive = Callable[[asyncpg.Connection, UUID, str], Awaitable[JobRecord]]
Waiter = Callable[[asyncio.Event, float], Awaitable[bool]]


async def _wait_until_stopped(stop: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


class JobLease:
    """Maintain one worker lease and expose explicit cooperative checkpoints."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        job_id: UUID,
        owner: str,
        lease_seconds: int,
        heartbeat_seconds: float,
        *,
        heartbeat_fn: Heartbeat = repository.heartbeat,
        assert_active_fn: AssertActive = repository.assert_active,
        waiter: Waiter = _wait_until_stopped,
    ) -> None:
        if not owner.strip():
            raise ValueError("owner must not be empty")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        if (
            isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, Real)
            or not isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat_seconds must be finite and positive")
        if heartbeat_seconds >= lease_seconds:
            raise ValueError("heartbeat_seconds must be less than lease_seconds")
        self.pool = pool
        self.job_id = job_id
        self.owner = owner
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._heartbeat_fn = heartbeat_fn
        self._assert_active_fn = assert_active_fn
        self._waiter = waiter
        self._stop = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_failure: Exception | None = None

    @property
    def heartbeat_task(self) -> asyncio.Task[None] | None:
        return self._heartbeat_task

    @property
    def heartbeat_failure(self) -> Exception | None:
        return self._heartbeat_failure

    async def __aenter__(self) -> JobLease:
        if self._heartbeat_task is not None:
            raise RuntimeError("job lease is already active")
        self._stop.clear()
        self._heartbeat_failure = None
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        self._stop.set()
        task = self._heartbeat_task
        if task is None:
            return False
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled() and exc_value is not None:
                return False
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise
        finally:
            self._heartbeat_task = None
        return False

    async def checkpoint(self, conn: asyncpg.Connection | None = None) -> JobRecord:
        """Assert the lease is active, optionally inside a caller-owned transaction.

        A supplied connection must already be inside the caller's explicit final
        business transaction so the row lock remains held through its writes.
        """
        if self._heartbeat_failure is not None:
            raise self._heartbeat_failure
        if conn is not None:
            if not conn.is_in_transaction():
                raise RuntimeError("checkpoint connection must already be inside an explicit transaction")
            return await self._assert_active_fn(conn, self.job_id, self.owner)
        async with self.pool.acquire() as conn, conn.transaction():
            return await self._assert_active_fn(conn, self.job_id, self.owner)

    async def _heartbeat_loop(self) -> None:
        while not await self._waiter(self._stop, self.heartbeat_seconds):
            try:
                async with self.pool.acquire() as conn, conn.transaction():
                    await self._heartbeat_fn(
                        conn,
                        self.job_id,
                        self.owner,
                        self.lease_seconds,
                    )
            except Exception as exc:  # noqa: BLE001 - checkpoint must surface any heartbeat failure.
                self._heartbeat_failure = exc
                self._stop.set()
                return
