"""ARQ delivery adapter for the durable PostgreSQL job ledger."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import asyncpg
from arq.connections import ArqRedis

logger = logging.getLogger(__name__)


def _validate_positive_integer(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


_SELECT_DUE_JOB_IDS = """
WITH dispatch_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at
)
SELECT job.id
FROM background_jobs AS job
CROSS JOIN dispatch_clock
WHERE job.state IN ('queued', 'retry_wait')
  AND job.run_after <= dispatch_clock.checked_at
  AND job.cancel_requested_at IS NULL
  AND job.attempt_count < job.max_attempts
  AND (
      job.last_dispatched_at IS NULL
      OR job.last_dispatched_at
         < dispatch_clock.checked_at - make_interval(secs => $1::double precision)
  )
ORDER BY job.run_after, job.created_at, job.id
FOR UPDATE OF job SKIP LOCKED
LIMIT $2
"""


async def select_due_job_ids(
    conn: asyncpg.Connection,
    *,
    batch_size: int,
    redeliver_seconds: int,
) -> list[UUID]:
    """Lock and return a deterministic bounded batch of database-due job IDs."""
    _validate_positive_integer(batch_size, "batch_size")
    _validate_positive_integer(redeliver_seconds, "redeliver_seconds")
    rows = await conn.fetch(_SELECT_DUE_JOB_IDS, redeliver_seconds, batch_size)
    return [row["id"] if isinstance(row["id"], UUID) else UUID(str(row["id"])) for row in rows]


_MARK_DISPATCHED = """
WITH dispatch_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at
)
UPDATE background_jobs AS job
SET
    last_dispatched_at = dispatch_clock.checked_at,
    dispatch_attempts = job.dispatch_attempts + 1
FROM dispatch_clock
WHERE job.id = $1
  AND job.state IN ('queued', 'retry_wait', 'running')
RETURNING true
"""


async def mark_dispatched(conn: asyncpg.Connection, job_id: UUID) -> bool:
    """Record one delivered transport message without changing durable job state."""
    return bool(await conn.fetchval(_MARK_DISPATCHED, job_id))


@dataclass(frozen=True, slots=True)
class DispatchSummary:
    selected: int = 0
    enqueued: int = 0
    already_present: int = 0
    marked: int = 0
    enqueue_failed: int = 0
    mark_failed: int = 0


async def dispatch_due_jobs(
    pool: asyncpg.Pool,
    arq_redis: ArqRedis,
    *,
    batch_size: int,
    redeliver_seconds: int,
) -> DispatchSummary:
    """Deliver due UUIDs after releasing their short PostgreSQL selection locks."""
    _validate_positive_integer(batch_size, "batch_size")
    _validate_positive_integer(redeliver_seconds, "redeliver_seconds")

    async with pool.acquire() as conn, conn.transaction():
        job_ids = await select_due_job_ids(
            conn,
            batch_size=batch_size,
            redeliver_seconds=redeliver_seconds,
        )

    enqueued = 0
    already_present = 0
    marked = 0
    enqueue_failed = 0
    mark_failed = 0

    for job_id in job_ids:
        try:
            arq_job = await arq_redis.enqueue_job(
                "run_job",
                str(job_id),
                _job_id=str(job_id),
            )
        except Exception as exc:  # noqa: BLE001 - one transport failure must not abort the batch.
            enqueue_failed += 1
            logger.warning(
                "job enqueue failed job_id=%s error_type=%s",
                job_id,
                type(exc).__name__,
            )
            continue

        if arq_job is None:
            already_present += 1
        else:
            enqueued += 1

        try:
            async with pool.acquire() as conn, conn.transaction():
                was_marked = await mark_dispatched(conn, job_id)
        except Exception as exc:  # noqa: BLE001 - durable redelivery recovers this mismatch.
            mark_failed += 1
            logger.warning(
                "job dispatch mark failed job_id=%s error_type=%s",
                job_id,
                type(exc).__name__,
            )
            continue

        if was_marked:
            marked += 1
        else:
            mark_failed += 1
            logger.info("job dispatch mark skipped job_id=%s", job_id)

    return DispatchSummary(
        selected=len(job_ids),
        enqueued=enqueued,
        already_present=already_present,
        marked=marked,
        enqueue_failed=enqueue_failed,
        mark_failed=mark_failed,
    )


async def dispatch_cron(ctx: dict) -> None:
    """ARQ cron entry point for one durable delivery scan."""
    await dispatch_due_jobs(
        ctx["pool"],
        ctx["redis"],
        batch_size=ctx["dispatch_batch_size"],
        redeliver_seconds=ctx["redeliver_seconds"],
    )
