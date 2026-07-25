"""ARQ delivery adapter for the durable PostgreSQL job ledger."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import asyncpg
from arq.connections import ArqRedis
from telemetry import emit

logger = logging.getLogger(__name__)


def _validate_positive_integer(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class DispatchCandidate:
    """One database-selected delivery generation."""

    job_id: UUID
    run_after: datetime
    job_type: str | None = None
    state: str | None = None
    dispatch_attempts: int = 0


def delivery_transport_id(candidate: DispatchCandidate) -> str:
    """Return the stable ARQ dedupe key for one persisted delivery generation."""
    if candidate.run_after.tzinfo is None or candidate.run_after.utcoffset() is None:
        raise ValueError("run_after must be timezone-aware")
    generation = candidate.run_after.astimezone(UTC).isoformat(timespec="microseconds")
    return f"{candidate.job_id}:{generation}"


_SELECT_DUE_JOB_IDS = """
WITH dispatch_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at
)
SELECT job.id, job.run_after, job.job_type, job.state, job.dispatch_attempts
FROM background_jobs AS job
CROSS JOIN dispatch_clock
WHERE job.state IN ('queued', 'retry_wait')
  AND job.run_after <= dispatch_clock.checked_at
  AND job.cancel_requested_at IS NULL
  AND job.attempt_count < job.max_attempts
  AND (
      job.last_dispatched_at IS NULL
      OR job.last_dispatched_at < job.run_after
      OR job.last_dispatched_at
         < dispatch_clock.checked_at - make_interval(secs => $1::double precision)
  )
ORDER BY job.run_after, job.created_at, job.id
FOR UPDATE OF job SKIP LOCKED
LIMIT $2
"""


async def select_due_jobs(
    conn: asyncpg.Connection,
    *,
    batch_size: int,
    redeliver_seconds: int,
) -> list[DispatchCandidate]:
    """Lock due job IDs with the run_after token for their selected generation."""
    _validate_positive_integer(batch_size, "batch_size")
    _validate_positive_integer(redeliver_seconds, "redeliver_seconds")
    rows = await conn.fetch(_SELECT_DUE_JOB_IDS, redeliver_seconds, batch_size)
    return [
        DispatchCandidate(
            job_id=row["id"] if isinstance(row["id"], UUID) else UUID(str(row["id"])),
            run_after=row["run_after"],
            job_type=row.get("job_type"),
            state=row.get("state"),
            dispatch_attempts=row.get("dispatch_attempts", 0),
        )
        for row in rows
    ]


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
  AND job.run_after = $2
  AND job.state IN ('queued', 'retry_wait', 'running')
RETURNING true
"""


async def mark_dispatched(
    conn: asyncpg.Connection,
    job_id: UUID,
    selected_run_after: datetime,
) -> bool:
    """Record delivery only while the selected generation remains current."""
    return bool(await conn.fetchval(_MARK_DISPATCHED, job_id, selected_run_after))


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
        selected_jobs = await select_due_jobs(
            conn,
            batch_size=batch_size,
            redeliver_seconds=redeliver_seconds,
        )

    enqueued = 0
    already_present = 0
    marked = 0
    enqueue_failed = 0
    mark_failed = 0

    for candidate in selected_jobs:
        job_id = candidate.job_id
        try:
            arq_job = await arq_redis.enqueue_job(
                "run_job",
                str(job_id),
                _job_id=delivery_transport_id(candidate),
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
                was_marked = await mark_dispatched(conn, job_id, candidate.run_after)
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
            dispatch_lag_ms = max(
                0,
                int((datetime.now(UTC) - candidate.run_after.astimezone(UTC)).total_seconds() * 1000),
            )
            emit(
                logger,
                "durable_job_dispatched",
                job_id=job_id,
                job_type=candidate.job_type,
                state=candidate.state,
                dispatch_attempts=candidate.dispatch_attempts + 1,
                dispatch_lag_ms=dispatch_lag_ms,
                replica_role="worker",
            )
        else:
            mark_failed += 1
            logger.info("job dispatch mark skipped job_id=%s", job_id)

    return DispatchSummary(
        selected=len(selected_jobs),
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
