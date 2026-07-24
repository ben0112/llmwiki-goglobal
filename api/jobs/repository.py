"""Asyncpg transport for durable background-job records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from jobs.models import (
    ERROR_CODE_MAX_CHARS,
    ERROR_MESSAGE_MAX_CHARS,
    JobCancelled,
    JobCreate,
    JobRecord,
    JobState,
    JobType,
    JSONValue,
    LeaseLost,
    retry_delay_seconds,
    to_json_value,
)

_ATTEMPTS_EXHAUSTED_MESSAGE = "The job could not be completed after retrying."


def _validate_owner(owner: str) -> None:
    if not owner.strip():
        raise ValueError("owner must not be empty")


def _validate_positive(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _validate_error(error_code: str, error_message: str) -> None:
    if len(error_code) > ERROR_CODE_MAX_CHARS:
        raise ValueError(f"error_code must not exceed {ERROR_CODE_MAX_CHARS} characters")
    if len(error_message) > ERROR_MESSAGE_MAX_CHARS:
        raise ValueError(f"error_message must not exceed {ERROR_MESSAGE_MAX_CHARS} characters")


def _uuid(value: object, field: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a UUID") from exc


def _datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    return value


def _optional_datetime(value: object, field: str) -> datetime | None:
    return None if value is None else _datetime(value, field)


def _json_object(value: object, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TypeError(f"{field} must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field} JSON object keys must be strings")
    return dict(value)


def _optional_json_object(value: object, field: str) -> dict[str, Any] | None:
    return None if value is None else _json_object(value, field)


def _row_to_record(row: Mapping[str, object]) -> JobRecord:
    """Convert an asyncpg row into the transport-independent domain record."""
    try:
        job_type = JobType(str(row["job_type"]))
        state = JobState(str(row["state"]))
    except ValueError as exc:
        raise TypeError("background job contains an unsupported enum value") from exc

    def optional_uuid(field: str) -> UUID | None:
        value = row[field]
        return None if value is None else _uuid(value, field)

    def optional_string(field: str) -> str | None:
        value = row[field]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{field} must be a string or None")
        return value

    def integer(field: str) -> int:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        return value

    return JobRecord(
        id=_uuid(row["id"], "id"),
        job_type=job_type,
        user_id=_uuid(row["user_id"], "user_id"),
        state=state,
        knowledge_base_id=optional_uuid("knowledge_base_id"),
        document_id=optional_uuid("document_id"),
        payload=_json_object(row["payload"], "payload"),
        progress=_optional_json_object(row["progress"], "progress"),
        result=_optional_json_object(row["result"], "result"),
        idempotency_key=optional_string("idempotency_key"),
        attempt_count=integer("attempt_count"),
        max_attempts=integer("max_attempts"),
        run_after=_datetime(row["run_after"], "run_after"),
        lease_owner=optional_string("lease_owner"),
        lease_expires_at=_optional_datetime(row["lease_expires_at"], "lease_expires_at"),
        heartbeat_at=_optional_datetime(row["heartbeat_at"], "heartbeat_at"),
        last_dispatched_at=_optional_datetime(row["last_dispatched_at"], "last_dispatched_at"),
        dispatch_attempts=integer("dispatch_attempts"),
        error_code=optional_string("error_code"),
        error_message=optional_string("error_message"),
        cancel_requested_at=_optional_datetime(row["cancel_requested_at"], "cancel_requested_at"),
        created_at=_datetime(row["created_at"], "created_at"),
        updated_at=_datetime(row["updated_at"], "updated_at"),
    )


_CREATE_WITH_DEFAULT_RUN_AFTER = """
INSERT INTO background_jobs (
    job_type, user_id, knowledge_base_id, document_id, payload, idempotency_key, max_attempts
) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
ON CONFLICT (user_id, job_type, idempotency_key)
WHERE idempotency_key IS NOT NULL
DO UPDATE SET idempotency_key = EXCLUDED.idempotency_key
RETURNING *
"""

_CREATE_WITH_RUN_AFTER = """
INSERT INTO background_jobs (
    job_type, user_id, knowledge_base_id, document_id, payload, idempotency_key, max_attempts, run_after
) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
ON CONFLICT (user_id, job_type, idempotency_key)
WHERE idempotency_key IS NOT NULL
DO UPDATE SET idempotency_key = EXCLUDED.idempotency_key
RETURNING *
"""


async def create(conn: asyncpg.Connection, command: JobCreate) -> JobRecord:
    """Insert a job, or return the existing record for its idempotency key."""
    values: list[object] = [
        command.job_type.value,
        command.user_id,
        command.knowledge_base_id,
        command.document_id,
        json.dumps(to_json_value(command.payload)),
        command.idempotency_key,
        command.max_attempts,
    ]
    query = _CREATE_WITH_DEFAULT_RUN_AFTER
    if command.run_after is not None:
        query = _CREATE_WITH_RUN_AFTER
        values.append(command.run_after)
    row = await conn.fetchrow(query, *values)
    if row is None:  # pragma: no cover - INSERT ... RETURNING always returns a row.
        raise RuntimeError("background job create returned no row")
    return _row_to_record(row)


async def get_for_user(conn: asyncpg.Connection, job_id: UUID, user_id: UUID) -> JobRecord | None:
    """Return one job only when its explicit tenant scope matches."""
    row = await conn.fetchrow(
        "SELECT * FROM background_jobs WHERE id = $1 AND user_id = $2",
        job_id,
        user_id,
    )
    return None if row is None else _row_to_record(row)


_REQUEST_CANCEL = """
WITH target AS (
    SELECT *
    FROM background_jobs
    WHERE id = $1 AND user_id = $2
    FOR UPDATE
), changed AS (
    UPDATE background_jobs AS job
    SET
        state = CASE
            WHEN target.state IN ('queued', 'retry_wait') THEN 'cancelled'
            ELSE target.state
        END,
        cancel_requested_at = COALESCE(job.cancel_requested_at, now())
    FROM target
    WHERE job.id = target.id
      AND (
          target.state IN ('queued', 'retry_wait')
          OR (target.state = 'running' AND target.cancel_requested_at IS NULL)
      )
    RETURNING job.*
)
SELECT * FROM changed
UNION ALL
SELECT * FROM target
WHERE state IN ('succeeded', 'failed', 'cancelled')
   OR (state = 'running' AND cancel_requested_at IS NOT NULL)
"""


async def request_cancel(conn: asyncpg.Connection, job_id: UUID, user_id: UUID) -> JobRecord | None:
    """Persist a cancellation request without touching terminal job rows."""
    row = await conn.fetchrow(_REQUEST_CANCEL, job_id, user_id)
    return None if row is None else _row_to_record(row)


_CLAIM = """
WITH target AS MATERIALIZED (
    SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at FROM target
)
UPDATE background_jobs AS job
SET
    state = 'running',
    lease_owner = $2,
    heartbeat_at = lease_clock.checked_at,
    lease_expires_at = lease_clock.checked_at + make_interval(secs => $3::double precision),
    attempt_count = job.attempt_count + 1
FROM target, lease_clock
WHERE job.id = target.id
  AND job.state IN ('queued', 'retry_wait')
  AND job.run_after <= lease_clock.checked_at
  AND job.cancel_requested_at IS NULL
  AND job.attempt_count < job.max_attempts
RETURNING job.*
"""


async def claim(
    conn: asyncpg.Connection,
    job_id: UUID,
    owner: str,
    lease_seconds: int,
) -> JobRecord | None:
    """Atomically acquire one due job for a worker."""
    _validate_owner(owner)
    _validate_positive(lease_seconds, "lease_seconds")
    row = await conn.fetchrow(_CLAIM, job_id, owner, lease_seconds)
    return None if row is None else _row_to_record(row)


_HEARTBEAT = """
WITH target AS MATERIALIZED (
    SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at FROM target
)
UPDATE background_jobs AS job
SET
    heartbeat_at = lease_clock.checked_at,
    lease_expires_at = lease_clock.checked_at + make_interval(secs => $3::double precision)
FROM target, lease_clock
WHERE job.id = target.id
  AND job.state = 'running'
  AND job.lease_owner = $2
  AND job.lease_expires_at > lease_clock.checked_at
RETURNING job.*
"""


async def heartbeat(
    conn: asyncpg.Connection,
    job_id: UUID,
    owner: str,
    lease_seconds: int,
) -> JobRecord:
    """Extend a live owned lease, or report that ownership was lost."""
    _validate_owner(owner)
    _validate_positive(lease_seconds, "lease_seconds")
    row = await conn.fetchrow(_HEARTBEAT, job_id, owner, lease_seconds)
    if row is None:
        raise LeaseLost("background job lease is no longer active")
    return _row_to_record(row)


_ASSERT_ACTIVE = """
WITH target AS MATERIALIZED (
    SELECT job.*
    FROM background_jobs AS job
    WHERE job.id = $1
      AND job.lease_owner = $2
    FOR UPDATE
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at FROM target
)
SELECT target.*, lease_clock.checked_at AS lease_checked_at
FROM target
CROSS JOIN lease_clock
"""


async def assert_active(conn: asyncpg.Connection, job_id: UUID, owner: str) -> JobRecord:
    """Check ownership, expiry, and cooperative cancellation at a worker checkpoint."""
    _validate_owner(owner)
    row = await conn.fetchrow(_ASSERT_ACTIVE, job_id, owner)
    if (
        row is None
        or row["state"] != JobState.RUNNING.value
        or row["lease_expires_at"] is None
        or row["lease_expires_at"] <= row["lease_checked_at"]
    ):
        raise LeaseLost("background job lease is no longer active")
    if row["cancel_requested_at"] is not None:
        raise JobCancelled("background job cancellation was requested")
    return _row_to_record(row)


_SUCCEED = """
WITH target AS MATERIALIZED (
    SELECT job.* FROM background_jobs AS job WHERE job.id = $1 FOR UPDATE
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at FROM target
), changed AS (
UPDATE background_jobs AS job
SET
    state = 'succeeded',
    result = $3::jsonb,
    progress = NULL,
    error_code = NULL,
    error_message = NULL,
    lease_owner = NULL,
    lease_expires_at = NULL,
    heartbeat_at = NULL
FROM target, lease_clock
WHERE job.id = target.id
  AND job.state = 'running'
  AND job.lease_owner = $2
  AND job.lease_expires_at > lease_clock.checked_at
  AND job.cancel_requested_at IS NULL
RETURNING job.*, 'succeeded'::text AS lease_outcome
)
SELECT changed.* FROM changed
UNION ALL
SELECT target.*, 'cancelled'::text AS lease_outcome
FROM target
CROSS JOIN lease_clock
WHERE NOT EXISTS (SELECT 1 FROM changed)
  AND target.state = 'running'
  AND target.lease_owner = $2
  AND target.lease_expires_at > lease_clock.checked_at
  AND target.cancel_requested_at IS NOT NULL
"""


async def succeed(
    conn: asyncpg.Connection,
    job_id: UUID,
    owner: str,
    result: Mapping[str, JSONValue],
) -> JobRecord:
    """Persist successful output only while the worker still owns a live lease."""
    _validate_owner(owner)
    serialized_result = json.dumps(to_json_value(result))
    row = await conn.fetchrow(_SUCCEED, job_id, owner, serialized_result)
    if row is None:
        raise LeaseLost("background job lease is no longer active")
    if row["lease_outcome"] == "cancelled":
        raise JobCancelled("background job cancellation was requested")
    return _row_to_record(row)


_FAIL_OR_RETRY = """
WITH target AS MATERIALIZED (
    SELECT id FROM background_jobs WHERE id = $1 FOR UPDATE
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at FROM target
)
UPDATE background_jobs AS job
SET
    state = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN 'cancelled'
        WHEN $3 AND job.attempt_count < job.max_attempts THEN 'retry_wait'
        ELSE 'failed'
    END,
    run_after = CASE
        WHEN job.cancel_requested_at IS NULL
         AND $3
         AND job.attempt_count < job.max_attempts
        THEN lease_clock.checked_at
             + make_interval(secs => ($6::double precision[])[job.attempt_count])
        ELSE job.run_after
    END,
    error_code = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN NULL
        WHEN $3 AND job.attempt_count >= job.max_attempts THEN 'attempts_exhausted'
        ELSE $4
    END,
    error_message = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN NULL
        WHEN $3 AND job.attempt_count >= job.max_attempts THEN $7
        ELSE $5
    END,
    result = NULL,
    progress = NULL,
    lease_owner = NULL,
    lease_expires_at = NULL,
    heartbeat_at = NULL
FROM target, lease_clock
WHERE job.id = target.id
  AND job.state = 'running'
  AND job.lease_owner = $2
  AND job.lease_expires_at > lease_clock.checked_at
RETURNING job.*
"""


async def fail_or_retry(
    conn: asyncpg.Connection,
    job_id: UUID,
    owner: str,
    *,
    error_code: str,
    error_message: str,
    retryable: bool,
) -> JobRecord:
    """Record cancellation, deterministic retry, or terminal worker failure."""
    _validate_owner(owner)
    _validate_error(error_code, error_message)
    retry_delays = [retry_delay_seconds(attempt, jitter=0) for attempt in range(1, 21)]
    row = await conn.fetchrow(
        _FAIL_OR_RETRY,
        job_id,
        owner,
        retryable,
        error_code,
        error_message,
        retry_delays,
        _ATTEMPTS_EXHAUSTED_MESSAGE,
    )
    if row is None:
        raise LeaseLost("background job lease is no longer active")
    return _row_to_record(row)


_REAP_EXPIRED = """
WITH candidates AS MATERIALIZED (
    SELECT job.id
    FROM background_jobs AS job
    WHERE job.state = 'running'
    ORDER BY job.lease_expires_at, job.id
    -- A live ordered candidate means no later unlocked row can be earlier-expired.
    FOR UPDATE OF job SKIP LOCKED
    LIMIT $1
), lease_clock AS MATERIALIZED (
    SELECT clock_timestamp() AS checked_at, count(*) AS candidate_count
    FROM candidates
)
UPDATE background_jobs AS job
SET
    state = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN 'cancelled'
        WHEN job.attempt_count < job.max_attempts THEN 'retry_wait'
        ELSE 'failed'
    END,
    run_after = CASE
        WHEN job.cancel_requested_at IS NULL AND job.attempt_count < job.max_attempts
        THEN lease_clock.checked_at
        ELSE job.run_after
    END,
    error_code = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN NULL
        WHEN job.attempt_count < job.max_attempts THEN 'lease_expired'
        ELSE 'attempts_exhausted'
    END,
    error_message = CASE
        WHEN job.cancel_requested_at IS NOT NULL THEN NULL
        WHEN job.attempt_count < job.max_attempts THEN 'Worker lease expired.'
        ELSE $2
    END,
    result = NULL,
    progress = NULL,
    lease_owner = NULL,
    lease_expires_at = NULL,
    heartbeat_at = NULL
FROM candidates, lease_clock
WHERE job.id = candidates.id
  AND job.lease_expires_at <= lease_clock.checked_at
RETURNING job.id
"""


async def reap_expired(
    conn: asyncpg.Connection,
    *,
    limit: int = 100,
) -> list[UUID]:
    """Recover a bounded batch of expired leases without colliding with other reapers."""
    _validate_positive(limit, "limit")
    rows = await conn.fetch(_REAP_EXPIRED, limit, _ATTEMPTS_EXHAUSTED_MESSAGE)
    return [_uuid(row["id"], "id") for row in rows]
