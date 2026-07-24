"""Asyncpg transport for durable background-job records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from jobs.models import JobCreate, JobRecord, JobState, JobType, to_json_value


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
