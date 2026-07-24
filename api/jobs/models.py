"""Pure state contracts shared by job repositories, workers, and HTTP routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from random import uniform
from types import MappingProxyType
from typing import TypeAlias
from uuid import UUID

PAYLOAD_MAX_BYTES = 16_384
PROGRESS_MAX_BYTES = 8_192
RESULT_MAX_BYTES = 16_384
ERROR_CODE_MAX_CHARS = 2_000
ERROR_MESSAGE_MAX_CHARS = 2_000

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FrozenJSONValue: TypeAlias = JSONScalar | tuple["FrozenJSONValue", ...] | Mapping[str, "FrozenJSONValue"]
FrozenJSONMapping: TypeAlias = Mapping[str, FrozenJSONValue]
Jitter: TypeAlias = float | Callable[[float], float]


class JobType(StrEnum):
    DOCUMENT_EXTRACT = "document.extract"
    GRAPH_REBUILD = "graph.rebuild"
    UPLOAD_CLEANUP = "upload.cleanup"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class LeaseLost(RuntimeError):
    """Raised when a worker no longer owns a job lease."""


class JobCancelled(RuntimeError):
    """Raised when a job observes a durable cancellation request."""


def _freeze_json(value: object) -> FrozenJSONValue:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("JSON values cannot contain bytes")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON floats must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, Sequence):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _freeze_mapping(value: Mapping[object, object]) -> FrozenJSONMapping:
    frozen: dict[str, FrozenJSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings")
        frozen[key] = _freeze_json(item)
    return MappingProxyType(frozen)


def to_json_value(value: object) -> JSONValue:
    """Validate and deep-thaw a frozen job JSON value for persistence or HTTP."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("JSON values cannot contain bytes")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON floats must be finite")
        return value
    if isinstance(value, Mapping):
        thawed: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            thawed[key] = to_json_value(item)
        return thawed
    if isinstance(value, Sequence):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class JobCreate:
    """The durable fields required to enqueue a new background job."""

    job_type: JobType
    user_id: UUID
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None
    payload: FrozenJSONMapping = field(default_factory=dict)
    idempotency_key: str | None = None
    max_attempts: int = 3
    run_after: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class JobRecord:
    """A complete durable job row, independent of its database transport."""

    id: UUID
    job_type: JobType
    user_id: UUID
    state: JobState = JobState.QUEUED
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None
    payload: FrozenJSONMapping = field(default_factory=dict)
    progress: FrozenJSONMapping | None = None
    result: FrozenJSONMapping | None = None
    idempotency_key: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    run_after: datetime = field(default_factory=lambda: datetime.now(UTC))
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    last_dispatched_at: datetime | None = None
    dispatch_attempts: int = 0
    error_code: str | None = None
    error_message: str | None = None
    cancel_requested_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        if self.progress is not None:
            object.__setattr__(self, "progress", _freeze_mapping(self.progress))
        if self.result is not None:
            object.__setattr__(self, "result", _freeze_mapping(self.result))


def retry_delay_seconds(
    attempt: int,
    base: float = 2,
    cap: float = 60,
    jitter: Jitter = 0,
) -> float:
    """Return a bounded exponential retry delay plus optional bounded jitter.

    A numeric ``jitter`` is the maximum random additional seconds.  A callable
    receives that maximum and makes deterministic tests possible.  Its result
    is clamped to the same range, so jitter cannot make the delay negative or
    unbounded.
    """
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    if base <= 0:
        raise ValueError("base must be positive")
    if cap < 0:
        raise ValueError("cap must not be negative")

    delay = base
    for _ in range(attempt - 1):
        delay *= base
        if delay >= cap:
            delay = cap
            break
    delay = min(delay, cap)

    if callable(jitter):
        adjustment = jitter(delay)
    else:
        if jitter < 0:
            raise ValueError("jitter must not be negative")
        adjustment = uniform(0, min(jitter, delay))

    return delay + min(max(float(adjustment), 0.0), delay)


_PUBLIC_ERROR_MESSAGES = {
    "unsupported_job_type": "This job type is not supported.",
    "document_not_found": "The requested document was not found.",
    "knowledge_base_not_found": "The requested knowledge base was not found.",
    "unsupported_document_type": "This document type is not supported.",
    "quota_exceeded": "The account quota was exceeded.",
    "attempts_exhausted": "The job could not be completed after retrying.",
    "cancelled": "The job was cancelled.",
}


def _public_error(record: JobRecord) -> dict[str, str] | None:
    if record.error_code is None:
        return None
    if message := _PUBLIC_ERROR_MESSAGES.get(record.error_code):
        return {"code": record.error_code, "message": message}
    return {"code": "internal_error", "message": "The job could not be completed."}


def serialize_public_job(record: JobRecord) -> dict[str, JSONValue | None]:
    """Serialize the intentionally small public projection of a durable job."""
    return {
        "id": str(record.id),
        "type": record.job_type.value,
        "state": record.state.value,
        "progress": to_json_value(record.progress),
        "result": to_json_value(record.result),
        "attempt_count": record.attempt_count,
        "max_attempts": record.max_attempts,
        "cancel_requested": record.cancel_requested_at is not None,
        "error": _public_error(record),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
