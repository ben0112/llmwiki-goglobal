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
    DOCUMENT_EMBED = "document.embed"
    GRAPH_REBUILD = "graph.rebuild"
    UPLOAD_CLEANUP = "upload.cleanup"
    BUILD_WIKI = "build_wiki"


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
    payload: FrozenJSONMapping = field(default_factory=dict, repr=False)
    idempotency_key: str | None = None
    max_attempts: int = 3
    run_after: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_type, JobType):
            raise ValueError("job_type must be a JobType")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        if self.job_type is JobType.DOCUMENT_EMBED:
            _validate_document_embed_command(self)
        elif self.job_type is JobType.BUILD_WIKI:
            _validate_build_wiki_command(self)


@dataclass(frozen=True, slots=True)
class JobRecord:
    """A complete durable job row, independent of its database transport."""

    id: UUID
    job_type: JobType
    user_id: UUID
    state: JobState = JobState.QUEUED
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None
    payload: FrozenJSONMapping = field(default_factory=dict, repr=False)
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


_DOCUMENT_EMBED_PAYLOAD_KEYS = frozenset({"document_id", "document_version", "provider", "model", "dimensions"})


def _validate_document_embed_command(command: JobCreate) -> None:
    if command.document_id is None or command.knowledge_base_id is None:
        raise ValueError("document embedding command requires document tenant scope")
    if set(command.payload) != _DOCUMENT_EMBED_PAYLOAD_KEYS:
        raise ValueError("document embedding payload must contain exactly the public profile fields")
    raw_document_id = command.payload.get("document_id")
    try:
        payload_document_id = UUID(raw_document_id) if isinstance(raw_document_id, str) else None
    except ValueError:
        payload_document_id = None
    if payload_document_id != command.document_id or str(payload_document_id) != raw_document_id:
        raise ValueError("document embedding payload contains an invalid document_id")
    version = command.payload.get("document_version")
    if type(version) is not int or not 1 <= version <= 2_147_483_647:
        raise ValueError("document embedding version must be a positive PostgreSQL integer")
    provider = command.payload.get("provider")
    model = command.payload.get("model")
    dimensions = command.payload.get("dimensions")
    if not isinstance(provider, str) or not provider or provider.strip() != provider or len(provider) > 100:
        raise ValueError("document embedding provider is invalid")
    if not isinstance(model, str) or not model or model.strip() != model or len(model) > 200:
        raise ValueError("document embedding model is invalid")
    if type(dimensions) is not int or not 1 <= dimensions <= 4096:
        raise ValueError("document embedding dimensions must be between 1 and 4096")


def _validate_build_wiki_command(command: JobCreate) -> None:
    if set(command.payload) != {"run_id"}:
        raise ValueError("build wiki payload must contain exactly run_id")
    raw_run_id = command.payload.get("run_id")
    try:
        run_id = UUID(raw_run_id) if isinstance(raw_run_id, str) else None
    except ValueError:
        run_id = None
    if run_id is None or str(run_id) != raw_run_id:
        raise ValueError("build wiki payload must contain a canonical UUID run_id")
    if command.knowledge_base_id is None:
        raise ValueError("build wiki command requires a knowledge base")
    if command.document_id is not None:
        raise ValueError("build wiki command must not reference a document")
    idempotency_key = command.idempotency_key
    if (
        type(idempotency_key) is not str
        or not 1 <= len(idempotency_key) <= 200
        or idempotency_key.strip() != idempotency_key
    ):
        raise ValueError("build wiki command requires a normalized idempotency key")
    if type(command.max_attempts) is not int or not 1 <= command.max_attempts <= 20:
        raise ValueError("build wiki max_attempts must be an integer between 1 and 20")


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
    "rag_budget_exhausted": "The RAG budget was exhausted.",
    "rag_disabled": "Server-side RAG is disabled.",
    "rag_invalid_draft": "The generated draft was invalid.",
    "rag_invalid_plan": "The generated plan was invalid.",
    "rag_job_binding_invalid": "The RAG job binding is invalid.",
    "rag_prompt_invalid": "The RAG prompt inputs were invalid.",
    "rag_run_not_found": "The RAG run was not found.",
    "rag_version_conflict": "The conflict retry limit was exhausted.",
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
