"""Strict, payload-free structured telemetry shared by API and MCP runtimes."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from math import isfinite
from re import compile as compile_pattern
from uuid import UUID

from .signals import sanitized_boundary_signal_or_unknown

TELEMETRY_SCHEMA_VERSION = 1
_MAX_STRING_CHARS = 256
_MAX_COUNT = 2_147_483_647
_MAX_DURATION_MS = 604_800_000.0
_STABLE_CODE = compile_pattern(r"[a-z][a-z0-9_]{0,63}\Z")
_PROFILE_IDENTITY = compile_pattern(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
_MODEL_IDENTITY = compile_pattern(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")
_FORBIDDEN_KEY_FRAGMENTS = (
    "authorization",
    "chunk",
    "content",
    "dsn",
    "embedding",
    "exception",
    "metadata",
    "object_url",
    "path",
    "payload",
    "query",
    "raw",
    "s3_key",
    "secret",
    "sql",
    "token",
    "vector",
)
_RETRIEVAL_PROFILES = frozenset({"lexical", "hybrid", "lexical_fallback"})
_FALLBACK_REASONS = frozenset({"vector_unavailable"})
_EMBEDDING_OUTCOMES = frozenset({"success", "stale", "retry", "terminal"})
_RAG_STEP_TYPES = frozenset({"plan", "retrieve", "read", "draft", "validate", "write", "lint", "conflict"})
_RAG_STEP_OUTCOMES = frozenset({"succeeded", "failed"})

_NEW_EVENT_FIELDS = {
    "retrieval_finished": frozenset(
        {
            "schema_version",
            "retrieval_id",
            "profile",
            "result_count",
            "candidate_count",
            "duration_ms",
            "error_code",
        }
    ),
    "retrieval_fallback": frozenset(
        {
            "schema_version",
            "retrieval_id",
            "profile",
            "result_count",
            "candidate_count",
            "duration_ms",
            "reason",
        }
    ),
    "embedding_finished": frozenset(
        {
            "schema_version",
            "job_id",
            "attempt",
            "outcome",
            "error_code",
            "provider",
            "model",
            "dimensions",
            "chunk_count",
            "duration_ms",
            "replica_role",
        }
    ),
    "rag_run_started": frozenset(
        {
            "schema_version",
            "run_id",
            "job_id",
            "model_profile",
            "model_profile_version",
            "page_count",
            "step_count",
            "model_token_count",
            "duration_ms",
        }
    ),
    "rag_step_finished": frozenset(
        {
            "schema_version",
            "run_id",
            "job_id",
            "step_id",
            "page_id",
            "model_profile",
            "model_profile_version",
            "step_type",
            "outcome",
            "model_token_count",
            "latency_ms",
            "error_code",
        }
    ),
    "rag_page_committed": frozenset(
        {
            "schema_version",
            "run_id",
            "job_id",
            "page_id",
            "document_id",
            "model_profile",
            "model_profile_version",
            "page_count",
            "step_count",
            "model_token_count",
            "latency_ms",
        }
    ),
    "rag_run_finished": frozenset(
        {
            "schema_version",
            "run_id",
            "job_id",
            "model_profile",
            "model_profile_version",
            "page_count",
            "step_count",
            "model_token_count",
            "duration_ms",
        }
    ),
    "rag_run_failed": frozenset(
        {
            "schema_version",
            "run_id",
            "job_id",
            "model_profile",
            "model_profile_version",
            "page_count",
            "step_count",
            "model_token_count",
            "duration_ms",
            "error_code",
        }
    ),
}


class TelemetryContractError(ValueError):
    """A callsite attempted to cross the strict telemetry data boundary."""


def _safe_scalar(value: object) -> str | int | float | bool | None:
    if isinstance(value, Enum):
        value = value.value
    elif isinstance(value, (UUID, date, datetime)):
        value = str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > _MAX_STRING_CHARS:
            raise TelemetryContractError("telemetry string is too long")
        if isinstance(value, float) and not isfinite(value):
            raise TelemetryContractError("telemetry number must be finite")
        return value
    raise TypeError(f"unsupported telemetry value type: {type(value).__name__}")


def _require_exact_fields(event: str, fields: dict[str, object]) -> None:
    expected = _NEW_EVENT_FIELDS[event]
    if set(fields) != expected:
        raise TelemetryContractError("telemetry event fields do not match schema")


def _require_integer(fields: dict[str, object], name: str, *, minimum: int, maximum: int) -> int:
    value = fields[name]
    if type(value) is not int or not minimum <= value <= maximum:
        raise TelemetryContractError(f"telemetry {name} is outside its supported range")
    return value


def _require_duration(fields: dict[str, object]) -> None:
    value = fields["duration_ms"]
    if type(value) not in (int, float):
        raise TelemetryContractError("telemetry duration_ms must be numeric")
    normalized = float(value)
    if not isfinite(normalized) or not 0 <= normalized <= _MAX_DURATION_MS:
        raise TelemetryContractError("telemetry duration_ms is outside its supported range")


def _require_named_duration(fields: dict[str, object], name: str) -> None:
    value = fields[name]
    if type(value) not in (int, float):
        raise TelemetryContractError(f"telemetry {name} must be numeric")
    normalized = float(value)
    if not isfinite(normalized) or not 0 <= normalized <= _MAX_DURATION_MS:
        raise TelemetryContractError(f"telemetry {name} is outside its supported range")


def _require_uuid(fields: dict[str, object], name: str) -> None:
    value = fields[name]
    try:
        parsed = value if isinstance(value, UUID) else UUID(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or (isinstance(value, str) and str(parsed) != value):
        raise TelemetryContractError(f"telemetry {name} must be a canonical UUID")


def _require_optional_uuid(fields: dict[str, object], name: str) -> None:
    if fields[name] is None:
        return
    _require_uuid(fields, name)


def _require_choice(fields: dict[str, object], name: str, choices: frozenset[str]) -> None:
    value = fields[name]
    if not isinstance(value, str) or value not in choices:
        raise TelemetryContractError(f"telemetry {name} is unsupported")


def _require_stable_code(fields: dict[str, object], name: str, *, optional: bool = False) -> None:
    value = fields[name]
    if optional and value is None:
        return
    if not isinstance(value, str) or not _STABLE_CODE.fullmatch(value):
        raise TelemetryContractError(f"telemetry {name} must be a stable code")


def _require_profile_identity(fields: dict[str, object], name: str) -> None:
    value = fields[name]
    if not isinstance(value, str) or not _PROFILE_IDENTITY.fullmatch(value):
        raise TelemetryContractError(f"telemetry {name} must be a bounded profile identity")


def _require_model_identity(fields: dict[str, object]) -> None:
    value = fields["model"]
    if not isinstance(value, str) or not _MODEL_IDENTITY.fullmatch(value):
        raise TelemetryContractError("telemetry model must be a bounded model identity")
    if value.count("/") > 1 or value.count(":") > 1 or "://" in value or "//" in value:
        raise TelemetryContractError("telemetry model must be a bounded model identity")
    if "/" in value and ":" in value and value.index(":") < value.index("/"):
        raise TelemetryContractError("telemetry model must be a bounded model identity")
    if any(not segment or segment.startswith(".") for segment in value.split("/")):
        raise TelemetryContractError("telemetry model must be a bounded model identity")
    if any(not segment for segment in value.split(":")):
        raise TelemetryContractError("telemetry model must be a bounded model identity")


def _validate_new_event(event: str, fields: dict[str, object]) -> None:
    _require_exact_fields(event, fields)
    if fields["schema_version"] != TELEMETRY_SCHEMA_VERSION or type(fields["schema_version"]) is not int:
        raise TelemetryContractError("unsupported telemetry schema_version")
    if "duration_ms" in fields:
        _require_duration(fields)
    if event.startswith("retrieval_"):
        _require_uuid(fields, "retrieval_id")
        _require_choice(fields, "profile", _RETRIEVAL_PROFILES)
        _require_integer(fields, "result_count", minimum=0, maximum=_MAX_COUNT)
        _require_integer(fields, "candidate_count", minimum=0, maximum=_MAX_COUNT)
        if fields["result_count"] > fields["candidate_count"]:
            raise TelemetryContractError("telemetry result_count exceeds candidate_count")
        if event == "retrieval_finished":
            _require_stable_code(fields, "error_code", optional=True)
        else:
            if fields["profile"] != "lexical_fallback":
                raise TelemetryContractError("fallback profile must be lexical_fallback")
            _require_choice(fields, "reason", _FALLBACK_REASONS)
        return

    if event.startswith("rag_"):
        _validate_rag_event(event, fields)
        return

    _require_uuid(fields, "job_id")
    _require_integer(fields, "attempt", minimum=1, maximum=1_000_000)
    _require_choice(fields, "outcome", _EMBEDDING_OUTCOMES)
    _require_stable_code(fields, "error_code")
    _require_profile_identity(fields, "provider")
    _require_model_identity(fields)
    _require_integer(fields, "dimensions", minimum=1, maximum=4096)
    _require_integer(fields, "chunk_count", minimum=0, maximum=_MAX_COUNT)
    _require_choice(fields, "replica_role", frozenset({"worker"}))


def _validate_rag_event(event: str, fields: dict[str, object]) -> None:
    _require_uuid(fields, "run_id")
    _require_uuid(fields, "job_id")
    _require_profile_identity(fields, "model_profile")
    _require_profile_identity(fields, "model_profile_version")
    _require_integer(fields, "model_token_count", minimum=0, maximum=250_000)
    if event == "rag_step_finished":
        _require_uuid(fields, "step_id")
        _require_optional_uuid(fields, "page_id")
        _require_choice(fields, "step_type", _RAG_STEP_TYPES)
        _require_choice(fields, "outcome", _RAG_STEP_OUTCOMES)
        _require_named_duration(fields, "latency_ms")
        _require_stable_code(fields, "error_code", optional=True)
        return
    if event == "rag_page_committed":
        _require_uuid(fields, "page_id")
        _require_uuid(fields, "document_id")
        _require_named_duration(fields, "latency_ms")
    _require_integer(fields, "page_count", minimum=0, maximum=_MAX_COUNT)
    _require_integer(fields, "step_count", minimum=0, maximum=250_000)
    if event == "rag_run_failed":
        _require_stable_code(fields, "error_code")


def validated_event(event: str, /, **fields: object) -> dict[str, str | int | float | bool | None]:
    """Return a normalized event only after the complete schema boundary passes."""
    if not isinstance(event, str) or not _STABLE_CODE.fullmatch(event):
        raise TelemetryContractError("event must be a stable non-empty code")
    if event in _NEW_EVENT_FIELDS:
        _validate_new_event(event, fields)
    else:
        forbidden = sorted(
            name
            for name in fields
            if not isinstance(name, str) or any(fragment in name.lower() for fragment in _FORBIDDEN_KEY_FRAGMENTS)
        )
        if forbidden:
            raise TelemetryContractError("forbidden telemetry field")
    return {"event": event, **{name: _safe_scalar(value) for name, value in fields.items()}}


def emit(logger: object, event: str, /, **fields: object) -> None:
    """Validate and emit one compact JSON event without exposing sink failures."""
    body = validated_event(event, **fields)
    serialized = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    failure = None
    try:
        logger.info(serialized)
    except BaseException as caught:  # noqa: BLE001 - telemetry is a process boundary.
        failure = caught
    if failure is None:
        return
    if signal := sanitized_boundary_signal_or_unknown(failure):
        raise signal from None


__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryContractError",
    "emit",
    "validated_event",
]
