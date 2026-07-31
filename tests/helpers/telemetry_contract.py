"""Assertions for the public structured-telemetry event surface."""

from __future__ import annotations

import json

EVENT_SCHEMAS = {
    "retrieval_finished": {
        "schema_version": int,
        "retrieval_id": str,
        "profile": str,
        "result_count": int,
        "candidate_count": int,
        "duration_ms": (int, float),
        "error_code": (str, type(None)),
    },
    "retrieval_fallback": {
        "schema_version": int,
        "retrieval_id": str,
        "profile": str,
        "result_count": int,
        "candidate_count": int,
        "duration_ms": (int, float),
        "reason": str,
    },
    "embedding_finished": {
        "schema_version": int,
        "job_id": str,
        "attempt": int,
        "outcome": str,
        "error_code": str,
        "provider": str,
        "model": str,
        "dimensions": int,
        "chunk_count": int,
        "duration_ms": int,
        "replica_role": str,
    },
    "durable_job_dispatched": {
        "job_id": str,
        "job_type": str,
        "state": str,
        "dispatch_attempts": int,
        "dispatch_lag_ms": int,
        "replica_role": str,
    },
    "durable_job_finished": {
        "job_id": str,
        "job_type": str,
        "attempt": int,
        "state": str,
        "lease_owner": str,
        "duration_ms": int,
        "error_code": (str, type(None)),
        "replica_role": str,
    },
    "durable_job_lease_reaped": {
        "job_id": str,
        "state": str,
        "error_code": (str, type(None)),
        "replica_role": str,
    },
    "rag_run_started": {
        "schema_version": int,
        "run_id": str,
        "job_id": str,
        "model_profile": str,
        "model_profile_version": str,
        "page_count": int,
        "step_count": int,
        "model_token_count": int,
        "duration_ms": int,
    },
    "rag_step_finished": {
        "schema_version": int,
        "run_id": str,
        "job_id": str,
        "step_id": str,
        "page_id": (str, type(None)),
        "model_profile": str,
        "model_profile_version": str,
        "step_type": str,
        "outcome": str,
        "model_token_count": int,
        "latency_ms": (int, float),
        "error_code": (str, type(None)),
    },
    "rag_page_committed": {
        "schema_version": int,
        "run_id": str,
        "job_id": str,
        "page_id": str,
        "document_id": str,
        "model_profile": str,
        "model_profile_version": str,
        "page_count": int,
        "step_count": int,
        "model_token_count": int,
        "latency_ms": (int, float),
    },
    "rag_run_finished": {
        "schema_version": int,
        "run_id": str,
        "job_id": str,
        "model_profile": str,
        "model_profile_version": str,
        "page_count": int,
        "step_count": int,
        "model_token_count": int,
        "duration_ms": int,
    },
    "rag_run_failed": {
        "schema_version": int,
        "run_id": str,
        "job_id": str,
        "model_profile": str,
        "model_profile_version": str,
        "page_count": int,
        "step_count": int,
        "model_token_count": int,
        "duration_ms": int,
        "error_code": str,
    },
    "tus_session_created": {"upload_id": str, "state": str, "replica_role": str},
    "tus_session_completed": {"upload_id": str, "state": str, "replica_role": str},
    "tus_session_stale": {"upload_id": str, "state": str, "replica_role": str},
    "quota_reserved": {"upload_id": str, "byte_count": int, "replica_role": str},
    "quota_released": {"upload_id": str, "byte_count": int, "replica_role": str},
    "upload_cleanup_finished": {"upload_id": str, "state": str, "replica_role": str},
}

_FORBIDDEN_KEY_FRAGMENTS = (
    "payload",
    "object_url",
    "s3_key",
    "secret",
    "token",
    "raw",
    "content",
    "query",
    "vector",
    "authorization",
    "dsn",
    "sql",
    "exception",
    "metadata",
    "path",
    "goal",
    "prompt",
    "evidence",
    "api_key",
    "url",
    "exception",
)
_FORBIDDEN_VALUE_FRAGMENTS = (
    "postgresql://",
    "redis://",
    "s3://",
    "http://",
    "https://",
    "bearer ",
    "private key",
)


def assert_telemetry_event(
    caplog,
    event: str,
    *,
    expected: dict | None = None,
    sensitive=(),
    count: int = 1,
) -> dict:
    """Parse and validate one event emitted by a production callsite."""
    matches = []
    for record in caplog.records:
        try:
            body = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(body, dict) and body.get("event") == event:
            matches.append(body)
    assert len(matches) == count, f"expected {count} {event} events, got {matches!r}"
    schema = EVENT_SCHEMAS[event]
    for body in matches:
        assert set(body) == {"event", *schema}
        assert body["event"] == event
        for field, field_type in schema.items():
            allowed = field_type if isinstance(field_type, tuple) else (field_type,)
            assert type(body[field]) in allowed, f"{event}.{field} has unstable type {type(body[field]).__name__}"
        if "replica_role" in body:
            assert body["replica_role"] in {"api", "worker"}
        permitted_rag_fields = {"model_token_count"} if event.startswith("rag_") else set()
        assert not any(
            fragment in field.lower()
            for field in body
            if field not in permitted_rag_fields
            for fragment in _FORBIDDEN_KEY_FRAGMENTS
        )
        serialized = json.dumps(body, ensure_ascii=True, sort_keys=True).lower()
        assert not any(fragment in serialized for fragment in _FORBIDDEN_VALUE_FRAGMENTS)
        for value in sensitive:
            assert str(value).lower() not in serialized
    if count == 0:
        return {}
    body = matches[-1]
    if expected is not None:
        assert body.items() >= expected.items()
    return body
