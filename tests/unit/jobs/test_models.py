from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from jobs.models import (
    ERROR_MESSAGE_MAX_CHARS,
    PAYLOAD_MAX_BYTES,
    PROGRESS_MAX_BYTES,
    RESULT_MAX_BYTES,
    JobCancelled,
    JobCreate,
    JobRecord,
    JobState,
    JobType,
    LeaseLost,
    retry_delay_seconds,
    serialize_public_job,
)


def test_job_type_values_are_stable():
    assert [item.value for item in JobType] == [
        "document.extract",
        "graph.rebuild",
        "upload.cleanup",
    ]


def test_job_state_values_and_terminal_states_are_stable():
    assert [item.value for item in JobState] == [
        "queued",
        "running",
        "retry_wait",
        "succeeded",
        "failed",
        "cancelled",
    ]
    assert {state for state in JobState if state.is_terminal} == {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }


def test_retry_delay_is_capped():
    assert retry_delay_seconds(attempt=1, jitter=0) == 2
    assert retry_delay_seconds(attempt=5, jitter=0) == 32
    assert retry_delay_seconds(attempt=20, jitter=0) == 60


def test_retry_delay_validates_attempt_and_bounds_injected_jitter():
    with pytest.raises(ValueError, match="attempt"):
        retry_delay_seconds(attempt=0, jitter=0)

    assert retry_delay_seconds(attempt=1, jitter=lambda maximum: maximum) == 4
    assert retry_delay_seconds(attempt=1, jitter=lambda maximum: -maximum) == 2
    assert retry_delay_seconds(attempt=1, jitter=lambda maximum: maximum * 2) == 4


def test_job_dataclasses_are_frozen_and_do_not_share_json_defaults():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = JobCreate(job_type=JobType.DOCUMENT_EXTRACT, user_id=uuid4())
    second = JobCreate(job_type=JobType.DOCUMENT_EXTRACT, user_id=uuid4())
    record = JobRecord(
        id=uuid4(),
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=uuid4(),
        created_at=now,
        updated_at=now,
    )

    assert is_dataclass(first) and is_dataclass(record)
    assert first.payload == second.payload == {}
    assert first.payload is not second.payload
    with pytest.raises(FrozenInstanceError):
        record.state = JobState.RUNNING  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.payload["document_id"] = "not mutable"  # type: ignore[index]


def test_job_json_fields_are_recursively_defensive_and_immutable():
    create_payload = {"nested": {"items": ["create"]}}
    payload = {"nested": {"items": ["payload"]}}
    progress = {"nested": {"items": ["progress"]}}
    result = {"nested": {"items": ["result"]}}
    create = JobCreate(
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=uuid4(),
        payload=create_payload,
    )
    record = JobRecord(
        id=uuid4(),
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=uuid4(),
        payload=payload,
        progress=progress,
        result=result,
    )

    create_payload["nested"]["items"].append("changed")
    payload["nested"]["items"].append("changed")
    progress["nested"]["items"].append("changed")
    result["nested"]["items"].append("changed")

    assert create.payload == {"nested": {"items": ("create",)}}
    assert record.payload == {"nested": {"items": ("payload",)}}
    assert record.progress == {"nested": {"items": ("progress",)}}
    assert record.result == {"nested": {"items": ("result",)}}
    with pytest.raises(TypeError):
        record.payload["nested"]["new"] = "value"  # type: ignore[index]
    with pytest.raises(AttributeError):
        record.progress["nested"]["items"].append("value")  # type: ignore[union-attr]


def test_job_domain_exceptions_are_runtime_errors_with_messages():
    for exception_type in (LeaseLost, JobCancelled):
        with pytest.raises(RuntimeError, match="job state changed") as raised:
            raise exception_type("job state changed")
        assert isinstance(raised.value, exception_type)


def test_public_job_serialization_allow_lists_fields_and_sanitizes_errors():
    job_id = UUID("11111111-1111-1111-1111-111111111111")
    user_id = UUID("22222222-2222-2222-2222-222222222222")
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    record = JobRecord(
        id=job_id,
        job_type=JobType.DOCUMENT_EXTRACT,
        user_id=user_id,
        state=JobState.FAILED,
        payload={"secret": "do not expose"},
        progress={"percent": 50, "steps": ["queued", "running"]},
        result={"document_id": "abc", "artifacts": [{"name": "summary"}]},
        attempt_count=3,
        max_attempts=3,
        lease_owner="worker-private",
        lease_expires_at=created_at,
        heartbeat_at=created_at,
        last_dispatched_at=created_at,
        dispatch_attempts=7,
        error_code="document_not_found",
        error_message="database timeout: internal detail",
        cancel_requested_at=created_at,
        created_at=created_at,
        updated_at=created_at,
    )

    public = serialize_public_job(record)

    assert isinstance(public["progress"], dict)
    assert isinstance(public["progress"]["steps"], list)
    assert isinstance(public["result"], dict)
    assert isinstance(public["result"]["artifacts"], list)
    assert public == {
        "id": str(job_id),
        "type": "document.extract",
        "state": "failed",
        "progress": {"percent": 50, "steps": ["queued", "running"]},
        "result": {"document_id": "abc", "artifacts": [{"name": "summary"}]},
        "attempt_count": 3,
        "max_attempts": 3,
        "cancel_requested": True,
        "error": {
            "code": "document_not_found",
            "message": "The requested document was not found.",
        },
        "created_at": "2026-01-02T03:04:05+00:00",
        "updated_at": "2026-01-02T03:04:05+00:00",
    }
    assert not ({"payload", "lease_owner", "lease_expires_at", "heartbeat_at", "last_dispatched_at", "dispatch_attempts", "error_message"} & public.keys())


def test_unknown_error_code_is_replaced_with_generic_public_error():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    record = JobRecord(
        id=uuid4(),
        job_type=JobType.GRAPH_REBUILD,
        user_id=uuid4(),
        error_code="postgres_deadlock",
        error_message="leaks implementation details",
        created_at=now,
        updated_at=now,
    )

    assert serialize_public_job(record)["error"] == {
        "code": "internal_error",
        "message": "The job could not be completed.",
    }


def test_model_size_constants_match_database_contract():
    assert PAYLOAD_MAX_BYTES == 16_384
    assert PROGRESS_MAX_BYTES == 8_192
    assert RESULT_MAX_BYTES == 16_384
    assert ERROR_MESSAGE_MAX_CHARS == 2_000
