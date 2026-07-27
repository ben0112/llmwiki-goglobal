import asyncio
import logging
import time
from pathlib import Path
from uuid import uuid4

import pytest

from tests.helpers.telemetry_contract import assert_telemetry_event

ROOT = Path(__file__).parents[2]


def test_nine_production_callsites_have_json_contract_assertions():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "tests/integration/test_durable_failure_matrix.py",
            ROOT / "tests/unit/jobs/test_dispatcher.py",
            ROOT / "tests/unit/test_hosted_quota.py",
            ROOT / "tests/unit/test_hosted_tus_multipart.py",
        )
    )
    compact = "".join(sources.split())
    for event in (
        "durable_job_dispatched",
        "durable_job_finished",
        "durable_job_lease_reaped",
        "tus_session_created",
        "tus_session_completed",
        "tus_session_stale",
        "quota_reserved",
        "quota_released",
        "upload_cleanup_finished",
    ):
        assert f'assert_telemetry_event(caplog,"{event}"' in compact


def test_retrieval_events_have_exact_versioned_allowlisted_contract(caplog):
    from telemetry import emit

    retrieval_id = uuid4()
    with caplog.at_level(logging.INFO):
        emit(
            logging.getLogger("test.retrieval"),
            "retrieval_fallback",
            schema_version=1,
            retrieval_id=retrieval_id,
            profile="lexical_fallback",
            result_count=2,
            candidate_count=3,
            duration_ms=1.25,
            reason="vector_unavailable",
        )
        emit(
            logging.getLogger("test.retrieval"),
            "retrieval_finished",
            schema_version=1,
            retrieval_id=retrieval_id,
            profile="lexical_fallback",
            result_count=2,
            candidate_count=3,
            duration_ms=1.25,
            error_code=None,
        )

    fallback = assert_telemetry_event(
        caplog,
        "retrieval_fallback",
        expected={"retrieval_id": str(retrieval_id), "reason": "vector_unavailable"},
    )
    finished = assert_telemetry_event(
        caplog,
        "retrieval_finished",
        expected={"retrieval_id": str(retrieval_id), "profile": "lexical_fallback"},
    )
    assert fallback["schema_version"] == finished["schema_version"] == 1


@pytest.mark.parametrize(
    ("event", "fields"),
    [
        (
            "rag_run_started",
            {
                "schema_version": 1,
                "run_id": uuid4(),
                "job_id": uuid4(),
                "model_profile": "primary",
                "model_profile_version": "primary-v1",
                "page_count": 0,
                "step_count": 0,
                "model_token_count": 0,
                "duration_ms": 0,
            },
        ),
        (
            "rag_step_finished",
            {
                "schema_version": 1,
                "run_id": uuid4(),
                "job_id": uuid4(),
                "step_id": uuid4(),
                "page_id": None,
                "model_profile": "primary",
                "model_profile_version": "primary-v1",
                "step_type": "plan",
                "outcome": "succeeded",
                "model_token_count": 12,
                "latency_ms": 1.5,
                "error_code": None,
            },
        ),
        (
            "rag_page_committed",
            {
                "schema_version": 1,
                "run_id": uuid4(),
                "job_id": uuid4(),
                "page_id": uuid4(),
                "document_id": uuid4(),
                "model_profile": "primary",
                "model_profile_version": "primary-v1",
                "page_count": 1,
                "step_count": 6,
                "model_token_count": 12,
                "latency_ms": 2.0,
            },
        ),
        (
            "rag_run_finished",
            {
                "schema_version": 1,
                "run_id": uuid4(),
                "job_id": uuid4(),
                "model_profile": "primary",
                "model_profile_version": "primary-v1",
                "page_count": 1,
                "step_count": 6,
                "model_token_count": 12,
                "duration_ms": 3,
            },
        ),
        (
            "rag_run_failed",
            {
                "schema_version": 1,
                "run_id": uuid4(),
                "job_id": uuid4(),
                "model_profile": "primary",
                "model_profile_version": "primary-v1",
                "page_count": 0,
                "step_count": 1,
                "model_token_count": 12,
                "duration_ms": 3,
                "error_code": "rag_invalid_plan",
            },
        ),
    ],
)
def test_rag_events_have_exact_bounded_shared_contract(caplog, event, fields):
    from telemetry import emit

    with caplog.at_level(logging.INFO):
        emit(logging.getLogger("test.telemetry.rag"), event, **fields)
    assert_telemetry_event(caplog, event)

    first_key = next(key for key in fields if key.endswith("_count"))
    with pytest.raises(ValueError):
        emit(
            logging.getLogger("test.telemetry.rag"),
            event,
            **(fields | {first_key: -1}),
        )
    with pytest.raises(ValueError):
        emit(
            logging.getLogger("test.telemetry.rag"),
            event,
            **(fields | {"goal": "private"}),
        )


def test_generic_telemetry_still_rejects_token_named_fields():
    from telemetry import emit

    with pytest.raises(ValueError):
        emit(logging.getLogger("test.telemetry.generic"), "generic_event", model_token_count=1)


@pytest.mark.parametrize(
    ("event", "fields"),
    [
        (
            "retrieval_finished",
            {
                "schema_version": 1,
                "retrieval_id": uuid4(),
                "profile": "hybrid",
                "result_count": 1,
                "candidate_count": 2,
                "duration_ms": 1.0,
                "error_code": None,
            },
        ),
        (
            "embedding_finished",
            {
                "schema_version": 1,
                "job_id": uuid4(),
                "attempt": 1,
                "outcome": "success",
                "error_code": "embedding_succeeded",
                "provider": "openai_compatible",
                "model": "embed-v1",
                "dimensions": 3,
                "chunk_count": 2,
                "duration_ms": 1,
                "replica_role": "worker",
            },
        ),
    ],
)
def test_sensitive_nested_and_unstable_telemetry_inputs_fail_closed(event, fields):
    from telemetry import emit

    logger = logging.getLogger("test.telemetry.rejection")
    sensitive_values = (
        {"content": "private"},
        [0.1, 0.2],
        RuntimeError("Bearer private"),
        ExceptionGroup("https://user:key@private.invalid", [RuntimeError("secret")]),
    )
    for value in sensitive_values:
        with pytest.raises((TypeError, ValueError)):
            emit(logger, event, **fields, metadata=value)
    with pytest.raises(ValueError):
        emit(logger, event, **(fields | {"duration_ms": float("nan")}))
    with pytest.raises(ValueError):
        emit(logger, event, **(fields | {"candidate_count": True}))
    with pytest.raises(ValueError):
        emit(logger, event, **(fields | {"error_code": "https://private.invalid"}))


def test_telemetry_rejects_unicode_confusables_and_noncanonical_opaque_ids():
    from telemetry import emit

    fields = {
        "schema_version": 1,
        "retrieval_id": "tenant-written-id",
        "profile": "hybrid\N{FULLWIDTH SOLIDUS}private",
        "result_count": 1,
        "candidate_count": 1,
        "duration_ms": 0,
        "error_code": None,
    }
    with pytest.raises(ValueError):
        emit(logging.getLogger("test.telemetry.ids"), "retrieval_finished", **fields)


class _FailingLogger:
    def info(self, *_args, **_kwargs):
        raise RuntimeError("sink contains https://private.invalid")


def test_ordinary_telemetry_sink_failure_is_safe_and_does_not_render_payload():
    from telemetry import emit

    emit(
        _FailingLogger(),
        "retrieval_finished",
        schema_version=1,
        retrieval_id=uuid4(),
        profile="lexical",
        result_count=0,
        candidate_count=0,
        duration_ms=0,
        error_code=None,
    )


@pytest.mark.parametrize(
    "signal",
    [
        KeyboardInterrupt("private"),
        SystemExit("private"),
        asyncio.CancelledError("private"),
        BaseExceptionGroup(
            "private",
            [RuntimeError("ignore"), KeyboardInterrupt("private")],
        ),
    ],
)
def test_linked_or_grouped_control_signals_from_sink_are_sanitized(signal):
    from telemetry import emit

    class SignalLogger:
        def info(self, *_args, **_kwargs):
            raise signal

    expected = KeyboardInterrupt if isinstance(signal, (KeyboardInterrupt, BaseExceptionGroup)) else type(signal)
    with pytest.raises(expected) as raised:
        emit(
            SignalLogger(),
            "retrieval_finished",
            schema_version=1,
            retrieval_id=uuid4(),
            profile="lexical",
            result_count=0,
            candidate_count=0,
            duration_ms=0,
            error_code=None,
        )
    assert "private" not in str(raised.value)


class _UnknownProcessSignal(BaseException):
    pass


@pytest.mark.parametrize(
    "failure",
    [
        _UnknownProcessSignal("private direct"),
        BaseExceptionGroup(
            "private group",
            [RuntimeError("ordinary"), _UnknownProcessSignal("private nested")],
        ),
    ],
)
def test_unknown_base_exception_from_sink_uses_safe_generic_propagation(failure):
    from telemetry import emit

    class SignalLogger:
        def info(self, *_args, **_kwargs):
            raise failure

    with pytest.raises(BaseException) as raised:
        emit(
            SignalLogger(),
            "retrieval_finished",
            schema_version=1,
            retrieval_id=uuid4(),
            profile="lexical",
            result_count=0,
            candidate_count=0,
            duration_ms=0,
            error_code=None,
        )

    assert type(raised.value) is BaseException
    assert raised.value.args == ()
    assert raised.value.__cause__ is None and raised.value.__context__ is None


def test_embedding_finished_is_emitted_only_from_committed_worker_transition(caplog):
    worker_source = (ROOT / "api/jobs/worker.py").read_text(encoding="utf-8")
    provider_source = (ROOT / "api/services/embeddings.py").read_text(encoding="utf-8")

    assert '"embedding_finished"' in worker_source
    assert '"embedding_finished"' not in provider_source
    assert "repository.succeed" in worker_source


@pytest.mark.parametrize(
    ("state", "result", "transition_error", "outcome", "event_error", "chunk_count"),
    [
        ("succeeded", {"document_id": "opaque", "embedded_chunks": 2}, None, "success", "embedding_succeeded", 2),
        ("succeeded", {"document_id": "opaque", "stale": True}, None, "stale", "embedding_stale", 0),
        ("retry_wait", None, "embedding_unavailable", "retry", "embedding_unavailable", 0),
        ("failed", None, "invalid_embedding_response", "terminal", "invalid_embedding_response", 0),
    ],
)
def test_embedding_attempt_event_uses_only_committed_transition_outcome(
    caplog,
    state,
    result,
    transition_error,
    outcome,
    event_error,
    chunk_count,
):
    from jobs.models import JobRecord, JobState, JobType
    from jobs.worker import _emit_finished

    job_id = uuid4()
    document_id = uuid4()
    transition = JobRecord(
        id=job_id,
        job_type=JobType.DOCUMENT_EMBED,
        user_id=uuid4(),
        state=JobState(state),
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": 1,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
        },
        result=result,
        attempt_count=2,
        error_code=transition_error,
    )
    with caplog.at_level(logging.INFO):
        _emit_finished(transition, worker_id="worker-safe", started=time.monotonic())

    assert_telemetry_event(
        caplog,
        "embedding_finished",
        expected={
            "job_id": str(job_id),
            "attempt": 2,
            "outcome": outcome,
            "error_code": event_error,
            "provider": "openai_compatible",
            "model": "embed-v1",
            "dimensions": 3,
            "chunk_count": chunk_count,
        },
    )


def test_committed_embedding_attempt_emits_common_namespaced_model_identity(caplog):
    from jobs.models import JobRecord, JobState, JobType
    from jobs.worker import _emit_finished

    job_id = uuid4()
    document_id = uuid4()
    transition = JobRecord(
        id=job_id,
        job_type=JobType.DOCUMENT_EMBED,
        user_id=uuid4(),
        state=JobState.SUCCEEDED,
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": 1,
            "provider": "openai_compatible",
            "model": "vendor/embed:v1",
            "dimensions": 3,
        },
        result={"document_id": str(document_id), "embedded_chunks": 2},
        attempt_count=1,
    )

    with caplog.at_level(logging.INFO, logger="jobs.worker"):
        _emit_finished(transition, worker_id="worker-safe", started=time.monotonic())

    assert_telemetry_event(
        caplog,
        "embedding_finished",
        expected={
            "job_id": str(job_id),
            "model": "vendor/embed:v1",
            "outcome": "success",
        },
    )


@pytest.mark.parametrize(
    "model",
    [
        "https://user:pass@private.invalid/model",
        "vendor//embed:v1",
        "vendor\\embed:v1",
        "vendor/embed v1",
        "vendor/embed\nprivate",
        "vendor/embed?token=private",
        "vendor/embed#private",
        "vendor/embed%2fprivate",
        "/absolute/model",
        "vendor/../private",
        "vendor/.hidden",
        "a" * 201,
        "vendor/embed\N{FULLWIDTH COLON}v1",
    ],
)
def test_embedding_model_identity_rejects_unsafe_or_ambiguous_values_without_logging(
    caplog,
    model,
):
    from telemetry import emit

    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="model"):
        emit(
            logging.getLogger("test.telemetry.model"),
            "embedding_finished",
            schema_version=1,
            job_id=uuid4(),
            attempt=1,
            outcome="success",
            error_code="embedding_succeeded",
            provider="openai_compatible",
            model=model,
            dimensions=3,
            chunk_count=1,
            duration_ms=0,
            replica_role="worker",
        )

    assert "embedding_finished" not in caplog.text
    assert model not in caplog.text


def test_malformed_persisted_embedding_identity_fails_closed_without_masking_job_event(caplog):
    from jobs.models import JobRecord, JobState, JobType
    from jobs.worker import _emit_finished

    document_id = uuid4()
    transition = JobRecord(
        id=uuid4(),
        job_type=JobType.DOCUMENT_EMBED,
        user_id=uuid4(),
        state=JobState.SUCCEEDED,
        knowledge_base_id=uuid4(),
        document_id=document_id,
        payload={"provider": "https://key@private", "model": "embed-v1", "dimensions": 3},
        result={"document_id": str(document_id), "embedded_chunks": 1},
        attempt_count=1,
    )
    with caplog.at_level(logging.INFO):
        _emit_finished(transition, worker_id="worker-safe", started=time.monotonic())

    assert_telemetry_event(caplog, "durable_job_finished")
    assert_telemetry_event(caplog, "embedding_finished", count=0)
    assert "private" not in caplog.text.lower()
