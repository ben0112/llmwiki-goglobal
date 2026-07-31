from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import asyncpg
import pytest

from llmwiki_core.rag import RagPageState
from tests.helpers.telemetry_contract import assert_telemetry_event

RUN_ID = uuid4()
JOB_ID = uuid4()
USER_ID = uuid4()
KB_ID = uuid4()
NOW = datetime(2026, 7, 27, tzinfo=UTC)


class FakePool:
    def __init__(self) -> None:
        self.connection = self

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

    @asynccontextmanager
    async def transaction(self):
        yield


class FakeLease:
    async def checkpoint(self):
        return None


class FakeOrchestrator:
    def __init__(self, *, result=None, failure: BaseException | None = None) -> None:
        self.result = result or {
            "run_id": str(RUN_ID),
            "completion_reason": "completed",
            "pages_committed": 2,
        }
        self.failure = failure
        self.runs = []

    async def run(self, run, lease):
        self.runs.append((run, lease))
        if self.failure is not None:
            raise self.failure
        return self.result


def _job(**changes):
    from jobs.models import JobRecord, JobState, JobType

    record = JobRecord(
        id=JOB_ID,
        job_type=JobType.BUILD_WIKI,
        user_id=USER_ID,
        state=JobState.RUNNING,
        knowledge_base_id=KB_ID,
        payload={"run_id": str(RUN_ID)},
        lease_owner="worker-safe",
    )
    return replace(record, **changes)


def _run(**changes):
    from rag.records import RagRunRecord

    from llmwiki_core.rag import RagBudget, RagUsage

    record = RagRunRecord(
        id=RUN_ID,
        job_id=JOB_ID,
        root_run_id=RUN_ID,
        parent_run_id=None,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        goal="private goal",
        goal_digest=hashlib.sha256(b"private goal").hexdigest(),
        target_path_prefix="/wiki/private/",
        model_profile="primary",
        model_profile_version="primary-v1",
        retrieval_profile="lexical",
        dry_run=False,
        budget=RagBudget(),
        usage=RagUsage(steps=3, model_tokens=17),
        idempotency_key="private-key",
        request_digest="a" * 64,
        completion_reason=None,
        last_committed_ordinal=1,
        created_at=NOW,
        updated_at=NOW,
    )
    return replace(record, **changes)


def _page(ordinal: int, **changes):
    from rag.records import RagPageRecord

    from llmwiki_core.rag import RagPageState

    record = RagPageRecord(
        id=uuid4(),
        run_id=RUN_ID,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        ordinal=ordinal,
        path=f"/wiki/private/page-{ordinal}.md",
        intent="private intent",
        query="private query",
        state=RagPageState.COMMITTED,
        document_id=uuid4(),
        version_read=1,
        version_committed=2,
        attempt_count=1,
        conflict_retry_count=0,
        last_completed_step_sequence=ordinal + 1,
        preview=None,
        preview_digest=None,
        preview_full_char_count=None,
        preview_truncated=False,
        lint_summary={"warnings": 0},
        created_at=NOW,
        updated_at=NOW,
    )
    return replace(record, **changes)


def _context(orchestrator: FakeOrchestrator | None):
    from jobs.handlers import WorkerContext

    return WorkerContext(
        pool=FakePool(),
        s3=None,
        converter_url="https://converter.invalid",
        converter_secret="private-secret",
        rag_orchestrator_factory=None if orchestrator is None else lambda: orchestrator,
    )


class HostileResultMapping(Mapping):
    def __init__(self, *, reported_length: int) -> None:
        self.reported_length = reported_length
        self.len_calls = 0
        self.iter_calls = 0
        self.getitem_calls = 0

    def __len__(self):
        self.len_calls += 1
        return self.reported_length

    def __iter__(self):
        self.iter_calls += 1
        raise RuntimeError("TOP_SECRET_UNBOUNDED_ITERATION")

    def __getitem__(self, _key):
        self.getitem_calls += 1
        raise RuntimeError("TOP_SECRET_TIME_VARYING_VALUE")


class HostileResultDict(dict):
    def __init__(self, value):
        super().__init__(value)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        raise RuntimeError("TOP_SECRET_DICT_SUBCLASS")


class FakeOwnedModel:
    def __init__(self, *close_outcomes: BaseException | None) -> None:
        self.close_outcomes = list(close_outcomes)
        self.close_calls = 0

    async def complete_json(self, **_kwargs):
        raise AssertionError("fake build orchestrator owns operation execution")

    async def aclose(self):
        self.close_calls += 1
        outcome = self.close_outcomes.pop(0) if self.close_outcomes else None
        if outcome is not None:
            raise outcome


def _scoped_orchestrator(monkeypatch, owned_model, *, operation_result=None, operation_failure=None):
    from rag import handler
    from rag import model as rag_model
    from rag.model import ResolvedRagModelProfile

    class FakeBuildOrchestrator:
        def __init__(self, _ports):
            pass

        async def run(self, _run, _lease):
            if operation_failure is not None:
                raise operation_failure
            return operation_result or {
                "run_id": str(RUN_ID),
                "completion_reason": "completed",
                "pages_committed": 2,
            }

    profile = ResolvedRagModelProfile(
        name="primary",
        base_url="https://model.invalid/v1",
        model="model-v1",
        timeout_seconds=30,
        version="primary-v1",
        api_key="private-key",
    )
    monkeypatch.setattr(rag_model, "OpenAICompatibleRagModel", lambda *_args, **_kwargs: owned_model)
    monkeypatch.setattr(handler, "BuildWikiOrchestrator", FakeBuildOrchestrator)
    orchestrator = handler._RunScopedOrchestrator(
        object(), SimpleNamespace(SERVER_RAG_ENABLED=True), {"primary": profile}
    )
    return orchestrator, SimpleNamespace(owner="worker-safe")


@pytest.mark.asyncio
async def test_build_wiki_handler_loads_authoritative_exact_job_bound_run(monkeypatch, caplog):
    from rag import handler

    from llmwiki_core.rag import RagCompletionReason

    run = _run()
    orchestrator = FakeOrchestrator()
    context = _context(orchestrator)
    calls = []

    async def get_for_worker(conn, run_id, job_id):
        calls.append((conn, run_id, job_id))
        return run

    async def terminal_snapshot(conn, *, job, run_id):
        calls.append((conn, run_id, job.id))
        return replace(run, completion_reason=RagCompletionReason.COMPLETED), (_page(0), _page(1))

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(handler.repository, "get_terminal_snapshot_for_worker", terminal_snapshot)
    with caplog.at_level(logging.INFO, logger="rag.handler"):
        result = await handler.handle_build_wiki(_job(), FakeLease(), context)

    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
    }
    assert calls == [
        (context.pool.connection, RUN_ID, JOB_ID),
        (context.pool.connection, RUN_ID, JOB_ID),
    ]
    assert orchestrator.runs == [(run, orchestrator.runs[0][1])]
    assert_telemetry_event(
        caplog,
        "rag_run_started",
        expected={"run_id": str(RUN_ID), "job_id": str(JOB_ID), "model_profile": "primary"},
    )
    assert_telemetry_event(
        caplog,
        "rag_run_finished",
        expected={"run_id": str(RUN_ID), "job_id": str(JOB_ID), "page_count": 2},
        sensitive=("private goal", "/wiki/private/", "private-key", "https://converter.invalid"),
    )


@pytest.mark.asyncio
async def test_build_wiki_handler_accepts_partial_dry_run_replay_with_all_five_result_fields(monkeypatch):
    from rag import handler

    from llmwiki_core.rag import RagBudget, RagCompletionReason

    run = _run(dry_run=True, budget=RagBudget(max_pages=2))
    result = {
        "run_id": str(RUN_ID),
        "completion_reason": "dry_run",
        "pages_committed": 0,
        "pages_skipped": 1,
        "pages_dry_run": 2,
    }
    orchestrator = FakeOrchestrator(result=result)
    calls = []

    async def get_for_worker(*args):
        calls.append(args)
        return run

    async def terminal_snapshot(*_args, **_kwargs):
        pages = (
            _page(0, state=RagPageState.DRY_RUN_COMPLETE, document_id=None, version_read=None, version_committed=None),
            _page(1, state=RagPageState.DRY_RUN_COMPLETE, document_id=None, version_read=None, version_committed=None),
        )
        return replace(run, completion_reason=RagCompletionReason.DRY_RUN), pages

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(handler.repository, "get_terminal_snapshot_for_worker", terminal_snapshot)

    assert await handler.handle_build_wiki(_job(), FakeLease(), _context(orchestrator)) == result
    assert len(calls) == 1
    assert orchestrator.runs[0][0] is run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "run_changes", "pages", "result"),
    [
        (
            "completed",
            {},
            (_page(0), _page(1)),
            {"run_id": str(RUN_ID), "completion_reason": "completed", "pages_committed": 1},
        ),
        (
            "dry_run",
            {"dry_run": True},
            (
                _page(
                    0,
                    state=RagPageState.DRY_RUN_COMPLETE,
                    document_id=None,
                    version_read=None,
                    version_committed=None,
                ),
                _page(
                    1,
                    state=RagPageState.DRY_RUN_COMPLETE,
                    document_id=None,
                    version_read=None,
                    version_committed=None,
                ),
            ),
            {
                "run_id": str(RUN_ID),
                "completion_reason": "dry_run",
                "pages_committed": 0,
                "pages_dry_run": 1,
            },
        ),
        (
            "no_work",
            {},
            (_page(0),),
            {"run_id": str(RUN_ID), "completion_reason": "no_work", "pages_committed": 0},
        ),
    ],
)
async def test_build_wiki_handler_rejects_result_counters_that_disagree_with_authoritative_pages(
    monkeypatch,
    reason,
    run_changes,
    pages,
    result,
):
    from jobs.handlers import TerminalJobError
    from rag import handler

    from llmwiki_core.rag import RagCompletionReason

    run = _run(**run_changes)

    async def get_for_worker(*_args):
        return run

    async def terminal_snapshot(*_args, **_kwargs):
        return replace(run, completion_reason=RagCompletionReason(reason)), pages

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(handler.repository, "get_terminal_snapshot_for_worker", terminal_snapshot)

    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(result=result)),
        )

    assert raised.value.error_code == "rag_job_binding_invalid"


@pytest.mark.asyncio
async def test_build_wiki_handler_bounds_attempt_overlap_by_authoritative_page_total(monkeypatch):
    from jobs.handlers import TerminalJobError
    from rag import handler

    from llmwiki_core.rag import RagCompletionReason

    run = _run()
    result = {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 3,
        "pages_skipped": 3,
    }

    async def get_for_worker(*_args):
        return run

    async def terminal_snapshot(*_args, **_kwargs):
        return replace(run, completion_reason=RagCompletionReason.COMPLETED), (_page(0), _page(1))

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(handler.repository, "get_terminal_snapshot_for_worker", terminal_snapshot)

    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(FakeOrchestrator(result=result)))

    assert raised.value.error_code == "rag_job_binding_invalid"


@pytest.mark.asyncio
async def test_build_wiki_handler_maps_malformed_authoritative_snapshot_terminal(monkeypatch):
    from jobs.handlers import TerminalJobError
    from rag import handler

    from llmwiki_core.rag import RagDomainError

    run = _run()

    async def get_for_worker(*_args):
        return run

    async def malformed_snapshot(*_args, **_kwargs):
        raise RagDomainError("rag_job_binding_invalid", "TOP_SECRET_MALFORMED")

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    monkeypatch.setattr(handler.repository, "get_terminal_snapshot_for_worker", malformed_snapshot)

    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(FakeOrchestrator()))

    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_job_binding_invalid",
        "The RAG job binding is invalid.",
    )
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"knowledge_base_id": None},
        {"document_id": uuid4()},
        {"payload": {}},
        {"payload": {"run_id": str(RUN_ID), "goal": "private"}},
        {"payload": {"run_id": str(RUN_ID).upper()}},
    ],
)
async def test_build_wiki_handler_rejects_non_exact_job_shape_without_database(changes):
    from jobs.handlers import TerminalJobError
    from rag.handler import handle_build_wiki

    context = _context(FakeOrchestrator())
    with pytest.raises(TerminalJobError) as raised:
        await handle_build_wiki(_job(**changes), FakeLease(), context)
    assert raised.value.error_code == "rag_job_binding_invalid"


@pytest.mark.asyncio
async def test_build_wiki_handler_maps_missing_run_terminal(monkeypatch):
    from jobs.handlers import TerminalJobError
    from rag import handler

    async def get_for_worker(*_args):
        return None

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(FakeOrchestrator()))
    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_run_not_found",
        "The RAG run was not found.",
    )


@pytest.mark.asyncio
async def test_build_wiki_handler_maps_missing_factory_to_exact_disabled_terminal(monkeypatch, caplog):
    from jobs.handlers import TerminalJobError
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with caplog.at_level(logging.INFO, logger="rag.handler"), pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(None))

    assert type(raised.value) is TerminalJobError
    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_disabled",
        "Server-side RAG is disabled.",
    )
    assert_telemetry_event(caplog, "rag_run_started", count=0)
    assert_telemetry_event(caplog, "rag_run_failed", count=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message", "retryable", "expected"),
    [
        ("rag_disabled", "private", False, "terminal"),
        ("rag_model_unavailable", "private provider URL", True, "retryable"),
        ("rag_internal_error", "postgresql://private", True, "retryable"),
        ("rag_invalid_plan", "private prompt", False, "terminal"),
        ("rag_invalid_draft", "private draft", False, "terminal"),
        ("rag_budget_exhausted", "private budget", False, "terminal"),
        ("rag_version_conflict", "private path", False, "terminal"),
    ],
)
async def test_build_wiki_handler_maps_only_allowlisted_rag_failures(
    monkeypatch,
    code,
    message,
    retryable,
    expected,
):
    from jobs.handlers import RetryableJobError, TerminalJobError
    from rag import handler
    from rag.orchestrator import RagRunFailure

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    orchestrator = FakeOrchestrator(failure=RagRunFailure(code, message, retryable))
    error_type = RetryableJobError if expected == "retryable" else TerminalJobError
    with pytest.raises(error_type) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(orchestrator))
    assert raised.value.error_code == code
    assert "private" not in raised.value.error_message.lower()
    assert "postgresql" not in raised.value.error_message.lower()


@pytest.mark.asyncio
async def test_build_wiki_handler_maps_database_and_unknown_failures_retryable(monkeypatch):
    from jobs.handlers import RetryableJobError
    from rag import handler

    async def unavailable(*_args):
        raise asyncpg.ConnectionDoesNotExistError("postgresql://private.invalid")

    monkeypatch.setattr(handler.repository, "get_for_worker", unavailable)
    with pytest.raises(RetryableJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(FakeOrchestrator()))
    assert raised.value.error_code == "rag_internal_error"
    assert "postgresql" not in raised.value.error_message.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_kind",
    [
        "subclass-terminal-code-disguised-retryable",
        "exact-classification-mismatch",
        "unknown-code",
    ],
)
async def test_build_wiki_handler_rejects_hostile_or_mismatched_job_errors(monkeypatch, failure_kind):
    from jobs.handlers import RetryableJobError, TerminalJobError
    from rag import handler

    class HostileRetryable(RetryableJobError):
        pass

    failure = {
        "subclass-terminal-code-disguised-retryable": HostileRetryable("rag_disabled", "TOP_SECRET_TOKEN"),
        "exact-classification-mismatch": RetryableJobError("rag_disabled", "TOP_SECRET_TOKEN"),
        "unknown-code": TerminalJobError("rag_unknown_private", "TOP_SECRET_TOKEN"),
    }[failure_kind]

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(RetryableJobError) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )

    assert type(raised.value) is RetryableJobError
    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_internal_error",
        "The RAG request could not be completed.",
    )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET_TOKEN" not in repr(raised.value)


@pytest.mark.asyncio
async def test_build_wiki_handler_fails_closed_on_hostile_job_error_attributes(monkeypatch):
    from jobs.handlers import RetryableJobError
    from rag import handler

    class HostileRetryable(RetryableJobError):
        def __init__(self, *args):
            self._armed = False
            super().__init__(*args)
            self._armed = True

        def __getattribute__(self, name):
            armed = object.__getattribute__(self, "__dict__").get("_armed", False)
            if armed and name in {"__cause__", "__context__", "error_code", "error_message"}:
                raise RuntimeError("TOP_SECRET_ATTRIBUTE")
            return super().__getattribute__(name)

    async def get_for_worker(*_args):
        return _run()

    failure = HostileRetryable("rag_disabled", "TOP_SECRET_TOKEN")
    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(RetryableJobError) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )

    assert type(raised.value) is RetryableJobError
    assert raised.value.error_code == "rag_internal_error"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_build_wiki_handler_rebuilds_exact_job_error_without_exception_links(monkeypatch):
    from jobs.handlers import TerminalJobError
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    original = TerminalJobError("rag_disabled", "TOP_SECRET_TOKEN")
    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=original)),
        )

    assert type(raised.value) is TerminalJobError
    assert raised.value is not original
    assert (raised.value.error_code, raised.value.error_message) == (
        "rag_disabled",
        "Server-side RAG is disabled.",
    )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    "control",
    [asyncio.CancelledError("TOP_SECRET_TOKEN"), KeyboardInterrupt("TOP_SECRET_TOKEN")],
)
async def test_build_wiki_handler_prioritizes_nested_sanitized_controls(monkeypatch, grouped, control):
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    if grouped:
        failure = BaseExceptionGroup("TOP_SECRET_GROUP", [RuntimeError("private"), control])
    else:
        failure = RuntimeError("private")
        failure.__cause__ = control
    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(type(control)) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )

    assert type(raised.value) is type(control)
    assert str(raised.value) == ""
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_build_wiki_handler_failure_emits_no_failed_event_or_snapshot_reload(monkeypatch, caplog):
    from rag import handler
    from rag.orchestrator import RagRunFailure

    calls = []

    async def get_for_worker(*args):
        calls.append(args)
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    failure = RagRunFailure("rag_model_unavailable", "TOP_SECRET_TOKEN", True)
    with caplog.at_level(logging.INFO, logger="rag.handler"), pytest.raises(Exception):
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )

    assert len(calls) == 1
    assert_telemetry_event(caplog, "rag_run_started", count=1)
    assert_telemetry_event(caplog, "rag_run_failed", count=0)
    assert "TOP_SECRET_TOKEN" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError("private"), KeyboardInterrupt("private")])
async def test_build_wiki_handler_propagates_sanitized_process_controls(monkeypatch, failure):
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    with pytest.raises(type(failure)) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )
    assert "private" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["JobCancelled", "LeaseLost"])
async def test_build_wiki_handler_propagates_job_controls_unchanged(monkeypatch, name):
    from jobs import models
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    failure = getattr(models, name)("private")
    with pytest.raises(type(failure)) as raised:
        await handler.handle_build_wiki(
            _job(),
            FakeLease(),
            _context(FakeOrchestrator(failure=failure)),
        )
    assert raised.value is failure


@pytest.mark.asyncio
async def test_build_wiki_handler_rejects_unbounded_or_mismatched_result(monkeypatch):
    from jobs.handlers import TerminalJobError
    from rag import handler

    async def get_for_worker(*_args):
        return _run()

    monkeypatch.setattr(handler.repository, "get_for_worker", get_for_worker)
    result = {"run_id": str(uuid4()), "completion_reason": "completed", "pages_committed": 1}
    with pytest.raises(TerminalJobError) as raised:
        await handler.handle_build_wiki(_job(), FakeLease(), _context(FakeOrchestrator(result=result)))
    assert raised.value.error_code == "rag_job_binding_invalid"


@pytest.mark.parametrize("reported_length", [3, 250_000])
def test_bounded_result_rejects_hostile_mapping_without_len_or_iteration(reported_length):
    from jobs.handlers import TerminalJobError
    from rag import handler

    value = HostileResultMapping(reported_length=reported_length)

    with pytest.raises(TerminalJobError) as raised:
        handler._bounded_result(value, _run())

    assert raised.value.error_code == "rag_job_binding_invalid"
    assert value.len_calls == 0
    assert value.iter_calls == 0
    assert value.getitem_calls == 0
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


def test_bounded_result_rejects_dict_subclass_without_iteration():
    from jobs.handlers import TerminalJobError
    from rag import handler

    value = HostileResultDict(
        {
            "run_id": str(RUN_ID),
            "completion_reason": "completed",
            "pages_committed": 2,
        }
    )

    with pytest.raises(TerminalJobError) as raised:
        handler._bounded_result(value, _run())

    assert raised.value.error_code == "rag_job_binding_invalid"
    assert value.iter_calls == 0


def test_bounded_result_rejects_unknown_exact_dict_fields():
    from jobs.handlers import TerminalJobError
    from rag import handler

    private_field = {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
        "private_goal": "TOP_SECRET_GOAL",
    }

    with pytest.raises(TerminalJobError) as raised:
        handler._bounded_result(private_field, _run())
    assert raised.value.error_code == "rag_job_binding_invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


def test_bounded_result_accepts_completed_replay_where_skipped_overlaps_committed_total():
    from rag import handler

    from llmwiki_core.rag import RagBudget

    run = _run(budget=RagBudget(max_pages=2))
    result = {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
        "pages_skipped": 1,
    }

    assert handler._bounded_result(result, run) == result


@pytest.mark.parametrize(
    "result",
    [
        {
            "run_id": str(RUN_ID),
            "completion_reason": "dry_run",
            "pages_committed": 0,
            "pages_skipped": 2,
            "pages_dry_run": 1,
        },
        {
            "run_id": str(RUN_ID),
            "completion_reason": "completed",
            "pages_committed": 1,
            "pages_skipped": 2,
        },
        {
            "run_id": str(RUN_ID),
            "completion_reason": "completed",
            "pages_committed": 1,
            "pages_dry_run": 1,
        },
        {
            "run_id": str(RUN_ID),
            "completion_reason": "dry_run",
            "pages_committed": 1,
            "pages_dry_run": 1,
        },
        {
            "run_id": str(RUN_ID),
            "completion_reason": "no_work",
            "pages_committed": 1,
        },
    ],
)
def test_bounded_result_rejects_impossible_completion_counter_relationships(result):
    from jobs.handlers import TerminalJobError
    from rag import handler

    with pytest.raises(TerminalJobError) as raised:
        handler._bounded_result(result, _run())

    assert raised.value.error_code == "rag_job_binding_invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("field", ["run_id", "completion_reason"])
def test_bounded_result_rejects_hostile_text_values_without_comparison(field):
    from jobs.handlers import TerminalJobError
    from rag import handler

    class HostileText(str):
        comparisons = 0

        def __eq__(self, _other):
            type(self).comparisons += 1
            raise RuntimeError("TOP_SECRET_VALUE_COMPARISON")

        __hash__ = str.__hash__

    value = {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
    }
    value[field] = HostileText(value[field])

    with pytest.raises(TerminalJobError) as raised:
        handler._bounded_result(value, _run())

    assert raised.value.error_code == "rag_job_binding_invalid"
    assert HostileText.comparisons == 0
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_scoped_orchestrator_closes_model_when_composition_fails(monkeypatch):
    from rag import handler
    from rag.orchestrator import RagRunFailure

    model = FakeOwnedModel()
    orchestrator, lease = _scoped_orchestrator(monkeypatch, model)
    failure = RagRunFailure("rag_invalid_plan", "The generated plan was invalid.", False)

    def fail_store(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(handler, "_PostgresRunStore", fail_store)

    with pytest.raises(RagRunFailure) as raised:
        await orchestrator.run(_run(), lease)

    assert raised.value is failure
    assert model.close_calls == 1


@pytest.mark.asyncio
async def test_scoped_orchestrator_retries_one_ordinary_close_failure(monkeypatch):
    model = FakeOwnedModel(RuntimeError("TOP_SECRET_FIRST_CLOSE"), None)
    result = {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
    }
    orchestrator, lease = _scoped_orchestrator(monkeypatch, model, operation_result=result)

    assert await orchestrator.run(_run(), lease) is result
    assert model.close_calls == 2


@pytest.mark.asyncio
async def test_scoped_orchestrator_maps_two_close_failures_to_fixed_internal(monkeypatch):
    from rag.orchestrator import RagRunFailure

    model = FakeOwnedModel(
        RuntimeError("TOP_SECRET_FIRST_CLOSE"),
        RuntimeError("TOP_SECRET_SECOND_CLOSE"),
    )
    orchestrator, lease = _scoped_orchestrator(monkeypatch, model)

    with pytest.raises(RagRunFailure) as raised:
        await orchestrator.run(_run(), lease)

    assert (
        raised.value.code,
        raised.value.public_message,
        raised.value.retryable,
    ) == (
        "rag_internal_error",
        "The RAG request could not be completed.",
        True,
    )
    assert model.close_calls == 2
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_scoped_orchestrator_preserves_canonical_operation_failure_after_close_retries(monkeypatch):
    from rag.orchestrator import RagRunFailure

    operation_failure = RagRunFailure("rag_invalid_plan", "The generated plan was invalid.", False)
    model = FakeOwnedModel(
        RuntimeError("TOP_SECRET_FIRST_CLOSE"),
        RuntimeError("TOP_SECRET_SECOND_CLOSE"),
    )
    orchestrator, lease = _scoped_orchestrator(
        monkeypatch,
        model,
        operation_failure=operation_failure,
    )

    with pytest.raises(RagRunFailure) as raised:
        await orchestrator.run(_run(), lease)

    assert raised.value is operation_failure
    assert model.close_calls == 2
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "TOP_SECRET" not in repr(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["operation", "close"])
@pytest.mark.parametrize(
    "control",
    [KeyboardInterrupt("TOP_SECRET_CONTROL"), asyncio.CancelledError("TOP_SECRET_CONTROL")],
)
async def test_scoped_orchestrator_sanitizes_operation_and_close_controls(
    monkeypatch,
    boundary,
    control,
):
    operation_failure = control if boundary == "operation" else None
    model = FakeOwnedModel(control) if boundary == "close" else FakeOwnedModel()
    orchestrator, lease = _scoped_orchestrator(
        monkeypatch,
        model,
        operation_failure=operation_failure,
    )

    with pytest.raises(type(control)) as raised:
        await orchestrator.run(_run(), lease)

    assert type(raised.value) is type(control)
    assert raised.value is not control
    assert str(raised.value) == ""
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert model.close_calls == 1


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("rag_disabled", "Server-side RAG is disabled."),
        ("rag_invalid_plan", "The generated plan was invalid."),
        ("rag_invalid_draft", "The generated draft was invalid."),
        ("rag_budget_exhausted", "The RAG budget was exhausted."),
        ("rag_version_conflict", "The conflict retry limit was exhausted."),
    ],
)
def test_public_job_serialization_allows_only_terminal_rag_messages(code, message):
    from jobs.models import JobState, serialize_public_job

    public = serialize_public_job(replace(_job(), state=JobState.FAILED, error_code=code, error_message="private"))
    assert public["error"] == {"code": code, "message": message}


@pytest.mark.parametrize("code", ["rag_model_unavailable", "rag_retrieval_failed", "rag_internal_error"])
def test_retryable_and_internal_rag_messages_remain_generic(code):
    from jobs.models import JobState, serialize_public_job

    public = serialize_public_job(replace(_job(), state=JobState.FAILED, error_code=code, error_message="private"))
    assert public["error"] == {"code": "internal_error", "message": "The job could not be completed."}
