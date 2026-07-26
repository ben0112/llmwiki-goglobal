"""Public state-machine tests for durable RAG planning orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import UUID, uuid4

import pytest
from jobs.models import JobCancelled, JobRecord, JobState, JobType, LeaseLost
from rag.model import InvalidRagModelResponse, RagModelResponse, RagModelUnavailable, RagTokenUsage
from rag.orchestrator import (
    PLANNER_REPAIR_PROMPT_DIGEST,
    PLANNER_REPAIR_PROMPT_VERSION,
    BuildWikiOrchestrator,
    RagRunFailure,
)
from rag.ports import (
    AtomicPageCommit,
    AuthoritativeRagRun,
    OrchestratorPorts,
    PageExecutionResult,
    PlanAcceptance,
    PlanAttempt,
    PlanAttemptBinding,
    PlanAttemptSpec,
    RunStore,
    StructuredRagModel,
    WikiCatalogItem,
)
from rag.prompts import PLANNER_PROMPT_DIGEST, PLANNER_PROMPT_VERSION
from rag.records import RagPageRecord, RagRunRecord, RagStepRecord

from llmwiki_adapters.postgres.wiki import WikiWriteResult
from llmwiki_core.rag import (
    RagBudget,
    RagCompletionReason,
    RagPageState,
    RagStepStatus,
    RagStepType,
    RagUsage,
)
from llmwiki_core.search import SearchResult

RUN_ID = UUID("00000000-0000-0000-0000-000000000901")
JOB_ID = UUID("00000000-0000-0000-0000-000000000902")
USER_ID = UUID("00000000-0000-0000-0000-000000000903")
KB_ID = UUID("00000000-0000-0000-0000-000000000904")
NOW = datetime(2026, 7, 27, tzinfo=UTC)
VALID_PLAN = {
    "pages": [
        {
            "path": "/wiki/launch/overview.md",
            "intent": "Launch overview",
            "query": "launch overview",
        },
        {
            "path": "/wiki/launch/risks.md",
            "intent": "Launch risks",
            "query": "launch risks",
        },
    ]
}


def _run(
    *,
    parent: bool = False,
    dry_run: bool = False,
    budget: RagBudget | None = None,
    usage: RagUsage = RagUsage(),
    last_committed_ordinal: int = -1,
) -> RagRunRecord:
    return RagRunRecord(
        id=RUN_ID,
        job_id=JOB_ID,
        root_run_id=UUID(int=800) if parent else RUN_ID,
        parent_run_id=UUID(int=801) if parent else None,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        goal="Build a launch wiki",
        goal_digest=hashlib.sha256(b"Build a launch wiki").hexdigest(),
        target_path_prefix="/wiki/launch/",
        model_profile="primary",
        model_profile_version="primary-v1",
        retrieval_profile="lexical",
        dry_run=dry_run,
        budget=budget or RagBudget(),
        usage=usage,
        idempotency_key="rag-run-1",
        request_digest="b" * 64,
        completion_reason=None,
        last_committed_ordinal=last_committed_ordinal,
        created_at=NOW,
        updated_at=NOW,
    )


def _job(*, state: JobState = JobState.RUNNING) -> JobRecord:
    return JobRecord(
        id=JOB_ID,
        job_type=JobType.BUILD_WIKI,
        user_id=USER_ID,
        state=state,
        knowledge_base_id=KB_ID,
        payload={"run_id": str(RUN_ID)},
        idempotency_key="job-1",
        lease_owner="worker-1",
        lease_expires_at=NOW,
    )


def _page(
    ordinal: int,
    path: str,
    *,
    state: RagPageState = RagPageState.PLANNED,
) -> RagPageRecord:
    preview = f"preview {ordinal}" if state is RagPageState.DRY_RUN_COMPLETE else None
    return RagPageRecord(
        id=UUID(int=1_000 + ordinal),
        run_id=RUN_ID,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        ordinal=ordinal,
        path=path,
        intent=f"intent {ordinal}",
        query=f"query {ordinal}",
        state=state,
        document_id=UUID(int=2_000 + ordinal) if state is RagPageState.COMMITTED else None,
        version_read=None,
        version_committed=1 if state is RagPageState.COMMITTED else None,
        attempt_count=1 if state is RagPageState.DRY_RUN_COMPLETE else 0,
        conflict_retry_count=0,
        last_completed_step_sequence=0,
        preview=preview,
        preview_digest=hashlib.sha256(preview.encode()).hexdigest() if preview is not None else None,
        preview_full_char_count=len(preview) if preview is not None else None,
        preview_truncated=False,
        lint_summary=MappingProxyType({}) if state is RagPageState.COMMITTED else None,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeStore:
    def __init__(self, run: RagRunRecord, pages: tuple[RagPageRecord, ...] = ()) -> None:
        self.run = run
        self.job = _job()
        self.pages = pages
        self.events: list[tuple[str, object]] = []
        self.steps: list[RagStepRecord] = []
        self.running_step: RagStepRecord | None = None
        self.running_spec: PlanAttemptSpec | None = None
        self.invalid_specs: set[bool] = set()
        self.fatal_invalid_usage = False
        self.crash_after_accept = False
        self.crash_after_finish = False
        self.crash_after_finish_run = False
        self.crash_after_bind = False
        self.reload_failure: BaseException | None = None
        self.fail_accept_fence = False
        self.fail_finish_fence = False
        self.fail_finish_run_fence = False

    async def reload(self, run_id: UUID) -> AuthoritativeRagRun:
        self.events.append(("reload", run_id))
        if self.reload_failure is not None:
            raise self.reload_failure
        return AuthoritativeRagRun(self.run, self.job)

    async def list_pages(self, run_id: UUID) -> tuple[RagPageRecord, ...]:
        self.events.append(("list_pages", run_id))
        return self.pages

    def _assert_fence(self, kwargs: dict[str, object]) -> None:
        assert kwargs["job"] == self.job
        assert kwargs["lease_owner"] == self.job.lease_owner

    def _terminalize_running(self, status: RagStepStatus, token_usage: RagTokenUsage, error_code: str | None) -> None:
        assert self.running_step is not None
        terminal = replace(
            self.running_step,
            status=status,
            prompt_version=self.running_spec.prompt_version if self.running_spec else None,
            prompt_digest=self.running_spec.prompt_digest if self.running_spec else None,
            input_tokens=token_usage.prompt_tokens,
            output_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            error_code=error_code,
        )
        self.steps = [terminal if step.id == terminal.id else step for step in self.steps]

    async def begin_plan_attempt(self, **kwargs: object) -> PlanAttempt:
        self._assert_fence(kwargs)
        specs = kwargs["specs"]
        assert type(specs) is tuple and len(specs) == 2
        if self.running_step is not None:
            self.events.append(("begin_plan_attempt", dict(kwargs)))
            assert self.running_spec is not None and self.running_spec in specs
            return PlanAttempt(self.run, self.running_step, self.running_spec, True)
        if self.fatal_invalid_usage:
            from llmwiki_core.rag import RagDomainError

            raise RagDomainError("rag_invalid_plan", "private invalid provider usage")
        minimum_output = min(4_096, max(256, self.run.budget.max_pages * 256))
        self.run.usage.consume_step(self.run.budget).reserve_model_call(self.run.budget, minimum_output)
        self.events.append(("begin_plan_attempt", dict(kwargs)))
        if True in self.invalid_specs:
            from llmwiki_core.rag import RagDomainError

            raise RagDomainError("rag_invalid_plan", "private prior invalid plan")
        spec = specs[1] if False in self.invalid_specs else specs[0]
        sequence = len(self.steps) + 1
        step = RagStepRecord(
            id=uuid4(),
            run_id=RUN_ID,
            run_page_id=None,
            user_id=USER_ID,
            knowledge_base_id=KB_ID,
            sequence=sequence,
            step_type=RagStepType.PLAN,
            status=RagStepStatus.RUNNING,
            input_digest=spec.input_digest,
            output_summary=MappingProxyType({}),
            citation_identities=(),
            prompt_version=spec.prompt_version,
            prompt_digest=spec.prompt_digest,
            model_profile_version="primary-v1",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            error_code=None,
            error_message=None,
            created_at=NOW,
            updated_at=NOW,
        )
        self.steps.append(step)
        self.running_step = step
        self.running_spec = spec
        return PlanAttempt(self.run, step, spec, False)

    async def bind_plan_attempt_input(self, **kwargs: object) -> PlanAttemptBinding:
        self._assert_fence(kwargs)
        self.events.append(("bind_plan_attempt_input", dict(kwargs)))
        assert self.running_step is not None and self.running_spec is not None
        input_digest = kwargs["input_digest"]
        assert type(input_digest) is str
        if self.running_step.input_digest in {self.running_spec.input_digest, input_digest}:
            self.running_step = replace(self.running_step, input_digest=input_digest)
            self.steps = [self.running_step if step.id == self.running_step.id else step for step in self.steps]
            if self.crash_after_bind:
                self.crash_after_bind = False
                raise asyncio.CancelledError("crash after atomic input bind")
            attempt = PlanAttempt(self.run, self.running_step, self.running_spec, True)
            return PlanAttemptBinding(self.run, attempt, False)

        zero = RagTokenUsage(0, 0, 0)
        self._terminalize_running(RagStepStatus.FAILED, zero, "rag_plan_input_changed")
        self.run = replace(self.run, usage=self.run.usage.consume_step(self.run.budget))
        prior = self.running_step
        self.running_step = None
        if self.run.usage.steps >= self.run.budget.max_steps or prior.sequence >= self.run.budget.max_steps:
            return PlanAttemptBinding(self.run, None, True)
        sequence = prior.sequence + 1
        self.running_step = replace(
            prior,
            id=uuid4(),
            sequence=sequence,
            input_digest=input_digest,
            created_at=NOW,
            updated_at=NOW,
        )
        self.steps.append(self.running_step)
        attempt = PlanAttempt(self.run, self.running_step, self.running_spec, False)
        return PlanAttemptBinding(self.run, attempt, False)

    async def finish_plan_attempt(self, **kwargs: object) -> RagRunRecord:
        self._assert_fence(kwargs)
        if self.fail_finish_fence:
            self.fail_finish_fence = False
            raise LeaseLost("lost at failed-step fence")
        self.events.append(("finish_plan_attempt", dict(kwargs)))
        assert self.running_step is not None and kwargs["step"].id == self.running_step.id
        token_usage = kwargs["token_usage"]
        assert type(token_usage) is RagTokenUsage
        aggregate_usage = kwargs["aggregate_usage"]
        summary = kwargs["summary"]
        assert aggregate_usage.model_tokens == self.run.usage.model_tokens + summary["charged_tokens"]
        assert summary["charged_tokens"] == (
            token_usage.total_tokens if summary["usage_trusted"] else summary["charged_tokens"]
        )
        self.run = replace(self.run, usage=aggregate_usage)
        if kwargs["error_code"] == "rag_invalid_model_usage" and not summary["usage_trusted"]:
            self.fatal_invalid_usage = True
        elif kwargs["error_code"] in {"rag_invalid_model_usage", "rag_invalid_plan"}:
            assert self.running_spec is not None
            self.invalid_specs.add(self.running_spec.repair)
        self._terminalize_running(RagStepStatus.FAILED, token_usage, kwargs["error_code"])
        self.running_step = None
        self.running_spec = None
        if self.crash_after_finish:
            self.crash_after_finish = False
            raise asyncio.CancelledError("crash after atomic failed step")
        return self.run

    async def accept_plan(self, **kwargs: object) -> PlanAcceptance:
        self._assert_fence(kwargs)
        if self.fail_accept_fence:
            self.fail_accept_fence = False
            raise LeaseLost("lost at accept fence")
        self.events.append(("accept_plan", dict(kwargs)))
        assert self.running_step is not None and kwargs["step"].id == self.running_step.id
        token_usage = kwargs["token_usage"]
        aggregate_usage = kwargs["aggregate_usage"]
        assert type(token_usage) is RagTokenUsage
        assert aggregate_usage.model_tokens == self.run.usage.model_tokens + token_usage.total_tokens
        self.run = replace(
            self.run,
            usage=aggregate_usage,
            completion_reason=kwargs["completion_reason"],
        )
        self.pages = tuple(
            replace(
                _page(item.ordinal, item.path),
                intent=item.intent,
                query=item.query,
            )
            for item in kwargs["items"]  # type: ignore[union-attr]
        )
        self._terminalize_running(RagStepStatus.SUCCEEDED, token_usage, None)
        self.running_step = None
        self.running_spec = None
        accepted = PlanAcceptance(self.run, self.pages, kwargs["completion_reason"])
        if self.crash_after_accept:
            self.crash_after_accept = False
            raise asyncio.CancelledError("crash after atomic accept")
        return accepted

    async def finish_run(self, **kwargs: object) -> RagRunRecord:
        self._assert_fence(kwargs)
        if self.fail_finish_run_fence:
            self.fail_finish_run_fence = False
            raise LeaseLost("lost at final fence")
        self.events.append(("finish_run", dict(kwargs)))
        self.run = replace(self.run, completion_reason=kwargs["completion_reason"])
        if self.crash_after_finish_run:
            self.crash_after_finish_run = False
            raise asyncio.CancelledError("crash after atomic run completion")
        return self.run


class FakeLease:
    def __init__(self, *, fail_at: int | None = None, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.failure = failure

    async def checkpoint(self) -> JobRecord:
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.failure or LeaseLost("private lease detail")
        return _job()


class FakeFeatureGate:
    def __init__(self, *, fail_at: int | None = None, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.failure = failure

    async def ensure_enabled(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        if self.calls == self.fail_at:
            from llmwiki_core.rag import RagDomainError

            raise RagDomainError("rag_disabled", "secret disabled detail")


class FakeModel:
    def __init__(
        self,
        *payloads: object,
        failure: BaseException | None = None,
        usage: RagTokenUsage | None = None,
        usages: tuple[RagTokenUsage, ...] | None = None,
    ) -> None:
        self.payloads = list(payloads or (VALID_PLAN,))
        self.failure = failure
        self.usage = usage or RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        self.usages = list(usages) if usages is not None else None
        self.calls: list[dict[str, object]] = []
        self.events: list[tuple[str, object]] | None = None

    async def complete_json(self, **kwargs: object) -> RagModelResponse:
        self.calls.append(dict(kwargs))
        if self.events is not None:
            self.events.append(("model", len(self.calls)))
        if self.failure is not None:
            raise self.failure
        payload = self.payloads.pop(0)
        usage = self.usages.pop(0) if self.usages is not None else self.usage
        return RagModelResponse(
            payload=payload,  # type: ignore[arg-type]
            usage=usage,
        )


class FakeRetrieval:
    def __init__(self) -> None:
        self.calls = 0
        self.failure: BaseException | None = None

    async def retrieve(self, query: object, *, profile: str) -> SearchResult:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return SearchResult(hits=(), candidate_count=0, profile=profile)


class FakeEvidenceReader:
    def __init__(self) -> None:
        self.calls = 0

    async def read(self, *args: object) -> tuple[()]:
        self.calls += 1
        return ()


class FakeCatalog:
    def __init__(self) -> None:
        self.calls = 0
        self.items = (WikiCatalogItem("/wiki/existing.md", "Existing"),)

    async def list_for_planning(self, **kwargs: object) -> tuple[WikiCatalogItem, ...]:
        self.calls += 1
        return self.items


class UnusedPort:
    async def get_by_path(self, *args: object) -> None:
        return None

    async def commit(self, **kwargs: object) -> object:
        raise AssertionError("unused")

    async def lint(self, **kwargs: object) -> MappingProxyType:
        return MappingProxyType({})


class FakePageRunner:
    def __init__(self, store: FakeStore) -> None:
        self.store = store
        self.calls: list[int] = []
        self.failure: BaseException | None = None

    async def run(
        self,
        run: RagRunRecord,
        page: RagPageRecord,
        lease: object,
    ) -> PageExecutionResult:
        del lease
        self.calls.append(page.ordinal)
        if self.failure is not None:
            raise self.failure
        state = RagPageState.DRY_RUN_COMPLETE if run.dry_run else RagPageState.COMMITTED
        preview = f"preview {page.ordinal}" if run.dry_run else None
        committed_version = None if run.dry_run else (page.version_read + 1 if page.version_read is not None else 1)
        updated_page = replace(
            page,
            state=state,
            document_id=None if run.dry_run else page.document_id or UUID(int=2_000 + page.ordinal),
            version_committed=committed_version,
            attempt_count=page.attempt_count + 1,
            preview=preview,
            preview_digest=hashlib.sha256(preview.encode()).hexdigest() if preview is not None else None,
            preview_full_char_count=len(preview) if preview is not None else None,
            lint_summary=None if run.dry_run else MappingProxyType({}),
        )
        updated_run = replace(
            run,
            usage=RagUsage(run.usage.steps + 1, run.usage.model_tokens),
            last_committed_ordinal=page.ordinal if not run.dry_run else run.last_committed_ordinal,
        )
        self.store.run = updated_run
        self.store.pages = tuple(updated_page if item.id == page.id else item for item in self.store.pages)
        return PageExecutionResult(updated_run, updated_page)


def _ports(
    *,
    run: RagRunRecord | None = None,
    pages: tuple[RagPageRecord, ...] = (),
    model: FakeModel | None = None,
    feature_gate: object | None = None,
) -> tuple[OrchestratorPorts, FakeStore, FakeModel, FakePageRunner, FakeRetrieval, FakeCatalog]:
    store = FakeStore(run or _run(), pages)
    selected_model = model or FakeModel()
    selected_model.events = store.events
    runner = FakePageRunner(store)
    retrieval = FakeRetrieval()
    catalog = FakeCatalog()
    unused = UnusedPort()
    return (
        OrchestratorPorts(
            store=store,
            model=selected_model,
            retrieval=retrieval,
            evidence_reader=FakeEvidenceReader(),
            wiki_catalog=catalog,
            wiki_page_reader=unused,
            wiki_writer=unused,
            draft_linter=unused,
            feature_gate=feature_gate or FakeFeatureGate(),
            page_runner=runner,
        ),
        store,
        selected_model,
        runner,
        retrieval,
        catalog,
    )


@pytest.mark.asyncio
async def test_root_plan_is_persisted_before_model_and_accepted_atomically():
    ports, store, model, runner, _, _ = _ports()
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "completed",
        "pages_committed": 2,
    }
    assert runner.calls == [0, 1]
    names = [event[0] for event in store.events]
    assert (
        names.index("begin_plan_attempt")
        < names.index("bind_plan_attempt_input")
        < names.index("model")
        < names.index("accept_plan")
    )
    assert [event[0] for event in store.events].count("accept_plan") == 1
    assert [event[0] for event in store.events].count("finish_plan_attempt") == 0
    accepted = next(value for name, value in store.events if name == "accept_plan")
    assert accepted["summary"] == {"accepted_pages": 2, "repair": False}
    assert accepted["token_usage"] == RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    assert accepted["aggregate_usage"] == RagUsage(steps=1, model_tokens=14)
    assert (store.steps[0].input_tokens, store.steps[0].output_tokens, store.steps[0].total_tokens) == (10, 4, 14)
    bound = next(value for name, value in store.events if name == "bind_plan_attempt_input")
    canonical = {
        "messages": [dict(message) for message in model.calls[0]["messages"]],
        "planner": {"prompt_digest": PLANNER_PROMPT_DIGEST, "prompt_version": PLANNER_PROMPT_VERSION},
        "run_scope": {
            "goal_digest": store.run.goal_digest,
            "knowledge_base_id": str(store.run.knowledge_base_id),
            "model_profile_version": store.run.model_profile_version,
            "retrieval_profile": store.run.retrieval_profile,
            "target_path_prefix": store.run.target_path_prefix,
        },
    }
    expected_digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert bound["input_digest"] == expected_digest
    assert store.steps[0].input_digest == expected_digest


@pytest.mark.asyncio
async def test_empty_root_plan_finishes_no_work_without_page_side_effects():
    ports, store, model, runner, _, _ = _ports(model=FakeModel({"pages": []}))
    result = await BuildWikiOrchestrator(ports).run(_run(), FakeLease())

    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "no_work",
        "pages_committed": 0,
    }
    assert len(model.calls) == 1
    assert runner.calls == []
    assert store.run.completion_reason is RagCompletionReason.NO_WORK
    assert [name for name, _ in store.events].count("accept_plan") == 1
    assert [name for name, _ in store.events].count("finish_run") == 0


@pytest.mark.asyncio
async def test_resume_uses_persisted_worklist_and_skips_committed_prefix_without_planner():
    pages = (
        _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
        _page(1, "/wiki/launch/risks.md"),
    )
    model = FakeModel(failure=AssertionError("planner must not execute"))
    ports, _, _, runner, retrieval, catalog = _ports(
        run=_run(parent=True, last_committed_ordinal=0),
        pages=pages,
        model=model,
    )
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["pages_skipped"] == 1
    assert result["pages_committed"] == 2
    assert runner.calls == [1]
    assert model.calls == []
    assert retrieval.calls == catalog.calls == 0


@pytest.mark.asyncio
async def test_invalid_plan_gets_one_versioned_repair_and_never_persists_raw_output():
    secret = "raw-secret-plan-body"
    ports, store, model, _, _, _ = _ports(model=FakeModel({"bad": secret}, VALID_PLAN))
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "completed"
    assert len(model.calls) == 2
    failed = [value for name, value in store.events if name == "finish_plan_attempt"]
    assert len(failed) == 1
    assert failed[0]["prompt_version"] == PLANNER_PROMPT_VERSION
    assert failed[0]["token_usage"] == RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    assert failed[0]["summary"] == {
        "accepted_pages": 0,
        "charged_tokens": 14,
        "outcome": "failed",
        "usage_trusted": True,
    }
    accepted = next(value for name, value in store.events if name == "accept_plan")
    assert accepted["prompt_version"] == PLANNER_REPAIR_PROMPT_VERSION
    assert accepted["token_usage"] == RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    assert accepted["aggregate_usage"] == RagUsage(steps=2, model_tokens=28)
    assert secret not in repr(store.events)


@pytest.mark.asyncio
async def test_second_invalid_plan_is_terminal_and_both_attempts_are_failed():
    ports, store, _, runner, _, _ = _ports(model=FakeModel({"bad": "first"}, {"bad": "second"}))
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value == RagRunFailure("rag_invalid_plan", "The generated plan was invalid.", False)
    assert [name for name, _ in store.events].count("finish_plan_attempt") == 2
    assert runner.calls == []
    assert store.run.completion_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        _run(budget=RagBudget(max_steps=1), usage=RagUsage(steps=1)),
        _run(budget=RagBudget(max_model_tokens=255)),
    ],
)
async def test_step_and_token_budgets_fail_before_planning_external_calls(run: RagRunRecord):
    ports, store, model, _, retrieval, catalog = _ports(run=run)
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value.code == "rag_budget_exhausted"
    assert model.calls == []
    assert retrieval.calls == catalog.calls == 0
    assert not any(name == "begin_plan_attempt" for name, _ in store.events)


@pytest.mark.asyncio
async def test_feature_disable_is_checked_before_each_page():
    pages = (_page(0, "/wiki/launch/overview.md"),)
    gate = FakeFeatureGate(fail_at=2)
    ports, _, model, runner, _, _ = _ports(pages=pages, feature_gate=gate)
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value == RagRunFailure("rag_disabled", "Server-side RAG is disabled.", False)
    assert runner.calls == []
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "page"),
    [
        (_run(), _page(0, "/wiki/launch/overview.md", state=RagPageState.DRY_RUN_COMPLETE)),
        (
            _run(dry_run=True, last_committed_ordinal=0),
            _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
        ),
    ],
)
async def test_authoritative_pages_reject_mode_mismatch_before_page_runner(run, page):
    ports, _, model, runner, _, _ = _ports(run=run, pages=(page,))
    lease = FakeLease()
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, lease)
    assert raised.value.code == "rag_internal_error"
    assert model.calls == []
    assert runner.calls == []
    assert lease.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (asyncio.CancelledError("private"), asyncio.CancelledError),
        (KeyboardInterrupt("private"), KeyboardInterrupt),
        (SystemExit("private"), SystemExit),
        (GeneratorExit("private"), GeneratorExit),
        (JobCancelled("private"), JobCancelled),
        (LeaseLost("private"), LeaseLost),
    ],
)
async def test_control_and_durable_lease_signals_propagate_sanitized(
    failure: BaseException,
    expected: type[BaseException],
):
    ports, *_ = _ports(pages=(_page(0, "/wiki/launch/overview.md"),))
    with pytest.raises(expected) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease(fail_at=1, failure=failure))
    if expected not in {JobCancelled, LeaseLost}:
        assert not raised.value.args or "private" not in repr(raised.value.args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_code", "expected"),
    [
        (7, 7),
        (True, 1),
        (False, 0),
        ("private-system-exit-code", 1),
        (object(), 1),
    ],
)
async def test_system_exit_preserves_only_the_sanitized_integer_code_in_public_traceback(raw_code, expected):
    source = SystemExit(raw_code)
    ports, *_ = _ports(model=FakeModel(failure=source))

    with pytest.raises(SystemExit) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert type(raised.value.code) is int
    assert raised.value.code == expected
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    traceback = raised.value.__traceback__
    while traceback is not None:
        if "/api/rag/" in traceback.tb_frame.f_code.co_filename:
            assert all(value is not source for value in traceback.tb_frame.f_locals.values())
            if type(raw_code) not in {int, bool}:
                assert all(value is not raw_code for value in traceback.tb_frame.f_locals.values())
            rendered = "\n".join(repr(value) for value in traceback.tb_frame.f_locals.values())
            assert "private-system-exit-code" not in rendered
        traceback = traceback.tb_next


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", range(1, 9))
async def test_cancellation_at_each_planning_checkpoint_prevents_acceptance(fail_at: int):
    ports, store, *_ = _ports()
    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(
            RUN_ID,
            FakeLease(fail_at=fail_at, failure=asyncio.CancelledError("private")),
        )
    assert not any(name == "accept_plan" for name, _ in store.events)
    assert store.run.completion_reason is None


@pytest.mark.asyncio
async def test_lease_loss_at_page_checkpoint_does_not_invoke_page_runner():
    ports, _, _, runner, _, _ = _ports(pages=(_page(0, "/wiki/launch/overview.md"),))
    with pytest.raises(LeaseLost):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease(fail_at=2))
    assert runner.calls == []


@pytest.mark.asyncio
async def test_linked_cancellation_in_hostile_port_failure_wins_control_first():
    failure = RuntimeError("provider secret")
    failure.__cause__ = asyncio.CancelledError("cancel secret")
    ports, *_ = _ports(model=FakeModel(failure=failure))
    with pytest.raises(asyncio.CancelledError) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.args == ()


@pytest.mark.asyncio
async def test_provider_unavailability_is_retryable_and_persisted_failure_is_sanitized():
    ports, store, _, _, _, _ = _ports(model=FakeModel(failure=RagModelUnavailable()))
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value == RagRunFailure(
        "rag_model_unavailable",
        "The RAG model is temporarily unavailable.",
        True,
    )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    failure = next(value for name, value in store.events if name == "finish_plan_attempt")
    assert failure["error_code"] == "rag_model_unavailable"
    assert failure["token_usage"] == RagTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert "secret" not in repr(failure)


@pytest.mark.asyncio
async def test_falsified_usage_charges_reservation_and_is_terminal_without_repair():
    invalid = RagTokenUsage(prompt_tokens=700, completion_tokens=1, total_tokens=701)
    plan = {"pages": [VALID_PLAN["pages"][0]]}
    model = FakeModel(plan, usage=invalid)
    ports, store, *_ = _ports(
        run=_run(budget=RagBudget(max_pages=1, max_model_tokens=700)),
        model=model,
    )
    store.crash_after_finish = True

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert store.run.usage.steps == 1
    assert 0 < store.run.usage.model_tokens <= 700
    assert store.running_step is None
    assert len(model.calls) == 1

    with pytest.raises(RagRunFailure) as replayed:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert replayed.value == RagRunFailure("rag_invalid_plan", "The generated plan was invalid.", False)
    failed = next(value for name, value in store.events if name == "finish_plan_attempt")
    assert failed["error_code"] == "rag_invalid_model_usage"
    assert failed["token_usage"] == RagTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert failed["aggregate_usage"] == store.run.usage
    assert failed["summary"]["usage_trusted"] is False
    assert failed["summary"]["charged_tokens"] == store.run.usage.model_tokens
    assert not any(name == "accept_plan" for name, _ in store.events)
    assert store.running_step is None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_invalid_model_payload_with_trusted_usage_is_exactly_accounted():
    trusted = RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    ports, store, *_ = _ports(model=FakeModel(failure=InvalidRagModelResponse(trusted)))

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value.code == "rag_invalid_plan"
    failed = next(value for name, value in store.events if name == "finish_plan_attempt")
    assert failed["token_usage"] == trusted
    assert failed["aggregate_usage"] == RagUsage(steps=1, model_tokens=14)
    assert failed["summary"]["usage_trusted"] is True


@pytest.mark.asyncio
async def test_empty_plan_replays_after_multiple_durable_retryable_planner_failures():
    model = FakeModel(failure=RagModelUnavailable())
    ports, store, *_ = _ports(model=model)

    for _ in range(2):
        with pytest.raises(RagRunFailure) as retryable:
            await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
        assert retryable.value.retryable is True

    model.failure = None
    model.payloads = [{"pages": []}]
    store.crash_after_accept = True
    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert store.run.completion_reason is RagCompletionReason.NO_WORK
    assert store.run.usage == RagUsage(steps=3, model_tokens=14)
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "no_work",
        "pages_committed": 0,
    }
    assert store.run.usage == RagUsage(steps=3, model_tokens=14)
    assert len(model.calls) == 3


@pytest.mark.asyncio
async def test_dry_run_aggregates_page_results_with_exact_completion_reason():
    pages = (
        _page(0, "/wiki/launch/overview.md"),
        _page(1, "/wiki/launch/risks.md"),
    )
    ports, store, model, runner, _, _ = _ports(run=_run(dry_run=True), pages=pages)
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "dry_run",
        "pages_committed": 0,
        "pages_dry_run": 2,
    }
    assert model.calls == []
    assert runner.calls == [0, 1]
    assert store.run.completion_reason is RagCompletionReason.DRY_RUN


@pytest.mark.asyncio
async def test_supplied_run_is_only_an_identity_hint_and_authoritative_tamper_fails_closed():
    supplied = replace(_run(), goal="attacker controlled stale goal", job_id=uuid4())
    ports, _, model, _, _, _ = _ports()
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(supplied, FakeLease())
    assert raised.value.code == "rag_internal_error"
    assert model.calls == []
    assert "attacker" not in str(raised.value)


@pytest.mark.asyncio
async def test_hostile_port_result_and_exception_are_detached_to_internal_failure():
    class HostileStore(FakeStore):
        async def reload(self, run_id: UUID) -> object:
            del run_id
            return {"goal": "secret goal"}

    ports, _, _, _, _, _ = _ports()
    object.__setattr__(ports, "store", HostileStore(_run()))
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value == RagRunFailure(
        "rag_internal_error",
        "The RAG request could not be completed.",
        True,
    )
    assert "secret goal" not in str(raised.value)


@pytest.mark.asyncio
async def test_authoritative_direct_record_tamper_fails_before_any_model_or_page_call():
    ports, store, model, runner, _, _ = _ports()
    store.run = replace(store.run, goal_digest="f" * 64)
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.code == "rag_internal_error"
    assert model.calls == []
    assert runner.calls == []


@pytest.mark.asyncio
async def test_unknown_domain_error_uses_only_operation_allowlist():
    from llmwiki_core.rag import RagDomainError

    secret = "postgresql://user:private@db.invalid/secret"
    ports, store, *_ = _ports(model=FakeModel(failure=RagDomainError("attacker_code", secret, True)))
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.code == "rag_internal_error"
    assert secret not in str(raised.value)
    persisted = next(value for name, value in store.events if name == "finish_plan_attempt")
    assert persisted["error_code"] == "rag_internal_error"
    assert secret not in repr(persisted)


@pytest.mark.asyncio
async def test_hostile_port_cannot_inject_a_public_rag_run_failure():
    secret = "attacker supplied public message"
    ports, *_ = _ports(model=FakeModel(failure=RagRunFailure("rag_attacker", secret, False)))
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.code == "rag_internal_error"
    assert secret not in str(raised.value)


@pytest.mark.asyncio
async def test_running_plan_step_is_idempotently_recovered_across_repeated_lease_losses():
    model = FakeModel(VALID_PLAN, VALID_PLAN, VALID_PLAN)
    ports, store, _, _, _, _ = _ports(model=model)

    for _ in range(2):
        with pytest.raises(LeaseLost):
            await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease(fail_at=8))
        assert store.running_step is not None
        assert len(store.steps) == 1
        assert sum(step.status is RagStepStatus.RUNNING for step in store.steps) == 1
        assert store.run.usage == RagUsage()

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert result["completion_reason"] == "completed"
    assert len(store.steps) == 1
    assert store.running_step is None
    assert sum(step.status is RagStepStatus.RUNNING for step in store.steps) == 0


@pytest.mark.asyncio
async def test_begin_plan_attempt_cannot_commit_usage_before_the_terminal_boundary():
    class UsageMutatingStore(FakeStore):
        async def begin_plan_attempt(self, **kwargs: object) -> PlanAttempt:
            attempt = await super().begin_plan_attempt(**kwargs)
            mutated = replace(attempt.run, usage=attempt.run.usage.consume_step(attempt.run.budget))
            return PlanAttempt(mutated, attempt.step, attempt.spec, attempt.resumed)

    ports, _, model, runner, retrieval, catalog = _ports()
    hostile = UsageMutatingStore(_run())
    object.__setattr__(ports, "store", hostile)
    lease = FakeLease()

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, lease)

    assert raised.value.code == "rag_internal_error"
    assert lease.calls == 2
    assert model.calls == []
    assert runner.calls == []
    assert retrieval.calls == catalog.calls == 0


@pytest.mark.asyncio
async def test_bound_input_drift_fails_old_step_and_never_reuses_it_for_new_messages():
    ports, store, model, _, _, catalog = _ports()
    store.crash_after_bind = True

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    first = store.running_step
    assert first is not None
    assert model.calls == []
    catalog.items = (WikiCatalogItem("/wiki/existing.md", "Changed title"),)

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "completed"
    plan_steps = [step for step in store.steps if step.step_type is RagStepType.PLAN]
    assert len(plan_steps) == 2
    assert plan_steps[0].status is RagStepStatus.FAILED
    assert plan_steps[0].error_code == "rag_plan_input_changed"
    assert plan_steps[0].input_digest != plan_steps[1].input_digest
    assert plan_steps[0].id != plan_steps[1].id
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_bound_input_drift_at_last_step_is_atomically_exhausted_without_replacement():
    ports, store, model, _, _, catalog = _ports(
        run=_run(budget=RagBudget(max_steps=1)),
        model=FakeModel({"pages": []}),
    )
    store.crash_after_bind = True

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    catalog.items = (WikiCatalogItem("/wiki/existing.md", "Changed title"),)
    with pytest.raises(RagRunFailure) as exhausted:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert exhausted.value == RagRunFailure(
        "rag_budget_exhausted",
        "The RAG budget was exhausted.",
        False,
    )
    assert store.run.usage == RagUsage(steps=1)
    assert store.running_step is None
    assert len(store.steps) == 1
    assert store.steps[0].status is RagStepStatus.FAILED
    assert store.steps[0].error_code == "rag_plan_input_changed"
    assert model.calls == []

    with pytest.raises(RagRunFailure) as replayed:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert replayed.value == exhausted.value
    assert store.running_step is None
    assert len(store.steps) == 1
    assert model.calls == []


@pytest.mark.asyncio
async def test_bound_input_drift_can_use_the_exact_last_replacement_step():
    ports, store, model, _, _, catalog = _ports(
        run=_run(budget=RagBudget(max_steps=2)),
        model=FakeModel({"pages": []}),
    )
    store.crash_after_bind = True

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    catalog.items = (WikiCatalogItem("/wiki/existing.md", "Changed title"),)
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "no_work"
    assert store.run.usage == RagUsage(steps=2, model_tokens=14)
    assert [step.status for step in store.steps] == [RagStepStatus.FAILED, RagStepStatus.SUCCEEDED]
    assert store.running_step is None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_retry_after_durable_invalid_step_starts_the_only_repair_not_a_new_initial_attempt():
    model = FakeModel({"bad": "invalid"}, VALID_PLAN)
    ports, store, *_ = _ports(model=model)
    store.crash_after_finish = True
    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert store.invalid_specs == {False}
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert result["completion_reason"] == "completed"
    begins = [value for name, value in store.events if name == "begin_plan_attempt"]
    assert len(begins) == 2
    assert store.steps[1].prompt_version == PLANNER_REPAIR_PROMPT_VERSION
    accepted = next(value for name, value in store.events if name == "accept_plan")
    assert accepted["prompt_version"] == PLANNER_REPAIR_PROMPT_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        FakeModel(VALID_PLAN),
        FakeModel({"bad": "invalid"}),
        FakeModel(failure=RagModelUnavailable()),
    ],
)
async def test_loss_after_model_outcome_never_persists_plan_effects(model: FakeModel):
    ports, store, *_ = _ports(model=model)
    with pytest.raises(LeaseLost):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease(fail_at=8))
    assert store.running_step is not None
    assert store.run.usage == RagUsage()
    assert not any(name in {"finish_plan_attempt", "accept_plan"} for name, _ in store.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "fence"),
    [
        (FakeModel(VALID_PLAN), "accept"),
        (FakeModel({"bad": "invalid"}), "finish"),
        (FakeModel(failure=RagModelUnavailable()), "finish"),
    ],
)
async def test_atomic_plan_port_fence_rejects_loss_without_persistent_effects(model: FakeModel, fence: str):
    ports, store, *_ = _ports(model=model)
    if fence == "accept":
        store.fail_accept_fence = True
    else:
        store.fail_finish_fence = True
    with pytest.raises(LeaseLost):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert store.running_step is not None
    assert store.run.usage == RagUsage()
    assert store.pages == ()


@pytest.mark.asyncio
async def test_final_run_fence_rejects_loss_without_terminal_completion():
    ports, store, _, runner, _, _ = _ports(pages=(_page(0, "/wiki/launch/overview.md"),))
    store.fail_finish_run_fence = True
    with pytest.raises(LeaseLost):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert runner.calls == [0]
    assert store.run.completion_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dry_run", "completion", "expected"),
    [
        (
            False,
            RagCompletionReason.COMPLETED,
            {
                "run_id": str(RUN_ID),
                "completion_reason": "completed",
                "pages_committed": 2,
            },
        ),
        (
            True,
            RagCompletionReason.DRY_RUN,
            {
                "run_id": str(RUN_ID),
                "completion_reason": "dry_run",
                "pages_committed": 0,
                "pages_dry_run": 2,
            },
        ),
    ],
)
async def test_retry_after_atomic_finish_run_replays_the_exact_terminal_result(
    dry_run: bool,
    completion: RagCompletionReason,
    expected: dict[str, str | int],
):
    pages = (
        _page(0, "/wiki/launch/overview.md"),
        _page(1, "/wiki/launch/risks.md"),
    )
    ports, store, model, runner, retrieval, catalog = _ports(run=_run(dry_run=dry_run), pages=pages)
    store.crash_after_finish_run = True

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert store.run.completion_reason is completion
    usage = store.run.usage
    page_calls = tuple(runner.calls)
    finish_calls = sum(name == "finish_run" for name, _ in store.events)

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result == expected
    assert store.run.usage == usage
    assert tuple(runner.calls) == page_calls
    assert sum(name == "finish_run" for name, _ in store.events) == finish_calls == 1
    assert model.calls == []
    assert retrieval.calls == catalog.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "pages", "expected"),
    [
        (
            replace(
                _run(usage=RagUsage(steps=1, model_tokens=14)),
                completion_reason=RagCompletionReason.NO_WORK,
            ),
            (),
            {
                "run_id": str(RUN_ID),
                "completion_reason": "no_work",
                "pages_committed": 0,
            },
        ),
        (
            replace(
                _run(usage=RagUsage(steps=1), last_committed_ordinal=0),
                completion_reason=RagCompletionReason.COMPLETED,
            ),
            (_page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),),
            {
                "run_id": str(RUN_ID),
                "completion_reason": "completed",
                "pages_committed": 1,
            },
        ),
        (
            replace(
                _run(dry_run=True, usage=RagUsage(steps=1)),
                completion_reason=RagCompletionReason.DRY_RUN,
            ),
            (_page(0, "/wiki/launch/overview.md", state=RagPageState.DRY_RUN_COMPLETE),),
            {
                "run_id": str(RUN_ID),
                "completion_reason": "dry_run",
                "pages_committed": 0,
                "pages_dry_run": 1,
            },
        ),
    ],
)
async def test_disabled_feature_does_not_block_exact_terminal_replay(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
    expected: dict[str, str | int],
):
    gate = FakeFeatureGate(fail_at=1)
    ports, store, model, runner, retrieval, catalog = _ports(
        run=run,
        pages=pages,
        feature_gate=gate,
    )

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result == expected
    assert gate.calls == 0
    assert model.calls == []
    assert runner.calls == []
    assert retrieval.calls == catalog.calls == 0
    assert not any(
        name in {"begin_plan_attempt", "finish_plan_attempt", "accept_plan", "finish_run"} for name, _ in store.events
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "pages"),
    [
        (replace(_run(), completion_reason=RagCompletionReason.COMPLETED), ()),
        (
            replace(
                _run(
                    budget=RagBudget(max_steps=1),
                    usage=RagUsage(steps=2),
                    last_committed_ordinal=0,
                ),
                completion_reason=RagCompletionReason.COMPLETED,
            ),
            (_page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),),
        ),
        (replace(_run(), completion_reason=RagCompletionReason.PARTIAL_FAILURE), ()),
    ],
)
async def test_disabled_feature_cannot_make_malformed_terminal_state_replayable(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
):
    gate = FakeFeatureGate(fail_at=1)
    ports, store, model, runner, _, _ = _ports(run=run, pages=pages, feature_gate=gate)

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value.code == "rag_internal_error"
    assert gate.calls == 0
    assert model.calls == []
    assert runner.calls == []
    assert not any(
        name in {"begin_plan_attempt", "finish_plan_attempt", "accept_plan", "finish_run"} for name, _ in store.events
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "pages"),
    [
        (_run(), ()),
        (
            _run(parent=True),
            (_page(0, "/wiki/launch/overview.md"),),
        ),
    ],
)
async def test_disabled_feature_still_blocks_active_root_and_resume_without_side_effects(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
):
    gate = FakeFeatureGate(fail_at=1)
    ports, store, model, runner, retrieval, catalog = _ports(
        run=run,
        pages=pages,
        feature_gate=gate,
    )

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value.code == "rag_disabled"
    assert gate.calls == 1
    assert model.calls == []
    assert runner.calls == []
    assert retrieval.calls == catalog.calls == 0
    assert [name for name, _ in store.events] == ["reload"]


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_id", [UUID(int=800), UUID(int=802)])
async def test_first_and_multi_generation_resume_lineage_are_authoritative(parent_id):
    run = replace(
        _run(parent=True),
        root_run_id=UUID(int=800),
        parent_run_id=parent_id,
    )
    ports, _, model, runner, _, _ = _ports(
        run=run,
        pages=(_page(0, "/wiki/launch/overview.md"),),
    )

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "completed"
    assert runner.calls == [0]
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        replace(_run(parent=True), root_run_id=RUN_ID),
        replace(_run(parent=True), parent_run_id=RUN_ID),
    ],
)
async def test_resume_lineage_rejects_root_or_parent_self_cycles(run):
    ports, _, model, runner, _, _ = _ports(
        run=run,
        pages=(_page(0, "/wiki/launch/overview.md"),),
    )

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert raised.value.code == "rag_internal_error"
    assert runner.calls == []
    assert model.calls == []


@pytest.mark.asyncio
async def test_empty_acceptance_is_atomic_and_retry_returns_without_replanning_or_double_usage():
    model = FakeModel({"pages": []})
    ports, store, *_ = _ports(model=model)
    store.crash_after_accept = True
    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert store.run.completion_reason is RagCompletionReason.NO_WORK
    assert store.run.usage == RagUsage(steps=1, model_tokens=14)
    assert store.running_step is None
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert result == {
        "run_id": str(RUN_ID),
        "completion_reason": "no_work",
        "pages_committed": 0,
    }
    assert len(model.calls) == 1
    assert [name for name, _ in store.events].count("accept_plan") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run", "pages"),
    [
        (replace(_run(), completion_reason=RagCompletionReason.COMPLETED), ()),
        (replace(_run(), completion_reason=RagCompletionReason.NO_WORK), ()),
        (
            replace(_run(), completion_reason=RagCompletionReason.NO_WORK),
            (_page(0, "/wiki/launch/overview.md"),),
        ),
        (
            replace(
                _run(last_committed_ordinal=0),
                completion_reason=RagCompletionReason.COMPLETED,
            ),
            (
                _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
                _page(1, "/wiki/launch/risks.md"),
            ),
        ),
        (
            replace(
                _run(last_committed_ordinal=0),
                completion_reason=RagCompletionReason.COMPLETED,
            ),
            (
                _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
                _page(1, "/wiki/launch/risks.md", state=RagPageState.COMMITTED),
            ),
        ),
        (
            replace(_run(), completion_reason=RagCompletionReason.DRY_RUN),
            (_page(0, "/wiki/launch/overview.md", state=RagPageState.DRY_RUN_COMPLETE),),
        ),
        (
            replace(_run(dry_run=True), completion_reason=RagCompletionReason.DRY_RUN),
            (),
        ),
        (
            replace(
                _run(dry_run=True, last_committed_ordinal=0),
                completion_reason=RagCompletionReason.DRY_RUN,
            ),
            (_page(0, "/wiki/launch/overview.md", state=RagPageState.DRY_RUN_COMPLETE),),
        ),
        (
            replace(
                _run(last_committed_ordinal=0),
                completion_reason=RagCompletionReason.NO_WORK,
            ),
            (),
        ),
    ],
)
async def test_illegal_terminal_run_and_page_combinations_fail_closed(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
):
    ports, _, model, runner, _, _ = _ports(run=run, pages=pages)
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.code == "rag_internal_error"
    assert model.calls == []
    assert runner.calls == []


@pytest.mark.asyncio
async def test_feature_disable_after_model_return_prevents_acceptance():
    gate = FakeFeatureGate(fail_at=8)
    ports, store, *_ = _ports(feature_gate=gate)
    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.code == "rag_disabled"
    assert store.running_step is not None
    assert store.run.usage == RagUsage()
    assert not any(name == "accept_plan" for name, _ in store.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [RagPageState.RUNNING, RagPageState.FAILED])
@pytest.mark.parametrize("existing", [False, True])
async def test_active_page_projection_is_reconstructed_and_resumed(state, existing):
    page = replace(
        _page(0, "/wiki/launch/overview.md", state=state),
        document_id=UUID(int=2_000) if existing else None,
        version_read=3 if existing else None,
        attempt_count=1,
    )
    ports, store, model, runner, retrieval, catalog = _ports(pages=(page,))

    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "completed"
    assert runner.calls == [0]
    assert store.pages[0].state is RagPageState.COMMITTED
    assert store.pages[0].version_committed == (4 if existing else 1)
    assert model.calls == []
    assert retrieval.calls == catalog.calls == 0


@pytest.mark.asyncio
async def test_crash_reconstructs_the_same_running_page_before_retry():
    page = replace(
        _page(0, "/wiki/launch/overview.md", state=RagPageState.RUNNING),
        attempt_count=1,
    )
    ports, store, model, runner, _, _ = _ports(pages=(page,))
    runner.failure = asyncio.CancelledError("crash during page attempt")

    with pytest.raises(asyncio.CancelledError):
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert store.pages == (page,)
    runner.failure = None
    result = await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    assert result["completion_reason"] == "completed"
    assert runner.calls == [0, 0]
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["model", "retrieval", "store", "page", "feature"])
async def test_external_ordinary_failures_leave_a_fully_detached_public_failure(boundary: str):
    secret = RuntimeError(f"{boundary}-secret")
    pages: tuple[RagPageRecord, ...] = ()
    model = FakeModel()
    gate = FakeFeatureGate()
    if boundary == "model":
        model.failure = secret
    if boundary == "feature":
        gate.failure = secret
    if boundary == "page":
        pages = (_page(0, "/wiki/launch/overview.md"),)
    ports, store, _, runner, retrieval, _ = _ports(pages=pages, model=model, feature_gate=gate)
    if boundary == "retrieval":
        retrieval.failure = secret
    if boundary == "store":
        store.reload_failure = secret
    if boundary == "page":
        runner.failure = secret

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert f"{boundary}-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_public_failure_traceback_locals_do_not_retain_the_private_exception_graph():
    secret = "traceback-local-super-secret"
    inner = RuntimeError(secret)
    outer = RuntimeError("outer-private")
    outer.__cause__ = inner
    ports, *_ = _ports(model=FakeModel(failure=outer))

    with pytest.raises(RagRunFailure) as raised:
        await BuildWikiOrchestrator(ports).run(RUN_ID, FakeLease())

    pending = [raised.value]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        failure = pending.pop()
        if id(failure) in seen:
            continue
        seen.add(id(failure))
        traceback = failure.__traceback__
        while traceback is not None:
            if "/api/rag/" in traceback.tb_frame.f_code.co_filename:
                for value in traceback.tb_frame.f_locals.values():
                    rendered.append(repr(value))
                    if isinstance(value, BaseException):
                        pending.append(value)
            traceback = traceback.tb_next
        if failure.__cause__ is not None:
            pending.append(failure.__cause__)
        if failure.__context__ is not None:
            pending.append(failure.__context__)

    assert secret not in "\n".join(rendered)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_ports_are_runtime_typed_frozen_and_repr_private():
    ports, store, model, *_ = _ports()
    assert isinstance(store, RunStore)
    assert isinstance(model, StructuredRagModel)
    assert repr(ports) == "OrchestratorPorts(<redacted>)"
    with pytest.raises(Exception):
        ports.model = FakeModel()  # type: ignore[misc]


def test_ports_reject_missing_and_hostile_dependencies_without_leaking_repr():
    ports, *_ = _ports()
    values = {name: getattr(ports, name) for name in ports.__slots__}
    values["store"] = None
    with pytest.raises(TypeError, match="port is invalid"):
        OrchestratorPorts(**values)

    class HostilePort:
        @property
        def reload(self):
            raise RuntimeError("secret port detail")

    values["store"] = HostilePort()
    with pytest.raises(TypeError) as raised:
        OrchestratorPorts(**values)
    assert "secret" not in str(raised.value)


def test_catalog_and_failure_direct_construction_rejects_hostile_values():
    secret = "https://secret.invalid/?key=private"
    with pytest.raises(ValueError, match="catalog item"):
        WikiCatalogItem(f"/wiki/{secret}\x00.md", None)
    with pytest.raises(ValueError, match="run failure"):
        RagRunFailure("INVALID", secret, True)
    failure = RagRunFailure("rag_internal_error", "A fixed message.", True)
    with pytest.raises(AttributeError, match="immutable"):
        failure._code = "rag_disabled"  # type: ignore[misc]


def test_atomic_page_commit_is_a_public_frozen_boundary():
    source = {"ok": True, "nested": {"items": [1, 2]}}
    run = _run(last_committed_ordinal=0)
    page = replace(
        _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
        document_id=UUID(int=999),
        lint_summary=source,
    )
    written = WikiWriteResult(UUID(int=999), "overview.md", "/wiki/launch/", 1)
    commit = AtomicPageCommit(run, page, written, source)
    assert commit.written.version == 1
    source["nested"]["items"].append(3)
    assert commit.lint_summary["nested"]["items"] == (1, 2)
    assert commit.page.lint_summary["nested"]["items"] == (1, 2)
    with pytest.raises(Exception):
        commit.page = page  # type: ignore[misc]


@pytest.mark.parametrize(
    ("run", "page", "written", "summary"),
    [
        (
            _run(),
            _page(0, "/wiki/launch/overview.md"),
            WikiWriteResult(UUID(int=999), "overview.md", "/wiki/launch/", 1),
            {},
        ),
        (
            _run(last_committed_ordinal=0),
            replace(
                _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
                document_id=UUID(int=999),
                lint_summary={},
            ),
            WikiWriteResult(UUID(int=999), "wrong.md", "/wiki/launch/", 1),
            {},
        ),
        (
            _run(last_committed_ordinal=0),
            replace(
                _page(0, "/wiki/launch/overview.md", state=RagPageState.COMMITTED),
                document_id=UUID(int=999),
                lint_summary={"a": 1},
            ),
            WikiWriteResult(UUID(int=999), "overview.md", "/wiki/launch/", 1),
            {"a": 2},
        ),
    ],
)
def test_atomic_page_commit_rejects_non_atomic_or_mismatched_projections(run, page, written, summary):
    with pytest.raises(TypeError, match="atomic page commit"):
        AtomicPageCommit(run, page, written, summary)


def test_repair_digest_covers_the_complete_canonical_repair_template():
    template = {
        "base_prompt_digest": PLANNER_PROMPT_DIGEST,
        "base_prompt_version": PLANNER_PROMPT_VERSION,
        "repair_message": {
            "role": "system",
            "content": (
                "The previous response did not satisfy the required schema. "
                "Return a new complete JSON object following the original policy."
            ),
        },
        "repair_prompt_version": PLANNER_REPAIR_PROMPT_VERSION,
    }
    canonical = json.dumps(template, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == PLANNER_REPAIR_PROMPT_DIGEST
    template["repair_message"]["content"] += " changed"
    changed = json.dumps(template, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(changed).hexdigest() != PLANNER_REPAIR_PROMPT_DIGEST
