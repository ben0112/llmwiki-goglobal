"""Bounded page execution and retry contracts."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import UTC, date, datetime
from types import MappingProxyType
from uuid import UUID

import pytest
from jobs.models import JobCancelled, JobRecord, JobState, JobType, LeaseLost
from rag.model import InvalidRagModelResponse, RagModelResponse, RagModelUnavailable, RagTokenUsage
from rag.page_runner import PageRunner
from rag.ports import AtomicPageCommit, PageExecutionResult
from rag.records import RagPageRecord, RagRunRecord, RagStepRecord
from rag.retrieval import RagEvidence, RagWikiPage
from rag.wiki_writer import PersistedRagLintError

from llmwiki_adapters.postgres.wiki import WikiWriteResult
from llmwiki_core.documents import DocumentKind, DocumentStatus
from llmwiki_core.rag import RagBudget, RagCitation, RagDomainError, RagPageState, RagStepStatus, RagStepType, RagUsage
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchResult
from llmwiki_core.wiki import VersionConflict

RUN_ID = UUID("00000000-0000-0000-0000-000000001001")
JOB_ID = UUID("00000000-0000-0000-0000-000000001002")
USER_ID = UUID("00000000-0000-0000-0000-000000001003")
KB_ID = UUID("00000000-0000-0000-0000-000000001004")
PAGE_ID = UUID("00000000-0000-0000-0000-000000001005")
DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000001006")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000001007")
NOW = datetime(2026, 7, 27, tzinfo=UTC)


def _run(*, dry_run: bool = False, budget: RagBudget | None = None) -> RagRunRecord:
    return RagRunRecord(
        id=RUN_ID,
        job_id=JOB_ID,
        root_run_id=RUN_ID,
        parent_run_id=None,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        goal="Build a sourced launch wiki",
        goal_digest=hashlib.sha256(b"Build a sourced launch wiki").hexdigest(),
        target_path_prefix="/wiki/launch/",
        model_profile="primary",
        model_profile_version="primary-v1",
        retrieval_profile="lexical",
        dry_run=dry_run,
        budget=budget or RagBudget(),
        usage=RagUsage(),
        idempotency_key="page-run",
        request_digest="a" * 64,
        completion_reason=None,
        last_committed_ordinal=-1,
        created_at=NOW,
        updated_at=NOW,
    )


def _page(**changes) -> RagPageRecord:
    page = RagPageRecord(
        id=PAGE_ID,
        run_id=RUN_ID,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        ordinal=0,
        path="/wiki/launch/risks.md",
        intent="Explain launch risks",
        query="launch risk evidence",
        state=RagPageState.PLANNED,
        document_id=None,
        version_read=None,
        version_committed=None,
        attempt_count=0,
        conflict_retry_count=0,
        last_completed_step_sequence=0,
        preview=None,
        preview_digest=None,
        preview_full_char_count=None,
        preview_truncated=False,
        lint_summary=None,
        created_at=NOW,
        updated_at=NOW,
    )
    return replace(page, **changes)


def _job() -> JobRecord:
    return JobRecord(
        id=JOB_ID,
        job_type=JobType.BUILD_WIKI,
        user_id=USER_ID,
        state=JobState.RUNNING,
        knowledge_base_id=KB_ID,
        payload={"run_id": str(RUN_ID)},
        idempotency_key="page-job",
        lease_owner="worker-1",
        lease_expires_at=NOW,
    )


def _content(extra: str = "") -> str:
    return (
        "---\n"
        "title: Launch risks\n"
        "tags: [launch, risk]\n"
        "description: A sourced launch risk summary.\n"
        "date: 2026-07-27\n"
        "---\n"
        "# Launch risks\n\n"
        "```mermaid\ngraph TD\n  A --> B\n```\n\n"
        "Risk is documented.[^1]\n\n[^1]: source.pdf, p.5\n"
        f"{extra}"
    )


def _payload(*, content: str | None = None, citation: RagCitation | None = None):
    cited = citation or RagCitation(SOURCE_ID, 3, 4, 5)
    return {
        "content": _content() if content is None else content,
        "citations": [
            {
                "document_id": str(cited.document_id),
                "document_version": cited.document_version,
                "chunk_index": cited.chunk_index,
                "page": cited.page,
            }
        ],
    }


HIT = SearchHit(str(SOURCE_ID), 3, 4, "hit", 0.9, "/corpus/source.pdf", page=5)
EVIDENCE = RagEvidence(
    document_id=SOURCE_ID,
    document_version=3,
    chunk_index=4,
    page=5,
    filename="source.pdf",
    path="/corpus/",
    title="Source",
    content="Selected exact source text",
    status=DocumentStatus.READY,
    archived=False,
    score=0.9,
    document_kind=DocumentKind.SOURCE,
)


class FakeLease:
    def __init__(self, *, fail_at: int | None = None, failure: BaseException | None = None):
        self.calls = 0
        self.fail_at = fail_at
        self.failure = failure or JobCancelled("cancelled")

    async def checkpoint(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.failure
        return _job()


class FakeGate:
    def __init__(self, events):
        self.events = events

    async def ensure_enabled(self):
        self.events.append("feature")


class FakeStore:
    def __init__(self, run, page, events):
        self.run, self.page, self.events = run, page, events
        self.steps = []
        self.running_step = None
        self.finished_steps = []
        self.failpoint = None
        self.reload_mutator = None
        self.usage_mutator = None
        self.begin_mutator = None
        self.read_mutator = None
        self.start_mutator = None
        self.finish_mutators = []
        self.conflict_mutator = None
        self.dry_mutator = None

    async def reload_boundary(self, **kwargs):
        del kwargs
        result = PageExecutionResult(self.run, self.page)
        return result if self.reload_mutator is None else self.reload_mutator(result)

    async def load_usage(self, **kwargs):
        del kwargs
        usage = self.run.usage
        return usage if self.usage_mutator is None else self.usage_mutator(usage)

    async def begin_page_attempt(self, **kwargs):
        del kwargs
        if self.page.attempt_count >= self.run.budget.max_page_attempts:
            raise RagDomainError("rag_page_attempts_exhausted", "attempts exhausted")
        self.page = replace(
            self.page,
            state=RagPageState.RUNNING,
            attempt_count=self.page.attempt_count + 1,
        )
        self.events.append("attempt")
        return self.page if self.begin_mutator is None else self.begin_mutator(self.page)

    async def record_page_read(self, *, document_id, version, **kwargs):
        del kwargs
        self.page = replace(self.page, document_id=document_id, version_read=version)
        return self.page if self.read_mutator is None else self.read_mutator(self.page)

    async def start_step(self, *, run, page, step_type, input_digest, reserved_tokens=0, **kwargs):
        del kwargs
        if self.running_step is not None:
            raise AssertionError("stale running step")
        self.steps.append(step_type)
        self.events.append(step_type.value)
        self.running_step = RagStepRecord(
            id=UUID(int=10_000 + len(self.steps)),
            run_id=run.id,
            run_page_id=page.id,
            user_id=run.user_id,
            knowledge_base_id=run.knowledge_base_id,
            sequence=len(self.steps),
            step_type=step_type,
            status=RagStepStatus.RUNNING,
            input_digest=input_digest,
            output_summary=MappingProxyType({}),
            citation_identities=(),
            prompt_version=None,
            prompt_digest=None,
            model_profile_version=run.model_profile_version,
            reserved_tokens=reserved_tokens,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
            error_code=None,
            error_message=None,
            created_at=NOW,
            updated_at=NOW,
        )
        if self.start_mutator is not None:
            self.running_step = self.start_mutator(self.running_step)
        return self.running_step

    async def finish_step(self, *, token_usage=RagTokenUsage(0, 0, 0), **kwargs):
        assert kwargs["step"] == self.running_step
        finished = replace(
            self.running_step,
            status=kwargs["status"],
            output_summary=MappingProxyType(dict(kwargs["summary"])),
            citation_identities=tuple(kwargs["citations"]),
            prompt_version=kwargs.get("prompt_version"),
            prompt_digest=kwargs.get("prompt_digest"),
            input_tokens=token_usage.prompt_tokens,
            output_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            error_code=kwargs.get("error_code"),
        )
        if self.finish_mutators:
            return self.finish_mutators.pop(0)(finished)
        self.finished_steps.append(
            (
                self.steps[self.running_step.sequence - 1],
                kwargs["status"],
                kwargs["summary"],
                kwargs.get("error_code"),
                token_usage,
            )
        )
        self.running_step = None
        self.run = replace(
            self.run,
            usage=RagUsage(
                self.run.usage.steps + 1,
                self.run.usage.model_tokens + token_usage.total_tokens,
            ),
        )
        return finished

    async def record_conflict(self, **kwargs):
        del kwargs
        if self.page.conflict_retry_count >= self.run.budget.max_conflict_retries:
            raise RagDomainError("rag_version_conflict", "conflicts exhausted")
        self.page = replace(self.page, conflict_retry_count=self.page.conflict_retry_count + 1)
        self.events.append("conflict_recorded")
        return self.page if self.conflict_mutator is None else self.conflict_mutator(self.page)

    async def complete_dry_run(self, *, preview, preview_digest, preview_full_char_count, preview_truncated, **kwargs):
        del kwargs
        self.page = replace(
            self.page,
            state=RagPageState.DRY_RUN_COMPLETE,
            document_id=None,
            version_read=None,
            preview=preview,
            preview_digest=preview_digest,
            preview_full_char_count=preview_full_char_count,
            preview_truncated=preview_truncated,
            last_completed_step_sequence=self.run.usage.steps,
        )
        result = PageExecutionResult(self.run, self.page)
        return result if self.dry_mutator is None else self.dry_mutator(result)


class FakeRetrieval:
    def __init__(self, events, failure=None, result=None):
        self.events, self.queries, self.failure, self.result = events, [], failure, result

    async def retrieve(self, query, *, profile="lexical"):
        self.events.append("retrieve_port")
        self.queries.append((query, profile))
        if self.failure is not None:
            raise self.failure
        return self.result if self.result is not None else SearchResult((HIT,), 1, profile=profile)


class FakeEvidenceReader:
    def __init__(self, events, failure=None):
        self.events, self.calls, self.failure = events, [], failure

    async def read(self, user_id, knowledge_base_id, hits, max_chars):
        self.events.append("read_port")
        self.calls.append((user_id, knowledge_base_id, tuple(hits), max_chars))
        if self.failure is not None:
            raise self.failure
        return (EVIDENCE,)


class FakeWikiReader:
    def __init__(self, events, pages=()):
        self.events, self.pages, self.calls = events, list(pages), []

    async def get_by_path(self, user_id, knowledge_base_id, path):
        self.events.append("current_page")
        self.calls.append((user_id, knowledge_base_id, path))
        return self.pages.pop(0) if self.pages else None


class FakeModel:
    def __init__(self, events, payloads, failure=None, usage=RagTokenUsage(10, 20, 30)):
        self.events, self.payloads, self.messages, self.failure = events, list(payloads), [], failure
        self.usage = usage

    async def complete_json(self, *, messages, max_output_tokens, timeout_seconds):
        self.events.append("draft_port")
        self.messages.append((messages, max_output_tokens, timeout_seconds))
        if self.failure is not None:
            raise self.failure
        return RagModelResponse(self.payloads.pop(0), self.usage)


class FakeLinter:
    def __init__(self, events, failure=None, result=None):
        self.events, self.failure, self.result = events, failure, result

    async def lint(self, **kwargs):
        del kwargs
        self.events.append("lint_port")
        if self.failure is not None:
            raise self.failure
        return {"warnings": 0} if self.result is None else self.result


class FakeWriter:
    def __init__(self, store, events, failures=()):
        self.store, self.events, self.failures, self.calls = store, events, list(failures), []
        self.commit_mutator = None

    async def commit(self, **kwargs):
        self.events.append("write_port")
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        bundle = kwargs["bundle"]
        written = WikiWriteResult(
            UUID(bundle.document_id), bundle.filename, bundle.path, (bundle.expected_version or 0) + 1
        )
        committed_page = replace(
            self.store.page,
            state=RagPageState.COMMITTED,
            document_id=written.document_id,
            version_read=bundle.expected_version,
            version_committed=written.version,
            last_completed_step_sequence=self.store.run.usage.steps,
            lint_summary=MappingProxyType({"persisted": True}),
        )
        committed_run = replace(self.store.run, last_committed_ordinal=committed_page.ordinal)
        self.store.run, self.store.page = committed_run, committed_page
        committed = AtomicPageCommit(committed_run, committed_page, written, {"persisted": True})
        return committed if self.commit_mutator is None else self.commit_mutator(committed)


def _wiki_page(version=1):
    return RagWikiPage(
        document_id=DOCUMENT_ID,
        version=version,
        path="/wiki/launch/risks.md",
        filename="risks.md",
        content=f"old version {version}",
        title="Risks",
        tags=("risk",),
        date=date(2026, 7, 20),
        metadata=MappingProxyType({}),
    )


def _runner(
    *,
    run=None,
    page=None,
    payloads=None,
    current_pages=(),
    writer_failures=(),
    retrieval_failure=None,
    retrieval_result=None,
    reader_failure=None,
    model_failure=None,
    model_usage=RagTokenUsage(10, 20, 30),
    lint_failure=None,
    lint_result=None,
):
    events = []
    store = FakeStore(run or _run(), page or _page(), events)
    writer = FakeWriter(store, events, writer_failures)
    model = FakeModel(events, payloads or [_payload()], model_failure, model_usage)
    reader = FakeEvidenceReader(events, reader_failure)
    wiki_reader = FakeWikiReader(events, current_pages)
    runner = PageRunner(
        store=store,
        model=model,
        retrieval=FakeRetrieval(events, retrieval_failure, retrieval_result),
        evidence_reader=reader,
        wiki_page_reader=wiki_reader,
        wiki_writer=writer,
        draft_linter=FakeLinter(events, lint_failure, lint_result),
        feature_gate=FakeGate(events),
    )
    return runner, store, writer, model, reader, wiki_reader, events


async def test_page_flow_orders_numbered_phases_and_preserves_exact_evidence_identity():
    runner, store, writer, model, reader, _, events = _runner(current_pages=(_wiki_page(),))

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert store.steps == [
        RagStepType.RETRIEVE,
        RagStepType.READ,
        RagStepType.DRAFT,
        RagStepType.VALIDATE,
        RagStepType.WRITE,
        RagStepType.LINT,
    ]
    assert events.index("retrieve_port") < events.index("read_port") < events.index("draft_port")
    assert events.index("draft_port") < events.index("write") < events.index("lint")
    assert reader.calls[0][2] == (HIT,)
    prompt = "\n".join(message["content"] for message in model.messages[0][0])
    assert all(value in prompt for value in (str(SOURCE_ID), '"document_version":3', '"chunk_index":4'))
    assert writer.calls[0]["bundle"].expected_version == 1


async def test_new_page_uses_new_document_identity_and_no_expected_version():
    runner, store, writer, *_ = _runner()

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert writer.calls[0]["bundle"].expected_version is None
    assert writer.calls[0]["bundle"].document_id != str(SOURCE_ID)


async def test_invalid_draft_is_repaired_once_with_a_new_page_attempt():
    invalid = _payload(citation=RagCitation(UUID(int=999), 1, 0))
    runner, store, writer, model, *_ = _runner(payloads=[invalid, _payload()])

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert result.page.attempt_count == 2
    assert len(model.messages) == 2
    assert "repair" in model.messages[1][0][-1]["content"].lower()
    assert len(writer.calls) == 1


async def test_invalid_writer_schema_is_repaired_once_and_charges_the_failed_call():
    runner, store, writer, model, *_ = _runner(payloads=[{"private": "raw output"}, _payload()])

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert result.page.attempt_count == 2
    assert len(model.messages) == 2
    assert store.run.usage.model_tokens == 60
    assert len(writer.calls) == 1


async def test_unsupported_citation_after_repair_is_terminal_and_never_writes():
    invalid = _payload(citation=RagCitation(UUID(int=999), 1, 0))
    runner, store, writer, *_ = _runner(payloads=[invalid, invalid])

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_invalid_draft"
    assert writer.calls == []
    assert store.page.attempt_count == 2


async def test_dry_run_has_no_wiki_mutation_and_persists_bounded_full_digest_and_count():
    long_draft = _content("x" * 20_000)
    runner, store, writer, *_ = _runner(run=_run(dry_run=True), payloads=[_payload(content=long_draft)])

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.DRY_RUN_COMPLETE
    assert len(result.page.preview.encode("utf-8")) <= 16_384
    assert result.page.preview_truncated is True
    assert result.page.preview_digest == hashlib.sha256(long_draft.encode()).hexdigest()
    assert result.page.preview_full_char_count == len(long_draft)
    assert writer.calls == []


async def test_conflict_rereads_current_page_and_redrafts_with_both_counters_bounded():
    runner, store, writer, model, _, wiki_reader, _ = _runner(
        current_pages=(_wiki_page(1), _wiki_page(2)),
        payloads=[_payload(), _payload()],
        writer_failures=(VersionConflict("stale"),),
    )

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert result.page.attempt_count == 2
    assert result.page.conflict_retry_count == 1
    assert len(wiki_reader.calls) == len(model.messages) == len(writer.calls) == 2
    assert [call["bundle"].expected_version for call in writer.calls] == [1, 2]


async def test_conflict_cap_propagates_sanitized_domain_failure_without_stale_commit():
    budget = replace(RagBudget(), max_page_attempts=3, max_conflict_retries=1)
    runner, store, writer, *_ = _runner(
        run=_run(budget=budget),
        current_pages=(_wiki_page(1), _wiki_page(2)),
        payloads=[_payload(), _payload()],
        writer_failures=(VersionConflict("private stale one"), VersionConflict("private stale two")),
    )

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_version_conflict"
    assert store.run.last_committed_ordinal == -1
    assert store.page.conflict_retry_count == 1
    assert len(writer.calls) == 2
    assert store.steps.count(RagStepType.CONFLICT) == 2
    assert store.finished_steps[-1][:4] == (
        RagStepType.CONFLICT,
        RagStepStatus.FAILED,
        {"outcome": "failed", "phase": "conflict", "usage_trusted": False},
        "rag_version_conflict",
    )
    assert "private" not in repr(store.finished_steps[-1])


async def test_two_persisted_lint_rollbacks_are_both_audited_before_terminal_invalid_draft():
    runner, store, writer, model, *_ = _runner(
        payloads=[_payload(), _payload()],
        writer_failures=(PersistedRagLintError(), PersistedRagLintError()),
    )

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_invalid_draft"
    failed_validations = [
        finished
        for finished in store.finished_steps
        if finished[0] is RagStepType.VALIDATE
        and finished[1] is RagStepStatus.FAILED
        and finished[3] == "rag_persisted_lint_failed"
    ]
    assert len(failed_validations) == 2
    assert [finished[2] for finished in failed_validations] == [
        {"outcome": "persisted_lint_rollback"},
        {"outcome": "persisted_lint_rollback"},
    ]
    assert store.page.attempt_count == 2
    assert len(model.messages) == len(writer.calls) == 2
    assert "repair" not in model.messages[0][0][-1]["content"].lower()
    assert "repair" in model.messages[1][0][-1]["content"].lower()
    assert store.run.last_committed_ordinal == -1
    assert store.page.document_id is None
    assert store.page.version_committed is None


async def test_page_attempt_cap_is_checked_before_ports_or_model():
    budget = replace(RagBudget(), max_page_attempts=1)
    page = _page(state=RagPageState.FAILED, attempt_count=1)
    runner, store, writer, model, *_ = _runner(run=_run(budget=budget), page=page)

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_budget_exhausted"
    assert writer.calls == model.messages == []


@pytest.mark.parametrize("failure", [JobCancelled("cancelled"), LeaseLost("lost")])
async def test_control_signal_at_each_checkpoint_stops_before_publication(failure):
    baseline, store, *_ = _runner()
    lease = FakeLease()
    await baseline.run(store.run, store.page, lease)
    checkpoint_count = lease.calls
    assert checkpoint_count >= 7

    for fail_at in range(1, checkpoint_count + 1):
        runner, state, writer, *_ = _runner()
        with pytest.raises(type(failure)):
            await runner.run(state.run, state.page, FakeLease(fail_at=fail_at, failure=failure))
        if fail_at < checkpoint_count:
            assert state.run.last_committed_ordinal == -1
            assert writer.calls == []


async def test_lease_loss_immediately_before_atomic_write_never_calls_writer():
    runner, store, writer, *_ = _runner()
    lease = FakeLease()
    runner.before_commit = lambda: setattr(lease, "fail_at", lease.calls + 1)

    with pytest.raises(JobCancelled):
        await runner.run(store.run, store.page, lease)

    assert writer.calls == []
    assert store.run.last_committed_ordinal == -1


@pytest.mark.parametrize(
    ("failure_kwargs", "failure", "failed_phase", "error_code"),
    [
        (
            {"retrieval_failure": RetrieverUnavailable("private retrieval query")},
            RetrieverUnavailable,
            RagStepType.RETRIEVE,
            "rag_retrieval_failed",
        ),
        (
            {"reader_failure": RuntimeError("private reader row")},
            RuntimeError,
            RagStepType.READ,
            "rag_internal_error",
        ),
        (
            {"model_failure": RagModelUnavailable()},
            RagModelUnavailable,
            RagStepType.DRAFT,
            "rag_model_unavailable",
        ),
        (
            {"lint_failure": RagDomainError("rag_invalid_draft", "private lint output")},
            RagDomainError,
            RagStepType.LINT,
            "rag_invalid_draft",
        ),
    ],
)
async def test_ordinary_phase_failure_closes_started_step_with_bounded_audit(
    failure_kwargs,
    failure,
    failed_phase,
    error_code,
):
    runner, store, writer, *_ = _runner(payloads=[_payload(), _payload()], **failure_kwargs)

    with pytest.raises(failure):
        await runner.run(store.run, store.page, FakeLease())

    assert store.running_step is None
    phase, status, summary, recorded_code, _ = store.finished_steps[-1]
    assert (phase, status, recorded_code) == (failed_phase, RagStepStatus.FAILED, error_code)
    assert summary == {"outcome": "failed", "phase": failed_phase.value, "usage_trusted": False}
    assert "private" not in repr(store.finished_steps)
    assert writer.calls == []


async def test_control_signal_inside_phase_propagates_without_attempting_unsafe_finish():
    cancellation = asyncio.CancelledError("private cancellation")
    runner, store, writer, *_ = _runner(retrieval_failure=cancellation)

    with pytest.raises(asyncio.CancelledError):
        await runner.run(store.run, store.page, FakeLease())

    assert store.running_step.sequence == 1
    assert store.finished_steps == []
    assert writer.calls == []


async def test_writer_reserves_estimated_input_plus_output_but_sends_only_output(monkeypatch):
    messages = ({"role": "user", "content": "x" * 4_000},)
    monkeypatch.setattr("rag.page_runner.build_writer_messages", lambda **kwargs: messages)
    budget = RagBudget(max_model_tokens=20_000)
    usage = RagTokenUsage(prompt_tokens=1_000, completion_tokens=8_000, total_tokens=9_000)
    runner, store, writer, model, *_ = _runner(run=_run(budget=budget), model_usage=usage)

    result = await runner.run(store.run, store.page, FakeLease())

    assert result.page.state is RagPageState.COMMITTED
    assert model.messages[0][1] == 8_192
    assert store.run.usage.model_tokens == 9_000
    assert len(writer.calls) == 1


@pytest.mark.parametrize(
    ("reported", "usage_trusted"),
    [
        (RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14), True),
        (RagTokenUsage(prompt_tokens=20_000, completion_tokens=0, total_tokens=20_000), False),
        (None, False),
    ],
)
async def test_invalid_model_response_is_terminal_and_charges_only_trusted_or_reserved_usage(
    reported,
    usage_trusted,
):
    runner, store, writer, model, *_ = _runner(model_failure=InvalidRagModelResponse(reported))

    with pytest.raises(InvalidRagModelResponse):
        await runner.run(store.run, store.page, FakeLease())

    reservation = model.messages[0][1] + max(
        1,
        (sum(len(item["content"].encode("utf-8")) for item in model.messages[0][0]) + 3) // 4,
    )
    recorded = store.finished_steps[-1]
    expected_usage = (
        reported
        if usage_trusted
        else RagTokenUsage(reservation, 0, reservation)
        if reported is not None
        else RagTokenUsage(0, 0, 0)
    )
    assert recorded[:4] == (
        RagStepType.DRAFT,
        RagStepStatus.FAILED,
        {"outcome": "failed", "phase": "draft", "usage_trusted": usage_trusted},
        "rag_invalid_draft",
    )
    assert recorded[4] == expected_usage
    assert store.run.usage.model_tokens == expected_usage.total_tokens
    assert store.page.attempt_count == 1
    assert len(model.messages) == 1
    assert writer.calls == []


async def test_hostile_invalid_model_subclass_is_never_introspected_and_is_safely_detached():
    class HostileInvalidModelResponse(InvalidRagModelResponse):
        def __getattribute__(self, name):
            if name == "usage":
                raise AssertionError("private hostile usage getter")
            return super().__getattribute__(name)

    failure = HostileInvalidModelResponse()
    failure.__cause__ = RuntimeError("private linked provider response")
    runner, store, writer, model, *_ = _runner(model_failure=failure)

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_invalid_draft"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    reservation = model.messages[0][1] + max(
        1,
        (sum(len(item["content"].encode("utf-8")) for item in model.messages[0][0]) + 3) // 4,
    )
    assert store.finished_steps[-1][4] == RagTokenUsage(reservation, 0, reservation)
    assert store.finished_steps[-1][2]["usage_trusted"] is False
    assert store.running_step is None
    assert "private" not in repr(store.finished_steps)
    assert writer.calls == []


async def test_search_result_subclass_is_failed_before_retrieve_success_is_recorded():
    class HostileSearchResult(SearchResult):
        pass

    runner, store, writer, *_ = _runner(retrieval_result=HostileSearchResult((HIT,), 1))

    with pytest.raises(TypeError, match="retrieval result"):
        await runner.run(store.run, store.page, FakeLease())

    assert store.finished_steps[-1][:4] == (
        RagStepType.RETRIEVE,
        RagStepStatus.FAILED,
        {"outcome": "failed", "phase": "retrieve", "usage_trusted": False},
        "rag_internal_error",
    )
    assert store.running_step is None
    assert writer.calls == []


async def test_lint_mapping_subclass_is_failed_before_lint_success_is_recorded():
    class HostileLintSummary(dict):
        pass

    runner, store, writer, *_ = _runner(
        payloads=[_payload(), _payload()],
        lint_result=HostileLintSummary(warnings=0),
    )

    with pytest.raises(RagDomainError) as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.code == "rag_invalid_draft"
    lint_steps = [item for item in store.finished_steps if item[0] is RagStepType.LINT]
    assert len(lint_steps) == 2
    assert all(item[1] is RagStepStatus.FAILED and item[3] == "rag_invalid_draft" for item in lint_steps)
    assert store.running_step is None
    assert writer.calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: PageExecutionResult(replace(result.run, goal="hostile replacement"), result.page),
        lambda result: PageExecutionResult(
            replace(result.run, budget=replace(result.run.budget, max_model_tokens=1)), result.page
        ),
        lambda result: PageExecutionResult(replace(result.run, model_profile="hostile-model"), result.page),
        lambda result: PageExecutionResult(replace(result.run, retrieval_profile="hostile-retrieval"), result.page),
        lambda result: PageExecutionResult(result.run, replace(result.page, ordinal=1)),
        lambda result: PageExecutionResult(result.run, replace(result.page, path="/wiki/launch/hostile.md")),
        lambda result: PageExecutionResult(result.run, replace(result.page, intent="Hostile intent")),
        lambda result: PageExecutionResult(result.run, replace(result.page, query="hostile query")),
    ],
    ids=("goal", "budget", "model", "retrieval", "ordinal", "path", "intent", "query"),
)
async def test_hostile_reload_identity_is_rejected_before_any_remote_side_effect(mutate):
    runner, store, writer, model, *_ = _runner()
    store.reload_mutator = mutate

    with pytest.raises(TypeError, match="page runner state") as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert model.messages == writer.calls == []
    assert "retrieve_port" not in store.events
    assert store.finished_steps == []
    assert store.running_step is None


@pytest.mark.parametrize("dry_run", [True, False], ids=("dry-to-committed", "write-to-dry"))
async def test_hostile_reload_terminal_state_must_match_run_mode_before_remote_side_effect(dry_run):
    runner, store, writer, model, *_ = _runner(run=_run(dry_run=dry_run))
    if dry_run:
        store.reload_mutator = lambda result: PageExecutionResult(
            replace(result.run, last_committed_ordinal=result.page.ordinal),
            replace(
                result.page,
                state=RagPageState.COMMITTED,
                document_id=DOCUMENT_ID,
                version_committed=1,
            ),
        )
    else:
        preview = "hostile preview"
        store.reload_mutator = lambda result: PageExecutionResult(
            result.run,
            replace(
                result.page,
                state=RagPageState.DRY_RUN_COMPLETE,
                preview=preview,
                preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
                preview_full_char_count=len(preview),
            ),
        )

    with pytest.raises(TypeError, match="page runner state") as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.__cause__ is caught.value.__context__ is None
    assert model.messages == writer.calls == []
    assert "retrieve_port" not in store.events
    assert store.finished_steps == []
    assert store.running_step is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda page: replace(page, attempt_count=page.attempt_count + 1),
        lambda page: replace(page, conflict_retry_count=page.conflict_retry_count + 1),
        lambda page: replace(page, state=RagPageState.PLANNED),
        lambda page: replace(page, path="/wiki/launch/hostile.md"),
    ],
    ids=("attempt", "conflict", "state", "identity"),
)
async def test_hostile_attempt_transition_is_rejected_before_retrieval(mutate):
    runner, store, writer, model, *_ = _runner()
    store.begin_mutator = mutate

    with pytest.raises(TypeError, match="page result"):
        await runner.run(store.run, store.page, FakeLease())

    assert model.messages == writer.calls == []
    assert "retrieve_port" not in store.events
    assert store.finished_steps == []
    assert store.running_step is None


@pytest.mark.parametrize("usage", [RagUsage(), RagUsage(steps=201, model_tokens=0)], ids=("rollback", "over-budget"))
async def test_hostile_usage_is_rejected_before_retrieval(usage):
    run = replace(_run(), usage=RagUsage(steps=1, model_tokens=5))
    runner, store, writer, model, *_ = _runner(run=run)
    store.usage_mutator = lambda current: usage

    with pytest.raises(TypeError, match="usage"):
        await runner.run(store.run, store.page, FakeLease())

    assert model.messages == writer.calls == []
    assert "retrieve_port" not in store.events


@pytest.mark.parametrize(
    "mutate",
    [
        lambda step: replace(step, run_id=UUID(int=999)),
        lambda step: replace(step, run_page_id=UUID(int=998)),
        lambda step: replace(step, step_type=RagStepType.READ),
        lambda step: replace(step, reserved_tokens=1),
    ],
    ids=("run", "page", "type", "reservation"),
)
async def test_hostile_start_step_is_rejected_before_retrieval(mutate):
    runner, store, writer, model, *_ = _runner()
    store.start_mutator = mutate

    with pytest.raises(TypeError, match="step") as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.__cause__ is caught.value.__context__ is None
    assert model.messages == writer.calls == []
    assert "retrieve_port" not in store.events
    assert not any(item[1] is RagStepStatus.SUCCEEDED for item in store.finished_steps)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda step: replace(step, run_id=UUID(int=999)),
        lambda step: replace(step, run_page_id=UUID(int=998)),
        lambda step: replace(step, step_type=RagStepType.READ),
        lambda step: replace(step, reserved_tokens=1),
    ],
    ids=("run", "page", "type", "reservation"),
)
async def test_hostile_finish_step_is_failed_before_model_or_writer(mutate):
    runner, store, writer, model, *_ = _runner()
    store.finish_mutators = [mutate]

    with pytest.raises(TypeError, match="step") as caught:
        await runner.run(store.run, store.page, FakeLease())

    assert caught.value.__cause__ is caught.value.__context__ is None
    assert model.messages == writer.calls == []
    assert not any(item[1] is RagStepStatus.SUCCEEDED for item in store.finished_steps)
    assert store.running_step is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda page: replace(page, document_id=SOURCE_ID, version_read=1),
        lambda page: replace(page, attempt_count=page.attempt_count + 1),
        lambda page: replace(page, path="/wiki/launch/hostile.md"),
    ],
    ids=("document", "counter", "identity"),
)
async def test_hostile_read_output_closes_read_failed_before_model_or_writer(mutate):
    runner, store, writer, model, *_ = _runner()
    store.read_mutator = mutate

    with pytest.raises(TypeError, match="page result"):
        await runner.run(store.run, store.page, FakeLease())

    assert model.messages == writer.calls == []
    assert store.finished_steps[-1][0:2] == (RagStepType.READ, RagStepStatus.FAILED)
    assert not any(item[0] is RagStepType.READ and item[1] is RagStepStatus.SUCCEEDED for item in store.finished_steps)
    assert store.running_step is None


async def test_hostile_conflict_output_closes_conflict_failed_without_retrying():
    runner, store, writer, model, *_ = _runner(writer_failures=(VersionConflict("private stale"),))
    store.conflict_mutator = lambda page: replace(
        page,
        conflict_retry_count=page.conflict_retry_count + 1,
    )

    with pytest.raises(TypeError, match="page result"):
        await runner.run(store.run, store.page, FakeLease())

    assert len(model.messages) == len(writer.calls) == 1
    assert store.finished_steps[-1][0:2] == (RagStepType.CONFLICT, RagStepStatus.FAILED)
    assert not any(
        item[0] is RagStepType.CONFLICT and item[1] is RagStepStatus.SUCCEEDED for item in store.finished_steps
    )
    assert store.running_step is None


async def test_hostile_dry_result_is_rejected_at_terminal_boundary():
    runner, store, writer, model, *_ = _runner(run=_run(dry_run=True))
    store.dry_mutator = lambda result: PageExecutionResult(
        replace(result.run, usage=RagUsage()),
        replace(result.page, attempt_count=result.page.attempt_count + 1),
    )

    with pytest.raises(TypeError, match="page runner state"):
        await runner.run(store.run, store.page, FakeLease())

    assert len(model.messages) == 1
    assert writer.calls == []


async def test_hostile_commit_result_is_rejected_at_terminal_boundary():
    runner, store, writer, model, *_ = _runner()
    writer.commit_mutator = lambda committed: replace(
        committed,
        run=replace(committed.run, goal="hostile replacement"),
    )

    with pytest.raises(TypeError, match="atomic page commit"):
        await runner.run(store.run, store.page, FakeLease())

    assert len(model.messages) == len(writer.calls) == 1


def test_atomic_page_commit_is_the_single_shared_contract():
    from rag.wiki_writer import AtomicPageCommit as WriterCommit

    assert WriterCommit is AtomicPageCommit
