"""Durable worker boundary and concrete hosted adapters for wiki builds."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID

from jobs import repository as jobs_repository
from jobs.handlers import RetryableJobError, TerminalJobError, WorkerContext
from jobs.models import JobCancelled, JobRecord, JobState, JobType, LeaseLost
from telemetry import emit

from llmwiki_core.rag import (
    RagCompletionReason,
    RagDomainError,
    RagPageState,
    RagStepStatus,
    RagStepType,
    RagUsage,
)
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

from . import repository
from .orchestrator import BuildWikiOrchestrator, RagRunFailure
from .ports import (
    AuthoritativeRagRun,
    OrchestratorPorts,
    PlanAcceptance,
    PlanAttempt,
    PlanAttemptBinding,
    WikiCatalogItem,
)
from .records import RagPageRecord, RagRunRecord, RagStepRecord

logger = logging.getLogger(__name__)

_FAILURES = MappingProxyType(
    {
        "rag_budget_exhausted": ("The RAG budget was exhausted.", False),
        "rag_disabled": ("Server-side RAG is disabled.", False),
        "rag_internal_error": ("The RAG request could not be completed.", True),
        "rag_invalid_draft": ("The generated draft was invalid.", False),
        "rag_invalid_plan": ("The generated plan was invalid.", False),
        "rag_job_binding_invalid": ("The RAG job binding is invalid.", False),
        "rag_model_unavailable": ("The RAG model is temporarily unavailable.", True),
        "rag_prompt_invalid": ("The RAG prompt inputs were invalid.", False),
        "rag_retrieval_failed": ("RAG retrieval is temporarily unavailable.", True),
        "rag_run_not_found": ("The RAG run was not found.", False),
        "rag_version_conflict": ("The conflict retry limit was exhausted.", False),
    }
)
_INTERNAL = ("rag_internal_error", "The RAG request could not be completed.")
_RESULT_REQUIRED = frozenset({"run_id", "completion_reason", "pages_committed"})
_RESULT_OPTIONAL = frozenset({"pages_skipped", "pages_dry_run"})


def _shape_run_id(job: JobRecord) -> UUID:
    invalid = type(job) is not JobRecord
    invalid = invalid or job.job_type is not JobType.BUILD_WIKI
    invalid = invalid or job.knowledge_base_id is None or job.document_id is not None
    invalid = invalid or set(job.payload) != {"run_id"}
    raw_run_id = job.payload.get("run_id") if type(job) is JobRecord else None
    try:
        run_id = UUID(raw_run_id) if isinstance(raw_run_id, str) else None
    except ValueError:
        run_id = None
    invalid = invalid or run_id is None or str(run_id) != raw_run_id
    if invalid:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    return run_id


def _exact_binding(job: JobRecord, run: RagRunRecord, run_id: UUID) -> bool:
    return (
        type(run) is RagRunRecord
        and run.id == run_id
        and run.job_id == job.id
        and run.user_id == job.user_id
        and run.knowledge_base_id == job.knowledge_base_id
    )


def _bounded_result(value: object, run: RagRunRecord) -> dict[str, str | int]:
    if type(value) is not dict or len(value) not in {3, 4, 5}:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    result = {key: value[key] for key in _RESULT_REQUIRED if key in value}
    result.update({key: value[key] for key in _RESULT_OPTIONAL if key in value})
    if len(result) != len(value) or not _RESULT_REQUIRED.issubset(result):
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    result_run_id = result.get("run_id")
    reason = result.get("completion_reason")
    if (
        type(result_run_id) is not str
        or result_run_id != str(run.id)
        or type(reason) is not str
        or reason not in {"completed", "no_work", "dry_run"}
    ):
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    for key in {"pages_committed", "pages_skipped", "pages_dry_run"}.intersection(result):
        count = result[key]
        if type(count) is not int or not 0 <= count <= run.budget.max_pages:
            raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    pages_committed = result["pages_committed"]
    pages_skipped = result.get("pages_skipped", 0)
    pages_dry_run = result.get("pages_dry_run", 0)
    invalid_relationship = reason == "completed" and (
        pages_committed < 1 or pages_dry_run != 0 or pages_skipped > pages_committed
    )
    invalid_relationship = invalid_relationship or (
        reason == "dry_run" and (pages_committed != 0 or pages_dry_run < 1 or pages_skipped > pages_dry_run)
    )
    invalid_relationship = invalid_relationship or (
        reason == "no_work" and (pages_committed != 0 or pages_skipped != 0 or pages_dry_run != 0)
    )
    if invalid_relationship:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    return result  # type: ignore[return-value]


def _authoritative_success_result(
    run: RagRunRecord,
    pages: object,
    reported: Mapping[str, str | int] | None = None,
) -> dict[str, str | int]:
    """Derive the public success projection from one bounded durable snapshot."""
    invalid = type(run) is not RagRunRecord or type(pages) is not tuple
    invalid = invalid or len(pages) > run.budget.max_pages
    invalid = invalid or any(
        type(page) is not RagPageRecord
        or page.run_id != run.id
        or page.user_id != run.user_id
        or page.knowledge_base_id != run.knowledge_base_id
        or page.ordinal != ordinal
        for ordinal, page in enumerate(pages)
    )
    reason = run.completion_reason
    if invalid or reason not in {
        RagCompletionReason.COMPLETED,
        RagCompletionReason.NO_WORK,
        RagCompletionReason.DRY_RUN,
    }:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    total = len(pages)
    if reason is RagCompletionReason.COMPLETED:
        invalid = total == 0 or run.last_committed_ordinal != total - 1
        invalid = invalid or any(page.state is not RagPageState.COMMITTED for page in pages)
        canonical: dict[str, str | int] = {
            "run_id": str(run.id),
            "completion_reason": reason.value,
            "pages_committed": total,
        }
        overlap_bound = total
    elif reason is RagCompletionReason.DRY_RUN:
        invalid = not run.dry_run or total == 0
        invalid = invalid or any(page.state is not RagPageState.DRY_RUN_COMPLETE for page in pages)
        canonical = {
            "run_id": str(run.id),
            "completion_reason": reason.value,
            "pages_committed": 0,
            "pages_dry_run": total,
        }
        overlap_bound = total
    else:
        invalid = total != 0 or run.last_committed_ordinal != -1
        canonical = {
            "run_id": str(run.id),
            "completion_reason": reason.value,
            "pages_committed": 0,
        }
        overlap_bound = 0
    if invalid:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    if reported is None:
        return canonical
    if (
        reported.get("run_id") != canonical["run_id"]
        or reported.get("completion_reason") != canonical["completion_reason"]
        or reported.get("pages_committed") != canonical["pages_committed"]
        or reported.get("pages_dry_run", 0) != canonical.get("pages_dry_run", 0)
    ):
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    pages_skipped = reported.get("pages_skipped", 0)
    if type(pages_skipped) is not int or not 0 <= pages_skipped <= overlap_bound:
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    return dict(reported)


def _run_fields(run: RagRunRecord, *, duration_ms: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run.id,
        "job_id": run.job_id,
        "model_profile": run.model_profile,
        "model_profile_version": run.model_profile_version,
        "page_count": max(0, run.last_committed_ordinal + 1),
        "step_count": run.usage.steps,
        "model_token_count": run.usage.model_tokens,
        "duration_ms": duration_ms,
    }


def _mapped_failure(failure: BaseException) -> BaseException:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None:
        return RetryableJobError(*_INTERNAL) if type(signal) is BaseException else signal
    if isinstance(failure, (JobCancelled, LeaseLost)):
        return failure
    if type(failure) in {RetryableJobError, TerminalJobError}:
        expected = _FAILURES.get(failure.error_code)
        retryable = type(failure) is RetryableJobError
        if expected is not None and retryable is expected[1]:
            error_type = RetryableJobError if retryable else TerminalJobError
            return error_type(failure.error_code, expected[0])
    if type(failure) is RagRunFailure:
        expected = _FAILURES.get(failure.code)
        retryable = failure.retryable
        if expected is not None and retryable is expected[1]:
            error_type = RetryableJobError if retryable else TerminalJobError
            return error_type(failure.code, expected[0])
    if type(failure) is RagDomainError:
        expected = _FAILURES.get(failure.code)
        if expected is not None and failure.retryable is expected[1]:
            error_type = RetryableJobError if failure.retryable else TerminalJobError
            return error_type(failure.code, expected[0])
    return RetryableJobError(*_INTERNAL)


async def handle_build_wiki(
    job: JobRecord,
    lease: object,
    context: WorkerContext,
) -> Mapping[str, str | int]:
    """Load one authoritative run and map its bounded orchestrator outcome."""
    run_id = _shape_run_id(job)
    load_failure: BaseException | None = None
    run = None
    try:
        async with context.pool.acquire() as conn:
            run = await repository.get_for_worker(conn, run_id, job.id)
    except BaseException as failure:  # noqa: BLE001 - sanitize database adapter failures.
        load_failure = failure
    if load_failure is not None:
        raise _mapped_failure(load_failure) from None
    if run is None:
        raise TerminalJobError("rag_run_not_found", "The RAG run was not found.")
    if not _exact_binding(job, run, run_id):
        raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
    factory = context.rag_orchestrator_factory
    if not callable(factory):
        raise TerminalJobError("rag_disabled", "Server-side RAG is disabled.")

    started = time.monotonic()
    emit(logger, "rag_run_started", **_run_fields(run, duration_ms=0))
    operation_failure: BaseException | None = None
    try:
        orchestrator = factory()
        operation = getattr(orchestrator, "run", None)
        if not callable(operation):
            raise TypeError("RAG orchestrator is invalid")
        raw_result = await operation(run, lease)
        result = _bounded_result(raw_result, run)
        async with context.pool.acquire() as conn, conn.transaction():
            snapshot = await repository.get_terminal_snapshot_for_worker(conn, job=job, run_id=run_id)
        completed, pages = snapshot if snapshot is not None else (None, ())
        if (
            completed is None
            or not _exact_binding(job, completed, run_id)
            or completed.completion_reason is None
            or completed.completion_reason.value != result["completion_reason"]
        ):
            raise TerminalJobError("rag_job_binding_invalid", "The RAG job binding is invalid.")
        result = _authoritative_success_result(completed, pages, result)
        run = completed
    except BaseException as failure:  # noqa: BLE001 - detach private adapter failures.
        operation_failure = failure
    if operation_failure is not None:
        raise _mapped_failure(operation_failure) from None

    emit(
        logger,
        "rag_run_finished",
        **_run_fields(run, duration_ms=_duration_ms(started)),
    )
    return result


def emit_rag_run_failed(transition: JobRecord, run: RagRunRecord) -> None:
    """Emit one terminal failure event from post-commit authoritative records."""
    run_id = _shape_run_id(transition)
    if (
        type(transition) is not JobRecord
        or transition.state is not JobState.FAILED
        or type(run) is not RagRunRecord
        or not _exact_binding(transition, run, run_id)
        or run.completion_reason is None
        or run.completion_reason.value not in {"budget_exhausted", "partial_failure"}
        or type(transition.error_code) is not str
    ):
        raise ValueError("terminal RAG telemetry records are invalid")
    emit(
        logger,
        "rag_run_failed",
        **_run_fields(run, duration_ms=0),
        error_code=transition.error_code,
    )


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1_000))


class _FeatureGate:
    def __init__(self, settings: object) -> None:
        self._settings = settings

    async def ensure_enabled(self) -> None:
        if getattr(self._settings, "SERVER_RAG_ENABLED", False) is not True:
            raise RagDomainError("rag_disabled", "Server-side RAG is disabled.")


class _DraftLinter:
    async def lint(self, *, run, page, content, citations):
        if (
            type(run) is not RagRunRecord
            or type(page) is not RagPageRecord
            or page.run_id != run.id
            or type(content) is not str
            or type(citations) is not tuple
        ):
            raise RagDomainError("rag_invalid_draft", "The generated draft was invalid.")
        return {"warnings": 0}


class _WikiCatalog:
    def __init__(self, database) -> None:
        self._database = database

    async def list_for_planning(self, *, user_id, knowledge_base_id, target_path_prefix, limit):
        rows = await self._database.fetch(
            "SELECT path,filename,title FROM documents WHERE user_id=$1 AND knowledge_base_id=$2 "
            "AND source_kind='wiki' AND NOT archived "
            "AND left(path,char_length($3))=$3 ORDER BY path,filename LIMIT $4",
            user_id,
            knowledge_base_id,
            target_path_prefix,
            limit,
        )
        if len(rows) > limit:
            raise RuntimeError("RAG catalog exceeded its bound")
        return tuple(WikiCatalogItem(f"{row['path']}{row['filename']}", row["title"]) for row in rows)


class _TelemetryWikiWriter:
    def __init__(self, writer) -> None:
        self._writer = writer

    async def commit(self, **kwargs):
        started = time.monotonic()
        committed = await self._writer.commit(**kwargs)
        emit(
            logger,
            "rag_page_committed",
            schema_version=1,
            run_id=committed.run.id,
            job_id=committed.run.job_id,
            page_id=committed.page.id,
            document_id=committed.page.document_id,
            model_profile=committed.run.model_profile,
            model_profile_version=committed.run.model_profile_version,
            page_count=committed.run.last_committed_ordinal + 1,
            step_count=committed.run.usage.steps,
            model_token_count=committed.run.usage.model_tokens,
            latency_ms=float(_duration_ms(started)),
        )
        return committed


class _TelemetryPageStore:
    """Delegate page mutations and emit only their committed terminal records."""

    def __init__(self, store) -> None:
        self._store = store

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    async def finish_step(self, **kwargs):
        finished = await self._store.finish_step(**kwargs)
        _emit_step(kwargs["run"], finished)
        return finished


class _PostgresRunStore:
    def __init__(self, pool, *, run: RagRunRecord, owner: str) -> None:
        self._pool = pool
        self._run_id = run.id
        self._job_id = run.job_id
        self._owner = owner

    async def _active(self, conn) -> AuthoritativeRagRun:
        job = await jobs_repository.assert_active(conn, self._job_id, self._owner)
        run = await repository.get_for_worker(conn, self._run_id, self._job_id)
        if run is None:
            raise RagDomainError("rag_run_not_found", "The RAG run was not found.")
        return AuthoritativeRagRun(run, job)

    async def reload(self, run_id: UUID) -> AuthoritativeRagRun:
        if run_id != self._run_id:
            raise RagDomainError("rag_run_mismatch", "The supplied RAG run does not match persisted identity.")
        async with self._pool.acquire() as conn, conn.transaction():
            return await self._active(conn)

    async def list_pages(self, run_id: UUID) -> tuple[RagPageRecord, ...]:
        if run_id != self._run_id:
            raise RagDomainError("rag_run_mismatch", "The supplied RAG run does not match persisted identity.")
        async with self._pool.acquire() as conn:
            return await repository.list_pages(conn, run_id)

    async def begin_plan_attempt(self, *, run, job, lease_owner, specs):
        async with self._pool.acquire() as conn, conn.transaction():
            state = await self._active(conn)
            _require_store_snapshot(state, run, job, lease_owner)
            rows = await conn.fetch(
                "SELECT * FROM rag_steps WHERE run_id=$1 AND run_page_id IS NULL "
                "AND step_type='plan' ORDER BY sequence FOR UPDATE",
                run.id,
            )
            steps = tuple(repository._decode_db_step(row) for row in rows)
            running = [step for step in steps if step.status is RagStepStatus.RUNNING]
            if len(running) > 1:
                raise RagDomainError("rag_step_already_running", "A RAG step is already running.", True)
            if running:
                step = running[0]
                spec = _matching_spec(step, specs)
                return PlanAttempt(state.run, step, spec, True)
            terminal = [step for step in steps if step.status in {RagStepStatus.SUCCEEDED, RagStepStatus.FAILED}]
            if len(terminal) >= len(specs):
                raise RagDomainError("rag_invalid_plan", "The generated plan was invalid.")
            spec = specs[len(terminal)]
            step = await repository.start_step(
                conn,
                run_id=run.id,
                page_id=None,
                step_type=RagStepType.PLAN,
                input_digest=spec.input_digest,
                prompt_version=spec.prompt_version,
                prompt_digest=spec.prompt_digest,
            )
            return PlanAttempt(state.run, step, spec, False)

    async def bind_plan_attempt_input(self, *, run, job, lease_owner, attempt, input_digest):
        finished_to_emit = None
        async with self._pool.acquire() as conn, conn.transaction():
            state = await self._active(conn)
            _require_store_snapshot(state, run, job, lease_owner)
            row = await conn.fetchrow(
                "SELECT * FROM rag_steps WHERE id=$1 AND run_id=$2 AND status='running' FOR UPDATE",
                attempt.step.id,
                run.id,
            )
            if row is None:
                raise RagDomainError("rag_step_not_running", "The RAG step is not running.")
            step = repository._decode_db_step(row)
            if step.input_digest == input_digest:
                return PlanAttemptBinding(state.run, PlanAttempt(state.run, step, attempt.spec, True), False)
            if step.input_digest == attempt.spec.input_digest:
                updated = await conn.fetchrow(
                    "UPDATE rag_steps SET input_digest=$2 WHERE id=$1 AND status='running' RETURNING *",
                    step.id,
                    input_digest,
                )
                bound = repository._decode_db_step(updated)
                return PlanAttemptBinding(state.run, PlanAttempt(state.run, bound, attempt.spec, True), False)
            zero = RagUsage(steps=1, model_tokens=0)
            finished = await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.FAILED,
                summary={"outcome": "failed", "usage_trusted": False},
                citations=(),
                usage=zero,
                latency_ms=0,
                error_code="rag_plan_input_changed",
                prompt_version=attempt.spec.prompt_version,
                prompt_digest=attempt.spec.prompt_digest,
            )
            next_usage = RagUsage(state.run.usage.steps + 1, state.run.usage.model_tokens)
            current = await _update_usage(conn, state.run, next_usage)
            finished_to_emit = finished
            if next_usage.steps >= current.budget.max_steps:
                result = PlanAttemptBinding(current, None, True)
            else:
                replacement = await repository.start_step(
                    conn,
                    run_id=run.id,
                    page_id=None,
                    step_type=RagStepType.PLAN,
                    input_digest=input_digest,
                    prompt_version=attempt.spec.prompt_version,
                    prompt_digest=attempt.spec.prompt_digest,
                )
                result = PlanAttemptBinding(current, PlanAttempt(current, replacement, attempt.spec, False), False)
        if finished_to_emit is not None:
            _emit_step(current, finished_to_emit)
        return result

    async def finish_plan_attempt(self, **kwargs):
        async with self._pool.acquire() as conn, conn.transaction():
            state = await self._active(conn)
            _require_store_snapshot(state, kwargs["run"], kwargs["job"], kwargs["lease_owner"])
            finished = await repository.finish_step(
                conn,
                step_id=kwargs["step"].id,
                status=kwargs["status"],
                summary=kwargs["summary"],
                citations=kwargs["citations"],
                usage=RagUsage(1, kwargs["token_usage"].total_tokens),
                latency_ms=kwargs["latency_ms"],
                error_code=kwargs["error_code"],
                token_usage=kwargs["token_usage"],
                prompt_version=kwargs["prompt_version"],
                prompt_digest=kwargs["prompt_digest"],
            )
            updated = await _update_usage(conn, state.run, kwargs["aggregate_usage"])
        _emit_step(updated, finished)
        return updated

    async def accept_plan(self, **kwargs):
        async with self._pool.acquire() as conn, conn.transaction():
            state = await self._active(conn)
            _require_store_snapshot(state, kwargs["run"], kwargs["job"], kwargs["lease_owner"])
            finished = await repository.finish_step(
                conn,
                step_id=kwargs["step"].id,
                status=RagStepStatus.SUCCEEDED,
                summary=kwargs["summary"],
                citations=(),
                usage=RagUsage(1, kwargs["token_usage"].total_tokens),
                latency_ms=kwargs["latency_ms"],
                token_usage=kwargs["token_usage"],
                prompt_version=kwargs["prompt_version"],
                prompt_digest=kwargs["prompt_digest"],
            )
            updated = await _update_usage(conn, state.run, kwargs["aggregate_usage"])
            pages = await repository.insert_worklist(conn, updated, kwargs["items"]) if kwargs["items"] else ()
            if kwargs["completion_reason"] is not None:
                updated = await repository.finish_run(
                    conn,
                    run_id=updated.id,
                    completion_reason=kwargs["completion_reason"],
                    usage=kwargs["aggregate_usage"],
                )
        _emit_step(updated, finished)
        return PlanAcceptance(updated, pages, kwargs["completion_reason"])

    async def finish_run(self, *, run, job, lease_owner, completion_reason, usage):
        async with self._pool.acquire() as conn, conn.transaction():
            state = await self._active(conn)
            _require_store_snapshot(state, run, job, lease_owner)
            return await repository.finish_run(
                conn,
                run_id=run.id,
                completion_reason=completion_reason,
                usage=usage,
            )


def _matching_spec(step: RagStepRecord, specs):
    matched = [
        spec
        for spec in specs
        if step.prompt_version == spec.prompt_version and step.prompt_digest == spec.prompt_digest
    ]
    if len(matched) != 1:
        raise RagDomainError("rag_invalid_plan", "The generated plan was invalid.")
    return matched[0]


def _require_store_snapshot(state, run, job, owner) -> None:
    current = state.job
    if (
        state.run != run
        or type(job) is not JobRecord
        or current.id != job.id
        or current.job_type is not JobType.BUILD_WIKI
        or current.state is not JobState.RUNNING
        or current.user_id != job.user_id
        or current.knowledge_base_id != job.knowledge_base_id
        or current.document_id is not None
        or dict(current.payload) != dict(job.payload)
        or current.lease_owner != owner
        or job.lease_owner != owner
    ):
        raise LeaseLost("RAG job lease is no longer active")


async def _update_usage(conn, run: RagRunRecord, usage: RagUsage) -> RagRunRecord:
    row = await conn.fetchrow(
        "UPDATE rag_runs SET usage=$2::jsonb WHERE id=$1 AND completion_reason IS NULL RETURNING *",
        run.id,
        json.dumps({"steps": usage.steps, "model_tokens": usage.model_tokens}, separators=(",", ":")),
    )
    if row is None:
        raise RagDomainError("rag_run_finished", "The RAG run is already finished.")
    return repository._decode_db_run(row)


def _emit_step(run: RagRunRecord, step: RagStepRecord) -> None:
    emit(
        logger,
        "rag_step_finished",
        schema_version=1,
        run_id=run.id,
        job_id=run.job_id,
        step_id=step.id,
        page_id=step.run_page_id,
        model_profile=run.model_profile,
        model_profile_version=run.model_profile_version,
        step_type=step.step_type,
        outcome=step.status,
        model_token_count=step.total_tokens,
        latency_ms=step.latency_ms,
        error_code=step.error_code,
    )


async def _close_owned_model(model: object) -> tuple[tuple[BaseException, ...], bool]:
    """Make at most two idempotent close attempts, stopping on a control."""
    failures: list[BaseException] = []
    for _attempt in range(2):
        close_failure: BaseException | None = None
        try:
            await model.aclose()  # type: ignore[attr-defined]
        except BaseException as caught:  # noqa: BLE001 - bounded idempotent cleanup retry.
            close_failure = caught
        if close_failure is None:
            return tuple(failures), True
        failures.append(close_failure)
        close_signal = sanitized_boundary_signal_or_unknown(close_failure)
        if (close_signal is not None and type(close_signal) is not BaseException) or not isinstance(
            close_failure, Exception
        ):
            break
    return tuple(failures), False


class _RunScopedOrchestrator:
    def __init__(self, pool, settings: object, profiles: Mapping[str, object]) -> None:
        self._pool = pool
        self._settings = settings
        self._profiles = profiles

    async def run(self, run: RagRunRecord, lease: object):
        from .model import OpenAICompatibleRagModel, ResolvedRagModelProfile
        from .page_runner import PageRunner, PostgresPageStore
        from .retrieval import HostedRagRetrieval, PostgresEvidenceReader, PostgresWikiPageReader
        from .wiki_writer import PostgresRagWikiWriter

        profile = self._profiles.get(run.model_profile)
        if (
            not isinstance(profile, ResolvedRagModelProfile)
            or profile.name != run.model_profile
            or profile.version != run.model_profile_version
        ):
            raise RagRunFailure("rag_model_unavailable", "The RAG model is temporarily unavailable.", True)
        owner = getattr(lease, "owner", None) or getattr(lease, "_owner", None)
        if not isinstance(owner, str) or not owner:
            owner = ""
        if not owner:
            checkpoint = await lease.checkpoint()
            owner = checkpoint.lease_owner if type(checkpoint) is JobRecord else None
        if not isinstance(owner, str) or not owner:
            raise LeaseLost("RAG job lease is no longer active")
        operation_failure: BaseException | None = None
        close_failures: tuple[BaseException, ...] = ()
        cleanup_succeeded = False
        result = None
        model = OpenAICompatibleRagModel(profile, api_key=profile.api_key.get_secret_value())
        try:
            try:
                store = _PostgresRunStore(self._pool, run=run, owner=owner)
                retrieval = HostedRagRetrieval(
                    self._pool,
                    user_id=run.user_id,
                    knowledge_base_id=run.knowledge_base_id,
                    settings=self._settings,
                )
                evidence = PostgresEvidenceReader(self._pool)
                reader = PostgresWikiPageReader(self._pool)
                feature_gate = _FeatureGate(self._settings)
                writer = _TelemetryWikiWriter(PostgresRagWikiWriter(self._pool))
                page_runner = PageRunner(
                    store=_TelemetryPageStore(PostgresPageStore(self._pool)),
                    model=model,
                    retrieval=retrieval,
                    evidence_reader=evidence,
                    wiki_page_reader=reader,
                    wiki_writer=writer,
                    draft_linter=_DraftLinter(),
                    feature_gate=feature_gate,
                )
                orchestrator = BuildWikiOrchestrator(
                    OrchestratorPorts(
                        store=store,
                        model=model,
                        retrieval=retrieval,
                        evidence_reader=evidence,
                        wiki_catalog=_WikiCatalog(self._pool),
                        wiki_page_reader=reader,
                        wiki_writer=writer,
                        draft_linter=_DraftLinter(),
                        feature_gate=feature_gate,
                        page_runner=page_runner,
                    )
                )
                result = await orchestrator.run(run, lease)
            except BaseException as caught:  # noqa: BLE001 - preserve after owned model cleanup.
                operation_failure = caught
        finally:
            close_failures, cleanup_succeeded = await _close_owned_model(model)

        failures = (() if operation_failure is None else (operation_failure,)) + close_failures
        signal = sanitized_boundary_signal_or_unknown(*failures)
        if signal is not None and type(signal) is not BaseException:
            del failures, close_failures, operation_failure
            raise signal from None
        if operation_failure is not None:
            preserved_failure = operation_failure
            del failures, close_failures, operation_failure
            raise preserved_failure
        if close_failures and not cleanup_succeeded:
            del failures, close_failures
            raise RagRunFailure("rag_internal_error", "The RAG request could not be completed.", True) from None
        return result


def build_rag_orchestrator_factory(pool, settings: object, profiles: Mapping[str, object]):
    """Return a no-argument factory after startup has resolved all profiles."""
    from llmwiki_core.telemetry import validated_event

    from .model import ResolvedRagModelProfile

    if not profiles:
        raise RuntimeError("server RAG requires configured model profiles")
    for name, profile in profiles.items():
        if type(name) is not str or not isinstance(profile, ResolvedRagModelProfile) or profile.name != name:
            raise RuntimeError("server RAG model profile configuration is invalid")
        validated_event(
            "rag_run_started",
            schema_version=1,
            run_id=UUID(int=0),
            job_id=UUID(int=0),
            model_profile=profile.name,
            model_profile_version=profile.version,
            page_count=0,
            step_count=0,
            model_token_count=0,
            duration_ms=0,
        )

    def factory() -> _RunScopedOrchestrator:
        return _RunScopedOrchestrator(pool, settings, profiles)

    return factory


__all__ = ["build_rag_orchestrator_factory", "emit_rag_run_failed", "handle_build_wiki"]
