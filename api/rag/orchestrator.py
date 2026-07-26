"""Durable, bounded state machine for planning and dispatching wiki builds."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from time import perf_counter
from types import MappingProxyType
from typing import Any, NoReturn
from uuid import UUID

from jobs.models import JobCancelled, JobRecord, JobState, JobType, LeaseLost

from llmwiki_core.rag import (
    RagBudget,
    RagCompletionReason,
    RagDomainError,
    RagPageState,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
    validate_worklist,
)
from llmwiki_core.search import RetrieverUnavailable, SearchQuery, SearchResult
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

from .model import InvalidRagModelResponse, RagModelResponse, RagModelUnavailable, RagTokenUsage
from .ports import (
    AuthoritativeRagRun,
    LeaseCheckpoint,
    OrchestratorPorts,
    PageExecutionResult,
    PlanAcceptance,
    PlanAttempt,
    PlanAttemptBinding,
    PlanAttemptSpec,
    WikiCatalogItem,
)
from .prompts import (
    PLANNER_PROMPT_DIGEST,
    PLANNER_PROMPT_VERSION,
    build_planner_messages,
    parse_plan,
)
from .records import RagPageRecord, RagRunRecord, RagStepRecord
from .retrieval import RagEvidence

_PLAN_OUTPUT_TOKENS_PER_PAGE = 256
_MIN_PLAN_OUTPUT_TOKENS = 256
_MAX_PLAN_OUTPUT_TOKENS = 4_096
_MAX_PLANNER_CATALOG_ITEMS = 128
_REPAIR_SUFFIX = "repair-v1"
PLANNER_REPAIR_PROMPT_VERSION = f"{PLANNER_PROMPT_VERSION}-{_REPAIR_SUFFIX}"
_REPAIR_MESSAGE = MappingProxyType(
    {
        "role": "system",
        "content": (
            "The previous response did not satisfy the required schema. "
            "Return a new complete JSON object following the original policy."
        ),
    }
)
_REPAIR_TEMPLATE = MappingProxyType(
    {
        "base_prompt_digest": PLANNER_PROMPT_DIGEST,
        "base_prompt_version": PLANNER_PROMPT_VERSION,
        "repair_message": _REPAIR_MESSAGE,
        "repair_prompt_version": PLANNER_REPAIR_PROMPT_VERSION,
    }
)
PLANNER_REPAIR_PROMPT_DIGEST = hashlib.sha256(
    json.dumps(
        {
            "base_prompt_digest": _REPAIR_TEMPLATE["base_prompt_digest"],
            "base_prompt_version": _REPAIR_TEMPLATE["base_prompt_version"],
            "repair_message": dict(_REPAIR_MESSAGE),
            "repair_prompt_version": _REPAIR_TEMPLATE["repair_prompt_version"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()
_ZERO_TOKEN_USAGE = RagTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
_INTERNAL_CODE = "rag_internal_error"
_INTERNAL_MESSAGE = "The RAG request could not be completed."
_INVALID_PLAN_MESSAGE = "The generated plan was invalid."


class RagRunFailure(Exception):
    """A detached, stable failure safe for the durable job boundary."""

    __slots__ = ("_code", "_public_message", "_retryable")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("RagRunFailure cannot be subclassed")

    def __init__(self, code: str, public_message: str, retryable: bool) -> None:
        if (
            type(code) is not str
            or not code.startswith("rag_")
            or not 1 <= len(code) <= 128
            or type(public_message) is not str
            or not 1 <= len(public_message) <= 500
            or type(retryable) is not bool
        ):
            raise ValueError("RAG run failure is invalid")
        Exception.__init__(self, public_message)
        object.__setattr__(self, "_code", code)
        object.__setattr__(self, "_public_message", public_message)
        object.__setattr__(self, "_retryable", retryable)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"__traceback__", "__cause__", "__context__", "__suppress_context__"}:
            Exception.__setattr__(self, name, value)
            return
        raise AttributeError("RagRunFailure is immutable")

    @property
    def code(self) -> str:
        return self._code

    @property
    def public_message(self) -> str:
        return self._public_message

    @property
    def retryable(self) -> bool:
        return self._retryable

    def __eq__(self, other: object) -> bool:
        return type(other) is RagRunFailure and (
            self.code,
            self.public_message,
            self.retryable,
        ) == (other.code, other.public_message, other.retryable)

    def __hash__(self) -> int:
        return hash((self.code, self.public_message, self.retryable))

    def __repr__(self) -> str:
        return (
            f"RagRunFailure(code={self.code!r}, public_message={self.public_message!r}, retryable={self.retryable!r})"
        )


class _InvalidPlanAttempt(Exception):
    pass


class _BoundPlanBudgetExhausted(Exception):
    pass


class BuildWikiOrchestrator:
    """Plan once, resume from durable pages, and dispatch bounded page work."""

    def __init__(self, ports: OrchestratorPorts) -> None:
        if type(ports) is not OrchestratorPorts:
            raise TypeError("orchestrator ports are invalid")
        self._ports = ports

    def run(
        self,
        run: UUID | RagRunRecord,
        lease: LeaseCheckpoint,
    ) -> Any:
        return _run_public(self, run, lease)

    async def _run(
        self,
        run: UUID | RagRunRecord,
        lease: LeaseCheckpoint,
    ) -> dict[str, str | int]:
        run_id, supplied = _run_identity(run)
        state = await self._checkpoint(run_id, lease)
        if supplied is not None and not _same_run_identity(supplied, state.run):
            raise _internal_failure()
        pages = await self._list_pages(state.run)
        if state.run.completion_reason is not None:
            return _terminal_replay_result(state.run, pages)
        if not pages:
            if state.run.parent_run_id is not None:
                raise _internal_failure()
            state, pages = await self._ensure_plan(state, lease)
        if not pages:
            return _terminal_replay_result(state.run, pages)
        return await self._run_pages(state, pages, lease)

    async def _checkpoint(
        self,
        run_id: UUID,
        lease: LeaseCheckpoint,
    ) -> AuthoritativeRagRun:
        lease_job = await _boundary_call(lease, "checkpoint")
        if type(lease_job) is not JobRecord:
            raise _internal_failure()
        state = await _boundary_call(self._ports.store, "reload", run_id)
        if type(state) is not AuthoritativeRagRun:
            raise _internal_failure()
        _validate_authoritative_state(state, lease_job, run_id)
        completion = state.run.completion_reason
        if completion is not None:
            if type(completion) is not RagCompletionReason or completion not in {
                RagCompletionReason.NO_WORK,
                RagCompletionReason.COMPLETED,
                RagCompletionReason.DRY_RUN,
            }:
                raise _internal_failure()
            return state
        enabled = await _boundary_call(
            self._ports.feature_gate,
            "ensure_enabled",
            _allowed_domain_codes=frozenset({"rag_disabled"}),
        )
        if enabled is not None:
            raise _internal_failure()
        _require_within_budget(state.run)
        return state

    async def _list_pages(self, run: RagRunRecord) -> tuple[RagPageRecord, ...]:
        pages = await _boundary_call(self._ports.store, "list_pages", run.id)
        return _validate_pages(run, pages)

    async def _ensure_plan(
        self,
        state: AuthoritativeRagRun,
        lease: LeaseCheckpoint,
    ) -> tuple[AuthoritativeRagRun, tuple[RagPageRecord, ...]]:
        for _ in range(2):
            state = await self._checkpoint(state.run.id, lease)
            if await self._list_pages(state.run):
                raise _internal_failure()
            try:
                return await self._plan_attempt(state, lease)
            except _InvalidPlanAttempt:
                state = await self._checkpoint(state.run.id, lease)
            except RagDomainError as failure:
                if type(failure) is RagDomainError and failure.code == "rag_invalid_plan":
                    raise RagRunFailure("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False) from None
                raise
        raise RagRunFailure("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False)

    async def _plan_attempt(
        self,
        state: AuthoritativeRagRun,
        lease: LeaseCheckpoint,
    ) -> tuple[AuthoritativeRagRun, tuple[RagPageRecord, ...]]:
        run = state.run
        minimum_output = min(
            _MAX_PLAN_OUTPUT_TOKENS,
            max(_MIN_PLAN_OUTPUT_TOKENS, run.budget.max_pages * _PLAN_OUTPUT_TOKENS_PER_PAGE),
        )
        specs = _plan_attempt_specs(run)
        attempt = await _boundary_call(
            self._ports.store,
            "begin_plan_attempt",
            _allowed_domain_codes=frozenset({"rag_budget_exhausted", "rag_invalid_plan"}),
            run=run,
            job=state.job,
            lease_owner=_lease_owner(state.job),
            specs=specs,
        )
        run, step = _validate_plan_attempt(
            attempt,
            run,
            specs,
        )
        run.usage.consume_step(run.budget).reserve_model_call(run.budget, minimum_output)
        spec = attempt.spec
        started = perf_counter()
        reported_tokens = _ZERO_TOKEN_USAGE
        attempt_failure: BaseException | None = None
        invalid_code: str | None = None
        invalid_fatal = False
        reservation = 0
        try:
            catalog = await self._planning_catalog(run, lease)
            result = await self._planning_retrieval(run, lease)
            evidence = await self._planning_evidence(run, result, lease)
            messages = build_planner_messages(
                goal=run.goal,
                target_path_prefix=run.target_path_prefix,
                catalog=catalog,
                evidence=evidence,
            )
            if spec.repair:
                messages = (*messages, dict(_REPAIR_MESSAGE))
            input_digest = _planning_messages_digest(run, spec, messages)
            await self._planning_checkpoint(run, lease, run.usage.consume_step(run.budget), None)
            binding = await _boundary_call(
                self._ports.store,
                "bind_plan_attempt_input",
                run=run,
                job=state.job,
                lease_owner=_lease_owner(state.job),
                attempt=attempt,
                input_digest=input_digest,
            )
            run, bound_step = _validate_plan_attempt_binding(binding, run, spec, step, input_digest)
            if bound_step is None:
                raise _BoundPlanBudgetExhausted
            step = bound_step
            next_usage = run.usage.consume_step(run.budget)
            next_usage.reserve_model_call(run.budget, minimum_output)
            reservation, max_output_tokens = _reserve_plan_call(run, next_usage, messages)
            await self._planning_checkpoint(run, lease, next_usage, reservation)
            response = await _boundary_call(
                self._ports.model,
                "complete_json",
                messages=messages,
                max_output_tokens=max_output_tokens,
                timeout_seconds=float(run.budget.per_call_timeout_seconds),
            )
            items, reported_tokens, committed_usage, invalid_code, invalid_fatal = _validated_plan_response(
                response,
                run,
                next_usage,
                reservation,
            )
            del response, messages, evidence, result, catalog
        except BaseException as caught:  # noqa: BLE001 - classify after leaving active context.
            attempt_failure = caught

        if type(attempt_failure) is _BoundPlanBudgetExhausted:
            raise _budget_failure()
        aggregate_usage = _aggregate_plan_usage(run, reported_tokens)
        if attempt_failure is not None:
            if (signal := sanitized_boundary_signal_or_unknown(attempt_failure)) and type(signal) is not BaseException:
                raise signal
            if isinstance(attempt_failure, (JobCancelled, LeaseLost)):
                raise attempt_failure
            reported_tokens, aggregate_usage, usage_trusted, charged_tokens = _failure_token_accounting(
                attempt_failure,
                run,
                reservation,
            )
            mapped = _map_failure(attempt_failure, operation="plan")
            fenced = await self._post_model_checkpoint(run, lease, aggregate_usage)
            await self._finish_failed_plan(
                fenced,
                step,
                reported_tokens,
                aggregate_usage,
                started,
                spec.prompt_version,
                spec.prompt_digest,
                mapped.code,
                usage_trusted=usage_trusted,
                charged_tokens=charged_tokens,
            )
            raise mapped

        fenced = await self._post_model_checkpoint(run, lease, committed_usage)
        if invalid_code is not None:
            await self._finish_failed_plan(
                fenced,
                step,
                reported_tokens,
                committed_usage,
                started,
                spec.prompt_version,
                spec.prompt_digest,
                invalid_code,
                usage_trusted=not invalid_fatal,
                charged_tokens=committed_usage.model_tokens - run.usage.model_tokens,
            )
            if invalid_fatal or spec.repair:
                raise RagRunFailure("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False)
            raise _InvalidPlanAttempt()

        completion = RagCompletionReason.NO_WORK if not items else None
        accepted = await _boundary_call(
            self._ports.store,
            "accept_plan",
            run=fenced.run,
            job=fenced.job,
            lease_owner=_lease_owner(fenced.job),
            step=step,
            items=items,
            summary={"accepted_pages": len(items), "repair": spec.repair},
            token_usage=reported_tokens,
            aggregate_usage=committed_usage,
            latency_ms=_latency_ms(started),
            prompt_version=spec.prompt_version,
            prompt_digest=spec.prompt_digest,
            completion_reason=completion,
        )
        accepted_state = _validate_plan_acceptance(accepted, run, items, committed_usage, completion)
        return AuthoritativeRagRun(accepted_state[0], fenced.job), accepted_state[1]

    async def _planning_catalog(
        self,
        run: RagRunRecord,
        lease: LeaseCheckpoint,
    ) -> tuple[WikiCatalogItem, ...]:
        await self._planning_checkpoint(run, lease, run.usage.consume_step(run.budget), None)
        value = await _boundary_call(
            self._ports.wiki_catalog,
            "list_for_planning",
            user_id=run.user_id,
            knowledge_base_id=run.knowledge_base_id,
            target_path_prefix=run.target_path_prefix,
            limit=min(_MAX_PLANNER_CATALOG_ITEMS, run.budget.max_pages * 4),
        )
        if type(value) is not tuple or any(type(item) is not WikiCatalogItem for item in value):
            raise _internal_failure()
        if len(value) > min(_MAX_PLANNER_CATALOG_ITEMS, run.budget.max_pages * 4):
            raise _internal_failure()
        return value

    async def _planning_retrieval(
        self,
        run: RagRunRecord,
        lease: LeaseCheckpoint,
    ) -> SearchResult:
        await self._planning_checkpoint(run, lease, run.usage.consume_step(run.budget), None)
        limit = min(100, max(1, run.budget.max_pages * 4))
        result = await _boundary_call(
            self._ports.retrieval,
            "retrieve",
            SearchQuery(run.goal, limit=limit),
            profile=run.retrieval_profile,
        )
        if type(result) is not SearchResult:
            raise _internal_failure()
        return result

    async def _planning_evidence(
        self,
        run: RagRunRecord,
        result: SearchResult,
        lease: LeaseCheckpoint,
    ) -> tuple[RagEvidence, ...]:
        await self._planning_checkpoint(run, lease, run.usage.consume_step(run.budget), None)
        evidence = await _boundary_call(
            self._ports.evidence_reader,
            "read",
            run.user_id,
            run.knowledge_base_id,
            result.hits,
            run.budget.max_context_chars,
        )
        if type(evidence) is not tuple or any(type(item) is not RagEvidence for item in evidence):
            raise _internal_failure()
        if sum(len(item.content) for item in evidence) > run.budget.max_context_chars:
            raise _internal_failure()
        return evidence

    async def _planning_checkpoint(
        self,
        run: RagRunRecord,
        lease: LeaseCheckpoint,
        usage: RagUsage,
        reservation: int | None,
    ) -> None:
        state = await self._checkpoint(run.id, lease)
        if (
            not _same_run_identity(state.run, run)
            or state.run.usage != run.usage
            or state.run.completion_reason is not None
        ):
            raise _internal_failure()
        if usage.steps > run.budget.max_steps or usage.model_tokens > run.budget.max_model_tokens:
            raise _budget_failure()
        if reservation is not None:
            usage.reserve_model_call(run.budget, reservation)

    async def _post_model_checkpoint(
        self,
        run: RagRunRecord,
        lease: LeaseCheckpoint,
        aggregate_usage: RagUsage,
    ) -> AuthoritativeRagRun:
        state = await self._checkpoint(run.id, lease)
        if (
            not _same_run_identity(state.run, run)
            or state.run.usage != run.usage
            or state.run.completion_reason is not None
            or aggregate_usage.steps != run.usage.steps + 1
            or aggregate_usage.steps > run.budget.max_steps
            or aggregate_usage.model_tokens < run.usage.model_tokens
            or aggregate_usage.model_tokens > run.budget.max_model_tokens
        ):
            raise _internal_failure()
        return state

    async def _finish_failed_plan(
        self,
        state: AuthoritativeRagRun,
        step: RagStepRecord,
        token_usage: RagTokenUsage,
        aggregate_usage: RagUsage,
        started: float,
        prompt_version: str,
        prompt_digest: str,
        error_code: str,
        *,
        usage_trusted: bool = True,
        charged_tokens: int | None = None,
    ) -> None:
        charged = token_usage.total_tokens if charged_tokens is None else charged_tokens
        result = await _boundary_call(
            self._ports.store,
            "finish_plan_attempt",
            run=state.run,
            job=state.job,
            lease_owner=_lease_owner(state.job),
            step=step,
            status=RagStepStatus.FAILED,
            summary={
                "accepted_pages": 0,
                "charged_tokens": charged,
                "outcome": "failed",
                "usage_trusted": usage_trusted,
            },
            citations=(),
            token_usage=token_usage,
            aggregate_usage=aggregate_usage,
            latency_ms=_latency_ms(started),
            error_code=error_code,
            prompt_version=prompt_version,
            prompt_digest=prompt_digest,
        )
        if (
            type(result) is not RagRunRecord
            or not _same_run_identity(result, state.run)
            or result.usage != aggregate_usage
            or result.completion_reason is not None
        ):
            raise _internal_failure()

    async def _run_pages(
        self,
        initial: AuthoritativeRagRun,
        initial_pages: tuple[RagPageRecord, ...],
        lease: LeaseCheckpoint,
    ) -> dict[str, str | int]:
        pages_skipped = sum(page.ordinal <= initial.run.last_committed_ordinal for page in initial_pages)
        run = initial.run
        for ordinal in range(run.last_committed_ordinal + 1, len(initial_pages)):
            state = await self._checkpoint(run.id, lease)
            pages = await self._list_pages(state.run)
            page = pages[ordinal]
            if page.state in {RagPageState.COMMITTED, RagPageState.DRY_RUN_COMPLETE}:
                pages_skipped += 1
                run = state.run
                continue
            if page.state not in {RagPageState.PLANNED, RagPageState.RUNNING, RagPageState.FAILED}:
                raise _internal_failure()
            _require_page_budget(state.run)
            outcome = await _boundary_call(self._ports.page_runner, "run", state.run, page, lease)
            if type(outcome) is not PageExecutionResult:
                raise _internal_failure()
            _validate_page_outcome(state.run, page, outcome)
            run = outcome.run

        final = await self._checkpoint(run.id, lease)
        final_pages = await self._list_pages(final.run)
        completion = RagCompletionReason.DRY_RUN if final.run.dry_run else RagCompletionReason.COMPLETED
        _validate_finished_pages(final.run, final_pages, completion)
        finished = await _boundary_call(
            self._ports.store,
            "finish_run",
            run=final.run,
            job=final.job,
            lease_owner=_lease_owner(final.job),
            completion_reason=completion,
            usage=final.run.usage,
        )
        if (
            type(finished) is not RagRunRecord
            or not _same_run_identity(finished, final.run)
            or finished.usage != final.run.usage
            or finished.completion_reason is not completion
        ):
            raise _internal_failure()
        total_committed = sum(page.state is RagPageState.COMMITTED for page in final_pages)
        total_dry_run = sum(page.state is RagPageState.DRY_RUN_COMPLETE for page in final_pages)
        result: dict[str, str | int] = {
            "run_id": str(finished.id),
            "completion_reason": completion.value,
            "pages_committed": total_committed,
        }
        if pages_skipped:
            result["pages_skipped"] = pages_skipped
        if total_dry_run:
            result["pages_dry_run"] = total_dry_run
        return result


async def _run_public(
    orchestrator: BuildWikiOrchestrator,
    run_value: UUID | RagRunRecord,
    lease: LeaseCheckpoint,
) -> dict[str, str | int]:
    try:
        return await orchestrator._run(run_value, lease)
    except BaseException as caught:  # noqa: BLE001 - reduce to safe scalars before public raise.
        safe = _public_failure_scalars(caught)
    del orchestrator, run_value, lease
    return _raise_public_scalars(*safe)


async def _boundary_call(
    target: object,
    method_name: str,
    *args: object,
    _allowed_domain_codes: frozenset[str] = frozenset(),
    **kwargs: object,
) -> Any:
    failure: BaseException | None = None
    try:
        method = getattr(target, method_name)
        if not callable(method):
            raise TypeError
        return await method(*args, **kwargs)
    except BaseException as caught:  # noqa: BLE001 - detach hostile port failures.
        failure = caught
    if isinstance(failure, (JobCancelled, LeaseLost)):
        raise failure
    if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
        raise signal from None
    if type(failure) is RagRunFailure:
        raise _internal_failure() from None
    if type(failure) is RagDomainError and failure.code not in _allowed_domain_codes:
        raise _internal_failure() from None
    raise failure


def _run_identity(run: UUID | RagRunRecord) -> tuple[UUID, RagRunRecord | None]:
    if type(run) is UUID:
        return run, None
    if type(run) is RagRunRecord:
        return run.id, run
    raise TypeError("run must be a UUID or exact RagRunRecord")


def _validate_authoritative_state(state: AuthoritativeRagRun, lease_job: JobRecord, run_id: UUID) -> None:
    run = state.run
    job = state.job
    _validate_run_record(run)
    _validate_job_record(job, run)
    _validate_job_record(lease_job, run)
    if (
        run.id != run_id
        or run.job_id != job.id
        or lease_job.id != job.id
        or job.job_type is not JobType.BUILD_WIKI
        or job.state is not JobState.RUNNING
        or lease_job.state is not JobState.RUNNING
        or job.user_id != run.user_id
        or job.knowledge_base_id != run.knowledge_base_id
        or job.document_id is not None
        or set(job.payload) != {"run_id"}
        or job.payload.get("run_id") != str(run.id)
        or lease_job.job_type is not JobType.BUILD_WIKI
        or lease_job.user_id != run.user_id
        or lease_job.knowledge_base_id != run.knowledge_base_id
        or lease_job.document_id is not None
        or set(lease_job.payload) != {"run_id"}
        or lease_job.payload.get("run_id") != str(run.id)
        or type(job.lease_owner) is not str
        or not job.lease_owner
        or lease_job.lease_owner != job.lease_owner
    ):
        raise _internal_failure()


def _validate_pages(run: RagRunRecord, value: object) -> tuple[RagPageRecord, ...]:
    if type(value) is not tuple or any(type(page) is not RagPageRecord for page in value):
        raise _internal_failure()
    pages = value
    if len(pages) > run.budget.max_pages:
        raise _internal_failure()
    if (not pages and run.last_committed_ordinal != -1) or (
        pages and not -1 <= run.last_committed_ordinal < len(pages)
    ):
        raise _internal_failure()
    items = tuple(RagWorkItem(page.ordinal, page.path, page.intent, page.query) for page in pages)
    try:
        validate_worklist(items, target_path_prefix=run.target_path_prefix, max_pages=run.budget.max_pages)
    except (TypeError, ValueError):
        raise _internal_failure() from None
    if any(
        type(page.id) is not UUID
        or type(page.run_id) is not UUID
        or type(page.user_id) is not UUID
        or type(page.knowledge_base_id) is not UUID
        or page.run_id != run.id
        or page.user_id != run.user_id
        or page.knowledge_base_id != run.knowledge_base_id
        or type(page.ordinal) is not int
        or type(page.state) is not RagPageState
        or not _valid_page_projection(run, page)
        or not _valid_timestamp(page.created_at)
        or not _valid_timestamp(page.updated_at)
        or (page.ordinal <= run.last_committed_ordinal and page.state is not RagPageState.COMMITTED)
        or (not run.dry_run and page.ordinal > run.last_committed_ordinal and page.state is RagPageState.COMMITTED)
        or (not run.dry_run and page.state is RagPageState.DRY_RUN_COMPLETE)
        or (run.dry_run and page.state is RagPageState.COMMITTED)
        for page in pages
    ):
        raise _internal_failure()
    return pages


def _valid_page_projection(run: RagRunRecord, page: RagPageRecord) -> bool:
    if (
        type(page.state) is not RagPageState
        or type(page.attempt_count) is not int
        or not 0 <= page.attempt_count <= run.budget.max_page_attempts
        or type(page.conflict_retry_count) is not int
        or not 0 <= page.conflict_retry_count <= min(page.attempt_count, run.budget.max_conflict_retries)
        or type(page.last_completed_step_sequence) is not int
        or not 0 <= page.last_completed_step_sequence <= run.budget.max_steps
        or (page.document_id is not None and type(page.document_id) is not UUID)
        or (page.version_read is not None and (type(page.version_read) is not int or page.version_read < 1))
        or (
            page.version_committed is not None
            and (type(page.version_committed) is not int or page.version_committed < 1)
        )
        or not _valid_preview_projection(run, page)
    ):
        return False
    if page.state is RagPageState.PLANNED:
        return (
            page.document_id is None
            and page.version_read is None
            and page.version_committed is None
            and page.lint_summary is None
            and page.preview is None
        )
    if page.state in {RagPageState.RUNNING, RagPageState.FAILED}:
        return (
            (page.state is not RagPageState.RUNNING or page.attempt_count >= 1)
            and ((page.document_id is None) is (page.version_read is None))
            and page.version_committed is None
            and page.lint_summary is None
            and page.preview is None
        )
    if page.state is RagPageState.COMMITTED:
        if page.version_committed is None:
            return False
        expected_read = None if page.version_committed == 1 else page.version_committed - 1
        return (
            page.document_id is not None
            and page.version_committed is not None
            and page.version_read == expected_read
            and type(page.lint_summary) is MappingProxyType
            and page.preview is None
        )
    if page.state is RagPageState.DRY_RUN_COMPLETE:
        return (
            page.attempt_count >= 1
            and page.document_id is None
            and page.version_read is None
            and page.version_committed is None
            and page.lint_summary is None
            and page.preview is not None
        )
    return False


def _valid_preview_projection(run: RagRunRecord, page: RagPageRecord) -> bool:
    if type(page.preview_truncated) is not bool:
        return False
    if page.preview is None:
        return page.preview_digest is None and page.preview_full_char_count is None and not page.preview_truncated
    if (
        type(page.preview) is not str
        or len(page.preview) > 16_384
        or type(page.preview_digest) is not str
        or not _is_digest(page.preview_digest)
        or type(page.preview_full_char_count) is not int
        or not len(page.preview) <= page.preview_full_char_count <= run.budget.max_page_chars
        or page.preview_truncated is not (page.preview_full_char_count > len(page.preview))
    ):
        return False
    return page.preview_truncated or page.preview_digest == hashlib.sha256(page.preview.encode("utf-8")).hexdigest()


def _validate_plan_attempt(
    attempt: object,
    prior: RagRunRecord,
    specs: tuple[PlanAttemptSpec, PlanAttemptSpec],
) -> tuple[RagRunRecord, RagStepRecord]:
    if (
        type(attempt) is not PlanAttempt
        or not _same_run_identity(attempt.run, prior)
        or attempt.run.usage != prior.usage
        or attempt.run.completion_reason is not None
        or attempt.spec not in specs
    ):
        raise _internal_failure()
    step = attempt.step
    if not _valid_running_plan_step(step, attempt.run, attempt.spec) or (
        not attempt.resumed and step.input_digest != attempt.spec.input_digest
    ):
        raise _internal_failure()
    _validate_run_record(attempt.run)
    return attempt.run, step


def _validate_plan_attempt_binding(
    value: object,
    prior: RagRunRecord,
    spec: PlanAttemptSpec,
    prior_step: RagStepRecord,
    input_digest: str,
) -> tuple[RagRunRecord, RagStepRecord | None]:
    if type(value) is not PlanAttemptBinding or not _same_run_identity(value.run, prior):
        raise _internal_failure()
    if value.exhausted:
        if (
            value.attempt is not None
            or value.run.usage.model_tokens != prior.usage.model_tokens
            or value.run.usage.steps != prior.usage.steps + 1
            or (value.run.usage.steps < value.run.budget.max_steps and prior_step.sequence < value.run.budget.max_steps)
            or value.run.completion_reason is not None
        ):
            raise _internal_failure()
        _validate_run_record(value.run)
        return value.run, None
    attempt = value.attempt
    if (
        type(attempt) is not PlanAttempt
        or attempt.run != value.run
        or attempt.spec != spec
        or not _same_run_identity(attempt.run, prior)
        or attempt.run.usage.model_tokens != prior.usage.model_tokens
        or attempt.run.usage.steps - prior.usage.steps not in {0, 1}
        or attempt.run.completion_reason is not None
    ):
        raise _internal_failure()
    step = attempt.step
    if (
        not _valid_running_plan_step(step, attempt.run, spec)
        or step.input_digest != input_digest
        or (attempt.run.usage == prior.usage and (step.id != prior_step.id or step.sequence != prior_step.sequence))
        or (attempt.run.usage != prior.usage and (step.id == prior_step.id or step.sequence != prior_step.sequence + 1))
    ):
        raise _internal_failure()
    _validate_run_record(attempt.run)
    return attempt.run, step


def _planning_input_digest(run: RagRunRecord, prompt_digest: str) -> str:
    value = {
        "goal_digest": run.goal_digest,
        "knowledge_base_id": str(run.knowledge_base_id),
        "prompt_digest": prompt_digest,
        "retrieval_profile": run.retrieval_profile,
        "target_path_prefix": run.target_path_prefix,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _plan_attempt_specs(run: RagRunRecord) -> tuple[PlanAttemptSpec, PlanAttemptSpec]:
    return (
        PlanAttemptSpec(
            repair=False,
            input_digest=_planning_input_digest(run, PLANNER_PROMPT_DIGEST),
            prompt_version=PLANNER_PROMPT_VERSION,
            prompt_digest=PLANNER_PROMPT_DIGEST,
        ),
        PlanAttemptSpec(
            repair=True,
            input_digest=_planning_input_digest(run, PLANNER_REPAIR_PROMPT_DIGEST),
            prompt_version=PLANNER_REPAIR_PROMPT_VERSION,
            prompt_digest=PLANNER_REPAIR_PROMPT_DIGEST,
        ),
    )


def _planning_messages_digest(
    run: RagRunRecord,
    spec: PlanAttemptSpec,
    messages: Sequence[Mapping[str, str]],
) -> str:
    value = {
        "messages": [dict(message) for message in messages],
        "planner": {"prompt_digest": spec.prompt_digest, "prompt_version": spec.prompt_version},
        "run_scope": {
            "goal_digest": run.goal_digest,
            "knowledge_base_id": str(run.knowledge_base_id),
            "model_profile_version": run.model_profile_version,
            "retrieval_profile": run.retrieval_profile,
            "target_path_prefix": run.target_path_prefix,
        },
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _reserve_plan_call(
    run: RagRunRecord,
    usage: RagUsage,
    messages: Sequence[Mapping[str, str]],
) -> tuple[int, int]:
    max_output = min(
        _MAX_PLAN_OUTPUT_TOKENS,
        max(_MIN_PLAN_OUTPUT_TOKENS, run.budget.max_pages * _PLAN_OUTPUT_TOKENS_PER_PAGE),
    )
    input_tokens = max(1, (sum(len(item["content"].encode("utf-8")) for item in messages) + 3) // 4)
    reservation = input_tokens + max_output
    usage.reserve_model_call(run.budget, reservation)
    return reservation, max_output


def _run_config(run: RagRunRecord) -> RagRunConfig:
    return RagRunConfig.build(
        knowledge_base_id=run.knowledge_base_id,
        goal=run.goal,
        target_path_prefix=run.target_path_prefix,
        model_profile=run.model_profile,
        retrieval_profile=run.retrieval_profile,
        dry_run=run.dry_run,
        budget=run.budget,
    )


def _aggregate_plan_usage(run: RagRunRecord, token_usage: RagTokenUsage) -> RagUsage:
    if type(token_usage) is not RagTokenUsage:
        raise _internal_failure()
    return RagUsage(
        steps=run.usage.steps + 1,
        model_tokens=run.usage.model_tokens + token_usage.total_tokens,
    )


def _validated_plan_response(
    response: object,
    run: RagRunRecord,
    next_usage: RagUsage,
    reservation: int,
) -> tuple[tuple[RagWorkItem, ...], RagTokenUsage, RagUsage, str | None, bool]:
    if type(response) is not RagModelResponse or type(response.usage) is not RagTokenUsage:
        raise _internal_failure()
    reported_tokens = _validated_token_usage(response.usage)
    try:
        committed_usage = next_usage.commit_model_usage(
            run.budget,
            reservation,
            reported_tokens.total_tokens,
        )
    except RagDomainError as invalid_usage:
        if type(invalid_usage) is not RagDomainError or invalid_usage.code != "rag_invalid_model_usage":
            raise
        charged_usage = RagUsage(
            steps=next_usage.steps,
            model_tokens=next_usage.model_tokens + reservation,
        )
        return (), _ZERO_TOKEN_USAGE, charged_usage, "rag_invalid_model_usage", True
    try:
        items = parse_plan(response.payload, _run_config(run))
    except RagDomainError as invalid_plan:
        if type(invalid_plan) is not RagDomainError or invalid_plan.code != "rag_invalid_plan":
            raise
        return (), reported_tokens, committed_usage, "rag_invalid_plan", False
    return items, reported_tokens, committed_usage, None, False


def _validated_token_usage(value: object) -> RagTokenUsage:
    if type(value) is not RagTokenUsage:
        raise _internal_failure()
    try:
        validated = RagTokenUsage(value.prompt_tokens, value.completion_tokens, value.total_tokens)
    except (TypeError, ValueError):
        raise _internal_failure() from None
    return validated


def _failure_token_accounting(
    failure: BaseException,
    run: RagRunRecord,
    reservation: int,
) -> tuple[RagTokenUsage, RagUsage, bool, int]:
    usage = failure.usage if type(failure) is InvalidRagModelResponse else None
    if usage is None or reservation <= 0:
        return _ZERO_TOKEN_USAGE, _aggregate_plan_usage(run, _ZERO_TOKEN_USAGE), False, 0
    trusted = _validated_token_usage(usage)
    if 0 < trusted.total_tokens <= reservation:
        return trusted, _aggregate_plan_usage(run, trusted), True, trusted.total_tokens
    aggregate = RagUsage(run.usage.steps + 1, run.usage.model_tokens + reservation)
    return _ZERO_TOKEN_USAGE, aggregate, False, reservation


def _lease_owner(job: JobRecord) -> str:
    owner = job.lease_owner
    if type(owner) is not str or not owner or owner.strip() != owner:
        raise LeaseLost("RAG job lease is no longer active")
    return owner


def _terminal_replay_result(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
) -> dict[str, str | int]:
    if run.usage.steps > run.budget.max_steps or run.usage.model_tokens > run.budget.max_model_tokens:
        raise _internal_failure()
    completion = run.completion_reason
    if completion is RagCompletionReason.NO_WORK:
        if pages or run.last_committed_ordinal != -1 or run.usage.steps < 1 or run.usage.model_tokens < 1:
            raise _internal_failure()
        return {
            "run_id": str(run.id),
            "completion_reason": completion.value,
            "pages_committed": 0,
        }
    if completion is RagCompletionReason.COMPLETED:
        if (
            run.dry_run
            or not pages
            or run.last_committed_ordinal != len(pages) - 1
            or any(page.state is not RagPageState.COMMITTED for page in pages)
        ):
            raise _internal_failure()
        return {
            "run_id": str(run.id),
            "completion_reason": completion.value,
            "pages_committed": len(pages),
        }
    if completion is RagCompletionReason.DRY_RUN:
        if (
            not run.dry_run
            or not pages
            or run.last_committed_ordinal != -1
            or any(page.state is not RagPageState.DRY_RUN_COMPLETE for page in pages)
        ):
            raise _internal_failure()
        return {
            "run_id": str(run.id),
            "completion_reason": completion.value,
            "pages_committed": 0,
            "pages_dry_run": len(pages),
        }
    if completion is None:
        raise _internal_failure()
    raise _internal_failure()


def _validate_plan_acceptance(
    value: object,
    prior: RagRunRecord,
    items: tuple[RagWorkItem, ...],
    committed_usage: RagUsage,
    completion_reason: RagCompletionReason | None,
) -> tuple[RagRunRecord, tuple[RagPageRecord, ...]]:
    if type(value) is not PlanAcceptance:
        raise _internal_failure()
    run, pages = value.run, value.pages
    if (
        type(run) is not RagRunRecord
        or not _same_run_identity(run, prior)
        or run.usage != committed_usage
        or run.completion_reason is not completion_reason
        or value.completion_reason is not completion_reason
        or run.last_committed_ordinal != prior.last_committed_ordinal
    ):
        raise _internal_failure()
    _validate_run_record(run)
    validated_pages = _validate_pages(run, pages)
    if tuple((page.ordinal, page.path, page.intent, page.query) for page in validated_pages) != tuple(
        (item.ordinal, item.path, item.intent, item.query) for item in items
    ):
        raise _internal_failure()
    if (not items and completion_reason is not RagCompletionReason.NO_WORK) or (
        items and completion_reason is not None
    ):
        raise _internal_failure()
    return run, validated_pages


def _validate_page_outcome(
    prior_run: RagRunRecord,
    prior_page: RagPageRecord,
    outcome: PageExecutionResult,
) -> None:
    if (
        not _same_run_identity(outcome.run, prior_run)
        or outcome.page.id != prior_page.id
        or outcome.page.run_id != prior_run.id
        or outcome.page.ordinal != prior_page.ordinal
        or outcome.page.state not in {RagPageState.COMMITTED, RagPageState.DRY_RUN_COMPLETE}
        or (prior_run.dry_run and outcome.page.state is not RagPageState.DRY_RUN_COMPLETE)
        or (not prior_run.dry_run and outcome.page.state is not RagPageState.COMMITTED)
        or outcome.run.usage.steps < prior_run.usage.steps
        or outcome.run.usage.model_tokens < prior_run.usage.model_tokens
        or outcome.run.usage.steps > prior_run.budget.max_steps
        or outcome.run.usage.model_tokens > prior_run.budget.max_model_tokens
    ):
        raise _internal_failure()
    _validate_run_record(outcome.run)


def _validate_finished_pages(
    run: RagRunRecord,
    pages: tuple[RagPageRecord, ...],
    completion: RagCompletionReason,
) -> None:
    target = RagPageState.DRY_RUN_COMPLETE if completion is RagCompletionReason.DRY_RUN else RagPageState.COMMITTED
    if not pages:
        raise _internal_failure()
    if any(page.state is not target for page in pages):
        raise _internal_failure()


def _require_remaining_budget(run: RagRunRecord) -> None:
    if run.usage.steps >= run.budget.max_steps or run.usage.model_tokens >= run.budget.max_model_tokens:
        raise _budget_failure()


def _require_within_budget(run: RagRunRecord) -> None:
    if run.usage.steps > run.budget.max_steps or run.usage.model_tokens > run.budget.max_model_tokens:
        raise _budget_failure()


def _validate_run_record(run: RagRunRecord) -> None:
    try:
        config = _run_config(run)
        goal_digest = hashlib.sha256(run.goal.encode("utf-8")).hexdigest()
        valid = (
            type(run.id) is UUID
            and type(run.job_id) is UUID
            and type(run.root_run_id) is UUID
            and (run.parent_run_id is None or type(run.parent_run_id) is UUID)
            and type(run.user_id) is UUID
            and type(run.knowledge_base_id) is UUID
            and config.goal == run.goal
            and config.target_path_prefix == run.target_path_prefix
            and config.model_profile == run.model_profile
            and config.retrieval_profile == run.retrieval_profile
            and config.dry_run is run.dry_run
            and config.budget == run.budget
            and goal_digest == run.goal_digest
            and _is_digest(run.goal_digest)
            and _is_digest(run.request_digest)
            and type(run.idempotency_key) is str
            and bool(run.idempotency_key)
            and run.idempotency_key.strip() == run.idempotency_key
            and type(run.budget) is RagBudget
            and type(run.model_profile_version) is str
            and 1 <= len(run.model_profile_version) <= 128
            and run.model_profile_version.strip() == run.model_profile_version
            and type(run.usage) is RagUsage
            and type(run.usage.steps) is int
            and type(run.usage.model_tokens) is int
            and 0 <= run.usage.steps <= run.budget.max_steps
            and 0 <= run.usage.model_tokens <= run.budget.max_model_tokens
            and type(run.last_committed_ordinal) is int
            and run.last_committed_ordinal >= -1
            and (run.completion_reason is None or type(run.completion_reason) is RagCompletionReason)
            and _valid_timestamp(run.created_at)
            and _valid_timestamp(run.updated_at)
            and (
                (run.parent_run_id is None and run.root_run_id == run.id)
                or (type(run.parent_run_id) is UUID and run.root_run_id != run.id and run.parent_run_id != run.id)
            )
        )
    except BaseException as failure:  # noqa: BLE001 - strict hostile record validation.
        if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
            raise signal from None
        valid = False
    if not valid:
        raise _internal_failure()


def _valid_timestamp(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is not None and value.utcoffset() is not None


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_job_record(job: JobRecord, run: RagRunRecord) -> None:
    valid = (
        type(job) is JobRecord
        and type(job.id) is UUID
        and type(job.user_id) is UUID
        and type(job.knowledge_base_id) is UUID
        and job.document_id is None
        and type(job.job_type) is JobType
        and type(job.state) is JobState
        and job.job_type is JobType.BUILD_WIKI
        and job.state is JobState.RUNNING
        and job.id == run.job_id
        and job.user_id == run.user_id
        and job.knowledge_base_id == run.knowledge_base_id
        and type(job.attempt_count) is int
        and type(job.max_attempts) is int
        and 0 <= job.attempt_count <= job.max_attempts
        and job.max_attempts >= 1
        and type(job.dispatch_attempts) is int
        and job.dispatch_attempts >= 0
        and _valid_timestamp(job.run_after)
        and _valid_timestamp(job.created_at)
        and _valid_timestamp(job.updated_at)
        and type(job.lease_owner) is str
        and bool(job.lease_owner)
        and job.lease_owner.strip() == job.lease_owner
        and _valid_timestamp(job.lease_expires_at)
        and set(job.payload) == {"run_id"}
        and job.payload.get("run_id") == str(run.id)
    )
    if not valid:
        raise _internal_failure()


def _valid_running_plan_step(step: object, run: RagRunRecord, spec: PlanAttemptSpec) -> bool:
    return (
        type(step) is RagStepRecord
        and type(step.id) is UUID
        and type(step.run_id) is UUID
        and step.run_id == run.id
        and step.run_page_id is None
        and type(step.user_id) is UUID
        and step.user_id == run.user_id
        and type(step.knowledge_base_id) is UUID
        and step.knowledge_base_id == run.knowledge_base_id
        and type(step.sequence) is int
        and step.sequence >= 1
        and step.step_type is RagStepType.PLAN
        and step.status is RagStepStatus.RUNNING
        and _is_digest(step.input_digest)
        and type(step.output_summary) is MappingProxyType
        and not step.output_summary
        and type(step.citation_identities) is tuple
        and not step.citation_identities
        and step.prompt_version == spec.prompt_version
        and step.prompt_digest == spec.prompt_digest
        and step.model_profile_version == run.model_profile_version
        and step.input_tokens == step.output_tokens == step.total_tokens == 0
        and type(step.latency_ms) in {int, float}
        and step.latency_ms == 0
        and step.error_code is None
        and step.error_message is None
        and _valid_timestamp(step.created_at)
        and _valid_timestamp(step.updated_at)
    )


def _same_run_identity(left: RagRunRecord, right: RagRunRecord) -> bool:
    fields = (
        "id",
        "job_id",
        "root_run_id",
        "parent_run_id",
        "user_id",
        "knowledge_base_id",
        "goal",
        "goal_digest",
        "target_path_prefix",
        "model_profile",
        "model_profile_version",
        "retrieval_profile",
        "dry_run",
        "budget",
        "idempotency_key",
        "request_digest",
    )
    try:
        return all(getattr(left, field) == getattr(right, field) for field in fields)
    except BaseException as failure:  # noqa: BLE001 - strict hostile record validation.
        if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
            raise signal from None
        return False


def _require_page_budget(run: RagRunRecord) -> None:
    _require_remaining_budget(run)


def _latency_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1_000)


def _map_failure(failure: BaseException, *, operation: str) -> RagRunFailure:
    if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
        raise signal from None
    if isinstance(failure, (JobCancelled, LeaseLost)):
        raise failure
    if type(failure) is RagRunFailure:
        return failure
    if isinstance(failure, RagModelUnavailable):
        return RagRunFailure("rag_model_unavailable", "The RAG model is temporarily unavailable.", True)
    if isinstance(failure, InvalidRagModelResponse):
        return RagRunFailure("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False)
    if isinstance(failure, RetrieverUnavailable):
        return RagRunFailure("rag_retrieval_failed", "RAG retrieval is temporarily unavailable.", True)
    if type(failure) is RagDomainError:
        safe_by_operation = {
            "orchestrator": {
                "rag_budget_exhausted": ("rag_budget_exhausted", "The RAG budget was exhausted.", False),
                "rag_disabled": ("rag_disabled", "Server-side RAG is disabled.", False),
            },
            "plan": {
                "rag_budget_exhausted": ("rag_budget_exhausted", "The RAG budget was exhausted.", False),
                "rag_disabled": ("rag_disabled", "Server-side RAG is disabled.", False),
                "rag_invalid_plan": ("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False),
                "rag_invalid_model_usage": ("rag_invalid_plan", _INVALID_PLAN_MESSAGE, False),
                "rag_prompt_invalid": ("rag_prompt_invalid", "The RAG prompt inputs were invalid.", False),
            },
        }
        safe = safe_by_operation.get(operation, {}).get(failure.code)
        if safe is not None:
            return RagRunFailure(safe[0], safe[1], safe[2])
    return _internal_failure()


def _budget_failure() -> RagRunFailure:
    return RagRunFailure("rag_budget_exhausted", "The RAG budget was exhausted.", False)


def _internal_failure() -> RagRunFailure:
    return RagRunFailure(_INTERNAL_CODE, _INTERNAL_MESSAGE, True)


def _public_failure_scalars(failure: BaseException) -> tuple[str, str, str, bool, int]:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None and type(signal) is not BaseException:
        if isinstance(signal, JobCancelled):
            return "job_cancelled", "", "", False, 0
        if isinstance(signal, LeaseLost):
            return "lease_lost", "", "", False, 0
        if isinstance(signal, asyncio.CancelledError):
            return "cancelled", "", "", False, 0
        if isinstance(signal, KeyboardInterrupt):
            return "keyboard", "", "", False, 0
        if isinstance(signal, SystemExit):
            safe_code = signal.code if type(signal.code) is int else 1
            return "system_exit", "", "", False, safe_code
        if isinstance(signal, GeneratorExit):
            return "generator_exit", "", "", False, 0
        return "rag", _INTERNAL_CODE, _INTERNAL_MESSAGE, True, 0
    if isinstance(failure, JobCancelled):
        return "job_cancelled", "", "", False, 0
    if isinstance(failure, LeaseLost):
        return "lease_lost", "", "", False, 0
    mapped = failure if type(failure) is RagRunFailure else _map_failure(failure, operation="orchestrator")
    return "rag", mapped.code, mapped.public_message, mapped.retryable, 0


def _raise_public_scalars(
    kind: str,
    code: str,
    message: str,
    retryable: bool,
    system_exit_code: int,
) -> NoReturn:
    if kind == "cancelled":
        raise asyncio.CancelledError() from None
    if kind == "keyboard":
        raise KeyboardInterrupt() from None
    if kind == "system_exit":
        raise SystemExit(system_exit_code) from None
    if kind == "generator_exit":
        raise GeneratorExit() from None
    if kind == "job_cancelled":
        raise JobCancelled("background job cancellation was requested") from None
    if kind == "lease_lost":
        raise LeaseLost("background job lease is no longer active") from None
    raise RagRunFailure(code, message, retryable) from None


__all__ = [
    "BuildWikiOrchestrator",
    "PLANNER_REPAIR_PROMPT_DIGEST",
    "PLANNER_REPAIR_PROMPT_VERSION",
    "RagRunFailure",
]
