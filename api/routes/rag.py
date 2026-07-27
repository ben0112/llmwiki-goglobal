"""Hosted HTTP surface for durable server-side RAG runs."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Annotated
from uuid import UUID

from asyncpg.pgproto.pgproto import UUID as AsyncpgUUID
from deps import get_job_service, get_rag_service, get_user_id
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from jobs.models import JobRecord, JobState, JobType
from jobs.service import JobResourceNotFound, JobService
from pydantic import BaseModel, ConfigDict, Field
from rag.records import RagRunRecord, RagStepRecord
from rag.service import CreateRagRun, RagService

from llmwiki_core.rag import (
    DEFAULT_MAX_CONFLICT_RETRIES,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_MODEL_TOKENS,
    DEFAULT_MAX_PAGE_ATTEMPTS,
    DEFAULT_MAX_PAGE_CHARS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_STEPS,
    DEFAULT_PER_CALL_TIMEOUT_SECONDS,
    MAX_CONFLICT_RETRIES,
    MAX_CONTEXT_CHARS,
    MAX_GOAL_CHARS,
    MAX_MODEL_TOKENS,
    MAX_PAGE_ATTEMPTS,
    MAX_PAGE_CHARS,
    MAX_PAGES,
    MAX_PER_CALL_TIMEOUT_SECONDS,
    MAX_PROFILE_CHARS,
    MAX_STEPS,
    RAG_ERROR_CONTRACTS,
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagDomainError,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
)
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown


class _SanitizedValidationRoute(APIRoute):
    """Replace FastAPI's input-echoing validation details at the RAG boundary."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        original = super().get_route_handler()

        async def sanitized(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "rag_invalid_request",
                        "message": "The RAG request is invalid.",
                    },
                ) from None

        return sanitized


router = APIRouter(prefix="/v1/rag", tags=["rag"], route_class=_SanitizedValidationRoute)

IdempotencyKey = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=1, max_length=200),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuildWikiBudget(_StrictModel):
    max_pages: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGES)] = DEFAULT_MAX_PAGES
    max_steps: Annotated[int, Field(strict=True, ge=1, le=MAX_STEPS)] = DEFAULT_MAX_STEPS
    max_model_tokens: Annotated[int, Field(strict=True, ge=1, le=MAX_MODEL_TOKENS)] = DEFAULT_MAX_MODEL_TOKENS
    max_context_chars: Annotated[int, Field(strict=True, ge=1, le=MAX_CONTEXT_CHARS)] = DEFAULT_MAX_CONTEXT_CHARS
    max_page_chars: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_CHARS)] = DEFAULT_MAX_PAGE_CHARS
    per_call_timeout_seconds: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_PER_CALL_TIMEOUT_SECONDS),
    ] = DEFAULT_PER_CALL_TIMEOUT_SECONDS
    max_page_attempts: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_ATTEMPTS)] = DEFAULT_MAX_PAGE_ATTEMPTS
    max_conflict_retries: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CONFLICT_RETRIES),
    ] = DEFAULT_MAX_CONFLICT_RETRIES

    def to_domain(self) -> RagBudget:
        return RagBudget(**self.model_dump())


class ResumeBudgetOverride(_StrictModel):
    max_pages: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGES)] | None = None
    max_steps: Annotated[int, Field(strict=True, ge=1, le=MAX_STEPS)] | None = None
    max_model_tokens: Annotated[int, Field(strict=True, ge=1, le=MAX_MODEL_TOKENS)] | None = None
    max_context_chars: Annotated[int, Field(strict=True, ge=1, le=MAX_CONTEXT_CHARS)] | None = None
    max_page_chars: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_CHARS)] | None = None
    per_call_timeout_seconds: (
        Annotated[
            int,
            Field(strict=True, ge=1, le=MAX_PER_CALL_TIMEOUT_SECONDS),
        ]
        | None
    ) = None
    max_page_attempts: Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_ATTEMPTS)] | None = None
    max_conflict_retries: (
        Annotated[
            int,
            Field(strict=True, ge=1, le=MAX_CONFLICT_RETRIES),
        ]
        | None
    ) = None

    def explicit_values(self) -> dict[str, int]:
        return self.model_dump(exclude_none=True)


class BuildWikiRequest(_StrictModel):
    knowledge_base_id: UUID
    goal: Annotated[str, Field(min_length=1, max_length=MAX_GOAL_CHARS)]
    target_path_prefix: Annotated[str, Field(min_length=1, max_length=2_000)]
    model_profile: Annotated[str, Field(min_length=1, max_length=MAX_PROFILE_CHARS)]
    retrieval_profile: Annotated[str, Field(min_length=1, max_length=MAX_PROFILE_CHARS)] = "lexical"
    dry_run: Annotated[bool, Field(strict=True)] = False
    budget: BuildWikiBudget = Field(default_factory=BuildWikiBudget)


class ResumeRunRequest(_StrictModel):
    budget: ResumeBudgetOverride = Field(default_factory=ResumeBudgetOverride)


class PublicRagAccepted(_StrictModel):
    run_id: UUID
    job_id: UUID
    state: JobState
    run_url: str
    job_url: str


class PublicRagBudget(_StrictModel):
    max_pages: int
    max_steps: int
    max_model_tokens: int
    max_context_chars: int
    max_page_chars: int
    per_call_timeout_seconds: int
    max_page_attempts: int
    max_conflict_retries: int


class PublicRagUsage(_StrictModel):
    steps: int
    model_tokens: int


class PublicRagRun(_StrictModel):
    id: UUID
    job_id: UUID
    root_run_id: UUID
    parent_run_id: UUID | None
    knowledge_base_id: UUID
    goal_digest: str
    model_profile: str
    model_profile_version: str
    retrieval_profile: str
    dry_run: bool
    budget: PublicRagBudget
    usage: PublicRagUsage
    state: JobState
    completion_reason: RagCompletionReason | None
    last_committed_ordinal: int
    created_at: datetime
    updated_at: datetime


class PublicRagCitation(_StrictModel):
    document_id: UUID
    document_version: int
    chunk_index: int
    page: int | None


class PublicRagStep(_StrictModel):
    id: UUID
    run_id: UUID
    run_page_id: UUID | None
    knowledge_base_id: UUID
    sequence: int
    type: RagStepType
    status: RagStepStatus
    citation_identities: list[PublicRagCitation]
    model_profile_version: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    error_code: str | None
    created_at: datetime
    updated_at: datetime


class PublicRagStepsPage(_StrictModel):
    items: list[PublicRagStep]
    next_cursor: int | None


async def _get_authenticated_user_id(
    user_id: Annotated[str, Depends(get_user_id)],
) -> UUID:
    try:
        return UUID(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated subject",
        ) from None


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="RAG run not found")


_CANONICAL_ERROR_BY_CODE = {}
for _contract in RAG_ERROR_CONTRACTS.values():
    _CANONICAL_ERROR_BY_CODE.setdefault(_contract.code, _contract)

_UNPROCESSABLE_CODES = frozenset(
    {
        "rag_attempt_limit_mismatch",
        "rag_budget_exhausted",
        "rag_budget_too_small",
        "rag_model_profile_unavailable",
        "rag_retrieval_profile_unavailable",
    }
)
_NOT_FOUND_CODES = frozenset(
    {
        "rag_document_not_found",
        "rag_page_not_found",
        "rag_run_not_found",
    }
)
_PUBLIC_RAG_ERROR_CODES = frozenset(contract.code for contract in RAG_ERROR_CONTRACTS.values())


class _InvalidProjection(RuntimeError):
    pass


def _internal_error() -> HTTPException:
    contract = RAG_ERROR_CONTRACTS["internal_error"]
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": contract.code, "message": contract.public_message},
    )


def _raise_projection_failure(failure: BaseException | None = None) -> None:
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        if signal is not None and type(signal) is not BaseException:
            raise signal from None
    raise _InvalidProjection() from None


def _raise_route_failure(failure: BaseException) -> None:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None and type(signal) is not BaseException:
        raise signal from None
    if type(failure) is JobResourceNotFound:
        raise _not_found() from None
    if type(failure) is RagDomainError:
        raise _domain_http_error(failure) from None
    raise _internal_error() from None


def _valid_timestamp(value: object) -> bool:
    return type(value) is datetime and value.tzinfo is not None and value.utcoffset() is not None


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_run_record(
    run: object,
    *,
    authenticated_user_id: UUID,
    operation: str,
    resource_id: UUID | None = None,
    knowledge_base_id: UUID | None = None,
) -> RagRunRecord:
    try:
        if type(run) is not RagRunRecord:
            raise TypeError
        if (
            type(run.id) is not UUID
            or type(run.job_id) is not UUID
            or type(run.root_run_id) is not UUID
            or (run.parent_run_id is not None and type(run.parent_run_id) is not UUID)
            or type(run.user_id) is not UUID
            or type(run.knowledge_base_id) is not UUID
            or run.user_id != authenticated_user_id
            or run.job_id == run.id
            or run.job_id == run.root_run_id
            or run.job_id == run.parent_run_id
        ):
            raise ValueError
        if operation == "create" and (
            run.id != run.root_run_id or run.parent_run_id is not None or run.knowledge_base_id != knowledge_base_id
        ):
            raise ValueError
        if operation == "resume" and (
            run.id == resource_id or run.root_run_id == run.id or run.parent_run_id != resource_id
        ):
            raise ValueError
        if operation in {"get", "steps"} and run.id != resource_id:
            raise ValueError
        if (run.parent_run_id is None and run.root_run_id != run.id) or (
            run.parent_run_id is not None and (run.root_run_id == run.id or run.parent_run_id == run.id)
        ):
            raise ValueError
        if (
            type(run.goal) is not str
            or not 1 <= len(run.goal) <= MAX_GOAL_CHARS
            or type(run.target_path_prefix) is not str
            or not 1 <= len(run.target_path_prefix) <= 2_000
            or type(run.model_profile) is not str
            or not 1 <= len(run.model_profile) <= MAX_PROFILE_CHARS
            or type(run.retrieval_profile) is not str
            or not 1 <= len(run.retrieval_profile) <= MAX_PROFILE_CHARS
            or type(run.dry_run) is not bool
            or type(run.budget) is not RagBudget
        ):
            raise TypeError
        normalized = RagRunConfig.build(
            knowledge_base_id=run.knowledge_base_id,
            goal=run.goal,
            target_path_prefix=run.target_path_prefix,
            model_profile=run.model_profile,
            retrieval_profile=run.retrieval_profile,
            dry_run=run.dry_run,
            budget=run.budget,
        )
        if (
            normalized.goal != run.goal
            or normalized.target_path_prefix != run.target_path_prefix
            or normalized.model_profile != run.model_profile
            or normalized.retrieval_profile != run.retrieval_profile
            or hashlib.sha256(run.goal.encode("utf-8")).hexdigest() != run.goal_digest
            or not _is_digest(run.goal_digest)
            or not _is_digest(run.request_digest)
            or type(run.model_profile_version) is not str
            or not 1 <= len(run.model_profile_version) <= 128
            or run.model_profile_version.strip() != run.model_profile_version
            or type(run.idempotency_key) is not str
            or not 1 <= len(run.idempotency_key) <= 200
            or run.idempotency_key.strip() != run.idempotency_key
            or type(run.usage) is not RagUsage
            or type(run.usage.steps) is not int
            or not 0 <= run.usage.steps <= run.budget.max_steps
            or type(run.usage.model_tokens) is not int
            or not 0 <= run.usage.model_tokens <= run.budget.max_model_tokens
            or (run.completion_reason is not None and type(run.completion_reason) is not RagCompletionReason)
            or (run.completion_reason is RagCompletionReason.DRY_RUN and not run.dry_run)
            or type(run.last_committed_ordinal) is not int
            or not -1 <= run.last_committed_ordinal < run.budget.max_pages
            or (run.completion_reason is RagCompletionReason.NO_WORK and run.last_committed_ordinal != -1)
            or not _valid_timestamp(run.created_at)
            or not _valid_timestamp(run.updated_at)
            or run.created_at > run.updated_at
        ):
            raise ValueError
    except BaseException as failure:  # noqa: BLE001 - service projections are adapter-controlled.
        _raise_projection_failure(failure)
    return run


def _validate_citations(value: object) -> tuple[RagCitation, ...]:
    if type(value) is not tuple or len(value) > 128:
        raise ValueError
    seen = set()
    for item in value:
        if (
            type(item) is not RagCitation
            or type(item.document_id) is not UUID
            or type(item.document_version) is not int
            or item.document_version < 1
            or type(item.chunk_index) is not int
            or item.chunk_index < 0
            or (item.page is not None and (type(item.page) is not int or item.page < 1))
        ):
            raise ValueError
        identity = (item.document_id, item.document_version, item.chunk_index, item.page)
        if identity in seen:
            raise ValueError
        seen.add(identity)
    return value


def _validate_step_records(  # noqa: C901 - validates one bounded public trace page.
    records: object,
    *,
    run: RagRunRecord,
    after_sequence: int,
    limit: int,
) -> tuple[RagStepRecord, ...]:
    try:
        if type(records) is not tuple or len(records) > limit:
            raise TypeError
        previous = after_sequence
        step_ids = set()
        for step in records:
            if (
                type(step) is not RagStepRecord
                or type(step.id) is not UUID
                or type(step.run_id) is not UUID
                or step.run_id != run.id
                or (step.run_page_id is not None and type(step.run_page_id) is not UUID)
                or type(step.user_id) is not UUID
                or step.user_id != run.user_id
                or type(step.knowledge_base_id) is not UUID
                or step.knowledge_base_id != run.knowledge_base_id
                or type(step.sequence) is not int
                or not previous < step.sequence <= MAX_STEPS
                or type(step.step_type) is not RagStepType
                or type(step.status) is not RagStepStatus
                or not _is_digest(step.input_digest)
                or type(step.output_summary) is not MappingProxyType
                or type(step.model_profile_version) is not str
                or step.model_profile_version != run.model_profile_version
                or type(step.reserved_tokens) is not int
                or not 0 <= step.reserved_tokens <= MAX_MODEL_TOKENS
                or type(step.input_tokens) is not int
                or step.input_tokens < 0
                or type(step.output_tokens) is not int
                or step.output_tokens < 0
                or type(step.total_tokens) is not int
                or step.total_tokens != step.input_tokens + step.output_tokens
                or step.total_tokens > MAX_MODEL_TOKENS
                or type(step.latency_ms) not in {int, float}
                or not isfinite(step.latency_ms)
                or step.latency_ms < 0
                or (step.error_code is not None and (type(step.error_code) is not str or len(step.error_code) > 128))
                or (
                    step.error_message is not None
                    and (type(step.error_message) is not str or len(step.error_message) > 2_000)
                )
                or not _valid_timestamp(step.created_at)
                or not _valid_timestamp(step.updated_at)
                or step.created_at > step.updated_at
            ):
                raise ValueError
            if step.id in step_ids:
                raise ValueError
            step_ids.add(step.id)
            citations = _validate_citations(step.citation_identities)
            if (step.prompt_version is None) != (step.prompt_digest is None):
                raise ValueError
            if step.prompt_version is not None and (
                type(step.prompt_version) is not str
                or not 1 <= len(step.prompt_version) <= 128
                or step.prompt_version.strip() != step.prompt_version
                or not _is_digest(step.prompt_digest)
            ):
                raise ValueError
            if (step.step_type is RagStepType.DRAFT and step.reserved_tokens < 1) or (
                step.step_type is not RagStepType.DRAFT and step.reserved_tokens != 0
            ):
                raise ValueError
            if step.step_type is RagStepType.DRAFT and step.total_tokens > step.reserved_tokens:
                raise ValueError
            if step.status is RagStepStatus.RUNNING and (
                step.output_summary
                or citations
                or step.total_tokens
                or step.latency_ms
                or step.error_code is not None
                or step.error_message is not None
            ):
                raise ValueError
            if step.status is RagStepStatus.SUCCEEDED and (
                step.error_code is not None or step.error_message is not None
            ):
                raise ValueError
            if step.status is RagStepStatus.FAILED and step.error_code is None:
                raise ValueError
            previous = step.sequence
    except BaseException as failure:  # noqa: BLE001 - service projections are adapter-controlled.
        _raise_projection_failure(failure)
    return records


def _domain_http_error(error: RagDomainError) -> HTTPException:
    try:
        code = error.code
    except BaseException:  # noqa: BLE001 - hostile domain exception attributes fail closed.
        code = "rag_internal_error"
    if type(code) is not str or not 1 <= len(code) <= 128:
        code = "rag_internal_error"
    contract = _CANONICAL_ERROR_BY_CODE.get(code, RAG_ERROR_CONTRACTS["internal_error"])
    if contract.code in {"rag_disabled", "rag_internal_error"}:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif contract.code == "rag_invalid_request":
        status_code = status.HTTP_400_BAD_REQUEST
    elif contract.code in _UNPROCESSABLE_CODES:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif contract.code in _NOT_FOUND_CODES:
        status_code = status.HTTP_404_NOT_FOUND
    else:
        status_code = status.HTTP_409_CONFLICT
    return HTTPException(
        status_code=status_code,
        detail={"code": contract.code, "message": contract.public_message},
    )


async def _public_job_state(
    run: RagRunRecord,
    user_id: UUID,
    service: JobService,
) -> JobState:
    record = await service.get(run.job_id, authenticated_user_id=user_id)
    record = _normalize_trusted_job_uuids(record, service)
    if (
        type(record) is not JobRecord
        or type(record.id) is not UUID
        or record.id != run.job_id
        or type(record.user_id) is not UUID
        or record.user_id != user_id
        or record.user_id != run.user_id
        or type(record.knowledge_base_id) is not UUID
        or record.knowledge_base_id != run.knowledge_base_id
        or record.document_id is not None
        or type(record.job_type) is not JobType
        or record.job_type is not JobType.BUILD_WIKI
        or type(record.state) is not JobState
        or type(record.payload) is not MappingProxyType
        or set(record.payload) != {"run_id"}
        or record.payload.get("run_id") != str(run.id)
    ):
        raise _InvalidProjection()
    return record.state


def _normalize_trusted_job_uuids(record: object, service: object) -> object:
    """Normalize exact asyncpg UUIDs without touching untrusted UUID subclasses."""
    if type(service) is not JobService or type(record) is not JobRecord:
        return record
    normalized: dict[str, UUID] = {}
    changed = False
    try:
        for field_name in ("id", "user_id", "knowledge_base_id"):
            value = getattr(record, field_name)
            if type(value) is UUID:
                normalized[field_name] = value
            elif type(value) is AsyncpgUUID:
                normalized[field_name] = UUID(bytes=value.bytes)
                changed = True
            else:
                return record
        return replace(record, **normalized) if changed else record
    except BaseException as failure:  # noqa: BLE001 - fail closed at the adapter boundary.
        _raise_projection_failure(failure)


def _accepted(run: RagRunRecord, state: JobState) -> PublicRagAccepted:
    return PublicRagAccepted(
        run_id=run.id,
        job_id=run.job_id,
        state=state,
        run_url=f"/v1/rag/runs/{run.id}",
        job_url=f"/v1/jobs/{run.job_id}",
    )


def _public_run(run: RagRunRecord, state: JobState) -> PublicRagRun:
    return PublicRagRun(
        id=run.id,
        job_id=run.job_id,
        root_run_id=run.root_run_id,
        parent_run_id=run.parent_run_id,
        knowledge_base_id=run.knowledge_base_id,
        goal_digest=run.goal_digest,
        model_profile=run.model_profile,
        model_profile_version=run.model_profile_version,
        retrieval_profile=run.retrieval_profile,
        dry_run=run.dry_run,
        budget=PublicRagBudget(
            max_pages=run.budget.max_pages,
            max_steps=run.budget.max_steps,
            max_model_tokens=run.budget.max_model_tokens,
            max_context_chars=run.budget.max_context_chars,
            max_page_chars=run.budget.max_page_chars,
            per_call_timeout_seconds=run.budget.per_call_timeout_seconds,
            max_page_attempts=run.budget.max_page_attempts,
            max_conflict_retries=run.budget.max_conflict_retries,
        ),
        usage=PublicRagUsage(steps=run.usage.steps, model_tokens=run.usage.model_tokens),
        state=state,
        completion_reason=run.completion_reason,
        last_committed_ordinal=run.last_committed_ordinal,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _public_step(step: RagStepRecord) -> PublicRagStep:
    error_code = step.error_code
    if error_code is not None and error_code not in _PUBLIC_RAG_ERROR_CODES:
        error_code = RAG_ERROR_CONTRACTS["internal_error"].code
    return PublicRagStep(
        id=step.id,
        run_id=step.run_id,
        run_page_id=step.run_page_id,
        knowledge_base_id=step.knowledge_base_id,
        sequence=step.sequence,
        type=step.step_type,
        status=step.status,
        citation_identities=[
            PublicRagCitation(
                document_id=item.document_id,
                document_version=item.document_version,
                chunk_index=item.chunk_index,
                page=item.page,
            )
            for item in step.citation_identities
        ],
        model_profile_version=step.model_profile_version,
        input_tokens=step.input_tokens,
        output_tokens=step.output_tokens,
        total_tokens=step.total_tokens,
        latency_ms=step.latency_ms,
        error_code=error_code,
        created_at=step.created_at,
        updated_at=step.updated_at,
    )


@router.post(
    "/build-wiki",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=PublicRagAccepted,
)
async def create_build_wiki(
    body: BuildWikiRequest,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    idempotency_key: IdempotencyKey,
):
    failure: BaseException | None = None
    response = None
    try:
        raw_run = await rag_service.create(
            CreateRagRun(
                knowledge_base_id=body.knowledge_base_id,
                goal=body.goal,
                target_path_prefix=body.target_path_prefix,
                model_profile=body.model_profile,
                retrieval_profile=body.retrieval_profile,
                dry_run=body.dry_run,
                budget=body.budget.to_domain(),
            ),
            authenticated_user_id=authenticated_user_id,
            idempotency_key=idempotency_key,
        )
        run = _validate_run_record(
            raw_run,
            authenticated_user_id=authenticated_user_id,
            operation="create",
            knowledge_base_id=body.knowledge_base_id,
        )
        state = await _public_job_state(run, authenticated_user_id, job_service)
        response = _accepted(run, state)
    except BaseException as caught:  # noqa: BLE001 - sanitize service and projection boundaries.
        failure = caught
    if failure is not None:
        _raise_route_failure(failure)
    return response


@router.get("/runs/{run_id}", response_model=PublicRagRun)
async def get_run(
    run_id: UUID,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
    job_service: Annotated[JobService, Depends(get_job_service)],
):
    failure: BaseException | None = None
    response = None
    missing = False
    try:
        raw_run = await rag_service.get(run_id, authenticated_user_id=authenticated_user_id)
        missing = raw_run is None
        if not missing:
            run = _validate_run_record(
                raw_run,
                authenticated_user_id=authenticated_user_id,
                operation="get",
                resource_id=run_id,
            )
            state = await _public_job_state(run, authenticated_user_id, job_service)
            response = _public_run(run, state)
    except BaseException as caught:  # noqa: BLE001 - sanitize service and projection boundaries.
        failure = caught
    if failure is not None:
        _raise_route_failure(failure)
    if missing:
        raise _not_found()
    return response


@router.get("/runs/{run_id}/steps", response_model=PublicRagStepsPage)
async def get_run_steps(
    run_id: UUID,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
    after_sequence: Annotated[int, Query(alias="after", ge=0, le=MAX_STEPS)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    failure: BaseException | None = None
    response = None
    missing = False
    try:
        raw_run = await rag_service.get(run_id, authenticated_user_id=authenticated_user_id)
        missing = raw_run is None
        if not missing:
            run = _validate_run_record(
                raw_run,
                authenticated_user_id=authenticated_user_id,
                operation="steps",
                resource_id=run_id,
            )
            raw_records = await rag_service.steps(
                run_id,
                authenticated_user_id=authenticated_user_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            if raw_records is None:
                raise _InvalidProjection()
            records = _validate_step_records(
                raw_records,
                run=run,
                after_sequence=after_sequence,
                limit=limit,
            )
            items = [_public_step(item) for item in records]
            next_cursor = records[-1].sequence if len(records) == limit else None
            response = PublicRagStepsPage(items=items, next_cursor=next_cursor)
    except BaseException as caught:  # noqa: BLE001 - sanitize service and projection boundaries.
        failure = caught
    if failure is not None:
        _raise_route_failure(failure)
    if missing:
        raise _not_found()
    return response


@router.post(
    "/runs/{run_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=PublicRagAccepted,
)
async def resume_run(
    run_id: UUID,
    body: ResumeRunRequest,
    authenticated_user_id: Annotated[UUID, Depends(_get_authenticated_user_id)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
    job_service: Annotated[JobService, Depends(get_job_service)],
    idempotency_key: IdempotencyKey,
):
    failure: BaseException | None = None
    response = None
    missing = False
    try:
        raw_run = await rag_service.resume(
            run_id,
            authenticated_user_id=authenticated_user_id,
            idempotency_key=idempotency_key,
            budget_override=body.budget.explicit_values(),
        )
        missing = raw_run is None
        if not missing:
            run = _validate_run_record(
                raw_run,
                authenticated_user_id=authenticated_user_id,
                operation="resume",
                resource_id=run_id,
            )
            state = await _public_job_state(run, authenticated_user_id, job_service)
            response = _accepted(run, state)
    except BaseException as caught:  # noqa: BLE001 - sanitize service and projection boundaries.
        failure = caught
    if failure is not None:
        _raise_route_failure(failure)
    if missing:
        raise _not_found()
    return response


__all__ = [
    "BuildWikiRequest",
    "PublicRagRun",
    "PublicRagStep",
    "PublicRagStepsPage",
    "ResumeRunRequest",
    "router",
]
