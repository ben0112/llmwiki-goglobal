"""Bounded, checkpointed execution of one durable RAG page."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from uuid import UUID, uuid4

from jobs import repository as jobs_repository
from jobs.models import JobRecord, LeaseLost

from llmwiki_core.rag import (
    RagCitation,
    RagDomainError,
    RagPageState,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
    new_rag_error,
)
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery, SearchResult
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown
from llmwiki_core.wiki import DuplicateDocumentError, VersionConflict

from . import repository
from .model import InvalidRagModelResponse, RagModelResponse, RagModelUnavailable, RagTokenUsage
from .ports import AtomicPageCommit, PageExecutionResult
from .prompts import (
    WRITER_PROMPT_DIGEST,
    WRITER_PROMPT_VERSION,
    build_writer_messages,
    parse_draft,
)
from .records import RagPageRecord, RagRunRecord, RagStepRecord
from .retrieval import RagEvidence, RagWikiPage
from .validation import validate_draft
from .wiki_writer import InjectedCrash, PersistedRagLintError

_ZERO_TOKENS = RagTokenUsage(0, 0, 0)
_MAX_WRITER_OUTPUT_TOKENS = 8_192
_PREVIEW_BYTES = 16_384
_REPAIR_MESSAGE = {
    "role": "system",
    "content": "Repair the draft once. Return only a corrected response matching the required schema.",
}
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _invalid_draft_failure() -> RagDomainError:
    return RagDomainError("rag_invalid_draft", "The generated draft was invalid.")


def _phase_error_code(failure: BaseException) -> str:
    if isinstance(failure, RagModelUnavailable):
        return "rag_model_unavailable"
    if isinstance(failure, InvalidRagModelResponse):
        return "rag_invalid_draft"
    if isinstance(failure, RetrieverUnavailable):
        return "rag_retrieval_failed"
    if isinstance(failure, RagDomainError) and failure.code in {
        "rag_invalid_draft",
        "rag_invalid_model_usage",
        "rag_version_conflict",
    }:
        return failure.code
    return "rag_internal_error"


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_citations(evidence: tuple[RagEvidence, ...]) -> tuple[RagCitation, ...]:
    return tuple(RagCitation(item.document_id, item.document_version, item.chunk_index, item.page) for item in evidence)


def _bounded_preview(content: str) -> tuple[str, str, int, bool]:
    encoded = content.encode("utf-8")
    if len(encoded) <= _PREVIEW_BYTES:
        preview = content
    else:
        clipped = encoded[:_PREVIEW_BYTES]
        while clipped:
            try:
                preview = clipped.decode("utf-8")
                break
            except UnicodeDecodeError:
                clipped = clipped[:-1]
        else:  # pragma: no cover - every nonempty UTF-8 string has a first code point.
            preview = ""
    digest = hashlib.sha256(encoded).hexdigest()
    return preview, digest, len(content), preview != content


def _reserve_writer_call(run: RagRunRecord, usage: RagUsage, messages) -> tuple[int, int]:
    input_tokens = max(1, (sum(len(item["content"].encode("utf-8")) for item in messages) + 3) // 4)
    available_output = run.budget.max_model_tokens - usage.model_tokens - input_tokens
    max_output_tokens = min(_MAX_WRITER_OUTPUT_TOKENS, available_output)
    if max_output_tokens < 1:
        raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.")
    reservation = input_tokens + max_output_tokens
    usage.reserve_model_call(run.budget, reservation)
    return reservation, max_output_tokens


def _invalid_response_accounting(
    failure: BaseException,
    run: RagRunRecord,
    usage: RagUsage,
    reservation: int,
) -> tuple[RagTokenUsage, RagUsage, bool, bool]:
    charged = RagTokenUsage(reservation, 0, reservation)
    charged_usage = RagUsage(usage.steps, usage.model_tokens + reservation)
    if type(failure) is not InvalidRagModelResponse:
        return charged, charged_usage, False, False
    try:
        reported = failure.usage
    except BaseException:  # noqa: BLE001 - an exact instance may still have a corrupted attribute.
        return charged, charged_usage, False, False
    if reported is None:
        return _ZERO_TOKENS, usage, False, True
    if type(reported) is not RagTokenUsage:
        return charged, charged_usage, False, False
    if reported.total_tokens == 0:
        return _ZERO_TOKENS, usage, False, True
    if reported.total_tokens <= reservation:
        return reported, usage.commit_model_usage(run.budget, reservation, reported.total_tokens), True, True
    return charged, charged_usage, False, True


def _validated_search_result(value: object) -> SearchResult:
    if type(value) is not SearchResult:
        raise TypeError("RAG retrieval result is invalid")
    if type(value.hits) is not tuple or any(type(hit) is not SearchHit for hit in value.hits):
        raise TypeError("RAG retrieval result is invalid")
    validated: SearchResult | None = None
    try:
        hits = tuple(
            SearchHit(
                document_id=hit.document_id,
                document_version=hit.document_version,
                chunk_index=hit.chunk_index,
                content=hit.content,
                score=hit.score,
                path=hit.path,
                title=hit.title,
                page=hit.page,
                header_breadcrumb=hit.header_breadcrumb,
                tags=hit.tags,
                document_kind=hit.document_kind,
                metadata=hit.metadata,
            )
            for hit in value.hits
        )
        validated = SearchResult(
            hits=hits,
            candidate_count=value.candidate_count,
            latency_ms=value.latency_ms,
            profile=value.profile,
        )
    except (TypeError, ValueError):
        pass
    if validated is None:
        raise TypeError("RAG retrieval result is invalid") from None
    return validated


_RUN_IMMUTABLE_FIELDS = (
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
    "created_at",
)
_PAGE_IMMUTABLE_FIELDS = (
    "id",
    "run_id",
    "user_id",
    "knowledge_base_id",
    "ordinal",
    "path",
    "intent",
    "query",
    "created_at",
)
_STEP_IMMUTABLE_FIELDS = (
    "id",
    "run_id",
    "run_page_id",
    "user_id",
    "knowledge_base_id",
    "sequence",
    "step_type",
    "input_digest",
    "model_profile_version",
    "reserved_tokens",
    "created_at",
)


def _invalid_state(message: str) -> TypeError:
    return TypeError(message)


def _same_fields(left: object, right: object, fields: tuple[str, ...]) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _validated_usage(value: object, *, expected: RagUsage, run: RagRunRecord) -> RagUsage:
    if (
        type(value) is not RagUsage
        or value.steps < expected.steps
        or value.model_tokens < expected.model_tokens
        or value.steps > run.budget.max_steps
        or value.model_tokens > run.budget.max_model_tokens
    ):
        raise _invalid_state("RAG usage is invalid") from None
    return value


def _validated_run_progress(
    value: object,
    *,
    expected: RagRunRecord,
    usage: RagUsage | None = None,
    last_committed_ordinal: int | None = None,
) -> RagRunRecord:
    if (
        type(value) is not RagRunRecord
        or not _same_fields(value, expected, _RUN_IMMUTABLE_FIELDS)
        or value.completion_reason is not expected.completion_reason
        or type(value.last_committed_ordinal) is not int
        or type(value.updated_at) is not type(expected.updated_at)
        or value.updated_at < expected.updated_at
    ):
        raise _invalid_state("page runner state is invalid") from None
    _validated_usage(value.usage, expected=expected.usage, run=expected)
    if usage is not None and value.usage != usage:
        raise _invalid_state("page runner state is invalid") from None
    if last_committed_ordinal is not None and value.last_committed_ordinal != last_committed_ordinal:
        raise _invalid_state("page runner state is invalid") from None
    if value.last_committed_ordinal < expected.last_committed_ordinal:
        raise _invalid_state("page runner state is invalid") from None
    return value


def _validated_page_identity(value: object, *, expected: RagPageRecord, run: RagRunRecord) -> RagPageRecord:
    if (
        type(value) is not RagPageRecord
        or not _same_fields(value, expected, _PAGE_IMMUTABLE_FIELDS)
        or value.run_id != run.id
        or value.user_id != run.user_id
        or value.knowledge_base_id != run.knowledge_base_id
        or value.updated_at < expected.updated_at
        or type(value.attempt_count) is not int
        or type(value.conflict_retry_count) is not int
        or type(value.last_completed_step_sequence) is not int
    ):
        raise _invalid_state("RAG page result is invalid") from None
    return value


def _validated_reload_page(value: object, *, expected: RagPageRecord, run: RagRunRecord) -> RagPageRecord:
    page = _validated_page_identity(value, expected=expected, run=run)
    allowed_states = {
        RagPageState.PLANNED: {
            RagPageState.PLANNED,
            RagPageState.RUNNING,
            RagPageState.FAILED,
            RagPageState.COMMITTED,
            RagPageState.DRY_RUN_COMPLETE,
        },
        RagPageState.RUNNING: {
            RagPageState.RUNNING,
            RagPageState.FAILED,
            RagPageState.COMMITTED,
            RagPageState.DRY_RUN_COMPLETE,
        },
        RagPageState.FAILED: {
            RagPageState.FAILED,
            RagPageState.RUNNING,
            RagPageState.COMMITTED,
            RagPageState.DRY_RUN_COMPLETE,
        },
        RagPageState.COMMITTED: {RagPageState.COMMITTED},
        RagPageState.DRY_RUN_COMPLETE: {RagPageState.DRY_RUN_COMPLETE},
    }
    if (
        page.state not in allowed_states[expected.state]
        or not expected.attempt_count <= page.attempt_count <= run.budget.max_page_attempts
        or not expected.conflict_retry_count <= page.conflict_retry_count <= run.budget.max_conflict_retries
        or page.last_completed_step_sequence < expected.last_completed_step_sequence
        or page.last_completed_step_sequence > run.usage.steps
        or (page.state is RagPageState.PLANNED and page.attempt_count != 0)
        or (page.state is RagPageState.RUNNING and page.attempt_count < 1)
        or (page.state is not RagPageState.COMMITTED and (page.document_id is None) is not (page.version_read is None))
        or (page.state is not RagPageState.COMMITTED and page.version_committed is not None)
        or (
            page.state is RagPageState.COMMITTED
            and (
                page.document_id is None
                or page.version_committed is None
                or page.version_committed < 1
                or page.version_read != (page.version_committed - 1 or None)
            )
        )
        or (
            page.state is RagPageState.DRY_RUN_COMPLETE
            and (
                page.document_id is not None
                or page.version_read is not None
                or page.version_committed is not None
                or page.preview is None
                or page.preview_digest is None
                or page.preview_full_char_count is None
            )
        )
    ):
        raise _invalid_state("page runner state is invalid") from None
    return page


def _validated_attempt_page(value: object, *, expected: RagPageRecord, run: RagRunRecord) -> RagPageRecord:
    page = _validated_page_identity(value, expected=expected, run=run)
    unchanged = (
        "document_id",
        "version_read",
        "version_committed",
        "conflict_retry_count",
        "last_completed_step_sequence",
        "preview",
        "preview_digest",
        "preview_full_char_count",
        "preview_truncated",
        "lint_summary",
    )
    if (
        page.state is not RagPageState.RUNNING
        or page.attempt_count != expected.attempt_count + 1
        or page.attempt_count > run.budget.max_page_attempts
        or not _same_fields(page, expected, unchanged)
    ):
        raise _invalid_state("RAG page result is invalid") from None
    return page


def _validated_read_page(
    value: object,
    *,
    expected: RagPageRecord,
    run: RagRunRecord,
    document_id: UUID | None,
    version: int | None,
) -> RagPageRecord:
    page = _validated_page_identity(value, expected=expected, run=run)
    unchanged = (
        "state",
        "attempt_count",
        "conflict_retry_count",
        "version_committed",
        "last_completed_step_sequence",
        "preview",
        "preview_digest",
        "preview_full_char_count",
        "preview_truncated",
        "lint_summary",
    )
    if (
        page.state is not RagPageState.RUNNING
        or page.document_id != document_id
        or page.version_read != version
        or not _same_fields(page, expected, unchanged)
    ):
        raise _invalid_state("RAG page result is invalid") from None
    return page


def _validated_conflict_page(value: object, *, expected: RagPageRecord, run: RagRunRecord) -> RagPageRecord:
    page = _validated_page_identity(value, expected=expected, run=run)
    unchanged = (
        "state",
        "document_id",
        "version_read",
        "version_committed",
        "attempt_count",
        "last_completed_step_sequence",
        "preview",
        "preview_digest",
        "preview_full_char_count",
        "preview_truncated",
        "lint_summary",
    )
    if (
        page.state is not RagPageState.RUNNING
        or page.conflict_retry_count != expected.conflict_retry_count + 1
        or page.conflict_retry_count > run.budget.max_conflict_retries
        or not _same_fields(page, expected, unchanged)
    ):
        raise _invalid_state("RAG page result is invalid") from None
    return page


def _validated_execution_result(
    value: object,
    *,
    expected_run: RagRunRecord,
    expected_page: RagPageRecord,
) -> PageExecutionResult:
    if type(value) is not PageExecutionResult:
        raise _invalid_state("page runner state is invalid") from None
    invalid_page = False
    try:
        current_run = _validated_run_progress(value.run, expected=expected_run)
        current_page = _validated_reload_page(value.page, expected=expected_page, run=current_run)
    except TypeError:
        invalid_page = True
    if invalid_page:
        raise _invalid_state("page runner state is invalid") from None
    expected_last = expected_run.last_committed_ordinal
    if current_page.state is RagPageState.COMMITTED:
        expected_last = current_page.ordinal
    if current_run.last_committed_ordinal != expected_last:
        raise _invalid_state("page runner state is invalid") from None
    if (current_page.state is RagPageState.COMMITTED and current_run.dry_run) or (
        current_page.state is RagPageState.DRY_RUN_COMPLETE and not current_run.dry_run
    ):
        raise _invalid_state("page runner state is invalid") from None
    return value


def _validated_started_step(
    value: object,
    *,
    run: RagRunRecord,
    page: RagPageRecord,
    step_type: RagStepType,
    input_digest: str,
    reserved_tokens: int,
    expected_sequence: int,
) -> RagStepRecord:
    if (
        type(value) is not RagStepRecord
        or value.run_id != run.id
        or value.run_page_id != page.id
        or value.user_id != run.user_id
        or value.knowledge_base_id != run.knowledge_base_id
        or type(value.sequence) is not int
        or value.sequence != expected_sequence
        or value.step_type is not step_type
        or value.status is not RagStepStatus.RUNNING
        or value.input_digest != input_digest
        or value.model_profile_version != run.model_profile_version
        or type(value.reserved_tokens) is not int
        or value.reserved_tokens != reserved_tokens
        or type(value.output_summary) is not _MAPPING_PROXY_TYPE
        or bool(value.output_summary)
        or type(value.citation_identities) is not tuple
        or bool(value.citation_identities)
        or value.prompt_version is not None
        or value.prompt_digest is not None
        or type(value.input_tokens) is not int
        or value.input_tokens != 0
        or type(value.output_tokens) is not int
        or value.output_tokens != 0
        or type(value.total_tokens) is not int
        or value.total_tokens != 0
        or type(value.latency_ms) is not float
        or value.latency_ms != 0
        or value.error_code is not None
        or value.error_message is not None
        or type(value.updated_at) is not type(value.created_at)
        or value.updated_at < value.created_at
    ):
        raise _invalid_state("RAG step result is invalid") from None
    return value


def _validated_finished_step(
    value: object,
    *,
    step: RagStepRecord,
    run: RagRunRecord,
    page: RagPageRecord,
    status: RagStepStatus,
    summary: Mapping[str, object],
    citations: tuple[RagCitation, ...],
    token_usage: RagTokenUsage,
    latency_ms: float,
    error_code: str | None,
    prompt_version: str | None,
    prompt_digest: str | None,
) -> RagStepRecord:
    if (
        type(value) is not RagStepRecord
        or not _same_fields(value, step, _STEP_IMMUTABLE_FIELDS)
        or value.run_id != run.id
        or value.run_page_id != page.id
        or value.user_id != run.user_id
        or value.knowledge_base_id != run.knowledge_base_id
        or value.status is not status
        or type(value.output_summary) is not _MAPPING_PROXY_TYPE
        or value.output_summary != summary
        or type(value.citation_identities) is not tuple
        or value.citation_identities != citations
        or value.prompt_version != prompt_version
        or value.prompt_digest != prompt_digest
        or type(value.input_tokens) is not int
        or value.input_tokens != token_usage.prompt_tokens
        or type(value.output_tokens) is not int
        or value.output_tokens != token_usage.completion_tokens
        or type(value.total_tokens) is not int
        or value.total_tokens != token_usage.total_tokens
        or type(value.latency_ms) is not float
        or value.latency_ms != float(latency_ms)
        or value.error_code != error_code
        or value.error_message is not None
        or type(value.updated_at) is not type(step.updated_at)
        or value.updated_at < step.updated_at
    ):
        raise _invalid_state("RAG step result is invalid") from None
    return value


def _validated_dry_result(
    value: object,
    *,
    expected_run: RagRunRecord,
    expected_page: RagPageRecord,
    usage: RagUsage,
    preview: str,
    preview_digest: str,
    preview_full_char_count: int,
    preview_truncated: bool,
) -> PageExecutionResult:
    if type(value) is not PageExecutionResult:
        raise _invalid_state("page runner state is invalid") from None
    run = _validated_run_progress(
        value.run,
        expected=expected_run,
        usage=usage,
        last_committed_ordinal=expected_run.last_committed_ordinal,
    )
    page = _validated_page_identity(value.page, expected=expected_page, run=run)
    if (
        not run.dry_run
        or page.state is not RagPageState.DRY_RUN_COMPLETE
        or page.document_id is not None
        or page.version_read is not None
        or page.version_committed is not None
        or page.attempt_count != expected_page.attempt_count
        or page.conflict_retry_count != expected_page.conflict_retry_count
        or page.last_completed_step_sequence != usage.steps
        or page.preview != preview
        or page.preview_digest != preview_digest
        or page.preview_full_char_count != preview_full_char_count
        or page.preview_truncated is not preview_truncated
    ):
        raise _invalid_state("page runner state is invalid") from None
    return value


def _validated_commit(
    value: object,
    *,
    expected_run: RagRunRecord,
    expected_page: RagPageRecord,
    usage: RagUsage,
    bundle,
) -> AtomicPageCommit:
    if type(value) is not AtomicPageCommit:
        raise _invalid_state("atomic page commit is invalid") from None
    invalid_projection = False
    try:
        run = _validated_run_progress(
            value.run,
            expected=expected_run,
            usage=usage,
            last_committed_ordinal=expected_page.ordinal,
        )
        page = _validated_page_identity(value.page, expected=expected_page, run=run)
    except TypeError:
        invalid_projection = True
    if invalid_projection:
        raise _invalid_state("atomic page commit is invalid") from None
    expected_version = (bundle.expected_version or 0) + 1
    if (
        run.dry_run
        or expected_page.ordinal != expected_run.last_committed_ordinal + 1
        or page.state is not RagPageState.COMMITTED
        or page.document_id != value.written.document_id
        or str(page.document_id) != bundle.document_id
        or page.version_read != bundle.expected_version
        or page.version_committed != expected_version
        or value.written.version != expected_version
        or page.attempt_count != expected_page.attempt_count
        or page.conflict_retry_count != expected_page.conflict_retry_count
        or page.last_completed_step_sequence != usage.steps
        or page.preview is not None
        or page.preview_digest is not None
        or page.preview_full_char_count is not None
        or page.preview_truncated
        or dict(page.lint_summary or {}) != dict(value.lint_summary)
    ):
        raise _invalid_state("atomic page commit is invalid") from None
    return value


def _validated_lint_summary(value: object) -> dict[str, int]:
    if type(value) is not dict or len(value) != 1:
        raise _invalid_draft_failure()
    key = next(iter(value))
    if type(key) is not str or key != "warnings":
        raise _invalid_draft_failure()
    warnings = dict.__getitem__(value, "warnings")
    if type(warnings) is not int or not 0 <= warnings <= 1_000_000:
        raise _invalid_draft_failure()
    return {"warnings": warnings}


class PostgresPageStore:
    """Lease-fence page-step mutations while retaining repository transaction rules."""

    def __init__(self, pool) -> None:
        self._pool = pool

    @staticmethod
    async def _assert_binding(conn, *, job_id, lease_owner, run, page=None, step=None):
        job = await jobs_repository.assert_active(conn, job_id, lease_owner)
        return await repository.assert_page_job_binding(
            conn,
            job=job,
            run=run,
            page=page,
            step=step,
        )

    async def reload_boundary(self, *, run: RagRunRecord, page: RagPageRecord, job_id: UUID, lease_owner: str):
        async with self._pool.acquire() as conn, conn.transaction():
            current_run, current_page, _ = await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
        if current_page is None:  # pragma: no cover - page was supplied above.
            raise RagDomainError("rag_page_not_found", "The RAG page was not found.")
        return _validated_execution_result(
            PageExecutionResult(current_run, current_page),
            expected_run=run,
            expected_page=page,
        )

    async def load_usage(self, *, run: RagRunRecord, job_id: UUID, lease_owner: str) -> RagUsage:
        async with self._pool.acquire() as conn, conn.transaction():
            current_run, _, _ = await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
            )
            aggregate = await conn.fetchrow(
                "SELECT count(*) FILTER (WHERE status IN ('succeeded','failed')) AS steps,"
                "COALESCE(sum(total_tokens) FILTER (WHERE status IN ('succeeded','failed')),0) AS model_tokens "
                "FROM rag_steps WHERE run_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                current_run.id,
                current_run.user_id,
                current_run.knowledge_base_id,
            )
        return _validated_usage(
            RagUsage(steps=aggregate["steps"], model_tokens=aggregate["model_tokens"]),
            expected=run.usage,
            run=run,
        )

    async def begin_page_attempt(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        max_attempts: int,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
            attempted = await repository.begin_page_attempt(conn, page.id, max_attempts=max_attempts)
            if attempted is not None:
                attempted = _validated_attempt_page(attempted, expected=page, run=run)
        if attempted is None:
            raise new_rag_error("page_attempts_exhausted")
        return attempted

    async def record_page_read(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        document_id: UUID | None,
        version: int | None,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
            recorded = await repository.record_page_read(
                conn,
                run=run,
                page=page,
                document_id=document_id,
                version=version,
            )
            return _validated_read_page(
                recorded,
                expected=page,
                run=run,
                document_id=document_id,
                version=version,
            )

    async def start_step(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        step_type: RagStepType,
        input_digest: str,
        reserved_tokens: int = 0,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
            started = await repository.start_step(
                conn,
                run_id=run.id,
                page_id=page.id,
                step_type=step_type,
                input_digest=input_digest,
                reserved_tokens=reserved_tokens,
            )
            terminal_steps = await conn.fetchval(
                "SELECT count(*) FROM rag_steps WHERE run_id=$1 AND status IN ('succeeded','failed')",
                run.id,
            )
            return _validated_started_step(
                started,
                run=run,
                page=page,
                step_type=step_type,
                input_digest=input_digest,
                reserved_tokens=reserved_tokens,
                expected_sequence=terminal_steps + 1,
            )

    async def finish_step(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        step,
        status: RagStepStatus,
        summary: Mapping[str, object],
        citations: tuple[RagCitation, ...],
        token_usage: RagTokenUsage,
        latency_ms: float = 0.0,
        error_code: str | None = None,
        prompt_version: str | None = None,
        prompt_digest: str | None = None,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
                step=step,
            )
            finished = await repository.finish_step(
                conn,
                step_id=step.id,
                status=status,
                summary=summary,
                citations=citations,
                usage=RagUsage(steps=1, model_tokens=token_usage.total_tokens),
                latency_ms=latency_ms,
                error_code=error_code,
                token_usage=token_usage,
                prompt_version=prompt_version,
                prompt_digest=prompt_digest,
            )
            return _validated_finished_step(
                finished,
                step=step,
                run=run,
                page=page,
                status=status,
                summary=summary,
                citations=citations,
                token_usage=token_usage,
                latency_ms=latency_ms,
                error_code=error_code,
                prompt_version=prompt_version,
                prompt_digest=prompt_digest,
            )

    async def record_conflict(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        max_conflict_retries: int,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
            conflicted = await repository.record_page_conflict(
                conn,
                run=run,
                page=page,
                max_conflict_retries=max_conflict_retries,
            )
            return _validated_conflict_page(conflicted, expected=page, run=run)

    async def complete_dry_run(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        usage: RagUsage,
        preview: str,
        preview_digest: str,
        preview_full_char_count: int,
        preview_truncated: bool,
    ):
        async with self._pool.acquire() as conn, conn.transaction():
            await self._assert_binding(
                conn,
                job_id=job_id,
                lease_owner=lease_owner,
                run=run,
                page=page,
            )
            updated_run, updated_page = await repository.mark_dry_run_complete(
                conn,
                run=run,
                page=page,
                usage=usage,
                preview=preview,
                preview_digest=preview_digest,
                preview_full_char_count=preview_full_char_count,
                preview_truncated=preview_truncated,
            )
            return _validated_dry_result(
                PageExecutionResult(updated_run, updated_page),
                expected_run=run,
                expected_page=page,
                usage=usage,
                preview=preview,
                preview_digest=preview_digest,
                preview_full_char_count=preview_full_char_count,
                preview_truncated=preview_truncated,
            )


class PageRunner:
    """Execute retrieve/read/draft/validate/prepublish/lint and atomic commit."""

    def __init__(
        self,
        *,
        store,
        model,
        retrieval,
        evidence_reader,
        wiki_page_reader,
        wiki_writer,
        draft_linter,
        feature_gate,
    ) -> None:
        self._store = store
        self._model = model
        self._retrieval = retrieval
        self._evidence_reader = evidence_reader
        self._wiki_page_reader = wiki_page_reader
        self._wiki_writer = wiki_writer
        self._draft_linter = draft_linter
        self._feature_gate = feature_gate
        self.before_commit: Callable[[], object] | None = None

    @property
    def failpoint(self) -> str | None:
        value = getattr(self._wiki_writer, "failpoint", None)
        return value if type(value) is str else None

    @failpoint.setter
    def failpoint(self, value: str | None) -> None:
        if value is not None and type(value) is not str:
            raise ValueError("page runner failpoint is invalid")
        if not hasattr(self._wiki_writer, "failpoint"):
            raise ValueError("page runner writer does not support failpoints")
        self._wiki_writer.failpoint = value

    async def _checkpoint(self, run: RagRunRecord, lease, usage: RagUsage) -> JobRecord:
        job = await lease.checkpoint()
        if type(job) is not JobRecord or job.id != run.job_id:
            raise LeaseLost("RAG job lease is no longer active")
        await self._feature_gate.ensure_enabled()
        if usage.steps > run.budget.max_steps or usage.model_tokens > run.budget.max_model_tokens:
            raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.")
        return job

    @staticmethod
    def _owner(job: JobRecord) -> str:
        if type(job.lease_owner) is not str or not job.lease_owner:
            raise LeaseLost("RAG job lease is no longer active")
        return job.lease_owner

    async def _start(self, run, page, lease, usage, step_type, input_value, *, reserved_tokens=0):
        next_usage = usage.consume_step(run.budget)
        job = await self._checkpoint(run, lease, next_usage)
        input_digest = _digest(input_value)
        raw_step = await self._store.start_step(
            job_id=job.id,
            lease_owner=self._owner(job),
            run=run,
            page=page,
            step_type=step_type,
            input_digest=input_digest,
            reserved_tokens=reserved_tokens,
        )
        invalid_step = False
        try:
            step = _validated_started_step(
                raw_step,
                run=run,
                page=page,
                step_type=step_type,
                input_digest=input_digest,
                reserved_tokens=reserved_tokens,
                expected_sequence=next_usage.steps,
            )
        except TypeError:
            invalid_step = True
        if invalid_step:
            if type(raw_step) is RagStepRecord:
                try:
                    await self._store.finish_step(
                        job_id=job.id,
                        lease_owner=self._owner(job),
                        run=run,
                        page=page,
                        step=raw_step,
                        status=RagStepStatus.FAILED,
                        summary={"outcome": "failed", "phase": step_type.value, "usage_trusted": False},
                        citations=(),
                        token_usage=_ZERO_TOKENS,
                        error_code="rag_internal_error",
                        prompt_version=None,
                        prompt_digest=None,
                    )
                except BaseException as cleanup_failure:  # noqa: BLE001 - preserve only control signals.
                    if (signal := sanitized_boundary_signal_or_unknown(cleanup_failure)) is not None:
                        raise signal from None
            raise _invalid_state("RAG step result is invalid") from None
        return job, step, next_usage

    async def _finish(
        self,
        *,
        run,
        page,
        job,
        step,
        status=RagStepStatus.SUCCEEDED,
        summary,
        citations=(),
        token_usage=_ZERO_TOKENS,
        error_code=None,
        prompt_version=None,
        prompt_digest=None,
    ):
        result = await self._store.finish_step(
            job_id=job.id,
            lease_owner=self._owner(job),
            run=run,
            page=page,
            step=step,
            status=status,
            summary=summary,
            citations=tuple(citations),
            token_usage=token_usage,
            latency_ms=0.0,
            error_code=error_code,
            prompt_version=prompt_version,
            prompt_digest=prompt_digest,
        )
        return _validated_finished_step(
            result,
            step=step,
            run=run,
            page=page,
            status=status,
            summary=summary,
            citations=tuple(citations),
            token_usage=token_usage,
            latency_ms=0.0,
            error_code=error_code,
            prompt_version=prompt_version,
            prompt_digest=prompt_digest,
        )

    async def _close_failed_phase(
        self,
        *,
        run,
        page,
        lease,
        usage,
        step,
        step_type,
        failure,
        token_usage=_ZERO_TOKENS,
        usage_trusted=False,
        prompt_version=None,
        prompt_digest=None,
    ) -> None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        if signal is not None:
            raise signal from None
        try:
            job = await self._checkpoint(run, lease, usage)
            await self._finish(
                run=run,
                page=page,
                job=job,
                step=step,
                status=RagStepStatus.FAILED,
                summary={
                    "outcome": "failed",
                    "phase": step_type.value,
                    "usage_trusted": usage_trusted,
                },
                token_usage=token_usage,
                error_code=_phase_error_code(failure),
                prompt_version=prompt_version,
                prompt_digest=prompt_digest,
            )
        except BaseException as close_failure:  # noqa: BLE001 - never mask a control signal.
            signal = sanitized_boundary_signal_or_unknown(close_failure)
            if signal is not None:
                raise signal from None

    async def _successful_phase(
        self,
        run,
        page,
        lease,
        usage,
        step_type,
        input_value,
        operation,
        validator,
        summary,
    ):
        job, step, usage = await self._start(run, page, lease, usage, step_type, input_value)
        try:
            value = validator(await operation())
            await self._finish(
                run=run,
                page=page,
                job=job,
                step=step,
                summary=summary(value),
                citations=_evidence_citations(value) if step_type is RagStepType.READ else (),
            )
        except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
            await self._close_failed_phase(
                run=run,
                page=page,
                lease=lease,
                usage=usage,
                step=step,
                step_type=step_type,
                failure=failure,
            )
            raise
        return value, usage

    async def run(  # noqa: C901 - one bounded durable page state machine.
        self,
        run: RagRunRecord,
        page: RagPageRecord,
        lease,
    ) -> PageExecutionResult:
        if type(run) is not RagRunRecord or type(page) is not RagPageRecord or page.run_id != run.id:
            raise TypeError("page runner input is invalid")
        usage = run.usage
        initial_job = await self._checkpoint(run, lease, usage)
        reload_boundary = getattr(self._store, "reload_boundary", None)
        if callable(reload_boundary):
            authoritative = await reload_boundary(
                run=run,
                page=page,
                job_id=initial_job.id,
                lease_owner=self._owner(initial_job),
            )
            authoritative = _validated_execution_result(
                authoritative,
                expected_run=run,
                expected_page=page,
            )
            run, page, usage = authoritative.run, authoritative.page, authoritative.run.usage
            if page.state in {RagPageState.COMMITTED, RagPageState.DRY_RUN_COMPLETE}:
                return authoritative
            load_usage = getattr(self._store, "load_usage", None)
            if callable(load_usage):
                usage_before_load = usage
                usage = await load_usage(
                    run=run,
                    job_id=initial_job.id,
                    lease_owner=self._owner(initial_job),
                )
                usage = _validated_usage(usage, expected=usage_before_load, run=run)
        repair = False
        new_document_id: UUID | None = None
        while True:
            attempt_job = await self._checkpoint(run, lease, usage)
            attempt_input_page = page
            try:
                attempted_page = await self._store.begin_page_attempt(
                    job_id=attempt_job.id,
                    lease_owner=self._owner(attempt_job),
                    run=run,
                    page=page,
                    max_attempts=run.budget.max_page_attempts,
                )
                page = _validated_attempt_page(attempted_page, expected=attempt_input_page, run=run)
            except RagDomainError as failure:
                if failure.code == "rag_page_attempts_exhausted":
                    raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.") from None
                raise
            load_usage = getattr(self._store, "load_usage", None)
            if callable(load_usage):
                usage_before_load = usage
                usage = await load_usage(
                    run=run,
                    job_id=attempt_job.id,
                    lease_owner=self._owner(attempt_job),
                )
                usage = _validated_usage(usage, expected=usage_before_load, run=run)

            query = SearchQuery(page.query, limit=min(20, 100))
            result, usage = await self._successful_phase(
                run,
                page,
                lease,
                usage,
                RagStepType.RETRIEVE,
                {"query": page.query, "profile": run.retrieval_profile},
                lambda query=query, profile=run.retrieval_profile: self._retrieval.retrieve(query, profile=profile),
                _validated_search_result,
                lambda value: {
                    "candidate_count": value.candidate_count,
                    "returned_count": value.returned_count,
                },
            )

            async def read_inputs(result=result, target_page=page):
                evidence = await self._evidence_reader.read(
                    run.user_id,
                    run.knowledge_base_id,
                    result.hits,
                    run.budget.max_context_chars,
                )
                current = await self._wiki_page_reader.get_by_path(
                    run.user_id,
                    run.knowledge_base_id,
                    target_page.path,
                )
                if type(evidence) is not tuple or any(type(item) is not RagEvidence for item in evidence):
                    raise TypeError("RAG evidence is invalid")
                if current is not None and type(current) is not RagWikiPage:
                    raise TypeError("current wiki page is invalid")
                return evidence, current

            job, step, usage = await self._start(
                run,
                page,
                lease,
                usage,
                RagStepType.READ,
                {"hits": [hit.identity for hit in result.hits], "path": page.path},
            )
            try:
                evidence, current = await read_inputs()
                read_input_page = page
                read_document_id = None if current is None else current.document_id
                read_version = None if current is None else current.version
                read_page = await self._store.record_page_read(
                    job_id=job.id,
                    lease_owner=self._owner(job),
                    run=run,
                    page=page,
                    document_id=read_document_id,
                    version=read_version,
                )
                page = _validated_read_page(
                    read_page,
                    expected=read_input_page,
                    run=run,
                    document_id=read_document_id,
                    version=read_version,
                )
                citations = _evidence_citations(evidence)
                await self._finish(
                    run=run,
                    page=page,
                    job=job,
                    step=step,
                    summary={
                        "evidence_count": len(evidence),
                        "evidence_chars": sum(len(item.content) for item in evidence),
                    },
                    citations=citations,
                )
            except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=usage,
                    step=step,
                    step_type=RagStepType.READ,
                    failure=failure,
                )
                raise

            item = RagWorkItem(page.ordinal, page.path, page.intent, page.query)
            messages = build_writer_messages(
                goal=run.goal,
                item=item,
                current_page=current,
                evidence=evidence,
            )
            if repair:
                messages = (*messages, dict(_REPAIR_MESSAGE))
            reservation, max_output_tokens = _reserve_writer_call(
                run,
                usage.consume_step(run.budget),
                messages,
            )
            job, step, next_usage = await self._start(
                run,
                page,
                lease,
                usage,
                RagStepType.DRAFT,
                {"prompt_digest": WRITER_PROMPT_DIGEST, "repair": repair, "messages": messages},
                reserved_tokens=reservation,
            )
            detach_invalid_response = False
            try:
                response = await self._model.complete_json(
                    messages=messages,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=float(run.budget.per_call_timeout_seconds),
                )
                if type(response) is not RagModelResponse or type(response.usage) is not RagTokenUsage:
                    raise InvalidRagModelResponse()
                usage = next_usage.commit_model_usage(run.budget, reservation, response.usage.total_tokens)
            except InvalidRagModelResponse as failure:
                signal = sanitized_boundary_signal_or_unknown(failure)
                if signal is not None:
                    raise signal from None
                token_usage, aggregate_usage, usage_trusted, safe_exact_failure = _invalid_response_accounting(
                    failure,
                    run,
                    next_usage,
                    reservation,
                )
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=aggregate_usage,
                    step=step,
                    step_type=RagStepType.DRAFT,
                    failure=failure,
                    token_usage=token_usage,
                    usage_trusted=usage_trusted,
                    prompt_version=WRITER_PROMPT_VERSION,
                    prompt_digest=WRITER_PROMPT_DIGEST,
                )
                if safe_exact_failure:
                    raise
                detach_invalid_response = True
            except RagDomainError as failure:
                charged = (
                    RagTokenUsage(reservation, 0, reservation)
                    if failure.code == "rag_invalid_model_usage"
                    else _ZERO_TOKENS
                )
                aggregate_usage = (
                    RagUsage(next_usage.steps, next_usage.model_tokens + reservation)
                    if failure.code == "rag_invalid_model_usage"
                    else next_usage
                )
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=aggregate_usage,
                    step=step,
                    step_type=RagStepType.DRAFT,
                    failure=failure,
                    token_usage=charged,
                    prompt_version=WRITER_PROMPT_VERSION,
                    prompt_digest=WRITER_PROMPT_DIGEST,
                )
                raise
            except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=next_usage,
                    step=step,
                    step_type=RagStepType.DRAFT,
                    failure=failure,
                    prompt_version=WRITER_PROMPT_VERSION,
                    prompt_digest=WRITER_PROMPT_DIGEST,
                )
                raise
            if detach_invalid_response:
                raise _invalid_draft_failure() from None
            try:
                draft = parse_draft(response.payload)
            except RagDomainError as failure:
                await self._finish(
                    run=run,
                    page=page,
                    job=job,
                    step=step,
                    status=RagStepStatus.FAILED,
                    summary={"outcome": "rejected", "repair": repair},
                    token_usage=response.usage,
                    error_code=failure.code,
                    prompt_version=WRITER_PROMPT_VERSION,
                    prompt_digest=WRITER_PROMPT_DIGEST,
                )
                if repair:
                    raise _invalid_draft_failure() from None
                repair = True
                continue
            await self._finish(
                run=run,
                page=page,
                job=job,
                step=step,
                summary={
                    "citation_count": len(draft.citations),
                    "content_chars": len(draft.content),
                    "content_digest": hashlib.sha256(draft.content.encode("utf-8")).hexdigest(),
                    "repair": repair,
                },
                citations=draft.citations,
                token_usage=response.usage,
                prompt_version=WRITER_PROMPT_VERSION,
                prompt_digest=WRITER_PROMPT_DIGEST,
            )

            if current is None:
                new_document_id = new_document_id or uuid4()
                document_id, expected_version = new_document_id, None
            else:
                document_id, expected_version = current.document_id, current.version
            job, step, usage = await self._start(
                run,
                page,
                lease,
                usage,
                RagStepType.VALIDATE,
                {"content_digest": hashlib.sha256(draft.content.encode()).hexdigest(), "citations": draft.citations},
            )
            try:
                bundle, validation_summary = validate_draft(
                    draft,
                    evidence,
                    document_id=document_id,
                    expected_version=expected_version,
                    target_path=page.path,
                    max_page_chars=run.budget.max_page_chars,
                )
            except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                audit_failure = _invalid_draft_failure() if type(failure) is RagDomainError else failure
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=usage,
                    step=step,
                    step_type=RagStepType.VALIDATE,
                    failure=audit_failure,
                )
                if type(failure) is RagDomainError:
                    if repair:
                        raise _invalid_draft_failure() from None
                    repair = True
                    continue
                raise
            await self._finish(
                run=run,
                page=page,
                job=job,
                step=step,
                summary={"outcome": "accepted", **validation_summary},
                citations=draft.citations,
            )

            job, step, usage = await self._start(
                run,
                page,
                lease,
                usage,
                RagStepType.WRITE,
                {"document_id": bundle.document_id, "expected_version": bundle.expected_version},
            )
            try:
                await self._finish(
                    run=run,
                    page=page,
                    job=job,
                    step=step,
                    summary={
                        "outcome": "publication_prepared",
                        "content_digest": hashlib.sha256(bundle.content.encode("utf-8")).hexdigest(),
                    },
                    citations=draft.citations,
                )
            except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=usage,
                    step=step,
                    step_type=RagStepType.WRITE,
                    failure=failure,
                )
                raise

            job, step, usage = await self._start(
                run,
                page,
                lease,
                usage,
                RagStepType.LINT,
                {"content_digest": hashlib.sha256(bundle.content.encode()).hexdigest()},
            )
            try:
                lint_summary = _validated_lint_summary(
                    await self._draft_linter.lint(
                        run=run,
                        page=page,
                        content=bundle.content,
                        citations=draft.citations,
                    )
                )
            except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                await self._close_failed_phase(
                    run=run,
                    page=page,
                    lease=lease,
                    usage=usage,
                    step=step,
                    step_type=RagStepType.LINT,
                    failure=failure,
                )
                if isinstance(failure, RagDomainError) and failure.code == "rag_invalid_draft":
                    if repair:
                        raise _invalid_draft_failure() from None
                    repair = True
                    continue
                raise
            await self._finish(
                run=run,
                page=page,
                job=job,
                step=step,
                summary={"outcome": "prepublication_lint_complete", "warnings": lint_summary.get("warnings", 0)},
                citations=draft.citations,
            )

            if run.dry_run:
                preview, preview_digest, full_count, truncated = _bounded_preview(bundle.content)
                dry_job = await self._checkpoint(run, lease, usage)
                outcome = await self._store.complete_dry_run(
                    job_id=dry_job.id,
                    lease_owner=self._owner(dry_job),
                    run=run,
                    page=page,
                    usage=usage,
                    preview=preview,
                    preview_digest=preview_digest,
                    preview_full_char_count=full_count,
                    preview_truncated=truncated,
                )
                return _validated_dry_result(
                    outcome,
                    expected_run=run,
                    expected_page=page,
                    usage=usage,
                    preview=preview,
                    preview_digest=preview_digest,
                    preview_full_char_count=full_count,
                    preview_truncated=truncated,
                )

            if self.before_commit is not None:
                callback_result = self.before_commit()
                if inspect.isawaitable(callback_result):
                    await callback_result
            commit_job = await self._checkpoint(run, lease, usage)
            try:
                committed = await self._wiki_writer.commit(
                    job_id=commit_job.id,
                    lease_owner=self._owner(commit_job),
                    run=run,
                    page=page,
                    bundle=bundle,
                    usage=usage,
                )
            except (VersionConflict, DuplicateDocumentError):
                job, step, usage = await self._start(
                    run,
                    page,
                    lease,
                    usage,
                    RagStepType.CONFLICT,
                    {"document_id": bundle.document_id, "expected_version": bundle.expected_version},
                )
                if page.conflict_retry_count >= run.budget.max_conflict_retries:
                    terminal_conflict = RagDomainError(
                        "rag_version_conflict",
                        "The conflict retry limit was exhausted.",
                    )
                    await self._close_failed_phase(
                        run=run,
                        page=page,
                        lease=lease,
                        usage=usage,
                        step=step,
                        step_type=RagStepType.CONFLICT,
                        failure=terminal_conflict,
                    )
                    raise terminal_conflict from None
                try:
                    conflict_input_page = page
                    conflict_page = await self._store.record_conflict(
                        job_id=job.id,
                        lease_owner=self._owner(job),
                        run=run,
                        page=page,
                        max_conflict_retries=run.budget.max_conflict_retries,
                    )
                    page = _validated_conflict_page(conflict_page, expected=conflict_input_page, run=run)
                    await self._finish(
                        run=run,
                        page=page,
                        job=job,
                        step=step,
                        summary={"outcome": "retry", "conflict_retry_count": page.conflict_retry_count},
                    )
                except BaseException as failure:  # noqa: BLE001 - sanitize control flow before auditing.
                    await self._close_failed_phase(
                        run=run,
                        page=page,
                        lease=lease,
                        usage=usage,
                        step=step,
                        step_type=RagStepType.CONFLICT,
                        failure=failure,
                    )
                    raise
                repair = False
                continue
            except PersistedRagLintError as failure:
                job, step, usage = await self._start(
                    run,
                    page,
                    lease,
                    usage,
                    RagStepType.VALIDATE,
                    {"outcome": "persisted_lint_rollback"},
                )
                await self._finish(
                    run=run,
                    page=page,
                    job=job,
                    step=step,
                    status=RagStepStatus.FAILED,
                    summary={"outcome": "persisted_lint_rollback"},
                    error_code=failure.code,
                )
                if repair:
                    raise _invalid_draft_failure() from None
                repair = True
                continue
            if type(committed) is not AtomicPageCommit:
                raise TypeError("atomic page commit is invalid")
            committed = _validated_commit(
                committed,
                expected_run=run,
                expected_page=page,
                usage=usage,
                bundle=bundle,
            )
            return PageExecutionResult(committed.run, committed.page)


BoundedPageRunner = PageRunner

__all__ = ["BoundedPageRunner", "InjectedCrash", "PageRunner", "PostgresPageStore"]
