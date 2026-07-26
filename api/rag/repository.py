"""Transaction-aware SQL state transitions for durable RAG runs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from uuid import UUID

import asyncpg

from llmwiki_core.rag import (
    MAX_MODEL_TOKENS,
    MAX_PAGE_ATTEMPTS,
    RagBudget,
    RagCitation,
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

from .records import (
    RagPageRecord,
    RagRunRecord,
    RagStepRecord,
    _adapt_db_page_row,
    _adapt_db_run_row,
    _adapt_db_step_row,
    _decode_page,
    _decode_run,
    _decode_step,
    _require_json_depth,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
_MAX_STEP_PAGE_SIZE = 100
_BUDGET_FIELDS = (
    "max_pages",
    "max_steps",
    "max_model_tokens",
    "max_context_chars",
    "max_page_chars",
    "per_call_timeout_seconds",
    "max_page_attempts",
    "max_conflict_retries",
)
_RUN_IDENTITY_FIELDS = (
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
_PAGE_SNAPSHOT_FIELDS = (
    "id",
    "run_id",
    "user_id",
    "knowledge_base_id",
    "ordinal",
    "path",
    "intent",
    "query",
    "state",
    "document_id",
    "version_read",
    "version_committed",
    "attempt_count",
    "conflict_retry_count",
    "last_completed_step_sequence",
    "preview",
    "preview_digest",
    "preview_full_char_count",
    "preview_truncated",
    "lint_summary",
)

# Global mutation lock hierarchy: resolve identifiers without row locks, then
# acquire run -> page -> step/document locks. Resume lineage locks the true root
# run -> parent run -> complete root pages -> complete parent pages. Never
# acquire a run lock while holding a page, step, or document lock.


def _decode_db_run(row: Mapping[str, object]) -> RagRunRecord:
    return _decode_run(_adapt_db_run_row(row))


def _decode_db_page(row: Mapping[str, object]) -> RagPageRecord:
    return _decode_page(_adapt_db_page_row(row))


def _decode_db_step(row: Mapping[str, object]) -> RagStepRecord:
    return _decode_step(_adapt_db_step_row(row))


def _error(code: str, message: str, *, retryable: bool = False) -> RagDomainError:
    return RagDomainError(code, message, retryable)


def _require_transaction(conn: asyncpg.Connection) -> None:
    checker = getattr(conn, "is_in_transaction", None)
    if not callable(checker) or not checker():
        raise RuntimeError("RAG mutation requires an explicit transaction")


def _require_uuid(value: UUID, field: str) -> None:
    if not isinstance(value, UUID):
        raise ValueError(f"{field} must be a UUID")


def _require_digest(value: str, field: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _require_key(value: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 200 or value.strip() != value:
        raise ValueError("idempotency_key must contain 1 to 200 normalized characters")


def _require_profile_version(value: str) -> None:
    if type(value) is not str or not 1 <= len(value) <= 128 or value.strip() != value:
        raise ValueError("model_profile_version must contain 1 to 128 normalized characters")


def _require_usage(usage: RagUsage, budget: RagBudget | None = None) -> None:
    if not isinstance(usage, RagUsage):
        raise ValueError("usage must be a RagUsage")
    if budget is not None and (usage.steps > budget.max_steps or usage.model_tokens > budget.max_model_tokens):
        raise _error("rag_budget_exhausted", "The RAG budget was exhausted.")


def _budget_json(budget: RagBudget) -> str:
    if not isinstance(budget, RagBudget):
        raise ValueError("budget must be a RagBudget")
    return json.dumps(asdict(budget), separators=(",", ":"), sort_keys=True)


def _usage_json(usage: RagUsage) -> str:
    return json.dumps(asdict(usage), separators=(",", ":"), sort_keys=True)


def _json_object(value: Mapping[str, object], field: str, *, max_bytes: int) -> str:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ValueError(f"{field} must be a JSON object")
    try:
        _require_json_depth(value, field)
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except RecursionError as exc:
        raise ValueError(f"{field} exceeds the JSON nesting limit") from exc
    except (TypeError, ValueError) as exc:
        if "nesting limit" in str(exc):
            raise
        raise ValueError(f"{field} must contain only finite JSON values") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} exceeds its byte limit")
    return encoded


def _citation_json(citations: tuple[RagCitation, ...]) -> str:
    if type(citations) is not tuple or not all(isinstance(item, RagCitation) for item in citations):
        raise ValueError("citations must be a tuple of RagCitation values")
    if len(citations) > 128:
        raise ValueError("citations exceed the item limit")
    identities = [
        {
            "document_id": str(item.document_id),
            "document_version": item.document_version,
            "chunk_index": item.chunk_index,
            "page": item.page,
        }
        for item in citations
    ]
    if len({tuple(identity.values()) for identity in identities}) != len(identities):
        raise ValueError("citation identities must be unique")
    encoded = json.dumps(identities, sort_keys=True)
    if len(encoded.encode("utf-8")) > 16_384:
        raise ValueError("citations exceed the byte limit")
    return encoded


def _validate_config(config: RagRunConfig) -> None:
    if not isinstance(config, RagRunConfig):
        raise ValueError("config must be a RagRunConfig")
    normalized = RagRunConfig.build(
        knowledge_base_id=config.knowledge_base_id,
        goal=config.goal,
        target_path_prefix=config.target_path_prefix,
        model_profile=config.model_profile,
        retrieval_profile=config.retrieval_profile,
        dry_run=config.dry_run,
        budget=config.budget,
    )
    if normalized != config:
        raise ValueError("config must be normalized")


async def _lock_idempotency(conn: asyncpg.Connection, user_id: UUID, key: str) -> None:
    await conn.fetchval(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"rag-run:{user_id}:{key}",
    )


def _idempotency_conflict() -> RagDomainError:
    return _error(
        "rag_idempotency_conflict",
        "The idempotency key was already used for a different request.",
    )


def _root_replay_matches(
    existing: RagRunRecord,
    *,
    job_id: UUID,
    user_id: UUID,
    config: RagRunConfig,
    request_digest: str,
    model_profile_version: str,
) -> bool:
    goal_digest = hashlib.sha256(config.goal.encode("utf-8")).hexdigest()
    return (
        existing.job_id == job_id
        and existing.root_run_id == existing.id
        and existing.parent_run_id is None
        and existing.user_id == user_id
        and existing.knowledge_base_id == config.knowledge_base_id
        and existing.goal == config.goal
        and existing.goal_digest == goal_digest
        and existing.target_path_prefix == config.target_path_prefix
        and existing.model_profile == config.model_profile
        and existing.model_profile_version == model_profile_version
        and existing.retrieval_profile == config.retrieval_profile
        and existing.dry_run == config.dry_run
        and existing.budget == config.budget
        and existing.request_digest == request_digest
    )


async def find_by_idempotency(conn: asyncpg.Connection, user_id: UUID, key: str) -> RagRunRecord | None:
    _require_uuid(user_id, "user_id")
    _require_key(key)
    row = await conn.fetchrow(
        "SELECT * FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2",
        user_id,
        key,
    )
    return None if row is None else _decode_db_run(row)


async def create_root(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    job_id: UUID,
    user_id: UUID,
    config: RagRunConfig,
    idempotency_key: str,
    request_digest: str,
    model_profile_version: str,
) -> RagRunRecord:
    _require_transaction(conn)
    _require_uuid(run_id, "run_id")
    _require_uuid(job_id, "job_id")
    _require_uuid(user_id, "user_id")
    _validate_config(config)
    _require_key(idempotency_key)
    _require_digest(request_digest, "request_digest")
    _require_profile_version(model_profile_version)
    await _lock_idempotency(conn, user_id, idempotency_key)
    existing = await find_by_idempotency(conn, user_id, idempotency_key)
    if existing is not None:
        if not _root_replay_matches(
            existing,
            job_id=job_id,
            user_id=user_id,
            config=config,
            request_digest=request_digest,
            model_profile_version=model_profile_version,
        ):
            raise _idempotency_conflict()
        return existing
    goal_digest = hashlib.sha256(config.goal.encode("utf-8")).hexdigest()
    async with conn.transaction():
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_runs (
                    id,job_id,root_run_id,parent_run_id,user_id,knowledge_base_id,
                    goal,goal_digest,target_path_prefix,model_profile,model_profile_version,
                    retrieval_profile,dry_run,budget,usage,idempotency_key,request_digest,
                    completion_reason,last_committed_ordinal
                ) VALUES (
                    $1,$2,$1,NULL,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::jsonb,
                    $14,$15,NULL,-1
                ) RETURNING *
                """,
                run_id,
                job_id,
                user_id,
                config.knowledge_base_id,
                config.goal,
                goal_digest,
                config.target_path_prefix,
                config.model_profile,
                model_profile_version,
                config.retrieval_profile,
                config.dry_run,
                _budget_json(config.budget),
                _usage_json(RagUsage()),
                idempotency_key,
                request_digest,
            )
        except asyncpg.ForeignKeyViolationError as exc:
            raise _error("rag_job_binding_invalid", "The RAG job binding is invalid.") from exc
        except (asyncpg.UniqueViolationError, asyncpg.CheckViolationError) as exc:
            raise _error("rag_run_conflict", "The RAG run could not be created.") from exc
        if row is None:  # pragma: no cover
            raise RuntimeError("RAG root insert returned no row")
        created = _decode_db_run(row)
    return created


async def _validate_resume_source(  # noqa: C901 - validates a coupled durable snapshot.
    conn: asyncpg.Connection,
    *,
    parent: RagRunRecord,
    budget: RagBudget,
) -> tuple[RagRunRecord, tuple[RagPageRecord, ...], tuple[RagPageRecord, ...]]:
    if parent.completion_reason is None:
        raise _error("rag_parent_not_finished", "The parent RAG run is not finished.")
    if any(getattr(budget, field) < getattr(parent.budget, field) for field in _BUDGET_FIELDS):
        raise _error("rag_budget_too_small", "The resume budget cannot decrease.")
    root_row = await conn.fetchrow(
        "SELECT * FROM rag_runs WHERE id=$1 FOR UPDATE",
        parent.root_run_id,
    )
    if root_row is None:
        raise _error("rag_resume_boundary_invalid", "The root RAG run is invalid.")
    try:
        root = _decode_db_run(root_row)
    except (TypeError, ValueError) as exc:
        raise _error("rag_resume_boundary_invalid", "The root RAG run is invalid.") from exc
    immutable_fields = (
        "root_run_id",
        "user_id",
        "knowledge_base_id",
        "goal",
        "goal_digest",
        "target_path_prefix",
        "model_profile",
        "model_profile_version",
        "retrieval_profile",
        "dry_run",
    )
    if (
        root.id != root.root_run_id
        or root.parent_run_id is not None
        or root.completion_reason is None
        or any(getattr(root, field) != getattr(parent, field) for field in immutable_fields)
    ):
        raise _error("rag_resume_boundary_invalid", "The RAG lineage is invalid.")
    root_rows = await conn.fetch(
        "SELECT * FROM rag_run_pages WHERE run_id=$1 ORDER BY ordinal FOR UPDATE",
        root.id,
    )
    parent_rows = await conn.fetch(
        "SELECT * FROM rag_run_pages WHERE run_id=$1 ORDER BY ordinal FOR UPDATE",
        parent.id,
    )
    try:
        root_pages = tuple(_decode_db_page(row) for row in root_rows)
        parent_pages = tuple(_decode_db_page(row) for row in parent_rows)
    except (TypeError, ValueError) as exc:
        raise _error("rag_resume_boundary_invalid", "The RAG resume source is invalid.") from exc
    expected_ordinals = tuple(range(len(root_pages)))
    work_fields = ("ordinal", "path", "intent", "query", "user_id", "knowledge_base_id")
    if (
        tuple(page.ordinal for page in root_pages) != expected_ordinals
        or len(parent_pages) != len(root_pages)
        or tuple(page.ordinal for page in parent_pages) != expected_ordinals
        or any(
            any(getattr(root_page, field) != getattr(parent_page, field) for field in work_fields)
            for root_page, parent_page in zip(root_pages, parent_pages, strict=True)
        )
    ):
        raise _error("rag_resume_boundary_invalid", "The RAG worklist identity is invalid.")
    if len(root_pages) > budget.max_pages:
        raise _error("rag_budget_too_small", "The resume budget is smaller than the worklist.")
    for page in parent_pages:
        if page.ordinal <= parent.last_committed_ordinal:
            valid = (
                page.state is RagPageState.COMMITTED
                and page.document_id is not None
                and page.version_committed is not None
                and page.lint_summary is not None
            )
        else:
            valid = page.state in {
                RagPageState.PLANNED,
                RagPageState.RUNNING,
                RagPageState.FAILED,
                RagPageState.DRY_RUN_COMPLETE,
            }
        if not valid:
            raise _error("rag_resume_boundary_invalid", "The parent RAG boundary is invalid.")
    if parent.last_committed_ordinal >= len(parent_pages):
        raise _error("rag_resume_boundary_invalid", "The parent RAG boundary is invalid.")
    return root, root_pages, parent_pages


def _resume_replay_matches(
    existing: RagRunRecord,
    *,
    job_id: UUID,
    parent: RagRunRecord,
    budget: RagBudget,
    request_digest: str,
) -> bool:
    immutable_fields = (
        "root_run_id",
        "user_id",
        "knowledge_base_id",
        "goal",
        "goal_digest",
        "target_path_prefix",
        "model_profile",
        "model_profile_version",
        "retrieval_profile",
        "dry_run",
    )
    return (
        existing.job_id == job_id
        and existing.parent_run_id == parent.id
        and existing.budget == budget
        and existing.request_digest == request_digest
        and all(getattr(existing, field) == getattr(parent, field) for field in immutable_fields)
    )


async def _insert_resume_snapshot(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    job_id: UUID,
    parent: RagRunRecord,
    budget: RagBudget,
    idempotency_key: str,
    request_digest: str,
    root_pages: tuple[RagPageRecord, ...],
    parent_pages: tuple[RagPageRecord, ...],
) -> RagRunRecord:
    async with conn.transaction():
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_runs (
                    id,job_id,root_run_id,parent_run_id,user_id,knowledge_base_id,
                    goal,goal_digest,target_path_prefix,model_profile,model_profile_version,
                    retrieval_profile,dry_run,budget,usage,idempotency_key,request_digest,
                    completion_reason,last_committed_ordinal
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb,$15::jsonb,
                    $16,$17,NULL,$18
                ) RETURNING *
                """,
                run_id,
                job_id,
                parent.root_run_id,
                parent.id,
                parent.user_id,
                parent.knowledge_base_id,
                parent.goal,
                parent.goal_digest,
                parent.target_path_prefix,
                parent.model_profile,
                parent.model_profile_version,
                parent.retrieval_profile,
                parent.dry_run,
                _budget_json(budget),
                _usage_json(RagUsage()),
                idempotency_key,
                request_digest,
                parent.last_committed_ordinal,
            )
            inserted = 0
            for root_page, source_page in zip(root_pages, parent_pages, strict=True):
                committed = root_page.ordinal <= parent.last_committed_ordinal
                result = await conn.execute(
                    """
                    INSERT INTO rag_run_pages (
                        run_id,user_id,knowledge_base_id,ordinal,path,intent,query,state,
                        document_id,version_read,version_committed,attempt_count,
                        conflict_retry_count,last_completed_step_sequence,preview_truncated,lint_summary
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,0,0,0,false,$12::jsonb)
                    """,
                    run_id,
                    root_page.user_id,
                    root_page.knowledge_base_id,
                    root_page.ordinal,
                    root_page.path,
                    root_page.intent,
                    root_page.query,
                    "committed" if committed else "planned",
                    source_page.document_id if committed else None,
                    source_page.version_read if committed else None,
                    source_page.version_committed if committed else None,
                    _json_object(source_page.lint_summary, "lint_summary", max_bytes=16_384) if committed else None,
                )
                inserted += int(result.rsplit(" ", 1)[-1])
            if inserted != len(root_pages):
                raise _error("rag_resume_conflict", "The RAG resume worklist copy was incomplete.")
        except asyncpg.ForeignKeyViolationError as exc:
            raise _error("rag_job_binding_invalid", "The RAG resume job binding is invalid.") from exc
        except (asyncpg.UniqueViolationError, asyncpg.CheckViolationError) as exc:
            raise _error("rag_resume_conflict", "The RAG resume could not be created.") from exc
        if row is None:  # pragma: no cover
            raise RuntimeError("RAG resume insert returned no row")
        resumed = _decode_db_run(row)
    return resumed


async def create_resume(  # noqa: C901 - keeps the resume transaction atomic.
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    job_id: UUID,
    parent: RagRunRecord,
    budget: RagBudget,
    idempotency_key: str,
    request_digest: str,
) -> RagRunRecord:
    _require_transaction(conn)
    _require_uuid(run_id, "run_id")
    _require_uuid(job_id, "job_id")
    if not isinstance(parent, RagRunRecord):
        raise ValueError("parent must be a RagRunRecord")
    if parent.completion_reason is None:
        raise _error("rag_parent_not_finished", "The parent RAG run is not finished.")
    _budget_json(budget)
    _require_key(idempotency_key)
    _require_digest(request_digest, "request_digest")
    await _lock_idempotency(conn, parent.user_id, idempotency_key)
    existing = await find_by_idempotency(conn, parent.user_id, idempotency_key)
    if existing is not None:
        if not _resume_replay_matches(
            existing,
            job_id=job_id,
            parent=parent,
            budget=budget,
            request_digest=request_digest,
        ):
            raise _idempotency_conflict()
        return existing
    parent_identity = await conn.fetchrow(
        "SELECT root_run_id FROM rag_runs WHERE id=$1 AND user_id=$2",
        parent.id,
        parent.user_id,
    )
    if parent_identity is None:
        raise _error("rag_run_not_found", "The parent RAG run was not found.")
    await conn.fetchrow("SELECT id FROM rag_runs WHERE id=$1 FOR UPDATE", parent_identity["root_run_id"])
    locked_parent_row = await conn.fetchrow(
        "SELECT * FROM rag_runs WHERE id=$1 AND user_id=$2 FOR UPDATE",
        parent.id,
        parent.user_id,
    )
    if locked_parent_row is None:  # pragma: no cover - locked lineage cannot disappear.
        raise _error("rag_run_not_found", "The parent RAG run was not found.")
    locked_parent = _decode_db_run(locked_parent_row)
    if any(getattr(parent, field) != getattr(locked_parent, field) for field in _RUN_IDENTITY_FIELDS) or (
        parent.completion_reason != locked_parent.completion_reason
        or parent.last_committed_ordinal != locked_parent.last_committed_ordinal
    ):
        raise _error("rag_run_mismatch", "The supplied parent run is stale or mismatched.")
    _, root_pages, parent_pages = await _validate_resume_source(conn, parent=locked_parent, budget=budget)
    return await _insert_resume_snapshot(
        conn,
        run_id=run_id,
        job_id=job_id,
        parent=locked_parent,
        budget=budget,
        idempotency_key=idempotency_key,
        request_digest=request_digest,
        root_pages=root_pages,
        parent_pages=parent_pages,
    )


async def get_for_user(conn: asyncpg.Connection, run_id: UUID, user_id: UUID) -> RagRunRecord | None:
    _require_uuid(run_id, "run_id")
    _require_uuid(user_id, "user_id")
    row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 AND user_id=$2", run_id, user_id)
    return None if row is None else _decode_db_run(row)


async def get_for_worker(conn: asyncpg.Connection, run_id: UUID, job_id: UUID) -> RagRunRecord | None:
    _require_uuid(run_id, "run_id")
    _require_uuid(job_id, "job_id")
    row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 AND job_id=$2", run_id, job_id)
    return None if row is None else _decode_db_run(row)


async def insert_worklist(
    conn: asyncpg.Connection,
    run: RagRunRecord,
    items: tuple[RagWorkItem, ...],
) -> tuple[RagPageRecord, ...]:
    _require_transaction(conn)
    if not isinstance(run, RagRunRecord):
        raise ValueError("run must be a RagRunRecord")
    if type(items) is not tuple:
        raise ValueError("items must be a tuple")
    validate_worklist(items, target_path_prefix=run.target_path_prefix, max_pages=run.budget.max_pages)
    locked_row = await conn.fetchrow(
        "SELECT * FROM rag_runs WHERE id=$1 AND user_id=$2 FOR UPDATE",
        run.id,
        run.user_id,
    )
    if locked_row is None:
        raise _error("rag_run_not_found", "The RAG run was not found.")
    locked = _decode_db_run(locked_row)
    if any(getattr(run, field) != getattr(locked, field) for field in _RUN_IDENTITY_FIELDS):
        raise _error("rag_run_mismatch", "The supplied RAG run does not match persisted identity.")
    if locked.completion_reason is not None:
        raise _error("rag_run_finished", "The RAG run is already finished.")
    if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_run_pages WHERE run_id=$1)", run.id):
        raise _error("rag_worklist_exists", "The RAG worklist already exists.")
    async with conn.transaction():
        rows = []
        try:
            for item in items:
                row = await conn.fetchrow(
                    """
                    INSERT INTO rag_run_pages (
                        run_id,user_id,knowledge_base_id,ordinal,path,intent,query
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *
                    """,
                    run.id,
                    run.user_id,
                    run.knowledge_base_id,
                    item.ordinal,
                    item.path,
                    item.intent,
                    item.query,
                )
                if row is None:  # pragma: no cover
                    raise RuntimeError("RAG worklist insert returned no row")
                rows.append(row)
        except (
            asyncpg.UniqueViolationError,
            asyncpg.CheckViolationError,
            asyncpg.ForeignKeyViolationError,
        ) as exc:
            raise _error("rag_worklist_conflict", "The RAG worklist could not be stored.") from exc
        pages = tuple(_decode_db_page(row) for row in rows)
    return pages


async def list_pages(conn: asyncpg.Connection, run_id: UUID) -> tuple[RagPageRecord, ...]:
    _require_uuid(run_id, "run_id")
    rows = await conn.fetch("SELECT * FROM rag_run_pages WHERE run_id=$1 ORDER BY ordinal", run_id)
    return tuple(_decode_db_page(row) for row in rows)


async def start_step(  # noqa: C901 - validates the complete locked step context.
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    page_id: UUID | None,
    step_type: RagStepType,
    input_digest: str,
) -> RagStepRecord:
    _require_transaction(conn)
    _require_uuid(run_id, "run_id")
    if page_id is not None:
        _require_uuid(page_id, "page_id")
    if not isinstance(step_type, RagStepType):
        raise ValueError("step_type must be a RagStepType")
    _require_digest(input_digest, "input_digest")
    run_row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 FOR UPDATE", run_id)
    if run_row is None:
        raise _error("rag_run_not_found", "The RAG run was not found.")
    run = _decode_db_run(run_row)
    if run.completion_reason is not None:
        raise _error("rag_run_finished", "The RAG run is already finished.")
    if page_id is not None:
        page_row = await conn.fetchrow(
            "SELECT * FROM rag_run_pages WHERE id=$1 AND run_id=$2 FOR UPDATE",
            page_id,
            run_id,
        )
        if page_row is None:
            raise _error("rag_page_not_found", "The RAG page was not found.")
        page = _decode_db_page(page_row)
        if page.state is not RagPageState.RUNNING:
            raise _error("rag_page_not_running", "Page-scoped steps require a running page.")
    if await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_steps WHERE run_id=$1 AND status='running')", run_id):
        raise _error("rag_step_already_running", "A RAG step is already running.", retryable=True)
    sequence = await conn.fetchval("SELECT COALESCE(max(sequence),0)+1 FROM rag_steps WHERE run_id=$1", run_id)
    if type(sequence) is not int or not 1 <= sequence <= run.budget.max_steps:
        raise _error("rag_budget_exhausted", "The RAG budget was exhausted.")
    async with conn.transaction():
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_steps (
                    run_id,run_page_id,user_id,knowledge_base_id,sequence,step_type,status,
                    input_digest,model_profile_version
                ) VALUES ($1,$2,$3,$4,$5,$6,'running',$7,$8) RETURNING *
                """,
                run.id,
                page_id,
                run.user_id,
                run.knowledge_base_id,
                sequence,
                step_type.value,
                input_digest,
                run.model_profile_version,
            )
        except asyncpg.UniqueViolationError as exc:
            code = (
                "rag_step_already_running"
                if exc.constraint_name == "rag_steps_one_running_per_run"
                else "rag_step_sequence_conflict"
            )
            raise _error(code, "The RAG step could not be started.", retryable=True) from exc
        except asyncpg.ForeignKeyViolationError as exc:
            raise _error("rag_page_not_found", "The RAG page was not found.") from exc
        if row is None:  # pragma: no cover
            raise RuntimeError("RAG step insert returned no row")
        step = _decode_db_step(row)
    return step


async def finish_step(  # noqa: C901 - validates before the single terminal write.
    conn: asyncpg.Connection,
    *,
    step_id: UUID,
    status: RagStepStatus,
    summary: Mapping[str, object],
    citations: tuple[RagCitation, ...],
    usage: RagUsage,
    latency_ms: float,
    error_code: str | None = None,
) -> RagStepRecord:
    _require_transaction(conn)
    _require_uuid(step_id, "step_id")
    if status not in {RagStepStatus.SUCCEEDED, RagStepStatus.FAILED}:
        raise ValueError("finish_step requires a terminal status")
    summary_json = _json_object(summary, "summary", max_bytes=16_384)
    citations_json = _citation_json(citations)
    _require_usage(usage)
    if usage.steps != 1:
        raise ValueError("step usage must account for exactly one step")
    if usage.model_tokens > MAX_MODEL_TOKENS:
        raise ValueError(f"step model tokens exceed the core hard cap of {MAX_MODEL_TOKENS}")
    if type(latency_ms) not in {int, float} or not isfinite(float(latency_ms)) or latency_ms < 0:
        raise ValueError("latency_ms must be a finite nonnegative number")
    if status is RagStepStatus.SUCCEEDED:
        if error_code is not None:
            raise ValueError("succeeded steps cannot have an error_code")
    elif type(error_code) is not str or _ERROR_CODE.fullmatch(error_code) is None:
        raise ValueError("failed steps require a normalized error_code")
    identity = await conn.fetchrow("SELECT run_id,run_page_id FROM rag_steps WHERE id=$1", step_id)
    if identity is None:
        raise _error("rag_step_not_running", "The RAG step is not running.")
    run_row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 FOR UPDATE", identity["run_id"])
    if run_row is None:  # pragma: no cover - protected by the step foreign key.
        raise _error("rag_run_not_found", "The RAG run was not found.")
    run = _decode_db_run(run_row)
    if run.completion_reason is not None:
        raise _error("rag_run_finished", "The RAG run is already finished.")
    if identity["run_page_id"] is not None:
        page_row = await conn.fetchrow(
            "SELECT * FROM rag_run_pages WHERE id=$1 AND run_id=$2 FOR UPDATE",
            identity["run_page_id"],
            run.id,
        )
        if page_row is None:
            raise _error("rag_page_not_found", "The RAG page was not found.")
        if _decode_db_page(page_row).state is not RagPageState.RUNNING:
            raise _error("rag_page_not_running", "Page-scoped steps require a running page.")
    step_row = await conn.fetchrow("SELECT * FROM rag_steps WHERE id=$1 AND run_id=$2 FOR UPDATE", step_id, run.id)
    if step_row is None or _decode_db_step(step_row).status is not RagStepStatus.RUNNING:
        raise _error("rag_step_not_running", "The RAG step is not running.")
    aggregate = await conn.fetchrow(
        "SELECT count(*) FILTER (WHERE status IN ('succeeded','failed')) AS terminal_steps,"
        "COALESCE(sum(total_tokens) FILTER (WHERE status IN ('succeeded','failed')),0) AS model_tokens "
        "FROM rag_steps WHERE run_id=$1",
        run.id,
    )
    post_usage = RagUsage(
        steps=aggregate["terminal_steps"] + 1,
        model_tokens=aggregate["model_tokens"] + usage.model_tokens,
    )
    _require_usage(post_usage, run.budget)
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            UPDATE rag_steps SET
                status=$2,output_summary=$3::jsonb,citation_identities=$4::jsonb,
                input_tokens=$5,output_tokens=0,total_tokens=$5,latency_ms=$6,error_code=$7
            WHERE id=$1 AND status='running'
            RETURNING *
            """,
            step_id,
            status.value,
            summary_json,
            citations_json,
            usage.model_tokens,
            float(latency_ms),
            error_code,
        )
        if row is None:
            raise _error("rag_step_not_running", "The RAG step is not running.")
        finished = _decode_db_step(row)
    return finished


async def begin_page_attempt(conn: asyncpg.Connection, page_id: UUID, *, max_attempts: int) -> RagPageRecord:
    _require_transaction(conn)
    _require_uuid(page_id, "page_id")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_PAGE_ATTEMPTS:
        raise ValueError(f"max_attempts must be between 1 and {MAX_PAGE_ATTEMPTS}")
    run_id = await conn.fetchval("SELECT run_id FROM rag_run_pages WHERE id=$1", page_id)
    if run_id is None:
        raise _error("rag_page_not_found", "The RAG page was not found.")
    run_row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 FOR UPDATE", run_id)
    if run_row is None:  # pragma: no cover - protected by the page foreign key.
        raise _error("rag_run_not_found", "The RAG run was not found.")
    run = _decode_db_run(run_row)
    if run.completion_reason is not None:
        raise _error("rag_run_finished", "The RAG run is already finished.")
    if max_attempts > run.budget.max_page_attempts:
        raise _error("rag_attempt_limit_mismatch", "The page attempt limit exceeds the persisted run budget.")
    page_row = await conn.fetchrow("SELECT * FROM rag_run_pages WHERE id=$1 AND run_id=$2 FOR UPDATE", page_id, run.id)
    if page_row is None:
        raise _error("rag_page_not_found", "The RAG page was not found.")
    page = _decode_db_page(page_row)
    if page.attempt_count >= max_attempts:
        raise _error("rag_page_attempts_exhausted", "The page attempt limit was exhausted.")
    if page.state not in {RagPageState.PLANNED, RagPageState.RUNNING, RagPageState.FAILED}:
        raise _error("rag_page_not_attemptable", "The RAG page cannot be attempted.")
    async with conn.transaction():
        row = await conn.fetchrow(
            "UPDATE rag_run_pages SET attempt_count=attempt_count+1,state='running' WHERE id=$1 RETURNING *",
            page.id,
        )
        if row is None:  # pragma: no cover - page is locked above.
            raise _error("rag_page_not_found", "The RAG page was not found.")
        attempted = _decode_db_page(row)
    return attempted


async def _require_document_version(
    conn: asyncpg.Connection,
    *,
    document_id: UUID,
    run: RagRunRecord,
    committed_version: int,
) -> None:
    persisted_version = await conn.fetchval(
        "SELECT version FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3 AND NOT archived FOR UPDATE",
        document_id,
        run.user_id,
        run.knowledge_base_id,
    )
    if persisted_version is None:
        raise _error("rag_document_not_found", "The RAG document was not found.")
    if type(persisted_version) is not int or persisted_version != committed_version:
        raise _error(
            "rag_document_version_mismatch",
            "The committed document version does not match the RAG boundary.",
            retryable=True,
        )


async def mark_boundary(  # noqa: C901 - validates one authoritative boundary snapshot.
    conn: asyncpg.Connection,
    *,
    run: RagRunRecord,
    page: RagPageRecord,
    document_id: UUID,
    committed_version: int,
    usage: RagUsage,
    lint_summary: Mapping[str, object],
) -> tuple[RagRunRecord, RagPageRecord]:
    _require_transaction(conn)
    if not isinstance(run, RagRunRecord) or not isinstance(page, RagPageRecord):
        raise ValueError("run and page must be durable RAG records")
    _require_uuid(document_id, "document_id")
    if type(committed_version) is not int or committed_version < 1:
        raise ValueError("committed_version must be a positive integer")
    _require_usage(usage, run.budget)
    lint_json = _json_object(lint_summary, "lint_summary", max_bytes=16_384)
    if page.run_id != run.id or page.user_id != run.user_id or page.knowledge_base_id != run.knowledge_base_id:
        raise _error("rag_page_mismatch", "The RAG page does not belong to the run.")
    run_row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 AND user_id=$2 FOR UPDATE", run.id, run.user_id)
    if run_row is None:
        raise _error("rag_run_not_found", "The RAG run was not found.")
    current_run = _decode_db_run(run_row)
    page_row = await conn.fetchrow("SELECT * FROM rag_run_pages WHERE id=$1 AND run_id=$2 FOR UPDATE", page.id, run.id)
    if page_row is None:
        raise _error("rag_page_not_found", "The RAG page was not found.")
    current_page = _decode_db_page(page_row)
    if any(getattr(run, field) != getattr(current_run, field) for field in _RUN_IDENTITY_FIELDS) or (
        run.usage != current_run.usage
        or run.last_committed_ordinal != current_run.last_committed_ordinal
        or run.completion_reason != current_run.completion_reason
    ):
        raise _error("rag_run_mismatch", "The supplied RAG run is stale or mismatched.")
    if any(getattr(page, field) != getattr(current_page, field) for field in _PAGE_SNAPSHOT_FIELDS):
        raise _error("rag_page_mismatch", "The supplied RAG page is stale or mismatched.")
    if current_run.completion_reason is not None:
        raise _error("rag_run_finished", "The RAG run is already finished.")
    if current_page.ordinal != current_run.last_committed_ordinal + 1:
        raise _error("rag_boundary_out_of_order", "The RAG boundary update is out of order.")
    if current_page.state is not RagPageState.RUNNING:
        raise _error("rag_page_not_running", "The RAG page is not running.")
    step_rows = await conn.fetch(
        "SELECT * FROM rag_steps WHERE run_id=$1 ORDER BY sequence FOR UPDATE",
        current_run.id,
    )
    try:
        steps = tuple(_decode_db_step(row) for row in step_rows)
    except (TypeError, ValueError) as exc:
        raise _error("rag_step_invalid", "Persisted RAG step state is invalid.") from exc
    if any(step.status is RagStepStatus.RUNNING for step in steps):
        raise _error("rag_step_still_running", "A RAG step is still running.")
    authoritative = RagUsage(
        steps=len(steps),
        model_tokens=sum(step.total_tokens for step in steps),
    )
    _require_usage(authoritative, current_run.budget)
    if usage != authoritative:
        raise _error("rag_usage_mismatch", "The supplied RAG usage does not match persisted terminal steps.")
    if current_page.document_id is None:
        if current_page.version_read is not None or committed_version != 1:
            raise _error("rag_document_version_mismatch", "A new page must commit document version 1.", retryable=True)
    elif current_page.document_id != document_id or current_page.version_read != committed_version - 1:
        raise _error(
            "rag_document_version_mismatch",
            "The page read identity does not match the committed document version.",
            retryable=True,
        )
    await _require_document_version(
        conn,
        document_id=document_id,
        run=current_run,
        committed_version=committed_version,
    )
    last_sequence = max(
        (step.sequence for step in steps if step.run_page_id == current_page.id),
        default=0,
    )
    async with conn.transaction():
        page_updated = await conn.fetchrow(
            """
            UPDATE rag_run_pages SET state='committed',document_id=$2,
                version_read=CASE WHEN $3::integer > 1 THEN $3::integer-1 ELSE NULL END,
                version_committed=$3,last_completed_step_sequence=$4,lint_summary=$6::jsonb
            WHERE id=$1 AND run_id=$5 AND state='running'
            RETURNING *
            """,
            current_page.id,
            document_id,
            committed_version,
            last_sequence,
            current_run.id,
            lint_json,
        )
        run_updated = await conn.fetchrow(
            """
            UPDATE rag_runs SET last_committed_ordinal=$2,usage=$3::jsonb
            WHERE id=$1 AND last_committed_ordinal=$2-1 AND completion_reason IS NULL
            RETURNING *
            """,
            current_run.id,
            current_page.ordinal,
            _usage_json(usage),
        )
        if page_updated is None or run_updated is None:  # pragma: no cover - rows are locked above.
            raise _error("rag_boundary_conflict", "The RAG boundary could not be advanced.", retryable=True)
        decoded_run = _decode_db_run(run_updated)
        decoded_page = _decode_db_page(page_updated)
    return decoded_run, decoded_page


async def _authoritative_usage(conn: asyncpg.Connection, run: RagRunRecord) -> RagUsage:
    rows = await conn.fetch(
        "SELECT * FROM rag_steps WHERE run_id=$1 ORDER BY sequence FOR UPDATE",
        run.id,
    )
    try:
        steps = tuple(_decode_db_step(row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise _error("rag_step_invalid", "Persisted RAG step state is invalid.") from exc
    if any(step.status is RagStepStatus.RUNNING for step in steps):
        raise _error("rag_step_still_running", "A RAG step is still running.")
    usage = RagUsage(
        steps=len(steps),
        model_tokens=sum(step.total_tokens for step in steps),
    )
    _require_usage(usage, run.budget)
    return usage


async def _finish_locked_run(
    conn: asyncpg.Connection,
    *,
    run: RagRunRecord,
    completion_reason: RagCompletionReason,
    caller_usage: RagUsage | None,
) -> RagRunRecord:
    authoritative = await _authoritative_usage(conn, run)
    if caller_usage is not None and caller_usage != authoritative:
        raise _error(
            "rag_usage_mismatch",
            "The supplied RAG usage does not match persisted terminal steps.",
        )
    page_counts = await conn.fetchrow(
        "SELECT count(*) AS total,count(*) FILTER (WHERE state='committed') AS committed,"
        "count(*) FILTER (WHERE state='dry_run_complete') AS dry_run_complete "
        "FROM rag_run_pages WHERE run_id=$1",
        run.id,
    )
    total = page_counts["total"]
    if completion_reason is RagCompletionReason.NO_WORK and total != 0:
        raise _error("rag_invalid_completion", "A non-empty run cannot finish with no work.")
    if completion_reason is RagCompletionReason.COMPLETED and (total == 0 or page_counts["committed"] != total):
        raise _error("rag_invalid_completion", "The RAG worklist is not fully committed.")
    if completion_reason is RagCompletionReason.DRY_RUN and (
        not run.dry_run or total == 0 or page_counts["dry_run_complete"] != total
    ):
        raise _error("rag_invalid_completion", "The dry RAG run is not complete.")
    async with conn.transaction():
        updated = await conn.fetchrow(
            "UPDATE rag_runs SET completion_reason=$2,usage=$3::jsonb "
            "WHERE id=$1 AND completion_reason IS NULL RETURNING *",
            run.id,
            completion_reason.value,
            _usage_json(authoritative),
        )
        if updated is None:  # pragma: no cover - row is locked.
            raise _error("rag_run_already_finished", "The RAG run is already finished.")
        finished = _decode_db_run(updated)
    return finished


async def finish_run(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    completion_reason: RagCompletionReason,
    usage: RagUsage,
) -> RagRunRecord:
    _require_transaction(conn)
    _require_uuid(run_id, "run_id")
    if not isinstance(completion_reason, RagCompletionReason):
        raise ValueError("completion_reason must be a RagCompletionReason")
    if not isinstance(usage, RagUsage):
        raise ValueError("usage must be a RagUsage")
    row = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1 FOR UPDATE", run_id)
    if row is None:
        raise _error("rag_run_not_found", "The RAG run was not found.")
    run = _decode_db_run(row)
    if run.completion_reason is not None:
        raise _error("rag_run_already_finished", "The RAG run is already finished.")
    return await _finish_locked_run(
        conn,
        run=run,
        completion_reason=completion_reason,
        caller_usage=usage,
    )


async def record_terminal_job_state(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    completion_reason: RagCompletionReason,
) -> RagRunRecord:
    _require_transaction(conn)
    _require_uuid(run_id, "run_id")
    if not isinstance(completion_reason, RagCompletionReason):
        raise ValueError("completion_reason must be a RagCompletionReason")
    row = await conn.fetchrow(
        "SELECT run.*,job.state::text AS job_state FROM rag_runs AS run "
        "JOIN background_jobs AS job ON job.id=run.job_id WHERE run.id=$1 FOR UPDATE OF run,job",
        run_id,
    )
    if row is None:
        raise _error("rag_run_not_found", "The RAG run was not found.")
    if row["job_state"] not in {"succeeded", "failed", "cancelled"}:
        raise _error("rag_job_not_terminal", "The RAG job is not terminal.")
    run = _decode_db_run(row)
    if row["job_state"] == "cancelled":
        return run
    successful_reasons = {
        RagCompletionReason.COMPLETED,
        RagCompletionReason.NO_WORK,
        RagCompletionReason.DRY_RUN,
    }
    failure_reasons = {
        RagCompletionReason.BUDGET_EXHAUSTED,
        RagCompletionReason.PARTIAL_FAILURE,
    }
    if (row["job_state"] == "succeeded" and completion_reason not in successful_reasons) or (
        row["job_state"] == "failed" and completion_reason not in failure_reasons
    ):
        raise _error(
            "rag_invalid_completion",
            "The completion reason does not match the terminal job state.",
        )
    if run.completion_reason is not None:
        if run.completion_reason is completion_reason:
            return run
        raise _error("rag_run_already_finished", "The RAG run is already finished.")
    return await _finish_locked_run(
        conn,
        run=run,
        completion_reason=completion_reason,
        caller_usage=None,
    )


async def list_steps_for_user(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    user_id: UUID,
    after_sequence: int,
    limit: int,
) -> tuple[RagStepRecord, ...]:
    _require_uuid(run_id, "run_id")
    _require_uuid(user_id, "user_id")
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("after_sequence must be a nonnegative integer")
    if type(limit) is not int or not 1 <= limit <= _MAX_STEP_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {_MAX_STEP_PAGE_SIZE}")
    owned = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1 AND user_id=$2)", run_id, user_id)
    if not owned:
        return ()
    rows = await conn.fetch(
        "SELECT * FROM rag_steps WHERE run_id=$1 AND user_id=$2 AND sequence>$3 ORDER BY sequence LIMIT $4",
        run_id,
        user_id,
        after_sequence,
        limit,
    )
    return tuple(_decode_db_step(row) for row in rows)


__all__ = [
    "find_by_idempotency",
    "create_root",
    "create_resume",
    "get_for_user",
    "get_for_worker",
    "insert_worklist",
    "list_pages",
    "start_step",
    "finish_step",
    "begin_page_attempt",
    "mark_boundary",
    "finish_run",
    "record_terminal_job_state",
    "list_steps_for_user",
]
