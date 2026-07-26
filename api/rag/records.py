"""Strict immutable projections of durable RAG database rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Any
from uuid import UUID

from llmwiki_core.rag import (
    MAX_GOAL_CHARS,
    MAX_PAGE_CHARS,
    MAX_PAGES,
    MAX_PROFILE_CHARS,
    MAX_STEPS,
    MAX_WORK_ITEM_TEXT_CHARS,
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagPageState,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,128}$")
_BUDGET_KEYS = frozenset(
    {
        "max_pages",
        "max_steps",
        "max_model_tokens",
        "max_context_chars",
        "max_page_chars",
        "per_call_timeout_seconds",
        "max_page_attempts",
        "max_conflict_retries",
    }
)
_USAGE_KEYS = frozenset({"steps", "model_tokens"})
_CITATION_KEYS = frozenset({"document_id", "document_version", "chunk_index", "page"})
_MAX_JSON_DEPTH = 32
_JSON_COLUMN_BYTE_LIMITS = {
    "budget": 4_096,
    "usage": 4_096,
    "lint_summary": 16_384,
    "output_summary": 16_384,
    "citation_identities": 16_384,
}


@dataclass(frozen=True, slots=True)
class RagRunRecord:
    id: UUID
    job_id: UUID
    root_run_id: UUID
    parent_run_id: UUID | None
    user_id: UUID
    knowledge_base_id: UUID
    goal: str
    goal_digest: str
    target_path_prefix: str
    model_profile: str
    model_profile_version: str
    retrieval_profile: str
    dry_run: bool
    budget: RagBudget
    usage: RagUsage
    idempotency_key: str
    request_digest: str
    completion_reason: RagCompletionReason | None
    last_committed_ordinal: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RagPageRecord:
    id: UUID
    run_id: UUID
    user_id: UUID
    knowledge_base_id: UUID
    ordinal: int
    path: str
    intent: str
    query: str
    state: RagPageState
    document_id: UUID | None
    version_read: int | None
    version_committed: int | None
    attempt_count: int
    conflict_retry_count: int
    last_completed_step_sequence: int
    preview: str | None
    preview_digest: str | None
    preview_full_char_count: int | None
    preview_truncated: bool
    lint_summary: Mapping[str, object] | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RagStepRecord:
    id: UUID
    run_id: UUID
    run_page_id: UUID | None
    user_id: UUID
    knowledge_base_id: UUID
    sequence: int
    step_type: RagStepType
    status: RagStepStatus
    input_digest: str
    output_summary: Mapping[str, object]
    citation_identities: tuple[RagCitation, ...]
    prompt_version: str | None
    prompt_digest: str | None
    model_profile_version: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


def _uuid(value: object, field: str) -> UUID:
    if isinstance(value, UUID):
        return value
    if type(value) is not str:
        raise TypeError(f"{field} must be a UUID")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise TypeError(f"{field} must be a UUID") from exc
    if str(parsed) != value:
        raise TypeError(f"{field} must be a canonical UUID")
    return parsed


def _optional_uuid(value: object, field: str) -> UUID | None:
    return None if value is None else _uuid(value, field)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} is outside its allowed range")
    return value


def _optional_positive_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, minimum=1)


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a boolean")
    return value


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise TypeError(f"{field} must be timezone-aware")
    return value


def _text(
    value: object,
    field: str,
    *,
    min_chars: int = 1,
    max_chars: int,
    max_bytes: int | None = None,
    normalized: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be UTF-8 encodable") from exc
    if not min_chars <= len(value) <= max_chars:
        raise ValueError(f"{field} is outside its allowed length")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ValueError(f"{field} exceeds its byte limit")
    if "\x00" in value or (normalized and value.strip() != value):
        raise ValueError(f"{field} is not normalized")
    return value


def _optional_text(
    value: object,
    field: str,
    *,
    max_chars: int,
    max_bytes: int | None = None,
) -> str | None:
    if value is None:
        return None
    return _text(value, field, min_chars=0, max_chars=max_chars, max_bytes=max_bytes)


def _digest(value: object, field: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: object, field: str) -> str | None:
    return None if value is None else _digest(value, field)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"database JSON contains a non-finite value: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"database JSON contains a duplicate key: {key}")
        value[key] = item
    return value


def _require_json_depth(value: object, field: str) -> None:
    stack = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ValueError(f"{field} exceeds the JSON nesting limit")
        if isinstance(current, Mapping):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray, memoryview)):
            stack.extend((item, depth + 1) for item in current)


def _parse_db_json(value: object, field: str) -> object:
    if type(value) is not str:
        _require_json_depth(value, field)
        return value
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise TypeError(f"{field} contains invalid database JSON") from exc
    if len(encoded) > _JSON_COLUMN_BYTE_LIMITS[field]:
        raise ValueError(f"{field} exceeds its byte limit")
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        _require_json_depth(parsed, field)
        return parsed
    except RecursionError as exc:
        raise ValueError(f"{field} exceeds the JSON nesting limit") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        if "nesting limit" in str(exc):
            raise
        raise TypeError(f"{field} contains invalid database JSON") from exc


def _adapt_db_json_columns(row: Mapping[str, object], columns: tuple[str, ...]) -> Mapping[str, object]:
    adapted = dict(row)
    for column in columns:
        if adapted[column] is not None:
            adapted[column] = _parse_db_json(adapted[column], column)
    return adapted


def _adapt_db_run_row(row: Mapping[str, object]) -> Mapping[str, object]:
    return _adapt_db_json_columns(row, ("budget", "usage"))


def _adapt_db_page_row(row: Mapping[str, object]) -> Mapping[str, object]:
    return _adapt_db_json_columns(row, ("lint_summary",))


def _adapt_db_step_row(row: Mapping[str, object]) -> Mapping[str, object]:
    return _adapt_db_json_columns(row, ("output_summary", "citation_identities"))


def _freeze_json(value: object, field: str, *, depth: int = 1) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{field} exceeds the JSON nesting limit")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError(f"{field} keys must be strings")
        return MappingProxyType({key: _freeze_json(item, field, depth=depth + 1) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return tuple(_freeze_json(item, field, depth=depth + 1) for item in value)
    raise TypeError(f"{field} contains a non-JSON value")


def _json_object(value: object, field: str, *, max_bytes: int) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a JSON object")
    try:
        frozen = _freeze_json(value, field)
        serialized = json.dumps(dict(value), ensure_ascii=False, allow_nan=False)
    except RecursionError as exc:
        raise ValueError(f"{field} exceeds the JSON nesting limit") from exc
    if len(serialized.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} exceeds its byte limit")
    assert isinstance(frozen, Mapping)
    return frozen


def _budget(value: object) -> RagBudget:
    obj = value
    _require_json_depth(obj, "budget")
    if not isinstance(obj, Mapping) or set(obj) != _BUDGET_KEYS:
        raise TypeError("budget must contain exactly the supported keys")
    values = {key: _integer(obj[key], f"budget.{key}", minimum=1) for key in _BUDGET_KEYS}
    return RagBudget(**values)


def _usage(value: object, budget: RagBudget | None = None) -> RagUsage:
    obj = value
    _require_json_depth(obj, "usage")
    if not isinstance(obj, Mapping) or set(obj) != _USAGE_KEYS:
        raise TypeError("usage must contain exactly steps and model_tokens")
    usage = RagUsage(
        steps=_integer(obj["steps"], "usage.steps"),
        model_tokens=_integer(obj["model_tokens"], "usage.model_tokens"),
    )
    if budget is not None and (usage.steps > budget.max_steps or usage.model_tokens > budget.max_model_tokens):
        raise ValueError("usage exceeds budget")
    return usage


def _completion(value: object) -> RagCompletionReason | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError("completion_reason must be a string or None")
    try:
        return RagCompletionReason(value)
    except ValueError as exc:
        raise ValueError("completion_reason is unsupported") from exc


def _enum(enum_type: type[Any], value: object, field: str):
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{field} is unsupported") from exc


def _decode_run(row: Mapping[str, object]) -> RagRunRecord:
    budget = _budget(row["budget"])
    usage = _usage(row["usage"], budget)
    knowledge_base_id = _uuid(row["knowledge_base_id"], "knowledge_base_id")
    goal = _text(row["goal"], "goal", max_chars=MAX_GOAL_CHARS)
    target_path = _text(
        row["target_path_prefix"],
        "target_path_prefix",
        max_chars=8_000,
        max_bytes=8_000,
    )
    model_profile = _text(row["model_profile"], "model_profile", max_chars=MAX_PROFILE_CHARS)
    retrieval_profile = row["retrieval_profile"]
    dry_run = _boolean(row["dry_run"], "dry_run")
    try:
        normalized = RagRunConfig.build(
            knowledge_base_id=knowledge_base_id,
            goal=goal,
            target_path_prefix=target_path,
            model_profile=model_profile,
            retrieval_profile=retrieval_profile,  # type: ignore[arg-type]
            dry_run=dry_run,
            budget=budget,
        )
    except ValueError as exc:
        raise ValueError("run configuration is invalid") from exc
    if (
        normalized.goal != goal
        or normalized.target_path_prefix != target_path
        or normalized.model_profile != model_profile
    ):
        raise ValueError("run configuration is not normalized")
    run_id = _uuid(row["id"], "id")
    root_run_id = _uuid(row["root_run_id"], "root_run_id")
    parent_run_id = _optional_uuid(row["parent_run_id"], "parent_run_id")
    if parent_run_id is None and root_run_id != run_id:
        raise ValueError("root run lineage is inconsistent")
    if parent_run_id is not None and (parent_run_id == run_id or root_run_id == run_id):
        raise ValueError("resume run lineage is inconsistent")
    completion = _completion(row["completion_reason"])
    if completion is RagCompletionReason.DRY_RUN and not dry_run:
        raise ValueError("dry-run completion requires a dry-run configuration")
    goal_digest = _digest(row["goal_digest"], "goal_digest")
    if goal_digest != hashlib.sha256(goal.encode("utf-8")).hexdigest():
        raise ValueError("goal_digest does not match goal")
    last_ordinal = _integer(row["last_committed_ordinal"], "last_committed_ordinal", minimum=-1)
    if last_ordinal >= MAX_PAGES:
        raise ValueError("last_committed_ordinal exceeds the core page cap")
    if completion is RagCompletionReason.NO_WORK and last_ordinal != -1:
        raise ValueError("no-work run cannot have a committed boundary")
    return RagRunRecord(
        id=run_id,
        job_id=_uuid(row["job_id"], "job_id"),
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
        user_id=_uuid(row["user_id"], "user_id"),
        knowledge_base_id=knowledge_base_id,
        goal=goal,
        goal_digest=goal_digest,
        target_path_prefix=target_path,
        model_profile=model_profile,
        model_profile_version=_text(row["model_profile_version"], "model_profile_version", max_chars=128),
        retrieval_profile=normalized.retrieval_profile,
        dry_run=dry_run,
        budget=budget,
        usage=usage,
        idempotency_key=_text(row["idempotency_key"], "idempotency_key", max_chars=200, normalized=True),
        request_digest=_digest(row["request_digest"], "request_digest"),
        completion_reason=completion,
        last_committed_ordinal=last_ordinal,
        created_at=_timestamp(row["created_at"], "created_at"),
        updated_at=_timestamp(row["updated_at"], "updated_at"),
    )


def _decode_preview(row: Mapping[str, object]) -> tuple[str | None, str | None, int | None, bool]:
    preview = _optional_text(row["preview"], "preview", max_chars=16_384, max_bytes=16_384)
    preview_digest = _optional_digest(row["preview_digest"], "preview_digest")
    preview_count = (
        None
        if row["preview_full_char_count"] is None
        else _integer(row["preview_full_char_count"], "preview_full_char_count")
    )
    if preview_count is not None and preview_count > MAX_PAGE_CHARS:
        raise ValueError("preview_full_char_count exceeds the core page cap")
    preview_truncated = _boolean(row["preview_truncated"], "preview_truncated")
    if preview is None and (preview_digest is not None or preview_count is not None or preview_truncated):
        raise ValueError("preview metadata is inconsistent")
    if preview is not None and (preview_digest is None or preview_count is None or preview_count < len(preview)):
        raise ValueError("preview metadata is incomplete")
    if preview is not None and preview_count is not None and preview_truncated != (preview_count > len(preview)):
        raise ValueError("preview truncation metadata is inconsistent")
    if (
        preview is not None
        and not preview_truncated
        and preview_digest != hashlib.sha256(preview.encode("utf-8")).hexdigest()
    ):
        raise ValueError("complete preview digest does not match preview")
    return preview, preview_digest, preview_count, preview_truncated


def _validate_page_counters_and_versions(
    version_read: int | None,
    version_committed: int | None,
    attempt_count: int,
    conflict_count: int,
) -> None:
    if conflict_count > attempt_count:
        raise ValueError("conflict retries cannot exceed page attempts")
    if version_read is not None and version_committed is not None and version_read > version_committed:
        raise ValueError("version_read cannot exceed version_committed")


def _validate_active_page_state(
    state: RagPageState,
    document_id: UUID | None,
    version_read: int | None,
    version_committed: int | None,
    attempt_count: int,
    preview: str | None,
) -> None:
    if state is RagPageState.PLANNED and (document_id is not None or version_read is not None):
        raise ValueError("planned page cannot contain document identity")
    if state in {RagPageState.RUNNING, RagPageState.FAILED} and document_id is not None and version_read is None:
        raise ValueError("in-progress document identity requires version_read")
    if version_committed is not None:
        raise ValueError("uncommitted page has a committed version")
    if preview is not None:
        raise ValueError("only a dry-run-complete page can contain a preview")
    if state is RagPageState.RUNNING and attempt_count < 1:
        raise ValueError("running page has no attempt")


def _validate_page_state(
    state: RagPageState,
    document_id: UUID | None,
    version_read: int | None,
    version_committed: int | None,
    attempt_count: int,
    preview: str | None,
    lint_summary: Mapping[str, object] | None,
) -> None:
    if state is RagPageState.COMMITTED:
        if document_id is None or version_committed is None or lint_summary is None:
            raise ValueError("committed page is missing boundary identity")
        expected_read = None if version_committed == 1 else version_committed - 1
        if version_read != expected_read:
            raise ValueError("committed page version predecessor is inconsistent")
        if preview is not None:
            raise ValueError("committed page cannot contain a dry-run preview")
        return
    if lint_summary is not None:
        raise ValueError("uncommitted page cannot contain a lint summary")
    if state is RagPageState.DRY_RUN_COMPLETE:
        if document_id is not None or version_read is not None or version_committed is not None:
            raise ValueError("dry-run page cannot contain document identity")
        if preview is None:
            raise ValueError("dry-run page requires valid preview metadata")
        if attempt_count < 1:
            raise ValueError("dry-run-complete page has no attempt")
        return
    _validate_active_page_state(
        state,
        document_id,
        version_read,
        version_committed,
        attempt_count,
        preview,
    )


def _decode_page(row: Mapping[str, object]) -> RagPageRecord:
    ordinal = _integer(row["ordinal"], "ordinal")
    if ordinal >= MAX_PAGES:
        raise ValueError("ordinal exceeds the core page cap")
    path = _text(row["path"], "path", max_chars=2_000, max_bytes=8_000, normalized=True)
    intent = _text(row["intent"], "intent", max_chars=MAX_WORK_ITEM_TEXT_CHARS, normalized=True)
    query = _text(row["query"], "query", max_chars=MAX_WORK_ITEM_TEXT_CHARS, normalized=True)
    try:
        work_item = RagWorkItem(ordinal, path, intent, query)
    except ValueError as exc:
        raise ValueError("page work item is invalid") from exc
    state = _enum(RagPageState, row["state"], "state")
    document_id = _optional_uuid(row["document_id"], "document_id")
    version_read = _optional_positive_integer(row["version_read"], "version_read")
    version_committed = _optional_positive_integer(row["version_committed"], "version_committed")
    if document_id is None and (version_read is not None or version_committed is not None):
        raise ValueError("document versions require a document identity")
    attempt_count = _integer(row["attempt_count"], "attempt_count")
    conflict_count = _integer(row["conflict_retry_count"], "conflict_retry_count")
    if attempt_count > 3 or conflict_count > 3:
        raise ValueError("page counters exceed core caps")
    last_sequence = _integer(row["last_completed_step_sequence"], "last_completed_step_sequence")
    if last_sequence > MAX_STEPS:
        raise ValueError("last_completed_step_sequence exceeds the core step cap")
    preview, preview_digest, preview_count, preview_truncated = _decode_preview(row)
    lint_summary = (
        None if row["lint_summary"] is None else _json_object(row["lint_summary"], "lint_summary", max_bytes=16_384)
    )
    _validate_page_counters_and_versions(
        version_read,
        version_committed,
        attempt_count,
        conflict_count,
    )
    _validate_page_state(
        state,
        document_id,
        version_read,
        version_committed,
        attempt_count,
        preview,
        lint_summary,
    )
    return RagPageRecord(
        id=_uuid(row["id"], "id"),
        run_id=_uuid(row["run_id"], "run_id"),
        user_id=_uuid(row["user_id"], "user_id"),
        knowledge_base_id=_uuid(row["knowledge_base_id"], "knowledge_base_id"),
        ordinal=work_item.ordinal,
        path=work_item.path,
        intent=work_item.intent,
        query=work_item.query,
        state=state,
        document_id=document_id,
        version_read=version_read,
        version_committed=version_committed,
        attempt_count=attempt_count,
        conflict_retry_count=conflict_count,
        last_completed_step_sequence=last_sequence,
        preview=preview,
        preview_digest=preview_digest,
        preview_full_char_count=preview_count,
        preview_truncated=preview_truncated,
        lint_summary=lint_summary,
        created_at=_timestamp(row["created_at"], "created_at"),
        updated_at=_timestamp(row["updated_at"], "updated_at"),
    )


def _citations(value: object) -> tuple[RagCitation, ...]:
    parsed = value
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes, bytearray)):
        raise TypeError("citation_identities must be a JSON array")
    if len(parsed) > 128:
        raise ValueError("citation_identities exceeds its item limit")
    _require_json_depth(parsed, "citation_identities")
    try:
        serialized = json.dumps(parsed, ensure_ascii=False, allow_nan=False)
    except RecursionError as exc:
        raise ValueError("citation_identities exceeds the JSON nesting limit") from exc
    if len(serialized.encode("utf-8")) > 16_384:
        raise ValueError("citation_identities exceeds its byte limit")
    citations: list[RagCitation] = []
    seen: set[tuple[UUID, int, int, int | None]] = set()
    for raw in parsed:
        if not isinstance(raw, Mapping) or set(raw) != _CITATION_KEYS:
            raise TypeError("citation identity has an invalid shape")
        citation = RagCitation(
            document_id=_uuid(raw["document_id"], "citation.document_id"),
            document_version=_integer(raw["document_version"], "citation.document_version", minimum=1),
            chunk_index=_integer(raw["chunk_index"], "citation.chunk_index"),
            page=(None if raw["page"] is None else _integer(raw["page"], "citation.page", minimum=1)),
        )
        identity = (
            citation.document_id,
            citation.document_version,
            citation.chunk_index,
            citation.page,
        )
        if identity in seen:
            raise ValueError("citation identities must be unique")
        seen.add(identity)
        citations.append(citation)
    return tuple(citations)


def _decode_step(row: Mapping[str, object]) -> RagStepRecord:
    status = _enum(RagStepStatus, row["status"], "status")
    summary = _json_object(row["output_summary"], "output_summary", max_bytes=16_384)
    citations = _citations(row["citation_identities"])
    sequence = _integer(row["sequence"], "sequence", minimum=1)
    if sequence > MAX_STEPS:
        raise ValueError("sequence exceeds the core step cap")
    input_tokens = _integer(row["input_tokens"], "input_tokens")
    output_tokens = _integer(row["output_tokens"], "output_tokens")
    total_tokens = _integer(row["total_tokens"], "total_tokens")
    if total_tokens != input_tokens + output_tokens or total_tokens > 250_000:
        raise ValueError("step token totals are inconsistent")
    latency_raw = row["latency_ms"]
    if type(latency_raw) not in {int, float}:
        raise TypeError("latency_ms must be a number")
    latency = float(latency_raw)
    if not isfinite(latency) or latency < 0:
        raise ValueError("latency_ms is outside its allowed range")
    prompt_version = (
        None if row["prompt_version"] is None else _text(row["prompt_version"], "prompt_version", max_chars=128)
    )
    prompt_digest = _optional_digest(row["prompt_digest"], "prompt_digest")
    if (prompt_version is None) != (prompt_digest is None):
        raise ValueError("prompt metadata is inconsistent")
    error_code = row["error_code"]
    if error_code is not None and (type(error_code) is not str or _ERROR_CODE.fullmatch(error_code) is None):
        raise ValueError("error_code is invalid")
    error_message = _optional_text(row["error_message"], "error_message", max_chars=2_000, max_bytes=8_000)
    if error_message is not None and any(ord(char) < 32 for char in error_message):
        raise ValueError("error_message contains control characters")
    if status is RagStepStatus.RUNNING:
        if summary or citations or total_tokens or latency or error_code is not None or error_message is not None:
            raise ValueError("running step contains terminal output")
    elif status is RagStepStatus.SUCCEEDED and (error_code is not None or error_message is not None):
        raise ValueError("succeeded step contains an error")
    elif status is RagStepStatus.FAILED and error_code is None:
        raise ValueError("failed step is missing an error code")
    return RagStepRecord(
        id=_uuid(row["id"], "id"),
        run_id=_uuid(row["run_id"], "run_id"),
        run_page_id=_optional_uuid(row["run_page_id"], "run_page_id"),
        user_id=_uuid(row["user_id"], "user_id"),
        knowledge_base_id=_uuid(row["knowledge_base_id"], "knowledge_base_id"),
        sequence=sequence,
        step_type=_enum(RagStepType, row["step_type"], "step_type"),
        status=status,
        input_digest=_digest(row["input_digest"], "input_digest"),
        output_summary=summary,
        citation_identities=citations,
        prompt_version=prompt_version,
        prompt_digest=prompt_digest,
        model_profile_version=_text(row["model_profile_version"], "model_profile_version", max_chars=128),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency,
        error_code=error_code,
        error_message=error_message,
        created_at=_timestamp(row["created_at"], "created_at"),
        updated_at=_timestamp(row["updated_at"], "updated_at"),
    )


__all__ = ["RagPageRecord", "RagRunRecord", "RagStepRecord"]
