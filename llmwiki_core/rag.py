"""Pure contracts and bounds for server-side RAG runs."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import islice
from typing import Any
from uuid import UUID

from .documents import normalize_directory_path

DEFAULT_MAX_PAGES = 8
DEFAULT_MAX_STEPS = 96
DEFAULT_MAX_MODEL_TOKENS = 64_000
DEFAULT_MAX_CONTEXT_CHARS = 120_000
DEFAULT_MAX_PAGE_CHARS = 40_000
DEFAULT_PER_CALL_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PAGE_ATTEMPTS = 2
DEFAULT_MAX_CONFLICT_RETRIES = 1

MAX_PAGES = 32
MAX_STEPS = 512
MAX_MODEL_TOKENS = 250_000
MAX_CONTEXT_CHARS = 240_000
MAX_PAGE_CHARS = 120_000
MAX_PER_CALL_TIMEOUT_SECONDS = 180
MAX_PAGE_ATTEMPTS = 3
MAX_CONFLICT_RETRIES = 3
MAX_GOAL_CHARS = 4_000
MAX_PROFILE_CHARS = 100
MAX_WORK_ITEM_TEXT_CHARS = 2_000


class RagPageState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMMITTED = "committed"
    DRY_RUN_COMPLETE = "dry_run_complete"
    FAILED = "failed"


class RagStepType(StrEnum):
    PLAN = "plan"
    RETRIEVE = "retrieve"
    READ = "read"
    DRAFT = "draft"
    VALIDATE = "validate"
    WRITE = "write"
    LINT = "lint"
    CONFLICT = "conflict"


class RagStepStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RagCompletionReason(StrEnum):
    COMPLETED = "completed"
    NO_WORK = "no_work"
    DRY_RUN = "dry_run"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PARTIAL_FAILURE = "partial_failure"


class RagDomainError(RuntimeError):
    """A stable, sanitized RAG domain error suitable for public reporting."""

    def __init__(self, code: str, public_message: str, retryable: bool = False) -> None:
        self.code = code
        self.public_message = public_message
        self.retryable = retryable
        super().__init__(public_message)


@dataclass(frozen=True, slots=True)
class RagBudget:
    max_pages: int = DEFAULT_MAX_PAGES
    max_steps: int = DEFAULT_MAX_STEPS
    max_model_tokens: int = DEFAULT_MAX_MODEL_TOKENS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS
    per_call_timeout_seconds: int = DEFAULT_PER_CALL_TIMEOUT_SECONDS
    max_page_attempts: int = DEFAULT_MAX_PAGE_ATTEMPTS
    max_conflict_retries: int = DEFAULT_MAX_CONFLICT_RETRIES

    def __post_init__(self) -> None:
        _validate_limit("max_pages", self.max_pages, MAX_PAGES)
        _validate_limit("max_steps", self.max_steps, MAX_STEPS)
        _validate_limit("max_model_tokens", self.max_model_tokens, MAX_MODEL_TOKENS)
        _validate_limit("max_context_chars", self.max_context_chars, MAX_CONTEXT_CHARS)
        _validate_limit("max_page_chars", self.max_page_chars, MAX_PAGE_CHARS)
        _validate_limit(
            "per_call_timeout_seconds",
            self.per_call_timeout_seconds,
            MAX_PER_CALL_TIMEOUT_SECONDS,
        )
        _validate_limit("max_page_attempts", self.max_page_attempts, MAX_PAGE_ATTEMPTS)
        _validate_limit("max_conflict_retries", self.max_conflict_retries, MAX_CONFLICT_RETRIES)

    def consume_page_attempt(self, attempts: int) -> int:
        _validate_counter("page attempts", attempts)
        if attempts >= self.max_page_attempts:
            raise RagDomainError("rag_page_attempts_exhausted", "The page attempt limit was exhausted.")
        return attempts + 1

    def consume_conflict_retry(self, retries: int) -> int:
        _validate_counter("conflict retries", retries)
        if retries >= self.max_conflict_retries:
            raise RagDomainError("rag_version_conflict", "The conflict retry limit was exhausted.")
        return retries + 1


@dataclass(frozen=True, slots=True)
class RagRunConfig:
    knowledge_base_id: UUID
    goal: str
    target_path_prefix: str
    model_profile: str
    retrieval_profile: str
    dry_run: bool
    budget: RagBudget

    @classmethod
    def build(
        cls,
        *,
        knowledge_base_id: UUID,
        goal: str,
        target_path_prefix: str,
        model_profile: str,
        retrieval_profile: str = "lexical",
        dry_run: bool = False,
        budget: RagBudget | None = None,
        max_pages: int | None = None,
        max_steps: int | None = None,
        max_model_tokens: int | None = None,
        max_context_chars: int | None = None,
        max_page_chars: int | None = None,
        per_call_timeout_seconds: int | None = None,
        max_page_attempts: int | None = None,
        max_conflict_retries: int | None = None,
    ) -> "RagRunConfig":
        if not isinstance(knowledge_base_id, UUID):
            raise ValueError("knowledge_base_id must be a UUID")
        normalized_goal = _normalize_bounded_text("goal", goal, MAX_GOAL_CHARS)
        normalized_target_path = _normalize_target_path_prefix(target_path_prefix)
        normalized_model_profile = _normalize_bounded_text(
            "model profile", model_profile, MAX_PROFILE_CHARS
        )
        if type(retrieval_profile) is not str or retrieval_profile not in {"lexical", "hybrid"}:
            raise ValueError("retrieval profile is unknown")
        if type(dry_run) is not bool:
            raise ValueError("dry_run must be a boolean")
        if budget is not None and not isinstance(budget, RagBudget):
            raise ValueError("budget must be a RagBudget")

        overrides = {
            name: value
            for name, value in {
                "max_pages": max_pages,
                "max_steps": max_steps,
                "max_model_tokens": max_model_tokens,
                "max_context_chars": max_context_chars,
                "max_page_chars": max_page_chars,
                "per_call_timeout_seconds": per_call_timeout_seconds,
                "max_page_attempts": max_page_attempts,
                "max_conflict_retries": max_conflict_retries,
            }.items()
            if value is not None
        }
        if budget is not None and overrides:
            raise ValueError("budget cannot be combined with budget overrides")
        resolved_budget = replace(budget, **overrides) if budget is not None else RagBudget(**overrides)

        return cls(
            knowledge_base_id=knowledge_base_id,
            goal=normalized_goal,
            target_path_prefix=normalized_target_path,
            model_profile=normalized_model_profile,
            retrieval_profile=retrieval_profile,
            dry_run=dry_run,
            budget=resolved_budget,
        )


@dataclass(frozen=True, slots=True)
class RagWorkItem:
    ordinal: int
    path: str
    intent: str
    query: str

    def __post_init__(self) -> None:
        _validate_work_item_fields(self.ordinal, self.path, self.intent, self.query)

    @classmethod
    def build(cls, ordinal: int, path: str, intent: str, query: str) -> "RagWorkItem":
        return cls(
            ordinal=ordinal,
            path=_normalize_markdown_path(path),
            intent=_normalize_bounded_text("intent", intent, MAX_WORK_ITEM_TEXT_CHARS),
            query=_normalize_bounded_text("query", query, MAX_WORK_ITEM_TEXT_CHARS),
        )


@dataclass(frozen=True, slots=True)
class RagUsage:
    steps: int = 0
    model_tokens: int = 0

    def __post_init__(self) -> None:
        _validate_counter("steps", self.steps)
        _validate_counter("model tokens", self.model_tokens)

    def consume_step(self, budget: RagBudget) -> "RagUsage":
        if self.steps >= budget.max_steps:
            raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.")
        return replace(self, steps=self.steps + 1)

    def reserve_model_call(self, budget: RagBudget, reserved_tokens: int) -> int:
        if type(reserved_tokens) is not int or reserved_tokens <= 0:
            raise ValueError("reserved model tokens must be positive")
        if self.model_tokens + reserved_tokens > budget.max_model_tokens:
            raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.")
        return reserved_tokens

    def commit_model_usage(
        self,
        budget: RagBudget,
        reservation: int,
        used_tokens: int,
    ) -> "RagUsage":
        if type(reservation) is not int or reservation <= 0:
            raise RagDomainError("rag_invalid_model_usage", "The model response was invalid.")
        if type(used_tokens) is not int or used_tokens <= 0:
            raise RagDomainError("rag_invalid_model_usage", "The model response was invalid.")
        if used_tokens > reservation:
            raise RagDomainError("rag_invalid_model_usage", "The model response was invalid.")
        if self.model_tokens + used_tokens > budget.max_model_tokens:
            raise RagDomainError("rag_budget_exhausted", "The RAG budget was exhausted.")
        return replace(self, model_tokens=self.model_tokens + used_tokens)


@dataclass(frozen=True, slots=True)
class RagCitation:
    document_id: UUID
    document_version: int
    chunk_index: int
    page: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, UUID):
            raise ValueError("document_id must be a UUID")
        if type(self.document_version) is not int or self.document_version < 1:
            raise ValueError("document_version must be a positive integer")
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("chunk_index must be a nonnegative integer")
        if self.page is not None and (type(self.page) is not int or self.page < 1):
            raise ValueError("page must be a positive integer or None")


def validate_worklist(
    items: Iterable[RagWorkItem], *, target_path_prefix: str, max_pages: int
) -> None:
    normalized_target_path = _normalize_target_path_prefix(target_path_prefix)
    _validate_limit("max_pages", max_pages, MAX_PAGES)

    materialized = tuple(islice(items, max_pages + 1))
    if len(materialized) > max_pages:
        raise ValueError("worklist exceeds max_pages")

    paths: set[str] = set()
    for expected_ordinal, item in enumerate(materialized):
        if not isinstance(item, RagWorkItem):
            raise ValueError("worklist items must be RagWorkItems")
        _validate_work_item_fields(item.ordinal, item.path, item.intent, item.query)
        if item.ordinal != expected_ordinal:
            raise ValueError("worklist ordinals must be contiguous")
        if not item.path.startswith(normalized_target_path):
            raise ValueError("work item path is outside the target path")
        if item.path in paths:
            raise ValueError("worklist contains a duplicate path")
        paths.add(item.path)


def remaining_work_items(
    items: Iterable[RagWorkItem], last_committed_ordinal: int
) -> tuple[RagWorkItem, ...]:
    if type(last_committed_ordinal) is not int:
        raise ValueError("last_committed_ordinal must be an integer")
    materialized = tuple(items)
    for expected_ordinal, item in enumerate(materialized):
        if not isinstance(item, RagWorkItem):
            raise ValueError("worklist items must be RagWorkItems")
        _validate_work_item_ordinal(item.ordinal)
        if item.ordinal != expected_ordinal:
            raise ValueError("worklist ordinals must be contiguous")
    if (last_committed_ordinal < -1 or last_committed_ordinal >= len(materialized)) and not (
        last_committed_ordinal == -1 and not materialized
    ):
        raise ValueError("last_committed_ordinal is outside the worklist")
    return tuple(item for item in materialized if item.ordinal > last_committed_ordinal)


def _validate_limit(name: str, value: int, cap: int) -> None:
    if type(value) is not int or value <= 0 or value > cap:
        raise ValueError(f"{name} must be a positive integer at most {cap}")


def _validate_counter(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_work_item_fields(ordinal: int, path: str, intent: str, query: str) -> None:
    _validate_work_item_ordinal(ordinal)
    if path != _normalize_markdown_path(path):
        raise ValueError("path must be normalized")
    if intent != _normalize_bounded_text("intent", intent, MAX_WORK_ITEM_TEXT_CHARS):
        raise ValueError("intent must be normalized")
    if query != _normalize_bounded_text("query", query, MAX_WORK_ITEM_TEXT_CHARS):
        raise ValueError("query must be normalized")


def _validate_work_item_ordinal(ordinal: int) -> None:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("ordinal must be a nonnegative integer")


def _normalize_bounded_text(name: str, raw: Any, maximum: int) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string")
    _validate_utf8_text(name, raw)
    normalized = raw.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{name} must contain between 1 and {maximum} characters")
    return normalized


def _normalize_target_path_prefix(raw: str) -> str:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError("target path must be an absolute path under /wiki/")
    _validate_utf8_text("target path", raw)
    try:
        normalized = normalize_directory_path(raw)
    except ValueError as exc:
        raise ValueError("target path must be a safe path under /wiki/") from exc
    if not normalized.startswith("/wiki/"):
        raise ValueError("target path must be under /wiki/")
    return normalized


def _validate_utf8_text(name: str, raw: str) -> None:
    if "\x00" in raw:
        raise ValueError(f"{name} must not contain NUL")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be UTF-8 encodable") from exc


def _normalize_markdown_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError("path must be an absolute Markdown path")
    if "\x00" in raw:
        raise ValueError("path contains NUL")
    normalized_parts = raw.replace("\\", "/").split("/")
    if any(part == ".." for part in normalized_parts):
        raise ValueError("path traversal is not allowed")
    parts = [part for part in normalized_parts if part and part != "."]
    if not parts or not parts[-1].endswith(".md"):
        raise ValueError("path must name a .md file")
    normalized = "/" + "/".join(parts)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("path must be UTF-8 encodable") from exc
    return normalized
