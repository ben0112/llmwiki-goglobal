"""Narrow typed ports for the server-side RAG application boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import NoReturn, Protocol, runtime_checkable
from uuid import UUID

from jobs.models import JobRecord

from llmwiki_adapters.postgres.wiki import WikiWriteResult
from llmwiki_core.rag import (
    RagCitation,
    RagCompletionReason,
    RagPageState,
    RagStepStatus,
    RagUsage,
    RagWorkItem,
)
from llmwiki_core.search import SearchHit, SearchQuery, SearchResult
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown
from llmwiki_core.wiki import WikiWriteBundle

from .model import RagModelResponse, RagTokenUsage
from .records import RagPageRecord, RagRunRecord, RagStepRecord
from .retrieval import RagEvidence, RagWikiPage


@dataclass(frozen=True, slots=True, repr=False)
class AuthoritativeRagRun:
    """A run and its canonical durable scheduler record."""

    run: RagRunRecord
    job: JobRecord

    def __post_init__(self) -> None:
        if type(self.run) is not RagRunRecord or type(self.job) is not JobRecord:
            raise TypeError("authoritative RAG state is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class WikiCatalogItem:
    """The only existing-wiki fields exposed to planning."""

    path: str
    title: str | None

    def __post_init__(self) -> None:
        try:
            normalized_path = RagWorkItem.build(0, self.path, "catalog", "catalog").path
        except (TypeError, ValueError):
            raise ValueError("wiki catalog item is invalid") from None
        if normalized_path != self.path or not self.path.startswith("/wiki/") or len(self.path) > 4_096:
            raise ValueError("wiki catalog item is invalid")
        if self.title is not None and (
            type(self.title) is not str
            or not 1 <= len(self.title) <= 4_096
            or "\x00" in self.title
            or self.title.strip() != self.title
        ):
            raise ValueError("wiki catalog item is invalid")
        try:
            self.path.encode("utf-8")
            if self.title is not None:
                self.title.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("wiki catalog item is invalid") from exc


@dataclass(frozen=True, slots=True, repr=False)
class PageExecutionResult:
    """One page boundary returned by an injected page runner."""

    run: RagRunRecord
    page: RagPageRecord

    def __post_init__(self) -> None:
        if (
            type(self.run) is not RagRunRecord
            or type(self.page) is not RagPageRecord
            or self.page.run_id != self.run.id
        ):
            raise TypeError("page execution result is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PlanAttemptSpec:
    """One immutable initial-or-repair planner prompt identity."""

    repair: bool
    input_digest: str
    prompt_version: str
    prompt_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.repair) is not bool
            or type(self.input_digest) is not str
            or not _is_digest(self.input_digest)
            or type(self.prompt_version) is not str
            or not 1 <= len(self.prompt_version) <= 128
            or self.prompt_version.strip() != self.prompt_version
            or type(self.prompt_digest) is not str
            or not _is_digest(self.prompt_digest)
        ):
            raise TypeError("plan attempt specification is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PlanAttempt:
    """One lease-fenced running planner step, new or idempotently recovered."""

    run: RagRunRecord
    step: RagStepRecord
    spec: PlanAttemptSpec
    resumed: bool

    def __post_init__(self) -> None:
        if (
            type(self.run) is not RagRunRecord
            or type(self.step) is not RagStepRecord
            or type(self.spec) is not PlanAttemptSpec
            or type(self.resumed) is not bool
            or self.step.run_id != self.run.id
            or self.step.run_page_id is not None
            or self.step.status is not RagStepStatus.RUNNING
            or self.run.completion_reason is not None
        ):
            raise TypeError("plan attempt is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PlanAttemptBinding:
    """Atomic canonical-input bind, including terminal step-budget exhaustion."""

    run: RagRunRecord
    attempt: PlanAttempt | None
    exhausted: bool

    def __post_init__(self) -> None:
        if (
            type(self.run) is not RagRunRecord
            or type(self.exhausted) is not bool
            or self.exhausted is not (self.attempt is None)
            or (self.attempt is not None and (type(self.attempt) is not PlanAttempt or self.attempt.run != self.run))
            or self.run.completion_reason is not None
        ):
            raise TypeError("plan attempt binding is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class PlanAcceptance:
    """Atomic accepted worklist, terminal when the accepted plan is empty."""

    run: RagRunRecord
    pages: tuple[RagPageRecord, ...]
    completion_reason: RagCompletionReason | None

    def __post_init__(self) -> None:
        if (
            type(self.run) is not RagRunRecord
            or type(self.pages) is not tuple
            or any(type(page) is not RagPageRecord for page in self.pages)
            or (self.completion_reason is not None and type(self.completion_reason) is not RagCompletionReason)
            or any(page.run_id != self.run.id for page in self.pages)
            or (not self.pages and self.completion_reason is not RagCompletionReason.NO_WORK)
            or (self.pages and self.completion_reason is not None)
            or self.run.completion_reason is not self.completion_reason
        ):
            raise TypeError("plan acceptance is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class AtomicPageCommit:
    """Lease-fenced page publication and durable RAG boundary result."""

    run: RagRunRecord
    page: RagPageRecord
    written: WikiWriteResult
    lint_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            frozen_summary = _freeze_json_mapping(self.lint_summary)
            frozen_page_summary = _freeze_json_mapping(self.page.lint_summary)
            canonical_path = f"{self.written.path}{self.written.filename}"
        except BaseException as failure:  # noqa: BLE001 - hostile boundary projection.
            if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
                raise signal from None
            raise TypeError("atomic page commit is invalid") from None
        if (
            type(self.run) is not RagRunRecord
            or type(self.page) is not RagPageRecord
            or type(self.written) is not WikiWriteResult
            or type(self.run.id) is not UUID
            or type(self.run.user_id) is not UUID
            or type(self.run.knowledge_base_id) is not UUID
            or type(self.page.id) is not UUID
            or type(self.page.run_id) is not UUID
            or type(self.page.user_id) is not UUID
            or type(self.page.knowledge_base_id) is not UUID
            or self.page.run_id != self.run.id
            or self.page.user_id != self.run.user_id
            or self.page.knowledge_base_id != self.run.knowledge_base_id
            or self.page.state is not RagPageState.COMMITTED
            or self.run.dry_run
            or self.run.last_committed_ordinal != self.page.ordinal
            or type(self.written.document_id) is not UUID
            or self.page.document_id != self.written.document_id
            or self.page.version_committed != self.written.version
            or type(self.written.filename) is not str
            or not self.written.filename
            or "/" in self.written.filename
            or type(self.written.path) is not str
            or not self.written.path.endswith("/")
            or canonical_path != self.page.path
            or type(self.written.version) is not int
            or self.written.version < 1
            or frozen_page_summary != frozen_summary
        ):
            raise TypeError("atomic page commit is invalid")
        object.__setattr__(self, "lint_summary", frozen_summary)
        object.__setattr__(self, "page", replace(self.page, lint_summary=frozen_page_summary))


@runtime_checkable
class RunStore(Protocol):
    async def reload(self, run_id: UUID) -> AuthoritativeRagRun: ...

    async def list_pages(self, run_id: UUID) -> tuple[RagPageRecord, ...]: ...

    async def begin_plan_attempt(
        self,
        *,
        run: RagRunRecord,
        job: JobRecord,
        lease_owner: str,
        specs: tuple[PlanAttemptSpec, PlanAttemptSpec],
    ) -> PlanAttempt: ...

    async def bind_plan_attempt_input(
        self,
        *,
        run: RagRunRecord,
        job: JobRecord,
        lease_owner: str,
        attempt: PlanAttempt,
        input_digest: str,
    ) -> PlanAttemptBinding: ...

    async def finish_plan_attempt(
        self,
        *,
        run: RagRunRecord,
        job: JobRecord,
        lease_owner: str,
        step: RagStepRecord,
        status: RagStepStatus,
        summary: Mapping[str, object],
        citations: tuple[RagCitation, ...],
        token_usage: RagTokenUsage,
        aggregate_usage: RagUsage,
        latency_ms: float,
        error_code: str | None,
        prompt_version: str | None,
        prompt_digest: str | None,
    ) -> RagRunRecord: ...

    async def accept_plan(
        self,
        *,
        run: RagRunRecord,
        job: JobRecord,
        lease_owner: str,
        step: RagStepRecord,
        items: tuple[RagWorkItem, ...],
        summary: Mapping[str, object],
        token_usage: RagTokenUsage,
        aggregate_usage: RagUsage,
        latency_ms: float,
        prompt_version: str,
        prompt_digest: str,
        completion_reason: RagCompletionReason | None,
    ) -> PlanAcceptance: ...

    async def finish_run(
        self,
        *,
        run: RagRunRecord,
        job: JobRecord,
        lease_owner: str,
        completion_reason: RagCompletionReason,
        usage: RagUsage,
    ) -> RagRunRecord: ...


@runtime_checkable
class StructuredRagModel(Protocol):
    async def complete_json(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> RagModelResponse: ...


@runtime_checkable
class RagRetrieval(Protocol):
    async def retrieve(self, query: SearchQuery, *, profile: str = "lexical") -> SearchResult: ...


@runtime_checkable
class EvidenceReader(Protocol):
    async def read(
        self,
        user_id: UUID,
        knowledge_base_id: UUID,
        hits: Sequence[SearchHit],
        max_chars: int,
    ) -> tuple[RagEvidence, ...]: ...


@runtime_checkable
class WikiCatalog(Protocol):
    async def list_for_planning(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        target_path_prefix: str,
        limit: int,
    ) -> tuple[WikiCatalogItem, ...]: ...


@runtime_checkable
class WikiPageReader(Protocol):
    async def get_by_path(
        self,
        user_id: UUID,
        knowledge_base_id: UUID,
        path: str,
    ) -> RagWikiPage | None: ...


@runtime_checkable
class WikiWriter(Protocol):
    async def commit(
        self,
        *,
        job_id: UUID,
        lease_owner: str,
        run: RagRunRecord,
        page: RagPageRecord,
        bundle: WikiWriteBundle,
        usage: RagUsage,
    ) -> AtomicPageCommit: ...


@runtime_checkable
class DraftLinter(Protocol):
    async def lint(
        self,
        *,
        run: RagRunRecord,
        page: RagPageRecord,
        content: str,
        citations: tuple[RagCitation, ...],
    ) -> Mapping[str, object]: ...


@runtime_checkable
class FeatureGate(Protocol):
    async def ensure_enabled(self) -> None: ...


@runtime_checkable
class LeaseCheckpoint(Protocol):
    async def checkpoint(self) -> JobRecord: ...


@runtime_checkable
class PageRunner(Protocol):
    async def run(
        self,
        run: RagRunRecord,
        page: RagPageRecord,
        lease: LeaseCheckpoint,
    ) -> PageExecutionResult: ...


@dataclass(frozen=True, slots=True, repr=False)
class OrchestratorPorts:
    store: RunStore
    model: StructuredRagModel
    retrieval: RagRetrieval
    evidence_reader: EvidenceReader
    wiki_catalog: WikiCatalog
    wiki_page_reader: WikiPageReader
    wiki_writer: WikiWriter
    draft_linter: DraftLinter
    feature_gate: FeatureGate
    page_runner: PageRunner

    def __post_init__(self) -> None:
        requirements = (
            (
                self.store,
                (
                    "reload",
                    "list_pages",
                    "begin_plan_attempt",
                    "bind_plan_attempt_input",
                    "finish_plan_attempt",
                    "accept_plan",
                    "finish_run",
                ),
            ),
            (self.model, ("complete_json",)),
            (self.retrieval, ("retrieve",)),
            (self.evidence_reader, ("read",)),
            (self.wiki_catalog, ("list_for_planning",)),
            (self.wiki_page_reader, ("get_by_path",)),
            (self.wiki_writer, ("commit",)),
            (self.draft_linter, ("lint",)),
            (self.feature_gate, ("ensure_enabled",)),
            (self.page_runner, ("run",)),
        )
        for port, methods in requirements:
            _require_port(port, methods)

    def __repr__(self) -> str:
        return "OrchestratorPorts(<redacted>)"

    def __str__(self) -> str:
        return "OrchestratorPorts(<redacted>)"


def _require_port(port: object, methods: tuple[str, ...]) -> None:
    failure: BaseException | None = None
    try:
        if port is None or any(not callable(getattr(port, method)) for method in methods):
            raise TypeError
    except BaseException as caught:  # noqa: BLE001 - hostile dependency objects fail closed.
        failure = caught
    if failure is None:
        return
    _raise_port_error(failure)


def _freeze_json_mapping(value: object) -> Mapping[str, object]:
    try:
        item_count = [0]
        frozen, thawed = _freeze_json(value, 0, item_count)
        if item_count[0] > 4_096:
            raise ValueError
        encoded = json.dumps(thawed, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 16_384:
            raise ValueError
    except BaseException as failure:  # noqa: BLE001 - hostile boundary values fail closed.
        _raise_port_error(failure)
    if not isinstance(frozen, Mapping):
        raise TypeError("atomic page commit is invalid")
    return frozen


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _freeze_json(value: object, depth: int, item_count: list[int]) -> tuple[object, object]:
    if depth > 32:
        raise ValueError
    item_count[0] += 1
    if value is None or type(value) in {str, bool, int}:
        return value, value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError
        return value, value
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError
        frozen: dict[str, object] = {}
        thawed: dict[str, object] = {}
        for key, item in value.items():
            frozen_item, thawed_item = _freeze_json(item, depth + 1, item_count)
            frozen[key] = frozen_item
            thawed[key] = thawed_item
        return MappingProxyType(frozen), thawed
    if type(value) in {list, tuple}:
        values = tuple(_freeze_json(item, depth + 1, item_count) for item in value)
        return tuple(item[0] for item in values), [item[1] for item in values]
    raise TypeError


def _raise_port_error(failure: BaseException) -> NoReturn:
    if (signal := sanitized_boundary_signal_or_unknown(failure)) and type(signal) is not BaseException:
        raise signal from None
    raise TypeError("orchestrator port is invalid") from None


__all__ = [
    "AtomicPageCommit",
    "AuthoritativeRagRun",
    "DraftLinter",
    "EvidenceReader",
    "FeatureGate",
    "LeaseCheckpoint",
    "OrchestratorPorts",
    "PageExecutionResult",
    "PlanAcceptance",
    "PlanAttempt",
    "PlanAttemptBinding",
    "PlanAttemptSpec",
    "PageRunner",
    "RagRetrieval",
    "RunStore",
    "StructuredRagModel",
    "WikiCatalog",
    "WikiCatalogItem",
    "WikiPageReader",
    "WikiWriter",
]
