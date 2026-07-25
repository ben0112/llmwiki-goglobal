"""Deterministic, content-free retrieval evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from llmwiki_core import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationReport,
    EvaluationRun,
    RankedResult,
    SearchArea,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchScope,
    evaluate_rankings,
    evaluation_dataset_digest,
    load_cases,
    promotion_decision,
)
from llmwiki_core.documents import DocumentKind

REPORT_SCHEMA_VERSION = 1
MAX_CORPUS_BYTES = 8 * 1024 * 1024
MAX_CORPUS_LINE_BYTES = 256 * 1024
MAX_CORPUS_DOCUMENTS = 10_000
MAX_CORPUS_CHUNKS = 100_000
MAX_CORPUS_TEXT_CHARS = 1_000_000

_DOCUMENT_FIELDS = frozenset(
    {"schema_version", "document_id", "document_kind", "path", "title", "tags", "facets", "chunks"}
)
_DOCUMENT_REQUIRED_FIELDS = frozenset(
    {"schema_version", "document_id", "document_kind", "path", "title", "chunks"}
)
_CHUNK_FIELDS = frozenset({"chunk_index", "content", "source_content", "annotations_text"})
_CHUNK_REQUIRED_FIELDS = frozenset({"chunk_index", "content"})
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

RetrieverFactory = Callable[[str, Path], object]


class HybridConfigurationUnavailable(RuntimeError):
    """The later hosted hybrid-service wiring is not configured."""


class EvaluationDatasetError(RuntimeError):
    """The evaluation cohort or its sibling synthetic corpus is invalid."""


class RetrievalContractError(RuntimeError):
    """An injected retriever violated the shared retrieval contract."""


class RetrievalExecutionError(RuntimeError):
    """A retriever failed without exposing its backend exception."""


class OutputWriteError(RuntimeError):
    """The requested report output could not be written safely."""


_SAFE_BACKEND_EXCEPTIONS = (
    EvaluationDatasetError,
    HybridConfigurationUnavailable,
    RetrievalContractError,
    RetrievalExecutionError,
)


def _sanitized_process_control(error: BaseException) -> KeyboardInterrupt | SystemExit | None:
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt()
    if isinstance(error, SystemExit):
        if isinstance(error.code, bool):
            return SystemExit(int(error.code))
        return SystemExit(error.code if type(error.code) is int else 1)
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            if process_control := _sanitized_process_control(nested):
                return process_control
    return None


def _sanitized_backend_failure(error: BaseException) -> BaseException:
    if process_control := _sanitized_process_control(error):
        return process_control
    return RetrievalExecutionError()


def _backend_call(
    operation: Callable[[], Any],
    *,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Any:
    failure: BaseException | None = None
    try:
        return operation()
    except BaseException as error:  # noqa: BLE001 - the privacy boundary must classify BaseExceptionGroup.
        failure = error if isinstance(error, passthrough) else _sanitized_backend_failure(error)
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("backend boundary lost its failure")
    raise failure from None


async def _await_backend(awaitable: object) -> Any:
    failure: BaseException | None = None
    try:
        return await awaitable
    except BaseException as error:  # noqa: BLE001 - the privacy boundary must classify BaseExceptionGroup.
        failure = _sanitized_backend_failure(error)
    if failure is None:  # pragma: no cover - the except path always assigns it.
        raise RuntimeError("backend boundary lost its failure")
    raise failure from None


class _ArgumentError(ValueError):
    pass


class _HelpRequested(RuntimeError):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _ArgumentError

    def print_help(self, file: Any = None) -> None:
        del file
        raise _HelpRequested

    def exit(self, status: int = 0, _message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested
        raise _ArgumentError


@dataclass(frozen=True, slots=True)
class _CorpusChunk:
    document_id: str
    document_kind: DocumentKind
    path: str
    title: str
    tags: frozenset[str]
    facets: Mapping[str, object]
    chunk_index: int
    content: str
    source_content: str
    annotations_text: str

    @property
    def annotated(self) -> bool:
        return bool(self.annotations_text.strip())


class _SyntheticLexicalRetriever:
    """Small deterministic lexical adapter for the sibling synthetic corpus."""

    def __init__(self, chunks: Sequence[_CorpusChunk]) -> None:
        self._chunks = tuple(chunks)

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        ranked: list[tuple[int, _CorpusChunk]] = []
        query_tokens = _tokenize(query.text)
        for chunk in self._chunks:
            if not _matches_filters(chunk, query):
                continue
            searchable = _scope_content(chunk, query.scope)
            corpus_tokens = _tokenize(searchable)
            content_score = sum(corpus_tokens.count(token) for token in query_tokens)
            if content_score > 0:
                title_tokens = _tokenize(chunk.title)
                title_score = sum(title_tokens.count(token) for token in query_tokens)
                ranked.append((content_score * 100 + title_score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].document_id, item[1].chunk_index))
        candidates = ranked[: query.candidate_limit]
        hits = tuple(
            SearchHit(
                document_id=chunk.document_id,
                document_version=1,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=float(score),
                path=chunk.path,
                title=chunk.title,
                tags=tuple(sorted(chunk.tags)),
                document_kind=chunk.document_kind,
                metadata={"facets": dict(chunk.facets), "synthetic": True},
            )
            for score, chunk in candidates
        )
        return SearchResult(
            hits=hits,
            candidate_count=len(ranked),
            latency_ms=0.0,
            profile="lexical",
        )


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(value.casefold()))


def _scope_content(chunk: _CorpusChunk, scope: SearchScope) -> str:
    if scope is SearchScope.SOURCE:
        return chunk.source_content
    if scope is SearchScope.ANNOTATIONS:
        return chunk.annotations_text
    return f"{chunk.source_content}\n{chunk.annotations_text}"


def _matches_filters(chunk: _CorpusChunk, query: SearchQuery) -> bool:
    if query.area is SearchArea.WIKI and chunk.document_kind is not DocumentKind.WIKI:
        return False
    if query.area is SearchArea.SOURCES and chunk.document_kind is DocumentKind.WIKI:
        return False
    if query.document_kinds and chunk.document_kind not in query.document_kinds:
        return False
    if query.path_glob is not None and not _logical_glob_matches(query.path_glob, chunk.path):
        return False
    if query.tags and not set(query.tags).issubset(chunk.tags):
        return False
    if query.annotated_only and not chunk.annotated:
        return False
    if query.scope is SearchScope.SOURCE and not chunk.source_content.strip():
        return False
    if query.scope is SearchScope.ANNOTATIONS and not chunk.annotations_text.strip():
        return False
    return all(key in chunk.facets and chunk.facets[key] == value for key, value in query.facets.items())


def _logical_glob_matches(path_glob: str, path: str) -> bool:
    if path_glob == "/":
        return path.startswith("/")
    directory = path_glob.endswith("/") or (
        "*" not in path_glob and "." not in path_glob.rsplit("/", 1)[-1]
    )
    pattern = path_glob.rstrip("/") + "/*" if directory else path_glob
    expression: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        if pattern[index] == "*":
            while index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 1
            expression.append(".*")
        else:
            expression.append(re.escape(pattern[index]))
        index += 1
    expression.append("$")
    return re.fullmatch("".join(expression), path) is not None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate corpus JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_object(
    value: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("corpus value must be an object")
    if set(value) - allowed or required - set(value):
        raise ValueError("corpus object fields are invalid")
    return value


def _nonblank_string(value: object) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()) or len(normalized) > 4096:
        raise ValueError("corpus string is invalid")
    return normalized


def _text(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_CORPUS_TEXT_CHARS:
        raise ValueError("corpus text is invalid")
    return value


def _tags(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("corpus tags are invalid")
    normalized = []
    for item in value:
        if not isinstance(item, str) or not (tag := item.strip().casefold()) or len(tag) > 128:
            raise ValueError("corpus tags are invalid")
        normalized.append(tag)
    return frozenset(normalized)


def _facets(value: object) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping) or len(value) > 100:
        raise ValueError("corpus facets are invalid")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("corpus facets are invalid")
        if nested is not None and not isinstance(nested, (str, int, float, bool)):
            raise ValueError("corpus facets are invalid")
        if isinstance(nested, float) and not isfinite(nested):
            raise ValueError("corpus facets are invalid")
        result[key] = nested
    return MappingProxyType(result)


def _parse_corpus_line(raw_line: bytes) -> tuple[_CorpusChunk, ...]:
    if len(raw_line) > MAX_CORPUS_LINE_BYTES or not raw_line.strip():
        raise ValueError("corpus line is invalid")
    try:
        raw = json.loads(
            raw_line.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError("corpus line is invalid") from exc
    document = _strict_object(raw, allowed=_DOCUMENT_FIELDS, required=_DOCUMENT_REQUIRED_FIELDS)
    if type(document["schema_version"]) is not int or document["schema_version"] != EVALUATION_SCHEMA_VERSION:
        raise ValueError("corpus schema is unsupported")
    document_id = _nonblank_string(document["document_id"])
    try:
        document_kind = DocumentKind(document["document_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("corpus document kind is invalid") from exc
    path = _nonblank_string(document["path"])
    if not path.startswith("/") or "\x00" in path or any(part == ".." for part in path.split("/")):
        raise ValueError("corpus path is invalid")
    title = _nonblank_string(document["title"])
    tags = _tags(document.get("tags"))
    facets = _facets(document.get("facets"))
    raw_chunks = document["chunks"]
    if not isinstance(raw_chunks, list) or not raw_chunks or len(raw_chunks) > MAX_CORPUS_CHUNKS:
        raise ValueError("corpus chunks are invalid")

    chunks: list[_CorpusChunk] = []
    seen_indices: set[int] = set()
    for raw_chunk in raw_chunks:
        chunk = _strict_object(raw_chunk, allowed=_CHUNK_FIELDS, required=_CHUNK_REQUIRED_FIELDS)
        chunk_index = chunk["chunk_index"]
        if type(chunk_index) is not int or chunk_index < 0 or chunk_index in seen_indices:
            raise ValueError("corpus chunk index is invalid")
        seen_indices.add(chunk_index)
        content = _text(chunk["content"])
        source_content = _text(chunk.get("source_content", content))
        annotations_text = _text(chunk.get("annotations_text", ""))
        if content != "\n".join(part for part in (source_content, annotations_text) if part):
            raise ValueError("corpus chunk content does not match its retrieval fields")
        chunks.append(
            _CorpusChunk(
                document_id=document_id,
                document_kind=document_kind,
                path=path,
                title=title,
                tags=tags,
                facets=facets,
                chunk_index=chunk_index,
                content=content,
                source_content=source_content,
                annotations_text=annotations_text,
            )
        )
    return tuple(chunks)


def _open_corpus(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("corpus cannot be read") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_CORPUS_BYTES:
            raise ValueError("corpus file is invalid")
        return os.fdopen(descriptor, "rb")
    except (OSError, ValueError):
        with suppress(OSError):
            os.close(descriptor)
        raise


def _register_corpus_document(
    parsed: Sequence[_CorpusChunk],
    *,
    chunks: list[_CorpusChunk],
    identities: set[tuple[str, int]],
    document_ids: set[str],
) -> None:
    document_id = parsed[0].document_id
    if document_id in document_ids:
        raise ValueError("corpus document id is duplicated")
    document_ids.add(document_id)
    if len(document_ids) > MAX_CORPUS_DOCUMENTS:
        raise ValueError("corpus has too many documents")
    for chunk in parsed:
        identity = (chunk.document_id, chunk.chunk_index)
        if identity in identities:
            raise ValueError("corpus chunk identity is duplicated")
        identities.add(identity)
        chunks.append(chunk)
        if len(chunks) > MAX_CORPUS_CHUNKS:
            raise ValueError("corpus has too many chunks")


def _load_corpus(path: Path) -> tuple[_CorpusChunk, ...]:
    chunks: list[_CorpusChunk] = []
    identities: set[tuple[str, int]] = set()
    document_ids: set[str] = set()
    total_bytes = 0
    try:
        with _open_corpus(path) as corpus:
            while raw_line := corpus.readline(MAX_CORPUS_LINE_BYTES + 1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_CORPUS_BYTES:
                    raise ValueError("corpus is too large")
                _register_corpus_document(
                    _parse_corpus_line(raw_line),
                    chunks=chunks,
                    identities=identities,
                    document_ids=document_ids,
                )
    except OSError as exc:
        raise ValueError("corpus cannot be read") from exc
    if not chunks:
        raise ValueError("corpus must not be empty")
    return tuple(chunks)


def configured_hosted_hybrid_retriever() -> object:
    """Return the hosted hybrid boundary once Task 6-9 configuration exists."""

    raise HybridConfigurationUnavailable


def _default_retriever_factory(profile: str, dataset_path: Path) -> object:
    if profile == "lexical":
        try:
            chunks = _load_corpus(dataset_path.with_name("corpus.jsonl"))
        except (OSError, ValueError):
            raise EvaluationDatasetError from None
        return _SyntheticLexicalRetriever(chunks)
    if profile == "hybrid":
        return configured_hosted_hybrid_retriever()
    raise RetrievalContractError


def _parser() -> _SafeArgumentParser:
    parser = _SafeArgumentParser(prog="retrieval_eval")
    parser.add_argument("--dataset", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", choices=("lexical", "hybrid"))
    selection.add_argument("--compare", action="store_true")
    parser.add_argument("--require-promotion-gate", action="store_true")
    parser.add_argument("--output-json")
    return parser


async def _retrieve_case(profile: str, retriever: object, query: SearchQuery) -> SearchResult:
    retrieve = _backend_call(lambda: getattr(retriever, "retrieve", None))
    if not callable(retrieve):
        raise RetrievalContractError
    pending = _backend_call(lambda: retrieve(query))
    if not inspect.isawaitable(pending):
        raise RetrievalContractError
    result = await _await_backend(pending)
    if type(result) is not SearchResult or result.profile != profile:
        raise RetrievalContractError
    if any(type(hit) is not SearchHit for hit in result.hits):
        raise RetrievalContractError
    if result.candidate_count < len(result.hits) or len(result.hits) > query.candidate_limit:
        raise RetrievalContractError
    identities = {(hit.document_id, hit.chunk_index) for hit in result.hits}
    if len(identities) != len(result.hits):
        raise RetrievalContractError
    return result


async def _evaluate_profile(profile: str, retriever: object, cases: Sequence[object]) -> EvaluationReport:
    runs: list[EvaluationRun] = []
    for case in cases:
        result = await _retrieve_case(profile, retriever, case.query)
        runs.append(
            EvaluationRun(
                case_id=case.case_id,
                ranking=tuple(RankedResult(hit.document_id, hit.chunk_index) for hit in result.hits),
                latency_ms=result.latency_ms,
            )
        )
    return evaluate_rankings(cases, runs)


def _build_retriever(factory: RetrieverFactory, profile: str, dataset_path: Path) -> object:
    factory_call = _backend_call(
        lambda: getattr(factory, "__call__", None)  # noqa: B004 - attribute access is a guarded backend boundary.
    )
    if not callable(factory_call):
        raise RetrievalContractError
    retriever = _backend_call(
        lambda: factory_call(profile, dataset_path),
        passthrough=_SAFE_BACKEND_EXCEPTIONS,
    )
    retrieve = _backend_call(lambda: getattr(retriever, "retrieve", None))
    if not callable(retrieve):
        raise RetrievalContractError
    return retriever


def _run_async(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _backend_call(
            lambda: asyncio.run(factory()),
            passthrough=_SAFE_BACKEND_EXCEPTIONS,
        )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrieval-eval") as executor:
        return _backend_call(
            lambda: executor.submit(lambda: asyncio.run(factory())).result(),
            passthrough=_SAFE_BACKEND_EXCEPTIONS,
        )


def _stable_number(value: float) -> float:
    rounded = round(value, 12)
    return 0.0 if rounded == 0 else rounded


def _metrics_payload(report: EvaluationReport) -> dict[str, object]:
    return {
        "filtered_result_count": report.filtered_result_count,
        "latency_p50_ms": _stable_number(report.latency_p50_ms),
        "latency_p95_ms": _stable_number(report.latency_p95_ms),
        "mrr": _stable_number(report.mrr),
        "ndcg_at_10": _stable_number(report.ndcg_at_10),
        "recall_at_10": _stable_number(report.recall_at_10),
        "recall_at_20": _stable_number(report.recall_at_20),
        "recall_at_5": _stable_number(report.recall_at_5),
    }


def _profile_payload(profile: str, report: EvaluationReport) -> dict[str, object]:
    return {
        "case_count": report.case_count,
        "metrics": _metrics_payload(report),
        "profile": profile,
    }


def _base_payload(cases: Sequence[object], *, profile: str) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "dataset_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_dataset_digest": evaluation_dataset_digest(cases),
        "profile": profile,
        "schema_version": REPORT_SCHEMA_VERSION,
    }


def _single_report(cases: Sequence[object], profile: str, report: EvaluationReport) -> dict[str, object]:
    return {**_base_payload(cases, profile=profile), "metrics": _metrics_payload(report)}


def _compare_report(
    cases: Sequence[object],
    lexical: EvaluationReport,
    hybrid: EvaluationReport,
) -> tuple[dict[str, object], bool]:
    decision = promotion_decision(lexical, hybrid)
    payload = {
        **_base_payload(cases, profile="compare"),
        "profiles": {
            "hybrid": _profile_payload("hybrid", hybrid),
            "lexical": _profile_payload("lexical", lexical),
        },
        "promotion": {
            "eligible": decision.eligible,
            "latency_ratio": None if decision.latency_ratio is None else _stable_number(decision.latency_ratio),
            "reason": decision.reason,
            "recall_ratio": None if decision.recall_ratio is None else _stable_number(decision.recall_ratio),
        },
    }
    return payload, decision.eligible


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _raw_output_components(path: str | os.PathLike[str]) -> tuple[bool, tuple[str, ...], str]:
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path or raw_path.endswith("/"):
        raise OutputWriteError
    components = tuple(component for component in raw_path.split("/") if component)
    if not components or any(component in {".", ".."} for component in components):
        raise OutputWriteError
    return raw_path.startswith("/"), components[:-1], components[-1]


def _open_output_parent(path: str | os.PathLike[str]) -> tuple[int, int, str]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    cloexec = getattr(os, "O_CLOEXEC", None)
    if nofollow is None or directory is None or cloexec is None:
        raise OutputWriteError
    absolute, parent_components, basename = _raw_output_components(path)
    descriptor = -1
    try:
        flags = os.O_RDONLY | directory | nofollow | cloexec
        descriptor = os.open("/" if absolute else ".", flags)
        for component in parent_components:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            with suppress(OSError):
                os.close(descriptor)
            descriptor = next_descriptor
        result = (descriptor, nofollow, basename)
        descriptor = -1
        return result
    except OSError as exc:
        raise OutputWriteError from exc
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)


def _create_output_temporary(parent_descriptor: int, basename: str, nofollow: int) -> tuple[int, str]:
    create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow
    for _attempt in range(10):
        temporary_name = f".{basename}.{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                create_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            return descriptor, temporary_name
        except FileExistsError:
            continue
    raise OutputWriteError


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OutputWriteError
        remaining = remaining[written:]


def _write_output(path: str | os.PathLike[str], content: bytes) -> None:
    parent_descriptor = -1
    temporary_descriptor = -1
    temporary_name: str | None = None
    try:
        parent_descriptor, nofollow, basename = _open_output_parent(path)
        try:
            current = os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise OutputWriteError
        temporary_descriptor, temporary_name = _create_output_temporary(
            parent_descriptor,
            basename,
            nofollow,
        )
        _write_all(temporary_descriptor, content)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.replace(
            temporary_name,
            basename,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    except OutputWriteError:
        raise
    except OSError as exc:
        raise OutputWriteError from exc
    finally:
        if temporary_descriptor >= 0:
            with suppress(OSError):
                os.close(temporary_descriptor)
        if temporary_name is not None and parent_descriptor >= 0:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        if parent_descriptor >= 0:
            with suppress(OSError):
                os.close(parent_descriptor)


def _discard_failed_stream_buffer(buffer: Any) -> None:
    with suppress(AttributeError, OSError, TypeError, ValueError):
        buffer.seek(0)
        buffer.truncate(0)


def _silence_failed_stream(buffer: Any) -> None:
    try:
        descriptor = buffer.fileno()
    except (AttributeError, OSError, TypeError, ValueError):
        _discard_failed_stream_buffer(buffer)
        return

    null_descriptor = -1
    try:
        null_descriptor = os.open(
            os.devnull,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        )
        os.dup2(null_descriptor, descriptor, inheritable=False)
    except (OSError, TypeError, ValueError):
        _discard_failed_stream_buffer(buffer)
        return
    finally:
        if null_descriptor >= 0:
            with suppress(OSError):
                os.close(null_descriptor)

    with suppress(OSError, ValueError):
        buffer.flush()


def _emit_stream(stream: Any, content: bytes) -> bool:
    buffer = stream
    remaining = memoryview(content)
    try:
        buffer = getattr(stream, "buffer", stream)
        while remaining:
            written = buffer.write(remaining)
            if type(written) is not int or written <= 0 or written > len(remaining):
                raise OSError
            remaining = remaining[written:]
        buffer.flush()
    except (AttributeError, OSError, TypeError, ValueError):
        _silence_failed_stream(buffer)
        return False
    return True


def _emit_error(category: str, code: str) -> None:
    _emit_stream(sys.stderr, _json_bytes({"error": {"category": category, "code": code}}))


def _parse_cli_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace | None, int]:
    parser = _parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        if args.require_promotion_gate and not args.compare:
            raise _ArgumentError
    except _HelpRequested:
        emitted = _emit_stream(sys.stdout, parser.format_help().encode("utf-8"))
        return None, 0 if emitted else 4
    except (_ArgumentError, SystemExit):
        _emit_error("arguments", "invalid_arguments")
        return None, 2
    return args, 0


def _load_cli_cases(dataset_path: Path) -> tuple[Sequence[object] | None, int]:
    try:
        cases = load_cases(dataset_path)
    except (OSError, TypeError, ValueError):
        _emit_error("dataset", "dataset_invalid")
        return None, 2
    return cases, 0


def _evaluate_request(
    args: argparse.Namespace,
    cases: Sequence[object],
    factory: RetrieverFactory,
    dataset_path: Path,
) -> tuple[dict[str, object], int]:
    if not args.compare:
        profile = args.profile
        retriever = _build_retriever(factory, profile, dataset_path)
        report = _run_async(lambda: _evaluate_profile(profile, retriever, cases))
        return _single_report(cases, profile, report), 0

    lexical_retriever = _build_retriever(factory, "lexical", dataset_path)
    hybrid_retriever = _build_retriever(factory, "hybrid", dataset_path)

    async def evaluate_both() -> tuple[EvaluationReport, EvaluationReport]:
        lexical = await _evaluate_profile("lexical", lexical_retriever, cases)
        hybrid = await _evaluate_profile("hybrid", hybrid_retriever, cases)
        return lexical, hybrid

    lexical_report, hybrid_report = _run_async(evaluate_both)
    payload, eligible = _compare_report(cases, lexical_report, hybrid_report)
    exit_code = 0 if eligible or not args.require_promotion_gate else 3
    return payload, exit_code


def _safe_evaluate_request(
    args: argparse.Namespace,
    cases: Sequence[object],
    factory: RetrieverFactory,
    dataset_path: Path,
) -> tuple[dict[str, object] | None, int]:
    try:
        return _evaluate_request(args, cases, factory, dataset_path)
    except EvaluationDatasetError:
        _emit_error("dataset", "dataset_invalid")
        return None, 2
    except HybridConfigurationUnavailable:
        _emit_error("configuration", "hybrid_unavailable")
        return None, 2
    except RetrievalContractError:
        _emit_error("retrieval", "retrieval_contract_invalid")
        return None, 2
    except RetrievalExecutionError:
        _emit_error("retrieval", "retrieval_failed")
        return None, 2
    except (TypeError, ValueError):
        _emit_error("retrieval", "retrieval_contract_invalid")
        return None, 2
    except Exception:  # noqa: BLE001 - final privacy boundary intentionally discards all exception details.
        _emit_error("retrieval", "retrieval_failed")
        return None, 2


def _emit_report(payload: Mapping[str, object], output_json: str | None, exit_code: int) -> int:
    encoded = _json_bytes(payload)
    if output_json is not None:
        try:
            _write_output(output_json, encoded)
        except OutputWriteError:
            _emit_error("output", "output_write_failed")
            return 4
    if not _emit_stream(sys.stdout, encoded):
        return 4
    return exit_code


def main(argv: Sequence[str] | None = None, retriever_factory: RetrieverFactory | None = None) -> int:
    """Run retrieval evaluation without exposing query or backend content."""

    args, early_exit = _parse_cli_args(argv)
    if args is None:
        return early_exit
    dataset_path = Path(args.dataset)
    cases, early_exit = _load_cli_cases(dataset_path)
    if cases is None:
        return early_exit
    factory = _default_retriever_factory if retriever_factory is None else retriever_factory
    payload, exit_code = _safe_evaluate_request(args, cases, factory, dataset_path)
    if payload is None:
        return exit_code
    return _emit_report(payload, args.output_json, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
