"""Strict retrieval-evaluation fixtures, exact metrics, and promotion gates."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from itertools import islice
from math import ceil, isfinite, log2
from numbers import Real
from os import PathLike
from pathlib import Path
from stat import S_ISREG
from sys import float_info
from typing import Any, BinaryIO

from .search import SearchArea, SearchQuery, SearchScope

EVALUATION_SCHEMA_VERSION = 1
MAX_DATASET_BYTES = 8 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
MAX_CASES = 10_000
MAX_RELEVANCE_PER_CASE = 10_000
MAX_RANKING_LENGTH = 10_000

_CASE_FIELDS = frozenset({"schema_version", "case_id", "query", "relevance"})
_QUERY_FIELDS = frozenset(
    {
        "text",
        "limit",
        "area",
        "scope",
        "facets",
        "candidate_limit",
        "path_glob",
        "tags",
        "document_kinds",
        "annotated_only",
    }
)
_RELEVANCE_FIELDS = frozenset({"document_id", "chunk_index", "grade"})


def _nonblank_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{name} must be a nonblank string")
    if len(normalized) > 1024:
        raise ValueError(f"{name} is too long")
    return normalized


def _chunk_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("chunk_index must be a non-negative integer or null")
    return value


def _finite_non_negative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _bounded_tuple(value: object, *, limit: int, too_large: str) -> tuple[Any, ...]:
    try:
        items = tuple(islice(iter(value), limit + 1))
    except TypeError as exc:
        raise ValueError("value must be iterable") from exc
    if len(items) > limit:
        raise ValueError(too_large)
    return items


def _strict_fields(
    value: object,
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{label} field names must be strings")
    if unknown := sorted(keys - allowed):
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing := sorted(required - keys):
        prefix = "" if label == "case" else f"{label} has "
        raise ValueError(f"{prefix}missing fields: {', '.join(missing)}")
    return value


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """One positive graded judgment at document or chunk granularity."""

    document_id: str
    grade: int
    chunk_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _nonblank_string("document_id", self.document_id))
        if isinstance(self.grade, bool) or not isinstance(self.grade, int) or self.grade <= 0:
            raise ValueError("grade must be a positive integer")
        object.__setattr__(self, "chunk_index", _chunk_index(self.chunk_index))

    @property
    def identity(self) -> tuple[str, int | None]:
        return (self.document_id, self.chunk_index)


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A content-free ranked retrieval identity used by evaluation runs."""

    document_id: str
    chunk_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _nonblank_string("document_id", self.document_id))
        object.__setattr__(self, "chunk_index", _chunk_index(self.chunk_index))

    @property
    def identity(self) -> tuple[str, int | None]:
        return (self.document_id, self.chunk_index)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A validated v1 query and its non-overlapping relevance judgments."""

    schema_version: int
    case_id: str
    query: SearchQuery
    relevance: tuple[RelevanceJudgment, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        object.__setattr__(self, "case_id", _nonblank_string("case_id", self.case_id))
        if not isinstance(self.query, SearchQuery):
            raise ValueError("query must be a SearchQuery")
        relevance = _bounded_tuple(
            self.relevance,
            limit=MAX_RELEVANCE_PER_CASE,
            too_large="too many relevance judgments",
        )
        if not relevance:
            raise ValueError("relevance must not be empty")
        if any(not isinstance(item, RelevanceJudgment) for item in relevance):
            raise ValueError("relevance must contain RelevanceJudgment values")
        _validate_relevance_overlap(relevance)
        object.__setattr__(self, "relevance", relevance)

    @classmethod
    def build(
        cls,
        *,
        schema_version: int,
        case_id: str,
        query: Mapping[str, Any],
        relevance: Sequence[RelevanceJudgment],
    ) -> EvalCase:
        query_fields = _strict_fields(
            query,
            label="query",
            allowed=_QUERY_FIELDS,
            required=frozenset({"text"}),
        )
        try:
            search_query = SearchQuery.build(**query_fields)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid query: {exc}") from exc
        return cls(schema_version, case_id, search_query, relevance)


def _validate_relevance_overlap(relevance: Sequence[RelevanceJudgment]) -> None:
    identities: set[tuple[str, int | None]] = set()
    document_level: set[str] = set()
    chunk_level: set[str] = set()
    for judgment in relevance:
        if judgment.identity in identities:
            raise ValueError(f"duplicate relevance judgment: {judgment.document_id}")
        identities.add(judgment.identity)
        if judgment.chunk_index is None:
            if judgment.document_id in chunk_level:
                raise ValueError(f"document-level and chunk-level relevance overlap: {judgment.document_id}")
            document_level.add(judgment.document_id)
        else:
            if judgment.document_id in document_level:
                raise ValueError(f"document-level and chunk-level relevance overlap: {judgment.document_id}")
            chunk_level.add(judgment.document_id)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One ranked result list and observed latency for exactly one case."""

    case_id: str
    ranking: tuple[RankedResult, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonblank_string("case_id", self.case_id))
        ranking = _bounded_tuple(
            self.ranking,
            limit=MAX_RANKING_LENGTH,
            too_large="ranking is too large",
        )
        if any(not isinstance(result, RankedResult) for result in ranking):
            raise ValueError("ranking must contain RankedResult values")
        seen: set[tuple[str, int | None]] = set()
        for result in ranking:
            if result.identity in seen:
                raise ValueError(f"duplicate ranked result: {result.document_id}")
            seen.add(result.identity)
        object.__setattr__(self, "ranking", ranking)
        object.__setattr__(
            self,
            "latency_ms",
            _finite_non_negative("latency_ms", self.latency_ms),
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Macro-averaged retrieval metrics and nearest-rank latency summaries."""

    case_count: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    filtered_result_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    dataset_digest: str | None = None
    _recall_at_10_exact: Fraction = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.case_count, bool) or not isinstance(self.case_count, int) or self.case_count <= 0:
            raise ValueError("case_count must be a positive integer")
        for name in ("recall_at_5", "recall_at_10", "recall_at_20", "mrr", "ndcg_at_10"):
            normalized = _finite_non_negative(name, getattr(self, name))
            if normalized > 1:
                raise ValueError(f"{name} must not exceed 1")
            object.__setattr__(self, name, normalized)
        if not self.recall_at_5 <= self.recall_at_10 <= self.recall_at_20:
            raise ValueError("recall metrics must be monotonic")
        if (
            isinstance(self.filtered_result_count, bool)
            or not isinstance(self.filtered_result_count, int)
            or self.filtered_result_count < 0
        ):
            raise ValueError("filtered_result_count must be a non-negative integer")
        object.__setattr__(
            self,
            "latency_p50_ms",
            _finite_non_negative("latency_p50_ms", self.latency_p50_ms),
        )
        object.__setattr__(
            self,
            "latency_p95_ms",
            _finite_non_negative("latency_p95_ms", self.latency_p95_ms),
        )
        if self.latency_p50_ms > self.latency_p95_ms:
            raise ValueError("latency p50 must not exceed p95")
        if self.dataset_digest is not None and not _is_sha256_digest(self.dataset_digest):
            raise ValueError("dataset_digest must be a lowercase SHA-256 digest or None")
        object.__setattr__(
            self,
            "_recall_at_10_exact",
            Fraction(*self.recall_at_10.as_integer_ratio()),
        )

    @property
    def recall_at_10_exact(self) -> Fraction:
        return self._recall_at_10_exact


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Measured decision for promoting hybrid retrieval over lexical retrieval."""

    eligible: bool
    reason: str
    recall_ratio: float | None = None
    latency_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be a boolean")
        object.__setattr__(self, "reason", _nonblank_string("reason", self.reason))
        for name in ("recall_ratio", "latency_ratio"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_non_negative(name, value))
        has_ratios = self.recall_ratio is not None and self.latency_ratio is not None
        gates_pass = bool(has_ratios and self.recall_ratio >= 1.10 and self.latency_ratio <= 2.0)
        gate_failure_is_visible = bool(has_ratios and (self.recall_ratio <= 1.10 or self.latency_ratio >= 2.0))
        consistent = (
            (self.eligible and self.reason == "eligible" and gates_pass)
            or (not self.eligible and self.reason == "gate_failed" and gate_failure_is_visible)
            or (
                not self.eligible
                and self.reason == "baseline_recall_zero"
                and self.recall_ratio is None
                and self.latency_ratio is None
            )
        )
        if not consistent:
            raise ValueError("promotion decision fields are inconsistent")


def _is_sha256_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_relevance(value: object) -> tuple[RelevanceJudgment, ...]:
    if not isinstance(value, list):
        raise ValueError("relevance must be an array")
    if not value:
        raise ValueError("relevance must not be empty")
    if len(value) > MAX_RELEVANCE_PER_CASE:
        raise ValueError("too many relevance judgments")
    judgments: list[RelevanceJudgment] = []
    for raw_judgment in value:
        judgment = _strict_fields(
            raw_judgment,
            label="relevance",
            allowed=_RELEVANCE_FIELDS,
            required=frozenset({"document_id", "grade"}),
        )
        judgments.append(
            RelevanceJudgment(
                document_id=judgment["document_id"],
                grade=judgment["grade"],
                chunk_index=judgment.get("chunk_index"),
            )
        )
    return tuple(judgments)


def _parse_case(value: object) -> EvalCase:
    raw_case = _strict_fields(
        value,
        label="case",
        allowed=_CASE_FIELDS,
        required=_CASE_FIELDS,
    )
    schema_version = raw_case["schema_version"]
    if type(schema_version) is not int or schema_version != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    return EvalCase.build(
        schema_version=schema_version,
        case_id=raw_case["case_id"],
        query=raw_case["query"],
        relevance=_parse_relevance(raw_case["relevance"]),
    )


def _open_dataset(dataset_path: Path) -> BinaryIO:
    try:
        descriptor = os.open(dataset_path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise ValueError("evaluation dataset cannot be read") from exc
    try:
        dataset_stat = os.fstat(descriptor)
        if not S_ISREG(dataset_stat.st_mode):
            raise ValueError("evaluation dataset must be a regular file")
        if dataset_stat.st_size > MAX_DATASET_BYTES:
            raise ValueError("evaluation dataset exceeds the size limit")
        return os.fdopen(descriptor, "rb")
    except (OSError, ValueError):
        os.close(descriptor)
        raise


def _parse_jsonl_line(raw_line: bytes, line_number: int) -> EvalCase:
    if len(raw_line) > MAX_LINE_BYTES:
        raise ValueError(f"line exceeds size limit at line {line_number}")
    if not raw_line.strip():
        raise ValueError(f"blank JSONL line at line {line_number}")
    try:
        value = json.loads(raw_line.decode("utf-8"), object_pairs_hook=_unique_object)
        return _parse_case(value)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid evaluation case at line {line_number}: {exc}") from exc


def load_cases(path: str | PathLike[str]) -> tuple[EvalCase, ...]:
    """Load and strictly validate a bounded UTF-8 JSONL evaluation dataset."""

    dataset_path = Path(path)

    cases: list[EvalCase] = []
    case_ids: set[str] = set()
    total_bytes = 0
    line_number = 0
    try:
        with _open_dataset(dataset_path) as dataset:
            while raw_line := dataset.readline(MAX_LINE_BYTES + 1):
                line_number += 1
                total_bytes += len(raw_line)
                if total_bytes > MAX_DATASET_BYTES:
                    raise ValueError("evaluation dataset exceeds the size limit")
                case = _parse_jsonl_line(raw_line, line_number)
                if case.case_id in case_ids:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
                case_ids.add(case.case_id)
                cases.append(case)
                if len(cases) > MAX_CASES:
                    raise ValueError("evaluation dataset has too many cases")
    except OSError as exc:
        raise ValueError("evaluation dataset cannot be read") from exc
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    return tuple(cases)


def _is_filtered(query: SearchQuery) -> bool:
    return bool(
        query.area is not SearchArea.ALL
        or query.scope is not SearchScope.ALL
        or query.facets
        or query.path_glob is not None
        or query.tags
        or query.document_kinds
        or query.annotated_only
    )


def _canonical_json_value(value: object) -> object:
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", "negative" if value < 0 else "nonnegative", format(abs(value), "x")]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Mapping):
        items = [[_canonical_json_value(key), _canonical_json_value(nested)] for key, nested in value.items()]
        items.sort(key=lambda item: _canonical_sort_key(item[0]))
        return ["mapping", items]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_json_value(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_canonical_json_value(item) for item in value]]
    if isinstance(value, frozenset):
        values = [_canonical_json_value(item) for item in value]
        return ["frozenset", sorted(values, key=_canonical_sort_key)]
    raise ValueError(f"unsupported canonical evaluation value: {type(value).__name__}")


def _canonical_sort_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def evaluation_dataset_digest(cases: Sequence[EvalCase]) -> str:
    """Return an order-independent SHA-256 identity for complete case definitions."""

    case_values = _bounded_tuple(
        cases,
        limit=MAX_CASES,
        too_large="evaluation dataset has too many cases",
    )
    if not case_values:
        raise ValueError("evaluation dataset must contain at least one case")
    if any(not isinstance(case, EvalCase) for case in case_values):
        raise ValueError("cases must contain EvalCase values")
    if len({case.case_id for case in case_values}) != len(case_values):
        raise ValueError("evaluation dataset has duplicate case ids")

    payload = []
    for case in sorted(case_values, key=lambda item: item.case_id):
        query = case.query
        payload.append(
            {
                "schema_version": case.schema_version,
                "case_id": case.case_id,
                "query": {
                    "text": query.text,
                    "limit": query.limit,
                    "candidate_limit": query.candidate_limit,
                    "area": query.area.value,
                    "scope": query.scope.value,
                    "facets": query.facets,
                    "path_glob": query.path_glob,
                    "tags": list(query.tags),
                    "document_kinds": [kind.value for kind in query.document_kinds],
                    "annotated_only": query.annotated_only,
                },
                "relevance": [
                    {
                        "document_id": judgment.document_id,
                        "chunk_index": judgment.chunk_index,
                        "grade": judgment.grade,
                    }
                    for judgment in sorted(
                        case.relevance,
                        key=lambda item: (
                            item.document_id,
                            -1 if item.chunk_index is None else item.chunk_index,
                        ),
                    )
                ],
            }
        )
    encoded = json.dumps(
        _canonical_json_value(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _case_metrics(
    case: EvalCase,
    run: EvaluationRun,
) -> tuple[Fraction, Fraction, Fraction, float, float]:
    document_judgments = {item.document_id: item for item in case.relevance if item.chunk_index is None}
    chunk_judgments = {item.identity: item for item in case.relevance if item.chunk_index is not None}
    matched: set[tuple[str, int | None]] = set()
    matches: list[tuple[int, RelevanceJudgment]] = []
    for rank, result in enumerate(run.ranking, start=1):
        judgment = document_judgments.get(result.document_id)
        if judgment is None:
            judgment = chunk_judgments.get(result.identity)
        if judgment is not None and judgment.identity not in matched:
            matched.add(judgment.identity)
            matches.append((rank, judgment))

    relevant_count = len(case.relevance)
    exact_recalls = tuple(
        Fraction(sum(rank <= cutoff for rank, _judgment in matches), relevant_count) for cutoff in (5, 10, 20)
    )
    mrr = 0.0 if not matches else 1.0 / matches[0][0]
    max_grade = max(item.grade for item in case.relevance)
    dcg = sum(_scaled_gain(judgment.grade, max_grade) / log2(rank + 1) for rank, judgment in matches if rank <= 10)
    ideal_grades = sorted((item.grade for item in case.relevance), reverse=True)[:10]
    ideal_dcg = sum(_scaled_gain(grade, max_grade) / log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    ndcg = dcg / ideal_dcg
    return (*exact_recalls, mrr, ndcg)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile * len(ordered)) - 1)]


def _scaled_gain(grade: int, max_grade: int) -> float:
    """Return ``(2**grade - 1) * 2**-max_grade`` without huge powers."""

    scaled_unit = 0.0 if max_grade > 1074 else 2.0 ** (-max_grade)
    delta = grade - max_grade
    scaled_power = 0.0 if delta < -1074 else 2.0**delta
    return scaled_power - scaled_unit


def _index_cases_and_runs(
    cases: Sequence[EvalCase],
    runs: Sequence[EvaluationRun],
) -> tuple[dict[str, EvalCase], dict[str, EvaluationRun]]:
    cases_by_id: dict[str, EvalCase] = {}
    for case in cases:
        if case.case_id in cases_by_id:
            raise ValueError(f"duplicate evaluation case: {case.case_id}")
        cases_by_id[case.case_id] = case
    runs_by_id: dict[str, EvaluationRun] = {}
    for run in runs:
        if run.case_id not in cases_by_id:
            raise ValueError(f"unknown case_id in evaluation run: {run.case_id}")
        if run.case_id in runs_by_id:
            raise ValueError(f"duplicate evaluation run: {run.case_id}")
        runs_by_id[run.case_id] = run
    if runs_by_id.keys() != cases_by_id.keys():
        raise ValueError("exactly one run per case is required")
    return cases_by_id, runs_by_id


def _with_exact_recall(report: EvaluationReport, exact_recall: Fraction) -> EvaluationReport:
    if float(exact_recall) != report.recall_at_10:
        raise RuntimeError("exact recall must round to the public recall_at_10")
    object.__setattr__(report, "_recall_at_10_exact", exact_recall)
    return report


def evaluate_rankings(
    cases: Sequence[EvalCase],
    runs: Sequence[EvaluationRun],
) -> EvaluationReport:
    """Evaluate exactly one ranked run per case using macro-averaged IR metrics."""

    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise ValueError("cases must be a sequence")
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise ValueError("runs must be a sequence")
    case_values = tuple(cases)
    run_values = tuple(runs)
    if not case_values:
        raise ValueError("at least one evaluation case is required")
    if len(case_values) > MAX_CASES or len(run_values) > MAX_CASES:
        raise ValueError("evaluation input is too large")
    if any(not isinstance(case, EvalCase) for case in case_values):
        raise ValueError("cases must contain EvalCase values")
    if any(not isinstance(run, EvaluationRun) for run in run_values):
        raise ValueError("runs must contain EvaluationRun values")

    _cases_by_id, runs_by_id = _index_cases_and_runs(case_values, run_values)

    per_case = [_case_metrics(case, runs_by_id[case.case_id]) for case in case_values]
    count = len(case_values)
    exact_recalls = [sum((metrics[index] for metrics in per_case), start=Fraction()) / count for index in range(3)]
    averages = [
        *(float(recall) for recall in exact_recalls),
        *(sum(metrics[index] for metrics in per_case) / count for index in range(3, 5)),
    ]
    latencies = [runs_by_id[case.case_id].latency_ms for case in case_values]
    return _with_exact_recall(
        EvaluationReport(
            case_count=count,
            recall_at_5=averages[0],
            recall_at_10=averages[1],
            recall_at_20=averages[2],
            mrr=averages[3],
            ndcg_at_10=averages[4],
            filtered_result_count=sum(
                len(runs_by_id[case.case_id].ranking) for case in case_values if _is_filtered(case.query)
            ),
            latency_p50_ms=_nearest_rank(latencies, 0.50),
            latency_p95_ms=_nearest_rank(latencies, 0.95),
            dataset_digest=evaluation_dataset_digest(case_values),
        ),
        exact_recalls[1],
    )


def promotion_decision(
    lexical: EvaluationReport,
    hybrid: EvaluationReport,
) -> PromotionDecision:
    """Apply the inclusive quality and latency gates for hybrid promotion."""

    if not isinstance(lexical, EvaluationReport) or not isinstance(hybrid, EvaluationReport):
        raise ValueError("promotion inputs must be EvaluationReport values")
    if lexical.dataset_digest is None or hybrid.dataset_digest is None:
        raise ValueError("promotion requires an identified evaluation cohort")
    if lexical.case_count != hybrid.case_count or lexical.dataset_digest != hybrid.dataset_digest:
        raise ValueError("promotion requires the same evaluation cohort")
    if lexical.recall_at_10 <= 0:
        return PromotionDecision(False, "baseline_recall_zero")
    latency_baseline = max(lexical.latency_p95_ms, 0.001)
    recall_ratio = _bounded_fraction_ratio(hybrid.recall_at_10_exact / lexical.recall_at_10_exact)
    latency_ratio = _bounded_ratio(hybrid.latency_p95_ms, latency_baseline)
    lexical_recall = lexical.recall_at_10_exact
    hybrid_recall = hybrid.recall_at_10_exact
    if lexical_recall is None or hybrid_recall is None:  # pragma: no cover - normalized.
        raise RuntimeError("evaluation report exact recall is missing")
    quality_gate = hybrid_recall * 10 >= lexical_recall * 11
    hybrid_latency = Fraction(*hybrid.latency_p95_ms.as_integer_ratio())
    baseline_latency = Fraction(*latency_baseline.as_integer_ratio())
    latency_gate = hybrid_latency <= baseline_latency * 2
    eligible = quality_gate and latency_gate
    return PromotionDecision(
        eligible=eligible,
        reason="eligible" if eligible else "gate_failed",
        recall_ratio=recall_ratio,
        latency_ratio=latency_ratio,
    )


def _bounded_ratio(numerator: float, denominator: float) -> float:
    ratio = numerator / denominator
    return ratio if isfinite(ratio) else float_info.max


def _bounded_fraction_ratio(ratio: Fraction) -> float:
    try:
        value = float(ratio)
    except OverflowError:
        return float_info.max
    return value if isfinite(value) else float_info.max


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvalCase",
    "EvaluationReport",
    "EvaluationRun",
    "PromotionDecision",
    "RankedResult",
    "RelevanceJudgment",
    "evaluation_dataset_digest",
    "evaluate_rankings",
    "load_cases",
    "promotion_decision",
]
