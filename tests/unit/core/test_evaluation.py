import json
import math
from pathlib import Path

import pytest

from llmwiki_core.evaluation import (
    EvalCase,
    EvaluationReport,
    EvaluationRun,
    RankedResult,
    RelevanceJudgment,
    evaluate_rankings,
    load_cases,
    promotion_decision,
)
from llmwiki_core.search import SearchArea, SearchScope

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "retrieval" / "v1"


def _write_jsonl(tmp_path: Path, *values: object) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "".join(json.dumps(value, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )
    return path


def _case_payload(case_id: str = "case-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "query": {"text": "synthetic permit query"},
        "relevance": [{"document_id": "fixture-policy", "grade": 2}],
    }


def _case(
    case_id: str,
    relevance: tuple[RelevanceJudgment, ...],
    **query_fields: object,
) -> EvalCase:
    return EvalCase.build(
        schema_version=1,
        case_id=case_id,
        query={"text": f"query {case_id}", **query_fields},
        relevance=relevance,
    )


def _run(
    case_id: str,
    *ranking: tuple[str, int | None],
    latency_ms: float = 1.0,
) -> EvaluationRun:
    return EvaluationRun(
        case_id=case_id,
        ranking=tuple(RankedResult(document_id, chunk_index) for document_id, chunk_index in ranking),
        latency_ms=latency_ms,
    )


def _report(*, recall_at_10: float, latency_p95_ms: float) -> EvaluationReport:
    return EvaluationReport(
        case_count=1,
        recall_at_5=recall_at_10,
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_10,
        mrr=recall_at_10,
        ndcg_at_10=recall_at_10,
        filtered_result_count=0,
        latency_p50_ms=latency_p95_ms,
        latency_p95_ms=latency_p95_ms,
    )


def test_loads_synthetic_fixture_and_all_current_query_filters():
    cases = load_cases(FIXTURE_ROOT / "cases.jsonl")

    assert [case.case_id for case in cases] == ["export-idn", "annotated-wiki-sgp"]
    filtered = cases[1].query
    assert filtered.limit == 5
    assert filtered.candidate_limit == 15
    assert filtered.area is SearchArea.WIKI
    assert filtered.scope is SearchScope.ANNOTATIONS
    assert filtered.path_glob == "/corpus/sgp/**/*.md"
    assert filtered.tags == ("reviewed", "synthetic")
    assert tuple(kind.value for kind in filtered.document_kinds) == ("wiki",)
    assert filtered.annotated_only is True
    assert dict(filtered.facets) == {"country": "SGP"}
    assert cases[0].relevance[0].chunk_index == 0
    assert cases[0].relevance[1].chunk_index is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "unsupported schema_version"),
        (lambda payload: payload.pop("relevance"), "missing fields: relevance"),
        (lambda payload: payload.update(extra=True), "unknown fields: extra"),
        (lambda payload: payload.update(case_id="  "), "case_id must be a nonblank string"),
        (
            lambda payload: payload["query"].update(unknown="value"),
            "query has unknown fields: unknown",
        ),
        (
            lambda payload: payload["relevance"][0].update(extra=True),
            "relevance has unknown fields: extra",
        ),
        (
            lambda payload: payload.update(relevance=[]),
            "relevance must not be empty",
        ),
        (
            lambda payload: payload["relevance"][0].update(grade=0),
            "grade must be a positive integer",
        ),
        (
            lambda payload: payload["relevance"][0].update(grade=True),
            "grade must be a positive integer",
        ),
        (
            lambda payload: payload["relevance"][0].update(chunk_index=-1),
            "chunk_index must be a non-negative integer or null",
        ),
    ],
)
def test_parser_rejects_invalid_or_non_strict_cases(tmp_path, mutate, message):
    payload = _case_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        load_cases(_write_jsonl(tmp_path, payload))


def test_parser_rejects_duplicate_case_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate case_id: case-1"):
        load_cases(_write_jsonl(tmp_path, _case_payload(), _case_payload()))


def test_parser_rejects_duplicate_and_overlapping_relevance(tmp_path):
    duplicate = _case_payload()
    duplicate["relevance"] = [
        {"document_id": "fixture-policy", "chunk_index": 0, "grade": 2},
        {"document_id": "fixture-policy", "chunk_index": 0, "grade": 1},
    ]
    with pytest.raises(ValueError, match="duplicate relevance judgment"):
        load_cases(_write_jsonl(tmp_path, duplicate))

    overlap = _case_payload()
    overlap["relevance"] = [
        {"document_id": "fixture-policy", "grade": 2},
        {"document_id": "fixture-policy", "chunk_index": 0, "grade": 1},
    ]
    with pytest.raises(ValueError, match="document-level and chunk-level relevance overlap"):
        load_cases(_write_jsonl(tmp_path, overlap))


@pytest.mark.parametrize("content", ["", "\n\n", "not-json\n", "[]\n"])
def test_parser_rejects_empty_or_malformed_jsonl(tmp_path, content):
    path = tmp_path / "cases.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_parser_rejects_oversized_input(tmp_path):
    payload = _case_payload()
    payload["case_id"] = "x" * 300_000

    with pytest.raises(ValueError, match="line exceeds"):
        load_cases(_write_jsonl(tmp_path, payload))


def test_metrics_match_hand_calculated_rankings_and_macro_average():
    cases = (
        _case(
            "a",
            (
                RelevanceJudgment("doc-a", 3, 0),
                RelevanceJudgment("doc-b", 1, 1),
            ),
            facets={"country": "IDN"},
        ),
        _case(
            "b",
            (
                RelevanceJudgment("doc-c", 2),
                RelevanceJudgment("doc-d", 1),
            ),
        ),
    )
    runs = (
        _run("a", ("doc-a", 0), ("noise", 0), latency_ms=10),
        _run("b", ("noise", 0), ("doc-c", 7), ("doc-d", 9), latency_ms=20),
    )

    report = evaluate_rankings(cases, runs)

    assert report.case_count == 2
    assert report.recall_at_5 == pytest.approx(0.75)
    assert report.recall_at_10 == pytest.approx(0.75)
    assert report.recall_at_20 == pytest.approx(0.75)
    assert report.mrr == pytest.approx(0.75)
    first_ndcg = 7 / (7 + 1 / math.log2(3))
    second_ndcg = ((3 / math.log2(3)) + (1 / math.log2(4))) / (3 + 1 / math.log2(3))
    assert report.ndcg_at_10 == pytest.approx((first_ndcg + second_ndcg) / 2)
    assert report.filtered_result_count == 2
    assert report.latency_p50_ms == 10
    assert report.latency_p95_ms == 20


def test_document_level_relevance_is_not_double_counted_by_multiple_chunks():
    cases = (_case("doc", (RelevanceJudgment("relevant", 3),)),)
    runs = (_run("doc", ("relevant", 4), ("relevant", 8), ("noise", 0)),)

    report = evaluate_rankings(cases, runs)

    assert report.recall_at_5 == 1
    assert report.mrr == 1
    assert report.ndcg_at_10 == 1


def test_chunk_level_relevance_requires_the_exact_chunk():
    cases = (_case("chunk", (RelevanceJudgment("relevant", 3, 2),)),)
    runs = (_run("chunk", ("relevant", 1), ("relevant", 2)),)

    report = evaluate_rankings(cases, runs)

    assert report.recall_at_5 == 1
    assert report.mrr == pytest.approx(0.5)
    assert report.ndcg_at_10 == pytest.approx(1 / math.log2(3))


def test_nearest_rank_percentiles_are_stable():
    cases = tuple(_case(str(index), (RelevanceJudgment(f"d{index}", 1),)) for index in range(20))
    runs = tuple(_run(str(index), (f"d{index}", 0), latency_ms=float(index + 1)) for index in range(20))

    report = evaluate_rankings(cases, runs)

    assert report.latency_p50_ms == 10
    assert report.latency_p95_ms == 19


@pytest.mark.parametrize(
    ("cases", "runs", "message"),
    [
        (
            (_case("a", (RelevanceJudgment("a", 1),)),),
            (),
            "exactly one run per case",
        ),
        (
            (_case("a", (RelevanceJudgment("a", 1),)),),
            (_run("a", ("a", 0)), _run("a", ("a", 0))),
            "duplicate evaluation run",
        ),
        (
            (_case("a", (RelevanceJudgment("a", 1),)),),
            (_run("unknown", ("a", 0)),),
            "unknown case_id",
        ),
    ],
)
def test_evaluation_requires_exactly_one_run_per_case(cases, runs, message):
    with pytest.raises(ValueError, match=message):
        evaluate_rankings(cases, runs)


def test_value_types_reject_invalid_rankings_latencies_and_duplicate_cases():
    with pytest.raises(ValueError, match="duplicate ranked result"):
        EvaluationRun(
            case_id="a",
            ranking=(RankedResult("doc", 0), RankedResult("doc", 0)),
            latency_ms=1,
        )
    with pytest.raises(ValueError, match="latency_ms must be finite and non-negative"):
        EvaluationRun(case_id="a", ranking=(), latency_ms=float("nan"))
    with pytest.raises(ValueError, match="chunk_index must be a non-negative integer or null"):
        RankedResult("doc", True)
    duplicate = _case("a", (RelevanceJudgment("doc", 1),))
    with pytest.raises(ValueError, match="duplicate evaluation case"):
        evaluate_rankings((duplicate, duplicate), (_run("a", ("doc", 0)),))


@pytest.mark.parametrize(
    ("baseline_recall", "hybrid_recall", "baseline_latency", "hybrid_latency", "eligible"),
    [
        (0.50, 0.55, 10.0, 20.0, True),
        (0.50, 0.549999, 10.0, 20.0, False),
        (0.50, 0.55, 10.0, 20.000001, False),
    ],
)
def test_promotion_gate_has_stable_inclusive_boundaries(
    baseline_recall,
    hybrid_recall,
    baseline_latency,
    hybrid_latency,
    eligible,
):
    decision = promotion_decision(
        _report(recall_at_10=baseline_recall, latency_p95_ms=baseline_latency),
        _report(recall_at_10=hybrid_recall, latency_p95_ms=hybrid_latency),
    )

    assert decision.eligible is eligible
    assert decision.reason == ("eligible" if eligible else "gate_failed")


def test_promotion_gate_rejects_zero_baseline_recall():
    decision = promotion_decision(
        _report(recall_at_10=0, latency_p95_ms=0),
        _report(recall_at_10=1, latency_p95_ms=0),
    )

    assert decision.eligible is False
    assert decision.reason == "baseline_recall_zero"
    assert decision.recall_ratio is None
    assert decision.latency_ratio is None


def test_public_evaluation_values_are_immutable():
    judgment = RelevanceJudgment("doc", 1)
    run = _run("case", ("doc", 0))

    with pytest.raises((AttributeError, TypeError)):
        judgment.grade = 2
    with pytest.raises((AttributeError, TypeError)):
        run.ranking += (RankedResult("other", 0),)
