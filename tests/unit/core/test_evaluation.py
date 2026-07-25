import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmwiki_core import evaluation as evaluation_module
from llmwiki_core.evaluation import (
    EvalCase,
    EvaluationReport,
    EvaluationRun,
    PromotionDecision,
    RankedResult,
    RelevanceJudgment,
    evaluate_rankings,
    load_cases,
    promotion_decision,
)
from llmwiki_core.search import SearchArea, SearchQuery, SearchScope

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


def _report(
    *,
    recall_at_10: float,
    latency_p95_ms: float,
    case_count: int = 1,
    dataset_digest: str = "0" * 64,
) -> EvaluationReport:
    return EvaluationReport(
        case_count=case_count,
        recall_at_5=recall_at_10,
        recall_at_10=recall_at_10,
        recall_at_20=recall_at_10,
        mrr=recall_at_10,
        ndcg_at_10=recall_at_10,
        filtered_result_count=0,
        latency_p50_ms=latency_p95_ms,
        latency_p95_ms=latency_p95_ms,
        dataset_digest=dataset_digest,
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
        (lambda payload: payload.update(schema_version=1.0), "unsupported schema_version"),
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


def test_positive_integer_grades_are_not_subject_to_an_arbitrary_domain_cap():
    judgment = RelevanceJudgment("fixture-policy", 101)

    assert judgment.grade == 101


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


def test_parser_rejects_non_regular_files_before_opening(tmp_path):
    with pytest.raises(ValueError, match="must be a regular file"):
        load_cases(tmp_path)


def test_parser_enforces_cumulative_bytes_while_reading(tmp_path, monkeypatch):
    path = _write_jsonl(
        tmp_path,
        _case_payload("case-1"),
        _case_payload("case-2"),
        _case_payload("case-3"),
    )
    real_stat = path.stat()
    monkeypatch.setattr(evaluation_module, "MAX_DATASET_BYTES", 100)
    monkeypatch.setattr(
        evaluation_module.os,
        "fstat",
        lambda _descriptor: SimpleNamespace(st_size=0, st_mode=real_stat.st_mode),
    )

    with pytest.raises(ValueError, match="exceeds the size limit"):
        load_cases(path)


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


def test_recall_and_ndcg_include_exact_cutoff_ranks_only():
    ranks = (5, 6, 10, 11, 20, 21)
    cases = tuple(_case(str(rank), (RelevanceJudgment(f"d{rank}", 1),)) for rank in ranks)
    runs = tuple(
        _run(
            str(rank),
            *((f"noise-{rank}-{index}", 0) for index in range(1, rank)),
            (f"d{rank}", 0),
        )
        for rank in ranks
    )

    report = evaluate_rankings(cases, runs)

    assert report.recall_at_5 == pytest.approx(1 / 6)
    assert report.recall_at_10 == pytest.approx(3 / 6)
    assert report.recall_at_20 == pytest.approx(5 / 6)
    expected_ndcg = sum(1 / math.log2(rank + 1) for rank in (5, 6, 10)) / 6
    assert report.ndcg_at_10 == pytest.approx(expected_ndcg)


def test_very_large_grade_is_evaluated_without_materializing_a_huge_power():
    huge_grade = 10**100
    cases = (_case("huge", (RelevanceJudgment("relevant", huge_grade),)),)
    runs = (_run("huge", ("relevant", 0)),)

    report = evaluate_rankings(cases, runs)

    assert report.ndcg_at_10 == 1


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
    with pytest.raises(ValueError, match="latency_ms must be finite and non-negative"):
        EvaluationRun(case_id="a", ranking=(), latency_ms=10**10_000)
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


def test_promotion_gate_accepts_exact_boundary_after_macro_average_rounding():
    one = _case("one", (RelevanceJudgment("one", 1),))
    seven = _case(
        "seven",
        tuple(RelevanceJudgment(f"seven-{index}", 1) for index in range(7)),
    )
    cases = (one, seven)
    lexical = evaluate_rankings(
        cases,
        (
            _run("one", ("one", 0), latency_ms=1),
            _run(
                "seven",
                *((f"seven-{index}", 0) for index in range(3)),
                latency_ms=1,
            ),
        ),
    )
    hybrid = evaluate_rankings(
        cases,
        (
            _run("one", ("one", 0), latency_ms=2),
            _run(
                "seven",
                *((f"seven-{index}", 0) for index in range(4)),
                latency_ms=2,
            ),
        ),
    )
    decision = promotion_decision(lexical, hybrid)

    assert lexical.recall_at_10_exact == Fraction(5, 7)
    assert hybrid.recall_at_10_exact == Fraction(11, 14)
    assert decision.recall_ratio == 1.1
    assert decision.eligible is True
    assert decision.reason == "eligible"


def test_promotion_gate_strictly_rejects_one_float_below_quality_boundary():
    decision = promotion_decision(
        _report(recall_at_10=0.5, latency_p95_ms=10),
        _report(recall_at_10=math.nextafter(0.55, 0.0), latency_p95_ms=10),
    )

    assert decision.recall_ratio < 1.10
    assert decision.eligible is False
    assert decision.reason == "gate_failed"


def test_promotion_gate_strictly_rejects_one_float_above_latency_boundary():
    decision = promotion_decision(
        _report(recall_at_10=0.5, latency_p95_ms=10),
        _report(recall_at_10=0.55, latency_p95_ms=math.nextafter(20.0, math.inf)),
    )

    assert decision.latency_ratio > 2.0
    assert decision.eligible is False
    assert decision.reason == "gate_failed"


def test_promotion_gate_strictly_rejects_small_nextafter_quality_value():
    decision = promotion_decision(
        _report(recall_at_10=0.0003, latency_p95_ms=10),
        _report(
            recall_at_10=math.nextafter(0.00033, 0.0),
            latency_p95_ms=10,
        ),
    )

    assert decision.recall_ratio < 1.10
    assert decision.eligible is False


def test_promotion_gate_strictly_rejects_milliscale_nextafter_boundaries():
    baseline = 0.001
    quality = promotion_decision(
        _report(recall_at_10=baseline, latency_p95_ms=baseline),
        _report(
            recall_at_10=math.nextafter(baseline * 1.1, 0.0),
            latency_p95_ms=baseline,
        ),
    )
    latency = promotion_decision(
        _report(recall_at_10=0.5, latency_p95_ms=baseline),
        _report(
            recall_at_10=0.55,
            latency_p95_ms=math.nextafter(baseline * 2, math.inf),
        ),
    )

    assert quality.eligible is False
    assert latency.eligible is False


def test_promotion_gate_uses_exact_recall_from_real_macro_metrics():
    one = _case("one", (RelevanceJudgment("one", 1),))
    many_relevance = tuple(RelevanceJudgment(f"many-{index}", 1) for index in range(48))
    many = _case("many", many_relevance)
    cases = (one, many)
    lexical = evaluate_rankings(
        cases,
        (
            _run("one", ("one", 0)),
            _run("many", *((f"many-{index}", 0) for index in range(2))),
        ),
    )
    hybrid = evaluate_rankings(
        cases,
        (
            _run("one", ("one", 0)),
            _run("many", *((f"many-{index}", 0) for index in range(7))),
        ),
    )

    decision = promotion_decision(lexical, hybrid)

    assert lexical.recall_at_10 == pytest.approx(25 / 48)
    assert hybrid.recall_at_10 == pytest.approx(55 / 96)
    assert decision.eligible is True


def test_public_recalls_round_from_the_exact_macro_average():
    first_relevance = tuple(RelevanceJudgment(f"first-{index}", 1) for index in range(411))
    second_relevance = tuple(RelevanceJudgment(f"second-{index}", 1) for index in range(334))
    cases = (
        _case("first", first_relevance),
        _case("second", second_relevance),
    )
    runs = (
        _run("first", *((f"first-{index}", 0) for index in range(2))),
        _run("second", *((f"second-{index}", 0) for index in range(3))),
    )
    exact = (Fraction(2, 411) + Fraction(3, 334)) / 2

    report = evaluate_rankings(cases, runs)

    assert report.recall_at_10_exact == exact
    assert report.recall_at_5 == float(exact)
    assert report.recall_at_10 == float(exact)
    assert report.recall_at_20 == float(exact)


def test_promotion_decision_handles_finite_values_whose_ratio_overflows():
    minimum_positive = float.fromhex("0x0.0000000000001p-1022")
    maximum_finite = float.fromhex("0x1.fffffffffffffp+1023")

    quality = promotion_decision(
        _report(recall_at_10=minimum_positive, latency_p95_ms=1),
        _report(recall_at_10=1, latency_p95_ms=1),
    )
    latency = promotion_decision(
        _report(recall_at_10=1, latency_p95_ms=0),
        _report(recall_at_10=1, latency_p95_ms=maximum_finite),
    )

    assert quality.eligible is True
    assert math.isfinite(quality.recall_ratio)
    assert latency.eligible is False
    assert math.isfinite(latency.latency_ratio)


def test_promotion_gate_rejects_no_improvement_at_minimum_positive_recall():
    minimum_positive = float.fromhex("0x0.0000000000001p-1022")

    decision = promotion_decision(
        _report(recall_at_10=minimum_positive, latency_p95_ms=1),
        _report(recall_at_10=minimum_positive, latency_p95_ms=1),
    )

    assert decision.recall_ratio == 1
    assert decision.eligible is False
    assert decision.reason == "gate_failed"


def test_dataset_digest_is_order_independent_and_definition_sensitive():
    first = _case(
        "first",
        (RelevanceJudgment("doc-first", 2, 0),),
        facets={"country": "IDN"},
        tags=["reviewed"],
    )
    second = _case("second", (RelevanceJudgment("doc-second", 1),))
    changed_query = EvalCase.build(
        schema_version=1,
        case_id="first",
        query={"text": "changed query", "facets": {"country": "IDN"}, "tags": ["reviewed"]},
        relevance=(RelevanceJudgment("doc-first", 2, 0),),
    )
    changed_relevance = _case(
        "first",
        (RelevanceJudgment("doc-first", 3, 0),),
        facets={"country": "IDN"},
        tags=["reviewed"],
    )

    digest = evaluation_module.evaluation_dataset_digest((first, second))

    assert digest == evaluation_module.evaluation_dataset_digest((second, first))
    assert digest != evaluation_module.evaluation_dataset_digest((changed_query, second))
    assert digest != evaluation_module.evaluation_dataset_digest((changed_relevance, second))
    assert len(digest) == 64
    assert "query first" not in digest


def test_promotion_rejects_unknown_or_different_cohorts():
    baseline = _report(recall_at_10=0.5, latency_p95_ms=1)

    with pytest.raises(ValueError, match="same evaluation cohort"):
        promotion_decision(
            baseline,
            _report(recall_at_10=0.55, latency_p95_ms=1, case_count=2),
        )
    with pytest.raises(ValueError, match="same evaluation cohort"):
        promotion_decision(
            baseline,
            _report(recall_at_10=0.55, latency_p95_ms=1, dataset_digest="1" * 64),
        )
    with pytest.raises(ValueError, match="identified evaluation cohort"):
        promotion_decision(
            _report(recall_at_10=0.5, latency_p95_ms=1, dataset_digest=None),
            _report(recall_at_10=0.55, latency_p95_ms=1, dataset_digest=None),
        )


def test_report_and_decision_constructors_reject_contradictions():
    with pytest.raises(ValueError, match="recall metrics must be monotonic"):
        EvaluationReport(
            case_count=1,
            recall_at_5=0.6,
            recall_at_10=0.5,
            recall_at_20=0.7,
            mrr=0.5,
            ndcg_at_10=0.5,
            filtered_result_count=0,
            latency_p50_ms=1,
            latency_p95_ms=2,
            dataset_digest="0" * 64,
        )
    with pytest.raises(ValueError, match="p50 must not exceed p95"):
        EvaluationReport(
            case_count=1,
            recall_at_5=0.5,
            recall_at_10=0.5,
            recall_at_20=0.5,
            mrr=0.5,
            ndcg_at_10=0.5,
            filtered_result_count=0,
            latency_p50_ms=2,
            latency_p95_ms=1,
            dataset_digest="0" * 64,
        )
    with pytest.raises(TypeError):
        EvaluationReport(
            case_count=1,
            recall_at_5=0.5,
            recall_at_10=0.5,
            recall_at_20=0.5,
            mrr=0.5,
            ndcg_at_10=0.5,
            filtered_result_count=0,
            latency_p50_ms=1,
            latency_p95_ms=1,
            dataset_digest="0" * 64,
            recall_at_10_exact=Fraction(1, 2),
        )

    inconsistent = (
        (True, "gate_failed", 1.1, 1.0),
        (False, "eligible", 1.1, 1.0),
        (True, "eligible", 1.0, 1.0),
        (False, "baseline_recall_zero", 1.1, 1.0),
        (False, "gate_failed", None, None),
        (False, "unknown", 1.0, 1.0),
    )
    for args in inconsistent:
        with pytest.raises(ValueError, match="promotion decision fields are inconsistent"):
            PromotionDecision(*args)

    rounded_gate_failure = PromotionDecision(False, "gate_failed", 1.1, 2.0)
    assert rounded_gate_failure.eligible is False


class _GuardedIterable:
    def __init__(self, factory, *, allowed: int):
        self.factory = factory
        self.allowed = allowed
        self.consumed = 0

    def __iter__(self):
        while True:
            self.consumed += 1
            if self.consumed > self.allowed:
                raise AssertionError("iterable consumed beyond the bounded probe")
            yield self.factory(self.consumed)


def test_case_and_run_materialize_iterables_with_a_hard_bound(monkeypatch):
    monkeypatch.setattr(evaluation_module, "MAX_RELEVANCE_PER_CASE", 3)
    monkeypatch.setattr(evaluation_module, "MAX_RANKING_LENGTH", 3)
    relevance = _GuardedIterable(
        lambda index: RelevanceJudgment(f"doc-{index}", 1),
        allowed=4,
    )
    ranking = _GuardedIterable(
        lambda index: RankedResult(f"doc-{index}", 0),
        allowed=4,
    )

    with pytest.raises(ValueError, match="too many relevance judgments"):
        EvalCase(1, "bounded", SearchQuery.build(text="query"), relevance)
    with pytest.raises(ValueError, match="ranking is too large"):
        EvaluationRun("bounded", ranking, 1)

    assert relevance.consumed == 4
    assert ranking.consumed == 4


def test_exact_gate_failure_survives_ratio_rounding_to_public_threshold():
    baseline = float.fromhex("0x1.313542965b73bp-1")
    hybrid = float.fromhex("0x1.4fba960bcaff4p-1")

    decision = promotion_decision(
        _report(recall_at_10=baseline, latency_p95_ms=1),
        _report(recall_at_10=hybrid, latency_p95_ms=1),
    )

    assert decision.recall_ratio == 1.1
    assert decision.eligible is False
    assert decision.reason == "gate_failed"


def test_public_evaluation_values_are_immutable():
    judgment = RelevanceJudgment("doc", 1)
    run = _run("case", ("doc", 0))

    with pytest.raises((AttributeError, TypeError)):
        judgment.grade = 2
    with pytest.raises((AttributeError, TypeError)):
        run.ranking += (RankedResult("other", 0),)
