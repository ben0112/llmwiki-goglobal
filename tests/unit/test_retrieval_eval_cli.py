import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest
from scripts import retrieval_eval as retrieval_eval_module
from scripts.retrieval_eval import (
    _default_retriever_factory,
    _load_corpus,
    _logical_glob_matches,
    main,
)

from llmwiki_core import evaluation_dataset_digest, load_cases
from llmwiki_core.search import SearchHit, SearchResult

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "retrieval" / "v1"
DATASET = FIXTURE_ROOT / "cases.jsonl"


def _hit(document_id: str, chunk_index: int, *, content: str = "private fixture content") -> SearchHit:
    return SearchHit(
        document_id=document_id,
        document_version=1,
        chunk_index=chunk_index,
        content=content,
        score=1.0,
        path=f"/private/{document_id}.md",
    )


class _FakeRetriever:
    def __init__(
        self,
        profile: str,
        results: dict[str, tuple[tuple[SearchHit, ...], float]],
    ) -> None:
        self.profile = profile
        self.results = results
        self.calls: list[object] = []

    async def retrieve(self, query):
        self.calls.append(query)
        hits, latency_ms = self.results[query.text]
        return SearchResult(
            hits=hits,
            candidate_count=len(hits),
            latency_ms=latency_ms,
            profile=self.profile,
        )


class _FailingRetriever:
    async def retrieve(self, query):
        raise RuntimeError(f"api_key=sk-secret query={query.text} embedding=[1,2,3]")


class _SyncCancelledRetriever:
    def retrieve(self, query):
        raise asyncio.CancelledError(
            f"api_key=sk-cancelled query={query.text} content=private embedding=[1,2,3]"
        )


class _AsyncCancelledRetriever:
    async def retrieve(self, query):
        raise asyncio.CancelledError(
            f"api_key=sk-cancelled query={query.text} content=private embedding=[1,2,3]"
        )


def _cancelled_factory(_profile, _path):
    raise asyncio.CancelledError(
        "api_key=sk-cancelled query=private content=private embedding=[1,2,3]"
    )


def _profile_results(profile: str, *, eligible: bool = True) -> dict[str, tuple[tuple[SearchHit, ...], float]]:
    if profile == "lexical":
        return {
            "Indonesia export permit": ((_hit("fixture-policy-idn", 0),), 10.0),
            "Singapore synthetic compliance notes": ((), 20.0),
        }
    if eligible:
        return {
            "Indonesia export permit": (
                (_hit("fixture-policy-idn", 0), _hit("fixture-checklist-idn", 1)),
                12.0,
            ),
            "Singapore synthetic compliance notes": ((_hit("fixture-wiki-sgp", 1),), 15.0),
        }
    return {
        "Indonesia export permit": ((_hit("fixture-policy-idn", 0),), 21.0),
        "Singapore synthetic compliance notes": ((), 25.0),
    }


def _factory(*, eligible: bool = True):
    retrievers = {
        profile: _FakeRetriever(profile, _profile_results(profile, eligible=eligible))
        for profile in ("lexical", "hybrid")
    }
    factory_calls: list[tuple[str, Path]] = []

    def build(profile: str, dataset_path: Path):
        factory_calls.append((profile, dataset_path))
        return retrievers[profile]

    return build, retrievers, factory_calls


def _read_json(capsys) -> tuple[dict[str, object], str]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out), captured.out


@pytest.mark.parametrize("profile", ["lexical", "hybrid"])
def test_single_profile_report_is_deterministic_and_retrieves_each_case_once(profile, capsys):
    factory, retrievers, factory_calls = _factory()

    code = main(["--dataset", str(DATASET), "--profile", profile], retriever_factory=factory)
    payload, first_stdout = _read_json(capsys)

    second_factory, _, _ = _factory()
    second_code = main(
        ["--dataset", str(DATASET), "--profile", profile],
        retriever_factory=second_factory,
    )
    _second_payload, second_stdout = _read_json(capsys)

    cases = load_cases(DATASET)
    assert code == second_code == 0
    assert first_stdout == second_stdout
    assert first_stdout.endswith("\n")
    assert payload["schema_version"] == 1
    assert payload["dataset_schema_version"] == 1
    assert payload["evaluation_dataset_digest"] == evaluation_dataset_digest(cases)
    assert len(payload["evaluation_dataset_digest"]) == 64
    assert "dataset_digest" not in payload
    assert payload["profile"] == profile
    assert payload["case_count"] == len(cases)
    assert set(payload["metrics"]) == {
        "filtered_result_count",
        "latency_p50_ms",
        "latency_p95_ms",
        "mrr",
        "ndcg_at_10",
        "recall_at_10",
        "recall_at_20",
        "recall_at_5",
    }
    assert factory_calls == [(profile, DATASET)]
    assert len(retrievers[profile].calls) == len(cases)
    assert Counter(query.text for query in retrievers[profile].calls) == Counter(
        case.query.text for case in cases
    )


def test_compare_reports_both_profiles_and_eligible_gate(capsys):
    factory, retrievers, factory_calls = _factory(eligible=True)

    code = main(
        ["--dataset", str(DATASET), "--compare", "--require-promotion-gate"],
        retriever_factory=factory,
    )
    payload, _stdout = _read_json(capsys)

    assert code == 0
    assert payload["profile"] == "compare"
    assert set(payload["profiles"]) == {"lexical", "hybrid"}
    assert payload["profiles"]["lexical"]["profile"] == "lexical"
    assert payload["profiles"]["hybrid"]["profile"] == "hybrid"
    assert payload["promotion"] == {
        "eligible": True,
        "latency_ratio": 0.75,
        "reason": "eligible",
        "recall_ratio": 4.0,
    }
    assert [profile for profile, _path in factory_calls] == ["lexical", "hybrid"]
    assert all(len(retriever.calls) == 2 for retriever in retrievers.values())


def test_required_promotion_gate_returns_exit_three_when_ineligible(capsys):
    factory, _retrievers, _factory_calls = _factory(eligible=False)

    code = main(
        ["--dataset", str(DATASET), "--compare", "--require-promotion-gate"],
        retriever_factory=factory,
    )
    payload, _stdout = _read_json(capsys)

    assert code == 3
    assert payload["promotion"]["eligible"] is False
    assert payload["promotion"]["reason"] == "gate_failed"


def test_compare_without_required_gate_returns_zero_when_ineligible(capsys):
    factory, _retrievers, _factory_calls = _factory(eligible=False)

    code = main(["--dataset", str(DATASET), "--compare"], retriever_factory=factory)
    payload, _stdout = _read_json(capsys)

    assert code == 0
    assert payload["promotion"]["eligible"] is False


def test_output_file_is_byte_identical_to_stdout(tmp_path, capsys):
    factory, _retrievers, _factory_calls = _factory()
    output_path = tmp_path / "nested" / "report.json"
    output_path.parent.mkdir()

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", str(output_path)],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    assert output_path.read_bytes() == captured.out.encode("utf-8")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--dataset", "/private/dataset.jsonl", "--profile", "unknown"],
        ["--dataset", "/private/dataset.jsonl", "--compare", "--profile", "lexical"],
        ["--dataset", "/private/dataset.jsonl", "--require-promotion-gate"],
    ],
)
def test_argument_errors_are_stable_and_sanitized(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "arguments", "code": "invalid_arguments"}
    }
    assert "/private" not in captured.err


def test_invalid_dataset_is_nonzero_and_does_not_echo_path_or_content(tmp_path, capsys):
    secret_path = tmp_path / "customer-secret-name.jsonl"
    secret_path.write_text('{"query":"customer secret text","api_key":"sk-secret"}\n', encoding="utf-8")

    code = main(["--dataset", str(secret_path), "--profile", "lexical"])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "dataset", "code": "dataset_invalid"}
    }
    assert "customer" not in captured.err
    assert "sk-secret" not in captured.err
    assert str(secret_path) not in captured.err


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_default_lexical_rejects_non_finite_corpus_metadata_as_dataset_error(
    non_finite,
    tmp_path,
    capsys,
):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"schema_version":1,"case_id":"case","query":{"text":"private query"},'
        '"relevance":[{"document_id":"fixture-doc","grade":1}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "corpus.jsonl").write_text(
        '{"schema_version":1,"document_id":"fixture-doc","document_kind":"source",'
        f'"path":"/private.md","title":"Synthetic","facets":{{"score":{non_finite}}},'
        '"chunks":[{"chunk_index":0,"content":"private document content"}]}\n',
        encoding="utf-8",
    )

    corpus = tmp_path / "corpus.jsonl"
    with pytest.raises(ValueError, match="corpus .* invalid"):
        _load_corpus(corpus)

    code = main(["--dataset", str(dataset), "--profile", "lexical"])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "dataset", "code": "dataset_invalid"}
    }
    assert "private query" not in captured.err
    assert "private document content" not in captured.err


def test_default_hybrid_fails_closed_with_typed_configuration_error(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-do-not-print")

    code = main(["--dataset", str(DATASET), "--profile", "hybrid"])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "configuration", "code": "hybrid_unavailable"}
    }
    assert "sk-do-not-print" not in captured.err


@pytest.mark.parametrize(
    ("factory", "expected_code"),
    [
        (lambda _profile, _path: object(), "retrieval_contract_invalid"),
        (
            lambda _profile, _path: _FakeRetriever(
                "wrong-profile",
                _profile_results("lexical"),
            ),
            "retrieval_contract_invalid",
        ),
        (lambda _profile, _path: _FailingRetriever(), "retrieval_failed"),
        (
            lambda _profile, _path: (_ for _ in ()).throw(
                RuntimeError("api_key=sk-secret customer query embedding=[1,2,3]")
            ),
            "retrieval_failed",
        ),
    ],
)
def test_untrusted_factory_result_profile_and_exceptions_are_sanitized(factory, expected_code, capsys):
    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "retrieval", "code": expected_code}
    }
    assert "sk-secret" not in captured.err
    assert "customer" not in captured.err
    assert "embedding" not in captured.err
    assert "Indonesia export permit" not in captured.err


@pytest.mark.parametrize(
    "factory",
    [
        _cancelled_factory,
        lambda _profile, _path: _SyncCancelledRetriever(),
        lambda _profile, _path: _AsyncCancelledRetriever(),
    ],
)
def test_cancelled_factory_and_retrieval_are_sanitized(factory, capsys):
    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "retrieval", "code": "retrieval_failed"}
    }
    for secret in (
        "sk-cancelled",
        "Indonesia export permit",
        "private",
        "embedding",
        "CancelledError",
        "Traceback",
    ):
        assert secret not in captured.err


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_process_control_base_exceptions_are_not_swallowed(error_type):
    def factory(_profile, _path):
        raise error_type

    with pytest.raises(error_type):
        main(
            ["--dataset", str(DATASET), "--profile", "lexical"],
            retriever_factory=factory,
        )


def test_rejects_search_result_with_wrong_profile_without_leaking_hit_content(capsys):
    class WrongProfileRetriever:
        async def retrieve(self, _query):
            return SearchResult(
                hits=(_hit("fixture-policy-idn", 0, content="customer-secret-content"),),
                candidate_count=1,
                latency_ms=1,
                profile="hybrid",
            )

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=lambda _profile, _path: WrongProfileRetriever(),
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "retrieval_contract_invalid"
    assert "customer-secret-content" not in captured.err


@pytest.mark.asyncio
async def test_sync_main_can_run_when_caller_event_loop_is_active(capsys):
    factory, _retrievers, _factory_calls = _factory()

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )
    payload, _stdout = _read_json(capsys)

    assert code == 0
    assert payload["profile"] == "lexical"


def test_output_refuses_symlink_and_does_not_modify_target(tmp_path, capsys):
    target = tmp_path / "target.json"
    target.write_text("keep me", encoding="utf-8")
    output_path = tmp_path / "report.json"
    output_path.symlink_to(target)
    factory, _retrievers, _factory_calls = _factory()

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", str(output_path)],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "output", "code": "output_write_failed"}
    }
    assert target.read_text(encoding="utf-8") == "keep me"
    assert str(output_path) not in captured.err


def test_output_refuses_symlink_parent_without_writing_target(tmp_path, capsys):
    real_parent = tmp_path / "private-target"
    real_parent.mkdir()
    symlink_parent = tmp_path / "customer-secret-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    output_path = symlink_parent / "report.json"
    factory, _retrievers, _factory_calls = _factory()

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", str(output_path)],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "output", "code": "output_write_failed"}
    }
    assert not (real_parent / "report.json").exists()
    assert "customer-secret-parent" not in captured.err


def test_atomic_output_replaces_by_open_parent_dirfd(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "report.json"
    factory, _retrievers, _factory_calls = _factory()
    real_replace = retrieval_eval_module.os.replace
    observed: dict[str, object] = {}

    def replace(source, destination, **kwargs):
        observed.update(source=source, destination=destination, **kwargs)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(retrieval_eval_module.os, "replace", replace)

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", str(output_path)],
        retriever_factory=factory,
    )
    _payload, stdout = _read_json(capsys)

    assert code == 0
    assert observed["source"] == observed["source"].split("/")[-1]
    assert observed["destination"] == "report.json"
    assert isinstance(observed["src_dir_fd"], int)
    assert observed["src_dir_fd"] == observed["dst_dir_fd"]
    assert output_path.read_bytes() == stdout.encode("utf-8")


def test_default_synthetic_lexical_baseline_is_stable_and_content_free(capsys):
    code = main(["--dataset", str(DATASET), "--profile", "lexical"])
    payload, stdout = _read_json(capsys)

    assert code == 0
    assert payload == {
        "case_count": 2,
        "evaluation_dataset_digest": evaluation_dataset_digest(load_cases(DATASET)),
        "dataset_schema_version": 1,
        "metrics": {
            "filtered_result_count": 3,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "mrr": 1.0,
            "ndcg_at_10": 1.0,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "recall_at_5": 1.0,
        },
        "profile": "lexical",
        "schema_version": 1,
    }
    for secret in (
        "Indonesia export permit",
        "Singapore synthetic compliance notes",
        "Synthetic export permit guidance",
        "fixture-policy-idn",
    ):
        assert secret not in stdout


def test_synthetic_lexical_returns_full_candidate_limit_and_evaluates_recall_cutoff(tmp_path, capsys):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "candidate-window",
                "query": {"text": "candidate token", "limit": 1, "candidate_limit": 3},
                "relevance": [{"document_id": "fixture-doc-c", "grade": 1}],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_lines = [
        {
            "schema_version": 1,
            "document_id": f"fixture-doc-{suffix}",
            "document_kind": "source",
            "path": f"/fixture-{suffix}.md",
            "title": "Synthetic Candidate",
            "chunks": [{"chunk_index": 0, "content": "candidate token"}],
        }
        for suffix in ("a", "b", "c")
    ]
    (tmp_path / "corpus.jsonl").write_text(
        "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in corpus_lines),
        encoding="utf-8",
    )
    case = load_cases(dataset)[0]
    retriever = _default_retriever_factory("lexical", dataset)

    result = asyncio.run(retriever.retrieve(case.query))
    code = main(["--dataset", str(dataset), "--profile", "lexical"])
    payload, _stdout = _read_json(capsys)

    assert result.candidate_count == 3
    assert [hit.document_id for hit in result.hits] == [
        "fixture-doc-a",
        "fixture-doc-b",
        "fixture-doc-c",
    ]
    assert code == 0
    assert payload["metrics"]["recall_at_5"] == 1.0
    assert payload["metrics"]["mrr"] == 0.333333333333


@pytest.mark.parametrize(
    ("path_glob", "path", "expected"),
    [
        ("/", "/target/file.md", True),
        ("/target/", "/target/file.md", True),
        ("/target/", "/target/nested/file.md", True),
        ("/target", "/target/file.md", True),
        ("/target", "/targeted/file.md", False),
        ("/target/*.md", "/target/nested/file.md", True),
        ("/target/**/*.md", "/target/nested/file.md", True),
        ("/target/**/*.md", "/target/file.md", False),
        ("/target/100%_ready?.md", "/target/100%_ready?.md", True),
        ("/target/100%_ready?.md", "/target/100XXready1.md", False),
    ],
)
def test_synthetic_glob_matches_task4_native_contract(path_glob, path, expected):
    assert _logical_glob_matches(path_glob, path) is expected
