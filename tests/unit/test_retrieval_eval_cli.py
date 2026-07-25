import asyncio
import json
import os
import subprocess
import sys
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
REPO_ROOT = Path(__file__).parents[2]


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


def _secret_base_exception_group(*, process_control=None):
    leaves = [
        asyncio.CancelledError("cancelled-leaf-secret"),
        GeneratorExit("generator-leaf-secret"),
        RuntimeError("runtime-leaf-secret"),
    ]
    if process_control is not None:
        leaves.insert(1, process_control)
    return BaseExceptionGroup("backend-group-secret", leaves)


class _FactoryAttributeGroup:
    def __getattribute__(self, name):
        if name == "__call__":
            raise _secret_base_exception_group()
        return super().__getattribute__(name)

    def __call__(self, profile, _path):
        return _FakeRetriever(profile, _profile_results(profile))


class _RetrieveAttributeGroup:
    @property
    def retrieve(self):
        raise _secret_base_exception_group()


class _SyncGroupRetriever:
    def retrieve(self, _query):
        raise _secret_base_exception_group()


class _AsyncGroupRetriever:
    async def retrieve(self, _query):
        raise _secret_base_exception_group()


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


def _run_cli_with_closed_stream(
    argv: list[str],
    *,
    stream: str,
) -> subprocess.CompletedProcess[bytes]:
    read_descriptor, write_descriptor = os.pipe()
    os.close(read_descriptor)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "api")
    streams = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    streams[stream] = write_descriptor
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "scripts.retrieval_eval", *argv],
            cwd=REPO_ROOT,
            env=environment,
            stdout=streams["stdout"],
            stderr=streams["stderr"],
        )
    finally:
        os.close(write_descriptor)
    stdout, stderr = process.communicate(timeout=10)
    return subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout or b"",
        stderr or b"",
    )


def _run_cli_with_missing_descriptor(
    argv: list[str],
    *,
    descriptor: int,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "api")
    process = subprocess.Popen(
        [sys.executable, "-m", "scripts.retrieval_eval", *argv],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: os.close(descriptor),
    )
    stdout, stderr = process.communicate(timeout=10)
    return subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout,
        stderr,
    )


class _EmissionFailingBuffer:
    def __init__(self, failure: str) -> None:
        self.content = bytearray()
        self.failure = failure

    def write(self, content: bytes) -> int:
        if self.failure == "write":
            raise BrokenPipeError("private immediate write failure")
        self.content.extend(content)
        return len(content)

    def flush(self) -> None:
        if self.failure == "flush":
            raise BrokenPipeError("private delayed flush failure")

    def seek(self, offset: int) -> int:
        assert offset == 0
        return 0

    def truncate(self, size: int = 0) -> int:
        del self.content[size:]
        return size


class _EmissionFailingStream:
    def __init__(self, failure: str) -> None:
        self.buffer = _EmissionFailingBuffer(failure)


class _ScriptedEmissionBuffer:
    def __init__(self, actions: list[object]) -> None:
        self.actions = list(actions)
        self.content = bytearray()
        self.flush_count = 0

    def write(self, content: memoryview) -> object:
        action = self.actions.pop(0) if self.actions else len(content)
        if isinstance(action, BaseException):
            raise action
        result = len(content) if action == "all" else action
        if type(result) is int and 0 < result <= len(content):
            self.content.extend(content[:result])
        return result

    def flush(self) -> None:
        self.flush_count += 1

    def seek(self, offset: int) -> int:
        assert offset == 0
        return 0

    def truncate(self, size: int = 0) -> int:
        del self.content[size:]
        return size


class _ScriptedEmissionStream:
    def __init__(self, actions: list[object]) -> None:
        self.buffer = _ScriptedEmissionBuffer(actions)


class _ExceptionalEmissionBuffer:
    def __init__(self, location: str, error_type: type[Exception]) -> None:
        self.location = location
        self.error_type = error_type
        self.content = bytearray()

    def write(self, content: memoryview) -> int:
        if self.location == "write":
            raise self.error_type("private write failure")
        self.content.extend(content)
        return len(content)

    def flush(self) -> None:
        if self.location == "flush":
            raise self.error_type("private flush failure")

    def seek(self, offset: int) -> int:
        assert offset == 0
        return 0

    def truncate(self, size: int = 0) -> int:
        del self.content[size:]
        return size


class _ExceptionalEmissionStream:
    def __init__(self, location: str, error_type: type[Exception]) -> None:
        self.location = location
        self.error_type = error_type
        self._buffer = _ExceptionalEmissionBuffer(location, error_type)

    @property
    def buffer(self) -> _ExceptionalEmissionBuffer:
        if self.location == "buffer":
            raise self.error_type("private buffer access failure")
        return self._buffer


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


def test_real_closed_stdout_pipe_returns_four_without_shutdown_diagnostics():
    completed = _run_cli_with_closed_stream(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        stream="stdout",
    )

    assert completed.returncode == 4
    assert completed.stdout == b""
    assert b"Exception ignored" not in completed.stderr
    assert b"Traceback" not in completed.stderr
    assert b"BrokenPipeError" not in completed.stderr


def test_real_closed_stdout_help_returns_four_without_shutdown_diagnostics():
    completed = _run_cli_with_closed_stream(["--help"], stream="stdout")

    assert completed.returncode == 4
    assert completed.stdout == b""
    assert b"Exception ignored" not in completed.stderr
    assert b"Traceback" not in completed.stderr
    assert b"BrokenPipeError" not in completed.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset", str(DATASET), "--profile", "lexical"],
        ["--help"],
    ],
    ids=["report", "help"],
)
def test_missing_stdout_descriptor_returns_four_without_traceback(argv):
    completed = _run_cli_with_missing_descriptor(argv, descriptor=1)

    assert completed.returncode == 4
    assert completed.stdout == b""
    assert b"Traceback" not in completed.stderr
    assert b"AttributeError" not in completed.stderr
    assert b"TypeError" not in completed.stderr


def test_real_closed_stderr_pipe_preserves_invalid_arguments_exit_code():
    completed = _run_cli_with_closed_stream([], stream="stderr")

    assert completed.returncode == 2
    assert completed.stdout == b""


def test_missing_stderr_descriptor_preserves_invalid_arguments_exit_code():
    completed = _run_cli_with_missing_descriptor([], descriptor=2)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr == b""


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_stdout_emission_failure_without_file_descriptor_returns_four_and_clears_buffer(
    failure,
    monkeypatch,
):
    stream = _EmissionFailingStream(failure)
    factory, _retrievers, _factory_calls = _factory()
    monkeypatch.setattr(retrieval_eval_module.sys, "stdout", stream)

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )

    assert code == 4
    assert stream.buffer.content == b""


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_stderr_emission_failure_without_file_descriptor_preserves_business_exit_code(
    failure,
    monkeypatch,
):
    stream = _EmissionFailingStream(failure)
    monkeypatch.setattr(retrieval_eval_module.sys, "stderr", stream)

    code = main([])

    assert code == 2
    assert stream.buffer.content == b""


@pytest.mark.parametrize(
    "actions",
    [
        [1, 1, 1, 1, 1, 1],
        [2, 1, "all"],
    ],
    ids=["one-byte-at-a-time", "segmented"],
)
def test_emit_stream_retries_partial_writes_until_all_bytes_are_flushed(actions):
    stream = _ScriptedEmissionStream(actions)

    emitted = retrieval_eval_module._emit_stream(stream, b"abcdef")

    assert emitted is True
    assert stream.buffer.content == b"abcdef"
    assert stream.buffer.flush_count == 1


@pytest.mark.parametrize(
    "invalid_result",
    [None, 0, -1, True, False, "1", 7],
    ids=["none", "zero", "negative", "true", "false", "non-integer", "too-large"],
)
def test_emit_stream_rejects_invalid_write_results_without_partial_success(invalid_result):
    stream = _ScriptedEmissionStream([invalid_result])

    emitted = retrieval_eval_module._emit_stream(stream, b"abcdef")

    assert emitted is False
    assert stream.buffer.content == b""
    assert stream.buffer.flush_count == 0


def test_emit_stream_clears_partial_content_when_later_write_raises():
    stream = _ScriptedEmissionStream([2, BrokenPipeError("private second-write failure")])

    emitted = retrieval_eval_module._emit_stream(stream, b"abcdef")

    assert emitted is False
    assert stream.buffer.content == b""
    assert stream.buffer.flush_count == 0


def test_stdout_partial_then_invalid_write_returns_four_without_partial_success(monkeypatch):
    stream = _ScriptedEmissionStream([1, 0])
    factory, _retrievers, _factory_calls = _factory()
    monkeypatch.setattr(retrieval_eval_module.sys, "stdout", stream)

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )

    assert code == 4
    assert stream.buffer.content == b""


def test_stderr_partial_then_invalid_write_preserves_business_exit_code(monkeypatch):
    stream = _ScriptedEmissionStream([1, None])
    monkeypatch.setattr(retrieval_eval_module.sys, "stderr", stream)

    code = main([])

    assert code == 2
    assert stream.buffer.content == b""


def test_help_uses_normal_parser_format(capsys):
    expected = retrieval_eval_module._parser().format_help()

    code = main(["--help"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out == expected
    assert captured.err == ""


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_help_stdout_emission_failure_returns_four_and_clears_buffer(failure, monkeypatch):
    stream = _EmissionFailingStream(failure)
    monkeypatch.setattr(retrieval_eval_module.sys, "stdout", stream)

    code = main(["--help"])

    assert code == 4
    assert stream.buffer.content == b""


@pytest.mark.parametrize("location", ["buffer", "write", "flush"])
@pytest.mark.parametrize("error_type", [AttributeError, TypeError])
def test_stdout_stream_shape_failures_return_four_without_partial_success(
    location,
    error_type,
    monkeypatch,
):
    stream = _ExceptionalEmissionStream(location, error_type)
    factory, _retrievers, _factory_calls = _factory()
    monkeypatch.setattr(retrieval_eval_module.sys, "stdout", stream)

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical"],
        retriever_factory=factory,
    )

    assert code == 4
    assert stream._buffer.content == b""


@pytest.mark.parametrize("location", ["buffer", "write", "flush"])
@pytest.mark.parametrize("error_type", [AttributeError, TypeError])
def test_stderr_stream_shape_failures_preserve_business_exit_code(
    location,
    error_type,
    monkeypatch,
):
    stream = _ExceptionalEmissionStream(location, error_type)
    monkeypatch.setattr(retrieval_eval_module.sys, "stderr", stream)

    code = main([])

    assert code == 2
    assert stream._buffer.content == b""


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


@pytest.mark.parametrize(
    ("process_control", "expected_type", "expected_args"),
    [
        (KeyboardInterrupt("direct-process-control-secret"), KeyboardInterrupt, ()),
        (SystemExit(7), SystemExit, (7,)),
        (SystemExit("direct-process-control-secret"), SystemExit, (1,)),
        (SystemExit(True), SystemExit, (1,)),
        (SystemExit(False), SystemExit, (0,)),
    ],
)
def test_direct_process_control_is_rethrown_without_backend_secrets(
    process_control,
    expected_type,
    expected_args,
):
    def factory(_profile, _path):
        raise process_control

    with pytest.raises(expected_type) as caught:
        main(
            ["--dataset", str(DATASET), "--profile", "lexical"],
            retriever_factory=factory,
        )

    assert caught.value.args == expected_args
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("process_control", "expected_type", "expected_args"),
    [
        (KeyboardInterrupt("process-control-leaf-secret"), KeyboardInterrupt, ()),
        (SystemExit(7), SystemExit, (7,)),
        (SystemExit("process-control-leaf-secret"), SystemExit, (1,)),
    ],
)
def test_grouped_process_control_is_rethrown_without_secret_group(
    process_control,
    expected_type,
    expected_args,
):
    def factory(_profile, _path):
        raise _secret_base_exception_group(process_control=process_control)

    with pytest.raises(expected_type) as caught:
        main(
            ["--dataset", str(DATASET), "--profile", "lexical"],
            retriever_factory=factory,
        )

    assert caught.value.args == expected_args
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "factory",
    [
        _FactoryAttributeGroup(),
        lambda _profile, _path: (_ for _ in ()).throw(_secret_base_exception_group()),
        lambda _profile, _path: _RetrieveAttributeGroup(),
        lambda _profile, _path: _SyncGroupRetriever(),
        lambda _profile, _path: _AsyncGroupRetriever(),
    ],
)
def test_backend_base_exception_groups_are_sanitized_at_every_boundary(factory, capsys):
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
        "backend-group-secret",
        "cancelled-leaf-secret",
        "generator-leaf-secret",
        "runtime-leaf-secret",
        "Indonesia export permit",
        "Traceback",
    ):
        assert secret not in captured.err


def test_final_async_runner_base_exception_group_is_sanitized(monkeypatch, capsys):
    async def fail_after_retrieval(*_args):
        raise _secret_base_exception_group()

    factory, _retrievers, _factory_calls = _factory()
    monkeypatch.setattr(retrieval_eval_module, "_evaluate_profile", fail_after_retrieval)

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
    assert "backend-group-secret" not in captured.err
    assert "runtime-leaf-secret" not in captured.err


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


@pytest.mark.parametrize("absolute", [False, True])
def test_output_refuses_relative_and_absolute_ancestor_symlink(
    absolute,
    tmp_path,
    monkeypatch,
    capsys,
):
    real_parent = tmp_path / "private-target"
    nested_target = real_parent / "nested"
    nested_target.mkdir(parents=True)
    symlink_ancestor = tmp_path / "customer-secret-ancestor"
    symlink_ancestor.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    output_path = symlink_ancestor / "nested" / "report.json"
    output_argument = str(output_path if absolute else output_path.relative_to(tmp_path))
    factory, _retrievers, _factory_calls = _factory()
    descriptors_before = len(os.listdir("/dev/fd"))

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", output_argument],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "output", "code": "output_write_failed"}
    }
    assert not (nested_target / "report.json").exists()
    assert not list(real_parent.rglob(".*.tmp"))
    assert len(os.listdir("/dev/fd")) == descriptors_before
    assert "customer-secret-ancestor" not in captured.err


def test_output_walks_normal_relative_multilevel_directory_without_fd_or_temp_leak(
    tmp_path,
    monkeypatch,
    capsys,
):
    output_parent = tmp_path / "safe" / "nested" / "directory"
    output_parent.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    factory, _retrievers, _factory_calls = _factory()
    descriptors_before = len(os.listdir("/dev/fd"))

    code = main(
        [
            "--dataset",
            str(DATASET),
            "--profile",
            "lexical",
            "--output-json",
            "safe/nested/directory/report.json",
        ],
        retriever_factory=factory,
    )
    _payload, stdout = _read_json(capsys)

    assert code == 0
    assert (output_parent / "report.json").read_bytes() == stdout.encode("utf-8")
    assert not list(output_parent.glob(".*.tmp"))
    assert len(os.listdir("/dev/fd")) == descriptors_before


@pytest.mark.parametrize("output_argument", ["safe/./dot-secret.json", "safe/../dotdot-secret.json"])
def test_output_rejects_raw_dot_and_dotdot_components(output_argument, tmp_path, monkeypatch, capsys):
    (tmp_path / "safe").mkdir()
    monkeypatch.chdir(tmp_path)
    factory, _retrievers, _factory_calls = _factory()

    code = main(
        ["--dataset", str(DATASET), "--profile", "lexical", "--output-json", output_argument],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"category": "output", "code": "output_write_failed"}
    }
    assert not (tmp_path / "safe" / "dot-secret.json").exists()
    assert not (tmp_path / "dotdot-secret.json").exists()
    assert "secret" not in captured.err


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
