from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/test.yml"


def test_workflow_uses_complete_isolated_test_matrix_segments():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "mapfile" not in workflow
    assert "| xargs" not in workflow
    assert "ci_test_matrix integration-api |" not in workflow
    assert "ci_test_matrix integration-mcp-postgres |" not in workflow
    for segment in (
        "unit-core",
        "unit-api",
        "unit-mcp",
        "unit-corpus",
        "integration-api",
        "integration-mcp",
        "integration-mcp-postgres",
        "integration-retrieval",
        "integration-redis",
        "integration-minio",
        "integration-scaled",
    ):
        command = f"python -m tests.helpers.ci_test_matrix run {segment} --"
        assert command in workflow


def test_retrieval_evaluation_has_a_dedicated_pgvector_segment():
    from tests.helpers.ci_test_matrix import SEGMENTS

    assert SEGMENTS["integration-retrieval"] == (
        "tests/integration/test_chunk_embeddings_schema.py",
        "tests/integration/test_durable_embeddings.py",
        "tests/integration/test_hybrid_failure_matrix.py",
        "tests/integration/test_retrieval_evaluation.py",
        "tests/integration/test_vector_store.py",
    )
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Run deterministic retrieval promotion gate", 1)[1]
    step = step.split("- name:", 1)[0]

    assert "pgvector/pgvector:0.8.0-pg16" in workflow
    assert "python -m tests.helpers.ci_test_matrix run integration-retrieval --" in step
    assert "env PYTHONPATH=api MODE=hosted pytest -v" in step
    assert "|| true" not in step
    assert "|" not in step


def test_api_integration_files_use_fresh_pytest_session_fixtures():
    """Each file must get the session fixture that recreates the Postgres schema."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Run complete isolated API integration matrix", 1)[1]
    step = step.split("- name:", 1)[0]

    assert "python -m tests.helpers.ci_test_matrix run integration-api --" in step
    assert "env PYTHONPATH=api MODE=hosted pytest -v" in step
    assert "|" not in step


def test_mcp_postgres_files_load_module_plugins_in_separate_pytest_processes():
    """Collecting a fixture-provider module and its consumer together hides the plugin fixtures."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Run MCP Postgres isolation tests", 1)[1]
    step = step.split("\n\n", 1)[0]

    assert "python -m tests.helpers.ci_test_matrix run integration-mcp-postgres --" in step
    assert "env PYTHONPATH=mcp pytest -v" in step
    assert "|" not in step


@pytest.mark.parametrize("failure", ["error", "empty"])
def test_run_command_fails_closed_before_launching_tests(monkeypatch, failure):
    from tests.helpers import ci_test_matrix

    launched = []

    def generate(_segment):
        if failure == "error":
            raise RuntimeError("generator failed")
        return ()

    def launch(*args, **kwargs):
        launched.append((args, kwargs))
        raise AssertionError("test command must not launch")

    monkeypatch.setattr(ci_test_matrix, "_segment_files", generate)
    monkeypatch.setattr(ci_test_matrix.subprocess, "run", launch)

    assert ci_test_matrix.main(["run", "unit-core", "--", "pytest", "-v"]) != 0
    assert launched == []


def test_run_command_propagates_child_failure_and_preserves_segment_strategy(monkeypatch):
    from tests.helpers import ci_test_matrix

    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    monkeypatch.setattr(
        ci_test_matrix,
        "_segment_files",
        lambda _segment: ("tests/one.py", "tests/two.py"),
    )

    def launch(command, check):
        calls.append((command, check))
        return Result(7 if len(calls) == 2 else 0)

    monkeypatch.setattr(ci_test_matrix.subprocess, "run", launch)

    assert (
        ci_test_matrix.main(
            ["run", "integration-api", "--", "env", "PYTHONPATH=api", "pytest", "-v"]
        )
        == 7
    )
    assert calls == [
        (["env", "PYTHONPATH=api", "pytest", "-v", "tests/one.py"], False),
        (["env", "PYTHONPATH=api", "pytest", "-v", "tests/two.py"], False),
    ]


def test_each_test_file_has_exactly_one_primary_ci_segment():
    from tests.helpers.ci_test_matrix import SEGMENTS

    expected = {
        path.relative_to(ROOT).as_posix()
        for suite in (ROOT / "tests/unit", ROOT / "tests/integration")
        for path in suite.rglob("test_*.py")
    }
    owners = {}
    for segment, paths in SEGMENTS.items():
        for path in paths:
            owners.setdefault(path, []).append(segment)

    assert set(owners) == expected
    assert {path: segments for path, segments in owners.items() if len(segments) != 1} == {}
