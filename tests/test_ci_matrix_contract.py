from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/test.yml"


def test_workflow_uses_complete_isolated_test_matrix_segments():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "mapfile" not in workflow
    for segment in (
        "unit-core",
        "unit-api",
        "unit-mcp",
        "unit-corpus",
        "integration-api",
        "integration-mcp",
        "integration-mcp-postgres",
        "integration-redis",
        "integration-minio",
        "integration-scaled",
    ):
        command = f"python -m tests.helpers.ci_test_matrix {segment}"
        assert command in workflow
        if segment in {"integration-api", "integration-mcp-postgres"}:
            assert f"{command} |" in workflow
            assert "while IFS= read -r test_file; do" in workflow
            assert 'PYTHONPATH=api MODE=hosted pytest "$test_file" -v' in workflow
        else:
            assert f"{command} | xargs" in workflow


def test_api_integration_files_use_fresh_pytest_session_fixtures():
    """Each file must get the session fixture that recreates the Postgres schema."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Run complete isolated API integration matrix", 1)[1]
    step = step.split("- name:", 1)[0]

    assert "set -o pipefail" in step
    assert "while IFS= read -r test_file; do" in step
    assert 'pytest "$test_file"' in step
    assert "xargs env PYTHONPATH=api MODE=hosted pytest" not in step


def test_mcp_postgres_files_load_module_plugins_in_separate_pytest_processes():
    """Collecting a fixture-provider module and its consumer together hides the plugin fixtures."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Run MCP Postgres isolation tests", 1)[1]
    step = step.split("\n\n", 1)[0]

    assert "set -o pipefail" in step
    assert "while IFS= read -r test_file; do" in step
    assert 'PYTHONPATH=mcp pytest "$test_file" -v' in step
    assert "xargs env PYTHONPATH=mcp pytest" not in step


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
