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
        assert f"{command} | xargs" in workflow


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
