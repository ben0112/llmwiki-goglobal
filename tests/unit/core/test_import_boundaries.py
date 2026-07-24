import ast
from pathlib import Path

FORBIDDEN = {"fastapi", "mcp", "asyncpg", "aiosqlite", "aioboto3", "boto3"}


def test_core_imports_without_service_dependencies():
    import llmwiki_core

    assert llmwiki_core.__version__ == "0.1.0"


def test_core_source_does_not_import_infrastructure_packages():
    root = Path(__file__).parents[3] / "llmwiki_core"
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & FORBIDDEN:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
