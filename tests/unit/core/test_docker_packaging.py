from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_hosted_compose_builds_api_and_mcp_from_repo_root():
    compose = (ROOT / "deploy/docker-compose.selfhost.yml").read_text(encoding="utf-8")
    assert "context: ..\n      dockerfile: api/Dockerfile" in compose
    assert "context: ..\n      dockerfile: mcp/Dockerfile" in compose


def test_all_python_images_install_core_package():
    for dockerfile in ("api/Dockerfile", "mcp/Dockerfile", "Dockerfile.local"):
        text = (ROOT / dockerfile).read_text(encoding="utf-8")
        assert "pip install --no-deps" in text and "llmwiki_core" in text
