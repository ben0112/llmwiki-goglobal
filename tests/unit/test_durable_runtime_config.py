import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from config import Settings

_MCP_CONFIG_PATH = Path(__file__).parents[2] / "mcp" / "config.py"
_MCP_SPEC = importlib.util.spec_from_file_location("test_mcp_embedding_config", _MCP_CONFIG_PATH)
assert _MCP_SPEC is not None and _MCP_SPEC.loader is not None
_MCP_CONFIG = importlib.util.module_from_spec(_MCP_SPEC)
_MCP_SPEC.loader.exec_module(_MCP_CONFIG)
McpSettings = _MCP_CONFIG.Settings

RUNTIME_ENVIRONMENT = (
    "REDIS_URL",
    "DURABLE_JOBS_ENABLED",
    "TUS_MULTIPART_ENABLED",
    "JOB_LEASE_SECONDS",
    "JOB_HEARTBEAT_SECONDS",
    "JOB_DISPATCH_BATCH_SIZE",
    "JOB_REDELIVER_SECONDS",
    "TUS_SESSION_TTL_SECONDS",
    "TUS_STALE_SECONDS",
    "TUS_LOCK_SECONDS",
    "TUS_MAX_PATCH_BYTES",
    "HYBRID_SEARCH_ENABLED",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_TIMEOUT_SECONDS",
    "HYBRID_LEXICAL_CANDIDATES",
    "HYBRID_VECTOR_CANDIDATES",
    "HYBRID_RRF_K",
)


def _clear_runtime_environment(monkeypatch) -> None:
    for key in RUNTIME_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_local_mode_does_not_require_redis(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")

    settings = _settings()

    assert settings.REDIS_URL is None


def test_local_mode_ignores_durable_flags_for_redis_requirements(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "true")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")

    settings = _settings()

    assert settings.REDIS_URL is None


def test_local_mode_ignores_tus_dependency_on_durable_jobs(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "false")

    settings = _settings()

    assert settings.REDIS_URL is None
    assert settings.TUS_MULTIPART_ENABLED is True
    assert settings.DURABLE_JOBS_ENABLED is False


def test_hosted_durable_jobs_requires_redis(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "true")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")

    with pytest.raises(ValueError, match="REDIS_URL"):
        _settings()


def test_hosted_false_durable_jobs_flag_fails_fast(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "false")

    with pytest.raises(ValueError, match="DURABLE_JOBS_ENABLED"):
        _settings()


def test_hosted_tus_requires_redis(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "true")

    with pytest.raises(ValueError, match="REDIS_URL"):
        _settings()


def test_hosted_false_multipart_flag_fails_fast(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "true")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "false")

    with pytest.raises(ValueError, match="TUS_MULTIPART_ENABLED"):
        _settings()


@pytest.mark.parametrize(
    "name",
    (
        "JOB_LEASE_SECONDS",
        "JOB_HEARTBEAT_SECONDS",
        "JOB_DISPATCH_BATCH_SIZE",
        "JOB_REDELIVER_SECONDS",
        "TUS_SESSION_TTL_SECONDS",
        "TUS_STALE_SECONDS",
        "TUS_LOCK_SECONDS",
        "TUS_MAX_PATCH_BYTES",
    ),
)
def test_runtime_limits_must_be_positive(monkeypatch, name):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValueError, match=name):
        _settings()


@pytest.mark.parametrize(
    ("overrides", "error_field"),
    (
        (
            {"JOB_LEASE_SECONDS": "30", "JOB_HEARTBEAT_SECONDS": "30"},
            "JOB_HEARTBEAT_SECONDS",
        ),
        (
            {"JOB_LEASE_SECONDS": "30", "JOB_HEARTBEAT_SECONDS": "31"},
            "JOB_HEARTBEAT_SECONDS",
        ),
        (
            {"TUS_SESSION_TTL_SECONDS": "30", "TUS_STALE_SECONDS": "30"},
            "TUS_STALE_SECONDS",
        ),
        (
            {"TUS_SESSION_TTL_SECONDS": "30", "TUS_STALE_SECONDS": "31"},
            "TUS_STALE_SECONDS",
        ),
        (
            {"TUS_STALE_SECONDS": "30", "TUS_LOCK_SECONDS": "30"},
            "TUS_LOCK_SECONDS",
        ),
        (
            {"TUS_STALE_SECONDS": "30", "TUS_LOCK_SECONDS": "31"},
            "TUS_LOCK_SECONDS",
        ),
    ),
)
def test_runtime_timing_relationships_are_strict(monkeypatch, overrides, error_field):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=error_field):
        _settings()


def test_positive_runtime_limits_are_accepted(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")
    expected = {
        "JOB_LEASE_SECONDS": 120,
        "JOB_HEARTBEAT_SECONDS": 30,
        "JOB_DISPATCH_BATCH_SIZE": 100,
        "JOB_REDELIVER_SECONDS": 30,
        "TUS_SESSION_TTL_SECONDS": 172800,
        "TUS_STALE_SECONDS": 86400,
        "TUS_LOCK_SECONDS": 60,
        "TUS_MAX_PATCH_BYTES": 67108864,
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, str(value))

    settings = _settings()

    for name, value in expected.items():
        assert getattr(settings, name) == value


def test_runtime_defaults_in_a_clean_environment(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")

    settings = _settings()

    assert settings.REDIS_URL is None
    assert settings.DURABLE_JOBS_ENABLED is False
    assert settings.TUS_MULTIPART_ENABLED is False
    assert settings.JOB_LEASE_SECONDS == 120
    assert settings.JOB_HEARTBEAT_SECONDS == 30
    assert settings.JOB_DISPATCH_BATCH_SIZE == 100
    assert settings.JOB_REDELIVER_SECONDS == 30
    assert settings.TUS_SESSION_TTL_SECONDS == 172800
    assert settings.TUS_STALE_SECONDS == 86400
    assert settings.TUS_LOCK_SECONDS == 60
    assert settings.TUS_MAX_PATCH_BYTES == 67108864


def test_lexical_defaults_do_not_require_embedding_endpoint(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")

    runtime = _settings()

    assert runtime.HYBRID_SEARCH_ENABLED is False
    assert runtime.EMBEDDING_PROVIDER == "openai_compatible"
    assert runtime.EMBEDDING_BASE_URL == ""
    assert runtime.EMBEDDING_API_KEY.get_secret_value() == ""
    assert runtime.EMBEDDING_MODEL == ""
    assert runtime.EMBEDDING_DIMENSIONS == 0
    assert runtime.EMBEDDING_BATCH_SIZE == 32
    assert runtime.EMBEDDING_TIMEOUT_SECONDS == 30
    assert runtime.HYBRID_LEXICAL_CANDIDATES == 50
    assert runtime.HYBRID_VECTOR_CANDIDATES == 50
    assert runtime.HYBRID_RRF_K == 60
    assert runtime.embedding_profile is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"MODE": "local"}, "hosted"),
        ({"DURABLE_JOBS_ENABLED": "false"}, "DURABLE_JOBS_ENABLED"),
        ({"EMBEDDING_BASE_URL": ""}, "EMBEDDING_BASE_URL"),
        ({"EMBEDDING_MODEL": ""}, "EMBEDDING_MODEL"),
        ({"EMBEDDING_DIMENSIONS": "0"}, "EMBEDDING_DIMENSIONS"),
        ({"EMBEDDING_DIMENSIONS": "4097"}, "EMBEDDING_DIMENSIONS"),
        ({"HYBRID_LEXICAL_CANDIDATES": "0"}, "HYBRID_LEXICAL_CANDIDATES"),
        ({"HYBRID_VECTOR_CANDIDATES": "501"}, "HYBRID_VECTOR_CANDIDATES"),
    ),
)
def test_hybrid_settings_fail_closed(monkeypatch, overrides, match):
    _clear_runtime_environment(monkeypatch)
    valid = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "REDIS_URL": "redis://localhost:6379/0",
        "HYBRID_SEARCH_ENABLED": "true",
        "EMBEDDING_BASE_URL": "https://embedding.test/v1",
        "EMBEDDING_MODEL": "embed-v1",
        "EMBEDDING_DIMENSIONS": "3",
    }
    valid.update(overrides)
    for name, value in valid.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match):
        _settings()


def test_hybrid_profile_and_request_candidate_window(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "REDIS_URL": "redis://localhost:6379/0",
        "HYBRID_SEARCH_ENABLED": "true",
        "EMBEDDING_BASE_URL": "https://embedding.test/v1",
        "EMBEDDING_API_KEY": "private-key",
        "EMBEDDING_MODEL": "embed-v1",
        "EMBEDDING_DIMENSIONS": "3",
        "HYBRID_LEXICAL_CANDIDATES": "50",
        "HYBRID_VECTOR_CANDIDATES": "60",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    runtime = _settings()

    assert runtime.embedding_profile.identity == ("openai_compatible", "embed-v1", 3)
    assert "private-key" not in repr(runtime.embedding_profile)
    assert runtime.hybrid_candidate_limits(50) == (50, 60)
    with pytest.raises(ValueError, match="request limit"):
        runtime.hybrid_candidate_limits(51)
    with pytest.raises(ValueError, match="request limit"):
        runtime.hybrid_candidate_limits(True)


@pytest.mark.parametrize("settings_type", (Settings, McpSettings))
def test_api_and_mcp_embedding_defaults_have_identical_semantics(settings_type):
    runtime = settings_type(MODE="local", _env_file=None)

    assert runtime.HYBRID_SEARCH_ENABLED is False
    assert runtime.embedding_profile is None
    assert runtime.EMBEDDING_TIMEOUT_SECONDS == 30
    assert runtime.hybrid_candidate_limits(50) == (50, 50)


@pytest.mark.parametrize("settings_type", (Settings, McpSettings))
@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"MODE": "local"}, "hosted"),
        ({"DURABLE_JOBS_ENABLED": False}, "DURABLE_JOBS_ENABLED"),
        ({"EMBEDDING_BASE_URL": ""}, "EMBEDDING_BASE_URL"),
        ({"EMBEDDING_MODEL": ""}, "EMBEDDING_MODEL"),
        ({"EMBEDDING_DIMENSIONS": 0}, "EMBEDDING_DIMENSIONS"),
    ),
)
def test_api_and_mcp_hybrid_preconditions_have_identical_semantics(settings_type, overrides, match):
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "TUS_MULTIPART_ENABLED": True,
        "REDIS_URL": "redis://localhost:6379/0",
        "HYBRID_SEARCH_ENABLED": True,
        "EMBEDDING_BASE_URL": "https://embedding.test/v1",
        "EMBEDDING_MODEL": "embed-v1",
        "EMBEDDING_DIMENSIONS": 3,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=match):
        settings_type(_env_file=None, **values)


@pytest.mark.parametrize("settings_type", (Settings, McpSettings))
@pytest.mark.parametrize(
    "field",
    (
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_TIMEOUT_SECONDS",
        "HYBRID_LEXICAL_CANDIDATES",
        "HYBRID_VECTOR_CANDIDATES",
        "HYBRID_RRF_K",
    ),
)
def test_embedding_numeric_settings_reject_booleans(settings_type, field):
    with pytest.raises(ValueError, match=field):
        settings_type(MODE="local", _env_file=None, **{field: True})


@pytest.mark.parametrize("settings_type", (Settings, McpSettings))
def test_embedding_api_key_is_hidden_from_settings_repr(settings_type):
    secret = "sk-private-settings-repr"

    runtime = settings_type(MODE="local", EMBEDDING_API_KEY=secret, _env_file=None)

    assert runtime.EMBEDDING_API_KEY.get_secret_value() == secret
    assert secret not in repr(runtime)


@pytest.mark.parametrize("settings_type", (Settings, McpSettings))
def test_embedding_secrets_are_hidden_from_validation_errors_and_stderr(settings_type, capsys):
    secret = "sk-private-validation-input"

    with pytest.raises(ValueError) as caught:
        settings_type(
            MODE="local",
            EMBEDDING_API_KEY=secret,
            EMBEDDING_DIMENSIONS=secret,
            _env_file=None,
        )

    print(caught.value, file=sys.stderr)
    rendered = f"{caught.value!r}\n{caught.value}\n{capsys.readouterr().err}"
    assert secret not in rendered


def test_redis_module_does_not_create_client_on_import(monkeypatch):
    from redis.asyncio import Redis

    from_url = Mock()
    monkeypatch.setattr(Redis, "from_url", from_url)
    sys.modules.pop("infra.redis", None)

    importlib.import_module("infra.redis")

    from_url.assert_not_called()


def test_create_redis_configures_expected_client(monkeypatch):
    import infra.redis as redis

    expected_client = object()
    from_url = Mock(return_value=expected_client)
    monkeypatch.setattr(redis.Redis, "from_url", from_url)

    client = redis.create_redis("redis://localhost:6379/0")

    assert client is expected_client
    from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=False,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
