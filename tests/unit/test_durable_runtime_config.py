import importlib
import sys
from unittest.mock import Mock

import pytest
from config import Settings

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

    with pytest.raises(ValueError, match="REDIS_URL"):
        _settings()


def test_tus_multipart_requires_durable_jobs(monkeypatch):
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
