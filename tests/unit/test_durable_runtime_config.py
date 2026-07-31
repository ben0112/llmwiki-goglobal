import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from config import Settings
from pydantic import TypeAdapter

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
    "SERVER_RAG_ENABLED",
    "RAG_MODEL_PROFILES_JSON",
    "RAG_MODEL_API_KEYS_JSON",
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
    assert settings.SERVER_RAG_ENABLED is False
    assert settings.RAG_MODEL_PROFILES_JSON.get_secret_value() == "{}"
    assert settings.RAG_MODEL_API_KEYS_JSON.get_secret_value() == "{}"


def test_disabled_rag_does_not_require_profiles_or_hosted_runtime(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv("MODE", "local")

    runtime = _settings()

    assert runtime.SERVER_RAG_ENABLED is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"MODE": "local"}, "hosted"),
        ({"DURABLE_JOBS_ENABLED": "false"}, "DURABLE_JOBS_ENABLED"),
        ({"RAG_MODEL_PROFILES_JSON": "{}", "RAG_MODEL_API_KEYS_JSON": "{}"}, "profile"),
    ),
)
def test_enabled_rag_requires_hosted_durable_runtime_and_complete_profile(monkeypatch, overrides, match):
    _clear_runtime_environment(monkeypatch)
    valid = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": "true",
        "RAG_MODEL_PROFILES_JSON": json.dumps(
            {
                "primary": {
                    "base_url": "https://models.test/v1",
                    "model": "writer-v1",
                    "timeout_seconds": 60,
                    "version": "2026-07-26",
                }
            }
        ),
        "RAG_MODEL_API_KEYS_JSON": '{"primary":"sk-private"}',
    }
    valid.update(overrides)
    for name, value in valid.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=match):
        _settings()


def test_enabled_rag_accepts_complete_hosted_profile(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": "true",
        "RAG_MODEL_PROFILES_JSON": '{"primary":{"base_url":"https://models.test/v1","model":"writer-v1","timeout_seconds":60,"version":"v1"}}',
        "RAG_MODEL_API_KEYS_JSON": '{"primary":"sk-private"}',
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    runtime = _settings()

    assert runtime.SERVER_RAG_ENABLED is True


def test_rag_settings_secrets_are_hidden_from_repr_dump_and_errors(monkeypatch, capsys):
    _clear_runtime_environment(monkeypatch)
    profiles = '{"private-profile":{"base_url":"https://private-model.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version"}}'
    keys = '{"private-profile":"sk-private-rag-key"}'
    runtime = Settings(
        MODE="local",
        RAG_MODEL_PROFILES_JSON=profiles,
        RAG_MODEL_API_KEYS_JSON=keys,
        _env_file=None,
    )

    dumped = json.dumps(runtime.model_dump(mode="json"), sort_keys=True)
    rendered = f"{runtime!r}\n{runtime}\n{dumped}"
    for private in ("private-model.test", "private-model", "private-version", "sk-private-rag-key"):
        assert private not in rendered

    with pytest.raises(ValueError) as caught:
        Settings(
            MODE="local",
            SERVER_RAG_ENABLED="not-a-bool",
            RAG_MODEL_PROFILES_JSON=profiles,
            RAG_MODEL_API_KEYS_JSON=keys,
            _env_file=None,
        )
    print(caught.value, file=sys.stderr)
    rendered_error = f"{caught.value!r}\n{caught.value}\n{capsys.readouterr().err}"
    for private in ("private-model.test", "private-model", "private-version", "sk-private-rag-key"):
        assert private not in rendered_error


def _assert_machine_readable_validation_error_is_redacted(error, private_values):
    rendered_parts = [str(error), repr(error), repr(error.__context__), repr(error.__cause__)]
    if hasattr(error, "json"):
        rendered_parts.append(error.json())
    if hasattr(error, "errors"):
        rendered_parts.append(json.dumps(error.errors(), default=repr, sort_keys=True))
    rendered = "\n".join(rendered_parts)
    for private in private_values:
        assert private not in rendered


def _assert_detached_settings_error(error, private_values):
    assert type(error).__name__ == "SettingsConfigurationError"
    assert isinstance(error, ValueError)
    assert error.__context__ is None
    assert error.__cause__ is None
    rendered = "\n".join((str(error), repr(error), repr(error.args), repr(vars(error))))
    pickled = pickle.dumps(error)
    for private in private_values:
        assert private not in rendered
        assert private.encode() not in pickled
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("/api/config.py"):
            frame_locals = repr(traceback.tb_frame.f_locals)
            for private in private_values:
                assert private not in frame_locals
        traceback = traceback.tb_next


def _private_rag_inputs():
    profiles = '{"private-profile":{"base_url":"https://private-url.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version","extra":"invalid"}}'
    keys = '{"private-profile":"sk-private-key"}'
    markers = ("private-profile", "private-url.test", "private-model", "private-version", "sk-private-key")
    return profiles, keys, markers


def test_direct_settings_failure_is_detached_from_rag_inputs():
    profiles, keys, markers = _private_rag_inputs()

    with pytest.raises(ValueError, match="configuration is invalid") as caught:
        Settings(
            MODE="hosted",
            DURABLE_JOBS_ENABLED=True,
            TUS_MULTIPART_ENABLED=True,
            REDIS_URL="redis://localhost:6379/0",
            SERVER_RAG_ENABLED=True,
            RAG_MODEL_PROFILES_JSON=profiles,
            RAG_MODEL_API_KEYS_JSON=keys,
            _env_file=None,
        )

    _assert_detached_settings_error(caught.value, markers)


def test_environment_settings_failure_is_detached_from_rag_inputs(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    profiles, keys, markers = _private_rag_inputs()
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv("JOB_LEASE_SECONDS", "30")
    monkeypatch.setenv("JOB_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("RAG_MODEL_PROFILES_JSON", profiles)
    monkeypatch.setenv("RAG_MODEL_API_KEYS_JSON", keys)

    with pytest.raises(ValueError, match="JOB_HEARTBEAT_SECONDS") as caught:
        _settings()

    _assert_detached_settings_error(caught.value, markers)


def test_model_validate_failure_is_detached_and_does_not_mutate_input():
    profiles, keys, markers = _private_rag_inputs()
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "TUS_MULTIPART_ENABLED": True,
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": True,
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }
    original = values.copy()

    with pytest.raises(ValueError, match="configuration is invalid") as caught:
        Settings.model_validate(values)

    assert values == original
    _assert_detached_settings_error(caught.value, markers)


@pytest.mark.parametrize("entrypoint", ("model_validate_json", "model_validate_strings"))
def test_serialized_validation_entrypoints_detach_rag_inputs(entrypoint):
    profiles, keys, markers = _private_rag_inputs()
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": "true",
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }
    supplied = json.dumps(values) if entrypoint == "model_validate_json" else values

    with pytest.raises(ValueError, match="configuration is invalid") as caught:
        getattr(Settings, entrypoint)(supplied)

    _assert_detached_settings_error(caught.value, markers)


@pytest.mark.parametrize("entrypoint", ("direct", "model_validate", "model_validate_json", "model_validate_strings"))
def test_unrelated_field_failures_are_detached_from_rag_inputs(entrypoint):
    profiles, keys, markers = _private_rag_inputs()
    values = {
        "MODE": "local",
        "JOB_LEASE_SECONDS": "not-an-integer",
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }
    with pytest.raises(ValueError, match="JOB_LEASE_SECONDS") as caught:
        if entrypoint == "direct":
            Settings(_env_file=None, **values)
        elif entrypoint == "model_validate_json":
            Settings.model_validate_json(json.dumps(values))
        else:
            getattr(Settings, entrypoint)(values)

    _assert_detached_settings_error(caught.value, markers)


@pytest.mark.parametrize("field", ("RAG_MODEL_PROFILES_JSON", "RAG_MODEL_API_KEYS_JSON"))
@pytest.mark.parametrize("invalid", (1, ["private-invalid-secret"], {"private": "invalid"}))
def test_disabled_rag_rejects_non_string_secret_inputs_through_safe_boundary(field, invalid):
    values = {"MODE": "local", field: invalid}
    original = values.copy()

    with pytest.raises(ValueError, match="configuration is invalid") as caught:
        Settings.model_validate(values)

    assert values == original
    _assert_detached_settings_error(caught.value, ("private-invalid-secret", "'private': 'invalid'"))


def test_type_adapter_validate_python_detaches_semantic_failure_and_preserves_input():
    profiles, keys, markers = _private_rag_inputs()
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "TUS_MULTIPART_ENABLED": True,
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": True,
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }
    original = values.copy()

    with pytest.raises(ValueError) as caught:
        TypeAdapter(Settings).validate_python(values)

    assert values == original
    _assert_detached_settings_error(caught.value, markers)


def test_type_adapter_validate_python_detaches_invalid_secret_type_when_rag_disabled():
    values = {
        "MODE": "local",
        "RAG_MODEL_PROFILES_JSON": {"private-profile": "private-model"},
    }
    original = values.copy()

    with pytest.raises(ValueError) as caught:
        TypeAdapter(Settings).validate_python(values)

    assert values == original
    _assert_detached_settings_error(caught.value, ("private-profile", "private-model"))


@pytest.mark.parametrize(
    "values",
    (
        {
            "MODE": "hosted",
            "DURABLE_JOBS_ENABLED": True,
            "TUS_MULTIPART_ENABLED": True,
            "REDIS_URL": "redis://localhost:6379/0",
            "SERVER_RAG_ENABLED": True,
            "RAG_MODEL_PROFILES_JSON": '{"private-profile":{"base_url":"https://private-url.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version","extra":"invalid"}}',
            "RAG_MODEL_API_KEYS_JSON": '{"private-profile":"sk-private-key"}',
        },
        {
            "MODE": "local",
            "RAG_MODEL_API_KEYS_JSON": ["sk-private-key"],
        },
    ),
)
def test_type_adapter_validate_json_detaches_semantic_and_type_failures(values):
    private_values = ("private-profile", "private-url.test", "private-model", "private-version", "sk-private-key")

    with pytest.raises(ValueError) as caught:
        TypeAdapter(Settings).validate_json(json.dumps(values))

    _assert_detached_settings_error(caught.value, private_values)


def test_type_adapter_validate_json_detaches_malformed_private_json():
    malformed = b'{"RAG_MODEL_PROFILES_JSON":"private-profile-private-model"'

    with pytest.raises(ValueError) as caught:
        TypeAdapter(Settings).validate_json(malformed)

    _assert_detached_settings_error(caught.value, ("private-profile", "private-model"))


def test_type_adapter_valid_paths_and_model_rebuild_keep_safe_validator():
    profiles = '{"primary":{"base_url":"https://models.test/v1","model":"writer-v1","timeout_seconds":60,"version":"v1"}}'
    keys = '{"primary":"sk-valid"}'
    values = {
        "MODE": "local",
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }
    assert Settings.model_rebuild(force=True) is True

    from_python = TypeAdapter(Settings).validate_python(values)
    from_json = TypeAdapter(Settings).validate_json(json.dumps(values))

    assert from_python.RAG_MODEL_PROFILES_JSON.get_secret_value() == profiles
    assert from_json.RAG_MODEL_API_KEYS_JSON.get_secret_value() == keys

    invalid = {**values, "RAG_MODEL_API_KEYS_JSON": ["sk-private-after-rebuild"]}
    with pytest.raises(ValueError) as caught:
        TypeAdapter(Settings).validate_python(invalid)
    _assert_detached_settings_error(caught.value, ("sk-private-after-rebuild",))


def test_direct_settings_core_validator_raises_public_sanitized_error():
    values = {
        "MODE": "local",
        "RAG_MODEL_PROFILES_JSON": {"private-profile": "private-model"},
    }
    original = values.copy()

    with pytest.raises(ValueError) as caught:
        Settings.__pydantic_validator__.validate_python(values)

    assert values == original
    _assert_detached_settings_error(caught.value, ("private-profile", "private-model"))


class ChildSettings(Settings):
    CHILD_LABEL: str = "child"


@pytest.mark.parametrize("entrypoint", ("python", "json", "malformed_json", "direct_core"))
def test_child_settings_validation_boundaries_are_public_and_sanitized(entrypoint):
    profiles, keys, markers = _private_rag_inputs()
    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "TUS_MULTIPART_ENABLED": True,
        "REDIS_URL": "redis://localhost:6379/0",
        "SERVER_RAG_ENABLED": True,
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
    }

    with pytest.raises(ValueError) as caught:
        if entrypoint == "python":
            TypeAdapter(ChildSettings).validate_python(values)
        elif entrypoint == "json":
            TypeAdapter(ChildSettings).validate_json(json.dumps(values))
        elif entrypoint == "malformed_json":
            TypeAdapter(ChildSettings).validate_json(
                b'{"RAG_MODEL_PROFILES_JSON":"private-profile-private-model"'
            )
        else:
            ChildSettings.__pydantic_validator__.validate_python(values)

    _assert_detached_settings_error(caught.value, markers)


def test_child_settings_valid_paths_and_rebuild_preserve_safe_validator():
    profiles = '{"primary":{"base_url":"https://models.test/v1","model":"writer-v1","timeout_seconds":60,"version":"v1"}}'
    keys = '{"primary":"sk-valid"}'
    values = {
        "MODE": "local",
        "RAG_MODEL_PROFILES_JSON": profiles,
        "RAG_MODEL_API_KEYS_JSON": keys,
        "CHILD_LABEL": "ready",
    }

    before_python = TypeAdapter(ChildSettings).validate_python(values)
    before_json = TypeAdapter(ChildSettings).validate_json(json.dumps(values))
    assert before_python.CHILD_LABEL == "ready"
    assert before_json.RAG_MODEL_API_KEYS_JSON.get_secret_value() == keys

    assert ChildSettings.model_rebuild(force=True) is True
    after_python = TypeAdapter(ChildSettings).validate_python(values)
    after_json = TypeAdapter(ChildSettings).validate_json(json.dumps(values))
    assert after_python.CHILD_LABEL == "ready"
    assert after_json.RAG_MODEL_PROFILES_JSON.get_secret_value() == profiles

    invalid = {**values, "RAG_MODEL_API_KEYS_JSON": ["sk-private-child-after-rebuild"]}
    with pytest.raises(ValueError) as caught:
        TypeAdapter(ChildSettings).validate_python(invalid)
    _assert_detached_settings_error(caught.value, ("sk-private-child-after-rebuild",))


def test_rag_semantic_validation_error_masks_lowercase_direct_inputs():
    profiles = '{"private-profile":{"base_url":"https://private-url.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version","extra":"invalid"}}'
    keys = '{"private-profile":"sk-private-key"}'

    with pytest.raises(ValueError) as caught:
        Settings(
            MODE="hosted",
            DURABLE_JOBS_ENABLED=True,
            TUS_MULTIPART_ENABLED=True,
            REDIS_URL="redis://localhost:6379/0",
            SERVER_RAG_ENABLED=True,
            rag_model_profiles_json=profiles,
            rag_model_api_keys_json=keys,
            _env_file=None,
        )

    _assert_machine_readable_validation_error_is_redacted(
        caught.value,
        ("private-profile", "private-url.test", "private-model", "private-version", "sk-private-key"),
    )


def test_invalid_rag_profile_error_masks_machine_readable_uppercase_inputs():
    profiles = '{"private-profile":{"base_url":"https://private-url.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version","extra":"invalid"}}'
    keys = '{"private-profile":"sk-private-key"}'

    with pytest.raises(ValueError, match="configuration is invalid") as caught:
        Settings(
            MODE="hosted",
            DURABLE_JOBS_ENABLED=True,
            TUS_MULTIPART_ENABLED=True,
            REDIS_URL="redis://localhost:6379/0",
            SERVER_RAG_ENABLED=True,
            RAG_MODEL_PROFILES_JSON=profiles,
            RAG_MODEL_API_KEYS_JSON=keys,
            _env_file=None,
        )

    _assert_machine_readable_validation_error_is_redacted(
        caught.value,
        ("private-profile", "private-url.test", "private-model", "private-version", "sk-private-key"),
    )


def test_unrelated_after_validator_error_masks_rag_environment_inputs(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    profiles = '{"private-profile":{"base_url":"https://private-url.test/v1","model":"private-model","timeout_seconds":60,"version":"private-version"}}'
    keys = '{"private-profile":"sk-private-key"}'
    monkeypatch.setenv("MODE", "local")
    monkeypatch.setenv("JOB_LEASE_SECONDS", "30")
    monkeypatch.setenv("JOB_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("RAG_MODEL_PROFILES_JSON", profiles)
    monkeypatch.setenv("RAG_MODEL_API_KEYS_JSON", keys)

    with pytest.raises(ValueError) as caught:
        _settings()

    _assert_machine_readable_validation_error_is_redacted(
        caught.value,
        ("private-profile", "private-url.test", "private-model", "private-version", "sk-private-key"),
    )


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
