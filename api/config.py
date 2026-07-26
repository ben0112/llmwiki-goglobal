from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llmwiki_core.models import EmbeddingProfile

_SETTINGS_CONFIG_ERROR = "Settings configuration is invalid"


class SettingsConfigurationError(ValueError):
    """Detached, sanitized failure at a public Settings construction boundary."""

    __slots__ = ()


class _SafeSettingsValidator:
    """Sanitize failures outside the Pydantic core, including JSON decoding."""

    __slots__ = ("_settings_type", "_validator")

    def __init__(self, settings_type: type["Settings"], validator: Any) -> None:
        self._settings_type = settings_type
        self._validator = validator

    def _call(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, str | None]:
        safe_message: str | None = None
        try:
            result = getattr(self._validator, method_name)(*args, **kwargs)
        except ValidationError as caught:
            safe_message = _safe_settings_error_message(self._settings_type, caught)
        if safe_message is not None:
            return None, safe_message
        return result, None

    def validate_python(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("validate_python", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def validate_json(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("validate_json", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def validate_strings(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("validate_strings", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def validate_assignment(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("validate_assignment", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def isinstance_python(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("isinstance_python", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def get_default_value(self, *args: Any, **kwargs: Any) -> Any:
        result, safe_message = self._call("get_default_value", args, kwargs)
        if safe_message is not None:
            del args
            kwargs.clear()
            del kwargs
            _raise_settings_configuration_error(safe_message)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._validator, name)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore", hide_input_in_errors=True)

    MODE: Literal["local", "hosted"] = "local"
    WORKSPACE_PATH: str = "."

    DATABASE_URL: str = ""
    # Direct (non-pooler) connection used only for the long-lived LISTEN/NOTIFY
    # socket. Supavisor recycles pooled sessions, which silently kills LISTEN;
    # a direct connection sidesteps that. Falls back to DATABASE_URL when unset.
    DIRECT_DATABASE_URL: str = ""
    SUPABASE_URL: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "supavault-documents"
    # Self-hosting: point the S3 clients at MinIO or another S3-compatible
    # endpoint (e.g. "https://s3.example.internal:9000"). Empty = AWS S3.
    # MinIO needs path-style addressing unless wildcard DNS is configured.
    S3_ENDPOINT_URL: str = ""
    S3_FORCE_PATH_STYLE: bool = False
    MISTRAL_API_KEY: str = ""
    PDF_BACKEND: str = "opendataloader"  # "opendataloader" or "mistral"
    STAGE: str = "dev"
    APP_URL: str = "http://localhost:3000"
    API_URL: str = "http://localhost:8000"

    QUOTA_MAX_PAGES_PER_DOC: int = 300  # max pages per single document
    QUOTA_MAX_STORAGE_BYTES: int = 1_073_741_824  # 1 GB per user

    CONVERTER_URL: str = ""
    CONVERTER_SECRET: str = ""

    REDIS_URL: str | None = None
    DURABLE_JOBS_ENABLED: bool = False
    TUS_MULTIPART_ENABLED: bool = False
    SERVER_RAG_ENABLED: bool = False
    RAG_MODEL_PROFILES_JSON: SecretStr = Field(default_factory=lambda: SecretStr("{}"), repr=False)
    RAG_MODEL_API_KEYS_JSON: SecretStr = Field(default_factory=lambda: SecretStr("{}"), repr=False)
    JOB_LEASE_SECONDS: int = Field(default=120, gt=0)
    JOB_HEARTBEAT_SECONDS: int = Field(default=30, gt=0)
    JOB_DISPATCH_BATCH_SIZE: int = Field(default=100, gt=0)
    JOB_REDELIVER_SECONDS: int = Field(default=30, gt=0)
    TUS_SESSION_TTL_SECONDS: int = Field(default=172800, gt=0)
    TUS_STALE_SECONDS: int = Field(default=86400, gt=0)
    TUS_LOCK_SECONDS: int = Field(default=60, gt=0)
    TUS_MAX_PATCH_BYTES: int = Field(default=67108864, gt=0)

    HYBRID_SEARCH_ENABLED: bool = False
    EMBEDDING_PROVIDER: Literal["openai_compatible"] = "openai_compatible"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    EMBEDDING_MODEL: str = ""
    EMBEDDING_DIMENSIONS: int = Field(default=0, ge=0, le=4096)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=512)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=30, gt=0, le=300)
    HYBRID_LEXICAL_CANDIDATES: int = Field(default=50, ge=1, le=500)
    HYBRID_VECTOR_CANDIDATES: int = Field(default=50, ge=1, le=500)
    HYBRID_RRF_K: int = Field(default=60, ge=1, le=10_000)

    GLOBAL_OCR_ENABLED: bool = True
    GLOBAL_MAX_PAGES: int = 1_000_000
    GLOBAL_MAX_USERS: int = 10_000

    SENTRY_DSN: str = ""

    # 语料分类流水线(本地模式;设置页存储优先于这些环境变量)
    CORPUS_LLM_BASE_URL: str = ""
    CORPUS_LLM_MODEL: str = ""
    CORPUS_LLM_API_KEY: str = ""
    CORPUS_LLM_TIMEOUT: float = 120.0
    CORPUS_LLM_CONCURRENCY: int = 0  # LLM 请求并发数;0 = 端点感知默认(本地2/云端8)
    CORPUS_LLM_THINKING: bool = False  # 分类 LLM 思考模式(设置页显式值优先)
    EXTRACT_CONCURRENCY: int = 0  # 文档提取并发(LibreOffice/JVM);0 = CPU 感知默认
    CORPUS_AUTOCLASSIFY: bool = False  # 自动分类默认关(设置页可开)
    CORPUS_AUTO_INTERVAL: int = 30  # 自动分类轮询间隔(秒)

    def __init__(__pydantic_self__, **values: Any) -> None:
        safe_message: str | None = None
        try:
            super().__init__(**values)
        except (ValidationError, SettingsConfigurationError) as caught:
            safe_message = _safe_settings_boundary_message(__pydantic_self__.__class__, caught)
        if safe_message is not None:
            values.clear()
            del values
            del __pydantic_self__
            _raise_settings_configuration_error(safe_message)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Any = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "Settings":
        safe_message: str | None = None
        try:
            result = super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except (ValidationError, SettingsConfigurationError) as caught:
            safe_message = _safe_settings_boundary_message(cls, caught)
        if safe_message is not None:
            del obj
            del context
            _raise_settings_configuration_error(safe_message)
        return result

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "Settings":
        safe_message: str | None = None
        try:
            result = super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except (ValidationError, SettingsConfigurationError) as caught:
            safe_message = _safe_settings_boundary_message(cls, caught)
        if safe_message is not None:
            del json_data
            del context
            _raise_settings_configuration_error(safe_message)
        return result

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Any = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "Settings":
        safe_message: str | None = None
        try:
            result = super().model_validate_strings(
                obj,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )
        except (ValidationError, SettingsConfigurationError) as caught:
            safe_message = _safe_settings_boundary_message(cls, caught)
        if safe_message is not None:
            del obj
            del context
            _raise_settings_configuration_error(safe_message)
        return result

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        _install_safe_settings_validator(cls)

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Any = None,
    ) -> bool | None:
        rebuilt = super().model_rebuild(
            force=force,
            raise_errors=raise_errors,
            _parent_namespace_depth=_parent_namespace_depth,
            _types_namespace=_types_namespace,
        )
        _install_safe_settings_validator(cls)
        return rebuilt

    @model_validator(mode="before")
    @classmethod
    def validate_embedding_numeric_types(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        sanitized_values = dict(values)
        rag_secret_fields = {"RAG_MODEL_PROFILES_JSON", "RAG_MODEL_API_KEYS_JSON"}
        for field_name, value in tuple(sanitized_values.items()):
            if not isinstance(field_name, str) or field_name.upper() not in rag_secret_fields:
                continue
            if isinstance(value, SecretStr):
                continue
            if not isinstance(value, str):
                raise ValueError(_SETTINGS_CONFIG_ERROR)
            sanitized_values[field_name] = SecretStr(value)
        values = sanitized_values
        integer_fields = (
            "EMBEDDING_DIMENSIONS",
            "EMBEDDING_BATCH_SIZE",
            "HYBRID_LEXICAL_CANDIDATES",
            "HYBRID_VECTOR_CANDIDATES",
            "HYBRID_RRF_K",
        )
        for field_name in integer_fields:
            value = values.get(field_name)
            if value is not None and not isinstance(value, str) and type(value) is not int:
                raise ValueError(f"{field_name} must be an integer")
        timeout = values.get("EMBEDDING_TIMEOUT_SECONDS")
        if timeout is not None and not isinstance(timeout, str) and type(timeout) not in (int, float):
            raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be a finite number")
        return values

    @model_validator(mode="after")
    def validate_durable_runtime(self) -> "Settings":
        if self.MODE == "hosted" and not self.DURABLE_JOBS_ENABLED:
            raise ValueError("Hosted mode requires DURABLE_JOBS_ENABLED=true; legacy in-process jobs were removed")
        if self.MODE == "hosted" and not self.TUS_MULTIPART_ENABLED:
            raise ValueError("Hosted mode requires TUS_MULTIPART_ENABLED=true; process-local uploads were removed")

        if self.MODE == "hosted" and not self.REDIS_URL:
            raise ValueError("REDIS_URL is required for hosted durable runtime features")

        if self.JOB_HEARTBEAT_SECONDS >= self.JOB_LEASE_SECONDS:
            raise ValueError("JOB_HEARTBEAT_SECONDS must be less than JOB_LEASE_SECONDS")
        if self.TUS_STALE_SECONDS >= self.TUS_SESSION_TTL_SECONDS:
            raise ValueError("TUS_STALE_SECONDS must be less than TUS_SESSION_TTL_SECONDS")
        if self.TUS_LOCK_SECONDS >= self.TUS_STALE_SECONDS:
            raise ValueError("TUS_LOCK_SECONDS must be less than TUS_STALE_SECONDS")

        return self

    @model_validator(mode="after")
    def validate_hybrid_runtime(self) -> "Settings":
        if not self.HYBRID_SEARCH_ENABLED:
            return self
        if self.MODE != "hosted":
            raise ValueError("Hybrid search is supported only in hosted mode")
        if not self.DURABLE_JOBS_ENABLED:
            raise ValueError("Hybrid search requires DURABLE_JOBS_ENABLED=true")
        if not self.EMBEDDING_BASE_URL.strip():
            raise ValueError("EMBEDDING_BASE_URL is required for hybrid search")
        if not self.EMBEDDING_MODEL.strip():
            raise ValueError("EMBEDDING_MODEL is required for hybrid search")
        if not 1 <= self.EMBEDDING_DIMENSIONS <= 4096:
            raise ValueError("EMBEDDING_DIMENSIONS must be between 1 and 4096")
        return self

    @model_validator(mode="after")
    def validate_rag_runtime(self) -> "Settings":
        if not self.SERVER_RAG_ENABLED:
            return self
        if self.MODE != "hosted":
            raise ValueError("Server RAG is supported only in hosted mode")
        if not self.DURABLE_JOBS_ENABLED:
            raise ValueError("Server RAG requires DURABLE_JOBS_ENABLED=true")
        from rag.model import resolve_model_profiles

        if not resolve_model_profiles(self):
            raise ValueError("Server RAG requires at least one complete model profile")
        return self

    @property
    def embedding_profile(self) -> EmbeddingProfile | None:
        if not self.HYBRID_SEARCH_ENABLED:
            return None
        return EmbeddingProfile(
            provider=self.EMBEDDING_PROVIDER,
            model=self.EMBEDDING_MODEL,
            dimensions=self.EMBEDDING_DIMENSIONS,
        )

    def hybrid_candidate_limits(self, request_limit: int) -> tuple[int, int]:
        if type(request_limit) is not int or not 1 <= request_limit <= 500:
            raise ValueError("request limit must be between 1 and 500")
        if request_limit > self.HYBRID_LEXICAL_CANDIDATES:
            raise ValueError("HYBRID_LEXICAL_CANDIDATES must be at least the request limit")
        if request_limit > self.HYBRID_VECTOR_CANDIDATES:
            raise ValueError("HYBRID_VECTOR_CANDIDATES must be at least the request limit")
        return self.HYBRID_LEXICAL_CANDIDATES, self.HYBRID_VECTOR_CANDIDATES

    @property
    def listen_database_url(self) -> str:
        """Connection for the LISTEN loop — direct if configured, else the pooler."""
        return self.DIRECT_DATABASE_URL or self.DATABASE_URL


def _safe_settings_error_message(settings_type: type[Settings], error: ValidationError) -> str:
    safe_details: list[str] = []
    field_names = settings_type.model_fields.keys()
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = item.get("loc", ())
        message = str(item.get("msg", ""))
        for component in location:
            if isinstance(component, str) and component in field_names and component not in safe_details:
                safe_details.append(component)
        for field_name in field_names:
            if field_name in message and field_name not in safe_details:
                safe_details.append(field_name)
        lowered = message.lower()
        for marker in ("hosted", "profile", "integer", "finite number"):
            if marker in lowered and marker not in safe_details:
                safe_details.append(marker)
    if not safe_details:
        return _SETTINGS_CONFIG_ERROR
    return f"{_SETTINGS_CONFIG_ERROR}: {', '.join(safe_details)}"


def _safe_settings_boundary_message(
    settings_type: type[Settings],
    error: ValidationError | SettingsConfigurationError,
) -> str:
    if isinstance(error, ValidationError):
        return _safe_settings_error_message(settings_type, error)
    return str(error)


def _raise_settings_configuration_error(message: str) -> Any:
    raise SettingsConfigurationError(message) from None


def _install_safe_settings_validator(settings_type: type[Settings]) -> None:
    validator = settings_type.__dict__.get("__pydantic_validator__")
    if validator is not None and not isinstance(validator, _SafeSettingsValidator):
        settings_type.__pydantic_validator__ = _SafeSettingsValidator(settings_type, validator)


_install_safe_settings_validator(Settings)
settings = Settings()
