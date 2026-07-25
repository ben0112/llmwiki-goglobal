from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llmwiki_core.models import EmbeddingProfile

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")

    MODE: Literal["local", "hosted"] = "local"
    WORKSPACE_PATH: str = "."

    DATABASE_URL: str = ""
    SUPABASE_URL: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "supavault-documents"
    # Self-hosting: point the S3 client at MinIO or another S3-compatible
    # endpoint. Empty = AWS S3. Mirrors api/config.py.
    S3_ENDPOINT_URL: str = ""
    S3_FORCE_PATH_STYLE: bool = False
    STAGE: str = "dev"
    APP_URL: str = "http://localhost:3000"
    API_URL: str = "http://localhost:8000"
    MCP_URL: str = "http://localhost:8080/mcp"
    SENTRY_DSN: str = ""

    DURABLE_JOBS_ENABLED: bool = False
    HYBRID_SEARCH_ENABLED: bool = False
    EMBEDDING_PROVIDER: Literal["openai_compatible"] = "openai_compatible"
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = ""
    EMBEDDING_DIMENSIONS: int = Field(default=0, ge=0, le=4096)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=512)
    EMBEDDING_TIMEOUT_SECONDS: float = Field(default=30, gt=0, le=300)
    HYBRID_LEXICAL_CANDIDATES: int = Field(default=50, ge=1, le=500)
    HYBRID_VECTOR_CANDIDATES: int = Field(default=50, ge=1, le=500)
    HYBRID_RRF_K: int = Field(default=60, ge=1, le=10_000)

    @model_validator(mode="before")
    @classmethod
    def validate_embedding_numeric_types(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
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


settings = Settings()
