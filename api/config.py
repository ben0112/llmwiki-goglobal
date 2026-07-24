from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

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
    JOB_LEASE_SECONDS: int = Field(default=120, gt=0)
    JOB_HEARTBEAT_SECONDS: int = Field(default=30, gt=0)
    JOB_DISPATCH_BATCH_SIZE: int = Field(default=100, gt=0)
    JOB_REDELIVER_SECONDS: int = Field(default=30, gt=0)
    TUS_SESSION_TTL_SECONDS: int = Field(default=172800, gt=0)
    TUS_STALE_SECONDS: int = Field(default=86400, gt=0)
    TUS_LOCK_SECONDS: int = Field(default=60, gt=0)
    TUS_MAX_PATCH_BYTES: int = Field(default=67108864, gt=0)

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

    @model_validator(mode="after")
    def validate_durable_runtime(self) -> "Settings":
        if self.TUS_MULTIPART_ENABLED and not self.DURABLE_JOBS_ENABLED:
            raise ValueError("DURABLE_JOBS_ENABLED must be true when TUS_MULTIPART_ENABLED is true")

        if self.MODE == "hosted" and (
            self.DURABLE_JOBS_ENABLED or self.TUS_MULTIPART_ENABLED
        ) and not self.REDIS_URL:
            raise ValueError("REDIS_URL is required for hosted durable runtime features")

        if self.JOB_HEARTBEAT_SECONDS >= self.JOB_LEASE_SECONDS:
            raise ValueError("JOB_HEARTBEAT_SECONDS must be less than JOB_LEASE_SECONDS")
        if self.TUS_STALE_SECONDS >= self.TUS_SESSION_TTL_SECONDS:
            raise ValueError("TUS_STALE_SECONDS must be less than TUS_SESSION_TTL_SECONDS")
        if self.TUS_LOCK_SECONDS >= self.TUS_STALE_SECONDS:
            raise ValueError("TUS_LOCK_SECONDS must be less than TUS_STALE_SECONDS")

        return self

    @property
    def listen_database_url(self) -> str:
        """Connection for the LISTEN loop — direct if configured, else the pooler."""
        return self.DIRECT_DATABASE_URL or self.DATABASE_URL


settings = Settings()
