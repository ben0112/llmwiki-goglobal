"""ARQ worker envelope backed exclusively by the durable PostgreSQL job ledger."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import socket
import time
from collections.abc import Mapping
from uuid import UUID, uuid4

import asyncpg
import httpx
from arq import cron
from arq.connections import RedisSettings
from arq.worker import create_worker as arq_create_worker
from arq.worker import run_worker as arq_run_worker
from config import settings
from telemetry import emit

from jobs import repository
from jobs.dispatcher import dispatch_cron
from jobs.handlers import (
    HANDLERS,
    Handler,
    RetryableJobError,
    TerminalJobError,
    UnsupportedJobHandler,
    WorkerContext,
)
from jobs.lease import JobLease
from jobs.models import (
    RESULT_MAX_BYTES,
    JobCancelled,
    JobRecord,
    JSONValue,
    LeaseLost,
    to_json_value,
)

logger = logging.getLogger(__name__)

_CANCELLED_CODE = "cancelled"
_CANCELLED_MESSAGE = "Job cancellation was requested."
_UNHANDLED_CODE = "unhandled_worker_error"
_UNHANDLED_MESSAGE = "The job encountered an unexpected error."
_INVALID_RESULT_CODE = "invalid_job_result"
_INVALID_RESULT_MESSAGE = "The job produced an invalid result."
_SHUTDOWN_CODE = "worker_shutdown"
_SHUTDOWN_MESSAGE = "Worker shutdown interrupted the job."
_OWNED_RESOURCES_CTX_KEY = "_durable_worker_owned_resources"
WORKER_STARTUP_TIMEOUT_SECONDS = 10.0
WORKER_MAX_JOBS = 10
WORKER_POOL_RESERVED_CONNECTIONS = 2
WORKER_POOL_MAX_SIZE = WORKER_MAX_JOBS + WORKER_POOL_RESERVED_CONNECTIONS
_WORKER_RUNTIME_CTX_KEYS = (
    "worker_context",
    "handlers",
    "worker_id",
    "lease_seconds",
    "heartbeat_seconds",
    "dispatch_batch_size",
    "reap_batch_size",
    "redeliver_seconds",
)


async def _create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=2, max_size=WORKER_POOL_MAX_SIZE)


def _create_s3_service() -> object:
    from services.s3 import S3Service

    return S3Service()


def _make_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"


def _s3_is_configured(runtime_settings: object) -> bool:
    return bool(
        getattr(runtime_settings, "S3_BUCKET", None)
        and getattr(runtime_settings, "AWS_ACCESS_KEY_ID", None)
        and getattr(runtime_settings, "AWS_SECRET_ACCESS_KEY", None)
    )


async def _check_worker_readiness(
    *,
    pool: asyncpg.Pool,
    redis: object,
    s3: object | None,
    converter_url: str,
) -> None:
    """Verify every dependency needed before ARQ starts accepting jobs."""
    if s3 is None:
        raise RuntimeError("ARQ durable worker requires S3 configuration")
    if not isinstance(converter_url, str) or not converter_url.strip():
        raise RuntimeError("ARQ durable worker requires converter URL")
    await pool.fetchval("SELECT 1")
    await redis.ping()
    await s3.head_bucket()
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        response = await client.get(f"{converter_url.rstrip('/')}/health")
        response.raise_for_status()


def validate_worker_runtime(runtime_settings: object) -> None:
    """Reject unsafe worker configuration before ARQ constructs a Worker."""
    if getattr(runtime_settings, "MODE", None) != "hosted":
        raise RuntimeError("ARQ durable worker requires MODE=hosted")
    if getattr(runtime_settings, "DURABLE_JOBS_ENABLED", None) is not True:
        raise RuntimeError("ARQ durable worker requires DURABLE_JOBS_ENABLED=true")

    redis_url = getattr(runtime_settings, "REDIS_URL", None)
    if not isinstance(redis_url, str) or not redis_url.strip():
        raise RuntimeError("ARQ durable worker requires REDIS_URL")
    database_url = getattr(runtime_settings, "DATABASE_URL", None)
    if not isinstance(database_url, str) or not database_url.strip():
        raise RuntimeError("ARQ durable worker requires DATABASE_URL")
    converter_secret = getattr(runtime_settings, "CONVERTER_SECRET", None)
    if not isinstance(converter_secret, str) or not converter_secret.strip():
        raise RuntimeError("ARQ durable worker converter secret is required")


async def startup(ctx: dict) -> None:
    """Construct only the resources needed by a hosted durable worker."""
    runtime_settings = ctx.get("runtime_settings")
    if runtime_settings is None:
        raise RuntimeError("ARQ durable worker requires runtime_settings in ctx")
    validate_worker_runtime(runtime_settings)
    if ctx.get("redis") is None:
        raise RuntimeError("ARQ durable worker requires ctx['redis'] from ARQ")
    reserved_keys = {"pool", "s3", _OWNED_RESOURCES_CTX_KEY}.intersection(ctx)
    if reserved_keys:
        names = ", ".join(sorted(reserved_keys))
        raise RuntimeError(f"ARQ durable worker ctx contains reserved keys: {names}")

    created_resources: dict[str, object | None] = {}
    try:
        async with asyncio.timeout(WORKER_STARTUP_TIMEOUT_SECONDS):
            pool = await _create_pool(runtime_settings.DATABASE_URL)
            created_resources["pool"] = pool
            ctx["pool"] = pool
            s3 = _create_s3_service() if _s3_is_configured(runtime_settings) else None
            created_resources["s3"] = s3
            ctx["s3"] = s3
            tus_cleanup = None
            if getattr(runtime_settings, "TUS_MULTIPART_ENABLED", False):
                if s3 is None:
                    raise RuntimeError("multipart TUS cleanup worker requires S3")
                from infra.quota import HostedQuotaService
                from infra.tus import HostedTusCleanupService
                from infra.tus_sessions import TusSessionStore

                from jobs.service import JobService

                tus_cleanup = HostedTusCleanupService(
                    pool,
                    s3,
                    JobService(pool),
                    HostedQuotaService(pool, ctx["redis"]),
                    TusSessionStore(ctx["redis"]),
                    session_ttl_seconds=runtime_settings.TUS_SESSION_TTL_SECONDS,
                    stale_seconds=runtime_settings.TUS_STALE_SECONDS,
                    lock_seconds=runtime_settings.TUS_LOCK_SECONDS,
                )
            worker_context = WorkerContext(
                pool=pool,
                s3=s3,
                converter_url=runtime_settings.CONVERTER_URL,
                converter_secret=runtime_settings.CONVERTER_SECRET,
                tus_cleanup=tus_cleanup,
            )
            ctx.update(
                {
                    "worker_context": worker_context,
                    "handlers": HANDLERS,
                    "worker_id": _make_worker_id(),
                    "lease_seconds": runtime_settings.JOB_LEASE_SECONDS,
                    "heartbeat_seconds": runtime_settings.JOB_HEARTBEAT_SECONDS,
                    "dispatch_batch_size": runtime_settings.JOB_DISPATCH_BATCH_SIZE,
                    "reap_batch_size": runtime_settings.JOB_DISPATCH_BATCH_SIZE,
                    "redeliver_seconds": runtime_settings.JOB_REDELIVER_SECONDS,
                }
            )
            await _check_worker_readiness(
                pool=pool,
                redis=ctx["redis"],
                s3=s3,
                converter_url=runtime_settings.CONVERTER_URL,
            )
            ctx[_OWNED_RESOURCES_CTX_KEY] = created_resources
    except BaseException as exc:
        try:
            await _close_worker_resources(ctx, owned_resources=created_resources)
        finally:
            _clear_worker_runtime_context(ctx)
        if isinstance(exc, TimeoutError):
            raise RuntimeError("worker dependency readiness timed out") from None
        raise


async def _close_worker_resources(
    ctx: dict,
    *,
    owned_resources: Mapping[str, object | None],
) -> None:
    """Close each tracked worker-owned resource at most once."""
    pool = owned_resources.get("pool")
    s3 = owned_resources.get("s3")
    if "pool" in owned_resources and ctx.get("pool") is pool:
        ctx.pop("pool", None)
    if "s3" in owned_resources and ctx.get("s3") is s3:
        ctx.pop("s3", None)

    try:
        close_s3 = getattr(s3, "close", None)
        if callable(close_s3):
            result = close_s3()
            if inspect.isawaitable(result):
                await result
    finally:
        if pool is not None:
            await pool.close()


def _clear_worker_runtime_context(ctx: dict) -> None:
    for key in _WORKER_RUNTIME_CTX_KEYS:
        ctx.pop(key, None)


async def shutdown(ctx: dict) -> None:
    """Release worker-owned resources while leaving ARQ's Redis client alone."""
    owned_resources = ctx.pop(_OWNED_RESOURCES_CTX_KEY, None)
    if owned_resources is None:
        return
    try:
        await _close_worker_resources(ctx, owned_resources=owned_resources)
        logger.info("durable worker resources closed")
    finally:
        _clear_worker_runtime_context(ctx)


async def _record_failure(
    pool: asyncpg.Pool,
    job_id: UUID,
    worker_id: str,
    *,
    error_code: str,
    error_message: str,
    retryable: bool,
) -> bool:
    try:
        async with pool.acquire() as conn, conn.transaction():
            await repository.fail_or_retry(
                conn,
                job_id,
                worker_id,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
    except LeaseLost:
        return False
    return True


def _outcome(status: str, job_id: UUID | None = None) -> dict[str, str]:
    outcome = {"status": status}
    if job_id is not None:
        outcome["job_id"] = str(job_id)
    return outcome


def _invalid_result() -> TerminalJobError:
    return TerminalJobError(_INVALID_RESULT_CODE, _INVALID_RESULT_MESSAGE)


def _reject_obviously_oversized_strings(value: JSONValue) -> None:
    """Reject when cumulative string bytes alone cannot fit the result budget."""

    def accumulate_string_bytes(item: JSONValue, total: int) -> int:
        if isinstance(item, str):
            total += len(item.encode("utf-8"))
            if total > RESULT_MAX_BYTES:
                raise _invalid_result()
            return total
        if isinstance(item, list):
            for child in item:
                total = accumulate_string_bytes(child, total)
            return total
        if isinstance(item, dict):
            for key, child in item.items():
                total = accumulate_string_bytes(key, total)
                total = accumulate_string_bytes(child, total)
        return total

    accumulate_string_bytes(value, 0)


def _prepare_job_result(raw_result: object) -> tuple[dict[str, JSONValue], str]:
    """Validate strict JSON types and produce safe PostgreSQL JSON input text."""
    try:
        result = to_json_value(raw_result)
        if not isinstance(result, dict):
            raise TypeError("job handler result must be a JSON object")
        _reject_obviously_oversized_strings(result)
        serialized = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(", ", ": "),
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError):
        raise _invalid_result() from None
    return result, serialized


async def _validate_result_in_postgres(
    conn: asyncpg.Connection,
    serialized_json: str,
) -> None:
    """Use PostgreSQL's canonical jsonb text as the authoritative size boundary."""
    try:
        canonical_bytes = await conn.fetchval(
            "SELECT octet_length($1::jsonb::text)",
            serialized_json,
        )
    except asyncpg.PostgresError as exc:
        sqlstate = getattr(exc, "sqlstate", None)
        if isinstance(exc, asyncpg.DataError) or (isinstance(sqlstate, str) and sqlstate.startswith("22")):
            raise _invalid_result() from None
        raise
    if not isinstance(canonical_bytes, int) or canonical_bytes > RESULT_MAX_BYTES:
        raise _invalid_result()


async def _run_claimed_job(
    *,
    pool: asyncpg.Pool,
    job: JobRecord,
    worker_id: str,
    lease_seconds: int,
    heartbeat_seconds: int,
    handlers: Mapping,
    worker_context: WorkerContext,
) -> dict[str, str]:
    async with JobLease(
        pool,
        job.id,
        worker_id,
        lease_seconds,
        heartbeat_seconds,
    ) as lease:
        try:
            handler: Handler | None = handlers.get(job.job_type)
            if handler is None:
                raise UnsupportedJobHandler
            raw_result = await handler(job, lease, worker_context)
            result, serialized_result = _prepare_job_result(raw_result)
            async with pool.acquire() as conn, conn.transaction():
                await _validate_result_in_postgres(conn, serialized_result)
                await repository.succeed(conn, job.id, worker_id, result)
            return _outcome("succeeded", job.id)
        except asyncio.CancelledError:
            await asyncio.shield(
                _record_failure(
                    pool,
                    job.id,
                    worker_id,
                    error_code=_SHUTDOWN_CODE,
                    error_message=_SHUTDOWN_MESSAGE,
                    retryable=True,
                )
            )
            raise
        except LeaseLost:
            logger.info("worker lease lost job_id=%s", job.id)
            return _outcome("lease_lost", job.id)
        except JobCancelled:
            recorded = await _record_failure(
                pool,
                job.id,
                worker_id,
                error_code=_CANCELLED_CODE,
                error_message=_CANCELLED_MESSAGE,
                retryable=False,
            )
        except RetryableJobError as exc:
            recorded = await _record_failure(
                pool,
                job.id,
                worker_id,
                error_code=exc.error_code,
                error_message=exc.error_message,
                retryable=True,
            )
        except TerminalJobError as exc:
            recorded = await _record_failure(
                pool,
                job.id,
                worker_id,
                error_code=exc.error_code,
                error_message=exc.error_message,
                retryable=False,
            )
        except Exception as exc:  # noqa: BLE001 - raw failures are sanitized at the ledger boundary.
            logger.error(
                "unhandled worker failure job_id=%s error_type=%s",
                job.id,
                type(exc).__name__,
            )
            recorded = await _record_failure(
                pool,
                job.id,
                worker_id,
                error_code=_UNHANDLED_CODE,
                error_message=_UNHANDLED_MESSAGE,
                retryable=True,
            )

        if not recorded:
            logger.info("worker lease lost before failure transition job_id=%s", job.id)
            return _outcome("lease_lost", job.id)
        return _outcome("failed", job.id)


async def run_job(ctx: dict, job_id_text: str) -> dict[str, str]:
    """Claim and execute one database-owned job from an opaque ARQ UUID message."""
    try:
        job_id = UUID(job_id_text)
    except (AttributeError, TypeError, ValueError):
        logger.warning("worker rejected invalid job UUID")
        return _outcome("invalid_job_id")

    pool: asyncpg.Pool = ctx["pool"]
    worker_id: str = ctx["worker_id"]
    lease_seconds: int = ctx["lease_seconds"]
    async with pool.acquire() as conn, conn.transaction():
        job = await repository.claim(conn, job_id, worker_id, lease_seconds)
    if job is None:
        return _outcome("duplicate", job_id)

    started = time.monotonic()
    try:
        outcome = await _run_claimed_job(
            pool=pool,
            job=job,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            heartbeat_seconds=ctx["heartbeat_seconds"],
            handlers=ctx["handlers"],
            worker_context=ctx["worker_context"],
        )
    except asyncio.CancelledError:
        emit(
            logger,
            "durable_job_finished",
            job_id=job.id,
            job_type=job.job_type,
            attempt=job.attempt_count,
            state="retry_wait",
            lease_owner=worker_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            error_code=_SHUTDOWN_CODE,
            replica_role="worker",
        )
        raise
    emit(
        logger,
        "durable_job_finished",
        job_id=job.id,
        job_type=job.job_type,
        attempt=job.attempt_count,
        state=outcome["status"],
        lease_owner=worker_id,
        duration_ms=int((time.monotonic() - started) * 1000),
        error_code=None if outcome["status"] == "succeeded" else outcome["status"],
        replica_role="worker",
    )
    return outcome


async def reap_cron(ctx: dict) -> None:
    """Recover a bounded batch of jobs whose PostgreSQL leases expired."""
    async with ctx["pool"].acquire() as conn, conn.transaction():
        reaped = await repository.reap_expired(conn, limit=ctx["reap_batch_size"])
    for job_id in reaped:
        emit(
            logger,
            "durable_job_lease_reaped",
            job_id=job_id,
            state="retry_wait",
            error_code="lease_expired",
            replica_role="worker",
        )


async def upload_cleanup_cron(ctx: dict) -> None:
    """Discover stale upload sessions and append idempotent durable cleanup jobs."""
    cleanup = ctx["worker_context"].tus_cleanup
    if cleanup is None:
        return
    count = await cleanup.enqueue_stale_jobs()
    logger.info("enqueued stale upload cleanup jobs count=%d", count)


def _build_cron_jobs() -> list:
    """Build fresh CronJob instances because ARQ mutates their next_run state."""
    return [
        cron(
            dispatch_cron,
            name="durable_job_dispatch",
            second={0, 10, 20, 30, 40, 50},
            run_at_startup=True,
            unique=True,
            max_tries=1,
            keep_result=0,
        ),
        cron(
            reap_cron,
            name="durable_job_reaper",
            second={5, 35},
            run_at_startup=True,
            unique=True,
            max_tries=1,
            keep_result=0,
        ),
        cron(
            upload_cleanup_cron,
            name="durable_upload_cleanup_scan",
            minute=None,
            second={15},
            run_at_startup=True,
            unique=True,
            max_tries=1,
            keep_result=0,
        ),
    ]


def build_worker_settings(runtime_settings: object) -> dict[str, object]:
    """Build safe ARQ settings only after durable runtime preflight succeeds."""
    validate_worker_runtime(runtime_settings)
    redis_url = runtime_settings.REDIS_URL.strip()
    return {
        "functions": [run_job],
        "cron_jobs": _build_cron_jobs(),
        "max_tries": 1,
        "retry_jobs": False,
        "keep_result": 0,
        "job_timeout": 3600,
        "max_jobs": WORKER_MAX_JOBS,
        "on_startup": startup,
        "on_shutdown": shutdown,
        "redis_settings": RedisSettings.from_dsn(redis_url),
        "ctx": {"runtime_settings": runtime_settings},
    }


def _merge_worker_context(
    runtime_settings: object,
    kwargs: dict[str, object],
) -> None:
    if "ctx" not in kwargs:
        return
    supplied_ctx = kwargs["ctx"]
    if supplied_ctx is None:
        kwargs["ctx"] = {"runtime_settings": runtime_settings}
        return
    if not isinstance(supplied_ctx, Mapping):
        raise TypeError("ctx must be a mapping")
    kwargs["ctx"] = {**supplied_ctx, "runtime_settings": runtime_settings}


def create_durable_worker(runtime_settings: object = settings, **kwargs: object) -> object:
    """Construct an unconnected ARQ Worker after mandatory durable preflight."""
    worker_settings = build_worker_settings(runtime_settings)
    _merge_worker_context(runtime_settings, kwargs)
    return arq_create_worker(worker_settings, **kwargs)


def run_durable_worker(runtime_settings: object = settings, **kwargs: object) -> object:
    """Run the supported durable worker launcher after mandatory preflight."""
    worker_settings = build_worker_settings(runtime_settings)
    _merge_worker_context(runtime_settings, kwargs)
    return arq_run_worker(worker_settings, **kwargs)


class WorkerSettings:
    """ARQ CLI settings; startup performs the authoritative runtime validation."""

    functions = [run_job]
    cron_jobs = _build_cron_jobs()
    max_jobs = WORKER_MAX_JOBS
    max_tries = 1
    retry_jobs = False
    keep_result = 0
    job_timeout = 3600
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL or "redis://localhost:6379/0")
    ctx = {"runtime_settings": settings}


def main() -> None:
    """Launch the durable worker via ``python -m jobs.worker``."""
    run_durable_worker()


if __name__ == "__main__":
    main()
