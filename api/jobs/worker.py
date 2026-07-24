"""ARQ worker envelope backed exclusively by the durable PostgreSQL job ledger."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import socket
from collections.abc import Mapping
from uuid import UUID, uuid4

import asyncpg
from arq import cron
from arq.connections import RedisSettings
from arq.worker import create_worker as arq_create_worker
from arq.worker import run_worker as arq_run_worker
from config import settings

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


async def _create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=2, max_size=10)


def _create_s3_service() -> object:
    from services.s3 import S3Service

    return S3Service()


def _make_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"


def _s3_is_configured() -> bool:
    return bool(settings.S3_BUCKET and settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY)


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


async def startup(ctx: dict) -> None:
    """Construct only the resources needed by a hosted durable worker."""
    validate_worker_runtime(settings)
    if ctx.get("redis") is None:
        raise RuntimeError("ARQ durable worker requires ctx['redis'] from ARQ")

    pool: asyncpg.Pool | None = None
    try:
        pool = await _create_pool(settings.DATABASE_URL)
        s3 = _create_s3_service() if _s3_is_configured() else None
        worker_context = WorkerContext(
            pool=pool,
            s3=s3,
            converter_url=settings.CONVERTER_URL,
            converter_secret=settings.CONVERTER_SECRET,
        )
        ctx.update(
            {
                "pool": pool,
                "s3": s3,
                "worker_context": worker_context,
                "handlers": HANDLERS,
                "worker_id": _make_worker_id(),
                "lease_seconds": settings.JOB_LEASE_SECONDS,
                "heartbeat_seconds": settings.JOB_HEARTBEAT_SECONDS,
                "dispatch_batch_size": settings.JOB_DISPATCH_BATCH_SIZE,
                "reap_batch_size": settings.JOB_DISPATCH_BATCH_SIZE,
                "redeliver_seconds": settings.JOB_REDELIVER_SECONDS,
            }
        )
    except BaseException:
        if pool is not None:
            await pool.close()
        raise


async def shutdown(ctx: dict) -> None:
    """Release worker-owned resources while leaving ARQ's Redis client alone."""
    pool = ctx.pop("pool", None)
    s3 = ctx.pop("s3", None)
    ctx.pop("worker_context", None)
    ctx.pop("handlers", None)
    ctx.pop("worker_id", None)
    ctx.pop("lease_seconds", None)
    ctx.pop("heartbeat_seconds", None)
    ctx.pop("dispatch_batch_size", None)
    ctx.pop("reap_batch_size", None)
    ctx.pop("redeliver_seconds", None)

    try:
        close_s3 = getattr(s3, "close", None)
        if callable(close_s3):
            result = close_s3()
            if inspect.isawaitable(result):
                await result
    finally:
        if pool is not None:
            await pool.close()


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


def _prepare_job_result(raw_result: object) -> dict[str, JSONValue]:
    """Validate a handler result against the durable JSONB persistence boundary."""
    try:
        result = to_json_value(raw_result)
        if not isinstance(result, dict):
            raise TypeError("job handler result must be a JSON object")
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            separators=(", ", ": "),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError):
        raise TerminalJobError(_INVALID_RESULT_CODE, _INVALID_RESULT_MESSAGE) from None
    if len(encoded) > RESULT_MAX_BYTES:
        raise TerminalJobError(_INVALID_RESULT_CODE, _INVALID_RESULT_MESSAGE)
    return result


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
            result = _prepare_job_result(raw_result)
            async with pool.acquire() as conn, conn.transaction():
                await repository.succeed(conn, job.id, worker_id, result)
            return _outcome("succeeded", job.id)
        except asyncio.CancelledError:
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

    return await _run_claimed_job(
        pool=pool,
        job=job,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        heartbeat_seconds=ctx["heartbeat_seconds"],
        handlers=ctx["handlers"],
        worker_context=ctx["worker_context"],
    )


async def reap_cron(ctx: dict) -> None:
    """Recover a bounded batch of jobs whose PostgreSQL leases expired."""
    async with ctx["pool"].acquire() as conn, conn.transaction():
        reaped = await repository.reap_expired(conn, limit=ctx["reap_batch_size"])
    logger.info("reaped expired durable jobs count=%d", len(reaped))


class WorkerSettings:
    functions = [run_job]
    cron_jobs = [
        cron(
            dispatch_cron,
            name="durable_job_dispatch",
            job_id="durable-job-dispatch-cron",
            second={0, 10, 20, 30, 40, 50},
            run_at_startup=True,
            unique=True,
            max_tries=1,
            keep_result=0,
        ),
        cron(
            reap_cron,
            name="durable_job_reaper",
            job_id="durable-job-reaper-cron",
            second={5, 35},
            run_at_startup=True,
            unique=True,
            max_tries=1,
            keep_result=0,
        ),
    ]
    max_tries = 1
    retry_jobs = False
    keep_result = 0
    job_timeout = 3600
    on_startup = startup
    on_shutdown = shutdown


def build_worker_settings(runtime_settings: object) -> dict[str, object]:
    """Build safe ARQ settings only after durable runtime preflight succeeds."""
    validate_worker_runtime(runtime_settings)
    redis_url = runtime_settings.REDIS_URL.strip()
    return {
        "functions": WorkerSettings.functions,
        "cron_jobs": WorkerSettings.cron_jobs,
        "max_tries": WorkerSettings.max_tries,
        "retry_jobs": WorkerSettings.retry_jobs,
        "keep_result": WorkerSettings.keep_result,
        "job_timeout": WorkerSettings.job_timeout,
        "on_startup": WorkerSettings.on_startup,
        "on_shutdown": WorkerSettings.on_shutdown,
        "redis_settings": RedisSettings.from_dsn(redis_url),
    }


def create_durable_worker(runtime_settings: object = settings, **kwargs: object) -> object:
    """Construct an unconnected ARQ Worker after mandatory durable preflight."""
    worker_settings = build_worker_settings(runtime_settings)
    return arq_create_worker(worker_settings, **kwargs)


def run_durable_worker(runtime_settings: object = settings, **kwargs: object) -> object:
    """Run the supported durable worker launcher after mandatory preflight."""
    worker_settings = build_worker_settings(runtime_settings)
    return arq_run_worker(worker_settings, **kwargs)


def main() -> None:
    """Launch the durable worker via ``python -m jobs.worker``."""
    run_durable_worker()


if __name__ == "__main__":
    main()
