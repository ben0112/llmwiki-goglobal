import logging

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """就绪探针:校验本进程角色所拥有的依赖。

    /health 仅证明进程存活。本地模式只查 SQLite；托管模式检查
    Postgres，以及当前 Hosted lifespan 已启用并持有的 Redis/S3。
    """
    state = request.app.state
    try:
        if getattr(state, "mode", None) == "local":
            cursor = await state.sqlite_db.execute("SELECT 1")
            await cursor.fetchone()
        else:
            pool = getattr(state, "pool", None)
            if pool is None:
                raise RuntimeError("postgres is not initialized")
            await pool.fetchval("SELECT 1")

            if getattr(state, "readiness_requires_redis", False):
                redis = getattr(state, "redis", None)
                if redis is None:
                    raise RuntimeError("redis is not initialized")
                await redis.ping()

            if getattr(state, "readiness_requires_s3", False):
                s3_service = getattr(state, "s3_service", None)
                if s3_service is None:
                    raise RuntimeError("s3 is not initialized")
                await s3_service.head_bucket()
    except Exception as exc:  # noqa: BLE001 - dependency clients expose unrelated error types.
        logger.warning("readiness check failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="not ready") from None
    return {"status": "ready"}
