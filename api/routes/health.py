from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    """就绪探针:校验数据库连通(供 compose healthcheck / 反向代理探活)。

    /health 仅证明进程存活;本端点在本地模式查 SQLite、托管模式查
    Postgres,连不上时返回 503。
    """
    state = request.app.state
    try:
        if getattr(state, "pool", None) is not None:
            await state.pool.fetchval("SELECT 1")
        elif getattr(state, "sqlite_db", None) is not None:
            cursor = await state.sqlite_db.execute("SELECT 1")
            await cursor.fetchone()
        else:
            raise RuntimeError("数据库未初始化")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"not ready: {e}")
    return {"status": "ready"}
