import asyncio
import logging
import socket
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware as _BaseCORSMiddleware
from starlette.types import Receive, Scope, Send


class CORSMiddleware(_BaseCORSMiddleware):
    """CORS middleware that passes WebSocket connections through.

    WebSocket auth is handled by JWT verification in the handler, not by
    origin checks. HTTP requests still get full CORS protection.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class ReplicaIdentityMiddleware:
    """Expose a replica identity only to scaled smoke tests."""

    def __init__(self, app, *, stage: str, instance_id: str):
        self.app = app
        self.instance_id = instance_id if stage == "test" else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_identity(message):
            if self.instance_id is not None and message["type"] in {
                "http.response.start",
                "websocket.accept",
            }:
                headers = list(message.get("headers", []))
                headers.append((b"x-api-instance-id", self.instance_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_identity)


from config import settings  # noqa: E402 - middleware must be defined before app imports.
from infra.tasks import spawn_logged  # noqa: E402 - middleware must be defined before app imports.

logger = logging.getLogger(__name__)

HOSTED_LISTENER_STARTUP_TIMEOUT_SECONDS = 10.0

if settings.SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        send_default_pii=True,
        traces_sample_rate=0.1,
        environment=settings.STAGE,
    )


from routes.corpus_pipeline import router as corpus_pipeline_router  # noqa: E402
from routes.documents import router as documents_router  # noqa: E402
from routes.health import router as health_router  # noqa: E402
from routes.knowledge_bases import router as knowledge_bases_router  # noqa: E402
from routes.me import router as me_router  # noqa: E402
from routes.usage import router as usage_router  # noqa: E402


async def _repair_hosted_derived_drift(pool) -> list[dict]:
    """Reset ready documents with stale derived rows before recovery scans."""
    from infra.db.derived_documents import reset_inconsistent_ready_documents

    rows = await reset_inconsistent_ready_documents(pool)
    if rows:
        logger.warning(
            "Reset %d hosted document(s) with inconsistent derived versions",
            len(rows),
        )
    return rows


async def _recover_durable_extraction_jobs(pool, job_service) -> list:
    """Idempotently backfill extraction jobs in one startup transaction."""
    recovered = []
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "UPDATE documents SET status = 'failed', "
            "error_message = 'Document extraction was cancelled.', updated_at = now() "
            "WHERE status IN ('pending', 'processing') AND version = 0 "
            "AND NOT archived AND source_kind = 'source' "
            "AND (SELECT state::text FROM background_jobs "
            "     WHERE background_jobs.user_id = documents.user_id "
            "     AND background_jobs.job_type = 'document.extract' "
            "     AND background_jobs.document_id = documents.id "
            "     ORDER BY created_at DESC, id DESC LIMIT 1) = 'cancelled'"
        )
        rows = await conn.fetch(
            "SELECT id, user_id, knowledge_base_id FROM documents "
            "WHERE status IN ('pending', 'processing') AND NOT archived "
            "AND source_kind = 'source' ORDER BY id"
        )
        for row in rows:
            job, created = await job_service.ensure_document_extraction_in_transaction(
                conn,
                document_id=row["id"],
                user_id=row["user_id"],
                knowledge_base_id=row["knowledge_base_id"],
                restart_terminal=False,
            )
            if created:
                recovered.append(job)
    return recovered


async def _recover_hosted_extractions(
    pool,
    *,
    job_service,
) -> list:
    """Backfill only the durable ledger; Hosted process-local recovery is gone."""
    recovered = await _recover_durable_extraction_jobs(pool, job_service)
    if recovered:
        logger.info("Recovered %d durable document extraction job(s)", len(recovered))
    return recovered


async def _start_hosted_quota_runtime(pool, redis_url: str):
    """Create and verify the API replica's single shared quota Redis client."""
    from infra.quota import HostedQuotaService
    from infra.redis import create_redis

    redis = create_redis(redis_url)
    try:
        await redis.ping()
    except BaseException:
        await redis.aclose()
        raise
    return redis, HostedQuotaService(pool, redis)


async def _start_hosted_listener(app: FastAPI):
    """Start LISTEN supervision and wait for the first active subscription."""
    from routes.ws import setup_listener

    listener = await setup_listener(settings.listen_database_url)
    app.state.listener_ready = listener.ready
    try:
        await listener.wait_ready(timeout_seconds=HOSTED_LISTENER_STARTUP_TIMEOUT_SECONDS)
    except TimeoutError:
        await listener.close()
        raise RuntimeError("listener subscription timed out") from None
    except BaseException:
        await listener.close()
        raise
    return listener


def _initialize_hosted_rag_service(app: FastAPI, pool) -> None:
    """Attach RAG only after the canonical durable job service exists."""
    app.state.rag_service = None
    if not settings.SERVER_RAG_ENABLED:
        return
    from rag.service import RagService

    app.state.rag_service = RagService(
        pool,
        settings,
        job_service=app.state.job_service,
    )


async def _finish_hosted_startup(app: FastAPI, pool):
    """Build Hosted services and background tasks after core infra is ready."""
    await _repair_hosted_derived_drift(pool)

    s3_service = None
    ocr_service = None
    if settings.AWS_ACCESS_KEY_ID and settings.S3_BUCKET:
        from services.s3 import S3Service

        s3_service = S3Service()
    if s3_service:
        from services.ocr import OCRService

        ocr_service = OCRService(s3_service, pool)

    app.state.s3_service = s3_service
    app.state.ocr_service = ocr_service
    app.state.tus_service = None
    app.state.tus_session_store = None
    app.state.auth_provider = None  # Uses Supabase JWKS auth via deps.py

    from services.hosted import HostedServiceFactory

    app.state.factory = HostedServiceFactory(pool, s3_service, ocr_service)

    await _recover_hosted_extractions(
        pool,
        job_service=app.state.job_service,
    )

    if s3_service is None or app.state.quota_service is None or app.state.redis is None:
        raise RuntimeError("Hosted multipart TUS requires S3, Redis, and quota coordination")
    from infra.tus import HostedTusMultipartService
    from infra.tus_sessions import TusSessionStore

    app.state.tus_session_store = TusSessionStore(app.state.redis)
    app.state.tus_service = HostedTusMultipartService(
        pool,
        s3_service,
        app.state.job_service,
        app.state.quota_service,
        app.state.tus_session_store,
        session_ttl_seconds=settings.TUS_SESSION_TTL_SECONDS,
        stale_seconds=settings.TUS_STALE_SECONDS,
        lock_seconds=settings.TUS_LOCK_SECONDS,
        max_patch_bytes=settings.TUS_MAX_PATCH_BYTES,
    )

    listener = await _start_hosted_listener(app)
    return listener, None


async def _close_hosted_resources(
    app: FastAPI,
    *,
    pool,
    quota_redis,
    listener,
    cleanup_task,
) -> tuple[BaseException, ...]:
    """Attempt every hosted close exactly once and return bounded failures."""
    failures: list[BaseException] = []
    app.state.rag_service = None
    if cleanup_task is not None:
        try:
            cleanup_task.cancel()
            await cleanup_task
        except asyncio.CancelledError:
            pass
        except BaseException as failure:  # noqa: BLE001 - preserve until all resources close.
            failures.append(failure)
    for resource, close_name in (
        (listener, "close"),
        (quota_redis, "aclose"),
        (pool, "close"),
    ):
        if resource is None:
            continue
        try:
            await getattr(resource, close_name)()
        except BaseException as failure:  # noqa: BLE001 - later resources must still close.
            failures.append(failure)
    return tuple(failures)


def _raise_lifespan_failure(
    primary: BaseException | None,
    cleanup_failures: tuple[BaseException, ...],
) -> None:
    from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

    failures = (() if primary is None else (primary,)) + cleanup_failures
    signal = sanitized_boundary_signal_or_unknown(*failures)
    if signal is not None and type(signal) is not BaseException:
        raise signal from None
    if primary is not None:
        raise primary from None
    if cleanup_failures:
        raise cleanup_failures[0] from None


@asynccontextmanager
async def _hosted_lifespan(app: FastAPI):
    # ── Hosted mode ──
    # Prefetch the Supabase JWKS so the first authenticated request doesn't
    # pay the cold-cache cost and so a JWKS outage at boot is visible
    # immediately rather than masked behind the first auth error.
    import asyncpg
    from auth import prefetch_jwks

    pool = None
    quota_redis = None
    listener = None
    cleanup_task = None
    startup_failure: BaseException | None = None
    try:
        await prefetch_jwks()
        pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)
        app.state.pool = pool
        app.state.mode = "hosted"
        app.state.readiness_requires_redis = bool(settings.DURABLE_JOBS_ENABLED)
        app.state.readiness_requires_s3 = bool(
            settings.TUS_MULTIPART_ENABLED or (settings.AWS_ACCESS_KEY_ID and settings.S3_BUCKET)
        )
        app.state.readiness_requires_listener = True
        app.state.job_service = None
        app.state.quota_service = None
        app.state.redis = None
        if settings.DURABLE_JOBS_ENABLED:
            from jobs.service import JobService

            app.state.job_service = JobService(pool)
            _initialize_hosted_rag_service(app, pool)
            quota_redis, app.state.quota_service = await _start_hosted_quota_runtime(
                pool,
                settings.REDIS_URL,
            )
            app.state.redis = quota_redis
        listener, cleanup_task = await _finish_hosted_startup(app, pool)
    except BaseException as failure:  # noqa: BLE001 - close every established startup resource.
        startup_failure = failure
    if startup_failure is not None:
        cleanup_failures = await _close_hosted_resources(
            app,
            pool=pool,
            quota_redis=quota_redis,
            listener=listener,
            cleanup_task=cleanup_task,
        )
        _raise_lifespan_failure(startup_failure, cleanup_failures)

    body_failure: BaseException | None = None
    try:
        yield
    except BaseException as failure:  # noqa: BLE001 - preserve until every resource closes.
        body_failure = failure
    cleanup_failures = await _close_hosted_resources(
        app,
        pool=pool,
        quota_redis=quota_redis,
        listener=listener,
        cleanup_task=cleanup_task,
    )
    if body_failure is not None or cleanup_failures:
        _raise_lifespan_failure(body_failure, cleanup_failures)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    if settings.MODE == "local":
        async with _local_lifespan(app):
            yield
        return
    async with _hosted_lifespan(app):
        yield


async def _local_lifespan_inner(app: FastAPI):
    """Local mode: SQLite + local filesystem + single-user auth."""
    import uuid
    from pathlib import Path

    from infra.auth.local import LocalAuthProvider
    from infra.db.sqlite import create_pool as create_sqlite_pool
    from infra.storage.local import LocalStorageService

    workspace = Path(settings.WORKSPACE_PATH).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "wiki").mkdir(exist_ok=True)
    (workspace / ".llmwiki").mkdir(exist_ok=True)
    (workspace / ".llmwiki" / "cache").mkdir(exist_ok=True)

    db_path = str(workspace / ".llmwiki" / "index.db")
    db = await create_sqlite_pool(db_path)

    local_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "local"))
    auth_provider = LocalAuthProvider(local_user_id)
    storage = LocalStorageService(str(workspace), settings.API_URL)

    # Ensure workspace row exists
    cursor = await db.execute("SELECT id FROM workspace LIMIT 1")
    if not await cursor.fetchone():
        ws_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, ?, '', ?)",
            (ws_id, workspace.name, local_user_id),
        )
        await db.commit()
        logger.info("Initialized local workspace: %s", workspace)

    app.state.mode = "local"
    app.state.pool = None  # No asyncpg pool in local mode
    app.state.sqlite_db = db
    app.state.s3_service = None
    app.state.storage_service = storage
    app.state.ocr_service = None
    app.state.job_service = None
    app.state.rag_service = None
    app.state.quota_service = None
    app.state.redis = None
    app.state.readiness_requires_redis = False
    app.state.readiness_requires_s3 = False
    app.state.readiness_requires_listener = False
    app.state.listener_ready = None
    app.state.tus_service = None
    app.state.tus_session_store = None
    app.state.auth_provider = auth_provider
    app.state.workspace_path = str(workspace)

    from services.local import LocalServiceFactory

    app.state.factory = LocalServiceFactory(db, storage, local_user_id)

    logger.info("Local mode — workspace: %s", workspace)
    return db


@asynccontextmanager
async def _local_lifespan(app: FastAPI):
    db = await _local_lifespan_inner(app)
    from pathlib import Path

    from infra.db.sqlite import create_pool as create_sqlite_pool

    workspace = Path(app.state.workspace_path)
    db_path = str(workspace / ".llmwiki" / "index.db")

    # Each background writer gets its own connection so a commit can't flush
    # another writer's (or a request handler's) open transaction.
    reconcile_db = await create_sqlite_pool(db_path, init_schema=False)
    watcher_db = await create_sqlite_pool(db_path, init_schema=False)
    sweep_db = await create_sqlite_pool(db_path, init_schema=False)

    # 启动对账挂掉必须留痕:它负责接住停机断点的整个提取积压,静默死亡
    # 的表现就是"重启后 CPU 闲置、队列不动"
    from domain.local_processor import reconcile_workspace

    reconcile_task = spawn_logged(reconcile_workspace(reconcile_db, workspace), "startup-reconcile")

    # 磁盘↔索引定期对账:兜住 inotify 收不到的场景(Docker Desktop 宿主侧
    # 拷入 bind mount、容器停机期间的增删),启动即扫一轮
    from domain.watcher import sweep_loop

    sweep_task = spawn_logged(sweep_loop(sweep_db, workspace), "workspace-sweep")

    # 语料自动分类轮询(默认关;设置页/环境变量开启后才会真正跑)
    from routes.corpus_pipeline import auto_loop

    corpus_auto_task = asyncio.create_task(auto_loop(app))

    watcher_task = None
    try:
        from domain.watcher import watch_workspace

        # 同样走留痕封装:几万文件的工作区可能触发 inotify watch 上限
        # (ENOSPC)让 watcher 中途挂掉,裸 task 死了毫无声息
        watcher_task = spawn_logged(watch_workspace(watcher_db, workspace), "file-watcher")
        logger.info("File watcher started")
    except ImportError:
        logger.warning("watchfiles not installed — file watcher disabled")

    try:
        yield
    finally:
        pipeline_task = getattr(app.state, "corpus_pipeline_task", None)
        if pipeline_task is not None and not pipeline_task.done():
            # 逐条状态即时落库,取消只丢弃当前未完成的单条,下轮自动续跑
            pipeline_task.cancel()
            with suppress(asyncio.CancelledError):
                await pipeline_task
        corpus_auto_task.cancel()
        with suppress(asyncio.CancelledError):
            await corpus_auto_task
        reconcile_task.cancel()
        with suppress(asyncio.CancelledError):
            await reconcile_task
        sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweep_task
        if watcher_task:
            watcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await watcher_task
        await reconcile_db.close()
        await watcher_db.close()
        await sweep_db.close()
        await db.close()


app = FastAPI(title="LLM Wiki API", lifespan=lifespan)

app.add_middleware(
    ReplicaIdentityMiddleware,
    stage=settings.STAGE,
    instance_id=socket.gethostname(),
)

# Rate limiting — applied as middleware so every authenticated route gets a
# broad ceiling. Hot endpoints can add tighter `@limiter.limit(...)` overrides.
# Skip in local mode where there's only one user.
if settings.MODE != "local":
    from infra.rate_limit import limiter
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    # Local mode is single-user with no auth and binds wherever the operator
    # publishes it (localhost / LAN / overlay network) — accept any origin so
    # the web app works from whichever address the browser used.
    **({"allow_origin_regex": r"^https?://.*$"} if settings.MODE == "local" else {"allow_origins": [settings.APP_URL]}),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location",
        "Upload-Offset",
        "Upload-Length",
        "Tus-Resumable",
        "Tus-Version",
        "Tus-Max-Size",
        "Tus-Extension",
        "X-Document-Id",
        "X-Job-Id",
        "X-API-Instance-ID",
    ],
)


app.include_router(health_router)
app.include_router(corpus_pipeline_router)
app.include_router(me_router)
app.include_router(usage_router)
app.include_router(knowledge_bases_router)
app.include_router(documents_router)

if settings.MODE == "local":
    from routes.files import router as files_router
    from routes.files import set_workspace_root
    from routes.local_graph import router as local_graph_router
    from routes.local_upload import router as local_upload_router

    app.include_router(local_upload_router)
    app.include_router(files_router)
    app.include_router(local_graph_router)
    set_workspace_root(settings.WORKSPACE_PATH)
else:
    from infra.tus import router as tus_router
    from routes.api_keys import router as api_keys_router
    from routes.graph import router as graph_router
    from routes.jobs import router as jobs_router
    from routes.public import router as public_router
    from routes.rag import router as rag_router
    from routes.ws import router as ws_router

    app.include_router(api_keys_router)
    app.include_router(tus_router)
    app.include_router(graph_router)
    app.include_router(ws_router)
    app.include_router(public_router)
    app.include_router(jobs_router)
    app.include_router(rag_router)
