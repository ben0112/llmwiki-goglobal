from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException


class Probe:
    def __init__(self, name: str, calls: list[str], *, failure: Exception | None = None):
        self.name = name
        self.calls = calls
        self.failure = failure
        self.close_calls = 0

    async def fetchval(self, query: str):
        assert query == "SELECT 1"
        await self._call()
        return 1

    async def ping(self):
        await self._call()
        return True

    async def head_bucket(self):
        await self._call()

    async def close(self):
        self.close_calls += 1

    async def _call(self):
        self.calls.append(self.name)
        if self.failure is not None:
            raise self.failure


def hosted_request(
    *,
    failing: str | None = None,
    requires_redis: bool = True,
    requires_s3: bool = True,
):
    calls: list[str] = []

    def dependency(name: str) -> Probe:
        failure = RuntimeError(f"{name} unavailable credential=never-report-this") if failing == name else None
        return Probe(name, calls, failure=failure)

    state = SimpleNamespace(
        mode="hosted",
        pool=dependency("postgres"),
        redis=dependency("redis"),
        s3_service=dependency("s3"),
        readiness_requires_redis=requires_redis,
        readiness_requires_s3=requires_s3,
        readiness_requires_listener=True,
        listener_ready=asyncio.Event(),
    )
    state.listener_ready.set()
    return SimpleNamespace(app=SimpleNamespace(state=state)), calls


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["postgres", "redis", "s3"])
async def test_hosted_api_readiness_fails_for_each_dependency_without_leaking_details(failing):
    from routes.health import ready

    request, calls = hosted_request(failing=failing)

    with pytest.raises(HTTPException) as raised:
        await ready(request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "not ready"
    assert calls[-1] == failing


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["redis", "s3_service"])
async def test_hosted_api_readiness_rejects_missing_required_dependency_object(missing):
    from routes.health import ready

    request, _calls = hosted_request()
    setattr(request.app.state, missing, None)

    with pytest.raises(HTTPException) as raised:
        await ready(request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "not ready"


@pytest.mark.asyncio
async def test_hosted_api_readiness_rejects_listener_during_initial_subscribe_or_reconnect():
    from routes.health import ready

    request, calls = hosted_request()
    request.app.state.listener_ready.clear()

    with pytest.raises(HTTPException) as raised:
        await ready(request)

    assert raised.value.status_code == 503
    assert raised.value.detail == "not ready"
    assert calls == ["postgres", "redis", "s3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking", ["postgres", "redis", "s3"])
async def test_hosted_api_readiness_timeout_is_bounded_sanitized_and_cancels_probe(
    monkeypatch,
    blocking,
):
    from routes import health as health_routes

    cancelled = asyncio.Event()

    class Dependency(Probe):
        async def _call(self):
            self.calls.append(self.name)
            if self.name != blocking:
                return
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    calls: list[str] = []
    listener_ready = asyncio.Event()
    listener_ready.set()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                mode="hosted",
                pool=Dependency("postgres", calls),
                redis=Dependency("redis", calls),
                s3_service=Dependency("s3", calls),
                readiness_requires_redis=True,
                readiness_requires_s3=True,
                readiness_requires_listener=True,
                listener_ready=listener_ready,
            )
        )
    )
    monkeypatch.setattr(health_routes, "READINESS_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(HTTPException) as raised:
        await asyncio.wait_for(health_routes.ready(request), timeout=0.2)

    assert raised.value.status_code == 503
    assert raised.value.detail == "not ready"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_legacy_hosted_readiness_does_not_require_redis():
    from routes.health import ready

    request, calls = hosted_request(requires_redis=False)
    request.app.state.redis = None

    assert await ready(request) == {"status": "ready"}
    assert calls == ["postgres", "s3"]


@pytest.mark.asyncio
async def test_liveness_does_not_probe_temporarily_unavailable_dependencies():
    from routes.health import health

    _request, calls = hosted_request(failing="postgres")

    assert await health() == {"status": "ok"}
    assert calls == []


@pytest.mark.asyncio
async def test_local_readiness_checks_sqlite_without_redis():
    from routes.health import ready

    calls: list[str] = []

    class Cursor:
        async def fetchone(self):
            calls.append("sqlite-fetch")
            return (1,)

    class Sqlite:
        async def execute(self, query):
            assert query == "SELECT 1"
            calls.append("sqlite-execute")
            return Cursor()

    class RedisMustNotBeUsed:
        async def ping(self):
            raise AssertionError("Local readiness must not require Redis")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                mode="local",
                pool=None,
                sqlite_db=Sqlite(),
                redis=RedisMustNotBeUsed(),
                s3_service=None,
            )
        )
    )

    assert await ready(request) == {"status": "ready"}
    assert calls == ["sqlite-execute", "sqlite-fetch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["postgres", "redis", "s3", "converter"])
async def test_worker_startup_requires_every_role_dependency(monkeypatch, failing):
    from jobs import worker

    calls: list[str] = []

    def dependency(name: str) -> Probe:
        failure = RuntimeError(f"{name} unavailable") if failing == name else None
        return Probe(name, calls, failure=failure)

    pool = dependency("postgres")
    redis = dependency("redis")
    s3 = dependency("s3")

    async def create_pool(_database_url):
        return pool

    class Response:
        def raise_for_status(self):
            if failing == "converter":
                raise httpx.HTTPStatusError(
                    "converter unavailable",
                    request=httpx.Request("GET", "http://converter:8000/health"),
                    response=httpx.Response(503),
                )

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            assert url == "http://converter:8000/health"
            calls.append("converter")
            return Response()

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(httpx, "AsyncClient", Client)

    runtime_settings = SimpleNamespace(
        MODE="hosted",
        DURABLE_JOBS_ENABLED=True,
        REDIS_URL="redis://redis:6379/0",
        DATABASE_URL="postgresql://database/jobs",
        AWS_ACCESS_KEY_ID="access",
        AWS_SECRET_ACCESS_KEY="secret",
        S3_BUCKET="bucket",
        CONVERTER_URL="http://converter:8000",
        CONVERTER_SECRET="converter-secret",
        TUS_MULTIPART_ENABLED=False,
        JOB_LEASE_SECONDS=60,
        JOB_HEARTBEAT_SECONDS=15,
        JOB_DISPATCH_BATCH_SIZE=100,
        JOB_REDELIVER_SECONDS=30,
    )
    ctx = {"redis": redis, "runtime_settings": runtime_settings}

    with pytest.raises((RuntimeError, httpx.HTTPStatusError), match=failing):
        await worker.startup(ctx)

    assert calls[-1] == failing
    assert pool.close_calls == 1
    assert ctx["redis"] is redis
    assert "worker_context" not in ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking", ["postgres", "redis", "s3", "converter"])
async def test_worker_startup_timeout_cancels_probe_and_closes_only_owned_resources(
    monkeypatch,
    blocking,
):
    from jobs import worker

    cancelled = asyncio.Event()

    class Dependency(Probe):
        async def _call(self):
            self.calls.append(self.name)
            if self.name != blocking:
                return
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    calls: list[str] = []
    pool = Dependency("postgres", calls)
    redis = Dependency("redis", calls)
    s3 = Dependency("s3", calls)

    async def create_pool(_database_url):
        return pool

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, _url):
            calls.append("converter")
            if blocking == "converter":
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()
            return Response()

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(worker, "WORKER_STARTUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    ctx = {
        "redis": redis,
        "runtime_settings": SimpleNamespace(
            MODE="hosted",
            DURABLE_JOBS_ENABLED=True,
            REDIS_URL="redis://redis:6379/0",
            DATABASE_URL="postgresql://database/jobs",
            AWS_ACCESS_KEY_ID="access",
            AWS_SECRET_ACCESS_KEY="secret",
            S3_BUCKET="bucket",
            CONVERTER_URL="http://converter:8000",
            CONVERTER_SECRET="converter-secret",
            TUS_MULTIPART_ENABLED=False,
            JOB_LEASE_SECONDS=60,
            JOB_HEARTBEAT_SECONDS=15,
            JOB_DISPATCH_BATCH_SIZE=100,
            JOB_REDELIVER_SECONDS=30,
        ),
    }

    with pytest.raises(RuntimeError, match="worker dependency readiness timed out"):
        await asyncio.wait_for(worker.startup(ctx), timeout=0.2)

    assert cancelled.is_set()
    assert pool.close_calls == 1
    assert s3.close_calls == 1
    assert redis.close_calls == 0
    assert ctx["redis"] is redis
    assert "pool" not in ctx
    assert "s3" not in ctx
    assert "worker_context" not in ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("converter_secret", ["", "   "])
async def test_worker_rejects_blank_converter_secret_before_creating_resources(
    monkeypatch,
    converter_secret,
):
    from jobs import worker

    created = []

    async def create_pool(_database_url):
        created.append("pool")

    monkeypatch.setattr(worker, "_create_pool", create_pool)
    runtime_settings = SimpleNamespace(
        MODE="hosted",
        DURABLE_JOBS_ENABLED=True,
        REDIS_URL="redis://redis:6379/0",
        DATABASE_URL="postgresql://database/jobs",
        CONVERTER_SECRET=converter_secret,
    )

    with pytest.raises(RuntimeError, match="converter secret is required") as raised:
        await worker.startup({"redis": object(), "runtime_settings": runtime_settings})

    assert str(raised.value) == "ARQ durable worker converter secret is required"
    assert created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing,settings_change,expected_s3_closes",
    [
        ("S3", {"AWS_ACCESS_KEY_ID": ""}, 0),
        ("S3", {"AWS_SECRET_ACCESS_KEY": ""}, 0),
        ("S3", {"S3_BUCKET": ""}, 0),
        ("converter", {"CONVERTER_URL": ""}, 1),
    ],
)
async def test_worker_startup_rejects_missing_required_configuration(
    monkeypatch,
    missing,
    settings_change,
    expected_s3_closes,
):
    from jobs import worker

    calls: list[str] = []
    pool = Probe("postgres", calls)
    redis = Probe("redis", calls)
    s3 = Probe("s3", calls)

    async def create_pool(_database_url):
        return pool

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, _url):
            return Response()

    values = {
        "MODE": "hosted",
        "DURABLE_JOBS_ENABLED": True,
        "REDIS_URL": "redis://redis:6379/0",
        "DATABASE_URL": "postgresql://database/jobs",
        "AWS_ACCESS_KEY_ID": "access",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "S3_BUCKET": "bucket",
        "CONVERTER_URL": "http://converter:8000",
        "CONVERTER_SECRET": "converter-secret",
        "TUS_MULTIPART_ENABLED": False,
        "JOB_LEASE_SECONDS": 60,
        "JOB_HEARTBEAT_SECONDS": 15,
        "JOB_DISPATCH_BATCH_SIZE": 100,
        "JOB_REDELIVER_SECONDS": 30,
    }
    values.update(settings_change)
    monkeypatch.setattr(worker, "_create_pool", create_pool)
    monkeypatch.setattr(worker, "_create_s3_service", lambda: s3)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    ctx = {"redis": redis, "runtime_settings": SimpleNamespace(**values)}

    with pytest.raises(RuntimeError, match=missing):
        await worker.startup(ctx)

    assert pool.close_calls == 1
    assert s3.close_calls == expected_s3_closes
    assert redis.close_calls == 0
    assert ctx["redis"] is redis
    assert "pool" not in ctx
    assert "s3" not in ctx
    assert "worker_context" not in ctx


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,expected",
    [("test", "api-replica-1"), ("dev", None), ("prod", None)],
)
async def test_replica_identity_header_is_test_only(stage, expected):
    from main import ReplicaIdentityMiddleware

    sent = []

    async def downstream(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = ReplicaIdentityMiddleware(downstream, stage=stage, instance_id="api-replica-1")

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http"},
        lambda: None,
        send,
    )

    headers = dict(sent[0]["headers"])
    actual = headers.get(b"x-api-instance-id")
    assert (actual.decode() if actual else None) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,expected",
    [("test", "api-replica-1"), ("dev", None), ("prod", None)],
)
async def test_replica_identity_websocket_handshake_header_is_test_only(stage, expected):
    from main import ReplicaIdentityMiddleware

    sent = []

    async def downstream(_scope, _receive, send):
        await send({"type": "websocket.accept", "headers": []})

    middleware = ReplicaIdentityMiddleware(downstream, stage=stage, instance_id="api-replica-1")

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "websocket"},
        lambda: None,
        send,
    )

    headers = dict(sent[0]["headers"])
    actual = headers.get(b"x-api-instance-id")
    assert (actual.decode() if actual else None) == expected
