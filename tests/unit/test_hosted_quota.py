import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis.cluster import key_slot


def _module():
    import importlib

    return importlib.import_module("infra.quota")


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Connection:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def transaction(self):
        return _Transaction()

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.row


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Pool:
    def __init__(self, row):
        self.connection = _Connection(row)

    def acquire(self):
        return _Acquire(self.connection)


class _Redis:
    def __init__(self, *, set_results=None, eval_results=None):
        self.set_results = list(set_results or [])
        self.eval_results = list(eval_results or [])
        self.set_calls = []
        self.eval_calls = []

    async def set(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))
        return self.set_results.pop(0)

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        return self.eval_results.pop(0)


def test_quota_keys_share_one_redis_cluster_slot_per_user_and_isolate_tenants():
    quota = _module()
    first = uuid4()
    second = uuid4()
    first_keys = quota.quota_keys(first)
    second_keys = quota.quota_keys(second)

    assert len({key_slot(key.encode()) for key in first_keys}) == 1
    assert len({key_slot(key.encode()) for key in second_keys}) == 1
    assert set(first_keys).isdisjoint(second_keys)
    assert all(f"{{{first}}}" in key for key in first_keys)


async def test_reserve_reads_committed_usage_then_writes_under_token_lock():
    quota = _module()
    user_id = uuid4()
    upload_id = uuid4()
    pool = _Pool({"storage_limit_bytes": 100, "committed_bytes": 40})
    redis = _Redis(set_results=[True], eval_results=[[1, 70], 1])
    service = quota.HostedQuotaService(pool, redis)

    reservation = await service.reserve(user_id, upload_id, 30, ttl_seconds=60)

    assert reservation.user_id == user_id
    assert reservation.upload_id == upload_id
    assert reservation.bytes == 30
    assert reservation.owner_token
    with pytest.raises(FrozenInstanceError):
        reservation.bytes = 10
    assert "SUM(d.file_size)" in pool.connection.calls[0][0]
    assert pool.connection.calls[0][1] == (user_id,)
    lock_args, lock_kwargs = redis.set_calls[0]
    assert lock_args[0] == quota.quota_lock_key(user_id)
    assert lock_args[1]
    assert lock_kwargs == {"nx": True, "px": quota.DEFAULT_LOCK_TTL_MS}
    _, key_count, script_args = redis.eval_calls[0]
    assert key_count == 4
    assert script_args[:4] == quota.quota_keys(user_id)
    assert str(upload_id) in script_args
    assert reservation.owner_token in script_args


async def test_final_capacity_rejection_is_typed_and_lock_is_released():
    quota = _module()
    pool = _Pool({"storage_limit_bytes": 100, "committed_bytes": 90})
    redis = _Redis(set_results=[True], eval_results=[[2, 110, 100], 1])
    service = quota.HostedQuotaService(pool, redis)

    with pytest.raises(quota.QuotaExceeded) as raised:
        await service.reserve(uuid4(), uuid4(), 20, ttl_seconds=60)

    assert raised.value.used_bytes == 110
    assert raised.value.limit_bytes == 100
    assert len(redis.eval_calls) == 2


async def test_lock_lost_before_atomic_write_reacquires_and_rereads_postgres():
    quota = _module()
    pool = _Pool({"storage_limit_bytes": 100, "committed_bytes": 10})
    redis = _Redis(set_results=[True, True], eval_results=[[3], 0, [1, 30], 1])
    service = quota.HostedQuotaService(pool, redis, max_lock_attempts=2)

    reservation = await service.reserve(uuid4(), uuid4(), 20, ttl_seconds=60)

    assert reservation.bytes == 20
    assert len(pool.connection.calls) == 2
    assert len(redis.set_calls) == 2


@pytest.mark.parametrize("method", ["finalize", "release"])
async def test_settlement_is_owner_token_cas_and_repeated_safe(method):
    quota = _module()
    redis = _Redis(eval_results=[[1], [0]])
    service = quota.HostedQuotaService(_Pool({}), redis)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, "opaque-owner")

    assert await getattr(service, method)(reservation) is True
    assert await getattr(service, method)(reservation) is False
    assert all(call[1] == 3 for call in redis.eval_calls)
    assert all(call[2][-1] == "opaque-owner" for call in redis.eval_calls)


@pytest.mark.parametrize(
    "user_id,upload_id,byte_count,ttl_seconds",
    [
        ("not-a-uuid", uuid4(), 1, 60),
        (uuid4(), "not-a-uuid", 1, 60),
        (uuid4(), uuid4(), True, 60),
        (uuid4(), uuid4(), 0, 60),
        (uuid4(), uuid4(), -1, 60),
        (uuid4(), uuid4(), 1, 0),
    ],
)
async def test_invalid_reservation_inputs_fail_before_redis(user_id, upload_id, byte_count, ttl_seconds):
    quota = _module()
    redis = _Redis()
    service = quota.HostedQuotaService(_Pool({}), redis)

    with pytest.raises(ValueError):
        await service.reserve(user_id, upload_id, byte_count, ttl_seconds=ttl_seconds)

    assert redis.set_calls == []
    assert redis.eval_calls == []


async def test_malformed_or_unavailable_redis_fails_closed_without_leaking_owner_token():
    quota = _module()
    redis = _Redis(set_results=[True], eval_results=[[4], 1])
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )

    with pytest.raises(quota.QuotaUnavailable) as raised:
        await service.reserve(uuid4(), uuid4(), 1, ttl_seconds=60)

    assert "token" not in str(raised.value).lower()


async def test_hosted_runtime_builds_one_pinged_shared_quota_service(monkeypatch):
    import infra.redis as redis_module
    from main import _start_hosted_quota_runtime

    class Client:
        def __init__(self):
            self.pings = 0
            self.closes = 0

        async def ping(self):
            self.pings += 1

        async def aclose(self):
            self.closes += 1

    client = Client()
    monkeypatch.setattr(redis_module, "create_redis", lambda url: client)
    pool = object()

    returned_client, service = await _start_hosted_quota_runtime(pool, "redis://quota.test/0")

    assert returned_client is client
    assert service._pool is pool
    assert service._redis is client
    assert client.pings == 1
    assert client.closes == 0


async def test_hosted_runtime_ping_failure_closes_redis_before_failing(monkeypatch):
    import infra.redis as redis_module
    from main import _start_hosted_quota_runtime

    class Client:
        def __init__(self):
            self.closes = 0

        async def ping(self):
            raise ConnectionError("credential=secret")

        async def aclose(self):
            self.closes += 1

    client = Client()
    monkeypatch.setattr(redis_module, "create_redis", lambda url: client)

    with pytest.raises(ConnectionError, match="credential"):
        await _start_hosted_quota_runtime(object(), "redis://quota.test/0")

    assert client.closes == 1


async def test_hosted_lifespan_closes_quota_redis_and_pool_when_app_body_fails(monkeypatch):
    import asyncpg
    import auth
    import main

    class Pool:
        def __init__(self):
            self.closes = 0

        async def close(self):
            self.closes += 1

    class Redis:
        def __init__(self):
            self.closes = 0

        async def aclose(self):
            self.closes += 1

    async def idle():
        await asyncio.Event().wait()

    pool = Pool()
    redis = Redis()
    listener = asyncio.create_task(idle())
    cleanup = asyncio.create_task(idle())

    async def no_op():
        return None

    async def start_quota(_pool, _url):
        return redis, object()

    async def finish_startup(_app, _pool):
        return listener, cleanup

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "REDIS_URL", "redis://quota.test/0")
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(main, "_start_hosted_quota_runtime", start_quota)
    monkeypatch.setattr(main, "_finish_hosted_startup", finish_startup)
    app = SimpleNamespace(state=SimpleNamespace())

    with pytest.raises(RuntimeError, match="app body failed"):
        async with main.lifespan(app):
            raise RuntimeError("app body failed")

    assert redis.closes == 1
    assert pool.closes == 1
    assert listener.cancelled()
    assert cleanup.cancelled()


async def test_hosted_lifespan_closes_established_infra_when_later_startup_fails(monkeypatch):
    import asyncpg
    import auth
    import main

    class Resource:
        def __init__(self):
            self.closes = 0

        async def close(self):
            self.closes += 1

        async def aclose(self):
            self.closes += 1

    pool = Resource()
    redis = Resource()

    async def no_op():
        return None

    async def start_quota(_pool, _url):
        return redis, object()

    async def fail_startup(_app, _pool):
        raise RuntimeError("listener startup failed")

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "REDIS_URL", "redis://quota.test/0")
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(main, "_start_hosted_quota_runtime", start_quota)
    monkeypatch.setattr(main, "_finish_hosted_startup", fail_startup)
    app = SimpleNamespace(state=SimpleNamespace())

    with pytest.raises(RuntimeError, match="listener startup failed"):
        async with main.lifespan(app):
            pass

    assert redis.closes == 1
    assert pool.closes == 1


async def _async_value(value):
    return value
