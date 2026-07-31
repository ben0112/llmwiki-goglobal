import asyncio
import logging
import secrets
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from uuid import uuid4

import pytest
from redis.cluster import key_slot

from tests.helpers.telemetry_contract import assert_telemetry_event


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
    assert len(reservation.owner_token) == 32
    assert set(reservation.owner_token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
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


async def test_reserve_and_release_emit_stable_json_without_owner_token(caplog):
    quota = _module()
    user_id = uuid4()
    upload_id = uuid4()
    redis = _Redis(set_results=[True], eval_results=[[1, 70], 1, [1]])
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 40}),
        redis,
    )

    with caplog.at_level(logging.INFO, logger="infra.quota"):
        reservation = await service.reserve(user_id, upload_id, 30, ttl_seconds=60)
        assert await service.release(reservation) is True

    expected = {
        "upload_id": str(upload_id),
        "byte_count": 30,
        "replica_role": "api",
    }
    assert_telemetry_event(
        caplog,
        "quota_reserved",
        expected=expected,
        sensitive=(reservation.owner_token, user_id, "redis://private.invalid", "RAW_QUOTA_TEXT"),
    )
    assert_telemetry_event(
        caplog,
        "quota_released",
        expected=expected,
        sensitive=(reservation.owner_token, user_id, "redis://private.invalid", "RAW_QUOTA_TEXT"),
    )


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
    redis = (
        _Redis(set_results=[True, True], eval_results=[[1], 1, [0], 1])
        if method == "finalize"
        else _Redis(eval_results=[[1], [0]])
    )
    service = quota.HostedQuotaService(_Pool({}), redis)
    owner_token = secrets.token_urlsafe(24)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, owner_token)

    assert await getattr(service, method)(reservation) is True
    assert await getattr(service, method)(reservation) is False
    mutation_calls = [call for call in redis.eval_calls if call[1] != 1]
    assert all(call[1] == (4 if method == "finalize" else 3) for call in mutation_calls)
    assert all(call[2][-2:] == (owner_token, "10") for call in mutation_calls)


@pytest.mark.parametrize(
    ("redis_status", "expected"),
    [
        (0, "settled"),
        (1, "active"),
        (2, "stale_generation"),
        (4, "marker_conflict"),
    ],
)
async def test_atomic_tus_marker_settlement_distinguishes_absent_owner_and_conflicts(
    redis_status,
    expected,
):
    quota = _module()
    redis = _Redis(eval_results=[[redis_status]])
    service = quota.HostedQuotaService(_Pool({}), redis)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    result = await service.settle_tus_marker_if_absent(reservation)

    assert result is quota.QuotaMarkerSettlementStatus(expected)
    _, numkeys, args = redis.eval_calls[0]
    assert numkeys == 4
    assert args[3] == f"tus:reservation:{{{reservation.user_id}}}:{{{reservation.upload_id}}}"
    assert args[-3:] == (str(reservation.upload_id), reservation.owner_token, "10")


async def test_atomic_tus_marker_settlement_malformed_snapshot_fails_closed():
    quota = _module()
    service = quota.HostedQuotaService(_Pool({}), _Redis(eval_results=[[5]]))
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    with pytest.raises(quota.QuotaUnavailable):
        await service.settle_tus_marker_if_absent(reservation)


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


@pytest.mark.parametrize(
    "owner_token",
    [
        "short",
        "a" * 31,
        "a" * 33,
        "a" * 31 + "\n",
        "é" * 32,
        "a" * 31 + "+",
    ],
)
def test_reservation_rejects_owner_tokens_outside_generated_canonical_format(owner_token):
    quota = _module()

    with pytest.raises(ValueError, match="owner_token"):
        quota.QuotaReservation(uuid4(), uuid4(), 10, owner_token)


async def test_settlement_revalidates_a_forged_reservation_before_redis():
    quota = _module()
    redis = _Redis(eval_results=[[1]])
    service = quota.HostedQuotaService(_Pool({}), redis)
    forged = object.__new__(quota.QuotaReservation)
    object.__setattr__(forged, "user_id", uuid4())
    object.__setattr__(forged, "upload_id", uuid4())
    object.__setattr__(forged, "bytes", 10)
    object.__setattr__(forged, "owner_token", "short")

    with pytest.raises(ValueError, match="owner_token"):
        await service.release(forged)

    assert redis.eval_calls == []


async def test_settlement_cas_includes_immutable_reservation_bytes():
    quota = _module()
    redis = _Redis(eval_results=[[2]])
    service = quota.HostedQuotaService(_Pool({}), redis)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    assert await service.release(reservation) is False

    _, key_count, args = redis.eval_calls[0]
    assert key_count == 3
    assert args[-3:] == (str(reservation.upload_id), reservation.owner_token, "10")


async def test_finalize_acquires_user_lock_and_fences_four_key_lua():
    quota = _module()
    redis = _Redis(set_results=[True], eval_results=[[1], 1])
    service = quota.HostedQuotaService(_Pool({}), redis)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    assert await service.finalize(reservation) is True

    assert redis.set_calls[0][0][0] == quota.quota_lock_key(reservation.user_id)
    _, key_count, args = redis.eval_calls[0]
    assert key_count == 4
    assert args[:4] == quota.quota_keys(reservation.user_id)
    assert args[-3:] == (str(reservation.upload_id), reservation.owner_token, "10")


async def test_finalize_reacquires_when_lua_reports_lock_lost():
    quota = _module()
    redis = _Redis(set_results=[True, True], eval_results=[[4], 0, [1], 1])
    service = quota.HostedQuotaService(_Pool({}), redis, max_lock_attempts=2)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    assert await service.finalize(reservation) is True

    assert len(redis.set_calls) == 2


@pytest.mark.parametrize("status,expected", [(1, True), (0, False), (2, False)])
async def test_renew_is_owner_bytes_cas_under_user_lock(status, expected):
    quota = _module()
    redis = _Redis(set_results=[True], eval_results=[[status], 1])
    service = quota.HostedQuotaService(_Pool({}), redis)
    reservation = quota.QuotaReservation(uuid4(), uuid4(), 10, secrets.token_urlsafe(24))

    assert await service.renew(reservation, ttl_seconds=60) is expected

    _, key_count, args = redis.eval_calls[0]
    assert key_count == 4
    assert args[:4] == quota.quota_keys(reservation.user_id)
    assert args[-4:-1] == (str(reservation.upload_id), reservation.owner_token, "10")
    assert args[-1] == "60000"


class _AdmissionOutcomeRedis:
    def __init__(self, *, cancel_after_write: bool = False, fail_before_write: bool = False):
        self.cancel_after_write = cancel_after_write
        self.fail_before_write = fail_before_write
        self.owner = None
        self.byte_count = None
        self.set_calls = []
        self.eval_calls = []

    async def set(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))
        return True

    async def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        if numkeys == 4:
            if self.fail_before_write:
                raise ConnectionError("response lost before execution")
            self.owner = args[-5]
            self.byte_count = args[-2]
            if self.cancel_after_write:
                raise asyncio.CancelledError
            raise ConnectionError("response lost after execution")
        if numkeys == 3:
            upload_id, owner, byte_count = args[-3:]
            del upload_id
            if owner == self.owner and byte_count == self.byte_count:
                self.owner = None
                self.byte_count = None
                return [1]
            return [2]
        return 1


async def test_admission_response_loss_best_effort_releases_written_generation():
    quota = _module()
    redis = _AdmissionOutcomeRedis()
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )

    with pytest.raises(quota.QuotaUnavailable) as raised:
        await service.reserve(uuid4(), uuid4(), 10, ttl_seconds=60)

    assert isinstance(raised.value.__cause__, ConnectionError)
    assert redis.owner is None


async def test_admission_cancellation_after_write_releases_then_propagates_cancel():
    quota = _module()
    redis = _AdmissionOutcomeRedis(cancel_after_write=True)
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.reserve(uuid4(), uuid4(), 10, ttl_seconds=60)

    assert redis.owner is None


async def test_unknown_admission_cleanup_cannot_delete_an_old_generation():
    quota = _module()
    redis = _AdmissionOutcomeRedis(fail_before_write=True)
    redis.owner = secrets.token_urlsafe(24)
    redis.byte_count = "20"
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )

    with pytest.raises(quota.QuotaUnavailable):
        await service.reserve(uuid4(), uuid4(), 10, ttl_seconds=60)

    assert redis.owner is not None
    assert redis.byte_count == "20"


class _BlockingAdmissionRedis(_AdmissionOutcomeRedis):
    def __init__(self):
        super().__init__()
        self.written = asyncio.Event()
        self.allow_response = asyncio.Event()

    async def eval(self, script, numkeys, *args):
        if numkeys == 4:
            self.eval_calls.append((script, numkeys, args))
            self.owner = args[-5]
            self.byte_count = args[-2]
            self.written.set()
            await self.allow_response.wait()
            return [1, 123]
        return await super().eval(script, numkeys, *args)


class _NeverRespondingAdmissionRedis(_AdmissionOutcomeRedis):
    def __init__(self):
        super().__init__()
        self.written_owner = None

    async def eval(self, script, numkeys, *args):
        if numkeys == 4:
            self.eval_calls.append((script, numkeys, args))
            self.owner = args[-5]
            self.written_owner = self.owner
            self.byte_count = args[-2]
            await asyncio.Event().wait()
        return await super().eval(script, numkeys, *args)


async def test_admission_timeout_cancels_command_cas_releases_and_never_logs_owner_token(caplog):
    quota = _module()
    redis = _NeverRespondingAdmissionRedis()
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )
    service._redis_command_timeout_seconds = 0.01

    with pytest.raises(quota.QuotaUnavailable) as raised:
        await service.reserve(uuid4(), uuid4(), 10, ttl_seconds=60)

    assert isinstance(raised.value.__cause__, TimeoutError)
    assert redis.owner is None
    assert redis.written_owner not in caplog.text
    assert redis.written_owner not in str(raised.value)


async def test_outer_cancellation_waits_for_admission_response_then_cas_releases():
    quota = _module()
    redis = _BlockingAdmissionRedis()
    service = quota.HostedQuotaService(
        _Pool({"storage_limit_bytes": 100, "committed_bytes": 0}),
        redis,
    )
    reserve = asyncio.create_task(service.reserve(uuid4(), uuid4(), 10, ttl_seconds=60))
    await redis.written.wait()

    reserve.cancel()
    await asyncio.sleep(0)
    assert not reserve.done()
    redis.allow_response.set()
    with pytest.raises(asyncio.CancelledError):
        await reserve

    assert redis.owner is None


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
    listener_task = asyncio.create_task(idle())
    cleanup = asyncio.create_task(idle())

    class Listener:
        async def close(self):
            listener_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await listener_task

    listener = Listener()

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
    assert listener_task.cancelled()
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
