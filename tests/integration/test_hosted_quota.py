from __future__ import annotations

import asyncio
import os
import secrets
import time
from uuid import uuid4

import pytest
from infra.quota import HostedQuotaService, QuotaExceeded, QuotaReservation, QuotaUnavailable, quota_keys
from redis.asyncio import Redis

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def redis_client():
    url = os.environ.get("QUOTA_REDIS_TEST_URL")
    if not url:
        pytest.skip("QUOTA_REDIS_TEST_URL is required for the real quota integration suite")
    client = Redis.from_url(url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
    await client.ping()
    try:
        yield client
    finally:
        await client.aclose()


async def _seed_user(pool, *, limit: int):
    user_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, storage_limit_bytes) VALUES ($1, $2, $3)",
        user_id,
        f"{user_id}@quota.test",
        limit,
    )
    return user_id


async def _seed_committed_document(pool, user_id, *, byte_count: int):
    kb_id = uuid4()
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        kb_id,
        user_id,
        f"KB {kb_id}",
        f"kb-{kb_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, file_type, file_size, status) "
        "VALUES ($1, $2, $3, 'committed.pdf', '/', 'pdf', $4, 'ready')",
        document_id,
        kb_id,
        user_id,
        byte_count,
    )


async def _clear(redis_client, *user_ids):
    keys = [key for user_id in user_ids for key in quota_keys(user_id)]
    if keys:
        await redis_client.delete(*keys)


async def _reservation_state(redis_client, user_id):
    _, reservations_key, bytes_key, tokens_key = quota_keys(user_id)
    return (
        await redis_client.zrange(reservations_key, 0, -1, withscores=True),
        await redis_client.hgetall(bytes_key),
        await redis_client.hgetall(tokens_key),
    )


async def _seed_raw_reservation(
    redis_client,
    user_id,
    member: str,
    owner: str,
    *,
    expired: bool = False,
    raw_bytes: str = "10",
    score: float | int | None = None,
):
    _, reservations_key, bytes_key, tokens_key = quota_keys(user_id)
    score = score if score is not None else (0 if expired else int(time.time() * 1000) + 60_000)
    await redis_client.zadd(reservations_key, {member: score})
    await redis_client.hset(bytes_key, member, raw_bytes)
    await redis_client.hset(tokens_key, member, owner)


async def test_committed_plus_live_reservations_determine_acceptance(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    await _seed_committed_document(pool, user_id, byte_count=40)
    service = HostedQuotaService(pool, redis_client)
    try:
        first = await service.reserve(user_id, uuid4(), 30, ttl_seconds=60)
        second = await service.reserve(user_id, uuid4(), 30, ttl_seconds=60)
        with pytest.raises(QuotaExceeded):
            await service.reserve(user_id, uuid4(), 1, ttl_seconds=60)
        assert await service.finalize(first) is True
        assert await service.release(second) is True
    finally:
        await _clear(redis_client, user_id)


async def test_two_replicas_contending_for_final_bytes_accept_exactly_one(pool, redis_client):
    user_id = await _seed_user(pool, limit=10)
    services = [HostedQuotaService(pool, redis_client), HostedQuotaService(pool, redis_client)]
    ready = asyncio.Event()
    arrived = 0
    guard = asyncio.Lock()

    async def reserve(service):
        nonlocal arrived
        async with guard:
            arrived += 1
            if arrived == 2:
                ready.set()
        await ready.wait()
        try:
            return await service.reserve(user_id, uuid4(), 10, ttl_seconds=60)
        except QuotaExceeded as exc:
            return exc

    try:
        results = await asyncio.gather(*(reserve(service) for service in services))
        assert sum(not isinstance(result, QuotaExceeded) for result in results) == 1
        assert sum(isinstance(result, QuotaExceeded) for result in results) == 1
    finally:
        await _clear(redis_client, user_id)


async def test_owner_token_prevents_old_generation_from_releasing_replacement(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    upload_id = uuid4()
    service = HostedQuotaService(pool, redis_client)
    try:
        old = await service.reserve(user_id, upload_id, 10, ttl_seconds=60)
        new = await service.reserve(user_id, upload_id, 20, ttl_seconds=60)
        assert old.owner_token != new.owner_token
        assert await service.release(old) is False
        with pytest.raises(QuotaExceeded):
            await service.reserve(user_id, uuid4(), 81, ttl_seconds=60)
        assert await service.release(new) is True
        assert await service.release(new) is False
    finally:
        await _clear(redis_client, user_id)


async def test_expired_cleanup_and_tenant_isolation(pool, redis_client):
    first_user = await _seed_user(pool, limit=10)
    second_user = await _seed_user(pool, limit=10)
    first = HostedQuotaService(pool, redis_client)
    second = HostedQuotaService(pool, redis_client)
    try:
        reservation = await first.reserve(first_user, uuid4(), 10, ttl_seconds=60)
        await redis_client.zadd(quota_keys(first_user)[1], {str(reservation.upload_id): 0})
        assert await first.cleanup_expired(first_user) == 1
        await first.reserve(first_user, uuid4(), 10, ttl_seconds=60)
        await second.reserve(second_user, uuid4(), 10, ttl_seconds=60)
    finally:
        await _clear(redis_client, first_user, second_user)


async def test_malformed_reservation_index_fails_closed(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    keys = quota_keys(user_id)
    try:
        await redis_client.zadd(keys[1], {str(uuid4()): 9_999_999_999_999})
        with pytest.raises(QuotaUnavailable):
            await HostedQuotaService(pool, redis_client).reserve(user_id, uuid4(), 1, ttl_seconds=60)
    finally:
        await _clear(redis_client, user_id)


class _ExpireFirstLockBeforeEval:
    def __init__(self, redis_client):
        self._redis = redis_client
        self.expired = False

    def __getattr__(self, name):
        return getattr(self._redis, name)

    async def eval(self, script, numkeys, *args):
        if numkeys == 4 and not self.expired:
            self.expired = True
            await self._redis.delete(args[0])
        return await self._redis.eval(script, numkeys, *args)


async def test_stale_writer_after_lock_expiry_reacquires_before_reserving(pool, redis_client):
    user_id = await _seed_user(pool, limit=10)
    proxy = _ExpireFirstLockBeforeEval(redis_client)
    try:
        reservation = await HostedQuotaService(pool, proxy).reserve(user_id, uuid4(), 10, ttl_seconds=60)
        assert proxy.expired is True
        assert reservation.bytes == 10
        with pytest.raises(QuotaExceeded):
            await HostedQuotaService(pool, redis_client).reserve(user_id, uuid4(), 1, ttl_seconds=60)
    finally:
        await _clear(redis_client, user_id)


@pytest.mark.parametrize(
    "member,owner",
    [
        ("NOT-A-UUID", secrets.token_urlsafe(24)),
        ("AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", secrets.token_urlsafe(24)),
        (str(uuid4()), "short"),
        (str(uuid4()), "a" * 31 + "+"),
        (str(uuid4()), "a" * 31 + "\n"),
        (str(uuid4()), "é" * 32),
    ],
)
async def test_internally_consistent_noncanonical_record_blocks_admission_without_mutation(
    pool,
    redis_client,
    member,
    owner,
):
    user_id = await _seed_user(pool, limit=100)
    service = HostedQuotaService(pool, redis_client)
    try:
        await _seed_raw_reservation(redis_client, user_id, member, owner)
        before = await _reservation_state(redis_client, user_id)

        with pytest.raises(QuotaUnavailable):
            await service.reserve(user_id, uuid4(), 1, ttl_seconds=60)

        assert await _reservation_state(redis_client, user_id) == before
    finally:
        await _clear(redis_client, user_id)


async def test_cleanup_fails_closed_on_nonexpired_malformed_record(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    service = HostedQuotaService(pool, redis_client)
    try:
        await _seed_raw_reservation(redis_client, user_id, str(uuid4()), "short")
        before = await _reservation_state(redis_client, user_id)

        with pytest.raises(QuotaUnavailable):
            await service.cleanup_expired(user_id)

        assert await _reservation_state(redis_client, user_id) == before
    finally:
        await _clear(redis_client, user_id)


async def test_cleanup_does_not_guess_or_delete_expired_malformed_record(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    service = HostedQuotaService(pool, redis_client)
    try:
        await _seed_raw_reservation(redis_client, user_id, "malformed-expired-member", "short", expired=True)
        before = await _reservation_state(redis_client, user_id)

        with pytest.raises(QuotaUnavailable):
            await service.cleanup_expired(user_id)

        assert await _reservation_state(redis_client, user_id) == before
    finally:
        await _clear(redis_client, user_id)


@pytest.mark.parametrize(
    "raw_bytes,score",
    [
        ("01", None),
        ("1.5", None),
        ("10", 1.5),
    ],
)
async def test_noncanonical_bytes_or_score_remain_fail_closed(pool, redis_client, raw_bytes, score):
    user_id = await _seed_user(pool, limit=100)
    service = HostedQuotaService(pool, redis_client)
    try:
        await _seed_raw_reservation(
            redis_client,
            user_id,
            str(uuid4()),
            secrets.token_urlsafe(24),
            raw_bytes=raw_bytes,
            score=score,
        )
        before = await _reservation_state(redis_client, user_id)

        with pytest.raises(QuotaUnavailable):
            await service.reserve(user_id, uuid4(), 1, ttl_seconds=60)

        assert await _reservation_state(redis_client, user_id) == before
    finally:
        await _clear(redis_client, user_id)


async def test_canonical_raw_record_with_dash_and_underscore_remains_operable(pool, redis_client):
    user_id = await _seed_user(pool, limit=10)
    upload_id = uuid4()
    owner = "A_b-" * 8
    service = HostedQuotaService(pool, redis_client)
    try:
        await _seed_raw_reservation(redis_client, user_id, str(upload_id), owner)

        with pytest.raises(QuotaExceeded):
            await service.reserve(user_id, uuid4(), 1, ttl_seconds=60)
        assert await service.release(QuotaReservation(user_id, upload_id, 10, owner)) is True
        assert await _reservation_state(redis_client, user_id) == ([], {}, {})
    finally:
        await _clear(redis_client, user_id)


class _PauseAfterCommittedRead(HostedQuotaService):
    def __init__(self, *args, read_done: asyncio.Event, allow_write: asyncio.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_done = read_done
        self.allow_write = allow_write

    async def _read_committed_usage(self, user_id):
        usage = await super()._read_committed_usage(user_id)
        self.read_done.set()
        await self.allow_write.wait()
        return usage


class _ObserveLockAttempt:
    def __init__(self, redis_client, attempted: asyncio.Event):
        self._redis = redis_client
        self.attempted = attempted

    def __getattr__(self, name):
        return getattr(self._redis, name)

    async def set(self, *args, **kwargs):
        self.attempted.set()
        return await self._redis.set(*args, **kwargs)


async def test_finalize_waits_for_inflight_admission_snapshot_before_removing_live_bytes(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    try:
        b_service = HostedQuotaService(pool, redis_client)
        b = await b_service.reserve(user_id, uuid4(), 60, ttl_seconds=60)
        read_done = asyncio.Event()
        allow_a_write = asyncio.Event()
        a_service = _PauseAfterCommittedRead(
            pool,
            redis_client,
            read_done=read_done,
            allow_write=allow_a_write,
        )
        a_task = asyncio.create_task(a_service.reserve(user_id, uuid4(), 60, ttl_seconds=60))
        await read_done.wait()
        await _seed_committed_document(pool, user_id, byte_count=60)
        finalize_attempted = asyncio.Event()
        finalize_service = HostedQuotaService(pool, _ObserveLockAttempt(redis_client, finalize_attempted))
        finalize_task = asyncio.create_task(finalize_service.finalize(b))
        await finalize_attempted.wait()
        await asyncio.sleep(0)
        assert not finalize_task.done()

        allow_a_write.set()
        with pytest.raises(QuotaExceeded):
            await a_task
        assert await finalize_task is True
        assert await _reservation_state(redis_client, user_id) == ([], {}, {})
    finally:
        await _clear(redis_client, user_id)


async def test_renew_never_revives_missing_expired_or_stale_generation(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    upload_id = uuid4()
    service = HostedQuotaService(pool, redis_client)
    missing = QuotaReservation(user_id, upload_id, 10, secrets.token_urlsafe(24))
    try:
        assert await service.renew(missing, ttl_seconds=60) is False
        assert await _reservation_state(redis_client, user_id) == ([], {}, {})

        expired = await service.reserve(user_id, upload_id, 10, ttl_seconds=60)
        await redis_client.zadd(quota_keys(user_id)[1], {str(upload_id): 0})
        expired_state = await _reservation_state(redis_client, user_id)
        assert await service.renew(expired, ttl_seconds=60) is False
        assert await _reservation_state(redis_client, user_id) == expired_state

        await _clear(redis_client, user_id)
        old = await service.reserve(user_id, upload_id, 10, ttl_seconds=60)
        new = await service.reserve(user_id, upload_id, 20, ttl_seconds=60)
        new_state = await _reservation_state(redis_client, user_id)
        assert await service.renew(old, ttl_seconds=60) is False
        assert await _reservation_state(redis_client, user_id) == new_state
        assert await service.renew(new, ttl_seconds=120) is True
        assert (await _reservation_state(redis_client, user_id))[0][0][1] > new_state[0][0][1]
    finally:
        await _clear(redis_client, user_id)


class _AdmissionResponseProxy:
    def __init__(self, redis_client, *, block_response: bool = False):
        self._redis = redis_client
        self.block_response = block_response
        self.written = asyncio.Event()
        self.allow_response = asyncio.Event()
        self.intercepted = False

    def __getattr__(self, name):
        return getattr(self._redis, name)

    async def eval(self, script, numkeys, *args):
        result = await self._redis.eval(script, numkeys, *args)
        if numkeys == 4 and not self.intercepted:
            self.intercepted = True
            self.written.set()
            if self.block_response:
                await self.allow_response.wait()
                return result
            raise ConnectionError("response lost after Redis executed admission")
        return result


async def test_real_redis_admission_response_loss_cas_releases_written_generation(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    proxy = _AdmissionResponseProxy(redis_client)
    try:
        with pytest.raises(QuotaUnavailable):
            await HostedQuotaService(pool, proxy).reserve(user_id, uuid4(), 10, ttl_seconds=60)
        assert await _reservation_state(redis_client, user_id) == ([], {}, {})
    finally:
        await _clear(redis_client, user_id)


async def test_real_redis_outer_cancel_waits_for_response_and_cleans_written_generation(pool, redis_client):
    user_id = await _seed_user(pool, limit=100)
    proxy = _AdmissionResponseProxy(redis_client, block_response=True)
    reserve = asyncio.create_task(HostedQuotaService(pool, proxy).reserve(user_id, uuid4(), 10, ttl_seconds=60))
    try:
        await proxy.written.wait()
        reserve.cancel()
        await asyncio.sleep(0)
        assert not reserve.done()
        proxy.allow_response.set()
        with pytest.raises(asyncio.CancelledError):
            await reserve
        assert await _reservation_state(redis_client, user_id) == ([], {}, {})
    finally:
        proxy.allow_response.set()
        if not reserve.done():
            reserve.cancel()
        await _clear(redis_client, user_id)
