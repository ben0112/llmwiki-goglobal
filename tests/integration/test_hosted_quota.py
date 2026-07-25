from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from infra.quota import HostedQuotaService, QuotaExceeded, QuotaUnavailable, quota_keys
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
