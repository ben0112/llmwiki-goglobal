import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from redis.asyncio import Redis

pytestmark = pytest.mark.asyncio


def _module():
    import importlib

    return importlib.import_module("infra.tus_sessions")


@pytest.fixture
async def redis_client():
    url = os.environ.get("TUS_REDIS_TEST_URL")
    if not url:
        pytest.skip("TUS_REDIS_TEST_URL is required for the real Redis integration suite")
    client = Redis.from_url(url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
    await client.ping()
    try:
        yield client
    finally:
        await client.aclose()


def _session(module, upload_id, user_id, kb_id, *, total=10):
    now = datetime.now(UTC)
    return module.TusSession(
        upload_id=upload_id,
        user_id=user_id,
        knowledge_base_id=kb_id,
        filename="safe.pdf",
        content_type="application/pdf",
        total_length=total,
        offset=0,
        s3_key=f"uploads/{user_id}/{upload_id}.pdf",
        multipart_upload_id=f"multipart-{upload_id}",
        parts=(),
        created_at=now,
        updated_at=now,
        state=module.TusSessionState.UPLOADING,
        document_id=None,
        job_id=None,
        reservation_bytes=total,
    )


@pytest.fixture
async def namespace(redis_client):
    module = _module()
    upload_id = uuid4()
    user_id = uuid4()
    kb_id = uuid4()
    keys = [
        module.session_key(upload_id),
        module.lock_key(upload_id),
        module.reservation_key(user_id, upload_id),
    ]
    try:
        yield module, module.TusSessionStore(redis_client), upload_id, user_id, kb_id
    finally:
        await redis_client.delete(*keys)


async def test_real_redis_concurrent_cas_allows_exactly_one_append(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    assert (
        await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
        is module.SessionCreateStatus.CREATED
    )

    ready = asyncio.Event()
    arrived = 0
    guard = asyncio.Lock()

    async def append(etag):
        nonlocal arrived
        async with guard:
            arrived += 1
            if arrived == 2:
                ready.set()
        await ready.wait()
        return await store.append_part(
            upload_id,
            expected_offset=0,
            byte_count=5,
            part_number=1,
            etag=etag,
            ttl_seconds=60,
        )

    first, second = await asyncio.gather(append("etag-a"), append("etag-b"))
    assert sorted([first.status.value, second.status.value]) == ["appended", "offset_mismatch"]
    session = await store.get(upload_id)
    assert session is not None
    assert session.offset == 5
    assert len(session.parts) == 1
    assert session.parts[0].part_number == 1
    assert session.parts[0].etag in {"etag-a", "etag-b"}


async def test_real_redis_append_is_atomic_and_refreshes_session_ttl(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
    key = module.session_key(upload_id)
    before_session = await store.get(upload_id)
    assert before_session.created_at == before_session.updated_at
    await redis_client.pexpire(key, 100)

    result = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=5,
        part_number=1,
        etag="etag-1",
        ttl_seconds=3,
    )

    assert result == module.AppendPartResult(module.AppendPartStatus.APPENDED, 5)
    assert await redis_client.pttl(key) > 2_000
    session = await store.get(upload_id)
    assert session.offset == 5
    assert session.parts == (module.TusPart(1, "etag-1"),)
    assert session.updated_at > before_session.updated_at
    stored = json.loads(await redis_client.get(key))
    assert stored["updated_at"].isascii() and stored["updated_at"].isdecimal()
    assert "e" not in stored["updated_at"].lower()


async def test_real_redis_failed_append_changes_neither_record_nor_ttl(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
    key = module.session_key(upload_id)
    await redis_client.pexpire(key, 30_000)
    before = await redis_client.get(key)
    before_ttl = await redis_client.pttl(key)

    result = await store.append_part(
        upload_id,
        expected_offset=1,
        byte_count=5,
        part_number=1,
        etag="etag-1",
        ttl_seconds=60,
    )

    after_ttl = await redis_client.pttl(key)
    assert result == module.AppendPartResult(module.AppendPartStatus.OFFSET_MISMATCH, 0)
    assert await redis_client.get(key) == before
    assert 0 <= before_ttl - after_ttl < 1_000


async def test_real_redis_rejects_overflow_and_out_of_order_parts(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)

    overflow = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=6,
        part_number=1,
        etag="etag-1",
        ttl_seconds=60,
    )
    out_of_order = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=5,
        part_number=2,
        etag="etag-2",
        ttl_seconds=60,
    )
    empty = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=0,
        part_number=1,
        etag="etag-empty",
        ttl_seconds=60,
    )

    assert overflow.status is module.AppendPartStatus.LENGTH_EXCEEDED
    assert out_of_order.status is module.AppendPartStatus.PART_OUT_OF_ORDER
    assert empty.status is module.AppendPartStatus.INVALID_BYTE_COUNT
    session = await store.get(upload_id)
    assert session.offset == 0
    assert session.parts == ()


async def test_real_redis_preserves_exact_offsets_at_lua_cjson_boundary(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    maximum = module.MAX_UPLOAD_BYTES
    session = replace(
        _session(module, upload_id, user_id, kb_id, total=maximum),
        offset=maximum - 1,
    )
    await store.create(session, ttl_seconds=60)

    result = await store.append_part(upload_id, maximum - 1, 1, 1, "etag-final", 60)

    assert result == module.AppendPartResult(module.AppendPartStatus.APPENDED, maximum)
    stored = await store.get(upload_id)
    assert stored.offset == stored.total_length == maximum


async def test_real_redis_completion_marker_is_idempotent_and_immutable(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)
    await store.append_part(upload_id, 0, 5, 1, "etag-1", 60)
    document_id = uuid4()
    job_id = uuid4()

    before_complete = await store.get(upload_id)
    first = await store.mark_complete(upload_id, 5, document_id, job_id, ttl_seconds=3)
    after_first = await store.get(upload_id)
    duplicate = await store.mark_complete(upload_id, 5, document_id, job_id, ttl_seconds=3)
    after_duplicate = await store.get(upload_id)
    raw_before_conflict = await redis_client.get(module.session_key(upload_id))
    conflict = await store.mark_complete(upload_id, 5, uuid4(), uuid4(), ttl_seconds=3)
    assert await redis_client.get(module.session_key(upload_id)) == raw_before_conflict
    append_after = await store.append_part(upload_id, 5, 0, 2, "etag-2", 3)

    assert first.status is module.CompleteStatus.COMPLETED
    assert duplicate.status is module.CompleteStatus.ALREADY_COMPLETED
    assert conflict.status is module.CompleteStatus.IDENTIFIER_CONFLICT
    assert duplicate.document_id == conflict.document_id == document_id
    assert duplicate.job_id == conflict.job_id == job_id
    assert before_complete.updated_at < after_first.updated_at < after_duplicate.updated_at
    assert append_after.status is module.AppendPartStatus.ALREADY_COMPLETED
    assert append_after.offset == 5
    session = await store.get(upload_id)
    assert session.state is module.TusSessionState.COMPLETED
    assert session.document_id == document_id
    assert session.job_id == job_id
    assert session.parts == (module.TusPart(1, "etag-1"),)
    assert await redis_client.pttl(module.session_key(upload_id)) > 2_000


async def test_real_redis_zero_length_completion_preserves_empty_parts_array(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=0), ttl_seconds=60)
    document_id = uuid4()
    job_id = uuid4()

    first = await store.mark_complete(upload_id, 0, document_id, job_id, ttl_seconds=60)
    duplicate = await store.mark_complete(upload_id, 0, document_id, job_id, ttl_seconds=60)

    assert first.status is module.CompleteStatus.COMPLETED
    assert duplicate.status is module.CompleteStatus.ALREADY_COMPLETED
    session = await store.get(upload_id)
    assert session.parts == ()
    assert session.offset == session.total_length == 0
    raw = json.loads(await redis_client.get(module.session_key(upload_id)))
    assert raw["parts"] == []


async def test_real_redis_lock_is_token_owned_and_renewable(namespace, redis_client):
    module, store, upload_id, _, _ = namespace
    acquired = await store.acquire_lock(upload_id, ttl_seconds=3)
    contended = await store.acquire_lock(upload_id, ttl_seconds=3)
    assert acquired.status is module.LockAcquireStatus.ACQUIRED
    assert contended.status is module.LockAcquireStatus.CONTENDED

    key = module.lock_key(upload_id)
    await redis_client.pexpire(key, 100)
    assert await store.renew_lock(upload_id, "wrong-owner", 3) is module.LockMutationStatus.NOT_OWNER
    assert await redis_client.pttl(key) <= 100
    assert await store.renew_lock(upload_id, acquired.token, 3) is module.LockMutationStatus.RENEWED
    assert await redis_client.pttl(key) > 2_000
    assert await store.release_lock(upload_id, "wrong-owner") is module.LockMutationStatus.NOT_OWNER
    assert await redis_client.exists(key) == 1
    assert await store.release_lock(upload_id, acquired.token) is module.LockMutationStatus.RELEASED
    assert await redis_client.exists(key) == 0


async def test_real_redis_reservation_release_is_once_and_marker_has_no_credentials(namespace, redis_client):
    module, store, upload_id, user_id, _ = namespace
    assert (
        await store.create_reservation(user_id, upload_id, bytes_reserved=10, ttl_seconds=60)
        is module.ReservationCreateStatus.CREATED
    )
    key = module.reservation_key(user_id, upload_id)
    raw = await redis_client.get(key)
    lowered = raw.lower()
    assert b"token" not in lowered and b"secret" not in lowered and b"credential" not in lowered

    assert await store.release_reservation_once(user_id, upload_id) is module.ReservationReleaseStatus.RELEASED
    assert await store.release_reservation_once(user_id, upload_id) is module.ReservationReleaseStatus.ALREADY_RELEASED


async def test_real_redis_expired_session_returns_none_with_bounded_poll(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
    await redis_client.pexpire(module.session_key(upload_id), 20)

    for _ in range(20):
        if await store.get(upload_id) is None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Redis did not expire the TUS session within the bounded poll")


async def test_real_redis_malformed_records_fail_closed_without_mutation(namespace, redis_client):
    module, store, upload_id, _, _ = namespace
    key = module.session_key(upload_id)
    malformed = b'{"offset":0,"total_length":10,"parts":[],"state":"uploading","unknown":true}'
    await redis_client.set(key, malformed, ex=60)

    with pytest.raises(module.InvalidTusSessionError):
        await store.get(upload_id)
    result = await store.append_part(upload_id, 0, 5, 1, "etag-1", 60)
    assert result.status is module.AppendPartStatus.MALFORMED
    assert await redis_client.get(key) == malformed
