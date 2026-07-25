import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
        path="/",
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
        object_completed=False,
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


async def _lock_token(module, store, upload_id):
    acquired = await store.acquire_lock(upload_id, ttl_seconds=60)
    assert acquired.status is module.LockAcquireStatus.ACQUIRED
    assert acquired.token is not None
    return acquired.token


async def test_real_redis_reservation_scan_yields_only_canonical_marker_keys(namespace, redis_client):
    module, store, upload_id, user_id, _ = namespace
    owner = "Q" * 32
    assert (
        await store.create_reservation(user_id, upload_id, 10, owner_token=owner, ttl_seconds=60)
        is module.ReservationCreateStatus.CREATED
    )
    decoy = f"tus:reservation:{{{str(user_id).upper()}}}:{{{upload_id}}}"
    await redis_client.set(decoy, module.TusQuotaReservation(10, owner, module.TusReservationState.RESERVED).to_json())
    try:
        records = [record async for record in store.iter_reservations()]
    finally:
        await redis_client.delete(decoy)

    assert records == [
        (
            user_id,
            upload_id,
            module.TusQuotaReservation(10, owner, module.TusReservationState.RESERVED),
        )
    ]


async def test_real_redis_reservation_scan_fails_closed_on_malformed_canonical_value(namespace, redis_client):
    module, store, upload_id, user_id, _ = namespace
    key = module.reservation_key(user_id, upload_id)
    malformed = b'{"bytes":10,"owner":"wrong","state":"reserved"}'
    await redis_client.set(key, malformed, ex=60)

    with pytest.raises(module.InvalidTusSessionError):
        _ = [record async for record in store.iter_reservations()]

    assert await redis_client.get(key) == malformed


async def test_real_redis_concurrent_cas_allows_exactly_one_append(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    assert (
        await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
        is module.SessionCreateStatus.CREATED
    )
    token = await _lock_token(module, store, upload_id)

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
            lock_token=token,
        )

    first, second = await asyncio.gather(append("etag-a"), append("etag-b"))
    assert sorted([first.status.value, second.status.value]) == ["appended", "offset_mismatch"]
    session = await store.get(upload_id)
    assert session is not None
    assert session.offset == 5
    assert len(session.parts) == 1
    assert session.parts[0].part_number == 1
    assert session.parts[0].etag in {"etag-a", "etag-b"}


async def test_real_redis_create_under_lock_is_atomic_with_lock_ownership(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    session = _session(module, upload_id, user_id, kb_id)
    stale = await _lock_token(module, store, upload_id)
    await redis_client.delete(module.lock_key(upload_id))
    owner = await _lock_token(module, store, upload_id)

    lost = await store.create_under_lock(session, ttl_seconds=60, lock_token=stale)
    assert lost is module.SessionCreateStatus.LOCK_LOST
    assert await store.get(upload_id) is None

    created = await store.create_under_lock(session, ttl_seconds=60, lock_token=owner)
    assert created is module.SessionCreateStatus.CREATED
    assert await store.get(upload_id) is not None


async def test_real_redis_stale_lock_owner_cannot_append_or_complete(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)
    first_token = await _lock_token(module, store, upload_id)
    await redis_client.delete(module.lock_key(upload_id))  # deterministic equivalent of lock expiry
    second_token = await _lock_token(module, store, upload_id)
    key = module.session_key(upload_id)
    before = await redis_client.get(key)
    before_ttl = await redis_client.pttl(key)

    stale_append = await store.append_part(upload_id, 0, 5, 1, "etag-stale", 60, lock_token=first_token)
    stale_complete = await store.mark_complete(upload_id, 0, uuid4(), uuid4(), 60, lock_token=first_token)

    after_ttl = await redis_client.pttl(key)
    assert stale_append.status is module.AppendPartStatus.LOCK_LOST
    assert stale_complete.status is module.CompleteStatus.LOCK_LOST
    assert await redis_client.get(key) == before
    assert 0 <= before_ttl - after_ttl < 1_000

    appended = await store.append_part(upload_id, 0, 5, 1, "etag-owner", 60, lock_token=second_token)
    document_id = uuid4()
    job_id = uuid4()
    completed = await store.mark_complete(upload_id, 5, document_id, job_id, 60, lock_token=second_token)
    assert appended.status is module.AppendPartStatus.APPENDED
    assert completed.status is module.CompleteStatus.COMPLETED

    await redis_client.delete(module.lock_key(upload_id))
    await _lock_token(module, store, upload_id)
    completed_record = await redis_client.get(key)
    completed_ttl = await redis_client.pttl(key)
    stale_duplicate = await store.mark_complete(upload_id, 5, document_id, job_id, 60, lock_token=second_token)
    assert stale_duplicate.status is module.CompleteStatus.LOCK_LOST
    assert await redis_client.get(key) == completed_record
    assert 0 <= completed_ttl - await redis_client.pttl(key) < 1_000


async def test_real_redis_stale_owner_cannot_mark_cleanup_or_delete_new_progress(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)
    stale = await _lock_token(module, store, upload_id)
    await redis_client.delete(module.lock_key(upload_id))
    owner = await _lock_token(module, store, upload_id)
    await store.append_part(upload_id, 0, 5, 1, "etag-owner", 60, lock_token=owner)
    before = await redis_client.get(module.session_key(upload_id))

    assert not await store.mark_cleanup_required(upload_id, 0, 60, lock_token=stale)
    assert not await store.delete_locked(upload_id, lock_token=stale)
    assert await redis_client.get(module.session_key(upload_id)) == before


async def test_real_redis_chinese_filename_round_trips_and_mutates(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    filename = "汉" * 50 + ".pdf"
    await store.create(
        replace(_session(module, upload_id, user_id, kb_id, total=1), filename=filename),
        ttl_seconds=60,
    )
    token = await _lock_token(module, store, upload_id)

    result = await store.append_part(upload_id, 0, 1, 1, "etag-cn", 60, lock_token=token)

    assert result.status is module.AppendPartStatus.APPENDED
    assert (await store.get(upload_id)).filename == filename


@pytest.mark.parametrize(
    "filename",
    [
        'report "final".pdf',
        'report ""draft"".pdf',
        '中文"最终".pdf',
        'section\u2028"final".pdf',
    ],
)
async def test_real_redis_quoted_filename_survives_full_completion(namespace, filename):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(
        replace(_session(module, upload_id, user_id, kb_id, total=1), filename=filename),
        ttl_seconds=60,
    )
    token = await _lock_token(module, store, upload_id)
    append = await store.append_part(upload_id, 0, 1, 1, "etag-quoted", 60, lock_token=token)
    complete = await store.mark_complete(upload_id, 1, uuid4(), uuid4(), 60, lock_token=token)

    stored = await store.get(upload_id)
    assert append.status is module.AppendPartStatus.APPENDED
    assert complete.status is module.CompleteStatus.COMPLETED
    assert stored.filename == filename
    assert stored.state is module.TusSessionState.COMPLETED


async def test_real_redis_append_is_atomic_and_refreshes_session_ttl(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
    token = await _lock_token(module, store, upload_id)
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
        lock_token=token,
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
    token = await _lock_token(module, store, upload_id)
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
        lock_token=token,
    )

    after_ttl = await redis_client.pttl(key)
    assert result == module.AppendPartResult(module.AppendPartStatus.OFFSET_MISMATCH, 0)
    assert await redis_client.get(key) == before
    assert 0 <= before_ttl - after_ttl < 1_000


async def test_real_redis_rejects_overflow_and_out_of_order_parts(namespace):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)
    token = await _lock_token(module, store, upload_id)

    overflow = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=6,
        part_number=1,
        etag="etag-1",
        ttl_seconds=60,
        lock_token=token,
    )
    out_of_order = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=5,
        part_number=2,
        etag="etag-2",
        ttl_seconds=60,
        lock_token=token,
    )
    empty = await store.append_part(
        upload_id,
        expected_offset=0,
        byte_count=0,
        part_number=1,
        etag="etag-empty",
        ttl_seconds=60,
        lock_token=token,
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
    token = await _lock_token(module, store, upload_id)

    result = await store.append_part(upload_id, maximum - 1, 1, 1, "etag-final", 60, lock_token=token)

    assert result == module.AppendPartResult(module.AppendPartStatus.APPENDED, maximum)
    stored = await store.get(upload_id)
    assert stored.offset == stored.total_length == maximum


async def test_real_redis_completion_marker_is_idempotent_and_immutable(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id, total=5), ttl_seconds=60)
    token = await _lock_token(module, store, upload_id)
    await store.append_part(upload_id, 0, 5, 1, "etag-1", 60, lock_token=token)
    document_id = uuid4()
    job_id = uuid4()

    before_complete = await store.get(upload_id)
    first = await store.mark_complete(upload_id, 5, document_id, job_id, ttl_seconds=3, lock_token=token)
    after_first = await store.get(upload_id)
    duplicate = await store.mark_complete(upload_id, 5, document_id, job_id, ttl_seconds=3, lock_token=token)
    after_duplicate = await store.get(upload_id)
    raw_before_conflict = await redis_client.get(module.session_key(upload_id))
    conflict = await store.mark_complete(upload_id, 5, uuid4(), uuid4(), ttl_seconds=3, lock_token=token)
    assert await redis_client.get(module.session_key(upload_id)) == raw_before_conflict
    append_after = await store.append_part(upload_id, 5, 0, 2, "etag-2", 3, lock_token=token)

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
    token = await _lock_token(module, store, upload_id)
    document_id = uuid4()
    job_id = uuid4()

    first = await store.mark_complete(upload_id, 0, document_id, job_id, ttl_seconds=60, lock_token=token)
    duplicate = await store.mark_complete(upload_id, 0, document_id, job_id, ttl_seconds=60, lock_token=token)

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
    owner = "R" * 32
    assert (
        await store.create_reservation(
            user_id,
            upload_id,
            bytes_reserved=10,
            owner_token=owner,
            ttl_seconds=60,
        )
        is module.ReservationCreateStatus.CREATED
    )
    key = module.reservation_key(user_id, upload_id)
    raw = await redis_client.get(key)
    lowered = raw.lower()
    assert b"token" not in lowered and b"secret" not in lowered and b"credential" not in lowered

    assert (
        await store.release_reservation_once(user_id, upload_id, "S" * 32) is module.ReservationReleaseStatus.NOT_OWNER
    )
    assert await store.release_reservation_once(user_id, upload_id, owner) is module.ReservationReleaseStatus.RELEASED
    assert (
        await store.release_reservation_once(user_id, upload_id, owner)
        is module.ReservationReleaseStatus.ALREADY_RELEASED
    )


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
    token = await _lock_token(module, store, upload_id)
    key = module.session_key(upload_id)
    malformed = b'{"offset":0,"total_length":10,"parts":[],"state":"uploading","unknown":true}'
    await redis_client.set(key, malformed, ex=60)

    with pytest.raises(module.InvalidTusSessionError):
        await store.get(upload_id)
    result = await store.append_part(upload_id, 0, 5, 1, "etag-1", 60, lock_token=token)
    assert result.status is module.AppendPartStatus.MALFORMED
    assert await redis_client.get(key) == malformed


def _replace_wire(old, new):
    def mutate(raw):
        assert old in raw
        return raw.replace(old, new, 1)

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_replace_wire(b'"parts":[]', b'"parts":{}'), id="parts-object"),
        pytest.param(_replace_wire(b'"offset":0', b'"offset":0.0'), id="offset-float"),
        pytest.param(_replace_wire(b'"total_length":10', b'"total_length":1e1'), id="length-exponent"),
        pytest.param(_replace_wire(b'"reservation_bytes":10', b'"reservation_bytes":10.0'), id="reservation-float"),
        pytest.param(
            _replace_wire(
                b'"parts":[]',
                b'"parts":[{"etag":"etag","part_number":1e0}]',
            ),
            id="part-number-exponent",
        ),
        pytest.param(
            _replace_wire(
                b'"parts":[]',
                '"parts":[{"etag":"bad\u00a0etag","part_number":1}]'.encode(),
            ),
            id="nbsp-etag",
        ),
        pytest.param(_replace_wire(b'"filename":"safe.pdf"', b'"filename":"bad\xff.pdf"'), id="invalid-utf8"),
        pytest.param(lambda raw: raw[:-1] + b',"unknown":true}', id="unknown-field"),
        pytest.param(_replace_wire(b'"content_type":"application/pdf",', b""), id="missing-field"),
        pytest.param(_replace_wire(b'"state":"uploading"', b'"state":"UPLOADING"'), id="state"),
        pytest.param(
            lambda raw: raw.replace(
                f'"upload_id":"{json.loads(raw)["upload_id"]}"'.encode(),
                f'"upload_id":"{json.loads(raw)["upload_id"].upper()}"'.encode(),
                1,
            ),
            id="uppercase-uuid",
        ),
        pytest.param(
            lambda raw: raw.replace(
                json.loads(raw)["updated_at"].encode(),
                b"0" + json.loads(raw)["updated_at"].encode(),
                1,
            ),
            id="timestamp-leading-zero",
        ),
    ],
)
async def test_real_redis_malformed_variants_block_append_and_complete_without_ttl_refresh(
    namespace, redis_client, mutate
):
    module, store, upload_id, user_id, kb_id = namespace
    await store.create(_session(module, upload_id, user_id, kb_id), ttl_seconds=60)
    token = await _lock_token(module, store, upload_id)
    key = module.session_key(upload_id)
    malformed = mutate(await redis_client.get(key))
    await redis_client.set(key, malformed, px=30_000)
    before_ttl = await redis_client.pttl(key)

    append = await store.append_part(upload_id, 0, 5, 1, "etag", 60, lock_token=token)
    complete = await store.mark_complete(upload_id, 0, uuid4(), uuid4(), 60, lock_token=token)

    after_ttl = await redis_client.pttl(key)
    assert append.status is module.AppendPartStatus.MALFORMED
    assert complete.status is module.CompleteStatus.MALFORMED
    assert await redis_client.get(key) == malformed
    assert 0 <= before_ttl - after_ttl < 1_000
    with pytest.raises(module.InvalidTusSessionError):
        await store.get(upload_id)


async def test_real_redis_max_timestamp_cannot_overflow_on_append_or_complete(namespace, redis_client):
    module, store, upload_id, user_id, kb_id = namespace
    token = await _lock_token(module, store, upload_id)
    maximum_time = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=module.MAX_SAFE_INTEGER)
    key = module.session_key(upload_id)

    append_session = replace(
        _session(module, upload_id, user_id, kb_id, total=1),
        created_at=maximum_time,
        updated_at=maximum_time,
    )
    await redis_client.set(key, append_session.to_json(), px=30_000)
    append_before = await redis_client.get(key)
    append_ttl = await redis_client.pttl(key)
    append = await store.append_part(upload_id, 0, 1, 1, "etag", 60, lock_token=token)
    assert append.status is module.AppendPartStatus.MALFORMED
    assert await redis_client.get(key) == append_before
    assert 0 <= append_ttl - await redis_client.pttl(key) < 1_000

    complete_session = replace(
        _session(module, upload_id, user_id, kb_id, total=0),
        created_at=maximum_time,
        updated_at=maximum_time,
    )
    await redis_client.set(key, complete_session.to_json(), px=30_000)
    complete_before = await redis_client.get(key)
    complete_ttl = await redis_client.pttl(key)
    complete = await store.mark_complete(upload_id, 0, uuid4(), uuid4(), 60, lock_token=token)
    assert complete.status is module.CompleteStatus.MALFORMED
    assert await redis_client.get(key) == complete_before
    assert 0 <= complete_ttl - await redis_client.pttl(key) < 1_000


async def test_real_redis_released_reservation_without_ttl_is_malformed(namespace, redis_client):
    module, store, upload_id, user_id, _ = namespace
    key = module.reservation_key(user_id, upload_id)
    owner = "T" * 32
    await redis_client.set(key, f'{{"bytes":10,"owner":"{owner}","state":"released"}}')

    result = await store.release_reservation_once(user_id, upload_id, owner)

    assert result is module.ReservationReleaseStatus.MALFORMED
    assert await redis_client.pttl(key) == -1


async def test_real_redis_zero_byte_reservation_is_malformed_and_never_renewed(namespace, redis_client):
    module, store, upload_id, user_id, _ = namespace
    key = module.reservation_key(user_id, upload_id)
    owner = "T" * 32
    raw = f'{{"bytes":0,"owner":"{owner}","state":"reserved"}}'
    await redis_client.set(key, raw, ex=60)

    with pytest.raises(module.InvalidTusSessionError):
        await store.get_reservation(user_id, upload_id)
    assert await store.release_reservation_once(user_id, upload_id, owner) is module.ReservationReleaseStatus.MALFORMED
    assert (
        await store.renew_reservation(user_id, upload_id, owner, ttl_seconds=60) is module.LockMutationStatus.NOT_OWNER
    )
    assert await redis_client.get(key) == raw.encode()
