import asyncio
import importlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from redis.cluster import key_slot


def _module():
    return importlib.import_module("infra.tus_sessions")


def _valid_session(module, **overrides):
    now = datetime.now(UTC).replace(microsecond=123456)
    values = {
        "upload_id": uuid4(),
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "filename": "quarterly-report.pdf",
        "path": "/reports/2026/",
        "content_type": "application/pdf",
        "total_length": 12,
        "offset": 5,
        "s3_key": "uploads/tenant/report.pdf",
        "multipart_upload_id": "multipart-opaque-id",
        "parts": (module.TusPart(part_number=1, etag="etag-1"),),
        "created_at": now,
        "updated_at": now + timedelta(seconds=1),
        "state": module.TusSessionState.UPLOADING,
        "document_id": None,
        "job_id": None,
        "reservation_bytes": 12,
        "object_completed": False,
    }
    values.update(overrides)
    return module.TusSession(**values)


class FakeRedis:
    def __init__(self, *, set_results=None, get_result=None, eval_results=None):
        self.set_results = list(set_results or [])
        self.get_result = get_result
        self.eval_results = list(eval_results or [])
        self.set_calls = []
        self.get_calls = []
        self.eval_calls = []

    async def set(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))
        return self.set_results.pop(0) if self.set_results else True

    async def get(self, key):
        self.get_calls.append(key)
        return self.get_result

    async def eval(self, script, numkeys, *keys_and_args):
        self.eval_calls.append((script, numkeys, keys_and_args))
        return self.eval_results.pop(0)


def test_session_round_trip_is_explicit_and_contains_no_auth_material():
    module = _module()
    session = _valid_session(module)

    encoded = session.to_json()
    payload = json.loads(encoded)

    assert module.TusSession.from_json(encoded) == session
    assert payload["created_at"].isascii() and payload["created_at"].isdecimal()
    assert payload["updated_at"].isascii() and payload["updated_at"].isdecimal()
    assert set(payload) == {
        "upload_id",
        "user_id",
        "knowledge_base_id",
        "filename",
        "path",
        "content_type",
        "total_length",
        "offset",
        "s3_key",
        "multipart_upload_id",
        "parts",
        "created_at",
        "updated_at",
        "state",
        "document_id",
        "job_id",
        "reservation_bytes",
        "object_completed",
    }
    lowered = encoded.lower()
    assert "token" not in lowered
    assert "credential" not in lowered
    assert "authorization" not in lowered
    assert "secret" not in lowered


def test_wire_json_is_canonical_and_filename_limits_use_utf8_bytes():
    module = _module()
    session = _valid_session(module, filename="汉" * 50 + ".pdf")
    assert module.TusSession.from_json(session.to_json()) == session

    noncanonical = json.dumps(json.loads(session.to_json()), ensure_ascii=False)
    with pytest.raises(module.InvalidTusSessionError):
        module.TusSession.from_json(noncanonical)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        _valid_session(module, filename="汉" * 86)


@pytest.mark.parametrize(
    "filename",
    [
        'report "final".pdf',
        'report ""draft"".pdf',
        '中文"最终".pdf',
        'section\u2028"final".pdf',
    ],
)
def test_quoted_filenames_use_canonical_json_escaping(filename):
    module = _module()
    session = _valid_session(module, filename=filename)

    encoded = session.to_json()

    assert '\\"' in encoded
    assert json.loads(encoded)["filename"] == filename
    assert module.TusSession.from_json(encoded) == session


def test_filename_backslash_and_quotes_in_other_opaque_fields_remain_rejected():
    module = _module()
    with pytest.raises(ValueError):
        _valid_session(module, filename="folder\\report.pdf")
    with pytest.raises(ValueError):
        _valid_session(module, content_type='application/"pdf')
    with pytest.raises(ValueError):
        module.TusPart(1, 'etag"quoted')


def test_session_reservation_bytes_must_equal_declared_length():
    module = _module()
    with pytest.raises(ValueError, match="reservation_bytes"):
        _valid_session(module, reservation_bytes=11)


def test_cleanup_required_session_preserves_offset_and_multipart_identity():
    module = _module()
    session = _valid_session(
        module,
        state=module.TusSessionState.CLEANUP_REQUIRED,
    )

    decoded = module.TusSession.from_json(session.to_json())

    assert decoded.offset == 5
    assert decoded.parts == (module.TusPart(1, "etag-1"),)
    assert decoded.multipart_upload_id == "multipart-opaque-id"


def test_object_completed_phase_requires_full_offset():
    module = _module()

    completed_object = _valid_session(module, offset=12, object_completed=True)
    assert module.TusSession.from_json(completed_object.to_json()) == completed_object

    with pytest.raises(ValueError, match="object_completed"):
        _valid_session(module, object_completed=True)


def test_hash_tagged_keys_are_exact_and_mutation_keys_share_cluster_slot():
    module = _module()
    upload_id = uuid4()
    user_id = uuid4()
    session = f"tus:session:{{{upload_id}}}"
    lock = f"tus:lock:{{{upload_id}}}"
    reservation = f"tus:reservation:{{{user_id}}}:{{{upload_id}}}"

    assert module.session_key(upload_id) == session
    assert module.lock_key(upload_id) == lock
    assert module.reservation_key(user_id, upload_id) == reservation
    assert key_slot(session.encode()) == key_slot(lock.encode())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update({"unknown": "field"}),
        lambda p: p.update({"upload_id": "not-a-uuid"}),
        lambda p: p.update({"created_at": "2026-01-01T00:00:00"}),
        lambda p: p.update({"updated_at": True}),
        lambda p: p.update({"updated_at": -1}),
        lambda p: p.update({"updated_at": 2**53}),
        lambda p: p.update({"state": "surprising"}),
        lambda p: p.update({"offset": True}),
        lambda p: p.update({"offset": -1}),
        lambda p: p.update({"offset": 13}),
        lambda p: p.update({"total_length": 2**53}),
        lambda p: p.update({"reservation_bytes": 2**53}),
        lambda p: p.update({"parts": [{"part_number": 1, "etag": "ok", "extra": 1}]}),
        lambda p: p.update({"parts": [{"part_number": True, "etag": "ok"}]}),
        lambda p: p.update({"parts": [{"part_number": 2, "etag": "ok"}]}),
        lambda p: p.update({"parts": [{"part_number": 1, "etag": "a"}, {"part_number": 1, "etag": "b"}]}),
        lambda p: p.update({"parts": [{"part_number": 1, "etag": "bad\nvalue"}]}),
        lambda p: p.update({"filename": "../escape.pdf"}),
        lambda p: p.update({"content_type": "application/pdf\r\nX-Test: bad"}),
        lambda p: p.update({"document_id": str(uuid4())}),
    ],
)
def test_session_decoder_rejects_malformed_or_inconsistent_records(mutate):
    module = _module()
    payload = json.loads(_valid_session(module).to_json())
    mutate(payload)

    with pytest.raises(module.InvalidTusSessionError):
        module.TusSession.from_json(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def test_completed_state_requires_full_offset_and_stable_document_and_job_ids():
    module = _module()
    document_id = uuid4()
    job_id = uuid4()
    session = _valid_session(
        module,
        offset=12,
        state=module.TusSessionState.COMPLETED,
        document_id=document_id,
        job_id=job_id,
        object_completed=True,
    )
    assert module.TusSession.from_json(session.to_json()) == session

    for changes in (
        {"offset": 11},
        {"document_id": None},
        {"job_id": None},
    ):
        payload = json.loads(session.to_json())
        payload.update(
            {key: str(value) if key.endswith("_id") and value is not None else value for key, value in changes.items()}
        )
        with pytest.raises(module.InvalidTusSessionError):
            module.TusSession.from_json(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


@pytest.mark.parametrize("field", ["total_length", "offset", "reservation_bytes"])
def test_session_constructor_rejects_bool_and_lua_unsafe_integers(field):
    module = _module()
    with pytest.raises(ValueError):
        _valid_session(module, **{field: True})
    with pytest.raises(ValueError):
        _valid_session(module, **{field: 2**53})


def test_store_uses_exact_session_key_and_single_key_create_lua():
    module = _module()
    redis = FakeRedis(eval_results=[1, 0])
    store = module.TusSessionStore(redis)
    session = _valid_session(module)

    first = asyncio.run(store.create(session, ttl_seconds=120))
    second = asyncio.run(store.create(session, ttl_seconds=120))

    assert first is module.SessionCreateStatus.CREATED
    assert second is module.SessionCreateStatus.ALREADY_EXISTS
    assert redis.set_calls == []
    assert all(call[1] == 1 for call in redis.eval_calls)
    _, _, args = redis.eval_calls[0]
    assert args[0] == f"tus:session:{{{session.upload_id}}}"
    assert json.loads(args[1])["upload_id"] == str(session.upload_id)
    assert args[2] == "120"


def test_get_returns_none_for_missing_record_and_fails_closed_for_malformed_record():
    module = _module()
    upload_id = uuid4()
    missing_store = module.TusSessionStore(FakeRedis(get_result=None))
    assert asyncio.run(missing_store.get(upload_id)) is None

    malformed_store = module.TusSessionStore(FakeRedis(get_result=b'{"unexpected":true}'))
    with pytest.raises(module.InvalidTusSessionError):
        asyncio.run(malformed_store.get(upload_id))


def test_append_part_uses_one_key_lua_and_maps_only_bounded_status_codes():
    module = _module()
    upload_id = uuid4()
    redis = FakeRedis(eval_results=[[9, 5], [0, 9], [1, 9], [99, 0]])
    store = module.TusSessionStore(redis)

    fenced = asyncio.run(store.append_part(upload_id, 5, 4, 2, "etag-2", 60, lock_token="stale-token"))

    appended = asyncio.run(
        store.append_part(
            upload_id,
            expected_offset=5,
            byte_count=4,
            part_number=2,
            etag="etag-2",
            ttl_seconds=60,
            lock_token="owner-token",
        )
    )
    mismatch = asyncio.run(
        store.append_part(
            upload_id,
            expected_offset=5,
            byte_count=4,
            part_number=2,
            etag="etag-2",
            ttl_seconds=60,
            lock_token="owner-token",
        )
    )

    assert fenced == module.AppendPartResult(module.AppendPartStatus.LOCK_LOST, 5)
    assert appended == module.AppendPartResult(module.AppendPartStatus.APPENDED, 9)
    assert mismatch == module.AppendPartResult(module.AppendPartStatus.OFFSET_MISMATCH, 9)
    _, numkeys, args = redis.eval_calls[1]
    assert numkeys == 2
    assert args[:2] == (f"tus:session:{{{upload_id}}}", f"tus:lock:{{{upload_id}}}")
    assert args[2:] == ("5", "4", "2", "etag-2", "60", "owner-token")
    with pytest.raises(module.TusSessionProtocolError):
        asyncio.run(
            store.append_part(
                upload_id,
                expected_offset=0,
                byte_count=1,
                part_number=1,
                etag="etag-1",
                ttl_seconds=60,
                lock_token="owner-token",
            )
        )


def test_append_validates_lua_safe_inputs_before_calling_redis():
    module = _module()
    redis = FakeRedis(eval_results=[])
    store = module.TusSessionStore(redis)

    for kwargs in (
        {"expected_offset": True, "byte_count": 1, "part_number": 1},
        {"expected_offset": 0, "byte_count": -1, "part_number": 1},
        {"expected_offset": 2**53, "byte_count": 1, "part_number": 1},
        {"expected_offset": 0, "byte_count": 2**53, "part_number": 1},
        {"expected_offset": 0, "byte_count": 1, "part_number": 10_001},
    ):
        with pytest.raises(ValueError):
            asyncio.run(
                store.append_part(
                    uuid4(),
                    etag="etag",
                    ttl_seconds=60,
                    lock_token="owner-token",
                    **kwargs,
                )
            )
    assert redis.eval_calls == []


def test_mark_complete_maps_first_duplicate_and_conflicting_markers():
    module = _module()
    upload_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    redis = FakeRedis(
        eval_results=[
            [0, 12, str(document_id), str(job_id)],
            [1, 12, str(document_id), str(job_id)],
            [2, 12, str(document_id), str(job_id)],
        ]
    )
    store = module.TusSessionStore(redis)

    results = [
        asyncio.run(
            store.mark_complete(
                upload_id,
                expected_offset=12,
                document_id=document_id,
                job_id=job_id,
                ttl_seconds=120,
                lock_token="owner-token",
            )
        )
        for _ in range(3)
    ]

    assert [result.status for result in results] == [
        module.CompleteStatus.COMPLETED,
        module.CompleteStatus.ALREADY_COMPLETED,
        module.CompleteStatus.IDENTIFIER_CONFLICT,
    ]
    assert all(result.offset == 12 for result in results)
    assert all(result.document_id == document_id for result in results)
    assert all(result.job_id == job_id for result in results)
    assert all(call[1] == 2 for call in redis.eval_calls)
    assert all(
        call[2][:2] == (f"tus:session:{{{upload_id}}}", f"tus:lock:{{{upload_id}}}") for call in redis.eval_calls
    )


def test_impossible_offsets_and_noncanonical_uuid_responses_fail_protocol_closed():
    module = _module()
    upload_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    redis = FakeRedis(
        eval_results=[
            [0, module.MAX_UPLOAD_BYTES + 1],
            [0, 0, str(document_id).upper(), str(job_id)],
        ]
    )
    store = module.TusSessionStore(redis)

    with pytest.raises(module.TusSessionProtocolError):
        asyncio.run(store.append_part(upload_id, 0, 1, 1, "etag", 60, lock_token="owner-token"))
    with pytest.raises(module.TusSessionProtocolError):
        asyncio.run(store.mark_complete(upload_id, 0, document_id, job_id, 60, lock_token="owner-token"))


def test_lock_tokens_are_opaque_and_owner_checked_with_single_key_scripts():
    module = _module()
    upload_id = uuid4()
    redis = FakeRedis(set_results=[True, None], eval_results=[1, 0, 1])
    store = module.TusSessionStore(redis)

    acquired = asyncio.run(store.acquire_lock(upload_id, ttl_seconds=20))
    contended = asyncio.run(store.acquire_lock(upload_id, ttl_seconds=20))
    renewed = asyncio.run(store.renew_lock(upload_id, acquired.token, ttl_seconds=30))
    rejected = asyncio.run(store.release_lock(upload_id, "not-the-owner"))
    released = asyncio.run(store.release_lock(upload_id, acquired.token))

    assert acquired.status is module.LockAcquireStatus.ACQUIRED
    assert isinstance(acquired.token, str) and len(acquired.token) >= 32
    assert contended == module.LockAcquireResult(module.LockAcquireStatus.CONTENDED, None)
    assert renewed is module.LockMutationStatus.RENEWED
    assert rejected is module.LockMutationStatus.NOT_OWNER
    assert released is module.LockMutationStatus.RELEASED
    assert redis.set_calls[0][0] == f"tus:lock:{{{upload_id}}}"
    assert redis.set_calls[0][2] == {"nx": True, "ex": 20}
    assert all(call[1] == 1 for call in redis.eval_calls)
    assert all(call[2][0] == f"tus:lock:{{{upload_id}}}" for call in redis.eval_calls)


def test_stale_owner_cannot_mark_cleanup_or_delete_a_new_owners_session():
    module = _module()
    upload_id = uuid4()
    redis = FakeRedis(eval_results=[5, 0])
    store = module.TusSessionStore(redis)

    marked = asyncio.run(
        store.mark_cleanup_required(
            upload_id,
            expected_offset=5,
            ttl_seconds=60,
            lock_token="stale-owner",
        )
    )
    deleted = asyncio.run(store.delete_locked(upload_id, lock_token="stale-owner"))

    assert marked is False
    assert deleted is False
    assert redis.eval_calls[0][1] == 2
    assert redis.eval_calls[1][1] == 2


def test_reservation_marker_persists_quota_owner_and_release_is_token_fenced():
    module = _module()
    user_id = uuid4()
    upload_id = uuid4()
    owner_token = "B" * 32
    payload = json.dumps(
        {"bytes": 12, "owner": owner_token, "state": "reserved"},
        sort_keys=True,
        separators=(",", ":"),
    )
    redis = FakeRedis(set_results=[True, None], get_result=payload, eval_results=[4, 1, 2])
    store = module.TusSessionStore(redis)

    created = asyncio.run(
        store.create_reservation(user_id, upload_id, 12, owner_token=owner_token, ttl_seconds=90)
    )
    duplicate = asyncio.run(
        store.create_reservation(user_id, upload_id, 12, owner_token=owner_token, ttl_seconds=90)
    )
    marker = asyncio.run(store.get_reservation(user_id, upload_id))
    wrong_owner = asyncio.run(store.release_reservation_once(user_id, upload_id, "C" * 32))
    released = asyncio.run(store.release_reservation_once(user_id, upload_id, owner_token))
    repeated = asyncio.run(store.release_reservation_once(user_id, upload_id, owner_token))

    assert created is module.ReservationCreateStatus.CREATED
    assert duplicate is module.ReservationCreateStatus.ALREADY_EXISTS
    assert marker == module.TusQuotaReservation(12, owner_token, module.TusReservationState.RESERVED)
    assert wrong_owner is module.ReservationReleaseStatus.NOT_OWNER
    assert released is module.ReservationReleaseStatus.RELEASED
    assert repeated is module.ReservationReleaseStatus.ALREADY_RELEASED
    key = f"tus:reservation:{{{user_id}}}:{{{upload_id}}}"
    assert redis.set_calls[0][0] == key
    reservation_payload = json.loads(redis.set_calls[0][1])
    assert reservation_payload == {"bytes": 12, "owner": owner_token, "state": "reserved"}
    assert all(call[1] == 1 and call[2][0] == key for call in redis.eval_calls)
    assert [call[2][1] for call in redis.eval_calls] == ["C" * 32, owner_token, owner_token]


def test_reservation_marker_ttl_renewal_requires_the_quota_owner():
    module = _module()
    user_id = uuid4()
    upload_id = uuid4()
    owner_token = "D" * 32
    redis = FakeRedis(eval_results=[1, 0])
    store = module.TusSessionStore(redis)

    renewed = asyncio.run(store.renew_reservation(user_id, upload_id, owner_token, ttl_seconds=120))
    rejected = asyncio.run(store.renew_reservation(user_id, upload_id, "E" * 32, ttl_seconds=120))

    assert renewed is module.LockMutationStatus.RENEWED
    assert rejected is module.LockMutationStatus.NOT_OWNER
    key = f"tus:reservation:{{{user_id}}}:{{{upload_id}}}"
    assert [call[1:] for call in redis.eval_calls] == [
        (1, (key, owner_token, "120")),
        (1, (key, "E" * 32, "120")),
    ]
