from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from infra.quota import QuotaReservation, QuotaUnavailable
from infra.tus_sessions import (
    AppendPartResult,
    AppendPartStatus,
    CompleteResult,
    CompleteStatus,
    LockAcquireResult,
    LockAcquireStatus,
    LockMutationStatus,
    ReservationCreateStatus,
    ReservationReleaseStatus,
    SessionCreateStatus,
    TusPart,
    TusQuotaReservation,
    TusReservationState,
    TusSession,
    TusSessionState,
)
from services.s3 import ObjectMetadata

pytestmark = pytest.mark.asyncio

MIB = 1024 * 1024
USER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
OTHER_USER_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
KB_ID = UUID("11111111-1111-1111-1111-111111111111")
OWNER = "Q" * 32


def _metadata(*, filename: str = "fault.pdf", path: str = "/faults/") -> str:
    values = {"filename": filename, "knowledge_base_id": str(KB_ID), "path": path}
    return ",".join(f"{key} {base64.b64encode(value.encode()).decode()}" for key, value in values.items())


def _session(*, total: int = 10 * MIB, offset: int = 0, parts=(), object_completed=False, state=None):
    now = datetime.now(UTC)
    return TusSession(
        upload_id=UUID("22222222-2222-2222-2222-222222222222"),
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        filename="fault.pdf",
        path="/faults/",
        content_type="application/pdf",
        total_length=total,
        offset=offset,
        s3_key=f"{USER_ID}/22222222-2222-2222-2222-222222222222/source.pdf",
        multipart_upload_id="multipart-id",
        parts=tuple(parts),
        created_at=now,
        updated_at=now,
        state=state or TusSessionState.UPLOADING,
        document_id=None,
        job_id=None,
        reservation_bytes=total,
        object_completed=object_completed,
    )


class Pool:
    async def fetchval(self, query, *args):
        if "knowledge_bases" in query:
            return str(USER_ID)
        return False

    async def fetchrow(self, query, *args):
        return None


class Quota:
    def __init__(
        self,
        events,
        *,
        reserve_error=None,
        release_error=None,
        finalize_error=None,
        release_result=True,
        finalize_result=True,
        marker_settlement_status="settled",
    ):
        self.events = events
        self.reserve_error = reserve_error
        self.release_error = release_error
        self.finalize_error = finalize_error
        self.release_result = release_result
        self.finalize_result = finalize_result
        self.marker_settlement_status_result = marker_settlement_status
        self.reservation = None

    async def reserve(self, user_id, upload_id, byte_count, ttl_seconds):
        self.events.append("quota.reserve")
        if self.reserve_error:
            raise self.reserve_error
        self.reservation = QuotaReservation(user_id, upload_id, byte_count, OWNER)
        return self.reservation

    async def renew(self, reservation, ttl_seconds):
        self.events.append("quota.renew")
        return True

    async def release(self, reservation):
        self.events.append("quota.release")
        if self.release_error:
            raise self.release_error
        return self.release_result

    async def finalize(self, reservation):
        self.events.append("quota.finalize")
        if self.finalize_error:
            raise self.finalize_error
        return self.finalize_result

    async def settle_tus_marker_if_absent(self, reservation):
        from infra.quota import QuotaMarkerSettlementStatus

        self.events.append("quota.marker_settle")
        return QuotaMarkerSettlementStatus(self.marker_settlement_status_result)


class S3:
    def __init__(
        self,
        events,
        *,
        create_error=None,
        abort_error=None,
        upload_gate=None,
        head=None,
        signature=b"%PDF-",
    ):
        self.events = events
        self.create_error = create_error
        self.abort_error = abort_error
        self.upload_gate = upload_gate
        self.head = head
        self.signature = signature

    async def create_multipart(self, key, content_type):
        self.events.append("s3.create")
        if self.create_error:
            raise self.create_error
        return "multipart-id"

    async def abort_multipart(self, key, upload_id):
        self.events.append("s3.abort")
        if self.abort_error:
            raise self.abort_error

    async def upload_part(self, key, upload_id, part_number, body):
        self.events.append("s3.upload.start")
        if self.upload_gate:
            self.upload_gate[0].set()
            await self.upload_gate[1].wait()
        self.events.append("s3.upload.etag")
        return f"etag-{part_number}"

    async def complete_multipart(self, key, upload_id, parts):
        self.events.append("s3.complete")

    async def head_object(self, key):
        self.events.append("s3.head")
        return self.head

    async def read_range(self, key, start, end):
        self.events.append("s3.range")
        return self.signature

    async def delete_object(self, key):
        self.events.append("s3.delete")


class Store:
    def __init__(
        self,
        events,
        *,
        session=None,
        create_error=None,
        append_status=AppendPartStatus.APPENDED,
        marker_release_status=ReservationReleaseStatus.RELEASED,
    ):
        self.events = events
        self.session = session
        self.create_error = create_error
        self.append_status = append_status
        self.marker_release_status = marker_release_status
        self.marker = None
        self.marker_reads = 0
        self.renew_status = LockMutationStatus.RENEWED

    async def create_reservation(self, user_id, upload_id, bytes_reserved, *, owner_token, ttl_seconds):
        self.events.append("marker.create")
        self.marker = TusQuotaReservation(bytes_reserved, owner_token, TusReservationState.RESERVED)
        return ReservationCreateStatus.CREATED

    async def get_reservation(self, user_id, upload_id):
        self.marker_reads += 1
        if self.marker is not None:
            return self.marker
        if self.session is None:
            return None
        return TusQuotaReservation(self.session.total_length, OWNER, TusReservationState.RESERVED)

    async def release_reservation_once(self, user_id, upload_id, owner_token):
        self.events.append("marker.release")
        return self.marker_release_status

    async def renew_reservation(self, user_id, upload_id, owner_token, *, ttl_seconds):
        self.events.append("marker.renew")
        return LockMutationStatus.RENEWED

    async def create(self, session, ttl_seconds):
        self.events.append("session.create")
        if self.create_error:
            error = self.create_error
            self.create_error = None
            raise error
        self.session = session
        return SessionCreateStatus.CREATED

    async def get(self, upload_id):
        return self.session

    async def acquire_lock(self, upload_id, ttl_seconds):
        return LockAcquireResult(LockAcquireStatus.ACQUIRED, "L" * 32)

    async def renew_lock(self, upload_id, token, ttl_seconds):
        self.events.append("lock.renew")
        return self.renew_status

    async def release_lock(self, upload_id, token):
        self.events.append("lock.release")
        return LockMutationStatus.RELEASED

    async def append_part(self, upload_id, expected_offset, byte_count, part_number, etag, ttl_seconds, *, lock_token):
        self.events.append("redis.append")
        if self.append_status is AppendPartStatus.APPENDED:
            self.session = replace(
                self.session,
                offset=expected_offset + byte_count,
                parts=(*self.session.parts, TusPart(part_number, etag)),
            )
        return AppendPartResult(
            self.append_status,
            expected_offset + byte_count if self.append_status is AppendPartStatus.APPENDED else expected_offset,
        )

    async def mark_cleanup_required(self, upload_id, expected_offset, ttl_seconds, *, lock_token):
        self.events.append("redis.cleanup_required")
        self.session = replace(self.session, state=TusSessionState.CLEANUP_REQUIRED)
        return True

    async def mark_object_completed(self, upload_id, expected_offset, ttl_seconds, *, lock_token):
        self.events.append("redis.object_completed")
        return True

    async def mark_complete(
        self,
        upload_id,
        expected_offset,
        document_id,
        job_id,
        ttl_seconds,
        *,
        lock_token,
    ):
        self.events.append("redis.complete")
        self.session = replace(
            self.session,
            state=TusSessionState.COMPLETED,
            object_completed=True,
            document_id=document_id,
            job_id=job_id,
        )
        return CompleteResult(CompleteStatus.COMPLETED, expected_offset, document_id, job_id)

    async def delete_locked(self, upload_id, *, lock_token):
        self.events.append("session.delete")
        self.session = None
        return True


def _app(monkeypatch, s3, quota, store, *, lock_seconds=3, max_patch_bytes=8 * MIB):
    from infra import tus

    async def authenticated(request):
        return request.headers.get("X-Test-User", str(USER_ID))

    monkeypatch.setattr(tus, "_get_user_id", authenticated)
    monkeypatch.setattr(tus.settings, "TUS_MULTIPART_ENABLED", True)
    app = FastAPI()
    app.state.tus_service = tus.HostedTusMultipartService(
        Pool(),
        s3,
        SimpleNamespace(),
        quota,
        store,
        session_ttl_seconds=300,
        stale_seconds=180,
        lock_seconds=lock_seconds,
        max_patch_bytes=max_patch_bytes,
    )
    app.include_router(tus.router)
    return app


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_shielded_compensation_failure_does_not_replace_outer_cancellation():
    from infra.tus import _shielded

    started = asyncio.Event()
    finish = asyncio.Event()

    async def failing_compensation():
        started.set()
        await finish.wait()
        raise RuntimeError("cleanup failed")

    task = asyncio.create_task(_shielded(failing_compensation()))
    await started.wait()
    task.cancel()
    finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("stage", ["quota", "s3", "session"])
async def test_create_compensates_completed_stages_in_reverse_order(monkeypatch, stage):
    events = []
    quota = Quota(events, reserve_error=QuotaUnavailable("down") if stage == "quota" else None)
    s3 = S3(events, create_error=RuntimeError("s3 down") if stage == "s3" else None)
    store = Store(events, create_error=RuntimeError("redis down") if stage == "session" else None)
    app = _app(monkeypatch, s3, quota, store)

    async with await _client(app) as client:
        response = await client.post(
            "/v1/uploads",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": _metadata(),
            },
        )

    assert response.status_code >= 400
    if stage == "quota":
        assert events == ["quota.reserve"]
    elif stage == "s3":
        assert events == ["quota.reserve", "s3.create", "quota.release", "marker.release"]
    else:
        assert events[-3:] == ["s3.abort", "quota.release", "marker.release"]


@pytest.mark.parametrize(
    ("filename", "path"),
    [
        (f"{'a' * 252}.pdf", "/faults/"),
        ("fault\x00.pdf", "/faults/"),
        ("fault.pdf", '/bad"path/'),
        ("fault.pdf", "/bad\x1fpath/"),
        ("fault.pdf", f"/{'a' * 1024}/"),
    ],
)
async def test_create_rejects_invalid_session_metadata_before_quota_or_s3(
    monkeypatch,
    filename,
    path,
):
    events = []
    app = _app(monkeypatch, S3(events), Quota(events), Store(events))

    async with await _client(app) as client:
        response = await client.post(
            "/v1/uploads",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": _metadata(filename=filename, path=path),
            },
        )

    assert response.status_code == 400
    assert "quota.reserve" not in events
    assert "s3.create" not in events


async def test_offset_is_not_committed_until_s3_returns_an_etag(monkeypatch):
    events = []
    started, release = asyncio.Event(), asyncio.Event()
    session = _session()
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events, upload_gate=(started, release)), Quota(events), store)
    body = b"%PDF-" + b"a" * (5 * MIB - 5)

    async with await _client(app) as client:
        patch = asyncio.create_task(
            client.patch(
                f"/v1/uploads/{session.upload_id}",
                headers={
                    "X-Test-User": str(USER_ID),
                    "Tus-Resumable": "1.0.0",
                    "Upload-Offset": "0",
                    "Content-Type": "application/offset+octet-stream",
                },
                content=body,
            )
        )
        await started.wait()
        assert store.session.offset == 0
        assert store.session.parts == ()
        assert "redis.append" not in events
        release.set()
        response = await patch

    assert response.status_code == 204
    assert events.index("s3.upload.etag") < events.index("redis.append")


async def test_single_huge_stream_chunk_buffers_only_cap_plus_one_and_never_reaches_s3(monkeypatch):
    from infra import tus

    events = []
    session = _session(total=10 * MIB)
    app = _app(monkeypatch, S3(events), Quota(events), Store(events, session=session), max_patch_bytes=1024)

    class TrackingBytearray(bytearray):
        peak = 0

        def extend(self, value):
            super().extend(value)
            type(self).peak = max(type(self).peak, len(self))

    monkeypatch.setattr(tus, "bytearray", TrackingBytearray, raising=False)

    async def one_huge_chunk():
        yield b"x" * (2 * MIB)

    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=one_huge_chunk(),
        )

    assert response.status_code == 413
    assert TrackingBytearray.peak <= 1025
    assert "s3.upload.start" not in events


async def test_unexpected_append_cas_failure_marks_cleanup_required_without_guessing_offset(monkeypatch):
    events = []
    session = _session()
    store = Store(events, session=session, append_status=AppendPartStatus.OFFSET_MISMATCH)
    app = _app(monkeypatch, S3(events), Quota(events), store)
    body = b"%PDF-" + b"a" * (5 * MIB - 5)

    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=body,
        )

    assert response.status_code in {409, 423, 500, 503}
    assert store.session.offset == 0
    assert store.session.state is TusSessionState.CLEANUP_REQUIRED
    assert events.index("s3.upload.etag") < events.index("redis.append") < events.index("redis.cleanup_required")


async def test_size_or_signature_failure_deletes_object_aborts_and_releases_quota(monkeypatch):
    events = []
    first = TusPart(1, "etag-1")
    session = _session(total=5 * MIB + 4, offset=5 * MIB, parts=(first,))
    store = Store(events, session=session)
    s3 = S3(events, head=ObjectMetadata(size=session.total_length + 1, etag="object", content_type="application/pdf"))
    app = _app(monkeypatch, s3, Quota(events), store)

    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": str(5 * MIB),
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"tail",
        )

    assert response.status_code == 400
    assert "s3.delete" in events
    assert "s3.abort" in events
    assert events.count("quota.release") == 1
    assert events.count("marker.release") == 1
    assert events.count("session.delete") == 1


async def test_lock_renewal_loss_fences_s3_and_offset_commit(monkeypatch):
    events = []
    session = _session()
    store = Store(events, session=session)
    store.renew_status = LockMutationStatus.NOT_OWNER
    app = _app(monkeypatch, S3(events), Quota(events), store, lock_seconds=1)

    async def slow_body():
        yield b"%PDF-"
        await asyncio.sleep(0.45)
        yield b"a" * (5 * MIB - 5)

    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=slow_body(),
        )

    assert response.status_code in {409, 423}
    assert "lock.renew" in events
    assert "s3.upload.start" not in events
    assert "redis.append" not in events


async def test_create_abort_failure_persists_cleanup_and_retains_quota_marker(monkeypatch):
    events = []
    quota = Quota(events)
    s3 = S3(events, abort_error=RuntimeError("abort unavailable"))
    store = Store(events, create_error=RuntimeError("session unavailable"))
    app = _app(monkeypatch, s3, quota, store)

    async with await _client(app) as client:
        response = await client.post(
            "/v1/uploads",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": _metadata(),
            },
        )

    assert response.status_code >= 400
    assert "quota.release" not in events
    assert "marker.release" not in events
    assert store.session is not None
    assert store.session.state is TusSessionState.CLEANUP_REQUIRED


@pytest.mark.parametrize("stage", ["before_marker", "after_marker"])
@pytest.mark.parametrize("atomic_status", ["active", "settled"])
async def test_create_compensation_recovers_quota_release_response_loss_atomically(
    monkeypatch,
    stage,
    atomic_status,
):
    events = []
    quota = Quota(
        events,
        release_error=RuntimeError("quota response lost"),
        marker_settlement_status=atomic_status,
    )
    s3 = S3(events, create_error=RuntimeError("s3 down") if stage == "before_marker" else None)
    store = Store(events, create_error=RuntimeError("redis down") if stage == "after_marker" else None)
    app = _app(monkeypatch, s3, quota, store)

    async with await _client(app) as client:
        response = await client.post(
            "/v1/uploads",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": _metadata(),
            },
        )

    assert response.status_code == 503
    assert events.index("quota.release") < events.index("quota.marker_settle")
    expect_marker = stage == "after_marker" or atomic_status == "active"
    assert ("marker.create" in events) is expect_marker
    assert (store.marker == TusQuotaReservation(100, OWNER, TusReservationState.RESERVED)) is expect_marker
    assert "marker.release" not in events


async def test_create_compensation_recovers_marker_cas_response_loss_atomically(monkeypatch):
    events = []

    class LostMarkerStore(Store):
        async def release_reservation_once(self, user_id, upload_id, owner_token):
            events.append("marker.release")
            raise RuntimeError("marker response lost")

    app = _app(
        monkeypatch,
        S3(events),
        Quota(events, marker_settlement_status="settled"),
        LostMarkerStore(events, create_error=RuntimeError("redis down")),
    )

    async with await _client(app) as client:
        response = await client.post(
            "/v1/uploads",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Length": "100",
                "Upload-Metadata": _metadata(),
            },
        )

    assert response.status_code == 503
    assert events.index("marker.release") < events.index("quota.marker_settle")


async def test_committed_settlement_keeps_marker_reserved_when_quota_finalize_fails(monkeypatch):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(events, finalize_error=QuotaUnavailable("redis unavailable"))
    app = _app(monkeypatch, S3(events), quota, store)
    service = app.state.tus_service
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    with pytest.raises(Exception):
        await service._settle_committed(session, reservation, "L" * 32, session.upload_id, uuid4())

    assert "quota.finalize" in events
    assert "marker.release" not in events


async def test_committed_retry_releases_marker_when_same_quota_generation_is_absent(monkeypatch):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(events, finalize_result=False, marker_settlement_status="settled")
    app = _app(monkeypatch, S3(events), quota, store)
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    await app.state.tus_service._settle_committed(
        session,
        reservation,
        "L" * 32,
        session.upload_id,
        uuid4(),
    )

    assert events.index("quota.finalize") < events.index("quota.marker_settle")


async def test_committed_retry_never_releases_marker_for_stale_quota_generation(monkeypatch):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(events, finalize_result=False, marker_settlement_status="stale_generation")
    app = _app(monkeypatch, S3(events), quota, store)
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    with pytest.raises(Exception):
        await app.state.tus_service._settle_committed(
            session,
            reservation,
            "L" * 32,
            session.upload_id,
            uuid4(),
        )

    assert "quota.marker_settle" in events
    assert "marker.release" not in events


@pytest.mark.parametrize("loss_point", ["before_marker_cas", "after_marker_cas"])
async def test_completed_duplicate_recovers_quota_finalize_response_loss_around_marker_cas(
    monkeypatch,
    loss_point,
):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)

    class LostMarkerResponseStore(Store):
        def __init__(self):
            super().__init__(events, session=session)
            self.marker_calls = 0

        async def release_reservation_once(self, user_id, upload_id, owner_token):
            self.marker_calls += 1
            events.append("marker.release")
            if self.marker_calls == 1:
                if loss_point == "after_marker_cas":
                    self.marker_release_status = ReservationReleaseStatus.ALREADY_RELEASED
                    self.marker = TusQuotaReservation(
                        session.total_length,
                        OWNER,
                        TusReservationState.RELEASED,
                    )
                raise RuntimeError("marker response lost")
            return self.marker_release_status

    class LostFinalizeResponseQuota(Quota):
        def __init__(self):
            super().__init__(events, marker_settlement_status="settled")
            self.finalize_calls = 0

        async def finalize(self, reservation):
            self.finalize_calls += 1
            events.append("quota.finalize")
            return self.finalize_calls == 1

    store = LostMarkerResponseStore()
    quota = LostFinalizeResponseQuota()
    app = _app(monkeypatch, S3(events), quota, store)
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    with pytest.raises(RuntimeError):
        await app.state.tus_service._settle_committed(
            session,
            reservation,
            "L" * 32,
            session.upload_id,
            uuid4(),
        )

    await app.state.tus_service._retry_completed_settlement(store.session)

    if loss_point == "before_marker_cas":
        assert quota.finalize_calls == 2
        assert "quota.marker_settle" in events
    else:
        assert quota.finalize_calls == 1
        assert store.marker_release_status is ReservationReleaseStatus.ALREADY_RELEASED
        assert store.marker.state is TusReservationState.RELEASED


async def test_completed_duplicate_retries_reserved_quota_settlement(monkeypatch):
    events = []
    document_id = UUID("22222222-2222-2222-2222-222222222222")
    job_id = uuid4()
    session = replace(
        _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True),
        state=TusSessionState.COMPLETED,
        document_id=document_id,
        job_id=job_id,
    )
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)

    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "5",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"",
        )

    assert response.status_code == 204
    assert events[-2:] == ["quota.finalize", "marker.release"]


async def test_partial_discard_failure_keeps_cleanup_session_and_reserved_marker(monkeypatch):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(events)
    app = _app(monkeypatch, S3(events, abort_error=RuntimeError("abort down")), quota, store)
    service = app.state.tus_service
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    await service._discard(session, reservation, "L" * 32)

    assert store.session is not None
    assert store.session.state is TusSessionState.CLEANUP_REQUIRED
    assert "session.delete" not in events
    assert "marker.release" not in events


async def test_discard_replay_recovers_quota_release_response_loss_and_deletes_session(monkeypatch):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(
        events,
        release_error=RuntimeError("quota response lost"),
        marker_settlement_status="settled",
    )
    app = _app(monkeypatch, S3(events), quota, store)
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    await app.state.tus_service._discard(session, reservation, "L" * 32)

    assert events.index("quota.release") < events.index("quota.marker_settle")
    assert "session.delete" in events
    assert store.session is None


@pytest.mark.parametrize("operation", ["release", "finalize"])
async def test_false_quota_generation_result_never_releases_marker_or_session(monkeypatch, operation):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    quota = Quota(
        events,
        release_result=operation != "release",
        finalize_result=operation != "finalize",
        marker_settlement_status="stale_generation",
    )
    app = _app(monkeypatch, S3(events), quota, store)
    service = app.state.tus_service
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    if operation == "release":
        await service._discard(session, reservation, "L" * 32)
    else:
        with pytest.raises(Exception):
            await service._settle_committed(session, reservation, "L" * 32, session.upload_id, uuid4())

    assert "marker.release" not in events
    assert "session.delete" not in events


@pytest.mark.parametrize(
    "marker_status",
    [ReservationReleaseStatus.NOT_OWNER, ReservationReleaseStatus.MALFORMED],
)
async def test_stale_or_malformed_marker_cas_blocks_session_delete(monkeypatch, marker_status):
    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session, marker_release_status=marker_status)
    app = _app(monkeypatch, S3(events), Quota(events), store)
    reservation = QuotaReservation(USER_ID, session.upload_id, session.total_length, OWNER)

    await app.state.tus_service._discard(session, reservation, "L" * 32)

    assert "marker.release" in events
    assert "session.delete" not in events
    assert store.session is not None
    assert store.session.state is TusSessionState.CLEANUP_REQUIRED


@pytest.mark.parametrize("committed", [False, None])
async def test_database_failure_deletes_only_confirmed_rollback_orphan(monkeypatch, committed):
    from infra.tus import _CommitOutcomeError

    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)
    service = app.state.tus_service

    async def fail_persistence(_session):
        raise _CommitOutcomeError(committed, session.upload_id, uuid4())

    monkeypatch.setattr(service, "_persist_document_job", fail_persistence)

    with pytest.raises(Exception):
        await service._finalize(session, "L" * 32)

    if committed is False:
        assert "s3.delete" in events
        assert "quota.release" in events
        assert "session.delete" in events
    else:
        assert "s3.delete" not in events
        assert "quota.release" not in events
        assert store.session is not None
        assert store.session.object_completed is True


@pytest.mark.parametrize("committed", [False, None])
async def test_database_cancellation_preserves_cancelled_error_after_safe_compensation(
    monkeypatch,
    committed,
):
    from infra.tus import _CommitOutcomeError

    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)
    store = Store(events, session=session)
    app = _app(monkeypatch, S3(events), Quota(events), store)

    async def cancelled_persistence(_session):
        try:
            raise asyncio.CancelledError
        except asyncio.CancelledError as cancelled:
            raise _CommitOutcomeError(committed, session.upload_id, uuid4()) from cancelled

    monkeypatch.setattr(app.state.tus_service, "_persist_document_job", cancelled_persistence)

    with pytest.raises(asyncio.CancelledError):
        await app.state.tus_service._finalize(session, "L" * 32)

    if committed is False:
        assert "session.delete" in events
    else:
        assert "s3.delete" not in events
        assert store.session is not None


@pytest.mark.parametrize(
    ("rollback_succeeds", "confirm_result", "expected"),
    [(True, False, False), (False, False, None), (False, True, True)],
)
async def test_commit_attempted_is_false_only_after_original_transaction_rollback(
    monkeypatch,
    rollback_succeeds,
    confirm_result,
    expected,
):
    from infra.tus import _CommitOutcomeError

    events = []
    session = _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True)

    class Transaction:
        async def start(self):
            events.append("tx.start")

        async def rollback(self):
            events.append("tx.rollback")
            if not rollback_succeeds:
                raise RuntimeError("connection outcome unknown")

    class Connection:
        def transaction(self):
            return Transaction()

        async def execute(self, *args):
            return "INSERT 0 1"

        async def fetchval(self, *args):
            return True

    class PersistencePool:
        async def acquire(self):
            return Connection()

        async def release(self, conn):
            return None

        async def fetchval(self, *args):
            events.append("confirm")
            return confirm_result

    class Jobs:
        async def ensure_document_extraction_in_transaction(self, *args, **kwargs):
            return SimpleNamespace(id=uuid4()), True

    app = _app(monkeypatch, S3(events), Quota(events), Store(events, session=session))
    service = app.state.tus_service
    service.pool = PersistencePool()
    service.jobs = Jobs()

    async def ambiguous_commit(_transaction):
        raise RuntimeError("commit response lost")

    monkeypatch.setattr(service, "_commit_transaction", ambiguous_commit)

    with pytest.raises(_CommitOutcomeError) as captured:
        await service._persist_document_job(session)

    assert captured.value.committed is expected
    assert "tx.rollback" in events
    assert ("confirm" in events) is (not rollback_succeeds)


async def test_head_retries_completed_quota_marker_settlement(monkeypatch):
    events = []
    session = replace(
        _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True),
        state=TusSessionState.COMPLETED,
        document_id=UUID("22222222-2222-2222-2222-222222222222"),
        job_id=uuid4(),
    )
    store = Store(events, session=session)
    app = _app(
        monkeypatch,
        S3(events),
        Quota(events, finalize_result=False, marker_settlement_status="settled"),
        store,
    )

    async with await _client(app) as client:
        response = await client.head(
            f"/v1/uploads/{session.upload_id}",
            headers={"X-Test-User": str(USER_ID), "Tus-Resumable": "1.0.0"},
        )

    assert response.status_code == 200
    assert events.index("quota.finalize") < events.index("quota.marker_settle")


async def test_upload_cleanup_handler_is_replay_safe_and_releases_quota_once():
    from jobs.handlers import WorkerContext, handle_upload_cleanup
    from jobs.models import JobRecord, JobType

    upload_id = uuid4()
    calls = 0

    class Cleanup:
        async def cleanup(self, actual_upload_id, user_id):
            nonlocal calls
            assert actual_upload_id == upload_id
            assert user_id == USER_ID
            calls += 1
            return {"upload_id": str(upload_id), "status": "cleaned" if calls == 1 else "already_clean"}

    class Lease:
        async def checkpoint(self, conn=None):
            return None

    job = JobRecord(
        id=uuid4(),
        job_type=JobType.UPLOAD_CLEANUP,
        user_id=USER_ID,
        knowledge_base_id=None,
        payload={"upload_id": str(upload_id)},
    )
    context = WorkerContext(
        pool=SimpleNamespace(), s3=None, converter_url="", converter_secret="", tus_cleanup=Cleanup()
    )

    first = await handle_upload_cleanup(job, Lease(), context)
    second = await handle_upload_cleanup(job, Lease(), context)

    assert first == {"upload_id": str(upload_id), "status": "cleaned"}
    assert second == {"upload_id": str(upload_id), "status": "already_clean"}
    assert calls == 2


async def test_stale_scan_enqueues_all_sessions_without_live_knowledge_bases():
    from infra.tus import HostedTusCleanupService

    sessions = [
        _session(state=TusSessionState.CLEANUP_REQUIRED),
        replace(
            _session(state=TusSessionState.CLEANUP_REQUIRED),
            upload_id=UUID("33333333-3333-3333-3333-333333333333"),
        ),
    ]
    commands = []

    class Sessions:
        async def iter_sessions(self):
            for session in sessions:
                yield session

        async def iter_reservations(self):
            if False:
                yield None

    class Jobs:
        async def ensure_upload_cleanup(self, command, *, authenticated_user_id):
            assert authenticated_user_id == USER_ID
            assert command.knowledge_base_id is None
            assert command.document_id is None
            commands.append(command)
            return SimpleNamespace(), True

    cleanup = HostedTusCleanupService(
        Pool(),
        S3([]),
        Jobs(),
        Quota([]),
        Sessions(),
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    assert await cleanup.enqueue_stale_jobs() == 2
    assert [command.payload["upload_id"] for command in commands] == [str(session.upload_id) for session in sessions]


async def test_cleanup_scan_uses_minute_successor_keys_and_enqueues_completed_settlement():
    from infra.tus import HostedTusCleanupService

    completed = replace(
        _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True),
        state=TusSessionState.COMPLETED,
        document_id=UUID("22222222-2222-2222-2222-222222222222"),
        job_id=uuid4(),
    )
    marker = TusQuotaReservation(5, OWNER, TusReservationState.RESERVED)

    class Sessions:
        async def iter_sessions(self):
            yield completed

        async def iter_reservations(self):
            if False:
                yield None

        async def get_reservation(self, user_id, upload_id):
            return marker

    class Jobs:
        def __init__(self):
            self.keys = set()

        async def ensure_upload_cleanup(self, command, *, authenticated_user_id):
            if any(key.rsplit(":", 1)[0] != command.idempotency_key.rsplit(":", 1)[0] for key in self.keys):
                raise AssertionError("unexpected upload identity")
            if self.keys:
                return SimpleNamespace(), False
            self.keys.add(command.idempotency_key)
            return SimpleNamespace(), True

    jobs = Jobs()
    cleanup = HostedTusCleanupService(
        Pool(),
        S3([]),
        jobs,
        Quota([]),
        Sessions(),
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )
    first = datetime(2026, 1, 1, 0, 0, 15, tzinfo=UTC)

    await cleanup.enqueue_stale_jobs(now=first)
    await cleanup.enqueue_stale_jobs(now=first.replace(second=45))
    assert len(jobs.keys) == 1
    await cleanup.enqueue_stale_jobs(now=first.replace(minute=1))
    assert len(jobs.keys) == 1


async def test_cleanup_scan_enqueues_successor_only_after_active_job_is_terminal():
    from infra.tus import HostedTusCleanupService

    session = _session(state=TusSessionState.CLEANUP_REQUIRED)

    class Sessions:
        async def iter_sessions(self):
            yield session

        async def iter_reservations(self):
            if False:
                yield None

    class Jobs:
        def __init__(self):
            self.active = False
            self.keys = []

        async def ensure_upload_cleanup(self, command, *, authenticated_user_id):
            if self.active:
                return SimpleNamespace(), False
            self.active = True
            self.keys.append(command.idempotency_key)
            return SimpleNamespace(), True

    jobs = Jobs()
    cleanup = HostedTusCleanupService(
        Pool(),
        S3([]),
        jobs,
        Quota([]),
        Sessions(),
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )
    first = datetime(2026, 1, 1, 0, 0, 15, tzinfo=UTC)

    assert await cleanup.enqueue_stale_jobs(now=first) == 1
    assert await cleanup.enqueue_stale_jobs(now=first + timedelta(minutes=1)) == 0
    jobs.active = False
    assert await cleanup.enqueue_stale_jobs(now=first + timedelta(minutes=2)) == 1
    assert len(jobs.keys) == 2
    assert jobs.keys[0] != jobs.keys[1]


async def test_cleanup_scan_enqueues_reserved_marker_without_session():
    from infra.tus import HostedTusCleanupService

    upload_id = uuid4()
    marker = TusQuotaReservation(100, OWNER, TusReservationState.RESERVED)
    commands = []

    class Sessions:
        async def iter_sessions(self):
            if False:
                yield None

        async def iter_reservations(self):
            yield USER_ID, upload_id, marker

    class Jobs:
        async def ensure_upload_cleanup(self, command, *, authenticated_user_id):
            commands.append(command)
            return SimpleNamespace(), True

    cleanup = HostedTusCleanupService(
        Pool(),
        S3([]),
        Jobs(),
        Quota([]),
        Sessions(),
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    assert await cleanup.enqueue_stale_jobs() == 1
    assert commands[0].user_id == USER_ID
    assert commands[0].payload == {"upload_id": str(upload_id)}


async def test_cleanup_scan_immediately_enqueues_fresh_object_completed_unknown_commit():
    from infra.tus import HostedTusCleanupService

    fresh = _session(
        total=5,
        offset=5,
        parts=(TusPart(1, "etag"),),
        object_completed=True,
    )
    commands = []

    class Sessions:
        async def iter_sessions(self):
            yield fresh

        async def iter_reservations(self):
            if False:
                yield None

    class Jobs:
        async def ensure_upload_cleanup(self, command, *, authenticated_user_id):
            commands.append(command)
            return SimpleNamespace(), True

    cleanup = HostedTusCleanupService(
        Pool(),
        S3([]),
        Jobs(),
        Quota([]),
        Sessions(),
        session_ttl_seconds=300,
        stale_seconds=86_400,
        lock_seconds=10,
    )

    await cleanup.enqueue_stale_jobs(now=fresh.updated_at)

    assert len(commands) == 1
    assert commands[0].payload["upload_id"] == str(fresh.upload_id)


async def test_worker_reconciles_object_completed_document_job_without_s3_cleanup():
    from infra.tus import HostedTusCleanupService

    events = []
    job_id = uuid4()
    session = _session(
        total=5,
        offset=5,
        parts=(TusPart(1, "etag"),),
        object_completed=True,
        state=TusSessionState.UPLOADING,
    )
    store = Store(events, session=session)

    class ReconcilePool(Pool):
        async def fetchrow(self, query, *args):
            return {"document_id": session.upload_id, "job_id": job_id}

    cleanup = HostedTusCleanupService(
        ReconcilePool(),
        S3(events),
        SimpleNamespace(),
        Quota(events),
        store,
        session_ttl_seconds=300,
        stale_seconds=86_400,
        lock_seconds=10,
    )

    result = await cleanup.cleanup(session.upload_id, USER_ID)

    assert result["status"] == "committed"
    assert store.session.state is TusSessionState.COMPLETED
    assert store.session.document_id == session.upload_id
    assert store.session.job_id == job_id
    assert "s3.delete" not in events
    assert "s3.abort" not in events
    assert "session.delete" not in events


async def test_worker_fences_and_cleans_stale_object_completed_without_database_row():
    from infra.tus import HostedTusCleanupService

    events = []
    stale_at = datetime.now(UTC) - timedelta(minutes=5)
    session = replace(
        _session(
            total=5,
            offset=5,
            parts=(TusPart(1, "etag"),),
            object_completed=True,
            state=TusSessionState.UPLOADING,
        ),
        created_at=stale_at,
        updated_at=stale_at,
    )
    store = Store(events, session=session)
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    result = await cleanup.cleanup(session.upload_id, USER_ID)

    assert result["status"] == "cleaned"
    assert events.index("redis.cleanup_required") < events.index("s3.delete")
    assert events.index("redis.cleanup_required") < events.index("s3.abort")
    assert store.session is None


@pytest.mark.parametrize(
    ("settlement_status", "expected_status"),
    [("active", "retry"), ("settled", "cleaned")],
)
async def test_worker_settles_reserved_marker_without_session(settlement_status, expected_status):
    from infra.tus import HostedTusCleanupService

    events = []
    upload_id = uuid4()
    store = Store(events, session=None)
    store.marker = TusQuotaReservation(100, OWNER, TusReservationState.RESERVED)
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events, release_error=RuntimeError("response lost"), marker_settlement_status=settlement_status),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    result = await cleanup.cleanup(upload_id, USER_ID)

    assert result["status"] == expected_status
    assert events == ["quota.release", "quota.marker_settle", "lock.release"]
    assert store.marker_reads == 1


async def test_worker_settles_completed_session_without_deleting_committed_data():
    from infra.tus import HostedTusCleanupService

    events = []
    session = replace(
        _session(total=5, offset=5, parts=(TusPart(1, "etag"),), object_completed=True),
        state=TusSessionState.COMPLETED,
        document_id=UUID("22222222-2222-2222-2222-222222222222"),
        job_id=uuid4(),
    )
    store = Store(events, session=session)
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events, finalize_result=False, marker_settlement_status="settled"),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    assert (await cleanup.cleanup(session.upload_id, USER_ID))["status"] == "committed"
    assert events.index("quota.finalize") < events.index("quota.marker_settle")
    assert "s3.delete" not in events
    assert "s3.abort" not in events
    assert "session.delete" not in events
    assert store.session.state is TusSessionState.COMPLETED


async def test_worker_cleanup_release_response_loss_uses_atomic_marker_recovery():
    from infra.tus import HostedTusCleanupService

    events = []
    session = _session(state=TusSessionState.CLEANUP_REQUIRED)
    store = Store(events, session=session)
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(
            events,
            release_error=RuntimeError("quota response lost"),
            marker_settlement_status="settled",
        ),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    assert (await cleanup.cleanup(session.upload_id, USER_ID))["status"] == "cleaned"
    assert events.index("quota.release") < events.index("quota.marker_settle")
    assert store.session is None


async def test_cleanup_service_treats_missing_objects_as_success_and_is_repeat_safe():
    from infra.tus import HostedTusCleanupService

    class MissingObject(Exception):
        response = {"Error": {"Code": "NoSuchUpload"}}

    events = []
    session = _session(state=TusSessionState.CLEANUP_REQUIRED)
    store = Store(events, session=session)
    s3 = S3(events, abort_error=MissingObject())

    async def missing_delete(_key):
        events.append("s3.delete")
        raise MissingObject

    s3.delete_object = missing_delete
    cleanup = HostedTusCleanupService(
        Pool(),
        s3,
        SimpleNamespace(),
        Quota(events),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    first = await cleanup.cleanup(session.upload_id, USER_ID)
    second = await cleanup.cleanup(session.upload_id, USER_ID)

    assert first["status"] == "cleaned"
    assert second["status"] == "already_clean"
    assert events.count("quota.release") == 1
    assert events.count("marker.release") == 1
    assert events.count("session.delete") == 1


async def test_cleanup_service_stale_owner_never_deletes_after_losing_lock():
    from infra.tus import HostedTusCleanupService

    events = []
    session = _session(state=TusSessionState.CLEANUP_REQUIRED)

    class LoseBeforeDelete(Store):
        def __init__(self):
            super().__init__(events, session=session)
            self.renewals = 0

        async def renew_lock(self, upload_id, token, ttl_seconds):
            self.renewals += 1
            return LockMutationStatus.RENEWED if self.renewals == 1 else LockMutationStatus.NOT_OWNER

    store = LoseBeforeDelete()
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events),
        store,
        session_ttl_seconds=300,
        stale_seconds=60,
        lock_seconds=10,
    )

    result = await cleanup.cleanup(session.upload_id, USER_ID)

    assert result["status"] == "lock_lost"
    assert "quota.release" not in events
    assert "marker.release" not in events
    assert "session.delete" not in events


async def test_stale_uploading_cleanup_sets_fenced_state_before_any_s3_side_effect():
    from infra.tus import HostedTusCleanupService

    events = []
    session = _session()
    store = Store(events, session=session)
    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events),
        store,
        session_ttl_seconds=300,
        stale_seconds=-1,
        lock_seconds=10,
    )

    result = await cleanup.cleanup(session.upload_id, USER_ID)

    assert result["status"] == "cleaned"
    assert events.index("redis.cleanup_required") < events.index("s3.delete")


async def test_cleanup_state_fence_blocks_new_patch_even_if_cleanup_lock_expires(monkeypatch):
    from infra.tus import HostedTusCleanupService

    events = []
    started = asyncio.Event()
    finish = asyncio.Event()
    session = _session()
    store = Store(events, session=session)

    class GatedDeleteS3(S3):
        async def delete_object(self, key):
            self.events.append("s3.delete")
            started.set()
            await finish.wait()

    s3 = GatedDeleteS3(events)
    quota = Quota(events)
    cleanup = HostedTusCleanupService(
        Pool(),
        s3,
        SimpleNamespace(),
        quota,
        store,
        session_ttl_seconds=300,
        stale_seconds=-1,
        lock_seconds=1,
    )
    cleanup_task = asyncio.create_task(cleanup.cleanup(session.upload_id, USER_ID))
    await started.wait()
    assert store.session.state is TusSessionState.CLEANUP_REQUIRED

    app = _app(monkeypatch, s3, quota, store, lock_seconds=1)
    async with await _client(app) as client:
        response = await client.patch(
            f"/v1/uploads/{session.upload_id}",
            headers={
                "X-Test-User": str(USER_ID),
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"%PDF-" + b"a" * (5 * MIB - 5),
        )

    assert response.status_code == 409
    assert "s3.upload.start" not in events
    finish.set()
    assert (await cleanup_task)["status"] == "cleaned"


@pytest.mark.parametrize("cas_outcome", [False, RuntimeError("redis unavailable")])
async def test_stale_uploading_cleanup_cas_failure_has_zero_s3_side_effects(cas_outcome):
    from infra.tus import HostedTusCleanupService

    events = []
    session = _session()

    class FailingFenceStore(Store):
        async def mark_cleanup_required(self, *args, **kwargs):
            events.append("redis.cleanup_required")
            if isinstance(cas_outcome, Exception):
                raise cas_outcome
            return cas_outcome

    cleanup = HostedTusCleanupService(
        Pool(),
        S3(events),
        SimpleNamespace(),
        Quota(events),
        FailingFenceStore(events, session=session),
        session_ttl_seconds=300,
        stale_seconds=-1,
        lock_seconds=10,
    )

    if isinstance(cas_outcome, Exception):
        with pytest.raises(RuntimeError):
            await cleanup.cleanup(session.upload_id, USER_ID)
    else:
        assert (await cleanup.cleanup(session.upload_id, USER_ID))["status"] == "lock_lost"

    assert "s3.delete" not in events
    assert "s3.abort" not in events
