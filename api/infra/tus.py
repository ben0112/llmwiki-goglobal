import asyncio
import logging
import time
from base64 import b64decode, b64encode
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from auth import get_current_user
from config import settings
from fastapi import APIRouter, HTTPException, Request, Response
from infra.tasks import spawn_logged
from starlette.requests import ClientDisconnect
from telemetry import emit, replica_role

logger = logging.getLogger(__name__)

# Bytes buffered in memory before flushing a PATCH body to the temp file.
FLUSH_SIZE = 1_048_576


def sanitize_upload_path(raw: str) -> str:
    """Normalize a client-supplied logical path to `/a/b/` form.

    逐段过滤(丢弃空段、`.`、`..`)而非正则替换:re.sub 不回扫,
    `../..` 这类输入曾在单轮替换后仍残留 `/../`。
    """
    parts = [p for p in (raw or "").strip().replace("\\", "/").split("/")
             if p not in ("", ".", "..")]
    return "/" + "/".join(parts) + "/" if parts else "/"


def _append_file(path: Path, data: bytes):
    with open(path, "ab") as f:
        f.write(data)


@dataclass
class _StreamResult:
    bytes_written: int
    overflow: bool
    disconnected: bool


async def _drain_to_temp(request: Request, temp_path: Path, remaining: int) -> _StreamResult:
    """Append the PATCH body to temp_path, capped at `remaining` bytes."""
    buf = bytearray()
    bytes_written = 0
    try:
        async for chunk in request.stream():
            if bytes_written + len(buf) + len(chunk) > remaining:
                return _StreamResult(bytes_written, overflow=True, disconnected=False)
            buf.extend(chunk)
            if len(buf) >= FLUSH_SIZE:
                await asyncio.to_thread(_append_file, temp_path, bytes(buf))
                bytes_written += len(buf)
                buf.clear()
    except ClientDisconnect:
        bytes_written += await _flush(temp_path, buf)
        return _StreamResult(bytes_written, overflow=False, disconnected=True)
    bytes_written += await _flush(temp_path, buf)
    return _StreamResult(bytes_written, overflow=False, disconnected=False)


async def _flush(temp_path: Path, buf: bytearray) -> int:
    """Append any buffered bytes to disk; return how many were written."""
    if not buf:
        return 0
    await asyncio.to_thread(_append_file, temp_path, bytes(buf))
    return len(buf)


# Magic-byte signatures we check at finalize time. We don't try to handle
# every file type — just the major ones that downstream code routes by
# extension. Office/OOXML files are ZIP containers, so they share a magic.
_FILE_SIGNATURES: dict[str, tuple[tuple[bytes, ...], ...]] = {
    "pdf":  ((b"%PDF-",),),
    "png":  ((b"\x89PNG\r\n\x1a\n",),),
    "jpg":  ((b"\xff\xd8\xff",),),
    "jpeg": ((b"\xff\xd8\xff",),),
    "webp": ((b"RIFF",),),    # also check "WEBP" further in but RIFF is enough
    "gif":  ((b"GIF87a", b"GIF89a"),),
    # OOXML — pptx/docx/xlsx are ZIP containers (PK\x03\x04). Legacy
    # ppt/doc/xls are CFB (Compound File Binary) starting with D0CF11E0.
    "pptx": ((b"PK\x03\x04",),),
    "docx": ((b"PK\x03\x04",),),
    "xlsx": ((b"PK\x03\x04",),),
    "ppt":  ((b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),),
    "doc":  ((b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),),
    "xls":  ((b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),),
    # CSV and HTML are text — no reliable magic bytes. Skip the strict check
    # for these; the parser layer is more forgiving and they're low-risk.
}


def _validate_file_signature(temp_path: Path, ext: str) -> None:
    """Refuse files whose first bytes don't match their declared extension."""
    signatures = _FILE_SIGNATURES.get(ext)
    if not signatures:
        return
    try:
        with open(temp_path, "rb") as f:
            head = f.read(16)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}") from e
    # WebP is a RIFF container; check both the RIFF magic and the WEBP fourcc.
    if ext == "webp":
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return
        raise HTTPException(status_code=400, detail="File content does not match declared type .webp")
    for group in signatures:
        for prefix in group:
            if head.startswith(prefix):
                return
    raise HTTPException(
        status_code=400,
        detail=f"File content does not match declared type .{ext}",
    )

TUS_VERSION = "1.0.0"
MAX_SIZE = 1_073_741_824  # 1 GiB(前端 tus-js 以 50MB 分块 PATCH,反向代理 body 上限无需同步放大)
UPLOAD_DIR = Path("/tmp/supavault_tus_uploads")
STALE_SECONDS = 3600
MIN_MULTIPART_PART_SIZE = 5 * 1024 * 1024
FINALIZATION_TIMEOUT_SECONDS = 20

ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".pptx": "pptx", ".ppt": "ppt",
    ".docx": "docx", ".doc": "doc",
    ".png": "png", ".jpg": "jpg", ".jpeg": "jpeg",
    ".webp": "webp", ".gif": "gif",
    ".xlsx": "xlsx", ".xls": "xls", ".csv": "csv",
    ".html": "html", ".htm": "html",
}
CONTENT_TYPES = {
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "webp": "image/webp", "gif": "image/gif",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel", "csv": "text/csv",
    "html": "text/html", "htm": "text/html",
}

router = APIRouter(prefix="/v1/uploads", tags=["tus"])


@dataclass
class TusUpload:
    upload_id: str
    user_id: str
    upload_length: int
    upload_offset: int
    filename: str
    knowledge_base_id: str
    temp_path: Path
    path: str = "/"
    last_activity: float = field(default_factory=time.time)


class _LocalTusState:
    """Process-local TUS state used only by single-process Local mode."""

    def __init__(self) -> None:
        self.uploads: dict[str, TusUpload] = {}


_local_tus_state = _LocalTusState()
_uploads = _local_tus_state.uploads


def _check_tus_version(request: Request):
    version = request.headers.get("Tus-Resumable")
    if version != TUS_VERSION:
        raise HTTPException(status_code=412, detail=f"Unsupported Tus-Resumable version (expected {TUS_VERSION})")


def _parse_metadata(header: str) -> dict[str, str]:
    result = {}
    if not header:
        return result
    for pair in header.split(","):
        pair = pair.strip()
        parts = pair.split(" ", 1)
        key = parts[0]
        if len(parts) > 1:
            try:
                value = b64decode(parts[1]).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 in Upload-Metadata key '{key}'",
                ) from None
        else:
            value = ""
        result[key] = value
    return result


def _tus_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Tus-Resumable": TUS_VERSION}
    if extra:
        headers.update(extra)
    return headers


def _get_upload(upload_id: str, user_id: str) -> TusUpload:
    upload = _uploads.get(upload_id)
    if not upload or upload.user_id != user_id:
        raise HTTPException(status_code=404, detail="Upload not found")
    return upload


async def _finalize(upload: TusUpload, app_state) -> str:
    document_id = str(uuid4())
    user_id = upload.user_id
    ext = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else "pdf"
    file_type = ALLOWED_EXTENSIONS.get(f".{ext}", ext)
    s3_key = f"{user_id}/{document_id}/source.{ext}"
    title = upload.filename.rsplit(".", 1)[0] if "." in upload.filename else upload.filename
    file_size = upload.upload_length

    # Magic-byte + size checks. On mismatch we discard the temp file and pop
    # the upload entry before raising so a rejected upload doesn't linger.
    try:
        _validate_file_signature(upload.temp_path, ext)
        actual_size = upload.temp_path.stat().st_size
        if actual_size != upload.upload_length:
            raise HTTPException(
                status_code=400,
                detail=f"Upload size mismatch: declared {upload.upload_length}, actual {actual_size}",
            )
    except HTTPException:
        upload.temp_path.unlink(missing_ok=True)
        _uploads.pop(upload.upload_id, None)
        raise

    s3_service = app_state.s3_service
    if not s3_service:
        raise ValueError("File storage not configured")
    await s3_service.upload_file(s3_key, str(upload.temp_path), CONTENT_TYPES.get(ext, "application/octet-stream"))

    pool = app_state.pool
    try:
        await pool.execute(
            "INSERT INTO documents (id, knowledge_base_id, user_id, filename, path, title, "
            "source_kind, file_type, file_size, status) "
            "VALUES ($1::uuid, $2::uuid, $3, $4, $8, $5, 'source', $6, $7, 'pending')",
            document_id,
            upload.knowledge_base_id,
            user_id,
            upload.filename,
            title,
            file_type,
            file_size,
            upload.path,
        )
    finally:
        upload.temp_path.unlink(missing_ok=True)
        _uploads.pop(upload.upload_id, None)

    ocr_service = app_state.ocr_service
    if ocr_service:
        spawn_logged(ocr_service.process_document(document_id, user_id),
                     f"tus-process:{document_id[:8]}")

    logger.info("TUS finalized: doc=%s file=%s", document_id[:8], upload.filename)
    return document_id


async def cleanup_stale_uploads():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [uid for uid, u in _uploads.items() if now - u.last_activity > STALE_SECONDS]
        for uid in stale:
            upload = _uploads.pop(uid, None)
            if upload:
                upload.temp_path.unlink(missing_ok=True)
                logger.info("Cleaned stale TUS upload: %s", uid)


async def _get_user_id(request: Request) -> str:
    return await get_current_user(request)


def _hosted_tus_service(request: Request):
    service = getattr(request.app.state, "tus_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Resumable upload coordination is unavailable")
    return service


def _canonical_upload_id(raw: str) -> UUID:
    try:
        upload_id = UUID(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=404, detail="Upload not found") from None
    if str(upload_id) != raw:
        raise HTTPException(status_code=404, detail="Upload not found")
    return upload_id


def _validate_signature_bytes(head: bytes, ext: str) -> None:
    signatures = _FILE_SIGNATURES.get(ext)
    if not signatures:
        return
    valid = head[:4] == b"RIFF" and head[8:12] == b"WEBP" if ext == "webp" else any(
        head.startswith(prefix) for group in signatures for prefix in group
    )
    if not valid:
        raise HTTPException(status_code=400, detail=f"File content does not match declared type .{ext}")


def _reservation_marker_settled(status) -> bool:
    from infra.tus_sessions import ReservationReleaseStatus

    return status in {
        ReservationReleaseStatus.RELEASED,
        ReservationReleaseStatus.ALREADY_RELEASED,
        ReservationReleaseStatus.NOT_FOUND,
    }


async def _release_quota_and_marker(quota, sessions, reservation) -> bool:
    """Settle a released quota generation and its marker across response loss."""
    from infra.quota import QuotaMarkerSettlementStatus

    try:
        released = await quota.release(reservation)
    except Exception:  # noqa: BLE001 -- an executed Redis CAS may have lost its response
        released = False
    if released is True:
        try:
            marker_status = await sessions.release_reservation_once(
                reservation.user_id,
                reservation.upload_id,
                reservation.owner_token,
            )
        except Exception:  # noqa: BLE001 -- marker CAS may have executed before its response was lost
            marker_status = None
        if marker_status is not None:
            return _reservation_marker_settled(marker_status)
    marker_settlement = await quota.settle_tus_marker_if_absent(reservation)
    return marker_settlement is QuotaMarkerSettlementStatus.SETTLED


async def _ensure_reservation_marker(sessions, reservation, ttl_seconds: int) -> bool:
    """Ensure an exact durable marker exists for quota-only recovery."""
    from infra.tus_sessions import ReservationCreateStatus, TusReservationState

    status = await sessions.create_reservation(
        reservation.user_id,
        reservation.upload_id,
        reservation.bytes,
        owner_token=reservation.owner_token,
        ttl_seconds=ttl_seconds,
    )
    if status is ReservationCreateStatus.CREATED:
        return True
    marker = await sessions.get_reservation(reservation.user_id, reservation.upload_id)
    return (
        marker is not None
        and marker.bytes == reservation.bytes
        and marker.owner_token == reservation.owner_token
        and marker.state is TusReservationState.RESERVED
    )


async def _shielded(operation) -> None:
    task = asyncio.create_task(operation)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        try:
            await task
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 -- preserve outer cancellation
            logger.error("Shielded TUS compensation failed error_type=%s", type(exc).__name__)
        raise cancelled


class HostedTusMultipartService:
    """Replica-safe Hosted TUS protocol over Redis sessions and S3 multipart."""

    def __init__(
        self,
        pool,
        s3_service,
        job_service,
        quota_service,
        session_store,
        *,
        session_ttl_seconds: int,
        stale_seconds: int,
        lock_seconds: int,
        max_patch_bytes: int,
    ) -> None:
        self.pool = pool
        self.s3 = s3_service
        self.jobs = job_service
        self.quota = quota_service
        self.sessions = session_store
        self.session_ttl_seconds = session_ttl_seconds
        self.stale_seconds = stale_seconds
        self.lock_seconds = lock_seconds
        self.max_patch_bytes = max_patch_bytes

    async def create(self, request: Request, user_id: str) -> Response:  # noqa: C901
        from infra.quota import QuotaExceeded, QuotaUnavailable
        from infra.tus_sessions import (
            LockAcquireStatus,
            ReservationCreateStatus,
            SessionCreateStatus,
            TusSession,
            TusSessionState,
            validate_tus_upload_metadata,
        )

        _check_tus_version(request)
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid authenticated user") from None
        upload_length = self._upload_length(request)
        metadata = _parse_metadata(request.headers.get("Upload-Metadata", ""))
        filename, ext = self._filename(metadata)
        path = sanitize_upload_path(metadata.get("path", "/"))
        try:
            validate_tus_upload_metadata(filename, path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid Upload-Metadata: {exc}") from None
        try:
            kb_id = UUID(metadata.get("knowledge_base_id", "").strip())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid knowledge_base_id format") from None
        if not metadata.get("knowledge_base_id", "").strip():
            raise HTTPException(status_code=400, detail="Missing knowledge_base_id in Upload-Metadata")
        owned = await self.pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM knowledge_bases WHERE id = $1 AND user_id = $2)",
            kb_id,
            user_uuid,
        )
        if not owned:
            raise HTTPException(status_code=403, detail="Knowledge base not found or not owned by you")

        upload_id = uuid4()
        try:
            reservation = await self.quota.reserve(
                user_uuid,
                upload_id,
                upload_length,
                ttl_seconds=self.session_ttl_seconds,
            )
        except QuotaExceeded as exc:
            raise HTTPException(status_code=413, detail=f"Storage quota exceeded ({exc.used_bytes}/{exc.limit_bytes} bytes)") from None
        except QuotaUnavailable:
            raise HTTPException(status_code=503, detail="Storage quota coordination is unavailable") from None

        key = f"{user_uuid}/{upload_id}/source.{ext}"
        multipart_id = None
        marker_created = False
        session = None
        lock_token = None
        lock_lost = asyncio.Event()
        lock_renewal = None
        try:
            multipart_id = await self.s3.create_multipart(key, CONTENT_TYPES.get(ext, "application/octet-stream"))
            now = datetime.now(UTC)
            session = TusSession(
                upload_id=upload_id,
                user_id=user_uuid,
                knowledge_base_id=kb_id,
                filename=filename,
                path=path,
                content_type=CONTENT_TYPES.get(ext, "application/octet-stream"),
                total_length=upload_length,
                offset=0,
                s3_key=key,
                multipart_upload_id=multipart_id,
                parts=(),
                created_at=now,
                updated_at=now,
                state=TusSessionState.UPLOADING,
                document_id=None,
                job_id=None,
                reservation_bytes=upload_length,
                object_completed=False,
            )
            acquired = await self.sessions.acquire_lock(upload_id, self.lock_seconds)
            if acquired.status is not LockAcquireStatus.ACQUIRED or acquired.token is None:
                raise RuntimeError("upload initialization lock is unavailable")
            lock_token = acquired.token
            lock_renewal = asyncio.create_task(self._renew_upload_lock(upload_id, lock_token, lock_lost))
            marker = await self.sessions.create_reservation(
                user_uuid,
                upload_id,
                upload_length,
                owner_token=reservation.owner_token,
                ttl_seconds=self.session_ttl_seconds,
            )
            if marker is not ReservationCreateStatus.CREATED:
                raise RuntimeError("reservation marker collision")
            marker_created = True
            created = await self.sessions.create_under_lock(
                session,
                self.session_ttl_seconds,
                lock_token=lock_token,
            )
            if created is not SessionCreateStatus.CREATED:
                raise RuntimeError("upload session collision")
        except asyncio.CancelledError:
            await _shielded(
                self._compensate_create(
                    reservation,
                    key,
                    multipart_id,
                    marker_created,
                    session,
                    lock_token=lock_token,
                )
            )
            raise
        except Exception:  # noqa: BLE001 -- all adapter failures require the same compensation
            await _shielded(
                self._compensate_create(
                    reservation,
                    key,
                    multipart_id,
                    marker_created,
                    session,
                    lock_token=lock_token,
                )
            )
            raise HTTPException(status_code=503, detail="Could not initialize resumable upload") from None
        finally:
            if lock_renewal is not None:
                lock_renewal.cancel()
                with suppress(asyncio.CancelledError):
                    await lock_renewal
            if lock_token is not None:
                with suppress(Exception):
                    await asyncio.shield(self.sessions.release_lock(upload_id, lock_token))
        emit(
            logger,
            "tus_session_created",
            upload_id=upload_id,
            state="uploading",
            replica_role=replica_role("api"),
        )
        return Response(status_code=201, headers=_tus_headers({"Location": f"/v1/uploads/{upload_id}"}))

    async def _compensate_create(  # noqa: C901
        self,
        reservation,
        key: str,
        multipart_id: str | None,
        marker_created: bool,
        session,
        *,
        lock_token: str | None,
    ) -> None:
        abort_ok = True
        if multipart_id is not None:
            try:
                await self._abort_ignoring_missing(key, multipart_id)
            except Exception as exc:  # noqa: BLE001 -- S3 SDK error types are adapter-specific
                abort_ok = False
                logger.error(
                    "TUS create abort failed upload_id=%s error_type=%s",
                    reservation.upload_id,
                    type(exc).__name__,
                )
        if not abort_ok and session is None:
            try:
                anchored = await _ensure_reservation_marker(
                    self.sessions,
                    reservation,
                    self.session_ttl_seconds,
                )
            except Exception as exc:  # noqa: BLE001 -- retain quota after uncertain object cleanup
                anchored = False
                logger.error(
                    "TUS create recovery marker failed upload_id=%s error_type=%s",
                    reservation.upload_id,
                    type(exc).__name__,
                )
            if not anchored:
                logger.error("TUS create recovery is unanchored upload_id=%s", reservation.upload_id)
            return
        if not abort_ok and session is not None:
            marker_anchored = marker_created
            from infra.tus_sessions import SessionCreateStatus, TusSessionState

            session_anchored = False
            try:
                cleanup_session = replace(session, state=TusSessionState.CLEANUP_REQUIRED)
                created = await self.sessions.create(cleanup_session, self.session_ttl_seconds)
                session_anchored = created is SessionCreateStatus.CREATED
                if not session_anchored:
                    stored = await self.sessions.get(session.upload_id)
                    session_anchored = (
                        stored is not None
                        and stored.user_id == cleanup_session.user_id
                        and stored.s3_key == cleanup_session.s3_key
                        and stored.multipart_upload_id == cleanup_session.multipart_upload_id
                        and stored.state is TusSessionState.CLEANUP_REQUIRED
                    )
                if created is SessionCreateStatus.ALREADY_EXISTS and not session_anchored:
                    cleanup_token = lock_token
                    acquired = None
                    if cleanup_token is None:
                        acquired = await self.sessions.acquire_lock(session.upload_id, self.lock_seconds)
                        cleanup_token = acquired.token
                    if cleanup_token is not None:
                        try:
                            marked = await self.sessions.mark_cleanup_required(
                                session.upload_id,
                                session.offset,
                                self.session_ttl_seconds,
                                lock_token=cleanup_token,
                            )
                            session_anchored = bool(marked)
                        finally:
                            if acquired is not None:
                                await self.sessions.release_lock(session.upload_id, cleanup_token)
            except Exception as exc:  # noqa: BLE001 -- marker recovery failure must not block this attempt
                logger.error(
                    "TUS create recovery session failed upload_id=%s error_type=%s",
                    reservation.upload_id,
                    type(exc).__name__,
                )
            if session_anchored and not marker_anchored:
                try:
                    marker_anchored = await _ensure_reservation_marker(
                        self.sessions,
                        reservation,
                        self.session_ttl_seconds,
                    )
                except Exception as exc:  # noqa: BLE001 -- the durable session remains independently scannable
                    logger.error(
                        "TUS create recovery marker failed upload_id=%s error_type=%s",
                        reservation.upload_id,
                        type(exc).__name__,
                    )
            if not marker_anchored:
                logger.error("TUS create recovery marker is uncertain upload_id=%s", reservation.upload_id)
            if not session_anchored:
                logger.error("TUS create recovery session is uncertain upload_id=%s", reservation.upload_id)
            return
        if session is not None:
            try:
                stored = await self.sessions.get(session.upload_id)
                owned = (
                    stored is not None
                    and stored.user_id == session.user_id
                    and stored.knowledge_base_id == session.knowledge_base_id
                    and stored.s3_key == session.s3_key
                    and stored.multipart_upload_id == session.multipart_upload_id
                    and stored.reservation_bytes == session.reservation_bytes
                )
                if stored is not None and (not owned or lock_token is None):
                    return
                if owned and not await self.sessions.delete_locked(session.upload_id, lock_token=lock_token):
                    return
            except Exception as exc:  # noqa: BLE001 -- retain quota when session ownership is uncertain
                logger.error(
                    "TUS create session compensation failed upload_id=%s error_type=%s",
                    reservation.upload_id,
                    type(exc).__name__,
                )
                return
        settled = False
        try:
            settled = await _release_quota_and_marker(self.quota, self.sessions, reservation)
        except Exception as exc:  # noqa: BLE001 -- quota backends expose adapter-specific failures
            logger.error(
                "TUS create quota release failed upload_id=%s error_type=%s",
                reservation.upload_id,
                type(exc).__name__,
            )
        if not settled:
            try:
                anchored = await _ensure_reservation_marker(
                    self.sessions,
                    reservation,
                    self.session_ttl_seconds,
                )
            except Exception as exc:  # noqa: BLE001 -- preserve quota ownership on recovery failure
                anchored = False
                logger.error(
                    "TUS create quota recovery marker failed upload_id=%s error_type=%s",
                    reservation.upload_id,
                    type(exc).__name__,
                )
            if not anchored:
                logger.error("TUS create quota recovery is unanchored upload_id=%s", reservation.upload_id)

    async def head(self, raw_upload_id: str, request: Request, user_id: str) -> Response:
        del request
        session = await self._owned_session(_canonical_upload_id(raw_upload_id), user_id)
        from infra.tus_sessions import TusSessionState

        if session.state is TusSessionState.COMPLETED:
            await self._retry_completed_settlement(session)
        metadata = {
            "filename": session.filename,
            "knowledge_base_id": str(session.knowledge_base_id),
            "path": session.path,
        }
        encoded_metadata = ",".join(
            f"{key} {b64encode(value.encode('utf-8')).decode('ascii')}" for key, value in metadata.items()
        )
        headers = _tus_headers(
            {
                "Upload-Offset": str(session.offset),
                "Upload-Length": str(session.total_length),
                "Upload-Metadata": encoded_metadata,
                "Cache-Control": "no-store",
            }
        )
        if session.document_id is not None and session.job_id is not None:
            headers.update({"X-Document-Id": str(session.document_id), "X-Job-Id": str(session.job_id)})
        return Response(status_code=200, headers=headers)

    async def patch(self, raw_upload_id: str, request: Request, user_id: str) -> Response:  # noqa: C901
        from infra.tus_sessions import AppendPartStatus, LockAcquireStatus, TusSessionState

        _check_tus_version(request)
        if request.headers.get("Content-Type", "") != "application/offset+octet-stream":
            raise HTTPException(status_code=415, detail="Content-Type must be application/offset+octet-stream")
        upload_id = _canonical_upload_id(raw_upload_id)
        session = await self._owned_session(upload_id, user_id)
        client_offset = self._client_offset(request)
        if session.state is TusSessionState.COMPLETED:
            await self._retry_completed_settlement(session)
            return self._completed_response(session, client_offset)
        if session.state is TusSessionState.CLEANUP_REQUIRED:
            raise HTTPException(status_code=409, detail="Upload requires cleanup", headers={"Retry-After": "1"})

        acquired = await self.sessions.acquire_lock(upload_id, self.lock_seconds)
        if acquired.status is not LockAcquireStatus.ACQUIRED or acquired.token is None:
            raise HTTPException(status_code=423, detail="Upload is busy", headers={"Retry-After": "1"})
        token = acquired.token
        lost = asyncio.Event()
        renewal = asyncio.create_task(self._renew_upload_lock(upload_id, token, lost))
        try:
            session = await self._owned_session(upload_id, user_id)
            if session.state is TusSessionState.COMPLETED:
                await self._retry_completed_settlement(session)
                return self._completed_response(session, client_offset)
            if session.state is TusSessionState.CLEANUP_REQUIRED:
                raise HTTPException(status_code=409, detail="Upload requires cleanup", headers={"Retry-After": "1"})
            if client_offset != session.offset:
                raise HTTPException(
                    status_code=409,
                    detail="Offset mismatch",
                    headers=_tus_headers({"Upload-Offset": str(session.offset)}),
                )
            await self._renew_session_reservation(session)

            body, too_large = await self._read_patch_body(request, session.total_length - session.offset)
            if too_large:
                raise HTTPException(
                    status_code=413,
                    detail="PATCH body exceeds the configured limit",
                    headers=_tus_headers({"Upload-Offset": str(session.offset)}),
                )
            if lost.is_set():
                raise HTTPException(status_code=423, detail="Upload lock was lost", headers={"Retry-After": "1"})
            if not body and session.offset != session.total_length:
                raise HTTPException(status_code=400, detail="PATCH body must not be empty")

            if body:
                next_offset = session.offset + len(body)
                if next_offset > session.total_length:
                    raise HTTPException(
                        status_code=413,
                        detail="Body exceeds declared Upload-Length",
                        headers=_tus_headers({"Upload-Offset": str(session.offset)}),
                    )
                if next_offset < session.total_length and len(body) < MIN_MULTIPART_PART_SIZE:
                    raise HTTPException(
                        status_code=422,
                        detail="Non-final multipart chunks must be at least 5 MiB",
                        headers=_tus_headers({"Upload-Offset": str(session.offset)}),
                    )
                part_number = len(session.parts) + 1
                etag = await self.s3.upload_part(
                    session.s3_key,
                    session.multipart_upload_id,
                    part_number,
                    body,
                )
                if lost.is_set():
                    raise HTTPException(status_code=423, detail="Upload lock was lost", headers={"Retry-After": "1"})
                appended = await self.sessions.append_part(
                    upload_id,
                    session.offset,
                    len(body),
                    part_number,
                    etag,
                    self.session_ttl_seconds,
                    lock_token=token,
                )
                if appended.status is not AppendPartStatus.APPENDED:
                    if appended.status is not AppendPartStatus.LOCK_LOST and not lost.is_set():
                        await self.sessions.mark_cleanup_required(
                            upload_id,
                            session.offset,
                            self.session_ttl_seconds,
                            lock_token=token,
                        )
                    raise HTTPException(
                        status_code=423 if appended.status is AppendPartStatus.LOCK_LOST else 409,
                        detail="Upload offset could not be committed",
                        headers=_tus_headers({"Upload-Offset": str(appended.offset), "Retry-After": "1"}),
                    )
                session = await self.sessions.get(upload_id)
                if session is None:
                    raise HTTPException(status_code=503, detail="Upload session disappeared")

            if session.offset == session.total_length:
                return await self._finalize(session, token)
            return Response(status_code=204, headers=_tus_headers({"Upload-Offset": str(session.offset)}))
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
            with suppress(Exception):
                await asyncio.shield(self.sessions.release_lock(upload_id, token))

    async def _renew_upload_lock(self, upload_id: UUID, token: str, lost: asyncio.Event) -> None:
        from infra.tus_sessions import LockMutationStatus

        interval = max(self.lock_seconds / 3, 0.05)
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await self.sessions.renew_lock(upload_id, token, self.lock_seconds)
            except Exception:  # noqa: BLE001 -- any coordination failure loses the renewable lock
                lost.set()
                return
            if renewed is not LockMutationStatus.RENEWED:
                lost.set()
                return

    async def _read_patch_body(self, request: Request, remaining: int) -> tuple[bytes, bool]:
        content_length = request.headers.get("Content-Length")
        if content_length:
            with suppress(ValueError):
                if int(content_length) > self.max_patch_bytes or int(content_length) > remaining:
                    return b"", True
        body = bytearray()
        limit = min(self.max_patch_bytes, remaining)
        try:
            async for chunk in request.stream():
                available = limit + 1 - len(body)
                if available > 0:
                    body.extend(chunk[:available])
                if len(chunk) > available or len(body) > limit:
                    return bytes(body), True
        except ClientDisconnect:
            raise HTTPException(status_code=400, detail="PATCH body was interrupted") from None
        return bytes(body), False

    async def _finalize(self, session, token: str) -> Response:  # noqa: C901
        from infra.quota import QuotaReservation, QuotaUnavailable
        from services.s3 import MultipartPart

        marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
        if marker is None or marker.state.value != "reserved" or marker.bytes != session.reservation_bytes:
            await self.sessions.mark_cleanup_required(
                session.upload_id,
                session.offset,
                self.session_ttl_seconds,
                lock_token=token,
            )
            raise HTTPException(status_code=503, detail="Upload reservation is unavailable")
        reservation = QuotaReservation(session.user_id, session.upload_id, marker.bytes, marker.owner_token)
        if not session.object_completed:
            try:
                await self.s3.complete_multipart(
                    session.s3_key,
                    session.multipart_upload_id,
                    [MultipartPart(part.part_number, part.etag) for part in session.parts],
                )
            except Exception as exc:  # noqa: BLE001 -- normalize S3 adapter-specific missing errors
                if not self._is_missing(exc) or await self.s3.head_object(session.s3_key) is None:
                    raise HTTPException(status_code=502, detail="Could not complete multipart upload") from None
            metadata = await self.s3.head_object(session.s3_key)
            if metadata is None or metadata.size != session.total_length:
                await _shielded(self._discard(session, reservation, token))
                raise HTTPException(status_code=400, detail="Upload size mismatch")
            head = await self.s3.read_range(session.s3_key, 0, min(15, session.total_length - 1))
            ext = session.filename.rsplit(".", 1)[-1].lower()
            try:
                _validate_signature_bytes(head, ext)
            except HTTPException:
                await _shielded(self._discard(session, reservation, token))
                raise
            if not await self.sessions.mark_object_completed(
                session.upload_id,
                session.offset,
                self.session_ttl_seconds,
                lock_token=token,
            ):
                await self.sessions.mark_cleanup_required(
                    session.upload_id,
                    session.offset,
                    self.session_ttl_seconds,
                    lock_token=token,
                )
                raise HTTPException(status_code=503, detail="Upload completion state is unavailable")
            session = await self.sessions.get(session.upload_id)
            if session is None:
                raise HTTPException(status_code=503, detail="Upload session disappeared")

        try:
            renewed = await self.quota.renew(reservation, self.session_ttl_seconds)
            marker_renewed = await self.sessions.renew_reservation(
                session.user_id,
                session.upload_id,
                marker.owner_token,
                ttl_seconds=self.session_ttl_seconds,
            )
        except QuotaUnavailable:
            raise HTTPException(status_code=503, detail="Storage quota coordination is unavailable") from None
        if not renewed or marker_renewed.value != "renewed":
            await _shielded(self._discard(session, reservation, token))
            raise HTTPException(status_code=503, detail="Storage quota reservation expired before persistence")

        try:
            document_id, job_id = await self._persist_document_job(session)
        except _CommitOutcomeError as exc:
            cancelled = exc.__cause__ if isinstance(exc.__cause__, asyncio.CancelledError) else None
            try:
                if exc.committed is False:
                    await _shielded(self._discard(session, reservation, token))
                elif exc.committed is True and exc.job_id is not None:
                    await _shielded(
                        self._settle_committed(session, reservation, token, exc.document_id, exc.job_id)
                    )
            except (Exception, asyncio.CancelledError) as compensation_error:  # noqa: BLE001
                if cancelled is None:
                    raise
                logger.error(
                    "TUS cancellation compensation failed upload_id=%s error_type=%s",
                    session.upload_id,
                    type(compensation_error).__name__,
                )
            if cancelled is not None:
                raise cancelled from None
            raise HTTPException(status_code=503, detail="Upload persistence outcome is unavailable") from None
        await self._settle_committed(session, reservation, token, document_id, job_id)
        completed = await self.sessions.get(session.upload_id)
        if completed is None:
            raise HTTPException(status_code=503, detail="Upload completion state is unavailable")
        emit(
            logger,
            "tus_session_completed",
            upload_id=session.upload_id,
            state="completed",
            replica_role=replica_role("api"),
        )
        return self._completed_response(completed, completed.offset)

    async def _renew_session_reservation(self, session):
        from infra.quota import QuotaReservation, QuotaUnavailable
        from infra.tus_sessions import LockMutationStatus, TusReservationState

        marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
        if marker is None or marker.state is not TusReservationState.RESERVED or marker.bytes != session.reservation_bytes:
            raise HTTPException(status_code=503, detail="Upload reservation is unavailable")
        reservation = QuotaReservation(session.user_id, session.upload_id, marker.bytes, marker.owner_token)
        try:
            renewed = await self.quota.renew(reservation, self.session_ttl_seconds)
            marker_renewed = await self.sessions.renew_reservation(
                session.user_id,
                session.upload_id,
                marker.owner_token,
                ttl_seconds=self.session_ttl_seconds,
            )
        except QuotaUnavailable:
            raise HTTPException(status_code=503, detail="Storage quota coordination is unavailable") from None
        if renewed is not True or marker_renewed is not LockMutationStatus.RENEWED:
            raise HTTPException(status_code=503, detail="Storage quota reservation expired")
        return reservation

    async def _persist_document_job(self, session) -> tuple[UUID, UUID]:
        document_id = session.upload_id
        title = session.filename.rsplit(".", 1)[0]
        ext = session.filename.rsplit(".", 1)[-1].lower()
        conn = None
        transaction = None
        commit_attempted = False
        job = None
        try:
            conn = await self.pool.acquire()
            async with asyncio.timeout(FINALIZATION_TIMEOUT_SECONDS):
                transaction = conn.transaction()
                await transaction.start()
                await conn.execute(
                    "INSERT INTO documents (id, knowledge_base_id, user_id, filename, path, title, "
                    "source_kind, file_type, file_size, status) "
                    "VALUES ($1, $2, $3, $4, $5, $6, 'source', $7, $8, 'pending') "
                    "ON CONFLICT (id) DO NOTHING",
                    document_id,
                    session.knowledge_base_id,
                    session.user_id,
                    session.filename,
                    session.path,
                    title,
                    ALLOWED_EXTENSIONS.get(f".{ext}", ext),
                    session.total_length,
                )
                owned = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1 AND user_id = $2 "
                    "AND knowledge_base_id = $3 AND filename = $4 AND file_size = $5)",
                    document_id,
                    session.user_id,
                    session.knowledge_base_id,
                    session.filename,
                    session.total_length,
                )
                if not owned:
                    raise RuntimeError("document id conflicts with another resource")
                job, _ = await self.jobs.ensure_document_extraction_in_transaction(
                    conn,
                    document_id=document_id,
                    user_id=session.user_id,
                    knowledge_base_id=session.knowledge_base_id,
                    restart_terminal=False,
                )
                commit_attempted = True
                await self._commit_transaction(transaction)
        except (Exception, asyncio.CancelledError) as exc:
            committed: bool | None = False if transaction is None else None
            if transaction is not None:
                try:
                    await asyncio.shield(transaction.rollback())
                    committed = False
                except (Exception, asyncio.CancelledError):  # noqa: BLE001 -- rollback outcome stays unknown
                    committed = None
            if committed is not False and commit_attempted and job is not None:
                confirmed = await asyncio.shield(
                    self._confirm_document_job_committed(session, document_id, job.id)
                )
                committed = True if confirmed is True else None
            raise _CommitOutcomeError(committed, document_id, None if job is None else job.id) from exc
        finally:
            if conn is not None:
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.shield(self.pool.release(conn))
        return document_id, job.id

    async def _commit_transaction(self, transaction) -> None:
        await transaction.commit()

    async def _confirm_document_job_committed(self, session, document_id: UUID, job_id: UUID) -> bool | None:
        try:
            return bool(
                await self.pool.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM documents d JOIN background_jobs j ON j.document_id = d.id "
                    "WHERE d.id = $1 AND d.user_id = $2 AND d.knowledge_base_id = $3 "
                    "AND j.id = $4 AND j.user_id = d.user_id)",
                    document_id,
                    session.user_id,
                    session.knowledge_base_id,
                    job_id,
                )
            )
        except Exception:  # noqa: BLE001 -- read-after-write failures make the outcome unknown
            return None

    async def _settle_committed(self, session, reservation, token: str, document_id: UUID, job_id: UUID) -> None:
        from infra.quota import QuotaMarkerSettlementStatus
        from infra.tus_sessions import CompleteStatus

        result = await self.sessions.mark_complete(
            session.upload_id,
            session.offset,
            document_id,
            job_id,
            self.session_ttl_seconds,
            lock_token=token,
        )
        if result.status not in {CompleteStatus.COMPLETED, CompleteStatus.ALREADY_COMPLETED}:
            raise HTTPException(status_code=503, detail="Upload completion state is unavailable")
        finalized = await self.quota.finalize(reservation)
        if finalized is not True:
            marker_settlement = await self.quota.settle_tus_marker_if_absent(reservation)
            if marker_settlement is not QuotaMarkerSettlementStatus.SETTLED:
                raise RuntimeError("quota reservation generation was not finalized")
            return
        marker_status = await self.sessions.release_reservation_once(
            session.user_id,
            session.upload_id,
            reservation.owner_token,
        )
        if not _reservation_marker_settled(marker_status):
            raise RuntimeError("upload reservation marker was not owner-settled")

    async def _retry_completed_settlement(self, session) -> None:
        from infra.quota import QuotaMarkerSettlementStatus, QuotaReservation
        from infra.tus_sessions import TusReservationState

        try:
            marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
            if marker is None or marker.state is TusReservationState.RELEASED:
                return
            reservation = QuotaReservation(session.user_id, session.upload_id, marker.bytes, marker.owner_token)
            finalized = await self.quota.finalize(reservation)
            if finalized is not True:
                marker_settlement = await self.quota.settle_tus_marker_if_absent(reservation)
                if marker_settlement is not QuotaMarkerSettlementStatus.SETTLED:
                    return
                return
            marker_status = await self.sessions.release_reservation_once(
                session.user_id,
                session.upload_id,
                marker.owner_token,
            )
            if not _reservation_marker_settled(marker_status):
                return
        except Exception as exc:  # noqa: BLE001 -- settlement is intentionally retried on duplicate PATCH
            logger.error(
                "TUS completed quota settlement deferred upload_id=%s error_type=%s",
                session.upload_id,
                type(exc).__name__,
            )

    async def _discard(self, session, reservation, token: str) -> None:
        with suppress(Exception):
            await self.sessions.mark_cleanup_required(
                session.upload_id,
                session.offset,
                self.session_ttl_seconds,
                lock_token=token,
            )
        objects_clean = True
        try:
            await self.s3.delete_object(session.s3_key)
        except Exception as exc:  # noqa: BLE001 -- normalize S3 adapter-specific missing errors
            objects_clean = self._is_missing(exc)
        try:
            await self._abort_ignoring_missing(session.s3_key, session.multipart_upload_id)
        except Exception:  # noqa: BLE001 -- incomplete object cleanup retains all ownership state
            objects_clean = False
        if not objects_clean:
            return
        try:
            settled = await _release_quota_and_marker(self.quota, self.sessions, reservation)
        except Exception:  # noqa: BLE001 -- marker failure retains the cleanup session
            return
        if not settled:
            return
        with suppress(Exception):
            await self.sessions.delete_locked(session.upload_id, lock_token=token)

    async def _abort_ignoring_missing(self, key: str, multipart_id: str) -> None:
        try:
            await self.s3.abort_multipart(key, multipart_id)
        except Exception as exc:
            if not self._is_missing(exc):
                raise

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        return code in {"404", "NoSuchKey", "NoSuchUpload", "NotFound"}

    async def _owned_session(self, upload_id: UUID, user_id: str):
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Upload not found") from None
        try:
            session = await self.sessions.get(upload_id)
        except Exception:  # noqa: BLE001 -- coordination backends map uniformly to service unavailable
            raise HTTPException(status_code=503, detail="Upload coordination is unavailable") from None
        if session is None or session.user_id != user_uuid:
            raise HTTPException(status_code=404, detail="Upload not found")
        return session

    @staticmethod
    def _client_offset(request: Request) -> int:
        raw = request.headers.get("Upload-Offset")
        if raw is None:
            raise HTTPException(status_code=400, detail="Missing Upload-Offset header")
        try:
            offset = int(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Upload-Offset") from None
        if offset < 0:
            raise HTTPException(status_code=400, detail="Invalid Upload-Offset")
        return offset

    @staticmethod
    def _upload_length(request: Request) -> int:
        raw = request.headers.get("Upload-Length")
        if raw is None:
            raise HTTPException(status_code=400, detail="Missing Upload-Length header")
        try:
            length = int(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Upload-Length") from None
        if length < 1:
            raise HTTPException(status_code=400, detail="Upload-Length must be positive")
        if length > MAX_SIZE:
            raise HTTPException(status_code=413, detail=f"Upload-Length exceeds maximum of {MAX_SIZE} bytes")
        return length

    @staticmethod
    def _filename(metadata: dict[str, str]) -> tuple[str, str]:
        raw = metadata.get("filename", "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Missing filename in Upload-Metadata")
        filename = raw.replace("\\", "/").rsplit("/", 1)[-1]
        if not filename or filename in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid filename in Upload-Metadata")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if f".{ext}" not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        return filename, ext

    @staticmethod
    def _completed_response(session, client_offset: int) -> Response:
        if client_offset != session.offset:
            raise HTTPException(
                status_code=409,
                detail="Offset mismatch",
                headers=_tus_headers({"Upload-Offset": str(session.offset)}),
            )
        return Response(
            status_code=204,
            headers=_tus_headers(
                {
                    "Upload-Offset": str(session.offset),
                    "X-Document-Id": str(session.document_id),
                    "X-Job-Id": str(session.job_id),
                }
            ),
        )


class _CommitOutcomeError(RuntimeError):
    def __init__(self, committed: bool | None, document_id: UUID, job_id: UUID | None) -> None:
        self.committed = committed
        self.document_id = document_id
        self.job_id = job_id
        super().__init__("upload database commit outcome was not confirmed")


class HostedTusCleanupService:
    """Worker-only stale multipart discovery and token-fenced reclamation."""

    def __init__(
        self,
        pool,
        s3_service,
        job_service,
        quota_service,
        session_store,
        *,
        session_ttl_seconds: int,
        stale_seconds: int,
        lock_seconds: int,
    ) -> None:
        self.pool = pool
        self.s3 = s3_service
        self.jobs = job_service
        self.quota = quota_service
        self.sessions = session_store
        self.session_ttl_seconds = session_ttl_seconds
        self.stale_seconds = stale_seconds
        self.lock_seconds = lock_seconds

    async def enqueue_stale_jobs(self, *, now: datetime | None = None) -> int:
        from infra.tus_sessions import TusReservationState, TusSessionState
        from jobs.models import JobCreate, JobType

        current = now or datetime.now(UTC)
        stale_before = current.timestamp() - self.stale_seconds
        scan_bucket = int(current.timestamp() // 60)
        created = 0

        async def enqueue(user_id: UUID, upload_id: UUID) -> None:
            nonlocal created
            _, was_created = await self.jobs.ensure_upload_cleanup(
                JobCreate(
                    job_type=JobType.UPLOAD_CLEANUP,
                    user_id=user_id,
                    payload={"upload_id": str(upload_id)},
                    idempotency_key=f"upload.cleanup:{upload_id}:scan:{scan_bucket}",
                ),
                authenticated_user_id=user_id,
            )
            created += int(was_created)
            if was_created:
                emit(
                    logger,
                    "tus_session_stale",
                    upload_id=upload_id,
                    state="cleanup_required",
                    replica_role=replica_role("worker"),
                )

        async for session in self.sessions.iter_sessions():
            if session.state is TusSessionState.COMPLETED:
                marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
                if marker is None or marker.state is not TusReservationState.RESERVED:
                    continue
            elif session.object_completed:
                pass
            elif session.state is not TusSessionState.CLEANUP_REQUIRED and session.updated_at.timestamp() > stale_before:
                continue
            await enqueue(session.user_id, session.upload_id)
        async for user_id, upload_id, marker in self.sessions.iter_reservations():
            if marker.state is TusReservationState.RESERVED and await self.sessions.get(upload_id) is None:
                await enqueue(user_id, upload_id)
        return created

    async def cleanup(self, upload_id: UUID, expected_user_id: UUID) -> dict[str, object]:  # noqa: C901
        from infra.quota import QuotaReservation
        from infra.tus_sessions import (
            LockAcquireStatus,
            LockMutationStatus,
            TusReservationState,
            TusSessionState,
        )

        def outcome(status: str) -> dict[str, object]:
            emit(
                logger,
                "upload_cleanup_finished",
                upload_id=upload_id,
                state=status,
                replica_role=replica_role("worker"),
            )
            return {"upload_id": str(upload_id), "status": status}

        acquired = await self.sessions.acquire_lock(upload_id, self.lock_seconds)
        if acquired.status is not LockAcquireStatus.ACQUIRED or acquired.token is None:
            return outcome("contended")
        token = acquired.token
        try:
            session = await self.sessions.get(upload_id)
            if session is None:
                marker = await self.sessions.get_reservation(expected_user_id, upload_id)
                if marker is None or marker.state is TusReservationState.RELEASED:
                    return outcome("already_clean")
                reservation = QuotaReservation(expected_user_id, upload_id, marker.bytes, marker.owner_token)
                settled = await _release_quota_and_marker(self.quota, self.sessions, reservation)
                return outcome("cleaned" if settled else "retry")
            if session.user_id != expected_user_id:
                return outcome("owner_mismatch")
            if session.state is TusSessionState.COMPLETED:
                settled = await self._settle_completed(session)
                return outcome("committed" if settled else "retry")
            if await self.sessions.renew_lock(upload_id, token, self.lock_seconds) is not LockMutationStatus.RENEWED:
                return outcome("lock_lost")
            stale_before = datetime.now(UTC).timestamp() - self.stale_seconds
            stale = session.updated_at.timestamp() <= stale_before
            if session.object_completed:
                reconciled = await self._reconcile_object_completed(session, token)
                if reconciled is True:
                    return outcome("committed")
                if reconciled is False or (session.state is TusSessionState.UPLOADING and not stale):
                    return outcome("retry")
            if session.state is not TusSessionState.CLEANUP_REQUIRED and not stale:
                return outcome("active")
            if session.state is TusSessionState.UPLOADING:
                fenced = await self.sessions.mark_cleanup_required(
                    upload_id,
                    session.offset,
                    self.session_ttl_seconds,
                    lock_token=token,
                )
                if not fenced:
                    return outcome("lock_lost")
                session = replace(session, state=TusSessionState.CLEANUP_REQUIRED)
            objects_clean = await self._cleanup_objects(session)
            if not objects_clean:
                return outcome("retry")
            if await self.sessions.renew_lock(upload_id, token, self.lock_seconds) is not LockMutationStatus.RENEWED:
                return outcome("lock_lost")

            marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
            if marker is not None and marker.state is TusReservationState.RESERVED:
                reservation = QuotaReservation(
                    session.user_id,
                    session.upload_id,
                    marker.bytes,
                    marker.owner_token,
                )
                if not await _release_quota_and_marker(self.quota, self.sessions, reservation):
                    return outcome("retry")
            deleted = await self.sessions.delete_locked(upload_id, lock_token=token)
            return outcome("cleaned" if deleted else "lock_lost")
        finally:
            with suppress(Exception):
                await asyncio.shield(self.sessions.release_lock(upload_id, token))

    async def _settle_completed(self, session) -> bool:
        from infra.quota import QuotaMarkerSettlementStatus, QuotaReservation
        from infra.tus_sessions import TusReservationState

        marker = await self.sessions.get_reservation(session.user_id, session.upload_id)
        if marker is None or marker.state is TusReservationState.RELEASED:
            return True
        reservation = QuotaReservation(session.user_id, session.upload_id, marker.bytes, marker.owner_token)
        finalized = await self.quota.finalize(reservation)
        if finalized is not True:
            status = await self.quota.settle_tus_marker_if_absent(reservation)
            return status is QuotaMarkerSettlementStatus.SETTLED
        marker_status = await self.sessions.release_reservation_once(
            session.user_id,
            session.upload_id,
            marker.owner_token,
        )
        return _reservation_marker_settled(marker_status)

    async def _reconcile_object_completed(self, session, token: str) -> bool | None:
        from infra.tus_sessions import CompleteStatus

        row = await self.pool.fetchrow(
            "SELECT d.id AS document_id, j.id AS job_id FROM documents d "
            "JOIN background_jobs j ON j.document_id = d.id AND j.user_id = d.user_id "
            "WHERE d.id = $1 AND d.user_id = $2 AND d.knowledge_base_id = $3 "
            "AND d.filename = $4 AND d.file_size = $5 AND j.job_type = 'document.extract' "
            "ORDER BY j.created_at ASC, j.id ASC LIMIT 1",
            session.upload_id,
            session.user_id,
            session.knowledge_base_id,
            session.filename,
            session.total_length,
        )
        if row is None:
            return None
        result = await self.sessions.mark_complete(
            session.upload_id,
            session.offset,
            row["document_id"],
            row["job_id"],
            self.session_ttl_seconds,
            lock_token=token,
        )
        if result.status not in {CompleteStatus.COMPLETED, CompleteStatus.ALREADY_COMPLETED}:
            return False
        completed = await self.sessions.get(session.upload_id)
        return completed is not None and await self._settle_completed(completed)

    async def _cleanup_objects(self, session) -> bool:
        try:
            await self.s3.delete_object(session.s3_key)
        except Exception as exc:  # noqa: BLE001 -- normalize S3 adapter-specific missing errors
            if not HostedTusMultipartService._is_missing(exc):
                return False
        try:
            await self.s3.abort_multipart(session.s3_key, session.multipart_upload_id)
        except Exception as exc:  # noqa: BLE001 -- normalize S3 adapter-specific missing errors
            if not HostedTusMultipartService._is_missing(exc):
                return False
        return True


@router.options("")
async def tus_options():
    return Response(
        status_code=204,
        headers=_tus_headers({
            "Tus-Version": TUS_VERSION,
            "Tus-Max-Size": str(MAX_SIZE),
            "Tus-Extension": "creation",
        }),
    )


@router.post("", status_code=201)
async def tus_create(request: Request):
    user_id = await _get_user_id(request)
    if settings.MODE == "hosted":
        return await _hosted_tus_service(request).create(request, user_id)
    _check_tus_version(request)

    upload_length_str = request.headers.get("Upload-Length")
    if not upload_length_str:
        raise HTTPException(status_code=400, detail="Missing Upload-Length header")
    try:
        upload_length = int(upload_length_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Upload-Length") from None
    if upload_length < 1:
        raise HTTPException(status_code=400, detail="Upload-Length must be positive")
    if upload_length > MAX_SIZE:
        raise HTTPException(status_code=413, detail=f"Upload-Length exceeds maximum of {MAX_SIZE} bytes")

    meta_header = request.headers.get("Upload-Metadata", "")
    metadata = _parse_metadata(meta_header)

    filename = metadata.get("filename", "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename in Upload-Metadata")

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS.keys())}",
        )

    kb_id = metadata.get("knowledge_base_id", "").strip()
    if not kb_id:
        raise HTTPException(status_code=400, detail="Missing knowledge_base_id in Upload-Metadata")

    pool = request.app.state.pool

    try:
        import uuid as _uuid
        _uuid.UUID(kb_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid knowledge_base_id format") from None

    kb_owner = await pool.fetchval(
        "SELECT user_id::text FROM knowledge_bases WHERE id = $1::uuid",
        kb_id,
    )
    if kb_owner != user_id:
        raise HTTPException(status_code=403, detail="Knowledge base not found or not owned by you")
    user_limits = await pool.fetchrow(
        "SELECT storage_limit_bytes FROM users WHERE id = $1",
        user_id,
    )
    storage_limit = user_limits["storage_limit_bytes"] if user_limits else settings.QUOTA_MAX_STORAGE_BYTES

    current_bytes = await pool.fetchval(
        "SELECT COALESCE(SUM(file_size), 0) FROM documents WHERE user_id = $1",
        user_id,
    )
    in_progress_bytes = sum(u.upload_length for u in _uploads.values() if u.user_id == user_id)
    if current_bytes + in_progress_bytes + upload_length > storage_limit:
        used_mb = current_bytes / (1024 * 1024)
        max_mb = storage_limit / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Storage quota exceeded. Using {used_mb:.0f} MB of {max_mb:.0f} MB.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = str(uuid4())
    temp_path = UPLOAD_DIR / upload_id
    temp_path.touch()

    # Sanitize path: must start with /, no traversal, no double slashes
    upload_path = sanitize_upload_path(metadata.get("path", "/"))

    upload = TusUpload(
        upload_id=upload_id,
        user_id=user_id,
        upload_length=upload_length,
        upload_offset=0,
        filename=filename,
        knowledge_base_id=kb_id,
        temp_path=temp_path,
        path=upload_path,
    )
    _uploads[upload_id] = upload

    location = f"/v1/uploads/{upload_id}"
    return Response(status_code=201, headers=_tus_headers({"Location": location}))


@router.head("/{upload_id}")
async def tus_head(upload_id: str, request: Request):
    user_id = await _get_user_id(request)
    if settings.MODE == "hosted":
        return await _hosted_tus_service(request).head(upload_id, request, user_id)
    upload = _get_upload(upload_id, user_id)
    return Response(
        status_code=200,
        headers=_tus_headers({
            "Upload-Offset": str(upload.upload_offset),
            "Upload-Length": str(upload.upload_length),
            "Cache-Control": "no-store",
        }),
    )


@router.patch("/{upload_id}", status_code=204)
async def tus_patch(upload_id: str, request: Request):
    user_id = await _get_user_id(request)
    if settings.MODE == "hosted":
        return await _hosted_tus_service(request).patch(upload_id, request, user_id)
    _check_tus_version(request)

    content_type = request.headers.get("Content-Type", "")
    if content_type != "application/offset+octet-stream":
        raise HTTPException(status_code=415, detail="Content-Type must be application/offset+octet-stream")

    upload = _get_upload(upload_id, user_id)

    offset_str = request.headers.get("Upload-Offset")
    if offset_str is None:
        raise HTTPException(status_code=400, detail="Missing Upload-Offset header")
    try:
        client_offset = int(offset_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Upload-Offset") from None

    if client_offset != upload.upload_offset:
        raise HTTPException(status_code=409, detail="Offset mismatch")

    # Hard ceiling on how many bytes this PATCH can write — never let the
    # client exceed Upload-Length, regardless of streamed body size.
    remaining = upload.upload_length - upload.upload_offset
    result = await _drain_to_temp(request, upload.temp_path, remaining)
    upload.upload_offset += result.bytes_written
    upload.last_activity = time.time()

    if result.overflow:
        raise HTTPException(status_code=413, detail="Body exceeds declared Upload-Length")

    if result.disconnected:
        # Client hung up mid-PATCH. Bytes received so far are persisted and the
        # offset advanced, so the client resumes from HEAD + the next PATCH.
        return Response(
            status_code=204,
            headers=_tus_headers({"Upload-Offset": str(upload.upload_offset)}),
        )

    if upload.upload_offset > upload.upload_length:
        upload.temp_path.unlink(missing_ok=True)
        _uploads.pop(upload_id, None)
        raise HTTPException(status_code=400, detail="Upload exceeded declared length")

    headers = _tus_headers({"Upload-Offset": str(upload.upload_offset)})

    if upload.upload_offset == upload.upload_length:
        try:
            document_id = await _finalize(upload, request.app.state)
            headers["X-Document-Id"] = document_id
        except HTTPException:
            raise
        except Exception:
            logger.exception("TUS finalization failed for upload %s", upload_id)
            raise HTTPException(status_code=500, detail="Finalization failed") from None

    return Response(status_code=204, headers=headers)
