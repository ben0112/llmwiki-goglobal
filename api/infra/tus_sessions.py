"""Strict Redis-backed TUS session, lock, and release-marker primitives."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID

MAX_SAFE_INTEGER = 9_007_199_254_740_991
# Redis embeds lua-cjson with at most 14 significant digits. Keep mutable upload
# counters below that boundary so a decode/mutate/encode cycle is exact.
MAX_UPLOAD_BYTES = 9_999_999_999_999
MAX_PART_NUMBER = 10_000
MAX_TTL_SECONDS = 2_147_483_647
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_SESSION_FIELDS = frozenset(
    {
        "upload_id",
        "user_id",
        "knowledge_base_id",
        "filename",
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
    }
)
_PART_FIELDS = frozenset({"part_number", "etag"})


class RedisClient(Protocol):
    async def set(self, key: str, value: str, **kwargs: object) -> object: ...

    async def get(self, key: str) -> bytes | str | None: ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...


class InvalidTusSessionError(ValueError):
    """A stored record does not match the closed TUS JSON schema."""


class TusSessionProtocolError(RuntimeError):
    """Redis returned a response outside this module's bounded protocol."""


class TusSessionState(StrEnum):
    UPLOADING = "uploading"
    COMPLETED = "completed"


class SessionCreateStatus(StrEnum):
    CREATED = "created"
    ALREADY_EXISTS = "already_exists"


class AppendPartStatus(StrEnum):
    APPENDED = "appended"
    OFFSET_MISMATCH = "offset_mismatch"
    NOT_FOUND = "not_found"
    NOT_UPLOADING = "not_uploading"
    LENGTH_EXCEEDED = "length_exceeded"
    PART_OUT_OF_ORDER = "part_out_of_order"
    ALREADY_COMPLETED = "already_completed"
    MALFORMED = "malformed"
    INVALID_BYTE_COUNT = "invalid_byte_count"


class CompleteStatus(StrEnum):
    COMPLETED = "completed"
    ALREADY_COMPLETED = "already_completed"
    IDENTIFIER_CONFLICT = "identifier_conflict"
    OFFSET_MISMATCH = "offset_mismatch"
    NOT_FOUND = "not_found"
    NOT_UPLOADING = "not_uploading"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


class LockAcquireStatus(StrEnum):
    ACQUIRED = "acquired"
    CONTENDED = "contended"


class LockMutationStatus(StrEnum):
    RENEWED = "renewed"
    RELEASED = "released"
    NOT_OWNER = "not_owner"


class ReservationCreateStatus(StrEnum):
    CREATED = "created"
    ALREADY_EXISTS = "already_exists"


class ReservationReleaseStatus(StrEnum):
    NOT_FOUND = "not_found"
    RELEASED = "released"
    ALREADY_RELEASED = "already_released"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class AppendPartResult:
    status: AppendPartStatus
    offset: int


@dataclass(frozen=True, slots=True)
class CompleteResult:
    status: CompleteStatus
    offset: int
    document_id: UUID | None
    job_id: UUID | None


@dataclass(frozen=True, slots=True)
class LockAcquireResult:
    status: LockAcquireStatus
    token: str | None


def _require_uuid(value: object, name: str) -> UUID:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} must be a UUID")
    return value


def _parse_uuid(value: object, name: str, *, optional: bool = False) -> UUID | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise InvalidTusSessionError(f"{name} must be a UUID string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise InvalidTusSessionError(f"{name} must be a UUID string") from exc
    if str(parsed) != value:
        raise InvalidTusSessionError(f"{name} must use canonical UUID encoding")
    return parsed


def _require_safe_integer(value: object, name: str, *, minimum: int = 0, maximum: int = MAX_SAFE_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _parse_safe_integer(value: object, name: str, *, minimum: int = 0, maximum: int = MAX_SAFE_INTEGER) -> int:
    try:
        return _require_safe_integer(value, name, minimum=minimum, maximum=maximum)
    except ValueError as exc:
        raise InvalidTusSessionError(str(exc)) from exc


def _require_ttl(ttl_seconds: object) -> int:
    return _require_safe_integer(ttl_seconds, "ttl_seconds", minimum=1, maximum=MAX_TTL_SECONDS)


def _validate_text(value: object, name: str, *, max_length: int, forbid_path: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string of at most {max_length} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    if forbid_path and (value in {".", ".."} or "/" in value or "\\" in value):
        raise ValueError(f"{name} must be a sanitized basename")
    return value


def _validate_etag(value: object) -> str:
    etag = _validate_text(value, "etag", max_length=512)
    if any(character.isspace() for character in etag):
        raise ValueError("etag must not contain whitespace")
    return etag


def _require_utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")
    return value.astimezone(UTC)


def _datetime_to_microseconds(value: datetime) -> int:
    delta = value - _EPOCH
    microseconds = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return _require_safe_integer(microseconds, "timestamp")


def _microseconds_to_datetime(value: object, name: str) -> datetime:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise InvalidTusSessionError(f"{name} must be a canonical decimal microsecond string")
    microseconds = _parse_safe_integer(int(value), name)
    return _EPOCH + timedelta(microseconds=microseconds)


@dataclass(frozen=True, slots=True)
class TusPart:
    part_number: int
    etag: str

    def __post_init__(self) -> None:
        _require_safe_integer(self.part_number, "part_number", minimum=1, maximum=MAX_PART_NUMBER)
        _validate_etag(self.etag)


@dataclass(frozen=True, slots=True)
class TusSession:
    upload_id: UUID
    user_id: UUID
    knowledge_base_id: UUID
    filename: str
    content_type: str
    total_length: int
    offset: int
    s3_key: str
    multipart_upload_id: str
    parts: tuple[TusPart, ...]
    created_at: datetime
    updated_at: datetime
    state: TusSessionState
    document_id: UUID | None
    job_id: UUID | None
    reservation_bytes: int

    def __post_init__(self) -> None:
        _require_uuid(self.upload_id, "upload_id")
        _require_uuid(self.user_id, "user_id")
        _require_uuid(self.knowledge_base_id, "knowledge_base_id")
        _validate_text(self.filename, "filename", max_length=255, forbid_path=True)
        _validate_text(self.content_type, "content_type", max_length=255)
        total_length = _require_safe_integer(self.total_length, "total_length", maximum=MAX_UPLOAD_BYTES)
        offset = _require_safe_integer(self.offset, "offset", maximum=MAX_UPLOAD_BYTES)
        if offset > total_length:
            raise ValueError("offset must not exceed total_length")
        _validate_text(self.s3_key, "s3_key", max_length=1_024)
        _validate_text(self.multipart_upload_id, "multipart_upload_id", max_length=1_024)
        if not isinstance(self.parts, tuple) or any(not isinstance(part, TusPart) for part in self.parts):
            raise ValueError("parts must be a tuple of TusPart values")
        for expected, part in enumerate(self.parts, start=1):
            if part.part_number != expected:
                raise ValueError("parts must have unique, contiguous, ordered part numbers")
        created_at = _require_utc(self.created_at, "created_at")
        updated_at = _require_utc(self.updated_at, "updated_at")
        _datetime_to_microseconds(created_at)
        _datetime_to_microseconds(updated_at)
        if updated_at < created_at:
            raise ValueError("updated_at must not precede created_at")
        if not isinstance(self.state, TusSessionState):
            raise ValueError("state must be a TusSessionState")
        _require_safe_integer(self.reservation_bytes, "reservation_bytes", maximum=MAX_UPLOAD_BYTES)
        if self.state is TusSessionState.UPLOADING:
            if self.document_id is not None or self.job_id is not None:
                raise ValueError("uploading sessions cannot contain document or job IDs")
        elif (
            self.offset != self.total_length
            or not isinstance(self.document_id, UUID)
            or not isinstance(self.job_id, UUID)
        ):
            raise ValueError("completed sessions require full offset plus document and job IDs")

    def to_json(self) -> str:
        payload = {
            "upload_id": str(self.upload_id),
            "user_id": str(self.user_id),
            "knowledge_base_id": str(self.knowledge_base_id),
            "filename": self.filename,
            "content_type": self.content_type,
            "total_length": self.total_length,
            "offset": self.offset,
            "s3_key": self.s3_key,
            "multipart_upload_id": self.multipart_upload_id,
            "parts": [{"part_number": part.part_number, "etag": part.etag} for part in self.parts],
            "created_at": str(_datetime_to_microseconds(self.created_at)),
            "updated_at": str(_datetime_to_microseconds(self.updated_at)),
            "state": self.state.value,
            "document_id": str(self.document_id) if self.document_id is not None else None,
            "job_id": str(self.job_id) if self.job_id is not None else None,
            "reservation_bytes": self.reservation_bytes,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: bytes | str) -> TusSession:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise InvalidTusSessionError("session JSON must be UTF-8") from exc
        if not isinstance(raw, str):
            raise InvalidTusSessionError("session JSON must be bytes or text")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise InvalidTusSessionError("session JSON is invalid") from exc
        if not isinstance(payload, dict) or set(payload) != _SESSION_FIELDS:
            raise InvalidTusSessionError("session JSON fields do not match the closed schema")
        raw_parts = payload["parts"]
        if not isinstance(raw_parts, list):
            raise InvalidTusSessionError("parts must be a JSON array")
        parts: list[TusPart] = []
        try:
            for raw_part in raw_parts:
                if not isinstance(raw_part, dict) or set(raw_part) != _PART_FIELDS:
                    raise InvalidTusSessionError("part fields do not match the closed schema")
                parts.append(
                    TusPart(
                        part_number=_parse_safe_integer(
                            raw_part["part_number"],
                            "part_number",
                            minimum=1,
                            maximum=MAX_PART_NUMBER,
                        ),
                        etag=_validate_etag(raw_part["etag"]),
                    )
                )
            state = TusSessionState(payload["state"])
            return cls(
                upload_id=_parse_uuid(payload["upload_id"], "upload_id"),
                user_id=_parse_uuid(payload["user_id"], "user_id"),
                knowledge_base_id=_parse_uuid(payload["knowledge_base_id"], "knowledge_base_id"),
                filename=_validate_text(payload["filename"], "filename", max_length=255, forbid_path=True),
                content_type=_validate_text(payload["content_type"], "content_type", max_length=255),
                total_length=_parse_safe_integer(payload["total_length"], "total_length", maximum=MAX_UPLOAD_BYTES),
                offset=_parse_safe_integer(payload["offset"], "offset", maximum=MAX_UPLOAD_BYTES),
                s3_key=_validate_text(payload["s3_key"], "s3_key", max_length=1_024),
                multipart_upload_id=_validate_text(
                    payload["multipart_upload_id"], "multipart_upload_id", max_length=1_024
                ),
                parts=tuple(parts),
                created_at=_microseconds_to_datetime(payload["created_at"], "created_at"),
                updated_at=_microseconds_to_datetime(payload["updated_at"], "updated_at"),
                state=state,
                document_id=_parse_uuid(payload["document_id"], "document_id", optional=True),
                job_id=_parse_uuid(payload["job_id"], "job_id", optional=True),
                reservation_bytes=_parse_safe_integer(
                    payload["reservation_bytes"], "reservation_bytes", maximum=MAX_UPLOAD_BYTES
                ),
            )
        except InvalidTusSessionError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidTusSessionError(str(exc)) from exc


def session_key(upload_id: UUID) -> str:
    return f"tus:session:{_require_uuid(upload_id, 'upload_id')}"


def lock_key(upload_id: UUID) -> str:
    return f"tus:lock:{_require_uuid(upload_id, 'upload_id')}"


def reservation_key(user_id: UUID, upload_id: UUID) -> str:
    return f"tus:reservation:{_require_uuid(user_id, 'user_id')}:{_require_uuid(upload_id, 'upload_id')}"


_LUA_RECORD_VALIDATOR = r"""
local MAX_SAFE = 9007199254740991
local MAX_UPLOAD = 9999999999999
local function count(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  return n
end
local function safe_integer(value, minimum, maximum)
  return type(value) == 'number' and value == math.floor(value) and value >= minimum and value <= maximum
end
local function nonempty(value, maximum)
  return type(value) == 'string' and #value > 0 and #value <= maximum and not string.find(value, '%c')
end
local function decimal_microseconds(value)
  if type(value) ~= 'string' or #value == 0 or #value > 16 or string.find(value, '[^0-9]') then return nil end
  if #value > 1 and string.sub(value, 1, 1) == '0' then return nil end
  local number = tonumber(value)
  if not safe_integer(number, 0, MAX_SAFE) then return nil end
  return number
end
local function next_timestamp(previous)
  local now = redis.call('TIME')
  local timestamp = now[1] .. string.format('%06d', tonumber(now[2]))
  local timestamp_number = tonumber(timestamp)
  local previous_number = decimal_microseconds(previous)
  if not previous_number then return nil end
  if timestamp_number <= previous_number then
    return string.format('%.0f', previous_number + 1)
  end
  return timestamp
end
local function uuid(value)
  if type(value) ~= 'string' or #value ~= 36
    or string.sub(value, 9, 9) ~= '-' or string.sub(value, 14, 14) ~= '-'
    or string.sub(value, 19, 19) ~= '-' or string.sub(value, 24, 24) ~= '-'
    or string.find(value, '[^0-9a-f%-]') then return false end
  local _, hyphens = string.gsub(value, '%-', '')
  return hyphens == 4
end
local function valid_record(session)
  if type(session) ~= 'table' or count(session) ~= 16 then return false end
  if not uuid(session.upload_id) or not uuid(session.user_id) or not uuid(session.knowledge_base_id) then return false end
  if not nonempty(session.filename, 255) or session.filename == '.' or session.filename == '..'
    or string.find(session.filename, '[/\\]') then return false end
  if not nonempty(session.content_type, 255) or not nonempty(session.s3_key, 1024)
    or not nonempty(session.multipart_upload_id, 1024) then return false end
  if not safe_integer(session.total_length, 0, MAX_UPLOAD)
    or not safe_integer(session.offset, 0, MAX_UPLOAD)
    or session.offset > session.total_length
    or not safe_integer(session.reservation_bytes, 0, MAX_UPLOAD) then return false end
  local created_at = decimal_microseconds(session.created_at)
  local updated_at = decimal_microseconds(session.updated_at)
  if not created_at or not updated_at or updated_at < created_at then return false end
  if type(session.parts) ~= 'table' then return false end
  local part_count = count(session.parts)
  if part_count ~= #session.parts or part_count > 10000 then return false end
  for index = 1, #session.parts do
    local part = session.parts[index]
    if type(part) ~= 'table' or count(part) ~= 2 or part.part_number ~= index
      or not safe_integer(part.part_number, 1, 10000)
      or not nonempty(part.etag, 512) or string.find(part.etag, '%s') then return false end
  end
  if session.state == 'uploading' then
    return session.document_id == cjson.null and session.job_id == cjson.null
  end
  if session.state == 'completed' then
    return session.offset == session.total_length and uuid(session.document_id) and uuid(session.job_id)
  end
  return false
end
"""


_CREATE_SESSION_LUA = (
    _LUA_RECORD_VALIDATOR
    + r"""
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
local decoded, session = pcall(cjson.decode, ARGV[1])
if not decoded or not valid_record(session) then return 2 end
local now = redis.call('TIME')
local timestamp = now[1] .. string.format('%06d', tonumber(now[2]))
local encoded, created_replacements = string.gsub(
  ARGV[1], '"created_at":"%d+"', '"created_at":"' .. timestamp .. '"', 1
)
local updated_replacements
encoded, updated_replacements = string.gsub(
  encoded, '"updated_at":"%d+"', '"updated_at":"' .. timestamp .. '"', 1
)
if created_replacements ~= 1 or updated_replacements ~= 1 then return 2 end
redis.call('SET', KEYS[1], encoded, 'EX', tonumber(ARGV[2]))
return 1
"""
)


_APPEND_PART_LUA = (
    _LUA_RECORD_VALIDATOR
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return {2, 0} end
local decoded, session = pcall(cjson.decode, raw)
if not decoded or not valid_record(session) then return {7, 0} end
if session.state == 'completed' then return {6, session.offset} end
if session.state ~= 'uploading' then return {3, session.offset} end
local expected = tonumber(ARGV[1])
local byte_count = tonumber(ARGV[2])
local part_number = tonumber(ARGV[3])
local ttl = tonumber(ARGV[5])
if session.offset ~= expected then return {1, session.offset} end
if byte_count <= 0 then return {8, session.offset} end
if session.offset + byte_count > session.total_length then return {4, session.offset} end
if part_number ~= #session.parts + 1 then return {5, session.offset} end
table.insert(session.parts, {part_number = part_number, etag = ARGV[4]})
session.offset = session.offset + byte_count
session.updated_at = next_timestamp(session.updated_at)
cjson.encode_number_precision(14)
redis.call('SET', KEYS[1], cjson.encode(session), 'EX', ttl)
return {0, session.offset}
"""
)


_MARK_COMPLETE_LUA = (
    _LUA_RECORD_VALIDATOR
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return {4, 0, '', ''} end
local decoded, session = pcall(cjson.decode, raw)
if not decoded or not valid_record(session) then return {7, 0, '', ''} end
local document_id = ARGV[2]
local job_id = ARGV[3]
local ttl = tonumber(ARGV[4])
if session.state == 'completed' then
  if session.document_id ~= document_id or session.job_id ~= job_id then
    return {2, session.offset, session.document_id, session.job_id}
  end
  local timestamp = next_timestamp(session.updated_at)
  local encoded, replacements = string.gsub(
    raw, '"updated_at":"%d+"', '"updated_at":"' .. timestamp .. '"', 1
  )
  if replacements ~= 1 then return {7, 0, '', ''} end
  redis.call('SET', KEYS[1], encoded, 'EX', ttl)
  return {1, session.offset, session.document_id, session.job_id}
end
if session.state ~= 'uploading' then return {5, session.offset, '', ''} end
if session.offset ~= tonumber(ARGV[1]) then return {3, session.offset, '', ''} end
if session.offset ~= session.total_length then return {6, session.offset, '', ''} end
session.state = 'completed'
session.document_id = document_id
session.job_id = job_id
local timestamp = next_timestamp(session.updated_at)
local encoded, state_replacements = string.gsub(raw, '"state":"uploading"', '"state":"completed"', 1)
local document_replacements
encoded, document_replacements = string.gsub(
  encoded, '"document_id":null', '"document_id":"' .. document_id .. '"', 1
)
local job_replacements
encoded, job_replacements = string.gsub(encoded, '"job_id":null', '"job_id":"' .. job_id .. '"', 1)
local updated_replacements
encoded, updated_replacements = string.gsub(
  encoded, '"updated_at":"%d+"', '"updated_at":"' .. timestamp .. '"', 1
)
if state_replacements ~= 1 or document_replacements ~= 1 or job_replacements ~= 1
  or updated_replacements ~= 1 then return {7, 0, '', ''} end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return {0, session.offset, session.document_id, session.job_id}
"""
)


_RENEW_LOCK_LUA = r"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
return 1
"""


_RELEASE_LOCK_LUA = r"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
return 1
"""


_RELEASE_RESERVATION_LUA = r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local decoded, reservation = pcall(cjson.decode, raw)
if not decoded or type(reservation) ~= 'table' then return 3 end
local count = 0
for _ in pairs(reservation) do count = count + 1 end
if count ~= 2 or type(reservation.bytes) ~= 'number' or reservation.bytes ~= math.floor(reservation.bytes)
  or reservation.bytes < 0 or reservation.bytes > 9999999999999 then return 3 end
if reservation.state == 'released' then return 2 end
if reservation.state ~= 'reserved' or redis.call('PTTL', KEYS[1]) <= 0 then return 3 end
reservation.state = 'released'
redis.call('SET', KEYS[1], cjson.encode(reservation), 'KEEPTTL')
return 1
"""


_APPEND_CODES = {
    0: AppendPartStatus.APPENDED,
    1: AppendPartStatus.OFFSET_MISMATCH,
    2: AppendPartStatus.NOT_FOUND,
    3: AppendPartStatus.NOT_UPLOADING,
    4: AppendPartStatus.LENGTH_EXCEEDED,
    5: AppendPartStatus.PART_OUT_OF_ORDER,
    6: AppendPartStatus.ALREADY_COMPLETED,
    7: AppendPartStatus.MALFORMED,
    8: AppendPartStatus.INVALID_BYTE_COUNT,
}
_COMPLETE_CODES = {
    0: CompleteStatus.COMPLETED,
    1: CompleteStatus.ALREADY_COMPLETED,
    2: CompleteStatus.IDENTIFIER_CONFLICT,
    3: CompleteStatus.OFFSET_MISMATCH,
    4: CompleteStatus.NOT_FOUND,
    5: CompleteStatus.NOT_UPLOADING,
    6: CompleteStatus.INCOMPLETE,
    7: CompleteStatus.MALFORMED,
}
_RESERVATION_RELEASE_CODES = {
    0: ReservationReleaseStatus.NOT_FOUND,
    1: ReservationReleaseStatus.RELEASED,
    2: ReservationReleaseStatus.ALREADY_RELEASED,
    3: ReservationReleaseStatus.MALFORMED,
}


def _protocol_integer(value: object, name: str, *, maximum: int = MAX_SAFE_INTEGER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise TusSessionProtocolError(f"Redis returned an invalid {name}")
    return value


def _protocol_sequence(value: object, expected_length: int) -> list[object]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise TusSessionProtocolError("Redis returned an invalid Lua response")
    return list(value)


def _decode_protocol_text(value: object, name: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TusSessionProtocolError(f"Redis returned invalid {name}") from exc
    if not isinstance(value, str):
        raise TusSessionProtocolError(f"Redis returned invalid {name}")
    return value


class TusSessionStore:
    """Operate one Redis key per call so every Lua script is cluster-slot safe."""

    def __init__(self, redis: RedisClient):
        self._redis = redis

    async def create(self, session: TusSession, ttl_seconds: int) -> SessionCreateStatus:
        if not isinstance(session, TusSession):
            raise ValueError("session must be a TusSession")
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(
            _CREATE_SESSION_LUA,
            1,
            session_key(session.upload_id),
            session.to_json(),
            str(ttl),
        )
        code = _protocol_integer(raw, "session create status", maximum=2)
        if code == 2:
            raise TusSessionProtocolError("Redis rejected a locally validated session record")
        return SessionCreateStatus.CREATED if code == 1 else SessionCreateStatus.ALREADY_EXISTS

    async def get(self, upload_id: UUID) -> TusSession | None:
        raw = await self._redis.get(session_key(upload_id))
        return None if raw is None else TusSession.from_json(raw)

    async def append_part(
        self,
        upload_id: UUID,
        expected_offset: int,
        byte_count: int,
        part_number: int,
        etag: str,
        ttl_seconds: int,
    ) -> AppendPartResult:
        expected = _require_safe_integer(expected_offset, "expected_offset", maximum=MAX_UPLOAD_BYTES)
        count = _require_safe_integer(byte_count, "byte_count", maximum=MAX_UPLOAD_BYTES)
        number = _require_safe_integer(part_number, "part_number", minimum=1, maximum=MAX_PART_NUMBER)
        validated_etag = _validate_etag(etag)
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(
            _APPEND_PART_LUA,
            1,
            session_key(upload_id),
            str(expected),
            str(count),
            str(number),
            validated_etag,
            str(ttl),
        )
        values = _protocol_sequence(raw, 2)
        code = _protocol_integer(values[0], "append status code", maximum=max(_APPEND_CODES))
        status = _APPEND_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown append status code")
        offset = _protocol_integer(values[1], "offset")
        return AppendPartResult(status=status, offset=offset)

    async def mark_complete(
        self,
        upload_id: UUID,
        expected_offset: int,
        document_id: UUID,
        job_id: UUID,
        ttl_seconds: int,
    ) -> CompleteResult:
        expected = _require_safe_integer(expected_offset, "expected_offset", maximum=MAX_UPLOAD_BYTES)
        document = _require_uuid(document_id, "document_id")
        job = _require_uuid(job_id, "job_id")
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(
            _MARK_COMPLETE_LUA,
            1,
            session_key(upload_id),
            str(expected),
            str(document),
            str(job),
            str(ttl),
        )
        values = _protocol_sequence(raw, 4)
        code = _protocol_integer(values[0], "completion status code", maximum=max(_COMPLETE_CODES))
        status = _COMPLETE_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown completion status code")
        offset = _protocol_integer(values[1], "offset")
        document_text = _decode_protocol_text(values[2], "document_id")
        job_text = _decode_protocol_text(values[3], "job_id")
        if status in {
            CompleteStatus.COMPLETED,
            CompleteStatus.ALREADY_COMPLETED,
            CompleteStatus.IDENTIFIER_CONFLICT,
        }:
            try:
                committed_document = UUID(document_text)
                committed_job = UUID(job_text)
            except ValueError as exc:
                raise TusSessionProtocolError("Redis returned invalid committed identifiers") from exc
        else:
            if document_text or job_text:
                raise TusSessionProtocolError("Redis returned unexpected committed identifiers")
            committed_document = None
            committed_job = None
        return CompleteResult(status, offset, committed_document, committed_job)

    async def acquire_lock(self, upload_id: UUID, ttl_seconds: int) -> LockAcquireResult:
        ttl = _require_ttl(ttl_seconds)
        token = secrets.token_urlsafe(32)
        acquired = await self._redis.set(lock_key(upload_id), token, nx=True, ex=ttl)
        if not acquired:
            return LockAcquireResult(LockAcquireStatus.CONTENDED, None)
        return LockAcquireResult(LockAcquireStatus.ACQUIRED, token)

    async def renew_lock(self, upload_id: UUID, token: str, ttl_seconds: int) -> LockMutationStatus:
        validated_token = _validate_text(token, "lock token", max_length=512)
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(_RENEW_LOCK_LUA, 1, lock_key(upload_id), validated_token, str(ttl))
        code = _protocol_integer(raw, "lock renewal status", maximum=1)
        return LockMutationStatus.RENEWED if code == 1 else LockMutationStatus.NOT_OWNER

    async def release_lock(self, upload_id: UUID, token: str) -> LockMutationStatus:
        validated_token = _validate_text(token, "lock token", max_length=512)
        raw = await self._redis.eval(_RELEASE_LOCK_LUA, 1, lock_key(upload_id), validated_token)
        code = _protocol_integer(raw, "lock release status", maximum=1)
        return LockMutationStatus.RELEASED if code == 1 else LockMutationStatus.NOT_OWNER

    async def create_reservation(
        self,
        user_id: UUID,
        upload_id: UUID,
        bytes_reserved: int,
        ttl_seconds: int,
    ) -> ReservationCreateStatus:
        """Create the reservation marker separately from the session key."""
        reserved = _require_safe_integer(bytes_reserved, "bytes_reserved", maximum=MAX_UPLOAD_BYTES)
        ttl = _require_ttl(ttl_seconds)
        payload = json.dumps({"bytes": reserved, "state": "reserved"}, sort_keys=True, separators=(",", ":"))
        created = await self._redis.set(
            reservation_key(user_id, upload_id),
            payload,
            nx=True,
            ex=ttl,
        )
        return ReservationCreateStatus.CREATED if created else ReservationCreateStatus.ALREADY_EXISTS

    async def release_reservation_once(self, user_id: UUID, upload_id: UUID) -> ReservationReleaseStatus:
        raw = await self._redis.eval(
            _RELEASE_RESERVATION_LUA,
            1,
            reservation_key(user_id, upload_id),
        )
        code = _protocol_integer(raw, "reservation release status", maximum=max(_RESERVATION_RELEASE_CODES))
        status = _RESERVATION_RELEASE_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown reservation release status code")
        return status
