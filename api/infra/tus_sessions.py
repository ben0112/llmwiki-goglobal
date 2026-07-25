"""Strict Redis-backed TUS session, lock, and release-marker primitives."""

from __future__ import annotations

import json
import re
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
)
_PART_FIELDS = frozenset({"part_number", "etag"})
_SESSION_KEY_PATTERN = re.compile(r"tus:session:\{([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\}\Z")


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
    CLEANUP_REQUIRED = "cleanup_required"
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
    LOCK_LOST = "lock_lost"


class CompleteStatus(StrEnum):
    COMPLETED = "completed"
    ALREADY_COMPLETED = "already_completed"
    IDENTIFIER_CONFLICT = "identifier_conflict"
    OFFSET_MISMATCH = "offset_mismatch"
    NOT_FOUND = "not_found"
    NOT_UPLOADING = "not_uploading"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"
    LOCK_LOST = "lock_lost"


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
    NOT_OWNER = "not_owner"


class TusReservationState(StrEnum):
    RESERVED = "reserved"
    RELEASED = "released"


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


@dataclass(frozen=True, slots=True)
class TusQuotaReservation:
    bytes: int
    owner_token: str
    state: TusReservationState

    def __post_init__(self) -> None:
        _require_safe_integer(self.bytes, "bytes", minimum=1, maximum=MAX_UPLOAD_BYTES)
        _validate_owner_token(self.owner_token)
        if not isinstance(self.state, TusReservationState):
            raise ValueError("state must be a TusReservationState")

    def to_json(self) -> str:
        return json.dumps(
            {"bytes": self.bytes, "owner": self.owner_token, "state": self.state.value},
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: bytes | str) -> TusQuotaReservation:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise InvalidTusSessionError("reservation marker must be UTF-8") from exc
        if not isinstance(raw, str):
            raise InvalidTusSessionError("reservation marker must be bytes or text")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != {"bytes", "owner", "state"}:
                raise InvalidTusSessionError("reservation marker fields do not match the closed schema")
            marker = cls(
                _parse_safe_integer(payload["bytes"], "bytes", minimum=1, maximum=MAX_UPLOAD_BYTES),
                _validate_owner_token(payload["owner"]),
                TusReservationState(payload["state"]),
            )
        except InvalidTusSessionError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise InvalidTusSessionError(str(exc)) from exc
        if marker.to_json() != raw:
            raise InvalidTusSessionError("reservation marker must use canonical wire encoding")
        return marker


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


def _validate_text(
    value: object,
    name: str,
    *,
    max_bytes: int,
    forbid_path: bool = False,
    allow_quote: bool = False,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} must be at most {max_bytes} UTF-8 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    if "\\" in value or ('"' in value and not allow_quote):
        raise ValueError(f"{name} must not contain JSON metacharacters")
    if forbid_path and (value in {".", ".."} or "/" in value):
        raise ValueError(f"{name} must be a sanitized basename")
    return value


def _validate_ascii_opaque(value: object, name: str, *, max_bytes: int) -> str:
    text = _validate_text(value, name, max_bytes=max_bytes)
    if any(not 33 <= ord(character) <= 126 for character in text):
        raise ValueError(f"{name} must contain only non-whitespace printable ASCII")
    return text


def _validate_etag(value: object) -> str:
    return _validate_ascii_opaque(value, "etag", max_bytes=512)


def _validate_owner_token(value: object) -> str:
    token = _validate_ascii_opaque(value, "owner token", max_bytes=32)
    if len(token) != 32 or any(
        not (character.isascii() and (character.isalnum() or character in "_-")) for character in token
    ):
        raise ValueError("owner token must use the generated URL-safe format")
    return token


def _validate_path(value: object) -> str:
    path = _validate_text(value, "path", max_bytes=1_024, allow_quote=False)
    if not path.startswith("/") or not path.endswith("/") or "//" in path:
        raise ValueError("path must use normalized absolute directory form")
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ValueError("path must not contain traversal segments")
    return path


def validate_tus_upload_metadata(filename: object, path: object) -> tuple[str, str]:
    """Validate the client-derived metadata stored in the closed session schema."""
    return (
        _validate_text(filename, "filename", max_bytes=255, forbid_path=True, allow_quote=True),
        _validate_path(path),
    )


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
    path: str
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
    object_completed: bool

    def __post_init__(self) -> None:  # noqa: C901 -- this is the closed-schema invariant boundary
        _require_uuid(self.upload_id, "upload_id")
        _require_uuid(self.user_id, "user_id")
        _require_uuid(self.knowledge_base_id, "knowledge_base_id")
        validate_tus_upload_metadata(self.filename, self.path)
        _validate_ascii_opaque(self.content_type, "content_type", max_bytes=255)
        total_length = _require_safe_integer(self.total_length, "total_length", maximum=MAX_UPLOAD_BYTES)
        offset = _require_safe_integer(self.offset, "offset", maximum=MAX_UPLOAD_BYTES)
        if offset > total_length:
            raise ValueError("offset must not exceed total_length")
        _validate_ascii_opaque(self.s3_key, "s3_key", max_bytes=1_024)
        _validate_ascii_opaque(self.multipart_upload_id, "multipart_upload_id", max_bytes=1_024)
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
        if self.reservation_bytes != self.total_length:
            raise ValueError("reservation_bytes must equal total_length")
        if not isinstance(self.object_completed, bool):
            raise ValueError("object_completed must be a boolean")
        if self.object_completed and self.offset != self.total_length:
            raise ValueError("object_completed requires the full offset")
        if self.state is TusSessionState.UPLOADING:
            if self.document_id is not None or self.job_id is not None:
                raise ValueError("uploading sessions cannot contain document or job IDs")
        elif self.state is TusSessionState.CLEANUP_REQUIRED:
            if self.document_id is not None or self.job_id is not None:
                raise ValueError("cleanup-required sessions cannot contain document or job IDs")
        elif (
            self.offset != self.total_length
            or not self.object_completed
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
            "path": self.path,
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
            "object_completed": self.object_completed,
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
            session = cls(
                upload_id=_parse_uuid(payload["upload_id"], "upload_id"),
                user_id=_parse_uuid(payload["user_id"], "user_id"),
                knowledge_base_id=_parse_uuid(payload["knowledge_base_id"], "knowledge_base_id"),
                filename=_validate_text(
                    payload["filename"],
                    "filename",
                    max_bytes=255,
                    forbid_path=True,
                    allow_quote=True,
                ),
                path=_validate_path(payload["path"]),
                content_type=_validate_ascii_opaque(payload["content_type"], "content_type", max_bytes=255),
                total_length=_parse_safe_integer(payload["total_length"], "total_length", maximum=MAX_UPLOAD_BYTES),
                offset=_parse_safe_integer(payload["offset"], "offset", maximum=MAX_UPLOAD_BYTES),
                s3_key=_validate_ascii_opaque(payload["s3_key"], "s3_key", max_bytes=1_024),
                multipart_upload_id=_validate_ascii_opaque(
                    payload["multipart_upload_id"], "multipart_upload_id", max_bytes=1_024
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
                object_completed=payload["object_completed"],
            )
            if raw != session.to_json():
                raise InvalidTusSessionError("session JSON must use the canonical wire encoding")
            return session
        except InvalidTusSessionError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidTusSessionError(str(exc)) from exc


def session_key(upload_id: UUID) -> str:
    return f"tus:session:{{{_require_uuid(upload_id, 'upload_id')}}}"


def lock_key(upload_id: UUID) -> str:
    return f"tus:lock:{{{_require_uuid(upload_id, 'upload_id')}}}"


def reservation_key(user_id: UUID, upload_id: UUID) -> str:
    return f"tus:reservation:{{{_require_uuid(user_id, 'user_id')}}}:{{{_require_uuid(upload_id, 'upload_id')}}}"


_LUA_RECORD_CODEC = r"""
local MAX_SAFE = 9007199254740991
local MAX_UPLOAD = 9999999999999
local MAX_TTL = 2147483647

local function count(table_value)
  local size = 0
  for _ in pairs(table_value) do size = size + 1 end
  return size
end

local function safe_integer(value, minimum, maximum)
  return type(value) == 'number' and value == math.floor(value) and value >= minimum and value <= maximum
end

local function canonical_decimal(value, minimum, maximum)
  if type(value) ~= 'string' or #value == 0 or #value > 16 or string.find(value, '[^0-9]') then return nil end
  if #value > 1 and string.sub(value, 1, 1) == '0' then return nil end
  local number = tonumber(value)
  if not safe_integer(number, minimum, maximum) then return nil end
  if string.format('%.0f', number) ~= value then return nil end
  return number
end

local function valid_utf8(value, maximum_bytes)
  if type(value) ~= 'string' or #value == 0 or #value > maximum_bytes then return false end
  local index = 1
  while index <= #value do
    local first = string.byte(value, index)
    if first <= 0x7f then
      index = index + 1
    elseif first >= 0xc2 and first <= 0xdf then
      local second = string.byte(value, index + 1)
      if not second or second < 0x80 or second > 0xbf then return false end
      index = index + 2
    elseif first >= 0xe0 and first <= 0xef then
      local second = string.byte(value, index + 1)
      local third = string.byte(value, index + 2)
      if not second or not third or third < 0x80 or third > 0xbf then return false end
      if first == 0xe0 and (second < 0xa0 or second > 0xbf) then return false end
      if first == 0xed and (second < 0x80 or second > 0x9f) then return false end
      if first ~= 0xe0 and first ~= 0xed and (second < 0x80 or second > 0xbf) then return false end
      index = index + 3
    elseif first >= 0xf0 and first <= 0xf4 then
      local second = string.byte(value, index + 1)
      local third = string.byte(value, index + 2)
      local fourth = string.byte(value, index + 3)
      if not second or not third or not fourth
        or third < 0x80 or third > 0xbf or fourth < 0x80 or fourth > 0xbf then return false end
      if first == 0xf0 and (second < 0x90 or second > 0xbf) then return false end
      if first == 0xf4 and (second < 0x80 or second > 0x8f) then return false end
      if first ~= 0xf0 and first ~= 0xf4 and (second < 0x80 or second > 0xbf) then return false end
      index = index + 4
    else
      return false
    end
  end
  return true
end

local function valid_filename(value)
  if not valid_utf8(value, 255) or value == '.' or value == '..' then return false end
  for index = 1, #value do
    local byte = string.byte(value, index)
    if byte < 32 or byte == 47 or byte == 92 or byte == 127 then return false end
  end
  return true
end

local function escape_filename(value)
  if not valid_filename(value) then return nil end
  local encoded = {}
  for index = 1, #value do
    if string.byte(value, index) == 34 then
      encoded[#encoded + 1] = '\\"'
    else
      encoded[#encoded + 1] = string.sub(value, index, index)
    end
  end
  return '"' .. table.concat(encoded) .. '"'
end

local function valid_path(value)
  if not valid_utf8(value, 1024) or string.sub(value, 1, 1) ~= '/'
    or string.sub(value, -1) ~= '/' or string.find(value, '//', 1, true) then return false end
  for index = 1, #value do
    local byte = string.byte(value, index)
    if byte < 32 or byte == 34 or byte == 92 or byte == 127 then return false end
  end
  for segment in string.gmatch(value, '[^/]+') do
    if segment == '.' or segment == '..' then return false end
  end
  return true
end

local function ascii_opaque(value, maximum_bytes)
  if type(value) ~= 'string' or #value == 0 or #value > maximum_bytes then return false end
  for index = 1, #value do
    local byte = string.byte(value, index)
    if byte < 33 or byte > 126 or byte == 34 or byte == 92 then return false end
  end
  return true
end

local function uuid(value)
  if type(value) ~= 'string' or #value ~= 36
    or string.sub(value, 9, 9) ~= '-' or string.sub(value, 14, 14) ~= '-'
    or string.sub(value, 19, 19) ~= '-' or string.sub(value, 24, 24) ~= '-'
    or string.find(value, '[^0-9a-f%-]') then return false end
  local _, hyphens = string.gsub(value, '%-', '')
  return hyphens == 4
end

local function parts_json(parts)
  if type(parts) ~= 'table' then return nil end
  local part_count = count(parts)
  if part_count ~= #parts or part_count > 10000 then return nil end
  local encoded = {}
  for index = 1, #parts do
    local part = parts[index]
    if type(part) ~= 'table' or count(part) ~= 2 or part.part_number ~= index
      or not safe_integer(part.part_number, 1, 10000) or not ascii_opaque(part.etag, 512) then return nil end
    encoded[index] = '{"etag":"' .. part.etag .. '","part_number":' .. string.format('%.0f', part.part_number) .. '}'
  end
  return '[' .. table.concat(encoded, ',') .. ']'
end

local function canonical_record(session)
  if type(session) ~= 'table' or count(session) ~= 18 then return nil end
  local encoded_filename = escape_filename(session.filename)
  if not uuid(session.upload_id) or not uuid(session.user_id) or not uuid(session.knowledge_base_id)
    or not encoded_filename or not valid_path(session.path) or not ascii_opaque(session.content_type, 255)
    or not ascii_opaque(session.s3_key, 1024) or not ascii_opaque(session.multipart_upload_id, 1024)
    or not safe_integer(session.total_length, 0, MAX_UPLOAD)
    or not safe_integer(session.offset, 0, MAX_UPLOAD) or session.offset > session.total_length
    or not safe_integer(session.reservation_bytes, 0, MAX_UPLOAD)
    or session.reservation_bytes ~= session.total_length or type(session.object_completed) ~= 'boolean'
    or (session.object_completed and session.offset ~= session.total_length) then return nil end
  local created_at = canonical_decimal(session.created_at, 0, MAX_SAFE)
  local updated_at = canonical_decimal(session.updated_at, 0, MAX_SAFE)
  if not created_at or not updated_at or updated_at < created_at then return nil end
  local encoded_parts = parts_json(session.parts)
  if not encoded_parts then return nil end
  local document_json
  local job_json
  if session.state == 'uploading' or session.state == 'cleanup_required' then
    if session.document_id ~= cjson.null or session.job_id ~= cjson.null then return nil end
    document_json = 'null'
    job_json = 'null'
  elseif session.state == 'completed' then
    if session.offset ~= session.total_length or not session.object_completed
      or not uuid(session.document_id) or not uuid(session.job_id) then return nil end
    document_json = '"' .. session.document_id .. '"'
    job_json = '"' .. session.job_id .. '"'
  else
    return nil
  end
  return '{"content_type":"' .. session.content_type
    .. '","created_at":"' .. session.created_at
    .. '","document_id":' .. document_json
    .. ',"filename":' .. encoded_filename
    .. ',"job_id":' .. job_json
    .. ',"knowledge_base_id":"' .. session.knowledge_base_id
    .. '","multipart_upload_id":"' .. session.multipart_upload_id
    .. '","object_completed":' .. (session.object_completed and 'true' or 'false')
    .. ',"offset":' .. string.format('%.0f', session.offset)
    .. ',"parts":' .. encoded_parts
    .. ',"path":"' .. session.path
    .. '","reservation_bytes":' .. string.format('%.0f', session.reservation_bytes)
    .. ',"s3_key":"' .. session.s3_key
    .. '","state":"' .. session.state
    .. '","total_length":' .. string.format('%.0f', session.total_length)
    .. ',"updated_at":"' .. session.updated_at
    .. '","upload_id":"' .. session.upload_id
    .. '","user_id":"' .. session.user_id .. '"}'
end

local function decode_canonical(raw)
  local decoded, session = pcall(cjson.decode, raw)
  if not decoded then return nil end
  local encoded = canonical_record(session)
  if not encoded or encoded ~= raw then return nil end
  return session
end

local function redis_timestamp()
  local now = redis.call('TIME')
  local timestamp = now[1] .. string.format('%06d', tonumber(now[2]))
  if not canonical_decimal(timestamp, 0, MAX_SAFE) then return nil end
  return timestamp
end

local function next_timestamp(previous)
  local previous_number = canonical_decimal(previous, 0, MAX_SAFE)
  local timestamp = redis_timestamp()
  if not previous_number or not timestamp then return nil end
  local timestamp_number = tonumber(timestamp)
  if timestamp_number > previous_number then return timestamp end
  if previous_number >= MAX_SAFE then return nil end
  return string.format('%.0f', previous_number + 1)
end

local function canonical_argument(value, minimum, maximum)
  return canonical_decimal(value, minimum, maximum)
end
"""


_CREATE_SESSION_LUA = (
    _LUA_RECORD_CODEC
    + r"""
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
local session = decode_canonical(ARGV[1])
local ttl = canonical_argument(ARGV[2], 1, MAX_TTL)
if not session or not ttl then return 2 end
local timestamp = redis_timestamp()
if not timestamp then return 2 end
session.created_at = timestamp
session.updated_at = timestamp
local encoded = canonical_record(session)
if not encoded then return 2 end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return 1
"""
)


_APPEND_PART_LUA = (
    _LUA_RECORD_CODEC
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return {2, 0} end
local session = decode_canonical(raw)
if not session then return {7, 0} end
if redis.call('GET', KEYS[2]) ~= ARGV[6] then return {9, session.offset} end
if session.state == 'completed' then return {6, session.offset} end
if session.state ~= 'uploading' then return {3, session.offset} end
local expected = canonical_argument(ARGV[1], 0, MAX_UPLOAD)
local byte_count = canonical_argument(ARGV[2], 0, MAX_UPLOAD)
local part_number = canonical_argument(ARGV[3], 1, 10000)
local ttl = canonical_argument(ARGV[5], 1, MAX_TTL)
if not expected or not byte_count or not part_number or not ttl or not ascii_opaque(ARGV[4], 512) then
  return {7, session.offset}
end
if session.offset ~= expected then return {1, session.offset} end
if byte_count <= 0 then return {8, session.offset} end
if session.offset + byte_count > session.total_length then return {4, session.offset} end
if part_number ~= #session.parts + 1 then return {5, session.offset} end
local timestamp = next_timestamp(session.updated_at)
if not timestamp then return {7, session.offset} end
table.insert(session.parts, {part_number = part_number, etag = ARGV[4]})
session.offset = session.offset + byte_count
session.updated_at = timestamp
local encoded = canonical_record(session)
if not encoded then return {7, session.offset - byte_count} end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return {0, session.offset}
"""
)


_MARK_COMPLETE_LUA = (
    _LUA_RECORD_CODEC
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return {4, 0, '', ''} end
local session = decode_canonical(raw)
if not session then return {7, 0, '', ''} end
if redis.call('GET', KEYS[2]) ~= ARGV[5] then return {8, session.offset, '', ''} end
local expected = canonical_argument(ARGV[1], 0, MAX_UPLOAD)
local ttl = canonical_argument(ARGV[4], 1, MAX_TTL)
local document_id = ARGV[2]
local job_id = ARGV[3]
if not expected or not ttl or not uuid(document_id) or not uuid(job_id) then return {7, session.offset, '', ''} end
if session.state == 'completed' then
  if session.document_id ~= document_id or session.job_id ~= job_id then
    return {2, session.offset, session.document_id, session.job_id}
  end
  local timestamp = next_timestamp(session.updated_at)
  if not timestamp then return {7, session.offset, '', ''} end
  session.updated_at = timestamp
  local encoded = canonical_record(session)
  if not encoded then return {7, session.offset, '', ''} end
  redis.call('SET', KEYS[1], encoded, 'EX', ttl)
  return {1, session.offset, session.document_id, session.job_id}
end
if session.state ~= 'uploading' then return {5, session.offset, '', ''} end
if session.offset ~= expected then return {3, session.offset, '', ''} end
if session.offset ~= session.total_length then return {6, session.offset, '', ''} end
local timestamp = next_timestamp(session.updated_at)
if not timestamp then return {7, session.offset, '', ''} end
session.state = 'completed'
session.object_completed = true
session.document_id = document_id
session.job_id = job_id
session.updated_at = timestamp
local encoded = canonical_record(session)
if not encoded then return {7, session.offset, '', ''} end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return {0, session.offset, session.document_id, session.job_id}
"""
)


_MARK_OBJECT_COMPLETED_LUA = (
    _LUA_RECORD_CODEC
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local session = decode_canonical(raw)
if not session then return 4 end
if redis.call('GET', KEYS[2]) ~= ARGV[3] then return 5 end
local expected = canonical_argument(ARGV[1], 0, MAX_UPLOAD)
local ttl = canonical_argument(ARGV[2], 1, MAX_TTL)
if not expected or not ttl then return 4 end
if session.offset ~= expected then return 6 end
if session.state ~= 'uploading' or session.offset ~= session.total_length then return 3 end
if session.object_completed then return 2 end
local timestamp = next_timestamp(session.updated_at)
if not timestamp then return 4 end
session.object_completed = true
session.updated_at = timestamp
local encoded = canonical_record(session)
if not encoded then return 4 end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return 1
"""
)


_MARK_CLEANUP_REQUIRED_LUA = (
    _LUA_RECORD_CODEC
    + r"""
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local session = decode_canonical(raw)
if not session then return 4 end
if redis.call('GET', KEYS[2]) ~= ARGV[3] then return 5 end
local expected = canonical_argument(ARGV[1], 0, MAX_UPLOAD)
local ttl = canonical_argument(ARGV[2], 1, MAX_TTL)
if not expected or not ttl then return 4 end
if session.offset ~= expected then return 6 end
if session.state == 'completed' then return 3 end
if session.state == 'cleanup_required' then return 2 end
if session.state ~= 'uploading' then return 3 end
local timestamp = next_timestamp(session.updated_at)
if not timestamp then return 4 end
session.state = 'cleanup_required'
session.updated_at = timestamp
local encoded = canonical_record(session)
if not encoded then return 4 end
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return 1
"""
)


_DELETE_LOCKED_LUA = r"""
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


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
if redis.call('PTTL', KEYS[1]) <= 0 then return 3 end
local decoded, reservation = pcall(cjson.decode, raw)
if not decoded or type(reservation) ~= 'table' then return 3 end
local count = 0
for _ in pairs(reservation) do count = count + 1 end
if count ~= 3 or type(reservation.bytes) ~= 'number' or reservation.bytes ~= math.floor(reservation.bytes)
  or reservation.bytes < 1 or reservation.bytes > 9999999999999 then return 3 end
if type(reservation.owner) ~= 'string' or #reservation.owner ~= 32
  or string.find(reservation.owner, '[^A-Za-z0-9_%-]') then return 3 end
if reservation.state ~= 'reserved' and reservation.state ~= 'released' then return 3 end
local canonical = '{"bytes":' .. string.format('%.0f', reservation.bytes)
  .. ',"owner":"' .. reservation.owner .. '","state":"' .. reservation.state .. '"}'
if canonical ~= raw then return 3 end
if reservation.owner ~= ARGV[1] then return 4 end
if reservation.state == 'released' then return 2 end
local released = '{"bytes":' .. string.format('%.0f', reservation.bytes)
  .. ',"owner":"' .. reservation.owner .. '","state":"released"}'
redis.call('SET', KEYS[1], released, 'KEEPTTL')
return 1
"""


_RENEW_RESERVATION_LUA = r"""
local raw = redis.call('GET', KEYS[1])
if not raw or redis.call('PTTL', KEYS[1]) <= 0 then return 0 end
local decoded, reservation = pcall(cjson.decode, raw)
if not decoded or type(reservation) ~= 'table' then return 0 end
local count = 0
for _ in pairs(reservation) do count = count + 1 end
if count ~= 3 or type(reservation.bytes) ~= 'number' or reservation.bytes ~= math.floor(reservation.bytes)
  or reservation.bytes < 1 or reservation.bytes > 9999999999999
  or type(reservation.owner) ~= 'string' or #reservation.owner ~= 32
  or string.find(reservation.owner, '[^A-Za-z0-9_%-]')
  or reservation.state ~= 'reserved' then return 0 end
local canonical = '{"bytes":' .. string.format('%.0f', reservation.bytes)
  .. ',"owner":"' .. reservation.owner .. '","state":"reserved"}'
if canonical ~= raw or reservation.owner ~= ARGV[1] then return 0 end
local ttl = tonumber(ARGV[2])
if not ttl or ttl < 1 or ttl > 2147483647 or ttl ~= math.floor(ttl) then return 0 end
redis.call('EXPIRE', KEYS[1], ttl)
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
    9: AppendPartStatus.LOCK_LOST,
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
    8: CompleteStatus.LOCK_LOST,
}
_RESERVATION_RELEASE_CODES = {
    0: ReservationReleaseStatus.NOT_FOUND,
    1: ReservationReleaseStatus.RELEASED,
    2: ReservationReleaseStatus.ALREADY_RELEASED,
    3: ReservationReleaseStatus.MALFORMED,
    4: ReservationReleaseStatus.NOT_OWNER,
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
    """Use one key or a hash-tagged same-slot key pair in every Lua call."""

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

    async def iter_sessions(self):
        """Scan canonical session keys without a cross-slot Lua operation."""
        scan_iter = getattr(self._redis, "scan_iter", None)
        if not callable(scan_iter):
            raise TusSessionProtocolError("Redis client does not support bounded session scans")
        async for raw_key in scan_iter(match="tus:session:{*}", count=100):
            key = _decode_protocol_text(raw_key, "session key")
            match = _SESSION_KEY_PATTERN.fullmatch(key)
            if match is None:
                continue
            upload_id = UUID(match.group(1))
            session = await self.get(upload_id)
            if session is not None:
                yield session

    async def append_part(
        self,
        upload_id: UUID,
        expected_offset: int,
        byte_count: int,
        part_number: int,
        etag: str,
        ttl_seconds: int,
        *,
        lock_token: str,
    ) -> AppendPartResult:
        expected = _require_safe_integer(expected_offset, "expected_offset", maximum=MAX_UPLOAD_BYTES)
        count = _require_safe_integer(byte_count, "byte_count", maximum=MAX_UPLOAD_BYTES)
        number = _require_safe_integer(part_number, "part_number", minimum=1, maximum=MAX_PART_NUMBER)
        validated_etag = _validate_etag(etag)
        ttl = _require_ttl(ttl_seconds)
        token = _validate_ascii_opaque(lock_token, "lock token", max_bytes=512)
        raw = await self._redis.eval(
            _APPEND_PART_LUA,
            2,
            session_key(upload_id),
            lock_key(upload_id),
            str(expected),
            str(count),
            str(number),
            validated_etag,
            str(ttl),
            token,
        )
        values = _protocol_sequence(raw, 2)
        code = _protocol_integer(values[0], "append status code", maximum=max(_APPEND_CODES))
        status = _APPEND_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown append status code")
        offset = _protocol_integer(values[1], "offset", maximum=MAX_UPLOAD_BYTES)
        return AppendPartResult(status=status, offset=offset)

    async def mark_complete(
        self,
        upload_id: UUID,
        expected_offset: int,
        document_id: UUID,
        job_id: UUID,
        ttl_seconds: int,
        *,
        lock_token: str,
    ) -> CompleteResult:
        expected = _require_safe_integer(expected_offset, "expected_offset", maximum=MAX_UPLOAD_BYTES)
        document = _require_uuid(document_id, "document_id")
        job = _require_uuid(job_id, "job_id")
        ttl = _require_ttl(ttl_seconds)
        token = _validate_ascii_opaque(lock_token, "lock token", max_bytes=512)
        raw = await self._redis.eval(
            _MARK_COMPLETE_LUA,
            2,
            session_key(upload_id),
            lock_key(upload_id),
            str(expected),
            str(document),
            str(job),
            str(ttl),
            token,
        )
        values = _protocol_sequence(raw, 4)
        code = _protocol_integer(values[0], "completion status code", maximum=max(_COMPLETE_CODES))
        status = _COMPLETE_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown completion status code")
        offset = _protocol_integer(values[1], "offset", maximum=MAX_UPLOAD_BYTES)
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
            if str(committed_document) != document_text or str(committed_job) != job_text:
                raise TusSessionProtocolError("Redis returned noncanonical committed identifiers")
        else:
            if document_text or job_text:
                raise TusSessionProtocolError("Redis returned unexpected committed identifiers")
            committed_document = None
            committed_job = None
        return CompleteResult(status, offset, committed_document, committed_job)

    async def mark_object_completed(
        self,
        upload_id: UUID,
        expected_offset: int,
        ttl_seconds: int,
        *,
        lock_token: str,
    ) -> bool:
        return await self._phase_mutation(
            _MARK_OBJECT_COMPLETED_LUA,
            upload_id,
            expected_offset,
            ttl_seconds,
            lock_token,
        )

    async def mark_cleanup_required(
        self,
        upload_id: UUID,
        expected_offset: int,
        ttl_seconds: int,
        *,
        lock_token: str,
    ) -> bool:
        return await self._phase_mutation(
            _MARK_CLEANUP_REQUIRED_LUA,
            upload_id,
            expected_offset,
            ttl_seconds,
            lock_token,
        )

    async def _phase_mutation(
        self,
        script: str,
        upload_id: UUID,
        expected_offset: int,
        ttl_seconds: int,
        lock_token: str,
    ) -> bool:
        expected = _require_safe_integer(expected_offset, "expected_offset", maximum=MAX_UPLOAD_BYTES)
        ttl = _require_ttl(ttl_seconds)
        token = _validate_ascii_opaque(lock_token, "lock token", max_bytes=512)
        raw = await self._redis.eval(
            script,
            2,
            session_key(upload_id),
            lock_key(upload_id),
            str(expected),
            str(ttl),
            token,
        )
        code = _protocol_integer(raw, "session phase status", maximum=6)
        if code in {1, 2}:
            return True
        if code in {0, 3, 5, 6}:
            return False
        raise TusSessionProtocolError("Redis rejected a session phase mutation")

    async def delete_locked(self, upload_id: UUID, *, lock_token: str) -> bool:
        token = _validate_ascii_opaque(lock_token, "lock token", max_bytes=512)
        raw = await self._redis.eval(
            _DELETE_LOCKED_LUA,
            2,
            session_key(upload_id),
            lock_key(upload_id),
            token,
        )
        return _protocol_integer(raw, "session delete status", maximum=1) == 1

    async def acquire_lock(self, upload_id: UUID, ttl_seconds: int) -> LockAcquireResult:
        ttl = _require_ttl(ttl_seconds)
        token = secrets.token_urlsafe(32)
        acquired = await self._redis.set(lock_key(upload_id), token, nx=True, ex=ttl)
        if not acquired:
            return LockAcquireResult(LockAcquireStatus.CONTENDED, None)
        return LockAcquireResult(LockAcquireStatus.ACQUIRED, token)

    async def renew_lock(self, upload_id: UUID, token: str, ttl_seconds: int) -> LockMutationStatus:
        validated_token = _validate_ascii_opaque(token, "lock token", max_bytes=512)
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(_RENEW_LOCK_LUA, 1, lock_key(upload_id), validated_token, str(ttl))
        code = _protocol_integer(raw, "lock renewal status", maximum=1)
        return LockMutationStatus.RENEWED if code == 1 else LockMutationStatus.NOT_OWNER

    async def release_lock(self, upload_id: UUID, token: str) -> LockMutationStatus:
        validated_token = _validate_ascii_opaque(token, "lock token", max_bytes=512)
        raw = await self._redis.eval(_RELEASE_LOCK_LUA, 1, lock_key(upload_id), validated_token)
        code = _protocol_integer(raw, "lock release status", maximum=1)
        return LockMutationStatus.RELEASED if code == 1 else LockMutationStatus.NOT_OWNER

    async def create_reservation(
        self,
        user_id: UUID,
        upload_id: UUID,
        bytes_reserved: int,
        *,
        owner_token: str,
        ttl_seconds: int,
    ) -> ReservationCreateStatus:
        """Create the reservation marker separately from the session key."""
        reserved = _require_safe_integer(bytes_reserved, "bytes_reserved", minimum=1, maximum=MAX_UPLOAD_BYTES)
        owner = _validate_owner_token(owner_token)
        ttl = _require_ttl(ttl_seconds)
        payload = TusQuotaReservation(reserved, owner, TusReservationState.RESERVED).to_json()
        created = await self._redis.set(
            reservation_key(user_id, upload_id),
            payload,
            nx=True,
            ex=ttl,
        )
        return ReservationCreateStatus.CREATED if created else ReservationCreateStatus.ALREADY_EXISTS

    async def get_reservation(self, user_id: UUID, upload_id: UUID) -> TusQuotaReservation | None:
        raw = await self._redis.get(reservation_key(user_id, upload_id))
        return None if raw is None else TusQuotaReservation.from_json(raw)

    async def renew_reservation(
        self,
        user_id: UUID,
        upload_id: UUID,
        owner_token: str,
        *,
        ttl_seconds: int,
    ) -> LockMutationStatus:
        owner = _validate_owner_token(owner_token)
        ttl = _require_ttl(ttl_seconds)
        raw = await self._redis.eval(
            _RENEW_RESERVATION_LUA,
            1,
            reservation_key(user_id, upload_id),
            owner,
            str(ttl),
        )
        return (
            LockMutationStatus.RENEWED
            if _protocol_integer(raw, "reservation renewal status", maximum=1)
            else LockMutationStatus.NOT_OWNER
        )

    async def release_reservation_once(
        self,
        user_id: UUID,
        upload_id: UUID,
        owner_token: str,
    ) -> ReservationReleaseStatus:
        owner = _validate_owner_token(owner_token)
        raw = await self._redis.eval(
            _RELEASE_RESERVATION_LUA,
            1,
            reservation_key(user_id, upload_id),
            owner,
        )
        code = _protocol_integer(raw, "reservation release status", maximum=max(_RESERVATION_RELEASE_CODES))
        status = _RESERVATION_RELEASE_CODES.get(code)
        if status is None:
            raise TusSessionProtocolError("Redis returned an unknown reservation release status code")
        return status
