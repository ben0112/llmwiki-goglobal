"""Cross-replica Hosted storage quota reservations backed by Postgres + Redis."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

# Lua numbers are IEEE-754 doubles. This bound also stays below lua-cjson's
# practical 14-significant-digit serialization boundary.
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_QUOTA_BYTES = 9_999_999_999_999
MAX_TTL_SECONDS = 2_147_483_647
DEFAULT_LOCK_TTL_MS = 5_000
DEFAULT_LOCK_ATTEMPTS = 512
DEFAULT_REDIS_COMMAND_TIMEOUT_SECONDS = 6


class RedisClient(Protocol):
    async def set(self, key: str, value: str, **kwargs: object) -> object: ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...


class QuotaExceeded(RuntimeError):
    def __init__(self, used_bytes: int, limit_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.limit_bytes = limit_bytes
        super().__init__("Hosted storage quota exceeded")


class QuotaUnavailable(RuntimeError):
    """Quota admission could not be decided safely."""


class QuotaMarkerSettlementStatus(StrEnum):
    SETTLED = "settled"
    ACTIVE = "active"
    STALE_GENERATION = "stale_generation"
    MARKER_CONFLICT = "marker_conflict"


@dataclass(frozen=True, slots=True)
class QuotaReservation:
    user_id: UUID
    upload_id: UUID
    bytes: int
    owner_token: str

    def __post_init__(self) -> None:
        _require_uuid(self.user_id, "user_id")
        _require_uuid(self.upload_id, "upload_id")
        _require_integer(self.bytes, "bytes", maximum=MAX_QUOTA_BYTES)
        _require_owner_token(self.owner_token)


class QuotaService(Protocol):
    """Shared reservation contract for URL ingest and resumable uploads."""

    async def reserve(
        self,
        user_id: UUID,
        upload_id: UUID,
        byte_count: int,
        ttl_seconds: int,
    ) -> QuotaReservation: ...

    async def finalize(self, reservation: QuotaReservation) -> bool: ...

    async def release(self, reservation: QuotaReservation) -> bool: ...

    async def renew(self, reservation: QuotaReservation, ttl_seconds: int) -> bool: ...

    async def cleanup_expired(self, user_id: UUID) -> int: ...


def _require_uuid(value: object, name: str) -> UUID:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} must be a UUID")
    return value


def _require_integer(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _require_owner_token(value: object) -> str:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    if not isinstance(value, str) or len(value) != 32 or any(character not in allowed for character in value):
        raise ValueError("owner_token must match the generated URL-safe format")
    return value


def quota_lock_key(user_id: UUID) -> str:
    return f"quota:lock:{{{user_id}}}"


def quota_reservations_key(user_id: UUID) -> str:
    return f"quota:reservations:{{{user_id}}}"


def quota_bytes_key(user_id: UUID) -> str:
    return f"quota:bytes:{{{user_id}}}"


def quota_tokens_key(user_id: UUID) -> str:
    return f"quota:tokens:{{{user_id}}}"


def quota_keys(user_id: UUID) -> tuple[str, str, str, str]:
    return (
        quota_lock_key(user_id),
        quota_reservations_key(user_id),
        quota_bytes_key(user_id),
        quota_tokens_key(user_id),
    )


_RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


_LUA_RECORD_VALIDATORS = f"""
local function canonical_uuid(value)
  if type(value) ~= 'string' or string.len(value) ~= 36
     or string.sub(value, 9, 9) ~= '-' or string.sub(value, 14, 14) ~= '-'
     or string.sub(value, 19, 19) ~= '-' or string.sub(value, 24, 24) ~= '-' then
    return false
  end
  local compact, hyphens = string.gsub(value, '%-', '')
  return hyphens == 4 and string.len(compact) == 32
     and string.match(compact, '^[0-9a-f]+$') ~= nil
end

local function canonical_owner(value)
  return type(value) == 'string' and string.len(value) == 32
     and string.match(value, '^[A-Za-z0-9_%-]+$') ~= nil
end

local function canonical_score(value)
  local parsed = tonumber(value)
  return parsed and parsed >= 0 and parsed <= {MAX_SAFE_INTEGER}
     and parsed % 1 == 0 and string.format('%.0f', parsed) == value
end

local function canonical_bytes(value)
  local parsed = tonumber(value)
  if not parsed or parsed < 1 or parsed > {MAX_QUOTA_BYTES}
     or parsed % 1 ~= 0 or string.format('%.0f', parsed) ~= value then
    return nil
  end
  return parsed
end
"""


# Validate the complete snapshot before removing even an expired member. An
# expired malformed record is retained and blocks admission until repaired;
# guessing which correlated fields to delete could erase another generation.
_RESERVE_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + f"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return {{3}}
end

local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local members = redis.call('ZRANGE', KEYS[2], 0, -1, 'WITHSCORES')
local member_count = #members / 2
if redis.call('HLEN', KEYS[3]) ~= member_count or redis.call('HLEN', KEYS[4]) ~= member_count then
  return {{4}}
end

local live = 0
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local score = tonumber(raw_score)
  local raw_bytes = redis.call('HGET', KEYS[3], upload_id)
  local owner = redis.call('HGET', KEYS[4], upload_id)
  local byte_count = canonical_bytes(raw_bytes)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not byte_count or not canonical_owner(owner) then
    return {{4}}
  end
  if score > now_ms then
    live = live + byte_count
    if live > {MAX_QUOTA_BYTES} then
      return {{4}}
    end
  end
end

local committed = tonumber(ARGV[4])
local storage_limit = tonumber(ARGV[5])
local incoming = tonumber(ARGV[6])
local ttl_ms = tonumber(ARGV[7])
if not canonical_uuid(ARGV[2]) or not canonical_owner(ARGV[3])
   or not committed or not storage_limit or not incoming or not ttl_ms
   or committed < 0 or committed > {MAX_QUOTA_BYTES}
   or storage_limit < 0 or storage_limit > {MAX_QUOTA_BYTES}
   or incoming < 1 or incoming > {MAX_QUOTA_BYTES}
   or ttl_ms < 1000 or ttl_ms > {MAX_TTL_SECONDS * 1000} then
  return {{4}}
end

local replaced = redis.call('HGET', KEYS[3], ARGV[2])
local replaced_score = redis.call('ZSCORE', KEYS[2], ARGV[2])
if replaced and replaced_score and tonumber(replaced_score) > now_ms then
  live = live - tonumber(replaced)
end
local used = committed + live + incoming
if used > storage_limit then
  return {{2, string.format('%.0f', used), string.format('%.0f', storage_limit)}}
end

local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now_ms)
for _, upload_id in ipairs(expired) do
  redis.call('ZREM', KEYS[2], upload_id)
  redis.call('HDEL', KEYS[3], upload_id)
  redis.call('HDEL', KEYS[4], upload_id)
end
local expires_at = now_ms + ttl_ms
redis.call('ZADD', KEYS[2], expires_at, ARGV[2])
redis.call('HSET', KEYS[3], ARGV[2], ARGV[6])
redis.call('HSET', KEYS[4], ARGV[2], ARGV[3])
return {{1, string.format('%.0f', expires_at)}}
"""
)


_SETTLE_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + """
if not canonical_uuid(ARGV[1]) or not canonical_owner(ARGV[2]) or not canonical_bytes(ARGV[3]) then
  return {3}
end
local count = redis.call('ZCARD', KEYS[1])
if redis.call('HLEN', KEYS[2]) ~= count or redis.call('HLEN', KEYS[3]) ~= count then
  return {3}
end
local members = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local raw_bytes = redis.call('HGET', KEYS[2], upload_id)
  local owner = redis.call('HGET', KEYS[3], upload_id)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not canonical_bytes(raw_bytes) or not canonical_owner(owner) then
    return {3}
  end
end
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
local byte_count = redis.call('HGET', KEYS[2], ARGV[1])
local owner = redis.call('HGET', KEYS[3], ARGV[1])
if not score and not byte_count and not owner then
  return {0}
end
if not score or not byte_count or not owner then
  return {3}
end
if owner ~= ARGV[2] or byte_count ~= ARGV[3] then
  return {2}
end
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1])
return {1}
"""
)


_SETTLE_TUS_MARKER_IF_ABSENT_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + f"""
if not canonical_uuid(ARGV[1]) or not canonical_owner(ARGV[2]) or not canonical_bytes(ARGV[3]) then
  return {{5}}
end
local count = redis.call('ZCARD', KEYS[1])
if redis.call('HLEN', KEYS[2]) ~= count or redis.call('HLEN', KEYS[3]) ~= count then
  return {{5}}
end
local members = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local raw_bytes = redis.call('HGET', KEYS[2], upload_id)
  local owner = redis.call('HGET', KEYS[3], upload_id)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not canonical_bytes(raw_bytes) or not canonical_owner(owner) then
    return {{5}}
  end
end
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
local byte_count = redis.call('HGET', KEYS[2], ARGV[1])
local owner = redis.call('HGET', KEYS[3], ARGV[1])
if score or byte_count or owner then
  if not score or not byte_count or not owner then
    return {{5}}
  end
  if owner == ARGV[2] and byte_count == ARGV[3] then
    return {{1}}
  end
  return {{2}}
end

local raw = redis.call('GET', KEYS[4])
if not raw then return {{0}} end
if redis.call('PTTL', KEYS[4]) <= 0 then return {{5}} end
local decoded, marker = pcall(cjson.decode, raw)
if not decoded or type(marker) ~= 'table' then return {{5}} end
local marker_count = 0
for _ in pairs(marker) do marker_count = marker_count + 1 end
if marker_count ~= 3 or type(marker.bytes) ~= 'number' or marker.bytes ~= math.floor(marker.bytes)
   or marker.bytes < 1 or marker.bytes > {MAX_QUOTA_BYTES}
   or not canonical_owner(marker.owner)
   or (marker.state ~= 'reserved' and marker.state ~= 'released') then
  return {{5}}
end
local canonical = '{{"bytes":' .. string.format('%.0f', marker.bytes)
  .. ',"owner":"' .. marker.owner .. '","state":"' .. marker.state .. '"}}'
if canonical ~= raw then return {{5}} end
if marker.owner ~= ARGV[2] or string.format('%.0f', marker.bytes) ~= ARGV[3] then
  return {{4}}
end
if marker.state == 'released' then return {{0}} end
local released = '{{"bytes":' .. ARGV[3] .. ',"owner":"' .. ARGV[2] .. '","state":"released"}}'
redis.call('SET', KEYS[4], released, 'KEEPTTL')
return {{0}}
"""
)


_FINALIZE_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return {4}
end
if not canonical_uuid(ARGV[2]) or not canonical_owner(ARGV[3]) or not canonical_bytes(ARGV[4]) then
  return {3}
end
local count = redis.call('ZCARD', KEYS[2])
if redis.call('HLEN', KEYS[3]) ~= count or redis.call('HLEN', KEYS[4]) ~= count then
  return {3}
end
local members = redis.call('ZRANGE', KEYS[2], 0, -1, 'WITHSCORES')
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local raw_bytes = redis.call('HGET', KEYS[3], upload_id)
  local owner = redis.call('HGET', KEYS[4], upload_id)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not canonical_bytes(raw_bytes) or not canonical_owner(owner) then
    return {3}
  end
end
local score = redis.call('ZSCORE', KEYS[2], ARGV[2])
local byte_count = redis.call('HGET', KEYS[3], ARGV[2])
local owner = redis.call('HGET', KEYS[4], ARGV[2])
if not score and not byte_count and not owner then
  return {0}
end
if not score or not byte_count or not owner then
  return {3}
end
if owner ~= ARGV[3] or byte_count ~= ARGV[4] then
  return {2}
end
redis.call('ZREM', KEYS[2], ARGV[2])
redis.call('HDEL', KEYS[3], ARGV[2])
redis.call('HDEL', KEYS[4], ARGV[2])
return {1}
"""
)


_RENEW_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + f"""
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return {{4}}
end
local ttl_ms = tonumber(ARGV[5])
if not canonical_uuid(ARGV[2]) or not canonical_owner(ARGV[3])
   or not canonical_bytes(ARGV[4]) or not ttl_ms
   or ttl_ms < 1000 or ttl_ms > {MAX_TTL_SECONDS * 1000} then
  return {{3}}
end
local count = redis.call('ZCARD', KEYS[2])
if redis.call('HLEN', KEYS[3]) ~= count or redis.call('HLEN', KEYS[4]) ~= count then
  return {{3}}
end
local members = redis.call('ZRANGE', KEYS[2], 0, -1, 'WITHSCORES')
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local raw_bytes = redis.call('HGET', KEYS[3], upload_id)
  local owner = redis.call('HGET', KEYS[4], upload_id)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not canonical_bytes(raw_bytes) or not canonical_owner(owner) then
    return {{3}}
  end
end
local score = redis.call('ZSCORE', KEYS[2], ARGV[2])
local byte_count = redis.call('HGET', KEYS[3], ARGV[2])
local owner = redis.call('HGET', KEYS[4], ARGV[2])
if not score and not byte_count and not owner then
  return {{0}}
end
if not score or not byte_count or not owner then
  return {{3}}
end
if owner ~= ARGV[3] or byte_count ~= ARGV[4] then
  return {{2}}
end
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
if tonumber(score) <= now_ms then
  return {{0}}
end
redis.call('ZADD', KEYS[2], now_ms + ttl_ms, ARGV[2])
return {{1}}
"""
)


_CLEANUP_SCRIPT = (
    _LUA_RECORD_VALIDATORS
    + """
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local count = redis.call('ZCARD', KEYS[1])
if redis.call('HLEN', KEYS[2]) ~= count or redis.call('HLEN', KEYS[3]) ~= count then
  return {0}
end
local members = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
for index = 1, #members, 2 do
  local upload_id = members[index]
  local raw_score = members[index + 1]
  local raw_bytes = redis.call('HGET', KEYS[2], upload_id)
  local owner = redis.call('HGET', KEYS[3], upload_id)
  if not canonical_uuid(upload_id) or not canonical_score(raw_score)
     or not canonical_bytes(raw_bytes) or not canonical_owner(owner) then
    return {0}
  end
end
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now_ms)
for _, upload_id in ipairs(expired) do
  redis.call('ZREM', KEYS[1], upload_id)
  redis.call('HDEL', KEYS[2], upload_id)
  redis.call('HDEL', KEYS[3], upload_id)
end
return {1, #expired}
"""
)


def _response_items(response: object) -> list[object]:
    if not isinstance(response, (list, tuple)) or not response:
        raise QuotaUnavailable("Quota coordination returned an invalid response")
    return list(response)


def _response_int(value: object) -> int:
    if isinstance(value, bool):
        raise QuotaUnavailable("Quota coordination returned an invalid response")
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise QuotaUnavailable("Quota coordination returned an invalid response") from exc
    if not isinstance(value, str) or not value.isdecimal() or (len(value) > 1 and value.startswith("0")):
        raise QuotaUnavailable("Quota coordination returned an invalid response")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise QuotaUnavailable("Quota coordination returned an invalid response") from exc
    return parsed


def _interpret_reserve_response(response: object) -> bool:
    """Return whether the lock was lost; raise for every other non-success status."""
    items = _response_items(response)
    status = _response_int(items[0])
    if status == 1 and len(items) == 2:
        _response_int(items[1])
        return False
    if status == 2 and len(items) == 3:
        raise QuotaExceeded(_response_int(items[1]), _response_int(items[2]))
    if status == 3 and len(items) == 1:
        return True
    if status == 4 and len(items) == 1:
        raise QuotaUnavailable("Quota reservation state is invalid")
    raise QuotaUnavailable("Quota coordination returned an invalid response")


class HostedQuotaService:
    def __init__(
        self,
        pool: object,
        redis: RedisClient,
        *,
        lock_ttl_ms: int = DEFAULT_LOCK_TTL_MS,
        max_lock_attempts: int = DEFAULT_LOCK_ATTEMPTS,
        redis_command_timeout_seconds: int = DEFAULT_REDIS_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self._pool = pool
        self._redis = redis
        self._lock_ttl_ms = _require_integer(lock_ttl_ms, "lock_ttl_ms", maximum=MAX_TTL_SECONDS * 1000)
        self._max_lock_attempts = _require_integer(max_lock_attempts, "max_lock_attempts", maximum=10_000)
        self._redis_command_timeout_seconds = _require_integer(
            redis_command_timeout_seconds,
            "redis_command_timeout_seconds",
            maximum=MAX_TTL_SECONDS,
        )

    async def reserve(
        self,
        user_id: UUID,
        upload_id: UUID,
        byte_count: int,
        ttl_seconds: int,
    ) -> QuotaReservation:
        user_id = _require_uuid(user_id, "user_id")
        upload_id = _require_uuid(upload_id, "upload_id")
        byte_count = _require_integer(byte_count, "byte_count", maximum=MAX_QUOTA_BYTES)
        ttl_seconds = _require_integer(ttl_seconds, "ttl_seconds", maximum=MAX_TTL_SECONDS)
        owner_token = secrets.token_urlsafe(24)
        reservation = QuotaReservation(user_id, upload_id, byte_count, owner_token)

        for attempt in range(self._max_lock_attempts):
            lock_token = secrets.token_urlsafe(24)
            try:
                acquired = await self._redis.set(
                    quota_lock_key(user_id),
                    lock_token,
                    nx=True,
                    px=self._lock_ttl_ms,
                )
            except Exception as exc:  # noqa: BLE001 - Redis clients expose backend-specific failures.
                raise QuotaUnavailable("Quota coordination is unavailable") from exc
            if not acquired:
                await asyncio.sleep(min(0.001 * (attempt + 1), 0.025))
                continue

            retry = False
            error: BaseException | None = None
            try:
                committed_bytes, storage_limit = await self._read_committed_usage(user_id)
                response = await self._run_admission_eval(
                    reservation,
                    (
                        *quota_keys(user_id),
                        lock_token,
                        str(upload_id),
                        owner_token,
                        str(committed_bytes),
                        str(storage_limit),
                        str(byte_count),
                        str(ttl_seconds * 1000),
                    ),
                )
                try:
                    retry = _interpret_reserve_response(response)
                except QuotaExceeded:
                    raise
                except QuotaUnavailable:
                    await self._best_effort_release(reservation)
                    raise
                if not retry:
                    return reservation
            except (QuotaExceeded, QuotaUnavailable) as exc:
                error = exc
            except Exception as exc:  # noqa: BLE001 - fail closed on Postgres or Redis errors.
                error = QuotaUnavailable("Quota coordination is unavailable")
                error.__cause__ = exc
            finally:
                await self._release_lock(user_id, lock_token)

            if error is not None:
                raise error
            if retry:
                await asyncio.sleep(0)
                continue

        raise QuotaUnavailable("Quota coordination is busy")

    async def finalize(self, reservation: QuotaReservation) -> bool:
        return await self._run_fenced_mutation(reservation, _FINALIZE_SCRIPT)

    async def release(self, reservation: QuotaReservation) -> bool:
        return await self._settle(reservation)

    async def settle_tus_marker_if_absent(
        self,
        reservation: QuotaReservation,
    ) -> QuotaMarkerSettlementStatus:
        """Atomically settle this owner's TUS marker only when its quota row is absent."""
        self._validate_reservation(reservation)
        _, reservations_key, bytes_key, tokens_key = quota_keys(reservation.user_id)
        marker_key = f"tus:reservation:{{{reservation.user_id}}}:{{{reservation.upload_id}}}"
        try:
            items = _response_items(
                await self._redis.eval(
                    _SETTLE_TUS_MARKER_IF_ABSENT_SCRIPT,
                    4,
                    reservations_key,
                    bytes_key,
                    tokens_key,
                    marker_key,
                    str(reservation.upload_id),
                    reservation.owner_token,
                    str(reservation.bytes),
                )
            )
        except QuotaUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - Redis clients expose backend-specific failures.
            raise QuotaUnavailable("Quota coordination is unavailable") from exc
        if len(items) != 1:
            raise QuotaUnavailable("Quota coordination returned an invalid response")
        status = _response_int(items[0])
        if status == 0:
            return QuotaMarkerSettlementStatus.SETTLED
        if status == 1:
            return QuotaMarkerSettlementStatus.ACTIVE
        if status == 2:
            return QuotaMarkerSettlementStatus.STALE_GENERATION
        if status == 4:
            return QuotaMarkerSettlementStatus.MARKER_CONFLICT
        if status == 5:
            raise QuotaUnavailable("Quota or TUS marker state is invalid")
        raise QuotaUnavailable("Quota coordination returned an invalid response")

    async def renew(self, reservation: QuotaReservation, ttl_seconds: int) -> bool:
        ttl_seconds = _require_integer(ttl_seconds, "ttl_seconds", maximum=MAX_TTL_SECONDS)
        return await self._run_fenced_mutation(
            reservation,
            _RENEW_SCRIPT,
            str(ttl_seconds * 1000),
        )

    async def cleanup_expired(self, user_id: UUID) -> int:
        user_id = _require_uuid(user_id, "user_id")
        _, reservations_key, bytes_key, tokens_key = quota_keys(user_id)
        try:
            items = _response_items(
                await self._redis.eval(
                    _CLEANUP_SCRIPT,
                    3,
                    reservations_key,
                    bytes_key,
                    tokens_key,
                )
            )
        except QuotaUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - Redis clients expose backend-specific failures.
            raise QuotaUnavailable("Quota coordination is unavailable") from exc
        if len(items) == 2 and _response_int(items[0]) == 1:
            return _response_int(items[1])
        if len(items) == 1 and _response_int(items[0]) == 0:
            raise QuotaUnavailable("Quota reservation state is invalid")
        raise QuotaUnavailable("Quota coordination returned an invalid response")

    async def _run_admission_eval(
        self,
        reservation: QuotaReservation,
        keys_and_args: tuple[str, ...],
    ) -> object:
        task = asyncio.create_task(self._redis.eval(_RESERVE_SCRIPT, 4, *keys_and_args))
        try:
            async with asyncio.timeout(self._redis_command_timeout_seconds):
                return await asyncio.shield(task)
        except asyncio.CancelledError:
            await self._finish_redis_task(task)
            await self._best_effort_release(reservation)
            raise
        except Exception:
            await self._finish_redis_task(task)
            await self._best_effort_release(reservation)
            raise

    async def _finish_redis_task(self, task: asyncio.Task) -> None:
        if not task.done():
            try:
                async with asyncio.timeout(self._redis_command_timeout_seconds):
                    await asyncio.shield(task)
            except TimeoutError:
                task.cancel()
            except (Exception, asyncio.CancelledError):  # noqa: BLE001 - cleanup observes backend task failures.
                pass
        with suppress(Exception, asyncio.CancelledError):
            await task

    async def _best_effort_release(self, reservation: QuotaReservation) -> None:
        task = asyncio.create_task(self._settle(reservation))
        try:
            async with asyncio.timeout(self._redis_command_timeout_seconds):
                await asyncio.shield(task)
        except (Exception, asyncio.CancelledError):  # noqa: BLE001 - cleanup is deliberately best-effort.
            task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await task

    async def _run_fenced_mutation(
        self,
        reservation: QuotaReservation,
        script: str,
        *extra_args: str,
    ) -> bool:
        self._validate_reservation(reservation)
        for attempt in range(self._max_lock_attempts):
            lock_token = secrets.token_urlsafe(24)
            try:
                acquired = await self._redis.set(
                    quota_lock_key(reservation.user_id),
                    lock_token,
                    nx=True,
                    px=self._lock_ttl_ms,
                )
            except Exception as exc:  # noqa: BLE001 - Redis clients expose backend-specific failures.
                raise QuotaUnavailable("Quota coordination is unavailable") from exc
            if not acquired:
                await asyncio.sleep(min(0.001 * (attempt + 1), 0.025))
                continue

            retry = False
            try:
                items = _response_items(
                    await self._redis.eval(
                        script,
                        4,
                        *quota_keys(reservation.user_id),
                        lock_token,
                        str(reservation.upload_id),
                        reservation.owner_token,
                        str(reservation.bytes),
                        *extra_args,
                    )
                )
                if len(items) != 1:
                    raise QuotaUnavailable("Quota coordination returned an invalid response")
                status = _response_int(items[0])
                if status == 1:
                    return True
                if status in {0, 2}:
                    return False
                if status == 3:
                    raise QuotaUnavailable("Quota reservation state is invalid")
                if status == 4:
                    retry = True
                else:
                    raise QuotaUnavailable("Quota coordination returned an invalid response")
            except QuotaUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - fail conservative and retain the reservation.
                raise QuotaUnavailable("Quota coordination is unavailable") from exc
            finally:
                await self._release_lock(reservation.user_id, lock_token)
            if retry:
                await asyncio.sleep(0)

        raise QuotaUnavailable("Quota coordination is busy")

    @staticmethod
    def _validate_reservation(reservation: QuotaReservation) -> None:
        if not isinstance(reservation, QuotaReservation):
            raise ValueError("reservation must be a QuotaReservation")
        _require_uuid(reservation.user_id, "user_id")
        _require_uuid(reservation.upload_id, "upload_id")
        _require_integer(reservation.bytes, "bytes", maximum=MAX_QUOTA_BYTES)
        _require_owner_token(reservation.owner_token)

    async def _read_committed_usage(self, user_id: UUID) -> tuple[int, int]:
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                row = await conn.fetchrow(
                    "SELECT u.storage_limit_bytes, "
                    "COALESCE((SELECT SUM(d.file_size) FROM documents d WHERE d.user_id = u.id), 0)::bigint "
                    "AS committed_bytes FROM users u WHERE u.id = $1::uuid",
                    user_id,
                )
        except Exception as exc:  # noqa: BLE001 - asyncpg exposes several operational subclasses.
            raise QuotaUnavailable("Quota billing data is unavailable") from exc
        if row is None:
            raise QuotaUnavailable("Quota billing subject is unavailable")
        committed_bytes = row["committed_bytes"]
        storage_limit = row["storage_limit_bytes"]
        if (
            isinstance(committed_bytes, bool)
            or not isinstance(committed_bytes, int)
            or not 0 <= committed_bytes <= MAX_QUOTA_BYTES
            or isinstance(storage_limit, bool)
            or not isinstance(storage_limit, int)
            or not 0 <= storage_limit <= MAX_QUOTA_BYTES
        ):
            raise QuotaUnavailable("Quota billing data is invalid")
        return committed_bytes, storage_limit

    async def _settle(self, reservation: QuotaReservation) -> bool:
        self._validate_reservation(reservation)
        _, reservations_key, bytes_key, tokens_key = quota_keys(reservation.user_id)
        try:
            items = _response_items(
                await self._redis.eval(
                    _SETTLE_SCRIPT,
                    3,
                    reservations_key,
                    bytes_key,
                    tokens_key,
                    str(reservation.upload_id),
                    reservation.owner_token,
                    str(reservation.bytes),
                )
            )
        except QuotaUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - Redis clients expose backend-specific failures.
            raise QuotaUnavailable("Quota coordination is unavailable") from exc
        if len(items) != 1:
            raise QuotaUnavailable("Quota coordination returned an invalid response")
        status = _response_int(items[0])
        if status == 1:
            return True
        if status in {0, 2}:
            return False
        if status == 3:
            raise QuotaUnavailable("Quota reservation state is invalid")
        raise QuotaUnavailable("Quota coordination returned an invalid response")

    async def _release_lock(self, user_id: UUID, token: str) -> None:
        try:
            await self._redis.eval(_RELEASE_LOCK_SCRIPT, 1, quota_lock_key(user_id), token)
        except Exception as exc:  # noqa: BLE001 - lock expires shortly; never expose its owner token.
            logger.warning("Hosted quota lock release failed error_type=%s", type(exc).__name__)
