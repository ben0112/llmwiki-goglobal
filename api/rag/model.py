"""Strict configuration and HTTP boundary for server-side RAG models."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, NoReturn
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import SecretStr

from llmwiki_core.rag import MAX_MODEL_TOKENS, MAX_PER_CALL_TIMEOUT_SECONDS, RagDomainError
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

MAX_PROFILE_NAME_CHARS = 100
MAX_PROFILE_TEXT_CHARS = 200
MAX_MESSAGE_CHARS = 240_000
MAX_RESPONSE_BYTES = 256 * 1024
MAX_JSON_DEPTH = 32
MAX_CONFIG_JSON_BYTES = 64 * 1024
MAX_MODEL_PROFILES = 32
MAX_API_KEY_CHARS = 8192
MAX_MESSAGES = 64
MAX_REQUEST_BYTES = 1024 * 1024

_CONFIG_ERROR = "RAG model configuration is invalid"
_INPUT_ERROR = "RAG model request is invalid"
_UNAVAILABLE_MESSAGE = "The RAG model is temporarily unavailable."
_INVALID_RESPONSE_MESSAGE = "The RAG model returned an invalid response."
_PROFILE_FIELDS = frozenset({"base_url", "model", "timeout_seconds", "version"})
_MESSAGE_FIELDS = frozenset({"role", "content"})
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_INVALID_JSON = object()


class RagModelUnavailable(RagDomainError):
    """A sanitized, retryable provider or transport failure."""

    def __init__(self) -> None:
        super().__init__("rag_model_unavailable", _UNAVAILABLE_MESSAGE, retryable=True)


class InvalidRagModelResponse(RagDomainError):
    """A sanitized provider response contract violation."""

    def __init__(self, usage: RagTokenUsage | None = None) -> None:
        if usage is not None and type(usage) is not RagTokenUsage:
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        self.usage = usage
        super().__init__("rag_model_invalid_response", _INVALID_RESPONSE_MESSAGE, retryable=False)


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedRagModelProfile:
    name: str
    base_url: str
    model: str
    timeout_seconds: float
    version: str
    api_key: SecretStr = field(default_factory=lambda: SecretStr(""), repr=False, compare=False)

    def __post_init__(self) -> None:
        failure: BaseException | None = None
        try:
            name = _bounded_text(self.name, maximum=MAX_PROFILE_NAME_CHARS)
            model = _bounded_text(self.model, maximum=MAX_PROFILE_TEXT_CHARS)
            version = _bounded_text(self.version, maximum=MAX_PROFILE_TEXT_CHARS)
            base_url = _normalized_base_url(self.base_url)
            timeout = _bounded_timeout(self.timeout_seconds)
            api_key = self.api_key
            if isinstance(api_key, str):
                api_key = SecretStr(api_key)
            if not isinstance(api_key, SecretStr):
                raise ValueError
        except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
            failure = caught
        if failure is not None:
            _raise_config_failure(failure)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "api_key", api_key)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def __repr__(self) -> str:
        return "ResolvedRagModelProfile(<redacted>)"

    def __str__(self) -> str:
        return "ResolvedRagModelProfile(<redacted>)"


@dataclass(frozen=True, slots=True)
class RagTokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(type(value) is not int or not 0 <= value <= MAX_MODEL_TOKENS for value in values):
            raise ValueError(_INVALID_RESPONSE_MESSAGE)
        if self.prompt_tokens + self.completion_tokens != self.total_tokens:
            raise ValueError(_INVALID_RESPONSE_MESSAGE)


@dataclass(frozen=True, slots=True)
class RagModelResponse:
    payload: Mapping[str, Any]
    usage: RagTokenUsage


class _ResolvedRagModelProfiles(Mapping[str, ResolvedRagModelProfile]):
    __slots__ = ("_profiles",)

    def __init__(self, profiles: Mapping[str, ResolvedRagModelProfile]) -> None:
        self._profiles = MappingProxyType(dict(profiles))

    def __getitem__(self, name: str) -> ResolvedRagModelProfile:
        return self._profiles[name]

    def __iter__(self):
        return iter(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)

    def __repr__(self) -> str:
        return "ResolvedRagModelProfiles(<redacted>)"

    def __str__(self) -> str:
        return "ResolvedRagModelProfiles(<redacted>)"


def resolve_model_profiles(settings: object) -> Mapping[str, ResolvedRagModelProfile]:
    """Resolve exact named profiles and separately supplied credentials."""
    failure: BaseException | None = None
    try:
        raw_profiles = _secret_text(settings.RAG_MODEL_PROFILES_JSON)  # type: ignore[attr-defined]
        raw_keys = _secret_text(settings.RAG_MODEL_API_KEYS_JSON)  # type: ignore[attr-defined]
        if not _within_utf8_byte_limit(raw_profiles, MAX_CONFIG_JSON_BYTES):
            raise ValueError
        if not _within_utf8_byte_limit(raw_keys, MAX_CONFIG_JSON_BYTES):
            raise ValueError
        profiles_value = _load_strict_json(raw_profiles)
        keys_value = _load_strict_json(raw_keys)
        if profiles_value is _INVALID_JSON or keys_value is _INVALID_JSON:
            raise ValueError
        if not isinstance(profiles_value, dict) or not isinstance(keys_value, dict):
            raise ValueError
        if len(profiles_value) > MAX_MODEL_PROFILES or len(keys_value) > MAX_MODEL_PROFILES:
            raise ValueError

        profiles = _normalized_named_mapping(profiles_value)
        keys = _normalized_named_mapping(keys_value)
        if profiles.keys() != keys.keys():
            raise ValueError

        resolved: dict[str, ResolvedRagModelProfile] = {}
        for name, raw_profile in profiles.items():
            if not isinstance(raw_profile, dict) or set(raw_profile) != _PROFILE_FIELDS:
                raise ValueError
            secret = keys[name]
            if not _valid_api_key(secret):
                raise ValueError
            resolved[name] = ResolvedRagModelProfile(
                name=name,
                base_url=raw_profile["base_url"],
                model=raw_profile["model"],
                timeout_seconds=raw_profile["timeout_seconds"],
                version=raw_profile["version"],
                api_key=secret,
            )
    except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
        failure = caught
    if failure is not None:
        _raise_config_failure(failure)
    return _ResolvedRagModelProfiles(resolved)


class _RetryableCloseTransport(httpx.AsyncBaseTransport):
    """Retry indeterminate closes; custom transports must make repeated close safe."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        if self.closed:
            return
        failure: BaseException | None = None
        try:
            await self._transport.aclose()
        except BaseException as caught:  # noqa: BLE001 -- preserve for detached boundary mapping
            failure = caught
        if failure is None:
            self.closed = True
        if failure is not None:
            raise failure


class OpenAICompatibleRagModel:
    """Bounded structured-output client for an allowlisted model profile."""

    def __init__(
        self,
        profile: ResolvedRagModelProfile,
        *,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(profile, ResolvedRagModelProfile):
            raise ValueError(_CONFIG_ERROR)
        if not _valid_api_key(api_key):
            raise ValueError(_CONFIG_ERROR)
        self._profile = profile
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        owned_transport = transport if transport is not None else httpx.AsyncHTTPTransport(trust_env=False)
        self._transport = _RetryableCloseTransport(owned_transport)
        self._client = httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
            },
        )

    @property
    def profile(self) -> ResolvedRagModelProfile:
        return self._profile

    async def __aenter__(self) -> OpenAICompatibleRagModel:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def complete_json(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> RagModelResponse:
        normalized_messages = _validated_messages(messages)
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= MAX_MODEL_TOKENS:
            raise ValueError(_INPUT_ERROR)
        timeout = _request_timeout(timeout_seconds, profile_timeout=self.profile.timeout_seconds)
        body = {
            "model": self.profile.model,
            "messages": list(normalized_messages),
            "response_format": {"type": "json_object"},
            "max_tokens": max_output_tokens,
        }
        request_bytes = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(request_bytes) > MAX_REQUEST_BYTES:
            raise ValueError(_INPUT_ERROR)
        deadline = asyncio.get_running_loop().time() + timeout
        request_task = asyncio.create_task(self._perform_request(request_bytes, timeout))
        request_result, boundary_failure = await self._await_request_task(request_task, deadline)
        del request_task
        if boundary_failure is not None:
            raise boundary_failure from None
        if request_result is None:
            raise RagModelUnavailable() from None
        _response_status, unavailable_status, invalid_response, raw = request_result
        if unavailable_status:
            raise RagModelUnavailable()
        if invalid_response:
            raise InvalidRagModelResponse

        outer = _load_strict_json(bytes(raw))
        if outer is _INVALID_JSON:
            raise InvalidRagModelResponse
        return _parse_response(outer)

    async def _await_request_task(
        self,
        request_task: asyncio.Task[tuple[int, bool, bool, bytearray]],
        deadline: float,
    ) -> tuple[tuple[int, bool, bool, bytearray] | None, BaseException | None]:
        wait_failure: BaseException | None = None
        completed: set[asyncio.Task[tuple[int, bool, bool, bytearray]]] = set()
        pending: set[asyncio.Task[tuple[int, bool, bool, bytearray]]] = set()
        try:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            completed, pending = await asyncio.wait({request_task}, timeout=remaining)
        except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
            wait_failure = caught
        if wait_failure is not None:
            request_task.cancel()
            self._track_cleanup_task(request_task)
            replacement = _boundary_replacement(
                wait_failure,
                RagModelUnavailable(),
            )
            completed.clear()
            pending.clear()
            del wait_failure
            del request_task
            return None, replacement
        if not completed:
            request_task.cancel()
            self._track_cleanup_task(request_task)
            cancellation_failure: BaseException | None = None
            try:
                await asyncio.sleep(0)
            except BaseException as caught:  # noqa: BLE001 -- sanitize external control signals
                cancellation_failure = caught
            completed.clear()
            pending.clear()
            del request_task
            if cancellation_failure is not None:
                replacement = _boundary_replacement(
                    cancellation_failure,
                    RagModelUnavailable(),
                )
                del cancellation_failure
                return None, replacement
            return None, RagModelUnavailable()

        transport_failure: BaseException | None = None
        try:
            request_result = request_task.result()
        except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
            transport_failure = caught
        completed.clear()
        pending.clear()
        del request_task
        if transport_failure is not None:
            replacement = _boundary_replacement(
                transport_failure,
                RagModelUnavailable(),
            )
            del transport_failure
            return None, replacement
        return request_result, None

    async def _perform_request(
        self,
        request_bytes: bytes,
        timeout: float,
    ) -> tuple[int, bool, bool, bytearray]:
        async with self._client.stream(
            "POST",
            self.profile.endpoint,
            content=request_bytes,
            timeout=httpx.Timeout(timeout),
        ) as response:
            return await _read_bounded_response(response)

    def _track_cleanup_task(self, task: asyncio.Task[Any]) -> None:
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._consume_cleanup_task)

    def _consume_cleanup_task(self, task: asyncio.Task[Any]) -> None:
        self._cleanup_tasks.discard(task)
        with suppress(BaseException):
            task.exception()

    async def aclose(self) -> None:
        failure: BaseException | None = None
        try:
            async with self._close_lock:
                if self._closed:
                    return
                cleanup_tasks = tuple(self._cleanup_tasks)
                for cleanup_task in cleanup_tasks:
                    cleanup_task.cancel()
                if cleanup_tasks:
                    await asyncio.wait(cleanup_tasks, timeout=0)
                if self._transport.closed:
                    self._closed = True
                    return
                if self._client.is_closed:
                    await self._transport.aclose()
                else:
                    await self._client.aclose()
                self._closed = self._transport.closed
        except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
            failure = caught
        if self._transport.closed:
            self._closed = True
        if failure is not None:
            _raise_boundary_failure(failure, RagModelUnavailable())


def _secret_text(value: object) -> str:
    if not isinstance(value, SecretStr):
        raise ValueError
    return value.get_secret_value()


def _within_utf8_byte_limit(value: str, maximum: int) -> bool:
    total = 0
    for character in value:
        codepoint = ord(character)
        if codepoint <= 0x7F:
            total += 1
        elif codepoint <= 0x7FF:
            total += 2
        elif 0xD800 <= codepoint <= 0xDFFF:
            return False
        elif codepoint <= 0xFFFF:
            total += 3
        else:
            total += 4
        if total > maximum:
            return False
    return True


def _normalized_named_mapping(raw: dict[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in raw.items():
        name = _bounded_text(raw_name, maximum=MAX_PROFILE_NAME_CHARS)
        if name in normalized:
            raise ValueError
        normalized[name] = value
    return normalized


def _bounded_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError
    value.encode("utf-8")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError
    return normalized


def _bounded_timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError
    normalized = float(value)
    if not isfinite(normalized) or not 0 < normalized <= MAX_PER_CALL_TIMEOUT_SECONDS:
        raise ValueError
    return normalized


def _request_timeout(value: object, *, profile_timeout: float) -> float:
    failure = False
    try:
        normalized = _bounded_timeout(value)
    except (TypeError, ValueError, OverflowError):
        failure = True
        normalized = 0.0
    if failure or normalized > profile_timeout:
        raise ValueError(_INPUT_ERROR)
    return normalized


def _normalized_base_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError
    raw = value
    if not raw or raw != raw.strip() or any(character.isspace() for character in raw):
        raise ValueError
    if "?" in raw or "#" in raw or "\\" in raw or "%" in raw:
        raise ValueError
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError
    # Reachability is intentionally unrestricted: operators may allowlist private/internal HTTP.
    if parsed.username is not None or parsed.password is not None or parsed.hostname is None:
        raise ValueError
    segments = parsed.path.split("/")
    for index, segment in enumerate(segments):
        decoded = unquote(segment)
        if decoded in {".", ".."} or "/" in decoded or "\\" in decoded:
            raise ValueError
        if not segment and index not in {0, len(segments) - 1}:
            raise ValueError
        if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
            raise ValueError
    try:
        url = httpx.URL(raw)
        normalized = str(url).rstrip("/")
    except (httpx.InvalidURL, TypeError, ValueError):
        normalized = ""
    if not normalized:
        raise ValueError
    return normalized


def _safe_response_headers(headers: httpx.Headers) -> bool:
    content_encoding = headers.get("content-encoding")
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        return False
    content_length = headers.get("content-length")
    if content_length is None:
        return True
    if not content_length or not all("0" <= character <= "9" for character in content_length):
        return False
    normalized_length = content_length.lstrip("0") or "0"
    maximum = str(MAX_RESPONSE_BYTES)
    return len(normalized_length) < len(maximum) or (
        len(normalized_length) == len(maximum) and normalized_length <= maximum
    )


async def _read_bounded_response(
    response: httpx.Response,
) -> tuple[int, bool, bool, bytearray]:
    raw = bytearray()
    status = response.status_code
    if not 200 <= status < 300:
        return status, True, False, raw
    if not _safe_response_headers(response.headers):
        return status, False, True, raw
    if response.is_stream_consumed:
        if len(response.content) > MAX_RESPONSE_BYTES:
            return status, False, True, raw
        raw.extend(response.content)
        return status, False, False, raw
    async for chunk in response.aiter_raw():
        if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
            return status, False, True, raw
        raw.extend(chunk)
    return status, False, False, raw


def _valid_api_key(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and len(value) <= MAX_API_KEY_CHARS
        and value.isascii()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _validated_messages(messages: object) -> tuple[dict[str, str], ...]:
    if type(messages) is not tuple or not messages or len(messages) > MAX_MESSAGES:
        raise ValueError(_INPUT_ERROR)
    normalized: list[dict[str, str]] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != _MESSAGE_FIELDS:
            raise ValueError(_INPUT_ERROR)
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            raise ValueError(_INPUT_ERROR)
        if not isinstance(content, str) or not content or "\x00" in content:
            raise ValueError(_INPUT_ERROR)
        utf8_failure = False
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            utf8_failure = True
        if utf8_failure:
            raise ValueError(_INPUT_ERROR)
        total_chars += len(content)
        if len(content) > MAX_MESSAGE_CHARS or total_chars > MAX_MESSAGE_CHARS:
            raise ValueError(_INPUT_ERROR)
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


def _strict_json(raw: str | bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(_value: str) -> NoReturn:
        raise ValueError

    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    _validate_json_depth(value)
    return value


def _load_strict_json(raw: str | bytes) -> Any:
    failure: BaseException | None = None
    try:
        value = _strict_json(raw)
    except BaseException as caught:  # noqa: BLE001 -- detach and sanitize below
        failure = caught
    if failure is not None:
        signal = sanitized_boundary_signal_or_unknown(failure)
        if signal is not None:
            raise signal from None
        return _INVALID_JSON
    return value


def _validate_json_depth(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
        elif isinstance(current, float) and not isfinite(current):
            raise ValueError


def _parse_response(outer: Any) -> RagModelResponse:
    if not isinstance(outer, dict):
        raise InvalidRagModelResponse
    token_usage = _parse_response_usage(outer)
    choices = outer.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise InvalidRagModelResponse(token_usage)
    choice = choices[0]
    if not isinstance(choice, dict):
        raise InvalidRagModelResponse(token_usage)
    message = choice.get("message")
    if not isinstance(message, dict):
        raise InvalidRagModelResponse(token_usage)
    content = message.get("content")
    if not isinstance(content, str):
        raise InvalidRagModelResponse(token_usage)
    payload = _load_strict_json(content)
    if payload is _INVALID_JSON or not isinstance(payload, dict):
        raise InvalidRagModelResponse(token_usage)
    return RagModelResponse(payload=payload, usage=token_usage)


def _parse_response_usage(outer: Mapping[str, object]) -> RagTokenUsage:
    usage = outer.get("usage")
    expected_usage = {"prompt_tokens", "completion_tokens", "total_tokens"}
    if not isinstance(usage, dict) or set(usage) != expected_usage:
        raise InvalidRagModelResponse
    invalid = False
    try:
        token_usage = RagTokenUsage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        )
    except ValueError:
        invalid = True
        token_usage = None
    if invalid or token_usage is None:
        raise InvalidRagModelResponse from None
    return token_usage


def _raise_boundary_failure(failure: BaseException, replacement: Exception) -> NoReturn:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None:
        raise signal from None
    raise replacement from None


def _boundary_replacement(failure: BaseException, replacement: BaseException) -> BaseException:
    signal = sanitized_boundary_signal_or_unknown(failure)
    return signal if signal is not None else replacement


def _raise_config_failure(failure: BaseException) -> NoReturn:
    signal = sanitized_boundary_signal_or_unknown(failure)
    if signal is not None:
        raise signal from None
    raise ValueError(_CONFIG_ERROR) from None


__all__ = [
    "InvalidRagModelResponse",
    "MAX_RESPONSE_BYTES",
    "OpenAICompatibleRagModel",
    "RagModelResponse",
    "RagModelUnavailable",
    "RagTokenUsage",
    "ResolvedRagModelProfile",
    "resolve_model_profiles",
]
