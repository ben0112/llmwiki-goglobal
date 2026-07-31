from __future__ import annotations

import asyncio
import gzip
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest
import rag.model as rag_model
from rag.model import (
    InvalidRagModelResponse,
    OpenAICompatibleRagModel,
    RagModelUnavailable,
    RagTokenUsage,
    ResolvedRagModelProfile,
    resolve_model_profiles,
)

PROFILE = ResolvedRagModelProfile(
    name="primary",
    base_url="https://models.test/v1",
    model="writer-v1",
    timeout_seconds=60,
    version="2026-07-26",
)
VALID_RESPONSE = {
    "choices": [{"message": {"content": '{"pages":[]}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
}

_PRIVATE_MARKERS = (
    "private",
    "models.test",
    "writer-v1",
    "2026-07-26",
    "secret",
)
_MAX_CONFIG_JSON_BYTES = 64 * 1024
_MAX_MODEL_PROFILES = 32
_MAX_API_KEY_CHARS = 8192
_MAX_MESSAGES = 64
_MAX_REQUEST_BYTES = 1024 * 1024


def _exception_graph(root: BaseException) -> tuple[BaseException, ...]:
    seen: set[int] = set()
    pending = [root]
    graph: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return tuple(graph)


def _assert_detached_and_sanitized(error: BaseException) -> None:
    assert error.__context__ is None
    assert error.__cause__ is None
    graph = _exception_graph(error)
    assert graph == (error,)
    rendered = "\n".join(f"{node!r}\n{node}\n{node.args!r}" for node in graph)
    for marker in _PRIVATE_MARKERS:
        assert marker not in rendered


class _Settings:
    RAG_MODEL_PROFILES_JSON: Any
    RAG_MODEL_API_KEYS_JSON: Any

    def __init__(self, profiles: str, keys: str) -> None:
        from pydantic import SecretStr

        self.RAG_MODEL_PROFILES_JSON = SecretStr(profiles)
        self.RAG_MODEL_API_KEYS_JSON = SecretStr(keys)


def _profile_json(**overrides: object) -> str:
    profile: dict[str, object] = {
        "base_url": "https://models.test/v1/",
        "model": " writer-v1 ",
        "timeout_seconds": 60,
        "version": " 2026-07-26 ",
    }
    profile.update(overrides)
    return json.dumps({" primary ": profile})


def _resolved(*, profiles: str | None = None, keys: str | None = None):
    return resolve_model_profiles(_Settings(profiles or _profile_json(), keys or '{" primary ":"sk-private"}'))


def test_profile_resolution_is_exact_normalized_and_secret_safe():
    profiles = _resolved()

    assert isinstance(profiles, Mapping)
    assert tuple(profiles) == ("primary",)
    profile = profiles["primary"]
    assert profile.name == "primary"
    assert profile.base_url == "https://models.test/v1"
    assert profile.endpoint == "https://models.test/v1/chat/completions"
    assert profile.model == "writer-v1"
    assert profile.timeout_seconds == 60
    assert profile.version == "2026-07-26"
    assert profile.api_key.get_secret_value() == "sk-private"
    rendered = f"{profile!r}\n{profile}\n{profiles!r}"
    assert repr(profiles) == "ResolvedRagModelProfiles(<redacted>)"
    assert repr(profile) == "ResolvedRagModelProfile(<redacted>)"
    for private in ("primary", "models.test", "writer-v1", "2026-07-26", "sk-private"):
        assert private not in rendered
    assert profile == ResolvedRagModelProfile(
        name="primary",
        base_url="https://models.test/v1",
        model="writer-v1",
        timeout_seconds=60,
        version="2026-07-26",
        api_key="different-secret",
    )


@pytest.mark.parametrize(
    ("profiles", "keys"),
    (
        ("[]", "{}"),
        ('{"a":{},"a":{}}', '{"a":"secret"}'),
        ('{"a":{"base_url":"https://x.test","model":"m","timeout_seconds":NaN,"version":"v"}}', '{"a":"secret"}'),
        (_profile_json(extra="no"), '{" primary ":"secret"}'),
        (
            json.dumps({"": {"base_url": "https://x.test", "model": "m", "timeout_seconds": 1, "version": "v"}}),
            '{"":"secret"}',
        ),
        (_profile_json(), "[]"),
        (_profile_json(), '{" primary ":"one"," primary ":"two"}'),
        (_profile_json(), "{}"),
        (_profile_json(), '{" primary ":"secret","extra":"secret"}'),
        (_profile_json(), '{" primary ":1}'),
        (_profile_json(), '{" primary ":""}'),
    ),
)
def test_profile_resolution_rejects_malformed_or_incomplete_mappings(profiles, keys):
    with pytest.raises(ValueError, match="RAG model configuration is invalid") as caught:
        resolve_model_profiles(_Settings(profiles, keys))

    rendered = f"{caught.value!r}\n{caught.value}"
    assert "models.test" not in rendered
    assert "writer-v1" not in rendered
    assert "secret" not in rendered
    _assert_detached_and_sanitized(caught.value)


@pytest.mark.parametrize(
    ("profiles", "keys"),
    (
        ('{"private-profile":{"base_url":"https://private.test"', '{"private-profile":"secret"}'),
        (_profile_json(), '{" primary ":"secret-key"'),
    ),
)
def test_secret_json_decode_failures_are_not_reachable_from_public_error(profiles, keys):
    with pytest.raises(ValueError) as caught:
        resolve_model_profiles(_Settings(profiles, keys))

    _assert_detached_and_sanitized(caught.value)


@pytest.mark.parametrize(
    "base_url",
    (
        "models.test/v1",
        "ftp://models.test/v1",
        "https:///v1",
        "https://user:pass@models.test/v1",
        "https://models.test/v1?q=1",
        "https://models.test/v1#frag",
        "https://models.test/v1\nheader",
        "https://models.test/v1//nested",
        "https://models.test/v1/../escape",
        "https://models.test/v1/%2e%2e/escape",
    ),
)
def test_profile_resolution_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError, match="RAG model configuration is invalid"):
        _resolved(profiles=_profile_json(base_url=base_url))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model", ""),
        ("model", "m" * 201),
        ("model", "bad\x00model"),
        ("version", ""),
        ("version", "v" * 201),
        ("timeout_seconds", True),
        ("timeout_seconds", "30"),
        ("timeout_seconds", 0),
        ("timeout_seconds", 181),
        ("timeout_seconds", float("inf")),
    ),
)
def test_profile_resolution_enforces_text_and_timeout_bounds(field, value):
    with pytest.raises(ValueError, match="RAG model configuration is invalid"):
        _resolved(profiles=_profile_json(**{field: value}))


def test_profile_resolution_rejects_names_that_collide_after_normalization():
    profiles = json.dumps(
        {
            "primary": {"base_url": "https://one.test", "model": "one", "timeout_seconds": 1, "version": "v1"},
            " primary ": {"base_url": "https://two.test", "model": "two", "timeout_seconds": 1, "version": "v2"},
        }
    )
    keys = '{"primary":"one"," primary ":"two"}'

    with pytest.raises(ValueError, match="RAG model configuration is invalid"):
        resolve_model_profiles(_Settings(profiles, keys))


def test_profile_json_byte_cap_is_checked_before_parsing(monkeypatch):
    calls = 0
    real_loads = rag_model.json.loads

    def count_loads(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(rag_model.json, "loads", count_loads)

    exact = "{}" + " " * (_MAX_CONFIG_JSON_BYTES - 2)
    assert resolve_model_profiles(_Settings(exact, "{}")) == {}
    assert calls == 2

    calls = 0
    with pytest.raises(ValueError) as caught:
        resolve_model_profiles(_Settings(exact + " ", "{}"))
    assert calls == 0
    _assert_detached_and_sanitized(caught.value)


def test_profile_count_has_exact_hard_cap():
    profiles = {
        f"p{index}": {
            "base_url": f"http://10.0.0.{index + 1}:8080/internal",
            "model": "m",
            "timeout_seconds": 1,
            "version": "v",
        }
        for index in range(_MAX_MODEL_PROFILES + 1)
    }
    keys = {name: "k" for name in profiles}

    exact_profiles = dict(list(profiles.items())[:_MAX_MODEL_PROFILES])
    exact_keys = dict(list(keys.items())[:_MAX_MODEL_PROFILES])
    assert len(resolve_model_profiles(_Settings(json.dumps(exact_profiles), json.dumps(exact_keys)))) == 32

    with pytest.raises(ValueError):
        resolve_model_profiles(_Settings(json.dumps(profiles), json.dumps(keys)))


def test_api_key_length_has_exact_hard_cap():
    exact = _resolved(keys=json.dumps({" primary ": "k" * _MAX_API_KEY_CHARS}))
    assert len(exact["primary"].api_key.get_secret_value()) == _MAX_API_KEY_CHARS

    with pytest.raises(ValueError) as caught:
        _resolved(keys=json.dumps({" primary ": "k" * (_MAX_API_KEY_CHARS + 1)}))
    _assert_detached_and_sanitized(caught.value)


def test_direct_profile_validation_error_is_fully_detached():
    with pytest.raises(ValueError) as caught:
        ResolvedRagModelProfile(
            name="private-name",
            base_url="https://private.test/v1?secret=query",
            model="private-model",
            timeout_seconds=30,
            version="private-version",
        )

    _assert_detached_and_sanitized(caught.value)


@pytest.mark.parametrize(
    "base_url",
    (
        " https://models.test/v1",
        "https://model s.test/v1",
        "https://models.test/%76%31",
        "https://models%2etest/v1",
        "https://models.test/%252e%252e/escape",
    ),
)
def test_base_url_rejects_whitespace_and_percent_ambiguity(base_url):
    with pytest.raises(ValueError):
        _resolved(profiles=_profile_json(base_url=base_url))


def test_private_internal_http_base_url_remains_supported():
    profile = _resolved(profiles=_profile_json(base_url="http://10.0.0.8:8080/internal"))["primary"]

    assert profile.endpoint == "http://10.0.0.8:8080/internal/chat/completions"


def test_client_requires_an_explicit_nonempty_api_key():
    signature = inspect.signature(OpenAICompatibleRagModel)
    assert signature.parameters["api_key"].default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        OpenAICompatibleRagModel(PROFILE)  # type: ignore[call-arg]
    with pytest.raises(ValueError) as caught:
        OpenAICompatibleRagModel(PROFILE, api_key="")
    _assert_detached_and_sanitized(caught.value)


def test_default_transport_also_disables_environment_trust(monkeypatch):
    seen: list[dict[str, object]] = []
    transport = _CloseTransport()

    def build_transport(**kwargs: object) -> httpx.AsyncBaseTransport:
        seen.append(kwargs)
        return transport

    monkeypatch.setattr(rag_model.httpx, "AsyncHTTPTransport", build_transport)

    OpenAICompatibleRagModel(PROFILE, api_key="secret")

    assert seen == [{"trust_env": False}]


@pytest.mark.asyncio
async def test_structured_client_returns_payload_and_exact_usage_and_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    response = await client.complete_json(
        messages=({"role": "system", "content": "policy"},),
        max_output_tokens=100,
        timeout_seconds=30,
    )

    assert response.payload == {"pages": []}
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 14
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://models.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"
    assert json.loads(request.content) == {
        "model": "writer-v1",
        "messages": [{"role": "system", "content": "policy"}],
        "response_format": {"type": "json_object"},
        "max_tokens": 100,
    }
    assert request.extensions["timeout"] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}
    assert client._client.follow_redirects is False
    assert client._client._trust_env is False
    await client.aclose()


@pytest.mark.parametrize(
    ("messages", "max_tokens", "timeout"),
    (
        ([{"role": "system", "content": "policy"}], 1, 1),
        ((), 1, 1),
        (({"role": "system", "content": "policy", "name": "hidden"},), 1, 1),
        (({"role": "tool", "content": "policy"},), 1, 1),
        (({"role": "system", "content": ""},), 1, 1),
        (({"role": "system", "content": "bad\x00text"},), 1, 1),
        (({"role": "system", "content": "x" * 240_001},), 1, 1),
        (({"role": "system", "content": "policy"},), True, 1),
        (({"role": "system", "content": "policy"},), 250_001, 1),
        (({"role": "system", "content": "policy"},), 1, True),
        (({"role": "system", "content": "policy"},), 1, 0),
        (({"role": "system", "content": "policy"},), 1, 61),
        (({"role": "system", "content": "policy"},), 1, float("nan")),
    ),
)
@pytest.mark.asyncio
async def test_request_validation_is_strict(messages, max_tokens, timeout):
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=VALID_RESPONSE)

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        await client.complete_json(
            messages=messages,
            max_output_tokens=max_tokens,
            timeout_seconds=timeout,
        )
    assert called is False
    await client.aclose()


@pytest.mark.asyncio
async def test_message_count_has_exact_hard_cap():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=VALID_RESPONSE)

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    exact = tuple({"role": "user", "content": "x"} for _ in range(_MAX_MESSAGES))
    await client.complete_json(messages=exact, max_output_tokens=1, timeout_seconds=1)

    with pytest.raises(ValueError):
        await client.complete_json(
            messages=exact + ({"role": "user", "content": "x"},),
            max_output_tokens=1,
            timeout_seconds=1,
        )
    assert calls == 1
    await client.aclose()


def _serialized_request_size(content: str) -> int:
    body = {
        "model": "writer-v1",
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "max_tokens": 1,
    }
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


@pytest.mark.asyncio
async def test_serialized_request_utf8_size_has_exact_hard_cap_with_escaping():
    base_size = _serialized_request_size("")
    remaining = _MAX_REQUEST_BYTES - base_size
    escaped_count, plain_count = divmod(remaining, 6)
    exact_content = "\x01" * escaped_count + "x" * plain_count
    assert len(exact_content) < 240_000
    assert _serialized_request_size(exact_content) == _MAX_REQUEST_BYTES
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=VALID_RESPONSE)

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    await client.complete_json(
        messages=({"role": "user", "content": exact_content},),
        max_output_tokens=1,
        timeout_seconds=1,
    )
    assert len(seen[0].content) == _MAX_REQUEST_BYTES

    with pytest.raises(ValueError):
        await client.complete_json(
            messages=({"role": "user", "content": exact_content + "x"},),
            max_output_tokens=1,
            timeout_seconds=1,
        )
    assert len(seen) == 1
    await client.aclose()


def test_unavailable_error_has_stable_no_arg_retryable_contract():
    error = RagModelUnavailable()

    assert error.code == "rag_model_unavailable"
    assert error.public_message == "The RAG model is temporarily unavailable."
    assert error.retryable is True
    with pytest.raises(TypeError):
        RagModelUnavailable(False)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "failure",
    (
        httpx.ConnectError("private"),
        httpx.ReadError("private"),
        httpx.ReadTimeout("private"),
        RuntimeError("private"),
    ),
)
@pytest.mark.asyncio
async def test_runtime_failures_are_sanitized_and_retryable(failure):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise failure

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(RagModelUnavailable) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    assert caught.value.code == "rag_model_unavailable"
    assert caught.value.public_message == "The RAG model is temporarily unavailable."
    assert caught.value.retryable is True
    assert "private" not in f"{caught.value!r}\n{caught.value}"
    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.parametrize(
    "failure",
    (
        httpx.ReadTimeout("private timeout"),
        ExceptionGroup("private provider group", [RuntimeError("private one"), ValueError("private two")]),
    ),
)
@pytest.mark.asyncio
async def test_timeout_and_ordinary_exception_groups_are_fully_detached(failure):
    def handler(_request: httpx.Request) -> httpx.Response:
        raise failure

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(RagModelUnavailable) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.parametrize("status", (300, 307, 400, 401, 403, 404, 408, 422, 425, 429, 500, 503))
@pytest.mark.asyncio
async def test_non_success_status_is_retryable_and_no_redirect_is_followed(status):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, headers={"location": "https://private.test/secret"}, text="private body")

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(RagModelUnavailable) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )
    assert calls == 1
    assert caught.value.retryable is True
    _assert_detached_and_sanitized(caught.value)
    assert "private" not in f"{caught.value!r}\n{caught.value}"
    await client.aclose()


class _CountingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: int, chunk_size: int) -> None:
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.yielded = 0
        self.closed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            self.yielded += 1
            yield b"x" * self.chunk_size

    async def aclose(self) -> None:
        self.closed += 1


class _BytesStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.yielded = 0
        self.closed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.yielded += 1
        yield self.content

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("content_length", ("-1", "abc", "262145"))
@pytest.mark.asyncio
async def test_invalid_or_oversized_content_length_rejects_before_iteration(content_length):
    stream = _BytesStream(b"private body")
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, headers={"content-length": content_length}, stream=stream)
        ),
    )

    with pytest.raises(InvalidRagModelResponse):
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )
    assert stream.yielded == 0
    assert stream.closed == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_gzip_response_is_rejected_without_decompression_or_iteration():
    compressed = gzip.compress(b"x" * (32 * 1024 * 1024))
    assert len(compressed) < 64 * 1024
    stream = _BytesStream(compressed)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=stream)

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(InvalidRagModelResponse):
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    assert seen[0].headers["accept-encoding"] == "identity"
    assert stream.yielded == 0
    assert stream.closed == 1
    await client.aclose()


class _NeverEndingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.Event().wait()
        yield b"unreachable"

    async def aclose(self) -> None:
        self.closed += 1


class _DeadlineCleanupStream(httpx.AsyncByteStream):
    def __init__(self, *, close_delay: float | None = 0, close_failure: BaseException | None = None) -> None:
        self.close_delay = close_delay
        self.close_failure = close_failure
        self.iter_started = asyncio.Event()
        self.iter_cancelled = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_finished = asyncio.Event()
        self.release_close = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iter_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.iter_cancelled.set()
            raise
        yield b"unreachable"

    async def aclose(self) -> None:
        self.close_started.set()
        if self.close_failure is not None:
            raise self.close_failure
        if self.close_delay is None:
            await self.release_close.wait()
        elif self.close_delay:
            await asyncio.sleep(self.close_delay)
        self.close_finished.set()


@pytest.mark.asyncio
async def test_total_deadline_stops_slow_stream_and_closes_response():
    stream = _NeverEndingStream()
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)),
    )

    with pytest.raises(RagModelUnavailable) as caught:
        await asyncio.wait_for(
            client.complete_json(
                messages=({"role": "user", "content": "go"},),
                max_output_tokens=1,
                timeout_seconds=0.01,
            ),
            timeout=0.2,
        )

    assert caught.value.retryable is True
    assert stream.closed == 1
    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_deadline_returns_without_waiting_for_slow_response_close():
    stream = _DeadlineCleanupStream(close_delay=0.08)
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)),
    )
    started = asyncio.get_running_loop().time()

    with pytest.raises(RagModelUnavailable) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},),
            max_output_tokens=1,
            timeout_seconds=0.01,
        )

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.05
    assert caught.value.retryable is True
    await asyncio.wait_for(stream.iter_cancelled.wait(), timeout=0.1)
    await asyncio.wait_for(stream.close_started.wait(), timeout=0.1)
    await client.aclose()


@pytest.mark.asyncio
async def test_never_ending_response_close_does_not_hold_complete_json():
    stream = _DeadlineCleanupStream(close_delay=None)
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)),
    )

    with pytest.raises(RagModelUnavailable):
        await asyncio.wait_for(
            client.complete_json(
                messages=({"role": "user", "content": "go"},),
                max_output_tokens=1,
                timeout_seconds=0.01,
            ),
            timeout=0.05,
        )

    await asyncio.wait_for(stream.iter_cancelled.wait(), timeout=0.1)
    await asyncio.wait_for(stream.close_started.wait(), timeout=0.1)
    stream.release_close.set()
    await asyncio.wait_for(stream.close_finished.wait(), timeout=0.1)
    await client.aclose()


@pytest.mark.asyncio
async def test_deadline_ignores_response_close_failure_and_consumes_background_exception():
    stream = _DeadlineCleanupStream(close_failure=RuntimeError("private close failure"))
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)),
    )
    try:
        with pytest.raises(RagModelUnavailable) as caught:
            await client.complete_json(
                messages=({"role": "user", "content": "go"},),
                max_output_tokens=1,
                timeout_seconds=0.01,
            )
        await asyncio.wait_for(stream.close_started.wait(), timeout=0.1)
        await asyncio.sleep(0)
        assert caught.value.retryable is True
        _assert_detached_and_sanitized(caught.value)
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)
        await client.aclose()


@pytest.mark.asyncio
async def test_external_cancellation_cancels_and_tracks_request_child():
    stream = _DeadlineCleanupStream(close_delay=None)
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream)),
    )
    operation = asyncio.create_task(
        client.complete_json(
            messages=({"role": "user", "content": "go"},),
            max_output_tokens=1,
            timeout_seconds=1,
        )
    )
    await asyncio.wait_for(stream.iter_started.wait(), timeout=0.1)
    operation.cancel("private cancellation")

    with pytest.raises(asyncio.CancelledError) as caught:
        await operation

    assert caught.value.args == ()
    _assert_detached_and_sanitized(caught.value)
    await asyncio.wait_for(stream.iter_cancelled.wait(), timeout=0.1)
    await asyncio.wait_for(stream.close_started.wait(), timeout=0.1)
    stream.release_close.set()
    await asyncio.sleep(0)
    await client.aclose()


@pytest.mark.asyncio
async def test_response_is_streamed_and_stopped_at_256_kib_cap():
    stream = _CountingStream(chunks=100, chunk_size=16 * 1024)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    with pytest.raises(InvalidRagModelResponse):
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    assert stream.yielded == 17
    assert stream.closed == 1
    await client.aclose()


@pytest.mark.parametrize(
    "raw",
    (
        b'{"choices":[],"choices":[],"usage":{}}',
        b'{"choices":[],"usage":{},"x":NaN}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"x":1e999}',
        b'{"choices":[],"usage":{}}',
        b'{"choices":[{},{}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":null}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":1}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"[]"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"{\\"x\\":1,\\"x\\":2}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"{\\"x\\":NaN}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"{\\"x\\":1e999}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"{}"}}]}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":true,"completion_tokens":1,"total_tokens":2}}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":-1,"completion_tokens":1,"total_tokens":0}}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":3}}',
        b'{"choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"extra":0}}',
    ),
)
@pytest.mark.asyncio
async def test_malformed_outer_content_and_usage_are_invalid(raw):
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw)),
    )
    with pytest.raises(InvalidRagModelResponse) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )
    assert caught.value.code == "rag_model_invalid_response"
    assert caught.value.retryable is False
    assert raw.decode("utf-8", errors="ignore") not in str(caught.value)
    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_payload_preserves_only_exact_trusted_usage():
    raw = (
        b'{"choices":[{"message":{"content":"[]"}}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}'
    )
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw)),
    )

    with pytest.raises(InvalidRagModelResponse) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=20, timeout_seconds=1
        )

    assert caught.value.usage == RagTokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    await client.aclose()


@pytest.mark.parametrize(
    "values",
    [
        (True, 0, 0),
        (0, True, 0),
        (0, 0, True),
        (250_001, 0, 250_001),
        (0, 250_001, 250_001),
        (125_001, 125_000, 250_001),
    ],
)
def test_token_usage_rejects_non_exact_or_over_hard_cap_values(values: tuple[object, object, object]):
    with pytest.raises(ValueError):
        RagTokenUsage(prompt_tokens=values[0], completion_tokens=values[1], total_tokens=values[2])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_raw_private_response_json_decode_failure_is_fully_detached():
    raw = b'{"private-response":"secret-body"'
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=raw)),
    )

    with pytest.raises(InvalidRagModelResponse) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_strict_response_parser_sanitizes_linked_control_signal(monkeypatch):
    failure = RuntimeError("private parser failure")
    failure.__cause__ = KeyboardInterrupt("private parser interrupt")

    def fail_parse(*_args: object, **_kwargs: object) -> object:
        raise failure

    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"{}")),
    )
    monkeypatch.setattr(rag_model.json, "loads", fail_parse)

    with pytest.raises(KeyboardInterrupt) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )

    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_excessive_json_depth_is_invalid_at_both_layers():
    deep_content = "{" + '"x":{' * 40 + '"y":1' + "}" * 40 + "}"
    outer = json.dumps(
        {
            "choices": [{"message": {"content": deep_content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()
    client = OpenAICompatibleRagModel(
        PROFILE,
        api_key="secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=outer)),
    )
    with pytest.raises(InvalidRagModelResponse):
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_linked_and_grouped_control_signals_are_preserved_and_sanitized():
    hidden = asyncio.CancelledError("secret cancellation")
    wrapper = httpx.ReadError("private provider detail")
    wrapper.__cause__ = BaseExceptionGroup("private group", [RuntimeError("private"), hidden])

    def handler(_request: httpx.Request) -> httpx.Response:
        raise wrapper

    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=httpx.MockTransport(handler))
    with pytest.raises(asyncio.CancelledError) as caught:
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )
    assert caught.value.args == ()
    _assert_detached_and_sanitized(caught.value)
    await client.aclose()


class _CloseTransport(httpx.AsyncBaseTransport):
    def __init__(self, failures: tuple[BaseException, ...] = ()) -> None:
        self.close_calls = 0
        self.failures = list(failures)
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=VALID_RESPONSE)

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.failures:
            raise self.failures.pop(0)
        self.closed = True


class _ClosedThenFailsTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=VALID_RESPONSE)

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            self.closed = True
            raise RuntimeError("private indeterminate close failure")


@pytest.mark.asyncio
async def test_aclose_closes_once_and_after_close_calls_fail_safely():
    transport = _CloseTransport()
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    await client.aclose()
    await client.aclose()
    assert transport.close_calls == 1
    with pytest.raises(RagModelUnavailable):
        await client.complete_json(
            messages=({"role": "user", "content": "go"},), max_output_tokens=1, timeout_seconds=1
        )


@pytest.mark.asyncio
async def test_aclose_failure_is_sanitized_and_next_call_retries_to_success():
    transport = _CloseTransport((RuntimeError("secret close detail"),))
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    with pytest.raises(RagModelUnavailable) as caught:
        await client.aclose()
    await client.aclose()

    assert transport.close_calls == 2
    assert transport.closed is True
    _assert_detached_and_sanitized(caught.value)


@pytest.mark.asyncio
async def test_aclose_retries_even_if_transport_private_state_claims_closed():
    transport = _ClosedThenFailsTransport()
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    with pytest.raises(RagModelUnavailable) as caught:
        await client.aclose()
    await client.aclose()

    assert transport.close_calls == 2
    assert transport.closed is True
    _assert_detached_and_sanitized(caught.value)


@pytest.mark.asyncio
async def test_concurrent_and_repeated_close_calls_close_transport_once():
    transport = _CloseTransport()
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    await asyncio.gather(*(client.aclose() for _ in range(20)))
    await client.aclose()

    assert transport.close_calls == 1
    assert transport.closed is True


@pytest.mark.asyncio
async def test_aclose_preserves_sanitized_control_signal():
    failure = RuntimeError("secret close")
    failure.__cause__ = KeyboardInterrupt("secret interrupt")
    transport = _CloseTransport((failure,))
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    with pytest.raises(KeyboardInterrupt) as caught:
        await client.aclose()
    assert caught.value.args == ()
    _assert_detached_and_sanitized(caught.value)

    await client.aclose()
    assert transport.close_calls == 2
    assert transport.closed is True


@pytest.mark.asyncio
async def test_aclose_cancellation_is_detached_and_retryable():
    transport = _CloseTransport((asyncio.CancelledError("private cancellation"),))
    client = OpenAICompatibleRagModel(PROFILE, api_key="secret", transport=transport)

    with pytest.raises(asyncio.CancelledError) as caught:
        await client.aclose()
    _assert_detached_and_sanitized(caught.value)

    await client.aclose()
    assert transport.close_calls == 2
    assert transport.closed is True
