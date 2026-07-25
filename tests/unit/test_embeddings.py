import json
import math

import httpx
import pytest
from services.embeddings import OpenAIEmbeddingClient

from llmwiki_core.models import (
    EmbeddingInputError,
    EmbeddingProfile,
    EmbeddingUnavailable,
    InvalidEmbeddingResponse,
)

pytestmark = pytest.mark.asyncio


def _profile(dimensions: int = 3) -> EmbeddingProfile:
    return EmbeddingProfile(provider="openai_compatible", model="embed-v1", dimensions=dimensions)


def _response(request: httpx.Request, vectors: list[list[float]], *, indices=None) -> httpx.Response:
    if indices is None:
        indices = range(len(vectors))
    return httpx.Response(
        200,
        request=request,
        headers={"content-type": "application/json"},
        content=json.dumps(
            {"data": [{"index": index, "embedding": vector} for index, vector in zip(indices, vectors)]}
        ),
    )


async def test_openai_adapter_posts_to_embeddings_with_optional_bearer_auth():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request, [[1.0, 2.0, 3.0]])

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1/",
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert await client.embed(["safe fixture"]) == ((1.0, 2.0, 3.0),)

    request = requests[0]
    assert request.url == httpx.URL("https://embedding.test/v1/embeddings")
    assert request.headers["authorization"] == "Bearer secret-key"
    assert request.headers["content-type"] == "application/json"
    assert request.read() == b'{"input":["safe fixture"],"model":"embed-v1"}'


async def test_openai_adapter_omits_authorization_when_key_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return _response(request, [[1.0, 2.0, 3.0]])

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        await client.embed(["safe fixture"])


async def test_openai_adapter_preserves_order_across_batches_and_response_indices():
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batches.append(inputs)
        offset = sum(len(batch) for batch in batches[:-1])
        vectors = [[float(offset + index), 0.0, 1.0] for index in range(len(inputs))]
        return _response(request, list(reversed(vectors)), indices=reversed(range(len(inputs))))

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.embed(["one", "two", "three", "four", "five"])

    assert batches == [["one", "two"], ["three", "four"], ["five"]]
    assert result == (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (3.0, 0.0, 1.0),
        (4.0, 0.0, 1.0),
    )


@pytest.mark.parametrize(
    ("inputs", "vectors", "indices"),
    (
        (["one"], [[1.0, 2.0]], None),
        (["one", "two"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 0]),
        (["one", "two"], [[1.0, 2.0, 3.0]], [0]),
        (["one"], [[1.0, 2.0, 3.0]], [1]),
        (["one"], [[1.0, 2.0, 3.0]], [True]),
    ),
)
async def test_openai_adapter_rejects_wrong_dimensions_or_invalid_indices(inputs, vectors, indices):
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, vectors, indices=indices)

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(InvalidEmbeddingResponse, match="invalid embedding response"):
            await client.embed(inputs)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0"])
async def test_openai_adapter_rejects_nonfinite_or_non_numeric_coordinates(value):
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, [[value, 2.0, 3.0]])

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(InvalidEmbeddingResponse, match="invalid embedding response"):
            await client.embed(["safe fixture"])


@pytest.mark.parametrize(
    ("inputs", "kwargs"),
    (
        ([], {}),
        ([""], {}),
        (["ok", 3], {}),
        (["a", "b", "c"], {"max_inputs": 2}),
        (["abc", "def"], {"max_total_chars": 5}),
    ),
)
async def test_openai_adapter_bounds_and_validates_inputs(inputs, kwargs):
    transport = httpx.MockTransport(lambda request: pytest.fail("invalid input reached transport"))
    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/v1",
        transport=transport,
        **kwargs,
    ) as client:
        with pytest.raises(EmbeddingInputError, match="embedding input"):
            await client.embed(inputs)


async def test_openai_adapter_maps_timeout_to_sanitized_unavailable_error():
    api_key = "sk-private"
    private_input = "private user input"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timeout {api_key} {private_input} {request.url}", request=request)

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/private",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(EmbeddingUnavailable) as caught:
            await client.embed([private_input])

    message = str(caught.value)
    assert message == "embedding provider unavailable"
    assert api_key not in message
    assert private_input not in message
    assert "embedding.test" not in message


async def test_openai_adapter_sanitizes_backend_error_body_and_url():
    private_body = "private backend body"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, text=private_body)

    async with OpenAIEmbeddingClient(
        profile=_profile(),
        base_url="https://embedding.test/private",
        api_key="private-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(EmbeddingUnavailable) as caught:
            await client.embed(["private input"])

    assert str(caught.value) == "embedding provider unavailable"
    assert private_body not in str(caught.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "ftp://embedding.test/v1",
        "https:///v1",
        "https://embedding.test:private/v1",
        "https://user:password@embedding.test/v1",
        "https://embedding.test/v1?key=secret",
        "https://embedding.test/v1#fragment",
    ],
)
async def test_openai_adapter_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError, match="embedding base URL"):
        OpenAIEmbeddingClient(profile=_profile(), base_url=base_url)


async def test_openai_adapter_rejects_embed_after_close_without_leaking_details():
    client = OpenAIEmbeddingClient(profile=_profile(), base_url="https://embedding.test/v1")
    await client.aclose()

    with pytest.raises(EmbeddingUnavailable, match="embedding provider unavailable"):
        await client.embed(["private input"])


@pytest.mark.parametrize("api_key", ["private\r\nkey", "私密密钥"])
async def test_openai_adapter_rejects_unsafe_api_keys_with_sanitized_error(api_key):
    with pytest.raises(ValueError) as caught:
        OpenAIEmbeddingClient(
            profile=_profile(),
            base_url="https://embedding.test/v1",
            api_key=api_key,
        )

    assert str(caught.value) == "embedding API key contains invalid characters"
    assert api_key not in str(caught.value)
