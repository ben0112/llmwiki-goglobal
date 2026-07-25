"""OpenAI-compatible HTTP embedding adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import httpx

from llmwiki_core.models import (
    EmbeddingInputError,
    EmbeddingProfile,
    EmbeddingUnavailable,
    InvalidEmbeddingResponse,
)

DEFAULT_MAX_INPUTS = 512
DEFAULT_MAX_TOTAL_CHARS = 200_000
_INVALID_RESPONSE = "invalid embedding response"
_UNAVAILABLE = "embedding provider unavailable"


class OpenAIEmbeddingClient:
    """Bounded, ordered adapter for an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        profile: EmbeddingProfile,
        base_url: str,
        api_key: str = "",
        batch_size: int = 32,
        timeout_seconds: float = 30,
        max_inputs: int = DEFAULT_MAX_INPUTS,
        max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(profile, EmbeddingProfile):
            raise ValueError("embedding profile is required")
        if profile.provider != "openai_compatible":
            raise ValueError("embedding profile provider is unsupported")
        self._profile = profile
        self._endpoint = _embedding_endpoint(base_url)
        validated_api_key = _validated_api_key(api_key)
        self._batch_size = _bounded_integer(batch_size, label="embedding batch size", maximum=512)
        self._max_inputs = _bounded_integer(max_inputs, label="maximum embedding inputs", maximum=10_000)
        self._max_total_chars = _bounded_integer(
            max_total_chars,
            label="maximum embedding input characters",
            maximum=10_000_000,
        )
        if type(timeout_seconds) not in (int, float) or not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("embedding timeout must be a positive finite number")
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(float(timeout_seconds)),
            follow_redirects=False,
            trust_env=False,
            headers={"Authorization": f"Bearer {validated_api_key}"} if validated_api_key else None,
        )

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    async def __aenter__(self) -> OpenAIEmbeddingClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        normalized = self._validated_inputs(texts)
        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(normalized), self._batch_size):
            vectors.extend(await self._embed_batch(normalized[offset : offset + self._batch_size]))
        return tuple(vectors)

    def _validated_inputs(self, texts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise EmbeddingInputError("embedding input must be a sequence of text")
        if not texts or len(texts) > self._max_inputs:
            raise EmbeddingInputError("embedding input count is outside the supported range")
        normalized: list[str] = []
        total_chars = 0
        for text in texts:
            if not isinstance(text, str) or not text:
                raise EmbeddingInputError("embedding input must contain non-empty text")
            total_chars += len(text)
            if total_chars > self._max_total_chars:
                raise EmbeddingInputError("embedding input character limit exceeded")
            normalized.append(text)
        return tuple(normalized)

    async def _embed_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        try:
            response = await self._client.post(
                self._endpoint,
                json={"input": list(texts), "model": self.profile.model},
            )
        except httpx.HTTPError:
            raise EmbeddingUnavailable(_UNAVAILABLE) from None
        except RuntimeError:
            raise EmbeddingUnavailable(_UNAVAILABLE) from None

        if not 200 <= response.status_code < 300:
            raise EmbeddingUnavailable(_UNAVAILABLE)
        try:
            payload = response.json()
        except (ValueError, TypeError):
            raise InvalidEmbeddingResponse(_INVALID_RESPONSE) from None
        return _ordered_vectors(payload, expected_count=len(texts), dimensions=self.profile.dimensions)


def _embedding_endpoint(base_url: object) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("embedding base URL is required")
    try:
        url = httpx.URL(base_url.strip())
    except (httpx.InvalidURL, TypeError, ValueError):
        raise ValueError("embedding base URL is invalid") from None
    if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("embedding base URL is invalid")
    return f"{str(url).rstrip('/')}/embeddings"


def _validated_api_key(api_key: object) -> str:
    if not isinstance(api_key, str):
        raise ValueError("embedding API key must be text")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("embedding API key contains invalid characters") from None
    if "\r" in api_key or "\n" in api_key:
        raise ValueError("embedding API key contains invalid characters")
    return api_key


def _bounded_integer(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}")
    return value


def _ordered_vectors(
    payload: Any,
    *,
    expected_count: int,
    dimensions: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(payload, Mapping) or not isinstance(data := payload.get("data"), list):
        raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
    if len(data) != expected_count:
        raise InvalidEmbeddingResponse(_INVALID_RESPONSE)

    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for item in data:
        if not isinstance(item, Mapping):
            raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
        index = item.get("index")
        embedding = item.get("embedding")
        if type(index) is not int or not 0 <= index < expected_count or ordered[index] is not None:
            raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
        vector: list[float] = []
        for coordinate in embedding:
            if type(coordinate) not in (int, float) or not isfinite(coordinate):
                raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
            vector.append(float(coordinate))
        ordered[index] = tuple(vector)
    if any(vector is None for vector in ordered):
        raise InvalidEmbeddingResponse(_INVALID_RESPONSE)
    return tuple(vector for vector in ordered if vector is not None)


__all__ = ["OpenAIEmbeddingClient"]
