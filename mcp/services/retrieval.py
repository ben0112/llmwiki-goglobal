"""Opt-in hosted hybrid retrieval composition for the MCP runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any, Literal

import httpx

from llmwiki_core.models import (
    EmbeddingError,
    EmbeddingProfile,
    EmbeddingUnavailable,
    InvalidEmbeddingResponse,
)
from llmwiki_core.search import (
    HybridRetrievalService,
    RetrieverUnavailable,
    SearchHit,
    SearchQuery,
    SearchResult,
)
from llmwiki_core.signals import sanitized_process_signal

logger = logging.getLogger(__name__)

RetrievalProfile = Literal["lexical", "hybrid"]
FallbackSignal = Callable[..., None]


class HostedRetrievalService:
    """Build the shared hybrid service without changing the lexical default."""

    def __init__(
        self,
        vault,
        knowledge_base_id: str,
        *,
        settings=None,
        embedding_client_factory: Callable[[], object] | None = None,
        reranker=None,
        fallback_signal: FallbackSignal | None = None,
    ) -> None:
        if settings is None:
            from config import settings as runtime_settings

            settings = runtime_settings
        self._vault = vault
        self._knowledge_base_id = knowledge_base_id
        self._settings = settings
        self._embedding_client_factory = embedding_client_factory
        self._reranker = reranker
        self._fallback_signal = fallback_signal or _log_fallback

    async def retrieve(
        self,
        query: SearchQuery,
        *,
        profile: RetrievalProfile = "lexical",
    ) -> SearchResult:
        if profile == "lexical":
            return await self._vault.retrieve(self._knowledge_base_id, query)
        if profile != "hybrid":
            raise ValueError("unsupported retrieval profile")
        embedding_profile = self._validated_hybrid_profile()

        lexical_limit, vector_limit = self._candidate_limits(query.limit)
        service_query = _query_with_candidate_limit(
            query, max(lexical_limit, vector_limit)
        )
        lexical = _LexicalRetriever(
            self._vault, self._knowledge_base_id, lexical_limit
        )
        vector = _VectorRetriever(
            self._vault,
            self._knowledge_base_id,
            vector_limit,
            embedding_profile,
            self._new_embedding_client,
        )
        reranker = (
            _PreservingReranker(self._reranker) if self._reranker is not None else None
        )
        expander = _GraphExpander(self._vault, self._knowledge_base_id, vector)
        service = HybridRetrievalService(
            lexical=lexical,
            vector=vector,
            reranker=reranker,
            expander=expander,
            rrf_k=self._settings.HYBRID_RRF_K,
        )
        result = await service.retrieve(service_query)
        if result.profile == "lexical_fallback":
            try:
                self._fallback_signal(
                    reason="vector_unavailable",
                    candidate_count=result.candidate_count,
                )
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception:  # noqa: BLE001 - signals cannot fail retrieval.
                pass
        return result

    def _validated_hybrid_profile(self) -> EmbeddingProfile:
        profile = getattr(self._settings, "embedding_profile", None)
        if (
            getattr(self._settings, "MODE", None) != "hosted"
            or getattr(self._settings, "HYBRID_SEARCH_ENABLED", False) is not True
            or not isinstance(profile, EmbeddingProfile)
        ):
            raise ValueError("hybrid retrieval is unavailable")
        return profile

    def _candidate_limits(self, request_limit: int) -> tuple[int, int]:
        helper = getattr(self._settings, "hybrid_candidate_limits", None)
        if callable(helper):
            return helper(request_limit)
        lexical = self._settings.HYBRID_LEXICAL_CANDIDATES
        vector = self._settings.HYBRID_VECTOR_CANDIDATES
        if request_limit > lexical or request_limit > vector:
            raise ValueError("hybrid candidate limits must be at least the result limit")
        return lexical, vector

    def _new_embedding_client(self):
        if self._embedding_client_factory is not None:
            client = self._embedding_client_factory()
        else:
            api_key = self._settings.EMBEDDING_API_KEY.get_secret_value()
            client = _OpenAICompatibleQueryEmbeddingClient(
                profile=self._validated_hybrid_profile(),
                base_url=self._settings.EMBEDDING_BASE_URL,
                api_key=api_key,
                timeout_seconds=self._settings.EMBEDDING_TIMEOUT_SECONDS,
            )
        return client


class _LexicalRetriever:
    def __init__(self, vault, knowledge_base_id: str, candidate_limit: int) -> None:
        self._vault = vault
        self._knowledge_base_id = knowledge_base_id
        self._candidate_limit = candidate_limit

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        return await self._vault.retrieve(
            self._knowledge_base_id,
            _query_with_candidate_limit(query, self._candidate_limit),
        )


class _VectorRetriever:
    def __init__(
        self,
        vault,
        knowledge_base_id: str,
        candidate_limit: int,
        profile: EmbeddingProfile,
        embedding_client_factory: Callable[[], object],
    ) -> None:
        self._vault = vault
        self._knowledge_base_id = knowledge_base_id
        self._candidate_limit = candidate_limit
        self._profile = profile
        self._embedding_client_factory = embedding_client_factory
        self.available: bool | None = None

    async def retrieve(self, query: SearchQuery) -> SearchResult:
        client = None
        result = None
        main_failure: BaseException | None = None
        close_failure: BaseException | None = None
        factory_failed = False
        try:
            client = self._embedding_client_factory()
            if getattr(client, "profile", None) != self._profile:
                raise EmbeddingUnavailable("query embedding is unavailable")
            vectors = await client.embed((query.text,))
            embedding = _validated_query_embedding(vectors, profile=self._profile)
            vector_query = _query_with_candidate_limit(query, self._candidate_limit)
            result = await self._vault.retrieve_vector(
                self._knowledge_base_id,
                vector_query,
                embedding=embedding,
                profile=self._profile,
            )
        except BaseException as failure:
            main_failure = failure
            factory_failed = client is None
        if client is not None:
            try:
                await _close_embedding_client(client)
            except BaseException as failure:
                close_failure = failure

        failures = tuple(
            failure
            for failure in (main_failure, close_failure)
            if failure is not None
        )
        if signal := sanitized_process_signal(*failures):
            self.available = False
            raise signal from None
        if main_failure is not None:
            self.available = False
            if factory_failed or isinstance(main_failure, (EmbeddingError, RetrieverUnavailable)):
                raise RetrieverUnavailable("query embedding is unavailable") from None
            raise main_failure
        if result is None:
            self.available = False
            raise RetrieverUnavailable("query embedding is unavailable")
        self.available = True
        return result


class _GraphExpander:
    def __init__(self, vault, knowledge_base_id: str, vector: _VectorRetriever) -> None:
        self._vault = vault
        self._knowledge_base_id = knowledge_base_id
        self._vector = vector

    async def expand(
        self, query: SearchQuery, hits: Sequence[SearchHit]
    ) -> Sequence[SearchHit]:
        if self._vector.available is not True:
            return ()
        remaining = max(0, query.limit - len(hits))
        if not remaining:
            return ()
        try:
            return await self._vault.expand_references(
                self._knowledge_base_id,
                query,
                tuple(hits),
                limit=remaining,
            )
        except RetrieverUnavailable:
            return ()


class _PreservingReranker:
    def __init__(self, reranker) -> None:
        self._reranker = reranker

    async def rerank(
        self, query: SearchQuery, hits: Sequence[SearchHit]
    ) -> Sequence[SearchHit]:
        try:
            output = await self._reranker.rerank(query, hits)
            iterator = iter(output)
            bounded: list[SearchHit] = []
            scan_limit = min(600, max(16, query.candidate_limit + query.limit))
            for _ in range(scan_limit + 1):
                try:
                    bounded.append(next(iterator))
                except StopIteration:
                    return tuple(bounded)
            return tuple(hits)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 - optional reranking must not break retrieval.
            return tuple(hits)


class _OpenAICompatibleQueryEmbeddingClient:
    """Small MCP-side query adapter; document embedding remains an API job."""

    def __init__(
        self,
        *,
        profile: EmbeddingProfile,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.profile = profile
        endpoint = _embedding_endpoint(base_url)
        validated_api_key = _validated_api_key(api_key)
        headers = (
            {"Authorization": f"Bearer {validated_api_key}"}
            if validated_api_key
            else None
        )
        self._endpoint = endpoint
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        )

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        try:
            response = await self._client.post(
                self._endpoint,
                json={"input": list(texts), "model": self.profile.model},
            )
        except (httpx.HTTPError, RuntimeError):
            raise EmbeddingUnavailable("embedding provider unavailable") from None
        if not 200 <= response.status_code < 300:
            raise EmbeddingUnavailable("embedding provider unavailable")
        try:
            payload = response.json()
            return _ordered_vectors(
                payload,
                expected_count=len(texts),
                dimensions=self.profile.dimensions,
            )
        except (TypeError, ValueError, KeyError, OverflowError):
            raise InvalidEmbeddingResponse("invalid embedding response") from None

    async def aclose(self) -> None:
        await self._client.aclose()


async def _close_embedding_client(client: object) -> None:
    close = getattr(client, "aclose", None)
    if callable(close):
        await close()


def _embedding_endpoint(base_url: object) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("embedding base URL is required")
    try:
        url = httpx.URL(base_url.strip())
    except (httpx.InvalidURL, TypeError, ValueError):
        raise ValueError("embedding base URL is invalid") from None
    if (
        url.scheme not in ("http", "https")
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("embedding base URL is invalid")
    return f"{str(url).rstrip('/')}/embeddings"


def _validated_api_key(api_key: object) -> str:
    if not isinstance(api_key, str):
        raise ValueError("embedding API key is invalid")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("embedding API key is invalid") from None
    if "\r" in api_key or "\n" in api_key:
        raise ValueError("embedding API key is invalid")
    return api_key


def _ordered_vectors(
    payload: Any, *, expected_count: int, dimensions: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise ValueError("invalid embedding response")
    data = payload["data"]
    if len(data) != expected_count:
        raise ValueError("invalid embedding response")
    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("invalid embedding response")
        index = item.get("index")
        raw = item.get("embedding")
        if (
            type(index) is not int
            or not 0 <= index < expected_count
            or ordered[index] is not None
            or not isinstance(raw, list)
            or len(raw) != dimensions
        ):
            raise ValueError("invalid embedding response")
        vector = tuple(float(value) for value in raw)
        if any(type(value) not in (int, float) for value in raw) or not all(
            isfinite(value) for value in vector
        ):
            raise ValueError("invalid embedding response")
        ordered[index] = vector
    if any(vector is None for vector in ordered):
        raise ValueError("invalid embedding response")
    return tuple(vector for vector in ordered if vector is not None)


def _validated_query_embedding(
    vectors: object,
    *,
    profile: EmbeddingProfile,
) -> tuple[float, ...]:
    if (
        isinstance(vectors, (str, bytes))
        or not isinstance(vectors, Sequence)
        or len(vectors) != 1
        or isinstance(vectors[0], (str, bytes))
        or not isinstance(vectors[0], Sequence)
        or len(vectors[0]) != profile.dimensions
    ):
        raise InvalidEmbeddingResponse("invalid embedding response")
    embedding: list[float] = []
    for coordinate in vectors[0]:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise InvalidEmbeddingResponse("invalid embedding response")
        try:
            normalized = float(coordinate)
        except (OverflowError, ValueError):
            raise InvalidEmbeddingResponse("invalid embedding response") from None
        if not isfinite(normalized):
            raise InvalidEmbeddingResponse("invalid embedding response")
        embedding.append(normalized)
    if not any(embedding):
        raise InvalidEmbeddingResponse("invalid embedding response")
    return tuple(embedding)


def _query_with_candidate_limit(query: SearchQuery, candidate_limit: int) -> SearchQuery:
    return SearchQuery.build(
        text=query.text,
        limit=query.limit,
        candidate_limit=candidate_limit,
        area=query.area,
        scope=query.scope,
        facets=query.facets,
        path_glob=query.path_glob,
        tags=query.tags,
        document_kinds=query.document_kinds,
        annotated_only=query.annotated_only,
    )


def _log_fallback(*, reason: str, candidate_count: int) -> None:
    logger.info(
        "retrieval fallback reason=%s candidate_count=%d",
        reason,
        candidate_count,
    )


__all__ = ["HostedRetrievalService", "RetrievalProfile"]
