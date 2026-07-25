import asyncio
from types import SimpleNamespace

import pytest

from llmwiki_core.documents import DocumentKind
from llmwiki_core.models import EmbeddingProfile, EmbeddingUnavailable
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery, SearchResult

PROFILE = EmbeddingProfile("openai_compatible", "embed-v1", 3)


def _settings(**overrides):
    values = {
        "MODE": "hosted",
        "HYBRID_SEARCH_ENABLED": True,
        "embedding_profile": PROFILE,
        "HYBRID_LEXICAL_CANDIDATES": 20,
        "HYBRID_VECTOR_CANDIDATES": 30,
        "HYBRID_RRF_K": 60,
        "EMBEDDING_BASE_URL": "https://embeddings.invalid/v1",
        "EMBEDDING_API_KEY": SimpleNamespace(get_secret_value=lambda: "secret"),
        "EMBEDDING_BATCH_SIZE": 32,
        "EMBEDDING_TIMEOUT_SECONDS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _hit(document_id, *, chunk=0, score=1.0, path=None, metadata=None):
    return SearchHit(
        document_id=document_id,
        document_version=1,
        chunk_index=chunk,
        content=f"content-{document_id}-{chunk}",
        score=score,
        path=path or f"/{document_id}.md",
        document_kind=DocumentKind.SOURCE,
        metadata={} if metadata is None else metadata,
    )


class _EmbeddingClient:
    profile = PROFILE

    def __init__(self, *, failure=None):
        self.failure = failure
        self.texts = []

    async def embed(self, texts):
        self.texts.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        return ((1.0, 0.0, 0.0),)


class _RawEmbeddingClient:
    profile = PROFILE

    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, _texts):
        return self.vectors


class _Vault:
    def __init__(self):
        self.lexical = SearchResult((_hit("lexical"),), 1, profile="lexical")
        self.vector = SearchResult((_hit("vector"),), 1, profile="vector")
        self.lexical_queries = []
        self.vector_queries = []
        self.vector_profile = None
        self.vector_embedding = None
        self.expansions = []

    async def retrieve(self, kb_id, query):
        self.lexical_queries.append((kb_id, query))
        return self.lexical

    async def retrieve_vector(self, kb_id, query, *, embedding, profile):
        self.vector_queries.append((kb_id, query))
        self.vector_profile = profile
        self.vector_embedding = embedding
        if isinstance(self.vector, BaseException):
            raise self.vector
        return self.vector

    async def expand_references(self, kb_id, query, hits, *, limit):
        self.expansions.append((kb_id, query, tuple(hits), limit))
        return ()


@pytest.mark.asyncio
async def test_lexical_profile_preserves_exact_adapter_result_and_never_builds_embeddings():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    built = []
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(MODE="local", HYBRID_SEARCH_ENABLED=False, embedding_profile=None),
        embedding_client_factory=lambda: built.append(True),
    )
    query = SearchQuery.build(text="permit", limit=1)

    result = await service.retrieve(query, profile="lexical")

    assert result is vault.lexical
    assert vault.lexical_queries == [("kb-1", query)]
    assert built == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        _settings(MODE="local"),
        _settings(HYBRID_SEARCH_ENABLED=False),
        _settings(embedding_profile=None),
    ],
)
async def test_hybrid_requires_valid_opt_in_hosted_configuration(settings):
    from services.retrieval import HostedRetrievalService

    service = HostedRetrievalService(_Vault(), "kb-1", settings=settings)

    with pytest.raises(ValueError, match="hybrid retrieval is unavailable"):
        await service.retrieve(SearchQuery.build(text="permit"), profile="hybrid")


@pytest.mark.asyncio
async def test_unknown_profile_is_strictly_rejected():
    from services.retrieval import HostedRetrievalService

    service = HostedRetrievalService(_Vault(), "kb-1", settings=_settings())

    with pytest.raises(ValueError, match="unsupported retrieval profile"):
        await service.retrieve(SearchQuery.build(text="permit"), profile="semantic")


@pytest.mark.asyncio
async def test_hybrid_forwards_identical_filters_and_configured_candidate_limits():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    client = _EmbeddingClient()
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=lambda: client,
    )
    query = SearchQuery.build(
        text="permit",
        limit=2,
        candidate_limit=2,
        path_glob="/target/*.pdf",
        tags=["Reviewed", "science"],
        document_kinds=["source"],
        annotated_only=True,
        scope="all",
        facets={"country": "IDN"},
    )

    result = await service.retrieve(query, profile="hybrid")

    lexical = vault.lexical_queries[0][1]
    vector = vault.vector_queries[0][1]
    for field in (
        "text", "path_glob", "tags", "area", "scope", "facets",
        "document_kinds", "annotated_only",
    ):
        assert getattr(lexical, field) == getattr(vector, field) == getattr(query, field)
    assert lexical.candidate_limit == 20
    assert vector.candidate_limit == 30
    assert vault.vector_profile is PROFILE
    assert vault.vector_embedding == (1.0, 0.0, 0.0)
    assert client.texts == [("permit",)]
    assert result.profile == "hybrid"


@pytest.mark.asyncio
async def test_typed_embedding_unavailability_falls_back_with_safe_structured_signal():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    events = []
    client = _EmbeddingClient(failure=EmbeddingUnavailable("private query key URL"))
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=lambda: client,
        fallback_signal=lambda **fields: events.append(fields),
    )

    result = await service.retrieve(SearchQuery.build(text="super secret", limit=2), profile="hybrid")

    assert result.hits == vault.lexical.hits
    assert result.profile == "lexical_fallback"
    assert events == [{"reason": "vector_unavailable", "candidate_count": 1}]
    assert "secret" not in repr(events).lower()
    assert vault.expansions == []


@pytest.mark.asyncio
async def test_typed_vector_unavailability_falls_back_but_lexical_failure_is_visible():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    vault.vector = RetrieverUnavailable("pgvector unavailable")
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=_EmbeddingClient,
    )
    result = await service.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")
    assert result.profile == "lexical_fallback"

    async def broken_lexical(_kb, _query):
        raise RuntimeError("lexical failed")

    vault.retrieve = broken_lexical
    with pytest.raises(RuntimeError, match="lexical failed"):
        await service.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [((1.0, 2.0),), ((0.0, 0.0, 0.0),), ((float("nan"), 0.0, 1.0),)],
)
async def test_invalid_query_embedding_fails_closed_to_lexical(vectors):
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=lambda: _RawEmbeddingClient(vectors),
    )

    result = await service.retrieve(
        SearchQuery.build(text="q", limit=1), profile="hybrid"
    )

    assert result.profile == "lexical_fallback"
    assert vault.vector_queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.CancelledError()])
async def test_process_signals_are_never_downgraded_to_fallback(failure):
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    vault.vector = failure
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=_EmbeddingClient,
    )

    with pytest.raises(type(failure)):
        await service.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")


@pytest.mark.asyncio
async def test_hybrid_collapses_chunk_identity_without_polluting_user_metadata():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    metadata = {"citation": "source-1"}
    shared = _hit("shared", metadata=metadata)
    vault.lexical = SearchResult((shared,), 1)
    vault.vector = SearchResult((_hit("shared", metadata={"internal": "ignored"}),), 1)
    service = HostedRetrievalService(
        vault, "kb-1", settings=_settings(), embedding_client_factory=_EmbeddingClient
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2), profile="hybrid")

    assert [hit.identity for hit in result.hits] == [shared.identity]
    assert dict(result.hits[0].metadata) == metadata


@pytest.mark.asyncio
async def test_injected_reranker_may_only_reorder_known_bounded_hits():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    a, b = _hit("a"), _hit("b")
    vault.lexical = SearchResult((a, b), 2)
    vault.vector = SearchResult((), 0)

    class Reranker:
        async def rerank(self, query, hits):
            return (_hit("injected"), hits[1], hits[1], hits[0])

    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=_EmbeddingClient,
        reranker=Reranker(),
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2), profile="hybrid")

    assert [hit.document_id for hit in result.hits] == ["b", "a"]


@pytest.mark.asyncio
async def test_graph_expansion_only_fills_remaining_budget_after_direct_hits():
    from services.retrieval import HostedRetrievalService

    vault = _Vault()
    direct = _hit("direct")
    related = _hit("related")
    vault.lexical = SearchResult((direct,), 1)
    vault.vector = SearchResult((), 0)

    async def expand(_kb, _query, hits, *, limit):
        assert tuple(hit.identity for hit in hits) == (direct.identity,)
        assert limit == 1
        return (direct, related, _hit("overflow"))

    vault.expand_references = expand
    service = HostedRetrievalService(
        vault, "kb-1", settings=_settings(), embedding_client_factory=_EmbeddingClient
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2), profile="hybrid")

    assert [hit.document_id for hit in result.hits] == ["direct", "related"]
