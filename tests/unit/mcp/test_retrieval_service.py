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


class _BoundaryEmbeddingClient:
    profile = PROFILE

    def __init__(self, *, embed_failure=None, close_failure=None):
        self.embed_failure = embed_failure
        self.close_failure = close_failure

    async def embed(self, _texts):
        if self.embed_failure is not None:
            raise self.embed_failure
        return ((1.0, 0.0, 0.0),)

    async def aclose(self):
        if self.close_failure is not None:
            raise self.close_failure


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
    [
        ((1.0, 2.0),),
        ((0.0, 0.0, 0.0),),
        ((float("nan"), 0.0, 1.0),),
        ((10**10_000, 0.0, 1.0),),
    ],
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


def test_process_signal_sanitizer_recurses_with_fixed_priority_and_safe_exit_codes():
    from llmwiki_core.signals import sanitized_process_signal

    private = RuntimeError("private backend URL and key")
    private.__cause__ = asyncio.CancelledError("private cancellation")
    system_exit = SystemExit("private unsafe exit")
    group = BaseExceptionGroup("private group", [private, system_exit])
    hidden_keyboard = RuntimeError("private wrapper")
    hidden_keyboard.__context__ = KeyboardInterrupt("private interrupt")

    signal = sanitized_process_signal(group, hidden_keyboard)
    assert type(signal) is KeyboardInterrupt
    assert signal.args == ()
    assert signal.__cause__ is None and signal.__context__ is None

    assert sanitized_process_signal(SystemExit(True)).code == 1
    assert sanitized_process_signal(SystemExit(False)).code == 0
    assert sanitized_process_signal(SystemExit(17)).code == 17
    assert sanitized_process_signal(SystemExit(None)).code == 1
    assert sanitized_process_signal(SystemExit("private")).code == 1
    cycle = RuntimeError("private cycle")
    cycle.__cause__ = cycle
    assert sanitized_process_signal(cycle) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [
        "embed_direct",
        "embed_linked",
        "embed_group",
        "close_direct",
        "close_linked",
        "close_group",
        "factory_direct",
        "factory_linked",
        "factory_group",
        "mixed",
    ],
)
async def test_hidden_cancellation_in_embedding_or_cleanup_is_never_swallowed(location):
    from services.retrieval import HostedRetrievalService

    linked = EmbeddingUnavailable("private embedding failure")
    linked.__cause__ = asyncio.CancelledError("private cancellation")
    group = BaseExceptionGroup(
        "private group", [EmbeddingUnavailable("private"), asyncio.CancelledError("private")]
    )
    embed_failure = (
        asyncio.CancelledError("private")
        if location == "embed_direct"
        else linked
        if location == "embed_linked"
        else group
        if location == "embed_group"
        else None
    )
    close_failure = (
        asyncio.CancelledError("private")
        if location == "close_direct"
        else linked
        if location == "close_linked"
        else group
        if location == "close_group"
        else None
    )
    if location == "mixed":
        embed_failure = EmbeddingUnavailable("private ordinary")
        close_failure = group
    client = _BoundaryEmbeddingClient(
        embed_failure=embed_failure,
        close_failure=close_failure,
    )
    factory_failure = (
        asyncio.CancelledError("private")
        if location == "factory_direct"
        else linked
        if location == "factory_linked"
        else group
        if location == "factory_group"
        else None
    )

    def factory():
        if factory_failure is not None:
            raise factory_failure
        return client

    service = HostedRetrievalService(
        _Vault(),
        "kb-1",
        settings=_settings(),
        embedding_client_factory=factory,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await service.retrieve(SearchQuery.build(text="private query", limit=1), profile="hybrid")

    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_ordinary_cleanup_failure_never_masks_success_or_typed_fallback():
    from services.retrieval import HostedRetrievalService

    success = HostedRetrievalService(
        _Vault(),
        "kb-1",
        settings=_settings(),
        embedding_client_factory=lambda: _BoundaryEmbeddingClient(
            close_failure=RuntimeError("private close")
        ),
    )
    assert (await success.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")).profile == "hybrid"

    fallback = HostedRetrievalService(
        _Vault(),
        "kb-1",
        settings=_settings(),
        embedding_client_factory=lambda: _BoundaryEmbeddingClient(
            embed_failure=EmbeddingUnavailable("private embed"),
            close_failure=RuntimeError("private close"),
        ),
    )
    assert (await fallback.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")).profile == "lexical_fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", ["秘密-key", "private\r\nkey"])
async def test_invalid_api_key_factory_failure_is_sanitized_to_lexical_fallback(api_key):
    from services.retrieval import HostedRetrievalService

    settings = _settings(
        EMBEDDING_API_KEY=SimpleNamespace(get_secret_value=lambda: api_key),
        EMBEDDING_BASE_URL="https://private.invalid/v1",
    )
    result = await HostedRetrievalService(_Vault(), "kb-1", settings=settings).retrieve(
        SearchQuery.build(text="private query", limit=1), profile="hybrid"
    )

    assert result.profile == "lexical_fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("bug"), TypeError("bug"), AssertionError("bug")])
async def test_injected_factory_programming_errors_remain_visible(failure):
    from services.retrieval import HostedRetrievalService

    def factory():
        raise failure

    service = HostedRetrievalService(
        _Vault(), "kb-1", settings=_settings(), embedding_client_factory=factory
    )

    with pytest.raises(type(failure), match="bug"):
        await service.retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["fallback_signal", "graph", "reranker"])
@pytest.mark.parametrize("shape", ["linked", "group"])
async def test_optional_callback_hidden_cancellation_is_sanitized(boundary, shape):
    from services.retrieval import HostedRetrievalService

    hidden = RuntimeError("private callback")
    hidden.__cause__ = asyncio.CancelledError("private cancellation")
    failure = (
        hidden
        if shape == "linked"
        else BaseExceptionGroup(
            "private group", [RuntimeError("private"), asyncio.CancelledError("private")]
        )
    )
    vault = _Vault()
    kwargs = {}
    query = SearchQuery.build(text="q", limit=1)
    if boundary == "fallback_signal":
        vault.vector = RetrieverUnavailable("unavailable")

        def callback(**_fields):
            raise failure

        kwargs["fallback_signal"] = callback
    elif boundary == "graph":
        vault.lexical = SearchResult((_hit("direct"),), 1)
        vault.vector = SearchResult((), 0)
        query = SearchQuery.build(text="q", limit=2)

        async def expand(*_args, **_kwargs):
            raise failure

        vault.expand_references = expand
    else:
        class Reranker:
            async def rerank(self, _query, _hits):
                raise failure

        kwargs["reranker"] = Reranker()
    service = HostedRetrievalService(
        vault,
        "kb-1",
        settings=_settings(),
        embedding_client_factory=_EmbeddingClient,
        **kwargs,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await service.retrieve(query, profile="hybrid")

    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["factory", "embed", "close", "fallback_signal", "graph", "reranker"]
)
async def test_generator_exit_is_never_treated_as_an_ordinary_callback_error(boundary):
    from services.retrieval import HostedRetrievalService

    failure = GeneratorExit("private generator")
    vault = _Vault()
    kwargs = {}
    query = SearchQuery.build(text="q", limit=1)
    if boundary == "factory":
        def factory():
            raise failure

        kwargs["embedding_client_factory"] = factory
    elif boundary in ("embed", "close"):
        kwargs["embedding_client_factory"] = lambda: _BoundaryEmbeddingClient(
            embed_failure=failure if boundary == "embed" else None,
            close_failure=failure if boundary == "close" else None,
        )
    elif boundary == "fallback_signal":
        vault.vector = RetrieverUnavailable("unavailable")

        def callback(**_fields):
            raise failure

        kwargs["fallback_signal"] = callback
    elif boundary == "graph":
        vault.lexical = SearchResult((_hit("direct"),), 1)
        vault.vector = SearchResult((), 0)
        query = SearchQuery.build(text="q", limit=2)

        async def expand(*_args, **_kwargs):
            raise failure

        vault.expand_references = expand
    else:
        class Reranker:
            async def rerank(self, _query, _hits):
                raise failure

        kwargs["reranker"] = Reranker()
    kwargs.setdefault("embedding_client_factory", _EmbeddingClient)
    service = HostedRetrievalService(vault, "kb-1", settings=_settings(), **kwargs)

    expected = (GeneratorExit, RuntimeError) if boundary in ("factory", "embed", "close") else GeneratorExit
    with pytest.raises(expected) as caught:
        await service.retrieve(query, profile="hybrid")
    assert not isinstance(caught.value, RetrieverUnavailable)


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
