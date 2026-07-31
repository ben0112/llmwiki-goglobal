import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from datetime import date
from types import SimpleNamespace
from uuid import UUID

import pytest

from llmwiki_core.documents import DocumentKind
from llmwiki_core.models import EmbeddingProfile, EmbeddingUnavailable
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery, SearchResult

USER_ID = UUID("10000000-0000-0000-0000-000000000001")
KNOWLEDGE_BASE_ID = UUID("20000000-0000-0000-0000-000000000002")
DOCUMENT_ID = UUID("30000000-0000-0000-0000-000000000003")
PROFILE = EmbeddingProfile("openai_compatible", "embed-v1", 3)


def _settings(**overrides):
    values = {
        "MODE": "hosted",
        "HYBRID_SEARCH_ENABLED": True,
        "embedding_profile": PROFILE,
        "HYBRID_LEXICAL_CANDIDATES": 4,
        "HYBRID_VECTOR_CANDIDATES": 5,
        "HYBRID_RRF_K": 60,
        "EMBEDDING_BASE_URL": "https://embedding.invalid/v1",
        "EMBEDDING_API_KEY": SimpleNamespace(get_secret_value=lambda: "secret"),
        "EMBEDDING_BATCH_SIZE": 32,
        "EMBEDDING_TIMEOUT_SECONDS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_hosted_rag_retrieval_settings_is_shared_strict_and_pure():
    from rag.retrieval import HostedRagRetrievalSettings, validate_hosted_rag_retrieval_settings

    snapshot = validate_hosted_rag_retrieval_settings(_settings(), request_limit=3)
    assert snapshot == HostedRagRetrievalSettings(
        embedding_profile=PROFILE,
        lexical_limit=4,
        vector_limit=5,
        rrf_k=60,
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.rrf_k = 1  # type: ignore[misc]
    invalid_settings = (
        _settings(MODE="local"),
        _settings(HYBRID_SEARCH_ENABLED=False),
        _settings(embedding_profile=object()),
        _settings(HYBRID_LEXICAL_CANDIDATES=True),
        _settings(HYBRID_VECTOR_CANDIDATES=501),
        _settings(HYBRID_RRF_K=True),
        _settings(HYBRID_RRF_K=1_000_001),
    )
    for settings in invalid_settings:
        with pytest.raises(ValueError, match="hybrid retrieval is unavailable") as exc_info:
            validate_hosted_rag_retrieval_settings(settings, request_limit=3)
        assert exc_info.value.__cause__ is exc_info.value.__context__ is None


def test_validate_hosted_rag_retrieval_settings_checks_candidate_helper_result():
    from rag.retrieval import validate_hosted_rag_retrieval_settings

    for result in ([4, 5], (True, 5), (2, 5), (4, 501), (4,), (4, 5, 6)):
        settings = _settings()
        settings.hybrid_candidate_limits = lambda _limit, result=result: result
        with pytest.raises(ValueError, match="hybrid retrieval is unavailable"):
            validate_hosted_rag_retrieval_settings(settings, request_limit=3)

    with pytest.raises(ValueError, match="hybrid retrieval is unavailable"):
        validate_hosted_rag_retrieval_settings(_settings(), request_limit=True)


def test_validate_hosted_rag_retrieval_settings_fails_closed_on_hostile_exception_graph():
    from rag.retrieval import validate_hosted_rag_retrieval_settings

    class HostileFailure(RuntimeError):
        def __getattribute__(self, name):
            if name in {"__cause__", "__context__"}:
                raise RuntimeError("private settings graph secret")
            return super().__getattribute__(name)

    settings = _settings()
    settings.hybrid_candidate_limits = lambda _limit: (_ for _ in ()).throw(HostileFailure("private settings secret"))

    with pytest.raises(ValueError, match="hybrid retrieval is unavailable") as exc_info:
        validate_hosted_rag_retrieval_settings(settings, request_limit=3)
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None


def _row(document_id=DOCUMENT_ID, *, score=3.5, candidate_count=1):
    return {
        "document_id": document_id,
        "document_version": 2,
        "chunk_index": 4,
        "content": "bounded source",
        "score": score,
        "path": "/corpus/",
        "filename": "source.pdf",
        "title": "Source",
        "page": 5,
        "header_breadcrumb": "Heading",
        "tags": ["Policy"],
        "source_kind": "source",
        "metadata": {"country": "IDN"},
        "candidate_count": candidate_count,
    }


def _hit(
    document_id=DOCUMENT_ID,
    *,
    version=2,
    chunk=4,
    score=0.75,
    path="/search/transport.md",
    document_kind=DocumentKind.SOURCE,
):
    return SearchHit(
        document_id=str(document_id),
        document_version=version,
        chunk_index=chunk,
        content="search-only content",
        score=score,
        path=path,
        document_kind=document_kind,
    )


class _OrdinaryBaseFailure(BaseException):
    pass


class _ExplodingMapping(Mapping):
    def __iter__(self):
        return iter(("secret-key",))

    def __len__(self):
        return 1

    def __getitem__(self, _key):
        raise RuntimeError("private mapping secret")


class _CancellingMapping(_ExplodingMapping):
    def __getitem__(self, _key):
        raise asyncio.CancelledError("private mapping cancellation")


class _ExplodingSequence(Sequence):
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        raise RuntimeError("private sequence secret")


class _CancellingSequence(_ExplodingSequence):
    def __getitem__(self, _index):
        raise asyncio.CancelledError("private sequence cancellation")


class _LenExplodingSequence(_ExplodingSequence):
    def __len__(self):
        raise RuntimeError("private length secret")


class _IterationExplodingSequence(_ExplodingSequence):
    def __iter__(self):
        raise RuntimeError("private iteration secret")


class _ExplodingFloat(float):
    def __float__(self):
        raise RuntimeError("private score secret")


def _evidence_row(
    document_id=DOCUMENT_ID,
    *,
    version=2,
    chunk=4,
    ordinal=1,
    content="fresh",
):
    return {
        "ordinal": ordinal,
        "document_id": document_id,
        "document_version": version,
        "chunk_index": chunk,
        "page": 5,
        "filename": "source.pdf",
        "path": "/corpus/",
        "title": "Source",
        "content": content,
        "status": "ready",
        "archived": False,
        "tags": ["policy"],
        "source_kind": "source",
        "metadata": {"country": "IDN"},
    }


class _RecordingDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.closed = False

    async def fetch(self, sql, *params):
        self.calls.append((sql, params))
        return self.rows

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_lexical_retriever_executes_the_shared_compiled_query_without_owning_database(
    monkeypatch,
):
    from rag.retrieval import PostgresLexicalRetriever

    compiled = SimpleNamespace(sql="SELECT shared_compiler", params=("a", "b"))
    compile_calls = []

    def compile_query(user_id, knowledge_base_id, query):
        compile_calls.append((user_id, knowledge_base_id, query))
        return compiled

    monkeypatch.setattr(
        "rag.retrieval.compile_postgres_lexical_query",
        compile_query,
    )
    database = _RecordingDatabase([_row(candidate_count=7)])
    query = SearchQuery.build(text="permit", limit=1, candidate_limit=4)

    result = await PostgresLexicalRetriever(
        database,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
    ).retrieve(query)

    assert compile_calls == [(USER_ID, KNOWLEDGE_BASE_ID, query)]
    assert database.calls == [(compiled.sql, compiled.params)]
    assert database.closed is False
    assert result.candidate_count == 7
    assert result.profile == "lexical"
    assert [hit.identity for hit in result.hits] == [(str(DOCUMENT_ID), 2, 4)]
    assert result.hits[0].path == "/corpus/source.pdf"
    assert result.hits[0].document_kind is DocumentKind.SOURCE


@pytest.mark.asyncio
async def test_vector_retriever_embeds_once_searches_the_store_and_closes_client(monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    expected = SearchResult(hits=(), candidate_count=0, profile="vector")
    calls = []

    class Store:
        def __init__(self, database, *, profile):
            calls.append(("store", database, profile))

        async def search(self, **kwargs):
            calls.append(("search", kwargs))
            return expected

    class Client:
        profile = PROFILE

        async def embed(self, texts):
            calls.append(("embed", tuple(texts)))
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            calls.append(("close",))

    database = object()
    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    query = SearchQuery.build(text="permit", limit=1, candidate_limit=2)

    result = await PostgresVectorRetriever(
        database,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        profile=PROFILE,
        embedding_client_factory=Client,
    ).retrieve(query)

    assert result is expected
    assert calls == [
        ("store", database, PROFILE),
        ("embed", ("permit",)),
        (
            "search",
            {
                "user_id": USER_ID,
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "query": query,
                "embedding": (1.0, 0.0, 0.0),
            },
        ),
        ("close",),
    ]


@pytest.mark.asyncio
async def test_vector_retriever_maps_typed_embedding_failure_and_still_closes(monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    calls = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            raise EmbeddingUnavailable("private query and endpoint")

        async def aclose(self):
            calls.append("close")

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)

    with pytest.raises(RetrieverUnavailable, match="query embedding is unavailable") as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=1))

    assert "private" not in str(caught.value)
    assert calls == ["close"]


@pytest.mark.asyncio
async def test_vector_retriever_validates_store_result_after_cleanup(monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    calls = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **_kwargs):
            return object()

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            calls.append("close")

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    with pytest.raises(RuntimeError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="q", limit=1))
    assert caught.value.args == ("query embedding failed",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert calls == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "failure"),
    [
        ("factory", RuntimeError("private factory key")),
        ("embed", TypeError("private embed payload")),
        ("embed", _OrdinaryBaseFailure("private base failure")),
        (
            "embed",
            ExceptionGroup(
                "private exception group",
                [RuntimeError("private first"), ValueError("private second")],
            ),
        ),
    ],
    ids=["factory", "embed", "base-exception", "exception-group"],
)
async def test_unexpected_factory_and_embedding_failures_are_fixed_sanitized_terminal(boundary, failure, monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    linked = failure
    if boundary == "embed" and type(failure) is TypeError:
        linked.__cause__ = RuntimeError("private explicit cause")
        linked.__context__ = ValueError("private explicit context")

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            raise linked

        async def aclose(self):
            pass

    def factory():
        if boundary == "factory":
            raise linked
        return Client()

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    with pytest.raises(RuntimeError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=factory,
        ).retrieve(SearchQuery.build(text="private query", limit=1))

    assert caught.value.args == ("query embedding failed",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_embedding_base_exception_group_with_control_propagates_sanitized_control(monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    failure = BaseExceptionGroup(
        "private group",
        [RuntimeError("private ordinary"), asyncio.CancelledError("private cancellation")],
    )

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            raise failure

        async def aclose(self):
            pass

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    with pytest.raises(asyncio.CancelledError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=1))
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [
        SearchResult(hits=(), candidate_count=0, profile="hybrid"),
        SearchResult(hits=(_hit("NOT-A-UUID"),), candidate_count=1, profile="vector"),
        SearchResult(hits=(_hit(), _hit()), candidate_count=2, profile="vector"),
        SearchResult(hits=(_hit(score=2.0),), candidate_count=1, profile="vector"),
        SearchResult(hits=(), candidate_count=2_147_483_648, profile="vector"),
        SearchResult(hits=(_hit(document_kind=None),), candidate_count=1, profile="vector"),
        SearchResult(hits=(_hit(path="/../private"),), candidate_count=1, profile="vector"),
        SearchResult(hits=(_hit(path="/safe/./private"),), candidate_count=1, profile="vector"),
        SearchResult(hits=(_hit(path="//private"),), candidate_count=1, profile="vector"),
    ],
    ids=[
        "profile",
        "uuid",
        "duplicate",
        "score",
        "candidate-bound",
        "document-kind",
        "traversal-path",
        "dot-path",
        "double-slash-path",
    ],
)
async def test_vector_retriever_fails_closed_on_inconsistent_exact_search_result(invalid_result, monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **_kwargs):
            return invalid_result

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            pass

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    with pytest.raises(RuntimeError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=2, candidate_limit=2))
    assert caught.value.args == ("query embedding failed",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "main_case",
    ["success", "embedding_unavailable", "store_unavailable", "unexpected"],
)
async def test_ordinary_embedding_cleanup_failure_is_sanitized_terminal_and_wins(main_case, monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    expected = SearchResult(hits=(), candidate_count=0, profile="vector")

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **_kwargs):
            if main_case == "store_unavailable":
                raise RetrieverUnavailable("private vector dsn")
            return expected

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            if main_case == "embedding_unavailable":
                raise EmbeddingUnavailable("private provider key")
            if main_case == "unexpected":
                raise TypeError("private programming state")
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            raise RuntimeError("private cleanup secret")

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)

    with pytest.raises(RuntimeError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=1))

    assert caught.value.args == ("query embedding cleanup failed",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_cleanup_control_signal_stays_sanitized_and_prevents_vector_return(monkeypatch):
    from rag.retrieval import PostgresVectorRetriever

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **_kwargs):
            return SearchResult(hits=(), candidate_count=0, profile="vector")

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            raise asyncio.CancelledError("private cleanup cancellation")

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    with pytest.raises(asyncio.CancelledError) as caught:
        await PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=1))
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_hosted_rag_defaults_to_lexical_without_constructing_embedding(monkeypatch):
    from rag.retrieval import HostedRagRetrieval

    monkeypatch.setattr(
        "rag.retrieval.OpenAIEmbeddingClient",
        lambda **_kwargs: pytest.fail("lexical retrieval must not construct embeddings"),
    )
    database = _RecordingDatabase([_row()])

    result = await HostedRagRetrieval(
        database,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=_settings(MODE="local", HYBRID_SEARCH_ENABLED=False, embedding_profile=None),
    ).retrieve(SearchQuery.build(text="permit", limit=1))

    assert result.profile == "lexical"
    assert [hit.document_id for hit in result.hits] == [str(DOCUMENT_ID)]


@pytest.mark.asyncio
async def test_hosted_lexical_retrieval_reads_no_hybrid_settings():
    from rag.retrieval import HostedRagRetrieval

    class UnreadableSettings:
        def __getattribute__(self, name):
            if not name.startswith("__"):
                raise RuntimeError("private settings secret")
            return super().__getattribute__(name)

    result = await HostedRagRetrieval(
        _RecordingDatabase([_row()]),
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=UnreadableSettings(),
    ).retrieve(SearchQuery.build(text="permit", limit=1), profile="lexical")

    assert result.profile == "lexical"


@pytest.mark.asyncio
@pytest.mark.parametrize("second_read_secret", ("helper", "profile", "rrf_k"))
async def test_hosted_hybrid_consumes_one_frozen_settings_snapshot(monkeypatch, second_read_secret):
    from rag.retrieval import HostedRagRetrieval

    class StatefulSettings:
        EMBEDDING_BASE_URL = "https://embedding.invalid/v1"
        EMBEDDING_API_KEY = SimpleNamespace(get_secret_value=lambda: "secret")
        EMBEDDING_BATCH_SIZE = 32
        EMBEDDING_TIMEOUT_SECONDS = 15

        def __init__(self):
            self.reads = {name: 0 for name in ("mode", "enabled", "profile", "helper", "rrf_k")}

        def _read(self, name, value):
            self.reads[name] += 1
            if name == second_read_secret and self.reads[name] > 1:
                raise RuntimeError(f"private second {name} settings secret")
            return value

        @property
        def MODE(self):
            return self._read("mode", "hosted")

        @property
        def HYBRID_SEARCH_ENABLED(self):
            return self._read("enabled", True)

        @property
        def embedding_profile(self):
            return self._read("profile", PROFILE)

        def hybrid_candidate_limits(self, _request_limit):
            return self._read("helper", (4, 5))

        @property
        def HYBRID_RRF_K(self):
            return self._read("rrf_k", 60)

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **_kwargs):
            return SearchResult(hits=(), candidate_count=0, profile="vector")

    class Client:
        def __init__(self, *, profile, **_kwargs):
            self.profile = profile

        async def embed(self, _texts):
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            pass

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    monkeypatch.setattr("rag.retrieval.OpenAIEmbeddingClient", Client)
    settings = StatefulSettings()
    result = await HostedRagRetrieval(
        _RecordingDatabase([_row()]),
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=settings,
    ).retrieve(SearchQuery.build(text="permit", limit=1), profile="hybrid")

    assert result.profile == "hybrid"
    assert settings.reads == {name: 1 for name in settings.reads}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "settings", "message"),
    [
        ("hybrid", _settings(HYBRID_SEARCH_ENABLED=False), "hybrid retrieval is unavailable"),
        ("semantic", _settings(), "unsupported retrieval profile"),
    ],
)
async def test_hosted_rag_strictly_rejects_disabled_hybrid_and_profile_mismatch(profile, settings, message):
    from rag.retrieval import HostedRagRetrieval

    service = HostedRagRetrieval(
        _RecordingDatabase([]),
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=settings,
    )

    with pytest.raises(ValueError, match=message):
        await service.retrieve(SearchQuery.build(text="permit"), profile=profile)


@pytest.mark.asyncio
async def test_hosted_scoped_hybrid_falls_back_before_embedding_client_factory():
    from rag.retrieval import HostedRagRetrieval

    calls = []
    result = await HostedRagRetrieval(
        _RecordingDatabase([_row()]),
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=_settings(),
        embedding_client_factory=lambda: calls.append("client"),
    ).retrieve(
        SearchQuery.build(text="permit", limit=1, scope="source"),
        profile="hybrid",
    )

    assert result.profile == "lexical_fallback"
    assert calls == []


@pytest.mark.asyncio
async def test_default_embedding_client_configuration_failure_is_typed_fallback(monkeypatch):
    from rag.retrieval import HostedRagRetrieval

    monkeypatch.setattr(
        "rag.retrieval.OpenAIEmbeddingClient",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("private endpoint and key")),
    )
    result = await HostedRagRetrieval(
        _RecordingDatabase([_row()]),
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=_settings(),
    ).retrieve(SearchQuery.build(text="private query", limit=1), profile="hybrid")

    assert result.profile == "lexical_fallback"


@pytest.mark.asyncio
async def test_injected_embedding_factory_programming_failure_is_visible_but_sanitized():
    from rag.retrieval import HostedRagRetrieval

    def broken_factory():
        raise TypeError("factory bug")

    with pytest.raises(RuntimeError) as caught:
        await HostedRagRetrieval(
            _RecordingDatabase([_row()]),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            settings=_settings(),
            embedding_client_factory=broken_factory,
        ).retrieve(SearchQuery.build(text="q", limit=1), profile="hybrid")
    assert caught.value.args == ("query embedding failed",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_hosted_hybrid_preserves_filters_before_configured_backend_limits(monkeypatch):
    from rag.retrieval import HostedRagRetrieval

    vector_queries = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        async def search(self, **kwargs):
            vector_queries.append(kwargs["query"])
            return SearchResult(hits=(), candidate_count=0, profile="vector")

    class Client:
        profile = PROFILE

        async def embed(self, _texts):
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            pass

    monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
    query = SearchQuery.build(
        text="permit",
        limit=2,
        candidate_limit=2,
        path_glob="/corpus/*.pdf",
        tags=("policy",),
        document_kinds=("source",),
        annotated_only=True,
        facets={"country": "IDN"},
    )
    database = _RecordingDatabase([_row()])

    result = await HostedRagRetrieval(
        database,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=_settings(),
        embedding_client_factory=Client,
    ).retrieve(query, profile="hybrid")

    lexical_sql, lexical_params = database.calls[0]
    assert "LIMIT $" in lexical_sql
    assert lexical_params[-1] == 4
    vector_query = vector_queries[0]
    for name in (
        "text",
        "path_glob",
        "tags",
        "document_kinds",
        "annotated_only",
        "facets",
    ):
        assert getattr(vector_query, name) == getattr(query, name)
    assert vector_query.candidate_limit == 5
    assert result.profile == "hybrid"


@pytest.mark.asyncio
async def test_evidence_reader_queries_selected_identities_once_and_preserves_hit_order():
    from rag.retrieval import PostgresEvidenceReader

    second_id = UUID("30000000-0000-0000-0000-000000000004")
    database = _RecordingDatabase(
        [
            _evidence_row(second_id, chunk=1, ordinal=2, content="second"),
            _evidence_row(ordinal=1, content="first"),
        ]
    )
    hits = (_hit(), _hit(second_id, chunk=1, score=0.5))

    evidence = await PostgresEvidenceReader(database).read(
        USER_ID,
        KNOWLEDGE_BASE_ID,
        hits,
        max_chars=11,
    )

    assert [item.document_id for item in evidence] == [DOCUMENT_ID, second_id]
    assert [item.content for item in evidence] == ["first", "second"]
    assert [item.score for item in evidence] == [0.75, 0.5]
    assert all(item.status.value == "ready" and item.archived is False for item in evidence)
    sql, params = database.calls[0]
    assert "WITH ORDINALITY" in sql
    assert "JOIN documents" in sql and "JOIN document_chunks" in sql
    assert "d.user_id = $1" in sql and "dc.user_id = $1" in sql
    assert "d.knowledge_base_id = $2" in sql and "dc.knowledge_base_id = $2" in sql
    assert "dc.document_version = d.version" in sql
    assert "d.status IN ('pending', 'processing', 'ready')" in sql
    assert "NOT d.archived" in sql
    assert params == (
        USER_ID,
        KNOWLEDGE_BASE_ID,
        [DOCUMENT_ID, second_id],
        [2, 2],
        [4, 1],
    )
    with pytest.raises(FrozenInstanceError):
        evidence[0].content = "changed"


@pytest.mark.asyncio
async def test_evidence_reader_exact_cap_stops_before_the_first_oversized_next_chunk():
    from rag.retrieval import PostgresEvidenceReader

    second_id = UUID("30000000-0000-0000-0000-000000000004")
    third_id = UUID("30000000-0000-0000-0000-000000000005")
    database = _RecordingDatabase(
        [
            _evidence_row(ordinal=1, content="12345"),
            _evidence_row(second_id, chunk=1, ordinal=2, content="6"),
            _evidence_row(third_id, chunk=2, ordinal=3, content=""),
        ]
    )
    hits = (_hit(), _hit(second_id, chunk=1), _hit(third_id, chunk=2))

    evidence = await PostgresEvidenceReader(database).read(USER_ID, KNOWLEDGE_BASE_ID, hits, max_chars=5)

    assert [item.content for item in evidence] == ["12345"]


@pytest.mark.asyncio
async def test_evidence_reader_rejects_duplicate_selected_identity_before_database():
    from rag.retrieval import PostgresEvidenceReader

    database = _RecordingDatabase([])

    with pytest.raises(ValueError) as caught:
        await PostgresEvidenceReader(database).read(
            USER_ID,
            KNOWLEDGE_BASE_ID,
            (_hit(), _hit()),
            max_chars=100,
        )

    assert caught.value.args == ("selected evidence hits are invalid",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert database.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hits",
    [_LenExplodingSequence(), _IterationExplodingSequence(), _ExplodingSequence()],
    ids=["len", "iteration", "getitem"],
)
async def test_evidence_reader_sanitizes_hostile_caller_hits_container(hits):
    from rag.retrieval import PostgresEvidenceReader

    database = _RecordingDatabase([])
    with pytest.raises(ValueError) as caught:
        await PostgresEvidenceReader(database).read(USER_ID, KNOWLEDGE_BASE_ID, hits, max_chars=100)
    assert caught.value.args == ("selected evidence hits are invalid",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "secret" not in str(caught.value)
    assert database.calls == []


@pytest.mark.asyncio
async def test_evidence_reader_preserves_control_from_hostile_caller_hits_container():
    from rag.retrieval import PostgresEvidenceReader

    with pytest.raises(asyncio.CancelledError) as caught:
        await PostgresEvidenceReader(_RecordingDatabase([])).read(
            USER_ID, KNOWLEDGE_BASE_ID, _CancellingSequence(), max_chars=100
        )
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_evidence_reader_sanitizes_hostile_exact_hit_field_value():
    from rag.retrieval import PostgresEvidenceReader

    hit = _hit()
    object.__setattr__(hit, "score", _ExplodingFloat(0.5))
    with pytest.raises(ValueError) as caught:
        await PostgresEvidenceReader(_RecordingDatabase([])).read(USER_ID, KNOWLEDGE_BASE_ID, (hit,), max_chars=100)
    assert caught.value.args == ("selected evidence hits are invalid",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_evidence_reader_rejects_empty_chunk_text_as_malformed_backend_data():
    from rag.retrieval import PostgresEvidenceReader

    with pytest.raises(RetrieverUnavailable, match="RAG evidence is unavailable"):
        await PostgresEvidenceReader(_RecordingDatabase([_evidence_row(content="")])).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )


@pytest.mark.parametrize("dto", ["evidence", "wiki"])
@pytest.mark.parametrize("field", ["metadata", "tags"])
def test_frozen_rag_dtos_sanitize_malicious_metadata_and_tag_containers(dto, field):
    from rag.retrieval import RagEvidence, RagWikiPage

    values = {"metadata": {"safe": "value"}, "tags": ("safe",)}
    values[field] = _ExplodingMapping() if field == "metadata" else _ExplodingSequence()

    with pytest.raises(ValueError) as caught:
        if dto == "evidence":
            RagEvidence(
                document_id=DOCUMENT_ID,
                document_version=2,
                chunk_index=4,
                page=5,
                filename="source.pdf",
                path="/corpus/",
                title="Source",
                content="source",
                status="ready",
                archived=False,
                score=0.5,
                tags=values["tags"],
                metadata=values["metadata"],
            )
        else:
            RagWikiPage(
                document_id=DOCUMENT_ID,
                version=2,
                path="/wiki/page.md",
                filename="page.md",
                content="page",
                title="Page",
                tags=values["tags"],
                date=date(2026, 7, 27),
                metadata=values["metadata"],
            )

    expected = "metadata is invalid" if field == "metadata" else "tags are invalid"
    assert caught.value.args == (expected,)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "secret" not in str(caught.value)


def test_rag_read_dto_repr_and_str_do_not_expose_private_payloads():
    from rag.retrieval import RagEvidence, RagWikiPage

    marker = "private-content-metadata-path-marker"
    evidence = RagEvidence(
        document_id=DOCUMENT_ID,
        document_version=2,
        chunk_index=4,
        page=5,
        filename="private.pdf",
        path=f"/corpus/{marker}/",
        title=marker,
        content=marker,
        status="ready",
        archived=False,
        score=0.5,
        tags=(marker,),
        metadata={"private": marker},
    )
    page = RagWikiPage(
        document_id=DOCUMENT_ID,
        version=2,
        path=f"/wiki/{marker}.md",
        filename=f"{marker}.md",
        content=marker,
        title=marker,
        tags=(marker,),
        date=date(2026, 7, 27),
        metadata={"private": marker},
    )

    for record in (evidence, page):
        assert marker not in repr(record)
        assert marker not in str(record)


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["evidence", "wiki"])
async def test_database_reader_sanitizes_malicious_metadata_container(reader):
    from rag.retrieval import PostgresEvidenceReader, PostgresWikiPageReader

    if reader == "evidence":
        row = _evidence_row()
        row["metadata"] = _ExplodingMapping()
        operation = PostgresEvidenceReader(_RecordingDatabase([row])).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )
        message = "RAG evidence is unavailable"
    else:
        row = {
            "document_id": DOCUMENT_ID,
            "version": 2,
            "path": "/wiki/",
            "filename": "page.md",
            "content": "page",
            "title": "Page",
            "tags": ["safe"],
            "date": "2026-07-27",
            "metadata": _ExplodingMapping(),
        }
        operation = PostgresWikiPageReader(_RecordingDatabase([row])).get_by_path(
            USER_ID, KNOWLEDGE_BASE_ID, "/wiki/page.md"
        )
        message = "wiki page is unavailable"

    with pytest.raises(RetrieverUnavailable, match=message) as caught:
        await operation
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_database_reader_preserves_sanitized_control_from_malicious_metadata():
    from rag.retrieval import PostgresEvidenceReader

    row = _evidence_row()
    row["metadata"] = _CancellingMapping()
    with pytest.raises(asyncio.CancelledError) as caught:
        await PostgresEvidenceReader(_RecordingDatabase([row])).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_database_reader_sanitizes_hostile_row_sequence_iteration():
    from rag.retrieval import PostgresEvidenceReader

    with pytest.raises(RetrieverUnavailable) as caught:
        await PostgresEvidenceReader(_RecordingDatabase(_ExplodingSequence())).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )
    assert caught.value.args == ("database returned invalid rows",)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_database_reader_preserves_control_from_hostile_row_sequence_iteration():
    from rag.retrieval import PostgresEvidenceReader

    with pytest.raises(asyncio.CancelledError) as caught:
        await PostgresEvidenceReader(_RecordingDatabase(_CancellingSequence())).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_lexical_and_evidence_database_failures_are_visible_but_sanitized():
    from rag.retrieval import PostgresEvidenceReader, PostgresLexicalRetriever

    class FailingDatabase:
        async def fetch(self, *_args):
            raise RuntimeError("private query content dsn=postgres://secret")

    with pytest.raises(RetrieverUnavailable, match="lexical retrieval is unavailable") as lexical:
        await PostgresLexicalRetriever(
            FailingDatabase(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
        ).retrieve(SearchQuery.build(text="private query", limit=1))
    with pytest.raises(RetrieverUnavailable, match="RAG evidence is unavailable") as evidence:
        await PostgresEvidenceReader(FailingDatabase()).read(USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100)
    assert "secret" not in str(lexical.value)
    assert "secret" not in str(evidence.value)
    assert lexical.value.__cause__ is None and lexical.value.__context__ is None
    assert evidence.value.__cause__ is None and evidence.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [_row(), _row(UUID("30000000-0000-0000-0000-000000000004"))],
        [{**_row(), "metadata": True}],
        [{**_row(), "tags": ["ok", True]}],
        [{**_row(), "content": "\ud800"}],
    ],
    ids=["row-count", "metadata", "tags", "utf8"],
)
async def test_lexical_retriever_rejects_malformed_or_oversized_backend_rows(rows):
    from rag.retrieval import PostgresLexicalRetriever

    with pytest.raises(RetrieverUnavailable, match="lexical retrieval is unavailable") as caught:
        await PostgresLexicalRetriever(
            _RecordingDatabase(rows),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
        ).retrieve(SearchQuery.build(text="q", limit=1, candidate_limit=1))
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["lexical", "evidence", "embedding"])
async def test_cancellation_is_sanitized_and_never_downgraded(boundary, monkeypatch):
    from rag.retrieval import (
        PostgresEvidenceReader,
        PostgresLexicalRetriever,
        PostgresVectorRetriever,
    )

    class CancellingDatabase:
        async def fetch(self, *_args):
            raise asyncio.CancelledError("private cancellation")

    if boundary == "lexical":
        operation = PostgresLexicalRetriever(
            CancellingDatabase(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
        ).retrieve(SearchQuery.build(text="private query", limit=1))
    elif boundary == "evidence":
        operation = PostgresEvidenceReader(CancellingDatabase()).read(
            USER_ID, KNOWLEDGE_BASE_ID, (_hit(),), max_chars=100
        )
    else:

        class Store:
            def __init__(self, *_args, **_kwargs):
                pass

        class Client:
            profile = PROFILE

            async def embed(self, _texts):
                raise asyncio.CancelledError("private cancellation")

        monkeypatch.setattr("rag.retrieval.PostgresVectorStore", Store)
        operation = PostgresVectorRetriever(
            object(),
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            profile=PROFILE,
            embedding_client_factory=Client,
        ).retrieve(SearchQuery.build(text="private query", limit=1))

    with pytest.raises(asyncio.CancelledError) as caught:
        await operation
    assert caught.value.args == ()
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.asyncio
async def test_wiki_page_reader_normalizes_exact_path_and_returns_frozen_current_page():
    from rag.retrieval import PostgresWikiPageReader

    database = _RecordingDatabase(
        [
            {
                "document_id": DOCUMENT_ID,
                "version": 3,
                "path": "/wiki/launch/",
                "filename": "page.md",
                "content": "Current page",
                "title": "Page",
                "tags": ["launch"],
                "date": "2026-07-27",
                "metadata": {"description": "Current"},
            }
        ]
    )

    page = await PostgresWikiPageReader(database).get_by_path(
        USER_ID,
        KNOWLEDGE_BASE_ID,
        "/wiki//launch/./page.md",
    )

    assert page.document_id == DOCUMENT_ID
    assert page.version == 3
    assert page.path == "/wiki/launch/page.md"
    assert page.filename == "page.md"
    assert page.date == date(2026, 7, 27)
    sql, params = database.calls[0]
    assert "d.source_kind = 'wiki'" in sql
    assert "d.user_id = $1" in sql and "d.knowledge_base_id = $2" in sql
    assert "dc.document_version = d.version" in sql
    assert "NOT d.archived" in sql
    assert params == (USER_ID, KNOWLEDGE_BASE_ID, "/wiki/launch/", "page.md")
    with pytest.raises(FrozenInstanceError):
        page.title = "changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../private.md", "/wiki/../private.md", "/wiki/a\x00.md", "/wiki/\ud800.md", True])
async def test_wiki_page_reader_rejects_unsafe_or_nontext_paths(path):
    from rag.retrieval import PostgresWikiPageReader

    database = _RecordingDatabase([])
    with pytest.raises((TypeError, ValueError)):
        await PostgresWikiPageReader(database).get_by_path(USER_ID, KNOWLEDGE_BASE_ID, path)
    assert database.calls == []
