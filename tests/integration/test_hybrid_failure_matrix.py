import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmwiki_core.models import EmbeddingProfile, EmbeddingUnavailable, InvalidEmbeddingResponse
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery, SearchResult

PROFILE = EmbeddingProfile("openai_compatible", "embed-v1", 3)


class _UnknownRetrievalSignal(BaseException):
    pass


def _hosted_retrieval_service():
    path = Path(__file__).parents[2] / "mcp/services/retrieval.py"
    spec = importlib.util.spec_from_file_location("task10_mcp_retrieval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HostedRetrievalService


def _settings(**changes):
    values = {
        "MODE": "hosted",
        "HYBRID_SEARCH_ENABLED": True,
        "embedding_profile": PROFILE,
        "HYBRID_LEXICAL_CANDIDATES": 10,
        "HYBRID_VECTOR_CANDIDATES": 10,
        "HYBRID_RRF_K": 60,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _hit(name):
    return SearchHit(name, 1, 0, f"private-{name}", 1.0, f"/{name}.md")


class _Client:
    profile = PROFILE

    def __init__(self, failure=None, vectors=((1.0, 0.0, 0.0),)):
        self.failure = failure
        self.vectors = vectors

    async def embed(self, _texts):
        if self.failure is not None:
            raise self.failure
        return self.vectors


class _Vault:
    def __init__(self, *, lexical_failure=None, vector_failure=None, vector_hits=None, graph_failure=None):
        self.lexical_failure = lexical_failure
        self.lexical = SearchResult((_hit("lexical"),), 1, profile="lexical")
        self.vector_failure = vector_failure
        self.vector_hits = (_hit("vector"),) if vector_hits is None else vector_hits
        self.graph_failure = graph_failure
        self.graph_calls = 0

    async def retrieve(self, _kb, _query):
        if self.lexical_failure is not None:
            raise self.lexical_failure
        return self.lexical

    async def retrieve_vector(self, _kb, _query, **_kwargs):
        if self.vector_failure is not None:
            raise self.vector_failure
        return SearchResult(tuple(self.vector_hits), len(self.vector_hits), profile="vector")

    async def expand_references(self, *_args, **_kwargs):
        self.graph_calls += 1
        if self.graph_failure is not None:
            raise self.graph_failure
        return ()


def _sink(events):
    def record(event, **fields):
        events.append((event, json.loads(json.dumps(fields, default=str))))

    return record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_failure", "vector_failure"),
    [
        (EmbeddingUnavailable("https://private:token@endpoint.invalid"), None),
        (None, RetrieverUnavailable("pgvector SQL private")),
        (InvalidEmbeddingResponse("wrong dimension private"), None),
        (None, RetrieverUnavailable("stale embedding version private")),
        (None, RetrieverUnavailable("vector scope/filter unavailable private")),
    ],
)
async def test_only_typed_vector_availability_failures_fallback_once(client_failure, vector_failure):
    HostedRetrievalService = _hosted_retrieval_service()

    events = []
    vault = _Vault(vector_failure=vector_failure)
    service = HostedRetrievalService(
        vault,
        "tenant-private-kb",
        settings=_settings(),
        embedding_client_factory=lambda: _Client(client_failure),
        telemetry_sink=_sink(events),
    )

    result = await service.retrieve(
        SearchQuery.build(text="super secret query", limit=2), profile="hybrid"
    )

    assert result.profile == "lexical_fallback"
    assert result.hits == (_hit("lexical"),)
    assert vault.graph_calls == 0
    assert [event for event, _fields in events] == ["retrieval_fallback", "retrieval_finished"]
    assert events[0][1]["retrieval_id"] == events[1][1]["retrieval_id"]
    assert "private" not in json.dumps(events).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        _settings(HYBRID_SEARCH_ENABLED=False),
        _settings(embedding_profile=None),
        _settings(MODE="local"),
    ],
)
async def test_disabled_or_absent_hybrid_configuration_is_visible_without_events(settings):
    HostedRetrievalService = _hosted_retrieval_service()

    events = []
    service = HostedRetrievalService(
        _Vault(), "kb", settings=settings, telemetry_sink=_sink(events)
    )
    with pytest.raises(ValueError, match="hybrid retrieval is unavailable"):
        await service.retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")
    assert events == []


@pytest.mark.asyncio
async def test_no_vector_hits_is_hybrid_not_fallback_and_emits_one_finished():
    HostedRetrievalService = _hosted_retrieval_service()

    events = []
    vault = _Vault(vector_hits=())
    result = await HostedRetrievalService(
        vault,
        "kb",
        settings=_settings(),
        embedding_client_factory=_Client,
        telemetry_sink=_sink(events),
    ).retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")

    assert result.profile == "hybrid"
    assert [event for event, _fields in events] == ["retrieval_finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("lexical SQL private"), RetrieverUnavailable("lexical unavailable private")],
)
async def test_lexical_backend_failure_never_degrades_to_vector_or_reports_success(failure):
    HostedRetrievalService = _hosted_retrieval_service()

    events = []
    service = HostedRetrievalService(
        _Vault(lexical_failure=failure),
        "kb",
        settings=_settings(),
        embedding_client_factory=_Client,
        telemetry_sink=_sink(events),
    )
    with pytest.raises(type(failure)):
        await service.retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (asyncio.CancelledError("private"), asyncio.CancelledError),
        (GeneratorExit("private"), (GeneratorExit, RuntimeError)),
    ],
)
async def test_process_controls_are_visible_sanitized_and_never_emit_success(signal, expected):
    HostedRetrievalService = _hosted_retrieval_service()

    events = []
    service = HostedRetrievalService(
        _Vault(),
        "kb",
        settings=_settings(),
        embedding_client_factory=lambda: _Client(signal),
        telemetry_sink=_sink(events),
    )
    with pytest.raises(expected) as raised:
        await service.retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")
    assert "private" not in str(raised.value)
    assert events == []


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (KeyboardInterrupt("private"), KeyboardInterrupt),
        (SystemExit("private"), SystemExit),
        (
            BaseExceptionGroup(
                "private", [RuntimeError("ordinary"), KeyboardInterrupt("private")]
            ),
            KeyboardInterrupt,
        ),
    ],
)
def test_sync_telemetry_boundary_preserves_ki_system_exit_and_group_controls(signal, expected):
    HostedRetrievalService = _hosted_retrieval_service()

    def signal_sink(*_args, **_kwargs):
        raise signal

    service = HostedRetrievalService(
        _Vault(), "kb", settings=_settings(), telemetry_sink=signal_sink
    )
    with pytest.raises(expected) as raised:
        service._emit_telemetry(
            "retrieval_finished",
            schema_version=1,
            retrieval_id="00000000-0000-0000-0000-000000000001",
            profile="lexical",
            result_count=0,
            candidate_count=0,
            duration_ms=0,
            error_code=None,
        )
    assert "private" not in str(raised.value)


@pytest.mark.asyncio
async def test_optional_graph_and_reranker_failures_preserve_result_without_fallback():
    HostedRetrievalService = _hosted_retrieval_service()

    class Reranker:
        async def rerank(self, *_args):
            raise RuntimeError("reranker private")

    events = []
    vault = _Vault(vector_hits=(), graph_failure=RetrieverUnavailable("graph private"))
    result = await HostedRetrievalService(
        vault,
        "kb",
        settings=_settings(),
        embedding_client_factory=_Client,
        reranker=Reranker(),
        telemetry_sink=_sink(events),
    ).retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")

    assert [hit.identity for hit in result.hits] == [_hit("lexical").identity]
    assert result.profile == "hybrid"
    assert [event for event, _fields in events] == ["retrieval_finished"]


@pytest.mark.asyncio
async def test_ordinary_sink_failure_does_not_mask_success_or_create_duplicate_callbacks():
    HostedRetrievalService = _hosted_retrieval_service()

    calls = []

    def failing_sink(event, **_fields):
        calls.append(event)
        raise RuntimeError("telemetry sink private")

    result = await HostedRetrievalService(
        _Vault(vector_failure=RetrieverUnavailable("private")),
        "kb",
        settings=_settings(),
        embedding_client_factory=_Client,
        telemetry_sink=failing_sink,
    ).retrieve(SearchQuery.build(text="secret", limit=2), profile="hybrid")

    assert result.profile == "lexical_fallback"
    assert calls == ["retrieval_fallback", "retrieval_finished"]


@pytest.mark.asyncio
async def test_ordinary_sink_failure_does_not_mask_default_lexical_result():
    HostedRetrievalService = _hosted_retrieval_service()

    calls = []

    def failing_sink(event, **_fields):
        calls.append(event)
        raise RuntimeError("telemetry sink private")

    vault = _Vault()
    result = await HostedRetrievalService(
        vault,
        "kb",
        settings=_settings(MODE="local", HYBRID_SEARCH_ENABLED=False, embedding_profile=None),
        telemetry_sink=failing_sink,
    ).retrieve(SearchQuery.build(text="secret", limit=2), profile="lexical")

    assert result is vault.lexical
    assert calls == ["retrieval_finished"]


def _linked_control(signal):
    failure = RuntimeError("private wrapper")
    failure.__cause__ = signal
    return failure


def _context_control(signal):
    failure = RuntimeError("private wrapper")
    failure.__context__ = signal
    return failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "expected", "exit_code"),
    [
        (KeyboardInterrupt("private"), KeyboardInterrupt, None),
        (SystemExit("private"), SystemExit, 1),
        (asyncio.CancelledError("private"), asyncio.CancelledError, None),
        (GeneratorExit("private"), GeneratorExit, None),
        (_linked_control(asyncio.CancelledError("private")), asyncio.CancelledError, None),
        (_linked_control(GeneratorExit("private")), GeneratorExit, None),
        (_context_control(GeneratorExit("private")), GeneratorExit, None),
        (
            BaseExceptionGroup(
                "private outer",
                [BaseExceptionGroup("private inner", [GeneratorExit("private")])],
            ),
            GeneratorExit,
            None,
        ),
        (
            BaseExceptionGroup(
                "private priority",
                [GeneratorExit("private"), asyncio.CancelledError("private")],
            ),
            asyncio.CancelledError,
            None,
        ),
        (_UnknownRetrievalSignal("private"), BaseException, None),
        (
            BaseExceptionGroup(
                "private unknown",
                [RuntimeError("ordinary"), _UnknownRetrievalSignal("private")],
            ),
            BaseException,
            None,
        ),
        (
            BaseExceptionGroup(
                "private", [RuntimeError("ordinary"), KeyboardInterrupt("private")]
            ),
            KeyboardInterrupt,
            None,
        ),
    ],
)
async def test_default_lexical_sink_controls_propagate_sanitized(
    signal,
    expected,
    exit_code,
):
    HostedRetrievalService = _hosted_retrieval_service()

    def signal_sink(*_args, **_kwargs):
        raise signal

    service = HostedRetrievalService(
        _Vault(),
        "kb",
        settings=_settings(MODE="local", HYBRID_SEARCH_ENABLED=False, embedding_profile=None),
        telemetry_sink=signal_sink,
    )
    with pytest.raises(expected) as raised:
        await service.retrieve(
            SearchQuery.build(text="private query", limit=1), profile="lexical"
        )

    assert raised.value.args in ((), (exit_code,))
    if expected is BaseException:
        assert type(raised.value) is BaseException
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__context__ is None


@pytest.mark.asyncio
async def test_injected_sink_still_passes_through_shared_fail_closed_sanitizer():
    HostedRetrievalService = _hosted_retrieval_service()

    vault = _Vault()
    vault.lexical = SearchResult((_hit("lexical"),), 1, profile="https://key@private")
    events = []
    result = await HostedRetrievalService(
        vault,
        "tenant-written-private-id",
        settings=_settings(),
        telemetry_sink=_sink(events),
    ).retrieve(SearchQuery.build(text="super secret", limit=1), profile="lexical")

    assert result is vault.lexical
    assert events == []
