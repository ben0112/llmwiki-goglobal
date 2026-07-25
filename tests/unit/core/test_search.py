from inspect import iscoroutinefunction
from typing import get_type_hints

import pytest

from llmwiki_core import (
    ContextExpander,
    DocumentKind,
    HybridRetrievalService,
    Reranker,
    Retriever,
    RetrieverUnavailable,
    SearchResult,
)
from llmwiki_core.search import (
    SearchArea,
    SearchHit,
    SearchQuery,
    SearchScope,
)


def _hit(
    document_id: str,
    *,
    version: int = 1,
    chunk: int = 0,
    score: float = 1.0,
) -> SearchHit:
    return SearchHit(
        document_id,
        version,
        chunk,
        f"content-{document_id}",
        score,
        f"/{document_id}.md",
    )


class _FakeRetriever:
    def __init__(
        self,
        hits,
        *,
        candidate_count=None,
        latency_ms=0.0,
        profile="backend",
    ):
        self.hits = tuple(hits)
        self.candidate_count = len(self.hits) if candidate_count is None else candidate_count
        self.latency_ms = latency_ms
        self.profile = profile
        self.queries = []

    async def retrieve(self, query):
        self.queries.append(query)
        return SearchResult(
            hits=self.hits,
            candidate_count=self.candidate_count,
            latency_ms=self.latency_ms,
            profile=self.profile,
        )


class _UnavailableRetriever:
    async def retrieve(self, query):
        raise RetrieverUnavailable("vector backend unavailable")


class _ErrorRetriever:
    def __init__(self, error):
        self.error = error

    async def retrieve(self, query):
        raise self.error


class _FakeReranker:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    async def rerank(self, query, hits):
        self.inputs.append(tuple(hits))
        return self.output


class _FakeExpander:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    async def expand(self, query, hits):
        self.inputs.append(tuple(hits))
        return self.output


def test_search_query_normalizes_shared_filters():
    query = SearchQuery.build(
        text="  data localization  ",
        limit=20,
        area="wiki",
        scope="annotations",
        facets={"stage": "S2"},
    )
    assert query.text == "data localization"
    assert query.area is SearchArea.WIKI
    assert query.scope is SearchScope.ANNOTATIONS
    assert query.facets == {"stage": "S2"}


@pytest.mark.parametrize("text,limit", [("", 10), ("query", 0), ("query", 101)])
def test_search_query_rejects_invalid_inputs(text, limit):
    with pytest.raises(ValueError):
        SearchQuery.build(text=text, limit=limit)


def test_search_query_copies_facets():
    facets = {"stage": "S2"}
    query = SearchQuery.build(text="query", facets=facets)
    facets["stage"] = "S3"
    assert query.facets == {"stage": "S2"}
    with pytest.raises(TypeError):
        query.facets["stage"] = "S4"


def test_search_query_normalizes_filters_and_candidate_limit():
    query = SearchQuery.build(
        text="  export controls  ",
        limit=10,
        candidate_limit=40,
        path_glob=" corpus/**/*.md ",
        tags=["Policy", " policy ", "ASEAN"],
        document_kinds=["wiki", DocumentKind.SOURCE, "source"],
        annotated_only=True,
    )

    assert query.text == "export controls"
    assert query.path_glob == "/corpus/**/*.md"
    assert query.tags == ("asean", "policy")
    assert query.document_kinds == (DocumentKind.SOURCE, DocumentKind.WIKI)
    assert query.annotated_only is True
    assert query.candidate_limit == 40


def test_search_query_keeps_existing_builder_calls_compatible():
    query = SearchQuery.build(
        text="query",
        limit=7,
        area="sources",
        scope="source",
        facets={"country": "IDN"},
    )

    assert query.candidate_limit == 7
    assert query.path_glob is None
    assert query.tags == ()
    assert query.document_kinds == ()
    assert query.annotated_only is False


def test_search_query_treats_only_none_area_as_all():
    assert SearchQuery(text="query", area=None).area is SearchArea.ALL
    assert SearchQuery.build(text="query", area=None).area is SearchArea.ALL


@pytest.mark.parametrize("area", [False, 0, "", []])
@pytest.mark.parametrize("constructor", [SearchQuery, SearchQuery.build])
def test_search_query_rejects_falsy_invalid_areas(area, constructor):
    with pytest.raises(ValueError, match="unsupported search area"):
        constructor(text="query", area=area)


@pytest.mark.parametrize("constructor", [SearchQuery, SearchQuery.build])
def test_search_query_scope_remains_fail_closed(constructor):
    with pytest.raises(ValueError, match="unsupported search scope"):
        constructor(text="query", scope=False)


def test_search_query_direct_constructor_normalizes_and_freezes_inputs():
    facets = {"country": "IDN"}
    tags = ["Policy", " asean "]
    document_kinds = ["wiki", DocumentKind.SOURCE]
    query = SearchQuery(
        "  query  ",
        40,
        "wiki",
        "source",
        facets,
        tags=tags,
        document_kinds=document_kinds,
        path_glob="corpus/*.md",
    )
    facets["country"] = "SGP"
    tags.append("new")
    document_kinds.append("asset")

    assert query.text == "query"
    assert query.limit == 40
    assert query.candidate_limit == 40
    assert query.area is SearchArea.WIKI
    assert query.scope is SearchScope.SOURCE
    assert query.facets == {"country": "IDN"}
    assert query.tags == ("asean", "policy")
    assert query.document_kinds == (DocumentKind.SOURCE, DocumentKind.WIKI)
    assert query.path_glob == "/corpus/*.md"
    with pytest.raises(TypeError):
        query.facets["country"] = "MYS"


def test_search_query_deep_freezes_facets_and_isolates_source_mutation():
    facets = {
        "countries": ["IDN", {"code": "SGP"}],
        "labels": {"policy", "reviewed"},
    }
    query = SearchQuery(text="query", facets=facets)
    facets["countries"].append("MYS")
    facets["countries"][1]["code"] = "MYS"
    facets["labels"].add("new")

    countries = query.facets["countries"]
    assert countries == ("IDN", {"code": "SGP"})
    assert query.facets["labels"] == frozenset({"policy", "reviewed"})
    with pytest.raises(TypeError):
        countries[1]["code"] = "MYS"


def test_search_query_candidate_limit_has_non_optional_public_type():
    assert get_type_hints(SearchQuery)["candidate_limit"] is int
    assert SearchQuery("query", 40).candidate_limit == 40


def test_search_query_preserves_positional_candidate_limit_input():
    query = SearchQuery("query", 10, "all", "all", {}, 40)

    assert query.candidate_limit == 40


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": True},
        {"limit": 2.5},
        {"candidate_limit": True},
        {"candidate_limit": 20.0},
    ],
)
def test_search_query_rejects_non_integer_counts(kwargs):
    with pytest.raises(ValueError, match="must be an integer"):
        SearchQuery(text="query", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tags": "policy"},
        {"tags": b"policy"},
        {"tags": ["policy", 7]},
        {"document_kinds": "source"},
        {"document_kinds": b"source"},
        {"document_kinds": [7]},
    ],
)
def test_search_query_rejects_scalar_or_invalid_filter_sequences(kwargs):
    with pytest.raises(ValueError, match="must be a sequence"):
        SearchQuery(text="query", **kwargs)


def test_search_query_rejects_non_boolean_annotated_only():
    with pytest.raises(ValueError, match="annotated_only must be a boolean"):
        SearchQuery(text="query", annotated_only=1)


@pytest.mark.parametrize("path_glob", [7, b"corpus/*.md"])
def test_search_query_rejects_non_string_path_glob(path_glob):
    with pytest.raises(ValueError, match="path_glob must be a string or None"):
        SearchQuery(text="query", path_glob=path_glob)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 10, "candidate_limit": 9}, "at least the search limit"),
        ({"candidate_limit": 501}, "at most 500"),
        ({"document_kinds": ["memo"]}, "unsupported document kind"),
        ({"tags": ["policy", " "]}, "tags must not be empty"),
        ({"path_glob": "corpus/\x00secret"}, "path glob contains NUL"),
    ],
)
def test_search_query_rejects_invalid_retrieval_filters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SearchQuery.build(text="query", **kwargs)


def test_search_hit_has_stable_identity_fields():
    hit = SearchHit(
        document_id="doc",
        document_version=3,
        chunk_index=2,
        content="matched text",
        score=0.75,
        path="/wiki/policy.md",
        title="Policy",
    )
    assert hit.identity == ("doc", 3, 2)


def test_search_hit_keeps_old_construction_and_defaults_metadata():
    hit = SearchHit("doc", 3, 2, "matched text", 0.75, "/wiki/policy.md", "Policy")

    assert hit.identity == ("doc", 3, 2)
    assert hit.page is None
    assert hit.header_breadcrumb is None
    assert hit.tags == ()
    assert hit.document_kind is None
    assert hit.metadata == {}


def test_search_hit_normalizes_path_and_keeps_legacy_scalar_values():
    hit = SearchHit("doc", 0, 0, "", 1, " relative/policy.md ", None, 0, None)

    assert hit.content == ""
    assert hit.path == "relative/policy.md"
    assert hit.page == 0
    assert hit.score == 1.0


def test_search_hit_copies_and_freezes_metadata():
    metadata = {"country": "IDN"}
    hit = SearchHit(
        "doc",
        3,
        2,
        "matched text",
        0.75,
        "/wiki/policy.md",
        tags=["policy"],
        document_kind=DocumentKind.WIKI,
        metadata=metadata,
    )
    metadata["country"] = "SGP"

    assert hit.tags == ("policy",)
    assert hit.metadata == {"country": "IDN"}
    with pytest.raises(TypeError):
        hit.metadata["country"] = "MYS"


def test_search_hit_deep_freezes_metadata_and_isolates_source_mutation():
    metadata = {
        "classification": {"countries": ["IDN", {"code": "SGP"}]},
        "labels": {"reviewed", "policy"},
    }
    hit = SearchHit("doc", 3, 2, "text", 0.75, "/doc.md", metadata=metadata)
    metadata["classification"]["countries"].append("MYS")
    metadata["classification"]["countries"][1]["code"] = "MYS"
    metadata["labels"].add("new")

    countries = hit.metadata["classification"]["countries"]
    assert countries == ("IDN", {"code": "SGP"})
    assert hit.metadata["labels"] == frozenset({"reviewed", "policy"})
    with pytest.raises(TypeError):
        countries[1]["code"] = "MYS"


@pytest.mark.parametrize("metadata", [{1: "value"}, {"value": object()}])
def test_search_hit_rejects_unsupported_metadata(metadata):
    with pytest.raises(TypeError, match="metadata"):
        SearchHit("doc", 1, 0, "text", 0.5, "/doc.md", metadata=metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_version", True),
        ("document_version", -1),
        ("document_version", 1.5),
        ("chunk_index", False),
        ("chunk_index", -1),
        ("chunk_index", 1.5),
        ("score", True),
        ("score", float("nan")),
        ("score", float("inf")),
    ],
)
def test_search_hit_rejects_invalid_identity_or_score_numbers(field, value):
    values = {
        "document_id": "doc",
        "document_version": 1,
        "chunk_index": 0,
        "content": "text",
        "score": 0.5,
        "path": "/doc.md",
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        SearchHit(**values)


@pytest.mark.parametrize("document_id", ["", "   ", 7])
def test_search_hit_rejects_invalid_document_id(document_id):
    with pytest.raises(ValueError, match="document_id must be a nonblank string"):
        SearchHit(document_id, 1, 0, "text", 0.5, "/doc.md")


@pytest.mark.parametrize("path", ["", "   ", 7])
def test_search_hit_rejects_invalid_path(path):
    with pytest.raises(ValueError, match="path must be a nonblank string"):
        SearchHit("doc", 1, 0, "text", 0.5, path)


def test_search_hit_rejects_non_string_content():
    with pytest.raises(ValueError, match="content must be a string"):
        SearchHit("doc", 1, 0, 7, 0.5, "/doc.md")


def test_search_hit_rejects_non_string_title():
    with pytest.raises(ValueError, match="title must be a string or None"):
        SearchHit("doc", 1, 0, "text", 0.5, "/doc.md", 7)


@pytest.mark.parametrize("page", [True, -1, 1.5])
def test_search_hit_rejects_invalid_page(page):
    with pytest.raises(ValueError, match="page must be a non-negative integer or None"):
        SearchHit("doc", 1, 0, "text", 0.5, "/doc.md", page=page)


def test_search_hit_rejects_non_string_header_breadcrumb():
    with pytest.raises(ValueError, match="header_breadcrumb must be a string or None"):
        SearchHit("doc", 1, 0, "text", 0.5, "/doc.md", header_breadcrumb=7)


def test_search_result_distinguishes_candidates_from_returned_hits():
    hit = SearchHit("doc", 1, 0, "text", 0.5, "/doc.md")
    result = SearchResult(hits=(hit,), candidate_count=17)

    assert result.returned_count == 1
    assert result.candidate_count == 17
    assert result.latency_ms == 0.0
    assert result.profile == "lexical"
    with pytest.raises((AttributeError, TypeError)):
        result.hits = ()


def test_search_result_copies_hit_sequence():
    hit = SearchHit("doc", 1, 0, "text", 0.5, "/doc.md")
    hits = [hit]
    result = SearchResult(hits=hits, candidate_count=1)
    hits.clear()

    assert result.hits == (hit,)


def test_search_result_allows_expansion_to_exceed_candidate_count():
    hits = (
        SearchHit("doc-1", 1, 0, "text", 0.5, "/one.md"),
        SearchHit("doc-2", 1, 0, "context", 0.4, "/two.md"),
    )
    result = SearchResult(hits=hits, candidate_count=1)

    assert result.returned_count == 2
    assert result.candidate_count == 1


@pytest.mark.parametrize("candidate_count", [True, -1, 1.5])
def test_search_result_rejects_invalid_candidate_count(candidate_count):
    hit = SearchHit("doc", 1, 0, "text", 0.5, "/doc.md")
    with pytest.raises(ValueError, match="candidate_count"):
        SearchResult(hits=(hit,), candidate_count=candidate_count)


@pytest.mark.parametrize("latency_ms", [True, -0.1, float("nan"), float("inf")])
def test_search_result_rejects_invalid_latency(latency_ms):
    with pytest.raises(ValueError, match="latency_ms"):
        SearchResult(hits=(), candidate_count=0, latency_ms=latency_ms)


@pytest.mark.parametrize("profile", ["", "   ", 7])
def test_search_result_rejects_invalid_profile(profile):
    with pytest.raises(ValueError, match="profile"):
        SearchResult(hits=(), candidate_count=0, profile=profile)


def test_search_result_rejects_non_hit_members():
    with pytest.raises(ValueError, match="SearchHit"):
        SearchResult(hits=(object(),), candidate_count=1)


def test_retrieval_ports_are_public_protocols():
    assert iscoroutinefunction(Retriever.retrieve)
    assert iscoroutinefunction(Reranker.rerank)
    assert iscoroutinefunction(ContextExpander.expand)


@pytest.mark.asyncio
async def test_hybrid_fuses_by_rrf_and_deduplicates_identity():
    lexical_b = _hit("b", score=0.8)
    vector_b = _hit("b", score=0.7)
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("a"), lexical_b], candidate_count=7),
        vector=_FakeRetriever([vector_b, _hit("c")], candidate_count=9),
        rrf_k=60,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=3, candidate_limit=10))

    assert [item.document_id for item in result.hits] == ["b", "a", "c"]
    assert result.hits[0].content == lexical_b.content
    assert result.hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert result.candidate_count == 16
    assert result.profile == "hybrid"


@pytest.mark.asyncio
async def test_hybrid_uses_identity_for_stable_rrf_tie_breaking():
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("z", version=2), _hit("middle")]),
        vector=_FakeRetriever([_hit("a", version=3), _hit("other")]),
        rrf_k=10,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=4))

    assert [item.document_id for item in result.hits] == ["a", "z", "middle", "other"]


@pytest.mark.asyncio
async def test_hybrid_counts_duplicate_only_once_per_retriever():
    duplicate = _hit("a")
    service = HybridRetrievalService(
        lexical=_FakeRetriever([duplicate, duplicate, _hit("b")]),
        vector=_FakeRetriever([]),
        rrf_k=10,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2))

    assert [item.document_id for item in result.hits] == ["a", "b"]
    assert result.hits[0].score == pytest.approx(1 / 11)
    assert result.hits[1].score == pytest.approx(1 / 12)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lexical_hits", "vector_hits", "expected"),
    [
        ([], [_hit("v")], ["v"]),
        ([_hit("l")], [], ["l"]),
        ([], [], []),
    ],
)
async def test_hybrid_handles_empty_backend_results(lexical_hits, vector_hits, expected):
    service = HybridRetrievalService(
        lexical=_FakeRetriever(lexical_hits, candidate_count=2),
        vector=_FakeRetriever(vector_hits, candidate_count=3),
    )

    result = await service.retrieve(SearchQuery.build(text="q"))

    assert [item.document_id for item in result.hits] == expected
    assert result.candidate_count == 5
    assert result.profile == "hybrid"


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_lexical_when_vector_is_unavailable():
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("a")], candidate_count=11, latency_ms=2.5),
        vector=_UnavailableRetriever(),
    )

    result = await service.retrieve(SearchQuery.build(text="q"))

    assert [item.document_id for item in result.hits] == ["a"]
    assert result.candidate_count == 11
    assert result.latency_ms == 2.5
    assert result.profile == "lexical_fallback"


@pytest.mark.asyncio
async def test_hybrid_does_not_swallow_vector_programming_errors():
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("a")]),
        vector=_ErrorRetriever(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        await service.retrieve(SearchQuery.build(text="q"))


@pytest.mark.asyncio
async def test_hybrid_propagates_lexical_unavailability_without_fallback():
    service = HybridRetrievalService(
        lexical=_UnavailableRetriever(),
        vector=_FakeRetriever([_hit("v")]),
    )

    with pytest.raises(RetrieverUnavailable, match="vector backend unavailable"):
        await service.retrieve(SearchQuery.build(text="q"))


@pytest.mark.asyncio
async def test_service_without_vector_preserves_lexical_profile_and_counts():
    lexical = _FakeRetriever([_hit("b"), _hit("a")], candidate_count=13, latency_ms=1.25)
    service = HybridRetrievalService(lexical=lexical)
    query = SearchQuery.build(text="q", limit=1, candidate_limit=4)

    result = await service.retrieve(query)

    assert [item.document_id for item in result.hits] == ["b"]
    assert result.candidate_count == 13
    assert result.latency_ms == 1.25
    assert result.profile == "lexical"
    assert lexical.queries == [query]


@pytest.mark.asyncio
async def test_reranker_can_only_reorder_known_unique_direct_hits():
    original_a = _hit("a", score=0.9)
    original_b = _hit("b", score=0.8)
    altered_b = SearchHit("b", 1, 0, "untrusted", 999, "/untrusted.md")
    reranker = _FakeReranker([altered_b, altered_b, _hit("unknown")])
    service = HybridRetrievalService(
        lexical=_FakeRetriever([original_a, original_b]),
        reranker=reranker,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2))

    assert result.hits == (original_b, original_a)
    assert reranker.inputs == [(original_a, original_b)]


@pytest.mark.asyncio
async def test_reranker_receives_only_the_bounded_direct_set():
    reranker = _FakeReranker([])
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("a"), _hit("b"), _hit("c")]),
        reranker=reranker,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2))

    assert [item.document_id for item in reranker.inputs[0]] == ["a", "b"]
    assert [item.document_id for item in result.hits] == ["a", "b"]


@pytest.mark.asyncio
async def test_expander_appends_unique_new_hits_without_displacing_direct_hits():
    direct = _hit("direct")
    expansion = _hit("context")
    expander = _FakeExpander(
        [direct, expansion, expansion, object(), _hit("overflow")]
    )
    service = HybridRetrievalService(
        lexical=_FakeRetriever([direct]),
        expander=expander,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2))

    assert result.hits == (direct, expansion)
    assert expander.inputs == [(direct,)]
    assert result.candidate_count == 1


@pytest.mark.asyncio
async def test_expander_is_not_called_when_direct_hits_fill_the_limit():
    expander = _FakeExpander([_hit("context")])
    service = HybridRetrievalService(
        lexical=_FakeRetriever([_hit("a"), _hit("b")]),
        expander=expander,
    )

    result = await service.retrieve(SearchQuery.build(text="q", limit=2))

    assert [item.document_id for item in result.hits] == ["a", "b"]
    assert expander.inputs == []


@pytest.mark.parametrize("rrf_k", [True, 0, -1, 1.5, float("inf")])
def test_hybrid_rejects_invalid_rrf_k(rrf_k):
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetrievalService(lexical=_FakeRetriever([]), rrf_k=rrf_k)
