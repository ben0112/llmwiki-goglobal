from inspect import iscoroutinefunction

import pytest

from llmwiki_core import (
    ContextExpander,
    DocumentKind,
    Reranker,
    Retriever,
    SearchResult,
)
from llmwiki_core.search import SearchArea, SearchHit, SearchQuery, SearchScope


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


def test_search_result_distinguishes_candidates_from_returned_hits():
    hit = SearchHit("doc", 1, 0, "text", 0.5, "/doc.md")
    result = SearchResult(hits=(hit,), candidate_count=17)

    assert result.returned_count == 1
    assert result.candidate_count == 17
    assert result.latency_ms == 0.0
    assert result.profile == "lexical"
    with pytest.raises((AttributeError, TypeError)):
        result.hits = ()


def test_retrieval_ports_are_public_protocols():
    assert iscoroutinefunction(Retriever.retrieve)
    assert iscoroutinefunction(Reranker.rerank)
    assert iscoroutinefunction(ContextExpander.expand)
