import pytest

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
