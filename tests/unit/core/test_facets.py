import pytest

from llmwiki_core.facets import (
    UnknownFacetError,
    apply_rollup,
    rollup_from_metas,
    sqlite_facet_conditions,
    validate_facets,
)


def test_validate_facets_rejects_unknown_keys():
    with pytest.raises(UnknownFacetError):
        validate_facets({"planet": "Mars"})


def test_rollup_merges_cited_corpus_dimensions():
    rollup = rollup_from_metas(
        [
            {"entry_id": "E-1", "stage": "S2", "geo_country": ["IDN"], "timeliness": "M2"},
            {"entry_id": "E-2", "stage": "S3", "geo_country": ["VNM"], "timeliness": "M1"},
        ],
        "2026-07-24",
    )
    assert rollup["stage"] == ["S2", "S3"]
    assert rollup["country"] == ["IDN", "VNM"]
    assert rollup["timeliness_worst"] == "M1"

    metadata = {}
    assert apply_rollup(metadata, rollup) is True
    assert metadata["facet_rollup"] == rollup


def test_sqlite_facet_conditions_are_parameterized_and_invalid_json_safe():
    conditions, params = sqlite_facet_conditions(
        {"stage": "S2", "layer": "G", "geo": "越南", "business": "B4"},
        doc_alias="doc",
    )

    sql = " AND ".join(conditions)
    assert "json_valid(doc.metadata)" in sql
    assert "S2" not in sql and "B4" not in sql
    assert params == [
        "S2",
        "S2",
        "S2",
        "G",
        "越南",
        "越南",
        "B4",
        "B4.%",
        "B4",
    ]
