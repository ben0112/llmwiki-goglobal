import pytest

from llmwiki_core.facets import (
    UnknownFacetError,
    apply_rollup,
    rollup_from_metas,
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
