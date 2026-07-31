from pathlib import Path
from uuid import UUID

import pytest

from llmwiki_core.documents import DocumentKind
from llmwiki_core.postgres_retrieval import (
    compile_postgres_lexical_query,
    logical_glob_to_sql_like,
    postgres_document_filter_conditions,
)
from llmwiki_core.search import SearchQuery

USER_ID = UUID("00000000-0000-0000-0000-000000000001")
KNOWLEDGE_BASE_ID = UUID("00000000-0000-0000-0000-000000000002")
REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    ("path_glob", "expected"),
    [
        ("/", "/%"),
        ("/target/", "/target/%"),
        ("/target", "/target/%"),
        ("/target/*.md", "/target/%.md"),
        (r"/literal/%_\\?.md", r"/literal/\%\_\\\\?.md"),
    ],
)
def test_shared_postgres_glob_compiler_preserves_logical_semantics(path_glob, expected):
    assert logical_glob_to_sql_like(path_glob) == expected


def test_shared_postgres_document_filters_bind_every_filter_before_limit():
    query = SearchQuery.build(
        text="private query",
        limit=2,
        candidate_limit=7,
        area="sources",
        scope="annotations",
        facets={"stage": "S2", "business": "02"},
        path_glob="/target/*.md",
        tags=["Alpha"],
        document_kinds=[DocumentKind.SOURCE],
        annotated_only=True,
    )
    params = [KNOWLEDGE_BASE_ID, query.text, USER_ID]

    conditions = postgres_document_filter_conditions(
        query,
        params,
        doc_alias="d",
        chunk_alias="dc",
    )

    assert conditions == [
        "dc.has_highlight = true",
        "d.source_kind != 'wiki'",
        "d.source_kind = ANY($4::text[])",
        "(d.path || d.filename) LIKE $5 ESCAPE '\\'",
        "ARRAY(SELECT lower(tag) FROM unnest(COALESCE(d.tags, ARRAY[]::text[])) tag) @> $6::text[]",
        "(d.metadata->>'stage' = $7 OR d.metadata->'stage_ext' ? $7 OR d.metadata#>'{facet_rollup,stage}' ? $7)",
        "(d.metadata#>>'{business,code}' = $8 OR d.metadata#>>'{business,code}' LIKE $9 "
        "OR d.metadata#>'{facet_rollup,business}' ? $8)",
    ]
    assert params == [
        KNOWLEDGE_BASE_ID,
        "private query",
        USER_ID,
        ["source"],
        "/target/%.md",
        ["alpha"],
        "S2",
        "02",
        "02.%",
    ]


@pytest.mark.parametrize(
    ("scope", "scope_clause"),
    [
        ("all", "SELECT * FROM labeled "),
        ("source", "SELECT * FROM labeled WHERE source_hit"),
        ("annotations", "SELECT * FROM labeled WHERE annotation_hit"),
    ],
)
def test_shared_postgres_lexical_compiler_is_exact_pgroonga_serving_query(
    scope,
    scope_clause,
):
    query = SearchQuery.build(
        text="private query",
        limit=2,
        candidate_limit=7,
        scope=scope,
    )

    compiled = compile_postgres_lexical_query(USER_ID, KNOWLEDGE_BASE_ID, query)

    assert compiled.params == (KNOWLEDGE_BASE_ID, "private query", USER_ID, 7)
    assert "dc.content &@~ $2" in compiled.sql
    assert "pgroonga_score(dc.tableoid, dc.ctid) AS score" in compiled.sql
    assert "dc.document_version = d.version" in compiled.sql
    assert "d.status != 'failed'" in compiled.sql
    assert "NOT d.archived" in compiled.sql
    assert "(dc.source_content &@~ $2) AS source_hit" in compiled.sql
    assert "dc.annotations_text &@~ $2" in compiled.sql
    assert scope_clause in compiled.sql
    assert "COUNT(*) OVER () AS candidate_count" in compiled.sql
    assert "ORDER BY score DESC, document_id, document_version, chunk_index" in compiled.sql
    assert compiled.sql.endswith("LIMIT $4")
    assert "to_tsvector" not in compiled.sql
    assert "plainto_tsquery" not in compiled.sql


def test_production_postgres_retrieval_has_no_legacy_tsvector_query_copy():
    for relative_path in (
        "api/scripts/retrieval_eval.py",
        "api/services/vector_store.py",
        "mcp/vaultfs/postgres.py",
        "llmwiki_core/postgres_retrieval.py",
    ):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "to_tsvector" not in source
        assert "plainto_tsquery" not in source
        assert "ts_rank_cd" not in source
