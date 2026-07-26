"""Pure Postgres retrieval SQL compilation shared by serving and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .facets import postgres_facet_conditions
from .search import SearchArea, SearchQuery, SearchScope


@dataclass(frozen=True, slots=True)
class PostgresLexicalQuery:
    sql: str
    params: tuple[object, ...]


def logical_glob_to_sql_like(path_glob: str) -> str:
    """Translate a normalized logical glob into an escaped SQL LIKE value."""
    directory = path_glob.endswith("/") or (
        "*" not in path_glob
        and "." not in path_glob.rsplit("/", 1)[-1]
        and path_glob != "/"
    )
    escaped: list[str] = []
    index = 0
    while index < len(path_glob):
        char = path_glob[index]
        if char == "*":
            if index + 1 < len(path_glob) and path_glob[index + 1] == "*":
                index += 1
            escaped.append("%")
        elif char in {"%", "_", "\\"}:
            escaped.append("\\" + char)
        else:
            escaped.append(char)
        index += 1
    pattern = "".join(escaped)
    if path_glob == "/":
        return "/%"
    if directory:
        return pattern.rstrip("/") + "/%"
    return pattern


def postgres_document_filter_conditions(
    query: SearchQuery,
    params: list[Any],
    *,
    doc_alias: str,
    chunk_alias: str,
) -> list[str]:
    """Append all document/chunk filter params and return their predicates."""

    def bind(value: object) -> str:
        params.append(value)
        return f"${len(params)}"

    conditions: list[str] = []
    if query.annotated_only:
        conditions.append(f"{chunk_alias}.has_highlight = true")
    if query.area is SearchArea.WIKI:
        conditions.append(f"{doc_alias}.source_kind = 'wiki'")
    elif query.area is SearchArea.SOURCES:
        conditions.append(f"{doc_alias}.source_kind != 'wiki'")
    if query.document_kinds:
        kinds = bind([kind.value for kind in query.document_kinds])
        conditions.append(f"{doc_alias}.source_kind = ANY({kinds}::text[])")
    if query.path_glob is not None:
        path_pattern = bind(logical_glob_to_sql_like(query.path_glob))
        conditions.append(
            f"({doc_alias}.path || {doc_alias}.filename) LIKE {path_pattern} ESCAPE '\\'"
        )
    if query.tags:
        tags = bind(list(query.tags))
        conditions.append(
            "ARRAY(SELECT lower(tag) FROM unnest(COALESCE("
            f"{doc_alias}.tags, ARRAY[]::text[])) tag) @> {tags}::text[]"
        )
    facet_conditions, facet_params = postgres_facet_conditions(
        dict(query.facets),
        start_index=len(params) + 1,
        doc_alias=doc_alias,
    )
    conditions.extend(facet_conditions)
    params.extend(facet_params)
    return conditions


def compile_postgres_lexical_query(
    user_id: object,
    knowledge_base_id: object,
    query: SearchQuery,
) -> PostgresLexicalQuery:
    """Compile the exact serving PGroonga lexical candidate query."""
    params: list[Any] = [knowledge_base_id, query.text, user_id]

    def bind(value: object) -> str:
        params.append(value)
        return f"${len(params)}"

    where = [
        "dc.knowledge_base_id = $1",
        "d.knowledge_base_id = $1",
        "dc.user_id = $3",
        "d.user_id = $3",
        "dc.content &@~ $2",
        "d.status != 'failed'",
        "NOT d.archived",
    ]
    where.extend(
        postgres_document_filter_conditions(
            query,
            params,
            doc_alias="d",
            chunk_alias="dc",
        )
    )

    scope_where = ""
    if query.scope is SearchScope.SOURCE:
        scope_where = "WHERE source_hit"
    elif query.scope is SearchScope.ANNOTATIONS:
        scope_where = "WHERE annotation_hit"
    limit_param = bind(query.candidate_limit)

    sql = (
        "WITH labeled AS ("
        "SELECT dc.document_id, dc.document_version, dc.content, "
        "dc.source_content, dc.annotations_text, dc.has_highlight, "
        "dc.page, dc.header_breadcrumb, dc.chunk_index, "
        "d.filename, d.title, d.path, d.file_type, d.tags, d.metadata, "
        "d.source_kind, pgroonga_score(dc.tableoid, dc.ctid) AS score, "
        "(dc.source_content &@~ $2) AS source_hit, "
        "(dc.annotations_text IS NOT NULL AND dc.annotations_text &@~ $2) "
        "AS annotation_hit "
        "FROM document_chunks dc JOIN documents d ON dc.document_id = d.id "
        f"WHERE {' AND '.join(where)}"
        "), filtered AS ("
        f"SELECT * FROM labeled {scope_where}"
        "), counted AS ("
        "SELECT *, COUNT(*) OVER () AS candidate_count FROM filtered"
        ") SELECT * FROM counted "
        "ORDER BY score DESC, document_id, document_version, chunk_index "
        f"LIMIT {limit_param}"
    )
    return PostgresLexicalQuery(sql=sql, params=tuple(params))


__all__ = [
    "PostgresLexicalQuery",
    "compile_postgres_lexical_query",
    "logical_glob_to_sql_like",
    "postgres_document_filter_conditions",
]
