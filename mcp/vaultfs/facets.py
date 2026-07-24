"""Facet filters over documents.metadata — the 八维 classification dimensions.

Corpus entries imported by corpus/import_annotations.py carry a structured
eight-dimension record in documents.metadata. Each facet below maps to a SQL
condition on that JSON; multi-value dimensions match the primary label or any
secondary label. Documents without corpus metadata simply never match, so a
facet filter implicitly narrows the search to classified corpus entries.

The same facet keys work on both backends; only the SQL dialect differs.
"""

from llmwiki_core.facets import FACET_KEYS, UnknownFacetError, validate_facets

__all__ = [
    "FACET_KEYS",
    "UnknownFacetError",
    "postgres_facet_conditions",
    "sqlite_facet_conditions",
    "validate_facets",
]

# facet key -> ("scalar", json path) | ("array", json path) | special-cased
_SCALAR = {
    "genre": "$.genre",
    "evidence": "$.evidence",
    "origin": "$.origin",
    "timeliness": "$.timeliness",
    "state": "$.lifecycle_state",
    "entry_id": "$.entry_id",
}
_ARRAY = {
    "rule": "$.rule_type",
    "dept": "$.gov_dept",
    "region": "$.geo_region",
    "industry": "$.industry",
    "mode": "$.mode",
}
_PRIMARY_EXT = {
    "stage": ("$.stage", "$.stage_ext"),
    "domain": ("$.domain", "$.domain_ext"),
}


def sqlite_facet_conditions(facets: dict[str, str], doc_alias: str = "d") -> tuple[list[str], list]:
    """(conditions, params) for a SQLite WHERE clause. Facets must be validated."""
    raw_meta = f"{doc_alias}.metadata"
    # json_extract/json_each 遇到非法 JSON 或 BLOB 会让整条查询报错;
    # 用 CASE(求值顺序有保证)把坏值替换为空对象,坏行仅不命中分面。
    meta = f"CASE WHEN typeof({raw_meta})='text' AND json_valid({raw_meta}) THEN {raw_meta} ELSE '{{}}' END"
    conds: list[str] = []
    params: list = []

    def array_contains(path: str) -> str:
        return f"EXISTS (SELECT 1 FROM json_each({meta}, '{path}') WHERE json_each.value = ?)"

    # 维基页面经 facet_rollup(从引用条目聚合)也可被分面命中,
    # 让"找覆盖 S2×IDN 的现有页面"这类查询直接可用。
    for key, value in facets.items():
        if key == "timeliness":
            conds.append(
                f"(json_extract({meta}, '$.timeliness') = ? "
                f"OR json_extract({meta}, '$.facet_rollup.timeliness_worst') = ?)"
            )
            params.extend([value, value])
        elif key in _SCALAR:
            conds.append(f"json_extract({meta}, '{_SCALAR[key]}') = ?")
            params.append(value)
        elif key in _ARRAY:
            conds.append(array_contains(_ARRAY[key]))
            params.append(value)
        elif key in _PRIMARY_EXT:
            primary, ext = _PRIMARY_EXT[key]
            conds.append(
                f"(json_extract({meta}, '{primary}') = ? OR {array_contains(ext)} "
                f"OR {array_contains(f'$.facet_rollup.{key}')})"
            )
            params.extend([value, value, value])
        elif key == "country":
            conds.append(
                f"({array_contains('$.geo_country')} OR {array_contains('$.geo_country_names')} "
                f"OR {array_contains('$.facet_rollup.country')})"
            )
            params.extend([value, value, value])
        elif key == "business":
            if "." in value:
                conds.append(
                    f"(json_extract({meta}, '$.business.code') = ? OR {array_contains('$.facet_rollup.business')})"
                )
                params.extend([value, value])
            else:
                conds.append(
                    f"(json_extract({meta}, '$.business.code') = ? "
                    f"OR json_extract({meta}, '$.business.code') LIKE ? "
                    f"OR {array_contains('$.facet_rollup.business')})"
                )
                params.extend([value, f"{value}.%", value])
    if conds:
        conds.insert(0, f"{raw_meta} IS NOT NULL")
    return conds, params


def postgres_facet_conditions(
    facets: dict[str, str],
    start_index: int,
    doc_alias: str = "d",
) -> tuple[list[str], list]:
    """(conditions, params) for Postgres; placeholders start at $start_index."""
    meta = f"{doc_alias}.metadata"
    conds: list[str] = []
    params: list = []
    n = start_index

    def nxt(value) -> int:
        nonlocal n
        params.append(value)
        n += 1
        return n - 1

    for key, value in facets.items():
        if key == "timeliness":
            i = nxt(value)
            conds.append(f"({meta}->>'timeliness' = ${i} OR {meta}#>>'{{facet_rollup,timeliness_worst}}' = ${i})")
        elif key in _SCALAR:
            field = _SCALAR[key].removeprefix("$.")
            conds.append(f"{meta}->>'{field}' = ${nxt(value)}")
        elif key in _ARRAY:
            field = _ARRAY[key].removeprefix("$.")
            conds.append(f"{meta}->'{field}' ? ${nxt(value)}")
        elif key in _PRIMARY_EXT:
            primary, ext = (p.removeprefix("$.") for p in _PRIMARY_EXT[key])
            i = nxt(value)
            conds.append(
                f"({meta}->>'{primary}' = ${i} OR {meta}->'{ext}' ? ${i} OR {meta}#>'{{facet_rollup,{key}}}' ? ${i})"
            )
        elif key == "country":
            i = nxt(value)
            conds.append(
                f"({meta}->'geo_country' ? ${i} OR {meta}->'geo_country_names' ? ${i} "
                f"OR {meta}#>'{{facet_rollup,country}}' ? ${i})"
            )
        elif key == "business":
            if "." in value:
                i = nxt(value)
                conds.append(f"({meta}#>>'{{business,code}}' = ${i} OR {meta}#>'{{facet_rollup,business}}' ? ${i})")
            else:
                i = nxt(value)
                j = nxt(f"{value}.%")
                conds.append(
                    f"({meta}#>>'{{business,code}}' = ${i} OR {meta}#>>'{{business,code}}' LIKE ${j} "
                    f"OR {meta}#>'{{facet_rollup,business}}' ? ${i})"
                )
    return conds, params
