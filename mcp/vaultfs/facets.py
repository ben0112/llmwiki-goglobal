"""Facet filters over documents.metadata — the 八维 classification dimensions.

Corpus entries imported by corpus/import_annotations.py carry a structured
eight-dimension record in documents.metadata. Each facet below maps to a SQL
condition on that JSON; multi-value dimensions match the primary label or any
secondary label. Documents without corpus metadata simply never match, so a
facet filter implicitly narrows the search to classified corpus entries.

The same facet keys work on both backends; only the SQL dialect differs.
"""

FACET_KEYS = (
    "stage",       # S0–S4 (primary or secondary)
    "domain",      # G1–G4 / C1–C5 / O1–O6 / Z1–Z4 / X9 (primary or secondary)
    "genre",       # 体裁 canonical name, e.g. 政策法规
    "rule",        # R0–R6 (any)
    "evidence",    # E0–E4
    "origin",      # 目的地国 / 国际 / 国内 / 混合
    "dept",        # 归口部门 canonical name (any)
    "country",     # ISO 3166 alpha-3 or Chinese name (any)
    "region",      # 区域 name, e.g. 东盟 (any)
    "industry",    # 行业 (any)
    "mode",        # 出海形态/进入模式 (any)
    "timeliness",  # M1–M3
    "state",       # 生命周期状态, e.g. 待复核
    "business",    # 业务码: scene "B4.14" exact, or class "B4" prefix
    "entry_id",    # composite entry id, exact
)

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


class UnknownFacetError(ValueError):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"unknown facet '{key}'; valid facets: {', '.join(FACET_KEYS)}")


def validate_facets(facets: dict | None) -> dict[str, str]:
    """Normalize a facets dict: string keys/values, known keys only."""
    if not facets:
        return {}
    clean: dict[str, str] = {}
    for key, value in facets.items():
        k = str(key).strip()
        if k not in FACET_KEYS:
            raise UnknownFacetError(k)
        v = str(value).strip()
        if v:
            clean[k] = v
    return clean


def sqlite_facet_conditions(facets: dict[str, str], doc_alias: str = "d") -> tuple[list[str], list]:
    """(conditions, params) for a SQLite WHERE clause. Facets must be validated."""
    meta = f"{doc_alias}.metadata"
    conds: list[str] = []
    params: list = []

    def array_contains(path: str) -> str:
        return f"EXISTS (SELECT 1 FROM json_each({meta}, '{path}') WHERE json_each.value = ?)"

    for key, value in facets.items():
        if key in _SCALAR:
            conds.append(f"json_extract({meta}, '{_SCALAR[key]}') = ?")
            params.append(value)
        elif key in _ARRAY:
            conds.append(array_contains(_ARRAY[key]))
            params.append(value)
        elif key in _PRIMARY_EXT:
            primary, ext = _PRIMARY_EXT[key]
            conds.append(f"(json_extract({meta}, '{primary}') = ? OR {array_contains(ext)})")
            params.extend([value, value])
        elif key == "country":
            conds.append(f"({array_contains('$.geo_country')} OR {array_contains('$.geo_country_names')})")
            params.extend([value, value])
        elif key == "business":
            if "." in value:
                conds.append(f"json_extract({meta}, '$.business.code') = ?")
                params.append(value)
            else:
                conds.append(
                    f"(json_extract({meta}, '$.business.code') = ? "
                    f"OR json_extract({meta}, '$.business.code') LIKE ?)"
                )
                params.extend([value, f"{value}.%"])
    if conds:
        conds.insert(0, f"{meta} IS NOT NULL")
    return conds, params


def postgres_facet_conditions(
    facets: dict[str, str], start_index: int, doc_alias: str = "d",
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
        if key in _SCALAR:
            field = _SCALAR[key].removeprefix("$.")
            conds.append(f"{meta}->>'{field}' = ${nxt(value)}")
        elif key in _ARRAY:
            field = _ARRAY[key].removeprefix("$.")
            conds.append(f"{meta}->'{field}' ? ${nxt(value)}")
        elif key in _PRIMARY_EXT:
            primary, ext = (p.removeprefix("$.") for p in _PRIMARY_EXT[key])
            i = nxt(value)
            conds.append(f"({meta}->>'{primary}' = ${i} OR {meta}->'{ext}' ? ${i})")
        elif key == "country":
            i = nxt(value)
            conds.append(f"({meta}->'geo_country' ? ${i} OR {meta}->'geo_country_names' ? ${i})")
        elif key == "business":
            if "." in value:
                conds.append(f"{meta}#>>'{{business,code}}' = ${nxt(value)}")
            else:
                i = nxt(value)
                j = nxt(f"{value}.%")
                conds.append(f"({meta}#>>'{{business,code}}' = ${i} OR {meta}#>>'{{business,code}}' LIKE ${j})")
    return conds, params
