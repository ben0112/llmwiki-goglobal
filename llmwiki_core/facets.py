"""Backend-neutral facet validation and wiki-page facet rollups."""

FACET_KEYS = (
    "stage",
    "domain",
    "genre",
    "rule",
    "evidence",
    "origin",
    "dept",
    "country",
    "region",
    "industry",
    "mode",
    "timeliness",
    "state",
    "business",
    "entry_id",
)

_TIMELINESS_ORDER = {"M1": 0, "M2": 1, "M3": 2}

SCALAR_FACET_PATHS = {
    "genre": "genre",
    "evidence": "evidence",
    "origin": "origin",
    "timeliness": "timeliness",
    "state": "lifecycle_state",
    "entry_id": "entry_id",
}
ARRAY_FACET_PATHS = {
    "rule": "rule_type",
    "dept": "gov_dept",
    "region": "geo_region",
    "industry": "industry",
    "mode": "mode",
}
PRIMARY_EXTENSION_FACET_PATHS = {
    "stage": ("stage", "stage_ext"),
    "domain": ("domain", "domain_ext"),
}


class UnknownFacetError(ValueError):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"unknown facet '{key}'; valid facets: {', '.join(FACET_KEYS)}")


def validate_facets(facets: dict | None) -> dict[str, str]:
    """Normalize facet keys and values while rejecting unknown dimensions."""
    if not facets:
        return {}
    clean: dict[str, str] = {}
    for key, value in facets.items():
        normalized_key = str(key).strip()
        if normalized_key not in FACET_KEYS:
            raise UnknownFacetError(normalized_key)
        normalized_value = str(value).strip()
        if normalized_value:
            clean[normalized_key] = normalized_value
    return clean


def postgres_facet_conditions(
    facets: dict | None,
    start_index: int,
    doc_alias: str = "d",
) -> tuple[list[str], list[object]]:
    """Compile validated facet predicates with numbered Postgres parameters."""
    meta = f"{doc_alias}.metadata"
    conditions: list[str] = []
    params: list[object] = []
    next_index = start_index

    def bind(value: object) -> int:
        nonlocal next_index
        params.append(value)
        index = next_index
        next_index += 1
        return index

    for key, value in validate_facets(facets).items():
        if key == "timeliness":
            index = bind(value)
            conditions.append(
                f"({meta}->>'timeliness' = ${index} OR "
                f"{meta}#>>'{{facet_rollup,timeliness_worst}}' = ${index})"
            )
        elif key in SCALAR_FACET_PATHS:
            conditions.append(
                f"{meta}->>'{SCALAR_FACET_PATHS[key]}' = ${bind(value)}"
            )
        elif key in ARRAY_FACET_PATHS:
            conditions.append(
                f"{meta}->'{ARRAY_FACET_PATHS[key]}' ? ${bind(value)}"
            )
        elif key in PRIMARY_EXTENSION_FACET_PATHS:
            primary, extension = PRIMARY_EXTENSION_FACET_PATHS[key]
            index = bind(value)
            conditions.append(
                f"({meta}->>'{primary}' = ${index} OR "
                f"{meta}->'{extension}' ? ${index} OR "
                f"{meta}#>'{{facet_rollup,{key}}}' ? ${index})"
            )
        elif key == "country":
            index = bind(value)
            conditions.append(
                f"({meta}->'geo_country' ? ${index} OR "
                f"{meta}->'geo_country_names' ? ${index} OR "
                f"{meta}#>'{{facet_rollup,country}}' ? ${index})"
            )
        elif key == "business":
            index = bind(value)
            if "." in value:
                conditions.append(
                    f"({meta}#>>'{{business,code}}' = ${index} OR "
                    f"{meta}#>'{{facet_rollup,business}}' ? ${index})"
                )
            else:
                prefix_index = bind(f"{value}.%")
                conditions.append(
                    f"({meta}#>>'{{business,code}}' = ${index} OR "
                    f"{meta}#>>'{{business,code}}' LIKE ${prefix_index} OR "
                    f"{meta}#>'{{facet_rollup,business}}' ? ${index})"
                )
    return conditions, params


def rollup_from_metas(metas: list[dict], computed_at: str) -> dict | None:
    """Aggregate classified corpus-entry metadata for a citing wiki page."""
    stages: set[str] = set()
    domains: set[str] = set()
    countries: set[str] = set()
    business: set[str] = set()
    worst: str | None = None
    count = 0
    for metadata in metas:
        if not isinstance(metadata, dict) or not metadata.get("entry_id"):
            continue
        count += 1
        for key, accumulator in (("stage", stages), ("domain", domains)):
            value = metadata.get(key)
            if value:
                accumulator.add(str(value))
            for extension in metadata.get(f"{key}_ext") or []:
                if extension:
                    accumulator.add(str(extension))
        for code in metadata.get("geo_country") or []:
            if code:
                countries.add(str(code))
        business_value = metadata.get("business")
        if isinstance(business_value, dict) and business_value.get("code"):
            business.add(str(business_value["code"]))
        timeliness = metadata.get("timeliness")
        if timeliness in _TIMELINESS_ORDER and (
            worst is None or _TIMELINESS_ORDER[timeliness] < _TIMELINESS_ORDER[worst]
        ):
            worst = timeliness
    if count == 0:
        return None
    return {
        "stage": sorted(stages),
        "domain": sorted(domains),
        "country": sorted(countries),
        "business": sorted(business),
        "timeliness_worst": worst,
        "entry_count": count,
        "computed_at": computed_at,
    }


def apply_rollup(metadata: dict, rollup: dict | None) -> bool:
    """Mutate page metadata with a rollup and report substantive changes."""

    def signature(value: dict | None) -> dict:
        return {key: item for key, item in (value or {}).items() if key != "computed_at"}

    current = metadata.get("facet_rollup")
    if not isinstance(current, dict):
        current = None
    if rollup is None:
        if "facet_rollup" in metadata:
            del metadata["facet_rollup"]
            return True
        return False
    if signature(current) != signature(rollup):
        metadata["facet_rollup"] = rollup
        return True
    return False
