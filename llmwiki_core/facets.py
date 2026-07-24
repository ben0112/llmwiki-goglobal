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
