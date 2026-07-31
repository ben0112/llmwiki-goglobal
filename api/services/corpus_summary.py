"""Backend-neutral corpus metadata aggregation for bounded read models."""

from datetime import date
from typing import Any

from .read_models import CorpusSummary


def _list_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float)) and item]


def _scalar_value(value: Any) -> list[str]:
    return [str(value)] if isinstance(value, (str, int, float)) and value else []


def _facet_values(metadata: dict[str, Any], key: str) -> list[str]:
    if key == "stage":
        return [
            *_scalar_value(metadata.get("stage")),
            *_list_values(metadata.get("stage_ext")),
        ]
    if key == "layer":
        domain = _scalar_value(metadata.get("domain"))
        return [domain[0][0]] if domain and domain[0] else []
    if key == "domain":
        return [
            *_scalar_value(metadata.get("domain")),
            *_list_values(metadata.get("domain_ext")),
        ]
    if key == "geo":
        return [
            *_list_values(metadata.get("geo_region")),
            *_list_values(metadata.get("geo_country_names")),
        ]
    if key == "country":
        return [
            *_list_values(metadata.get("geo_country")),
            *_list_values(metadata.get("geo_country_names")),
        ]
    paths = {
        "genre": "genre",
        "rule": "rule_type",
        "evidence": "evidence",
        "origin": "origin",
        "dept": "gov_dept",
        "region": "geo_region",
        "industry": "industry",
        "timeliness": "timeliness",
        "state": "lifecycle_state",
    }
    if key == "business":
        business = metadata.get("business")
        values = _scalar_value(business.get("code")) if isinstance(business, dict) else []
        return [value for value in values if value != "待定"]
    value = metadata.get(paths.get(key, key))
    values = _list_values(value) if isinstance(value, list) else _scalar_value(value)
    if key == "rule":
        return [item for item in values if item != "R0"]
    if key == "industry":
        return [item for item in values if item != "通用"]
    return values


def _matches(metadata: dict[str, Any], filters: dict[str, str], skip: str | None = None) -> bool:
    for key, value in filters.items():
        if key == skip:
            continue
        values = _facet_values(metadata, key)
        if key == "business" and "." not in value:
            if not any(item == value or item.startswith(f"{value}.") for item in values):
                return False
        elif value not in values:
            return False
    return True


def build_summary(rows: list[dict[str, Any]], filters: dict[str, str], revision: int) -> CorpusSummary:
    """Match the corpus UI's facet, coverage, business, and quality semantics."""
    filtered = [row for row in rows if _matches(row["metadata"], filters)]
    facet_keys = (
        "stage",
        "layer",
        "domain",
        "genre",
        "rule",
        "evidence",
        "origin",
        "dept",
        "geo",
        "country",
        "region",
        "industry",
        "mode",
        "timeliness",
        "state",
        "business",
    )
    facets: dict[str, dict[str, int]] = {}
    for key in facet_keys:
        counts: dict[str, int] = {}
        for row in rows:
            if not _matches(row["metadata"], filters, skip=key):
                continue
            for value in set(_facet_values(row["metadata"], key)):
                counts[value] = counts.get(value, 0) + 1
        facets[key] = counts

    stages = ("S0", "S1", "S2", "S3", "S4")
    layers = ("G", "C", "O", "Z", "X")
    coverage_counts = {stage: {layer: 0 for layer in layers} for stage in stages}
    classes: dict[str, int] = {}
    scenes: dict[str, int] = {}
    completeness = on_time = pending = 0
    required = (
        "stage",
        "domain",
        "genre",
        "evidence",
        "origin",
        "gov_dept",
        "timeliness",
        "lifecycle_state",
        "review_due",
    )
    today = date.today().isoformat()
    for row in filtered:
        metadata = row["metadata"]
        stage = metadata.get("stage")
        domain = str(metadata.get("domain") or "")
        if stage in coverage_counts and domain[:1] in coverage_counts[stage]:
            coverage_counts[stage][domain[0]] += 1
        business = metadata.get("business")
        if isinstance(business, dict) and business.get("code") and business.get("code") != "待定":
            code = str(business["code"])
            class_code = code.split(".")[0]
            classes[class_code] = classes.get(class_code, 0) + 1
            scenes[code] = scenes.get(code, 0) + 1
        if all(metadata.get(field) for field in required):
            completeness += 1
        if str(metadata.get("review_due") or "")[:10] >= today:
            on_time += 1
        if metadata.get("lifecycle_state") == "待复核":
            pending += 1
    filled = sum(coverage_counts[stage][layer] > 0 for stage in stages for layer in ("G", "C", "O", "Z"))
    return CorpusSummary(
        revision=revision,
        total_count=len(rows),
        filtered_count=len(filtered),
        facets=facets,
        coverage={"counts": coverage_counts, "total": len(filtered)},
        business_classes=classes,
        business_scenes=scenes,
        kpis={
            "total": len(filtered),
            "completeness": completeness,
            "shelf_filled": filled,
            "shelf_total": 20,
            "on_time": on_time,
            "pending_review": pending,
            "cited": None,
            "wiki_covered": None,
            "wiki_cells_with_entries": 0,
        },
    )


__all__ = ["build_summary"]
