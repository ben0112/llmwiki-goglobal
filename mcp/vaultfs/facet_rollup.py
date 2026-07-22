"""维基页面分面聚合(facet_rollup):从页面引用的语料条目自动汇总八维范围。

页面级不设人工八维,全部由引用条目聚合而来:stage/domain/country/business
取并集,时效取最紧(M1 > M2 > M3)。此文件在 api/services 与 mcp/vaultfs
各有一份拷贝(与 parse_citation_filename 同一模式),修改需两处同步。
"""

from __future__ import annotations

_TIMELINESS_ORDER = {"M1": 0, "M2": 1, "M3": 2}


def rollup_from_metas(metas: list[dict], computed_at: str) -> dict | None:
    """聚合被引条目的 metadata;无有效条目时返回 None(页面不带 rollup)。"""
    stages: set[str] = set()
    domains: set[str] = set()
    countries: set[str] = set()
    business: set[str] = set()
    worst: str | None = None
    count = 0
    for meta in metas:
        if not isinstance(meta, dict) or not meta.get("entry_id"):
            continue  # 只聚合语料条目(entry_id 为条目标识)
        count += 1
        for key, acc in (("stage", stages), ("domain", domains)):
            value = meta.get(key)
            if value:
                acc.add(str(value))
            for ext in meta.get(f"{key}_ext") or []:
                if ext:
                    acc.add(str(ext))
        for code in meta.get("geo_country") or []:
            if code:
                countries.add(str(code))
        biz = meta.get("business")
        if isinstance(biz, dict) and biz.get("code"):
            business.add(str(biz["code"]))
        t = meta.get("timeliness")
        if t in _TIMELINESS_ORDER and (worst is None or _TIMELINESS_ORDER[t] < _TIMELINESS_ORDER[worst]):
            worst = t
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


def apply_rollup(meta: dict, rollup: dict | None) -> bool:
    """把 rollup 并入页面 metadata;返回是否有实质变化(忽略 computed_at)。"""
    def _sig(d: dict | None) -> dict:
        return {k: v for k, v in (d or {}).items() if k != "computed_at"}

    current = meta.get("facet_rollup") if isinstance(meta.get("facet_rollup"), dict) else None
    if rollup is None:
        if "facet_rollup" in meta:
            del meta["facet_rollup"]
            return True
        return False
    if _sig(current) != _sig(rollup):
        meta["facet_rollup"] = rollup
        return True
    return False
