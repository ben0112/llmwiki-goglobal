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


# ---------------------------------------------------------------------------
# 批量刷新(api 侧独有):挂在引用图重建之后,rollup 与引用边同源同步。
# ---------------------------------------------------------------------------

import json
import logging
from datetime import date

logger = logging.getLogger(__name__)


def _parse_meta(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def refresh_rollups_local(db) -> int:
    """重算所有本地维基页的 facet_rollup;返回发生变化的页面数。"""
    from infra.db.sqlite import serialized_write

    cursor = await db.execute(
        "SELECT r.source_document_id, d.metadata FROM document_references r "
        "JOIN documents d ON d.id = r.target_document_id "
        "JOIN documents w ON w.id = r.source_document_id "
        "WHERE r.reference_type = 'cites' AND w.source_kind = 'wiki' "
        "AND d.relative_path LIKE 'corpus/%'"
    )
    by_page: dict[str, list[dict]] = {}
    for page_id, meta_raw in await cursor.fetchall():
        by_page.setdefault(page_id, []).append(_parse_meta(meta_raw))

    cursor = await db.execute(
        "SELECT id, metadata FROM documents WHERE source_kind = 'wiki'"
    )
    pages = await cursor.fetchall()

    today = date.today().isoformat()
    updated = 0
    async with serialized_write():
        for page_id, meta_raw in pages:
            meta = _parse_meta(meta_raw)
            rollup = rollup_from_metas(by_page.get(page_id, []), today)
            if apply_rollup(meta, rollup):
                await db.execute(
                    "UPDATE documents SET metadata = ? WHERE id = ?",
                    (json.dumps(meta, ensure_ascii=False), page_id),
                )
                updated += 1
        await db.commit()
    return updated


async def refresh_rollups_hosted(conn, kb_id, user_id: str) -> int:
    """托管模式同逻辑:jsonb 全量读改写。"""
    rows = await conn.fetch(
        "SELECT r.source_document_id::text AS page_id, d.metadata "
        "FROM document_references r "
        "JOIN documents d ON d.id = r.target_document_id "
        "JOIN documents w ON w.id = r.source_document_id "
        "WHERE r.reference_type = 'cites' AND w.path LIKE '/wiki/%' "
        "AND d.path LIKE '/corpus/%' AND w.knowledge_base_id = $1 AND w.user_id = $2",
        kb_id, user_id,
    )
    by_page: dict[str, list[dict]] = {}
    for row in rows:
        by_page.setdefault(row["page_id"], []).append(_parse_meta(row["metadata"]))

    pages = await conn.fetch(
        "SELECT id::text, metadata FROM documents "
        "WHERE knowledge_base_id = $1 AND user_id = $2 AND path LIKE '/wiki/%' AND NOT archived",
        kb_id, user_id,
    )
    today = date.today().isoformat()
    updated = 0
    for row in pages:
        meta = _parse_meta(row["metadata"])
        rollup = rollup_from_metas(by_page.get(row["id"], []), today)
        if apply_rollup(meta, rollup):
            await conn.execute(
                "UPDATE documents SET metadata = $1::jsonb WHERE id = $2::uuid AND user_id = $3",
                json.dumps(meta, ensure_ascii=False), row["id"], user_id,
            )
            updated += 1
    return updated
