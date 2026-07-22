"""复核队列操作(L3-P3,本地模式):通过 / 修改标签 / 不收录。

条目以 documents.metadata 里的结构化八维记录为准直接重建 EntryRecord
(不走 parse_row 的容错解析——复核是人工校准,取值必须命中码表,否则 400)。
文件即真相源:每次操作重渲染条目 markdown 并落盘;阶段/大类等主标签变更导致
entry_id 与货架路径变化时,旧文件与旧索引行一并迁移。审计记录写入
documents.metadata.review(操作/时间/备注),供页面与报表追溯。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .codetable import CodeTable, DEFAULT_VERSION, load
from .derive import derive_business_code
from .import_annotations import entry_relative_path, render_entry_markdown, upsert_document
from .schema import EntryRecord, make_entry_id

# 复核对话框可改的主标签字段 → EntryRecord 属性
_LABEL_FIELDS = ("stage", "domain", "genre", "rule_type", "evidence",
                 "origin", "timeliness")

# 规范强制人工复核类(逐条,不允许批量):法律 C1、数据出境 R1
MANDATORY_DOMAINS = {"C1"}
MANDATORY_RULES = {"R1"}


class ReviewError(Exception):
    """取值非法 / 条目不存在等,由路由映射为 4xx。"""


def record_from_metadata(meta: dict, title: str) -> EntryRecord:
    rec = EntryRecord(title=title)
    for key, value in meta.items():
        if key in ("business", "review"):
            continue
        if hasattr(rec, key):
            setattr(rec, key, value)
    business = meta.get("business") or {}
    rec.business_code = business.get("code", "")
    rec.business_scene = business.get("scene", "")
    rec.business_class = business.get("class", "")
    rec.business_priority = business.get("priority", "")
    rec.business_pending = bool(business.get("pending", False))
    rec.issues = []
    return rec


def _validate_labels(labels: dict, table: CodeTable) -> dict:
    """校验并规范化人工标签;非法取值直接报错(复核不走兜底)。"""
    clean: dict = {}
    for key, value in labels.items():
        if key not in _LABEL_FIELDS:
            raise ReviewError(f"不支持修改的字段: {key}")
        if key == "stage":
            if value not in table.stage_codes:
                raise ReviewError(f"阶段取值不在码表: {value!r}")
        elif key == "domain":
            if value not in table.domain_codes:
                raise ReviewError(f"服务大类取值不在码表: {value!r}")
        elif key == "genre":
            name, _exact = table.normalize_genre(str(value))
            if name is None:
                raise ReviewError(f"体裁取值不在码表: {value!r}")
            value = name
        elif key == "rule_type":
            rules = value if isinstance(value, list) else [v for v in str(value).replace("、", "/").split("/") if v]
            valid = set(table.raw("rules"))
            bad = [r for r in rules if r not in valid]
            if bad:
                raise ReviewError(f"隐性规则取值不在码表: {bad}")
            value = rules or ["R0"]
        elif key == "evidence":
            if value not in table.raw("evidence"):
                raise ReviewError(f"证据强度取值不在码表: {value!r}")
        elif key == "origin":
            name = table.normalize_origin(str(value))
            if name is None:
                raise ReviewError(f"来源域取值不在码表: {value!r}")
            value = name
        elif key == "timeliness":
            if value not in table.raw("timeliness"):
                raise ReviewError(f"时效取值不在码表: {value!r}")
        clean[key] = value
    return clean


def _rederive_business(rec: EntryRecord, table: CodeTable) -> None:
    text = " ".join([rec.title, rec.reason, *rec.geo_country_names, *rec.industry])
    code = derive_business_code(rec.domain, "/".join(rec.rule_type), rec.genre,
                                rec.timeliness, rec.stage, text)
    rec.business_code = code
    rec.business_pending = code == "待定"
    if code != "待定":
        cls, cls_name, priority = table.business_class_of(code)
        rec.business_scene = table.business_scenes.get(code, "")
        rec.business_class = cls_name
        rec.business_priority = priority


def _citing_wiki_pages(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    """引用(cites)该条目的维基页面 id。条目删除/换架会级联清边,须先反查。"""
    return [r[0] for r in conn.execute(
        "SELECT r.source_document_id FROM document_references r "
        "JOIN documents d ON d.id = r.source_document_id "
        "WHERE r.reference_type = 'cites' AND r.target_document_id = ? "
        "AND d.source_kind = 'wiki'", (doc_id,))]


def _mark_pages_stale(conn: sqlite3.Connection, page_ids: list[str]) -> int:
    """把页面标记为待复查(stale);已标记的不重复计时。返回新标记数。"""
    if not page_ids:
        return 0
    placeholders = ",".join("?" for _ in page_ids)
    cur = conn.execute(
        f"UPDATE documents SET stale_since = datetime('now') "
        f"WHERE id IN ({placeholders}) AND stale_since IS NULL", page_ids)
    return cur.rowcount


def apply_review(workspace: Path, doc_id: str, action: str,
                 labels: dict | None = None, note: str = "",
                 version: str = DEFAULT_VERSION) -> dict:
    if action not in ("approve", "update", "exclude"):
        raise ReviewError(f"未知操作: {action}")
    table = load(version)
    today = date.today()
    conn = sqlite3.connect(str(workspace / ".llmwiki" / "index.db"))
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        # 改标/剔除会改变条目口径:引用它的维基页面需要标记复查
        citing = _citing_wiki_pages(conn, doc_id) if action in ("update", "exclude") else []
        row = conn.execute(
            "SELECT relative_path, title, metadata, user_id FROM documents WHERE id = ?",
            (doc_id,)).fetchone()
        if row is None:
            raise ReviewError("条目不存在")
        old_rel, title, meta_json, user_id = row
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            meta = {}
        if not meta.get("entry_id"):
            raise ReviewError("该文档不是语料条目")

        audit = {"action": action, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "note": note[:500]}

        if action == "exclude":
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            src = meta.get("source_relpath", "")
            if src:
                src_row = conn.execute(
                    "SELECT id FROM documents WHERE relative_path = ?", (src,)).fetchone()
                if src_row:
                    conn.execute(
                        "INSERT INTO corpus_pipeline (doc_id, state, attempts, error, entry_id, updated_at) "
                        "VALUES (?, 'excluded', 1, ?, ?, ?) "
                        "ON CONFLICT(doc_id) DO UPDATE SET state='excluded', "
                        "error=excluded.error, updated_at=excluded.updated_at",
                        (src_row[0], f"人工不收录: {note[:200]}", meta.get("entry_id", ""),
                         audit["at"]))
            stale_pages = _mark_pages_stale(conn, citing)
            conn.commit()
            (workspace / old_rel).unlink(missing_ok=True)
            return {"action": "exclude", "entry_id": meta.get("entry_id", ""),
                    "stale_pages": stale_pages}

        rec = record_from_metadata(meta, title or "")
        if action == "update" and labels:
            for key, value in _validate_labels(labels, table).items():
                setattr(rec, key, value)
            if "timeliness" in labels:
                rec.review_due = (today + timedelta(days=table.review_days(rec.timeliness))).isoformat()
            _rederive_business(rec, table)
            rec.entry_id = make_entry_id(rec, rec.source_relpath or old_rel, table)
        # 人工结论:进入已入库,清空机器阶段的问题记录
        rec.lifecycle_state = "已入库"
        rec.spec_version = rec.spec_version or table.version

        src_body = None
        if rec.source_relpath:
            src_row = conn.execute(
                "SELECT content FROM documents WHERE relative_path = ?",
                (rec.source_relpath,)).fetchone()
            src_body = src_row[0] if src_row else None
        content = render_entry_markdown(rec, src_body, today.isoformat())
        new_rel = entry_relative_path(rec)

        if new_rel != old_rel:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            (workspace / old_rel).unlink(missing_ok=True)
        full = workspace / new_rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        upsert_document(conn, user_id, rec, new_rel, content, full.stat().st_mtime_ns)

        # 审计写入索引 metadata(操作性数据,不入文件 frontmatter)
        new_meta = rec.to_metadata()
        if rec.business_code:
            new_meta["business"] = {"code": rec.business_code, "scene": rec.business_scene,
                                    "class": rec.business_class, "priority": rec.business_priority,
                                    "pending": rec.business_pending}
        history = meta.get("review", [])
        if isinstance(history, dict):
            history = [history]
        new_meta["review"] = ([*history, audit])[-10:]
        conn.execute("UPDATE documents SET metadata = ? WHERE relative_path = ?",
                     (json.dumps(new_meta, ensure_ascii=False), new_rel))
        # 仅"改标"联动复查(approve 无口径变化);labels 为空的 update 同 approve
        stale_pages = _mark_pages_stale(conn, citing) if (action == "update" and labels) else 0
        conn.commit()
        return {"action": action, "entry_id": rec.entry_id, "relative_path": new_rel,
                "moved": new_rel != old_rel, "stale_pages": stale_pages}
    finally:
        conn.close()


def prepare_reprocess(workspace: Path, doc_id: str) -> dict:
    """重新识别入库:删除旧条目(引用页标 stale)、清空分类状态、源文档
    重置为待提取 —— 供「重新识别入库分类」按钮走完整链路。

    调用方随后应触发源文档重新提取(带 OCR 兜底)并起一轮分类。
    """
    conn = sqlite3.connect(str(workspace / ".llmwiki" / "index.db"))
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        citing = _citing_wiki_pages(conn, doc_id)
        row = conn.execute(
            "SELECT relative_path, metadata FROM documents WHERE id = ?",
            (doc_id,)).fetchone()
        if row is None:
            raise ReviewError("条目不存在")
        old_rel, meta_json = row
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            meta = {}
        if not meta.get("entry_id"):
            raise ReviewError("该文档不是语料条目")
        src = meta.get("source_relpath", "")
        src_row = conn.execute(
            "SELECT id FROM documents WHERE relative_path = ?", (src,)).fetchone() if src else None
        if not src_row:
            raise ReviewError("找不到对应源文件(可能已被删除或移动)")
        source_id = src_row[0]

        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("DELETE FROM corpus_pipeline WHERE doc_id = ?", (source_id,))
        conn.execute(
            "UPDATE documents SET status = 'pending', error_message = NULL, "
            "extraction_attempts = 0, "   # 手动重来 = 解除失败隔离
            "updated_at = datetime('now') WHERE id = ?", (source_id,))
        stale_pages = _mark_pages_stale(conn, citing)
        conn.commit()
        (workspace / old_rel).unlink(missing_ok=True)
        return {"entry_id": meta.get("entry_id", ""), "source_doc_id": source_id,
                "stale_pages": stale_pages}
    finally:
        conn.close()


def codetable_options(version: str = DEFAULT_VERSION) -> dict:
    """复核对话框的下拉取值(码表主值,含中文名)。"""
    table = load(version)
    raw = table._data
    def pairs(key):
        v = raw.get(key, {})
        out = []
        for code, val in v.items():
            name = val.get("name") if isinstance(val, dict) else val
            out.append({"code": code, "name": str(name)})
        return out
    return {
        "version": table.version,
        "stages": pairs("stages"),
        "domains": pairs("domains"),
        "genres": [{"code": g["name"], "name": g["name"]} for g in raw.get("genres", {}).values()],
        "rules": pairs("rules"),
        "evidence": pairs("evidence"),
        "origins": [{"code": v, "name": v} for v in raw.get("origins", {}).values()],
        "timeliness": pairs("timeliness"),
    }
