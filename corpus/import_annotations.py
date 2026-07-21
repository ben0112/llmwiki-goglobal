#!/usr/bin/env python3
"""导入八维标注明细 into an LLM Wiki workspace.

Reads the LLM 标注工具包 output (标注明细.csv, or 标注明细_业务视图.csv with the
derived business-view columns), validates every row against the versioned 码表,
then materializes each entry as:

  1. a markdown file with YAML frontmatter under <workspace>/corpus/<阶段-大类>/
     — the filesystem stays the source of truth, exactly like wiki pages;
  2. a documents row in <workspace>/.llmwiki/index.db whose metadata column
     carries the full structured 八维 record (facet search in a later phase
     reads this), with facet tags so today's tag filtering already works.

Chunking is left to the app: rows are written with parser=NULL, which the local
reconcile loop treats as "never chunked" and backfills on next `llmwiki open`.

Reports (导入报告.md, 复核队列.csv) land in <workspace>/.llmwiki/corpus_import/
so they are never indexed as corpus content.

Usage:
  python3 -m corpus.import_annotations --csv 标注明细_业务视图.csv \
      --workspace ~/goglobal-ws [--raw ../审核结果_deepseek/收录] [--dry-run]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import SPEC_VERSION
from .codetable import DEFAULT_VERSION, CodeTable, load
from .schema import ERROR, EntryRecord, parse_row

SUBDIR = "corpus"

_HDR_RE = re.compile(r"^(标题|URL|命中关键词|链接|来源|发布时间|栏目|接口ID|Title|Matched keywords)\s*[：:]\s*(.*)$", re.I)
_UNSAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z一-鿿·_-]+")


@dataclass
class ImportStats:
    total: int = 0
    imported: int = 0
    review: int = 0
    errors: int = 0
    warnings: int = 0
    skipped: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source text loading (收录/*.txt with the audit toolkit's header block)
# ---------------------------------------------------------------------------

def load_source_body(raw_root: Path, relpath: str) -> str | None:
    """Body text of a 收录 file, header lines stripped (mirrors parse_doc)."""
    path = raw_root / relpath
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines[:10]):
        if _HDR_RE.match(line.strip()) or (line.strip() == "" and body_start == i):
            body_start = i + 1
    return "\n".join(lines[body_start:]).strip()


# ---------------------------------------------------------------------------
# Markdown entry rendering
# ---------------------------------------------------------------------------

def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return json.dumps(str(value), ensure_ascii=False)  # double-quoted, YAML-safe


def _yaml_line(key: str, value, indent: int = 0) -> list:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{pad}{key}: {{}}"]
        lines = [f"{pad}{key}:"]
        for k, v in value.items():
            lines += _yaml_line(k, v, indent + 2)
        return lines
    if isinstance(value, list):
        rendered = ", ".join(_yaml_scalar(v) for v in value)
        return [f"{pad}{key}: [{rendered}]"]
    return [f"{pad}{key}: {_yaml_scalar(value)}"]


def render_entry_markdown(rec: EntryRecord, body: str | None, imported_on: str) -> str:
    fm = [
        *(_yaml_line("title", rec.title)),
        *(_yaml_line("date", rec.effective_date or imported_on)),
        *(_yaml_line("tags", rec.to_tags())),
        *(_yaml_line("entry_id", rec.entry_id)),
        *(_yaml_line("corpus", rec.to_metadata())),
    ]
    lines = ["---", *fm, "---", "", f"# {rec.title}", ""]
    if body:
        lines += [body, ""]
    else:
        lines += ["(元数据条目 — 原文未随导入提供;见来源链接。)", ""]
    footer = f"来源: {rec.source_site or '未知站点'}"
    if rec.source_url:
        footer = f"来源: [{rec.source_site or rec.source_url}]({rec.source_url})"
    lines += ["---", "", footer, ""]
    return "\n".join(lines)


def entry_relative_path(rec: EntryRecord) -> str:
    safe_id = _UNSAFE_FILENAME_RE.sub("_", rec.entry_id)
    return f"{SUBDIR}/{rec.stage}-{rec.domain}/{safe_id}.md"


# ---------------------------------------------------------------------------
# SQLite upsert
# ---------------------------------------------------------------------------

def upsert_document(conn: sqlite3.Connection, user_id: str, rec: EntryRecord,
                    rel_path: str, content: str, mtime_ns: int) -> str:
    """Insert or refresh one entry document; returns 'inserted' | 'updated'."""
    filename = rel_path.rsplit("/", 1)[-1]
    dir_path = "/" + rel_path.rsplit("/", 1)[0] + "/"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    tags_json = json.dumps(rec.to_tags(), ensure_ascii=False)
    metadata_json = json.dumps(rec.to_metadata(), ensure_ascii=False)

    row = conn.execute(
        "SELECT id, content_hash FROM documents WHERE relative_path = ?", (rel_path,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, file_size, status, content, tags, date, metadata, "
            "version, parser, content_hash, mtime_ns, last_indexed_at, document_number) "
            "VALUES (?, ?, ?, ?, ?, ?, 'source', 'md', ?, 'ready', ?, ?, ?, ?, 0, NULL, ?, ?, "
            "datetime('now'), (SELECT COALESCE(MAX(document_number), 0) + 1 FROM documents))",
            (str(uuid.uuid4()), user_id, filename, rec.title, dir_path, rel_path,
             len(content.encode("utf-8")), content, tags_json,
             rec.effective_date, metadata_json, content_hash, mtime_ns),
        )
        return "inserted"

    doc_id, old_hash = row
    if old_hash == content_hash:
        # Still refresh metadata/tags: review_due moves even when content doesn't.
        conn.execute(
            "UPDATE documents SET metadata = ?, tags = ?, updated_at = datetime('now') WHERE id = ?",
            (metadata_json, tags_json, doc_id),
        )
        return "updated"
    conn.execute(
        "UPDATE documents SET title = ?, content = ?, tags = ?, metadata = ?, file_size = ?, "
        "content_hash = ?, mtime_ns = ?, parser = NULL, status = 'ready', version = version + 1, "
        "last_indexed_at = datetime('now'), updated_at = datetime('now') WHERE id = ?",
        (rec.title, content, tags_json, metadata_json, len(content.encode("utf-8")),
         content_hash, mtime_ns, doc_id),
    )
    # Stale chunks are dropped; reconcile re-chunks because parser is NULL again.
    conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
    return "updated"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(report_dir: Path, records: list, stats: ImportStats, table: CodeTable) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    review = [r for r in records if r.needs_review]
    with open(report_dir / "复核队列.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "标题", "服务大类", "体裁", "置信度", "问题", "人工结论"])
        for r in review:
            w.writerow([r.entry_id, r.title, r.domain, r.genre, r.confidence_raw,
                        "; ".join(str(i) for i in r.issues), ""])

    stages = ["S0", "S1", "S2", "S3", "S4"]
    layers = ["G", "C", "O", "Z", "X"]
    grid = {s: dict.fromkeys(layers, 0) for s in stages}
    for r in records:
        grid[r.stage][r.domain[0]] += 1

    lines = [
        "# 语料导入报告",
        "",
        f"- 码表版本: {table.version}",
        f"- 导入日期: {date.today().isoformat()}",
        f"- 总条数: {stats.total} · 已导入: {stats.imported} · 进复核队列: {stats.review}",
        f"- 校验错误(兜底导入): {stats.errors} · 提示: {stats.warnings} · 跳过: {len(stats.skipped)}",
        "",
        "## 覆盖率账本(主阶段 × 大类层)",
        "",
        "| 阶段＼层 | G政府 | C合规 | O运营 | Z专项 | X兜底 |",
        "|---|---|---|---|---|---|",
    ]
    for s in stages:
        lines.append(f"| {s} | " + " | ".join(str(grid[s][layer]) for layer in layers) + " |")

    empty_cells = [f"{s}×{layer}" for s in stages for layer in ("G", "C", "O", "Z")
                   if grid[s][layer] == 0]
    if empty_cells:
        lines += ["", "## 空格(下一轮补采方向)", "", "- " + " · ".join(empty_cells)]

    issue_lines = []
    for r in records:
        for i in r.issues:
            issue_lines.append(f"- `{r.entry_id}` {i}")
    if stats.skipped:
        issue_lines += [f"- (跳过) {s}" for s in stats.skipped]
    if issue_lines:
        lines += ["", "## 校验明细", ""] + issue_lines

    (report_dir / "导入报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_records(csv_path: Path, table: CodeTable, today: date) -> tuple[list[EntryRecord], ImportStats]:
    """Parse + dedupe an annotation CSV. Shared by the local and hosted paths."""
    stats = ImportStats()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    stats.total = len(rows)

    records: list[EntryRecord] = []
    seen_paths: set[str] = set()
    for idx, row in enumerate(rows, start=2):  # header is line 1
        rec = parse_row(row, table, today=today)
        rel_path = entry_relative_path(rec)
        if rel_path in seen_paths:
            stats.skipped.append(f"CSV 第{idx}行: entry_id 重复 {rec.entry_id}")
            continue
        seen_paths.add(rel_path)
        records.append(rec)
        stats.errors += sum(1 for i in rec.issues if i.level == ERROR)
        stats.warnings += sum(1 for i in rec.issues if i.level != ERROR)
        if rec.needs_review:
            stats.review += 1
    return records, stats


def run(csv_path: Path, workspace: Path, raw_root: Path | None = None,
        version: str = DEFAULT_VERSION, dry_run: bool = False,
        today: date | None = None) -> ImportStats:
    table = load(version)
    today = today or date.today()
    imported_on = today.isoformat()

    db_path = workspace / ".llmwiki" / "index.db"
    if not dry_run and not db_path.exists():
        raise SystemExit(
            f"工作区未初始化: {db_path} 不存在。先运行: ./llmwiki init {workspace}"
        )

    records, stats = load_records(csv_path, table, today)

    conn = None
    user_id = None
    if not dry_run:
        conn = sqlite3.connect(str(db_path))
        ws_row = conn.execute("SELECT user_id FROM workspace LIMIT 1").fetchone()
        if ws_row is None:
            raise SystemExit(f"index.db 缺少 workspace 记录: {db_path}")
        user_id = ws_row[0]

    try:
        for rec in records:
            body = load_source_body(raw_root, rec.source_relpath) if raw_root else None
            content = render_entry_markdown(rec, body, imported_on)
            rel_path = entry_relative_path(rec)
            if not dry_run:
                full = workspace / rel_path
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
                upsert_document(conn, user_id, rec, rel_path, content,
                                full.stat().st_mtime_ns)
            stats.imported += 1
        if conn is not None:
            conn.commit()
    finally:
        if conn is not None:
            conn.close()

    report_dir = (workspace / ".llmwiki" / "corpus_import") if not dry_run \
        else csv_path.resolve().parent / "corpus_import_dryrun"
    write_reports(report_dir, records, stats, table)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="导入八维标注明细 — 本地工作区(--workspace)或 hosted 部署(--database-url)",
    )
    ap.add_argument("--csv", required=True, help="标注明细.csv 或 标注明细_业务视图.csv")
    ap.add_argument("--workspace", default=None, help="本地模式: LLM Wiki 工作区目录(已 init)")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""),
                    help="hosted 模式: Postgres 连接串(或环境变量 DATABASE_URL)")
    ap.add_argument("--user-email", default=None, help="hosted 模式: 语料归属账号邮箱(须已注册)")
    ap.add_argument("--kb", default="goglobal-corpus", help="hosted 模式: 知识库 slug(不存在则创建)")
    ap.add_argument("--raw", default=None, help="收录语料根目录(可选;提供则正文入条目)")
    ap.add_argument("--codetable", default=DEFAULT_VERSION, help=f"码表版本(默认 {SPEC_VERSION})")
    ap.add_argument("--dry-run", action="store_true", help="只校验并出报告,不写入")
    args = ap.parse_args()

    hosted = bool(args.database_url and args.user_email)
    if not hosted and not args.workspace:
        ap.error("需要 --workspace(本地)或 --database-url + --user-email(hosted)")

    raw_root = Path(args.raw).resolve() if args.raw else None
    if hosted:
        import asyncio

        from .hosted_import import run_hosted
        stats = asyncio.run(run_hosted(
            csv_path=Path(args.csv).resolve(),
            database_url=args.database_url,
            user_email=args.user_email,
            kb_slug=args.kb,
            raw_root=raw_root,
            version=args.codetable,
            dry_run=args.dry_run,
        ))
    else:
        stats = run(
            csv_path=Path(args.csv).resolve(),
            workspace=Path(args.workspace).resolve(),
            raw_root=raw_root,
            version=args.codetable,
            dry_run=args.dry_run,
        )
    mode = "校验" if args.dry_run else "导入"
    print(f"[✓] {mode}完成: {stats.imported}/{stats.total} 条"
          f" · 复核队列 {stats.review} · 错误 {stats.errors} · 提示 {stats.warnings}")
    if stats.skipped:
        print(f"[!] 跳过 {len(stats.skipped)} 条(详见 导入报告.md)")


if __name__ == "__main__":
    main()
