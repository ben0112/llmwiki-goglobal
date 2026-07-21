"""导入八维标注明细 into a hosted-mode deployment (Postgres).

The hosted twin of the local workspace importer: each annotated entry becomes
a `documents` row (content inline, metadata JSONB carrying the structured 八维
record, facet tags) plus `document_chunks` rows so search works immediately —
the same shape the MCP write tools produce, so facet search (phase 2), the web
corpus browser (phase 3), and lint/relations (phase 4) all work unchanged.

Corpus entries are markdown, so S3 is not required: hosted mode stores text
content in Postgres and only uses S3 for binary sources, which the annotation
pipeline does not produce.

Usage (from the repository root):
  python3 -m corpus.import_annotations \
      --csv 标注结果/标注明细_业务视图.csv \
      --database-url postgresql://...:5432/postgres \
      --user-email corpus-admin@example.com \
      --kb goglobal-corpus \
      [--raw 审核结果_deepseek/收录] [--dry-run]

The knowledge base is created (with the standard wiki scaffold pages) if the
slug does not exist yet. Re-imports are idempotent: entries are keyed by
(knowledge_base, path, filename); content changes bump version and rebuild
chunks, unchanged content only refreshes metadata/tags.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .codetable import DEFAULT_VERSION, load
from .import_annotations import (
    ImportStats,
    entry_relative_path,
    load_records,
    load_source_body,
    render_entry_markdown,
    write_reports,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_chunker():
    """Load mcp/services/chunker.py by path (the mcp tree is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "corpus_hosted_chunker", _REPO_ROOT / "mcp" / "services" / "chunker.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class HostedTarget:
    user_id: str
    kb_id: str
    kb_slug: str


async def _resolve_target(conn, user_email: str, kb_slug: str) -> HostedTarget:
    user = await conn.fetchrow("SELECT id FROM users WHERE email = $1", user_email)
    if user is None:
        raise SystemExit(
            f"用户 {user_email!r} 不存在 — 请先在应用中注册该账号(签入一次即可)。"
        )
    user_id = str(user["id"])

    kb = await conn.fetchrow(
        "SELECT id, slug FROM knowledge_bases WHERE user_id = $1 AND slug = $2",
        user["id"], kb_slug,
    )
    if kb is None:
        kb = await conn.fetchrow(
            "INSERT INTO knowledge_bases (user_id, name, slug, description) "
            "VALUES ($1, $2, $3, $4) RETURNING id, slug",
            user["id"], kb_slug, kb_slug, "出海智能体语料库 (corpus import)",
        )
        print(f"[i] 知识库 {kb_slug!r} 不存在,已创建 (id={kb['id']})")
    return HostedTarget(user_id=user_id, kb_id=str(kb["id"]), kb_slug=kb["slug"])


async def _upsert_entry(conn, target: HostedTarget, rec, rel_path: str,
                        content: str, chunker) -> str:
    dir_path = "/" + rel_path.rsplit("/", 1)[0] + "/"
    filename = rel_path.rsplit("/", 1)[-1]
    tags = rec.to_tags()
    metadata_json = json.dumps(rec.to_metadata(), ensure_ascii=False)

    existing = await conn.fetchrow(
        "SELECT id, content FROM documents "
        "WHERE knowledge_base_id = $1 AND user_id = $2 AND path = $3 AND filename = $4 "
        "AND NOT archived",
        target.kb_id, target.user_id, dir_path, filename,
    )

    if existing is None:
        row = await conn.fetchrow(
            "INSERT INTO documents (knowledge_base_id, user_id, filename, title, path, "
            "file_type, file_size, status, content, tags, date, metadata) "
            "VALUES ($1, $2, $3, $4, $5, 'md', $6, 'ready', $7, $8, $9, $10::jsonb) "
            "RETURNING id",
            target.kb_id, target.user_id, filename, rec.title, dir_path,
            len(content.encode("utf-8")), content, tags, rec.effective_date, metadata_json,
        )
        doc_id = row["id"]
        outcome = "inserted"
    else:
        doc_id = existing["id"]
        if existing["content"] == content:
            await conn.execute(
                "UPDATE documents SET metadata = $1::jsonb, tags = $2, updated_at = now() "
                "WHERE id = $3",
                metadata_json, tags, doc_id,
            )
            return "unchanged"
        await conn.execute(
            "UPDATE documents SET title = $1, content = $2, tags = $3, metadata = $4::jsonb, "
            "file_size = $5, version = version + 1, updated_at = now() WHERE id = $6",
            rec.title, content, tags, metadata_json,
            len(content.encode("utf-8")), doc_id,
        )
        await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", doc_id)
        outcome = "updated"

    chunks = chunker.chunk_text(content)
    for c in chunks:
        await conn.execute(
            "INSERT INTO document_chunks (document_id, user_id, knowledge_base_id, "
            "chunk_index, content, source_content, page, start_char, token_count, header_breadcrumb) "
            "VALUES ($1, $2, $3, $4, $5, $5, $6, $7, $8, $9)",
            doc_id, target.user_id, target.kb_id,
            c.index, c.content, c.page, c.start_char, c.token_count, c.header_breadcrumb,
        )
    return outcome


async def run_hosted(csv_path: Path, database_url: str, user_email: str, kb_slug: str,
                     raw_root: Path | None = None, version: str = DEFAULT_VERSION,
                     dry_run: bool = False, today: date | None = None) -> ImportStats:
    import asyncpg

    table = load(version)
    today = today or date.today()
    imported_on = today.isoformat()
    records, stats = load_records(csv_path, table, today)
    chunker = _load_chunker()

    conn = None
    if not dry_run:
        conn = await asyncpg.connect(database_url)
    try:
        if conn is not None:
            target = await _resolve_target(conn, user_email, kb_slug)
            # One transaction: a mid-import failure leaves the KB untouched.
            async with conn.transaction():
                for rec in records:
                    body = load_source_body(raw_root, rec.source_relpath) if raw_root else None
                    content = render_entry_markdown(rec, body, imported_on)
                    await _upsert_entry(conn, target, rec, entry_relative_path(rec), content, chunker)
                    stats.imported += 1
        else:
            stats.imported = len(records)
    finally:
        if conn is not None:
            await conn.close()

    report_dir = csv_path.resolve().parent / ("corpus_import_dryrun" if dry_run else "corpus_import_hosted")
    write_reports(report_dir, records, stats, table)
    return stats
