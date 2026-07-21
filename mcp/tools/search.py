"""Search tool — browse, search, and query references in the knowledge vault."""

import json
import logging
from datetime import date
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from vaultfs import VaultFS
from vaultfs.base import RELATION_TYPES
from vaultfs.facets import UnknownFacetError

from .helpers import MAX_LIST, MAX_SEARCH, deep_link, glob_match, resolve_path

logger = logging.getLogger(__name__)

_CONTEXT_CHARS = 120


def _extract_snippet(content: str, query: str) -> str:
    """Extract a context snippet around a query match."""
    if not content:
        return "(empty)"
    idx = content.lower().find(query.lower())
    if idx < 0:
        return content[:_CONTEXT_CHARS * 2].strip()
    start = max(0, idx - _CONTEXT_CHARS)
    end = min(len(content), idx + len(query) + _CONTEXT_CHARS)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


class SearchHandler:
    """Executes list, search, and reference queries on the knowledge vault."""

    def __init__(self, fs: VaultFS, kb: dict):
        self.fs = fs
        self.kb = kb
        self.kb_id = str(kb["id"])
        self.slug = kb["slug"]

    async def list_documents(self, target: str, tags: list[str] | None,
                             facets: dict | None = None) -> str:
        """List documents matching a glob pattern and optional tag/facet filters."""
        docs = await self.fs.list_documents(self.kb_id, facets=facets)

        if target not in ("*", "**", "**/*"):
            glob_pat = "/" + target.lstrip("/") if not target.startswith("/") else target
            docs = [d for d in docs if glob_match(d["path"] + d["filename"], glob_pat)]

        if tags:
            tag_set = {t.lower() for t in tags}
            docs = [d for d in docs if tag_set.issubset({t.lower() for t in (d.get("tags") or [])})]

        if not docs:
            return f"No matches for `{target}` in {self.slug}."

        sources = [d for d in docs if not d["path"].startswith("/wiki/")]
        wiki_pages = [d for d in docs if d["path"].startswith("/wiki/")]

        scope_parts = []
        if target not in ("*", "**", "**/*"):
            scope_parts.append(f"`{target}`")
        if tags:
            scope_parts.append(f"tags: {', '.join(tags)}")
        scope = f" ({' — '.join(scope_parts)})" if scope_parts else ""
        lines = [f"**{self.kb['name']}**{scope}:\n"]

        if sources:
            lines.append(f"**Sources ({len(sources)}):**")
            for doc in sources[:MAX_LIST]:
                lines.append(self._format_source_line(doc))
            if len(sources) > MAX_LIST:
                lines.append(f"  ... {len(sources) - MAX_LIST} more")

        if wiki_pages:
            if sources:
                lines.append("")
            lines.append(f"**Wiki ({len(wiki_pages)} pages):**")
            for doc in wiki_pages[:MAX_LIST]:
                lines.append(self._format_wiki_line(doc))

        return "\n".join(lines)

    async def search_chunks(
        self, query: str, path: str, tags: list[str] | None, limit: int,
        annotated_only: bool = False, scope: str = "all",
        facets: dict | None = None,
    ) -> str:
        """Full-text search across document chunks.

        `annotated_only=True` restricts to chunks the user has highlighted.
        `scope='annotations'` matches only within the user's notes/quotes;
        `scope='source'` matches only within original document content;
        `scope='all'` (default) matches either.
        `facets` filters by the corpus 八维 classification metadata.
        """
        path_filter = self._path_filter_key(path)

        matches = await self.fs.search_chunks(
            self.kb_id, query, limit, path_filter,
            annotated_only=annotated_only, scope=scope, facets=facets,
        )
        matches = self._apply_path_glob(matches, path)

        if tags:
            tag_set = {t.lower() for t in tags}
            matches = [m for m in matches if tag_set.issubset({t.lower() for t in (m.get("tags") or [])})]

        if not matches:
            scope_msg = ""
            if scope != "all":
                scope_msg = f" (scope: {scope})"
            if annotated_only:
                scope_msg = f" (annotated only{scope_msg})"
            return f"No matches for `{query}` in {self.slug}{scope_msg}."

        lines = [f"**{len(matches)} result(s)** for `{query}`:\n"]
        for m in matches:
            lines.append(self._format_search_result(m, query))

        return "\n".join(lines)

    async def query_references(self, path: str, query: str) -> str:
        """Query the citation/link graph."""
        if query == "uncited":
            return await self._find_uncited()
        if query == "stale":
            return await self._find_stale()
        if query == "due":
            return await self._find_due()
        return await self._document_references(path)

    async def _find_due(self) -> str:
        """Corpus review worklist: overdue review_due + 待复核 entries."""
        docs = await self.fs.list_documents(self.kb_id)
        today = date.today().isoformat()
        due: list[tuple[str, dict, dict]] = []
        for doc in docs:
            meta = doc.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict) or "spec_version" not in meta or "stage" not in meta:
                continue
            review_due = str(meta.get("review_due") or "")[:10]
            if meta.get("lifecycle_state") == "待复核":
                due.append(("待复核", doc, meta))
            elif review_due and review_due < today:
                due.append((f"复审到期 {review_due}", doc, meta))
        if not due:
            return "复审工作清单为空 — 没有到期或待复核的语料条目。"
        due.sort(key=lambda item: str(item[2].get("review_due") or ""))
        lines = [f"**复审工作清单 ({len(due)} 条)** — 按 review_due 升序:\n"]
        for reason, doc, meta in due:
            lines.append(
                f"  {doc['path']}{doc['filename']} — {reason} "
                f"({meta.get('entry_id', '?')} · {meta.get('timeliness', '?')})"
            )
        lines.append("\n复核后请更新条目的 lifecycle_state 与 review_due (重导入或编辑 frontmatter)。")
        return "\n".join(lines)

    async def _find_uncited(self) -> str:
        """Find source documents not cited by any wiki page."""
        rows = await self.fs.find_uncited_sources(self.kb_id)
        if not rows:
            return "All sources are cited in at least one wiki page."
        lines = [f"**{len(rows)} uncited source(s)** — not referenced by any wiki page:\n"]
        for r in rows:
            lines.append(f"  {r['path']}{r['filename']} ({r['file_type']})")
        return "\n".join(lines)

    async def _find_stale(self) -> str:
        """Find wiki pages flagged as potentially stale."""
        rows = await self.fs.find_stale_pages(self.kb_id)
        if not rows:
            return "No stale pages found."
        lines = [f"**{len(rows)} potentially stale page(s)** — a page they reference was updated:\n"]
        for r in rows:
            stale = r["stale_since"]
            if hasattr(stale, "strftime"):
                stale = stale.strftime("%Y-%m-%d %H:%M")
            title = r["title"] or r["filename"]
            lines.append(f"  {r['path']}{r['filename']} ({title}) — stale since {stale or '?'}")
        return "\n".join(lines)

    async def _document_references(self, path: str) -> str:
        """Show forward references and backlinks for a specific document."""
        if not path or path in ("*", "**"):
            return "references mode requires a `path` to a specific document, or `query=\"uncited\"` / `query=\"stale\"`."

        dir_path, filename = resolve_path(path)
        doc = await self.fs.get_document(self.kb_id, filename, dir_path)
        if not doc:
            return f"Document '{path}' not found."

        doc_id = str(doc["id"])
        forward = await self.fs.get_forward_references(doc_id)
        backlinks = await self.fs.get_backlinks(doc_id)

        title = doc.get("title") or doc["filename"]
        lines = [f"**References for {title}** (`{dir_path}{filename}`):\n"]

        if forward:
            cites = [r for r in forward if r["reference_type"] == "cites"]
            links = [r for r in forward if r["reference_type"] == "links_to"]
            relations = [r for r in forward if r["reference_type"] in RELATION_TYPES]
            if cites:
                lines.append(f"**Cites ({len(cites)} sources):**")
                for r in cites:
                    page_str = f", p.{r['page']}" if r.get("page") else ""
                    lines.append(f"  {r['path']}{r['filename']}{page_str}")
            if links:
                lines.append(f"\n**Links to ({len(links)} pages):**")
                for r in links:
                    lines.append(f"  {r['path']}{r['filename']} ({r.get('title') or r['filename']})")
            if relations:
                lines.append(f"\n**Relations ({len(relations)}, curated):**")
                for r in relations:
                    label = RELATION_TYPES[r["reference_type"]]
                    lines.append(
                        f"  —{r['reference_type']} ({label})→ {r['path']}{r['filename']} "
                        f"({r.get('title') or r['filename']})"
                    )
        else:
            lines.append("No outgoing references.")

        lines.append("")
        if backlinks:
            lines.append(f"**Referenced by ({len(backlinks)} pages):**")
            for r in backlinks:
                ref_type = r["reference_type"]
                if ref_type in RELATION_TYPES:
                    ref = f"{ref_type} ({RELATION_TYPES[ref_type]})"
                else:
                    ref = "cites" if ref_type == "cites" else "links to"
                lines.append(f"  {r['path']}{r['filename']} ({r.get('title') or r['filename']}) — {ref}")
        else:
            lines.append("No incoming references (backlinks).")

        return "\n".join(lines)

    def _path_filter_key(self, path: str) -> str | None:
        """Map a path pattern to a coarse wiki/sources search filter key."""
        if path in ("*", "**", "**/*"):
            return None
        if path.startswith("/wiki"):
            return "wiki"
        if path in ("/", "/*"):
            return "sources"
        return None

    def _apply_path_glob(self, matches: list[dict], path: str) -> list[dict]:
        """Narrow results by a file-level glob (e.g. `*.pdf`).

        The coarse wiki/sources filter is pushed into SQL; finer globs the SQL
        can't express are applied here, mirroring list/read/delete. Bare
        directory scopes (no wildcard) are left to the coarse filter.
        """
        if path in ("*", "**", "**/*") or not ("*" in path or "?" in path):
            return matches
        glob_pat = path if path.startswith("/") else "/" + path.lstrip("/")
        return [m for m in matches if glob_match(m["path"] + m["filename"], glob_pat)]

    def _format_source_line(self, doc: dict) -> str:
        """Format a single source document for list output."""
        tag_str = f" [{', '.join(doc['tags'])}]" if doc.get("tags") else ""
        date_part = ""
        if doc.get("updated_at"):
            date_val = doc["updated_at"]
            date_part = f", {date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else date_val}"
        pages_part = f", {doc['page_count']}p" if doc.get("page_count") else ""
        return f"  {doc['path']}{doc['filename']} ({doc.get('file_type', '')}{pages_part}{date_part}){tag_str}"

    def _format_wiki_line(self, doc: dict) -> str:
        """Format a single wiki page for list output."""
        date_part = ""
        if doc.get("updated_at"):
            date_val = doc["updated_at"]
            date_part = f", {date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else date_val}"
        return f"  {doc['path']}{doc['filename']}{date_part}"

    def _format_search_result(self, match: dict, query: str) -> str:
        """Format a single search result with snippet.

        Snippet is taken from `content` — which already contains the
        materialized source + annotations footnote block, so the LLM sees
        both sides in their proper context without us re-stitching them.
        A small marker after the header signals whether the match came
        from the user's annotations.
        """
        filepath = f"{match['path']}{match['filename']}"
        page_str = f" (p.{match['page']})" if match.get("page") else ""
        breadcrumb = f"\n  {match['header_breadcrumb']}" if match.get("header_breadcrumb") else ""
        link = deep_link(self.slug, match["path"], match["filename"])
        score = match.get("score", 0)
        score_str = f" [{score:.1f}]" if score else ""

        # Mark which side of the chunk produced the match so the LLM can
        # attribute correctly — "note" = user's voice, "source" = doc body.
        # `[annotated]` is a weaker signal: the chunk has user notes but the
        # match itself came only from the source.
        marker = ""
        if match.get("annotation_hit") and not match.get("source_hit"):
            marker = " [matched: note]"
        elif match.get("annotation_hit"):
            marker = " [matched: source+note]"
        elif match.get("has_highlight"):
            marker = " [annotated]"

        snippet = _extract_snippet(match.get("content", ""), query)
        return f"**{filepath}**{page_str}{score_str}{marker} — [view]({link}){breadcrumb}\n```\n{snippet}\n```\n"


async def _list_all_kbs(fs: VaultFS) -> str:
    """List all knowledge bases for the user."""
    kbs = await fs.list_knowledge_bases()
    if not kbs:
        return "No knowledge bases found."

    lines = ["**Knowledge Bases:**\n"]
    for kb in kbs:
        lines.append(f"  {kb['slug']}/ — {kb['name']}")
    return "\n".join(lines)


def register(mcp: FastMCP, get_user_id, fs_factory) -> None:

    @mcp.tool(
        name="search",
        description=(
            "Browse or search the knowledge vault.\n\n"
            "Sources (raw documents) live at `/`. Wiki pages (LLM-compiled) live at `/wiki/`.\n\n"
            "Modes:\n"
            "- list: browse files and folders\n"
            "- search: keyword search across document content (searches chunks for precise results with page numbers)\n"
            "- references: query the citation/link graph for a document\n\n"
            "References mode examples:\n"
            "- `search(mode=\"references\", path=\"/wiki/concepts/scaling.md\")` — what it cites + what links to it\n"
            "- `search(mode=\"references\", path=\"paper.pdf\")` — which wiki pages cite this source\n"
            "- `search(mode=\"references\", query=\"uncited\")` — sources with no wiki citations\n"
            "- `search(mode=\"references\", query=\"stale\")` — pages flagged as potentially stale\n"
            "- `search(mode=\"references\", query=\"due\")` — corpus review worklist "
            "(overdue review_due + 待复核 entries)\n"
            "Reference output includes curated relation edges (see the `relate` tool): "
            "is_a 上下位, next 前后置, routes_to 路径衔接, governed_by 归口映射, serves 阶段服务包.\n\n"
            "Use `path` to scope: `*` for root, `/wiki/**` for wiki only, `*.pdf` for PDFs, etc.\n"
            "Use `tags` to filter by document tags.\n\n"
            "Corpus facet filtering (list + search modes): when the knowledge base holds "
            "classified corpus entries (八维标注), pass `facets` to filter by classification "
            "metadata, e.g. `facets={\"domain\": \"C2\", \"country\": \"IDN\"}`. Keys: "
            "stage (S0-S4), domain (G/C/O/Z codes or X9), genre (体裁 name), rule (R0-R6), "
            "evidence (E0-E4), origin (目的地国/国际/国内/混合), dept (归口部门), "
            "country (ISO alpha-3 or Chinese name), region (e.g. 东盟), industry, mode, "
            "timeliness (M1-M3), state (生命周期), business (业务码 B4.14 or class B4), entry_id. "
            "Multi-value dimensions match primary or secondary labels."
        ),
    )
    async def search(
        ctx: Context,
        knowledge_base: str,
        mode: Literal["list", "search", "references"] = "list",
        query: str = "",
        path: str = "*",
        tags: list[str] | None = None,
        limit: int = 10,
        annotated_only: bool = False,
        scope: Literal["all", "annotations", "source"] = "all",
        facets: dict[str, str] | None = None,
    ) -> str:
        user_id = get_user_id(ctx)
        fs = fs_factory(user_id)

        if not knowledge_base:
            return await _list_all_kbs(fs)

        kb = await fs.resolve_kb(knowledge_base)
        if not kb:
            return f"Knowledge base '{knowledge_base}' not found."

        handler = SearchHandler(fs, kb)

        try:
            if mode == "list":
                return await handler.list_documents(path, tags, facets=facets)
            if mode == "search":
                if not query:
                    return "search mode requires a query."
                return await handler.search_chunks(
                    query, path, tags, min(limit, MAX_SEARCH),
                    annotated_only=annotated_only, scope=scope, facets=facets,
                )
            if mode == "references":
                return await handler.query_references(path, query)
        except UnknownFacetError as e:
            return f"{e} (facets filter classified corpus entries by 八维 metadata)"

        return f"Unknown mode: {mode}"
