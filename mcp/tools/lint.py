"""Lint tool — deterministic hygiene checks for wiki pages and sources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from vaultfs import VaultFS

from .helpers import glob_match
from .references import _parse_citation_filename, _parse_wiki_links
from .write import (
    _extract_frontmatter_tags,
    _extract_metadata,
    _is_footnote_suffix_line,
    _parse_frontmatter,
)

_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:\s*(.+)$", re.MULTILINE)
_FOOTNOTE_USE_RE = re.compile(r"\[\^([^\]]+)\](?!:)")
_SOURCE_EXT_RE = re.compile(r"\.(pdf|docx?|pptx?|xlsx?|csv|html?|md|txt)$", re.IGNORECASE)
_ROOT_PAGES = frozenset({"/wiki/overview.md", "/wiki/index.md", "/wiki/readme.md", "/wiki/log.md"})
# Append-only chronological ledgers — frontmatter/footnote conventions don't apply.
_LEDGER_PAGES = frozenset({"/wiki/log.md"})
_MATCH_ALL_PATHS = frozenset({"*", "**", "**/*"})
_MAX_ISSUES_PER_GROUP = 40

# 八维 corpus entry validation (码表 v2026.06 closed lists; the authoritative
# versioned tables live in corpus/codetables/ — these frozen sets exist so the
# MCP server can lint without importing outside its package root).
_CORPUS_STAGES = frozenset({"S0", "S1", "S2", "S3", "S4"})
_CORPUS_DOMAINS = frozenset(
    {f"G{i}" for i in range(1, 5)} | {f"C{i}" for i in range(1, 6)}
    | {f"O{i}" for i in range(1, 7)} | {f"Z{i}" for i in range(1, 5)} | {"X9"}
)
_CORPUS_GENRES = frozenset({
    "政策法规", "办事指南流程", "国别研究", "案例经验", "风险预警",
    "实操指引方法论", "知识问答QA", "数据指标", "资源名录", "规则提炼", "其他",
})
_CORPUS_RULES = frozenset({f"R{i}" for i in range(7)})
_CORPUS_EVIDENCE = frozenset({f"E{i}" for i in range(5)})
_CORPUS_ORIGINS = frozenset({"目的地国", "国际", "国内", "混合"})
_CORPUS_TIMELINESS = frozenset({"M1", "M2", "M3"})
_CORPUS_STATES = frozenset({"草拟", "待复核", "已入库", "已发布", "待更新", "已过期", "已归档"})
# 公理一/公理三: every dimension holds a value, provenance is mandatory.
_CORPUS_REQUIRED = ("stage", "domain", "genre", "evidence", "origin", "gov_dept",
                    "timeliness", "lifecycle_state", "review_due")
_CORPUS_LAYERS = ("G", "C", "O", "Z", "X")


def wiki_coverage_summary(entry_metas: list[dict], wiki_rollups: list[dict]) -> dict:
    """有语料的货架格(阶段×大类层)与业务场景,被维基页 rollup 覆盖的比例。"""
    cell_counts: dict[tuple[str, str], int] = {}
    scene_counts: dict[str, int] = {}
    for meta in entry_metas:
        stage = meta.get("stage")
        domain = str(meta.get("domain") or "")
        if stage and domain[:1] in ("G", "C", "O", "Z"):
            key = (str(stage), domain[:1])
            cell_counts[key] = cell_counts.get(key, 0) + 1
        biz = meta.get("business")
        code = biz.get("code") if isinstance(biz, dict) else None
        if code:
            scene_counts[str(code)] = scene_counts.get(str(code), 0) + 1

    page_cells: set[tuple[str, str]] = set()
    page_scenes: set[str] = set()
    for rollup in wiki_rollups:
        layers = {str(d)[:1] for d in rollup.get("domain") or []}
        for stage in rollup.get("stage") or []:
            for layer in layers:
                page_cells.add((str(stage), layer))
        for code in rollup.get("business") or []:
            page_scenes.add(str(code))

    cells = sorted(cell_counts)
    scenes = sorted(scene_counts)
    return {
        "cells_total": len(cells),
        "cells_covered": sum(1 for c in cells if c in page_cells),
        "cell_gaps": [f"{s}×{layer}({cell_counts[(s, layer)]}条)"
                      for s, layer in cells if (s, layer) not in page_cells],
        "scenes_total": len(scenes),
        "scenes_covered": sum(1 for c in scenes if c in page_scenes),
        "scene_gaps": [f"{c}({scene_counts[c]}条)" for c in scenes if c not in page_scenes],
    }

Scope = Literal["all", "wiki", "sources"]


@dataclass(frozen=True)
class LintIssue:
    severity: Literal["error", "warn"]
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class LintContext:
    """Lookups and counts computed once per run and shared across checks."""

    source_lookup: dict[str, dict]
    wiki_lookup: dict[str, dict]
    wiki_page_count: int


class LintHandler:
    """Runs deterministic checks across a knowledge base."""

    def __init__(self, fs: VaultFS, kb: dict):
        self.fs = fs
        self.kb = kb
        self.kb_id = str(kb["id"])
        self.slug = kb["slug"]

    async def run(
        self,
        path: str = "*",
        scope: Scope = "all",
        include_graph: bool = True,
    ) -> str:
        all_docs = await self.fs.list_documents_with_content(self.kb_id)
        docs = self._filter_docs(all_docs, path, scope)
        if not docs:
            return f"No documents matched `{path}` in {self.slug}."

        ctx = LintContext(
            source_lookup=self._source_lookup(all_docs),
            wiki_lookup=self._wiki_lookup(all_docs),
            wiki_page_count=sum(1 for doc in all_docs if self._is_wiki_page(doc)),
        )

        issues: list[LintIssue] = []
        corpus_entries: list[tuple[dict, dict]] = []
        overdue_entry_ids: list[str] = []
        for doc in docs:
            if self._is_wiki_page(doc):
                issues.extend(await self._lint_wiki_page(doc, ctx, include_graph))
            else:
                meta = self._corpus_meta(doc)
                if meta is not None:
                    corpus_entries.append((doc, meta))
                    entry_issues = self._lint_corpus_entry(doc, meta)
                    issues.extend(entry_issues)
                    if any(i.code == "corpus-review-overdue" for i in entry_issues):
                        overdue_entry_ids.append(str(doc["id"]))

        # 复审到期联动:到期条目的引用维基页标记待复查(search query="stale" 可查)
        if overdue_entry_ids:
            flagged = await self.fs.mark_cites_stale(overdue_entry_ids)
            if flagged:
                issues.append(LintIssue(
                    "warn", "stale-pages-flagged", f"{len(overdue_entry_ids)} 个到期条目",
                    f"已把 {flagged} 个引用这些条目的维基页标记为待复查 — search(mode=\"references\", query=\"stale\") 查看",
                ))

        if include_graph:
            issues.extend(await self._lint_kb_wide(path, scope))

        report = self._format_report(issues, docs)
        if corpus_entries:
            uncited_paths = {
                f"{r['path']}{r['filename']}"
                for r in await self.fs.find_uncited_sources(self.kb_id)
            }
            report += "\n\n" + self._format_coverage_ledger(corpus_entries, uncited_paths)
            report += "\n\n" + self._format_wiki_coverage(corpus_entries, all_docs)
        return report

    # ----- selection -------------------------------------------------------

    def _filter_docs(self, docs: list[dict], path: str, scope: Scope) -> list[dict]:
        if scope == "wiki":
            docs = [d for d in docs if self._is_wiki_page(d)]
        elif scope == "sources":
            docs = [d for d in docs if not self._is_wiki_page(d)]
        return [d for d in docs if self._path_matches(self._doc_path(d), path)]

    def _path_matches(self, doc_path: str, path: str) -> bool:
        if path in _MATCH_ALL_PATHS:
            return True
        glob_pat = path if path.startswith("/") else "/" + path
        return glob_match(doc_path, glob_pat)

    # ----- per-page checks -------------------------------------------------

    async def _lint_wiki_page(self, doc: dict, ctx: LintContext, include_graph: bool) -> list[LintIssue]:
        path = self._doc_path(doc)
        content = doc.get("content") or ""
        meta = _parse_frontmatter(content)

        issues: list[LintIssue] = []
        if not self._is_ledger_page(doc):
            issues.extend(self._lint_frontmatter(doc, meta))
            issues.extend(self._lint_footnotes(path, content))
        issues.extend(self._lint_citations(doc, content, ctx.source_lookup))
        issues.extend(self._lint_wiki_links(doc, content, ctx.wiki_lookup))

        if include_graph:
            issues.extend(await self._lint_reference_graph(doc, content, ctx.source_lookup))
            issues.extend(await self._lint_orphan(doc, ctx.wiki_page_count))

        return issues

    def _lint_frontmatter(self, doc: dict, meta: dict) -> list[LintIssue]:
        path = self._doc_path(doc)
        if not meta:
            return [LintIssue("error", "missing-frontmatter", path, "wiki page has no YAML frontmatter")]

        issues: list[LintIssue] = []
        title = meta.get("title")
        description = meta.get("description")
        fm_date_raw, _ = _extract_metadata(meta)
        fm_date = self._normalize_date(fm_date_raw)
        fm_tags = _extract_frontmatter_tags(meta)

        if not isinstance(title, str) or not title.strip():
            issues.append(LintIssue("error", "missing-title", path, "frontmatter is missing `title`"))
        if not isinstance(description, str) or not description.strip():
            issues.append(LintIssue("warn", "missing-description", path, "frontmatter is missing `description`"))
        if not fm_date:
            issues.append(LintIssue("warn", "missing-date", path, "frontmatter is missing `date`"))
        if fm_tags is None:
            issues.append(LintIssue("error", "missing-tags", path, "frontmatter is missing `tags`"))
        elif len(fm_tags) < 2:
            issues.append(LintIssue("warn", "too-few-tags", path, "frontmatter should include at least two tags"))

        indexed_tags = [str(t) for t in (doc.get("tags") or [])]
        if fm_tags is not None and self._normalize_tags(fm_tags) != self._normalize_tags(indexed_tags):
            issues.append(LintIssue(
                "warn",
                "tag-index-mismatch",
                path,
                f"frontmatter tags {fm_tags} do not match indexed tags {indexed_tags}",
            ))

        indexed_date = self._normalize_date(doc.get("date"))
        if fm_date and indexed_date and fm_date != indexed_date:
            issues.append(LintIssue(
                "warn",
                "date-index-mismatch",
                path,
                f"frontmatter date `{fm_date}` does not match indexed date `{indexed_date}`",
            ))
        elif fm_date and not indexed_date:
            issues.append(LintIssue("warn", "date-not-indexed", path, "frontmatter date is not indexed"))

        return issues

    def _lint_footnotes(self, path: str, content: str) -> list[LintIssue]:
        issues: list[LintIssue] = []
        def_ids = [footnote_id for footnote_id, _ in _FOOTNOTE_DEF_RE.findall(content)]
        used_ids = _FOOTNOTE_USE_RE.findall(content)

        for footnote_id in sorted({fid for fid in def_ids if def_ids.count(fid) > 1}, key=self._footnote_sort_key):
            issues.append(LintIssue("error", "duplicate-footnote", path, f"footnote `^{footnote_id}` is defined more than once"))

        for footnote_id in sorted(set(used_ids) - set(def_ids), key=self._footnote_sort_key):
            issues.append(LintIssue("error", "footnote-without-definition", path, f"footnote `^{footnote_id}` is used but not defined"))

        for footnote_id in sorted(set(def_ids) - set(used_ids), key=self._footnote_sort_key):
            issues.append(LintIssue("warn", "unused-footnote-definition", path, f"footnote `^{footnote_id}` is defined but not used"))

        if self._has_mid_document_footnotes(content):
            issues.append(LintIssue(
                "warn",
                "footnotes-not-at-tail",
                path,
                "footnote definitions should be grouped at the end of the page",
            ))

        return issues

    def _lint_citations(self, doc: dict, content: str, source_lookup: dict[str, dict]) -> list[LintIssue]:
        path = self._doc_path(doc)
        issues: list[LintIssue] = []
        for footnote_id, raw in _FOOTNOTE_DEF_RE.findall(content):
            filename, _page = _parse_citation_filename(raw)
            if not self._resolve_source(filename, source_lookup):
                issues.append(LintIssue(
                    "error",
                    "unresolved-citation",
                    path,
                    f"footnote `^{footnote_id}` cites `{filename}`, but no matching source exists",
                ))
        return issues

    def _lint_wiki_links(self, doc: dict, content: str, wiki_lookup: dict[str, dict]) -> list[LintIssue]:
        path = self._doc_path(doc)
        current_dir = doc["path"].replace("/wiki/", "", 1) if doc["path"].startswith("/wiki/") else ""
        issues: list[LintIssue] = []
        for link_path in _parse_wiki_links(content, current_dir):
            if not self._resolve_wiki_link(link_path, wiki_lookup):
                issues.append(LintIssue(
                    "error",
                    "dangling-link",
                    path,
                    f"wiki link `{link_path}` does not resolve to a page",
                ))
        return issues

    async def _lint_reference_graph(self, doc: dict, content: str, source_lookup: dict[str, dict]) -> list[LintIssue]:
        path = self._doc_path(doc)
        expected_source_ids: set[str] = set()
        for _footnote_id, raw in _FOOTNOTE_DEF_RE.findall(content):
            filename, _page = _parse_citation_filename(raw)
            target = self._resolve_source(filename, source_lookup)
            if target and str(target["id"]) != str(doc["id"]):
                expected_source_ids.add(str(target["id"]))

        if not expected_source_ids:
            return []

        forward = await self.fs.get_forward_references(str(doc["id"]))
        actual_source_ids = {
            str(ref["id"])
            for ref in forward
            if ref.get("reference_type") == "cites" and ref.get("id")
        }
        missing = expected_source_ids - actual_source_ids
        if not missing:
            return []

        missing_names = sorted(self._doc_path(d) for d in source_lookup.values() if str(d["id"]) in missing)
        return [LintIssue(
            "error",
            "citation-graph-mismatch",
            path,
            f"citation footnotes were not materialized into graph edges: {', '.join(missing_names)}",
        )]

    async def _lint_orphan(self, doc: dict, wiki_page_count: int) -> list[LintIssue]:
        if self._is_root_page(doc) or wiki_page_count <= 1:
            return []
        if await self.fs.get_backlinks(str(doc["id"])):
            return []
        return [LintIssue("warn", "orphan-page", self._doc_path(doc), "wiki page has no incoming links or citations")]

    # ----- knowledge-base-wide checks --------------------------------------

    async def _lint_kb_wide(self, path: str, scope: Scope) -> list[LintIssue]:
        """Graph-wide checks. Each issue is kept only when its path matches the
        run's `path` filter, so narrowing the run narrows these too (rather than
        silently dropping the check)."""
        issues: list[LintIssue] = []
        if scope in ("all", "sources"):
            issues.extend(i for i in await self._lint_uncited_sources() if self._path_matches(i.path, path))
        if scope in ("all", "wiki"):
            issues.extend(i for i in await self._lint_stale_pages() if self._path_matches(i.path, path))
        return issues

    async def _lint_uncited_sources(self) -> list[LintIssue]:
        rows = await self.fs.find_uncited_sources(self.kb_id)
        return [
            LintIssue("warn", "uncited-source", f"{row['path']}{row['filename']}", "source is not cited by any wiki page")
            for row in rows
        ]

    async def _lint_stale_pages(self) -> list[LintIssue]:
        rows = await self.fs.find_stale_pages(self.kb_id)
        return [
            LintIssue("warn", "stale-page", f"{row['path']}{row['filename']}", f"page is stale since {row.get('stale_since') or '?'}")
            for row in rows
        ]

    # ----- corpus (八维) entry checks ---------------------------------------

    def _corpus_meta(self, doc: dict) -> dict | None:
        """Structured 八维 metadata for a classified corpus entry, else None."""
        meta = doc.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(meta, dict):
            return None
        return meta if "spec_version" in meta and "stage" in meta else None

    def _lint_corpus_entry(self, doc: dict, meta: dict) -> list[LintIssue]:
        path = self._doc_path(doc)
        issues: list[LintIssue] = []

        missing = [f for f in _CORPUS_REQUIRED if not meta.get(f)]
        if missing:
            issues.append(LintIssue(
                "error", "corpus-missing-dimension", path,
                f"八维元数据缺失必填维度: {', '.join(missing)} (公理一/公理三)",
            ))

        def check_code(value, valid: frozenset, dim: str) -> None:
            if value and value not in valid:
                issues.append(LintIssue(
                    "error", "corpus-unknown-code", path,
                    f"{dim} 取值 `{value}` 不在码表内",
                ))

        check_code(meta.get("stage"), _CORPUS_STAGES, "阶段")
        for ext in meta.get("stage_ext") or []:
            check_code(ext, _CORPUS_STAGES, "阶段(副)")
        check_code(meta.get("domain"), _CORPUS_DOMAINS, "服务大类")
        for ext in meta.get("domain_ext") or []:
            check_code(ext, _CORPUS_DOMAINS, "服务大类(副)")
        check_code(meta.get("genre"), _CORPUS_GENRES, "体裁")
        for rule in meta.get("rule_type") or []:
            check_code(rule, _CORPUS_RULES, "隐性规则")
        check_code(meta.get("evidence"), _CORPUS_EVIDENCE, "证据强度")
        check_code(meta.get("origin"), _CORPUS_ORIGINS, "来源域")
        check_code(meta.get("timeliness"), _CORPUS_TIMELINESS, "时效")
        check_code(meta.get("lifecycle_state"), _CORPUS_STATES, "生命周期状态")

        review_due = str(meta.get("review_due") or "")
        if review_due and review_due[:10] < date.today().isoformat():
            issues.append(LintIssue(
                "warn", "corpus-review-overdue", path,
                f"复审已到期 ({review_due[:10]}, 时效 {meta.get('timeliness')}) — 需复核后更新 review_due",
            ))

        if meta.get("lifecycle_state") == "待复核":
            issues.append(LintIssue(
                "warn", "corpus-pending-review", path,
                "条目在人工复核队列 (低置信/X9/校验错误)",
            ))

        return issues

    def _format_wiki_coverage(self, entries: list[tuple[dict, dict]], all_docs: list[dict]) -> str:
        """Wiki 覆盖率:有语料的货架格/业务场景是否已有维基页覆盖(P2A)。

        缺口清单即建维基页的任务单——先按 guide 的分面检索规则拉语料再建页。"""
        rollups: list[dict] = []
        for doc in all_docs:
            if not self._is_wiki_page(doc):
                continue
            meta = doc.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    continue
            rollup = meta.get("facet_rollup") if isinstance(meta, dict) else None
            if isinstance(rollup, dict):
                rollups.append(rollup)
        summary = wiki_coverage_summary([m for _d, m in entries], rollups)

        lines = [
            f"**Wiki 覆盖率**:货架 {summary['cells_covered']}/{summary['cells_total']} 格有维基页 · "
            f"业务场景 {summary['scenes_covered']}/{summary['scenes_total']} 个有维基页",
        ]
        if summary["cell_gaps"]:
            lines.append("货架缺口(建页方向): " + " · ".join(summary["cell_gaps"][:10]))
        if summary["scene_gaps"]:
            lines.append("场景缺口: " + " · ".join(summary["scene_gaps"][:10]))
        if not summary["cell_gaps"] and not summary["scene_gaps"] and summary["cells_total"]:
            lines.append("有语料的货架与场景均已有对应维基页 ✓")
        return "\n".join(lines)

    def _format_coverage_ledger(self, entries: list[tuple[dict, dict]], uncited_paths: set[str]) -> str:
        """货架覆盖率账本 + 质量 KPI (spec §5.4) over classified corpus entries."""
        stages = sorted(_CORPUS_STAGES)
        grid = {s: dict.fromkeys(_CORPUS_LAYERS, 0) for s in stages}
        pending = 0
        complete = 0
        on_time = 0
        cited = 0
        today = date.today().isoformat()
        for doc, meta in entries:
            stage = meta.get("stage")
            domain = str(meta.get("domain") or "X9")
            if stage in grid and domain[:1] in _CORPUS_LAYERS:
                grid[stage][domain[:1]] += 1
            if meta.get("lifecycle_state") == "待复核":
                pending += 1
            if all(meta.get(f) for f in _CORPUS_REQUIRED):
                complete += 1
            review_due = str(meta.get("review_due") or "")[:10]
            if review_due and review_due >= today:
                on_time += 1
            if self._doc_path(doc) not in uncited_paths:
                cited += 1

        n = len(entries)
        filled = sum(1 for s in stages for layer in ("G", "C", "O", "Z") if grid[s][layer] > 0)
        total_cells = len(stages) * 4

        def pct(x: int) -> str:
            return f"{x / n * 100:.0f}%" if n else "—"

        lines = [
            f"**覆盖率账本** ({n} 条语料; 待复核 {pending}):",
            "",
            f"KPI: 分面完备率 {pct(complete)} · 货架覆盖率 {filled}/{total_cells}格 "
            f"({filled / total_cells * 100:.0f}%) · 时效达标率 {pct(on_time)} · "
            f"引用溯源率 {pct(cited)}",
            "",
            "| 阶段＼层 | G政府 | C合规 | O运营 | Z专项 | X兜底 |",
            "|---|---|---|---|---|---|",
        ]
        for s in stages:
            lines.append(f"| {s} | " + " | ".join(str(grid[s][layer]) for layer in _CORPUS_LAYERS) + " |")
        empty = [f"{s}×{layer}" for s in stages for layer in ("G", "C", "O", "Z") if grid[s][layer] == 0]
        if empty:
            lines += ["", f"空格 (补采方向): {' · '.join(empty)}"]
        return "\n".join(lines)

    # ----- report ----------------------------------------------------------

    def _format_report(self, issues: list[LintIssue], docs: list[dict]) -> str:
        if not issues:
            return f"**Lint passed** for {self.kb['name']} ({len(docs)} document(s) checked)."

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warn"]
        lines = [
            f"**Lint found {len(issues)} issue(s)** in {self.kb['name']} "
            f"({len(errors)} error, {len(warnings)} warning; {len(docs)} document(s) checked).",
        ]

        if errors:
            lines.append("\n**Errors**")
            lines.extend(self._format_issue_lines(errors))
        if warnings:
            lines.append("\n**Warnings**")
            lines.extend(self._format_issue_lines(warnings))

        return "\n".join(lines)

    def _format_issue_lines(self, issues: list[LintIssue]) -> list[str]:
        lines = [
            f"- [{issue.code}] `{issue.path}` — {issue.message}"
            for issue in issues[:_MAX_ISSUES_PER_GROUP]
        ]
        if len(issues) > _MAX_ISSUES_PER_GROUP:
            lines.append(f"- ... {len(issues) - _MAX_ISSUES_PER_GROUP} more")
        return lines

    # ----- lookups & helpers ----------------------------------------------

    def _source_lookup(self, docs: list[dict]) -> dict[str, dict]:
        lookup: dict[str, dict] = {}
        for doc in docs:
            if self._is_wiki_page(doc):
                continue
            for key in self._doc_keys(doc):
                lookup.setdefault(key, doc)
        return lookup

    def _wiki_lookup(self, docs: list[dict]) -> dict[str, dict]:
        lookup: dict[str, dict] = {}
        for doc in docs:
            if not self._is_wiki_page(doc):
                continue
            relative = self._doc_path(doc).replace("/wiki/", "", 1)
            lookup[relative.lower()] = doc
            lookup.setdefault(doc["filename"].lower(), doc)
        return lookup

    def _resolve_source(self, filename: str, source_lookup: dict[str, dict]) -> dict | None:
        key = filename.strip().lower()
        if key in source_lookup:
            return source_lookup[key]
        return source_lookup.get(_SOURCE_EXT_RE.sub("", key))

    def _resolve_wiki_link(self, link_path: str, wiki_lookup: dict[str, dict]) -> dict | None:
        key = link_path.split("#", 1)[0].lower()
        return (
            wiki_lookup.get(key)
            or wiki_lookup.get(f"{key}.md")
            or wiki_lookup.get(key.split("/")[-1])
        )

    def _doc_keys(self, doc: dict) -> list[str]:
        filename = doc["filename"].lower()
        title = str(doc.get("title") or "").lower()
        keys = [filename, _SOURCE_EXT_RE.sub("", filename)]
        if title:
            keys.extend([title, _SOURCE_EXT_RE.sub("", title)])
        return [k for k in keys if k]

    def _has_mid_document_footnotes(self, content: str) -> bool:
        """True when a footnote definition is followed by non-footnote prose.

        Uses the same tail-compatibility rule as the write tool's append logic
        (`_is_footnote_suffix_line`), so "grouped at the end" means the same
        thing in both places.
        """
        lines = content.rstrip().splitlines()
        for idx, line in enumerate(lines):
            if _FOOTNOTE_DEF_RE.match(line):
                return not all(_is_footnote_suffix_line(suffix) for suffix in lines[idx + 1:])
        return False

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        return sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()})

    def _normalize_date(self, value) -> str | None:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat().split("T", 1)[0]
        return str(value).split("T", 1)[0]

    def _doc_path(self, doc: dict) -> str:
        return f"{doc['path']}{doc['filename']}"

    def _is_wiki_page(self, doc: dict) -> bool:
        return doc.get("path", "").startswith("/wiki/")

    def _is_root_page(self, doc: dict) -> bool:
        return self._doc_path(doc) in _ROOT_PAGES

    def _is_ledger_page(self, doc: dict) -> bool:
        return self._doc_path(doc) in _LEDGER_PAGES

    def _footnote_sort_key(self, value: str) -> tuple[int, str]:
        return (0, f"{int(value):08d}") if value.isdigit() else (1, value)


def register(mcp: FastMCP, get_user_id, fs_factory) -> None:
    @mcp.tool(
        name="lint",
        description=(
            "Run deterministic hygiene checks across a knowledge base.\n\n"
            "Checks wiki frontmatter, tag/date index consistency, footnote hygiene, "
            "citation resolution, citation graph edges, dangling wiki links, orphan pages, "
            "uncited sources, and stale pages.\n\n"
            "For classified corpus entries (八维标注) it additionally checks dimension "
            "completeness (公理一), code-table membership, overdue reviews (review_due), "
            "and pending manual reviews — and appends the 阶段×大类 coverage ledger "
            "(货架覆盖率) with empty shelf cells to guide the next collection round, plus "
            "quality KPIs: 分面完备率, 货架覆盖率, 时效达标率, 引用溯源率.\n\n"
            "Use `path` to scope the run, e.g. `*`, `/wiki/**`, or `/wiki/concepts/*.md`."
        ),
    )
    async def lint(
        ctx: Context,
        knowledge_base: str,
        path: str = "*",
        scope: Scope = "all",
        include_graph: bool = True,
    ) -> str:
        user_id = get_user_id(ctx)
        fs = fs_factory(user_id)

        kb = await fs.resolve_kb(knowledge_base)
        if not kb:
            return f"Knowledge base '{knowledge_base}' not found."

        handler = LintHandler(fs, kb)
        return await handler.run(path=path, scope=scope, include_graph=include_graph)
