"""Pure citation and wiki-link parsing shared by API and MCP."""

import re

_CITATION_RE = re.compile(r"\[\^\d+\]:\s*(.+)$", re.MULTILINE)
_WIKI_LINK_RE = re.compile(r"(?<!!)\[(?:[^\]]*)\]\(([^)]+)\)")
_EXTENSION_RE = re.compile(r"\.(pdf|docx?|pptx?|xlsx?|csv|html?|md|txt)$")


def parse_citation_filename(raw: str) -> tuple[str, int | None]:
    """Extract a filename and optional page number from a citation target."""
    raw = raw.strip().lstrip("*").rstrip("*")
    link_match = re.match(r"\[([^\]]+)\]\([^)]*\)(.*)$", raw)
    if link_match:
        raw = f"{link_match.group(1)}{link_match.group(2)}"
    raw = re.split(r"\s+[-–—]\s+", raw, maxsplit=1)[0].strip()
    page_match = re.search(r",\s*p\.?\s*(\d+)\b", raw)
    if page_match:
        return raw[: page_match.start()].strip(), int(page_match.group(1))
    chinese_match = re.search(
        r"[,，]\s*第\s*([0-9〇零一二三四五六七八九十百千]+)\s*([条页章节款])",
        raw,
    )
    if chinese_match:
        number = chinese_match.group(1)
        page = int(number) if chinese_match.group(2) == "页" and number.isdigit() else None
        return raw[: chinese_match.start()].strip(), page
    return raw, None


def parse_wiki_links(content: str, current_dir: str) -> list[str]:
    """Extract non-asset internal links resolved relative to the current wiki directory."""
    paths: list[str] = []
    for match in _WIKI_LINK_RE.finditer(content):
        href = match.group(1)
        if href.startswith(("http", "#", "mailto:", "data:")):
            continue
        if re.search(r"\.(png|jpg|jpeg|gif|webp|svg)$", href, re.IGNORECASE):
            continue
        resolved = _resolve_wiki_href(href, current_dir)
        if resolved:
            paths.append(resolved)
    return paths


def _resolve_wiki_href(href: str, current_dir: str) -> str:
    if href.startswith("/wiki/"):
        return href.replace("/wiki/", "", 1)
    if href.startswith("./"):
        return current_dir + href[2:] if current_dir else href[2:]
    if href.startswith("../"):
        resolved_parts: list[str] = []
        for part in (current_dir.rstrip("/") + "/" + href).split("/"):
            if part == "..":
                if resolved_parts:
                    resolved_parts.pop()
            elif part and part != ".":
                resolved_parts.append(part)
        return "/".join(resolved_parts)
    if "/" not in href:
        return current_dir + href if current_dir else href
    return href


def build_lookup_maps(
    all_docs: list[dict],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Build filename, stem, and wiki-relative-path document lookup maps."""
    filename_to_doc: dict[str, dict] = {}
    base_to_doc: dict[str, dict] = {}
    wiki_path_to_doc: dict[str, dict] = {}
    for document in all_docs:
        filename = document["filename"].lower()
        filename_to_doc.setdefault(filename, document)
        if document.get("title"):
            filename_to_doc.setdefault(document["title"].lower(), document)
        base_to_doc.setdefault(_EXTENSION_RE.sub("", filename), document)
        if document["path"].startswith("/wiki/"):
            relative = (document["path"] + document["filename"]).replace("/wiki/", "", 1)
            wiki_path_to_doc[relative.lower()] = document
    return filename_to_doc, base_to_doc, wiki_path_to_doc


def extract_references(
    content: str,
    doc_id: str,
    wiki_dir: str,
    filename_to_doc: dict[str, dict],
    base_to_doc: dict[str, dict],
    wiki_path_to_doc: dict[str, dict],
) -> list[dict]:
    """Return deduplicated content-derived citation and wiki-link edges."""
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    source_id = str(doc_id)

    for match in _CITATION_RE.finditer(content):
        filename, page = parse_citation_filename(match.group(1))
        normalized = filename.lower()
        target = filename_to_doc.get(normalized) or base_to_doc.get(_EXTENSION_RE.sub("", normalized))
        if target:
            _append_edge(edges, seen, source_id, target, "cites", page)

    for link_path in parse_wiki_links(content, wiki_dir):
        normalized = link_path.lower()
        target = wiki_path_to_doc.get(normalized) or wiki_path_to_doc.get(normalized + ".md")
        if not target:
            target = wiki_path_to_doc.get(normalized.split("/")[-1])
        if target:
            _append_edge(edges, seen, source_id, target, "links_to", None)
    return edges


def _append_edge(
    edges: list[dict],
    seen: set[tuple[str, str]],
    source_id: str,
    target: dict,
    reference_type: str,
    page: int | None,
) -> None:
    target_id = str(target["id"])
    key = (target_id, reference_type)
    if target_id != source_id and key not in seen:
        seen.add(key)
        edges.append({"target_id": target_id, "type": reference_type, "page": page})
