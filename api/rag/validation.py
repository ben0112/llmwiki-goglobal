"""Deterministic validation for generated RAG wiki drafts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any, NoReturn
from urllib.parse import urlsplit
from uuid import UUID

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

from llmwiki_core.rag import MAX_PAGE_CHARS, RagCitation, RagDomainError, RagWorkItem
from llmwiki_core.references import ReferenceEdge, parse_citation_filename, parse_wiki_links
from llmwiki_core.wiki import WikiWriteBundle

from .prompts import RagDraft

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.+?\n)---[ \t]*\n", re.DOTALL)
_FOOTNOTE_DEFINITION_RE = re.compile(r"^\[\^(\d+)\]:\s*(.+)$", re.MULTILINE)
_FOOTNOTE_USE_RE = re.compile(r"\[\^(\d+)\]")
_EXTERNAL_SCHEMES = frozenset({"http", "https", "mailto", "data"})
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})
_NONFAILED_EVIDENCE_STATUSES = frozenset({"pending", "processing", "ready"})
_MAX_METADATA_DEPTH = 32
_MAX_YAML_DEPTH = 32
_MAX_YAML_NODES = 4_096
_MAX_YAML_TOKENS = 4_096
_MAX_MARKDOWN_SCAN_STEPS = MAX_PAGE_CHARS * 8 + 8
_MAX_MARKDOWN_NESTING = 32
_MAX_LINT_SUMMARY_BYTES = 16_384
_POSTGRES_INTEGER_MAX = 2_147_483_647
_ASCII_ESCAPABLE_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ESCAPED_PUNCTUATION_BASE = 0xDC00


@dataclass(frozen=True, slots=True)
class _InlineToken:
    kind: str
    destination: str


def validate_draft(
    draft: RagDraft,
    selected_evidence: Sequence[object],
    *,
    document_id: UUID | str,
    expected_version: int | None,
    target_path: str,
    max_page_chars: int,
) -> tuple[WikiWriteBundle, dict[str, object]]:
    """Validate a draft without repairing it and derive its atomic write bundle."""
    try:
        return _validated_draft(
            draft,
            selected_evidence,
            document_id=document_id,
            expected_version=expected_version,
            target_path=target_path,
            max_page_chars=max_page_chars,
        )
    except RagDomainError:
        raise
    except Exception:  # noqa: BLE001 -- sanitize untrusted model/evidence values
        raise RagDomainError("rag_invalid_draft", "The generated draft was invalid.") from None


def _validated_draft(
    draft: RagDraft,
    selected_evidence: Sequence[object],
    *,
    document_id: UUID | str,
    expected_version: int | None,
    target_path: str,
    max_page_chars: int,
) -> tuple[WikiWriteBundle, dict[str, object]]:
    if type(draft) is not RagDraft:
        _invalid_draft()
    content = _valid_content(draft.content, max_page_chars)
    canonical_document_id = _document_id(document_id)
    if expected_version is not None and (
        type(expected_version) is not int or not 1 <= expected_version < _POSTGRES_INTEGER_MAX
    ):
        _invalid_draft()
    canonical_target = _target_path(target_path)
    directory, filename = canonical_target.rsplit("/", 1)
    directory = directory + "/"

    metadata = _frontmatter(content)
    title, tags, description, document_date = _required_metadata(metadata)
    scan_content, mermaid_count, inline_tokens = _markdown_scan_content(content)
    visual_count = _validate_images(inline_tokens) + mermaid_count
    if visual_count < 1:
        _invalid_draft()

    evidence_by_identity, evidence_items = _selected_evidence(selected_evidence)
    edges, citation_count = _citation_edges(
        scan_content,
        draft.citations,
        evidence_by_identity,
        evidence_items,
    )
    internal_link_count = _validate_links(inline_tokens, canonical_target)

    normalized_metadata = _json_value(metadata)
    if type(normalized_metadata) is not dict:
        _invalid_draft()
    normalized_metadata.update(
        {
            "title": title,
            "tags": list(tags),
            "description": description,
            "date": document_date,
        }
    )
    lint_summary = _lint_summary(citation_count, internal_link_count, visual_count)
    bundle = WikiWriteBundle.build(
        document_id=canonical_document_id,
        expected_version=expected_version,
        filename=filename,
        path=directory,
        file_type="md",
        content=content,
        title=title,
        tags=tags,
        date=document_date,
        metadata=normalized_metadata,
        edges=edges,
    )
    return bundle, lint_summary


def _valid_content(raw: object, maximum: object) -> str:
    if type(maximum) is not int or not 1 <= maximum <= MAX_PAGE_CHARS:
        _invalid_draft()
    if type(raw) is not str or not raw or len(raw) > maximum or "\x00" in raw:
        _invalid_draft()
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        _invalid_draft()
    return raw


def _document_id(raw: object) -> str:
    if isinstance(raw, UUID):
        return str(raw)
    if type(raw) is not str:
        _invalid_draft()
    try:
        parsed = UUID(raw)
    except (ValueError, AttributeError):
        _invalid_draft()
    if str(parsed) != raw:
        _invalid_draft()
    return raw


def _target_path(raw: object) -> str:
    try:
        normalized = RagWorkItem.build(0, raw, "validation", "validation").path
    except (TypeError, ValueError):
        _invalid_draft()
    if not normalized.startswith("/wiki/"):
        _invalid_draft()
    return normalized


def _frontmatter(content: str) -> dict[str, Any]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        _invalid_draft()
    raw = match.group(1)
    try:
        _validate_yaml_tokens(raw)
        node = yaml.compose(raw, Loader=yaml.SafeLoader)
        _validate_yaml_nodes(node)
        parsed = yaml.safe_load(raw)
    except (TypeError, ValueError, RecursionError, yaml.YAMLError):
        _invalid_draft()
    if type(parsed) is not dict or not all(type(key) is str for key in parsed):
        _invalid_draft()
    return parsed


def _validate_yaml_tokens(raw: str) -> None:
    for count, token in enumerate(yaml.scan(raw, Loader=yaml.SafeLoader), start=1):
        if count > _MAX_YAML_TOKENS or isinstance(token, (AnchorToken, AliasToken, TagToken)):
            raise ValueError


def _validate_yaml_nodes(node: object) -> None:
    if node is None:
        raise ValueError
    pending = [(node, 0)]
    count = 0
    while pending:
        current, depth = pending.pop()
        count += 1
        if count > _MAX_YAML_NODES or depth > _MAX_YAML_DEPTH:
            raise ValueError
        if isinstance(current, MappingNode):
            seen: set[str] = set()
            for key, value in current.value:
                if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                    raise ValueError
                if key.value == "<<" or key.value in seen:
                    raise ValueError
                seen.add(key.value)
                pending.append((value, depth + 1))
        elif isinstance(current, SequenceNode):
            pending.extend((value, depth + 1) for value in current.value)
        elif not isinstance(current, ScalarNode):
            raise ValueError


class _MarkdownScanBudget:
    __slots__ = ("remaining",)

    def __init__(self) -> None:
        self.remaining = _MAX_MARKDOWN_SCAN_STEPS

    def consume(self, amount: int) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            _invalid_draft()


def _markdown_scan_content(content: str) -> tuple[str, int, tuple[_InlineToken, ...]]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        _invalid_draft()
    body = content[match.end() :]
    budget = _MarkdownScanBudget()
    without_comments = _strip_html_comments(body, budget)
    outside_fences, mermaid_count = _strip_fenced_blocks(without_comments, budget)
    without_inline_code = _strip_inline_code(outside_fences, budget)
    active = _strip_backslash_escapes(without_inline_code, budget)
    _reject_raw_html(active, budget)
    _reject_reference_definitions(active, budget)
    tokens = _tokenize_inline_links(active, budget)
    footnote_content = _mask_blockquote_lines(active, budget)
    return footnote_content, mermaid_count, tokens


def _strip_html_comments(content: str, budget: _MarkdownScanBudget) -> str:
    budget.consume(len(content))
    pieces: list[str] = []
    position = 0
    while position < len(content):
        opening = content.find("<!--", position)
        if opening < 0:
            pieces.append(content[position:])
            break
        pieces.append(content[position:opening])
        closing = content.find("-->", opening + 4)
        if closing < 0:
            pieces.append(_preserved_line_breaks(content[opening:]))
            break
        pieces.append(_preserved_line_breaks(content[opening : closing + 3]))
        position = closing + 3
    return "".join(pieces)


def _strip_fenced_blocks(
    content: str,
    budget: _MarkdownScanBudget,
) -> tuple[str, int]:
    lines = content.splitlines(keepends=True)
    budget.consume(sum(len(line) for line in lines))
    outside: list[str] = []
    opened: tuple[str, int, str, int] | None = None
    mermaid_count = 0
    for line in lines:
        quote_depth, container_content = _container_content(line)
        if opened is None:
            if _is_indented_code(container_content):
                outside.append(_preserved_line_breaks(line))
                continue
            candidate = _opening_fence(container_content)
            if candidate is None:
                outside.append(line)
                continue
            opened = (*candidate, quote_depth)
            outside.append(_preserved_line_breaks(line))
            continue
        outside.append(_preserved_line_breaks(line))
        if quote_depth == opened[3] and _is_closing_fence(
            container_content,
            opened[0],
            opened[1],
        ):
            if opened[2] == "mermaid":
                mermaid_count += 1
            opened = None
    return "".join(outside), mermaid_count


def _opening_fence(line: str) -> tuple[str, int, str] | None:
    logical = line.rstrip("\r\n")
    indent = len(logical) - len(logical.lstrip(" "))
    if indent > 3:
        return None
    candidate = logical[indent:]
    if not candidate or candidate[0] not in {"`", "~"}:
        return None
    marker = candidate[0]
    run_length = len(candidate) - len(candidate.lstrip(marker))
    if run_length < 3:
        return None
    info = candidate[run_length:].strip()
    if marker == "`" and "`" in info:
        return None
    return marker, run_length, info


def _is_closing_fence(line: str, marker: str, opening_length: int) -> bool:
    logical = line.rstrip("\r\n")
    indent = len(logical) - len(logical.lstrip(" "))
    if indent > 3:
        return False
    candidate = logical[indent:]
    run_length = len(candidate) - len(candidate.lstrip(marker))
    return run_length >= opening_length and not candidate[run_length:].strip()


def _container_content(line: str) -> tuple[int, str]:
    logical = line.rstrip("\r\n")
    line_break = line[len(logical) :]
    position = 0
    while position < min(3, len(logical)) and logical[position] == " ":
        position += 1
    quote_depth = 0
    while position < len(logical) and logical[position] == ">":
        quote_depth += 1
        position += 1
        if position < len(logical) and logical[position] in {" ", "\t"}:
            position += 1
    if quote_depth == 0:
        position = 0
    return quote_depth, logical[position:] + line_break


def _is_indented_code(line: str) -> bool:
    logical = line.rstrip("\r\n")
    return logical.startswith("\t") or logical.startswith("    ")


def _strip_inline_code(content: str, budget: _MarkdownScanBudget) -> str:
    budget.consume(len(content))
    rendered = list(content)
    position = 0
    while position < len(content):
        if content[position] != "`" or _is_escaped(content, position):
            position += 1
            continue
        opening_end = _run_end(content, position, "`")
        opening_length = opening_end - position
        search = opening_end
        closing_end: int | None = None
        while search < len(content):
            if content[search] != "`":
                search += 1
                continue
            candidate_end = _run_end(content, search, "`")
            if not _is_escaped(content, search) and candidate_end - search == opening_length:
                closing_end = candidate_end
                break
            search = candidate_end
        if closing_end is None:
            _invalid_draft()
        for index in range(position, closing_end):
            if rendered[index] not in {"\r", "\n"}:
                rendered[index] = " "
        position = closing_end
    return "".join(rendered)


def _strip_backslash_escapes(content: str, budget: _MarkdownScanBudget) -> str:
    budget.consume(len(content))
    rendered: list[str] = []
    position = 0
    while position < len(content):
        if content[position] != "\\":
            rendered.append(content[position])
            position += 1
            continue
        run_end = _run_end(content, position, "\\")
        run_length = run_end - position
        rendered.extend(_encoded_punctuation("\\") for _ in range(run_length // 2))
        if run_length % 2 == 1:
            if run_end < len(content) and content[run_end] in _ASCII_ESCAPABLE_PUNCTUATION:
                rendered.append(_encoded_punctuation(content[run_end]))
                run_end += 1
            else:
                rendered.append("\\")
        position = run_end
    return "".join(rendered)


def _encoded_punctuation(value: str) -> str:
    return chr(_ESCAPED_PUNCTUATION_BASE + ord(value))


def _decoded_punctuation(value: str) -> str:
    codepoint = ord(value)
    minimum = _ESCAPED_PUNCTUATION_BASE
    maximum = minimum + 127
    if minimum <= codepoint <= maximum:
        return chr(codepoint - minimum)
    return value


def _reject_raw_html(content: str, budget: _MarkdownScanBudget) -> None:
    budget.consume(len(content))
    position = 0
    while position < len(content):
        if content[position] != "<":
            position += 1
            continue
        candidate = position + 1
        if candidate < len(content) and content[candidate] == "/":
            candidate += 1
        if candidate < len(content) and _is_ascii_letter(content[candidate]):
            _link_invalid()
        position += 1


def _is_ascii_letter(character: str) -> bool:
    return "A" <= character <= "Z" or "a" <= character <= "z"


def _reject_reference_definitions(content: str, budget: _MarkdownScanBudget) -> None:
    budget.consume(len(content))
    for line in content.splitlines():
        candidate = _reference_definition_candidate(line)
        if candidate is None or not candidate.startswith("["):
            continue
        closing = 1
        while closing + 1 < len(candidate):
            if candidate[closing] == "]" and candidate[closing + 1] == ":":
                label = candidate[1:closing]
                if not (label.startswith("^") and label[1:].isdigit()):
                    _link_invalid()
                break
            closing += 1


def _reference_definition_candidate(line: str) -> str | None:
    position = 0
    while position < len(line) and position < 3 and line[position] == " ":
        position += 1
    if position < len(line) and line[position] == " ":
        return None
    while position < len(line) and line[position] == ">":
        position += 1
        if position < len(line) and line[position] in {" ", "\t"}:
            position += 1
    list_end = _list_prefix_end(line, position)
    if list_end is not None:
        position = list_end
    return line[position:]


def _list_prefix_end(line: str, position: int) -> int | None:
    if position >= len(line):
        return None
    if line[position] in {"-", "+", "*"}:
        end = position + 1
    elif line[position].isdigit():
        end = position
        while end < len(line) and line[end].isdigit():
            end += 1
        if end >= len(line) or line[end] not in {".", ")"}:
            return None
        end += 1
    else:
        return None
    if end >= len(line) or line[end] not in {" ", "\t"}:
        return None
    while end < len(line) and line[end] in {" ", "\t"}:
        end += 1
    return end


def _mask_blockquote_lines(content: str, budget: _MarkdownScanBudget) -> str:
    lines = content.splitlines(keepends=True)
    budget.consume(sum(len(line) for line in lines))
    return "".join(_preserved_line_breaks(line) if _container_content(line)[0] else line for line in lines)


def _run_end(content: str, start: int, marker: str) -> int:
    end = start
    while end < len(content) and content[end] == marker:
        end += 1
    return end


def _is_escaped(content: str, position: int) -> bool:
    backslashes = 0
    cursor = position - 1
    while cursor >= 0 and content[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _preserved_line_breaks(content: str) -> str:
    return "".join(character for character in content if character in {"\r", "\n"})


def _required_metadata(metadata: dict[str, Any]) -> tuple[str, tuple[str, ...], str, str]:
    title = _nonempty_text(metadata.get("title"))
    description = _nonempty_text(metadata.get("description"))
    raw_tags = metadata.get("tags")
    if type(raw_tags) is not list or not raw_tags:
        _invalid_draft()
    tags = tuple(_nonempty_text(tag) for tag in raw_tags)
    if len(set(tags)) != len(tags):
        _invalid_draft()
    raw_date = metadata.get("date")
    if type(raw_date) is date:
        document_date = raw_date.isoformat()
    elif type(raw_date) is str:
        normalized_date = _nonempty_text(raw_date)
        try:
            parsed_date = date.fromisoformat(normalized_date)
        except ValueError:
            _invalid_draft()
        if parsed_date.isoformat() != normalized_date:
            _invalid_draft()
        document_date = normalized_date
    else:
        _invalid_draft()
    return title, tags, description, document_date


def _nonempty_text(raw: object) -> str:
    if type(raw) is not str or "\x00" in raw:
        _invalid_draft()
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        _invalid_draft()
    normalized = raw.strip()
    if not normalized:
        _invalid_draft()
    return normalized


def _selected_evidence(
    selected: Sequence[object],
) -> tuple[dict[tuple[UUID, int, int, int | None], dict[str, object]], tuple[dict[str, object], ...]]:
    if isinstance(selected, (str, bytes, bytearray)) or not isinstance(selected, Sequence):
        _citation_invalid()
    identities: dict[tuple[UUID, int, int, int | None], dict[str, object]] = {}
    items: list[dict[str, object]] = []
    for raw in selected:
        item = _evidence_projection(raw)
        status = item["status"]
        status_value = status.value if hasattr(status, "value") else status
        if (
            type(status_value) is not str
            or status_value not in _NONFAILED_EVIDENCE_STATUSES
            or type(item["archived"]) is not bool
            or item["archived"]
        ):
            _citation_invalid()
        identity = _evidence_identity(item)
        if identity in identities:
            _citation_invalid()
        identities[identity] = item
        items.append(item)
    return identities, tuple(items)


def _evidence_projection(raw: object) -> dict[str, object]:
    required = (
        "document_id",
        "document_version",
        "chunk_index",
        "page",
        "filename",
        "status",
        "archived",
    )
    if isinstance(raw, Mapping):
        if not all(name in raw for name in required):
            _citation_invalid()
        item = {name: raw[name] for name in required}
    else:
        try:
            item = {name: getattr(raw, name) for name in required}
        except (AttributeError, TypeError):
            _citation_invalid()
    filename = item["filename"]
    if type(filename) is not str or not filename or "\x00" in filename:
        _citation_invalid()
    try:
        filename.encode("utf-8")
    except UnicodeEncodeError:
        _citation_invalid()
    if PurePosixPath(filename).name != filename:
        _citation_invalid()
    return item


def _evidence_identity(item: Mapping[str, object]) -> tuple[UUID, int, int, int | None]:
    raw_id = item["document_id"]
    if isinstance(raw_id, UUID):
        document_id = raw_id
    elif type(raw_id) is str:
        try:
            document_id = UUID(raw_id)
        except ValueError:
            _citation_invalid()
        if str(document_id) != raw_id:
            _citation_invalid()
    else:
        _citation_invalid()
    document_version = item["document_version"]
    chunk_index = item["chunk_index"]
    page = item["page"]
    if type(document_version) is not int or document_version < 1:
        _citation_invalid()
    if type(chunk_index) is not int or chunk_index < 0:
        _citation_invalid()
    if page is not None and (type(page) is not int or page < 1):
        _citation_invalid()
    return document_id, document_version, chunk_index, page


def _citation_edges(
    content: str,
    citations: object,
    selected: Mapping[tuple[UUID, int, int, int | None], dict[str, object]],
    evidence_items: Sequence[dict[str, object]],
) -> tuple[tuple[ReferenceEdge, ...], int]:
    citation_identities = _citation_identity_set(citations, selected)
    definitions = _footnote_definitions(content)

    resolved: set[tuple[UUID, int, int, int | None]] = set()
    edges: list[ReferenceEdge] = []
    for definition in definitions:
        identity = _resolve_definition(definition.group(2), citation_identities, evidence_items)
        if identity in resolved:
            _citation_invalid()
        resolved.add(identity)
        edges.append(ReferenceEdge(str(identity[0]), "cites", identity[3]))
    if resolved != citation_identities:
        _citation_invalid()
    return tuple(edges), len(definitions)


def _citation_identity_set(
    citations: object,
    selected: Mapping[tuple[UUID, int, int, int | None], dict[str, object]],
) -> set[tuple[UUID, int, int, int | None]]:
    if type(citations) is not tuple or not all(type(item) is RagCitation for item in citations):
        _citation_invalid()
    identities = {(item.document_id, item.document_version, item.chunk_index, item.page) for item in citations}
    if len(identities) != len(citations) or not identities <= set(selected):
        _citation_invalid()
    return identities


def _footnote_definitions(content: str) -> list[re.Match[str]]:
    definitions = list(_FOOTNOTE_DEFINITION_RE.finditer(content))
    identifiers = [match.group(1) for match in definitions]
    if len(set(identifiers)) != len(identifiers):
        _citation_invalid()
    body_without_definitions = _FOOTNOTE_DEFINITION_RE.sub("", content)
    if set(_FOOTNOTE_USE_RE.findall(body_without_definitions)) != set(identifiers):
        _citation_invalid()
    return definitions


def _resolve_definition(
    raw: str,
    citation_identities: set[tuple[UUID, int, int, int | None]],
    evidence_items: Sequence[dict[str, object]],
) -> tuple[UUID, int, int, int | None]:
    filename, page = parse_citation_filename(raw)
    candidates = []
    for item in evidence_items:
        identity = _evidence_identity(item)
        if (
            identity in citation_identities
            and str(item["filename"]).casefold() == filename.casefold()
            and identity[3] == page
        ):
            candidates.append(identity)
    if len(candidates) != 1:
        _citation_invalid()
    return candidates[0]


def _tokenize_inline_links(
    content: str,
    budget: _MarkdownScanBudget,
) -> tuple[_InlineToken, ...]:
    budget.consume(len(content))
    tokens: list[_InlineToken] = []
    position = 0
    while position < len(content):
        if content.startswith("![", position):
            kind = "image"
            label_open = position + 1
        elif content[position] == "[" and not content.startswith("[^", position):
            kind = "link"
            label_open = position
        else:
            position += 1
            continue
        label_close = _matching_label_close(content, label_open)
        if label_close is None:
            if kind == "image":
                _link_invalid()
            break
        destination_open = label_close + 1
        if destination_open < len(content) and content[destination_open] == "(":
            destination, token_end = _parse_inline_destination(content, destination_open)
            tokens.append(_InlineToken(kind, destination))
            position = token_end
            continue
        reference_open = destination_open
        while reference_open < len(content) and content[reference_open] in {" ", "\t"}:
            reference_open += 1
        if reference_open < len(content) and content[reference_open] == "[":
            _link_invalid()
        if kind == "image":
            _link_invalid()
        position = label_close + 1
    return tuple(tokens)


def _matching_label_close(content: str, opening: int) -> int | None:
    position = opening + 1
    while position < len(content):
        if content[position] == "[":
            _link_invalid()
        elif content[position] == "]":
            return position
        position += 1
    return None


def _parse_inline_destination(content: str, opening: int) -> tuple[str, int]:
    position = opening + 1
    while position < len(content) and content[position] in {" ", "\t"}:
        position += 1
    if position >= len(content):
        _link_invalid()
    if content[position] == "<":
        destination, position = _angle_destination(content, position)
        return _finish_inline_destination(content, position, destination, needs_separator=True)
    return _plain_destination(content, position)


def _angle_destination(content: str, opening: int) -> tuple[str, int]:
    position = opening + 1
    start = position
    while position < len(content):
        character = content[position]
        if character in {"\r", "\n", "<"}:
            _link_invalid()
        if character == ">":
            return _decode_destination(content[start:position]), position + 1
        position += 1
    _link_invalid()
    raise AssertionError("unreachable")


def _plain_destination(content: str, start: int) -> tuple[str, int]:
    position = start
    depth = 0
    while position < len(content):
        character = content[position]
        if character in {"\r", "\n", "<", ">"}:
            _link_invalid()
        if character == "(":
            depth += 1
            if depth > _MAX_MARKDOWN_NESTING:
                _link_invalid()
        elif character == ")":
            if depth == 0:
                destination = _decode_destination(content[start:position])
                return destination, position + 1
            depth -= 1
        elif character in {" ", "\t"} and depth == 0:
            destination = _decode_destination(content[start:position])
            return _finish_inline_destination(
                content,
                position,
                destination,
                needs_separator=False,
            )
        position += 1
    _link_invalid()
    raise AssertionError("unreachable")


def _finish_inline_destination(
    content: str,
    position: int,
    destination: str,
    *,
    needs_separator: bool,
) -> tuple[str, int]:
    separator_start = position
    while position < len(content) and content[position] in {" ", "\t"}:
        position += 1
    if position < len(content) and content[position] == ")":
        return destination, position + 1
    if needs_separator and position == separator_start:
        _link_invalid()
    if position >= len(content) or content[position] not in {'"', "'", "("}:
        _link_invalid()
    title_open = content[position]
    title_close = ")" if title_open == "(" else title_open
    position += 1
    while position < len(content) and content[position] != title_close:
        if content[position] in {"\r", "\n"}:
            _link_invalid()
        position += 1
    if position >= len(content):
        _link_invalid()
    position += 1
    while position < len(content) and content[position] in {" ", "\t"}:
        position += 1
    if position >= len(content) or content[position] != ")":
        _link_invalid()
    return destination, position + 1


def _decode_destination(raw: str) -> str:
    return "".join(_decoded_punctuation(character) for character in raw)


def _validate_links(tokens: Sequence[_InlineToken], target_path: str) -> int:
    current_dir = target_path.removeprefix("/wiki/").rsplit("/", 1)[0]
    if current_dir:
        current_dir += "/"
    count = 0
    for token in tokens:
        if token.kind != "link":
            continue
        href = token.destination
        if not href or href.startswith("#"):
            continue
        if _is_external_href(href):
            continue
        _validated_internal_href(href, current_dir)
        count += 1
    return count


def _validate_images(tokens: Sequence[_InlineToken]) -> int:
    count = 0
    for token in tokens:
        if token.kind != "image":
            continue
        uri = token.destination
        if not uri:
            _link_invalid()
        try:
            uri.encode("utf-8")
        except UnicodeEncodeError:
            _link_invalid()
        if "\x00" in uri or "\\" in uri or "%" in uri or uri.startswith("//"):
            _link_invalid()
        parsed = urlsplit(uri)
        if parsed.scheme and parsed.scheme.casefold() not in {"http", "https"}:
            _link_invalid()
        if not parsed.scheme:
            if parsed.path.startswith("/") and not parsed.path.startswith("/wiki/"):
                _link_invalid()
            if ".." in parsed.path.split("/"):
                _link_invalid()
        if PurePosixPath(parsed.path).suffix.casefold() not in _IMAGE_EXTENSIONS:
            _link_invalid()
        count += 1
    return count


def _lint_summary(
    citation_count: int,
    internal_link_count: int,
    visual_count: int,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "citation_count": citation_count,
        "internal_link_count": internal_link_count,
        "visual_count": visual_count,
        "warnings": 0,
    }
    encoded = json.dumps(summary, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_LINT_SUMMARY_BYTES:
        _invalid_draft()
    return summary


def _is_external_href(href: str) -> bool:
    try:
        href.encode("utf-8")
    except UnicodeEncodeError:
        _link_invalid()
    if "\x00" in href or "\\" in href or "%" in href or href.startswith("//"):
        _link_invalid()
    scheme = urlsplit(href).scheme
    if not scheme:
        return False
    if scheme.casefold() not in _EXTERNAL_SCHEMES:
        _link_invalid()
    return True


def _validated_internal_href(href: str, current_dir: str) -> None:
    raw_path = urlsplit(href).path
    if (raw_path.startswith("/") and not raw_path.startswith("/wiki/")) or (".." in raw_path.split("/")):
        _link_invalid()
    if "(" in raw_path or ")" in raw_path:
        relative = _relative_wiki_path(raw_path, current_dir)
    else:
        parsed_paths = parse_wiki_links(f"[link]({raw_path})", current_dir)
        if len(parsed_paths) != 1:
            _link_invalid()
        relative = parsed_paths[0]
    logical = "/wiki/" + relative.lstrip("/")
    parts = [part for part in logical.split("/") if part and part != "."]
    if not parts or parts[0] != "wiki" or ".." in parts:
        _link_invalid()
    suffix = PurePosixPath(parts[-1]).suffix.casefold()
    if suffix and suffix != ".md":
        _link_invalid()


def _relative_wiki_path(raw_path: str, current_dir: str) -> str:
    if raw_path.startswith("/wiki/"):
        return raw_path.removeprefix("/wiki/")
    if raw_path.startswith("./"):
        return current_dir + raw_path[2:]
    if "/" not in raw_path:
        return current_dir + raw_path
    return raw_path


def _json_value(value: object, depth: int = 0) -> object:
    if depth > _MAX_METADATA_DEPTH:
        _invalid_draft()
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _invalid_draft()
        return value
    if type(value) is str:
        if "\x00" in value:
            _invalid_draft()
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            _invalid_draft()
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if type(value) is list:
        return [_json_value(item, depth + 1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    _invalid_draft()
    raise AssertionError("unreachable")


def _invalid_draft() -> NoReturn:
    raise RagDomainError("rag_invalid_draft", "The generated draft was invalid.")


def _citation_invalid() -> NoReturn:
    raise RagDomainError("rag_citation_invalid", "The generated citations were invalid.")


def _link_invalid() -> NoReturn:
    raise RagDomainError("rag_link_invalid", "The generated wiki links were invalid.")
