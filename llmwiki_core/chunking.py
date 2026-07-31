"""Pure text chunking shared by API, MCP, and local runtimes."""

import re
from dataclasses import dataclass

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_CHUNK_TOKENS = 32
MAX_CHUNK_CHARS = 10_000

SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    index: int
    content: str
    page: int | None
    start_char: int
    token_count: int
    header_breadcrumb: str = ""


def chunk_text(
    content: str | None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    page: int | None = None,
    start_char_offset: int = 0,
) -> list[Chunk]:
    """Chunk text into overlapping segments while tracking Markdown headers."""
    if not content or not content.strip():
        return []

    paragraphs = _split_paragraphs(content)
    header_stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    current_blocks: list[str] = []
    current_tokens = 0
    current_start = start_char_offset
    char_pos = start_char_offset

    for paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)

        header_match = HEADER_RE.match(paragraph)
        if header_match:
            level = len(header_match.group(1))
            heading = header_match.group(2).strip()
            header_stack = [(item_level, text) for item_level, text in header_stack if item_level < level]
            header_stack.append((level, heading))

        if current_tokens + paragraph_tokens > chunk_size and current_blocks:
            content_part = "\n\n".join(current_blocks)
            if _estimate_tokens(content_part) >= MIN_CHUNK_TOKENS:
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        content=content_part,
                        page=page,
                        start_char=current_start,
                        token_count=_estimate_tokens(content_part),
                        header_breadcrumb=" > ".join(text for _, text in header_stack),
                    )
                )

            overlap_blocks, overlap_tokens = _get_overlap(current_blocks, overlap)
            current_blocks = overlap_blocks
            current_tokens = overlap_tokens
            current_start = char_pos - sum(len(block) + 2 for block in overlap_blocks)

        current_blocks.append(paragraph)
        current_tokens += paragraph_tokens
        char_pos += len(paragraph) + 2

    if current_blocks:
        content_part = "\n\n".join(current_blocks)
        if _estimate_tokens(content_part) >= MIN_CHUNK_TOKENS:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    content=content_part,
                    page=page,
                    start_char=current_start,
                    token_count=_estimate_tokens(content_part),
                    header_breadcrumb=" > ".join(text for _, text in header_stack),
                )
            )

    return _enforce_max_chars(chunks)


def chunk_pages(page_contents: list[tuple[int, str]]) -> list[Chunk]:
    """Chunk multiple pages while assigning globally sequential indexes."""
    all_chunks: list[Chunk] = []
    for page_number, content in page_contents:
        for chunk in chunk_text(content, page=page_number):
            chunk.index = len(all_chunks)
            all_chunks.append(chunk)
    return all_chunks


def _enforce_max_chars(chunks: list[Chunk]) -> list[Chunk]:
    if not any(len(chunk.content) > MAX_CHUNK_CHARS for chunk in chunks):
        return chunks

    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) <= MAX_CHUNK_CHARS:
            result.append(
                Chunk(
                    index=len(result),
                    content=chunk.content,
                    page=chunk.page,
                    start_char=chunk.start_char,
                    token_count=chunk.token_count,
                    header_breadcrumb=chunk.header_breadcrumb,
                )
            )
            continue
        offset = 0
        for piece in _split_oversized(chunk.content):
            result.append(
                Chunk(
                    index=len(result),
                    content=piece,
                    page=chunk.page,
                    start_char=chunk.start_char + offset,
                    token_count=_estimate_tokens(piece),
                    header_breadcrumb=chunk.header_breadcrumb,
                )
            )
            offset += len(piece)
    return result


def _split_oversized(text: str) -> list[str]:
    parts = SENTENCE_RE.split(text)
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(part) <= MAX_CHUNK_CHARS:
            current = part
        else:
            pieces.extend(part[index : index + MAX_CHUNK_CHARS] for index in range(0, len(part), MAX_CHUNK_CHARS))
            current = ""
    if current:
        pieces.append(current)
    return pieces


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _get_overlap(blocks: list[str], target_tokens: int) -> tuple[list[str], int]:
    result: list[str] = []
    tokens = 0
    for block in reversed(blocks):
        block_tokens = _estimate_tokens(block)
        if tokens + block_tokens > target_tokens:
            break
        result.insert(0, block)
        tokens += block_tokens
    return result, tokens
