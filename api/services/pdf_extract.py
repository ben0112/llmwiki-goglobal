"""PDF extraction via opendataloader-pdf, with a pypdf text-layer fallback.

Shared module used by both the hosted OCR service and the local processor.
No server-specific dependencies (no asyncpg, S3, httpx).
"""

import base64
import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import opendataloader_pdf

logger = logging.getLogger(__name__)


def _element_to_markdown(el: dict) -> str:
    """Convert a single JSON element to markdown."""
    t = el.get("type", "")
    content = el.get("content", "")

    if t == "heading":
        level = max(1, min(el.get("heading level", 1), 6))
        prefix = "#" * level
        return f"{prefix} {content}"

    if t == "paragraph":
        return content

    if t == "list":
        lines = []
        for item in el.get("list items", []):
            lines.append(f"- {item.get('content', '')}")
            for child in item.get("kids", []):
                lines.append(f"  - {child.get('content', '')}")
        return "\n".join(lines)

    if t == "image":
        # Don't include the data URI in markdown — images are stored separately
        return ""

    if t == "caption":
        return f"*{content}*" if content else ""

    # Skip headers, footers, and unknown types
    return ""


def _parse_data_uri(data_uri: str) -> tuple[bytes, str] | None:
    """Parse a data URI into (bytes, format). Returns None on failure."""
    if not data_uri.startswith("data:"):
        return None
    try:
        header, b64 = data_uri.split(",", 1)
        fmt = "png"
        if "jpeg" in header or "jpg" in header:
            fmt = "jpeg"
        return base64.b64decode(b64), fmt
    except Exception:
        return None


def _elements_to_pages(
    elements: list[dict], total_pages: int,
) -> list[tuple[int, str, list[dict]]]:
    """Group JSON elements by page number and reconstruct markdown per page.

    Returns a list of (page_num, markdown, images) for every page up to total_pages.
    Each image dict has: {"id": str, "bytes": bytes, "format": str}
    """
    page_elements: dict[int, list[dict]] = defaultdict(list)

    for el in elements:
        page_num = el.get("page number")
        if page_num is None or el.get("type") in ("header", "footer"):
            continue
        page_elements[page_num].append(el)

    pages = []
    img_counter = 0
    for page_num in range(1, total_pages + 1):
        parts = []
        images = []
        for el in page_elements.get(page_num, []):
            if el.get("type") == "image":
                src = el.get("source", "")
                parsed = _parse_data_uri(src) if src else None
                if parsed:
                    img_bytes, fmt = parsed
                    img_id = f"img_{page_num}_{img_counter}.{fmt}"
                    img_counter += 1
                    images.append({"id": img_id, "bytes": img_bytes, "format": fmt})
                continue
            md = _element_to_markdown(el)
            if md:
                parts.append(md)
        pages.append((page_num, "\n\n".join(parts), images))

    return pages


def _extract_pdf_fallback(pdf_path: str) -> list[tuple[int, str, list[dict]]]:
    """pypdf 文本层兜底:无版面结构/图片,但比整份失败强。"""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        reader.decrypt("")   # 空口令加密(常见于政务附件);真加密会抛错
    pages: list[tuple[int, str, list[dict]]] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        pages.append((i, text, []))
    if not any(md for _, md, _ in pages):
        raise RuntimeError("pypdf 兜底也未提取到文本")
    return pages


def extract_pdf(pdf_path: str) -> list[tuple[int, str, list[dict]]]:
    """Run opendataloader-pdf and return per-page markdown with images.

    Returns list of (page_num, markdown, images) where images is a list of
    {"id": str, "bytes": bytes, "format": str} dicts.

    opendataloader 失败(损坏/非标准 PDF、Java 异常)时回退 pypdf 文本层;
    两者都失败才抛 RuntimeError。
    """
    try:
        with tempfile.TemporaryDirectory() as extract_dir:
            opendataloader_pdf.convert(
                input_path=pdf_path,
                output_dir=extract_dir,
                format="json",
                image_output="embedded",
                quiet=True,
            )

            json_files = list(Path(extract_dir).glob("*.json"))
            if not json_files:
                raise RuntimeError("opendataloader-pdf produced no output")

            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)

        total_pages = data.get("number of pages", 0)
        elements = data.get("kids", [])
        return _elements_to_pages(elements, total_pages)
    except Exception as e:
        primary = e if isinstance(e, RuntimeError) else RuntimeError(f"PDF extraction failed: {e}")
        try:
            pages = _extract_pdf_fallback(pdf_path)
        except Exception as fb:
            raise RuntimeError(f"{primary}(pypdf 兜底亦失败: {fb})") from e
        logger.warning("opendataloader failed for %s, used pypdf fallback: %s",
                       Path(pdf_path).name, primary)
        return pages
