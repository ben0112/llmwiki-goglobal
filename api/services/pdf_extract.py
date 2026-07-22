"""PDF extraction via opendataloader-pdf, with pypdf and OCR fallbacks.

Shared module used by both the hosted OCR service and the local processor.
No server-specific dependencies (no asyncpg, S3, httpx).

OCR 兜底(tesseract + poppler,镜像内置):设计类报告常见「文本层编码
损坏」—— 字体缺失/伪造 ToUnicode 映射时,任何文本引擎都提不出中文,
只有按页渲染再 OCR 能恢复。文本层产出可疑时抽样对照,OCR 显著更高才
全文 OCR,避免误伤正常稀疏文档(如图册)。
"""

import base64
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

import opendataloader_pdf

logger = logging.getLogger(__name__)

OCR_LANGS = os.environ.get("PDF_OCR_LANGS", "chi_sim+eng")
OCR_MAX_PAGES = int(os.environ.get("PDF_OCR_MAX_PAGES", "300") or 300)
# 全局 OCR 页并发预算:所有文档共享一个进程级线程池(子进程释放 GIL),
# 总并发 = OCR_WORKERS,与同时 OCR 的文档数无关 —— 每文档各开线程池会让
# 文档级并发 × 页级并发相乘超订 CPU(8 文档 × 4 页 = 32 路 tesseract)。
# 每个 tesseract 进程限制为单 OpenMP 线程,预算即真实核占用。
OCR_WORKERS = (int(os.environ.get("PDF_OCR_WORKERS", "0") or 0)
               or max(2, min(8, (os.cpu_count() or 4) - 1)))
OCR_DISABLED = os.environ.get("PDF_OCR_DISABLE", "").lower() in ("1", "true", "yes")
_OCR_PROBE_TRIGGER_AVG = 400   # 文本层平均每页字符低于此值才启动探针
_OCR_PROBE_PAGES = 3
_OCR_WIN_RATIO = 2.0           # OCR 采样均值须超过文本层均值的倍数
_OCR_MIN_PAGE_CHARS = 200      # 且 OCR 采样均值本身要有实质内容


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


def _ocr_available() -> bool:
    return bool(shutil.which("pdftoppm") and shutil.which("tesseract"))


def _pdf_page_count(pdf_path: str) -> int:
    """poppler pdfinfo 的真实页数;失败返回 0(调用方回退用提取结果页数)。"""
    if not shutil.which("pdfinfo"):
        return 0
    try:
        out = subprocess.run(["pdfinfo", pdf_path], capture_output=True, timeout=30)
        m = re.search(rb"^Pages:\s+(\d+)", out.stdout, re.M)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def _ocr_single_page(pdf_path: str, page_num: int, workdir: Path) -> str:
    """渲染单页(200dpi 灰度)并 OCR;失败返回空串。"""
    prefix = workdir / f"p{page_num}"
    try:
        subprocess.run(
            ["pdftoppm", "-f", str(page_num), "-l", str(page_num),
             "-r", "200", "-gray", "-png", pdf_path, str(prefix)],
            capture_output=True, timeout=120, check=True)
        pngs = sorted(workdir.glob(f"p{page_num}-*.png"))
        if not pngs:
            return ""
        out = subprocess.run(
            ["tesseract", str(pngs[0]), "-", "-l", OCR_LANGS, "--psm", "3"],
            capture_output=True, timeout=180,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        for p in pngs:
            p.unlink(missing_ok=True)
        return out.stdout.decode("utf-8", errors="replace").strip()
    except Exception as e:
        logger.warning("OCR page %d failed for %s: %s", page_num, Path(pdf_path).name, e)
        return ""


_ocr_pool: concurrent.futures.ThreadPoolExecutor | None = None
_ocr_pool_lock = threading.Lock()


def _ocr_executor() -> concurrent.futures.ThreadPoolExecutor:
    """进程级共享 OCR 线程池,首次使用时按 OCR_WORKERS 创建,进程存续期复用。"""
    global _ocr_pool
    with _ocr_pool_lock:
        if _ocr_pool is None:
            _ocr_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=OCR_WORKERS, thread_name_prefix="pdf-ocr")
        return _ocr_pool


def _ocr_pages_parallel(pdf_path: str, page_nums: list[int], workdir: Path) -> dict[int, str]:
    """页级并行 OCR:每页(渲染→识别)一个任务,提交到全局共享线程池。

    多份文档同时 OCR 时自然分时共享预算;单份文档独跑时可用满全部预算。
    任务本身不向池内再提交任务,无嵌套等待死锁风险。
    """
    if not page_nums:
        return {}
    ex = _ocr_executor()
    futures = {ex.submit(_ocr_single_page, pdf_path, n, workdir): n for n in page_nums}
    results: dict[int, str] = {}
    for fut in concurrent.futures.as_completed(futures):
        results[futures[fut]] = fut.result()
    return results


def _extract_pdf_ocr(pdf_path: str, page_count: int) -> list[tuple[int, str, list[dict]]]:
    """整份 OCR(页数封顶 OCR_MAX_PAGES),按页并行、逐页渲染不占满磁盘。"""
    n = min(page_count, OCR_MAX_PAGES)
    with tempfile.TemporaryDirectory() as td:
        results = _ocr_pages_parallel(pdf_path, list(range(1, n + 1)), Path(td))
    if page_count > OCR_MAX_PAGES:
        logger.warning("OCR capped at %d/%d pages for %s",
                       OCR_MAX_PAGES, page_count, Path(pdf_path).name)
    return [(i, results.get(i, ""), []) for i in range(1, n + 1)]


def _maybe_ocr(pdf_path: str,
               pages: list[tuple[int, str, list[dict]]]) -> list[tuple[int, str, list[dict]]]:
    """文本层产出可疑时抽样对照 OCR,显著更高则整份改用 OCR 结果。"""
    if OCR_DISABLED or not _ocr_available():
        return pages
    text_total = sum(len(md) for _, md, _ in pages)
    text_avg = text_total / max(len(pages), 1)
    if text_avg >= _OCR_PROBE_TRIGGER_AVG:
        return pages
    page_count = _pdf_page_count(pdf_path) or len(pages)
    if page_count <= 0:
        return pages

    # 均匀抽样探针页
    k = min(_OCR_PROBE_PAGES, page_count)
    probe_nums = sorted({max(1, round(page_count * (i + 1) / (k + 1))) for i in range(k)})
    with tempfile.TemporaryDirectory() as td:
        samples = list(_ocr_pages_parallel(pdf_path, probe_nums, Path(td)).values())
    ocr_avg = sum(len(s) for s in samples) / max(len(samples), 1)

    if ocr_avg < _OCR_MIN_PAGE_CHARS or ocr_avg < _OCR_WIN_RATIO * max(text_avg, 1.0):
        return pages

    logger.warning(
        "Text layer looks broken for %s (text %d chars/page vs OCR sample %d) — "
        "OCR-ing all %d pages", Path(pdf_path).name, int(text_avg), int(ocr_avg), page_count)
    ocr_pages = _extract_pdf_ocr(pdf_path, page_count)
    ocr_total = sum(len(md) for _, md, _ in ocr_pages)
    return ocr_pages if ocr_total > text_total else pages


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
        return _maybe_ocr(pdf_path, _elements_to_pages(elements, total_pages))
    except Exception as e:
        primary = e if isinstance(e, RuntimeError) else RuntimeError(f"PDF extraction failed: {e}")
        try:
            pages = _extract_pdf_fallback(pdf_path)
        except Exception as fb:
            raise RuntimeError(f"{primary}(pypdf 兜底亦失败: {fb})") from e
        logger.warning("opendataloader failed for %s, used pypdf fallback: %s",
                       Path(pdf_path).name, primary)
        return _maybe_ocr(pdf_path, pages)
