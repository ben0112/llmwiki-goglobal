import os
import sys
import hmac
import json
import signal
import asyncio
import logging
import subprocess
import tempfile
import contextlib
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from collections.abc import Iterator

import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supavault Converter")

OFFICE_EXTENSIONS = {"pptx", "ppt", "docx", "doc"}
PDF_EXTENSIONS = {"pdf"}
SUPPORTED_EXTENSIONS = OFFICE_EXTENSIONS | PDF_EXTENSIONS
CONVERT_TIMEOUT = 120  # LibreOffice subprocess timeout (seconds)
EXTRACT_TIMEOUT = 180  # opendataloader extraction timeout (seconds)
MAX_SOURCE_BYTES = 200 * 1024 * 1024  # 200 MB — hard cap on downloaded file size
# Defense-in-depth cap on the page-assembly loop. `number of pages` comes from
# the untrusted PDF; a crafted /Count would otherwise drive an unbounded loop
# that runs after the subprocess timeout no longer applies.
MAX_PAGES = 10_000
# Concurrent /extract jobs allowed at once. Each can spawn LibreOffice + a JVM,
# so unbounded concurrency OOMs the container. Tune per container memory.
MAX_CONCURRENT_EXTRACTIONS = int(os.environ.get("MAX_CONCURRENT_EXTRACTIONS", "2"))

# Extraction runs opendataloader in a child process group (not the in-process
# library call) so a timeout can kill the JVM instead of orphaning it.
_EXTRACT_SCRIPT = (
    "import sys, opendataloader_pdf\n"
    "opendataloader_pdf.convert(input_path=sys.argv[1], output_dir=sys.argv[2], format='json', quiet=True)\n"
)

_active_extractions = 0

CONVERTER_SECRET = os.environ.get("CONVERTER_SECRET", "")
# Bucket name from env — used to lock the S3 URL allowlist to OUR bucket
# rather than any `*.amazonaws.com` URL (which would let any S3-hosted file
# be processed by this service).
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Self-hosting: the S3-compatible endpoint presigned URLs point at
# (e.g. "http://minio:9000" or "https://s3.example.internal"). When set, the
# allowlist accepts that host instead of *.amazonaws.com — still locked to
# S3_BUCKET. Matches S3_ENDPOINT_URL on the API service.
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", os.environ.get("S3_ENDPOINT_URL", ""))

# Refuse to start in "public mode". Setting CONVERTER_SECRET is mandatory.
if not CONVERTER_SECRET:
    raise RuntimeError(
        "CONVERTER_SECRET environment variable is required. "
        "Generate a random string and set it on both the API and converter services."
    )


class ExtractRequest(BaseModel):
    source_url: str
    source_ext: str
    request_id: str | None = None


@contextlib.contextmanager
def _extraction_slot() -> Iterator[None]:
    """Admit one extraction; reject with 503 once the container is at capacity."""
    global _active_extractions
    if _active_extractions >= MAX_CONCURRENT_EXTRACTIONS:
        logger.warning("extract rejected: at capacity (%d concurrent)", _active_extractions)
        raise HTTPException(503, "Converter at capacity, retry shortly")
    _active_extractions += 1
    try:
        yield
    finally:
        _active_extractions -= 1


def _run_in_process_group(command: list[str], timeout: int, what: str) -> None:
    """Run a command in its own process group; kill the whole group on timeout."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise TimeoutError(f"{what} timed out after {timeout}s")
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: {stderr.decode(errors='replace')[:500]}")


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL an entire process group, then reap the leader."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _validate_s3_url(url: str) -> None:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise HTTPException(400, "URL has no hostname")
    if S3_ENDPOINT:
        # Self-hosted mode: lock to the configured endpoint (host + port),
        # and to the bucket when one is configured. Accept path style
        # (`endpoint/{bucket}/...`) and virtual-host (`{bucket}.endpoint`).
        ep = urlparse(S3_ENDPOINT if "://" in S3_ENDPOINT else f"//{S3_ENDPOINT}")
        host_ok = parsed.hostname == ep.hostname or (
            S3_BUCKET and parsed.hostname == f"{S3_BUCKET}.{ep.hostname}"
        )
        port_ok = ep.port is None or parsed.port == ep.port
        if not (host_ok and port_ok):
            raise HTTPException(400, "URL does not point to the configured S3 endpoint")
        if S3_BUCKET and parsed.hostname == ep.hostname \
                and not parsed.path.lstrip("/").startswith(f"{S3_BUCKET}/"):
            raise HTTPException(400, "URL does not point to the configured S3 bucket")
        return
    if not S3_BUCKET:
        # Fall back to broad check if bucket isn't configured. Less safe but
        # the service still works for self-hosters.
        if not parsed.hostname.endswith(".amazonaws.com"):
            raise HTTPException(400, "URLs must point to S3")
        return
    # Strict: only allow URLs pointing at our specific bucket. Accept both
    # virtual-host style (`{bucket}.s3.{region}.amazonaws.com`) and path
    # style (`s3.{region}.amazonaws.com/{bucket}/...`).
    expected_vhost = f"{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
    expected_vhost_global = f"{S3_BUCKET}.s3.amazonaws.com"
    vhost_ok = parsed.hostname in (expected_vhost, expected_vhost_global)
    path_ok = (
        parsed.hostname in (f"s3.{S3_REGION}.amazonaws.com", "s3.amazonaws.com")
        and parsed.path.lstrip("/").startswith(f"{S3_BUCKET}/")
    )
    if not (vhost_ok or path_ok):
        raise HTTPException(400, "URL does not point to the configured S3 bucket")


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
        src = el.get("source", "")
        return f"![image]({src})" if src else ""

    if t == "caption":
        return f"*{content}*" if content else ""

    return ""


def _extract_pages(pdf_path: str, output_dir: str) -> list[dict]:
    """Run opendataloader-pdf with JSON output and return per-page markdown."""
    _run_in_process_group(
        [sys.executable, "-c", _EXTRACT_SCRIPT, pdf_path, output_dir],
        timeout=EXTRACT_TIMEOUT,
        what="opendataloader",
    )

    json_files = list(Path(output_dir).glob("*.json"))
    if not json_files:
        raise RuntimeError("opendataloader-pdf produced no output")

    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)

    total_pages = data.get("number of pages", 0)
    if not isinstance(total_pages, int) or total_pages < 0:
        total_pages = 0
    total_pages = min(total_pages, MAX_PAGES)
    elements = data.get("kids", [])
    page_elements: dict[int, list[dict]] = defaultdict(list)

    for el in elements:
        page_num = el.get("page number")
        if page_num is None or el.get("type") in ("header", "footer"):
            continue
        page_elements[page_num].append(el)

    pages = []
    for page_num in range(1, total_pages + 1):
        parts = []
        for el in page_elements.get(page_num, []):
            md = _element_to_markdown(el)
            if md:
                parts.append(md)
        pages.append({"page": page_num, "content": "\n\n".join(parts)})

    return pages


def _convert_to_pdf(source_path: Path, tmpdir: str) -> Path:
    """Convert an Office file to PDF via a private, killable LibreOffice instance."""
    # A per-conversion UserInstallation profile avoids the headless profile-lock
    # contention that leaves zombie soffice/Java processes under concurrency.
    profile = f"file://{tmpdir}/lo-profile"
    _run_in_process_group(
        [
            "libreoffice", f"-env:UserInstallation={profile}",
            "--headless", "--norestore", "--nofirststartwizard",
            "--convert-to", "pdf", "--outdir", tmpdir, str(source_path),
        ],
        timeout=CONVERT_TIMEOUT,
        what="LibreOffice",
    )
    pdf_path = Path(tmpdir) / f"{source_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("LibreOffice did not produce a PDF")
    return pdf_path


async def _download_source(url: str, dest: Path, request_id: str) -> None:
    """Stream an S3 source file to dest, bounded by MAX_SOURCE_BYTES."""
    host = urlparse(url).hostname
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(64 * 1024):
                        total += len(chunk)
                        if total > MAX_SOURCE_BYTES:
                            raise HTTPException(413, f"Source file exceeds {MAX_SOURCE_BYTES} bytes")
                        f.write(chunk)
    except httpx.HTTPStatusError as e:
        logger.error("source download rejected: status=%s host=%s request_id=%s",
                     e.response.status_code, host, request_id)
        raise HTTPException(502, f"Could not fetch source (HTTP {e.response.status_code})")
    except httpx.HTTPError as e:
        logger.error("source download error: %s host=%s request_id=%s",
                     type(e).__name__, host, request_id)
        raise HTTPException(502, "Could not fetch source file")


async def _to_pdf(source_path: Path, ext: str, tmpdir: str, request_id: str) -> Path:
    """Return a PDF path — converting via LibreOffice when the source is an Office file."""
    if ext not in OFFICE_EXTENSIONS:
        return source_path
    try:
        return await asyncio.to_thread(_convert_to_pdf, source_path, tmpdir)
    except TimeoutError:
        logger.error("office->pdf conversion timed out after %ds request_id=%s", CONVERT_TIMEOUT, request_id)
        raise HTTPException(504, "Office conversion timed out")
    except RuntimeError as e:
        logger.error("office->pdf conversion failed: %s request_id=%s", e, request_id)
        raise HTTPException(500, "Office conversion failed")


async def _run_extraction(pdf_path: Path, tmpdir: str, request_id: str) -> list[dict]:
    """Run opendataloader extraction; the worker subprocess enforces the wall-clock bound."""
    extract_dir = Path(tmpdir) / "extract"
    extract_dir.mkdir()
    try:
        return await asyncio.to_thread(_extract_pages, str(pdf_path), str(extract_dir))
    except TimeoutError:
        logger.error("pdf extraction timed out after %ds request_id=%s", EXTRACT_TIMEOUT, request_id)
        raise HTTPException(504, f"PDF extraction exceeded {EXTRACT_TIMEOUT}s timeout")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    req: ExtractRequest,
    authorization: str = Header(default=""),
):
    """Extract markdown pages from PDF or Office files.

    For Office files, converts to PDF first via LibreOffice.
    Returns per-page markdown content.
    """
    if CONVERTER_SECRET:
        expected = f"Bearer {CONVERTER_SECRET}"
        if not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "Unauthorized")

    ext = req.source_ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported extension: {ext}")

    _validate_s3_url(req.source_url)

    request_id = req.request_id or "none"
    with _extraction_slot():
        logger.info("extract start: ext=%s active=%d request_id=%s", ext, _active_extractions, request_id)
        try:
            with tempfile.TemporaryDirectory(dir="/tmp/conversions") as tmpdir:
                source_path = Path(tmpdir) / f"source.{ext}"
                await _download_source(req.source_url, source_path, request_id)
                pdf_path = await _to_pdf(source_path, ext, tmpdir, request_id)
                pages = await _run_extraction(pdf_path, tmpdir, request_id)
        except HTTPException:
            raise
        except Exception:
            logger.exception("extract failed: ext=%s request_id=%s", ext, request_id)
            raise HTTPException(500, "Extraction failed")

    page_count = len(pages)
    logger.info("extract done: ext=%s pages=%d request_id=%s", ext, page_count, request_id)
    response = {"pages": pages, "page_count": page_count}
    if req.request_id:
        response["request_id"] = req.request_id
    return response
