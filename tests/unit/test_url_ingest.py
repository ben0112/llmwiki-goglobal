"""PDF-by-URL ingestion tests: URL normalization, filename derivation, and the
SSRF/size/magic-byte guards on the download path. No live network — getaddrinfo
and the httpx transport are mocked, matching test_webclip_ssrf.py."""

import socket
from types import SimpleNamespace

import httpx
import infra.safe_fetch as sf
import pytest
import services.url_ingest as ui
from fastapi import HTTPException
from routes import documents as document_routes
from services.url_ingest import UrlIngestService, _derive_filename, _normalize_pdf_url, _sanitize_filename
from starlette.requests import Request
from starlette.responses import Response

_REAL_ASYNC_CLIENT = httpx.AsyncClient

PDF_BYTES = b"%PDF-1.7\n" + b"\x00" * 64


def _gai_return(*ips: str) -> list:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


def _fake_getaddrinfo(mapping: dict[str, list[str]]):
    def _gai(host, *args, **kwargs):
        if host in mapping:
            return _gai_return(*mapping[host])
        raise socket.gaierror(f"unknown host {host}")

    return _gai


def _client_with_transport(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    return factory


def _service() -> UrlIngestService:
    return UrlIngestService(pool=None, s3_service=None, job_service=None, quota_service=None)


def test_url_ingest_accepts_durable_job_service_instead_of_ocr_service():
    job_service = object()
    quota_service = object()

    service = UrlIngestService(
        pool=None,
        s3_service=None,
        job_service=job_service,
        quota_service=quota_service,
    )

    assert service.jobs is job_service
    assert service.quota is quota_service


class TestNormalizePdfUrl:
    def test_arxiv_abs_rewritten_to_pdf(self):
        assert _normalize_pdf_url("https://arxiv.org/abs/2506.06266") == "https://arxiv.org/pdf/2506.06266"
        assert _normalize_pdf_url("https://www.arxiv.org/abs/2506.06266v3") == "https://www.arxiv.org/pdf/2506.06266v3"

    def test_non_arxiv_untouched(self):
        assert _normalize_pdf_url("https://example.com/paper.pdf") == "https://example.com/paper.pdf"

    def test_arxiv_pdf_url_untouched(self):
        assert _normalize_pdf_url("https://arxiv.org/pdf/2506.06266") == "https://arxiv.org/pdf/2506.06266"


async def test_create_from_url_uses_durable_job_service_and_sets_response_header(monkeypatch):
    job_service = object()
    quota_service = object()
    state = SimpleNamespace(
        s3_service=object(),
        job_service=job_service,
        quota_service=quota_service,
        pool=object(),
    )
    app = SimpleNamespace(state=state)
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "app": app})
    response = Response()
    captured = {}

    async def current_user(_request):
        return "00000000-0000-0000-0000-000000000001"

    class FakeUrlIngestService:
        def __init__(self, pool, s3_service, jobs, quota):
            captured["args"] = (pool, s3_service, jobs, quota)

        async def ingest_pdf(self, user_id, kb_id, url, path):
            captured["call"] = (user_id, kb_id, url, path)
            return {
                "id": "00000000-0000-0000-0000-000000000010",
                "filename": "paper.pdf",
                "status": "pending",
                "already_exists": False,
                "job_id": "00000000-0000-0000-0000-000000000020",
            }

    monkeypatch.setattr(document_routes, "get_current_user", current_user)
    monkeypatch.setattr(document_routes.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(document_routes, "UrlIngestService", FakeUrlIngestService)
    body = document_routes.CreateFromUrl(
        knowledge_base_id="00000000-0000-0000-0000-000000000002",
        url="https://example.test/paper.pdf",
    )

    result = await document_routes.create_document_from_url.__wrapped__(request, body, response)

    assert captured["args"] == (state.pool, state.s3_service, job_service, quota_service)
    assert response.headers["X-Job-Id"] == result["job_id"]


def test_cors_exposes_durable_job_header():
    from main import app

    cors = next(middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware")
    assert "X-Job-Id" in cors.kwargs["expose_headers"]


async def test_from_url_uses_legacy_ocr_producer_when_durable_jobs_are_disabled(monkeypatch):
    ocr_service = object()
    state = SimpleNamespace(
        s3_service=object(),
        ocr_service=ocr_service,
        job_service=None,
        pool=object(),
    )
    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": [], "app": SimpleNamespace(state=state)}
    )
    response = Response()

    async def current_user(_request):
        return "00000000-0000-0000-0000-000000000001"

    captured = {}

    class FakeLegacyUrlIngest:
        def __init__(self, pool, s3_service, ocr):
            captured["args"] = (pool, s3_service, ocr)

        async def ingest_pdf(self, user_id, kb_id, url, path):
            captured["call"] = (user_id, kb_id, url, path)
            return {
                "id": "00000000-0000-0000-0000-000000000010",
                "filename": "paper.pdf",
                "status": "pending",
                "already_exists": False,
            }

    monkeypatch.setattr(document_routes, "get_current_user", current_user)
    monkeypatch.setattr(document_routes.settings, "DURABLE_JOBS_ENABLED", False)
    monkeypatch.setattr(document_routes, "_LegacyUrlIngestCompatibility", FakeLegacyUrlIngest)
    body = document_routes.CreateFromUrl(
        knowledge_base_id="00000000-0000-0000-0000-000000000002",
        url="https://example.test/paper.pdf",
    )

    result = await document_routes.create_document_from_url.__wrapped__(request, body, response)

    assert result["status"] == "pending"
    assert "X-Job-Id" not in response.headers
    assert captured["args"] == (state.pool, state.s3_service, ocr_service)


class TestDeriveFilename:
    def test_content_disposition_wins(self):
        resp = httpx.Response(200, headers={"content-disposition": 'inline; filename="2506.06266v3.pdf"'})
        assert _derive_filename(resp, "https://arxiv.org/pdf/2506.06266") == "2506.06266v3.pdf"

    def test_falls_back_to_url_segment(self):
        resp = httpx.Response(200)
        assert _derive_filename(resp, "https://example.com/papers/attention.pdf") == "attention.pdf"

    def test_extensionless_segment_gets_pdf_suffix(self):
        resp = httpx.Response(200)
        assert _derive_filename(resp, "https://arxiv.org/pdf/2506.06266") == "2506.06266.pdf"

    def test_sanitizes_traversal_and_junk(self):
        assert _sanitize_filename("../../etc/passwd") == "passwd.pdf"
        assert _sanitize_filename("") == "document.pdf"
        assert _sanitize_filename("a" * 300 + ".pdf").endswith(".pdf")
        assert len(_sanitize_filename("a" * 300 + ".pdf")) <= 124


class TestDownloadGuards:
    async def test_downloads_pdf_with_pinned_ip(self, monkeypatch):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url_host"] = request.url.host
            captured["host_header"] = request.headers.get("host")
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, content=PDF_BYTES)

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"arxiv.org": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        pdf = await _service()._download("https://arxiv.org/pdf/2506.06266")

        assert pdf.data == PDF_BYTES
        assert pdf.filename == "2506.06266.pdf"
        assert captured["url_host"] == "93.184.216.34"
        assert captured["host_header"] == "arxiv.org"
        assert captured["user_agent"] == ui.USER_AGENT

    async def test_private_host_rejected_without_sending(self, monkeypatch):
        sent = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            sent["count"] += 1
            return httpx.Response(200, content=PDF_BYTES)

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"internal.test": ["10.0.0.5"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://internal.test/doc.pdf")
        assert exc.value.status_code == 400
        assert sent["count"] == 0

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com:444/doc.pdf",
            "https://user:pass@example.com/doc.pdf",
            "https://api.railway.internal/doc.pdf",
        ],
    )
    async def test_unsafe_url_shape_rejected_without_sending(self, monkeypatch, url):
        sent = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            sent["count"] += 1
            return httpx.Response(200, content=PDF_BYTES)

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"example.com": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download(url)
        assert exc.value.status_code == 400
        assert sent["count"] == 0

    async def test_redirect_to_private_host_rejected(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://meta.test/latest"})

        monkeypatch.setattr(
            sf.socket,
            "getaddrinfo",
            _fake_getaddrinfo(
                {
                    "cdn.test": ["93.184.216.34"],
                    "meta.test": ["169.254.169.254"],
                }
            ),
        )
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://cdn.test/doc.pdf")
        assert exc.value.status_code == 400

    async def test_redirect_to_unsafe_port_rejected(self, monkeypatch):
        sent_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent_urls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://cdn.test:444/private.pdf"})

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"cdn.test": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://cdn.test/doc.pdf")
        assert exc.value.status_code == 400
        assert len(sent_urls) == 1

    async def test_redirect_to_internal_hostname_rejected(self, monkeypatch):
        sent_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent_urls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://api.railway.internal/private.pdf"})

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"cdn.test": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://cdn.test/doc.pdf")
        assert exc.value.status_code == 400
        assert len(sent_urls) == 1

    async def test_non_pdf_content_rejected(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"<html>nope</html>")

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"example.com": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://example.com/fake.pdf")
        assert exc.value.status_code == 400
        assert "browser extension" in exc.value.detail

    async def test_oversized_pdf_rejected(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=PDF_BYTES)

        monkeypatch.setattr(ui, "MAX_PDF_BYTES", 16)
        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"example.com": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://example.com/big.pdf")
        assert exc.value.status_code == 413

    async def test_http_error_status_rejected(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=b"not found")

        monkeypatch.setattr(sf.socket, "getaddrinfo", _fake_getaddrinfo({"example.com": ["93.184.216.34"]}))
        monkeypatch.setattr(ui.httpx, "AsyncClient", _client_with_transport(handler))

        with pytest.raises(HTTPException) as exc:
            await _service()._download("https://example.com/gone.pdf")
        assert exc.value.status_code == 400
        assert "404" in exc.value.detail

    async def test_non_http_scheme_rejected(self, monkeypatch):
        with pytest.raises(HTTPException) as exc:
            await _service()._download("file:///etc/passwd")
        assert exc.value.status_code == 400
