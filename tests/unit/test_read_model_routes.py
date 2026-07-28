from deps import get_read_service
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routes.read_models import router
from services.read_local import StaleReadCursor

from llmwiki_core.read_cursor import CursorError


class FakeReadService:
    def __init__(self):
        self.browse_calls = 0
        self.corpus_calls = 0
        self.current_revision = 7
        self.stale = False

    async def revision(self, kb_id):
        return self.current_revision

    async def browse(self, *args, **kwargs):
        self.browse_calls += 1
        if kwargs.get("cursor") == "malformed":
            raise CursorError("private decoder detail")
        if self.stale:
            raise StaleReadCursor(self.current_revision)
        return {
            "revision": self.current_revision,
            "items": [],
            "folders": [],
            "next_cursor": None,
            "total_count": 0,
        }

    async def upload_preflight(self, kb_id, descriptors):
        return {"revision": self.current_revision, "items": descriptors}

    async def corpus_entries(self, *args, **kwargs):
        self.corpus_calls += 1
        return {
            "revision": self.current_revision,
            "items": [],
            "next_cursor": None,
            "total_count": 0,
        }

    async def corpus_summary(self, *args, **kwargs):
        self.corpus_calls += 1
        return {
            "revision": self.current_revision,
            "total_count": 0,
            "filtered_count": 0,
            "facets": {},
            "coverage": {},
            "business_classes": {},
            "business_scenes": {},
            "kpis": {},
        }


def _client(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_read_service] = lambda: service
    return TestClient(app)


def test_browse_returns_quoted_etag_and_304_before_adapter_data_read():
    service = FakeReadService()
    client = _client(service)
    url = "/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/documents/browse"
    response = client.get(url, params={"path": "/"})
    assert response.status_code == 200
    assert response.headers["etag"] == '"kb-read-7"'

    response = client.get(url, params={"path": "/"}, headers={"If-None-Match": '"kb-read-7"'})
    assert response.status_code == 304
    assert service.browse_calls == 1


def test_stale_cursor_and_invalid_cursor_have_stable_errors():
    service = FakeReadService()
    service.stale = True
    client = _client(service)
    url = "/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/documents/browse"
    response = client.get(url, params={"path": "/", "cursor": "valid-to-fake"})
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "stale_cursor", "revision": 7}}

    service.stale = False
    response = client.get(url, params={"path": "/", "cursor": "malformed"})
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "invalid_cursor"}}
    assert "private decoder detail" not in response.text


def test_upload_preflight_rejects_more_than_200_descriptors():
    client = _client(FakeReadService())
    url = "/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/documents/upload-preflight"
    items = [{"path": "/", "filename": f"{index}.md", "size": 1} for index in range(201)]
    response = client.post(url, json={"items": items})
    assert response.status_code == 422


def test_corpus_routes_use_etag_and_short_circuit_before_summary_reads():
    service = FakeReadService()
    client = _client(service)
    root = "/v1/knowledge-bases/00000000-0000-0000-0000-000000000001/corpus"

    response = client.get(
        f"{root}/entries",
        params={"stage": "S2", "query": "permit", "sort": "domain"},
    )
    assert response.status_code == 200
    assert response.headers["etag"] == '"kb-read-7"'

    response = client.get(
        f"{root}/summary",
        headers={"If-None-Match": 'W/"kb-read-7"'},
    )
    assert response.status_code == 304
    assert service.corpus_calls == 1
