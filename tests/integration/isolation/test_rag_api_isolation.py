"""Tenant-isolation and projection contracts for the hosted RAG HTTP API."""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from jobs.service import JobService
from pydantic import SecretStr
from rag.service import RagService

from tests.helpers.jwt import auth_headers, seed_jwks_cache
from tests.integration.isolation.conftest import (
    KB_A_ID,
    KB_B_ID,
    USER_A_ID,
    USER_B_ID,
)


@pytest.fixture(autouse=True)
async def clean_rag_jobs(pool, seed_two_tenants):
    await pool.execute("DELETE FROM background_jobs")
    yield
    await pool.execute("DELETE FROM background_jobs")


@pytest.fixture
def rag_settings():
    profiles = {
        "primary": {
            "base_url": "https://provider-private.invalid/v1",
            "model": "writer-private",
            "timeout_seconds": 60,
            "version": "profile-v1",
        }
    }
    return SimpleNamespace(
        SERVER_RAG_ENABLED=True,
        RAG_MODEL_PROFILES_JSON=SecretStr(json.dumps(profiles)),
        RAG_MODEL_API_KEYS_JSON=SecretStr(json.dumps({"primary": "provider-key-private"})),
        HYBRID_SEARCH_ENABLED=False,
        embedding_profile=None,
        MODE="hosted",
        HYBRID_LEXICAL_CANDIDATES=50,
        HYBRID_VECTOR_CANDIDATES=50,
        HYBRID_RRF_K=60,
        EMBEDDING_BASE_URL="https://embedding-private.invalid/v1",
        EMBEDDING_API_KEY=SecretStr("embedding-key-private"),
        EMBEDDING_BATCH_SIZE=8,
        EMBEDDING_TIMEOUT_SECONDS=30,
    )


@pytest.fixture
async def rag_client(pool, rag_settings, monkeypatch):
    from config import settings
    from main import app

    sentinel = object()
    previous = {
        name: getattr(app.state, name, sentinel)
        for name in ("pool", "mode", "auth_provider", "job_service", "rag_service")
    }
    job_service = JobService(pool)
    app.state.pool = pool
    app.state.mode = "hosted"
    app.state.auth_provider = None
    app.state.job_service = job_service
    app.state.rag_service = RagService(pool, rag_settings, job_service=job_service)
    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(settings, "SERVER_RAG_ENABLED", True)
    seed_jwks_cache()

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        for name, value in previous.items():
            if value is sentinel:
                if hasattr(app.state, name):
                    delattr(app.state, name)
            else:
                setattr(app.state, name, value)


def _create_body(knowledge_base_id=KB_A_ID):
    return {
        "knowledge_base_id": knowledge_base_id,
        "goal": "PRIVATE TENANT GOAL",
        "target_path_prefix": "/wiki/private-tenant/",
        "model_profile": "primary",
    }


async def _create_owner_run(client, *, key="owner-create"):
    response = await client.post(
        "/v1/rag/build-wiki",
        headers={**auth_headers(USER_A_ID), "Idempotency-Key": key},
        json=_create_body(),
    )
    assert response.status_code == 202, response.text
    return response.json()


@pytest.mark.asyncio
async def test_create_and_get_return_only_public_tenant_projection(rag_client):
    accepted = await _create_owner_run(rag_client)

    response = await rag_client.get(
        f"/v1/rag/runs/{accepted['run_id']}",
        headers=auth_headers(USER_A_ID),
    )

    assert response.status_code == 200
    assert response.json()["state"] == "queued"
    assert response.json()["knowledge_base_id"] == KB_A_ID
    for private in (
        "PRIVATE TENANT GOAL",
        "/wiki/private-tenant/",
        "owner-create",
        "provider-private",
        "provider-key-private",
        "embedding-private",
        "embedding-key-private",
        "writer-private",
        "request_digest",
        "user_id",
    ):
        assert private not in response.text

    job = await rag_client.get(accepted["job_url"], headers=auth_headers(USER_A_ID))
    assert job.status_code == 200
    assert job.json()["type"] == "build_wiki"
    for private in ("PRIVATE TENANT GOAL", "/wiki/private-tenant/", "goal", "step", "run_id"):
        assert private not in job.text


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["run", "steps", "resume"])
async def test_other_tenant_and_missing_run_are_indistinguishable(rag_client, resource):
    accepted = await _create_owner_run(rag_client, key=f"isolation-{resource}")
    existing = accepted["run_id"]
    missing = str(uuid4())

    def path(run_id):
        suffix = "/steps" if resource == "steps" else "/resume" if resource == "resume" else ""
        return f"/v1/rag/runs/{run_id}{suffix}"

    kwargs = {"headers": auth_headers(USER_B_ID)}
    if resource == "resume":
        kwargs["headers"] = {**kwargs["headers"], "Idempotency-Key": "tenant-resume"}
        kwargs["json"] = {}
        other = await rag_client.post(path(existing), **kwargs)
        absent = await rag_client.post(path(missing), **kwargs)
    else:
        other = await rag_client.get(path(existing), **kwargs)
        absent = await rag_client.get(path(missing), **kwargs)

    assert other.status_code == absent.status_code == 404
    assert other.json() == absent.json()
    assert "PRIVATE TENANT GOAL" not in other.text


@pytest.mark.asyncio
async def test_cross_tenant_and_missing_knowledge_base_create_are_indistinguishable(rag_client):
    responses = []
    for key, knowledge_base_id in (("foreign-kb", KB_B_ID), ("missing-kb", str(uuid4()))):
        responses.append(
            await rag_client.post(
                "/v1/rag/build-wiki",
                headers={**auth_headers(USER_A_ID), "Idempotency-Key": key},
                json=_create_body(knowledge_base_id),
            )
        )

    assert [response.status_code for response in responses] == [404, 404]
    assert responses[0].json() == responses[1].json()
    assert "PRIVATE TENANT GOAL" not in responses[0].text


@pytest.mark.asyncio
async def test_disabled_rag_rejects_new_work_with_stable_code(rag_client, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "SERVER_RAG_ENABLED", False)
    response = await rag_client.post(
        "/v1/rag/build-wiki",
        headers={**auth_headers(USER_A_ID), "Idempotency-Key": "disabled"},
        json=_create_body(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "rag_disabled", "message": "Server-side RAG is disabled."}}
    assert "PRIVATE TENANT GOAL" not in response.text


@pytest.mark.asyncio
async def test_document_websocket_projection_discards_rag_goal_and_step_content(monkeypatch):
    from routes import ws

    delivered = []

    async def capture(user_id, knowledge_base_id, event):
        delivered.append((user_id, knowledge_base_id, event))

    monkeypatch.setattr(ws.manager, "broadcast", capture)
    await ws._handle_notify(
        json.dumps(
            {
                "event": "UPDATE",
                "id": "public-document-id",
                "user_id": USER_A_ID,
                "knowledge_base_id": KB_A_ID,
                "job_type": "build_wiki",
                "goal": "PRIVATE TENANT GOAL",
                "step": {"prompt": "PRIVATE PROMPT"},
                "path": "/wiki/private-tenant/page.md",
            }
        )
    )

    assert delivered == [
        (
            USER_A_ID,
            KB_A_ID,
            {"event": "UPDATE", "id": "public-document-id"},
        )
    ]
