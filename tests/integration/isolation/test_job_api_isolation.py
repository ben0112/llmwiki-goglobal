"""Tenant-isolation contracts for the durable job HTTP API."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from jobs.service import JobService

from tests.helpers.jwt import auth_headers, seed_jwks_cache
from tests.integration.isolation.conftest import USER_A_ID, USER_B_ID


@pytest.fixture(autouse=True)
async def clean_background_jobs(pool, seed_two_tenants):
    """Keep jobs from holding tenant rows across isolation tests."""
    await pool.execute("DELETE FROM background_jobs")
    yield
    await pool.execute("DELETE FROM background_jobs")


@pytest.fixture
async def job_client(pool, monkeypatch):
    from config import settings
    from main import app

    previous_service = getattr(app.state, "job_service", None)
    had_service = hasattr(app.state, "job_service")
    previous_error = getattr(app.state, "job_service_error", None)
    had_error = hasattr(app.state, "job_service_error")

    app.state.pool = pool
    app.state.auth_provider = None
    app.state.job_service = JobService(pool)
    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", True)
    seed_jwks_cache()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        if had_service:
            app.state.job_service = previous_service
        else:
            del app.state.job_service
        if had_error:
            app.state.job_service_error = previous_error
        elif hasattr(app.state, "job_service_error"):
            del app.state.job_service_error


async def _insert_job(pool, user_id: str, *, state: str = "queued", **values) -> UUID:
    columns = ["user_id", "job_type", "state", *values]
    parameters = [user_id, "document.extract", state, *values.values()]
    placeholders = [f"${index}" for index in range(1, len(parameters) + 1)]
    json_columns = {"payload", "progress", "result"}
    casts = ["::jsonb" if column in json_columns else "" for column in columns]
    return await pool.fetchval(
        "INSERT INTO background_jobs ("
        + ", ".join(columns)
        + ") VALUES ("
        + ", ".join(placeholder + cast for placeholder, cast in zip(placeholders, casts, strict=True))
        + ") RETURNING id",
        *parameters,
    )


@pytest.mark.asyncio
async def test_owner_gets_only_public_job_projection_with_allowlisted_error(job_client, pool):
    job_id = await _insert_job(
        pool,
        USER_A_ID,
        state="failed",
        payload='{"private_prompt":"never expose"}',
        progress='{"percent":75}',
        result='{"document_id":"public-result"}',
        attempt_count=2,
        lease_owner="worker-private-name",
        last_dispatched_at=datetime(2026, 1, 2, 3, tzinfo=UTC),
        dispatch_attempts=9,
        error_code="document_not_found",
        error_message="stack: password=super-secret",
    )

    response = await job_client.get(f"/v1/jobs/{job_id}", headers=auth_headers(USER_A_ID))

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "id": str(job_id),
        "type": "document.extract",
        "state": "failed",
        "progress": {"percent": 75},
        "result": {"document_id": "public-result"},
        "attempt_count": 2,
        "max_attempts": 3,
        "cancel_requested": False,
        "error": {
            "code": "document_not_found",
            "message": "The requested document was not found.",
        },
        "created_at": body["created_at"],
        "updated_at": body["updated_at"],
    }
    serialized = response.text
    for private_value in (
        "private_prompt",
        "never expose",
        "lease_owner",
        "worker-private-name",
        "last_dispatched_at",
        "dispatch_attempts",
        "password",
        "super-secret",
        "stack",
    ):
        assert private_value not in serialized


@pytest.mark.asyncio
async def test_unknown_internal_error_is_replaced_with_generic_public_error(job_client, pool):
    job_id = await _insert_job(
        pool,
        USER_A_ID,
        state="failed",
        error_code="postgres_deadlock",
        error_message="relation private_table does not exist",
    )

    response = await job_client.get(f"/v1/jobs/{job_id}", headers=auth_headers(USER_A_ID))

    assert response.status_code == 200
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "The job could not be completed.",
    }
    assert "postgres_deadlock" not in response.text
    assert "private_table" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_other_tenant_and_missing_job_are_indistinguishable(method, job_client, pool):
    job_id = await _insert_job(pool, USER_A_ID, payload='{"tenant_secret":true}')
    missing_id = uuid4()

    other_tenant = await job_client.request(
        method,
        f"/v1/jobs/{job_id}" + ("/cancel" if method == "POST" else ""),
        headers=auth_headers(USER_B_ID),
    )
    missing = await job_client.request(
        method,
        f"/v1/jobs/{missing_id}" + ("/cancel" if method == "POST" else ""),
        headers=auth_headers(USER_B_ID),
    )

    assert other_tenant.status_code == missing.status_code == 404
    assert other_tenant.json() == missing.json()
    assert "tenant_secret" not in other_tenant.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/v1/jobs/aaaaaaaa-0000-0000-0000-000000000001",
        "/v1/jobs/aaaaaaaa-0000-0000-0000-000000000001/cancel",
    ],
)
async def test_job_routes_require_authentication(path, job_client):
    response = await job_client.request("POST" if path.endswith("/cancel") else "GET", path)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_non_uuid_authenticated_subject_is_rejected_before_service_call(job_client, pool):
    job_id = await _insert_job(pool, USER_A_ID)

    response = await job_client.get(f"/v1/jobs/{job_id}", headers=auth_headers("not-a-uuid"))

    assert response.status_code == 401
    assert "UUID" not in response.text
    assert "not-a-uuid" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_state", ["queued", "retry_wait"])
async def test_pending_job_cancellation_is_immediate_and_idempotent(initial_state, job_client, pool):
    job_id = await _insert_job(pool, USER_A_ID, state=initial_state)
    url = f"/v1/jobs/{job_id}/cancel"

    first = await job_client.post(url, headers=auth_headers(USER_A_ID))
    second = await job_client.post(url, headers=auth_headers(USER_A_ID))

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["state"] == "cancelled"
    assert first.json()["cancel_requested"] is True
    row = await pool.fetchrow("SELECT state, cancel_requested_at FROM background_jobs WHERE id = $1", job_id)
    assert row["state"] == "cancelled"
    assert row["cancel_requested_at"] is not None


@pytest.mark.asyncio
async def test_running_job_cancellation_requests_stop_without_rewriting_state(job_client, pool):
    job_id = await _insert_job(
        pool,
        USER_A_ID,
        state="running",
        attempt_count=1,
        lease_owner="worker-a",
        lease_expires_at=datetime(2030, 1, 2, 3, tzinfo=UTC),
    )
    url = f"/v1/jobs/{job_id}/cancel"

    first = await job_client.post(url, headers=auth_headers(USER_A_ID))
    second = await job_client.post(url, headers=auth_headers(USER_A_ID))

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["state"] == "running"
    assert first.json()["cancel_requested"] is True
    assert "worker-a" not in first.text
    assert await pool.fetchval("SELECT state FROM background_jobs WHERE id = $1", job_id) == "running"


@pytest.mark.asyncio
async def test_enabled_route_fails_closed_when_job_service_is_unavailable(job_client):
    from main import app

    app.state.job_service = None
    secret = "postgresql://admin:password@private-db/jobs"
    app.state.job_service_error = RuntimeError(secret)

    response = await job_client.get(f"/v1/jobs/{uuid4()}", headers=auth_headers(USER_A_ID))

    assert response.status_code == 503
    assert secret not in response.text
    assert "password" not in response.text
    assert "RuntimeError" not in response.text


@pytest.mark.asyncio
async def test_unavailable_service_does_not_preempt_authentication(job_client):
    from main import app

    app.state.job_service = None

    response = await job_client.get(f"/v1/jobs/{uuid4()}")

    assert response.status_code == 401


def test_local_configuration_does_not_register_or_import_job_router():
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(
        {
            "MODE": "local",
            "PYTHONPATH": str(root / "api"),
            "DURABLE_JOBS_ENABLED": "false",
            "TUS_MULTIPART_ENABLED": "false",
            "REDIS_URL": "",
        }
    )
    probe = (
        "import sys; import main; "
        "paths={r.path for r in main.app.routes}; "
        "assert '/v1/jobs/{job_id}' not in paths; "
        "assert 'routes.jobs' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
