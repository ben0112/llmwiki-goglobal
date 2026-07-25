"""Two-API/two-worker smoke test for the self-hosted Compose deployment.

The live test is intentionally opt-in because it requires a running self-hosted
Supabase stack and a real JWT for a seeded user. See ``deploy/.env.selfhost.example``
for the exact ``SCALED_TEST_*`` inputs and run this file only after Compose is up.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import websockets

ROOT = Path(__file__).parents[2]
COMPOSE = ROOT / "deploy/docker-compose.selfhost.yml"
NGINX = ROOT / "deploy/nginx.conf"
LIVE = os.getenv("SCALED_COMPOSE_TEST") == "1"


def _service_block(text: str, service: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(service)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text)
    assert match is not None, f"missing Compose service {service}"
    return match.group(1)


def test_scaled_compose_declares_private_replicas_and_gateway():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "redis:7.4-alpine" in text
    assert '["redis-server", "--appendonly", "yes", "--appendfsync", "everysec"]' in text
    assert "redis-data:/data" in text
    assert 'command: ["arq", "jobs.worker.WorkerSettings"]' in text
    assert "REDIS_URL: redis://redis:6379/0" in text
    assert 'DURABLE_JOBS_ENABLED: "true"' in text
    assert 'TUS_MULTIPART_ENABLED: "true"' in text
    assert "gateway:" in text
    assert '      - "8000:8000"' in text

    api_block = _service_block(text, "api")
    assert 'expose:\n      - "8000"' in api_block
    assert "ports:" not in api_block
    redis_block = _service_block(text, "redis")
    assert "ports:" not in redis_block
    converter_block = _service_block(text, "converter")
    assert "ports:" not in converter_block


def test_gateway_configuration_supports_dynamic_http_and_websocket_proxying():
    text = NGINX.read_text(encoding="utf-8")
    expected = (
        "map $http_upgrade $connection_upgrade",
        "client_max_body_size 65m;",
        "resolver 127.0.0.11 valid=10s ipv6=off;",
        "set $api_backend http://api:8000;",
        "proxy_pass $api_backend;",
        "proxy_set_header Upgrade $http_upgrade;",
        "proxy_set_header Connection $connection_upgrade;",
        "proxy_request_buffering off;",
        "proxy_read_timeout 3600s;",
    )
    for directive in expected:
        assert directive in text


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.fail(f"SCALED_COMPOSE_TEST=1 requires {name}")
    return value


def _mini_pdf() -> bytes:
    stream = b"BT /F1 24 Tf 72 700 Td (Scaled Compose Smoke) Tj ET"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(stream), stream),
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_at = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        output += b"%010d 00000 n \n" % offset
    output += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(output)


def _metadata(**values: str) -> str:
    return ",".join(
        f"{key} {base64.b64encode(value.encode()).decode()}" for key, value in values.items()
    )


async def _wait_for_job(client: httpx.AsyncClient, job_id: str, timeout: float = 240) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        last = response.json()
        if last["state"] == "succeeded":
            return last
        if last["state"] in {"failed", "cancelled", "attempts_exhausted"}:
            pytest.fail(f"job {job_id} ended in {last['state']}: {last.get('error')}")
        await asyncio.sleep(1)
    pytest.fail(f"job {job_id} did not succeed within {timeout}s; last={last}")


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
        "--env-file",
        str(ROOT / "deploy/.env.selfhost"),
        *args,
    ]
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _worker_container_for_owner(owner: str) -> str:
    hostname = owner.split(":", 1)[0]
    container_ids = _compose("ps", "-q", "worker").stdout.split()
    for container_id in container_ids:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.Hostname}}", container_id],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if inspected == hostname:
            return container_id
    pytest.fail(f"no worker container matched lease hostname {hostname!r}")


@pytest.mark.skipif(
    not LIVE,
    reason=(
        "requires the opt-in scaled Compose stack, a self-hosted Supabase database, "
        "and SCALED_TEST_DATABASE_URL/TOKEN/USER_ID"
    ),
)
@pytest.mark.asyncio
async def test_two_api_two_worker_recovery_smoke():
    api_url = os.getenv("SCALED_TEST_API_URL", "http://127.0.0.1:8000").rstrip("/")
    database_url = _required_env("SCALED_TEST_DATABASE_URL")
    token = _required_env("SCALED_TEST_TOKEN")
    user_id = UUID(_required_env("SCALED_TEST_USER_ID"))
    auth_headers = {"Authorization": f"Bearer {token}"}
    kb_id = uuid4()
    filename = f"scaled-{uuid4()}.pdf"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=3)

    try:
        await pool.execute(
            "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
            kb_id,
            user_id,
            "Scaled compose smoke",
            f"scaled-{kb_id}",
        )

        instance_ids = set()
        async with httpx.AsyncClient(base_url=api_url, timeout=20) as client:
            for _ in range(80):
                response = await client.get("/health", headers={"Connection": "close"})
                response.raise_for_status()
                instance_id = response.headers.get("x-api-instance-id")
                assert instance_id, "compose must set STAGE=test for the smoke test"
                instance_ids.add(instance_id)
                if len(instance_ids) == 2:
                    break
        assert len(instance_ids) == 2

        websocket_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(f"{websocket_url}/v1/ws/documents/{kb_id}") as websocket:
            await websocket.send(token)
            event_id = str(uuid4())
            payload = json.dumps(
                {
                    "event": "scaled-smoke",
                    "id": event_id,
                    "user_id": str(user_id),
                    "knowledge_base_id": str(kb_id),
                }
            )
            await asyncio.sleep(0.2)
            await pool.execute("SELECT pg_notify('document_changes', $1)", payload)
            notification = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15))
            assert notification == {"event": "scaled-smoke", "id": event_id}

        pdf = _mini_pdf() + b" " * (5 * 1024 * 1024)
        tus_headers = {
            **auth_headers,
            "Tus-Resumable": "1.0.0",
            "Upload-Length": str(len(pdf)),
            "Upload-Metadata": _metadata(filename=filename, knowledge_base_id=str(kb_id), path="/"),
        }
        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=90) as client:
            created = await client.post("/v1/uploads", headers=tus_headers)
            created.raise_for_status()
            location = created.headers["location"]
            first = pdf[: 5 * 1024 * 1024]
            patched = await client.patch(
                location,
                headers={
                    "Tus-Resumable": "1.0.0",
                    "Upload-Offset": "0",
                    "Content-Type": "application/offset+octet-stream",
                },
                content=first,
            )
            patched.raise_for_status()

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=90) as client:
            resumed = await client.head(location, headers={"Tus-Resumable": "1.0.0"})
            resumed.raise_for_status()
            assert int(resumed.headers["upload-offset"]) == len(first)
            completed = await client.patch(
                location,
                headers={
                    "Tus-Resumable": "1.0.0",
                    "Upload-Offset": str(len(first)),
                    "Content-Type": "application/offset+octet-stream",
                },
                content=pdf[len(first) :],
            )
            completed.raise_for_status()
            document_id = completed.headers["x-document-id"]
            extraction_job_id = completed.headers["x-job-id"]
            await _wait_for_job(client, extraction_job_id)

            graph = await client.post(f"/v1/knowledge-bases/{kb_id}/graph/rebuild")
            graph.raise_for_status()
            await _wait_for_job(client, graph.json()["job_id"])

        lock_connection = await pool.acquire()
        transaction = lock_connection.transaction()
        await transaction.start()
        await lock_connection.execute("LOCK TABLE document_references IN ACCESS EXCLUSIVE MODE")
        try:
            async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
                leased = await client.post(f"/v1/knowledge-bases/{kb_id}/graph/rebuild")
                leased.raise_for_status()
                leased_job_id = UUID(leased.json()["job_id"])
                deadline = time.monotonic() + 40
                owner = None
                while time.monotonic() < deadline:
                    row = await pool.fetchrow(
                        "SELECT state::text, lease_owner FROM background_jobs WHERE id = $1",
                        leased_job_id,
                    )
                    if row and row["state"] == "running":
                        owner = row["lease_owner"]
                        break
                    await asyncio.sleep(0.5)
                assert owner, "test graph job was not leased by either worker"
                container_id = _worker_container_for_owner(owner)
                subprocess.run(["docker", "stop", "--time", "1", container_id], check=True)
        finally:
            await transaction.rollback()
            await pool.release(lock_connection)

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
            recovered = await _wait_for_job(client, str(leased_job_id), timeout=300)
            assert recovered["attempt_count"] >= 2

        duplicate_counts = await pool.fetchrow(
            "SELECT "
            "(SELECT count(*) FROM documents WHERE id = $1) AS documents, "
            "(SELECT count(*) FROM document_pages WHERE document_id = $1) AS pages, "
            "(SELECT count(DISTINCT page) FROM document_pages WHERE document_id = $1) AS unique_pages, "
            "(SELECT count(*) FROM document_chunks WHERE document_id = $1) AS chunks, "
            "(SELECT count(DISTINCT chunk_index) FROM document_chunks WHERE document_id = $1) AS unique_chunks, "
            "(SELECT count(*) FROM document_references WHERE source_document_id = $1) AS refs, "
            "(SELECT count(DISTINCT (source_document_id, target_document_id, reference_type)) "
            " FROM document_references WHERE source_document_id = $1) AS unique_refs",
            UUID(document_id),
        )
        assert duplicate_counts["documents"] == 1
        assert duplicate_counts["pages"] == duplicate_counts["unique_pages"]
        assert duplicate_counts["chunks"] == duplicate_counts["unique_chunks"]
        assert duplicate_counts["refs"] == duplicate_counts["unique_refs"]
    finally:
        await pool.execute("DELETE FROM knowledge_bases WHERE id = $1", kb_id)
        await pool.close()
