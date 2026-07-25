"""Two-API/two-worker smoke test for the self-hosted Compose deployment.

The live test is intentionally opt-in because it requires a running self-hosted
Supabase stack and a real JWT for a seeded user. See ``deploy/.env.selfhost.example``
for the exact ``SCALED_TEST_*`` inputs and run this file only after Compose is up.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import subprocess
import time
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import websockets
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).parents[2]
COMPOSE = ROOT / "deploy/docker-compose.selfhost.yml"
NGINX = ROOT / "deploy/nginx.conf"
DOCS = ROOT / "docs/self-hosting.md"
SELFHOST_ENV = ROOT / "deploy/.env.selfhost"
LIVE = os.getenv("SCALED_COMPOSE_TEST") == "1"


def _service_block(text: str, service: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(service)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text)
    assert match is not None, f"missing Compose service {service}"
    return match.group(1)


def test_scaled_compose_declares_private_replicas_and_gateway():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "redis:7.4.2-alpine" in text
    assert "Redis 7.4.2" in DOCS.read_text(encoding="utf-8")
    assert '["redis-server", "--appendonly", "yes", "--appendfsync", "everysec"]' in text
    assert "redis-data:/data" in text
    assert 'command: ["arq", "jobs.worker.WorkerSettings"]' in text
    assert "REDIS_URL: redis://redis:6379/0" in text
    assert "DURABLE_JOBS_ENABLED: ${DURABLE_JOBS_ENABLED:-true}" in text
    assert "TUS_MULTIPART_ENABLED: ${TUS_MULTIPART_ENABLED:-true}" in text
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


def test_self_host_docs_make_rollback_and_scaled_smoke_commands_executable():
    text = DOCS.read_text(encoding="utf-8")
    rollback_section = text.split("Copyable incident rollback", 1)[1].split("### Security notes", 1)[0]
    rollback_block = rollback_section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    assert rollback_block.startswith("(\n  set -e\n")
    assert rollback_block.endswith("\n)")
    assert "ROLLBACK_REF" in rollback_block
    assert 'git archive "$ROLLBACK_REF"' in rollback_block
    assert "DURABLE_JOBS_ENABLED=false" not in rollback_block
    assert "TUS_MULTIPART_ENABLED=false" not in rollback_block
    assert "--scale worker=1" in rollback_block
    assert "--scale api=1" in rollback_block
    assert "--force-recreate" in rollback_block
    assert "api worker gateway" in rollback_block
    assert "trap cleanup EXIT" in text
    assert "SCALED_COMPOSE_TEST=1" in text
    assert "safely reads `deploy/.env.selfhost`" in text
    assert "does not send `CONVERTER_SECRET`" in text
    assert "cannot detect a converter-secret mismatch" in text
    assert "current Postgres LISTEN subscription" in text
    scaling = text.split("## Scaling verification", 1)[1]
    assert "```bash\n(\n  set -e" in scaling
    assert "trap cleanup EXIT" in scaling
    assert "\n)\n```" in scaling


def test_exported_rollback_flags_override_env_file_for_api_and_worker():
    environment = {
        **os.environ,
        "DURABLE_JOBS_ENABLED": "false",
        "TUS_MULTIPART_ENABLED": "false",
    }
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE),
            "--env-file",
            str(ROOT / "deploy/.env.selfhost.example"),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    rendered = json.loads(result.stdout)
    for service in ("api", "worker"):
        assert rendered["services"][service]["environment"]["DURABLE_JOBS_ENABLED"] == "false"
        assert rendered["services"][service]["environment"]["TUS_MULTIPART_ENABLED"] == "false"


def test_live_scaled_smoke_exercises_cross_replica_and_sigkill_recovery_paths():
    source = inspect.getsource(test_two_api_two_worker_recovery_smoke)
    assert 'subprocess.run(["docker", "kill", "--signal", "KILL"' in source
    assert "websocket.response.headers" in source
    assert "len(websockets_by_instance) == 2" in source
    assert "tus_instances" in source
    assert "first_patch_instance" in source
    assert "resume_instance != first_patch_instance" in source
    assert "await asyncio.sleep(0.2)" not in source
    assert "lock_connection = None" in source
    assert '"--scale", "worker=2", "worker"' in source
    assert "timeout=240" in source
    assert 'recovered["attempt_count"] >= 2' in source
    assert "_restart_redis_and_wait()" in source
    assert "_api_container_for_instance(completed_instance)" in source
    assert 'subprocess.run(["docker", "kill", "--signal", "TERM"' in source
    assert "old_owner" in source
    assert "graceful" in source
    term_at = source.index('subprocess.run(["docker", "kill", "--signal", "TERM"')
    next_job_at = source.index("draining_response = await client.post", term_at)
    guarded_exit_at = source.index("await _wait_for_exit_without_old_owner_claim", next_job_at)
    assert term_at < next_job_at < guarded_exit_at
    assert '_container_state(graceful_worker_container)["Running"]' in source
    assert '"durable worker resources closed" in graceful_logs' in source
    guard_source = inspect.getsource(_wait_for_exit_without_old_owner_claim)
    assert 'last_job["lease_owner"] != old_owner' in guard_source
    assert "await asyncio.sleep(0.05)" in guard_source


def test_redis_restart_requires_a_local_waitaof_fsync(monkeypatch):
    waitaof_stdout = ["0\n0\n"]
    calls = []

    def fake_compose(*args, check=True):
        calls.append(args)
        if "WAITAOF" in args:
            return subprocess.CompletedProcess(args, 0, waitaof_stdout[0], "")
        if args[-1] == "ping":
            return subprocess.CompletedProcess(args, 0, "PONG\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setitem(_restart_redis_and_wait.__globals__, "_compose", fake_compose)

    with pytest.raises(AssertionError, match="local AOF fsync"):
        _restart_redis_and_wait()
    assert calls == [("exec", "-T", "redis", "redis-cli", "WAITAOF", "1", "0", "5000")]

    calls.clear()
    waitaof_stdout[0] = "1\n0\n"
    _restart_redis_and_wait()
    assert calls == [
        ("exec", "-T", "redis", "redis-cli", "WAITAOF", "1", "0", "5000"),
        ("restart", "redis"),
        ("exec", "-T", "redis", "redis-cli", "ping"),
    ]


def test_scaled_pdf_fixture_is_a_complete_multipart_sized_document():
    pdf = _mini_pdf()
    assert len(pdf) > 5 * 1024 * 1024
    assert pdf.rstrip().endswith(b"%%EOF")
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) == 1
    assert "Scaled Compose Smoke" in reader.pages[0].extract_text()


def test_compose_recovery_reuses_explicit_scaled_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "scaled.env"
    env_file.write_text(
        "DATABASE_URL=postgresql://postgres@ci-postgres/postgres\n"
        "AWS_SECRET_ACCESS_KEY=scaled-secret\n"
    )
    monkeypatch.setenv("SCALED_COMPOSE_ENV_FILE", str(env_file))
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres@localhost:5434/postgres")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    captured = []

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _compose("up", "-d", "--no-deps", "--scale", "worker=2", "worker")

    assert len(captured) == 1
    command, kwargs = captured[0]
    assert command == [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
        "--env-file",
        str(env_file),
        "up",
        "-d",
        "--no-deps",
        "--scale",
        "worker=2",
        "worker",
    ]
    child_env = kwargs.pop("env")
    assert kwargs == {"check": True, "text": True, "capture_output": True}
    assert "DATABASE_URL" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert child_env["PATH"] == os.environ["PATH"]


def test_compose_command_omits_missing_default_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("SCALED_COMPOSE_ENV_FILE", raising=False)
    monkeypatch.setitem(_compose.__globals__, "SELFHOST_ENV", tmp_path / "missing.env")
    captured = []

    def fake_run(command, **kwargs):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _compose("ps", "-q", "worker")

    assert len(captured) == 1
    assert "--env-file" not in captured[0]


def _selfhost_env_value(name: str) -> str:
    try:
        lines = SELFHOST_ENV.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = stripped.partition("=")
        if separator and key.strip() == name:
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value
    return ""


def _env_value(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or _selfhost_env_value(name) or default


def _required_env(name: str) -> str:
    value = _env_value(name)
    if not value:
        pytest.fail(f"SCALED_COMPOSE_TEST=1 requires {name}")
    return value


def _mini_pdf() -> bytes:
    stream = b"BT /F1 24 Tf 72 700 Td (Scaled Compose Smoke) Tj ET"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
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
    # Put the multipart-sizing bytes inside a standards-compliant embedded
    # stream. Appending bytes after %%EOF is tolerated by pypdf but rejected
    # by the real OpenDataLoader/PDFBox converter exercised in scaled CI.
    writer = PdfWriter(clone_from=BytesIO(bytes(output)))
    writer.add_attachment("multipart-padding.bin", b"0" * (5 * 1024 * 1024))
    padded = BytesIO()
    writer.write(padded)
    return padded.getvalue()


def _metadata(**values: str) -> str:
    return ",".join(f"{key} {base64.b64encode(value.encode()).decode()}" for key, value in values.items())


async def _wait_for_job(
    client: httpx.AsyncClient,
    job_id: str,
    timeout: float = 240,
    *,
    forbidden_instance: str | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"/v1/jobs/{job_id}", headers={"Connection": "close"})
        except httpx.TransportError:
            await asyncio.sleep(0.5)
            continue
        if response.status_code >= 500:
            await asyncio.sleep(0.5)
            continue
        response.raise_for_status()
        if forbidden_instance is not None:
            responding_instance = response.headers.get("x-api-instance-id")
            assert responding_instance and responding_instance != forbidden_instance
        last = response.json()
        if last["state"] == "succeeded":
            return last
        if last["state"] in {"failed", "cancelled", "attempts_exhausted"}:
            pytest.fail(f"job {job_id} ended in {last['state']}: {last.get('error')}")
        await asyncio.sleep(1)
    pytest.fail(f"job {job_id} did not succeed within {timeout}s; last={last}")


async def _wait_for_running_job(pool, job_id: UUID, timeout: float = 40):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await pool.fetchrow(
            "SELECT state::text, lease_owner, lease_expires_at, attempt_count "
            "FROM background_jobs WHERE id = $1",
            job_id,
        )
        if last and last["state"] == "running" and last["lease_owner"]:
            return last
        await asyncio.sleep(0.25)
    pytest.fail(f"job {job_id} was not claimed within {timeout}s; last={dict(last) if last else None}")


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
    ]
    configured_env_file = os.getenv("SCALED_COMPOSE_ENV_FILE", "").strip()
    env_file = Path(configured_env_file) if configured_env_file else SELFHOST_ENV
    if not env_file.is_absolute():
        env_file = ROOT / env_file
    run_kwargs = {"check": check, "text": True, "capture_output": True}
    if configured_env_file or env_file.is_file():
        command.extend(("--env-file", str(env_file)))
        child_env = os.environ.copy()
        try:
            env_lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            env_lines = ()
        for line in env_lines:
            assignment = line.strip()
            if not assignment or assignment.startswith("#"):
                continue
            if assignment.startswith("export "):
                assignment = assignment.removeprefix("export ").lstrip()
            key, separator, _ = assignment.partition("=")
            key = key.strip()
            if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                child_env.pop(key, None)
        run_kwargs["env"] = child_env
    command.extend(args)
    return subprocess.run(command, **run_kwargs)


def _container_for_instance(service: str, instance_id: str) -> str:
    hostname = instance_id.split(":", 1)[0]
    container_ids = _compose("ps", "-q", service).stdout.split()
    for container_id in container_ids:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.Hostname}}", container_id],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if inspected == hostname:
            return container_id
    pytest.fail(f"no {service} container matched instance hostname {hostname!r}")


def _worker_container_for_owner(owner: str) -> str:
    return _container_for_instance("worker", owner)


def _api_container_for_instance(instance_id: str) -> str:
    return _container_for_instance("api", instance_id)


def _restart_redis_and_wait(timeout: float = 40) -> None:
    # Force the first PATCH state through Redis 7.4's local AOF before the
    # restart so this verifies persistence rather than a graceful memory copy.
    waitaof = _compose("exec", "-T", "redis", "redis-cli", "WAITAOF", "1", "0", "5000")
    response_lines = [line.strip() for line in waitaof.stdout.splitlines() if line.strip()]
    try:
        local_fsync_count = int(response_lines[0])
    except (IndexError, ValueError):
        pytest.fail("Redis WAITAOF returned an invalid local fsync count")
    assert local_fsync_count >= 1, (
        f"Redis WAITAOF local AOF fsync count must be at least 1; got {local_fsync_count}"
    )
    _compose("restart", "redis")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _compose("exec", "-T", "redis", "redis-cli", "ping", check=False)
        if last.returncode == 0 and last.stdout.strip() == "PONG":
            return
        time.sleep(0.25)
    pytest.fail(f"Redis did not become ready after restart; returncode={getattr(last, 'returncode', None)}")


def _disable_restart_and_kill(container_id: str, signal: str) -> None:
    subprocess.run(["docker", "update", "--restart=no", container_id], check=True)
    subprocess.run(["docker", "kill", "--signal", signal, container_id], check=True)


def _container_state(container_id: str) -> dict:
    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", container_id],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(inspected.stdout)


def _wait_for_container_exit(container_id: str, timeout: float = 40) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _container_state(container_id)
        if not last["Running"]:
            return last
        time.sleep(0.25)
    pytest.fail(f"container {container_id} did not exit after signal; state={last}")


async def _wait_for_exit_without_old_owner_claim(
    pool,
    job_id: UUID,
    old_owner: str,
    container_id: str,
    timeout: float = 40,
) -> tuple[dict, str | None]:
    """Observe every drain-window lease state until the signalled worker exits."""
    deadline = time.monotonic() + timeout
    last_job = None
    last_container = None
    replacement_owner = None
    while time.monotonic() < deadline:
        last_job = await pool.fetchrow(
            "SELECT state::text, lease_owner FROM background_jobs WHERE id = $1",
            job_id,
        )
        if last_job and last_job["lease_owner"]:
            assert last_job["lease_owner"] != old_owner, (
                "SIGTERM-draining worker claimed a new durable job"
            )
            replacement_owner = last_job["lease_owner"]
        last_container = _container_state(container_id)
        if not last_container["Running"]:
            return last_container, replacement_owner
        await asyncio.sleep(0.05)
    pytest.fail(
        f"container did not exit during guarded drain window; "
        f"job_state={dict(last_job) if last_job else None}, container_state={last_container}"
    )


@pytest.mark.skipif(
    not LIVE,
    reason=(
        "requires the opt-in scaled Compose stack, a self-hosted Supabase database, "
        "and SCALED_TEST_DATABASE_URL/TOKEN/USER_ID"
    ),
)
@pytest.mark.asyncio
async def test_two_api_two_worker_recovery_smoke():
    api_url = _env_value("SCALED_TEST_API_URL", "http://127.0.0.1:8000").rstrip("/")
    database_url = _required_env("SCALED_TEST_DATABASE_URL")
    token = _required_env("SCALED_TEST_TOKEN")
    user_id = UUID(_required_env("SCALED_TEST_USER_ID"))
    auth_headers = {"Authorization": f"Bearer {token}"}
    kb_id = uuid4()
    graceful_kb_id = uuid4()
    draining_kb_id = uuid4()
    filename = f"scaled-{uuid4()}.pdf"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=3)

    try:
        await pool.executemany(
            "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
            (
                (kb_id, user_id, "Scaled compose smoke", f"scaled-{kb_id}"),
                (graceful_kb_id, user_id, "Graceful worker smoke", f"graceful-{graceful_kb_id}"),
                (draining_kb_id, user_id, "Draining worker smoke", f"draining-{draining_kb_id}"),
            ),
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
        websockets_by_instance = {}
        try:
            for _ in range(80):
                websocket = await websockets.connect(f"{websocket_url}/v1/ws/documents/{kb_id}")
                await websocket.send(token)
                instance_id = websocket.response.headers.get("x-api-instance-id")
                assert instance_id, "STAGE=test must identify each WebSocket handshake"
                if instance_id in websockets_by_instance:
                    await websocket.close()
                    continue
                websockets_by_instance[instance_id] = websocket
                if len(websockets_by_instance) == 2:
                    break
            assert len(websockets_by_instance) == 2, (
                "gateway did not route retained WebSockets to two API instances; "
                f"observed={sorted(websockets_by_instance)}"
            )

            notifications = None
            for _ in range(10):
                event_id = str(uuid4())
                payload = json.dumps(
                    {
                        "event": "scaled-smoke",
                        "id": event_id,
                        "user_id": str(user_id),
                        "knowledge_base_id": str(kb_id),
                    }
                )
                await pool.execute("SELECT pg_notify('document_changes', $1)", payload)
                try:
                    messages = await asyncio.wait_for(
                        asyncio.gather(*(websocket.recv() for websocket in websockets_by_instance.values())),
                        timeout=3,
                    )
                except TimeoutError:
                    continue
                decoded = [json.loads(message) for message in messages]
                expected = {"event": "scaled-smoke", "id": event_id}
                if all(notification == expected for notification in decoded):
                    notifications = decoded
                    break
            assert notifications is not None, (
                "two retained API WebSockets did not receive the same PostgreSQL NOTIFY; "
                f"instances={sorted(websockets_by_instance)}"
            )
        finally:
            await asyncio.gather(
                *(websocket.close() for websocket in websockets_by_instance.values()),
                return_exceptions=True,
            )

        pdf = _mini_pdf()
        tus_headers = {
            **auth_headers,
            "Tus-Resumable": "1.0.0",
            "Upload-Length": str(len(pdf)),
            "Upload-Metadata": _metadata(filename=filename, knowledge_base_id=str(kb_id), path="/"),
        }
        tus_instances = []
        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=90) as client:
            created = await client.post("/v1/uploads", headers=tus_headers)
            created.raise_for_status()
            created_instance = created.headers.get("x-api-instance-id")
            assert created_instance, "TUS create response omitted API instance identity"
            tus_instances.append(("create", created_instance))
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
            first_patch_instance = patched.headers.get("x-api-instance-id")
            assert first_patch_instance, "first TUS PATCH response omitted API instance identity"
            tus_instances.append(("patch-0", first_patch_instance))

        _restart_redis_and_wait()

        resume_instance = None
        for attempt in range(80):
            async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=90) as client:
                resumed = await client.head(
                    location,
                    headers={"Tus-Resumable": "1.0.0", "Connection": "close"},
                )
            resumed.raise_for_status()
            head_instance = resumed.headers.get("x-api-instance-id")
            assert head_instance, f"TUS HEAD attempt {attempt} omitted API instance identity"
            tus_instances.append((f"head-{attempt}", head_instance))
            if head_instance != first_patch_instance:
                resume_instance = head_instance
                assert int(resumed.headers["upload-offset"]) == len(first)
                break
        assert resume_instance != first_patch_instance, (
            f"gateway did not route TUS resume HEAD across API replicas; responses={tus_instances}"
        )

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=90) as client:
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
            completed_instance = completed.headers.get("x-api-instance-id")
            assert completed_instance, "final TUS PATCH response omitted API instance identity"
            tus_instances.append(("patch-final", completed_instance))
            assert len({instance_id for _operation, instance_id in tus_instances}) >= 2
            document_id = completed.headers["x-document-id"]
            extraction_job_id = completed.headers["x-job-id"]
            accepting_api_container = _api_container_for_instance(completed_instance)
            _disable_restart_and_kill(accepting_api_container, "KILL")

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
            await _wait_for_job(client, extraction_job_id, forbidden_instance=completed_instance)

        _compose("up", "-d", "--no-deps", "--scale", "api=2", "api")
        restored_api_instances = set()
        async with httpx.AsyncClient(base_url=api_url, timeout=20) as client:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and len(restored_api_instances) < 2:
                try:
                    response = await client.get("/health", headers={"Connection": "close"})
                except httpx.TransportError:
                    await asyncio.sleep(0.5)
                    continue
                if response.status_code == 200:
                    restored_instance = response.headers.get("x-api-instance-id")
                    if restored_instance:
                        restored_api_instances.add(restored_instance)
                await asyncio.sleep(0.25)
        assert len(restored_api_instances) == 2

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
            graph = await client.post(f"/v1/knowledge-bases/{kb_id}/graph/rebuild")
            graph.raise_for_status()
            await _wait_for_job(client, graph.json()["job_id"])

        graceful_lock_connection = None
        graceful_transaction = None
        graceful_transaction_started = False
        try:
            graceful_lock_connection = await pool.acquire()
            graceful_transaction = graceful_lock_connection.transaction()
            await graceful_transaction.start()
            graceful_transaction_started = True
            await graceful_lock_connection.execute("LOCK TABLE document_references IN ACCESS EXCLUSIVE MODE")
            async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
                graceful_response = await client.post(
                    f"/v1/knowledge-bases/{graceful_kb_id}/graph/rebuild"
                )
                graceful_response.raise_for_status()
                graceful_job_id = UUID(graceful_response.json()["job_id"])

            graceful_running = await _wait_for_running_job(pool, graceful_job_id)
            old_owner = graceful_running["lease_owner"]
            graceful_worker_container = _worker_container_for_owner(old_owner)
            subprocess.run(["docker", "update", "--restart=no", graceful_worker_container], check=True)
            subprocess.run(["docker", "kill", "--signal", "TERM", graceful_worker_container], check=True)

            async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
                draining_response = await client.post(
                    f"/v1/knowledge-bases/{draining_kb_id}/graph/rebuild"
                )
                draining_response.raise_for_status()
                draining_job_id = UUID(draining_response.json()["job_id"])
            assert _container_state(graceful_worker_container)["Running"], (
                "signalled worker exited before the new job entered the drain window"
            )
            graceful_exit, observed_replacement_owner = (
                await _wait_for_exit_without_old_owner_claim(
                    pool,
                    draining_job_id,
                    old_owner,
                    graceful_worker_container,
                )
            )

            deadline = time.monotonic() + 40
            graceful_transition = None
            while time.monotonic() < deadline:
                graceful_transition = await pool.fetchrow(
                    "SELECT state::text, lease_owner, lease_expires_at, error_code "
                    "FROM background_jobs WHERE id = $1",
                    graceful_job_id,
                )
                if (
                    graceful_transition
                    and graceful_transition["state"] in {"retry_wait", "succeeded"}
                    and graceful_transition["lease_owner"] is None
                    and graceful_transition["lease_expires_at"] is None
                ):
                    break
                await asyncio.sleep(0.05)
            assert graceful_transition is not None
            assert graceful_transition["state"] in {"retry_wait", "succeeded"}
            assert graceful_transition["lease_owner"] is None
            assert graceful_transition["lease_expires_at"] is None

            graceful_logs = subprocess.run(
                ["docker", "logs", graceful_worker_container],
                check=True,
                text=True,
                capture_output=True,
            )
            assert graceful_exit["ExitCode"] == 0
            assert "shutdown on SIGTERM" in graceful_logs.stderr + graceful_logs.stdout
            assert "durable worker resources closed" in graceful_logs.stderr + graceful_logs.stdout

            draining_running = await _wait_for_running_job(pool, draining_job_id)
            assert draining_running["lease_owner"] != old_owner
            if observed_replacement_owner is not None:
                assert draining_running["lease_owner"] == observed_replacement_owner
            old_owner_claims = await pool.fetchval(
                "SELECT count(*) FROM background_jobs "
                "WHERE state = 'running' AND lease_owner = $1",
                old_owner,
            )
            assert old_owner_claims == 0
        finally:
            try:
                if graceful_transaction_started:
                    await graceful_transaction.rollback()
            finally:
                if graceful_lock_connection is not None:
                    await pool.release(graceful_lock_connection)

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
            graceful_finished = await _wait_for_job(client, str(graceful_job_id), timeout=120)
            assert graceful_finished["attempt_count"] >= 2
            await _wait_for_job(client, str(draining_job_id), timeout=120)

        _compose("up", "-d", "--no-deps", "--scale", "worker=2", "worker")

        lock_connection = None
        transaction = None
        transaction_started = False
        try:
            lock_connection = await pool.acquire()
            transaction = lock_connection.transaction()
            await transaction.start()
            transaction_started = True
            await lock_connection.execute("LOCK TABLE document_references IN ACCESS EXCLUSIVE MODE")
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
                kill_started = time.monotonic()
                subprocess.run(["docker", "kill", "--signal", "KILL", container_id], check=True)
        finally:
            try:
                if transaction_started:
                    await transaction.rollback()
            finally:
                if lock_connection is not None:
                    await pool.release(lock_connection)

        _compose("up", "-d", "--no-deps", "--scale", "worker=2", "worker")

        async with httpx.AsyncClient(base_url=api_url, headers=auth_headers, timeout=20) as client:
            recovered = await _wait_for_job(client, str(leased_job_id), timeout=240)
            assert recovered["attempt_count"] >= 2
            assert time.monotonic() - kill_started < 240

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
        try:
            _compose(
                "up",
                "-d",
                "--no-deps",
                "--scale",
                "api=2",
                "--scale",
                "worker=2",
                "api",
                "worker",
                check=False,
            )
        finally:
            await pool.execute(
                "DELETE FROM knowledge_bases WHERE id = ANY($1::uuid[])",
                [kb_id, graceful_kb_id, draining_kb_id],
            )
            await pool.close()
