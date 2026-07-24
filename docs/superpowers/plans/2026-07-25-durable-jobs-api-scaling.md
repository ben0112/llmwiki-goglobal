# Durable Jobs and Hosted API Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hosted document extraction, graph rebuilds, and resumable uploads durable across API and worker failures, then remove the single-API-replica deployment constraint without changing Local mode.

**Architecture:** Postgres is the source of truth for job state and tenant ownership. ARQ/Redis is a replaceable delivery adapter plus the store for short-lived upload coordination; workers claim Postgres leases before running handlers. Hosted TUS stores bytes directly in S3-compatible multipart uploads and commits a document plus extraction job atomically. API and worker processes scale separately behind Nginx, while each API replica keeps its own WebSocket connections and Postgres listener.

**Tech Stack:** Python 3.11, FastAPI, asyncpg, ARQ 0.28.0, redis-py 5.3.1, Redis 7.4 with AOF, aioboto3/S3 multipart upload, Postgres 16, Nginx, pytest, Docker Compose, GitHub Actions.

---

## Ground rules

- Execute tasks in order. Each task starts with a failing test and ends with a focused commit and push to feat/platform-architecture-evolution.
- Keep Local mode unchanged. MODE=local must neither require Redis nor create a Postgres job table.
- Keep business state in Postgres. Redis loss may delay delivery or invalidate an unfinished upload, but cannot erase an accepted document or job.
- Disable ARQ retries and result ownership. The Postgres ledger alone decides attempts, backoff, cancellation, leases, and terminal state.
- Keep all ARQ imports inside api/jobs/dispatcher.py and api/jobs/worker.py. ARQ 0.28.0 is maintenance-only, so handlers and repositories must remain transport-agnostic.
- Preserve existing API response fields and add job_id only where an accepted durable operation creates a job.
- Never put document bytes, extracted text, credentials, database URLs, JWTs, or presigned URLs in background_jobs payload/result or Redis job messages.
- Run every command from the repository root unless a command includes cd.

## Task 1: Pin the hosted runtime boundary

**Files:**

- Modify: api/requirements.txt
- Modify: api/requirements.lock
- Modify: api/config.py
- Create: api/infra/redis.py
- Create: tests/unit/test_durable_runtime_config.py

- [ ] **Step 1: Write failing settings tests**

Add tests that prove Local mode can start without REDIS_URL and Hosted mode validates the two rollout flags:

~~~python
def test_local_mode_does_not_require_redis(monkeypatch):
    monkeypatch.setenv("MODE", "local")
    monkeypatch.delenv("REDIS_URL", raising=False)
    settings = Settings()
    assert settings.REDIS_URL is None


def test_hosted_durable_jobs_requires_redis(monkeypatch):
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "true")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="REDIS_URL"):
        Settings()


def test_tus_multipart_requires_durable_jobs(monkeypatch):
    monkeypatch.setenv("MODE", "hosted")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("TUS_MULTIPART_ENABLED", "true")
    monkeypatch.setenv("DURABLE_JOBS_ENABLED", "false")
    with pytest.raises(ValueError, match="DURABLE_JOBS_ENABLED"):
        Settings()
~~~

- [ ] **Step 2: Run the focused test and verify failure**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_durable_runtime_config.py -q
~~~

Expected: import or assertion failure because the new settings do not exist.

- [ ] **Step 3: Add typed settings and a lazy Redis factory**

Add these settings with conservative defaults:

~~~python
REDIS_URL: str | None = None
DURABLE_JOBS_ENABLED: bool = False
TUS_MULTIPART_ENABLED: bool = False
JOB_LEASE_SECONDS: int = 120
JOB_HEARTBEAT_SECONDS: int = 30
JOB_DISPATCH_BATCH_SIZE: int = 100
JOB_REDELIVER_SECONDS: int = 30
TUS_SESSION_TTL_SECONDS: int = 172800
TUS_STALE_SECONDS: int = 86400
TUS_LOCK_SECONDS: int = 60
TUS_MAX_PATCH_BYTES: int = 67108864
~~~

Validate positive timing relationships, require REDIS_URL only for Hosted durable features, and reject TUS_MULTIPART_ENABLED=true when DURABLE_JOBS_ENABLED=false.

In api/infra/redis.py expose only:

~~~python
from redis.asyncio import Redis


def create_redis(url: str) -> Redis:
    return Redis.from_url(
        url,
        decode_responses=False,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
~~~

Do not create a client at import time.

- [ ] **Step 4: Pin compatible dependencies and regenerate the hash lock**

Add exact pins:

~~~text
arq==0.28.0
redis[hiredis]==5.3.1
~~~

Then run:

~~~bash
cd api && /opt/homebrew/bin/uv pip compile requirements.txt --python-version 3.11 --output-file requirements.lock --generate-hashes
cd .. && /opt/homebrew/bin/uv pip install --python .venv/bin/python -r api/requirements.txt
~~~

Review the diff and confirm no unrequested package is removed.

- [ ] **Step 5: Run focused verification**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_durable_runtime_config.py -q
.venv/bin/ruff check api/config.py api/infra/redis.py tests/unit/test_durable_runtime_config.py
~~~

Expected: all pass.

- [ ] **Step 6: Commit and push**

~~~bash
git add api/requirements.txt api/requirements.lock api/config.py api/infra/redis.py tests/unit/test_durable_runtime_config.py
git commit -m "build: add durable hosted runtime boundary"
git push origin feat/platform-architecture-evolution
~~~

## Task 2: Add the Postgres job schema and pure state model

**Files:**

- Create: supabase/migrations/012_background_jobs.sql
- Modify: tests/helpers/schema.sql
- Create: api/jobs/__init__.py
- Create: api/jobs/models.py
- Create: tests/unit/jobs/__init__.py
- Create: tests/unit/jobs/test_models.py
- Create: tests/integration/test_background_job_schema.py

- [ ] **Step 1: Write failing model tests**

Cover exact job/state values, deterministic exponential backoff, terminal states, and payload limits:

~~~python
def test_retry_delay_is_capped():
    assert retry_delay_seconds(attempt=1, jitter=0) == 2
    assert retry_delay_seconds(attempt=5, jitter=0) == 32
    assert retry_delay_seconds(attempt=20, jitter=0) == 60


def test_terminal_states():
    assert JobState.SUCCEEDED.is_terminal
    assert JobState.FAILED.is_terminal
    assert JobState.CANCELLED.is_terminal
    assert not JobState.RUNNING.is_terminal
~~~

- [ ] **Step 2: Run the model test and verify failure**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/jobs/test_models.py -q
~~~

Expected: api.jobs.models does not exist.

- [ ] **Step 3: Implement transport-neutral job types**

Define:

~~~python
class JobType(StrEnum):
    DOCUMENT_EXTRACT = "document.extract"
    GRAPH_REBUILD = "graph.rebuild"
    UPLOAD_CLEANUP = "upload.cleanup"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
~~~

Add frozen JobCreate and JobRecord dataclasses, LeaseLost and JobCancelled exceptions, retry_delay_seconds(attempt, base=2, cap=60, jitter) and serialize_public_job(). Keep ARQ and Redis out of this module.

- [ ] **Step 4: Write the schema contract test**

The test must insert all three job types, reject an invalid state, reject oversized payload/result/progress JSON, and prove uniqueness of (user_id, job_type, idempotency_key) only when idempotency_key is non-null.

Run:

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_background_job_schema.py -q
~~~

Expected: relation background_jobs does not exist.

- [ ] **Step 5: Add migration 012 and mirror it in the test schema**

Create background_jobs with:

- UUID primary key and created_at/updated_at timestamps.
- user_id required; knowledge_base_id and document_id nullable, all with cascading tenant cleanup.
- job_type constrained to document.extract, graph.rebuild, upload.cleanup.
- state constrained to the six Python enum values.
- payload JSONB required and capped at 16 KiB; progress capped at 8 KiB; result capped at 16 KiB.
- attempt_count >= 0, max_attempts from 1 through 20, run_after required.
- lease_owner, lease_expires_at, heartbeat_at, last_dispatched_at, dispatch_attempts.
- error_code, error_message capped at 2,000 characters, cancel_requested_at.
- a partial idempotency unique index, due-dispatch index, lease-expiry index, and user-created index.
- RLS enabled with authenticated SELECT restricted to auth.uid() = user_id. Mutations continue through scoped application transactions/service role, not a client write policy.

The migration and tests/helpers/schema.sql definitions must be byte-for-byte equivalent for the table, constraints, and indexes apart from migration comments.

- [ ] **Step 6: Run focused verification**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/jobs/test_models.py -q
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_background_job_schema.py -q
.venv/bin/ruff check api/jobs tests/unit/jobs tests/integration/test_background_job_schema.py
~~~

- [ ] **Step 7: Commit and push**

~~~bash
git add supabase/migrations/012_background_jobs.sql tests/helpers/schema.sql api/jobs tests/unit/jobs tests/integration/test_background_job_schema.py
git commit -m "feat: add durable background job ledger"
git push origin feat/platform-architecture-evolution
~~~

## Task 3: Implement create, read, and cancel semantics

**Files:**

- Create: api/jobs/repository.py
- Create: api/jobs/service.py
- Create: tests/integration/test_background_job_repository.py

- [ ] **Step 1: Write failing repository tests**

Prove:

- create() returns the existing row for the same tenant/type/idempotency key.
- the same key in a second tenant creates a different job.
- get_for_user() cannot read another user's job.
- cancel() changes queued/retry_wait to cancelled.
- cancel() marks cancel_requested_at for running work.
- cancel() is idempotent for every terminal state.

Use the existing scoped Postgres fixtures and explicit tenant UUIDs.

- [ ] **Step 2: Run the tests and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_background_job_repository.py -q
~~~

Expected: repository/service imports fail.

- [ ] **Step 3: Implement SQL repository operations**

Expose async methods with explicit connection arguments:

~~~python
async def create(conn: asyncpg.Connection, command: JobCreate) -> JobRecord
async def get_for_user(conn: asyncpg.Connection, job_id: UUID, user_id: UUID) -> JobRecord | None
async def request_cancel(conn: asyncpg.Connection, job_id: UUID, user_id: UUID) -> JobRecord | None
~~~

create() uses INSERT ... ON CONFLICT ... WHERE idempotency_key IS NOT NULL DO UPDATE SET idempotency_key = EXCLUDED.idempotency_key RETURNING *. Never overwrite payload or ownership on conflict.

request_cancel() uses one UPDATE ... CASE expression so queued/retry_wait become cancelled, running receives cancel_requested_at, and terminal rows remain unchanged.

- [ ] **Step 4: Add a thin transaction service**

JobService accepts the pool, opens scoped transactions, validates that referenced document/knowledge base belongs to the authenticated user, and delegates to repository functions. It must not import ARQ.

- [ ] **Step 5: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_background_job_repository.py -q
.venv/bin/ruff check api/jobs/repository.py api/jobs/service.py tests/integration/test_background_job_repository.py
~~~

- [ ] **Step 6: Commit and push**

~~~bash
git add api/jobs/repository.py api/jobs/service.py tests/integration/test_background_job_repository.py
git commit -m "feat: persist background job commands"
git push origin feat/platform-architecture-evolution
~~~

## Task 4: Implement leases, retries, heartbeats, and recovery

**Files:**

- Modify: api/jobs/repository.py
- Create: api/jobs/lease.py
- Create: tests/unit/jobs/test_lease.py
- Create: tests/integration/test_background_job_leases.py

- [ ] **Step 1: Write failing lease transition tests**

Cover:

- one of two workers can claim a due queued job.
- attempt_count increments exactly once per successful claim.
- a worker cannot heartbeat or finish another worker's lease.
- cancel_requested_at raises JobCancelled at a checkpoint.
- retryable failure moves to retry_wait with run_after in the future.
- max attempts produces failed, not retry_wait.
- an expired running lease is requeued unless cancellation was requested.
- stale completion after lease expiry is rejected.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_lease.py tests/integration/test_background_job_leases.py -q
~~~

- [ ] **Step 3: Implement conditional state transitions**

Add repository methods:

~~~python
async def claim(conn, job_id, owner, lease_seconds) -> JobRecord | None
async def heartbeat(conn, job_id, owner, lease_seconds) -> JobRecord
async def assert_active(conn, job_id, owner) -> JobRecord
async def succeed(conn, job_id, owner, result) -> JobRecord
async def fail_or_retry(conn, job_id, owner, error, retryable, now) -> JobRecord
async def reap_expired(conn, now, limit) -> list[UUID]
~~~

Every worker mutation must include:

~~~sql
WHERE id = $1
  AND state = 'running'
  AND lease_owner = $2
  AND lease_expires_at > now()
~~~

claim accepts queued or due retry_wait, rejects cancel_requested_at, and atomically sets running, owner, heartbeat, lease expiry, and attempt_count + 1.

- [ ] **Step 4: Implement JobLease checkpoints**

JobLease owns job_id/owner/repository/pool, starts a heartbeat loop at JOB_HEARTBEAT_SECONDS, exposes checkpoint(), and guarantees heartbeat shutdown in __aexit__. The handler must call checkpoint before irreversible work and inside the final database transaction.

- [ ] **Step 5: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_lease.py tests/integration/test_background_job_leases.py -q
.venv/bin/ruff check api/jobs tests/unit/jobs/test_lease.py tests/integration/test_background_job_leases.py
~~~

- [ ] **Step 6: Commit and push**

~~~bash
git add api/jobs/repository.py api/jobs/lease.py tests/unit/jobs/test_lease.py tests/integration/test_background_job_leases.py
git commit -m "feat: enforce durable job leases"
git push origin feat/platform-architecture-evolution
~~~

## Task 5: Add the replaceable ARQ delivery adapter

**Files:**

- Create: api/jobs/dispatcher.py
- Create: api/jobs/worker.py
- Create: api/jobs/handlers.py
- Create: tests/unit/jobs/test_dispatcher.py
- Create: tests/integration/test_job_delivery.py

- [ ] **Step 1: Write failing dispatcher tests**

Use a fake enqueue adapter to prove:

- due rows are selected with FOR UPDATE SKIP LOCKED.
- Redis receives only the database UUID as both argument and _job_id.
- successful enqueue records last_dispatched_at and increments dispatch_attempts.
- enqueue failure leaves the row due for the next scan.
- queued/running terminal mismatches do not become duplicate execution.
- rows dispatched before JOB_REDELIVER_SECONDS become eligible again.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_dispatcher.py tests/integration/test_job_delivery.py -q
~~~

- [ ] **Step 3: Implement dispatcher and due-row SQL**

The repository returns candidate UUIDs without holding a database transaction across a Redis call. Dispatcher flow for each UUID:

1. enqueue run_job(str(job_id), _job_id=str(job_id));
2. in a short transaction mark_dispatched(job_id);
3. log and continue on Redis failure.

Use a periodic scan, not an in-memory queue. ARQ enqueue_job() returning None because the deterministic _job_id is already present counts as delivered and updates last_dispatched_at.

- [ ] **Step 4: Implement the worker envelope**

api/jobs/worker.py contains:

~~~python
class WorkerSettings:
    functions = [run_job]
    cron_jobs = [
        cron(dispatch_due_jobs, second={0, 10, 20, 30, 40, 50}),
        cron(reap_expired_jobs, second={5, 35}),
    ]
    max_tries = 1
    retry_jobs = False
    keep_result = 0
    job_timeout = 3600
    on_startup = startup
    on_shutdown = shutdown
~~~

startup creates Postgres, Redis, S3, converter, repository, and handler registry resources. run_job parses only the UUID, claims the Postgres row, selects a handler by persisted job_type, manages JobLease, and records succeed/fail_or_retry. Unknown job types fail terminally with error_code=unsupported_job_type.

Add cron coroutines for dispatch_due_jobs and reap_expired_jobs. Do not let ARQ retry exceptions.

- [ ] **Step 5: Add an explicit handler registry**

api/jobs/handlers.py exposes:

~~~python
Handler = Callable[[JobRecord, JobLease, WorkerContext], Awaitable[dict[str, object]]]
HANDLERS: dict[JobType, Handler]
~~~

Initially register stubs that raise UnsupportedJobHandler; the next tasks replace document.extract, graph.rebuild, and upload.cleanup one at a time.

- [ ] **Step 6: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_dispatcher.py tests/integration/test_job_delivery.py -q
.venv/bin/ruff check api/jobs tests/unit/jobs/test_dispatcher.py tests/integration/test_job_delivery.py
~~~

- [ ] **Step 7: Commit and push**

~~~bash
git add api/jobs/dispatcher.py api/jobs/worker.py api/jobs/handlers.py tests/unit/jobs/test_dispatcher.py tests/integration/test_job_delivery.py
git commit -m "feat: dispatch durable jobs through arq"
git push origin feat/platform-architecture-evolution
~~~

## Task 6: Expose tenant-scoped job status and cancellation

**Files:**

- Create: api/routes/jobs.py
- Modify: api/deps.py
- Modify: api/main.py
- Create: tests/integration/isolation/test_job_api_isolation.py

- [ ] **Step 1: Write failing API isolation tests**

Cover GET /v1/jobs/{job_id} and POST /v1/jobs/{job_id}/cancel:

- owner receives the public job shape.
- a second tenant receives 404, not 403.
- unauthenticated requests receive 401.
- queued cancellation returns state=cancelled.
- running cancellation returns state=running with cancel_requested=true.
- repeated cancellation is idempotent.
- payload, lease_owner, Redis metadata, stack traces, and internal error details never appear.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/isolation/test_job_api_isolation.py -q
~~~

- [ ] **Step 3: Add dependency and router**

get_job_service(request) reads the initialized Hosted JobService from app.state and returns 503 if durable jobs are enabled but unavailable.

Public response:

~~~json
{
  "id": "uuid",
  "type": "document.extract",
  "state": "queued",
  "progress": {},
  "result": null,
  "attempt_count": 0,
  "max_attempts": 3,
  "cancel_requested": false,
  "error": null,
  "created_at": "...",
  "updated_at": "..."
}
~~~

Map only allow-listed error codes to a sanitized error object.

- [ ] **Step 4: Register the router only in Hosted mode**

Keep Local mode route loading and startup free from Redis/job dependencies.

- [ ] **Step 5: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/isolation/test_job_api_isolation.py -q
.venv/bin/ruff check api/routes/jobs.py api/deps.py api/main.py tests/integration/isolation/test_job_api_isolation.py
~~~

- [ ] **Step 6: Commit and push**

~~~bash
git add api/routes/jobs.py api/deps.py api/main.py tests/integration/isolation/test_job_api_isolation.py
git commit -m "feat: expose durable job status and cancellation"
git push origin feat/platform-architecture-evolution
~~~

## Task 7: Move Hosted document extraction to durable jobs

**Files:**

- Modify: api/services/ocr.py
- Modify: api/services/url_ingest.py
- Modify: api/jobs/handlers.py
- Modify: api/main.py
- Modify: api/routes/documents.py
- Modify: api/services/types.py
- Modify: tests/unit/test_url_ingest.py
- Create: tests/integration/test_durable_extraction.py

- [ ] **Step 1: Write failing producer and handler tests**

Prove:

- URL ingest uploads the source, then commits the pending document and document.extract job in the same Postgres transaction.
- a transaction failure deletes the just-uploaded orphan S3 object.
- idempotent producer calls return one document and one job.
- handler reads document_id from persisted payload after validating tenant ownership.
- duplicate execution produces one current derived document set.
- lease loss before replace_derived_content prevents the final write.
- transient S3/converter errors retry; missing document, ownership mismatch, unsupported type, and quota errors fail terminally.
- cancellation before final write leaves the previous derived version current.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_url_ingest.py tests/integration/test_durable_extraction.py -q
~~~

- [ ] **Step 3: Refactor OCRService into a worker-safe operation**

Keep parsing and derived-content behavior, but stop swallowing all exceptions inside _do_process. Introduce typed retryable/terminal extraction exceptions. Accept before_write: Callable[[asyncpg.Connection], Awaitable[None]] and pass it to replace_derived_content so JobLease.assert_active runs in the same final transaction.

The handler returns bounded metadata only:

~~~python
{"document_id": str(document_id), "derived_version": version}
~~~

- [ ] **Step 4: Replace URL ingest spawn_logged with atomic job creation**

URLIngestService receives JobService instead of OCRService. After S3 upload:

~~~python
async with pool.acquire() as conn:
    async with conn.transaction():
        document = await insert_pending_document(conn, ...)
        job = await job_repository.create(
            conn,
            JobCreate(
                job_type=JobType.DOCUMENT_EXTRACT,
                user_id=user_id,
                knowledge_base_id=kb_id,
                document_id=document.id,
                payload={"document_id": str(document.id)},
                idempotency_key=f"document.extract:{document.id}",
            ),
        )
~~~

Return existing fields plus job_id and add X-Job-Id to exposed CORS headers.

- [ ] **Step 5: Stop Hosted startup extraction**

When DURABLE_JOBS_ENABLED=true:

- initialize JobService;
- do not scan pending/processing documents into spawn_logged;
- run one recovery transaction that inserts missing document.extract jobs for pending/processing Hosted documents using the deterministic idempotency key;
- rely on the dispatcher for delivery.

When the flag is false, preserve the old path for rollback. Local startup remains unchanged.

- [ ] **Step 6: Run focused and regression verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_url_ingest.py tests/integration/test_durable_extraction.py tests/integration/test_converter_isolation.py -q
PYTHONPATH=api .venv/bin/pytest tests/unit/test_extraction_fixes.py tests/unit/test_local_processor_extraction.py -q
.venv/bin/ruff check api/services/ocr.py api/services/url_ingest.py api/jobs/handlers.py api/main.py api/routes/documents.py api/services/types.py tests/unit/test_url_ingest.py tests/integration/test_durable_extraction.py
~~~

- [ ] **Step 7: Commit and push**

~~~bash
git add api/services/ocr.py api/services/url_ingest.py api/jobs/handlers.py api/main.py api/routes/documents.py api/services/types.py tests/unit/test_url_ingest.py tests/integration/test_durable_extraction.py
git commit -m "feat: run hosted extraction as durable jobs"
git push origin feat/platform-architecture-evolution
~~~

## Task 8: Move Hosted graph rebuilds to durable jobs

**Files:**

- Modify: api/routes/graph.py
- Modify: api/services/graph.py
- Modify: api/jobs/handlers.py
- Create: tests/integration/test_durable_graph_rebuild.py
- Modify: tests/unit/test_graph_local.py

- [ ] **Step 1: Write failing graph job tests**

Prove:

- POST Hosted rebuild returns 202 with job_id.
- repeated active requests for the same user/knowledge base return the same job.
- two API instances cannot create two active graph jobs.
- a worker validates lease in the same transaction that replaces graph edges.
- a stale worker cannot replace a newer graph.
- another tenant cannot observe or cancel the job.
- Local graph tests continue to call the synchronous local implementation.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_durable_graph_rebuild.py -q
PYTHONPATH=api .venv/bin/pytest tests/unit/test_graph_local.py -q
~~~

- [ ] **Step 3: Add an active-job database invariant**

Add a partial unique index in migration 012 and tests/helpers/schema.sql:

~~~sql
CREATE UNIQUE INDEX background_jobs_one_active_graph
ON background_jobs (user_id, knowledge_base_id, job_type)
WHERE job_type = 'graph.rebuild'
  AND state IN ('queued', 'running', 'retry_wait');
~~~

Because migration 012 has not shipped outside this feature branch, amend that migration rather than create a follow-up migration.

- [ ] **Step 4: Replace process-local Hosted coordination**

Remove _rebuild_locks and _rebuild_last_run from api/routes/graph.py. Create/return graph.rebuild through JobService; database uniqueness replaces the per-process lock/cooldown.

Modify rebuild_hosted(..., before_write=None) and invoke before_write(conn) immediately before atomic edge/facet replacement.

- [ ] **Step 5: Register the graph handler**

Validate job knowledge_base_id/user_id, call lease.checkpoint before compute and inside before_write, and return bounded counts only.

- [ ] **Step 6: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_durable_graph_rebuild.py -q
PYTHONPATH=api .venv/bin/pytest tests/unit/test_graph_local.py -q
.venv/bin/ruff check api/routes/graph.py api/services/graph.py api/jobs/handlers.py tests/integration/test_durable_graph_rebuild.py tests/unit/test_graph_local.py
~~~

- [ ] **Step 7: Commit and push**

~~~bash
git add supabase/migrations/012_background_jobs.sql tests/helpers/schema.sql api/routes/graph.py api/services/graph.py api/jobs/handlers.py tests/integration/test_durable_graph_rebuild.py tests/unit/test_graph_local.py
git commit -m "feat: run hosted graph rebuilds as durable jobs"
git push origin feat/platform-architecture-evolution
~~~

## Task 9: Add S3 multipart primitives

**Files:**

- Modify: api/services/s3.py
- Modify: tests/unit/test_s3_selfhost.py
- Create: tests/integration/test_s3_multipart.py

- [ ] **Step 1: Write failing S3 contract tests**

Using a fake client in unit tests and MinIO in integration tests, cover create, upload part, complete, abort, head object, ranged read, and delete object. Assert that ETags are returned without surrounding quotes and completion sends ordered PartNumber/ETag pairs.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_s3_selfhost.py tests/integration/test_s3_multipart.py -q
~~~

- [ ] **Step 3: Implement narrow multipart methods**

Add:

~~~python
async def create_multipart(self, key: str, content_type: str) -> str
async def upload_part(self, key: str, upload_id: str, part_number: int, body: bytes) -> str
async def complete_multipart(self, key: str, upload_id: str, parts: list[MultipartPart]) -> None
async def abort_multipart(self, key: str, upload_id: str) -> None
async def head_object(self, key: str) -> ObjectMetadata | None
async def read_range(self, key: str, start: int, end: int) -> bytes
async def delete_object(self, key: str) -> None
async def head_bucket(self) -> None
~~~

Reject part numbers outside 1..10,000. The TUS layer, not S3Service, enforces the 5 MiB non-final-part rule.

- [ ] **Step 4: Run focused verification**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_s3_selfhost.py tests/integration/test_s3_multipart.py -q
.venv/bin/ruff check api/services/s3.py tests/unit/test_s3_selfhost.py tests/integration/test_s3_multipart.py
~~~

- [ ] **Step 5: Commit and push**

~~~bash
git add api/services/s3.py tests/unit/test_s3_selfhost.py tests/integration/test_s3_multipart.py
git commit -m "feat: add object store multipart primitives"
git push origin feat/platform-architecture-evolution
~~~

## Task 10: Store TUS sessions and locks in Redis

**Files:**

- Create: api/infra/tus_sessions.py
- Create: tests/unit/test_tus_sessions.py
- Create: tests/integration/test_tus_sessions_redis.py

- [ ] **Step 1: Write failing session/lock tests**

Cover:

- all fields round-trip without credentials or auth material;
- compare-and-set offset succeeds only at the expected offset;
- part number and ETag append atomically with the offset;
- duplicate completed PATCH returns the committed offset without appending;
- lock release and renewal require the owner token;
- TTL refreshes on successful activity;
- expired session returns not found;
- reservation release and completion markers are idempotent.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_tus_sessions.py tests/integration/test_tus_sessions_redis.py -q
~~~

- [ ] **Step 3: Implement opaque Redis records**

Use keys:

~~~text
tus:session:{upload_uuid}
tus:lock:{upload_uuid}
tus:reservation:{user_uuid}:{upload_uuid}
~~~

TusSession contains upload UUID, authenticated user/KB UUIDs, sanitized filename, content type, total length, current offset, S3 key, multipart upload ID, ordered parts, created/updated timestamps, state, document_id/job_id after commit, and reservation bytes.

Use Lua for:

- append_part(expected_offset, byte_count, part_number, etag, ttl);
- mark_complete(expected_offset, document_id, job_id, ttl);
- release_lock(token);
- renew_lock(token, ttl);
- release_reservation_once.

Never deserialize arbitrary Python objects; use explicit JSON validation and reject unknown/invalid numeric state.

- [ ] **Step 4: Run focused verification**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_tus_sessions.py tests/integration/test_tus_sessions_redis.py -q
.venv/bin/ruff check api/infra/tus_sessions.py tests/unit/test_tus_sessions.py tests/integration/test_tus_sessions_redis.py
~~~

- [ ] **Step 5: Commit and push**

~~~bash
git add api/infra/tus_sessions.py tests/unit/test_tus_sessions.py tests/integration/test_tus_sessions_redis.py
git commit -m "feat: persist tus coordination in redis"
git push origin feat/platform-architecture-evolution
~~~

## Task 11: Centralize Hosted quota reservations

**Files:**

- Create: api/infra/quota.py
- Modify: api/services/url_ingest.py
- Create: tests/unit/test_hosted_quota.py
- Create: tests/integration/test_hosted_quota.py

- [ ] **Step 1: Write failing quota tests**

Cover:

- committed Postgres bytes plus all live Redis reservations determine acceptance.
- two concurrent upload creates cannot both consume the final available bytes.
- release is token/idempotency safe.
- expired reservation no longer counts.
- final document commit occurs before reservation release.
- URL ingest and TUS use the same HostedQuotaService.
- tenant IDs isolate reservations.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_hosted_quota.py tests/integration/test_hosted_quota.py -q
~~~

- [ ] **Step 3: Implement reservation protocol**

HostedQuotaService.reserve(user_id, upload_id, bytes, ttl) acquires a short token lock at quota:lock:{user_id}, reads committed usage in a scoped Postgres transaction, sums live reservations from a per-user Redis sorted set/hash, rejects overflow, writes the reservation with expiry, then releases the token lock.

finalize() deletes a reservation only after the caller commits the document. release() is safe to repeat. A cleanup method removes expired sorted-set members.

Do not use Redis as authoritative billing data; it only prevents concurrent in-flight over-allocation.

- [ ] **Step 4: Route URL ingest through the shared service**

Replace its bespoke quota calculation with HostedQuotaService for the downloaded byte length. Use a deterministic reservation ID tied to the ingest command, commit the document/job, then release.

- [ ] **Step 5: Run focused verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_hosted_quota.py tests/integration/test_hosted_quota.py tests/unit/test_url_ingest.py -q
.venv/bin/ruff check api/infra/quota.py api/services/url_ingest.py tests/unit/test_hosted_quota.py tests/integration/test_hosted_quota.py
~~~

- [ ] **Step 6: Commit and push**

~~~bash
git add api/infra/quota.py api/services/url_ingest.py tests/unit/test_hosted_quota.py tests/integration/test_hosted_quota.py
git commit -m "feat: coordinate hosted quota reservations"
git push origin feat/platform-architecture-evolution
~~~

## Task 12: Replace Hosted TUS disk state with Redis and S3 multipart

**Files:**

- Modify: api/infra/tus.py
- Modify: api/main.py
- Modify: api/jobs/handlers.py
- Modify: tests/unit/test_tus_sanitize_and_tasks.py
- Modify: tests/unit/test_resumable_upload.py
- Modify: tests/integration/test_converter_isolation.py
- Create: tests/integration/test_tus_multipart.py

- [ ] **Step 1: Write failing cross-replica protocol tests**

Build two FastAPI app instances sharing Postgres/Redis/MinIO and cover:

- POST on replica A, HEAD/PATCH on replica B.
- Upload-Offset mismatch returns 409 and the current offset.
- a non-final PATCH smaller than 5 MiB returns 413/422 and keeps the old offset.
- a request above TUS_MAX_PATCH_BYTES is rejected before S3 upload.
- final part may be smaller than 5 MiB.
- Redis offset changes only after S3 returns an ETag.
- duplicate final PATCH creates exactly one document and one document.extract job.
- transaction failure after multipart completion deletes the orphan object and preserves a recoverable/cleanup state.
- lock contention returns 423 or 409 with Retry-After.
- authorization is checked against the session owner on HEAD and PATCH.
- no file appears under /tmp/supavault_tus_uploads.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_tus_multipart.py tests/unit/test_resumable_upload.py -q
~~~

- [ ] **Step 3: Implement create and HEAD**

POST:

1. validate metadata, length, type, tenant, and TUS headers;
2. reserve the full declared length;
3. create an opaque S3 object key and multipart upload;
4. write the Redis session;
5. compensate by aborting S3/releasing quota if session creation fails.

HEAD loads Redis, checks owner, and returns Upload-Length, Upload-Offset, and sanitized Upload-Metadata without local disk access.

- [ ] **Step 4: Implement PATCH with bounded memory**

Acquire the token lock, reload session, verify owner/offset/state, and read at most TUS_MAX_PATCH_BYTES + 1 into a bytearray. The browser currently sends 50 MiB chunks, so the default 64 MiB cap admits existing clients while bounding per-request memory.

Reject any non-final chunk below 5 MiB. Upload one S3 part; only after receiving ETag call append_part Lua. If Redis CAS unexpectedly fails, keep the lock, re-read state, and mark the session cleanup_required rather than advancing a guessed offset.

- [ ] **Step 5: Implement finalization transaction**

After the final ETag:

1. complete multipart with ordered parts;
2. verify object length with head_object;
3. in one Postgres transaction insert the pending document and document.extract job with deterministic idempotency keys;
4. mark Redis complete with document_id/job_id;
5. release quota exactly once.

Use persisted completion IDs to make a repeated final PATCH return 204 with the final offset and X-Document-Id/X-Job-Id.

- [ ] **Step 6: Implement upload.cleanup**

Replace the API process cleanup loop with an ARQ cron scan that creates/executes upload.cleanup work for stale sessions. Under the same token lock it aborts unfinished multipart uploads, deletes completed orphan objects, releases quota once, and deletes the session. Treat NoSuchUpload/NoSuchKey as successful idempotent cleanup.

Keep sanitize_upload_path compatibility for existing imports, but remove the process-local _uploads dictionary and /tmp upload directory from Hosted behavior.

- [ ] **Step 7: Preserve rollback flag behavior**

When TUS_MULTIPART_ENABLED=false, retain the legacy Hosted path during rollout. Put legacy state in a clearly named private compatibility class so the final rollout task can delete it only after the full matrix passes. Local upload behavior is unaffected.

- [ ] **Step 8: Run focused and isolation verification**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_tus_multipart.py tests/integration/test_converter_isolation.py tests/unit/test_tus_sanitize_and_tasks.py tests/unit/test_resumable_upload.py -q
.venv/bin/ruff check api/infra/tus.py api/jobs/handlers.py api/main.py tests/integration/test_tus_multipart.py tests/unit/test_tus_sanitize_and_tasks.py tests/unit/test_resumable_upload.py
~~~

- [ ] **Step 9: Commit and push**

~~~bash
git add api/infra/tus.py api/main.py api/jobs/handlers.py tests/unit/test_tus_sanitize_and_tasks.py tests/unit/test_resumable_upload.py tests/integration/test_converter_isolation.py tests/integration/test_tus_multipart.py
git commit -m "feat: make hosted tus uploads replica safe"
git push origin feat/platform-architecture-evolution
~~~

## Task 13: Deploy independently scalable API and workers

**Files:**

- Modify: api/routes/health.py
- Modify: api/main.py
- Modify: api/jobs/worker.py
- Create: deploy/nginx.conf
- Modify: deploy/docker-compose.selfhost.yml
- Modify: deploy/.env.selfhost.example
- Modify: docs/self-hosting.md
- Create: tests/unit/test_health_roles.py
- Create: tests/integration/test_scaled_compose.py

- [ ] **Step 1: Write failing role-readiness tests**

API readiness must fail independently for Postgres, Redis, and S3; worker startup readiness must include converter. Liveness must not fail merely because a dependency is temporarily unavailable. Local readiness remains its current no-Redis behavior.

- [ ] **Step 2: Run and verify failure**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_health_roles.py -q
~~~

- [ ] **Step 3: Wire Hosted lifecycle resources**

In API lifespan, create one Redis client per process only when a Hosted durable flag is enabled; ping during startup, attach it to app.state, and await aclose() during shutdown. Keep one local WebSocket manager and one Postgres LISTEN task per API replica.

API /ready checks Postgres SELECT 1, Redis PING, and S3 head_bucket. Worker startup performs the same checks plus converter health before accepting jobs. On dependency loss after startup, a job fails through the normal Postgres retry classifier rather than being acknowledged as successful.

- [ ] **Step 4: Add Redis, worker, and Nginx services**

Pin Redis:

~~~yaml
redis:
  image: redis:7.4-alpine
  command: ["redis-server", "--appendonly", "yes", "--appendfsync", "everysec"]
  volumes:
    - redis-data:/data
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
~~~

The worker reuses api/Dockerfile and overrides command:

~~~yaml
command: ["arq", "jobs.worker.WorkerSettings"]
~~~

Name the Nginx service gateway. API uses expose: ["8000"] without a host port. Gateway alone maps 8000:8000, resolves the Compose api service through Docker DNS on every request window, forwards HTTP and WebSocket Upgrade headers, uses conservative upload/body/read timeouts, and never exposes Redis/converter:

~~~nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
}

server {
    listen 8000;
    client_max_body_size 65m;
    resolver 127.0.0.11 valid=10s ipv6=off;
    set $api_backend http://api:8000;

    location / {
        proxy_pass $api_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_request_buffering off;
        proxy_read_timeout 3600s;
    }
}
~~~

Add REDIS_URL=redis://redis:6379/0, DURABLE_JOBS_ENABLED=true, and TUS_MULTIPART_ENABLED=true to API and worker environments. Add redis-data volume. Remove the exact-one-API warning.

- [ ] **Step 5: Add scaled Compose smoke test**

The test command:

~~~bash
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost up -d --build --scale api=2 --scale worker=2
PYTHONPATH=api .venv/bin/pytest tests/integration/test_scaled_compose.py -q
~~~

The test must:

- make repeated HTTP calls through Nginx and observe two API instance IDs from a test-only response header enabled in STAGE=test;
- open a WebSocket and receive a Postgres notification;
- create a multipart upload, resume it across requests, and wait for extraction success;
- enqueue graph rebuild and wait for success;
- terminate one worker during a leased test job and observe reaper-driven completion by the other worker;
- assert no duplicate document/derived rows.

Always tear down with:

~~~bash
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost down -v
~~~

- [ ] **Step 6: Document scaling and operating commands**

Document migrations through 012, Redis AOF backup/restore, role-specific readiness, worker concurrency, zero-downtime order from the design, rollback flags, and:

~~~bash
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost up -d --scale api=2 --scale worker=2
~~~

After changing replica counts, restart the gateway to drop cached upstream connections:

~~~bash
docker compose -f deploy/docker-compose.selfhost.yml --env-file deploy/.env.selfhost restart gateway
~~~

- [ ] **Step 7: Run focused verification**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_health_roles.py -q
docker compose -f deploy/docker-compose.selfhost.yml config --quiet
.venv/bin/ruff check api/routes/health.py api/main.py api/jobs/worker.py tests/unit/test_health_roles.py tests/integration/test_scaled_compose.py
~~~

- [ ] **Step 8: Commit and push**

~~~bash
git add api/routes/health.py api/main.py api/jobs/worker.py deploy/nginx.conf deploy/docker-compose.selfhost.yml deploy/.env.selfhost.example docs/self-hosting.md tests/unit/test_health_roles.py tests/integration/test_scaled_compose.py
git commit -m "deploy: scale hosted api and workers independently"
git push origin feat/platform-architecture-evolution
~~~

## Task 14: Add CI, observability, rollout verification, and close the milestone

**Files:**

- Modify: api/jobs/dispatcher.py
- Modify: api/jobs/worker.py
- Modify: api/config.py
- Modify: api/main.py
- Modify: api/infra/quota.py
- Modify: api/infra/tus_sessions.py
- Modify: api/infra/tus.py
- Modify: .github/workflows/test.yml
- Modify: README.md
- Modify: docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md
- Create: docs/architecture/durable-jobs.md
- Create: tests/integration/test_durable_failure_matrix.py

- [ ] **Step 1: Write the failure-matrix tests before changing CI**

Cover with deterministic fault injection:

- Redis outage after document/job transaction, followed by eventual dispatch.
- API termination immediately after acceptance.
- worker termination before final derived write, followed by lease recovery.
- duplicate ARQ delivery.
- Redis offset-write failure after successful S3 part.
- multipart completion followed by Postgres rollback.
- Redis restart with AOF and session resume.
- repeated cleanup with one quota release.
- graceful SIGTERM stops new claims, finishes or relinquishes the current lease, and closes resources.

- [ ] **Step 2: Run and verify at least one expected failure**

~~~bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_durable_failure_matrix.py -q
~~~

- [ ] **Step 3: Add structured operational telemetry**

Log structured fields job_id, job_type, attempt, state, lease_owner, duration_ms, error_code, dispatch_attempts, upload_id, and replica role. Never log payloads or object URLs.

Emit stable structured JSON log events because this repository does not currently ship a metrics backend:

- durable_job_dispatched with state/type/dispatch lag;
- durable_job_finished with retry count/latency/outcome;
- durable_job_lease_reaped;
- tus_session_created, tus_session_completed, and tus_session_stale;
- quota_reserved and quota_released with byte counts;
- upload_cleanup_finished.

Document these event names and example log queries; do not add a second monitoring stack in this milestone.

- [ ] **Step 4: Add Redis and MinIO to GitHub Actions**

In postgres-integration add Redis 7.4 as a service with health check. Start MinIO and create the test bucket in explicit steps. Add REDIS_URL/S3 test environment variables and run all new integration files.

Add a required scaled-compose GitHub Actions job that builds the stack, starts two API and two worker replicas, runs tests/integration/test_scaled_compose.py, prints service logs on failure, and always runs docker compose down -v. Keep docker compose config --quiet as an earlier fast validation step. Do not silently skip the failure matrix.

- [ ] **Step 5: Run the full verification matrix**

~~~bash
PYTHONPATH=api .venv/bin/pytest tests/unit/ -q
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/ -q
(cd mcp && ../.venv/bin/pytest ../tests/unit/mcp/ ../tests/integration/mcp/ -q)
.venv/bin/ruff check api/jobs api/infra/redis.py api/infra/quota.py api/infra/tus_sessions.py api/infra/tus.py api/routes/jobs.py api/routes/health.py api/services/ocr.py api/services/graph.py api/services/s3.py api/services/url_ingest.py tests/unit/jobs tests/unit/test_durable_runtime_config.py tests/unit/test_tus_sessions.py tests/unit/test_hosted_quota.py tests/unit/test_health_roles.py tests/integration/test_background_job_schema.py tests/integration/test_background_job_repository.py tests/integration/test_background_job_leases.py tests/integration/test_job_delivery.py tests/integration/test_durable_extraction.py tests/integration/test_durable_graph_rebuild.py tests/integration/test_s3_multipart.py tests/integration/test_tus_sessions_redis.py tests/integration/test_hosted_quota.py tests/integration/test_tus_multipart.py tests/integration/test_durable_failure_matrix.py
docker compose -f deploy/docker-compose.selfhost.yml config --quiet
~~~

Run the scaled smoke test from Task 13 with two API and two worker replicas. Record commands, pass counts, and any pre-existing repository-wide Ruff debt; do not claim repository-wide lint clean unless it is actually clean.

- [ ] **Step 6: Verify all acceptance criteria manually**

Check each acceptance criterion in the confirmed design:

- accepted extraction/graph jobs survive API loss;
- worker crash recovers exactly once at the materialized-data boundary;
- duplicate ARQ/final PATCH does not duplicate business rows;
- cross-replica TUS resume works;
- two API/two worker smoke passes;
- Hosted startup has no extraction coroutine scan when the flag is on;
- Local full suite starts without Redis/Postgres/S3/ARQ worker.

- [ ] **Step 7: Complete architecture and rollout documentation**

docs/architecture/durable-jobs.md must include:

- state transition diagram and lease/CAS rules;
- transaction boundaries for URL ingest and TUS finalization;
- retryable versus terminal error table;
- Redis key inventory and TTLs;
- S3 multipart limits (5 MiB non-final minimum, 10,000 parts);
- security/tenant boundaries;
- ARQ maintenance-only risk and the two-file replacement boundary;
- backup, recovery, rollback, and scaling runbooks.

Update README deployment overview. Mark the design status implemented only after every acceptance item passes, and append evidence with commit SHA and test commands.

- [ ] **Step 8: Remove rollout-only legacy Hosted behavior**

After the full matrix passes with both flags enabled:

- remove Hosted startup spawn_logged extraction recovery;
- remove the process-local Hosted TUS dictionary/temp-file compatibility class;
- keep both flags in settings for one release, but in Hosted mode make false fail fast with a message that durable workers and multipart storage are now required;
- do not remove any Local-mode implementation.

Re-run the full verification matrix after deletion.

- [ ] **Step 9: Commit, push, and inspect GitHub checks**

~~~bash
git add api/jobs/dispatcher.py api/jobs/worker.py api/config.py api/main.py api/infra/quota.py api/infra/tus_sessions.py api/infra/tus.py .github/workflows/test.yml README.md docs/architecture/durable-jobs.md docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md tests/integration/test_durable_failure_matrix.py
git commit -m "docs: verify durable hosted execution milestone"
git push origin feat/platform-architecture-evolution
gh run list -R ben0112/llmwiki-goglobal --branch feat/platform-architecture-evolution --limit 5
~~~

Wait for the pushed workflow and inspect any failure with:

~~~bash
gh run view RUN_ID -R ben0112/llmwiki-goglobal --log-failed
~~~

The milestone is complete only when required GitHub checks are green and the branch is clean.

## Final implementation review checklist

- [ ] Every persisted job transition is conditional on state and, for workers, a live matching lease.
- [ ] Every producer commits business row plus job row in one Postgres transaction.
- [ ] Redis messages contain only a job UUID; Postgres/Redis records contain no secrets or large content.
- [ ] ARQ retry/result behavior is disabled and ARQ imports remain confined to dispatcher.py/worker.py.
- [ ] Handler final writes validate the lease inside the same transaction as the business replacement.
- [ ] TUS offset advances only after S3 returns an ETag and finalization is idempotent.
- [ ] Quota combines committed Postgres bytes with live Redis reservations and releases once.
- [ ] API and worker readiness checks their own dependencies; Local mode has no Hosted dependency.
- [ ] Two API and two worker replicas pass HTTP, WebSocket, upload, extraction, graph, crash recovery, and shutdown smoke tests.
- [ ] Documentation, Compose config, migration mirror, dependency lock, and CI agree with the implemented behavior.
