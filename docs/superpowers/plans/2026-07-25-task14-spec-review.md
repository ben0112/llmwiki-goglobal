# Task 14 Specification Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Task 14 specification gaps with real scaled failure tests, a complete isolated CI matrix, and stable structured telemetry contracts.

**Architecture:** Extend the required two-API/two-worker Compose smoke with observable Docker and database state transitions while preserving the existing SIGKILL path. Keep API and MCP tests in separate Python processes to avoid their legacy top-level module names sharing `sys.modules`. Exercise telemetry through production callsites and validate each emitted JSON record as a public contract.

**Tech Stack:** Python 3.11, pytest, asyncpg, httpx, Docker Compose, Redis 7.4 AOF, ARQ, GitHub Actions, Ruff.

---

### Task 1: Required scaled failure scenarios

**Files:**
- Modify: `tests/integration/test_scaled_compose.py`
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Write failing static contract tests**

Require the live test source to call helpers that restart Redis and wait for health, map an `x-api-instance-id` to an API container and kill it after final PATCH, and send `TERM` to the worker owning a blocked job while asserting its lease/claim/exit lifecycle.

```python
assert "_restart_redis_and_wait()" in source
assert "_api_container_for_instance(completed_instance)" in source
assert '["docker", "kill", "--signal", "TERM"' in source
assert "old_owner" in source
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=api .venv/bin/pytest tests/integration/test_scaled_compose.py -q
```

Expected: static assertions fail because the three real scenarios are absent.

- [ ] **Step 3: Add bounded Docker/database helpers and live assertions**

Implement container lookup using Compose container IDs plus `docker inspect` hostname, wait for Redis with `compose exec -T redis redis-cli ping`, and wait for API/worker scale and container exit using monotonic deadlines. After the first PATCH restart only Redis so its named AOF volume survives, then require the cross-replica HEAD offset and final PATCH. Kill the accepting API before polling the returned job through the gateway, then restore `api=2`. For graceful shutdown, block a graph job with a database lock, TERM its lease owner, create a second graph job, release the lock, and prove the old owner stops claiming, the first lease clears or completes, the container exits, and replacement workers finish outstanding work. Retain the existing KILL/reaper scenario.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=api .venv/bin/pytest tests/integration/test_scaled_compose.py -q
.venv/bin/ruff check tests/integration/test_scaled_compose.py
git add tests/integration/test_scaled_compose.py .github/workflows/test.yml
git commit -m "test: exercise scaled service failure lifecycle"
```

### Task 2: Complete isolated Python test matrix

**Files:**
- Modify: `.github/workflows/test.yml`
- Create: `tests/test_ci_matrix_contract.py`

- [ ] **Step 1: Reproduce collection pollution**

Run the raw aggregate commands and capture their top-level API/MCP module collisions, then enumerate all files with `rg --files tests/unit tests/integration`.

```bash
PYTHONPATH=api .venv/bin/pytest tests/unit --collect-only -q
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration --collect-only -q
```

- [ ] **Step 2: Write RED matrix coverage test**

Parse workflow commands, classify every test file into core, corpus, API, MCP SQLite, MCP Postgres, Redis, MinIO, or scaled segments, and fail with the exact omitted path set. The test must also require separate pytest processes for API and MCP roots.

- [ ] **Step 3: Replace selected lists with complete directory/file partitions**

Use generated complete path sets or directory partitions whose overlaps are explicitly excluded. Run each partition in its own process and working directory so API and MCP top-level modules never coexist in one interpreter.

- [ ] **Step 4: Verify collection and execution**

```bash
.venv/bin/pytest tests/test_ci_matrix_contract.py -q
# Run every exact segment printed in .github/workflows/test.yml and record pass/skip counts.
```

Expected: every test path is owned by exactly one CI segment and all segments collect without error.

### Task 3: Nine production telemetry contracts

**Files:**
- Create: `tests/unit/test_telemetry_contracts.py`
- Modify only if RED finds a contract defect: `api/telemetry.py`, `api/jobs/dispatcher.py`, `api/jobs/worker.py`, `api/infra/tus.py`, `api/infra/quota.py`

- [ ] **Step 1: Write one parameterized RED suite around real callsites**

Trigger or call the focused production methods for `durable_job_dispatched`, `durable_job_finished`, `durable_job_lease_reaped`, `tus_session_created`, `tus_session_completed`, `tus_session_stale`, `quota_reserved`, `quota_released`, and `upload_cleanup_finished`. Parse every `LogRecord.message` with `json.loads`.

```python
assert event["event"] == expected_name
assert event["replica_role"] in {"api", "worker"}
assert all(type(value) in {str, int, float, bool, type(None)} for value in event.values())
assert forbidden_keys.isdisjoint(event)
assert not sensitive_fragments.intersection(json.dumps(event).lower())
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_telemetry_contracts.py -q
```

Expected: missing or unstable real-callsite fields fail explicitly.

- [ ] **Step 3: Make the narrowest production corrections and run GREEN**

Keep IDs, bounded state/error codes, duration/count values, and replica role; never include payload, object URLs, credentials, database URLs, tokens, or raw error text.

```bash
PYTHONPATH=api .venv/bin/pytest tests/unit/test_telemetry_contracts.py -q
.venv/bin/ruff check api/telemetry.py api/jobs/dispatcher.py api/jobs/worker.py api/infra/tus.py api/infra/quota.py tests/unit/test_telemetry_contracts.py
```

### Task 4: Final evidence and branch verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md`

- [ ] **Step 1: Run all exact isolated segments and focused checks**

Record each command and its pass/skip count. Run repository-wide Ruff separately and record pre-existing debt honestly if it is not clean.

- [ ] **Step 2: Commit implementation, then backfill immutable evidence**

Replace provisional status with implemented/verified only after all local segments pass. Cite real implementation commit SHAs, exact commands/counts, and the successful Actions run without claiming the invalid raw aggregate commands passed.

- [ ] **Step 3: Push and monitor required CI**

```bash
git push origin feat/platform-architecture-evolution
gh run watch <run-id> -R ben0112/llmwiki-goglobal --exit-status
```

Expected: all six jobs pass, including the non-skipped scaled Compose live test.
