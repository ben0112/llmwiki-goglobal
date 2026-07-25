# Durable Jobs Quality Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quality review by making CI matrix execution fail closed, deriving worker telemetry from committed database transitions, and pinning MinIO images.

**Architecture:** A Python `run` command owns matrix generation and process execution so shell pipelines cannot hide failure or accept an empty segment. Repository mutations expose their committed `JobRecord` outcomes to the worker while preserving existing default reaper callers. Compose uses explicit, compatible MinIO release tags guarded by static tests and documentation.

**Tech Stack:** Python 3.11, pytest, asyncpg/PostgreSQL, GitHub Actions YAML, Docker Compose.

---

### Task 1: Fail-closed CI matrix runner

**Files:**
- Modify: `tests/helpers/ci_test_matrix.py`
- Modify: `tests/test_ci_matrix_contract.py`
- Modify: `.github/workflows/test.yml`

- [ ] Write tests that invoke the wished-for `run` command, force segment generation to raise or return empty, and assert non-zero exit without launching pytest.
- [ ] Run `pytest tests/test_ci_matrix_contract.py -v` and verify the new tests fail because `run` does not exist.
- [ ] Implement `run SEGMENT -- COMMAND...`, validate non-empty generated files, execute merged segments once and PostgreSQL-isolated segments once per file, and return the child failure code.
- [ ] Replace every workflow matrix pipeline with the safe runner and strengthen the workflow contract to reject `| xargs` and generator pipelines.
- [ ] Re-run `pytest tests/test_ci_matrix_contract.py -v` and verify green.

### Task 2: Persisted-transition telemetry

**Files:**
- Modify: `api/jobs/repository.py`
- Modify: `api/jobs/worker.py`
- Modify: `tests/unit/jobs/test_dispatcher.py`
- Modify: `tests/integration/test_background_job_leases.py`
- Modify: `tests/helpers/telemetry_contract.py`

- [ ] Write worker tests for retry, terminal failure, exhausted attempts, requested cancellation, shutdown cancellation, LeaseLost, and reaper retry/exhausted/cancelled outcomes using explicit `JobRecord` transitions.
- [ ] Run focused unit tests and verify they fail because telemetry currently infers state and error code.
- [ ] Make `_record_failure` return the persisted `JobRecord | None`; retain `fail_or_retry`'s existing `JobRecord` API; extend reaping with an explicit outcome mode while keeping existing UUID-list callers valid.
- [ ] Emit `durable_job_finished` and `durable_job_lease_reaped` only from returned records; emit nothing when no transition was persisted.
- [ ] Update exact JSON schemas and run dispatcher plus PostgreSQL lease tests.

### Task 3: Reproducible MinIO images

**Files:**
- Modify: `deploy/docker-compose.selfhost.yml`
- Modify: `tests/integration/test_scaled_compose.py`
- Modify: `docs/self-hosting.md`

- [ ] Add static assertions rejecting `latest` and requiring the selected server/client release tags.
- [ ] Run the static scaled test and verify RED against the current Compose file.
- [ ] Pin server to `RELEASE.2025-04-22T22-12-26Z`, select a published compatible `mc` release tag, and document both.
- [ ] Re-run scaled static tests and Compose config validation.

### Task 4: Verification and handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-durable-jobs-api-scaling-design.md`

- [ ] Run matrix contract, dispatcher unit tests, local PostgreSQL lease tests, scaled static tests, changed-path Ruff, and Compose validation.
- [ ] Record truthful local evidence while keeping design status incomplete pending independent re-review.
- [ ] Commit and push `feat/platform-architecture-evolution`.
- [ ] Wait for all six GitHub Actions jobs; hand the immutable SHA and run URL to the independent reviewer before restoring verified status.
