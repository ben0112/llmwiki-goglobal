"""Transport contracts for the hosted server-side RAG REST API."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from jobs.models import JobRecord, JobState, JobType
from rag.records import RagRunRecord, RagStepRecord

from llmwiki_core.rag import (
    RAG_ERROR_CONTRACTS,
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagDomainError,
    RagStepStatus,
    RagStepType,
    RagUsage,
)

USER_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
KB_ID = UUID("11111111-1111-1111-1111-111111111111")
RUN_ID = UUID("33333333-3333-3333-3333-333333333333")
JOB_ID = UUID("44444444-4444-4444-4444-444444444444")
NOW = datetime(2026, 7, 26, 12, tzinfo=UTC)


class HostileUUID(UUID):
    equality_calls = 0

    def __eq__(self, other):
        type(self).equality_calls += 1
        raise AssertionError("PRIVATE UUID equality must not execute")

    __hash__ = UUID.__hash__


def _run_record(**changes) -> RagRunRecord:
    values = {
        "id": RUN_ID,
        "job_id": JOB_ID,
        "root_run_id": RUN_ID,
        "parent_run_id": None,
        "user_id": USER_ID,
        "knowledge_base_id": KB_ID,
        "goal": "PRIVATE GOAL SHOULD NEVER CROSS HTTP",
        "goal_digest": hashlib.sha256(b"PRIVATE GOAL SHOULD NEVER CROSS HTTP").hexdigest(),
        "target_path_prefix": "/wiki/private/",
        "model_profile": "primary",
        "model_profile_version": "profile-v1",
        "retrieval_profile": "lexical",
        "dry_run": False,
        "budget": RagBudget(),
        "usage": RagUsage(steps=2, model_tokens=30),
        "idempotency_key": "private-idempotency-key",
        "request_digest": "b" * 64,
        "completion_reason": None,
        "last_committed_ordinal": -1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return RagRunRecord(**values)


def _step_record(sequence: int, **changes) -> RagStepRecord:
    values = {
        "id": UUID(int=100 + sequence),
        "run_id": RUN_ID,
        "run_page_id": UUID(int=200 + sequence),
        "user_id": USER_ID,
        "knowledge_base_id": KB_ID,
        "sequence": sequence,
        "step_type": RagStepType.RETRIEVE,
        "status": RagStepStatus.SUCCEEDED,
        "input_digest": "c" * 64,
        "output_summary": MappingProxyType({"private_path": "/wiki/private/page.md", "goal": "PRIVATE GOAL"}),
        "citation_identities": (
            RagCitation(document_id=UUID(int=300 + sequence), document_version=1, chunk_index=0, page=1),
        ),
        "prompt_version": "prompt-v1",
        "prompt_digest": "d" * 64,
        "model_profile_version": "profile-v1",
        "reserved_tokens": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "latency_ms": 12.5,
        "error_code": None,
        "error_message": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return RagStepRecord(**values)


class FakeRagService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.run = _run_record()
        self.resume_record = _run_record(id=UUID(int=999), parent_run_id=RUN_ID, job_id=UUID(int=998))
        self.step_rows = (_step_record(8), _step_record(9))
        self.failure: BaseException | None = None

    async def create(self, command, *, authenticated_user_id, idempotency_key):
        self.calls.append(("create", command, authenticated_user_id, idempotency_key))
        if self.failure:
            raise self.failure
        return self.run

    async def get(self, run_id, *, authenticated_user_id):
        self.calls.append(("get", run_id, authenticated_user_id))
        if self.failure:
            raise self.failure
        return self.run

    async def steps(self, run_id, *, authenticated_user_id, after_sequence, limit):
        self.calls.append(("steps", run_id, authenticated_user_id, after_sequence, limit))
        if self.failure:
            raise self.failure
        return self.step_rows

    async def resume(self, run_id, *, authenticated_user_id, idempotency_key, budget_override):
        self.calls.append(("resume", run_id, authenticated_user_id, idempotency_key, budget_override))
        if self.failure:
            raise self.failure
        return self.resume_record


class FakeJobService:
    def __init__(self) -> None:
        self.state = JobState.QUEUED
        self.record = None
        self.failure = None

    async def get(self, job_id, *, authenticated_user_id):
        if self.failure:
            raise self.failure
        run_id = RUN_ID if job_id == JOB_ID else UUID(int=999)
        return self.record or JobRecord(
            id=job_id,
            job_type=JobType.BUILD_WIKI,
            user_id=authenticated_user_id,
            knowledge_base_id=KB_ID,
            state=self.state,
            payload={"run_id": str(run_id)},
            created_at=NOW,
            updated_at=NOW,
        )


@pytest.fixture
async def rag_client():
    import deps
    from routes.rag import router

    rag_service = FakeRagService()
    job_service = FakeJobService()
    application = FastAPI()
    application.include_router(router)
    application.dependency_overrides[deps.get_user_id] = lambda: str(USER_ID)
    application.dependency_overrides[deps.get_rag_service] = lambda: rag_service
    application.dependency_overrides[deps.get_job_service] = lambda: job_service
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, application, rag_service, job_service


def _create_body(**changes):
    body = {
        "knowledge_base_id": str(KB_ID),
        "goal": "Build launch pages",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
    }
    body.update(changes)
    return body


@pytest.mark.asyncio
async def test_create_returns_accepted_links_and_applies_defaults(rag_client):
    client, _app, service, _jobs = rag_client

    response = await client.post(
        "/v1/rag/build-wiki",
        headers={"Idempotency-Key": "create-1"},
        json=_create_body(),
    )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": str(RUN_ID),
        "job_id": str(JOB_ID),
        "state": "queued",
        "run_url": f"/v1/rag/runs/{RUN_ID}",
        "job_url": f"/v1/jobs/{JOB_ID}",
    }
    _kind, command, user_id, key = service.calls[-1]
    assert user_id == USER_ID
    assert key == "create-1"
    assert command.retrieval_profile == "lexical"
    assert command.dry_run is False
    assert asdict(command.budget) == asdict(RagBudget())
    assert "Build launch pages" not in response.text


@pytest.mark.asyncio
async def test_create_rejects_exact_run_subclass_and_post_create_job_failure_without_leak(rag_client):
    client, _app, service, jobs = rag_client

    class RunSubclass(RagRunRecord):
        pass

    record = service.run
    service.run = RunSubclass(**{item.name: getattr(record, item.name) for item in fields(RagRunRecord)})
    response = await client.post(
        "/v1/rag/build-wiki",
        headers={"Idempotency-Key": "hostile-run"},
        json=_create_body(),
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"

    service.run = record
    jobs.failure = RuntimeError("postgresql://private:secret@database")
    response = await client.post(
        "/v1/rag/build-wiki",
        headers={"Idempotency-Key": "job-failure"},
        json=_create_body(),
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"
    assert "private" not in response.text
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_service_dependencies_reject_rag_and_job_subclasses(monkeypatch):
    import deps
    from config import settings
    from jobs.service import JobService
    from rag.service import RagService

    class RagSubclass(RagService):
        pass

    class JobSubclass(JobService):
        pass

    state = SimpleNamespace(
        mode="hosted",
        rag_service=object.__new__(RagSubclass),
        job_service=object.__new__(JobSubclass),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(settings, "SERVER_RAG_ENABLED", True)

    for dependency in (deps.get_rag_service, deps.get_job_service):
        with pytest.raises(Exception) as raised:
            await dependency(request)
        assert raised.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "changes"),
    [
        ({}, {}),
        ({"Idempotency-Key": "create-1"}, {"provider_endpoint": "https://private.invalid"}),
        ({"Idempotency-Key": "create-1"}, {"budget": {"max_pages": 33}}),
        ({"Idempotency-Key": "create-1"}, {"dry_run": 1}),
    ],
)
async def test_create_rejects_missing_key_extra_fields_and_invalid_caps(rag_client, headers, changes):
    client, _app, service, _jobs = rag_client

    response = await client.post("/v1/rag/build-wiki", headers=headers, json=_create_body(**changes))

    assert response.status_code == 422
    assert service.calls == []
    assert "private.invalid" not in response.text
    assert "Build launch pages" not in response.text


@pytest.mark.asyncio
async def test_create_translates_explicit_bounded_budget(rag_client):
    client, _app, service, _jobs = rag_client

    response = await client.post(
        "/v1/rag/build-wiki",
        headers={"Idempotency-Key": "create-budget"},
        json=_create_body(
            retrieval_profile="hybrid",
            dry_run=True,
            budget={"max_pages": 2, "max_model_tokens": 1000},
        ),
    )

    assert response.status_code == 202
    command = service.calls[-1][1]
    assert command.retrieval_profile == "hybrid"
    assert command.dry_run is True
    assert command.budget.max_pages == 2
    assert command.budget.max_model_tokens == 1000
    assert command.budget.max_steps == RagBudget().max_steps


@pytest.mark.asyncio
async def test_get_run_returns_only_public_digest_configuration_and_job_state(rag_client):
    client, _app, _service, jobs = rag_client
    jobs.state = JobState.RUNNING

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(RUN_ID)
    assert body["job_id"] == str(JOB_ID)
    assert body["state"] == "running"
    assert body["goal_digest"] == hashlib.sha256(b"PRIVATE GOAL SHOULD NEVER CROSS HTTP").hexdigest()
    assert body["budget"] == asdict(RagBudget())
    assert body["usage"] == {"steps": 2, "model_tokens": 30}
    assert body["completion_reason"] is None
    forbidden = (
        "PRIVATE GOAL",
        "/wiki/private/",
        "private-idempotency-key",
        "request_digest",
        "user_id",
        "provider_endpoint",
        "api_key",
    )
    for value in forbidden:
        assert value not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        _run_record(id=UUID(int=701)),
        _run_record(user_id=UUID(int=702)),
        _run_record(root_run_id=UUID(int=703), parent_run_id=None),
    ],
)
async def test_get_rejects_foreign_or_malformed_exact_run_records(rag_client, run):
    client, _app, service, _jobs = rag_client
    service.run = run

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        _run_record(dry_run=False, completion_reason=RagCompletionReason.DRY_RUN),
        _run_record(completion_reason=RagCompletionReason.NO_WORK, last_committed_ordinal=0),
    ],
)
async def test_get_rejects_completion_records_that_violate_durable_decoder_invariants(rag_client, run):
    client, _app, service, _jobs = rag_client
    service.run = run

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    contract = RAG_ERROR_CONTRACTS["internal_error"]
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": contract.code, "message": contract.public_message}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        _run_record(dry_run=True, completion_reason=RagCompletionReason.DRY_RUN),
        _run_record(completion_reason=RagCompletionReason.NO_WORK, last_committed_ordinal=-1),
    ],
)
async def test_get_accepts_completion_records_that_match_durable_decoder_invariants(rag_client, run):
    client, _app, service, _jobs = rag_client
    service.run = run

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["completion_reason"] == run.completion_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_changes",
    [
        {"user_id": UUID(int=801)},
        {"knowledge_base_id": UUID(int=802)},
        {"document_id": UUID(int=803)},
        {"payload": {"run_id": str(UUID(int=804))}},
        {"payload": {"run_id": str(RUN_ID), "goal": "PRIVATE JOB GOAL"}},
    ],
)
async def test_get_rejects_wrong_job_binding_without_private_projection(rag_client, job_changes):
    client, _app, _service, jobs = rag_client
    jobs.record = JobRecord(
        id=JOB_ID,
        job_type=JobType.BUILD_WIKI,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        payload={"run_id": str(RUN_ID)},
        created_at=NOW,
        updated_at=NOW,
    )
    jobs.record = replace(jobs.record, **job_changes)

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"
    assert "PRIVATE JOB GOAL" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", HostileUUID(str(JOB_ID))),
        ("user_id", HostileUUID(str(USER_ID))),
        ("knowledge_base_id", HostileUUID(str(KB_ID))),
    ],
)
async def test_get_rejects_hostile_uuid_subclass_job_bindings_without_executing_equality(rag_client, field, value):
    client, _app, _service, jobs = rag_client
    jobs.record = JobRecord(
        id=JOB_ID,
        job_type=JobType.BUILD_WIKI,
        user_id=USER_ID,
        knowledge_base_id=KB_ID,
        payload={"run_id": str(RUN_ID)},
        created_at=NOW,
        updated_at=NOW,
    )
    jobs.record = replace(jobs.record, **{field: value})
    HostileUUID.equality_calls = 0

    response = await client.get(f"/v1/rag/runs/{RUN_ID}")

    contract = RAG_ERROR_CONTRACTS["internal_error"]
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": contract.code, "message": contract.public_message}}
    assert HostileUUID.equality_calls == 0
    assert "PRIVATE UUID" not in response.text


@pytest.mark.asyncio
async def test_steps_are_bounded_cursor_paginated_and_strip_private_trace(rag_client):
    client, _app, service, _jobs = rag_client

    response = await client.get(f"/v1/rag/runs/{RUN_ID}/steps?after=7&limit=2")

    assert response.status_code == 200
    assert service.calls[-1] == ("steps", RUN_ID, USER_ID, 7, 2)
    body = response.json()
    assert body["next_cursor"] == 9
    assert [item["sequence"] for item in body["items"]] == [8, 9]
    assert set(body["items"][0]) == {
        "id",
        "run_id",
        "run_page_id",
        "knowledge_base_id",
        "sequence",
        "type",
        "status",
        "citation_identities",
        "model_profile_version",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "error_code",
        "created_at",
        "updated_at",
    }
    for private in ("PRIVATE GOAL", "/wiki/private/", "input_digest", "prompt", "reserved_tokens"):
        assert private not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        [_step_record(8)],
        (_step_record(1), _step_record(2), _step_record(3)),
        (_step_record(8), _step_record(8)),
        (_step_record(8, run_id=UUID(int=901)),),
        (_step_record(8, user_id=UUID(int=902)),),
        (_step_record(8, citation_identities=tuple(_step_record(8).citation_identities * 129)),),
    ],
)
async def test_steps_reject_oversized_or_malformed_service_projections(rag_client, rows):
    client, _app, service, _jobs = rag_client
    service.step_rows = rows

    response = await client.get(f"/v1/rag/runs/{RUN_ID}/steps?after=7&limit=2")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"
    assert "private" not in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["after=-1", "limit=0", "limit=101", "after=true", "after=513", "after=" + "9" * 10_000],
)
async def test_steps_reject_unbounded_or_malformed_cursor_before_service(rag_client, query):
    client, _app, service, _jobs = rag_client

    response = await client.get(f"/v1/rag/runs/{RUN_ID}/steps?{query}")

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.asyncio
async def test_service_control_signal_is_sanitized_and_propagated(rag_client):
    client, _app, service, _jobs = rag_client
    service.failure = asyncio.CancelledError("PRIVATE CONTROL")

    with pytest.raises(asyncio.CancelledError) as raised:
        await client.get(f"/v1/rag/runs/{RUN_ID}")

    assert str(raised.value) == ""


@pytest.mark.asyncio
async def test_resume_accepts_only_explicit_higher_budget_overrides(rag_client):
    client, _app, service, _jobs = rag_client

    response = await client.post(
        f"/v1/rag/runs/{RUN_ID}/resume",
        headers={"Idempotency-Key": "resume-1"},
        json={"budget": {"max_model_tokens": 100_000}},
    )

    assert response.status_code == 202
    assert response.json()["run_id"] == str(UUID(int=999))
    assert service.calls[-1] == ("resume", RUN_ID, USER_ID, "resume-1", {"max_model_tokens": 100_000})


@pytest.mark.asyncio
async def test_resume_rejects_wrong_parent_binding(rag_client):
    client, _app, service, _jobs = rag_client
    service.resume_record = _run_record(id=UUID(int=999), parent_run_id=UUID(int=997), job_id=UUID(int=998))

    response = await client.post(
        f"/v1/rag/runs/{RUN_ID}/resume",
        headers={"Idempotency-Key": "resume-wrong-parent"},
        json={},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rag_internal_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "headers", "body"),
    [
        ("GET", f"/v1/rag/runs/{RUN_ID}", {}, None),
        ("GET", f"/v1/rag/runs/{RUN_ID}/steps", {}, None),
        (
            "POST",
            "/v1/rag/build-wiki",
            {"Idempotency-Key": "bad-subject-create"},
            _create_body(),
        ),
        (
            "POST",
            f"/v1/rag/runs/{RUN_ID}/resume",
            {"Idempotency-Key": "bad-subject-resume"},
            {},
        ),
    ],
)
async def test_malformed_auth_subject_is_rejected_before_rag_or_job_service(rag_client, method, path, headers, body):
    import deps

    client, app, service, _jobs = rag_client
    service_calls = 0

    async def malformed_subject():
        return "not-a-uuid"

    async def fail_if_rag_service_runs():
        nonlocal service_calls
        service_calls += 1
        raise AssertionError("RAG service dependency ran before subject validation")

    app.dependency_overrides[deps.get_user_id] = malformed_subject
    app.dependency_overrides[deps.get_rag_service] = fail_if_rag_service_runs
    app.dependency_overrides[deps.get_job_service] = fail_if_rag_service_runs
    response = await client.request(method, path, headers=headers, json=body)

    assert response.status_code == 401
    assert service_calls == 0
    assert service.calls == []
    assert "not-a-uuid" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("contract_name", "expected_status"),
    [
        ("invalid_request", 400),
        ("idempotency_conflict", 409),
        ("model_profile_unavailable", 422),
        ("retrieval_profile_unavailable", 422),
        ("disabled", 503),
        ("internal_error", 503),
    ],
)
async def test_domain_errors_map_to_stable_public_status_and_canonical_message(
    rag_client, contract_name, expected_status
):
    client, _app, service, _jobs = rag_client
    contract = RAG_ERROR_CONTRACTS[contract_name]
    service.failure = RagDomainError(contract.code, "INTERNAL MESSAGE MUST NOT LEAK", contract.retryable)

    response = await client.post(
        "/v1/rag/build-wiki",
        headers={"Idempotency-Key": "failure"},
        json=_create_body(),
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": {"code": contract.code, "message": contract.public_message}}
    assert "INTERNAL MESSAGE" not in response.text


@pytest.mark.asyncio
async def test_disabled_dependency_returns_stable_503_after_authentication(monkeypatch):
    import deps
    from config import settings
    from routes.rag import router

    application = FastAPI()
    application.state.mode = "hosted"
    application.state.job_service = object()
    application.state.rag_service = None
    application.include_router(router)
    application.dependency_overrides[deps.get_user_id] = lambda: str(USER_ID)
    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(settings, "SERVER_RAG_ENABLED", False)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/rag/build-wiki",
            headers={"Idempotency-Key": "disabled"},
            json=_create_body(),
        )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "rag_disabled", "message": "Server-side RAG is disabled."}}


@pytest.mark.asyncio
async def test_hosted_lifespan_constructs_rag_service_after_job_service(monkeypatch):
    import asyncpg
    import auth
    import main
    from jobs.service import JobService
    from rag.service import RagService

    class Resource:
        async def close(self):
            return None

        async def aclose(self):
            return None

    class Listener:
        async def close(self):
            return None

    pool = Resource()

    async def no_op():
        return None

    async def finish_startup(app, supplied_pool):
        assert supplied_pool is pool
        assert isinstance(app.state.job_service, JobService)
        assert isinstance(app.state.rag_service, RagService)
        assert app.state.rag_service._job_service is app.state.job_service
        return Listener(), None

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "SERVER_RAG_ENABLED", True)
    monkeypatch.setattr(main.settings, "REDIS_URL", "redis://test")
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(main, "_start_hosted_quota_runtime", lambda *_args: _async_value((Resource(), object())))
    monkeypatch.setattr(main, "_finish_hosted_startup", finish_startup)
    application = SimpleNamespace(state=SimpleNamespace())

    async with main.lifespan(application):
        pass


@pytest.mark.asyncio
async def test_failed_hosted_startup_clears_rag_service(monkeypatch):
    import asyncpg
    import auth
    import main

    class Resource:
        async def close(self):
            return None

        async def aclose(self):
            return None

    pool = Resource()

    async def no_op():
        return None

    async def fail_startup(_app, _pool):
        raise RuntimeError("listener failed")

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "SERVER_RAG_ENABLED", True)
    monkeypatch.setattr(main.settings, "REDIS_URL", "redis://test")
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(main, "_start_hosted_quota_runtime", lambda *_args: _async_value((Resource(), object())))
    monkeypatch.setattr(main, "_finish_hosted_startup", fail_startup)
    application = SimpleNamespace(state=SimpleNamespace())

    with pytest.raises(RuntimeError, match="listener failed"):
        async with main.lifespan(application):
            pass

    assert application.state.rag_service is None


@pytest.mark.asyncio
async def test_job_service_initialization_failure_closes_pool_and_clears_rag(monkeypatch):
    import asyncpg
    import auth
    import jobs.service as job_service_module
    import main

    class Pool:
        def __init__(self):
            self.closes = 0

        async def close(self):
            self.closes += 1

    pool = Pool()

    async def no_op():
        return None

    class FailingJobService:
        def __init__(self, _pool):
            raise RuntimeError("PRIVATE JOB INIT")

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "SERVER_RAG_ENABLED", True)
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(job_service_module, "JobService", FailingJobService)
    application = SimpleNamespace(state=SimpleNamespace(rag_service=object()))

    with pytest.raises(RuntimeError, match="PRIVATE JOB INIT"):
        async with main.lifespan(application):
            pass

    assert application.state.rag_service is None
    assert pool.closes == 1


@pytest.mark.asyncio
async def test_listener_close_failure_still_closes_redis_pool_and_preserves_body_failure(monkeypatch):
    import asyncpg
    import auth
    import main

    class Resource:
        def __init__(self):
            self.closes = 0

        async def close(self):
            self.closes += 1

        async def aclose(self):
            self.closes += 1

    class Listener:
        async def close(self):
            raise RuntimeError("PRIVATE LISTENER CLOSE")

    pool = Resource()
    redis = Resource()

    async def no_op():
        return None

    async def finish_startup(_app, _pool):
        return Listener(), None

    monkeypatch.setattr(main.settings, "MODE", "hosted")
    monkeypatch.setattr(main.settings, "DURABLE_JOBS_ENABLED", True)
    monkeypatch.setattr(main.settings, "SERVER_RAG_ENABLED", True)
    monkeypatch.setattr(main.settings, "REDIS_URL", "redis://test")
    monkeypatch.setattr(auth, "prefetch_jwks", no_op)
    monkeypatch.setattr(asyncpg, "create_pool", lambda *args, **kwargs: _async_value(pool))
    monkeypatch.setattr(main, "_start_hosted_quota_runtime", lambda *_args: _async_value((redis, object())))
    monkeypatch.setattr(main, "_finish_hosted_startup", finish_startup)
    application = SimpleNamespace(state=SimpleNamespace())

    with pytest.raises(RuntimeError, match="PRIMARY BODY FAILURE"):
        async with main.lifespan(application):
            raise RuntimeError("PRIMARY BODY FAILURE")

    assert application.state.rag_service is None
    assert redis.closes == 1
    assert pool.closes == 1


def _async_value(value):
    async def result():
        return value

    return result()


def test_local_configuration_does_not_register_or_import_rag_router():
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.update(
        {
            "MODE": "local",
            "PYTHONPATH": str(root / "api"),
            "DURABLE_JOBS_ENABLED": "false",
            "SERVER_RAG_ENABLED": "false",
            "TUS_MULTIPART_ENABLED": "false",
            "REDIS_URL": "",
        }
    )
    probe = (
        "import sys; import main; "
        "paths={r.path for r in main.app.routes}; "
        "assert '/v1/rag/build-wiki' not in paths; "
        "assert 'routes.rag' not in sys.modules; "
        "assert 'rag.service' not in sys.modules"
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
