"""Hosted graph rebuilds are durable, tenant scoped, and lease fenced."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import FastAPI
from jobs import repository
from jobs.handlers import (
    RetryableJobError,
    TerminalJobError,
    WorkerContext,
    handle_graph_rebuild,
)
from jobs.models import JobCancelled, JobCreate, JobRecord, JobType, LeaseLost
from jobs.service import JobService

from tests.helpers.jwt import auth_headers, seed_jwks_cache


async def _seed_tenant(pool) -> tuple[UUID, UUID]:
    user_id = uuid4()
    kb_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@durable-graph.test",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        kb_id,
        user_id,
        f"KB {kb_id}",
        f"kb-{kb_id}",
    )
    return user_id, kb_id


async def _seed_document(
    pool,
    user_id: UUID,
    kb_id: UUID,
    *,
    filename: str,
    path: str,
    source_kind: str,
    content: str | None = None,
    metadata: str | None = None,
) -> UUID:
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, source_kind, file_type, "
        "status, content, metadata) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, 'ready', $8, $9::jsonb)",
        document_id,
        kb_id,
        user_id,
        filename,
        path,
        source_kind,
        filename.rsplit(".", 1)[-1],
        content,
        metadata,
    )
    return document_id


async def _seed_graph_job(pool, user_id: UUID, kb_id: UUID, *, owner: str = "graph-worker") -> JobRecord:
    command = JobCreate(
        job_type=JobType.GRAPH_REBUILD,
        user_id=user_id,
        knowledge_base_id=kb_id,
        payload={"knowledge_base_id": str(kb_id)},
    )
    async with pool.acquire() as conn, conn.transaction():
        created = await repository.create(conn, command)
    claimed = await repository.claim(pool, created.id, owner, 120)
    assert claimed is not None
    return claimed


def _worker_context(pool) -> WorkerContext:
    return WorkerContext(
        pool=pool,
        s3=None,
        converter_url="http://converter.invalid",
        converter_secret="not-used",
    )


class RecordingLease:
    def __init__(self, failure: Exception | None = None, *, fail_connection_call: int | None = None):
        self.failure = failure
        self.fail_connection_call = fail_connection_call
        self.calls = 0
        self.connection_calls = 0

    async def checkpoint(self, conn=None):
        self.calls += 1
        if conn is not None:
            assert conn.is_in_transaction()
            self.connection_calls += 1
            if self.failure is not None and self.connection_calls == self.fail_connection_call:
                raise self.failure


def _api(pool, service: JobService) -> FastAPI:
    from routes.graph import router as graph_router
    from routes.jobs import router as jobs_router

    app = FastAPI()
    app.state.pool = pool
    app.state.auth_provider = None
    app.state.job_service = service
    app.include_router(graph_router)
    app.include_router(jobs_router)
    return app


@pytest.fixture
async def graph_api_clients(pool, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", True)
    seed_jwks_cache()
    clients = []
    for _ in range(2):
        app = _api(pool, JobService(pool))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clients.append((app, client))
    try:
        yield clients
    finally:
        await asyncio.gather(*(client.aclose() for _, client in clients))


@pytest.mark.asyncio
async def test_hosted_rebuild_returns_202_and_persists_minimal_graph_job(pool, graph_api_clients):
    user_id, kb_id = await _seed_tenant(pool)
    _, client = graph_api_clients[0]

    response = await client.post(
        f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
        headers=auth_headers(user_id),
    )

    assert response.status_code == 202
    assert set(response.json()) == {"job_id"}
    job_id = UUID(response.json()["job_id"])
    row = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", job_id)
    assert row["job_type"] == "graph.rebuild"
    assert row["user_id"] == user_id
    assert row["knowledge_base_id"] == kb_id
    assert row["document_id"] is None
    assert json.loads(row["payload"]) == {"knowledge_base_id": str(kb_id)}
    assert row["idempotency_key"] is None


@pytest.mark.asyncio
async def test_two_api_instances_return_one_active_graph_job(pool, graph_api_clients):
    user_id, kb_id = await _seed_tenant(pool)
    (_, client_a), (_, client_b) = graph_api_clients

    first, second = await asyncio.gather(
        client_a.post(
            f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
            headers=auth_headers(user_id),
        ),
        client_b.post(
            f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
            headers=auth_headers(user_id),
        ),
    )

    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id = $1 AND knowledge_base_id = $2 "
            "AND job_type = 'graph.rebuild' AND state IN ('queued', 'running', 'retry_wait')",
            user_id,
            kb_id,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_terminal_graph_job_does_not_permanently_deduplicate_future_rebuild(pool, graph_api_clients):
    user_id, kb_id = await _seed_tenant(pool)
    _, client = graph_api_clients[0]
    url = f"/v1/knowledge-bases/{kb_id}/graph/rebuild"

    first = await client.post(url, headers=auth_headers(user_id))
    first_id = UUID(first.json()["job_id"])
    await pool.execute("UPDATE background_jobs SET state = 'succeeded' WHERE id = $1", first_id)
    second = await client.post(url, headers=auth_headers(user_id))

    assert second.status_code == 202
    assert UUID(second.json()["job_id"]) != first_id
    rows = await pool.fetch(
        "SELECT idempotency_key FROM background_jobs WHERE user_id = $1 "
        "AND knowledge_base_id = $2 AND job_type = 'graph.rebuild' ORDER BY created_at",
        user_id,
        kb_id,
    )
    assert len(rows) == 2
    assert all(row["idempotency_key"] is None for row in rows)


@pytest.mark.asyncio
async def test_repeated_request_never_rewrites_active_graph_job_contract(pool, graph_api_clients):
    user_id, kb_id = await _seed_tenant(pool)
    _, client = graph_api_clients[0]
    url = f"/v1/knowledge-bases/{kb_id}/graph/rebuild"
    first = await client.post(url, headers=auth_headers(user_id))
    job_id = UUID(first.json()["job_id"])
    await pool.execute(
        "UPDATE background_jobs SET payload = $2::jsonb, attempt_count = 1 WHERE id = $1",
        job_id,
        '{"knowledge_base_id":"immutable","marker":"kept"}',
    )

    repeated = await client.post(url, headers=auth_headers(user_id))

    assert repeated.status_code == 202
    assert UUID(repeated.json()["job_id"]) == job_id
    row = await pool.fetchrow("SELECT user_id, payload, attempt_count FROM background_jobs WHERE id = $1", job_id)
    assert row["user_id"] == user_id
    assert json.loads(row["payload"]) == {"knowledge_base_id": "immutable", "marker": "kept"}
    assert row["attempt_count"] == 1


@pytest.mark.asyncio
async def test_rebuild_authentication_precedes_service_availability(pool, graph_api_clients):
    user_id, kb_id = await _seed_tenant(pool)
    app, client = graph_api_clients[0]
    app.state.job_service = None

    unauthenticated = await client.post(f"/v1/knowledge-bases/{kb_id}/graph/rebuild")
    authenticated = await client.post(
        f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
        headers=auth_headers(user_id),
    )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 503


@pytest.mark.asyncio
async def test_non_uuid_subject_is_rejected_before_job_service_dependency(pool, graph_api_clients):
    import deps

    _, kb_id = await _seed_tenant(pool)
    app, client = graph_api_clients[0]
    service_calls = 0

    async def fail_if_service_runs():
        nonlocal service_calls
        service_calls += 1
        raise AssertionError("job service dependency ran before UUID authentication")

    app.dependency_overrides[deps.get_job_service] = fail_if_service_runs
    try:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
            headers=auth_headers("not-a-uuid"),
        )
    finally:
        app.dependency_overrides.pop(deps.get_job_service, None)

    assert response.status_code == 401
    assert service_calls == 0
    assert "not-a-uuid" not in response.text


@pytest.mark.asyncio
async def test_hosted_rollout_flag_false_returns_503_without_synchronous_rebuild(
    pool,
    graph_api_clients,
    monkeypatch,
):
    from config import settings

    user_id, kb_id = await _seed_tenant(pool)
    _, client = graph_api_clients[0]
    monkeypatch.setattr(settings, "DURABLE_JOBS_ENABLED", False)

    response = await client.post(
        f"/v1/knowledge-bases/{kb_id}/graph/rebuild",
        headers=auth_headers(user_id),
    )

    assert response.status_code == 503
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id = $1 AND knowledge_base_id = $2",
            user_id,
            kb_id,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_cross_tenant_rebuild_and_job_access_are_indistinguishable_from_missing(
    pool,
    graph_api_clients,
):
    owner_id, owner_kb_id = await _seed_tenant(pool)
    attacker_id, _ = await _seed_tenant(pool)
    _, client = graph_api_clients[0]
    created = await client.post(
        f"/v1/knowledge-bases/{owner_kb_id}/graph/rebuild",
        headers=auth_headers(owner_id),
    )
    job_id = created.json()["job_id"]

    forbidden_create = await client.post(
        f"/v1/knowledge-bases/{owner_kb_id}/graph/rebuild",
        headers=auth_headers(attacker_id),
    )
    missing_create = await client.post(
        f"/v1/knowledge-bases/{uuid4()}/graph/rebuild",
        headers=auth_headers(attacker_id),
    )
    forbidden_get = await client.get(f"/v1/jobs/{job_id}", headers=auth_headers(attacker_id))
    forbidden_cancel = await client.post(
        f"/v1/jobs/{job_id}/cancel",
        headers=auth_headers(attacker_id),
    )

    assert forbidden_create.status_code == missing_create.status_code == 404
    assert forbidden_create.json() == missing_create.json()
    assert forbidden_get.status_code == forbidden_cancel.status_code == 404
    assert await pool.fetchval("SELECT state FROM background_jobs WHERE id = $1", UUID(job_id)) == "queued"


@pytest.mark.asyncio
async def test_graph_handler_fences_edge_and_facet_replacement_in_one_transaction(pool):
    user_id, kb_id = await _seed_tenant(pool)
    corpus_id = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="entry.md",
        path="/corpus/S2-G1/",
        source_kind="source",
        metadata='{"entry_id":"E-1","stage":"S2","domain":"G1"}',
    )
    wiki_id = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="page.md",
        path="/wiki/",
        source_kind="wiki",
        content="A claim.\n\n[^1]: entry.md, p.7",
        metadata="{}",
    )
    curated_target = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="curated.md",
        path="/wiki/",
        source_kind="wiki",
        content="curated",
    )
    await pool.execute(
        "INSERT INTO document_references "
        "(source_document_id, target_document_id, knowledge_base_id, reference_type) "
        "VALUES ($1, $2, $3, 'is_a')",
        wiki_id,
        curated_target,
        kb_id,
    )
    job = await _seed_graph_job(pool, user_id, kb_id)
    lease = RecordingLease()

    result = await handle_graph_rebuild(job, lease, _worker_context(pool))

    assert result == {"citations": 1, "links": 0, "facet_rollups": 1}
    assert lease.connection_calls == 2
    rows = await pool.fetch(
        "SELECT target_document_id, reference_type, page FROM document_references "
        "WHERE source_document_id = $1 ORDER BY reference_type",
        wiki_id,
    )
    assert [(row["target_document_id"], row["reference_type"], row["page"]) for row in rows] == [
        (corpus_id, "cites", 7),
        (curated_target, "is_a", None),
    ]
    rollup = json.loads(await pool.fetchval("SELECT metadata->'facet_rollup' FROM documents WHERE id = $1", wiki_id))
    assert rollup["entry_count"] == 1
    assert rollup["stage"] == ["S2"]


@pytest.mark.asyncio
async def test_final_lease_loss_preserves_previous_edges_and_facet_rollups(pool):
    user_id, kb_id = await _seed_tenant(pool)
    old_target = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="old.md",
        path="/corpus/old/",
        source_kind="source",
        metadata='{"entry_id":"OLD","stage":"S1"}',
    )
    new_target = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="new.md",
        path="/corpus/new/",
        source_kind="source",
        metadata='{"entry_id":"NEW","stage":"S2"}',
    )
    wiki_id = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="page.md",
        path="/wiki/",
        source_kind="wiki",
        content="New graph.\n\n[^1]: new.md, p.2",
        metadata='{"facet_rollup":{"stage":["S1"],"entry_count":1}}',
    )
    await pool.execute(
        "INSERT INTO document_references "
        "(source_document_id, target_document_id, knowledge_base_id, reference_type, page) "
        "VALUES ($1, $2, $3, 'cites', 9)",
        wiki_id,
        old_target,
        kb_id,
    )
    job = await _seed_graph_job(pool, user_id, kb_id)
    lease = RecordingLease(LeaseLost("stale"), fail_connection_call=2)

    with pytest.raises(LeaseLost):
        await handle_graph_rebuild(job, lease, _worker_context(pool))

    assert (
        await pool.fetchval(
            "SELECT target_document_id FROM document_references "
            "WHERE source_document_id = $1 AND reference_type = 'cites'",
            wiki_id,
        )
        == old_target
    )
    metadata = json.loads(await pool.fetchval("SELECT metadata FROM documents WHERE id = $1", wiki_id))
    assert metadata["facet_rollup"] == {"stage": ["S1"], "entry_count": 1}
    assert new_target != old_target


@pytest.mark.asyncio
async def test_stale_worker_cannot_overwrite_graph_committed_by_newer_job(pool, monkeypatch):
    import services.graph as graph_service

    from llmwiki_core.references import ReferenceEdge

    user_id, kb_id = await _seed_tenant(pool)
    target = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="new.md",
        path="/wiki/",
        source_kind="wiki",
        content="target",
    )
    stale_target = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="stale.md",
        path="/wiki/",
        source_kind="wiki",
        content="stale target",
    )
    wiki_id = await _seed_document(
        pool,
        user_id,
        kb_id,
        filename="page.md",
        path="/wiki/",
        source_kind="wiki",
        content="[new](./new.md)",
    )
    stale_job = await _seed_graph_job(pool, user_id, kb_id, owner="stale-worker")
    await pool.execute(
        "UPDATE background_jobs SET state = 'failed', lease_owner = NULL, lease_expires_at = NULL WHERE id = $1",
        stale_job.id,
    )
    newer_job = await _seed_graph_job(pool, user_id, kb_id, owner="new-worker")
    async with pool.acquire() as conn:
        await graph_service.rebuild_hosted(
            conn,
            kb_id,
            str(user_id),
            before_write=lambda tx: repository.assert_active(tx, newer_job.id, "new-worker"),
        )

    def stale_snapshot(*args, **kwargs):
        return [ReferenceEdge(str(stale_target), "links_to")]

    monkeypatch.setattr(graph_service, "extract_references", stale_snapshot)

    async def stale_checkpoint(conn):
        assert conn.is_in_transaction()
        await repository.assert_active(conn, stale_job.id, "stale-worker")

    async with pool.acquire() as conn:
        with pytest.raises(LeaseLost):
            await graph_service.rebuild_hosted(conn, kb_id, str(user_id), before_write=stale_checkpoint)
    assert (
        await pool.fetchval(
            "SELECT target_document_id FROM document_references "
            "WHERE source_document_id = $1 AND reference_type = 'links_to'",
            wiki_id,
        )
        == target
    )
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM document_references "
        "WHERE source_document_id = $1 AND target_document_id = $2 AND reference_type = 'links_to')",
        wiki_id,
        stale_target,
    )


@pytest.mark.asyncio
async def test_graph_handler_rejects_invalid_or_cross_tenant_jobs_before_compute(pool, monkeypatch):
    import services.graph as graph_service

    owner_id, owner_kb_id = await _seed_tenant(pool)
    _, other_kb_id = await _seed_tenant(pool)
    valid = await _seed_graph_job(pool, owner_id, owner_kb_id)
    compute_calls = 0

    async def should_not_compute(*args, **kwargs):
        nonlocal compute_calls
        compute_calls += 1
        raise AssertionError("graph compute crossed validation boundary")

    monkeypatch.setattr(graph_service, "rebuild_hosted", should_not_compute)
    invalid_jobs = [
        (replace(valid, payload={}), "invalid_graph_job"),
        (
            replace(valid, payload={"knowledge_base_id": str(owner_kb_id), "secret": "x"}),
            "invalid_graph_job",
        ),
        (replace(valid, payload={"knowledge_base_id": str(uuid4())}), "invalid_graph_job"),
        (replace(valid, document_id=uuid4()), "invalid_graph_job"),
        (replace(valid, job_type=JobType.DOCUMENT_EXTRACT), "invalid_graph_job"),
        (
            replace(valid, knowledge_base_id=other_kb_id, payload={"knowledge_base_id": str(other_kb_id)}),
            "knowledge_base_not_found",
        ),
        (
            replace(
                valid,
                knowledge_base_id=(missing_kb_id := uuid4()),
                payload={"knowledge_base_id": str(missing_kb_id)},
            ),
            "knowledge_base_not_found",
        ),
    ]

    for invalid, expected_code in invalid_jobs:
        with pytest.raises(TerminalJobError) as raised:
            await handle_graph_rebuild(invalid, RecordingLease(), _worker_context(pool))
        assert raised.value.error_code == expected_code

    assert compute_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [JobCancelled("cancelled"), LeaseLost("lost")])
async def test_graph_handler_propagates_cancellation_and_lease_loss(pool, failure):
    user_id, kb_id = await _seed_tenant(pool)
    job = await _seed_graph_job(pool, user_id, kb_id)
    lease = RecordingLease(failure, fail_connection_call=1)

    with pytest.raises(type(failure)):
        await handle_graph_rebuild(job, lease, _worker_context(pool))


@pytest.mark.asyncio
async def test_graph_handler_retries_operational_postgres_errors_without_leaking_details(pool, monkeypatch):
    import services.graph as graph_service

    user_id, kb_id = await _seed_tenant(pool)
    job = await _seed_graph_job(pool, user_id, kb_id)

    async def fail_operationally(*args, **kwargs):
        raise asyncpg.CannotConnectNowError("password=private payload=secret")

    monkeypatch.setattr(graph_service, "rebuild_hosted", fail_operationally)

    with pytest.raises(RetryableJobError) as raised:
        await handle_graph_rebuild(job, RecordingLease(), _worker_context(pool))

    assert raised.value.error_code == "graph_transient"
    assert "private" not in raised.value.error_message
    assert "secret" not in raised.value.error_message
