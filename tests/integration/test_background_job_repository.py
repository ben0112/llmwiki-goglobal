from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from jobs import repository
from jobs.models import JobCreate, JobRecord, JobState, JobType
from jobs.service import JobResourceNotFound, JobService


async def _seed_user(pool):
    user_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@job-repository.test",
    )
    return user_id


async def _seed_knowledge_base(pool, user_id):
    knowledge_base_id = uuid4()
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    return knowledge_base_id


async def _seed_document(pool, user_id, knowledge_base_id):
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, file_type) "
        "VALUES ($1, $2, $3, 'source.md', '/', 'md')",
        document_id,
        knowledge_base_id,
        user_id,
    )
    return document_id


async def _insert_job(pool, user_id, *, state="queued", **values):
    columns = ["user_id", "job_type", "state", *values]
    parameters = [user_id, "document.extract", state, *values.values()]
    placeholders = [f"${index}" for index in range(1, len(parameters) + 1)]
    casts = ["" if key not in {"payload", "progress", "result"} else "::jsonb" for key in columns]
    row = await pool.fetchrow(
        "INSERT INTO background_jobs ("
        + ", ".join(columns)
        + ") VALUES ("
        + ", ".join(value + cast for value, cast in zip(placeholders, casts, strict=True))
        + ") RETURNING *",
        *parameters,
    )
    return row


def _command(user_id, **changes):
    values = {
        "job_type": JobType.DOCUMENT_EXTRACT,
        "user_id": user_id,
        "payload": {"nested": {"items": ["one", 2]}},
    }
    values.update(changes)
    return JobCreate(**values)


@pytest.mark.asyncio
async def test_create_returns_complete_typed_record_with_defaults_and_nested_payload(pool):
    user_id = await _seed_user(pool)

    async with pool.acquire() as conn:
        record = await repository.create(conn, _command(user_id))

    assert isinstance(record, JobRecord)
    assert record.user_id == user_id
    assert record.job_type is JobType.DOCUMENT_EXTRACT
    assert record.state is JobState.QUEUED
    assert record.payload == {"nested": {"items": ("one", 2)}}
    assert record.progress is None
    assert record.result is None
    assert record.run_after.tzinfo is not None
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    assert record.attempt_count == 0
    assert record.dispatch_attempts == 0
    assert record.max_attempts == 3


@pytest.mark.asyncio
async def test_create_idempotency_returns_original_tenant_job_without_overwriting_fields(pool):
    user_id = await _seed_user(pool)
    kb_id = await _seed_knowledge_base(pool, user_id)
    document_id = await _seed_document(pool, user_id, kb_id)
    alternative_kb_id = await _seed_knowledge_base(pool, user_id)
    alternative_document_id = await _seed_document(pool, user_id, alternative_kb_id)
    original_run_after = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    lease_expires_at = datetime(2030, 1, 2, 4, 5, 6, tzinfo=UTC)
    heartbeat_at = datetime(2030, 1, 2, 3, 5, 6, tzinfo=UTC)
    first = _command(
        user_id,
        knowledge_base_id=kb_id,
        document_id=document_id,
        payload={"original": True},
        idempotency_key="key-1",
        max_attempts=4,
        run_after=original_run_after,
    )
    second = _command(
        user_id,
        knowledge_base_id=alternative_kb_id,
        document_id=alternative_document_id,
        payload={"replacement": True},
        idempotency_key="key-1",
        max_attempts=8,
        run_after=datetime(2040, 1, 1, tzinfo=UTC),
    )

    async with pool.acquire() as conn:
        original = await repository.create(conn, first)
        await pool.execute(
            "UPDATE background_jobs SET state = 'running', attempt_count = 2, dispatch_attempts = 5, "
            "lease_owner = 'worker-a', lease_expires_at = $1, heartbeat_at = $2 WHERE id = $3",
            lease_expires_at,
            heartbeat_at,
            original.id,
        )
        duplicate = await repository.create(conn, second)

    assert duplicate.id == original.id
    assert duplicate.payload == {"original": True}
    assert duplicate.knowledge_base_id == kb_id
    assert duplicate.document_id == document_id
    assert duplicate.max_attempts == 4
    assert duplicate.run_after == original_run_after
    assert duplicate.state is JobState.RUNNING
    assert duplicate.attempt_count == 2
    assert duplicate.dispatch_attempts == 5
    assert duplicate.lease_owner == "worker-a"
    assert duplicate.lease_expires_at == lease_expires_at
    assert duplicate.heartbeat_at == heartbeat_at


@pytest.mark.asyncio
async def test_create_idempotency_scope_distinguishes_tenants_types_and_null_keys(pool):
    user_a = await _seed_user(pool)
    user_b = await _seed_user(pool)

    async with pool.acquire() as conn:
        first = await repository.create(conn, _command(user_a, idempotency_key="shared"))
        other_tenant = await repository.create(conn, _command(user_b, idempotency_key="shared"))
        other_type = await repository.create(
            conn,
            _command(user_a, job_type=JobType.GRAPH_REBUILD, idempotency_key="shared"),
        )
        null_first = await repository.create(conn, _command(user_a))
        null_second = await repository.create(conn, _command(user_a))

    assert other_tenant.id != first.id
    assert other_type.id != first.id
    assert null_first.id != null_second.id


@pytest.mark.asyncio
async def test_concurrent_create_with_one_idempotency_key_returns_one_job(pool):
    user_id = await _seed_user(pool)
    command = _command(user_id, idempotency_key="concurrent-create")

    async with pool.acquire() as conn_a, pool.acquire() as conn_b:
        first, second = await asyncio.gather(
            repository.create(conn_a, command),
            repository.create(conn_b, command),
        )

    assert first.id == second.id
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id = $1 AND job_type = $2 AND idempotency_key = $3",
            user_id,
            JobType.DOCUMENT_EXTRACT.value,
            "concurrent-create",
        )
        == 1
    )


@pytest.mark.asyncio
async def test_get_for_user_returns_only_the_owner_row(pool):
    owner_id = await _seed_user(pool)
    other_id = await _seed_user(pool)
    async with pool.acquire() as conn:
        created = await repository.create(conn, _command(owner_id))
        assert (await repository.get_for_user(conn, created.id, owner_id)).id == created.id
        assert await repository.get_for_user(conn, created.id, other_id) is None
        assert await repository.get_for_user(conn, uuid4(), owner_id) is None


@pytest.mark.asyncio
async def test_cancel_changes_queued_retry_wait_and_running_with_stable_timestamp(pool):
    user_id = await _seed_user(pool)
    async with pool.acquire() as conn:
        queued = await repository.create(conn, _command(user_id))
        retry_wait = await _insert_job(pool, user_id, state="retry_wait")
        running = await _insert_job(pool, user_id, state="running")

        queued_cancelled = await repository.request_cancel(conn, queued.id, user_id)
        retry_cancelled = await repository.request_cancel(conn, retry_wait["id"], user_id)
        running_requested = await repository.request_cancel(conn, running["id"], user_id)
        queued_repeat = await repository.request_cancel(conn, queued.id, user_id)
        retry_repeat = await repository.request_cancel(conn, retry_wait["id"], user_id)
        running_repeat = await repository.request_cancel(conn, running["id"], user_id)

    assert queued_cancelled.state is JobState.CANCELLED
    assert retry_cancelled.state is JobState.CANCELLED
    assert running_requested.state is JobState.RUNNING
    assert queued_cancelled.cancel_requested_at is not None
    assert retry_cancelled.cancel_requested_at is not None
    assert running_requested.cancel_requested_at is not None
    assert queued_repeat.cancel_requested_at == queued_cancelled.cancel_requested_at
    assert retry_repeat.cancel_requested_at == retry_cancelled.cancel_requested_at
    assert running_repeat.cancel_requested_at == running_requested.cancel_requested_at


@pytest.mark.asyncio
async def test_repeated_running_cancel_leaves_the_complete_row_unchanged(pool):
    user_id = await _seed_user(pool)
    running = await _insert_job(pool, user_id, state="running", payload='{"keep": true}')

    async with pool.acquire() as conn:
        first = await repository.request_cancel(conn, running["id"], user_id)
        repeated = await repository.request_cancel(conn, running["id"], user_id)

    assert first.state is JobState.RUNNING
    assert first.cancel_requested_at is not None
    assert repeated == first


@pytest.mark.asyncio
async def test_cancel_keeps_terminal_rows_fully_unchanged_and_typed(pool):
    user_id = await _seed_user(pool)
    terminal_rows = []
    for state in ("succeeded", "failed", "cancelled"):
        terminal_rows.append(await _insert_job(pool, user_id, state=state, payload='{"keep": true}'))

    async with pool.acquire() as conn:
        results = [await repository.request_cancel(conn, row["id"], user_id) for row in terminal_rows]

    for original, result in zip(terminal_rows, results, strict=True):
        after = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", original["id"])
        assert isinstance(result, JobRecord)
        assert result.state.value == original["state"]
        assert dict(after) == dict(original)


@pytest.mark.asyncio
async def test_cancel_race_returns_terminal_row_without_mutating_it(pool):
    user_id = await _seed_user(pool)
    async with pool.acquire() as conn:
        created = await repository.create(conn, _command(user_id))
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    transaction_a = conn_a.transaction()
    await transaction_a.start()
    committed = False
    try:
        terminal = await conn_a.fetchrow(
            "UPDATE background_jobs SET state = 'succeeded' WHERE id = $1 RETURNING *",
            created.id,
        )
        pending_cancel = asyncio.create_task(repository.request_cancel(conn_b, created.id, user_id))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending_cancel), timeout=0.05)

        await transaction_a.commit()
        committed = True
        cancelled = await asyncio.wait_for(pending_cancel, timeout=1)
        after = await pool.fetchrow("SELECT * FROM background_jobs WHERE id = $1", created.id)
    finally:
        if not committed:
            await transaction_a.rollback()
        await pool.release(conn_b)
        await pool.release(conn_a)

    assert cancelled.state is JobState.SUCCEEDED
    assert cancelled.cancel_requested_at is None
    assert cancelled.updated_at == terminal["updated_at"]
    assert dict(after) == dict(terminal)


@pytest.mark.asyncio
async def test_cancel_never_crosses_tenants_and_missing_returns_none(pool):
    owner_id = await _seed_user(pool)
    other_id = await _seed_user(pool)
    async with pool.acquire() as conn:
        created = await repository.create(conn, _command(owner_id))
        assert await repository.request_cancel(conn, created.id, other_id) is None
        assert await repository.request_cancel(conn, uuid4(), owner_id) is None

    unchanged = await pool.fetchrow("SELECT state, cancel_requested_at FROM background_jobs WHERE id = $1", created.id)
    assert dict(unchanged) == {"state": "queued", "cancel_requested_at": None}


@pytest.mark.asyncio
async def test_service_rejects_mismatched_user_and_non_owned_resources_without_creating_job(pool):
    user_a = await _seed_user(pool)
    user_b = await _seed_user(pool)
    service = JobService(pool)
    other_kb = await _seed_knowledge_base(pool, user_b)
    missing_kb = uuid4()
    other_document = await _seed_document(pool, user_b, other_kb)
    missing_document = uuid4()
    job_count_before = await pool.fetchval("SELECT count(*) FROM background_jobs")

    with pytest.raises(ValueError):
        await service.create(_command(user_b), authenticated_user_id=user_a)
    for command in (
        _command(user_a, knowledge_base_id=missing_kb),
        _command(user_a, knowledge_base_id=other_kb),
        _command(user_a, document_id=other_document),
        _command(user_a, document_id=missing_document),
    ):
        with pytest.raises(JobResourceNotFound):
            await service.create(command, authenticated_user_id=user_a)

    assert await pool.fetchval("SELECT count(*) FROM background_jobs") == job_count_before


@pytest.mark.asyncio
async def test_service_rejects_document_only_when_its_knowledge_base_has_another_owner(pool):
    user_a = await _seed_user(pool)
    user_b = await _seed_user(pool)
    other_kb = await _seed_knowledge_base(pool, user_b)
    inconsistent_document = await _seed_document(pool, user_a, other_kb)
    service = JobService(pool)
    job_count_before = await pool.fetchval("SELECT count(*) FROM background_jobs")

    with pytest.raises(JobResourceNotFound):
        await service.create(
            _command(user_a, document_id=inconsistent_document),
            authenticated_user_id=user_a,
        )

    assert await pool.fetchval("SELECT count(*) FROM background_jobs") == job_count_before


@pytest.mark.asyncio
async def test_service_rejects_document_knowledge_base_mismatch_for_the_same_tenant(pool):
    user_id = await _seed_user(pool)
    kb_a = await _seed_knowledge_base(pool, user_id)
    kb_b = await _seed_knowledge_base(pool, user_id)
    document_b = await _seed_document(pool, user_id, kb_b)
    service = JobService(pool)
    job_count_before = await pool.fetchval("SELECT count(*) FROM background_jobs")

    with pytest.raises(JobResourceNotFound):
        await service.create(
            _command(user_id, knowledge_base_id=kb_a, document_id=document_b),
            authenticated_user_id=user_id,
        )

    assert await pool.fetchval("SELECT count(*) FROM background_jobs") == job_count_before


@pytest.mark.asyncio
async def test_service_creates_and_delegates_scoped_get_and_cancel(pool):
    user_id = await _seed_user(pool)
    kb_id = await _seed_knowledge_base(pool, user_id)
    document_id = await _seed_document(pool, user_id, kb_id)
    service = JobService(pool)

    created = await service.create(
        _command(user_id, knowledge_base_id=kb_id, document_id=document_id),
        authenticated_user_id=user_id,
    )
    fetched = await service.get(created.id, authenticated_user_id=user_id)
    cancelled = await service.cancel(created.id, authenticated_user_id=user_id)

    assert fetched == created
    assert cancelled.state is JobState.CANCELLED


def test_row_mapper_handles_nullable_columns_and_rejects_non_object_json():
    row = {
        "id": str(uuid4()),
        "job_type": "document.extract",
        "user_id": str(uuid4()),
        "state": "queued",
        "knowledge_base_id": None,
        "document_id": None,
        "payload": '{"nested": [1]}',
        "progress": None,
        "result": None,
        "idempotency_key": None,
        "attempt_count": 0,
        "max_attempts": 3,
        "run_after": datetime(2030, 1, 1, tzinfo=UTC),
        "lease_owner": None,
        "lease_expires_at": None,
        "heartbeat_at": None,
        "last_dispatched_at": None,
        "dispatch_attempts": 0,
        "error_code": None,
        "error_message": None,
        "cancel_requested_at": None,
        "created_at": datetime(2030, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2030, 1, 1, tzinfo=UTC),
    }

    record = repository._row_to_record(row)

    assert record.knowledge_base_id is None
    assert record.progress is None
    assert record.payload == {"nested": (1,)}
    with pytest.raises(TypeError, match="JSON object"):
        repository._row_to_record({**row, "payload": "[]"})
