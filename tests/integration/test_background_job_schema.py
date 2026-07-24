import json
from contextlib import asynccontextmanager
from uuid import uuid4

import asyncpg
import pytest


async def _seed_user(pool, user_id=None):
    user_id = user_id or uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@jobs.test",
    )
    return user_id


async def _insert_job(pool, user_id, *, job_type="document.extract", **values):
    columns = ["user_id", "job_type"] + list(values)
    parameters = [user_id, job_type, *values.values()]
    placeholders = [f"${index}" for index in range(1, len(parameters) + 1)]
    casts = ["", ""] + ["::jsonb" if key in {"payload", "progress", "result"} else "" for key in values]
    values_sql = ", ".join(value + cast for value, cast in zip(placeholders, casts, strict=True))
    return await pool.fetchval(
        f"INSERT INTO background_jobs ({', '.join(columns)}) VALUES ({values_sql}) RETURNING id",
        *parameters,
    )


@asynccontextmanager
async def _authenticated_session(pool, user_id):
    conn = await pool.acquire()
    transaction = conn.transaction()
    await transaction.start()
    try:
        await conn.execute("SET LOCAL ROLE authenticated")
        await conn.execute("SELECT set_config('request.jwt.claims', $1, true)", json.dumps({"sub": str(user_id)}))
        yield conn
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise
    finally:
        await pool.release(conn)


@pytest.mark.asyncio
async def test_background_jobs_accepts_exact_job_types_and_states(pool):
    user_id = await _seed_user(pool)
    for job_type in ("document.extract", "graph.rebuild", "upload.cleanup"):
        await _insert_job(pool, user_id, job_type=job_type)
    for state in ("queued", "running", "retry_wait", "succeeded", "failed", "cancelled"):
        await _insert_job(pool, user_id, state=state)

    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id = $1", user_id) == 9


@pytest.mark.asyncio
async def test_background_jobs_rejects_unknown_job_type_and_state(pool):
    user_id = await _seed_user(pool)
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_job(pool, user_id, job_type="document.unknown")
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_job(pool, user_id, state="waiting")


@pytest.mark.asyncio
async def test_background_jobs_enforces_json_and_error_size_limits(pool):
    user_id = await _seed_user(pool)
    limits = {"payload": 16_384, "progress": 8_192, "result": 16_384}
    for column, limit in limits.items():
        boundary = json.dumps("x" * (limit - 2))
        over_limit = json.dumps("x" * (limit - 1))
        await _insert_job(pool, user_id, **{column: boundary})
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_job(pool, user_id, **{column: over_limit})

    await _insert_job(pool, user_id, error_message="x" * 2_000)
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_job(pool, user_id, error_message="x" * 2_001)


@pytest.mark.asyncio
async def test_background_jobs_enforces_attempt_ranges(pool):
    user_id = await _seed_user(pool)
    for values in (
        {"attempt_count": -1},
        {"max_attempts": 0},
        {"max_attempts": 21},
        {"dispatch_attempts": -1},
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_job(pool, user_id, **values)


@pytest.mark.asyncio
async def test_background_jobs_requires_user_and_allows_nullable_resource_links(pool):
    user_id = await _seed_user(pool)
    job_id = await _insert_job(pool, user_id)
    row = await pool.fetchrow(
        "SELECT user_id, knowledge_base_id, document_id FROM background_jobs WHERE id = $1", job_id
    )
    assert dict(row) == {"user_id": user_id, "knowledge_base_id": None, "document_id": None}

    with pytest.raises(asyncpg.NotNullViolationError):
        await pool.execute("INSERT INTO background_jobs (job_type) VALUES ('document.extract')")


@pytest.mark.asyncio
async def test_background_job_idempotency_is_tenant_scoped_and_partial(pool):
    user_a = await _seed_user(pool)
    user_b = await _seed_user(pool)
    await _insert_job(pool, user_a, idempotency_key="extract-1")
    with pytest.raises(asyncpg.UniqueViolationError):
        await _insert_job(pool, user_a, idempotency_key="extract-1")
    await _insert_job(pool, user_b, idempotency_key="extract-1")
    await _insert_job(pool, user_a)
    await _insert_job(pool, user_a)


@pytest.mark.asyncio
async def test_background_job_indexes_have_expected_columns_and_predicates(pool):
    rows = await pool.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'background_jobs'"
    )
    indexes = {row["indexname"]: row["indexdef"] for row in rows}

    assert "UNIQUE INDEX background_jobs_idempotency_key_unique" in indexes[
        "background_jobs_idempotency_key_unique"
    ]
    assert "(user_id, job_type, idempotency_key)" in indexes["background_jobs_idempotency_key_unique"]
    assert "WHERE (idempotency_key IS NOT NULL)" in indexes["background_jobs_idempotency_key_unique"]
    assert "(state, run_after, last_dispatched_at)" in indexes["background_jobs_due_dispatch_idx"]
    assert "WHERE (state = ANY (ARRAY['queued'::text, 'retry_wait'::text]))" in indexes[
        "background_jobs_due_dispatch_idx"
    ]
    assert "(state, lease_expires_at)" in indexes["background_jobs_lease_expiry_idx"]
    assert "WHERE (state = 'running'::text)" in indexes["background_jobs_lease_expiry_idx"]
    assert "(user_id, created_at DESC)" in indexes["background_jobs_user_created_idx"]


@pytest.mark.asyncio
async def test_background_jobs_rls_is_enabled_and_select_is_tenant_scoped(pool):
    user_a = await _seed_user(pool)
    user_b = await _seed_user(pool)
    job_a = await _insert_job(pool, user_a)
    job_b = await _insert_job(pool, user_b)

    assert await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE oid = 'background_jobs'::regclass"
    )
    policy = await pool.fetchrow(
        "SELECT cmd, roles FROM pg_policies WHERE schemaname = 'public' "
        "AND tablename = 'background_jobs' AND policyname = 'background_jobs_select'"
    )
    assert dict(policy) == {"cmd": "SELECT", "roles": ["authenticated"]}

    async with _authenticated_session(pool, user_a) as conn:
        visible = await conn.fetch("SELECT id FROM background_jobs ORDER BY id")
    assert [row["id"] for row in visible] == [job_a]
    assert job_b not in [row["id"] for row in visible]


@pytest.mark.asyncio
async def test_background_jobs_has_no_authenticated_mutation_policy_or_access(pool):
    user_id = await _seed_user(pool)
    job_id = await _insert_job(pool, user_id)
    policies = await pool.fetch(
        "SELECT policyname, cmd, roles FROM pg_policies WHERE schemaname = 'public' "
        "AND tablename = 'background_jobs' ORDER BY policyname"
    )
    assert [dict(policy) for policy in policies] == [
        {"policyname": "background_jobs_select", "cmd": "SELECT", "roles": ["authenticated"]}
    ]

    mutations = (
        ("INSERT INTO background_jobs (user_id, job_type) VALUES ($1, 'document.extract')", user_id),
        ("UPDATE background_jobs SET state = 'cancelled' WHERE id = $1", job_id),
        ("DELETE FROM background_jobs WHERE id = $1", job_id),
    )
    for statement, parameter in mutations:
        async with _authenticated_session(pool, user_id) as conn:
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(statement, parameter)
