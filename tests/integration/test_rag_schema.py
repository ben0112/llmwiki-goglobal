import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "supabase/migrations/015_server_rag.sql"


def _budget(**overrides):
    values = {
        "max_pages": 8,
        "max_steps": 96,
        "max_model_tokens": 64_000,
        "max_context_chars": 120_000,
        "max_page_chars": 40_000,
        "per_call_timeout_seconds": 60,
        "max_page_attempts": 2,
        "max_conflict_retries": 1,
    }
    values.update(overrides)
    return values


async def _seed_scope(pool):
    user_id, knowledge_base_id, document_id = uuid4(), uuid4(), uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@rag.test",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) "
        "VALUES ($1, $2, 'RAG schema', $3)",
        knowledge_base_id,
        user_id,
        f"rag-{knowledge_base_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, source_kind, file_type, status, version) "
        "VALUES ($1, $2, $3, 'page.md', '/wiki/page.md', 'wiki', 'md', 'ready', 1)",
        document_id,
        knowledge_base_id,
        user_id,
    )
    return user_id, knowledge_base_id, document_id


async def _insert_job(
    pool,
    user_id,
    knowledge_base_id,
    *,
    job_type="build_wiki",
    run_id=None,
    document_id=None,
    payload=None,
    idempotency_key=...,
):
    run_id = run_id or uuid4()
    if payload is None:
        payload = {"run_id": str(run_id)} if job_type == "build_wiki" else {}
    job_id = await pool.fetchval(
        "INSERT INTO background_jobs "
        "(job_type, user_id, knowledge_base_id, document_id, payload, idempotency_key) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6) RETURNING id",
        job_type,
        user_id,
        knowledge_base_id,
        document_id,
        json.dumps(payload),
        f"rag-job:{run_id}" if idempotency_key is ... else idempotency_key,
    )
    return job_id, run_id


async def _insert_run(
    pool,
    user_id,
    knowledge_base_id,
    *,
    job_id=None,
    run_id=None,
    root_run_id=None,
    parent_run_id=None,
    budget=...,
    usage=...,
    **values,
):
    if job_id is None:
        job_id, generated_run_id = await _insert_job(pool, user_id, knowledge_base_id)
        run_id = run_id or generated_run_id
    run_id = run_id or uuid4()
    root_run_id = root_run_id or run_id
    row = {
        "id": run_id,
        "job_id": job_id,
        "root_run_id": root_run_id,
        "parent_run_id": parent_run_id,
        "user_id": user_id,
        "knowledge_base_id": knowledge_base_id,
        "goal": "Build a concise wiki",
        "goal_digest": "a" * 64,
        "target_path_prefix": "/wiki/",
        "model_profile": "test-profile",
        "model_profile_version": "profile-v1",
        "retrieval_profile": "lexical",
        "budget": json.dumps(_budget() if budget is ... else budget),
        "usage": json.dumps(
            {"steps": 0, "model_tokens": 0} if usage is ... else usage
        ),
        "idempotency_key": f"rag-run:{run_id}",
        "request_digest": "b" * 64,
    }
    row.update(values)
    columns = list(row)
    parameters = list(row.values())
    placeholders = []
    for index, column in enumerate(columns, 1):
        cast = "::jsonb" if column in {"budget", "usage"} else ""
        placeholders.append(f"${index}{cast}")
    return await pool.fetchval(
        f"INSERT INTO rag_runs ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) RETURNING id",
        *parameters,
    )


async def _insert_page(pool, run_id, user_id, knowledge_base_id, *, ordinal=0, **values):
    row = {
        "run_id": run_id,
        "user_id": user_id,
        "knowledge_base_id": knowledge_base_id,
        "ordinal": ordinal,
        "path": f"/wiki/page-{ordinal}.md",
        "intent": f"Explain page {ordinal}",
        "query": f"page {ordinal} sources",
    }
    row.update(values)
    columns = list(row)
    parameters = list(row.values())
    placeholders = []
    for index, column in enumerate(columns, 1):
        cast = "::jsonb" if column == "lint_summary" else ""
        placeholders.append(f"${index}{cast}")
    return await pool.fetchval(
        f"INSERT INTO rag_run_pages ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) RETURNING id",
        *parameters,
    )


async def _insert_step(
    pool,
    run_id,
    user_id,
    knowledge_base_id,
    *,
    sequence=1,
    run_page_id=None,
    **values,
):
    row = {
        "run_id": run_id,
        "run_page_id": run_page_id,
        "user_id": user_id,
        "knowledge_base_id": knowledge_base_id,
        "sequence": sequence,
        "step_type": "retrieve",
        "status": "succeeded",
        "input_digest": "c" * 64,
        "model_profile_version": "profile-v1",
    }
    row.update(values)
    columns = list(row)
    parameters = list(row.values())
    placeholders = []
    for index, column in enumerate(columns, 1):
        cast = "::jsonb" if column in {"output_summary", "citation_identities"} else ""
        placeholders.append(f"${index}{cast}")
    return await pool.fetchval(
        f"INSERT INTO rag_steps ({', '.join(columns)}) "
        f"VALUES ({', '.join(placeholders)}) RETURNING id",
        *parameters,
    )


@asynccontextmanager
async def _authenticated_session(pool, user_id):
    conn = await pool.acquire()
    transaction = conn.transaction()
    await transaction.start()
    try:
        await conn.execute("SET LOCAL ROLE authenticated")
        await conn.execute(
            "SELECT set_config('request.jwt.claims', $1, true)",
            json.dumps({"sub": str(user_id)}),
        )
        yield conn
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise
    finally:
        await pool.release(conn)


async def _json_with_octet_length(pool, kind, target):
    constructor = "jsonb_build_object('text', '')" if kind == "object" else "jsonb_build_array('')"
    overhead = await pool.fetchval(f"SELECT octet_length({constructor}::text)")
    text_bytes = target - overhead
    euro_count, ascii_count = divmod(text_bytes, len("€".encode()))
    text = "€" * euro_count + "x" * ascii_count
    value = {"text": text} if kind == "object" else [text]
    encoded = json.dumps(value, ensure_ascii=False)
    assert await pool.fetchval("SELECT octet_length($1::jsonb::text)", encoded) == target
    return encoded


@pytest.mark.asyncio
async def test_rag_tables_exist(pool):
    assert await pool.fetchval("SELECT to_regclass('public.rag_runs')") == "rag_runs"
    assert await pool.fetchval("SELECT to_regclass('public.rag_run_pages')") == "rag_run_pages"
    assert await pool.fetchval("SELECT to_regclass('public.rag_steps')") == "rag_steps"


@pytest.mark.asyncio
async def test_build_wiki_jobs_require_exact_canonical_database_shape(pool):
    user_id, knowledge_base_id, document_id = await _seed_scope(pool)
    invalid_commands = (
        {"knowledge_base_id": None},
        {"document_id": document_id},
        {"payload": {}},
        {"payload": {"run_id": str(uuid4()), "goal": "private"}},
        {"payload": {"run_id": str(uuid4()).upper()}},
        {"payload": {"run_id": None}},
        {"payload": {"run_id": 1}},
        {"payload": []},
    )

    for changes in invalid_commands:
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_job(
                pool,
                user_id,
                changes.pop("knowledge_base_id", knowledge_base_id),
                **changes,
            )


@pytest.mark.asyncio
async def test_background_jobs_reject_cross_tenant_knowledge_base_ownership(pool):
    user_a, knowledge_base_a, _ = await _seed_scope(pool)
    user_b, _, _ = await _seed_scope(pool)

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_job(
            pool,
            user_b,
            knowledge_base_a,
            job_type="document.extract",
        )
    assert user_a != user_b


@pytest.mark.asyncio
async def test_build_wiki_jobs_require_normalized_bounded_database_idempotency_key(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    for key in (None, "", "   ", " leading", "trailing ", "x" * 201):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_job(
                pool,
                user_id,
                knowledge_base_id,
                idempotency_key=key,
            )

    await _insert_job(
        pool,
        user_id,
        knowledge_base_id,
        idempotency_key="x" * 200,
    )


@pytest.mark.asyncio
async def test_rag_run_id_trigger_derives_identity_and_ignores_lease_only_updates(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    job_id, run_id = await _insert_job(pool, user_id, knowledge_base_id)
    assert await pool.fetchval(
        "SELECT rag_run_id FROM background_jobs WHERE id=$1", job_id
    ) == run_id

    assert await pool.fetchval(
        "UPDATE background_jobs SET rag_run_id=$2 WHERE id=$1 RETURNING rag_run_id",
        job_id,
        uuid4(),
    ) == run_id
    await pool.execute(
        "UPDATE background_jobs SET heartbeat_at=clock_timestamp(), state='running' WHERE id=$1",
        job_id,
    )
    assert await pool.fetchval(
        "SELECT rag_run_id FROM background_jobs WHERE id=$1", job_id
    ) == run_id

    legacy_job_id, _ = await _insert_job(
        pool,
        user_id,
        knowledge_base_id,
        job_type="document.extract",
        payload={"run_id": "not-a-uuid"},
    )
    assert await pool.fetchval(
        "UPDATE background_jobs SET rag_run_id=$2 WHERE id=$1 RETURNING rag_run_id",
        legacy_job_id,
        uuid4(),
    ) is None

    trigger_definition = await pool.fetchval(
        "SELECT pg_get_triggerdef(oid) FROM pg_trigger "
        "WHERE tgrelid='background_jobs'::regclass "
        "AND tgname='set_background_job_rag_run_id'"
    )
    assert "BEFORE INSERT OR UPDATE OF job_type, payload, rag_run_id" in trigger_definition
    assert "heartbeat_at" not in trigger_definition
    assert "state" not in trigger_definition


@pytest.mark.asyncio
async def test_rag_run_requires_build_wiki_job_with_matching_payload_run_id(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    legacy_job_id, legacy_run_id = await _insert_job(
        pool,
        user_id,
        knowledge_base_id,
        job_type="document.extract",
    )
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_run(
            pool,
            user_id,
            knowledge_base_id,
            job_id=legacy_job_id,
            run_id=legacy_run_id,
        )

    job_id, _ = await _insert_job(pool, user_id, knowledge_base_id)
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_run(
            pool,
            user_id,
            knowledge_base_id,
            job_id=job_id,
            run_id=uuid4(),
        )


@pytest.mark.asyncio
async def test_background_job_updates_cannot_break_persisted_rag_run_binding(pool):
    user_id, knowledge_base_id, document_id = await _seed_scope(pool)
    job_id, run_id = await _insert_job(pool, user_id, knowledge_base_id)
    await _insert_run(
        pool,
        user_id,
        knowledge_base_id,
        job_id=job_id,
        run_id=run_id,
    )

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await pool.execute(
            "UPDATE background_jobs SET payload=$2::jsonb WHERE id=$1",
            job_id,
            json.dumps({"run_id": str(uuid4())}),
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "UPDATE background_jobs SET payload=$2::jsonb WHERE id=$1",
            job_id,
            json.dumps({"run_id": str(run_id), "goal": "private"}),
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute(
            "UPDATE background_jobs SET document_id=$2 WHERE id=$1",
            job_id,
            document_id,
        )


@pytest.mark.asyncio
async def test_rag_schema_accepts_one_root_two_pages_and_three_steps(pool):
    user_id, knowledge_base_id, document_id = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    page_a = await _insert_page(pool, run_id, user_id, knowledge_base_id, ordinal=0)
    page_b = await _insert_page(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        ordinal=1,
        state="committed",
        document_id=document_id,
        version_read=1,
        version_committed=2,
        lint_summary=json.dumps({"warnings": 0}),
    )
    await _insert_step(pool, run_id, user_id, knowledge_base_id, sequence=1, step_type="plan")
    await _insert_step(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        sequence=2,
        run_page_id=page_a,
        status="failed",
        error_code="retrieval_failed",
        error_message="Retrieval failed.",
    )
    await _insert_step(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        sequence=3,
        run_page_id=page_b,
        status="running",
    )

    assert await pool.fetchval("SELECT count(*) FROM rag_run_pages WHERE run_id=$1", run_id) == 2
    assert await pool.fetchval("SELECT count(*) FROM rag_steps WHERE run_id=$1", run_id) == 3


@pytest.mark.asyncio
async def test_rag_schema_rejects_unknown_states_and_step_types(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    for values in ({"state": "waiting"},):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_page(pool, run_id, user_id, knowledge_base_id, **values)
    for values in ({"step_type": "search"}, {"status": "waiting"}):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_step(pool, run_id, user_id, knowledge_base_id, **values)
    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute("UPDATE rag_runs SET completion_reason='unknown' WHERE id=$1", run_id)


@pytest.mark.asyncio
async def test_rag_run_budget_and_usage_have_exact_bounded_integer_shapes(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    await _insert_run(
        pool,
        user_id,
        knowledge_base_id,
        budget=_budget(
            max_pages=32,
            max_steps=512,
            max_model_tokens=250_000,
            max_context_chars=240_000,
            max_page_chars=120_000,
            per_call_timeout_seconds=180,
            max_page_attempts=3,
            max_conflict_retries=3,
        ),
        usage={"steps": 512, "model_tokens": 250_000},
    )

    invalid_budgets = [
        None,
        [],
        "not-an-object",
        {"max_pages": 8},
        _budget(max_pages=33),
        _budget(max_steps=0),
        _budget(max_pages=1.5),
        _budget(max_pages="1"),
        _budget(max_pages=None),
        _budget(max_model_tokens=True),
        {**_budget(), "private": 1},
    ]
    for budget in invalid_budgets:
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_run(pool, user_id, knowledge_base_id, budget=budget)

    for usage in (
        None,
        [],
        "not-an-object",
        {"steps": -1, "model_tokens": 0},
        {"steps": 0, "model_tokens": 250_001},
        {"steps": 1.5, "model_tokens": 0},
        {"steps": "1", "model_tokens": 0},
        {"steps": None, "model_tokens": 0},
        {"steps": True, "model_tokens": 0},
        {"steps": 0},
        {"steps": 0, "model_tokens": 0, "private": 1},
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_run(pool, user_id, knowledge_base_id, usage=usage)


@pytest.mark.asyncio
async def test_rag_usage_cannot_exceed_its_persisted_budget(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    for usage in (
        {"steps": 3, "model_tokens": 10},
        {"steps": 2, "model_tokens": 11},
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_run(
                pool,
                user_id,
                knowledge_base_id,
                budget=_budget(max_steps=2, max_model_tokens=10),
                usage=usage,
            )


@pytest.mark.asyncio
async def test_target_path_prefix_is_normalized_and_bounded_in_utf8_bytes(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    segment_bytes = 8_000 - len(b"/wiki//")
    euro_count, ascii_count = divmod(segment_bytes, len("€".encode()))
    segment = "€" * euro_count + "x" * ascii_count
    at_limit = f"/wiki/{segment}/"
    assert len(at_limit.encode()) == 8_000
    await _insert_run(
        pool,
        user_id,
        knowledge_base_id,
        target_path_prefix=at_limit,
    )

    invalid_prefixes = (
        at_limit[:-1] + "x/",
        "/wiki/../",
        "/wiki/./",
        "/wiki//nested/",
        r"/wiki/foo\bar/",
        " /wiki/",
        "/wiki/ ",
        "/wiki/control\nsegment/",
    )
    for prefix in invalid_prefixes:
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_run(
                pool,
                user_id,
                knowledge_base_id,
                target_path_prefix=prefix,
            )


@pytest.mark.asyncio
async def test_rag_text_and_json_caps_measure_utf8_bytes(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    preview_at_limit = "€" * 5_461 + "x"
    preview_over_limit = preview_at_limit + "x"
    page_id = await _insert_page(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        preview=preview_at_limit,
        preview_digest="d" * 64,
        preview_full_char_count=len(preview_at_limit),
        preview_truncated=False,
    )
    assert len(preview_at_limit.encode()) == 16_384
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_page(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            ordinal=1,
            preview=preview_over_limit,
        )

    summary_at_limit = await _json_with_octet_length(pool, "object", 16_384)
    summary_over_limit = await _json_with_octet_length(pool, "object", 16_385)
    citations_at_limit = await _json_with_octet_length(pool, "array", 16_384)
    citations_over_limit = await _json_with_octet_length(pool, "array", 16_385)
    await _insert_step(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        sequence=1,
        run_page_id=page_id,
        output_summary=summary_at_limit,
        citation_identities=citations_at_limit,
    )
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_step(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            sequence=2,
            output_summary=summary_over_limit,
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_step(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            sequence=2,
            citation_identities=citations_over_limit,
        )


@pytest.mark.asyncio
async def test_rag_pages_reject_invalid_fields_and_duplicate_work_items(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    await _insert_page(pool, run_id, user_id, knowledge_base_id, ordinal=0)
    invalid_values = (
        {"ordinal": -1},
        {"path": "/wiki//not-normalized.md"},
        {"path": r"/wiki/foo\bar.md"},
        {"path": "/wiki/not-markdown.txt"},
        {"intent": ""},
        {"intent": "x" * 2_001},
        {"query": "x" * 2_001},
        {"version_read": 0},
        {"version_committed": 0},
        {"attempt_count": -1},
        {"conflict_retry_count": -1},
        {"last_completed_step_sequence": -1},
        {"preview_digest": "not-a-digest"},
        {"preview_full_char_count": -1},
    )
    for offset, values in enumerate(invalid_values, 1):
        with pytest.raises(asyncpg.CheckViolationError):
            page_values = {"ordinal": offset, **values}
            await _insert_page(
                pool,
                run_id,
                user_id,
                knowledge_base_id,
                **page_values,
            )

    with pytest.raises(asyncpg.UniqueViolationError):
        await _insert_page(pool, run_id, user_id, knowledge_base_id, ordinal=0)
    with pytest.raises(asyncpg.UniqueViolationError):
        await _insert_page(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            ordinal=99,
            path="/wiki/page-0.md",
        )


@pytest.mark.asyncio
async def test_rag_page_lint_summary_is_bounded_object_and_matches_committed_state(pool):
    user_id, knowledge_base_id, document_id = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    summary_at_limit = await _json_with_octet_length(pool, "object", 16_384)
    summary_over_limit = await _json_with_octet_length(pool, "object", 16_385)
    page_id = await _insert_page(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        state="committed",
        document_id=document_id,
        version_committed=1,
        lint_summary=summary_at_limit,
    )
    assert await pool.fetchval("SELECT lint_summary FROM rag_run_pages WHERE id=$1", page_id)
    for ordinal, lint_summary in enumerate((json.dumps([]), summary_over_limit), 1):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_page(
                pool,
                run_id,
                user_id,
                knowledge_base_id,
                ordinal=ordinal,
                state="committed",
                document_id=document_id,
                version_committed=1,
                lint_summary=lint_summary,
            )
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_page(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            ordinal=1_000_001,
            state="committed",
            document_id=document_id,
            version_committed=1,
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_page(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            ordinal=1_000_002,
            lint_summary=json.dumps({}),
        )


@pytest.mark.asyncio
async def test_rag_steps_reject_invalid_fields_and_duplicate_running_step(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    run_id = await _insert_run(pool, user_id, knowledge_base_id)
    page_id = await _insert_page(pool, run_id, user_id, knowledge_base_id)
    await _insert_step(
        pool,
        run_id,
        user_id,
        knowledge_base_id,
        sequence=1,
        run_page_id=page_id,
        status="running",
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await _insert_step(
            pool,
            run_id,
            user_id,
            knowledge_base_id,
            sequence=2,
            status="running",
        )

    invalid_values = (
        {"sequence": 0},
        {"input_digest": "bad"},
        {"prompt_digest": "bad"},
        {"prompt_version": "x" * 129},
        {"input_tokens": -1},
        {"output_tokens": -1},
        {"total_tokens": -1},
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 4},
        {"latency_ms": -1},
        {"error_code": "UPPER CASE"},
        {"error_message": "x" * 2_001},
        {"citation_identities": json.dumps([{}] * 129)},
    )
    for offset, values in enumerate(invalid_values, 2):
        with pytest.raises(asyncpg.CheckViolationError):
            step_values = {"sequence": offset, **values}
            await _insert_step(
                pool,
                run_id,
                user_id,
                knowledge_base_id,
                **step_values,
            )


@pytest.mark.asyncio
async def test_rag_composite_foreign_keys_reject_cross_tenant_and_cross_run_links(pool):
    user_a, kb_a, _ = await _seed_scope(pool)
    user_b, kb_b, doc_b = await _seed_scope(pool)
    run_a = await _insert_run(pool, user_a, kb_a)
    job_b, run_b_id = await _insert_job(pool, user_b, kb_b)

    foreign_job_id, foreign_run_id = await _insert_job(pool, user_a, kb_a)
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_run(
            pool,
            user_b,
            kb_b,
            job_id=foreign_job_id,
            run_id=foreign_run_id,
        )

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_run(
            pool,
            user_b,
            kb_b,
            job_id=job_b,
            run_id=run_b_id,
            root_run_id=run_a,
        )
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_page(
            pool,
            run_a,
            user_a,
            kb_a,
            document_id=doc_b,
        )

    page_a = await _insert_page(pool, run_a, user_a, kb_a)
    same_scope_run = await _insert_run(pool, user_a, kb_a)
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_step(
            pool,
            same_scope_run,
            user_a,
            kb_a,
            run_page_id=page_a,
        )

    run_b = await _insert_run(pool, user_b, kb_b)
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await _insert_step(
            pool,
            run_b,
            user_b,
            kb_b,
            run_page_id=page_a,
        )


@pytest.mark.asyncio
async def test_rag_lineage_prevents_deleting_root_before_child(pool):
    user_id, knowledge_base_id, _ = await _seed_scope(pool)
    root_run_id = await _insert_run(pool, user_id, knowledge_base_id)
    child_job_id, child_run_id = await _insert_job(pool, user_id, knowledge_base_id)
    await _insert_run(
        pool,
        user_id,
        knowledge_base_id,
        job_id=child_job_id,
        run_id=child_run_id,
        root_run_id=root_run_id,
        parent_run_id=root_run_id,
    )

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await pool.execute("DELETE FROM rag_runs WHERE id=$1", root_run_id)
    await pool.execute("DELETE FROM rag_runs WHERE id=$1", child_run_id)
    assert await pool.fetchval("DELETE FROM rag_runs WHERE id=$1 RETURNING true", root_run_id)


@pytest.mark.asyncio
async def test_rag_rls_is_select_only_and_tenant_scoped(pool):
    user_a, kb_a, _ = await _seed_scope(pool)
    user_b, kb_b, _ = await _seed_scope(pool)
    run_a = await _insert_run(pool, user_a, kb_a)
    run_b = await _insert_run(pool, user_b, kb_b)
    page_a = await _insert_page(pool, run_a, user_a, kb_a)
    page_b = await _insert_page(pool, run_b, user_b, kb_b)
    step_a = await _insert_step(pool, run_a, user_a, kb_a, run_page_id=page_a)
    step_b = await _insert_step(pool, run_b, user_b, kb_b, run_page_id=page_b)

    for table in ("rag_runs", "rag_run_pages", "rag_steps"):
        assert await pool.fetchval(
            "SELECT relrowsecurity FROM pg_class WHERE oid=$1::regclass", table
        )
        policies = await pool.fetch(
            "SELECT cmd, roles FROM pg_policies WHERE schemaname='public' AND tablename=$1",
            table,
        )
        assert [dict(policy) for policy in policies] == [
            {"cmd": "SELECT", "roles": ["authenticated"]}
        ]

    async with _authenticated_session(pool, user_a) as conn:
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_runs")] == [run_a]
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_run_pages")] == [page_a]
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_steps")] == [step_a]

    async with _authenticated_session(pool, user_b) as conn:
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_runs")] == [run_b]
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_run_pages")] == [page_b]
        assert [row["id"] for row in await conn.fetch("SELECT id FROM rag_steps")] == [step_b]

    mutations = (
        ("UPDATE rag_runs SET goal='changed' WHERE id=$1", run_a),
        ("DELETE FROM rag_run_pages WHERE id=$1", page_a),
        (
            "INSERT INTO rag_steps "
            "(run_id,user_id,knowledge_base_id,sequence,step_type,status,input_digest,model_profile_version) "
            "VALUES($1,$2,$3,2,'read','succeeded',$4,'profile-v1')",
            run_a,
            user_a,
            kb_a,
            "e" * 64,
        ),
    )
    for statement in mutations:
        async with _authenticated_session(pool, user_a) as conn:
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(statement[0], *statement[1:])


@pytest.mark.asyncio
async def test_rag_owner_indexes_running_index_and_updated_at_triggers_exist(pool):
    indexes = {
        row["indexname"]: row["indexdef"]
        for row in await pool.fetch(
            "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' "
            "AND tablename IN ('background_jobs','rag_runs','rag_run_pages','rag_steps')"
        )
    }
    assert "(id, user_id, knowledge_base_id)" in indexes["background_jobs_rag_owner_ref"]
    assert "(run_id, ordinal)" in indexes["rag_run_pages_run_id_ordinal_key"]
    assert "(run_id, path)" in indexes["rag_run_pages_run_id_path_key"]
    assert "UNIQUE INDEX rag_steps_one_running_per_run" in indexes["rag_steps_one_running_per_run"]
    assert "WHERE (status = 'running'::text)" in indexes["rag_steps_one_running_per_run"]
    assert "rag_steps_run_sequence_idx" not in indexes
    assert "(id, user_id, knowledge_base_id, rag_run_id)" in indexes[
        "background_jobs_rag_run_ref"
    ]

    triggers = {
        row["tgname"]
        for row in await pool.fetch(
            "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND "
            "tgrelid IN ('rag_runs'::regclass,'rag_run_pages'::regclass,'rag_steps'::regclass)"
        )
    }
    assert triggers == {
        "set_rag_runs_updated_at",
        "set_rag_run_pages_updated_at",
        "set_rag_steps_updated_at",
    }

    step_columns = {
        row["column_name"]
        for row in await pool.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name='rag_steps'"
        )
    }
    assert {"prompt_version", "prompt_digest", "model_profile_version"} <= step_columns
    assert not ({"prompt", "credentials", "api_key"} & step_columns)


@pytest.mark.asyncio
async def test_migration_015_upgrades_populated_014_schema_additively(pool):
    """Serialize destructive DDL on one connection and always roll it back."""
    migration = MIGRATION.read_text(encoding="utf-8")
    user_id, knowledge_base_id = uuid4(), uuid4()
    legacy_types = {
        "document.extract",
        "document.embed",
        "graph.rebuild",
        "upload.cleanup",
    }

    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("DROP TABLE rag_steps, rag_run_pages, rag_runs")
            await conn.execute("DELETE FROM background_jobs WHERE job_type='build_wiki'")
            await conn.execute("DROP INDEX IF EXISTS background_jobs_rag_run_ref")
            await conn.execute("DROP INDEX background_jobs_rag_owner_ref")
            await conn.execute(
                "DROP TRIGGER IF EXISTS set_background_job_rag_run_id ON background_jobs"
            )
            await conn.execute("DROP FUNCTION IF EXISTS set_background_job_rag_run_id()")
            await conn.execute(
                "ALTER TABLE background_jobs "
                "DROP CONSTRAINT IF EXISTS background_jobs_build_wiki_shape_check, "
                "DROP CONSTRAINT IF EXISTS background_jobs_kb_owner_fk"
            )
            await conn.execute(
                "ALTER TABLE background_jobs DROP COLUMN IF EXISTS rag_run_id, "
                "DROP COLUMN IF EXISTS build_wiki_run_id"
            )
            await conn.execute(
                "ALTER TABLE background_jobs DROP CONSTRAINT background_jobs_job_type_check;"
                "ALTER TABLE background_jobs ADD CONSTRAINT background_jobs_job_type_check "
                "CHECK (job_type IN ("
                "'document.extract','document.embed','graph.rebuild','upload.cleanup'))"
            )
            await conn.execute(
                "INSERT INTO users(id,email) VALUES($1,$2)",
                user_id,
                f"{user_id}@rag-upgrade.test",
            )
            await conn.execute(
                "INSERT INTO knowledge_bases(id,user_id,name,slug) "
                "VALUES($1,$2,'RAG upgrade',$3)",
                knowledge_base_id,
                user_id,
                f"rag-upgrade-{knowledge_base_id}",
            )
            legacy_job_ids = []
            for index, job_type in enumerate(sorted(legacy_types)):
                payload = {"run_id": "not-a-uuid"} if index == 0 else {}
                legacy_job_ids.append(
                    await conn.fetchval(
                        "INSERT INTO background_jobs(job_type,user_id,knowledge_base_id,payload) "
                        "VALUES($1,$2,$3,$4::jsonb) RETURNING id",
                        job_type,
                        user_id,
                        knowledge_base_id,
                        json.dumps(payload),
                    )
                )

            before = {
                (row["id"], row["job_type"])
                for row in await conn.fetch("SELECT id,job_type FROM background_jobs")
            }
            relfilenode_before = await conn.fetchval(
                "SELECT relfilenode FROM pg_class WHERE oid='background_jobs'::regclass"
            )
            await conn.execute(migration)
            relfilenode_after = await conn.fetchval(
                "SELECT relfilenode FROM pg_class WHERE oid='background_jobs'::regclass"
            )
            after = {
                (row["id"], row["job_type"])
                for row in await conn.fetch("SELECT id,job_type FROM background_jobs")
            }

            assert relfilenode_after == relfilenode_before
            assert after == before
            assert legacy_types <= {job_type for _, job_type in after}
            assert await conn.fetchval(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name='background_jobs' AND column_name='rag_run_id'"
            ) == 1
            assert all(
                value is None
                for value in await conn.fetchval(
                    "SELECT array_agg(rag_run_id) FROM background_jobs WHERE id=ANY($1::uuid[])",
                    legacy_job_ids,
                )
            )

            run_id = uuid4()
            assert await conn.fetchval(
                "INSERT INTO background_jobs "
                "(job_type,user_id,knowledge_base_id,payload,idempotency_key) "
                "VALUES('build_wiki',$1,$2,$3::jsonb,'upgrade-build') RETURNING rag_run_id",
                user_id,
                knowledge_base_id,
                json.dumps({"run_id": str(run_id)}),
            ) == run_id
            with pytest.raises(asyncpg.CheckViolationError):
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO background_jobs "
                        "(job_type,user_id,knowledge_base_id,payload,idempotency_key) "
                        "VALUES('build_wiki',$1,$2,$3::jsonb,'upgrade-invalid')",
                        user_id,
                        knowledge_base_id,
                        json.dumps({"run_id": str(run_id), "goal": "private"}),
                    )
        finally:
            await transaction.rollback()


def test_rag_migration_and_test_schema_blocks_match():
    migration = MIGRATION.read_text(encoding="utf-8").strip()
    test_schema = (ROOT / "tests/helpers/schema.sql").read_text(encoding="utf-8")

    assert migration in test_schema
    for table in ("rag_runs", "rag_run_pages", "rag_steps"):
        assert f"REVOKE INSERT, UPDATE, DELETE ON {table} FROM authenticated;" in migration
