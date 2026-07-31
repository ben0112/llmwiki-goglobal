from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from jobs.models import JobState
from jobs.service import JobResourceNotFound
from pydantic import SecretStr
from rag import repository
from rag import service as rag_service_module
from rag.service import CreateRagRun, RagService, ResumeRagRun

from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.rag import (
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagDomainError,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
)


async def _wait_until_blocked_by(pool, *, blocked_by_pid: int) -> None:
    deadline = asyncio.get_running_loop().time() + 3
    while True:
        waiting = await pool.fetchval(
            "SELECT EXISTS("
            "SELECT 1 FROM pg_locks AS waiting JOIN pg_locks AS holding ON "
            "holding.locktype=waiting.locktype "
            "AND holding.database IS NOT DISTINCT FROM waiting.database "
            "AND holding.relation IS NOT DISTINCT FROM waiting.relation "
            "AND holding.page IS NOT DISTINCT FROM waiting.page "
            "AND holding.tuple IS NOT DISTINCT FROM waiting.tuple "
            "AND holding.virtualxid IS NOT DISTINCT FROM waiting.virtualxid "
            "AND holding.transactionid IS NOT DISTINCT FROM waiting.transactionid "
            "AND holding.classid IS NOT DISTINCT FROM waiting.classid "
            "AND holding.objid IS NOT DISTINCT FROM waiting.objid "
            "AND holding.objsubid IS NOT DISTINCT FROM waiting.objsubid "
            "WHERE NOT waiting.granted AND holding.granted "
            "AND holding.pid=$1 AND waiting.pid<>holding.pid)",
            blocked_by_pid,
        )
        if waiting:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("resume did not wait for the authoritative parent lock")
        await asyncio.sleep(0)


class ExplodingMapping(Mapping[str, int]):
    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("provider-key-do-not-leak")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> int:
        raise RuntimeError("provider-key-do-not-leak")


@pytest.fixture
def rag_settings():
    profiles = {
        "primary": {
            "base_url": "https://private-model.invalid/v1",
            "model": "writer",
            "timeout_seconds": 60,
            "version": "profile-v1",
        }
    }
    return SimpleNamespace(
        SERVER_RAG_ENABLED=True,
        RAG_MODEL_PROFILES_JSON=SecretStr(json.dumps(profiles)),
        RAG_MODEL_API_KEYS_JSON=SecretStr(json.dumps({"primary": "provider-key-do-not-leak"})),
        HYBRID_SEARCH_ENABLED=False,
        embedding_profile=None,
        MODE="hosted",
        HYBRID_LEXICAL_CANDIDATES=50,
        HYBRID_VECTOR_CANDIDATES=50,
        HYBRID_RRF_K=60,
        EMBEDDING_BASE_URL="https://embedding.invalid/v1",
        EMBEDDING_API_KEY=SecretStr("embedding-secret"),
        EMBEDDING_BATCH_SIZE=8,
        EMBEDDING_TIMEOUT_SECONDS=30,
    )


@pytest.fixture
def service(pool, rag_settings):
    return RagService(pool, settings=rag_settings)


@pytest.fixture
async def owned_kb(pool):
    user_id = uuid4()
    knowledge_base_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id,email,display_name) VALUES($1,$2,'RAG Service Test')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id,user_id,name,slug) VALUES($1,$2,$3,$4)",
        knowledge_base_id,
        user_id,
        f"RAG {knowledge_base_id}",
        f"rag-{knowledge_base_id}",
    )
    return user_id, knowledge_base_id


def _request(knowledge_base_id: UUID, **changes: object) -> CreateRagRun:
    values: dict[str, object] = {
        "knowledge_base_id": knowledge_base_id,
        "goal": "Build the launch documentation",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
    }
    values.update(changes)
    return CreateRagRun(**values)


async def _fail_run(pool, run, *, pages: int = 0):
    usage = RagUsage()
    if pages:
        items = tuple(
            RagWorkItem.build(
                ordinal,
                f"/wiki/launch/page-{ordinal}.md",
                f"Intent {ordinal}",
                f"Evidence {ordinal}",
            )
            for ordinal in range(pages)
        )
        async with pool.acquire() as conn, conn.transaction():
            await repository.insert_worklist(conn, run, items)
            step = await repository.start_step(
                conn,
                run_id=run.id,
                page_id=None,
                step_type=RagStepType.PLAN,
                input_digest="f" * 64,
            )
            await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.SUCCEEDED,
                summary={"accepted": pages},
                citations=(),
                usage=RagUsage(steps=1),
                latency_ms=1,
            )
            usage = RagUsage(steps=1)
    async with pool.acquire() as conn, conn.transaction():
        run = await repository.finish_run(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=usage,
        )
        await conn.execute(
            "UPDATE background_jobs SET state='failed',error_code='rag_internal_error',"
            "error_message='The RAG run failed.' WHERE id=$1",
            run.job_id,
        )
    return run


def test_commands_are_frozen_normalized_and_digest_complete_budget(owned_kb):
    _user_id, knowledge_base_id = owned_kb
    command = _request(
        knowledge_base_id,
        goal="  Build the launch documentation  ",
        target_path_prefix="/wiki/launch//",
    )
    expected = {
        "budget": asdict(RagBudget()),
        "dry_run": False,
        "goal": "Build the launch documentation",
        "knowledge_base_id": str(knowledge_base_id),
        "model_profile": "primary",
        "retrieval_profile": "lexical",
        "target_path_prefix": "/wiki/launch/",
    }
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert command.request_digest == hashlib.sha256(encoded.encode()).hexdigest()
    assert command.goal == expected["goal"]
    assert command.target_path_prefix == expected["target_path_prefix"]
    assert command.goal not in repr(command)
    with pytest.raises((AttributeError, TypeError)):
        command.goal = "changed"  # type: ignore[misc]

    resumed = ResumeRagRun(parent_run_id=uuid4(), budget=RagBudget())
    resume_expected = json.dumps(
        {"budget": asdict(resumed.budget), "parent_run_id": str(resumed.parent_run_id)},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert resumed.request_digest == hashlib.sha256(resume_expected.encode()).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("knowledge_base_id", "not-a-uuid"),
        ("dry_run", 1),
        ("budget", {"max_pages": 2}),
    ),
)
def test_create_command_direct_construction_rejects_noncanonical_values(owned_kb, field, value):
    _user_id, knowledge_base_id = owned_kb
    values = {
        "knowledge_base_id": knowledge_base_id,
        "goal": "Build the launch documentation",
        "target_path_prefix": "/wiki/launch/",
        "model_profile": "primary",
    }
    values[field] = value
    with pytest.raises(RagDomainError) as exc_info:
        CreateRagRun(**values)
    assert exc_info.value.code == "rag_invalid_request"
    assert "provider-key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_preflights_feature_and_profiles_before_database_writes(pool, owned_kb, rag_settings):
    user_id, knowledge_base_id = owned_kb
    jobs_before = await pool.fetchval("SELECT count(*) FROM background_jobs")
    runs_before = await pool.fetchval("SELECT count(*) FROM rag_runs")

    rag_settings.SERVER_RAG_ENABLED = False
    with pytest.raises(RagDomainError) as disabled:
        await RagService(pool, settings=rag_settings).create(
            _request(knowledge_base_id), authenticated_user_id=user_id, idempotency_key="disabled"
        )
    assert disabled.value.code == "rag_disabled"

    rag_settings.SERVER_RAG_ENABLED = True
    with pytest.raises(RagDomainError) as unavailable:
        await RagService(pool, settings=rag_settings).create(
            _request(knowledge_base_id, model_profile="missing"),
            authenticated_user_id=user_id,
            idempotency_key="missing-profile",
        )
    assert unavailable.value.code == "rag_model_profile_unavailable"

    with pytest.raises(RagDomainError) as retrieval:
        await RagService(pool, settings=rag_settings).create(
            _request(knowledge_base_id, retrieval_profile="hybrid"),
            authenticated_user_id=user_id,
            idempotency_key="missing-retrieval",
        )
    assert retrieval.value.code == "rag_retrieval_profile_unavailable"
    assert await pool.fetchval("SELECT count(*) FROM background_jobs") == jobs_before
    assert await pool.fetchval("SELECT count(*) FROM rag_runs") == runs_before
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM rag_runs WHERE user_id=$1 AND idempotency_key=ANY($2::text[])",
            user_id,
            ["disabled", "missing-profile", "missing-retrieval"],
        )
        == 0
    )


@pytest.mark.asyncio
async def test_concurrent_same_digest_create_has_one_run_and_one_job(pool, service, owned_kb, monkeypatch):
    user_id, knowledge_base_id = owned_kb
    barrier = asyncio.Barrier(8)
    original_create = service._job_service.create_in_transaction

    async def synchronized_create(*args, **kwargs):
        await barrier.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(service._job_service, "create_in_transaction", synchronized_create)
    results = await asyncio.gather(
        *(
            service.create(
                _request(knowledge_base_id),
                authenticated_user_id=user_id,
                idempotency_key="concurrent-same",
            )
            for _ in range(8)
        )
    )
    assert len({item.id for item in results}) == 1
    assert len({item.job_id for item in results}) == 1
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM rag_runs WHERE user_id=$1 AND idempotency_key='concurrent-same'",
            user_id,
        )
        == 1
    )
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id=$1 "
            "AND job_type='build_wiki' AND idempotency_key='concurrent-same'",
            user_id,
        )
        == 1
    )
    binding = await pool.fetchrow(
        "SELECT job.rag_run_id,run.id FROM background_jobs AS job "
        "JOIN rag_runs AS run ON run.job_id=job.id "
        "WHERE job.user_id=$1 AND job.idempotency_key='concurrent-same'",
        user_id,
    )
    assert binding["rag_run_id"] == binding["id"] == results[0].id


@pytest.mark.asyncio
async def test_concurrent_different_digest_create_has_one_winner_and_no_orphan(pool, service, owned_kb, monkeypatch):
    user_id, knowledge_base_id = owned_kb
    barrier = asyncio.Barrier(8)
    original_create = service._job_service.create_in_transaction

    async def synchronized_create(*args, **kwargs):
        await barrier.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(service._job_service, "create_in_transaction", synchronized_create)
    commands = tuple(
        _request(knowledge_base_id, goal="Winner A" if index % 2 == 0 else "Winner B") for index in range(8)
    )
    outcomes = await asyncio.gather(
        *(
            service.create(
                command,
                authenticated_user_id=user_id,
                idempotency_key="concurrent-conflict",
            )
            for command in commands
        ),
        return_exceptions=True,
    )
    winners = tuple(item for item in outcomes if not isinstance(item, BaseException))
    conflicts = tuple(item for item in outcomes if isinstance(item, RagDomainError))
    assert len(winners) == 4
    assert len({item.id for item in winners}) == 1
    assert len(conflicts) == 4
    assert {item.code for item in conflicts} == {"rag_idempotency_conflict"}
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM rag_runs WHERE user_id=$1 AND idempotency_key='concurrent-conflict'",
            user_id,
        )
        == 1
    )
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id=$1 "
            "AND job_type='build_wiki' AND idempotency_key='concurrent-conflict'",
            user_id,
        )
        == 1
    )
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM background_jobs AS job LEFT JOIN rag_runs AS run "
        "ON run.job_id=job.id WHERE job.user_id=$1 AND job.idempotency_key='concurrent-conflict' "
        "AND run.id IS NULL)",
        user_id,
    )


@pytest.mark.asyncio
async def test_resume_after_committed_boundary_copies_only_public_worklist_state(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    root = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="boundary-parent",
    )
    items = tuple(
        RagWorkItem.build(
            ordinal,
            f"/wiki/launch/boundary-{ordinal}.md",
            f"Intent {ordinal}",
            f"Query {ordinal}",
        )
        for ordinal in range(3)
    )
    document_id = uuid4()
    private_marker = "private-prompt-evidence-provider-key"
    citation = RagCitation(document_id, 1, 0, page=1)
    async with pool.acquire() as conn, conn.transaction():
        pages = await repository.insert_worklist(conn, root, items)
        await conn.execute(
            "INSERT INTO documents "
            "(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,content,version) "
            "VALUES($1,$2,$3,'boundary-0.md','/wiki/launch/','wiki','md','ready',$4,1)",
            document_id,
            knowledge_base_id,
            user_id,
            private_marker,
        )
        page = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        step = await repository.start_step(
            conn,
            run_id=root.id,
            page_id=page.id,
            step_type=RagStepType.WRITE,
            input_digest="c" * 64,
        )
        await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"private": private_marker},
            citations=(citation,),
            usage=RagUsage(steps=1, model_tokens=7),
            latency_ms=1,
        )
        parent, committed = await repository.mark_boundary(
            conn,
            run=root,
            page=page,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(steps=1, model_tokens=7),
            lint_summary={"clean": True},
        )
        parent = await repository.finish_run(
            conn,
            run_id=root.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(steps=1, model_tokens=7),
        )
        await conn.execute(
            "UPDATE background_jobs SET state='failed',error_code='rag_internal_error',"
            "error_message='The RAG run failed.' WHERE id=$1",
            root.job_id,
        )

    resumed = await service.resume(
        parent.id,
        authenticated_user_id=user_id,
        idempotency_key="boundary-resume",
        budget_override={"max_model_tokens": parent.budget.max_model_tokens + 100},
    )
    assert resumed is not None
    assert resumed.last_committed_ordinal == parent.last_committed_ordinal == 0
    copied = await repository.list_pages(pool, resumed.id)
    assert tuple((page.ordinal, page.path, page.intent, page.query) for page in copied) == tuple(
        (item.ordinal, item.path, item.intent, item.query) for item in items
    )
    assert copied[0].state.value == "committed"
    assert copied[0].document_id == committed.document_id == document_id
    assert copied[0].version_committed == committed.version_committed == 1
    assert copied[0].lint_summary == {"clean": True}
    for page in copied[1:]:
        assert page.state.value == "planned"
        assert page.document_id is page.version_read is page.version_committed is None
        assert page.attempt_count == page.conflict_retry_count == page.last_completed_step_sequence == 0
    assert (
        await service.steps(
            resumed.id,
            authenticated_user_id=user_id,
            after_sequence=0,
            limit=100,
        )
        == ()
    )
    persisted_text = await pool.fetchval(
        "SELECT row_to_json(run)::text || job.payload::text FROM rag_runs AS run "
        "JOIN background_jobs AS job ON job.id=run.job_id WHERE run.id=$1",
        resumed.id,
    )
    assert private_marker not in persisted_text
    assert private_marker not in repr(resumed)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("job", "run"))
async def test_resume_injected_failures_rollback_new_attempt_and_preserve_parent(
    pool, service, owned_kb, monkeypatch, caplog, failure_point
):
    user_id, knowledge_base_id = owned_kb
    parent = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key=f"resume-rollback-parent-{failure_point}",
    )
    parent = await _fail_run(pool, parent, pages=2)
    parent_before = await service.get(parent.id, authenticated_user_id=user_id)
    page_count_before = await pool.fetchval("SELECT count(*) FROM rag_run_pages")
    private_failure = "https://secret.invalid/provider-key/prompt/evidence/internal"
    if failure_point == "job":
        original = service._job_service.create_in_transaction

        async def fail_after_job(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError(private_failure)

        monkeypatch.setattr(service._job_service, "create_in_transaction", fail_after_job)
    else:
        original = repository.create_resume

        async def fail_after_run(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError(private_failure)

        monkeypatch.setattr(repository, "create_resume", fail_after_run)

    key = f"resume-rollback-{failure_point}"
    with pytest.raises(RagDomainError) as exc_info:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key=key,
            budget_override={"max_model_tokens": parent.budget.max_model_tokens + 1},
        )
    assert exc_info.value.code == "rag_internal_error"
    assert private_failure not in str(exc_info.value)
    assert private_failure not in caplog.text
    assert await service.get(parent.id, authenticated_user_id=user_id) == parent_before
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM background_jobs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        key,
    )
    assert await pool.fetchval("SELECT count(*) FROM rag_run_pages") == page_count_before
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        key,
    )
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM rag_run_pages AS page JOIN rag_runs AS run ON run.id=page.run_id "
        "WHERE run.user_id=$1 AND run.idempotency_key=$2)",
        user_id,
        key,
    )


@pytest.mark.asyncio
async def test_resume_waits_for_parent_lock_and_uses_post_lock_job_state(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    parent = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="locked-parent",
    )
    parent = await _fail_run(pool, parent)
    blocker = await pool.acquire()
    transaction = blocker.transaction()
    await transaction.start()
    try:
        await blocker.fetchrow("SELECT id FROM rag_runs WHERE id=$1 FOR UPDATE", parent.id)
        await blocker.execute("UPDATE background_jobs SET state='succeeded' WHERE id=$1", parent.job_id)
        blocker_pid = await blocker.fetchval("SELECT pg_backend_pid()")
        resume_task = asyncio.create_task(
            service.resume(
                parent.id,
                authenticated_user_id=user_id,
                idempotency_key="locked-resume",
                budget_override={},
            )
        )
        await _wait_until_blocked_by(pool, blocked_by_pid=blocker_pid)
        assert not resume_task.done()
        await transaction.commit()
        with pytest.raises(RagDomainError) as exc_info:
            await resume_task
        assert exc_info.value.code == "rag_resume_not_allowed"
    finally:
        if blocker.is_in_transaction():
            await transaction.rollback()
        await pool.release(blocker)
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM background_jobs WHERE user_id=$1 AND idempotency_key='locked-resume')",
        user_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create", "resume"))
async def test_control_signal_is_sanitized_propagated_and_rolls_back(
    pool, service, owned_kb, monkeypatch, caplog, operation
):
    user_id, knowledge_base_id = owned_kb
    private_message = "https://secret.invalid/provider-key/prompt/evidence"
    parent = None
    if operation == "resume":
        parent = await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key="cancel-parent",
        )
        parent = await _fail_run(pool, parent)
    original = service._job_service.create_in_transaction

    async def cancel_after_job(*args, **kwargs):
        await original(*args, **kwargs)
        raise asyncio.CancelledError(private_message)

    monkeypatch.setattr(service._job_service, "create_in_transaction", cancel_after_job)
    key = f"cancel-{operation}"
    with pytest.raises(asyncio.CancelledError) as exc_info:
        if operation == "create":
            await service.create(
                _request(knowledge_base_id),
                authenticated_user_id=user_id,
                idempotency_key=key,
            )
        else:
            assert parent is not None
            await service.resume(
                parent.id,
                authenticated_user_id=user_id,
                idempotency_key=key,
                budget_override={},
            )
    assert exc_info.value.args == ()
    assert private_message not in caplog.text
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM background_jobs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        key,
    )
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        key,
    )


@pytest.mark.asyncio
async def test_create_is_atomic_exact_and_idempotent(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    command = _request(knowledge_base_id, dry_run=True, budget=replace(RagBudget(), max_pages=2))
    created = await service.create(
        command,
        authenticated_user_id=user_id,
        idempotency_key="create-one",
    )
    _assert_exact_run_uuids(created)
    job = await pool.fetchrow("SELECT * FROM background_jobs WHERE id=$1", created.job_id)
    assert json.loads(job["payload"]) == {"run_id": str(created.id)}
    assert created.id == created.root_run_id
    assert created.parent_run_id is None
    assert created.user_id == user_id
    assert created.knowledge_base_id == knowledge_base_id
    assert created.model_profile_version == "profile-v1"
    assert created.request_digest == command.request_digest
    assert created.budget == command.budget
    assert created.dry_run

    replay = await service.create(
        _request(knowledge_base_id, dry_run=True, budget=replace(RagBudget(), max_pages=2)),
        authenticated_user_id=user_id,
        idempotency_key="create-one",
    )
    _assert_exact_run_uuids(replay)
    assert replay == created
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM background_jobs WHERE user_id=$1 AND job_type='build_wiki'",
            user_id,
        )
        == 1
    )
    assert await pool.fetchval("SELECT count(*) FROM rag_runs WHERE user_id=$1", user_id) == 1

    with pytest.raises(RagDomainError) as conflict:
        await service.create(
            _request(knowledge_base_id, goal="A different request"),
            authenticated_user_id=user_id,
            idempotency_key="create-one",
        )
    assert conflict.value.code == "rag_idempotency_conflict"
    assert await pool.fetchval("SELECT count(*) FROM rag_runs WHERE user_id=$1", user_id) == 1


@pytest.mark.asyncio
async def test_create_rejects_unowned_knowledge_base(pool, service, owned_kb):
    _owner_id, knowledge_base_id = owned_kb
    other_user = uuid4()
    await pool.execute(
        "INSERT INTO users (id,email) VALUES($1,$2)",
        other_user,
        f"{other_user}@test.invalid",
    )
    with pytest.raises(JobResourceNotFound):
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=other_user,
            idempotency_key="not-owned",
        )
    assert not await pool.fetchval("SELECT EXISTS(SELECT 1 FROM background_jobs WHERE user_id=$1)", other_user)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("job", "run"))
async def test_create_injected_failures_roll_back_job_and_run(
    pool, service, owned_kb, monkeypatch, caplog, failure_point
):
    user_id, knowledge_base_id = owned_kb
    if failure_point == "job":
        original = service._job_service.create_in_transaction

        async def fail_after_job(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("https://private-model.invalid/provider-key-do-not-leak")

        monkeypatch.setattr(service._job_service, "create_in_transaction", fail_after_job)
    else:
        original = repository.create_root

        async def fail_after_run(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("retrieved-content-do-not-leak")

        monkeypatch.setattr(repository, "create_root", fail_after_run)

    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key=f"rollback-{failure_point}",
        )
    assert exc_info.value.code == "rag_internal_error"
    assert "private-model" not in str(exc_info.value)
    assert "provider-key" not in str(exc_info.value)
    assert "retrieved-content" not in str(exc_info.value)
    assert "private-model" not in caplog.text
    assert "provider-key" not in caplog.text
    assert "retrieved-content" not in caplog.text
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM background_jobs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        f"rollback-{failure_point}",
    )
    assert not await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM rag_runs WHERE user_id=$1 AND idempotency_key=$2)",
        user_id,
        f"rollback-{failure_point}",
    )


@pytest.mark.asyncio
async def test_get_and_steps_are_tenant_scoped_and_strictly_bounded(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    created = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="observe",
    )
    async with pool.acquire() as conn, conn.transaction():
        step = await repository.start_step(
            conn,
            run_id=created.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="a" * 64,
        )
        await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"count": 1},
            citations=(),
            usage=RagUsage(steps=1),
            latency_ms=1,
        )

    fetched = await service.get(created.id, authenticated_user_id=user_id)
    assert fetched == created
    assert fetched is not None
    _assert_exact_run_uuids(fetched)
    observed = await service.steps(
        created.id,
        authenticated_user_id=user_id,
        after_sequence=0,
        limit=1,
    )
    assert observed is not None and tuple(item.sequence for item in observed) == (1,)
    for item in observed:
        for field in ("id", "run_id", "user_id", "knowledge_base_id"):
            assert type(getattr(item, field)) is UUID
    stranger = uuid4()
    assert await service.get(created.id, authenticated_user_id=stranger) is None
    assert (
        await service.steps(
            created.id,
            authenticated_user_id=stranger,
            after_sequence=0,
            limit=1,
        )
        is None
    )
    for after_sequence, limit in ((True, 1), (0, True), (-1, 1), (0, 0), (0, 101)):
        with pytest.raises(RagDomainError) as exc_info:
            await service.steps(
                created.id,
                authenticated_user_id=user_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        assert exc_info.value.code == "rag_invalid_request"


@pytest.mark.asyncio
async def test_resume_copies_worklist_preserves_scope_and_is_idempotent(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    root = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="resume-parent",
    )
    parent = await _fail_run(pool, root, pages=2)
    override = {"max_model_tokens": parent.budget.max_model_tokens + 1_000}
    resumed = await service.resume(
        parent.id,
        authenticated_user_id=user_id,
        idempotency_key="resume-one",
        budget_override=override,
    )
    assert resumed is not None
    _assert_exact_run_uuids(resumed)
    assert resumed.parent_run_id == parent.id
    assert resumed.root_run_id == parent.root_run_id
    assert resumed.user_id == parent.user_id
    assert resumed.knowledge_base_id == parent.knowledge_base_id
    assert resumed.goal == parent.goal
    assert resumed.target_path_prefix == parent.target_path_prefix
    assert resumed.model_profile == parent.model_profile
    assert resumed.model_profile_version == parent.model_profile_version
    assert resumed.retrieval_profile == parent.retrieval_profile
    assert resumed.dry_run == parent.dry_run
    assert resumed.budget.max_model_tokens == override["max_model_tokens"]
    assert await repository.list_pages(pool, resumed.id)
    parent_steps = await service.steps(
        parent.id,
        authenticated_user_id=user_id,
        after_sequence=0,
        limit=100,
    )
    assert parent_steps is not None and len(parent_steps) == 1
    assert (
        await service.steps(
            resumed.id,
            authenticated_user_id=user_id,
            after_sequence=0,
            limit=100,
        )
        == ()
    )
    job = await pool.fetchrow("SELECT payload,state::text FROM background_jobs WHERE id=$1", resumed.job_id)
    assert json.loads(job["payload"]) == {"run_id": str(resumed.id)}
    assert job["state"] == JobState.QUEUED.value

    replay = await service.resume(
        parent.id,
        authenticated_user_id=user_id,
        idempotency_key="resume-one",
        budget_override=dict(override),
    )
    assert replay == resumed
    assert replay is not None
    _assert_exact_run_uuids(replay)
    with pytest.raises(RagDomainError) as conflict:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key="resume-one",
            budget_override={"max_model_tokens": override["max_model_tokens"] + 1},
        )
    assert conflict.value.code == "rag_idempotency_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("queued", "running", "retry_wait", "cancelled", "succeeded"))
async def test_resume_requires_exact_failed_job_state(pool, service, owned_kb, state):
    user_id, knowledge_base_id = owned_kb
    parent = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key=f"state-parent-{state}",
    )
    await pool.execute(
        "UPDATE rag_runs SET completion_reason='partial_failure' WHERE id=$1",
        parent.id,
    )
    await pool.execute("UPDATE background_jobs SET state=$2 WHERE id=$1", parent.job_id, state)
    with pytest.raises(RagDomainError) as exc_info:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key=f"state-resume-{state}",
            budget_override={},
        )
    assert exc_info.value.code == "rag_resume_not_allowed"


@pytest.mark.asyncio
async def test_resume_rejects_profile_drift_and_invalid_budget_without_writes(pool, service, owned_kb, rag_settings):
    user_id, knowledge_base_id = owned_kb
    parent = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="validation-parent",
    )
    parent = await _fail_run(pool, parent)
    before = await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id=$1", user_id)

    bad_overrides = (
        {"unknown": 1},
        {"max_pages": True},
        {"max_pages": parent.budget.max_pages},
        {"max_pages": parent.budget.max_pages - 1},
        {"max_pages": 33},
        ExplodingMapping(),
    )
    for index, override in enumerate(bad_overrides):
        with pytest.raises(RagDomainError) as exc_info:
            await service.resume(
                parent.id,
                authenticated_user_id=user_id,
                idempotency_key=f"bad-budget-{index}",
                budget_override=override,
            )
        assert exc_info.value.code == "rag_invalid_request"
        assert "provider-key" not in str(exc_info.value)

    drifted = {
        "primary": {
            "base_url": "https://private-model.invalid/v1",
            "model": "writer",
            "timeout_seconds": 60,
            "version": "profile-v2",
        }
    }
    rag_settings.RAG_MODEL_PROFILES_JSON = SecretStr(json.dumps(drifted))
    with pytest.raises(RagDomainError) as drift:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key="profile-drift",
            budget_override={},
        )
    assert drift.value.code == "rag_model_profile_unavailable"
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id=$1", user_id) == before


@pytest.mark.asyncio
async def test_resume_is_tenant_scoped_and_disabled_blocks_new_work(pool, service, owned_kb, rag_settings):
    user_id, knowledge_base_id = owned_kb
    parent = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key="scope-parent",
    )
    parent = await _fail_run(pool, parent)
    assert (
        await service.resume(
            parent.id,
            authenticated_user_id=uuid4(),
            idempotency_key="wrong-tenant",
            budget_override={},
        )
        is None
    )
    rag_settings.SERVER_RAG_ENABLED = False
    with pytest.raises(RagDomainError) as disabled:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key="disabled-resume",
            budget_override={},
        )
    assert disabled.value.code == "rag_disabled"


class HostileUUID(UUID):
    def __str__(self) -> str:
        raise RuntimeError("uuid-secret-do-not-leak")


def _assert_exact_run_uuids(run):
    for field in ("id", "job_id", "root_run_id", "user_id", "knowledge_base_id"):
        assert type(getattr(run, field)) is UUID
    if run.parent_run_id is not None:
        assert type(run.parent_run_id) is UUID


class HostileRagDomainError(RagDomainError):
    def __getattribute__(self, name):
        if name in {"code", "public_message", "retryable", "__cause__", "__context__"}:
            raise RuntimeError("private-domain-attribute-secret")
        return super().__getattribute__(name)


class HostileExceptionGraph(RuntimeError):
    def __getattribute__(self, name):
        if name in {"__cause__", "__context__"}:
            raise RuntimeError("private-exception-graph-secret")
        return super().__getattribute__(name)


class HostileExceptionGroup(BaseExceptionGroup):
    def __getattribute__(self, name):
        if name == "exceptions":
            raise RuntimeError("private-exception-group-secret")
        return super().__getattribute__(name)


class OversizedMapping(Mapping[str, int]):
    def __init__(self) -> None:
        self.reads = 0

    def __iter__(self) -> Iterator[str]:
        yield from (
            "max_pages",
            "max_steps",
            "max_model_tokens",
            "max_context_chars",
            "max_page_chars",
            "per_call_timeout_seconds",
            "max_page_attempts",
            "max_conflict_retries",
        )
        for index in range(100_000):
            yield f"extra-{index}"

    def __len__(self) -> int:
        raise RuntimeError("mapping-len-secret-do-not-leak")

    def __getitem__(self, key: str) -> int:
        self.reads += 1
        return 250_000


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("job", "repository", "profile", "settings"))
async def test_public_boundary_rejects_malicious_domain_errors_without_chain(
    pool, service, owned_kb, monkeypatch, source
):
    user_id, knowledge_base_id = owned_kb
    malicious = RagDomainError("rag_idempotency_conflict", "private-provider-key", retryable=True)
    if source == "job":

        async def fail_job(*_args, **_kwargs):
            raise malicious

        monkeypatch.setattr(service._job_service, "create_in_transaction", fail_job)
    elif source == "repository":

        async def fail_repository(*_args, **_kwargs):
            raise malicious

        monkeypatch.setattr(repository, "create_root", fail_repository)
    elif source == "profile":
        monkeypatch.setattr(
            rag_service_module,
            "resolve_model_profiles",
            lambda _settings: (_ for _ in ()).throw(malicious),
        )
    else:

        class MaliciousSettings:
            @property
            def SERVER_RAG_ENABLED(self):
                raise malicious

        service = RagService(pool, settings=MaliciousSettings())

    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key=f"malicious-{source}",
        )
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "private-provider-key" not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("cause", "context", "group"))
async def test_nested_control_signal_has_priority_and_is_detached(pool, service, owned_kb, monkeypatch, shape):
    user_id, knowledge_base_id = owned_kb
    control = asyncio.CancelledError("private-control-message")
    wrapper = RagDomainError(
        "rag_idempotency_conflict",
        "The idempotency key was already used for a different request.",
    )
    if shape == "cause":
        wrapper.__cause__ = control
        failure: BaseException = wrapper
    elif shape == "context":
        wrapper.__context__ = control
        failure = wrapper
    else:
        failure = BaseExceptionGroup("private-group", (RuntimeError("private"), control))

    async def fail_job(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(service._job_service, "create_in_transaction", fail_job)
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key=f"control-{shape}",
        )
    assert exc_info.value.args == ()
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_exact_allowlisted_repository_error_contract_is_preserved(service, owned_kb, monkeypatch):
    user_id, knowledge_base_id = owned_kb
    expected_message = "The idempotency key was already used for a different request."

    async def fail_repository(*_args, **_kwargs):
        raise RagDomainError("rag_idempotency_conflict", expected_message, retryable=False)

    monkeypatch.setattr(repository, "create_root", fail_repository)
    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key="valid-contract",
        )
    assert (exc_info.value.code, str(exc_info.value), exc_info.value.retryable) == (
        "rag_idempotency_conflict",
        expected_message,
        False,
    )
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None


def test_repository_error_contracts_are_composed_into_exact_operation_allowlists():
    from llmwiki_core.rag import RAG_ERROR_CONTRACTS

    root_errors = frozenset(RAG_ERROR_CONTRACTS[name] for name in repository.CREATE_ROOT_RAG_ERROR_CONTRACT_NAMES)
    resume_errors = frozenset(RAG_ERROR_CONTRACTS[name] for name in repository.CREATE_RESUME_RAG_ERROR_CONTRACT_NAMES)
    create_preflight = frozenset(
        RAG_ERROR_CONTRACTS[name] for name in rag_service_module._CREATE_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES
    )
    resume_preflight = frozenset(
        RAG_ERROR_CONTRACTS[name] for name in rag_service_module._RESUME_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES
    )
    assert root_errors | create_preflight == rag_service_module._CREATE_RAG_ERRORS
    assert resume_errors | resume_preflight == rag_service_module._RESUME_RAG_ERRORS
    assert frozenset({RAG_ERROR_CONTRACTS["invalid_request"]}) == rag_service_module._READ_RAG_ERRORS


@pytest.mark.asyncio
async def test_job_service_cannot_inject_an_exact_allowlisted_rag_contract(service, owned_kb, monkeypatch):
    user_id, knowledge_base_id = owned_kb

    async def fail_job(*_args, **_kwargs):
        raise RagDomainError(
            "rag_idempotency_conflict",
            "The idempotency key was already used for a different request.",
            retryable=False,
        )

    monkeypatch.setattr(service._job_service, "create_in_transaction", fail_job)
    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key="exact-job-injection",
        )
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("subclass", "missing_attribute"))
async def test_public_boundary_fails_closed_for_hostile_domain_error_objects(service, owned_kb, monkeypatch, shape):
    user_id, knowledge_base_id = owned_kb
    if shape == "subclass":
        failure = HostileRagDomainError(
            "rag_idempotency_conflict",
            "The idempotency key was already used for a different request.",
        )
    else:
        failure = RagDomainError(
            "rag_idempotency_conflict",
            "The idempotency key was already used for a different request.",
        )
        del failure.code

    async def fail_repository(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(repository, "create_root", fail_repository)
    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key=f"hostile-domain-{shape}",
        )
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None
    assert "private-domain-attribute-secret" not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    (
        HostileExceptionGraph("private graph"),
        HostileExceptionGroup("private group", (RuntimeError("private child"),)),
    ),
)
async def test_public_boundary_fails_closed_for_hostile_exception_graphs(service, owned_kb, monkeypatch, failure):
    user_id, knowledge_base_id = owned_kb

    async def fail_repository(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(repository, "create_root", fail_repository)
    with pytest.raises(RagDomainError) as exc_info:
        await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key=f"hostile-graph-{type(failure).__name__}",
        )
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None
    assert "private-exception" not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "contract_name"),
    (
        ("create", "parent_not_finished"),
        ("resume", "job_binding_invalid_root"),
        ("get", "disabled"),
        ("steps", "idempotency_conflict"),
    ),
)
async def test_public_methods_reject_exact_cross_operation_rag_contracts(
    pool, service, owned_kb, monkeypatch, operation, contract_name
):
    from llmwiki_core.rag import new_rag_error

    user_id, knowledge_base_id = owned_kb
    parent = None
    if operation == "resume":
        parent = await service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key="cross-operation-parent",
        )
        parent = await _fail_run(pool, parent)

    async def fail_repository(*_args, **_kwargs):
        raise new_rag_error(contract_name)

    if operation == "create":
        monkeypatch.setattr(repository, "create_root", fail_repository)
        request = service.create(
            _request(knowledge_base_id),
            authenticated_user_id=user_id,
            idempotency_key="cross-operation-create",
        )
    elif operation == "resume":
        assert parent is not None
        monkeypatch.setattr(repository, "create_resume", fail_repository)
        request = service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key="cross-operation-resume",
            budget_override={},
        )
    elif operation == "get":
        monkeypatch.setattr(repository, "get_for_user", fail_repository)
        request = service.get(uuid4(), authenticated_user_id=user_id)
    else:
        monkeypatch.setattr(repository, "get_for_user", fail_repository)
        request = service.steps(
            uuid4(),
            authenticated_user_id=user_id,
            after_sequence=0,
            limit=1,
        )

    with pytest.raises(RagDomainError) as exc_info:
        await request
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("get", "steps"))
async def test_read_methods_do_not_publish_job_resource_not_found(service, owned_kb, monkeypatch, operation):
    user_id, _knowledge_base_id = owned_kb

    async def fail_repository(*_args, **_kwargs):
        raise JobResourceNotFound("private job resource")

    monkeypatch.setattr(repository, "get_for_user", fail_repository)
    if operation == "get":
        request = service.get(uuid4(), authenticated_user_id=user_id)
    else:
        request = service.steps(
            uuid4(),
            authenticated_user_id=user_id,
            after_sequence=0,
            limit=1,
        )

    with pytest.raises(RagDomainError) as exc_info:
        await request
    assert exc_info.value.code == "rag_internal_error"
    assert str(exc_info.value) == "The RAG request could not be completed."
    assert exc_info.value.__cause__ is exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_hybrid_preflight_reuses_complete_shared_configuration_validation(pool, owned_kb, rag_settings):
    user_id, knowledge_base_id = owned_kb
    rag_settings.HYBRID_SEARCH_ENABLED = True
    rag_settings.embedding_profile = EmbeddingProfile("openai_compatible", "embed", 8)
    valid = await RagService(pool, rag_settings).create(
        _request(knowledge_base_id, retrieval_profile="hybrid"),
        authenticated_user_id=user_id,
        idempotency_key="hybrid-valid",
    )
    assert valid.retrieval_profile == "hybrid"
    count_before = await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id=$1", user_id)

    invalid_values = (
        ("embedding_profile", object()),
        ("HYBRID_LEXICAL_CANDIDATES", True),
        ("HYBRID_VECTOR_CANDIDATES", 501),
        ("HYBRID_RRF_K", float("nan")),
    )
    for index, (field, value) in enumerate(invalid_values):
        invalid = SimpleNamespace(**vars(rag_settings))
        setattr(invalid, field, value)
        with pytest.raises(RagDomainError) as exc_info:
            await RagService(pool, invalid).create(
                _request(knowledge_base_id, retrieval_profile="hybrid"),
                authenticated_user_id=user_id,
                idempotency_key=f"hybrid-invalid-{index}",
            )
        assert exc_info.value.code == "rag_retrieval_profile_unavailable"
        assert exc_info.value.__cause__ is exc_info.value.__context__ is None

    class HostileHybridSettings:
        def __getattr__(self, name):
            return getattr(rag_settings, name)

        @property
        def embedding_profile(self):
            raise RuntimeError("private-embedding-settings-secret")

    with pytest.raises(RagDomainError) as hostile_error:
        await RagService(pool, HostileHybridSettings()).create(
            _request(knowledge_base_id, retrieval_profile="hybrid"),
            authenticated_user_id=user_id,
            idempotency_key="hybrid-hostile-settings",
        )
    assert hostile_error.value.code == "rag_retrieval_profile_unavailable"
    assert str(hostile_error.value) == "The requested RAG retrieval profile is unavailable."
    assert hostile_error.value.__cause__ is hostile_error.value.__context__ is None
    assert "private-embedding-settings-secret" not in repr(hostile_error.value)
    assert await pool.fetchval("SELECT count(*) FROM background_jobs WHERE user_id=$1", user_id) == count_before


@pytest.mark.asyncio
async def test_input_boundaries_copy_uuid_reject_bad_utf8_and_bound_mapping_reads(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    hostile_kb = HostileUUID(int=knowledge_base_id.int)
    hostile_user = HostileUUID(int=user_id.int)
    command = CreateRagRun(
        knowledge_base_id=hostile_kb,
        goal="Canonical UUID copy",
        target_path_prefix="/wiki/uuid/",
        model_profile="primary",
    )
    assert type(command.knowledge_base_id) is UUID
    created = await service.create(
        command,
        authenticated_user_id=hostile_user,
        idempotency_key="uuid-copy",
    )
    assert created.knowledge_base_id == knowledge_base_id

    for index, key in enumerate(("bad-\ud800", "x" * 201)):
        with pytest.raises(RagDomainError) as exc_info:
            await service.create(
                _request(knowledge_base_id),
                authenticated_user_id=user_id,
                idempotency_key=key,
            )
        assert exc_info.value.code == "rag_invalid_request"
        assert exc_info.value.__cause__ is exc_info.value.__context__ is None
    multibyte = "🔒" * 200
    accepted = await service.create(
        _request(knowledge_base_id),
        authenticated_user_id=user_id,
        idempotency_key=multibyte,
    )
    assert accepted.idempotency_key == multibyte

    parent = await _fail_run(pool, created)
    oversized = OversizedMapping()
    with pytest.raises(RagDomainError) as mapping_error:
        await service.resume(
            parent.id,
            authenticated_user_id=user_id,
            idempotency_key="bounded-mapping",
            budget_override=oversized,
        )
    assert mapping_error.value.code == "rag_invalid_request"
    assert oversized.reads <= 9
    assert "mapping-len-secret" not in str(mapping_error.value)


@pytest.mark.asyncio
async def test_all_rag_record_repr_and_str_are_redacted(pool, service, owned_kb):
    user_id, knowledge_base_id = owned_kb
    marker = "private-goal-idempotency-preview-lint-summary-citation-error"
    run = await service.create(
        _request(knowledge_base_id, goal=marker),
        authenticated_user_id=user_id,
        idempotency_key=marker,
    )
    async with pool.acquire() as conn, conn.transaction():
        page = (
            await repository.insert_worklist(
                conn,
                run,
                (RagWorkItem.build(0, "/wiki/launch/private.md", marker, marker),),
            )
        )[0]
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="e" * 64,
        )
        step = await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.FAILED,
            summary={"private": marker},
            citations=(RagCitation(uuid4(), 1, 0),),
            usage=RagUsage(steps=1),
            latency_ms=1,
            error_code="rag_internal_error",
        )
        await conn.execute("UPDATE rag_steps SET error_message=$2 WHERE id=$1", step.id, marker)
    page = (await repository.list_pages(pool, run.id))[0]
    page = replace(
        page,
        preview=marker,
        preview_digest=hashlib.sha256(marker.encode()).hexdigest(),
        preview_full_char_count=len(marker),
        lint_summary={"private": marker},
    )
    step = (
        await repository.list_steps_for_user(
            pool,
            run_id=run.id,
            user_id=user_id,
            after_sequence=0,
            limit=1,
        )
    )[0]
    for record in (run, page, step):
        assert marker not in repr(record)
        assert marker not in str(record)
