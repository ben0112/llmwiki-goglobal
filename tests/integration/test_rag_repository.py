from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from jobs import repository as jobs_repository
from jobs import worker as jobs_worker
from jobs.models import JobCreate, JobType
from jobs.service import JobService
from rag import records, repository
from rag.model import RagTokenUsage

from llmwiki_core.rag import (
    RagBudget,
    RagCitation,
    RagCompletionReason,
    RagDomainError,
    RagPageState,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
)


@dataclass(frozen=True, slots=True)
class SeededKnowledgeBase:
    id: UUID
    user_id: UUID


@pytest.fixture
async def seeded_kb(pool) -> SeededKnowledgeBase:
    user_id = uuid4()
    knowledge_base_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id,email,display_name) VALUES($1,$2,'RAG Repository Test')",
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
    return SeededKnowledgeBase(knowledge_base_id, user_id)


def _config(knowledge_base_id: UUID, *, budget: RagBudget | None = None) -> RagRunConfig:
    return RagRunConfig.build(
        knowledge_base_id=knowledge_base_id,
        goal="Document the hosted RAG architecture",
        target_path_prefix="/wiki/platform/",
        model_profile="balanced",
        retrieval_profile="hybrid",
        budget=budget,
    )


def _job_command(
    config: RagRunConfig,
    *,
    run_id: UUID,
    user_id: UUID,
    key: str = "build-one",
) -> JobCreate:
    return JobCreate(
        job_type=JobType.BUILD_WIKI,
        user_id=user_id,
        knowledge_base_id=config.knowledge_base_id,
        payload={"run_id": str(run_id)},
        idempotency_key=key,
    )


async def _create_root(
    pool,
    seeded_kb: SeededKnowledgeBase,
    *,
    run_id: UUID | None = None,
    key: str = "create-one",
    request_digest: str = "a" * 64,
    config: RagRunConfig | None = None,
):
    run_id = run_id or uuid4()
    config = config or _config(seeded_kb.id)
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(config, run_id=run_id, user_id=seeded_kb.user_id, key=key),
            authenticated_user_id=seeded_kb.user_id,
        )
        run = await repository.create_root(
            conn,
            run_id=run_id,
            job_id=job.id,
            user_id=seeded_kb.user_id,
            config=config,
            idempotency_key=key,
            request_digest=request_digest,
            model_profile_version="profile-v1",
        )
    return run, job


async def _insert_pages(pool, run, count: int = 2):
    items = tuple(
        RagWorkItem.build(
            ordinal,
            f"/wiki/platform/page-{ordinal}.md",
            f"Explain page {ordinal}",
            f"page {ordinal} evidence",
        )
        for ordinal in range(count)
    )
    async with pool.acquire() as conn, conn.transaction():
        return await repository.insert_worklist(conn, run, items)


def _decode_page_row(row):
    return records._decode_page(records._adapt_db_page_row(row))


@pytest.mark.asyncio
async def test_root_run_and_job_are_created_in_one_transaction(pool, seeded_kb):
    run_id = uuid4()
    config = _config(seeded_kb.id)
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(config, run_id=run_id, user_id=seeded_kb.user_id),
            authenticated_user_id=seeded_kb.user_id,
        )
        run = await repository.create_root(
            conn,
            run_id=run_id,
            job_id=job.id,
            user_id=seeded_kb.user_id,
            config=config,
            idempotency_key="create-one",
            request_digest="a" * 64,
            model_profile_version="profile-v1",
        )

    assert run.root_run_id == run.id == run_id
    assert run.job_id == job.id
    assert run.parent_run_id is None
    assert run.budget == config.budget
    assert run.usage == RagUsage()
    assert run.last_committed_ordinal == -1


@pytest.mark.asyncio
async def test_root_creation_rollback_leaves_no_job_or_run(pool, seeded_kb):
    run_id = uuid4()
    config = _config(seeded_kb.id)
    with pytest.raises(RuntimeError, match="rollback marker"):
        async with pool.acquire() as conn, conn.transaction():
            job = await JobService(pool).create_in_transaction(
                conn,
                _job_command(config, run_id=run_id, user_id=seeded_kb.user_id),
                authenticated_user_id=seeded_kb.user_id,
            )
            await repository.create_root(
                conn,
                run_id=run_id,
                job_id=job.id,
                user_id=seeded_kb.user_id,
                config=config,
                idempotency_key="rollback-run",
                request_digest="b" * 64,
                model_profile_version="profile-v1",
            )
            raise RuntimeError("rollback marker")

    assert not await pool.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1)", run_id)
    assert not await pool.fetchval("SELECT EXISTS(SELECT 1 FROM background_jobs WHERE rag_run_id=$1)", run_id)


@pytest.mark.asyncio
async def test_create_root_decoder_failure_rolls_back_operation_savepoint(pool, seeded_kb, monkeypatch):
    run_id = uuid4()
    config = _config(seeded_kb.id)
    original_decode = repository._decode_db_run

    def fail_target(row):
        if row["id"] == run_id:
            raise ValueError("decoder failure")
        return original_decode(row)

    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(config, run_id=run_id, user_id=seeded_kb.user_id, key="root-savepoint"),
            authenticated_user_id=seeded_kb.user_id,
        )
        with monkeypatch.context() as patch:
            patch.setattr(repository, "_decode_db_run", fail_target)
            with pytest.raises(ValueError, match="decoder failure"):
                await repository.create_root(
                    conn,
                    run_id=run_id,
                    job_id=job.id,
                    user_id=seeded_kb.user_id,
                    config=config,
                    idempotency_key="root-savepoint",
                    request_digest="1" * 64,
                    model_profile_version="profile-v1",
                )
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1)", run_id)
        assert await conn.fetchval("SELECT 40 + 2") == 42
    assert not await pool.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1)", run_id)


@pytest.mark.asyncio
async def test_idempotency_lookup_is_tenant_scoped_and_digest_conflicts_are_stable(pool, seeded_kb):
    original, job = await _create_root(pool, seeded_kb, key="repeat-key")
    async with pool.acquire() as conn, conn.transaction():
        replay = await repository.create_root(
            conn,
            run_id=uuid4(),
            job_id=job.id,
            user_id=seeded_kb.user_id,
            config=_config(seeded_kb.id),
            idempotency_key="repeat-key",
            request_digest="a" * 64,
            model_profile_version="profile-v1",
        )
        assert replay == original
        assert await repository.find_by_idempotency(conn, seeded_kb.user_id, "repeat-key") == original
        assert await repository.find_by_idempotency(conn, uuid4(), "repeat-key") is None
        with pytest.raises(RagDomainError) as exc_info:
            await repository.create_root(
                conn,
                run_id=uuid4(),
                job_id=uuid4(),
                user_id=seeded_kb.user_id,
                config=_config(seeded_kb.id),
                idempotency_key="repeat-key",
                request_digest="f" * 64,
                model_profile_version="profile-v1",
            )
    assert exc_info.value.code == "rag_idempotency_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ("job", "config", "profile", "budget"))
async def test_root_idempotent_replay_requires_exact_immutable_request(pool, seeded_kb, mismatch):
    original, job = await _create_root(pool, seeded_kb, key=f"root-replay-{mismatch}")
    config = _config(seeded_kb.id)
    job_id = job.id
    profile_version = "profile-v1"
    if mismatch == "job":
        job_id = uuid4()
    elif mismatch == "config":
        config = RagRunConfig.build(
            knowledge_base_id=seeded_kb.id,
            goal="A different normalized goal",
            target_path_prefix="/wiki/platform/",
            model_profile="balanced",
            retrieval_profile="hybrid",
        )
    elif mismatch == "profile":
        profile_version = "profile-v2"
    else:
        config = _config(seeded_kb.id, budget=replace(original.budget, max_pages=original.budget.max_pages - 1))
    async with pool.acquire() as conn, conn.transaction():
        with pytest.raises(RagDomainError) as exc_info:
            await repository.create_root(
                conn,
                run_id=uuid4(),
                job_id=job_id,
                user_id=seeded_kb.user_id,
                config=config,
                idempotency_key=f"root-replay-{mismatch}",
                request_digest="a" * 64,
                model_profile_version=profile_version,
            )
    assert exc_info.value.code == "rag_idempotency_conflict"


@pytest.mark.asyncio
async def test_worklist_is_validated_inserted_in_order_and_rolls_back(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    items = tuple(RagWorkItem.build(i, f"/wiki/platform/{i}.md", f"Intent {i}", f"Query {i}") for i in range(2))
    async with pool.acquire() as conn, conn.transaction():
        pages = await repository.insert_worklist(conn, run, items)
    assert tuple(page.ordinal for page in pages) == (0, 1)
    assert tuple(page.path for page in pages) == tuple(item.path for item in items)
    assert await repository.list_pages(pool, run.id) == pages

    rollback_run, _ = await _create_root(pool, seeded_kb, key="rollback-worklist")
    with pytest.raises(RuntimeError, match="rollback marker"):
        async with pool.acquire() as conn, conn.transaction():
            await repository.insert_worklist(conn, rollback_run, items)
            raise RuntimeError("rollback marker")
    assert await repository.list_pages(pool, rollback_run.id) == ()


@pytest.mark.asyncio
async def test_terminal_worker_snapshot_is_exact_bound_and_page_bounded(pool, seeded_kb):
    run, job = await _create_root(
        pool,
        seeded_kb,
        key="terminal-worker-snapshot",
        config=_config(seeded_kb.id, budget=replace(RagBudget(), max_pages=2)),
    )
    pages = await _insert_pages(pool, run, 2)
    async with pool.acquire() as conn, conn.transaction():
        claimed = await jobs_repository.claim(conn, job.id, "snapshot-worker", 120)
        assert claimed is not None
        assert await repository.get_terminal_snapshot_for_worker(conn, job=claimed, run_id=run.id) == (run, pages)
        assert (
            await repository.get_terminal_snapshot_for_worker(
                conn,
                job=replace(claimed, user_id=uuid4()),
                run_id=run.id,
            )
            is None
        )
        with pytest.raises(RagDomainError) as malformed:
            await repository.get_terminal_snapshot_for_worker(
                conn,
                job=replace(claimed, payload={"run_id": str(uuid4())}),
                run_id=run.id,
            )
        assert malformed.value.code == "rag_job_binding_invalid"
        await conn.execute(
            "UPDATE rag_runs SET budget=jsonb_set(budget,'{max_pages}','1'::jsonb) WHERE id=$1",
            run.id,
        )
        with pytest.raises(RagDomainError) as over_bound:
            await repository.get_terminal_snapshot_for_worker(conn, job=claimed, run_id=run.id)
        assert over_bound.value.code == "rag_job_binding_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["completed", "no_work", "dry_run"])
async def test_reaper_recovers_rag_success_crash_window_from_authoritative_snapshot(
    pool,
    seeded_kb,
    caplog,
    reason,
):
    dry_run = reason == "dry_run"
    config = RagRunConfig.build(
        knowledge_base_id=seeded_kb.id,
        goal=f"Recover {reason}",
        target_path_prefix="/wiki/platform/",
        model_profile="balanced",
        retrieval_profile="hybrid",
        dry_run=dry_run,
        budget=replace(RagBudget(), max_pages=1),
    )
    run, job = await _create_root(pool, seeded_kb, key=f"recover-{reason}", config=config)
    async with pool.acquire() as conn, conn.transaction():
        claimed = await jobs_repository.claim(conn, job.id, "crashed-rag-worker", 120)
        assert claimed is not None
        if reason == "completed":
            page = (
                await repository.insert_worklist(
                    conn,
                    run,
                    (RagWorkItem.build(0, "/wiki/platform/page-0.md", "Explain", "Evidence"),),
                )
            )[0]
            attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=1)
            document_id = await _seed_document(conn, seeded_kb)
            run, _ = await repository.mark_boundary(
                conn,
                run=run,
                page=attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(),
                lint_summary={},
            )
        elif reason == "dry_run":
            page = (
                await repository.insert_worklist(
                    conn,
                    run,
                    (RagWorkItem.build(0, "/wiki/platform/page-0.md", "Explain", "Evidence"),),
                )
            )[0]
            attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=1)
            preview = "dry preview"
            run, _ = await repository.mark_dry_run_complete(
                conn,
                run=run,
                page=attempt,
                usage=RagUsage(),
                preview=preview,
                preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
                preview_full_char_count=len(preview),
                preview_truncated=False,
            )
        run = await repository.finish_run(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason(reason),
            usage=RagUsage(),
        )
        await conn.execute(
            "UPDATE background_jobs SET lease_expires_at=clock_timestamp()-interval '1 second',"
            "attempt_count=max_attempts WHERE id=$1",
            job.id,
        )

    with caplog.at_level("INFO"):
        await jobs_worker.reap_cron({"pool": pool, "reap_batch_size": 1})

    recovered = await jobs_repository.get_for_user(pool, job.id, seeded_kb.user_id)
    assert recovered is not None
    assert recovered.state.value == "succeeded"
    assert recovered.error_code is None
    assert recovered.result == {
        "run_id": str(run.id),
        "completion_reason": reason,
        "pages_committed": 1 if reason == "completed" else 0,
        **({"pages_dry_run": 1} if reason == "dry_run" else {}),
    }
    assert "rag_run_failed" not in caplog.text


@pytest.mark.asyncio
async def test_reaper_pending_cancel_wins_over_terminal_rag_success_crash_window(
    pool,
    seeded_kb,
    caplog,
):
    run, job = await _create_root(pool, seeded_kb, key="recover-cancelled-success")
    async with pool.acquire() as conn, conn.transaction():
        assert await jobs_repository.claim(conn, job.id, "crashed-cancelled-worker", 120)
        run = await repository.finish_run(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(),
        )
        pending = await jobs_repository.request_cancel(conn, job.id, seeded_kb.user_id)
        assert pending is not None and pending.cancel_requested_at is not None
        await conn.execute(
            "UPDATE background_jobs SET lease_expires_at=clock_timestamp()-interval '1 second',"
            "attempt_count=max_attempts WHERE id=$1",
            job.id,
        )

    with caplog.at_level("INFO"):
        await jobs_worker.reap_cron({"pool": pool, "reap_batch_size": 1})

    cancelled = await jobs_repository.get_for_user(pool, job.id, seeded_kb.user_id)
    assert cancelled is not None
    assert cancelled.state.value == "cancelled"
    assert cancelled.result is None
    assert cancelled.cancel_requested_at is not None
    assert (await repository.get_for_user(pool, run.id, seeded_kb.user_id)).completion_reason is (
        RagCompletionReason.NO_WORK
    )
    assert "rag_run_failed" not in caplog.text


@pytest.mark.asyncio
async def test_worklist_decoder_failure_rolls_back_all_pages_savepoint(pool, seeded_kb, monkeypatch):
    run, _ = await _create_root(pool, seeded_kb, key="worklist-savepoint")
    items = tuple(RagWorkItem.build(i, f"/wiki/platform/save-{i}.md", f"Intent {i}", f"Query {i}") for i in range(2))
    async with pool.acquire() as conn, conn.transaction():
        with monkeypatch.context() as patch:
            patch.setattr(
                repository, "_decode_db_page", lambda _row: (_ for _ in ()).throw(ValueError("decoder failure"))
            )
            with pytest.raises(ValueError, match="decoder failure"):
                await repository.insert_worklist(conn, run, items)
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_run_pages WHERE run_id=$1)", run.id)
        assert await conn.fetchval("SELECT 40 + 2") == 42
    assert await repository.list_pages(pool, run.id) == ()


@pytest.mark.asyncio
async def test_mutators_require_an_explicit_transaction(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    with pytest.raises(RuntimeError, match="transaction"):
        await repository.insert_worklist(pool, run, ())
    with pytest.raises(RuntimeError, match="transaction"):
        await repository.start_step(
            pool,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="a" * 64,
        )


@pytest.mark.asyncio
async def test_step_append_is_ordered_and_only_one_can_be_running(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    pages = await _insert_pages(pool, run, 1)
    async with pool.acquire() as conn, conn.transaction():
        page = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        first = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.RETRIEVE,
            input_digest="1" * 64,
        )
        with pytest.raises(RagDomainError) as exc_info:
            await repository.start_step(
                conn,
                run_id=run.id,
                page_id=page.id,
                step_type=RagStepType.READ,
                input_digest="2" * 64,
            )
        completed = await repository.finish_step(
            conn,
            step_id=first.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"matches": 3},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=7),
            latency_ms=2.5,
        )
        second = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.READ,
            input_digest="2" * 64,
        )
    assert exc_info.value.code == "rag_step_already_running"
    assert completed.status is RagStepStatus.SUCCEEDED
    assert completed.output_summary == {"matches": 3}
    assert completed.total_tokens == 7
    assert second.sequence == first.sequence + 1


@pytest.mark.asyncio
async def test_start_and_finish_step_enforce_dedicated_draft_reservation(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="draft-reservation")
    page = (await _insert_pages(pool, run, 1))[0]
    async with pool.acquire() as conn, conn.transaction():
        page = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        for step_type, reservation in (
            (RagStepType.DRAFT, True),
            (RagStepType.DRAFT, -1),
            (RagStepType.DRAFT, run.budget.max_model_tokens + 1),
            (RagStepType.READ, 1),
        ):
            with pytest.raises((ValueError, RagDomainError)):
                await repository.start_step(
                    conn,
                    run_id=run.id,
                    page_id=page.id,
                    step_type=step_type,
                    input_digest="9" * 64,
                    reserved_tokens=reservation,
                )
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_steps WHERE run_id=$1)", run.id)

        draft = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.DRAFT,
            input_digest="8" * 64,
            reserved_tokens=100,
        )
        assert draft.reserved_tokens == 100
        with pytest.raises((ValueError, RagDomainError)):
            await repository.finish_step(
                conn,
                step_id=draft.id,
                status=RagStepStatus.FAILED,
                summary={"outcome": "failed"},
                citations=(),
                usage=RagUsage(steps=1, model_tokens=101),
                latency_ms=0,
                error_code="rag_invalid_model_usage",
                token_usage=RagTokenUsage(50, 51, 101),
            )
        finished = await repository.finish_step(
            conn,
            step_id=draft.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"outcome": "drafted"},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=90),
            latency_ms=0,
            token_usage=RagTokenUsage(50, 40, 90),
        )
    assert finished.total_tokens == 90
    assert finished.reserved_tokens == 100


@pytest.mark.asyncio
async def test_start_step_decoder_failure_rolls_back_step_savepoint(pool, seeded_kb, monkeypatch):
    run, _ = await _create_root(pool, seeded_kb, key="step-savepoint")
    async with pool.acquire() as conn, conn.transaction():
        with monkeypatch.context() as patch:
            patch.setattr(
                repository, "_decode_db_step", lambda _row: (_ for _ in ()).throw(ValueError("decoder failure"))
            )
            with pytest.raises(ValueError, match="decoder failure"):
                await repository.start_step(
                    conn,
                    run_id=run.id,
                    page_id=None,
                    step_type=RagStepType.PLAN,
                    input_digest="2" * 64,
                )
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_steps WHERE run_id=$1)", run.id)
        assert await conn.fetchval("SELECT 40 + 2") == 42


@pytest.mark.asyncio
async def test_concurrent_step_allocation_has_unique_monotonic_sequences(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)

    async def append(step_type: RagStepType, digest_character: str):
        async with pool.acquire() as conn, conn.transaction():
            step = await repository.start_step(
                conn,
                run_id=run.id,
                page_id=None,
                step_type=step_type,
                input_digest=digest_character * 64,
            )
            return await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.SUCCEEDED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1),
                latency_ms=0,
            )

    steps = await asyncio.gather(append(RagStepType.PLAN, "a"), append(RagStepType.RETRIEVE, "b"))
    assert sorted(step.sequence for step in steps) == [1, 2]


@pytest.mark.asyncio
async def test_finish_step_records_success_failure_and_rejects_double_finish(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    document_id = uuid4()
    citation = RagCitation(document_id, 3, 4, page=2)
    async with pool.acquire() as conn, conn.transaction():
        succeeded = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="a" * 64,
        )
        succeeded = await repository.finish_step(
            conn,
            step_id=succeeded.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"planned": True},
            citations=(citation,),
            usage=RagUsage(steps=1, model_tokens=9),
            latency_ms=1.25,
        )
        failed = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.RETRIEVE,
            input_digest="b" * 64,
        )
        failed = await repository.finish_step(
            conn,
            step_id=failed.id,
            status=RagStepStatus.FAILED,
            summary={"retryable": False},
            citations=(),
            usage=RagUsage(steps=1),
            latency_ms=3.5,
            error_code="retrieval_failed",
        )
        with pytest.raises(RagDomainError) as exc_info:
            await repository.finish_step(
                conn,
                step_id=failed.id,
                status=RagStepStatus.FAILED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1),
                latency_ms=0,
                error_code="retrieval_failed",
            )
    assert succeeded.citation_identities == (citation,)
    assert failed.error_code == "retrieval_failed"
    assert exc_info.value.code == "rag_step_not_running"


@pytest.mark.asyncio
async def test_finish_step_rejects_tokens_above_core_cap_before_mutation(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="step-token-cap")
    async with pool.acquire() as conn, conn.transaction():
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="a" * 64,
        )
        with pytest.raises(ValueError, match="core hard cap"):
            await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.SUCCEEDED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1, model_tokens=250_001),
                latency_ms=1,
            )
    persisted = await repository.list_steps_for_user(
        pool,
        run_id=run.id,
        user_id=seeded_kb.user_id,
        after_sequence=0,
        limit=10,
    )
    assert len(persisted) == 1
    assert persisted[0].status is RagStepStatus.RUNNING
    assert persisted[0].total_tokens == 0


@pytest.mark.asyncio
async def test_page_attempt_increment_is_atomic_and_respects_cap(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    page = (await _insert_pages(pool, run, 1))[0]

    async def begin():
        async with pool.acquire() as conn, conn.transaction():
            return await repository.begin_page_attempt(conn, page.id, max_attempts=2)

    attempts = await asyncio.gather(begin(), begin())
    assert sorted(item.attempt_count for item in attempts) == [1, 2]
    with pytest.raises(RagDomainError) as exc_info:
        await begin()
    assert exc_info.value.code == "rag_page_attempts_exhausted"
    assert (await repository.list_pages(pool, run.id))[0].attempt_count == 2


@pytest.mark.asyncio
async def test_page_attempt_caller_cap_cannot_exceed_persisted_budget(pool, seeded_kb):
    budget = replace(RagBudget(), max_page_attempts=1)
    run, _ = await _create_root(pool, seeded_kb, key="attempt-budget", config=_config(seeded_kb.id, budget=budget))
    page = (await _insert_pages(pool, run, 1))[0]
    async with pool.acquire() as conn, conn.transaction():
        with pytest.raises(RagDomainError) as mismatch:
            await repository.begin_page_attempt(conn, page.id, max_attempts=3)
        first = await repository.begin_page_attempt(conn, page.id, max_attempts=1)
        with pytest.raises(RagDomainError) as exhausted:
            await repository.begin_page_attempt(conn, page.id, max_attempts=1)
    assert mismatch.value.code == "rag_attempt_limit_mismatch"
    assert exhausted.value.code == "rag_page_attempts_exhausted"
    assert first.attempt_count == 1


@pytest.mark.asyncio
async def test_mutation_lock_order_has_no_run_page_deadlock(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="lock-order")
    page = (await _insert_pages(pool, run, 1))[0]
    conn_a = await pool.acquire()
    conn_b = await pool.acquire()
    tx_a = conn_a.transaction()
    tx_b = conn_b.transaction()
    await tx_a.start()
    await tx_b.start()
    try:
        attempted = await repository.begin_page_attempt(conn_a, page.id, max_attempts=2)
        pid_a = await conn_a.fetchval("SELECT pg_backend_pid()")
        pid_b = await conn_b.fetchval("SELECT pg_backend_pid()")

        async def mark():
            return await repository.mark_boundary(
                conn_b,
                run=run,
                page=attempted,
                document_id=uuid4(),
                committed_version=1,
                usage=RagUsage(),
                lint_summary={},
            )

        mark_task = asyncio.create_task(mark())
        for _ in range(100):
            if pid_a in await pool.fetchval("SELECT pg_blocking_pids($1)", pid_b):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("second mutation did not block on the first transaction")
        step = await asyncio.wait_for(
            repository.start_step(
                conn_a,
                run_id=run.id,
                page_id=page.id,
                step_type=RagStepType.WRITE,
                input_digest="d" * 64,
            ),
            timeout=3,
        )
        await tx_a.commit()
        with pytest.raises(RagDomainError):
            await asyncio.wait_for(mark_task, timeout=3)
        assert step.status is RagStepStatus.RUNNING
        await tx_b.rollback()
    except BaseException:
        if conn_a.is_in_transaction():
            await tx_a.rollback()
        if conn_b.is_in_transaction():
            await tx_b.rollback()
        raise
    finally:
        await pool.release(conn_a)
        await pool.release(conn_b)


@pytest.mark.asyncio
async def test_reads_are_tenant_scoped_and_steps_use_a_bounded_cursor(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        for sequence in range(3):
            step = await repository.start_step(
                conn,
                run_id=run.id,
                page_id=None,
                step_type=RagStepType.PLAN,
                input_digest=str(sequence) * 64,
            )
            await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.SUCCEEDED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1),
                latency_ms=0,
            )
        assert await repository.get_for_user(conn, run.id, seeded_kb.user_id) == run
        assert await repository.get_for_user(conn, run.id, uuid4()) is None
        assert await repository.get_for_worker(conn, run.id, run.job_id) == run
        assert await repository.get_for_worker(conn, run.id, uuid4()) is None
        assert (
            await repository.list_steps_for_user(
                conn,
                run_id=run.id,
                user_id=uuid4(),
                after_sequence=0,
                limit=2,
            )
            == ()
        )
        steps = await repository.list_steps_for_user(
            conn,
            run_id=run.id,
            user_id=seeded_kb.user_id,
            after_sequence=1,
            limit=2,
        )
    assert tuple(step.sequence for step in steps) == (2, 3)


@pytest.mark.asyncio
async def test_fabricated_tenant_record_cannot_mutate_a_run(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    foreign = replace(run, user_id=uuid4())
    async with pool.acquire() as conn, conn.transaction():
        with pytest.raises(RagDomainError) as exc_info:
            await repository.insert_worklist(conn, foreign, ())
    assert exc_info.value.code == "rag_run_not_found"


@pytest.mark.asyncio
async def test_forged_knowledge_base_record_cannot_leak_database_errors(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    forged = replace(run, knowledge_base_id=uuid4())
    items = (RagWorkItem.build(0, "/wiki/platform/forged.md", "Forged", "Forged"),)
    async with pool.acquire() as conn, conn.transaction():
        with pytest.raises(RagDomainError) as exc_info:
            await repository.insert_worklist(conn, forged, items)
    assert exc_info.value.code == "rag_run_mismatch"


async def _seed_document(pool, seeded_kb: SeededKnowledgeBase) -> UUID:
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,path,file_type,status,version,source_kind) "
        "VALUES($1,$2,$3,'page.md','/wiki/platform/page-0.md','md','ready',1,'source')",
        document_id,
        seeded_kb.id,
        seeded_kb.user_id,
    )
    return document_id


@pytest.mark.asyncio
async def test_boundary_updates_page_run_and_rollback_atomically(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    pages = await _insert_pages(pool, run, 2)
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=attempt.id,
            step_type=RagStepType.WRITE,
            input_digest="a" * 64,
        )
        completed_step = await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.SUCCEEDED,
            summary={},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=11),
            latency_ms=1,
        )
        advanced_run, committed = await repository.mark_boundary(
            conn,
            run=run,
            page=attempt,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(steps=1, model_tokens=11),
            lint_summary={"warnings": 0},
        )
    assert advanced_run.last_committed_ordinal == 0
    assert advanced_run.usage == RagUsage(steps=1, model_tokens=11)
    assert committed.state is RagPageState.COMMITTED
    assert committed.document_id == document_id
    assert committed.version_read is None
    assert committed.version_committed == 1
    assert committed.lint_summary == {"warnings": 0}
    assert committed.last_completed_step_sequence == completed_step.sequence

    with pytest.raises(RagDomainError) as exc_info:
        async with pool.acquire() as conn, conn.transaction():
            await repository.mark_boundary(
                conn,
                run=advanced_run,
                page=committed,
                document_id=document_id,
                committed_version=2,
                usage=advanced_run.usage,
                lint_summary={},
            )
    assert exc_info.value.code == "rag_boundary_out_of_order"

    with pytest.raises(RagDomainError) as version_exc:
        async with pool.acquire() as conn, conn.transaction():
            second_attempt = await repository.begin_page_attempt(conn, pages[1].id, max_attempts=2)
            await repository.mark_boundary(
                conn,
                run=advanced_run,
                page=second_attempt,
                document_id=document_id,
                committed_version=2,
                usage=RagUsage(steps=1, model_tokens=11),
                lint_summary={},
            )
    assert version_exc.value.code == "rag_document_version_mismatch"

    before = (await repository.get_for_user(pool, run.id, seeded_kb.user_id), await repository.list_pages(pool, run.id))
    with pytest.raises(RuntimeError, match="rollback marker"):
        async with pool.acquire() as conn, conn.transaction():
            second_attempt = await repository.begin_page_attempt(conn, pages[1].id, max_attempts=2)
            updated_run, updated_page = await repository.mark_boundary(
                conn,
                run=advanced_run,
                page=second_attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(steps=1, model_tokens=11),
                lint_summary={},
            )
            assert updated_run.last_committed_ordinal == 1
            assert updated_page.lint_summary == {}
            raise RuntimeError("rollback marker")
    assert (
        await repository.get_for_user(pool, run.id, seeded_kb.user_id),
        await repository.list_pages(pool, run.id),
    ) == before


@pytest.mark.asyncio
async def test_boundary_requires_exact_terminal_usage_and_no_running_step(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="boundary-authority")
    page = (await _insert_pages(pool, run, 1))[0]
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        with pytest.raises(RagDomainError) as fabricated:
            await repository.mark_boundary(
                conn,
                run=run,
                page=attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(steps=1, model_tokens=1),
                lint_summary={},
            )
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.WRITE,
            input_digest="e" * 64,
        )
        with pytest.raises(RagDomainError) as running:
            await repository.mark_boundary(
                conn,
                run=run,
                page=attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(),
                lint_summary={},
            )
        await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.SUCCEEDED,
            summary={},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=2),
            latency_ms=1,
        )
        with pytest.raises(RagDomainError) as stale_usage:
            await repository.mark_boundary(
                conn,
                run=run,
                page=attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(),
                lint_summary={},
            )
        advanced, committed = await repository.mark_boundary(
            conn,
            run=run,
            page=attempt,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(steps=1, model_tokens=2),
            lint_summary={},
        )
    assert fabricated.value.code == "rag_usage_mismatch"
    assert running.value.code == "rag_step_still_running"
    assert stale_usage.value.code == "rag_usage_mismatch"
    assert advanced.usage == RagUsage(steps=1, model_tokens=2)
    assert committed.last_completed_step_sequence == step.sequence


@pytest.mark.asyncio
async def test_boundary_rejects_stale_caller_page_before_writes(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="stale-page")
    page = (await _insert_pages(pool, run, 1))[0]
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        stale = replace(attempt, attempt_count=0, state=RagPageState.PLANNED)
        with pytest.raises(RagDomainError) as exc_info:
            await repository.mark_boundary(
                conn,
                run=run,
                page=stale,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(),
                lint_summary={},
            )
    assert exc_info.value.code == "rag_page_mismatch"
    persisted = (await repository.list_pages(pool, run.id))[0]
    assert persisted.state is RagPageState.RUNNING
    assert persisted.attempt_count == 1


@pytest.mark.asyncio
async def test_boundary_refresh_requires_locked_document_read_predecessor(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="refresh-boundary")
    page = (await _insert_pages(pool, run, 1))[0]
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        await conn.execute(
            "UPDATE rag_run_pages SET document_id=$2,version_read=1 WHERE id=$1",
            attempt.id,
            document_id,
        )
        attempt = _decode_page_row(await conn.fetchrow("SELECT * FROM rag_run_pages WHERE id=$1", attempt.id))
        await conn.execute("UPDATE documents SET version=2 WHERE id=$1", document_id)
        advanced, committed = await repository.mark_boundary(
            conn,
            run=run,
            page=attempt,
            document_id=document_id,
            committed_version=2,
            usage=RagUsage(),
            lint_summary={},
        )
    assert advanced.last_committed_ordinal == 0
    assert committed.document_id == document_id
    assert committed.version_read == 1
    assert committed.version_committed == 2


@pytest.mark.asyncio
async def test_boundary_rejects_lint_whose_database_json_exceeds_cap_and_rolls_back(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="oversized-lint")
    page = (await _insert_pages(pool, run, 1))[0]
    document_id = await _seed_document(pool, seeded_kb)
    lint_summary = {f"k{index}": 0 for index in range(1_458)}
    assert len(json.dumps(lint_summary, separators=(",", ":")).encode()) <= 16_384
    assert len(json.dumps(lint_summary).encode()) > 16_384
    with pytest.raises(ValueError, match="byte limit"):
        async with pool.acquire() as conn, conn.transaction():
            attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
            await repository.mark_boundary(
                conn,
                run=run,
                page=attempt,
                document_id=document_id,
                committed_version=1,
                usage=RagUsage(),
                lint_summary=lint_summary,
            )
    persisted = (await repository.list_pages(pool, run.id))[0]
    assert persisted.state is RagPageState.PLANNED
    assert persisted.lint_summary is None


@pytest.mark.asyncio
async def test_finish_and_terminal_job_bookkeeping_are_strict(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        finished = await repository.finish_run(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(),
        )
        with pytest.raises(RagDomainError) as exc_info:
            await repository.finish_run(
                conn,
                run_id=run.id,
                completion_reason=RagCompletionReason.COMPLETED,
                usage=RagUsage(),
            )
    assert finished.completion_reason is RagCompletionReason.NO_WORK
    assert exc_info.value.code == "rag_run_already_finished"

    run2, job2 = await _create_root(pool, seeded_kb, key="terminal-job")
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE background_jobs SET state='failed' WHERE id=$1", job2.id)
        terminal = await repository.record_terminal_job_state(
            conn,
            run_id=run2.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
        )
    assert terminal.completion_reason is RagCompletionReason.PARTIAL_FAILURE

    run3, job3 = await _create_root(pool, seeded_kb, key="invalid-terminal-job")
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE background_jobs SET state='succeeded' WHERE id=$1", job3.id)
        with pytest.raises(RagDomainError) as terminal_exc:
            await repository.record_terminal_job_state(
                conn,
                run_id=run3.id,
                completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            )
    assert terminal_exc.value.code == "rag_invalid_completion"

    run4, job4 = await _create_root(pool, seeded_kb, key="unfinished-success-job")
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE background_jobs SET state='succeeded' WHERE id=$1", job4.id)
        with pytest.raises(RagDomainError) as unfinished_exc:
            await repository.record_terminal_job_state(
                conn,
                run_id=run4.id,
                completion_reason=RagCompletionReason.COMPLETED,
            )
    assert unfinished_exc.value.code == "rag_invalid_completion"


@pytest.mark.asyncio
async def test_finish_run_decoder_failure_rolls_back_completion_savepoint(pool, seeded_kb, monkeypatch):
    run, _ = await _create_root(pool, seeded_kb, key="finish-savepoint")
    original_decode = repository._decode_db_run

    def fail_finished(row):
        if row["id"] == run.id and row["completion_reason"] is not None:
            raise ValueError("decoder failure")
        return original_decode(row)

    async with pool.acquire() as conn, conn.transaction():
        with monkeypatch.context() as patch:
            patch.setattr(repository, "_decode_db_run", fail_finished)
            with pytest.raises(ValueError, match="decoder failure"):
                await repository.finish_run(
                    conn,
                    run_id=run.id,
                    completion_reason=RagCompletionReason.NO_WORK,
                    usage=RagUsage(),
                )
        assert await conn.fetchval("SELECT completion_reason IS NULL FROM rag_runs WHERE id=$1", run.id)
        assert await conn.fetchval("SELECT 40 + 2") == 42
    assert (await repository.get_for_user(pool, run.id, seeded_kb.user_id)).completion_reason is None


@pytest.mark.asyncio
async def test_cancelled_job_is_a_noop_for_unfinished_run(pool, seeded_kb):
    run, job = await _create_root(pool, seeded_kb, key="cancelled-job")
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE background_jobs SET state='cancelled' WHERE id=$1", job.id)
        unchanged = await repository.record_terminal_job_state(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
        )
    assert unchanged.completion_reason is None
    assert unchanged.usage == RagUsage()


@pytest.mark.asyncio
async def test_empty_dry_run_must_finish_as_no_work(pool, seeded_kb):
    config = RagRunConfig.build(
        knowledge_base_id=seeded_kb.id,
        goal="Dry empty worklist",
        target_path_prefix="/wiki/platform/",
        model_profile="balanced",
        retrieval_profile="hybrid",
        dry_run=True,
    )
    run, _ = await _create_root(pool, seeded_kb, key="dry-empty", config=config)
    async with pool.acquire() as conn, conn.transaction():
        with pytest.raises(RagDomainError) as exc_info:
            await repository.finish_run(
                conn,
                run_id=run.id,
                completion_reason=RagCompletionReason.DRY_RUN,
                usage=RagUsage(),
            )
    assert exc_info.value.code == "rag_invalid_completion"


@pytest.mark.asyncio
async def test_finish_run_requires_exact_authoritative_terminal_step_usage(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        succeeded = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.PLAN,
            input_digest="a" * 64,
        )
        await repository.finish_step(
            conn,
            step_id=succeeded.id,
            status=RagStepStatus.SUCCEEDED,
            summary={},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=10),
            latency_ms=1,
        )
        with pytest.raises(RagDomainError) as mismatch:
            await repository.finish_run(
                conn,
                run_id=run.id,
                completion_reason=RagCompletionReason.NO_WORK,
                usage=RagUsage(),
            )
        finished = await repository.finish_run(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(steps=1, model_tokens=10),
        )
    assert mismatch.value.code == "rag_usage_mismatch"
    assert finished.usage == RagUsage(steps=1, model_tokens=10)


@pytest.mark.asyncio
async def test_authoritative_usage_includes_failed_terminal_steps(pool, seeded_kb):
    run, job = await _create_root(pool, seeded_kb, key="failed-usage")
    async with pool.acquire() as conn, conn.transaction():
        failed = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=None,
            step_type=RagStepType.RETRIEVE,
            input_digest="b" * 64,
        )
        await repository.finish_step(
            conn,
            step_id=failed.id,
            status=RagStepStatus.FAILED,
            summary={},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=7),
            latency_ms=1,
            error_code="retrieval_failed",
        )
        await conn.execute("UPDATE background_jobs SET state='failed' WHERE id=$1", job.id)
        terminal = await repository.record_terminal_job_state(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
        )
    assert terminal.usage == RagUsage(steps=1, model_tokens=7)


@pytest.mark.asyncio
async def test_finish_step_enforces_cumulative_persisted_budget_before_mutation(pool, seeded_kb):
    budget = replace(RagBudget(), max_model_tokens=10)
    run, _ = await _create_root(
        pool, seeded_kb, key="cumulative-step-budget", config=_config(seeded_kb.id, budget=budget)
    )
    async with pool.acquire() as conn, conn.transaction():
        first = await repository.start_step(
            conn, run_id=run.id, page_id=None, step_type=RagStepType.PLAN, input_digest="1" * 64
        )
        await repository.finish_step(
            conn,
            step_id=first.id,
            status=RagStepStatus.SUCCEEDED,
            summary={},
            citations=(),
            usage=RagUsage(steps=1, model_tokens=7),
            latency_ms=1,
        )
        second = await repository.start_step(
            conn, run_id=run.id, page_id=None, step_type=RagStepType.READ, input_digest="2" * 64
        )
        with pytest.raises(RagDomainError) as exhausted:
            await repository.finish_step(
                conn,
                step_id=second.id,
                status=RagStepStatus.SUCCEEDED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1, model_tokens=4),
                latency_ms=1,
            )
        persisted = await conn.fetchrow("SELECT status,total_tokens FROM rag_steps WHERE id=$1", second.id)
    assert exhausted.value.code == "rag_budget_exhausted"
    assert dict(persisted) == {"status": "running", "total_tokens": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ("planned", "failed", "dry_run_complete", "committed"))
async def test_page_steps_require_running_page(pool, seeded_kb, state):
    run, _ = await _create_root(pool, seeded_kb, key=f"page-step-state-{state}")
    page = (await _insert_pages(pool, run, 1))[0]
    async with pool.acquire() as conn, conn.transaction():
        if state == "failed":
            await conn.execute("UPDATE rag_run_pages SET state='failed' WHERE id=$1", page.id)
        elif state == "dry_run_complete":
            preview = "Dry preview"
            await conn.execute(
                "UPDATE rag_run_pages SET state='dry_run_complete',attempt_count=1,preview=$2,"
                "preview_digest=$3,preview_full_char_count=$4 WHERE id=$1",
                page.id,
                preview,
                hashlib.sha256(preview.encode()).hexdigest(),
                len(preview),
            )
        elif state == "committed":
            document_id = await _seed_document(pool, seeded_kb)
            await conn.execute(
                "UPDATE rag_run_pages SET state='committed',document_id=$2,version_committed=1,lint_summary='{}'::jsonb "
                "WHERE id=$1",
                page.id,
                document_id,
            )
        with pytest.raises(RagDomainError) as exc_info:
            await repository.start_step(
                conn,
                run_id=run.id,
                page_id=page.id,
                step_type=RagStepType.WRITE,
                input_digest="a" * 64,
            )
    assert exc_info.value.code == "rag_page_not_running"


@pytest.mark.asyncio
async def test_finish_page_step_rejects_nonrunning_page_before_mutation(pool, seeded_kb):
    run, _ = await _create_root(pool, seeded_kb, key="finish-page-state")
    page = (await _insert_pages(pool, run, 1))[0]
    async with pool.acquire() as conn, conn.transaction():
        page = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.WRITE,
            input_digest="f" * 64,
        )
        await conn.execute("UPDATE rag_run_pages SET state='failed' WHERE id=$1", page.id)
        with pytest.raises(RagDomainError) as exc_info:
            await repository.finish_step(
                conn,
                step_id=step.id,
                status=RagStepStatus.FAILED,
                summary={},
                citations=(),
                usage=RagUsage(steps=1),
                latency_ms=1,
                error_code="page_failed",
            )
        persisted = await conn.fetchrow("SELECT status,total_tokens FROM rag_steps WHERE id=$1", step.id)
    assert exc_info.value.code == "rag_page_not_running"
    assert dict(persisted) == {"status": "running", "total_tokens": 0}


@pytest.mark.asyncio
async def test_resume_preserves_identity_and_copies_only_worklist_state(pool, seeded_kb):
    root, _ = await _create_root(pool, seeded_kb)
    pages = await _insert_pages(pool, root, 2)
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        step = await repository.start_step(
            conn,
            run_id=root.id,
            page_id=attempt.id,
            step_type=RagStepType.WRITE,
            input_digest="a" * 64,
        )
        await repository.finish_step(
            conn,
            step_id=step.id,
            status=RagStepStatus.SUCCEEDED,
            summary={"private": "not copied"},
            citations=(),
            usage=RagUsage(steps=1),
            latency_ms=1,
        )
        parent, _ = await repository.mark_boundary(
            conn,
            run=root,
            page=attempt,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(steps=1),
            lint_summary={},
        )
        parent = await repository.finish_run(
            conn,
            run_id=root.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(steps=1),
        )

    resume_id = uuid4()
    resume_budget = replace(parent.budget, max_model_tokens=parent.budget.max_model_tokens + 1)
    config = _config(seeded_kb.id)
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(
                config,
                run_id=resume_id,
                user_id=seeded_kb.user_id,
                key="resume-one",
            ),
            authenticated_user_id=seeded_kb.user_id,
        )
        resumed = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=parent,
            budget=resume_budget,
            idempotency_key="resume-one",
            request_digest="c" * 64,
        )
    copied = await repository.list_pages(pool, resumed.id)
    assert resumed.root_run_id == root.id
    assert resumed.parent_run_id == parent.id
    assert resumed.goal_digest == parent.goal_digest
    assert resumed.model_profile_version == parent.model_profile_version
    assert resumed.budget == resume_budget
    assert resumed.usage == RagUsage()
    assert resumed.last_committed_ordinal == parent.last_committed_ordinal
    assert tuple(page.path for page in copied) == tuple(page.path for page in pages)
    assert copied[0].state is RagPageState.COMMITTED
    assert copied[0].document_id == document_id
    assert copied[0].version_committed == 1
    assert copied[0].lint_summary == {}
    assert copied[1].state is RagPageState.PLANNED
    assert copied[1].lint_summary is None
    assert (
        await repository.list_steps_for_user(
            pool,
            run_id=resumed.id,
            user_id=seeded_kb.user_id,
            after_sequence=0,
            limit=10,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_resume_decoder_failure_rolls_back_run_and_all_pages_savepoint(pool, seeded_kb, monkeypatch):
    parent, _ = await _create_root(pool, seeded_kb, key="resume-savepoint-parent")
    await _insert_pages(pool, parent, 2)
    async with pool.acquire() as conn, conn.transaction():
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
    resume_id = uuid4()
    original_decode = repository._decode_db_run

    def fail_resume(row):
        if row["id"] == resume_id:
            raise ValueError("decoder failure")
        return original_decode(row)

    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(_config(seeded_kb.id), run_id=resume_id, user_id=seeded_kb.user_id, key="resume-savepoint"),
            authenticated_user_id=seeded_kb.user_id,
        )
        with monkeypatch.context() as patch:
            patch.setattr(repository, "_decode_db_run", fail_resume)
            with pytest.raises(ValueError, match="decoder failure"):
                await repository.create_resume(
                    conn,
                    run_id=resume_id,
                    job_id=job.id,
                    parent=parent,
                    budget=parent.budget,
                    idempotency_key="resume-savepoint",
                    request_digest="3" * 64,
                )
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1)", resume_id)
        assert not await conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_run_pages WHERE run_id=$1)", resume_id)
        assert await conn.fetchval("SELECT 40 + 2") == 42
    assert not await pool.fetchval("SELECT EXISTS(SELECT 1 FROM rag_runs WHERE id=$1)", resume_id)


@pytest.mark.asyncio
async def test_resume_resets_repository_produced_running_suffix(pool, seeded_kb):
    parent, _ = await _create_root(pool, seeded_kb, key="running-suffix-parent")
    page = (await _insert_pages(pool, parent, 1))[0]
    async with pool.acquire() as conn, conn.transaction():
        running = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
    assert running.state is RagPageState.RUNNING
    resume_id = uuid4()
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(
                _config(seeded_kb.id), run_id=resume_id, user_id=seeded_kb.user_id, key="running-suffix-resume"
            ),
            authenticated_user_id=seeded_kb.user_id,
        )
        resumed = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=parent,
            budget=parent.budget,
            idempotency_key="running-suffix-resume",
            request_digest="4" * 64,
        )
    copied = (await repository.list_pages(pool, resumed.id))[0]
    assert copied.state is RagPageState.PLANNED
    assert copied.attempt_count == copied.conflict_retry_count == copied.last_completed_step_sequence == 0
    assert copied.document_id is copied.version_read is copied.version_committed is copied.lint_summary is None
    assert copied.preview is copied.preview_digest is copied.preview_full_char_count is None
    assert not copied.preview_truncated


@pytest.mark.asyncio
async def test_resume_resets_failed_dry_run_complete_suffix(pool, seeded_kb):
    dry_config = RagRunConfig.build(
        knowledge_base_id=seeded_kb.id,
        goal="Dry suffix resume",
        target_path_prefix="/wiki/platform/",
        model_profile="balanced",
        retrieval_profile="hybrid",
        dry_run=True,
    )
    parent, _ = await _create_root(pool, seeded_kb, key="dry-suffix-parent", config=dry_config)
    page = (await _insert_pages(pool, parent, 1))[0]
    preview = "Generated dry preview"
    async with pool.acquire() as conn, conn.transaction():
        attempted = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        await conn.execute(
            "UPDATE rag_run_pages SET state='dry_run_complete',preview=$2,preview_digest=$3,"
            "preview_full_char_count=$4,preview_truncated=false WHERE id=$1",
            attempted.id,
            preview,
            hashlib.sha256(preview.encode()).hexdigest(),
            len(preview),
        )
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
    resume_id = uuid4()
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(dry_config, run_id=resume_id, user_id=seeded_kb.user_id, key="dry-suffix-resume"),
            authenticated_user_id=seeded_kb.user_id,
        )
        resumed = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=parent,
            budget=parent.budget,
            idempotency_key="dry-suffix-resume",
            request_digest="5" * 64,
        )
    copied = (await repository.list_pages(pool, resumed.id))[0]
    assert resumed.dry_run
    assert copied.state is RagPageState.PLANNED
    assert copied.attempt_count == copied.conflict_retry_count == copied.last_completed_step_sequence == 0
    assert copied.preview is copied.preview_digest is copied.preview_full_char_count is None
    assert not copied.preview_truncated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    (
        "max_pages",
        "max_steps",
        "max_model_tokens",
        "max_context_chars",
        "max_page_chars",
        "per_call_timeout_seconds",
        "max_page_attempts",
        "max_conflict_retries",
    ),
)
async def test_resume_rejects_decreased_budget_dimensions(pool, seeded_kb, field):
    parent_budget = replace(RagBudget(), max_conflict_retries=2)
    parent, _ = await _create_root(
        pool,
        seeded_kb,
        key=f"budget-parent-{field}",
        config=_config(seeded_kb.id, budget=parent_budget),
    )
    async with pool.acquire() as conn, conn.transaction():
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(),
        )
        decreased = replace(parent.budget, **{field: getattr(parent.budget, field) - 1})
        with pytest.raises(RagDomainError) as exc_info:
            await repository.create_resume(
                conn,
                run_id=uuid4(),
                job_id=uuid4(),
                parent=parent,
                budget=decreased,
                idempotency_key=f"budget-resume-{field}",
                request_digest="d" * 64,
            )
    assert exc_info.value.code == "rag_budget_too_small"


@pytest.mark.asyncio
async def test_resume_accepts_same_budget(pool, seeded_kb):
    parent, _ = await _create_root(pool, seeded_kb, key="same-budget-parent")
    async with pool.acquire() as conn, conn.transaction():
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(),
        )
    resume_id = uuid4()
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(
                _config(seeded_kb.id),
                run_id=resume_id,
                user_id=seeded_kb.user_id,
                key="same-budget-resume",
            ),
            authenticated_user_id=seeded_kb.user_id,
        )
        resumed = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=parent,
            budget=parent.budget,
            idempotency_key="same-budget-resume",
            request_digest="e" * 64,
        )
    assert resumed.budget == parent.budget


@pytest.mark.asyncio
async def test_resume_idempotent_replay_requires_exact_job_parent_and_budget(pool, seeded_kb):
    parent, _ = await _create_root(pool, seeded_kb, key="resume-replay-parent")
    async with pool.acquire() as conn, conn.transaction():
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.NO_WORK,
            usage=RagUsage(),
        )
    resume_id = uuid4()
    config = _config(seeded_kb.id)
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(config, run_id=resume_id, user_id=seeded_kb.user_id, key="resume-replay"),
            authenticated_user_id=seeded_kb.user_id,
        )
        original = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=parent,
            budget=parent.budget,
            idempotency_key="resume-replay",
            request_digest="9" * 64,
        )
    for job_id, supplied_parent, budget in (
        (uuid4(), parent, parent.budget),
        (job.id, replace(parent, id=uuid4()), parent.budget),
        (job.id, parent, replace(parent.budget, max_model_tokens=parent.budget.max_model_tokens + 1)),
    ):
        async with pool.acquire() as conn, conn.transaction():
            with pytest.raises(RagDomainError) as exc_info:
                await repository.create_resume(
                    conn,
                    run_id=uuid4(),
                    job_id=job_id,
                    parent=supplied_parent,
                    budget=budget,
                    idempotency_key="resume-replay",
                    request_digest="9" * 64,
                )
        assert exc_info.value.code == "rag_idempotency_conflict"
    assert original.parent_run_id == parent.id


@pytest.mark.asyncio
async def test_resume_rejects_missing_committed_boundary_identity(pool, seeded_kb):
    parent, _ = await _create_root(pool, seeded_kb, key="invalid-boundary-parent")
    pages = await _insert_pages(pool, parent, 1)
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        parent, _ = await repository.mark_boundary(
            conn,
            run=parent,
            page=attempt,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(),
            lint_summary={},
        )
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
        await conn.execute("DELETE FROM rag_run_pages WHERE id=$1", pages[0].id)
        with pytest.raises(RagDomainError) as exc_info:
            await repository.create_resume(
                conn,
                run_id=uuid4(),
                job_id=uuid4(),
                parent=parent,
                budget=parent.budget,
                idempotency_key="invalid-boundary-resume",
                request_digest="f" * 64,
            )
    assert exc_info.value.code == "rag_resume_boundary_invalid"


@pytest.mark.asyncio
async def test_nested_resume_validates_and_copies_complete_root_identity(pool, seeded_kb):
    root, _ = await _create_root(pool, seeded_kb, key="nested-root")
    page = (await _insert_pages(pool, root, 1))[0]
    document_id = await _seed_document(pool, seeded_kb)
    async with pool.acquire() as conn, conn.transaction():
        attempt = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        root, _ = await repository.mark_boundary(
            conn,
            run=root,
            page=attempt,
            document_id=document_id,
            committed_version=1,
            usage=RagUsage(),
            lint_summary={"clean": True},
        )
        root = await repository.finish_run(
            conn,
            run_id=root.id,
            completion_reason=RagCompletionReason.COMPLETED,
            usage=RagUsage(),
        )

    async def resume(parent, key):
        run_id = uuid4()
        async with pool.acquire() as conn, conn.transaction():
            job = await JobService(pool).create_in_transaction(
                conn,
                _job_command(_config(seeded_kb.id), run_id=run_id, user_id=seeded_kb.user_id, key=key),
                authenticated_user_id=seeded_kb.user_id,
            )
            return await repository.create_resume(
                conn,
                run_id=run_id,
                job_id=job.id,
                parent=parent,
                budget=parent.budget,
                idempotency_key=key,
                request_digest=hashlib.sha256(key.encode()).hexdigest(),
            )

    first = await resume(root, "nested-first")
    async with pool.acquire() as conn, conn.transaction():
        first = await repository.finish_run(
            conn,
            run_id=first.id,
            completion_reason=RagCompletionReason.COMPLETED,
            usage=RagUsage(),
        )
    second = await resume(first, "nested-second")
    copied = await repository.list_pages(pool, second.id)
    assert second.root_run_id == root.id
    assert second.parent_run_id == first.id
    assert len(copied) == 1
    assert copied[0].path == page.path
    assert copied[0].document_id == document_id
    assert copied[0].lint_summary == {"clean": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("root_gap", "parent_work_item"))
async def test_nested_resume_rejects_malformed_locked_source(pool, seeded_kb, corruption):
    root, _ = await _create_root(pool, seeded_kb, key=f"malformed-root-{corruption}")
    await _insert_pages(pool, root, 2)
    async with pool.acquire() as conn, conn.transaction():
        root = await repository.finish_run(
            conn,
            run_id=root.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
    resume_id = uuid4()
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(
                _config(seeded_kb.id), run_id=resume_id, user_id=seeded_kb.user_id, key=f"malformed-parent-{corruption}"
            ),
            authenticated_user_id=seeded_kb.user_id,
        )
        parent = await repository.create_resume(
            conn,
            run_id=resume_id,
            job_id=job.id,
            parent=root,
            budget=root.budget,
            idempotency_key=f"malformed-parent-{corruption}",
            request_digest="8" * 64,
        )
        parent = await repository.finish_run(
            conn,
            run_id=parent.id,
            completion_reason=RagCompletionReason.PARTIAL_FAILURE,
            usage=RagUsage(),
        )
        if corruption == "root_gap":
            await conn.execute("DELETE FROM rag_run_pages WHERE run_id=$1 AND ordinal=1", root.id)
        else:
            await conn.execute(
                "UPDATE rag_run_pages SET intent='Mismatched parent intent' WHERE run_id=$1 AND ordinal=1",
                parent.id,
            )
        with pytest.raises(RagDomainError) as exc_info:
            await repository.create_resume(
                conn,
                run_id=uuid4(),
                job_id=uuid4(),
                parent=parent,
                budget=parent.budget,
                idempotency_key=f"malformed-child-{corruption}",
                request_digest="7" * 64,
            )
    assert exc_info.value.code == "rag_resume_boundary_invalid"


def _valid_run_row() -> dict[str, object]:
    now = datetime.now(UTC)
    run_id = uuid4()
    goal = "Goal"
    return {
        "id": run_id,
        "job_id": uuid4(),
        "root_run_id": run_id,
        "parent_run_id": None,
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "goal": goal,
        "goal_digest": hashlib.sha256(goal.encode()).hexdigest(),
        "target_path_prefix": "/wiki/platform/",
        "model_profile": "balanced",
        "model_profile_version": "profile-v1",
        "retrieval_profile": "hybrid",
        "dry_run": False,
        "budget": {
            "max_pages": 8,
            "max_steps": 96,
            "max_model_tokens": 64_000,
            "max_context_chars": 120_000,
            "max_page_chars": 40_000,
            "per_call_timeout_seconds": 60,
            "max_page_attempts": 2,
            "max_conflict_retries": 1,
        },
        "usage": {"steps": 0, "model_tokens": 0},
        "idempotency_key": "key",
        "request_digest": "b" * 64,
        "completion_reason": None,
        "last_committed_ordinal": -1,
        "created_at": now,
        "updated_at": now,
    }


def test_run_decoder_accepts_a_strict_valid_mapping():
    row = _valid_run_row()
    assert records._decode_run(row).id == row["id"]


def test_direct_mapping_decoder_rejects_uuid_subclasses_without_stringifying_them():
    class HostileUUID(UUID):
        def __str__(self):
            raise RuntimeError("private uuid mapping secret")

    row = _valid_run_row()
    row["id"] = HostileUUID(int=UUID(int=1).int)
    adapted = records._adapt_db_run_row(row)

    assert type(adapted["id"]) is HostileUUID
    with pytest.raises(TypeError, match="id must be a UUID") as exc_info:
        records._decode_run(adapted)
    assert "private uuid mapping secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_public_asyncpg_record_adapter_normalizes_uuid_columns_to_base_uuid(pool, seeded_kb):
    run_id = uuid4()
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            _job_command(
                _config(seeded_kb.id),
                run_id=run_id,
                user_id=seeded_kb.user_id,
                key="record-adapter",
            ),
            authenticated_user_id=seeded_kb.user_id,
        )
        await repository.create_root(
            conn,
            run_id=run_id,
            job_id=job.id,
            user_id=seeded_kb.user_id,
            config=_config(seeded_kb.id),
            idempotency_key="record-adapter",
            request_digest="a" * 64,
            model_profile_version="profile-v1",
        )
        raw = await conn.fetchrow("SELECT * FROM rag_runs WHERE id=$1", run_id)

    assert raw is not None
    adapted = records._adapt_db_run_row(raw)
    for field in ("id", "job_id", "root_run_id", "user_id", "knowledge_base_id"):
        assert type(adapted[field]) is UUID
    assert records._decode_run(adapted).id == run_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_committed_ordinal", True),
        ("created_at", datetime.now()),
        ("completion_reason", "unknown"),
        ("id", "not-a-uuid"),
        ("request_digest", "bad"),
        ("target_path_prefix", "/wiki//bad/"),
        ("goal", "x" * 4_001),
        ("budget", []),
        ("usage", []),
    ],
)
def test_run_decoder_rejects_malformed_database_values(field, value):
    row = _valid_run_row()
    row[field] = value
    with pytest.raises((TypeError, ValueError)):
        records._decode_run(row)


@pytest.mark.parametrize(
    "budget",
    [
        {"max_pages": 8},
        {**_valid_run_row()["budget"], "unknown": 1},
        {**_valid_run_row()["budget"], "max_pages": True},
        {**_valid_run_row()["budget"], "max_pages": 33},
    ],
)
def test_run_decoder_rejects_non_exact_or_out_of_cap_budget(budget):
    row = _valid_run_row()
    row["budget"] = budget
    with pytest.raises((TypeError, ValueError)):
        records._decode_run(row)


def test_run_decoder_rejects_goal_digest_mismatch_and_boundary_above_core_cap():
    for field, value in (("goal_digest", "f" * 64), ("last_committed_ordinal", 32)):
        row = _valid_run_row()
        row[field] = value
        with pytest.raises((TypeError, ValueError)):
            records._decode_run(row)


def _valid_page_row() -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "run_id": uuid4(),
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "ordinal": 0,
        "path": "/wiki/platform/page.md",
        "intent": "Explain the page",
        "query": "page evidence",
        "state": "planned",
        "document_id": None,
        "version_read": None,
        "version_committed": None,
        "attempt_count": 0,
        "conflict_retry_count": 0,
        "last_completed_step_sequence": 0,
        "preview": None,
        "preview_digest": None,
        "preview_full_char_count": None,
        "preview_truncated": False,
        "lint_summary": None,
        "created_at": now,
        "updated_at": now,
    }


def test_page_decoder_rejects_malformed_preview_counters_and_nullable_relationships():
    assert records._decode_page(_valid_page_row()).state is RagPageState.PLANNED
    for field, value in (
        ("attempt_count", True),
        ("attempt_count", 4),
        ("ordinal", 32),
        ("last_completed_step_sequence", 513),
        ("preview_truncated", True),
        ("version_read", 1),
    ):
        row = _valid_page_row()
        row[field] = value
        with pytest.raises((TypeError, ValueError)):
            records._decode_page(row)


def test_page_decoder_enforces_preview_state_version_and_lint_relationships():
    preview = "Complete preview"
    dry = _valid_page_row()
    dry.update(
        state="dry_run_complete",
        attempt_count=1,
        preview=preview,
        preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
        preview_full_char_count=len(preview),
    )
    assert records._decode_page(dry).state is RagPageState.DRY_RUN_COMPLETE

    committed = _valid_page_row()
    committed.update(
        state="committed",
        document_id=uuid4(),
        version_read=1,
        version_committed=2,
        lint_summary={"warnings": 0},
    )
    assert records._decode_page(committed).lint_summary == {"warnings": 0}

    invalid_rows = []
    wrong_digest = dict(dry)
    wrong_digest["preview_digest"] = "f" * 64
    invalid_rows.append(wrong_digest)
    dry_with_document = dict(dry)
    dry_with_document.update(document_id=uuid4(), version_read=1)
    invalid_rows.append(dry_with_document)
    committed_with_preview = dict(committed)
    committed_with_preview.update(
        preview=preview,
        preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
        preview_full_char_count=len(preview),
    )
    invalid_rows.append(committed_with_preview)
    descending_versions = dict(committed)
    descending_versions.update(version_read=3, version_committed=2)
    invalid_rows.append(descending_versions)
    retries_above_attempts = _valid_page_row()
    retries_above_attempts.update(attempt_count=1, conflict_retry_count=2)
    invalid_rows.append(retries_above_attempts)
    planned_with_document = _valid_page_row()
    planned_with_document["document_id"] = uuid4()
    invalid_rows.append(planned_with_document)
    committed_without_lint = dict(committed)
    committed_without_lint["lint_summary"] = None
    invalid_rows.append(committed_without_lint)
    for row in invalid_rows:
        with pytest.raises((TypeError, ValueError)):
            records._decode_page(row)


def test_page_decoder_requires_exact_committed_version_predecessor():
    def committed(version_read, version_committed):
        row = _valid_page_row()
        row.update(
            state="committed",
            document_id=uuid4(),
            version_read=version_read,
            version_committed=version_committed,
            lint_summary={},
        )
        return row

    assert records._decode_page(committed(None, 1)).version_read is None
    assert records._decode_page(committed(2, 3)).version_read == 2
    for row in (committed(1, 3), committed(None, 2)):
        with pytest.raises((TypeError, ValueError)):
            records._decode_page(row)


def test_dry_run_complete_requires_an_attempt():
    preview = "Dry preview"

    def dry_run(attempt_count):
        row = _valid_page_row()
        row.update(
            state="dry_run_complete",
            attempt_count=attempt_count,
            preview=preview,
            preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
            preview_full_char_count=len(preview),
        )
        return row

    with pytest.raises((TypeError, ValueError)):
        records._decode_page(dry_run(0))
    assert records._decode_page(dry_run(1)).attempt_count == 1


def test_step_decoder_rejects_invalid_json_tokens_and_citations():
    now = datetime.now(UTC)
    row = {
        "id": uuid4(),
        "run_id": uuid4(),
        "run_page_id": None,
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "sequence": 1,
        "step_type": "read",
        "status": "succeeded",
        "input_digest": "a" * 64,
        "output_summary": {},
        "citation_identities": [],
        "prompt_version": None,
        "prompt_digest": None,
        "model_profile_version": "profile-v1",
        "reserved_tokens": 0,
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "latency_ms": 1.5,
        "error_code": None,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    for field, value in (
        ("output_summary", []),
        ("citation_identities", {}),
        ("citation_identities", [{"document_id": "bad", "document_version": 1, "chunk_index": 0, "page": None}]),
        ("input_tokens", True),
        ("total_tokens", 6),
        ("latency_ms", True),
        ("status", "waiting"),
    ):
        malformed = dict(row)
        malformed[field] = value
        with pytest.raises((TypeError, ValueError)):
            records._decode_step(malformed)

    over_cap_sequence = dict(row)
    over_cap_sequence["sequence"] = 513
    with pytest.raises((TypeError, ValueError)):
        records._decode_step(over_cap_sequence)


def test_step_decoder_enforces_exact_draft_reservation_contract():
    now = datetime.now(UTC)
    row = {
        "id": uuid4(),
        "run_id": uuid4(),
        "run_page_id": uuid4(),
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "sequence": 1,
        "step_type": "draft",
        "status": "failed",
        "input_digest": "a" * 64,
        "output_summary": {"outcome": "failed"},
        "citation_identities": [],
        "prompt_version": "writer-v1",
        "prompt_digest": "b" * 64,
        "model_profile_version": "profile-v1",
        "reserved_tokens": 10,
        "input_tokens": 6,
        "output_tokens": 4,
        "total_tokens": 10,
        "latency_ms": 0.0,
        "error_code": "rag_invalid_draft",
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    assert records._decode_step(row).reserved_tokens == 10

    for field, value in (
        ("reserved_tokens", True),
        ("reserved_tokens", -1),
        ("reserved_tokens", 250_001),
    ):
        invalid = dict(row)
        invalid[field] = value
        with pytest.raises((TypeError, ValueError)):
            records._decode_step(invalid)

    over_reservation = dict(row)
    over_reservation.update(reserved_tokens=9)
    with pytest.raises((TypeError, ValueError)):
        records._decode_step(over_reservation)

    wrong_step_type = dict(row)
    wrong_step_type.update(step_type="read")
    with pytest.raises((TypeError, ValueError)):
        records._decode_step(wrong_step_type)


def test_direct_decoders_reject_stringified_json_boundaries():
    run = _valid_run_row()
    run["budget"] = json.dumps(run["budget"])
    with pytest.raises((TypeError, ValueError)):
        records._decode_run(run)

    run = _valid_run_row()
    run["usage"] = json.dumps(run["usage"])
    with pytest.raises((TypeError, ValueError)):
        records._decode_run(run)

    now = datetime.now(UTC)
    step = {
        "id": uuid4(),
        "run_id": uuid4(),
        "run_page_id": None,
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "sequence": 1,
        "step_type": "read",
        "status": "succeeded",
        "input_digest": "a" * 64,
        "output_summary": json.dumps({}),
        "citation_identities": json.dumps([]),
        "prompt_version": None,
        "prompt_digest": None,
        "model_profile_version": "profile-v1",
        "reserved_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "error_code": None,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    with pytest.raises((TypeError, ValueError)):
        records._decode_step(step)


@pytest.mark.parametrize(
    "encoded",
    ('{"max_pages":8,"max_pages":9}', '{"max_pages":NaN}'),
)
def test_database_json_adapter_rejects_duplicate_keys_and_nonfinite_values(encoded):
    row = _valid_run_row()
    row["budget"] = encoded
    with pytest.raises((TypeError, ValueError)):
        records._adapt_db_run_row(row)


def _nested_json(depth: int) -> dict[str, object]:
    value: dict[str, object] = {}
    for _ in range(depth):
        value = {"nested": value}
    return value


def test_json_boundaries_reject_oversized_wire_values_before_decode():
    row = _valid_run_row()
    row["budget"] = " " * 4_097
    with pytest.raises((TypeError, ValueError), match="byte limit"):
        records._adapt_db_run_row(row)


def test_json_boundaries_reject_excessive_nesting_for_wire_and_direct_values():
    row = _valid_run_row()
    row["usage"] = json.dumps({"steps": 0, "model_tokens": _nested_json(33)})
    with pytest.raises((TypeError, ValueError), match="nesting"):
        records._adapt_db_run_row(row)

    step = {
        "id": uuid4(),
        "run_id": uuid4(),
        "run_page_id": None,
        "user_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "sequence": 1,
        "step_type": "read",
        "status": "succeeded",
        "input_digest": "a" * 64,
        "output_summary": _nested_json(33),
        "citation_identities": [],
        "prompt_version": None,
        "prompt_digest": None,
        "model_profile_version": "profile-v1",
        "reserved_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
        "error_code": None,
        "error_message": None,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    with pytest.raises((TypeError, ValueError), match="nesting"):
        records._decode_step(step)
