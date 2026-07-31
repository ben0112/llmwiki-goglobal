"""Direct contract tests for the hosted Postgres wiki transaction adapter."""

import asyncio
import hashlib
import json
import uuid
from contextlib import suppress
from dataclasses import replace

import asyncpg
import pytest
from jobs import repository as jobs_repository
from jobs.models import JobCreate, JobType
from jobs.service import JobService
from rag import repository
from rag.model import RagModelResponse, RagTokenUsage
from rag.page_runner import PageRunner, PostgresPageStore
from rag.retrieval import RagEvidence
from rag.wiki_writer import InjectedCrash, PersistedRagLintError, PostgresRagWikiWriter

import llmwiki_adapters.postgres.wiki as postgres_wiki
from llmwiki_adapters.postgres.wiki import write_wiki_bundle_in_transaction
from llmwiki_core.documents import DocumentKind, DocumentStatus
from llmwiki_core.rag import (
    RagCompletionReason,
    RagDomainError,
    RagRunConfig,
    RagStepStatus,
    RagStepType,
    RagUsage,
    RagWorkItem,
)
from llmwiki_core.references import ReferenceEdge
from llmwiki_core.search import SearchHit, SearchResult
from llmwiki_core.wiki import VersionConflict, WikiWriteBundle


def _bundle(document_id: uuid.UUID, target_id: uuid.UUID) -> WikiWriteBundle:
    return WikiWriteBundle.build(
        document_id=str(document_id),
        expected_version=None,
        filename="atomic-rag-page.md",
        path="/wiki/",
        file_type="md",
        content="跨境 atomic content " * 120,
        title="Atomic RAG Page",
        tags=["rag", "跨境"],
        date="2026-07-26",
        metadata={"description": "Direct shared writer contract"},
        edges=[ReferenceEdge(str(target_id), "cites", 7)],
    )


def _nested_metadata(depth: int):
    value = "leaf"
    for _ in range(depth):
        value = {"nested": value}
    return value


class _NoTransactionConnection:
    def __init__(self) -> None:
        self.write_attempted = False

    def is_in_transaction(self) -> bool:
        return False

    def __getattr__(self, name):
        self.write_attempted = True
        raise AssertionError(f"database operation attempted through {name}")


class _InTransactionNoSqlConnection:
    def __init__(self) -> None:
        self.calls = []

    def is_in_transaction(self) -> bool:
        return True

    def __getattr__(self, name):
        async def reject_sql(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            raise AssertionError(f"invalid bundle reached SQL through {name}")

        return reject_sql


async def test_shared_writer_requires_explicit_transaction_before_any_write():
    conn = _NoTransactionConnection()

    with pytest.raises(RuntimeError, match="wiki writer requires an explicit transaction"):
        await write_wiki_bundle_in_transaction(
            conn,
            user_id=uuid.uuid4(),
            knowledge_base_id=uuid.uuid4(),
            bundle=_bundle(uuid.uuid4(), uuid.uuid4()),
        )

    assert conn.write_attempted is False


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("filename NUL", lambda bundle: replace(bundle, filename="bad\0.md")),
        ("content surrogate", lambda bundle: replace(bundle, content="bad\ud800")),
        ("title NUL", lambda bundle: replace(bundle, title="bad\0title")),
        ("tag surrogate", lambda bundle: replace(bundle, tags=("bad\udfff",))),
        ("date NUL", lambda bundle: replace(bundle, date="2026-07-26\0")),
        ("file_type surrogate", lambda bundle: replace(bundle, file_type="m\ud800d")),
        ("path surrogate", lambda bundle: replace(bundle, path="/wiki/\udfff/")),
        (
            "nested metadata key NUL",
            lambda bundle: replace(bundle, metadata={"outer": {"bad\0key": "value"}}),
        ),
        (
            "nested metadata value surrogate",
            lambda bundle: replace(bundle, metadata={"outer": ["bad\ud800"]}),
        ),
        (
            "reference type NUL",
            lambda bundle: replace(
                bundle,
                edges=(replace(bundle.edges[0], reference_type="cites\0"),),
            ),
        ),
        (
            "edge page int4 overflow",
            lambda bundle: replace(
                bundle,
                edges=(replace(bundle.edges[0], page=2_147_483_648),),
            ),
        ),
        (
            "metadata depth overflow",
            lambda bundle: replace(bundle, metadata={"root": _nested_metadata(33)}),
        ),
        (
            "metadata size overflow",
            lambda bundle: replace(bundle, metadata={"value": "x" * 1_048_577}),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_shared_writer_rejects_unrepresentable_values_before_sql(field, mutate):
    del field
    conn = _InTransactionNoSqlConnection()
    bundle = mutate(_bundle(uuid.uuid4(), uuid.uuid4()))

    with pytest.raises(ValueError):
        await write_wiki_bundle_in_transaction(
            conn,
            user_id=uuid.uuid4(),
            knowledge_base_id=uuid.uuid4(),
            bundle=bundle,
        )

    assert conn.calls == []


@pytest.mark.parametrize(
    "invalid_content",
    ["postgres NUL: \0", "postgres surrogate: \ud800"],
)
async def test_invalid_text_leaves_outer_postgres_transaction_usable(pool, invalid_content):
    async with pool.acquire() as conn:
        async with conn.transaction():
            with pytest.raises(ValueError):
                await write_wiki_bundle_in_transaction(
                    conn,
                    user_id=uuid.uuid4(),
                    knowledge_base_id=uuid.uuid4(),
                    bundle=replace(
                        _bundle(uuid.uuid4(), uuid.uuid4()),
                        content=invalid_content,
                    ),
                )
            assert conn.is_in_transaction()
            assert await conn.fetchval("SELECT 42") == 42
        assert not conn.is_in_transaction()


async def test_shared_writer_propagates_cancellation_and_rolls_back(pool, monkeypatch):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Cancelled Writer')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )

    async def cancel(*args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(postgres_wiki, "_replace_chunks", cancel)

    async with pool.acquire() as conn:
        with pytest.raises(asyncio.CancelledError):
            async with conn.transaction():
                await write_wiki_bundle_in_transaction(
                    conn,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    bundle=replace(
                        _bundle(document_id, uuid.uuid4()),
                        edges=(),
                    ),
                )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                document_id,
                user_id,
                knowledge_base_id,
            )
            == 0
        )


async def test_shared_writer_rolls_back_document_chunks_references_and_facet_rollup(pool):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    source_id = uuid.uuid4()
    document_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'RAG Writer')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, title, path, source_kind, "
        "file_type, status, content, metadata, version) "
        "VALUES ($1, $2, $3, 'source.md', 'Source', '/corpus/', 'source', "
        "'md', 'ready', 'source', $4::jsonb, 1)",
        source_id,
        knowledge_base_id,
        user_id,
        '{"entry_id":"E-ROLLBACK","stage":"S4","timeliness":"M1"}',
    )

    class RollbackSignal(Exception):
        pass

    async with pool.acquire() as conn:
        with pytest.raises(RollbackSignal):
            async with conn.transaction():
                result = await write_wiki_bundle_in_transaction(
                    conn,
                    user_id=user_id,
                    knowledge_base_id=knowledge_base_id,
                    bundle=_bundle(document_id, source_id),
                )
                assert result.document_id == document_id
                persisted = await conn.fetchrow(
                    "SELECT version, metadata FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                    document_id,
                    user_id,
                    knowledge_base_id,
                )
                assert persisted["version"] == 1
                metadata = persisted["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                assert metadata["facet_rollup"]["stage"] == ["S4"]
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM document_chunks "
                        "WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                        document_id,
                        user_id,
                        knowledge_base_id,
                    )
                    > 0
                )
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM document_references WHERE source_document_id=$1 AND knowledge_base_id=$2",
                        document_id,
                        knowledge_base_id,
                    )
                    == 1
                )
                raise RollbackSignal

        assert (
            await conn.fetchval(
                "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                document_id,
                user_id,
                knowledge_base_id,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM document_chunks WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                document_id,
                user_id,
                knowledge_base_id,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM document_references WHERE source_document_id=$1 AND knowledge_base_id=$2",
                document_id,
                knowledge_base_id,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM documents WHERE id=$1 AND metadata ? 'facet_rollup'",
                document_id,
            )
            == 0
        )


async def test_shared_writer_replaces_large_unicode_revision_with_zero_chunks(pool):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    source_id = uuid.uuid4()
    document_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Boundary Writer')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, title, path, source_kind, "
        "file_type, status, content, metadata, version) "
        "VALUES ($1, $2, $3, 'boundary-source.md', 'Boundary Source', '/corpus/', "
        "'source', 'md', 'ready', 'source', $4::jsonb, 1)",
        source_id,
        knowledge_base_id,
        user_id,
        '{"entry_id":"E-BOUNDARY","stage":"S3"}',
    )
    large_content = "# 跨境规则\n\n数据本地化与证据链。" * 2_000
    created_bundle = replace(_bundle(document_id, source_id), content=large_content)

    async with pool.acquire() as conn:
        async with conn.transaction():
            created = await write_wiki_bundle_in_transaction(
                conn,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                bundle=created_bundle,
            )
        assert created.version == 1
        chunk_lengths = await conn.fetch(
            "SELECT document_version, length(content) AS size FROM document_chunks "
            "WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        )
        assert len(chunk_lengths) > 1
        assert {row["document_version"] for row in chunk_lengths} == {1}
        assert max(row["size"] for row in chunk_lengths) <= 10_000

        empty_bundle = replace(
            created_bundle,
            expected_version=1,
            content="",
            edges=(),
        )
        async with conn.transaction():
            emptied = await write_wiki_bundle_in_transaction(
                conn,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                bundle=empty_bundle,
            )
        assert emptied.version == 2
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM document_chunks WHERE document_id=$1",
                document_id,
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM document_references WHERE source_document_id=$1",
                document_id,
            )
            == 0
        )
        persisted = await conn.fetchrow(
            "SELECT content, version, metadata FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        )
        metadata = persisted["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert (persisted["content"], persisted["version"]) == ("", 2)
        assert "facet_rollup" not in metadata

        with pytest.raises(VersionConflict, match="is not at version 1"):
            async with conn.transaction():
                await write_wiki_bundle_in_transaction(
                    conn,
                    user_id=uuid.uuid4(),
                    knowledge_base_id=knowledge_base_id,
                    bundle=replace(empty_bundle, content="cross-tenant overwrite"),
                )
        assert (
            await conn.fetchval(
                "SELECT content FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                document_id,
                user_id,
                knowledge_base_id,
            )
            == ""
        )


async def test_mutual_link_updates_do_not_deadlock(pool, monkeypatch):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document_a = uuid.uuid4()
    document_b = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Deadlock Writer')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    await pool.executemany(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, title, path, source_kind, "
        "file_type, status, content, metadata, version) "
        "VALUES ($1, $2, $3, $4, $4, '/wiki/', 'wiki', 'md', 'ready', $5, '{}', 1)",
        [
            (document_a, knowledge_base_id, user_id, "a.md", "A version 1"),
            (document_b, knowledge_base_id, user_id, "b.md", "B version 1"),
        ],
    )
    await pool.executemany(
        "INSERT INTO document_references "
        "(source_document_id, target_document_id, knowledge_base_id, reference_type) "
        "VALUES ($1, $2, $3, 'links_to')",
        [
            (document_a, document_b, knowledge_base_id),
            (document_b, document_a, knowledge_base_id),
        ],
    )
    arrivals = 0
    both_document_rows_locked = asyncio.Event()
    original_replace_chunks = postgres_wiki._replace_chunks

    async def synchronize_after_document_update(*args, **kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            both_document_rows_locked.set()
        with suppress(TimeoutError):
            await asyncio.wait_for(both_document_rows_locked.wait(), timeout=0.2)
        await original_replace_chunks(*args, **kwargs)

    monkeypatch.setattr(postgres_wiki, "_replace_chunks", synchronize_after_document_update)

    def update_bundle(document_id, filename, target_id):
        return WikiWriteBundle.build(
            document_id=str(document_id),
            expected_version=1,
            filename=filename,
            path="/wiki/",
            file_type="md",
            content=f"{filename} version 2",
            title=filename,
            tags=["mutual"],
            date="2026-07-26",
            metadata={"description": "Mutual link update"},
            edges=[ReferenceEdge(str(target_id), "links_to")],
        )

    async def update(bundle):
        async with pool.acquire() as conn, conn.transaction():
            return await write_wiki_bundle_in_transaction(
                conn,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                bundle=bundle,
            )

    results = await asyncio.gather(
        update(update_bundle(document_a, "a.md", document_b)),
        update(update_bundle(document_b, "b.md", document_a)),
        return_exceptions=True,
    )

    assert not any(isinstance(result, asyncpg.DeadlockDetectedError) for result in results)
    assert all(not isinstance(result, BaseException) for result in results), results
    assert {result.document_id: result.version for result in results} == {
        document_a: 2,
        document_b: 2,
    }
    rows = await pool.fetch(
        "SELECT id, version, stale_since FROM documents "
        "WHERE user_id=$1 AND knowledge_base_id=$2 AND id=ANY($3::uuid[]) ORDER BY id",
        user_id,
        knowledge_base_id,
        [document_a, document_b],
    )
    assert {row["version"] for row in rows} == {2}
    assert sum(row["stale_since"] is not None for row in rows) == 1
    references = await pool.fetch(
        "SELECT source_document_id, target_document_id FROM document_references "
        "WHERE knowledge_base_id=$1 ORDER BY source_document_id",
        knowledge_base_id,
    )
    assert {(row[0], row[1]) for row in references} == {
        (document_a, document_b),
        (document_b, document_a),
    }


async def _atomic_rag_state(pool):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    run_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id,email,display_name) VALUES($1,$2,'Atomic RAG Boundary')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id,user_id,name,slug) VALUES($1,$2,$3,$4)",
        knowledge_base_id,
        user_id,
        f"Atomic RAG {knowledge_base_id}",
        f"atomic-rag-{knowledge_base_id}",
    )
    config = RagRunConfig.build(
        knowledge_base_id=knowledge_base_id,
        goal="Build one atomic page",
        target_path_prefix="/wiki/atomic/",
        model_profile="primary",
    )
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            JobCreate(
                job_type=JobType.BUILD_WIKI,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                payload={"run_id": str(run_id)},
                idempotency_key=f"atomic-rag-{run_id}",
            ),
            authenticated_user_id=user_id,
        )
        run = await repository.create_root(
            conn,
            run_id=run_id,
            job_id=job.id,
            user_id=user_id,
            config=config,
            idempotency_key=f"atomic-rag-{run_id}",
            request_digest="d" * 64,
            model_profile_version="primary-v1",
        )
        pages = await repository.insert_worklist(
            conn,
            run,
            (RagWorkItem.build(0, "/wiki/atomic/page.md", "Atomic page", "atomic evidence"),),
        )
        page = await repository.begin_page_attempt(conn, pages[0].id, max_attempts=2)
        await conn.execute(
            "UPDATE background_jobs SET state='running',lease_owner='worker-atomic',"
            "lease_expires_at=clock_timestamp()+interval '5 minutes',heartbeat_at=clock_timestamp() "
            "WHERE id=$1",
            job.id,
        )
    bundle = WikiWriteBundle.build(
        document_id=str(uuid.uuid4()),
        expected_version=None,
        filename="page.md",
        path="/wiki/atomic/",
        file_type="md",
        content="# Atomic page\n\nPersist all facts together.",
        title="Atomic page",
        tags=["atomic"],
        date="2026-07-27",
        metadata={"description": "Atomic boundary"},
        edges=[],
    )
    return run, page, job, bundle


async def test_rag_writer_crash_between_shared_write_and_boundary_rolls_back_everything(pool):
    run, page, job, bundle = await _atomic_rag_state(pool)
    writer = PostgresRagWikiWriter(pool, failpoint="after_shared_writer_before_boundary")

    with pytest.raises(InjectedCrash):
        await writer.commit(
            job_id=job.id,
            lease_owner="worker-atomic",
            run=run,
            page=page,
            bundle=bundle,
            usage=RagUsage(),
        )

    assert (
        await pool.fetchval(
            "SELECT version FROM documents WHERE user_id=$1 AND knowledge_base_id=$2 "
            "AND path='/wiki/atomic/' AND filename='page.md'",
            run.user_id,
            run.knowledge_base_id,
        )
        is None
    )
    assert await pool.fetchval("SELECT last_committed_ordinal FROM rag_runs WHERE id=$1", run.id) == -1


async def test_rag_writer_persisted_lint_failure_rolls_back_page_and_boundary(pool):
    run, page, job, bundle = await _atomic_rag_state(pool)

    class RejectingPersistedLinter:
        async def lint_persisted_in_transaction(self, conn, **kwargs):
            del conn, kwargs
            raise PersistedRagLintError()

    writer = PostgresRagWikiWriter(pool, linter=RejectingPersistedLinter())

    with pytest.raises(PersistedRagLintError):
        await writer.commit(
            job_id=job.id,
            lease_owner="worker-atomic",
            run=run,
            page=page,
            bundle=bundle,
            usage=RagUsage(),
        )

    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE user_id=$1 AND knowledge_base_id=$2 "
            "AND path='/wiki/atomic/' AND filename='page.md'",
            run.user_id,
            run.knowledge_base_id,
        )
        == 0
    )
    assert await pool.fetchval("SELECT last_committed_ordinal FROM rag_runs WHERE id=$1", run.id) == -1


async def test_rag_writer_exact_lint_rejects_silently_missing_content_reference(pool, monkeypatch):
    run, page, job, bundle = await _atomic_rag_state(pool)
    target_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,title,path,source_kind,file_type,status,content,version) "
        "VALUES($1,$2,$3,'target.md','Target','/wiki/atomic/','wiki','md','ready','target',1)",
        target_id,
        run.knowledge_base_id,
        run.user_id,
    )
    bundle = replace(bundle, edges=(ReferenceEdge(str(target_id), "links_to", 7),))

    async def omit_content_references(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr(postgres_wiki, "_replace_content_references", omit_content_references)
    writer = PostgresRagWikiWriter(pool)

    with pytest.raises(PersistedRagLintError):
        await writer.commit(
            job_id=job.id,
            lease_owner="worker-atomic",
            run=run,
            page=page,
            bundle=bundle,
            usage=RagUsage(),
        )

    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            uuid.UUID(bundle.document_id),
            run.user_id,
            run.knowledge_base_id,
        )
        == 0
    )
    assert await pool.fetchval("SELECT last_committed_ordinal FROM rag_runs WHERE id=$1", run.id) == -1


async def test_rag_writer_crash_after_transaction_return_leaves_complete_atomic_boundary(pool):
    run, page, job, bundle = await _atomic_rag_state(pool)
    writer = PostgresRagWikiWriter(pool, failpoint="after_transaction_return")

    with pytest.raises(InjectedCrash):
        await writer.commit(
            job_id=job.id,
            lease_owner="worker-atomic",
            run=run,
            page=page,
            bundle=bundle,
            usage=RagUsage(),
        )

    persisted = await pool.fetchrow(
        "SELECT id,version FROM documents WHERE user_id=$1 AND knowledge_base_id=$2 "
        "AND path='/wiki/atomic/' AND filename='page.md'",
        run.user_id,
        run.knowledge_base_id,
    )
    assert persisted is not None and persisted["version"] == 1
    assert await pool.fetchval("SELECT last_committed_ordinal FROM rag_runs WHERE id=$1", run.id) == 0
    boundary = await pool.fetchrow(
        "SELECT state,document_id,version_committed FROM rag_run_pages WHERE id=$1",
        page.id,
    )
    assert tuple(boundary.values()) == ("committed", persisted["id"], 1)


async def test_real_page_runner_crash_replay_preserves_atomic_usage_and_job_binding(pool):
    run, page, job, _ = await _atomic_rag_state(pool)
    source_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,title,path,source_kind,file_type,status,content,version) "
        "VALUES($1,$2,$3,'source.pdf','Source','/corpus/','source','pdf','ready','source evidence',3)",
        source_id,
        run.knowledge_base_id,
        run.user_id,
    )
    hit = SearchHit(str(source_id), 3, 4, "source evidence", 0.9, "/corpus/source.pdf", page=5)
    evidence = RagEvidence(
        document_id=source_id,
        document_version=3,
        chunk_index=4,
        page=5,
        filename="source.pdf",
        path="/corpus/",
        title="Source",
        content="Selected exact source evidence",
        status=DocumentStatus.READY,
        archived=False,
        score=0.9,
        document_kind=DocumentKind.SOURCE,
    )

    class Lease:
        async def checkpoint(self):
            async with pool.acquire() as conn, conn.transaction():
                return await jobs_repository.assert_active(conn, job.id, "worker-atomic")

    class Gate:
        async def ensure_enabled(self):
            return

    class Retrieval:
        async def retrieve(self, query, *, profile):
            del query
            return SearchResult((hit,), 1, profile=profile)

    class EvidenceReader:
        async def read(self, user_id, knowledge_base_id, hits, max_chars):
            assert (user_id, knowledge_base_id, hits) == (
                run.user_id,
                run.knowledge_base_id,
                (hit,),
            )
            assert max_chars == run.budget.max_context_chars
            return (evidence,)

    class WikiPageReader:
        async def get_by_path(self, user_id, knowledge_base_id, path):
            assert (user_id, knowledge_base_id, path) == (
                run.user_id,
                run.knowledge_base_id,
                page.path,
            )

    class Model:
        calls = 0

        async def complete_json(self, **kwargs):
            del kwargs
            self.calls += 1
            content = (
                "---\n"
                "title: Atomic page\n"
                "tags: [atomic, evidence]\n"
                "description: A sourced atomic page.\n"
                "date: 2026-07-27\n"
                "---\n"
                "# Atomic page\n\n"
                "```mermaid\ngraph TD\n  A --> B\n```\n\n"
                "Atomic evidence is durable.[^1]\n\n[^1]: source.pdf, p.5\n"
            )
            return RagModelResponse(
                {
                    "content": content,
                    "citations": [
                        {
                            "document_id": str(source_id),
                            "document_version": 3,
                            "chunk_index": 4,
                            "page": 5,
                        }
                    ],
                },
                RagTokenUsage(10, 20, 30),
            )

    class Linter:
        async def lint(self, **kwargs):
            del kwargs
            return {"warnings": 0}

    model = Model()
    writer = PostgresRagWikiWriter(pool, failpoint="after_transaction_return")
    runner = PageRunner(
        store=PostgresPageStore(pool),
        model=model,
        retrieval=Retrieval(),
        evidence_reader=EvidenceReader(),
        wiki_page_reader=WikiPageReader(),
        wiki_writer=writer,
        draft_linter=Linter(),
        feature_gate=Gate(),
    )

    with pytest.raises(InjectedCrash):
        await runner.run(run, page, Lease())

    async with pool.acquire() as conn, conn.transaction():
        replay_run = await repository.get_for_worker(conn, run.id, job.id)
        replay_page = (await repository.list_pages(conn, run.id))[0]
    assert replay_run is not None
    assert replay_run.usage == RagUsage(steps=6, model_tokens=30)
    assert replay_page.state.value == "committed"
    assert replay_page.version_committed == 1
    assert model.calls == 1

    writer.failpoint = None
    replayed = await runner.run(replay_run, replay_page, Lease())

    assert replayed.run == replay_run
    assert replayed.page == replay_page
    assert model.calls == 1
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            replay_page.document_id,
            run.user_id,
            run.knowledge_base_id,
        )
        == 1
    )
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM rag_steps WHERE run_id=$1 AND status='succeeded'",
            run.id,
        )
        == 6
    )


async def test_two_page_dry_run_keeps_commit_boundary_negative_and_replays_terminal_state(pool):
    user_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    run_id = uuid.uuid4()
    await pool.execute(
        "INSERT INTO users (id,email,display_name) VALUES($1,$2,'Dry RAG Boundary')",
        user_id,
        f"{user_id}@test.invalid",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id,user_id,name,slug) VALUES($1,$2,$3,$4)",
        knowledge_base_id,
        user_id,
        f"Dry RAG {knowledge_base_id}",
        f"dry-rag-{knowledge_base_id}",
    )
    config = RagRunConfig.build(
        knowledge_base_id=knowledge_base_id,
        goal="Preview two pages",
        target_path_prefix="/wiki/dry/",
        model_profile="primary",
        dry_run=True,
    )
    async with pool.acquire() as conn, conn.transaction():
        job = await JobService(pool).create_in_transaction(
            conn,
            JobCreate(
                job_type=JobType.BUILD_WIKI,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                payload={"run_id": str(run_id)},
                idempotency_key=f"dry-rag-{run_id}",
            ),
            authenticated_user_id=user_id,
        )
        run = await repository.create_root(
            conn,
            run_id=run_id,
            job_id=job.id,
            user_id=user_id,
            config=config,
            idempotency_key=f"dry-rag-{run_id}",
            request_digest="e" * 64,
            model_profile_version="primary-v1",
        )
        pages = await repository.insert_worklist(
            conn,
            run,
            tuple(
                RagWorkItem.build(index, f"/wiki/dry/page-{index}.md", f"Page {index}", f"query {index}")
                for index in range(2)
            ),
        )

    current_run = run
    for index, planned in enumerate(pages):
        preview = f"preview {index}"
        async with pool.acquire() as conn, conn.transaction():
            running = await repository.begin_page_attempt(conn, planned.id, max_attempts=2)
            current_run, completed = await repository.mark_dry_run_complete(
                conn,
                run=current_run,
                page=running,
                usage=RagUsage(),
                preview=preview,
                preview_digest=hashlib.sha256(preview.encode()).hexdigest(),
                preview_full_char_count=len(preview),
                preview_truncated=False,
            )
        assert completed.state.value == "dry_run_complete"
        assert current_run.last_committed_ordinal == -1

    async with pool.acquire() as conn, conn.transaction():
        finished = await repository.finish_run(
            conn,
            run_id=current_run.id,
            completion_reason=RagCompletionReason.DRY_RUN,
            usage=RagUsage(),
        )
        replay_pages = await repository.list_pages(conn, run.id)

    assert finished.completion_reason is RagCompletionReason.DRY_RUN
    assert finished.last_committed_ordinal == -1
    assert [page.state.value for page in replay_pages] == ["dry_run_complete", "dry_run_complete"]
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE user_id=$1 AND knowledge_base_id=$2 AND source_kind='wiki'",
            user_id,
            knowledge_base_id,
        )
        == 0
    )


async def test_new_attempt_recovers_stale_running_page_step_before_replay(pool):
    run, page, _, _ = await _atomic_rag_state(pool)
    async with pool.acquire() as conn, conn.transaction():
        stale = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.RETRIEVE,
            input_digest="f" * 64,
        )

    async with pool.acquire() as conn, conn.transaction():
        replayed = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        recovered = await conn.fetchrow(
            "SELECT status,error_code,output_summary FROM rag_steps WHERE id=$1",
            stale.id,
        )
        next_step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.RETRIEVE,
            input_digest="a" * 64,
        )

    assert replayed.attempt_count == 2
    assert recovered["status"] == "failed"
    assert recovered["error_code"] == "rag_attempt_interrupted"
    assert "private" not in str(recovered["output_summary"])
    assert next_step.status.value == "running"


async def test_public_page_store_commits_stale_step_recovery_before_attempt_cap_failure(pool):
    run, page, job, _ = await _atomic_rag_state(pool)
    async with pool.acquire() as conn, conn.transaction():
        capped_page = await repository.begin_page_attempt(conn, page.id, max_attempts=2)
        stale = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.DRAFT,
            input_digest="e" * 64,
            reserved_tokens=100,
        )

    store = PostgresPageStore(pool)
    with pytest.raises(RagDomainError) as exhausted:
        await store.begin_page_attempt(
            job_id=job.id,
            lease_owner="worker-atomic",
            run=run,
            page=capped_page,
            max_attempts=2,
        )

    assert exhausted.value.code == "rag_page_attempts_exhausted"
    recovered = await pool.fetchrow(
        "SELECT status,error_code,output_summary FROM rag_steps WHERE id=$1",
        stale.id,
    )
    assert recovered["status"] == "failed"
    assert recovered["error_code"] == "rag_attempt_interrupted"
    recovered_summary = recovered["output_summary"]
    if isinstance(recovered_summary, str):
        recovered_summary = json.loads(recovered_summary)
    assert recovered_summary == {"outcome": "interrupted", "usage_trusted": False}

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE background_jobs SET state='failed' WHERE id=$1", job.id)
        terminal = await repository.record_terminal_job_state(
            conn,
            run_id=run.id,
            completion_reason=RagCompletionReason.BUDGET_EXHAUSTED,
        )
    assert terminal.completion_reason is RagCompletionReason.BUDGET_EXHAUSTED


async def test_interrupted_draft_charges_durable_reservation_before_replay_can_call_model_again(pool):
    run, page, job, _ = await _atomic_rag_state(pool)
    store = PostgresPageStore(pool)
    scope = {"job_id": job.id, "lease_owner": "worker-atomic", "run": run}
    stale = await store.start_step(
        page=page,
        step_type=RagStepType.DRAFT,
        input_digest="7" * 64,
        reserved_tokens=run.budget.max_model_tokens,
        **scope,
    )

    replayed = await store.begin_page_attempt(
        page=page,
        max_attempts=run.budget.max_page_attempts,
        **scope,
    )
    usage = await store.load_usage(**scope)

    recovered = await pool.fetchrow(
        "SELECT status,reserved_tokens,input_tokens,output_tokens,total_tokens,error_code,output_summary "
        "FROM rag_steps WHERE id=$1",
        stale.id,
    )
    summary = recovered["output_summary"]
    if isinstance(summary, str):
        summary = json.loads(summary)
    assert tuple(recovered.values())[:6] == (
        "failed",
        run.budget.max_model_tokens,
        run.budget.max_model_tokens,
        0,
        run.budget.max_model_tokens,
        "rag_attempt_interrupted",
    )
    assert summary == {"outcome": "interrupted", "usage_trusted": False}
    assert usage == RagUsage(steps=1, model_tokens=run.budget.max_model_tokens)

    with pytest.raises(RagDomainError) as exhausted:
        await store.start_step(
            page=replayed,
            step_type=RagStepType.DRAFT,
            input_digest="6" * 64,
            reserved_tokens=1,
            **scope,
        )
    assert exhausted.value.code == "rag_budget_exhausted"


async def test_attempt_recovery_never_consumes_a_running_step_from_another_page_or_run(pool):
    run, page, _, _ = await _atomic_rag_state(pool)
    async with pool.acquire() as conn, conn.transaction():
        stale = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.RETRIEVE,
            input_digest="d" * 64,
        )
        await conn.fetchrow(
            "INSERT INTO rag_run_pages(run_id,user_id,knowledge_base_id,ordinal,path,intent,query) "
            "VALUES($1,$2,$3,1,'/wiki/atomic/other.md','Other','other') RETURNING *",
            run.id,
            run.user_id,
            run.knowledge_base_id,
        )
        other_page = (await repository.list_pages(conn, run.id))[1]
        with pytest.raises(RagDomainError) as wrong_page:
            await repository.begin_page_attempt(conn, other_page.id, max_attempts=2)
        assert wrong_page.value.code == "rag_step_already_running"

    other_run, other_run_page, _, _ = await _atomic_rag_state(pool)
    async with pool.acquire() as conn, conn.transaction():
        attempted_other_run = await repository.begin_page_attempt(conn, other_run_page.id, max_attempts=2)

    assert attempted_other_run.run_id == other_run.id
    assert await pool.fetchval("SELECT status FROM rag_steps WHERE id=$1", stale.id) == "running"


async def test_postgres_page_adapters_reject_cross_tenant_active_job_before_any_side_effect(pool):
    _, _, foreign_job, _ = await _atomic_rag_state(pool)
    run, page, _, bundle = await _atomic_rag_state(pool)
    store = PostgresPageStore(pool)
    writer = PostgresRagWikiWriter(pool)
    foreign_scope = {"job_id": foreign_job.id, "lease_owner": "worker-atomic"}

    with pytest.raises(RagDomainError):
        await store.reload_boundary(run=run, page=page, **foreign_scope)
    with pytest.raises(RagDomainError):
        await store.load_usage(run=run, **foreign_scope)
    with pytest.raises(RagDomainError):
        await store.begin_page_attempt(run=run, page=page, max_attempts=2, **foreign_scope)
    with pytest.raises(RagDomainError):
        await store.record_page_read(
            run=run,
            page=page,
            document_id=None,
            version=None,
            **foreign_scope,
        )
    with pytest.raises(RagDomainError):
        await store.start_step(
            run=run,
            page=page,
            step_type=RagStepType.RETRIEVE,
            input_digest="c" * 64,
            **foreign_scope,
        )
    with pytest.raises(RagDomainError):
        await store.record_conflict(
            run=run,
            page=page,
            max_conflict_retries=run.budget.max_conflict_retries,
            **foreign_scope,
        )
    with pytest.raises(RagDomainError):
        await store.complete_dry_run(
            run=run,
            page=page,
            usage=RagUsage(),
            preview="bounded preview",
            preview_digest=hashlib.sha256(b"bounded preview").hexdigest(),
            preview_full_char_count=len("bounded preview"),
            preview_truncated=False,
            **foreign_scope,
        )
    with pytest.raises(RagDomainError):
        await writer.commit(
            run=run,
            page=page,
            bundle=bundle,
            usage=RagUsage(),
            **foreign_scope,
        )

    async with pool.acquire() as conn, conn.transaction():
        step = await repository.start_step(
            conn,
            run_id=run.id,
            page_id=page.id,
            step_type=RagStepType.READ,
            input_digest="b" * 64,
        )
    with pytest.raises(RagDomainError):
        await store.finish_step(
            run=run,
            page=page,
            step=step,
            status=RagStepStatus.FAILED,
            summary={"outcome": "failed"},
            citations=(),
            token_usage=RagTokenUsage(0, 0, 0),
            error_code="rag_internal_error",
            **foreign_scope,
        )

    persisted_page = await pool.fetchrow(
        "SELECT attempt_count,conflict_retry_count,document_id,version_read,state FROM rag_run_pages WHERE id=$1",
        page.id,
    )
    assert tuple(persisted_page.values()) == (1, 0, None, None, "running")
    assert await pool.fetchval("SELECT status FROM rag_steps WHERE id=$1", step.id) == "running"
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            uuid.UUID(bundle.document_id),
            run.user_id,
            run.knowledge_base_id,
        )
        == 0
    )
