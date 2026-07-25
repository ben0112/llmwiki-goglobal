import asyncio
from math import nan
from uuid import uuid4

import pytest
from jobs import repository
from jobs.handlers import (
    RetryableJobError,
    TerminalJobError,
    WorkerContext,
    handle_document_embed,
)
from jobs.lease import JobLease
from jobs.models import JobCreate, JobType, LeaseLost
from jobs.service import JobService
from scripts.enqueue_embeddings import reconcile_missing_embeddings

from llmwiki_core.models import EmbeddingProfile, EmbeddingUnavailable

PROFILE = EmbeddingProfile("openai_compatible", "embed-v1", 3)


async def _seed_document(pool, *, version=1, chunks=("first", "second")):
    user_id, kb_id, document_id = uuid4(), uuid4(), uuid4()
    await pool.execute(
        "INSERT INTO users (id,email) VALUES ($1,$2)", user_id, f"{user_id}@embed.test"
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id,user_id,name,slug) VALUES ($1,$2,$3,$4)",
        kb_id,
        user_id,
        f"KB {kb_id}",
        f"kb-{kb_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,version) "
        "VALUES ($1,$2,$3,'doc.md','/','source','md','ready',$4)",
        document_id,
        kb_id,
        user_id,
        version,
    )
    await pool.executemany(
        "INSERT INTO document_chunks "
        "(document_id,document_version,user_id,knowledge_base_id,chunk_index,content,source_content,token_count) "
        "VALUES ($1,$2,$3,$4,$5,$6,$6,1)",
        [(document_id, version, user_id, kb_id, index, text) for index, text in enumerate(chunks)],
    )
    return user_id, kb_id, document_id


async def _claimed_job(pool, ids, *, profile=PROFILE, version=1, owner="embed-worker"):
    user_id, kb_id, document_id = ids
    command = JobCreate(
        job_type=JobType.DOCUMENT_EMBED,
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=document_id,
        payload={
            "document_id": str(document_id),
            "document_version": version,
            "provider": profile.provider,
            "model": profile.model,
            "dimensions": profile.dimensions,
        },
        idempotency_key=(
            f"embed:{document_id}:{version}:{profile.provider}:{profile.model}:{profile.dimensions}"
        ),
    )
    async with pool.acquire() as conn, conn.transaction():
        created = await repository.create(conn, command)
    claimed = await repository.claim(pool, created.id, owner, 120)
    assert claimed is not None
    return claimed, JobLease(pool, claimed.id, owner, 120, 30)


def _context(pool):
    return WorkerContext(pool=pool, s3=None, converter_url="", converter_secret="")


def _install_profile(monkeypatch, profile=PROFILE):
    import jobs.handlers as handlers

    monkeypatch.setattr(handlers, "_configured_embedding_profile", lambda: profile)


@pytest.mark.asyncio
async def test_embedding_handler_batches_in_order_and_retry_is_idempotent(pool, monkeypatch):
    ids = await _seed_document(pool, chunks=("a", "b", "c"))
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)
    calls = []

    async def embed(profile, texts):
        calls.append(tuple(texts))
        return tuple((float(index + 1), 1.0, 0.0) for index in range(len(texts)))

    monkeypatch.setattr("jobs.handlers._embed_texts", embed)

    first = await handle_document_embed(job, lease, _context(pool))
    second = await handle_document_embed(job, lease, _context(pool))

    assert first == second == {"document_id": str(ids[2]), "embedded_chunks": 3}
    assert calls == [("a", "b", "c"), ("a", "b", "c")]
    rows = await pool.fetch(
        "SELECT chunk_index,embedding::text FROM chunk_embeddings "
        "WHERE document_id=$1 ORDER BY chunk_index",
        ids[2],
    )
    assert [row["chunk_index"] for row in rows] == [0, 1, 2]


@pytest.mark.asyncio
async def test_embedding_http_version_change_returns_stale_without_write(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)

    async def embed(_profile, texts):
        await pool.execute("DELETE FROM document_chunks WHERE document_id=$1", ids[2])
        await pool.execute("UPDATE documents SET version=2 WHERE id=$1", ids[2])
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    monkeypatch.setattr("jobs.handlers._embed_texts", embed)

    assert await handle_document_embed(job, lease, _context(pool)) == {
        "document_id": str(ids[2]),
        "stale": True,
    }
    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_embedding_lease_loss_before_commit_writes_nothing(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)

    async def embed(_profile, texts):
        await pool.execute(
            "UPDATE background_jobs SET lease_owner='new-worker' WHERE id=$1", job.id
        )
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    monkeypatch.setattr("jobs.handlers._embed_texts", embed)

    with pytest.raises(LeaseLost):
        await handle_document_embed(job, lease, _context(pool))
    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_embedding_profile_changed_before_execution_is_terminal(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch, EmbeddingProfile("openai_compatible", "embed-v2", 3))

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_embed(job, lease, _context(pool))

    assert raised.value.error_code == "embedding_profile_changed"
    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_embedding_profile_changed_during_http_cannot_write_old_profile(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    current = {"profile": PROFILE}
    monkeypatch.setattr(
        "jobs.handlers._configured_embedding_profile",
        lambda: current["profile"],
    )

    async def embed(_profile, texts):
        current["profile"] = EmbeddingProfile("openai_compatible", "embed-v2", 3)
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    monkeypatch.setattr("jobs.handlers._embed_texts", embed)

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_embed(job, lease, _context(pool))
    assert raised.value.error_code == "embedding_profile_changed"
    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_vector_commit_failure_rolls_back_complete_batch(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)
    monkeypatch.setattr(
        "jobs.handlers._embed_texts",
        lambda _profile, texts: asyncio.sleep(
            0,
            result=tuple((1.0, 0.0, 0.0) for _ in texts),
        ),
    )
    await pool.execute(
        f"CREATE FUNCTION fail_task8_vector_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
        f"BEGIN IF NEW.document_id='{ids[2]}'::uuid AND NEW.chunk_index=1 THEN "
        "RAISE EXCEPTION 'private dsn=postgres://token'; END IF; RETURN NEW; END $$"
    )
    await pool.execute(
        "CREATE TRIGGER fail_task8_vector_commit_trigger BEFORE INSERT ON chunk_embeddings "
        "FOR EACH ROW EXECUTE FUNCTION fail_task8_vector_commit()"
    )
    try:
        with pytest.raises(RetryableJobError) as raised:
            await handle_document_embed(job, lease, _context(pool))
        assert raised.value.error_code == "embedding_storage_transient"
        assert "private" not in str(raised.value)
        assert await pool.fetchval(
            "SELECT count(*) FROM chunk_embeddings WHERE document_id=$1",
            ids[2],
        ) == 0
    finally:
        await pool.execute("DROP TRIGGER fail_task8_vector_commit_trigger ON chunk_embeddings")
        await pool.execute("DROP FUNCTION fail_task8_vector_commit()")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        ((1.0, 0.0, 0.0),),
        ((1.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0, nan), (0.0, 1.0, 0.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ],
)
async def test_invalid_provider_output_is_terminal_and_atomic(pool, monkeypatch, response):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)
    monkeypatch.setattr("jobs.handlers._embed_texts", lambda _profile, _texts: asyncio.sleep(0, result=response))

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_embed(job, lease, _context(pool))

    assert raised.value.error_code == "invalid_embedding_response"
    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_missing_document_is_terminal(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    await pool.execute("UPDATE documents SET archived=true WHERE id=$1", ids[2])
    _install_profile(monkeypatch)

    with pytest.raises(TerminalJobError) as raised:
        await handle_document_embed(job, lease, _context(pool))
    assert raised.value.error_code == "document_not_found"


@pytest.mark.asyncio
async def test_endpoint_failure_is_retryable_and_sanitized(pool, monkeypatch):
    ids = await _seed_document(pool)
    job, lease = await _claimed_job(pool, ids)
    _install_profile(monkeypatch)

    async def unavailable(_profile, _texts):
        raise EmbeddingUnavailable("https://private:token@endpoint.invalid")

    monkeypatch.setattr("jobs.handlers._embed_texts", unavailable)

    with pytest.raises(RetryableJobError) as raised:
        await handle_document_embed(job, lease, _context(pool))
    assert raised.value.error_code == "embedding_unavailable"
    assert "private" not in str(raised.value)


@pytest.mark.asyncio
async def test_reconciliation_enqueues_only_ready_current_missing_complete_sets(pool):
    await pool.execute("UPDATE documents SET archived=true")
    missing = await _seed_document(pool)
    complete = await _seed_document(pool)
    await pool.executemany(
        "INSERT INTO chunk_embeddings "
        "(user_id,knowledge_base_id,document_id,document_version,chunk_index,provider,model,dimensions,embedding) "
        "VALUES ($1,$2,$3,1,$4,$5,$6,$7,$8::vector)",
        [
            (complete[0], complete[1], complete[2], index, PROFILE.provider, PROFILE.model, 3, "[1,0,0]")
            for index in (0, 1)
        ],
    )

    first = await reconcile_missing_embeddings(pool, PROFILE, page_size=1)
    second = await reconcile_missing_embeddings(pool, PROFILE, page_size=1)

    assert first == {"scanned": 1, "enqueued": 1}
    assert second == {"scanned": 1, "enqueued": 0}
    rows = await pool.fetch(
        "SELECT user_id,document_id,idempotency_key FROM background_jobs "
        "WHERE job_type='document.embed' AND document_id=ANY($1::uuid[])",
        [missing[2], complete[2]],
    )
    assert [(row["user_id"], row["document_id"]) for row in rows] == [(missing[0], missing[2])]


@pytest.mark.asyncio
async def test_concurrent_duplicate_embedding_enqueues_have_one_durable_winner(pool):
    ids = await _seed_document(pool)
    service = JobService(pool)

    jobs = await asyncio.gather(
        *[
            service.ensure_document_embedding(
                document_id=ids[2],
                document_version=1,
                user_id=ids[0],
                knowledge_base_id=ids[1],
                profile=PROFILE,
            )
            for _ in range(8)
        ]
    )

    assert len({job.id for job in jobs}) == 1
    assert await pool.fetchval(
        "SELECT count(*) FROM background_jobs WHERE document_id=$1 AND job_type='document.embed'",
        ids[2],
    ) == 1
