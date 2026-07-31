import hashlib
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

ROOT = Path(__file__).parents[2]
MIGRATION_013 = ROOT / "supabase/migrations/013_chunk_embeddings.sql"
MIGRATION_014 = ROOT / "supabase/migrations/014_document_embedding_jobs.sql"
TASK7_MIGRATION_013_SHA256 = "be5bcaf53e533147b539c8e91c3cc2b276bd030d0942e3091e5a1227d553ef6b"


def test_published_013_bytes_are_unchanged_and_014_is_additive():
    assert hashlib.sha256(MIGRATION_013.read_bytes()).hexdigest() == TASK7_MIGRATION_013_SHA256
    sql = MIGRATION_014.read_text(encoding="utf-8").lower()
    assert "document.embed" in sql
    assert "chunk_embeddings_reconciliation_idx" in sql
    assert "drop table" not in sql
    assert "delete from" not in sql


def test_fresh_helper_embeds_013_then_complete_014_in_order():
    helper = (ROOT / "tests/helpers/schema.sql").read_text(encoding="utf-8")
    migration_013 = MIGRATION_013.read_text(encoding="utf-8").strip()
    migration_014 = MIGRATION_014.read_text(encoding="utf-8").strip()

    assert migration_013 in helper
    assert migration_014 in helper
    assert helper.index(migration_013) < helper.index(migration_014)


@pytest.mark.asyncio
async def test_upgrade_through_013_rejects_embed_then_014_enables_it_idempotently(pool):
    user_id, kb_id, document_id = uuid4(), uuid4(), uuid4()
    await pool.execute(
        "INSERT INTO users(id,email) VALUES($1,$2)", user_id, f"{user_id}@migration.test"
    )
    await pool.execute(
        "INSERT INTO knowledge_bases(id,user_id,name,slug) VALUES($1,$2,'Migration','migration-$2')",
        kb_id,
        user_id,
    )
    await pool.execute(
        "INSERT INTO documents(id,knowledge_base_id,user_id,filename,file_type,status,version) "
        "VALUES($1,$2,$3,'doc.md','md','ready',1)",
        document_id,
        kb_id,
        user_id,
    )
    migration_014 = MIGRATION_014.read_text(encoding="utf-8")

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("DROP INDEX IF EXISTS chunk_embeddings_reconciliation_idx")
        await conn.execute(
            "ALTER TABLE background_jobs DROP CONSTRAINT background_jobs_job_type_check;"
            "ALTER TABLE background_jobs ADD CONSTRAINT background_jobs_job_type_check "
            "CHECK(job_type IN ('document.extract','graph.rebuild','upload.cleanup')) NOT VALID"
        )
        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO background_jobs(job_type,user_id,knowledge_base_id,document_id,payload) "
                    "VALUES('document.embed',$1,$2,$3,'{}'::jsonb)",
                    user_id,
                    kb_id,
                    document_id,
                )

        await conn.execute(migration_014)
        await conn.execute(
            "INSERT INTO background_jobs(job_type,user_id,knowledge_base_id,document_id,payload) "
            "VALUES('document.embed',$1,$2,$3,'{}'::jsonb)",
            user_id,
            kb_id,
            document_id,
        )
        assert await conn.fetchval(
            "SELECT count(*) FROM pg_indexes WHERE schemaname='public' "
            "AND indexname='chunk_embeddings_reconciliation_idx'"
        ) == 1
        constraint = await conn.fetchval(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid='background_jobs'::regclass AND conname='background_jobs_job_type_check'"
        )
        assert "document.embed" in constraint
        await conn.execute(migration_014)
