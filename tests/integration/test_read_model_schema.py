from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "supabase" / "migrations" / "016_read_models.sql"


def test_read_model_migration_is_additive_scoped_and_embedded_in_test_schema():
    sql = MIGRATION.read_text(encoding="utf-8")
    normalized = " ".join(sql.lower().split())
    helper = (ROOT / "tests" / "helpers" / "schema.sql").read_text(encoding="utf-8")

    assert "add column if not exists read_revision" in normalized
    assert "security definer" in normalized
    assert "set search_path = public, pg_temp" in normalized
    assert "revoke all on function bump_knowledge_base_read_revision() from public" in normalized
    assert "drop table" not in normalized
    assert sql.strip() in helper


@pytest.mark.asyncio
async def test_document_mutations_bump_only_affected_knowledge_bases(pool):
    migration = MIGRATION.read_text(encoding="utf-8")
    user_id, kb_a, kb_b, document_id = uuid4(), uuid4(), uuid4(), uuid4()
    await pool.execute(
        "INSERT INTO users(id,email) VALUES($1,$2)", user_id, f"{user_id}@read.test"
    )
    await pool.execute(
        "INSERT INTO knowledge_bases(id,user_id,name,slug) VALUES"
        "($1,$3,'Read A',$4),($2,$3,'Read B',$5)",
        kb_a,
        kb_b,
        user_id,
        f"read-a-{kb_a}",
        f"read-b-{kb_b}",
    )

    async with pool.acquire() as connection:
        await connection.execute(migration)
        await connection.execute(migration)
        initial_a, initial_b = await connection.fetchrow(
            "SELECT "
            "(SELECT read_revision FROM knowledge_bases WHERE id=$1),"
            "(SELECT read_revision FROM knowledge_bases WHERE id=$2)",
            kb_a,
            kb_b,
        )

        await connection.execute(
            "INSERT INTO documents"
            "(id,knowledge_base_id,user_id,filename,file_type,status) "
            "VALUES($1,$2,$3,'a.md','md','ready')",
            document_id,
            kb_a,
            user_id,
        )
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_a
        ) == initial_a + 1
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_b
        ) == initial_b

        await connection.execute(
            "UPDATE documents SET title='A' WHERE id=$1", document_id
        )
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_a
        ) == initial_a + 2

        await connection.execute(
            "UPDATE documents SET knowledge_base_id=$2 WHERE id=$1",
            document_id,
            kb_b,
        )
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_a
        ) == initial_a + 3
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_b
        ) == initial_b + 1

        await connection.execute("DELETE FROM documents WHERE id=$1", document_id)
        assert await connection.fetchval(
            "SELECT read_revision FROM knowledge_bases WHERE id=$1", kb_b
        ) == initial_b + 2


@pytest.mark.asyncio
async def test_read_indexes_are_tenant_first_and_partial(pool):
    rows = await pool.fetch(
        "SELECT indexname,indexdef FROM pg_indexes "
        "WHERE schemaname='public' AND indexname LIKE 'idx_documents_read_%'"
    )
    definitions = {row["indexname"]: row["indexdef"].lower() for row in rows}
    assert set(definitions) == {
        "idx_documents_read_browse_name",
        "idx_documents_read_browse_date",
        "idx_documents_read_wiki_path",
        "idx_documents_read_corpus",
        "idx_documents_read_number",
    }
    for definition in definitions.values():
        compact = " ".join(definition.split())
        assert "(knowledge_base_id, user_id," in compact
        assert "where (not archived)" in compact or "where not archived" in compact
