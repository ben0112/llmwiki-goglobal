from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "supabase/migrations/013_chunk_embeddings.sql"


def _normalized_sql(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").lower().split())


def test_migration_requires_pgvector_and_explicit_dimension_check():
    sql = _normalized_sql(MIGRATION)

    assert "create extension if not exists vector" in sql
    assert "embedding vector not null" in sql
    assert "dimensions integer not null" in sql
    assert "vector_dims(embedding) = dimensions" in sql
    assert "dimensions between 1 and 4096" in sql


def test_migration_has_versioned_tenant_owned_identity_and_cascades():
    sql = _normalized_sql(MIGRATION)

    for column in (
        "user_id uuid not null",
        "knowledge_base_id uuid not null",
        "document_id uuid not null",
        "document_version integer not null",
        "chunk_index integer not null",
        "provider text not null",
        "model text not null",
    ):
        assert column in sql
    assert "unique (document_id, document_version, chunk_index, provider, model, dimensions)" in sql
    assert "references users" in sql
    assert "references knowledge_bases" in sql
    assert "references documents" in sql
    assert "references document_chunks" in sql
    assert sql.count("on delete cascade") >= 4


def test_migration_enables_tenant_rls_without_authenticated_mutation_grants():
    sql = _normalized_sql(MIGRATION)

    assert "alter table chunk_embeddings enable row level security" in sql
    assert "create policy chunk_embeddings_select" in sql
    assert "for select to authenticated" in sql
    assert "user_id = auth.uid()" in sql
    assert "grant select on chunk_embeddings to authenticated" in sql
    assert "grant insert" not in sql
    assert "grant update" not in sql
    assert "grant delete" not in sql


def test_migration_uses_only_query_shaped_btree_indexes_and_no_ann_index():
    sql = _normalized_sql(MIGRATION)

    assert "using hnsw" not in sql
    assert "using ivfflat" not in sql
    assert "<=>" not in sql
    assert "user_id, knowledge_base_id, provider, model, dimensions" in sql
    assert "document_id, document_version" in sql


def test_test_schema_contains_the_exact_additive_migration():
    migration = MIGRATION.read_text(encoding="utf-8").strip()
    test_schema = (ROOT / "tests/helpers/schema.sql").read_text(encoding="utf-8")

    assert migration in test_schema


@pytest.mark.parametrize(
    "path",
    [ROOT / ".github/workflows/test.yml", ROOT / "deploy/docker-compose.selfhost.yml"],
)
def test_postgres_ci_images_are_fixed_pgvector(path: Path):
    text = path.read_text(encoding="utf-8")

    assert "pgvector/pgvector:0.8.0-pg16" in text
    assert "postgres:16.11-alpine" not in text


@pytest.mark.asyncio
async def test_real_schema_has_pgvector_checks_foreign_keys_and_no_ann_index(pool):
    assert await pool.fetchval("SELECT extversion FROM pg_extension WHERE extname='vector'") == "0.8.0"
    constraints = await pool.fetch(
        "SELECT contype, pg_get_constraintdef(oid) AS definition "
        "FROM pg_constraint WHERE conrelid='chunk_embeddings'::regclass"
    )
    definitions = [row["definition"] for row in constraints]
    assert any("vector_dims(embedding) = dimensions" in value for value in definitions)
    foreign_keys = [row for row in constraints if row["contype"] == b"f"]
    assert len(foreign_keys) == 4
    assert all("ON DELETE CASCADE" in row["definition"] for row in foreign_keys)

    indexes = await pool.fetch(
        "SELECT indexdef FROM pg_indexes WHERE schemaname='public' "
        "AND tablename='chunk_embeddings'"
    )
    assert all("hnsw" not in row["indexdef"].lower() for row in indexes)
    assert all("ivfflat" not in row["indexdef"].lower() for row in indexes)


@pytest.mark.asyncio
async def test_real_schema_rls_select_is_tenant_scoped_and_mutations_are_not_granted(pool):
    user_a, user_b = uuid4(), uuid4()
    kb_a, kb_b = uuid4(), uuid4()
    doc_a, doc_b = uuid4(), uuid4()
    for user_id, kb_id, document_id in (
        (user_a, kb_a, doc_a),
        (user_b, kb_b, doc_b),
    ):
        await pool.execute(
            "INSERT INTO users (id,email) VALUES ($1,$2)", user_id, f"{user_id}@rls.test"
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
            "(id,knowledge_base_id,user_id,filename,file_type,status,version) "
            "VALUES ($1,$2,$3,'doc.md','md','ready',1)",
            document_id,
            kb_id,
            user_id,
        )
        await pool.execute(
            "INSERT INTO document_chunks "
            "(document_id,document_version,user_id,knowledge_base_id,chunk_index,"
            "content,source_content,token_count) VALUES ($1,1,$2,$3,0,'x','x',1)",
            document_id,
            user_id,
            kb_id,
        )
        await pool.execute(
            "INSERT INTO chunk_embeddings "
            "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
            "provider,model,dimensions,embedding) "
            "VALUES ($1,$2,$3,1,0,'openai_compatible','rls',3,'[1,0,0]')",
            user_id,
            kb_id,
            document_id,
        )

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SET LOCAL ROLE authenticated")
        await conn.execute(
            "SELECT set_config('request.jwt.claims',$1,true)", f'{{"sub":"{user_a}"}}'
        )
        visible = await conn.fetch("SELECT document_id FROM chunk_embeddings")
        assert [row["document_id"] for row in visible] == [doc_a]
        with pytest.raises(asyncpg.InsufficientPrivilegeError, match="permission denied"):
            async with conn.transaction():
                await conn.execute("DELETE FROM chunk_embeddings WHERE document_id=$1", doc_a)
