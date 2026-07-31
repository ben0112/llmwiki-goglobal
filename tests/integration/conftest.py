import os
import uuid
from pathlib import Path

import asyncpg
import httpx
import pytest

from tests.helpers.jwt import seed_jwks_cache

DB_URL = os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
async def pool():
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=20)

    await pool.execute("DROP SCHEMA IF EXISTS public CASCADE")
    await pool.execute("CREATE SCHEMA public")

    schema_sql = (Path(__file__).parent.parent / "helpers" / "schema.sql").read_text()
    await pool.execute(schema_sql)

    yield pool
    pool.terminate()


@pytest.fixture
async def client(pool):
    from main import app
    from services.hosted import HostedServiceFactory

    app.state.pool = pool
    app.state.s3_service = None
    app.state.ocr_service = None
    app.state.auth_provider = None
    app.state.factory = HostedServiceFactory(pool)

    seed_jwks_cache()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def seed_pending_document(pool):
    """Create an isolated hosted document for derived-content invariant tests."""

    async def seed(*, status: str = "processing", version: int = 0) -> dict:
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        await pool.execute(
            "INSERT INTO users (id, email, display_name) VALUES ($1, $2, 'Invariant Test')",
            user_id,
            f"{user_id}@test.invalid",
        )
        await pool.execute(
            "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4)",
            kb_id,
            user_id,
            f"KB {kb_id}",
            f"kb-{kb_id}",
        )
        await pool.execute(
            "INSERT INTO documents "
            "(id, knowledge_base_id, user_id, filename, path, file_type, status, version, source_kind) "
            "VALUES ($1, $2, $3, 'source.md', '/', 'md', $4, $5, 'source')",
            doc_id,
            kb_id,
            user_id,
            status,
            version,
        )
        return {"id": doc_id, "user_id": user_id, "knowledge_base_id": kb_id}

    return seed
