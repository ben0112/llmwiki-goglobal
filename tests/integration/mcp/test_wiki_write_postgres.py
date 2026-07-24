"""Postgres entry point for the shared atomic wiki-write contract."""

import pytest

from tests.integration.mcp.test_mcp_isolation import KB_A_ID
from tests.integration.mcp.test_wiki_write_invariants import assert_atomic_wiki_write

pytest_plugins = ("tests.integration.mcp.test_mcp_isolation",)


@pytest.fixture
async def pg_atomic_fs(fs_alice, seed_and_bind_pool):
    return fs_alice, KB_A_ID


async def test_postgres_atomic_wiki_write(pg_atomic_fs):
    instance, kb_id = pg_atomic_fs
    await assert_atomic_wiki_write(instance, kb_id)
