import subprocess
import sys
from pathlib import Path

import aiosqlite
import pytest
from infra.db.sqlite import create_pool

ROOT = Path(__file__).parents[2]
SCHEMA = (ROOT / "shared" / "sqlite_schema.sql").read_text(encoding="utf-8")


async def _revision(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("SELECT read_revision FROM workspace WHERE id='ws1'")
    row = await cursor.fetchone()
    assert row is not None
    return row[0]


async def _seed_old_database(database_path: Path) -> None:
    old_schema = SCHEMA.replace(
        "    read_revision INTEGER NOT NULL DEFAULT 1 CHECK (read_revision > 0),\n",
        "",
    )
    db = await aiosqlite.connect(database_path)
    await db.executescript(old_schema)
    await db.execute(
        "INSERT INTO workspace (id,name,user_id) VALUES ('ws1','test','u1')"
    )
    await db.commit()
    await db.close()


async def _assert_read_schema(db: aiosqlite.Connection) -> None:
    columns = {
        row[1] for row in await db.execute_fetchall("PRAGMA table_info(workspace)")
    }
    assert "read_revision" in columns
    triggers = {
        row[0]
        for row in await db.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    assert {
        "documents_read_revision_insert",
        "documents_read_revision_update",
        "documents_read_revision_delete",
    } <= triggers


@pytest.mark.asyncio
async def test_document_insert_update_delete_each_bumps_read_revision(tmp_path):
    db = await create_pool(str(tmp_path / "index.db"))
    try:
        await db.execute(
            "INSERT INTO workspace (id,name,user_id) VALUES ('ws1','test','u1')"
        )
        await db.commit()
        initial = await _revision(db)

        await db.execute(
            "INSERT INTO documents "
            "(id,user_id,filename,path,relative_path,source_kind,file_type) "
            "VALUES ('d1','u1','a.md','/','a.md','source','md')"
        )
        await db.commit()
        assert await _revision(db) == initial + 1

        await db.execute("UPDATE documents SET title='A' WHERE id='d1'")
        await db.commit()
        assert await _revision(db) == initial + 2

        await db.execute("DELETE FROM documents WHERE id='d1'")
        await db.commit()
        assert await _revision(db) == initial + 3
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_api_initializer_backfills_old_schema_idempotently(tmp_path):
    database_path = tmp_path / "index.db"
    await _seed_old_database(database_path)

    for _ in range(2):
        db = await create_pool(str(database_path))
        try:
            await _assert_read_schema(db)
            assert await _revision(db) == 1
        finally:
            await db.close()


@pytest.mark.asyncio
async def test_mcp_initializer_backfills_old_schema_idempotently(tmp_path):
    workspace = tmp_path / "workspace"
    database_dir = workspace / ".llmwiki"
    database_dir.mkdir(parents=True)
    await _seed_old_database(database_dir / "index.db")
    probe = """
import asyncio
import sys

sys.path.insert(0, sys.argv[1])
from vaultfs.sqlite import SqliteVaultFS

async def main():
    for _ in range(2):
        await SqliteVaultFS.init(sys.argv[2])
        try:
            db = SqliteVaultFS._db_or_raise()
            columns = {row[1] for row in await db.execute_fetchall('PRAGMA table_info(workspace)')}
            assert 'read_revision' in columns
            triggers = {row[0] for row in await db.execute_fetchall(
                \"SELECT name FROM sqlite_master WHERE type='trigger'\"
            )}
            assert {
                'documents_read_revision_insert',
                'documents_read_revision_update',
                'documents_read_revision_delete',
            } <= triggers
            assert await db.execute_fetchall(
                \"SELECT read_revision FROM workspace WHERE id='ws1'\"
            ) == [(1,)]
        finally:
            await SqliteVaultFS.close()

asyncio.run(main())
"""
    subprocess.run(
        [sys.executable, "-c", probe, str(ROOT / "mcp"), str(workspace)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
