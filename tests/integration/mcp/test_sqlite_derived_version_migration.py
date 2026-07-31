from pathlib import Path

import aiosqlite


async def test_mcp_sqlite_init_backfills_derived_versions(tmp_path):
    from vaultfs.sqlite import SqliteVaultFS

    workspace = tmp_path / "workspace"
    database_dir = workspace / ".llmwiki"
    database_dir.mkdir(parents=True)
    database_path = database_dir / "index.db"
    schema_path = Path(__file__).parents[3] / "shared" / "sqlite_schema.sql"
    current_schema = schema_path.read_text(encoding="utf-8")
    old_schema = current_schema.replace(
        "    document_version INTEGER NOT NULL DEFAULT 0,\n",
        "",
    )
    assert old_schema != current_schema

    db = await aiosqlite.connect(database_path)
    await db.executescript(old_schema)
    await db.execute(
        "INSERT INTO documents "
        "(id, user_id, filename, path, relative_path, source_kind, file_type, status, version) "
        "VALUES ('d1', 'u1', 'doc.md', '/', 'doc.md', 'source', 'md', 'ready', 5)"
    )
    await db.execute(
        "INSERT INTO document_chunks "
        "(id, document_id, chunk_index, content, source_content, token_count) "
        "VALUES ('c1', 'd1', 0, 'chunk', 'chunk', 1)"
    )
    await db.commit()
    await db.close()

    await SqliteVaultFS.init(str(workspace))
    try:
        migrated = SqliteVaultFS._db_or_raise()
        assert await migrated.execute_fetchall("SELECT document_version FROM document_chunks WHERE id='c1'") == [(5,)]
    finally:
        await SqliteVaultFS.close()
