"""Direct contract tests for the hosted Postgres wiki transaction adapter."""

import asyncio
import json
import uuid
from contextlib import suppress
from dataclasses import replace

import asyncpg
import pytest

import llmwiki_adapters.postgres.wiki as postgres_wiki
from llmwiki_adapters.postgres.wiki import write_wiki_bundle_in_transaction
from llmwiki_core.references import ReferenceEdge
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
        assert await conn.fetchval(
            "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        ) == 0


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
                    "SELECT version, metadata FROM documents "
                    "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                    document_id,
                    user_id,
                    knowledge_base_id,
                )
                assert persisted["version"] == 1
                metadata = persisted["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                assert metadata["facet_rollup"]["stage"] == ["S4"]
                assert await conn.fetchval(
                    "SELECT count(*) FROM document_chunks "
                    "WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
                    document_id,
                    user_id,
                    knowledge_base_id,
                ) > 0
                assert await conn.fetchval(
                    "SELECT count(*) FROM document_references "
                    "WHERE source_document_id=$1 AND knowledge_base_id=$2",
                    document_id,
                    knowledge_base_id,
                ) == 1
                raise RollbackSignal

        assert await conn.fetchval(
            "SELECT count(*) FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        ) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM document_chunks "
            "WHERE document_id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        ) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM document_references "
            "WHERE source_document_id=$1 AND knowledge_base_id=$2",
            document_id,
            knowledge_base_id,
        ) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM documents "
            "WHERE id=$1 AND metadata ? 'facet_rollup'",
            document_id,
        ) == 0


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
        assert await conn.fetchval(
            "SELECT count(*) FROM document_chunks WHERE document_id=$1",
            document_id,
        ) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM document_references WHERE source_document_id=$1",
            document_id,
        ) == 0
        persisted = await conn.fetchrow(
            "SELECT content, version, metadata FROM documents "
            "WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
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
        assert await conn.fetchval(
            "SELECT content FROM documents WHERE id=$1 AND user_id=$2 AND knowledge_base_id=$3",
            document_id,
            user_id,
            knowledge_base_id,
        ) == ""


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
