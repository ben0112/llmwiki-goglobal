"""Postgres entry point for the shared atomic wiki-write contract."""

import asyncio
import uuid

import pytest
from vaultfs.base import DuplicateDocumentError as LegacyDuplicateDocumentError

import llmwiki_adapters.postgres.wiki as postgres_wiki
from llmwiki_adapters.postgres.wiki import WikiWriteResult
from llmwiki_core import DuplicateDocumentError as CoreDuplicateDocumentError
from llmwiki_core.references import ReferenceEdge
from llmwiki_core.wiki import VersionConflict, WikiWriteBundle
from tests.integration.mcp.test_mcp_isolation import KB_A_ID
from tests.integration.mcp.test_wiki_write_invariants import assert_atomic_wiki_write

pytest_plugins = ("tests.integration.mcp.test_mcp_isolation",)


class _CountingContext:
    def __init__(self, counts, prefix, value=None):
        self.counts = counts
        self.prefix = prefix
        self.value = value

    async def __aenter__(self):
        self.counts[f"{self.prefix}_enter"] += 1
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        self.counts[f"{self.prefix}_exit"] += 1
        self.counts[f"{self.prefix}_exc"] = exc


class _CountingConnection:
    def __init__(self, counts):
        self.counts = counts

    def transaction(self):
        self.counts["transaction"] += 1
        return _CountingContext(self.counts, "transaction")


class _CountingPool:
    def __init__(self, counts):
        self.counts = counts
        self.conn = _CountingConnection(counts)

    def acquire(self):
        self.counts["acquire"] += 1
        return _CountingContext(self.counts, "acquire", self.conn)


def _delegation_bundle(document_id):
    return WikiWriteBundle.build(
        document_id=document_id,
        expected_version=None,
        filename="delegated.md",
        path="/wiki/",
        file_type="md",
        content="delegated transaction " * 80,
        title="Delegated",
        tags=["delegated"],
        date="2026-07-26",
        metadata={"description": "Delegation contract"},
        edges=[ReferenceEdge(document_id, "links_to")],
    )


@pytest.fixture
async def pg_atomic_fs(fs_alice, seed_and_bind_pool):
    return fs_alice, KB_A_ID


async def test_postgres_atomic_wiki_write(pg_atomic_fs):
    instance, kb_id = pg_atomic_fs
    await assert_atomic_wiki_write(instance, kb_id)


async def test_postgres_vault_delegates_to_canonical_shared_writer(
    pg_atomic_fs,
    monkeypatch,
):
    instance, kb_id = pg_atomic_fs
    document_id = str(uuid.uuid4())
    bundle = _delegation_bundle(document_id)
    calls = []
    original = postgres_wiki.write_wiki_bundle_in_transaction

    async def spy(conn, **kwargs):
        calls.append((conn.is_in_transaction(), kwargs))
        return await original(conn, **kwargs)

    monkeypatch.setattr(postgres_wiki, "write_wiki_bundle_in_transaction", spy)

    result = await instance.write_wiki_bundle(kb_id, bundle)

    assert result == {
        "id": document_id,
        "filename": "delegated.md",
        "path": "/wiki/",
        "version": 1,
    }
    assert len(calls) == 1
    in_transaction, kwargs = calls[0]
    assert in_transaction is True
    assert kwargs == {
        "user_id": uuid.UUID(instance.user_id),
        "knowledge_base_id": uuid.UUID(kb_id),
        "bundle": bundle,
    }


def test_duplicate_document_error_is_the_shared_class_object():
    assert LegacyDuplicateDocumentError is CoreDuplicateDocumentError


async def test_postgres_vault_owns_exactly_one_connection_and_transaction(
    fs_alice,
    monkeypatch,
):
    counts = {
        "acquire": 0,
        "acquire_enter": 0,
        "acquire_exit": 0,
        "transaction": 0,
        "transaction_enter": 0,
        "transaction_exit": 0,
    }
    pool = _CountingPool(counts)
    document_id = uuid.uuid4()
    calls = []

    async def fake_get_pool():
        return pool

    async def fake_writer(conn, **kwargs):
        calls.append((conn, kwargs))
        return WikiWriteResult(
            document_id=document_id,
            filename="delegated.md",
            path="/wiki/",
            version=7,
        )

    monkeypatch.setattr("vaultfs.postgres.get_pool", fake_get_pool)
    monkeypatch.setattr(postgres_wiki, "write_wiki_bundle_in_transaction", fake_writer)

    result = await fs_alice.write_wiki_bundle(
        KB_A_ID,
        _delegation_bundle(str(document_id)),
    )

    assert result == {
        "id": str(document_id),
        "filename": "delegated.md",
        "path": "/wiki/",
        "version": 7,
    }
    assert len(calls) == 1
    assert calls[0][0] is pool.conn
    assert counts == {
        "acquire": 1,
        "acquire_enter": 1,
        "acquire_exit": 1,
        "transaction": 1,
        "transaction_enter": 1,
        "transaction_exit": 1,
        "transaction_exc": None,
        "acquire_exc": None,
    }


@pytest.mark.parametrize(
    "failure",
    [
        LegacyDuplicateDocumentError("/wiki/", "delegated.md"),
        VersionConflict("stale version"),
        PermissionError("invalid reference"),
        asyncio.CancelledError(),
    ],
    ids=["duplicate", "version", "reference", "cancellation"],
)
async def test_postgres_vault_propagates_canonical_writer_failures_exactly(
    fs_alice,
    monkeypatch,
    failure,
):
    counts = {
        "acquire": 0,
        "acquire_enter": 0,
        "acquire_exit": 0,
        "transaction": 0,
        "transaction_enter": 0,
        "transaction_exit": 0,
    }
    pool = _CountingPool(counts)

    async def fake_get_pool():
        return pool

    async def fake_writer(conn, **kwargs):
        del conn, kwargs
        raise failure

    monkeypatch.setattr("vaultfs.postgres.get_pool", fake_get_pool)
    monkeypatch.setattr(postgres_wiki, "write_wiki_bundle_in_transaction", fake_writer)

    with pytest.raises(type(failure)) as raised:
        await fs_alice.write_wiki_bundle(
            KB_A_ID,
            _delegation_bundle(str(uuid.uuid4())),
        )

    assert raised.value is failure
    assert counts["transaction_exc"] is failure
    assert counts["acquire_exc"] is failure
