import asyncio
import json
from dataclasses import replace
from math import inf, nan
from uuid import uuid4

import pytest

from llmwiki_core.documents import DocumentKind
from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.search import (
    RetrieverUnavailable,
    SearchArea,
    SearchQuery,
    SearchResult,
    SearchScope,
)

PROFILE = EmbeddingProfile("openai_compatible", "test-small", 3)


def _store(pool, profile=PROFILE):
    from services.vector_store import PostgresVectorStore

    return PostgresVectorStore(pool, profile=profile)


def test_store_rejects_profile_identity_too_large_for_schema():
    profile = EmbeddingProfile("openai_compatible", "x" * 201, 3)

    with pytest.raises(ValueError, match="model identity"):
        _store(object(), profile)


async def _seed_document(
    pool,
    *,
    user_id=None,
    knowledge_base_id=None,
    version=1,
    filename="doc.md",
    path="/corpus/",
    source_kind="source",
    tags=("alpha",),
    metadata=None,
    chunks=("first", "second"),
    annotated=(),
):
    user_id = user_id or uuid4()
    knowledge_base_id = knowledge_base_id or uuid4()
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
        user_id,
        f"{user_id}@vectors.test",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO NOTHING",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, source_kind, file_type, "
        "status, version, tags, metadata) "
        "VALUES ($1, $2, $3, $4, $5, $6, 'md', 'ready', $7, $8, $9::jsonb)",
        document_id,
        knowledge_base_id,
        user_id,
        filename,
        path,
        source_kind,
        version,
        list(tags),
        json.dumps(metadata) if metadata is not None else None,
    )
    await pool.executemany(
        "INSERT INTO document_chunks "
        "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
        "content, source_content, annotations_text, has_highlight, page, token_count, "
        "header_breadcrumb) VALUES ($1,$2,$3,$4,$5,$6,$6,$7,$8,$9,1,$10)",
        [
            (
                document_id,
                version,
                user_id,
                knowledge_base_id,
                index,
                content,
                "note" if index in annotated else None,
                index in annotated,
                index,
                f"H{index}",
            )
            for index, content in enumerate(chunks)
        ],
    )
    return user_id, knowledge_base_id, document_id


async def _replace(store, ids, vectors, *, version=1):
    user_id, kb_id, document_id = ids
    return await store.replace_document_embeddings(
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=document_id,
        document_version=version,
        embeddings=vectors,
    )


def _query(**changes):
    return replace(
        SearchQuery.build(text="semantic", limit=10, candidate_limit=10),
        **changes,
    )


@pytest.mark.asyncio
async def test_search_orders_cosine_candidates_and_returns_shared_result(pool):
    ids = await _seed_document(pool, chunks=("near", "far"))
    store = _store(pool)
    await _replace(store, ids, [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))])

    result = await store.search(user_id=ids[0], knowledge_base_id=ids[1], query=_query(), embedding=(0.9, 0.1, 0.0))

    assert isinstance(result, SearchResult)
    assert [hit.content for hit in result.hits] == ["near", "far"]
    assert result.candidate_count == 2
    assert result.profile == "vector"
    assert result.hits[0].score > result.hits[1].score


@pytest.mark.asyncio
async def test_search_isolates_tenant_kb_current_version_and_profile(pool):
    user_a = uuid4()
    kb_a = uuid4()
    owned = await _seed_document(pool, user_id=user_a, knowledge_base_id=kb_a, filename="owned.md")
    other_kb = await _seed_document(pool, user_id=user_a, filename="other-kb.md")
    other_user = await _seed_document(pool, filename="other-user.md")
    store = _store(pool)
    for ids in (owned, other_kb, other_user):
        await _replace(store, ids, [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))])

    alternate = _store(pool, EmbeddingProfile("openai_compatible", "other-model", 3))
    await _replace(alternate, owned, [(0, (0.0, 0.0, 1.0)), (1, (0.0, 0.0, 1.0))])
    alternate_dimensions = _store(pool, EmbeddingProfile("openai_compatible", "test-small", 2))
    await _replace(alternate_dimensions, owned, [(0, (1.0, 0.0)), (1, (0.0, 1.0))])
    current = await store.search(user_id=user_a, knowledge_base_id=kb_a, query=_query(), embedding=(1.0, 0.0, 0.0))
    assert {hit.document_id for hit in current.hits} == {str(owned[2])}
    assert len(current.hits) == 2

    # Existing MCP writes increment the document version before replacing chunks;
    # the ownership FK must not make that valid transaction impossible.
    await pool.execute("UPDATE documents SET version = 2 WHERE id = $1", owned[2])
    await pool.execute("DELETE FROM document_chunks WHERE document_id = $1", owned[2])
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
        "content, source_content, token_count) VALUES ($1,2,$2,$3,0,'new','new',1)",
        owned[2],
        user_a,
        kb_a,
    )

    result = await store.search(user_id=user_a, knowledge_base_id=kb_a, query=_query(), embedding=(1.0, 0.0, 0.0))

    assert result.hits == ()
    assert result.candidate_count == 0


@pytest.mark.asyncio
async def test_all_search_filters_apply_before_limit(pool):
    user_id = uuid4()
    kb_id = uuid4()
    excluded = await _seed_document(
        pool,
        user_id=user_id,
        knowledge_base_id=kb_id,
        filename="excluded.md",
        path="/other/",
        tags=("wrong",),
        metadata={"stage": "wrong"},
        chunks=("excluded",),
    )
    included = await _seed_document(
        pool,
        user_id=user_id,
        knowledge_base_id=kb_id,
        filename="included.md",
        path="/corpus/sgp/",
        source_kind="wiki",
        tags=("alpha", "beta"),
        metadata={"stage": "S2"},
        chunks=("included",),
        annotated=(0,),
    )
    store = _store(pool)
    await _replace(store, excluded, [(0, (1.0, 0.0, 0.0))])
    await _replace(store, included, [(0, (0.9, 0.1, 0.0))])
    query = SearchQuery.build(
        text="semantic",
        limit=1,
        candidate_limit=1,
        area=SearchArea.WIKI,
        scope=SearchScope.ANNOTATIONS,
        facets={"stage": "S2"},
        path_glob="/corpus/sgp/*.md",
        tags=("ALPHA", "beta"),
        document_kinds=(DocumentKind.WIKI,),
        annotated_only=True,
    )

    result = await store.search(user_id=user_id, knowledge_base_id=kb_id, query=query, embedding=(1.0, 0.0, 0.0))

    assert [hit.content for hit in result.hits] == ["included"]
    assert result.candidate_count == 1


@pytest.mark.asyncio
async def test_candidate_count_is_before_bounded_limit(pool):
    ids = await _seed_document(pool, chunks=("a", "b", "c"))
    store = _store(pool)
    await _replace(
        store,
        ids,
        [(0, (1.0, 0.0, 0.0)), (1, (0.9, 0.1, 0.0)), (2, (0.8, 0.2, 0.0))],
    )

    result = await store.search(
        user_id=ids[0],
        knowledge_base_id=ids[1],
        query=SearchQuery(text="semantic", limit=1, candidate_limit=1),
        embedding=(1.0, 0.0, 0.0),
    )

    assert len(result.hits) == 1
    assert result.candidate_count == 3


@pytest.mark.asyncio
async def test_same_version_replacement_is_idempotent(pool):
    ids = await _seed_document(pool)
    store = _store(pool)

    await _replace(store, ids, [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))])
    await _replace(store, ids, [(0, (0.0, 0.0, 1.0)), (1, (0.0, 0.0, 1.0))])

    rows = await pool.fetch(
        "SELECT chunk_index, embedding::text FROM chunk_embeddings WHERE document_id=$1 ORDER BY chunk_index",
        ids[2],
    )
    assert [(row["chunk_index"], row["embedding"]) for row in rows] == [
        (0, "[0,0,1]"),
        (1, "[0,0,1]"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embeddings, message",
    [
        ([(0, (1.0, 0.0, 0.0))], "complete current chunk set"),
        ([(0, (1.0, 0.0, 0.0)), (0, (0.0, 1.0, 0.0))], "duplicate chunk"),
        ([(0, (1.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0))], "complete current chunk set"),
    ],
)
async def test_replacement_rejects_missing_duplicate_and_extra_chunk_identities(pool, embeddings, message):
    ids = await _seed_document(pool)

    with pytest.raises(ValueError, match=message):
        await _replace(_store(pool), ids, embeddings)

    assert await pool.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", ids[2]) == 0


@pytest.mark.asyncio
async def test_replacement_rejects_stale_document_version(pool):
    ids = await _seed_document(pool, version=2)

    with pytest.raises(ValueError, match="current document version"):
        await _replace(
            _store(pool),
            ids,
            [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))],
            version=1,
        )


@pytest.mark.asyncio
async def test_partial_failure_rolls_back_and_preserves_old_complete_set(pool):
    ids = await _seed_document(pool)
    store = _store(pool)
    old = [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))]
    await _replace(store, ids, old)

    await pool.execute(
        "CREATE OR REPLACE FUNCTION fail_vector_write() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN IF NEW.chunk_index = 1 THEN RAISE EXCEPTION 'private dsn=postgres://secret'; "
        "END IF; RETURN NEW; END $$"
    )
    await pool.execute(
        "CREATE TRIGGER fail_vector_write BEFORE UPDATE ON chunk_embeddings "
        "FOR EACH ROW EXECUTE FUNCTION fail_vector_write()"
    )
    try:
        with pytest.raises(RetrieverUnavailable, match="vector store is unavailable") as raised:
            await _replace(store, ids, [(0, (0.0, 0.0, 1.0)), (1, (0.0, 0.0, 1.0))])
        assert "secret" not in str(raised.value)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
    finally:
        await pool.execute("DROP TRIGGER fail_vector_write ON chunk_embeddings")
        await pool.execute("DROP FUNCTION fail_vector_write()")

    rows = await pool.fetch(
        "SELECT chunk_index, embedding::text FROM chunk_embeddings WHERE document_id=$1 ORDER BY chunk_index",
        ids[2],
    )
    assert [row["embedding"] for row in rows] == ["[1,0,0]", "[0,1,0]"]


@pytest.mark.asyncio
async def test_cancellation_rolls_back_and_preserves_old_complete_set(pool):
    ids = await _seed_document(pool)
    store = _store(pool)
    await _replace(store, ids, [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))])
    await pool.execute(
        "CREATE OR REPLACE FUNCTION delay_vector_write() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN IF NEW.chunk_index = 1 THEN PERFORM pg_sleep(10); END IF; RETURN NEW; END $$"
    )
    await pool.execute(
        "CREATE TRIGGER delay_vector_write BEFORE UPDATE ON chunk_embeddings "
        "FOR EACH ROW EXECUTE FUNCTION delay_vector_write()"
    )
    try:
        task = asyncio.create_task(_replace(store, ids, [(0, (0.0, 0.0, 1.0)), (1, (0.0, 0.0, 1.0))]))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await pool.execute("DROP TRIGGER delay_vector_write ON chunk_embeddings")
        await pool.execute("DROP FUNCTION delay_vector_write()")

    rows = await pool.fetch(
        "SELECT embedding::text FROM chunk_embeddings WHERE document_id=$1 ORDER BY chunk_index",
        ids[2],
    )
    assert [row["embedding"] for row in rows] == ["[1,0,0]", "[0,1,0]"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedding",
    [
        (True, 0.0, 0.0),
        (nan, 0.0, 0.0),
        (inf, 0.0, 0.0),
        (1.0, 0.0),
    ],
)
async def test_invalid_vectors_are_rejected_before_database_access(pool, embedding):
    store = _store(pool)

    with pytest.raises(ValueError, match="embedding"):
        await store.search(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query=_query(),
            embedding=embedding,
        )


@pytest.mark.asyncio
async def test_wrong_types_and_oversize_limits_are_rejected(pool):
    store = _store(pool)

    with pytest.raises(ValueError):
        await store.search(
            user_id=True,
            knowledge_base_id=uuid4(),
            query=_query(),
            embedding=(1.0, 0.0, 0.0),
        )
    invalid_query = _query()
    object.__setattr__(invalid_query, "candidate_limit", 501)
    with pytest.raises(ValueError, match="candidate limit"):
        await store.search(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query=invalid_query,
            embedding=(1.0, 0.0, 0.0),
        )


@pytest.mark.asyncio
async def test_pgvector_backend_failures_are_sanitized(pool):
    class FailingPool:
        async def fetch(self, *args, **kwargs):
            raise RuntimeError("SELECT secret FROM private dsn=postgres://token")

    store = _store(FailingPool())
    with pytest.raises(RetrieverUnavailable, match="vector store is unavailable") as raised:
        await store.search(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query=_query(),
            embedding=(1.0, 0.0, 0.0),
        )
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
