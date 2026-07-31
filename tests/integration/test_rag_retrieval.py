from uuid import UUID, uuid4

import pytest

from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.rag import RagCitation, RagWorkItem
from llmwiki_core.search import SearchHit, SearchQuery

PROFILE = EmbeddingProfile("openai_compatible", "rag-test", 3)


async def _scope(pool):
    user_id = uuid4()
    knowledge_base_id = uuid4()
    await pool.execute(
        "INSERT INTO users (id, email) VALUES ($1, $2)",
        user_id,
        f"{user_id}@rag-retrieval.test",
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1,$2,$3,$4)",
        knowledge_base_id,
        user_id,
        f"KB {knowledge_base_id}",
        f"kb-{knowledge_base_id}",
    )
    return user_id, knowledge_base_id


async def _document(
    pool,
    user_id,
    knowledge_base_id,
    *,
    filename="source.pdf",
    path="/corpus/",
    source_kind="source",
    version=1,
    content=None,
    chunks=("source",),
    archived=False,
    status="ready",
):
    document_id = uuid4()
    await pool.execute(
        "INSERT INTO documents "
        "(id, user_id, knowledge_base_id, filename, path, source_kind, file_type, "
        "status, archived, version, content, title, tags, date, metadata) "
        "VALUES ($1,$2,$3,$4,$5,$6,'md',$7,$8,$9,$10,'Title',ARRAY['policy'],"
        "'2026-07-27',$11::jsonb)",
        document_id,
        user_id,
        knowledge_base_id,
        filename,
        path,
        source_kind,
        status,
        archived,
        version,
        content,
        '{"country":"IDN"}',
    )
    await pool.executemany(
        "INSERT INTO document_chunks "
        "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
        "content, source_content, page, token_count) VALUES ($1,$2,$3,$4,$5,$6,$6,$7,1)",
        [
            (
                document_id,
                version,
                user_id,
                knowledge_base_id,
                index,
                chunk,
                index + 1,
            )
            for index, chunk in enumerate(chunks)
        ],
    )
    return document_id


def _hit(document_id, *, version=1, chunk=0, score=0.5):
    return SearchHit(
        document_id=str(document_id),
        document_version=version,
        chunk_index=chunk,
        content="search transport content",
        score=score,
        path="/transport.md",
    )


@pytest.mark.asyncio
async def test_evidence_reader_enforces_tenant_kb_version_fences_order_and_exact_cap(pool):
    from rag.retrieval import PostgresEvidenceReader

    user_id, kb_id = await _scope(pool)
    first = await _document(pool, user_id, kb_id, filename="first.pdf", chunks=("12345",))
    second = await _document(pool, user_id, kb_id, filename="second.pdf", chunks=("67890",))
    third = await _document(pool, user_id, kb_id, filename="third.pdf", chunks=("x",))
    changed = await _document(pool, user_id, kb_id, filename="changed.pdf", chunks=("old",))
    await pool.execute("UPDATE documents SET version=2, content='new' WHERE id=$1", changed)
    other_user, other_kb = await _scope(pool)
    foreign = await _document(pool, other_user, other_kb, filename="private.pdf", chunks=("private",))
    hits = (
        _hit(first, score=0.9),
        _hit(changed, score=0.8),
        _hit(foreign, score=0.7),
        _hit(second, score=0.6),
        _hit(third, score=0.5),
    )

    evidence = await PostgresEvidenceReader(pool).read(
        user_id,
        kb_id,
        hits,
        max_chars=10,
    )

    assert [item.document_id for item in evidence] == [first, second]
    assert [item.content for item in evidence] == ["12345", "67890"]
    assert [item.score for item in evidence] == [0.9, 0.6]
    assert all(item.status.value == "ready" and item.archived is False for item in evidence)
    assert all(item.metadata == {"country": "IDN"} for item in evidence)


@pytest.mark.asyncio
async def test_wiki_reader_makes_missing_cross_tenant_and_stale_version_indistinguishable(pool):
    from rag.retrieval import PostgresWikiPageReader

    user_id, kb_id = await _scope(pool)
    document_id = await _document(
        pool,
        user_id,
        kb_id,
        filename="page.md",
        path="/wiki/launch/",
        source_kind="wiki",
        version=3,
        content="Current page",
        chunks=("Current page",),
    )
    reader = PostgresWikiPageReader(pool)

    page = await reader.get_by_path(user_id, kb_id, "/wiki/launch/page.md")
    assert page.document_id == document_id
    assert page.version == 3
    assert page.path == "/wiki/launch/page.md"
    assert page.metadata == {"country": "IDN"}
    other_user, other_kb = await _scope(pool)
    assert await reader.get_by_path(other_user, other_kb, "/wiki/launch/page.md") is None
    assert await reader.get_by_path(user_id, kb_id, "/wiki/launch/missing.md") is None

    await pool.execute("UPDATE documents SET version=4 WHERE id=$1", document_id)
    assert await reader.get_by_path(user_id, kb_id, "/wiki/launch/page.md") is None
    await pool.execute(
        "UPDATE document_chunks SET document_version=4 WHERE document_id=$1",
        document_id,
    )
    assert (await reader.get_by_path(user_id, kb_id, "/wiki/launch/page.md")).version == 4
    await pool.execute("UPDATE documents SET archived=true WHERE id=$1", document_id)
    assert await reader.get_by_path(user_id, kb_id, "/wiki/launch/page.md") is None


@pytest.mark.asyncio
async def test_postgres_vector_retriever_uses_one_query_embedding_and_keeps_pool_open(pool):
    from rag.retrieval import PostgresVectorRetriever
    from services.vector_store import PostgresVectorStore

    user_id, kb_id = await _scope(pool)
    document_id = await _document(pool, user_id, kb_id, chunks=("vector source",))
    await PostgresVectorStore(pool, profile=PROFILE).replace_document_embeddings(
        user_id=user_id,
        knowledge_base_id=kb_id,
        document_id=document_id,
        document_version=1,
        embeddings=((0, (1.0, 0.0, 0.0)),),
    )
    calls = []

    class Client:
        profile = PROFILE

        async def embed(self, texts):
            calls.append(("embed", tuple(texts)))
            return ((1.0, 0.0, 0.0),)

        async def aclose(self):
            calls.append(("close",))

    result = await PostgresVectorRetriever(
        pool,
        user_id=user_id,
        knowledge_base_id=kb_id,
        profile=PROFILE,
        embedding_client_factory=Client,
    ).retrieve(SearchQuery.build(text="vector query", limit=1))

    assert [hit.document_id for hit in result.hits] == [str(document_id)]
    assert calls == [("embed", ("vector query",)), ("close",)]
    assert await pool.fetchval("SELECT 1") == 1


@pytest.mark.asyncio
async def test_rag_evidence_is_consumed_by_task6_prompt_and_draft_validation(pool):
    from rag.prompts import RagDraft, build_writer_messages
    from rag.retrieval import PostgresEvidenceReader
    from rag.validation import validate_draft

    user_id, kb_id = await _scope(pool)
    source_id = await _document(pool, user_id, kb_id, chunks=("Selected source text",))
    evidence = await PostgresEvidenceReader(pool).read(
        user_id,
        kb_id,
        (_hit(source_id),),
        max_chars=100,
    )
    item = RagWorkItem.build(
        0,
        "/wiki/launch/risks.md",
        "Summarize risks",
        "launch risks",
    )
    messages = build_writer_messages(
        goal="Build launch guidance",
        item=item,
        current_page="",
        evidence=evidence,
    )
    assert str(source_id) in messages[-1]["content"]
    content = (
        "---\n"
        "title: Launch risks\n"
        "tags: [launch, risk]\n"
        "description: A sourced launch risk summary.\n"
        "date: 2026-07-27\n"
        "---\n"
        "# Launch risks\n\n"
        "```mermaid\ngraph TD\n  A --> B\n```\n\n"
        "[Overview](./overview.md)\n\n"
        "Risk is documented.[^1]\n\n[^1]: source.pdf, p.1\n"
    )
    draft = RagDraft(
        content=content,
        citations=(RagCitation(source_id, document_version=1, chunk_index=0, page=1),),
    )

    bundle, lint = validate_draft(
        draft,
        evidence,
        document_id=UUID("40000000-0000-0000-0000-000000000004"),
        expected_version=1,
        target_path="/wiki/launch/risks.md",
        max_page_chars=40_000,
    )

    assert bundle.content == content
    assert lint["citation_count"] == 1
