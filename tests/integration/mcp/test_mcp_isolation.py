"""MCP PostgresVaultFS multi-tenant isolation tests.

Verifies that PostgresVaultFS operations for User A cannot reach User B's data,
and vice versa, across reads, writes, graph queries, and S3 key derivation.

The test database installs a tiny deterministic text-search operator shim so
the real Postgres adapter query can exercise retrieval isolation without
requiring the PGroonga extension in this integration environment.
"""

import json
import os
import uuid

import asyncpg
import db as mcp_db
import pytest
from vaultfs.base import search_hit_to_legacy_dict
from vaultfs.postgres import PostgresVaultFS

from llmwiki_core.models import EmbeddingProfile
from llmwiki_core.search import RetrieverUnavailable, SearchHit, SearchQuery

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

KB_A_ID = "11111111-1111-1111-1111-111111111111"
KB_B_ID = "22222222-2222-2222-2222-222222222222"

DOC_A_ID = "aaaa1111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_B_ID = "bbbb1111-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

DOC_A2_ID = "aaaa4444-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC_B2_ID = "bbbb4444-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

ASSET_A_ID = "aaaa5555-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ASSET_B_ID = "bbbb5555-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

REF_A_ID = "aaaa3333-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
REF_B_ID = "bbbb3333-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture(scope="session")
async def pg_pool():
    """Shared Postgres pool for MCP isolation tests.

    Reuses the same test DB and schema as the API isolation tests.
    """
    from pathlib import Path
    db_url = os.environ["DATABASE_URL"]
    pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)

    await pool.execute("DROP SCHEMA IF EXISTS public CASCADE")
    await pool.execute("CREATE SCHEMA public")
    schema_sql = (Path(__file__).parent.parent.parent / "helpers" / "schema.sql").read_text()
    await pool.execute(schema_sql)
    await pool.execute(
        "CREATE FUNCTION test_text_search(text, text) RETURNS boolean "
        "LANGUAGE sql IMMUTABLE STRICT AS "
        "'SELECT strpos(lower($1), lower($2)) > 0'"
    )
    await pool.execute(
        "CREATE OPERATOR &@~ (LEFTARG = text, RIGHTARG = text, FUNCTION = test_text_search)"
    )
    await pool.execute(
        "CREATE FUNCTION pgroonga_score(oid, tid) RETURNS double precision "
        "LANGUAGE sql IMMUTABLE STRICT AS 'SELECT 0.0::double precision'"
    )

    yield pool
    pool.terminate()


@pytest.fixture(autouse=True)
async def seed_and_bind_pool(pg_pool):
    """Seed two tenants and point mcp.db's global pool at the test pool."""
    mcp_db._pool = pg_pool

    await pg_pool.execute("DELETE FROM document_references")
    await pg_pool.execute("DELETE FROM document_chunks")
    await pg_pool.execute("DELETE FROM document_pages")
    await pg_pool.execute("DELETE FROM documents")
    await pg_pool.execute("DELETE FROM api_keys")
    await pg_pool.execute("DELETE FROM knowledge_bases")
    await pg_pool.execute("DELETE FROM users")

    await pg_pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, 'alice@test.com', 'Alice')",
        USER_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO users (id, email, display_name) VALUES ($1, 'bob@test.com', 'Bob')",
        USER_B_ID,
    )
    await pg_pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, 'Alice KB', 'alice-kb')",
        KB_A_ID, USER_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO knowledge_bases (id, user_id, name, slug) VALUES ($1, $2, 'Bob KB', 'bob-kb')",
        KB_B_ID, USER_B_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version) "
        "VALUES ($1, $2, $3, 'notes.md', 'Notes', '/wiki/', 'md', 'ready', 'Alice secret', 1)",
        DOC_A_ID, KB_A_ID, USER_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version) "
        "VALUES ($1, $2, $3, 'notes.md', 'Notes', '/wiki/', 'md', 'ready', 'Bob secret', 1)",
        DOC_B_ID, KB_B_ID, USER_B_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version) "
        "VALUES ($1, $2, $3, 'source.pdf', 'Source', '/', 'pdf', 'ready', NULL, 1)",
        DOC_A2_ID, KB_A_ID, USER_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version) "
        "VALUES ($1, $2, $3, 'source.pdf', 'Source', '/', 'pdf', 'ready', NULL, 1)",
        DOC_B2_ID, KB_B_ID, USER_B_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version, metadata) "
        "VALUES ($1, $2, $3, 'image-01.png', 'image-01.png', '/webclipper/clip.assets/', "
        "'png', 'ready', NULL, 1, '{\"asset\": true}'::jsonb)",
        ASSET_A_ID, KB_A_ID, USER_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO documents (id, knowledge_base_id, user_id, filename, title, path, "
        "file_type, status, content, version, metadata) "
        "VALUES ($1, $2, $3, 'image-01.png', 'image-01.png', '/webclipper/clip.assets/', "
        "'png', 'ready', NULL, 1, '{\"asset\": true}'::jsonb)",
        ASSET_B_ID, KB_B_ID, USER_B_ID,
    )
    await pg_pool.execute(
        "INSERT INTO document_references (id, source_document_id, target_document_id, "
        "knowledge_base_id, reference_type) VALUES ($1, $2, $3, $4, 'cites')",
        REF_A_ID, DOC_A_ID, DOC_A2_ID, KB_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO document_references (id, source_document_id, target_document_id, "
        "knowledge_base_id, reference_type) VALUES ($1, $2, $3, $4, 'cites')",
        REF_B_ID, DOC_B_ID, DOC_B2_ID, KB_B_ID,
    )

    await pg_pool.execute(
        "INSERT INTO document_pages (id, document_id, page, content) VALUES ($1, $2, 1, 'Alice page 1')",
        str(uuid.uuid4()), DOC_A_ID,
    )
    await pg_pool.execute(
        "INSERT INTO document_pages (id, document_id, page, content) VALUES ($1, $2, 1, 'Bob page 1')",
        str(uuid.uuid4()), DOC_B_ID,
    )

    await pg_pool.execute(
        "UPDATE documents SET stale_since = now() WHERE id = $1", DOC_B_ID,
    )

    yield

    mcp_db._pool = None


@pytest.fixture
def fs_alice():
    return PostgresVaultFS(USER_A_ID)


@pytest.fixture
def fs_bob():
    return PostgresVaultFS(USER_B_ID)


class TestReadIsolation:
    """PostgresVaultFS reads are scoped by user_id (RLS + app-layer WHERE)."""

    async def test_list_kbs_returns_only_own(self, fs_alice):
        kbs = await fs_alice.list_knowledge_bases()
        slugs = [kb["slug"] for kb in kbs]
        assert "alice-kb" in slugs
        assert "bob-kb" not in slugs

    async def test_resolve_kb_other_tenant_returns_none(self, fs_alice):
        result = await fs_alice.resolve_kb("bob-kb")
        assert result is None

    async def test_resolve_kb_own_returns_data(self, fs_alice):
        result = await fs_alice.resolve_kb("alice-kb")
        assert result is not None
        assert result["slug"] == "alice-kb"

    async def test_list_documents_other_kb_returns_empty(self, fs_alice):
        docs = await fs_alice.list_documents(str(KB_B_ID))
        assert docs == []

    async def test_list_documents_own_kb_returns_data(self, fs_alice):
        docs = await fs_alice.list_documents(str(KB_A_ID))
        assert len(docs) == 2

    async def test_get_document_other_tenant_returns_none(self, fs_alice):
        doc = await fs_alice.get_document(str(KB_B_ID), "notes.md", "/wiki/")
        assert doc is None

    async def test_find_document_by_name_other_tenant_returns_none(self, fs_alice):
        doc = await fs_alice.find_document_by_name(str(KB_B_ID), "notes.md")
        assert doc is None

    async def test_create_document_uses_exact_wiki_path_segment(
        self,
        fs_alice,
        pg_pool,
    ):
        wikipedia = await fs_alice.create_document(
            KB_A_ID,
            "article.md",
            "Wikipedia article",
            "/wikipedia/",
            "pdf",
            "",
            ["reference"],
        )
        wiki = await fs_alice.create_document(
            KB_A_ID,
            "page.md",
            "Wiki page",
            "/wiki/",
            "pdf",
            "",
            ["wiki"],
        )

        rows = await pg_pool.fetch(
            "SELECT id, source_kind FROM documents WHERE id = ANY($1::uuid[])",
            [str(wikipedia["id"]), str(wiki["id"])],
        )
        by_id = {str(row["id"]): row["source_kind"] for row in rows}
        assert by_id[str(wikipedia["id"])] == "source"
        assert by_id[str(wiki["id"])] == "wiki"

    async def test_retrieve_filters_before_limit_and_isolates_tenant(
        self,
        fs_alice,
        pg_pool,
    ):
        async def insert_ranked(
            *,
            user_id: str,
            kb_id: str,
            path: str,
            filename: str,
            tags: list[str],
            country: str,
            source_content: str,
        ) -> None:
            doc_id = str(uuid.uuid4())
            await pg_pool.execute(
                "INSERT INTO documents "
                "(id, knowledge_base_id, user_id, filename, title, path, source_kind, "
                " file_type, status, tags, metadata, version) "
                "VALUES ($1, $2, $3, $4, $4, $5, 'source', 'md', 'ready', $6, $7, 1)",
                doc_id,
                kb_id,
                user_id,
                filename,
                path,
                tags,
                json.dumps(
                    {
                        "geo_country": [country],
                        "_legacy_tags": 7,
                        "source_hit": "user metadata source hit",
                    }
                ),
            )
            await pg_pool.execute(
                "INSERT INTO document_chunks "
                "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
                " content, source_content, annotations_text, has_highlight, token_count) "
                "VALUES ($1, 1, $2, $3, 0, $4, $5, 'permit annotation', true, 10)",
                doc_id,
                user_id,
                kb_id,
                f"{source_content}\npermit annotation",
                source_content,
            )

        for index in range(4):
            await insert_ranked(
                user_id=USER_A_ID,
                kb_id=KB_A_ID,
                path="/excluded/",
                filename=f"excluded-{index}.md",
                tags=["reviewed", "asean"],
                country="IDN",
                source_content="permit source",
            )
        for index in range(3):
            await insert_ranked(
                user_id=USER_A_ID,
                kb_id=KB_A_ID,
                path="/corpus/idn/",
                filename=f"eligible-{index}.md",
                tags=["Reviewed", "ASEAN"],
                country="IDN",
                source_content="permit source",
            )
        await insert_ranked(
            user_id=USER_B_ID,
            kb_id=KB_B_ID,
            path="/corpus/idn/",
            filename="bob-eligible.md",
            tags=["reviewed", "asean"],
            country="IDN",
            source_content="permit source",
        )

        result = await fs_alice.retrieve(
            KB_A_ID,
            SearchQuery.build(
                text="permit",
                limit=2,
                candidate_limit=2,
                path_glob="/corpus/idn/*.md",
                tags=["REVIEWED", "asean"],
                area="sources",
                document_kinds=["source"],
                annotated_only=True,
                scope="source",
                facets={"country": "IDN"},
            ),
        )

        assert result.returned_count == 2
        assert result.candidate_count == 3
        assert all(hit.path.startswith("/corpus/idn/eligible-") for hit in result.hits)
        assert all(hit.tags == ("asean", "reviewed") for hit in result.hits)
        assert all(
            dict(hit.metadata)
            == {
                "geo_country": ("IDN",),
                "_legacy_tags": 7,
                "source_hit": "user metadata source hit",
            }
            for hit in result.hits
        )
        assert all(
            search_hit_to_legacy_dict(hit)["tags"] == ["Reviewed", "ASEAN"]
            for hit in result.hits
        )

    async def test_load_asset_bytes_other_tenant_returns_none(self, fs_alice, monkeypatch):
        calls: list[str] = []

        async def fake_load_s3(self, key):
            calls.append(key)
            return b"asset-bytes"

        monkeypatch.setattr(PostgresVaultFS, "_load_s3", fake_load_s3)

        data = await fs_alice.load_asset_bytes(str(ASSET_B_ID))

        assert data is None
        assert calls == []

    async def test_load_asset_bytes_own_tenant_uses_own_s3_prefix(self, fs_alice, monkeypatch):
        calls: list[str] = []

        async def fake_load_s3(self, key):
            calls.append(key)
            return b"asset-bytes"

        monkeypatch.setattr(PostgresVaultFS, "_load_s3", fake_load_s3)

        data = await fs_alice.load_asset_bytes(str(ASSET_A_ID))

        assert data == b"asset-bytes"
        assert calls == [f"{USER_A_ID}/{ASSET_A_ID}/source.png"]

    async def test_load_image_bytes_uses_own_prefix(self, fs_alice, monkeypatch):
        """Image S3 keys are always prefixed with the caller's user_id, so a
        foreign document_id can never resolve to another tenant's object."""
        calls: list[str] = []

        async def fake_load_s3(self, key):
            calls.append(key)
            return b"img"

        monkeypatch.setattr(PostgresVaultFS, "_load_s3", fake_load_s3)

        await fs_alice.load_image_bytes(str(DOC_B_ID), "fig-1.png")

        assert calls == [f"{USER_A_ID}/{DOC_B_ID}/images/fig-1.png"]


class TestWriteIsolation:
    """PostgresVaultFS writes include WHERE user_id = $N (service-role)."""

    async def test_archive_documents_other_tenant_returns_zero(self, fs_alice, pg_pool):
        count = await fs_alice.archive_documents([str(DOC_B_ID)])
        assert count == 0
        row = await pg_pool.fetchrow("SELECT archived FROM documents WHERE id = $1", DOC_B_ID)
        assert row["archived"] is False

    async def test_archive_mixed_batch_archives_only_own(self, fs_alice, pg_pool):
        """A batch mixing Alice's and Bob's ids archives only Alice's."""
        count = await fs_alice.archive_documents([str(DOC_A2_ID), str(DOC_B2_ID)])
        assert count == 1
        rows = await pg_pool.fetch(
            "SELECT id, archived FROM documents WHERE id = ANY($1::uuid[])",
            [DOC_A2_ID, DOC_B2_ID],
        )
        archived = {str(r["id"]): r["archived"] for r in rows}
        assert archived[DOC_A2_ID] is True
        assert archived[DOC_B2_ID] is False

    async def test_update_document_other_tenant_does_not_modify(self, fs_alice, pg_pool):
        result = await fs_alice.update_document(str(DOC_B_ID), "pwned by alice")
        assert result is None
        row = await pg_pool.fetchrow("SELECT content FROM documents WHERE id = $1", DOC_B_ID)
        assert row["content"] == "Bob secret"

    async def test_update_knowledge_base_other_tenant_does_not_modify(self, fs_alice, pg_pool):
        result = await fs_alice.update_knowledge_base(str(KB_B_ID), kind="course")
        assert result is None
        row = await pg_pool.fetchrow("SELECT kind FROM knowledge_bases WHERE id = $1", KB_B_ID)
        assert row["kind"] == "wiki"

    async def test_update_knowledge_base_rename_other_tenant_does_not_modify(self, fs_alice, pg_pool):
        result = await fs_alice.update_knowledge_base(str(KB_B_ID), name="Pwned")
        assert result is None
        row = await pg_pool.fetchrow("SELECT name FROM knowledge_bases WHERE id = $1", KB_B_ID)
        assert row["name"] != "Pwned"

    async def test_update_knowledge_base_rename_own_regenerates_slug(self, fs_alice, pg_pool):
        updated = await fs_alice.update_knowledge_base(str(KB_A_ID), name="Deep Learning Notes")
        assert updated["name"] == "Deep Learning Notes"
        assert updated["slug"] == "deep-learning-notes"
        row = await pg_pool.fetchrow("SELECT slug, kind FROM knowledge_bases WHERE id = $1", KB_A_ID)
        assert row["slug"] == "deep-learning-notes"
        assert row["kind"] == "wiki"

    async def test_create_document_into_other_tenant_kb_rejected(self, fs_alice, pg_pool):
        """create_document writes into the caller's own KB but refuses a foreign one."""
        own = await fs_alice.create_document(
            str(KB_A_ID), "mine.md", "Mine", "/wiki/", "md", "x", ["t"],
        )
        assert own["filename"] == "mine.md"

        with pytest.raises(PermissionError):
            await fs_alice.create_document(
                str(KB_B_ID), "sneak.md", "Sneak", "/wiki/", "md", "x", ["t"],
            )
        rows = await pg_pool.fetch("SELECT id FROM documents WHERE filename = 'sneak.md'")
        assert rows == []


class TestReferenceIsolation:
    """Reference mutations now use scoped_execute (RLS-enforced)."""

    async def test_delete_references_other_tenant_deletes_nothing(self, fs_alice, pg_pool):
        await fs_alice.delete_references(str(DOC_B_ID))
        row = await pg_pool.fetchrow(
            "SELECT id FROM document_references WHERE id = $1", REF_B_ID,
        )
        assert row is not None

    async def test_upsert_reference_cross_tenant_kb_fails(self, fs_alice, pg_pool):
        """An edge written into Bob's KB is rejected by RLS and caught, leaving
        Bob's KB with only his own pre-existing reference."""
        await fs_alice.upsert_reference(
            str(DOC_A_ID), str(DOC_A2_ID), str(KB_B_ID), "cites", None,
        )
        rows = await pg_pool.fetch(
            "SELECT id FROM document_references WHERE knowledge_base_id = $1",
            KB_B_ID,
        )
        assert [str(r["id"]) for r in rows] == [REF_B_ID]

    async def test_delete_own_references_works(self, fs_alice, pg_pool):
        await fs_alice.delete_references(str(DOC_A_ID))
        row = await pg_pool.fetchrow(
            "SELECT id FROM document_references WHERE id = $1", REF_A_ID,
        )
        assert row is None


class TestListDocumentsWithContentIsolation:

    async def test_list_with_content_other_kb_returns_empty(self, fs_alice):
        docs = await fs_alice.list_documents_with_content(str(KB_B_ID))
        assert docs == []

    async def test_list_with_content_own_kb_returns_data(self, fs_alice):
        docs = await fs_alice.list_documents_with_content(str(KB_A_ID))
        assert len(docs) == 2
        contents = [d.get("content") for d in docs if d.get("content")]
        assert "Alice secret" in contents


class TestPageIsolation:
    """get_pages / get_all_pages use RLS on document_pages."""

    async def test_get_pages_other_tenant_doc_returns_empty(self, fs_alice):
        pages = await fs_alice.get_pages(str(DOC_B_ID), [1])
        assert pages == []

    async def test_get_pages_own_doc_returns_data(self, fs_alice):
        pages = await fs_alice.get_pages(str(DOC_A_ID), [1])
        assert len(pages) == 1
        assert pages[0]["content"] == "Alice page 1"

    async def test_get_all_pages_other_tenant_doc_returns_empty(self, fs_alice):
        pages = await fs_alice.get_all_pages(str(DOC_B_ID))
        assert pages == []

    async def test_get_all_pages_own_doc_returns_data(self, fs_alice):
        pages = await fs_alice.get_all_pages(str(DOC_A_ID))
        assert len(pages) == 1


class TestGraphQueryIsolation:
    """Backlinks, forward references, uncited sources, stale pages."""

    async def test_get_backlinks_other_tenant_doc_returns_empty(self, fs_alice):
        """DOC_B2 is the target of Bob's citation; Alice sees no backlinks to it."""
        links = await fs_alice.get_backlinks(str(DOC_B2_ID))
        assert links == []

    async def test_get_backlinks_own_doc_returns_data(self, fs_alice):
        """DOC_A2 is the target of Alice's citation (DOC_A cites DOC_A2)."""
        links = await fs_alice.get_backlinks(str(DOC_A2_ID))
        assert len(links) == 1
        assert links[0]["filename"] == "notes.md"

    async def test_get_forward_references_other_tenant_doc_returns_empty(self, fs_alice):
        refs = await fs_alice.get_forward_references(str(DOC_B_ID))
        assert refs == []

    async def test_get_forward_references_own_doc_returns_data(self, fs_alice):
        refs = await fs_alice.get_forward_references(str(DOC_A_ID))
        assert len(refs) == 1
        assert refs[0]["filename"] == "source.pdf"

    async def test_find_uncited_sources_other_kb_returns_empty(self, fs_alice):
        sources = await fs_alice.find_uncited_sources(str(KB_B_ID))
        assert sources == []

    async def test_find_stale_pages_other_kb_returns_empty(self, fs_alice):
        stale = await fs_alice.find_stale_pages(str(KB_B_ID))
        assert stale == []

    async def test_find_stale_pages_own_kb_excludes_other_tenant(self, fs_alice):
        """Alice's KB has no stale pages (only Bob's doc is stale)."""
        stale = await fs_alice.find_stale_pages(str(KB_A_ID))
        assert stale == []


class TestPropagateStalenesIsolation:
    """propagate_staleness uses service_execute with WHERE user_id."""

    async def test_propagate_staleness_does_not_affect_other_tenant(self, fs_alice, pg_pool):
        """Alice propagating staleness for Bob's doc should not mark Alice's docs stale."""
        await fs_alice.propagate_staleness(str(DOC_B2_ID))
        row = await pg_pool.fetchrow(
            "SELECT stale_since FROM documents WHERE id = $1", DOC_A_ID,
        )
        assert row["stale_since"] is None


class TestBidirectional:
    """Verify isolation works from Bob's perspective too."""

    async def test_bob_cannot_see_alice_kb(self, fs_bob):
        result = await fs_bob.resolve_kb("alice-kb")
        assert result is None

    async def test_bob_cannot_list_alice_docs(self, fs_bob):
        docs = await fs_bob.list_documents(str(KB_A_ID))
        assert docs == []

    async def test_bob_cannot_archive_alice_doc(self, fs_bob, pg_pool):
        count = await fs_bob.archive_documents([str(DOC_A_ID)])
        assert count == 0
        row = await pg_pool.fetchrow("SELECT archived FROM documents WHERE id = $1", DOC_A_ID)
        assert row["archived"] is False

    async def test_bob_cannot_read_alice_pages(self, fs_bob):
        pages = await fs_bob.get_pages(str(DOC_A_ID), [1])
        assert pages == []

    async def test_bob_cannot_get_alice_backlinks(self, fs_bob):
        links = await fs_bob.get_backlinks(str(DOC_A2_ID))
        assert links == []


class TestHostedRetrievalIsolation:
    profile = EmbeddingProfile("openai_compatible", "isolation-v1", 3)

    async def _seed_chunk_and_vector(self, pg_pool, *, user_id, kb_id, doc_id, vector):
        await pg_pool.execute(
            "INSERT INTO document_chunks "
            "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
            "content, source_content, token_count) VALUES ($1, 1, $2, $3, 0, $4, $4, 3)",
            doc_id,
            user_id,
            kb_id,
            f"content-{doc_id}",
        )
        await pg_pool.execute(
            "INSERT INTO chunk_embeddings "
            "(user_id, knowledge_base_id, document_id, document_version, chunk_index, "
            "provider, model, dimensions, embedding) "
            "VALUES ($1, $2, $3, 1, 0, $4, $5, 3, $6::vector)",
            user_id,
            kb_id,
            doc_id,
            self.profile.provider,
            self.profile.model,
            vector,
        )

    async def test_vector_retrieval_is_tenant_and_kb_scoped(
        self, fs_alice, pg_pool
    ):
        await self._seed_chunk_and_vector(
            pg_pool,
            user_id=USER_A_ID,
            kb_id=KB_A_ID,
            doc_id=DOC_A_ID,
            vector="[1,0,0]",
        )
        await self._seed_chunk_and_vector(
            pg_pool,
            user_id=USER_B_ID,
            kb_id=KB_B_ID,
            doc_id=DOC_B_ID,
            vector="[1,0,0]",
        )

        result = await fs_alice.retrieve_vector(
            KB_A_ID,
            SearchQuery.build(text="q", limit=5, candidate_limit=5),
            embedding=(1.0, 0.0, 0.0),
            profile=self.profile,
        )

        assert [hit.document_id for hit in result.hits] == [DOC_A_ID]
        assert result.candidate_count == 1

    async def test_absent_current_embedding_profile_is_typed_unavailable(
        self, fs_alice
    ):
        with pytest.raises(RetrieverUnavailable, match="current embeddings"):
            await fs_alice.retrieve_vector(
                KB_A_ID,
                SearchQuery.build(text="q", limit=1),
                embedding=(1.0, 0.0, 0.0),
                profile=self.profile,
            )

    async def test_graph_expansion_is_one_hop_current_and_tenant_scoped(
        self, fs_alice, pg_pool
    ):
        for chunk_index in range(2):
            await pg_pool.execute(
                "INSERT INTO document_chunks "
                "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
                "content, source_content, token_count) "
                "VALUES ($1, 1, $2, $3, $4, $5, $5, 3)",
                DOC_A2_ID,
                USER_A_ID,
                KB_A_ID,
                chunk_index,
                f"alice-related-{chunk_index}",
            )
        await pg_pool.execute(
            "INSERT INTO document_chunks "
            "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
            "content, source_content, token_count) VALUES ($1, 1, $2, $3, 0, 'bob', 'bob', 3)",
            DOC_B2_ID,
            USER_B_ID,
            KB_B_ID,
        )
        await pg_pool.execute(
            "INSERT INTO document_references "
            "(source_document_id, target_document_id, knowledge_base_id, reference_type) "
            "VALUES ($1, $2, $3, 'links_to')",
            DOC_A_ID,
            DOC_B2_ID,
            KB_A_ID,
        )
        await pg_pool.execute(
            "INSERT INTO document_references "
            "(source_document_id, target_document_id, knowledge_base_id, reference_type) "
            "VALUES ($1, $1, $2, 'links_to')",
            DOC_A_ID,
            KB_A_ID,
        )
        direct = SearchHit(DOC_A_ID, 1, 0, "direct", 1.0, "/wiki/notes.md")

        expanded = await fs_alice.expand_references(
            KB_A_ID,
            SearchQuery.build(text="q", limit=3),
            (direct,),
            limit=2,
        )

        assert [(hit.document_id, hit.chunk_index) for hit in expanded] == [
            (DOC_A2_ID, 0)
        ]
