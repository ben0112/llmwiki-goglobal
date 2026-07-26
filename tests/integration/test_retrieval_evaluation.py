import asyncio
import json
import os
from math import inf, nan
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import asyncpg
import pytest
from scripts import retrieval_eval
from scripts.retrieval_eval import main
from services.vector_store import PostgresVectorStore

from llmwiki_core import evaluation_dataset_digest, load_cases
from llmwiki_core.models import EmbeddingProfile

USER_ID = UUID("00000000-0000-0000-0000-000000000011")
KNOWLEDGE_BASE_ID = UUID("00000000-0000-0000-0000-000000000012")
POLICY_ID = UUID("00000000-0000-0000-0000-000000000101")
SEMANTIC_ID = UUID("00000000-0000-0000-0000-000000000102")
DISTRACTOR_ID = UUID("00000000-0000-0000-0000-000000000103")
OTHER_TENANT_ID = UUID("00000000-0000-0000-0000-000000000013")
OTHER_KB_ID = UUID("00000000-0000-0000-0000-000000000014")
OTHER_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000104")
SAME_USER_OTHER_KB_ID = UUID("00000000-0000-0000-0000-000000000015")
SAME_USER_OTHER_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000105")
STALE_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000106")
OTHER_PROFILE_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000107")
EMPTY_KB_ID = UUID("00000000-0000-0000-0000-000000000016")
PROFILE = EmbeddingProfile("openai_compatible", "evaluation-v1", 3)
OTHER_PROFILE = EmbeddingProfile("openai_compatible", "evaluation-other", 3)
OTHER_DIMENSIONS = EmbeddingProfile("openai_compatible", "evaluation-v1", 2)
EXPECTED_DATASET_DIGEST = "d45bf89f5b28694afe2b4af1d03d15e3ba59e02d3bc20129eff77b58ab39ab7f"


class _RecordingPool:
    def __init__(self):
        self.fetches = []
        self.returned_document_ids = []
        self.acquire_count = 0
        self.release_count = 0
        self.codec_calls = 0

    def acquire(self):
        return _RecordingAcquire(self)


class _RecordingAcquire:
    def __init__(self, pool):
        self._pool = pool
        self._connection = None

    async def __aenter__(self):
        self._connection = await asyncpg.connect(os.environ["DATABASE_URL"])
        self._pool.acquire_count += 1
        return _RecordingConnection(self._pool, self._connection)

    async def __aexit__(self, _error_type, _error, _traceback):
        await self._connection.close()
        self._pool.release_count += 1


class _RecordingConnection:
    def __init__(self, pool, connection):
        self._pool = pool
        self._connection = connection

    def transaction(self, **options):
        return self._connection.transaction(**options)

    async def set_type_codec(self, *args, **kwargs):
        self._pool.codec_calls += 1
        raise AssertionError("evaluation must not set connection codecs")

    async def reset_type_codec(self, *args, **kwargs):
        self._pool.codec_calls += 1
        raise AssertionError("evaluation must not reset connection codecs")

    async def fetch(self, sql, *params):
        self._pool.fetches.append((sql, params))
        rows = await self._connection.fetch(sql, *params)
        self._pool.returned_document_ids.extend(
            str(row["document_id"])
            for row in rows
            if "document_id" in row
        )
        return rows


class _FakeQueryEmbeddingClient:
    profile = PROFILE

    def __init__(self, calls):
        self._calls = calls

    async def embed(self, texts):
        self._calls.append(tuple(texts))
        vectors = {
            "export permit": (1.0, 0.0, 0.0),
            "semantic compliance": (0.0, 1.0, 0.0),
        }
        return tuple(vectors[text] for text in texts)

    async def aclose(self):
        return None


class _MutatingQueryEmbeddingClient(_FakeQueryEmbeddingClient):
    def __init__(self, calls, mutation, state):
        super().__init__(calls)
        self._mutation = mutation
        self._state = state

    async def embed(self, texts):
        if not self._state.done:
            self._state.done = True
            await self._mutation()
        return await super().embed(texts)


def _dataset(path: Path) -> Path:
    cases = (
        {
            "schema_version": 1,
            "case_id": "filtered-policy",
            "query": {
                "text": "export permit",
                "limit": 1,
                "candidate_limit": 1,
                "area": "sources",
                "scope": "all",
                "facets": {"stage": "S2"},
                "path_glob": "/target/*.md",
                "tags": ["alpha"],
                "document_kinds": ["source"],
                "annotated_only": True,
            },
            "relevance": [
                {"document_id": str(POLICY_ID), "chunk_index": 0, "grade": 2}
            ],
        },
        {
            "schema_version": 1,
            "case_id": "semantic-only",
            "query": {
                "text": "semantic compliance",
                "limit": 1,
                "candidate_limit": 1,
            },
            "relevance": [
                {"document_id": str(SEMANTIC_ID), "chunk_index": 0, "grade": 1}
            ],
        },
    )
    path.write_bytes(
        b"".join(
            (json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for case in cases
        )
    )
    return path


async def _seed_document(
    pool,
    *,
    user_id,
    knowledge_base_id,
    document_id,
    filename,
    path,
    content,
    tags=("alpha",),
    metadata=None,
    annotated=False,
):
    await pool.execute(
        "INSERT INTO documents "
        "(id, knowledge_base_id, user_id, filename, path, source_kind, file_type, "
        "status, version, tags, metadata) "
        "VALUES ($1,$2,$3,$4,$5,'source','md','ready',1,$6,$7::jsonb)",
        document_id,
        knowledge_base_id,
        user_id,
        filename,
        path,
        list(tags),
        json.dumps(metadata) if metadata is not None else None,
    )
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id, document_version, user_id, knowledge_base_id, chunk_index, "
        "content, source_content, annotations_text, has_highlight, token_count) "
        "VALUES ($1,1,$2,$3,0,$4,$4,$5,$6,2)",
        document_id,
        user_id,
        knowledge_base_id,
        content,
        "reviewed" if annotated else None,
        annotated,
    )


@pytest.fixture(scope="module")
async def evaluation_corpus(pool, tmp_path_factory):
    await pool.execute(
        "INSERT INTO users (id,email) VALUES ($1,'evaluation@test.invalid'),"
        "($2,'other-evaluation@test.invalid')",
        USER_ID,
        OTHER_TENANT_ID,
    )
    await pool.execute(
        "INSERT INTO knowledge_bases (id,user_id,name,slug) VALUES "
        "($1,$2,'Evaluation','evaluation'),($3,$4,'Other','other-evaluation'),"
        "($5,$2,'Same User Other','same-user-other'),($6,$2,'Empty','empty-evaluation')",
        KNOWLEDGE_BASE_ID,
        USER_ID,
        OTHER_KB_ID,
        OTHER_TENANT_ID,
        SAME_USER_OTHER_KB_ID,
        EMPTY_KB_ID,
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=POLICY_ID,
        filename="policy.md",
        path="/target/",
        content="export permit",
        metadata={"stage": "S2"},
        annotated=True,
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=SAME_USER_OTHER_KB_ID,
        document_id=SAME_USER_OTHER_DOCUMENT_ID,
        filename="same-user-private.md",
        path="/target/",
        content="export permit semantic compliance",
        metadata={"stage": "S2"},
        annotated=True,
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=STALE_DOCUMENT_ID,
        filename="stale.md",
        path="/target/",
        content="export permit semantic compliance",
        metadata={"stage": "S2"},
        annotated=True,
    )
    await pool.execute("UPDATE documents SET version=2 WHERE id=$1", STALE_DOCUMENT_ID)
    await pool.execute(
        "INSERT INTO document_chunks "
        "(document_id,document_version,user_id,knowledge_base_id,chunk_index,content,"
        "source_content,token_count) VALUES ($1,2,$2,$3,1,'current unrelated',"
        "'current unrelated',2)",
        STALE_DOCUMENT_ID,
        USER_ID,
        KNOWLEDGE_BASE_ID,
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        filename="other-profile.md",
        path="/target/",
        content="irrelevant alternate model",
        metadata={"stage": "S2"},
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=SEMANTIC_ID,
        filename="semantic.md",
        path="/target/",
        content="unrelated narrative",
        metadata={"stage": "S2"},
    )
    await _seed_document(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DISTRACTOR_ID,
        filename="distractor.md",
        path="/other/",
        content="semantic compliance",
        tags=("wrong",),
        metadata={"stage": "wrong"},
    )
    await _seed_document(
        pool,
        user_id=OTHER_TENANT_ID,
        knowledge_base_id=OTHER_KB_ID,
        document_id=OTHER_DOCUMENT_ID,
        filename="private.md",
        path="/target/",
        content="export permit semantic compliance",
        metadata={"stage": "S2"},
        annotated=True,
    )
    store = PostgresVectorStore(pool, profile=PROFILE)
    vectors = {
        POLICY_ID: (1.0, 0.0, 0.0),
        SEMANTIC_ID: (0.0, 1.0, 0.0),
        DISTRACTOR_ID: (0.0, 0.8, 0.2),
    }
    for document_id, vector in vectors.items():
        await store.replace_document_embeddings(
            user_id=USER_ID,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=document_id,
            document_version=1,
            embeddings=((0, vector),),
        )
    await PostgresVectorStore(pool, profile=PROFILE).replace_document_embeddings(
        user_id=OTHER_TENANT_ID,
        knowledge_base_id=OTHER_KB_ID,
        document_id=OTHER_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (1.0, 0.0, 0.0)),),
    )
    await PostgresVectorStore(pool, profile=PROFILE).replace_document_embeddings(
        user_id=USER_ID,
        knowledge_base_id=SAME_USER_OTHER_KB_ID,
        document_id=SAME_USER_OTHER_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (1.0, 0.0, 0.0)),),
    )
    await pool.execute(
        "INSERT INTO chunk_embeddings "
        "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
        "provider,model,dimensions,embedding) VALUES ($1,$2,$3,1,0,$4,$5,3,$6::vector)",
        USER_ID,
        KNOWLEDGE_BASE_ID,
        STALE_DOCUMENT_ID,
        PROFILE.provider,
        PROFILE.model,
        "[1,0,0]",
    )
    await pool.execute(
        "INSERT INTO chunk_embeddings "
        "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
        "provider,model,dimensions,embedding) VALUES ($1,$2,$3,2,1,$4,$5,3,$6::vector)",
        USER_ID,
        KNOWLEDGE_BASE_ID,
        STALE_DOCUMENT_ID,
        PROFILE.provider,
        PROFILE.model,
        "[0,0,1]",
    )
    await PostgresVectorStore(pool, profile=PROFILE).replace_document_embeddings(
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (0.0, 0.0, 1.0)),),
    )
    await PostgresVectorStore(pool, profile=OTHER_PROFILE).replace_document_embeddings(
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (0.0, 1.0, 0.0)),),
    )
    await pool.execute(
        "INSERT INTO chunk_embeddings "
        "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
        "provider,model,dimensions,embedding) VALUES ($1,$2,$3,1,0,'other_provider',"
        "$4,3,'[0,1,0]'::vector)",
        USER_ID,
        KNOWLEDGE_BASE_ID,
        OTHER_PROFILE_DOCUMENT_ID,
        PROFILE.model,
    )
    await PostgresVectorStore(pool, profile=OTHER_DIMENSIONS).replace_document_embeddings(
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (0.0, 1.0)),),
    )
    coverage = await pool.fetchrow(
        "WITH current_chunks AS ("
        "SELECT dc.document_id,dc.document_version,dc.chunk_index FROM document_chunks dc "
        "JOIN documents d ON d.id=dc.document_id WHERE dc.user_id=$1 "
        "AND dc.knowledge_base_id=$2 AND d.user_id=$1 AND d.knowledge_base_id=$2 "
        "AND dc.document_version=d.version AND d.status='ready' AND NOT d.archived"
        ") SELECT count(*) AS current_count, count(*) FILTER (WHERE EXISTS("
        "SELECT 1 FROM chunk_embeddings ce WHERE ce.user_id=$1 AND ce.knowledge_base_id=$2 "
        "AND ce.document_id=current_chunks.document_id "
        "AND ce.document_version=current_chunks.document_version "
        "AND ce.chunk_index=current_chunks.chunk_index AND ce.provider=$3 "
        "AND ce.model=$4 AND ce.dimensions=$5)) AS covered_count FROM current_chunks",
        USER_ID,
        KNOWLEDGE_BASE_ID,
        PROFILE.provider,
        PROFILE.model,
        PROFILE.dimensions,
    )
    fixture_root = tmp_path_factory.mktemp("retrieval-evaluation")
    return SimpleNamespace(
        dataset=_dataset(fixture_root / "cases.jsonl"),
        pool=pool,
        current_count=coverage["current_count"],
        covered_count=coverage["covered_count"],
    )


def _factory(
    corpus,
    *,
    hybrid_latency_ms,
    profile=PROFILE,
    knowledge_base_id=KNOWLEDGE_BASE_ID,
    embedding_client_factory=None,
):
    factory_builder = getattr(retrieval_eval, "postgres_evaluation_retriever_factory", None)
    assert callable(factory_builder), "Postgres evaluation adapter is not implemented"
    calls = []
    recording_pool = _RecordingPool()
    factory = factory_builder(
        recording_pool,
        user_id=USER_ID,
        knowledge_base_id=knowledge_base_id,
        embedding_profile=profile,
        embedding_client_factory=(
            (lambda: _FakeQueryEmbeddingClient(calls))
            if embedding_client_factory is None
            else embedding_client_factory(calls)
        ),
        lexical_candidate_limit=1,
        vector_candidate_limit=1,
        rrf_k=60,
        latency_ms=lambda selected, _query: 10.0 if selected == "lexical" else hybrid_latency_ms,
    )
    return factory, calls, recording_pool


def _invoke(capsys, corpus, factory, *, require_gate=True):
    arguments = ["--dataset", str(corpus.dataset), "--compare"]
    if require_gate:
        arguments.append("--require-promotion-gate")
    code = main(arguments, retriever_factory=factory)
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out else None, captured


def test_real_postgres_comparison_is_exact_filtered_and_byte_stable(
    evaluation_corpus, capsys
):
    factory, embedding_calls, recording_pool = _factory(
        evaluation_corpus, hybrid_latency_ms=20.0
    )

    first_code, first_payload, first = _invoke(capsys, evaluation_corpus, factory)
    second_code, second_payload, second = _invoke(capsys, evaluation_corpus, factory)

    assert first_code == second_code == 0
    assert first.err == second.err == ""
    assert first.out == second.out
    assert first_payload == second_payload
    assert recording_pool.acquire_count == recording_pool.release_count == 2
    assert recording_pool.codec_calls == 0
    assert evaluation_corpus.current_count == evaluation_corpus.covered_count == 5
    assert evaluation_dataset_digest(load_cases(evaluation_corpus.dataset)) == EXPECTED_DATASET_DIGEST
    assert first_payload == {
        "case_count": 2,
        "dataset_schema_version": 1,
        "evaluation_dataset_digest": EXPECTED_DATASET_DIGEST,
        "profile": "compare",
        "profiles": {
            "hybrid": {
                "case_count": 2,
                "metrics": {
                    "filtered_result_count": 1,
                    "latency_p50_ms": 20.0,
                    "latency_p95_ms": 20.0,
                    "mrr": 1.0,
                    "ndcg_at_10": 1.0,
                    "recall_at_10": 1.0,
                    "recall_at_20": 1.0,
                    "recall_at_5": 1.0,
                },
                "profile": "hybrid",
            },
            "lexical": {
                "case_count": 2,
                "metrics": {
                    "filtered_result_count": 1,
                    "latency_p50_ms": 10.0,
                    "latency_p95_ms": 10.0,
                    "mrr": 0.5,
                    "ndcg_at_10": 0.5,
                    "recall_at_10": 0.5,
                    "recall_at_20": 0.5,
                    "recall_at_5": 0.5,
                },
                "profile": "lexical",
            },
        },
        "promotion": {
            "eligible": True,
            "latency_ratio": 2.0,
            "reason": "eligible",
            "recall_ratio": 2.0,
        },
        "schema_version": 1,
    }
    assert embedding_calls == [
        ("export permit",),
        ("semantic compliance",),
        ("export permit",),
        ("semantic compliance",),
    ]
    retrieval_fetches = [item for item in recording_pool.fetches if "candidate_count" in item[0]]
    assert retrieval_fetches
    assert all(params[-1] == 1 for _sql, params in retrieval_fetches)
    assert sum("WITH current_chunks AS" in sql for sql, _params in recording_pool.fetches) == 2
    filtered_sql = next(sql for sql, params in retrieval_fetches if "S2" in params)
    assert "::text AS metadata" in filtered_sql
    for predicate in ("source_kind", "tags", "metadata", "has_highlight", "LIKE"):
        assert predicate in filtered_sql
    assert set(recording_pool.returned_document_ids) <= {
        str(POLICY_ID),
        str(SEMANTIC_ID),
        str(DISTRACTOR_ID),
    }
    for secret in (
        "export permit",
        "semantic compliance",
        "unrelated narrative",
        str(POLICY_ID),
        "postgresql://",
    ):
        assert secret not in first.out


def test_single_profile_uses_its_own_snapshot_without_vector_coverage(
    evaluation_corpus, capsys
):
    factory, embedding_calls, recording_pool = _factory(
        evaluation_corpus, hybrid_latency_ms=20.0
    )

    code = main(
        ["--dataset", str(evaluation_corpus.dataset), "--profile", "lexical"],
        retriever_factory=factory,
    )
    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["profile"] == "lexical"
    assert captured.err == ""
    assert embedding_calls == []
    assert recording_pool.acquire_count == recording_pool.release_count == 1
    assert not any("WITH current_chunks AS" in sql for sql, _ in recording_pool.fetches)


def test_promotion_gate_strictly_rejects_latency_above_exact_boundary(
    evaluation_corpus, capsys
):
    factory, _calls, _recording_pool = _factory(
        evaluation_corpus, hybrid_latency_ms=20.000000000001
    )

    code, payload, captured = _invoke(capsys, evaluation_corpus, factory)

    assert code == 3
    assert captured.err == ""
    assert payload["promotion"] == {
        "eligible": False,
        "latency_ratio": 2.0,
        "reason": "gate_failed",
        "recall_ratio": 2.0,
    }


def test_missing_vector_profile_fails_closed_instead_of_passing_promotion(
    evaluation_corpus, capsys
):
    unavailable_profile = EmbeddingProfile("openai_compatible", "missing-vectors", 3)
    factory, _calls, _recording_pool = _factory(
        evaluation_corpus,
        hybrid_latency_ms=20.0,
        profile=unavailable_profile,
    )

    code, payload, captured = _invoke(capsys, evaluation_corpus, factory)

    assert code == 2
    assert payload is None
    assert json.loads(captured.err) == {
        "error": {"category": "retrieval", "code": "retrieval_contract_invalid"}
    }
    for secret in ("missing-vectors", "postgresql://", "export permit"):
        assert secret not in captured.err


async def _assert_missing_vector_fails_closed(
    evaluation_corpus,
    capsys,
    *,
    document_id,
    document_version,
    chunk_index,
):
    await evaluation_corpus.pool.execute(
        "DELETE FROM chunk_embeddings WHERE user_id=$1 AND knowledge_base_id=$2 "
        "AND document_id=$3 AND document_version=$4 AND chunk_index=$5 "
        "AND provider=$6 AND model=$7 AND dimensions=$8",
        USER_ID,
        KNOWLEDGE_BASE_ID,
        document_id,
        document_version,
        chunk_index,
        PROFILE.provider,
        PROFILE.model,
        PROFILE.dimensions,
    )
    try:
        factory, calls, _recording_pool = _factory(
            evaluation_corpus,
            hybrid_latency_ms=20.0,
        )
        code, payload, captured = _invoke(capsys, evaluation_corpus, factory)
    finally:
        await evaluation_corpus.pool.execute(
            "INSERT INTO chunk_embeddings "
            "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
            "provider,model,dimensions,embedding) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::vector)",
            USER_ID,
            KNOWLEDGE_BASE_ID,
            document_id,
            document_version,
            chunk_index,
            PROFILE.provider,
            PROFILE.model,
            PROFILE.dimensions,
            "[0,1,0]",
        )

    assert code == 2
    assert payload is None
    assert calls == []
    assert json.loads(captured.err) == {
        "error": {"category": "retrieval", "code": "retrieval_contract_invalid"}
    }
    for secret in (str(document_id), "postgresql://", "semantic compliance", "[0,1,0]"):
        assert secret not in captured.err


@pytest.mark.asyncio
async def test_missing_related_current_chunk_vector_fails_closed_before_query_embedding(
    evaluation_corpus, capsys
):
    await _assert_missing_vector_fails_closed(
        evaluation_corpus,
        capsys,
        document_id=SEMANTIC_ID,
        document_version=1,
        chunk_index=0,
    )


@pytest.mark.asyncio
async def test_missing_filtered_nonrelevant_chunk_ignores_wrong_profiles_and_fails_closed(
    evaluation_corpus, capsys
):
    await _assert_missing_vector_fails_closed(
        evaluation_corpus,
        capsys,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        document_version=1,
        chunk_index=0,
    )


@pytest.mark.asyncio
async def test_stale_vector_does_not_cover_missing_current_document_version(
    evaluation_corpus, capsys
):
    assert await evaluation_corpus.pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM chunk_embeddings WHERE document_id=$1 "
        "AND document_version=1 AND provider=$2 AND model=$3 AND dimensions=$4)",
        STALE_DOCUMENT_ID,
        PROFILE.provider,
        PROFILE.model,
        PROFILE.dimensions,
    )
    await _assert_missing_vector_fails_closed(
        evaluation_corpus,
        capsys,
        document_id=STALE_DOCUMENT_ID,
        document_version=2,
        chunk_index=1,
    )


def test_zero_current_chunk_cohort_fails_closed(evaluation_corpus, capsys):
    factory, calls, _recording_pool = _factory(
        evaluation_corpus,
        hybrid_latency_ms=20.0,
        knowledge_base_id=EMPTY_KB_ID,
    )

    code, payload, captured = _invoke(capsys, evaluation_corpus, factory)

    assert code == 2
    assert payload is None
    assert calls == []
    assert json.loads(captured.err) == {
        "error": {"category": "retrieval", "code": "retrieval_contract_invalid"}
    }


async def _external_execute(sql, *params):
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await connection.execute(sql, *params)
    finally:
        await connection.close()


async def _external_add_current_chunk(document_id):
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        async with connection.transaction():
            await connection.execute(
                "INSERT INTO documents "
                "(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,version) "
                "VALUES ($1,$2,$3,'added.md','/target/','source','md','ready',1)",
                document_id,
                KNOWLEDGE_BASE_ID,
                USER_ID,
            )
            await connection.execute(
                "INSERT INTO document_chunks "
                "(document_id,document_version,user_id,knowledge_base_id,chunk_index,content,"
                "source_content,token_count) VALUES ($1,1,$2,$3,0,'added','added',1)",
                document_id,
                USER_ID,
                KNOWLEDGE_BASE_ID,
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation_kind", ["delete", "add", "version", "status"])
async def test_compare_uses_one_repeatable_read_snapshot_and_next_run_sees_partial_coverage(
    evaluation_corpus,
    capsys,
    mutation_kind,
):
    mutation_document_id = {
        "add": UUID("00000000-0000-0000-0000-000000000201"),
        "status": UUID("00000000-0000-0000-0000-000000000202"),
    }.get(mutation_kind)
    if mutation_kind == "version":
        await evaluation_corpus.pool.execute(
            "INSERT INTO document_chunks "
            "(document_id,document_version,user_id,knowledge_base_id,chunk_index,content,"
            "source_content,token_count) VALUES ($1,2,$2,$3,1,'future','future',1)",
            OTHER_PROFILE_DOCUMENT_ID,
            USER_ID,
            KNOWLEDGE_BASE_ID,
        )
    elif mutation_kind == "status":
        await evaluation_corpus.pool.execute(
            "INSERT INTO documents "
            "(id,knowledge_base_id,user_id,filename,path,source_kind,file_type,status,version) "
            "VALUES ($1,$2,$3,'pending.md','/target/','source','md','pending',1)",
            mutation_document_id,
            KNOWLEDGE_BASE_ID,
            USER_ID,
        )
        await evaluation_corpus.pool.execute(
            "INSERT INTO document_chunks "
            "(document_id,document_version,user_id,knowledge_base_id,chunk_index,content,"
            "source_content,token_count) VALUES ($1,1,$2,$3,0,'pending','pending',1)",
            mutation_document_id,
            USER_ID,
            KNOWLEDGE_BASE_ID,
        )

    async def mutate():
        if mutation_kind == "delete":
            await _external_execute(
                "DELETE FROM chunk_embeddings WHERE document_id=$1 AND document_version=1 "
                "AND chunk_index=0 AND provider=$2 AND model=$3 AND dimensions=$4",
                SEMANTIC_ID,
                PROFILE.provider,
                PROFILE.model,
                PROFILE.dimensions,
            )
        elif mutation_kind == "add":
            await _external_add_current_chunk(mutation_document_id)
        elif mutation_kind == "version":
            await _external_execute(
                "UPDATE documents SET version=2 WHERE id=$1",
                OTHER_PROFILE_DOCUMENT_ID,
            )
        else:
            await _external_execute(
                "UPDATE documents SET status='ready' WHERE id=$1",
                mutation_document_id,
            )

    state = SimpleNamespace(done=False)

    def mutating_factory(calls):
        return lambda: _MutatingQueryEmbeddingClient(calls, mutate, state)

    try:
        factory, calls, recording_pool = _factory(
            evaluation_corpus,
            hybrid_latency_ms=20.0,
            embedding_client_factory=mutating_factory,
        )
        first_code, first_payload, first = _invoke(capsys, evaluation_corpus, factory)
        next_factory, next_calls, next_pool = _factory(
            evaluation_corpus,
            hybrid_latency_ms=20.0,
        )
        next_code, next_payload, next_output = _invoke(
            capsys,
            evaluation_corpus,
            next_factory,
        )
    finally:
        if mutation_kind == "delete":
            await evaluation_corpus.pool.execute(
                "INSERT INTO chunk_embeddings "
                "(user_id,knowledge_base_id,document_id,document_version,chunk_index,"
                "provider,model,dimensions,embedding) VALUES ($1,$2,$3,1,0,$4,$5,$6,'[0,1,0]')",
                USER_ID,
                KNOWLEDGE_BASE_ID,
                SEMANTIC_ID,
                PROFILE.provider,
                PROFILE.model,
                PROFILE.dimensions,
            )
        elif mutation_kind == "add":
            await evaluation_corpus.pool.execute(
                "DELETE FROM documents WHERE id=$1",
                mutation_document_id,
            )
        elif mutation_kind == "version":
            await evaluation_corpus.pool.execute(
                "UPDATE documents SET version=1 WHERE id=$1",
                OTHER_PROFILE_DOCUMENT_ID,
            )
            await evaluation_corpus.pool.execute(
                "DELETE FROM document_chunks WHERE document_id=$1 AND document_version=2",
                OTHER_PROFILE_DOCUMENT_ID,
            )
        else:
            await evaluation_corpus.pool.execute(
                "DELETE FROM documents WHERE id=$1",
                mutation_document_id,
            )

    assert first_code == 0
    assert first_payload["promotion"]["eligible"] is True
    assert first.err == ""
    assert calls == [("export permit",), ("semantic compliance",)]
    assert recording_pool.acquire_count == recording_pool.release_count == 1
    assert sum("WITH current_chunks AS" in sql for sql, _ in recording_pool.fetches) == 1
    assert next_code == 2
    assert next_payload is None
    assert next_calls == []
    assert next_pool.acquire_count == next_pool.release_count == 1
    assert json.loads(next_output.err) == {
        "error": {"category": "retrieval", "code": "retrieval_contract_invalid"}
    }


def _valid_backend_row():
    return {
        "document_id": POLICY_ID,
        "document_version": 1,
        "chunk_index": 0,
        "content": "content",
        "score": 0.5,
        "path": "/target/",
        "filename": "policy.md",
        "title": "Policy",
        "page": 1,
        "header_breadcrumb": "Header",
        "tags": ["alpha"],
        "source_kind": "source",
        "metadata": '{"stage":"S2"}',
        "candidate_count": 1,
    }


@pytest.mark.parametrize("profile", ["lexical", "vector"])
@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("candidate_count", True),
        ("candidate_count", "1"),
        ("candidate_count", None),
        ("candidate_count", -1),
        ("candidate_count", 2**31),
        ("document_id", str(POLICY_ID)),
        ("document_id", None),
        ("document_id", 101),
        ("document_version", True),
        ("document_version", "1"),
        ("document_version", None),
        ("document_version", 0),
        ("document_version", 2**31),
        ("chunk_index", True),
        ("chunk_index", "0"),
        ("chunk_index", None),
        ("chunk_index", -1),
        ("chunk_index", 10_000),
        ("content", None),
        ("content", 1),
        pytest.param("content", "x" * 1_000_001, id="content-too-long"),
        ("path", None),
        ("path", 1),
        ("path", "relative/"),
        ("path", "/bad\x00/"),
        ("filename", None),
        ("filename", 1),
        ("filename", ""),
        ("filename", "nested/file.md"),
        ("score", True),
        ("score", "0.5"),
        ("score", None),
        ("score", nan),
        ("score", inf),
        ("score", -inf),
        ("page", True),
        ("page", "1"),
        ("page", 0),
        ("page", -1),
        ("metadata", None),
        ("metadata", {}),
        ("metadata", []),
        ("metadata", None),
    ],
)
def test_evaluation_backend_rows_reject_implicit_coercions(profile, field, invalid):
    validator = getattr(retrieval_eval, "_validated_evaluation_rows", None)
    assert callable(validator), "strict evaluation row validator is not implemented"
    row = _valid_backend_row()
    row[field] = invalid

    with pytest.raises(retrieval_eval.RetrieverUnavailable) as raised:
        validator((row,), profile=profile)

    assert type(raised.value).__name__ == "RetrieverUnavailable"
    assert raised.value.args == ("evaluation row is invalid",)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_evaluation_backend_rows_parse_exact_default_jsonb_text_shape():
    rows = retrieval_eval._validated_evaluation_rows(
        (_valid_backend_row(),),
        profile="lexical",
    )

    assert rows[0]["metadata"] == {"stage": "S2"}
    assert rows[0]["tags"] == ("alpha",)


@pytest.mark.parametrize(
    "metadata",
    [
        "{",
        '{"stage":"S2","stage":"S3"}',
        "[]",
        "null",
        '"scalar"',
        '{"value":NaN}',
        '{"value":1e999}',
        '{"value":"' + ("x" * 65_536) + '"}',
        ("{" + '"nested":{' * 34 + '"leaf":true' + "}" * 34 + "}"),
    ],
)
def test_evaluation_backend_rows_reject_invalid_metadata_json_text(metadata):
    row = {**_valid_backend_row(), "metadata": metadata}

    with pytest.raises(retrieval_eval.RetrieverUnavailable) as raised:
        retrieval_eval._validated_evaluation_rows((row,), profile="lexical")

    assert raised.value.args == ("evaluation row is invalid",)
    assert raised.value.__cause__ is raised.value.__context__ is None


@pytest.mark.parametrize(
    "tags",
    ["[]", (), {}, ["alpha", 1], [""], ["x" * 129], ["x"] * 101],
)
def test_evaluation_backend_rows_require_exact_string_array_tags(tags):
    row = {**_valid_backend_row(), "tags": tags}

    with pytest.raises(retrieval_eval.RetrieverUnavailable):
        retrieval_eval._validated_evaluation_rows((row,), profile="lexical")


class _SingleConnectionGuardLease:
    def __init__(self, pool):
        self._pool = pool
        self._lease = pool._pool.acquire()
        self._connection = None

    async def __aenter__(self):
        self._connection = await self._lease.__aenter__()
        return _SingleConnectionGuard(self._pool, self._connection)

    async def __aexit__(self, *args):
        return await self._lease.__aexit__(*args)


class _SingleConnectionGuard:
    def __init__(self, pool, connection):
        self._pool = pool
        self._connection = connection

    def transaction(self, **options):
        return self._connection.transaction(**options)

    async def fetch(self, sql, *params):
        return await self._connection.fetch(sql, *params)

    async def set_type_codec(self, *_args, **_kwargs):
        self._pool.codec_calls += 1
        raise AssertionError("evaluation must preserve default codecs")

    async def reset_type_codec(self, *_args, **_kwargs):
        self._pool.codec_calls += 1
        raise AssertionError("evaluation must preserve default codecs")


class _SingleConnectionGuardPool:
    def __init__(self, pool):
        self._pool = pool
        self.codec_calls = 0

    def acquire(self):
        return _SingleConnectionGuardLease(self)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_type",
    [None, RuntimeError, asyncio.CancelledError, GeneratorExit],
)
async def test_single_connection_evaluation_preserves_default_jsonb_codec(failure_type):
    pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=1,
        max_size=1,
    )
    guarded = _SingleConnectionGuardPool(pool)
    factory = _fake_snapshot_factory(guarded)
    try:
        if failure_type is None:
            async with factory.evaluation_session():
                rows = await factory._snapshot.fetch("SELECT '{}'::jsonb AS payload")
                assert rows[0]["payload"] == "{}"
        else:
            expected = (
                retrieval_eval.RetrievalExecutionError
                if failure_type is RuntimeError
                else failure_type
            )
            with pytest.raises(expected):
                async with factory.evaluation_session():
                    raise failure_type("private body")

        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT '{}'::jsonb") == "{}"
        assert guarded.codec_calls == 0
    finally:
        await pool.close()


def test_evaluation_backend_rows_reject_candidate_count_smaller_than_hits():
    validator = getattr(retrieval_eval, "_validated_evaluation_rows", None)
    assert callable(validator), "strict evaluation row validator is not implemented"
    first = _valid_backend_row()
    second = {**_valid_backend_row(), "document_id": SEMANTIC_ID, "candidate_count": 1}

    with pytest.raises(Exception) as raised:
        validator((first, second), profile="lexical")

    assert type(raised.value).__name__ == "RetrieverUnavailable"
    assert raised.value.args == ("evaluation row is invalid",)


@pytest.mark.parametrize(
    ("profile", "score"),
    [("lexical", -0.000001), ("vector", -1.000001), ("vector", 1.000001)],
)
def test_evaluation_backend_rows_reject_profile_score_out_of_range(profile, score):
    row = {**_valid_backend_row(), "score": score}

    with pytest.raises(Exception) as raised:
        retrieval_eval._validated_evaluation_rows((row,), profile=profile)

    assert type(raised.value).__name__ == "RetrieverUnavailable"
    assert raised.value.args == ("evaluation row is invalid",)
    assert raised.value.__cause__ is raised.value.__context__ is None


class _FakeTransaction:
    def __init__(self, events):
        self._events = events

    async def start(self):
        self._events.append("start")

    async def rollback(self):
        self._events.append("rollback")


class _FakeSnapshotConnection:
    def __init__(self, events):
        self._events = events

    async def set_type_codec(self, name, **options):
        raise AssertionError("evaluation must not set connection codecs")

    async def reset_type_codec(self, name, **options):
        raise AssertionError("evaluation must not reset connection codecs")

    def transaction(self, **options):
        assert options == {"isolation": "repeatable_read", "readonly": True}
        self._events.append("transaction")
        return _FakeTransaction(self._events)


class _FakeSnapshotLease:
    def __init__(self, events):
        self._events = events
        self._connection = _FakeSnapshotConnection(events)

    async def __aenter__(self):
        self._events.append("acquire")
        return self._connection

    async def __aexit__(self, error_type, error, traceback):
        assert error_type is error is traceback is None
        self._events.append("release")


class _FakeSnapshotPool:
    def __init__(self):
        self.events = []

    def acquire(self):
        return _FakeSnapshotLease(self.events)


def _fake_snapshot_factory(pool):
    return retrieval_eval.postgres_evaluation_retriever_factory(
        pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        embedding_profile=PROFILE,
        embedding_client_factory=None,
        lexical_candidate_limit=1,
        vector_candidate_limit=1,
        rrf_k=60,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_type", [asyncio.CancelledError, GeneratorExit])
async def test_snapshot_rolls_back_and_releases_for_boundary_signals(signal_type):
    pool = _FakeSnapshotPool()
    factory = _fake_snapshot_factory(pool)

    with pytest.raises(signal_type) as raised:
        async with factory.evaluation_session():
            raise signal_type("private")

    assert raised.value.args == ()
    assert pool.events == [
        "acquire",
        "transaction",
        "start",
        "rollback",
        "release",
    ]


class _UnknownCleanupFailure(BaseException):
    pass


def _cleanup_failure(kind):
    if kind == "ordinary":
        return RuntimeError("private cleanup dsn=postgres://secret")
    if kind == "keyboard":
        return KeyboardInterrupt("private cleanup")
    if kind == "system_exit":
        return SystemExit("private cleanup")
    if kind == "cancel":
        return asyncio.CancelledError("private cleanup")
    if kind == "generator_exit":
        return GeneratorExit("private cleanup")
    if kind == "unknown_group":
        return BaseExceptionGroup(
            "private cleanup group",
            [RuntimeError("private ordinary"), _UnknownCleanupFailure("private unknown")],
        )
    raise AssertionError(f"unsupported cleanup failure: {kind}")


def _expected_cleanup_failure(kind):
    return {
        "ordinary": retrieval_eval.RetrievalExecutionError,
        "keyboard": KeyboardInterrupt,
        "system_exit": SystemExit,
        "cancel": asyncio.CancelledError,
        "generator_exit": GeneratorExit,
        "unknown_group": BaseException,
    }[kind]


class _CleanupFailurePlan:
    def __init__(self, location, kind):
        self.location = location
        self.kind = kind
        self.events = []
        self.leased = False

    def getter(self, name, callback):
        location = f"{name}:get"
        self.events.append(location)
        if self.location == location:
            raise _cleanup_failure(self.kind)
        return callback

    async def call(self, name):
        location = f"{name}:call"
        self.events.append(location)
        if self.location == location:
            raise _cleanup_failure(self.kind)


class _CleanupFailureTransaction:
    def __init__(self, plan):
        self._plan = plan

    async def start(self):
        self._plan.events.append("start")

    @property
    def rollback(self):
        async def call():
            await self._plan.call("rollback")

        return self._plan.getter("rollback", call)


class _CleanupFailureConnection:
    def __init__(self, plan):
        self._plan = plan

    async def set_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not set connection codecs")

    @property
    def reset_type_codec(self):
        raise AssertionError("evaluation must not reset connection codecs")

    def transaction(self, **options):
        assert options == {"isolation": "repeatable_read", "readonly": True}
        self._plan.events.append("transaction")
        return _CleanupFailureTransaction(self._plan)


class _CleanupFailureLease:
    def __init__(self, plan):
        self._plan = plan
        self._connection = _CleanupFailureConnection(plan)

    async def __aenter__(self):
        self._plan.events.append("acquire")
        self._plan.leased = True
        return self._connection

    @property
    def __aexit__(self):
        async def call(*_args):
            await self._plan.call("release")
            self._plan.leased = False

        return self._plan.getter("release", call)


class _CleanupFailurePool:
    def __init__(self, plan):
        self._plan = plan

    def acquire(self):
        return _CleanupFailureLease(self._plan)

    @property
    def release(self):
        async def call(_connection):
            await self._plan.call("pool:release")
            self._plan.leased = False

        return self._plan.getter("pool:release", call)


_CLEANUP_LOCATIONS = [
    f"{operation}:{boundary}"
    for operation in ("rollback", "release")
    for boundary in ("get", "call")
]
_CLEANUP_FAILURE_KINDS = [
    "ordinary",
    "keyboard",
    "system_exit",
    "cancel",
    "generator_exit",
    "unknown_group",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("location", _CLEANUP_LOCATIONS)
@pytest.mark.parametrize("kind", _CLEANUP_FAILURE_KINDS)
async def test_snapshot_cleanup_guards_getters_and_calls_without_skipping_later_steps(
    location,
    kind,
):
    plan = _CleanupFailurePlan(location, kind)
    factory = _fake_snapshot_factory(_CleanupFailurePool(plan))

    with pytest.raises(_expected_cleanup_failure(kind)) as raised:
        async with factory.evaluation_session():
            assert factory._session_active is True

    expected = ["acquire", "transaction", "start", "rollback:get"]
    if location != "rollback:get":
        expected.append("rollback:call")
    expected.append("release:get")
    if location != "release:get":
        expected.append("release:call")
    if location in {"release:get", "release:call"}:
        expected.extend(["pool:release:get", "pool:release:call"])
    assert plan.events == expected
    assert plan.leased is False
    assert raised.value.args == ((1,) if kind == "system_exit" else ())
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert "private" not in str(raised.value)
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
async def test_cleanup_ordinary_failure_does_not_mask_primary_business_failure():
    plan = _CleanupFailurePlan("rollback:call", "ordinary")
    factory = _fake_snapshot_factory(_CleanupFailurePool(plan))

    with pytest.raises(retrieval_eval.RetrievalContractError) as raised:
        async with factory.evaluation_session():
            raise retrieval_eval.RetrievalContractError

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert plan.events == [
        "acquire",
        "transaction",
        "start",
        "rollback:get",
        "rollback:call",
        "release:get",
        "release:call",
    ]
    assert plan.leased is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["keyboard", "system_exit", "cancel", "generator_exit"])
async def test_cleanup_control_has_priority_over_primary_ordinary_failure(kind):
    plan = _CleanupFailurePlan("rollback:call", kind)
    factory = _fake_snapshot_factory(_CleanupFailurePool(plan))

    with pytest.raises(_expected_cleanup_failure(kind)) as raised:
        async with factory.evaluation_session():
            raise RuntimeError("private business failure")

    assert raised.value.args == ((1,) if kind == "system_exit" else ())
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert plan.events[-1] == "release:call"
    assert plan.leased is False


class _MultipleCleanupFailurePlan(_CleanupFailurePlan):
    def __init__(self):
        super().__init__(None, None)
        self.failures = {
            "rollback:call": "cancel",
            "release:call": "keyboard",
        }

    async def call(self, name):
        location = f"{name}:call"
        self.events.append(location)
        if kind := self.failures.get(location):
            raise _cleanup_failure(kind)


@pytest.mark.asyncio
async def test_cleanup_collects_all_failures_before_selecting_highest_priority_control():
    plan = _MultipleCleanupFailurePlan()
    factory = _fake_snapshot_factory(_CleanupFailurePool(plan))

    with pytest.raises(KeyboardInterrupt) as raised:
        async with factory.evaluation_session():
            pass

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert plan.events[-6:] == [
        "rollback:get",
        "rollback:call",
        "release:get",
        "release:call",
        "pool:release:get",
        "pool:release:call",
    ]
    assert plan.leased is False


class _ExitFallbackPlan:
    def __init__(self, mode):
        self.mode = mode
        self.events = []
        self.leased = False
        self.pool_release_attempts = 0


class _ExitFallbackTransaction:
    def __init__(self, plan):
        self._plan = plan

    async def start(self):
        self._plan.events.append("start")

    async def rollback(self):
        self._plan.events.append("rollback")


class _ExitFallbackConnection:
    def __init__(self, plan):
        self._plan = plan

    async def set_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not set connection codecs")

    async def reset_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not reset connection codecs")

    def transaction(self, **_options):
        self._plan.events.append("transaction")
        return _ExitFallbackTransaction(self._plan)


class _ExitFallbackLease:
    def __init__(self, plan):
        self._plan = plan
        self.connection = _ExitFallbackConnection(plan)

    async def __aenter__(self):
        self._plan.events.append("acquire")
        self._plan.leased = True
        return self.connection

    @property
    def __aexit__(self):
        self._plan.events.append("lease:exit:get")
        if self._plan.mode == "getter_raises":
            raise RuntimeError("private lease exit getter")
        if self._plan.mode == "noncallable":
            return object()

        async def call(*_args):
            self._plan.events.append("lease:exit:call")
            if self._plan.mode == "call_before_release":
                raise RuntimeError("private lease exit before release")
            self._plan.leased = False
            if self._plan.mode == "call_after_release":
                raise RuntimeError("private lease exit after release")

        return call


class _ExitFallbackPool:
    def __init__(self, mode):
        self.plan = _ExitFallbackPlan(mode)
        self.lease = _ExitFallbackLease(self.plan)

    def acquire(self):
        return self.lease

    async def release(self, connection):
        assert connection is self.lease.connection
        self.plan.events.append("pool:release:call")
        self.plan.pool_release_attempts += 1
        if not self.plan.leased:
            raise RuntimeError("private duplicate pool release")
        self.plan.leased = False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["getter_raises", "noncallable", "call_before_release", "call_after_release"],
)
async def test_failed_or_missing_lease_exit_falls_back_to_pool_release(mode):
    pool = _ExitFallbackPool(mode)
    factory = _fake_snapshot_factory(pool)

    if mode == "noncallable":
        async with factory.evaluation_session():
            pass
    else:
        with pytest.raises(retrieval_eval.RetrievalExecutionError) as raised:
            async with factory.evaluation_session():
                pass
        assert raised.value.args == ()
        assert raised.value.__cause__ is raised.value.__context__ is None

    assert pool.plan.pool_release_attempts == 1
    assert pool.plan.leased is False
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


class _PoolReleaseFailureLease:
    def __init__(self, plan):
        self._plan = plan
        self.connection = _CleanupFailureConnection(plan)

    async def __aenter__(self):
        self._plan.events.append("acquire")
        self._plan.leased = True
        return self.connection


class _PoolReleaseFailurePool(_CleanupFailurePool):
    def acquire(self):
        return _PoolReleaseFailureLease(self._plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["get", "call"])
@pytest.mark.parametrize("kind", _CLEANUP_FAILURE_KINDS)
async def test_pool_release_failures_are_inventoried_and_session_state_is_cleared(
    boundary,
    kind,
):
    plan = _CleanupFailurePlan(f"pool:release:{boundary}", kind)
    factory = _fake_snapshot_factory(_PoolReleaseFailurePool(plan))

    with pytest.raises(_expected_cleanup_failure(kind)) as raised:
        async with factory.evaluation_session():
            pass

    assert "pool:release:get" in plan.events
    if boundary == "call":
        assert "pool:release:call" in plan.events
    assert raised.value.args == ((1,) if kind == "system_exit" else ())
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert "private" not in str(raised.value)
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
async def test_pool_release_ordinary_failure_does_not_mask_primary_business_failure():
    plan = _CleanupFailurePlan("pool:release:call", "ordinary")
    factory = _fake_snapshot_factory(_PoolReleaseFailurePool(plan))

    with pytest.raises(retrieval_eval.RetrievalContractError) as raised:
        async with factory.evaluation_session():
            raise retrieval_eval.RetrievalContractError

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert plan.events[-2:] == ["pool:release:get", "pool:release:call"]
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
async def test_pool_release_control_failure_has_priority_over_primary_ordinary_failure():
    plan = _CleanupFailurePlan("pool:release:call", "keyboard")
    factory = _fake_snapshot_factory(_PoolReleaseFailurePool(plan))

    with pytest.raises(KeyboardInterrupt) as raised:
        async with factory.evaluation_session():
            raise RuntimeError("private primary")

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


class _ObligationPlan:
    def __init__(
        self,
        *,
        protocol="lease",
        lease_exit="success",
        lease_exit_failure_kind=None,
        pool_release="success",
        rollback="success",
        partial_setup=None,
    ):
        self.protocol = protocol
        self.lease_exit = lease_exit
        self.lease_exit_failure_kind = lease_exit_failure_kind
        self.pool_release = pool_release
        self.rollback = rollback
        self.partial_setup = partial_setup
        self.events = []
        self.leased = False
        self.transaction_active = False


def _obligation_operation(plan, name, mode, call):
    plan.events.append(f"{name}:get")
    if mode == "missing":
        return None
    if mode == "noncallable":
        return object()
    return call


class _ObligationTransaction:
    def __init__(self, plan):
        self._plan = plan

    async def start(self):
        self._plan.events.append("start")
        self._plan.transaction_active = True
        if self._plan.partial_setup == "transaction":
            raise RuntimeError("private partial transaction start")

    @property
    def rollback(self):
        async def call():
            self._plan.events.append("rollback:call")
            self._plan.transaction_active = False

        return _obligation_operation(self._plan, "rollback", self._plan.rollback, call)


class _ObligationConnection:
    def __init__(self, plan):
        self._plan = plan

    async def set_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not set connection codecs")

    @property
    def reset_type_codec(self):
        raise AssertionError("evaluation must not reset connection codecs")

    def transaction(self, **_options):
        self._plan.events.append("transaction")
        return _ObligationTransaction(self._plan)


class _ObligationLease:
    def __init__(self, plan, connection):
        self._plan = plan
        self._connection = connection

    async def __aenter__(self):
        self._plan.events.append("acquire")
        self._plan.leased = True
        return self._connection

    @property
    def __aexit__(self):
        async def call(*_args):
            self._plan.events.append("lease:exit:call")
            if self._plan.lease_exit_failure_kind is not None:
                raise _cleanup_failure(self._plan.lease_exit_failure_kind)
            self._plan.leased = False

        return _obligation_operation(
            self._plan,
            "lease:exit",
            self._plan.lease_exit,
            call,
        )


class _ObligationPool:
    def __init__(self, plan):
        self.plan = plan
        self.connection = _ObligationConnection(plan)
        self.lease = _ObligationLease(plan, self.connection)

    def acquire(self):
        if self.plan.protocol == "direct":
            self.plan.events.append("acquire")
            self.plan.leased = True
            return self.connection
        return self.lease

    @property
    def release(self):
        async def call(connection):
            assert connection is self.connection
            self.plan.events.append("pool:release:call")
            self.plan.leased = False

        return _obligation_operation(
            self.plan,
            "pool:release",
            self.plan.pool_release,
            call,
        )


def test_cleanup_obligation_validation_records_each_missing_proof():
    obligations = retrieval_eval._CleanupObligations(  # noqa: SLF001
        release=True,
        rollback=True,
    )
    proofs = retrieval_eval._CleanupProofs()  # noqa: SLF001
    failures = []

    retrieval_eval._validate_cleanup_obligations(  # noqa: SLF001
        obligations,
        proofs,
        failures,
    )

    assert proofs == retrieval_eval._CleanupProofs()  # noqa: SLF001
    assert [type(failure) for failure in failures] == [
        retrieval_eval.RetrievalExecutionError,
        retrieval_eval.RetrievalExecutionError,
    ]
    assert all(failure.args == () for failure in failures)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "lease_exit"),
    [("lease", "missing"), ("lease", "noncallable"), ("direct", "success")],
)
async def test_missing_all_release_paths_fails_the_release_obligation(
    protocol,
    lease_exit,
):
    plan = _ObligationPlan(
        protocol=protocol,
        lease_exit=lease_exit,
        pool_release="missing",
    )
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    with pytest.raises(retrieval_eval.RetrievalExecutionError) as raised:
        async with factory.evaluation_session():
            pass

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert "pool:release:get" in plan.events
    assert plan.leased is True
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", ["missing", "noncallable"])
async def test_started_transaction_requires_completed_rollback_proof(rollback):
    plan = _ObligationPlan(rollback=rollback)
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    with pytest.raises(retrieval_eval.RetrievalExecutionError) as raised:
        async with factory.evaluation_session():
            pass

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert "rollback:get" in plan.events
    assert "rollback:call" not in plan.events
    assert plan.transaction_active is True
    assert plan.leased is False


@pytest.mark.asyncio
async def test_partially_started_transaction_still_establishes_rollback_obligation():
    plan = _ObligationPlan(partial_setup="transaction")
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    with pytest.raises(retrieval_eval.RetrievalExecutionError) as raised:
        async with factory.evaluation_session():
            raise AssertionError("partial setup must not yield")

    assert raised.value.args == ()
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert plan.transaction_active is False
    assert plan.leased is False
    assert "rollback:call" in plan.events


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _CLEANUP_FAILURE_KINDS)
async def test_missing_obligation_combines_with_cleanup_failure_signal(kind):
    plan = _ObligationPlan(
        rollback="missing",
        lease_exit_failure_kind=kind,
    )
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    with pytest.raises(_expected_cleanup_failure(kind)) as raised:
        async with factory.evaluation_session():
            pass

    assert raised.value.args == ((1,) if kind == "system_exit" else ())
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert "private" not in str(raised.value)
    assert plan.transaction_active is True
    assert plan.leased is False
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
async def test_successful_lease_exit_is_a_release_proof_without_pool_release():
    plan = _ObligationPlan(pool_release="missing")
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    async with factory.evaluation_session():
        pass

    assert "lease:exit:call" in plan.events
    assert "pool:release:get" not in plan.events
    assert plan.leased is plan.transaction_active is False


@pytest.mark.asyncio
async def test_pool_release_proof_remedies_missing_lease_exit_only():
    plan = _ObligationPlan(lease_exit="missing")
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    async with factory.evaluation_session():
        pass

    assert plan.events[-3:] == [
        "lease:exit:get",
        "pool:release:get",
        "pool:release:call",
    ]
    assert plan.leased is plan.transaction_active is False


@pytest.mark.asyncio
async def test_release_proof_cannot_remedy_an_unproved_rollback():
    plan = _ObligationPlan(rollback="missing")
    factory = _fake_snapshot_factory(_ObligationPool(plan))

    with pytest.raises(retrieval_eval.RetrievalExecutionError):
        async with factory.evaluation_session():
            pass

    assert "lease:exit:call" in plan.events
    assert plan.leased is False


class _SetupBarrierPlan:
    def __init__(self, blocked_stage=None, *, fail_after_release=False):
        self.blocked_stage = blocked_stage
        self.fail_after_release = fail_after_release
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.events = []
        self.acquire_count = 0
        self.failures_remaining = 1 if fail_after_release else 0
        self.blocked_claimed = False

    async def step(self, stage):
        self.events.append(stage)
        if (
            stage == self.blocked_stage
            and self.failures_remaining
            and not self.blocked_claimed
        ):
            self.blocked_claimed = True
            self.started.set()
            await self.release.wait()
            self.failures_remaining -= 1
            raise RuntimeError(f"private {stage} setup failure")


class _SetupBarrierTransaction:
    def __init__(self, plan):
        self._plan = plan

    async def start(self):
        await self._plan.step("start")

    async def rollback(self):
        self._plan.events.append("rollback")


class _SetupBarrierConnection:
    def __init__(self, plan):
        self._plan = plan
        self.fetch_count = 0

    async def set_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not set connection codecs")

    async def reset_type_codec(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not reset connection codecs")

    def transaction(self, **_options):
        self._plan.events.append("transaction")
        return _SetupBarrierTransaction(self._plan)

    async def fetch(self, *_args):
        self.fetch_count += 1
        return ()


class _SetupBarrierLease:
    def __init__(self, plan):
        self._plan = plan
        self.connection = _SetupBarrierConnection(plan)

    async def __aenter__(self):
        self._plan.acquire_count += 1
        await self._plan.step("acquire")
        return self.connection

    async def __aexit__(self, *_args):
        self._plan.events.append("release")


class _SetupBarrierPool:
    def __init__(self, plan):
        self._plan = plan
        self.leases = []

    def acquire(self):
        lease = _SetupBarrierLease(self._plan)
        self.leases.append(lease)
        return lease


class _DirectAcquirePool:
    def __init__(self):
        self.events = []
        self.connection = _SetupBarrierConnection(_SetupBarrierPlan())

    def acquire(self):
        async def direct():
            self.events.append("acquire")
            return self.connection

        return direct()

    async def release(self, connection):
        assert connection is self.connection
        self.events.append("release")


class _NoExitLease:
    def __init__(self, pool):
        self._pool = pool

    async def __aenter__(self):
        self._pool.events.append("acquire")
        return self._pool.connection


class _NoExitLeasePool(_DirectAcquirePool):
    def acquire(self):
        return _NoExitLease(self)


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_type", [_DirectAcquirePool, _NoExitLeasePool])
async def test_snapshot_supports_direct_acquire_and_missing_lease_exit(pool_type):
    pool = pool_type()
    factory = _fake_snapshot_factory(pool)

    async with factory.evaluation_session():
        assert factory._session_active is True
        assert factory._snapshot is not None

    assert pool.events == ["acquire", "release"]
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}


@pytest.mark.asyncio
async def test_concurrent_and_nested_sessions_reject_without_acquiring_or_clearing_owner():
    plan = _SetupBarrierPlan()
    pool = _SetupBarrierPool(plan)
    factory = _fake_snapshot_factory(pool)
    owner_started = asyncio.Event()
    owner_release = asyncio.Event()

    async def owner():
        async with factory.evaluation_session():
            owner_started.set()
            await owner_release.wait()

    owner_task = asyncio.create_task(owner())
    await owner_started.wait()
    owner_snapshot = factory._snapshot

    with pytest.raises(retrieval_eval.RetrievalContractError) as concurrent:
        async with factory.evaluation_session():
            raise AssertionError("concurrent session must not enter")
    with pytest.raises(retrieval_eval.RetrievalContractError) as nested:
        async with factory.evaluation_session():
            raise AssertionError("nested session must not enter")

    assert concurrent.value.args == nested.value.args == ()
    assert concurrent.value.__cause__ is concurrent.value.__context__ is None
    assert nested.value.__cause__ is nested.value.__context__ is None
    assert plan.acquire_count == 1
    assert factory._session_active is True
    assert factory._snapshot is owner_snapshot
    owner_release.set()
    await owner_task
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}

    async with factory.evaluation_session():
        assert factory._snapshot is not None
    assert plan.acquire_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["acquire", "start"])
async def test_setup_failure_keeps_concurrent_session_out_and_allows_next_run(blocked_stage):
    plan = _SetupBarrierPlan(blocked_stage, fail_after_release=True)
    pool = _SetupBarrierPool(plan)
    factory = _fake_snapshot_factory(pool)

    async def first():
        async with factory.evaluation_session():
            raise AssertionError("failing setup must not yield")

    first_task = asyncio.create_task(first())
    await plan.started.wait()
    with pytest.raises(retrieval_eval.RetrievalContractError) as concurrent:
        async with factory.evaluation_session():
            raise AssertionError("concurrent setup must not enter")
    assert concurrent.value.args == ()
    assert plan.acquire_count == 1
    assert factory._session_active is True

    plan.release.set()
    with pytest.raises(retrieval_eval.RetrievalExecutionError) as failed:
        await first_task
    assert failed.value.args == ()
    assert failed.value.__cause__ is failed.value.__context__ is None
    assert factory._session_active is False
    assert factory._snapshot is None
    assert factory._coverage_cache == {}

    async with factory.evaluation_session():
        assert factory._snapshot is not None
    assert plan.acquire_count == 2


@pytest.mark.asyncio
async def test_retriever_snapshot_is_revoked_after_session_exit():
    plan = _SetupBarrierPlan()
    pool = _SetupBarrierPool(plan)
    factory = _fake_snapshot_factory(pool)

    async with factory.evaluation_session():
        retriever = factory("lexical", Path("unused"))
        snapshot = factory._snapshot
        assert snapshot is not None

    with pytest.raises(retrieval_eval.RetrieverUnavailable) as raised:
        await retriever.retrieve(
            retrieval_eval.SearchQuery(text="outside session", limit=1, candidate_limit=1)
        )

    assert type(raised.value).__name__ == "RetrieverUnavailable"
    assert raised.value.args == ("lexical evaluation store is unavailable",)
    assert raised.value.__cause__ is raised.value.__context__ is None
    assert pool.leases[0].connection.fetch_count == 0
