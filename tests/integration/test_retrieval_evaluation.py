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
        return await self._connection.set_type_codec(*args, **kwargs)

    async def reset_type_codec(self, *args, **kwargs):
        return await self._connection.reset_type_codec(*args, **kwargs)

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
        "metadata": {"stage": "S2"},
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
        ("content", "x" * 1_000_001),
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
        ("metadata", "{}"),
        ("metadata", []),
        ("metadata", {"nested": nan}),
        ("metadata", {"x" * 1025: "value"}),
    ],
)
def test_evaluation_backend_rows_reject_implicit_coercions(profile, field, invalid):
    validator = getattr(retrieval_eval, "_validated_evaluation_rows", None)
    assert callable(validator), "strict evaluation row validator is not implemented"
    row = _valid_backend_row()
    row[field] = invalid

    with pytest.raises(Exception) as raised:
        validator((row,), profile=profile)

    assert type(raised.value).__name__ == "RetrieverUnavailable"
    assert raised.value.args == ("evaluation row is invalid",)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


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
        assert name == "jsonb"
        assert options["schema"] == "pg_catalog"
        self._events.append("codec:set")

    async def reset_type_codec(self, name, **options):
        assert name == "jsonb"
        assert options["schema"] == "pg_catalog"
        self._events.append("codec:reset")

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
        "codec:set",
        "transaction",
        "start",
        "rollback",
        "codec:reset",
        "release",
    ]
