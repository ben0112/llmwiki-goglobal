import json
import os
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
PROFILE = EmbeddingProfile("openai_compatible", "evaluation-v1", 3)
OTHER_PROFILE = EmbeddingProfile("openai_compatible", "evaluation-other", 3)
EXPECTED_DATASET_DIGEST = "d45bf89f5b28694afe2b4af1d03d15e3ba59e02d3bc20129eff77b58ab39ab7f"


class _RecordingPool:
    def __init__(self):
        self.fetches = []
        self.returned_document_ids = []

    async def fetch(self, sql, *params):
        self.fetches.append((sql, params))
        connection = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            rows = await connection.fetch(sql, *params)
            self.returned_document_ids.extend(
                str(row["document_id"])
                for row in rows
                if "document_id" in row
            )
            return rows
        finally:
            await connection.close()


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
        "($5,$2,'Same User Other','same-user-other')",
        KNOWLEDGE_BASE_ID,
        USER_ID,
        OTHER_KB_ID,
        OTHER_TENANT_ID,
        SAME_USER_OTHER_KB_ID,
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
    await PostgresVectorStore(pool, profile=OTHER_PROFILE).replace_document_embeddings(
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=OTHER_PROFILE_DOCUMENT_ID,
        document_version=1,
        embeddings=((0, (0.0, 1.0, 0.0)),),
    )
    fixture_root = tmp_path_factory.mktemp("retrieval-evaluation")
    return SimpleNamespace(dataset=_dataset(fixture_root / "cases.jsonl"), pool=pool)


def _factory(corpus, *, hybrid_latency_ms, profile=PROFILE):
    factory_builder = getattr(retrieval_eval, "postgres_evaluation_retriever_factory", None)
    assert callable(factory_builder), "Postgres evaluation adapter is not implemented"
    calls = []
    recording_pool = _RecordingPool()
    factory = factory_builder(
        recording_pool,
        user_id=USER_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        embedding_profile=profile,
        embedding_client_factory=lambda: _FakeQueryEmbeddingClient(calls),
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
