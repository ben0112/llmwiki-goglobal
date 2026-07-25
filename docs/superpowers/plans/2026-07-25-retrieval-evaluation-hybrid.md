# Retrieval Evaluation and Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a deterministic lexical-search baseline, enforce all filters before retrieval limits, and add opt-in pgvector hybrid retrieval that can be promoted only by measured quality and latency gates.

**Architecture:** Extend the backend-neutral search contract in `llmwiki_core`, keep database-specific SQL in the existing SQLite and Postgres adapters, and make MCP's current search surface a compatibility facade over a shared retrieval service. Hosted embeddings are version-fenced durable jobs stored in pgvector; lexical search stays the default and remains the complete fallback whenever hybrid search is disabled, misconfigured, or unavailable.

**Tech Stack:** Python 3.11, dataclasses/protocols, pytest, SQLite FTS5 trigram, Postgres 16, PGroonga, pgvector, asyncpg, aiosqlite, httpx, durable ARQ/Postgres jobs, MCP FastMCP.

---

## File and responsibility map

- `llmwiki_core/search.py`: immutable search query/result contracts, retriever protocols, reciprocal-rank fusion, deduplication, and optional reranker/context-expander orchestration.
- `llmwiki_core/evaluation.py`: versioned JSONL case validation, exact IR metrics, latency summaries, and lexical-versus-hybrid promotion decisions.
- `llmwiki_core/models.py`: provider-neutral embedding profile and embedding client protocol.
- `mcp/vaultfs/sqlite.py`, `mcp/vaultfs/postgres.py`: backend SQL with every filter applied before `LIMIT`; legacy `search_chunks()` remains compatible.
- `mcp/services/retrieval.py`: adapters from `VaultFS` and hosted vector SQL to the core retrieval service, including bounded one-hop graph expansion.
- `mcp/tools/search.py`: compatible MCP surface with one new optional retrieval profile.
- `api/services/embeddings.py`: OpenAI-compatible embeddings HTTP adapter and validated batching.
- `api/services/vector_store.py`: tenant-scoped, document-version-fenced pgvector writes and candidate reads.
- `api/jobs/handlers.py`: durable `document.embed` handler.
- `api/scripts/retrieval_eval.py`: selected-retriever evaluation CLI and promotion-gate exit status.
- `supabase/migrations/013_chunk_embeddings.sql`: published Task 7 pgvector table, indexes, and RLS.
- `supabase/migrations/014_document_embedding_jobs.sql`: additive embedding job type and reconciliation index.
- `tests/fixtures/retrieval/v1/`: synthetic, non-user evaluation cases and fixture corpus.

### Task 1: Expand the shared search contract without breaking callers

**Files:**
- Modify: `llmwiki_core/search.py`
- Modify: `llmwiki_core/__init__.py`
- Modify: `tests/unit/core/test_search.py`

- [ ] **Step 1: Write failing contract tests**

Add tests proving normalized path globs/tags/kinds, bounded candidate counts, immutable results, stable hit identity, and backward-compatible construction:

```python
def test_search_query_normalizes_filters_and_candidate_limit():
    query = SearchQuery.build(
        text="  export controls  ",
        limit=10,
        candidate_limit=40,
        path_glob="corpus/**/*.md",
        tags=["Policy", " policy ", "ASEAN"],
        document_kinds=["source", "wiki"],
        annotated_only=True,
    )
    assert query.text == "export controls"
    assert query.path_glob == "/corpus/**/*.md"
    assert query.tags == ("asean", "policy")
    assert query.document_kinds == (DocumentKind.SOURCE, DocumentKind.WIKI)
    assert query.candidate_limit == 40


def test_search_result_distinguishes_candidates_from_returned_hits():
    result = SearchResult(hits=(make_hit("d1", 0),), candidate_count=17)
    assert result.returned_count == 1
    assert result.candidate_count == 17
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/unit/core/test_search.py -q`

Expected: failures because the new query fields and `SearchResult` do not exist.

- [ ] **Step 3: Implement immutable contracts and ports**

Keep existing `SearchQuery.build(text=..., limit=..., area=..., scope=..., facets=...)` valid. Add normalized optional fields and these interfaces:

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    candidate_count: int
    latency_ms: float = 0.0
    profile: str = "lexical"

    @property
    def returned_count(self) -> int:
        return len(self.hits)


class Retriever(Protocol):
    async def retrieve(self, query: SearchQuery) -> SearchResult: ...


class Reranker(Protocol):
    async def rerank(self, query: SearchQuery, hits: Sequence[SearchHit]) -> Sequence[SearchHit]: ...


class ContextExpander(Protocol):
    async def expand(self, query: SearchQuery, hits: Sequence[SearchHit]) -> Sequence[SearchHit]: ...
```

Reject `candidate_limit < limit`, `candidate_limit > 500`, unsupported kinds, empty tags, and path globs containing NUL. Extend `SearchHit` only with defaulted metadata fields so current positional/keyword users remain valid.

- [ ] **Step 4: Re-run focused tests**

Run: `.venv/bin/pytest tests/unit/core/test_search.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add llmwiki_core/search.py llmwiki_core/__init__.py tests/unit/core/test_search.py
git commit -m "feat: expand shared retrieval contracts"
git push origin feat/platform-architecture-evolution
```

### Task 2: Add deterministic rank fusion and bounded post-processing

**Files:**
- Modify: `llmwiki_core/search.py`
- Modify: `tests/unit/core/test_search.py`

- [ ] **Step 1: Write RED tests for RRF, deduplication, reranking, and expansion**

Cover deterministic tie-breaking by hit identity, duplicate hits from both retrievers, bounded expansion, a reranker returning duplicate/unknown hits, and one retriever failing:

```python
async def test_hybrid_fuses_by_rrf_and_deduplicates_identity():
    service = HybridRetrievalService(
        lexical=FakeRetriever([hit("a"), hit("b")]),
        vector=FakeRetriever([hit("b"), hit("c")]),
        rrf_k=60,
    )
    result = await service.retrieve(SearchQuery.build(text="q", limit=3, candidate_limit=10))
    assert [item.document_id for item in result.hits] == ["b", "a", "c"]


async def test_hybrid_falls_back_to_lexical_when_vector_is_unavailable():
    service = HybridRetrievalService(
        lexical=FakeRetriever([hit("a")]),
        vector=FailingRetriever(),
    )
    result = await service.retrieve(SearchQuery.build(text="q"))
    assert result.profile == "lexical_fallback"
    assert [item.document_id for item in result.hits] == ["a"]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/unit/core/test_search.py -q`

Expected: failures because fusion and orchestration are absent.

- [ ] **Step 3: Implement the pure service**

Run lexical and vector retrieval independently, fuse with `1 / (rrf_k + rank)`, sort by fused score then stable identity, apply an injected reranker only to known unique identities, apply at most `query.limit` direct/expanded unique hits, and retain the sum of backend candidate counts. Catch only a typed `RetrieverUnavailable` from the vector side; propagate lexical failures and programming errors.

- [ ] **Step 4: Verify green and lint**

Run: `.venv/bin/pytest tests/unit/core/test_search.py -q && .venv/bin/ruff check llmwiki_core/search.py tests/unit/core/test_search.py`

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit**

```bash
git add llmwiki_core/search.py tests/unit/core/test_search.py
git commit -m "feat: add deterministic hybrid rank fusion"
git push origin feat/platform-architecture-evolution
```

### Task 3: Define the versioned evaluation format and exact metrics

**Files:**
- Create: `llmwiki_core/evaluation.py`
- Create: `tests/unit/core/test_evaluation.py`
- Create: `tests/fixtures/retrieval/v1/cases.jsonl`
- Create: `tests/fixtures/retrieval/v1/corpus.jsonl`
- Modify: `llmwiki_core/__init__.py`

- [ ] **Step 1: Add a synthetic JSONL fixture and failing parser tests**

Every case line uses this shape; fixture identifiers are synthetic and contain no user content:

```json
{"schema_version":1,"case_id":"export-idn","query":{"text":"Indonesia export permit","facets":{"country":"IDN"}},"relevance":[{"document_id":"fixture-policy-idn","chunk_index":0,"grade":3},{"document_id":"fixture-checklist-idn","chunk_index":1,"grade":1}]}
```

Test duplicate case ids, unsupported versions, missing relevance, invalid grades, duplicate relevance identities, and query validation.

- [ ] **Step 2: Write failing exact-metric tests**

Use hand-calculated rankings and assert `Recall@5`, `Recall@10`, `Recall@20`, MRR, nDCG@10, filtered result counts, p50, and p95. Define recall against all positive-grade relevant identities and DCG as `(2**grade - 1) / log2(rank + 1)`.

```python
def test_metrics_match_hand_calculated_ranking():
    report = evaluate_rankings(cases, runs)
    assert report.recall_at_5 == pytest.approx(0.75)
    assert report.mrr == pytest.approx(0.75)
    assert report.filtered_result_count == 3
```

- [ ] **Step 3: Run tests and verify RED**

Run: `.venv/bin/pytest tests/unit/core/test_evaluation.py -q`

Expected: import failure for the missing evaluation module.

- [ ] **Step 4: Implement parser, report, and promotion decision**

Implement `load_cases(path)`, `evaluate_rankings(cases, runs)`, and:

```python
def promotion_decision(lexical: EvaluationReport, hybrid: EvaluationReport) -> PromotionDecision:
    if lexical.recall_at_10 <= 0:
        return PromotionDecision(False, "baseline_recall_zero")
    quality_ratio = hybrid.recall_at_10 / lexical.recall_at_10
    latency_ratio = hybrid.latency_p95_ms / max(lexical.latency_p95_ms, 0.001)
    return PromotionDecision(
        eligible=quality_ratio >= 1.10 and latency_ratio <= 2.0,
        reason="eligible" if quality_ratio >= 1.10 and latency_ratio <= 2.0 else "gate_failed",
        recall_ratio=quality_ratio,
        latency_ratio=latency_ratio,
    )
```

Use nearest-rank percentiles so fixture expectations are platform-independent.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/pytest tests/unit/core/test_evaluation.py tests/unit/core/test_search.py -q`

Expected: all tests pass.

```bash
git add llmwiki_core/evaluation.py llmwiki_core/__init__.py tests/unit/core/test_evaluation.py tests/fixtures/retrieval/v1
git commit -m "feat: add deterministic retrieval evaluation"
git push origin feat/platform-architecture-evolution
```

### Task 4: Push every lexical filter before the limit

**Files:**
- Modify: `mcp/vaultfs/base.py`
- Modify: `mcp/vaultfs/sqlite.py`
- Modify: `mcp/vaultfs/postgres.py`
- Modify: `mcp/tools/search.py`
- Modify: `tests/integration/mcp/test_vaultfs_contract.py`
- Modify: `tests/integration/mcp/test_corpus_facets.py`
- Modify: `tests/integration/mcp/test_mcp_isolation.py`
- Modify: `tests/integration/mcp/test_tool_handlers.py`

- [ ] **Step 1: Write adversarial RED tests**

Seed more high-scoring excluded rows than the requested limit, followed by eligible rows. Cover path glob, tags, document kind/area, `annotated_only`, scope, and facets independently and in combination. Assert `limit=2` still returns two eligible hits and candidate/returned counts are accurate.

```python
async def test_filters_are_applied_before_limit(fs, seeded_ranked_chunks):
    request = SearchQuery.build(
        text="permit",
        limit=2,
        candidate_limit=4,
        path_glob="corpus/idn/*.md",
        tags=["reviewed"],
        document_kinds=["source"],
        scope="source",
        facets={"country": "IDN"},
    )
    result = await fs.retrieve(request, kb_id=KB_ID)
    assert result.returned_count == 2
    assert all(hit.path.startswith("/corpus/idn/") for hit in result.hits)
```

- [ ] **Step 2: Verify RED on SQLite and Postgres**

Run:

```bash
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_vaultfs_contract.py tests/integration/mcp/test_corpus_facets.py -q
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_mcp_isolation.py -q
```

Expected: post-limit path/tag filtering underfills or the new `retrieve()` contract is absent.

- [ ] **Step 3: Implement backend-native filter pushdown**

Add `VaultFS.retrieve(kb_id, SearchQuery) -> SearchResult`; leave legacy `search_chunks()` delegating to it and converting hits to current dictionaries. In both backends, add all fixed SQL predicates before `ORDER BY ... LIMIT`:

- area/document kind via `source_kind`;
- exact tag containment (`json_each` in SQLite, `tags @> $n::jsonb` in Postgres);
- normalized logical glob translated to escaped SQL `LIKE` with only `*` and `**` treated as wildcards;
- annotated/source scope against the appropriate chunk column;
- validated facet predicates from the existing shared facet compiler.

Use a count CTE over the same filtered candidate query and return the count separately. Do not use Python filtering or over-fetch multipliers for correctness.

- [ ] **Step 4: Remove MCP post-limit path/tag filtering**

Build the complete `SearchQuery` in `SearchHandler.search_chunks()` and pass it once to `fs.retrieve()`. Keep formatting and corpus folding unchanged.

- [ ] **Step 5: Verify both adapters and legacy behavior**

Run:

```bash
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_vaultfs_contract.py tests/integration/mcp/test_corpus_facets.py tests/integration/mcp/test_tool_handlers.py -q
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_mcp_isolation.py -q
.venv/bin/ruff check mcp/vaultfs mcp/tools/search.py tests/integration/mcp
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mcp/vaultfs/base.py mcp/vaultfs/sqlite.py mcp/vaultfs/postgres.py mcp/tools/search.py tests/integration/mcp
git commit -m "fix: apply lexical filters before retrieval limits"
git push origin feat/platform-architecture-evolution
```

### Task 5: Add the evaluation CLI and lexical baseline report

**Files:**
- Create: `api/scripts/retrieval_eval.py`
- Create: `tests/unit/test_retrieval_eval_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write RED CLI tests**

Inject a fake retriever factory and assert deterministic JSON, selected `lexical`/`hybrid` profiles, no fixture content in error logs, nonzero exit on invalid datasets, and exit `3` when `--require-promotion-gate` fails.

```python
def test_cli_compares_profiles_and_enforces_gate(tmp_path, capsys):
    code = main(["--dataset", str(DATASET), "--compare", "--require-promotion-gate"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["promotion"]["eligible"] is True
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=api .venv/bin/pytest tests/unit/test_retrieval_eval_cli.py -q`

Expected: missing CLI module.

- [ ] **Step 3: Implement a dependency-injected CLI**

Support:

```text
python -m scripts.retrieval_eval --dataset FILE --profile lexical --output-json FILE
python -m scripts.retrieval_eval --dataset FILE --compare --require-promotion-gate
```

`--profile lexical` is always available. `--profile hybrid` and `--compare` use the configured hosted hybrid service. Print one stable JSON object containing dataset version/digest, profile, metrics, filtered counts, p50/p95 latency, and promotion decision. Never print query text, document content, API keys, or embeddings.

- [ ] **Step 4: Capture and commit the synthetic lexical baseline**

Run: `PYTHONPATH=api .venv/bin/python -m scripts.retrieval_eval --dataset tests/fixtures/retrieval/v1/cases.jsonl --profile lexical`

Expected: valid JSON with exact fixture metrics and a SHA-256 dataset digest.

Run: `PYTHONPATH=api .venv/bin/pytest tests/unit/test_retrieval_eval_cli.py tests/unit/core/test_evaluation.py -q`

```bash
git add api/scripts/retrieval_eval.py tests/unit/test_retrieval_eval_cli.py README.md
git commit -m "feat: add retrieval evaluation CLI"
git push origin feat/platform-architecture-evolution
```

### Task 6: Define validated embedding profiles and the OpenAI-compatible adapter

**Files:**
- Create: `llmwiki_core/models.py`
- Create: `api/services/embeddings.py`
- Create: `tests/unit/core/test_models.py`
- Create: `tests/unit/test_embeddings.py`
- Modify: `api/config.py`
- Modify: `mcp/config.py`
- Modify: `api/requirements.txt`
- Modify: `api/requirements.lock`

- [ ] **Step 1: Write RED profile/config tests**

Assert lexical defaults require no endpoint; hybrid is rejected outside hosted mode, without durable jobs, without base URL/model/dimensions, with dimensions outside `1..4096`, or with candidate limits outside `limit..500`.

- [ ] **Step 2: Write RED HTTP adapter tests**

Use `httpx.MockTransport` to assert `/embeddings`, ordered batching, optional bearer auth, dimension validation, finite floats, bounded input count/characters, timeout mapping to typed `EmbeddingUnavailable`, and sanitized errors.

```python
async def test_openai_adapter_rejects_wrong_dimensions():
    client = OpenAIEmbeddingClient(profile=profile(dimensions=3), transport=fake([[1.0, 2.0]]))
    with pytest.raises(InvalidEmbeddingResponse):
        await client.embed(["safe fixture"])
```

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/pytest tests/unit/core/test_models.py tests/unit/test_embeddings.py tests/unit/test_durable_runtime_config.py -q`

Expected: missing profile and adapter types.

- [ ] **Step 4: Implement the model port and settings**

Use these environment variables with inert defaults:

```text
HYBRID_SEARCH_ENABLED=false
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=
EMBEDDING_DIMENSIONS=0
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=30
HYBRID_LEXICAL_CANDIDATES=50
HYBRID_VECTOR_CANDIDATES=50
HYBRID_RRF_K=60
```

Secrets live only in settings and HTTP headers; profile identity contains provider/model/dimensions but no key.

- [ ] **Step 5: Verify, lock dependencies, and commit**

Run:

```bash
.venv/bin/pytest tests/unit/core/test_models.py tests/unit/test_embeddings.py tests/unit/test_durable_runtime_config.py -q
.venv/bin/ruff check llmwiki_core/models.py api/services/embeddings.py api/config.py mcp/config.py
```

Regenerate `api/requirements.lock` using the repository's existing lock command after adding no new direct dependency unless the lock is stale; `httpx` is already direct.

```bash
git add llmwiki_core/models.py api/services/embeddings.py api/config.py mcp/config.py api/requirements.txt api/requirements.lock tests/unit/core/test_models.py tests/unit/test_embeddings.py tests/unit/test_durable_runtime_config.py
git commit -m "feat: add validated embedding adapter"
git push origin feat/platform-architecture-evolution
```

### Task 7: Add the pgvector schema and tenant-safe vector store

**Files:**
- Create: `supabase/migrations/013_chunk_embeddings.sql`
- Create: `api/services/vector_store.py`
- Create: `tests/integration/test_chunk_embeddings_schema.py`
- Create: `tests/integration/test_vector_store.py`
- Modify: `tests/helpers/schema.sql`
- Modify: `.github/workflows/test.yml`
- Modify: `deploy/docker-compose.selfhost.yml`
- Modify: `docs/self-hosting.md`

- [ ] **Step 1: Write RED migration/static tests**

Require `CREATE EXTENSION IF NOT EXISTS vector`, versioned uniqueness, `vector_dims(embedding) = dimensions`, document/user/KB foreign ownership, RLS, and no broad cross-tenant grants. Require the pinned CI image `pgvector/pgvector:0.8.0-pg16` in workflow and the self-host CI profile.

- [ ] **Step 2: Write RED real-Postgres tests**

Against the pgvector service, cover ordered cosine candidates, same-tenant/KB isolation, current document version only, provider/model/dimension isolation, all filters before limit, idempotent same-version replacement, stale-version write rejection, and rollback preserving the previous complete vector set.

- [ ] **Step 3: Verify RED**

Run: `PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_chunk_embeddings_schema.py tests/integration/test_vector_store.py -q`

Expected: missing migration/table/store.

- [ ] **Step 4: Implement additive schema**

Create `chunk_embeddings` with `embedding vector NOT NULL`, explicit `dimensions`, tenant/KB/document/version/chunk/provider/model columns, timestamps, unique identity, and `ON DELETE CASCADE`. Keep exact cosine scan initially because mixed dimensions prevent one global ANN index; filter by dimensions/profile/tenant/KB/version before `<=>` and bounded `LIMIT`. Add B-tree indexes selected by this query shape, not speculative indexes.

- [ ] **Step 5: Implement version-fenced store**

`replace_document_embeddings()` must lock/read the document version, verify the submitted chunk identity set equals the current chunk set, write all rows in one transaction, and delete older profile rows only after the new set is complete. `search()` must return `SearchResult` and convert typed pgvector failures to `RetrieverUnavailable` so only the hybrid service can fall back.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_chunk_embeddings_schema.py tests/integration/test_vector_store.py -q
.venv/bin/ruff check api/services/vector_store.py tests/integration/test_chunk_embeddings_schema.py tests/integration/test_vector_store.py
```

```bash
git add supabase/migrations/013_chunk_embeddings.sql api/services/vector_store.py tests/helpers/schema.sql tests/integration/test_chunk_embeddings_schema.py tests/integration/test_vector_store.py .github/workflows/test.yml deploy/docker-compose.selfhost.yml docs/self-hosting.md
git commit -m "feat: add versioned pgvector storage"
git push origin feat/platform-architecture-evolution
```

### Task 8: Embed current document versions with durable jobs

**Files:**
- Modify: `api/jobs/models.py`
- Modify: `api/jobs/handlers.py`
- Modify: `api/jobs/service.py`
- Modify: `api/services/ocr.py`
- Create: `api/scripts/enqueue_embeddings.py`
- Create: `tests/unit/jobs/test_embedding_handler.py`
- Create: `tests/integration/test_durable_embeddings.py`
- Preserve: `supabase/migrations/013_chunk_embeddings.sql`
- Create: `supabase/migrations/014_document_embedding_jobs.sql`
- Modify: `tests/helpers/schema.sql`

- [ ] **Step 1: Write RED job-contract tests**

Add `JobType.DOCUMENT_EMBED = "document.embed"`. Validate the exact non-secret payload `{document_id, document_version, provider, model, dimensions}` and reject any key/token/base URL/content field.

- [ ] **Step 2: Write RED durable integration tests**

Cover successful batching, idempotent retry, crash before vector commit, lease loss before commit, document version changing during the embedding HTTP call, profile changing before execution, partial/wrong-dimension provider output, terminal missing document, retryable endpoint failure, and reconciliation of ready current versions missing embeddings.

- [ ] **Step 3: Verify RED**

Run: `PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_embedding_handler.py tests/integration/test_durable_embeddings.py -q`

Expected: unsupported job type/handler.

- [ ] **Step 4: Implement version-fenced durable embedding**

The handler reads current chunks, checkpoints its lease, calls the adapter outside a database transaction, then starts one transaction, checkpoints again, rechecks document/chunk versions, and atomically replaces embeddings. A superseded version returns `{"document_id": ..., "stale": true}` without writing. Endpoint/timeouts retry; invalid response/profile mismatch fails with stable bounded error codes.

After extraction commits version `N`, enqueue the embedding job with idempotency key `embed:{document_id}:{N}:{provider}:{model}:{dimensions}` only when hybrid is configured. If enqueue fails, extraction remains successfully searchable lexically; `enqueue_embeddings.py --missing` repairs the durable enqueue gap.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/jobs/test_embedding_handler.py tests/integration/test_durable_embeddings.py tests/integration/test_durable_extraction.py -q
.venv/bin/ruff check api/jobs api/services/ocr.py api/scripts/enqueue_embeddings.py tests/unit/jobs/test_embedding_handler.py tests/integration/test_durable_embeddings.py
```

```bash
git add api/jobs api/services/ocr.py api/scripts/enqueue_embeddings.py api/services/embeddings.py supabase/migrations/014_document_embedding_jobs.sql tests/helpers/schema.sql tests/unit/jobs/test_embedding_handler.py tests/integration/test_durable_embeddings.py tests/integration/test_document_embedding_jobs_migration.py
git commit -m "feat: add durable document embedding jobs"
git push origin feat/platform-architecture-evolution
```

### Task 9: Wire opt-in hybrid retrieval, reranking, and graph expansion

**Files:**
- Create: `mcp/services/retrieval.py`
- Create: `tests/unit/mcp/test_retrieval_service.py`
- Modify: `mcp/vaultfs/base.py`
- Modify: `mcp/vaultfs/postgres.py`
- Modify: `mcp/vaultfs/sqlite.py`
- Modify: `mcp/tools/search.py`
- Modify: `tests/integration/mcp/test_tool_handlers.py`
- Modify: `tests/integration/mcp/test_mcp_isolation.py`

- [ ] **Step 1: Write RED service and compatibility tests**

Assert:

- omitted profile and explicit `lexical` produce identical legacy output;
- `hybrid` is rejected in local/offline mode and when the flag/profile is invalid;
- vector unavailability returns lexical hits and emits only structured identifiers/counts;
- filters reach both lexical and vector retrievers unchanged;
- citation/source duplicates collapse by chunk identity;
- bounded one-hop graph expansion never crosses tenant/KB and never displaces higher-ranked direct hits;
- an injected deterministic reranker may reorder only the bounded known hit set.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=mcp .venv/bin/pytest tests/unit/mcp/test_retrieval_service.py tests/integration/mcp/test_tool_handlers.py tests/integration/mcp/test_mcp_isolation.py -q`

Expected: no hybrid service/profile argument.

- [ ] **Step 3: Implement hosted retrieval adapters**

`mcp/services/retrieval.py` builds the core service from a lexical `VaultFS.retrieve`, a Postgres vector retriever, an optional injected reranker, and a graph expander. The expander follows only existing `document_references` within the same user/KB, fetches at most one current chunk per related document, and respects the remaining result budget.

- [ ] **Step 4: Extend MCP compatibly**

Add optional `retrieval_profile: Literal["lexical", "hybrid"] = "lexical"` to the MCP search tool. Do not change existing names/defaults/formatting. Local mode exposes lexical only; hosted hybrid remains opt-in even when it passes promotion gates.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=mcp .venv/bin/pytest tests/unit/mcp/test_retrieval_service.py tests/integration/mcp/test_tool_handlers.py -q
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_mcp_isolation.py -q
.venv/bin/ruff check mcp/services/retrieval.py mcp/vaultfs mcp/tools/search.py tests/unit/mcp/test_retrieval_service.py
```

```bash
git add mcp/services/retrieval.py mcp/vaultfs mcp/tools/search.py tests/unit/mcp/test_retrieval_service.py tests/integration/mcp/test_tool_handlers.py tests/integration/mcp/test_mcp_isolation.py
git commit -m "feat: add opt-in hosted hybrid retrieval"
git push origin feat/platform-architecture-evolution
```

### Task 10: Add telemetry and prove fallback safety

**Files:**
- Modify: `api/telemetry.py`
- Modify: `mcp/services/retrieval.py`
- Modify: `api/services/embeddings.py`
- Modify: `tests/helpers/telemetry_contract.py`
- Modify: `tests/unit/test_telemetry_contracts.py`
- Create: `tests/integration/test_hybrid_failure_matrix.py`

- [ ] **Step 1: Write RED telemetry contract tests**

Require `retrieval_finished`, `retrieval_fallback`, and `embedding_finished` events with only profile, counts, duration, stable reason/error code, tenant-safe IDs, model/provider identity, and dimensions. Forbid query text, content, vector values, API keys, URLs containing credentials, and raw exception messages.

- [ ] **Step 2: Write RED failure-matrix tests**

Cover disabled flag, absent embeddings, endpoint timeout, pgvector unavailable, dimension mismatch, stale version, one retriever returning no hits, cancellation, and lexical backend failure. Only vector-side typed availability failures may fall back; lexical failure must remain visible.

- [ ] **Step 3: Implement events at committed outcomes**

Emit one retrieval event after the final result/fallback is known and one embedding event after the durable vector transaction commits or the durable job transition is known. Reuse the existing JSON telemetry sanitizer.

- [ ] **Step 4: Verify and commit**

Run:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_telemetry_contracts.py tests/integration/test_hybrid_failure_matrix.py -q
.venv/bin/ruff check api/telemetry.py api/services/embeddings.py mcp/services/retrieval.py tests/integration/test_hybrid_failure_matrix.py
```

```bash
git add api/telemetry.py api/services/embeddings.py mcp/services/retrieval.py tests/helpers/telemetry_contract.py tests/unit/test_telemetry_contracts.py tests/integration/test_hybrid_failure_matrix.py
git commit -m "test: cover hybrid retrieval failure modes"
git push origin feat/platform-architecture-evolution
```

### Task 11: Run comparative evaluation and enforce the promotion gate

**Files:**
- Modify: `api/scripts/retrieval_eval.py`
- Create: `tests/integration/test_retrieval_evaluation.py`
- Modify: `.github/workflows/test.yml`
- Modify: `tests/helpers/ci_test_matrix.py`
- Modify: `tests/test_ci_matrix_contract.py`

- [ ] **Step 1: Add RED end-to-end evaluation tests**

Seed the synthetic corpus in Postgres, generate deterministic fake embeddings, run lexical and hybrid profiles over the exact same cases, and assert exact metrics/dataset digest. Include one passing and one failing promotion report, plus p95 exactly at `2.0x` as passing.

- [ ] **Step 2: Add a dedicated primary CI segment**

Create `integration-retrieval` for the pgvector/evaluation files. The matrix ownership contract must still assign every unit/integration test file exactly once and reject empty or hidden-failure execution.

- [ ] **Step 3: Verify locally with fixed pgvector**

Run:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/integration/test_retrieval_evaluation.py -q
.venv/bin/pytest tests/test_ci_matrix_contract.py -q
python -m tests.helpers.ci_test_matrix run integration-retrieval -- env PYTHONPATH=api MODE=hosted pytest -v
```

Expected: the evaluation is deterministic and the primary matrix remains exhaustive.

- [ ] **Step 4: Commit**

```bash
git add api/scripts/retrieval_eval.py tests/integration/test_retrieval_evaluation.py .github/workflows/test.yml tests/helpers/ci_test_matrix.py tests/test_ci_matrix_contract.py
git commit -m "test: enforce retrieval promotion evidence"
git push origin feat/platform-architecture-evolution
```

### Task 12: Document rollout and run the full milestone gate

**Files:**
- Modify: `README.md`
- Modify: `docs/self-hosting.md`
- Modify: `docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md`
- Create: `docs/architecture/retrieval.md`

- [ ] **Step 1: Document operational contracts**

Describe lexical default/fallback, configuration validation, embedding backfill/re-embed, model changes, dataset privacy, evaluation CLI, promotion thresholds, pgvector requirement, additive migration, rollback by disabling `HYBRID_SEARCH_ENABLED`, and the fact that passing the gate makes hybrid eligible but never automatically changes the deployment default.

- [ ] **Step 2: Run focused and complete matrices**

Run:

```bash
.venv/bin/pytest tests/unit/core/test_search.py tests/unit/core/test_evaluation.py tests/unit/core/test_models.py -q
PYTHONPATH=api MODE=hosted .venv/bin/pytest tests/unit/test_embeddings.py tests/unit/jobs/test_embedding_handler.py tests/integration/test_chunk_embeddings_schema.py tests/integration/test_vector_store.py tests/integration/test_durable_embeddings.py tests/integration/test_hybrid_failure_matrix.py tests/integration/test_retrieval_evaluation.py -q
PYTHONPATH=mcp .venv/bin/pytest tests/unit/mcp/test_retrieval_service.py tests/integration/mcp/test_tool_handlers.py tests/integration/mcp/test_corpus_facets.py -q
PYTHONPATH=mcp .venv/bin/pytest tests/integration/mcp/test_mcp_isolation.py -q
.venv/bin/pytest tests/test_ci_matrix_contract.py -q
.venv/bin/ruff check llmwiki_core api mcp tests
git diff --check
```

Then push and require every GitHub Actions job, including the fixed pgvector retrieval segment and scaled Compose job, to pass at the exact head SHA.

- [ ] **Step 3: Request specification and quality reviews**

Specification review checks every Milestone 3 clause: deterministic metrics, filter-before-limit semantics, optional versioned pgvector, OpenAI-compatible/fake adapters, RRF, reranker/graph hooks, lexical fallback, and promotion thresholds. Quality review reports Critical/Important/Minor findings and `Ready: Yes/No`; fix and re-run until all severities are zero.

- [ ] **Step 4: Record final evidence and commit**

Update the design status with exact implementation SHA, Actions URL, per-segment counts, evaluation dataset digest/metrics, promotion result, pinned images, and both review conclusions.

```bash
git add README.md docs/self-hosting.md docs/architecture/retrieval.md docs/superpowers/specs/2026-07-24-platform-architecture-evolution-design.md
git commit -m "docs: record hybrid retrieval verification"
git push origin feat/platform-architecture-evolution
```

Wait for the documentation commit's own workflow to pass, then confirm local HEAD equals `origin/feat/platform-architecture-evolution` and the worktree is clean before beginning server-side RAG.

## Plan self-review

- Spec coverage: all Milestone 3 requirements map to Tasks 1-12, including filters before limits, exact metrics, pgvector version fencing, fake/OpenAI-compatible embeddings, opt-in fallback, RRF, optional reranking, graph expansion, and the 10%/2x gate.
- Compatibility: existing MCP arguments and lexical result formatting remain the default; local mode has no model/network requirement.
- Failure safety: only typed vector availability failures fall back; lexical errors, schema errors, tenant violations, and invalid embeddings fail visibly.
- Scope: server-side RAG, arbitrary agent behavior, vector-only search, production evaluation corpora, and automatic default promotion remain outside this milestone.
- Placeholder scan: the plan contains no deferred implementation markers; commands, interfaces, test expectations, and rollout gates are explicit.
