# Retrieval architecture and rollout

## Product contract

Lexical retrieval is the deployment default in both local and hosted mode. It
does not require an embedding endpoint and remains available when hybrid
retrieval is disabled or the vector side reports a typed availability failure.
Callers opt in per search with `retrieval_profile="hybrid"`; setting
`HYBRID_SEARCH_ENABLED=true` makes that profile available but does not change
the default argument.

Hosted hybrid retrieval applies every path, tag, document-kind, annotation,
scope, and corpus-facet filter inside both database candidate queries before
their limits. It retrieves bounded lexical and vector candidate sets in
parallel, combines them with reciprocal-rank fusion (RRF), preserves unique
source/chunk identities, and can invoke bounded reranker and graph-expansion
hooks. The hooks are optional and no model reranker is configured by default.

Fallback is deliberately narrow. A typed vector availability failure returns
the lexical result as profile `lexical_fallback` and emits fallback telemetry.
Lexical failures, invalid configuration, tenant violations, malformed backend
results, and unexpected programming errors remain visible; they are not
converted into an apparently successful lexical result. Vector store/provider
details are sanitized at their adapter boundaries before a typed availability
failure is eligible for fallback.

Local mode is always lexical-only and requires no model or network access.
Hosted callers that do not request `hybrid` also stay entirely on the lexical
path even when hybrid support is configured.

## Storage and embedding lifecycle

Migration `013_chunk_embeddings.sql` creates the `vector` extension and an
RLS-protected `chunk_embeddings` table. A vector is identified by tenant,
knowledge base, document, `document_version`, chunk index, provider, model,
and dimensions. Retrieval joins only the current ready, non-archived document
version and the exact configured embedding profile, so stale or mixed-profile
vectors cannot enter a result.

Migration `014_document_embedding_jobs.sql` adds durable `document.embed`
jobs and a reconciliation index. After extraction commits a new ready document
version, the worker idempotently enqueues that exact version/profile. Failed
delivery is repaired with the reconciliation command below; retrying cannot
create duplicate vectors.

```bash
PYTHONPATH=api MODE=hosted .venv/bin/python -m scripts.enqueue_embeddings \
  --missing --page-size 100
```

Run the command while the durable workers are running and the same hosted
environment variables are loaded. It scans only ready current source versions
whose configured vector set is incomplete. Repeat after workers settle until
it reports `{"enqueued":0,"scanned":0}`.

The embedding profile is the immutable tuple `(provider, model, dimensions)`.
Changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` never reinterprets old
vectors. Deploy the new profile, run `enqueue_embeddings --missing`, wait for
complete coverage, and evaluate that exact profile. Jobs carrying the old
profile stop without writing it after the configuration change. Old rows may
remain as additive rollback data; exact-profile queries ignore them.

## Hosted configuration

Hybrid retrieval is disabled by default. API, worker, and MCP processes must
receive the same profile and candidate settings:

```dotenv
HYBRID_SEARCH_ENABLED=true
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_BASE_URL=https://embeddings.example.com/v1
EMBEDDING_API_KEY=replace-with-secret
EMBEDDING_MODEL=text-embedding-model
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=30
HYBRID_LEXICAL_CANDIDATES=50
HYBRID_VECTOR_CANDIDATES=50
HYBRID_RRF_K=60
```

The current self-hosted Compose file is a starting skeleton and does not
forward these optional variables. Add explicit `${VARIABLE}` mappings for all
of the values above under the `api`, `worker`, and `mcp` service `environment`
blocks (or inject them with your orchestrator); putting them only in the
Compose `--env-file` does not place them inside the containers.

Configuration fails closed at process startup when hybrid is enabled outside
hosted mode, durable jobs are disabled, the base URL or model is blank, or the
dimensions/candidate/batch/timeout/RRF values are out of their validated
bounds. Each candidate limit must also be at least the caller's requested
result limit. `EMBEDDING_PROVIDER` currently accepts only
`openai_compatible`; the test suite uses an explicitly injected deterministic
fake adapter and never makes it a deployment default.

Postgres must provide pgvector 0.8.x; CI and the self-hosted smoke profile pin
`pgvector/pgvector:0.8.0-pg16`. PGroonga remains required for hosted lexical
retrieval. Apply migrations through `014` before enabling the flag. Both
migrations are additive and safe to leave in place when hybrid is disabled.

## Evaluation and promotion

Evaluation datasets use the strict versioned JSONL schema. Keep representative
deployment cohorts outside the repository: do not commit user queries,
documents, embeddings, API keys, database error text, or tenant-identifying
values. Use pseudonymous fixture identifiers and control access to reports as
production telemetry. Reports contain only dataset identity, profile,
case count, aggregate Recall@5/10/20, MRR, nDCG@10, filtered result count, and
latency percentiles.

The repository's content-free lexical smoke cohort is reproducible without a
database or network:

```bash
PYTHONPATH=api .venv/bin/python -m scripts.retrieval_eval \
  --dataset tests/fixtures/retrieval/v1/cases.jsonl \
  --profile lexical \
  --output-json retrieval-baseline.json
```

Comparative execution uses the same CLI boundary with `--compare` and the
real-Postgres retriever factory. The standalone repository command has no
deployment credentials or implicit tenant selection, so it intentionally
returns the stable `hybrid_unavailable` configuration error unless an
operator-owned wrapper injects that factory. This prevents accidentally
evaluating or disclosing the wrong tenant. The checked-in Postgres integration
gate exercises that boundary directly:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest \
  tests/integration/test_retrieval_evaluation.py -q
```

A deployment-specific wrapper should invoke the CLI equivalent:

```text
retrieval_eval --dataset <private-cases.jsonl> --compare \
  --require-promotion-gate --output-json <private-report.json>
```

The two profiles run against one read-only `REPEATABLE READ` snapshot and the
same dataset digest. The gate is inclusive: hybrid Recall@10 must be at least
110% of lexical Recall@10, and hybrid p95 latency must be no more than 2.0x
the lexical baseline. Exit code `3` means the comparison ran but did not pass;
configuration/dataset/retrieval errors use `2`, and report-write errors use
`4`.

Passing the gate only makes that exact hybrid profile eligible for a deliberate
deployment decision. It never changes configuration, never changes the search
tool's default, and never enables hybrid automatically. Re-run the same private
cohort after model, dimensions, chunking, filters, candidate limits, RRF,
reranker, graph-expansion, or representative corpus changes.

## Rollout and rollback

1. Confirm the database supports PGroonga and pgvector 0.8.x, then apply all
   additive migrations through `014` while hybrid remains disabled.
2. Deploy API, worker, and MCP with one identical, validated embedding profile.
3. Enable `HYBRID_SEARCH_ENABLED=true`, keep callers on lexical, and run the
   missing-embedding reconciliation until the current corpus is covered.
4. Run the private lexical/hybrid comparison. Review quality and latency in
   addition to the machine gate.
5. Allow selected callers to request `retrieval_profile="hybrid"`. Keep
   lexical as the default unless an operator explicitly changes client policy.

To roll back retrieval immediately, set `HYBRID_SEARCH_ENABLED=false` on API,
worker, and MCP, then roll/restart those processes. Requests return to lexical
behavior; no schema reversal or vector deletion is required. Preserve the
embedding rows and job history for auditability and a later re-enable. If the
application release itself is rolled back, roll API and worker together as
described in the self-hosting guide.
