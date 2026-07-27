# Retrieval architecture and rollout

## Product contract

Lexical retrieval is the deployment default in both local and hosted mode. It
does not require an embedding endpoint and remains available when hybrid
retrieval is disabled or the vector side reports a typed availability failure.
Callers opt in per search with `retrieval_profile="hybrid"`; setting
`HYBRID_SEARCH_ENABLED=true` makes that profile available but does not change
the default argument.

The [platform overview](overview.md) owns the end-to-end data flow. Hosted
embedding work follows the [durable-job contract](durable-jobs.md), and
server-side generation consumes this same retrieval contract as described in
[server-rag.md](server-rag.md).

For `scope=all`, hosted hybrid retrieval applies path, tag, document-kind,
annotated-only, area, and corpus-facet filters inside both lexical and vector
database candidate queries before their limits. It retrieves the bounded
candidate sets in parallel, combines them with reciprocal-rank fusion (RRF),
preserves unique source/chunk identities, and can invoke bounded reranker and
graph-expansion hooks. The hooks are optional and no model reranker is
configured by default.

Document embeddings represent the whole chunk, so the vector backend cannot
honor `scope=source` or `scope=annotations` independently. Those scoped hybrid
requests raise the typed vector-unavailable signal and the whole request
returns the lexical result as `lexical_fallback`; the lexical query still
applies its source/annotation scope before its own limit. Vector retrieval is
executed only for `scope=all`.

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

Hosted serving and promotion evaluation compile lexical candidates through the
same pure shared Postgres compiler. Both therefore execute the production
PGroonga `&@~` match and `pgroonga_score` ordering, use the same
`status != 'failed'` and archive rules, derive source/annotation scope from the
same labeled rows, and apply every document filter before `candidate_limit`.
The evaluator does not maintain a `to_tsvector` approximation or an independent
filter/status query.

## Storage and embedding lifecycle

Migration `013_chunk_embeddings.sql` creates the `vector` extension and an
RLS-protected `chunk_embeddings` table. A vector is identified by tenant,
knowledge base, document, `document_version`, chunk index, provider, model,
and dimensions. The durable writer verifies that the source is ready,
non-archived, and still on the submitted current version before committing a
complete vector set. Retrieval joins only the current document/chunk version
and the exact configured embedding profile, and excludes failed or archived
documents. It does not independently require `status=ready`; version fencing
prevents stale or mixed-profile vectors from entering a result.

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

Switch profiles without exposing callers to partial coverage:

1. Stop new hybrid opt-ins and keep every serving entry point on lexical.
2. Roll API producers and workers together onto the new immutable profile;
   keep embedding work enabled there, but keep MCP and other serving entry
   points from selecting hybrid.
3. Run `enqueue_embeddings --missing` after each worker drain until it reports
   no missing work, and confirm current-version coverage for the exact new
   provider/model/dimensions tuple.
4. Run the hosted comparison below in a separate evaluator process configured
   with that exact profile. Do not enable serving hybrid if coverage is
   incomplete or the gate exits `2` or `3`.
5. After a passing report and operator review, roll serving processes onto the
   new profile, then re-enable only the selected canary callers. Keep lexical
   as the default. To abort, disable serving hybrid and restore the previously
   tested profile; old vector rows remain usable because profiles never mix.

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

Comparative execution uses the supported `--hosted` CLI path. It builds the
real Postgres retrievers and OpenAI-compatible embedding client inside one
event loop; no Python-level retriever-factory injection or operator wrapper is
required. The evaluator requires an explicit tenant and knowledge base.
Missing or invalid static hosted settings (including an empty DSN, tenant,
knowledge base, or embedding profile) fail closed as `hybrid_unavailable`.
A non-empty but malformed, unreachable, or runtime-invalid DSN reaches pool
creation and is sanitized as `retrieval_failed`; neither classification emits
connection details. Load the ordinary deployment embedding and hybrid settings
first, and load `DATABASE_URL` and any API key from the deployment secret store
rather than command-line arguments. Then run:

```bash
: "${DATABASE_URL:?load DATABASE_URL from the deployment secret store}"
: "${EMBEDDING_BASE_URL:?load EMBEDDING_BASE_URL}"
: "${EMBEDDING_MODEL:?load EMBEDDING_MODEL}"
: "${EMBEDDING_DIMENSIONS:?load EMBEDDING_DIMENSIONS}"
export HYBRID_SEARCH_ENABLED=true
export RETRIEVAL_EVAL_USER_ID=00000000-0000-0000-0000-000000000001
export RETRIEVAL_EVAL_KNOWLEDGE_BASE_ID=00000000-0000-0000-0000-000000000002
PYTHONPATH=api MODE=hosted .venv/bin/python -m scripts.retrieval_eval \
  --dataset /controlled/private-retrieval-cases.jsonl \
  --compare \
  --hosted \
  --require-promotion-gate \
  --output-json /controlled/private-retrieval-report.json
```

The two UUIDs are required, validated parameters rather than inferred tenant
state; replace the examples with the intended deployment identifiers. The
command never prints the DSN, API key, UUIDs, queries, document content,
embeddings, or backend exception strings. Success, failure, cancellation, and
process-control paths all execute and validate asynchronous client/pool
cleanup; if cleanup itself raises a process-control signal, the evaluator
propagates a sanitized signal rather than claiming the underlying close
completed. The checked-in real-Postgres integration gate for this entry point
is:

```bash
PYTHONPATH=api MODE=hosted .venv/bin/pytest \
  tests/integration/test_retrieval_evaluation.py -q
```

The two profiles run against one read-only `REPEATABLE READ` snapshot and the
same dataset digest. Its lexical side is the exact shared production PGroonga
candidate query, not a separate evaluator ranking implementation. The gate is
inclusive: hybrid Recall@10 must be at least
110% of lexical Recall@10, and hybrid p95 latency must be no more than 2.0x
the lexical baseline. Exit code `3` means the comparison ran but did not pass;
configuration/dataset/retrieval errors use `2`, and report-write errors use
`4`.

Passing the gate only makes that exact hybrid profile eligible for a deliberate
deployment decision. It never changes configuration, never changes the search
tool's default, and never enables hybrid automatically. Re-run the same private
cohort after model, dimensions, chunking, filters, candidate limits, RRF,
reranker, graph-expansion, or representative corpus changes.

## Verification evidence

The closing retrieval candidate is
`66337e5c949220c8d167af46a0ad0e4e7519128c`. Its
[GitHub Actions run 30191640099](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30191640099)
passed all six jobs, and independent specification and quality reviews both
concluded Critical `0`, Important `0`, Minor `0`, `Ready: Yes`.

The representative real-Postgres cohort has dataset digest
`d45bf89f5b28694afe2b4af1d03d15e3ba59e02d3bc20129eff77b58ab39ab7f`.
Lexical Recall@10 and p95 are `0.5` and `10.0 ms`; hybrid reports `1.0` and
`20.0 ms`. The result passes at recall ratio `2.0` and the inclusive latency
boundary `2.0`. This evidence makes only that exact profile eligible for
operator promotion and does not change the lexical default.

The later platform publication baseline
`0173f560c6fec03b87ce4f6803f663d2d6983ead` also passed all six jobs in
[run 30251808426](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251808426).

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
