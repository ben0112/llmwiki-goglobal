# Platform Architecture Evolution Design

## Status and scope

This specification defines one long-lived implementation branch,
`feat/platform-architecture-evolution`, with four sequential, independently
testable milestones:

1. shared kernel and enforceable data invariants;
2. durable work and horizontally scalable hosted API;
3. retrieval evaluation and optional hybrid retrieval;
4. optional server-side RAG orchestration.

Each milestone must leave local and hosted modes usable. Changes that are not
yet ready for normal use remain behind configuration flags. The existing MCP
agent workflow remains a supported first-class path throughout the work.

The implementation preserves these product contracts:

- Local mode remains single-user, fully offline, and usable with only the
  filesystem and SQLite.
- Hosted mode remains multi-user with Postgres row-level isolation and S3
  compatible object storage.
- Existing REST and MCP tool signatures remain compatible unless a migration
  is explicitly documented and covered by contract tests.
- Lexical search remains available when embeddings or a model endpoint are not
  configured.

## Architectural approach

The branch uses incremental migration rather than a parallel rewrite. Shared
behavior moves into an installable `llmwiki_core` Python package. API, MCP,
worker, and CLI entrypoints call application services from that package and
keep transport-specific authentication, serialization, and lifecycle code at
their edges.

The target dependency direction is:

```text
web / REST / MCP / CLI
          |
          v
llmwiki_core application services
          |
          v
ports: repositories, object storage, jobs, search, models
          |
          v
SQLite + filesystem | Postgres + S3 | external model endpoints
```

Infrastructure adapters may depend on `llmwiki_core`; the core must not import
FastAPI, FastMCP, asyncpg, aiosqlite, boto, or frontend code. API and MCP may
temporarily retain legacy implementations while each use case is migrated,
but every migrated behavior must have one authoritative implementation.

## Milestone 1: shared kernel and data invariants

### Package boundary

Create an installable root package with these focused modules:

- `llmwiki_core/documents.py`: document kinds, statuses, state-transition
  rules, normalized paths, and immutable document identity.
- `llmwiki_core/chunking.py`: the authoritative chunk model and chunking
  algorithm currently duplicated by API and MCP.
- `llmwiki_core/facets.py`: facet keys, validation, and backend-neutral facet
  values. SQL compilation remains in adapters.
- `llmwiki_core/references.py`: citation and wiki-link parsing plus reference
  edge types.
- `llmwiki_core/search.py`: backend-neutral search query/result contracts.
- `llmwiki_core/jobs.py`: job types and state transitions used by milestone 2.
- `llmwiki_core/models.py`: model invocation and embedding interfaces used by
  milestones 3 and 4.

The package is installed into API, MCP, worker, CLI, and test environments.
Docker build contexts move to the repository root so all services consume the
same package source rather than copying it.

### Document invariants

The following rules are enforced in application code and, where practical, in
database constraints:

- A document has one explicit kind: `source`, `wiki`, or `asset`. Hosted mode
  receives a `source_kind` column instead of repeatedly inferring kind from
  path; existing rows are backfilled from `/wiki/` and asset metadata.
- Status transitions are limited to:
  `pending -> processing -> ready`,
  `pending|processing -> failed`,
  `failed -> pending`, and the explicit system-repair invalidation
  `ready -> pending`. The repair transition is not available to ordinary
  business flows. Hosted archival remains the orthogonal
  `archived=true` lifecycle flag used by the existing schema. Local deletion
  remains a physical delete because the filesystem is its source of truth.
- `ready` means the current document version has complete derived data. Pages,
  chunks, metadata patches, and the ready transition commit in one database
  transaction.
- Each derived index record carries `document_version`. A chunk set is valid
  only when every chunk version equals the parent document version.
- Reprocessing increments the document version and atomically replaces the
  prior derived page and chunk set.
- Paths use one normalized logical form at application boundaries; adapters
  translate to local `relative_path` or hosted `path` storage.
- Wiki writes update content, chunks, derived citation edges, staleness, and
  facet rollups transactionally where the backend supports transactions.

SQLite remains rebuildable from disk. A reconciliation pass detects ready
documents with missing or stale derived versions and queues them for repair.
Postgres uses the same invariant query for readiness audits.

### Compatibility strategy

Existing API service ABCs and MCP `VaultFS` remain as compatibility facades
during migration. Their implementations delegate to core use cases one use
case at a time. Once contract tests cover both SQLite and Postgres adapters,
duplicate chunking, facet, and reference logic is removed.

## Milestone 2: durable work and hosted horizontal scaling

### Durable job model

Add a job table to both schemas with these fields:

- stable job id, job type, tenant/user id, knowledge-base id, document id;
- state: `queued`, `running`, `succeeded`, `failed`, or `cancelled`;
- attempt count, maximum attempts, availability time, lease owner and expiry;
- JSON payload, progress, result, and bounded error text;
- idempotency key and created/updated timestamps.

Supported initial job types are `extract_document`, `classify_corpus`,
`rebuild_graph`, `refresh_facets`, `embed_document`, and `build_wiki`.

Postgres workers claim jobs with `FOR UPDATE SKIP LOCKED`, set a finite lease,
and renew it while working. SQLite uses a transactionally claimed row and one
local worker process. Expired leases return to `queued` until maximum attempts
is reached. Job handlers are idempotent: retrying the same job cannot create
duplicate chunks, corpus entries, reference edges, or wiki pages.

### Worker service

Introduce one worker executable and container. Hosted API requests only
validate input, persist source metadata, enqueue work, and return a job or
document identifier. The worker owns extraction, chunking, corpus
classification, embeddings, graph rebuilds, and server-side RAG work.

Local mode starts the same worker loop inside the all-in-one process by
default, backed by the SQLite job table. The local CLI may also run the worker
as a separate process for debugging without changing behavior.

### Upload scaling

Hosted resumable uploads stop depending on process-local `_uploads` state and
temporary files. Upload sessions are persisted in Postgres and backed by S3
multipart upload:

- create validates quota and starts an S3 multipart upload;
- each chunk is uploaded as a numbered S3 part and its ETag and committed
  offset are transactionally recorded;
- completion verifies size and ordered parts, completes the object, creates
  the document row, and enqueues extraction;
- abort and stale-session cleanup abort the S3 multipart upload.

The existing TUS-compatible HTTP surface is retained so the web client does
not need to change protocol. Offset conflicts return the persisted committed
offset. Quota accounting includes active upload sessions.

### Remaining process-local state

- Graph rebuild serialization moves to Postgres advisory locks keyed by
  knowledge-base id; cooldown timestamps are stored in the job record.
- Every API replica opens its own Postgres `LISTEN` connection and broadcasts
  notifications only to WebSockets connected to that replica. No shared
  WebSocket registry is required.
- Rate limits remain a reverse-proxy responsibility until a shared limiter is
  introduced; the application limiter is not used as a correctness boundary.
- MCP remains stateless and can scale independently.

After this milestone the hosted deployment no longer documents an API
single-replica requirement. API, MCP, worker, and converter replica counts can
be tuned independently.

## Milestone 3: retrieval evaluation and hybrid retrieval

### Evaluation before model-dependent ranking

Add a versioned JSONL evaluation dataset format. Each case contains tenant-safe
fixture identifiers, query text, optional path/tags/facets/scope filters,
relevant document or chunk ids, and graded relevance where available.

The evaluation CLI runs a selected retriever and reports:

- Recall@5, Recall@10, and Recall@20;
- mean reciprocal rank;
- nDCG@10;
- filtered-query result count and latency percentiles.

Tests use a small deterministic fixture set. Real deployment datasets remain
outside the repository and contain no user content.

### Filter semantics

Path glob, tags, document kind, annotated-only, area, and all corpus facets are
compiled into each supporting backend query before the retrieval limit is
applied. Current whole-chunk vectors support only `scope=all`; a hybrid request
for source-only or annotations-only scope makes the vector side typed
unavailable and returns the correctly scoped lexical result as
`lexical_fallback`. Search contracts distinguish candidate count from returned
count so post-processing cannot silently underfill a request.

Postgres adds indexes for high-traffic facet expressions or normalized facet
columns selected from evaluation evidence. It does not add a broad collection
of speculative indexes. SQLite retains trigram FTS and short-token LIKE
fallback behavior.

### Optional vector retrieval

Hosted mode gains an optional pgvector-backed embedding table keyed by
document id, document version, chunk index, provider, and embedding model.
Embedding dimensions are stored per configured model family in a validated
configuration; incompatible model changes enqueue a re-embed rather than
mixing vectors.

An OpenAI-compatible embeddings adapter is provided, along with a deterministic
fake adapter for tests. If no embedding endpoint is configured, vector
retrieval is disabled and lexical behavior is unchanged.

Hybrid retrieval performs:

1. filter pushdown;
2. independent lexical and vector candidate retrieval;
3. reciprocal-rank fusion with configured, bounded candidate counts;
4. optional reranking through a model adapter;
5. citation/source deduplication and graph-aware context expansion.

The default remains lexical-only. A configured hybrid profile is eligible to
become a deployment default only when, on the same representative evaluation
run, Recall@10 improves by at least ten percent relative to lexical retrieval
and p95 latency is no more than twice the lexical baseline. Deployments may
keep hybrid search opt-in even when those gates pass.

### Milestone 3 implementation evidence

Milestone 3's latest substantive hosted-evaluation candidate is
`2e8da85e1c55121204339ff2ef8cec95f299a4bb`. The implementation includes the
strict versioned evaluation schema and exact deterministic metrics,
filter-before-limit compilation, version/profile-fenced pgvector storage,
OpenAI-compatible and deterministic fake embedding adapters, durable embedding
reconciliation, bounded RRF, optional reranker and graph-expansion hooks, typed
lexical fallback, and the inclusive 110% Recall@10 / 2.0x p95 promotion gate.
The current follow-up also makes hosted serving and promotion evaluation call
one pure production lexical compiler with identical PGroonga `&@~`,
`pgroonga_score`, `status != 'failed'`, scope labeling, filter-before-limit,
candidate-count, and stable-order semantics; it removes the evaluator-only
`tsvector` approximation. The current local follow-up additionally fences every
lexical candidate to `dc.document_version = d.version`.

The representative two-case real-Postgres comparison uses dataset digest
`d45bf89f5b28694afe2b4af1d03d15e3ba59e02d3bc20129eff77b58ab39ab7f`.
Its exact lexical metrics are Recall@5/10/20 `0.5`, MRR `0.5`, nDCG@10 `0.5`,
filtered result count `1`, and p50/p95 `10.0 ms`; hybrid reports
Recall@5/10/20 `1.0`, MRR `1.0`, nDCG@10 `1.0`, filtered result count `1`,
and p50/p95 `20.0 ms`. The result is eligible at recall ratio `2.0` and the
inclusive latency boundary `2.0`.

Verification for `2e8da85e1c55121204339ff2ef8cec95f299a4bb` completed with all
six jobs successful in
[GitHub Actions run 30191342308](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30191342308).
The retrieval segment owns the chunk schema, vector store, durable embeddings,
hybrid failure matrix, and real-Postgres evaluation tests. The historical Task
11 specification and quality reviews of implementation commit `5cca143` both
concluded Critical `0`, Important `0`, Minor `0`, and `Ready: Yes`; this does
not claim the same result for the Task 12 candidate or its follow-ups. A Task 12
specification re-review of `20b8e7b` reported `3` Important and `1` Minor
follow-ups, which `396acbf` addressed. The subsequent quality re-review reported
`2` Important follow-ups: the outer async control-signal boundary and lexical
serving/evaluator drift, which `2e8da85` addressed. The specification re-review
of `2e8da85` reported Critical `0`, Important `1`, Minor `0`, and `Ready: No`
because lexical candidates were not fenced to the current document version.
The current local remediation addresses that finding but remains pending
specification and quality re-review; no zero-finding or readiness conclusion is
recorded yet. Operational rollout, private-dataset
handling, promotion, and rollback are documented in
`docs/architecture/retrieval.md`; passing the gate only makes hybrid eligible
and never changes the lexical deployment default.

The current Task 12 local milestone gate records `251` core tests,
`141` CLI boundary tests, `463` API embedding/retrieval tests, `206` MCP
retrieval/tool/facet tests, `48` MCP Postgres-isolation and hosted-corpus tests,
and `8` CI ownership contract tests passing. Its
verification workflow's configured image tags are
`pgvector/pgvector:0.8.0-pg16`, `python:3.11-alpine`,
`redis:7.4.2-alpine`, `minio/minio:RELEASE.2025-04-22T22-12-26Z`,
`minio/mc:RELEASE.2025-04-16T18-13-26Z`, and `nginx:1.27-alpine`.

## Milestone 4: optional server-side RAG orchestration

### Product boundary

Server-side RAG complements rather than replaces external MCP agents. It is
disabled by default and exposed through REST plus a CLI. MCP tools continue to
offer direct search/read/write access.

The first supported workflow is `build_wiki`: compile or refresh a bounded set
of wiki pages from a knowledge base. General autonomous agents, arbitrary code
execution, browser automation, and unbounded recursive planning are outside
this branch.

### Persistent run model

A RAG run is a durable `build_wiki` job with normalized configuration:
knowledge base, requested goal, model profile, retrieval profile, token and
step budgets, target page scope, and idempotency key. Steps persist their type,
inputs digest, bounded output, citations, token usage, latency, and status.

The orchestrator state machine is:

1. plan a bounded page worklist;
2. retrieve candidates with the shared search service;
3. read only selected source/page ranges;
4. draft or revise one page;
5. validate frontmatter, citations, links, and budget;
6. atomically write the page and derived data;
7. run deterministic lint and persist the result;
8. continue until the worklist or budget is exhausted.

Each page write uses an expected document version. A concurrent edit causes a
conflict step and a bounded retry after re-reading; the orchestrator never
silently overwrites newer user content.

### Model and safety behavior

Model access uses the shared model interface and an OpenAI-compatible adapter.
Prompts treat retrieved documents as untrusted content, preserve source
attribution, and do not execute instructions found inside documents. Secrets
are referenced by server configuration and are never stored in job payloads or
step outputs.

Every run enforces maximum steps, maximum model tokens, per-call timeouts, and
maximum page size. Cancellation is cooperative between steps. Failed runs keep
their trace and may be resumed only from the last committed page boundary.

## Error handling and observability

- Domain errors use typed codes; transports map them to HTTP/MCP responses.
- Job failures record bounded messages and structured error codes without
  secrets or full document contents.
- Logs carry job id, document id, knowledge-base id, and request id where
  available.
- Metrics cover queue depth, lease expiry, attempts, extraction duration,
  indexing duration, search latency, retrieval quality, model tokens, and RAG
  validation failures.
- Health checks report process liveness. Readiness checks verify required
  database and object-store dependencies for each service role.

## Testing strategy

Implementation follows red-green-refactor. Every new behavior begins with a
test that fails for the intended missing behavior.

Required test layers are:

- pure unit tests for core state transitions, paths, chunking, facets,
  references, rank fusion, budgets, and orchestration state;
- adapter contract tests run against SQLite and Postgres for document writes,
  jobs, filters, and version invariants;
- integration tests for S3 multipart upload, worker lease recovery, job
  idempotency, API replica independence, and WebSocket delivery;
- retrieval evaluation fixture tests with exact metric expectations;
- end-to-end hosted tests for upload through searchable ready document and for
  a deterministic fake-model wiki build;
- regression execution of the existing unit and integration suites.

Each milestone ends with its focused tests, the full Python suite, formatting
and lint checks, and the relevant Docker image builds. Completion is not
claimed when a required integration dependency is unavailable; the missing
verification is reported explicitly.

## Migration and rollout

Database migrations are forward-only and safe to apply before deploying new
code. New columns and tables are additive first; backfills are explicit;
constraints become strict only after audit queries show no invalid rows.

Feature flags are:

- `DURABLE_JOBS_ENABLED`, enabled only after job tables and worker are live;
- `HYBRID_SEARCH_ENABLED`, requiring a valid embedding profile;
- `SERVER_RAG_ENABLED`, requiring durable jobs and a valid model profile.

Rollback disables flags and returns traffic to existing lexical/MCP paths.
Migrations are not automatically reversed because job histories and embeddings
are non-destructive additive data.

## Explicit non-goals

- Splitting every domain into a separately deployed microservice.
- Replacing SQLite in local mode.
- Replacing PGroonga or FTS with vector-only retrieval.
- Removing MCP or requiring a platform-controlled LLM.
- Supporting arbitrary autonomous tool execution in the server orchestrator.
- Committing production evaluation corpora or user documents to the repository.

## Acceptance criteria

The branch is ready for review when all of the following hold:

- API and MCP use the shared core for the migrated document, chunk, facet,
  reference, search, and job contracts, with duplicate authoritative logic
  removed.
- Local mode remains offline and passes existing behavioral tests.
- Hosted document readiness is atomic and auditably consistent with derived
  versions.
- Interrupted jobs and uploads recover on another worker or API replica.
- Hosted API can run at least two replicas without session affinity for normal
  REST, upload, graph, and WebSocket workflows.
- Retrieval evaluation produces deterministic metrics, filters execute before
  limits, and hybrid search cleanly falls back to lexical search.
- A fake-model end-to-end RAG run produces a linted, cited wiki page with a
  persisted trace, while MCP direct authoring still works.
- Required tests, lint checks, and service builds pass with fresh evidence.
