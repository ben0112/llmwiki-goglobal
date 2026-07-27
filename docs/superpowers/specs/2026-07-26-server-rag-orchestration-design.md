# Server-side RAG Orchestration Design

## Status and decision summary

This specification defines Milestone 4 of the platform architecture evolution:
an optional server-side `build_wiki` workflow. It complements the existing MCP
agent workflow and does not replace direct MCP search, read, or write tools.

The accepted product decisions are:

- use the existing durable Postgres job ledger and worker lease model;
- implement a purpose-built `build_wiki` orchestrator, not a general workflow
  engine;
- let a constrained model planner derive a bounded page worklist from a goal
  and an allowed wiki directory;
- allow both new-page creation and refresh of existing pages;
- automatically execute the plan, with an optional no-write `dry_run` mode;
- accept only server-configured, allowlisted model profiles;
- retry transient failures a bounded number of times and require explicit user
  resume after a terminal failure;
- resume only from the last committed page boundary; and
- keep the entire capability disabled by default.

### Implementation and verification status

Implemented on `feat/platform-architecture-evolution`. Substantive candidate
`f4c22afa39c249e000ac5becf10d8bfb212be75c` passed all six jobs in GitHub
Actions [run 30251148719](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251148719),
including the fresh-process Postgres RAG matrix and required live scaled
Compose smoke. Independent specification and quality reviews both reported
Critical 0, Important 0, Minor 0, `Ready: Yes` for that exact SHA.

Local evidence includes 74 core tests, 926 RAG/job unit tests, 180 MCP/wiki
invariant tests, 212 fresh-process RAG integration tests, 402 API integration
tests with 36 skips, 391 retrieval integration tests, and 19 scaled/CI static
contract tests with only the live opt-in case skipped locally. The
deterministic E2E dataset verified seven scenarios and an exact successful
usage of seven steps and 42 model tokens. A separate local live scaled recovery
smoke passed in 361.66 seconds. Focused Ruff is clean; the unrelated
repository-wide baseline remains 258 errors, 149 automatically fixable.

Operational details and the rollout boundary are recorded in
[`docs/architecture/server-rag.md`](../../architecture/server-rag.md). The
feature stays off by default: deploy migration and matched API/worker code
first, then enable one internal allowlisted profile for a bounded tenant
cohort. Rollback is `SERVER_RAG_ENABLED=false` on both roles without removing
migration `015` or committed/audit state.

## Goals

The first release must:

1. create or refresh a bounded set of wiki pages from one authenticated
   knowledge base;
2. preserve tenant isolation, citations, wiki write invariants, and concurrent
   user edits;
3. persist enough bounded state to inspect, cancel, retry, and explicitly
   resume work without storing secrets or full retrieved documents;
4. reuse the shared retrieval service and the existing Postgres durable job
   semantics;
5. provide REST and CLI entry points while keeping model execution out of API
   request processes; and
6. support deterministic fake-model end-to-end verification without external
   network access.

## Non-goals

This milestone does not add:

- a general autonomous agent, arbitrary tool execution, or recursive planning;
- a user-defined DAG, workflow language, or plugin execution system;
- browser automation, arbitrary code execution, shell access, or external URL
  fetching by the model;
- client-supplied model endpoints, credentials, system prompts, or tools;
- vector-only retrieval or automatic promotion of hybrid retrieval;
- local/offline server orchestration; the local MCP workflow remains the
  supported offline path;
- automatic merging of concurrent edits; or
- unbounded worklists, context, retries, tokens, page sizes, or traces.

## Alternatives considered

### Accepted: purpose-built orchestration on the durable job ledger

`build_wiki` becomes a supported background job type. A pure core state model,
RAG-specific persistence, and explicit ports isolate orchestration from
Postgres, HTTP, model, retrieval, and wiki-write details. This preserves the
existing claim, retry, lease, cancellation, and reaper behavior while making
page-boundary recovery auditable.

### Rejected: one monolithic job handler

Putting planning, retrieval, generation, validation, and writes in a single
handler with one result JSON object would be initially smaller, but it would
make step bounds, page-boundary recovery, conflict handling, and trace privacy
implicit. It would also make a worker crash between a wiki write and trace
update difficult to reconcile.

### Rejected: general DAG workflow engine

A DAG scheduler would provide flexibility that the first workflow does not
need. It would introduce dependency semantics, generic node payloads, and a
workflow DSL before a second server workflow exists. The dedicated state
machine can later become one input to such a system without exposing a generic
engine now.

## Architectural boundaries

### Core domain

`llmwiki_core` owns immutable RAG contracts and pure rules:

- normalized run configuration;
- page work items and page states;
- step types, statuses, and stable error codes;
- budget reservation and accounting;
- worklist validation and target-path containment;
- allowed state transitions;
- page-boundary resume calculations;
- citation identity and draft validation inputs; and
- bounded public summaries.

The core does not import asyncpg, FastAPI, HTTP clients, model SDKs, MCP code,
or environment settings.

### Application orchestrator

The API application owns a `BuildWikiOrchestrator` that depends on narrow
ports:

- `RagRunRepository` for runs, page items, and step records;
- `RagPlannerModel` and `RagWriterModel` backed by one allowlisted model
  profile;
- `RagRetriever` assembled from the shared `HybridRetrievalService` and the
  same production Postgres lexical/vector adapters used by hosted search;
- `RagReader` for exact, tenant-scoped reads of selected current document
  chunks or page ranges;
- `RagWikiWriter` for compare-and-swap wiki commits; and
- `RagLinter` for deterministic frontmatter, citation, link, and hygiene
  checks.

The orchestrator drives one bounded transition at a time. The durable job
handler only validates job ownership and shape, constructs these ports, and
invokes the orchestrator. It does not contain prompt text, SQL, or page-state
rules.

### Transport boundary

REST creates and observes runs but never performs model work inline. The CLI is
an HTTP client for the REST API and does not connect directly to Postgres or a
model endpoint. Existing MCP tools continue to expose direct search, read,
write, and lint behavior independently.

## Persistent model

The existing `background_jobs` row remains the canonical source for queue,
claim, lease, attempt, retry, cancellation, and terminal job state.

### `rag_runs`

Each row is one immutable run attempt and has exactly one `background_jobs`
row. A resumed attempt receives a new run id and job id while linking to its
parent and root run.

Required columns are:

- `id`, `job_id`, `root_run_id`, and nullable `parent_run_id`;
- `user_id` and `knowledge_base_id` with composite ownership constraints;
- bounded `goal` plus `goal_digest`;
- normalized `target_path_prefix` under `/wiki/`;
- `model_profile`, `retrieval_profile`, and their resolved non-secret version
  identifiers;
- `dry_run`;
- `max_pages`, `max_steps`, `max_model_tokens`, `max_context_chars`,
  `max_page_chars`, `per_call_timeout_seconds`, `max_page_attempts`, and
  `max_conflict_retries`;
- accumulated step count, model tokens, pages committed, and pages inspected;
- `completion_reason`, `last_committed_ordinal`, and timestamps; and
- caller idempotency key and a digest of the normalized create request.

The create path enforces one row per `(user_id, idempotency_key)`. Reusing a key
with the same digest returns the existing run; reusing it with a different
digest returns `rag_idempotency_conflict`.

### `rag_run_pages`

The accepted worklist is normalized into rows instead of remaining model JSON.
Each row contains `run_id`, zero-based `ordinal`, normalized wiki path, bounded
intent, bounded retrieval query, state, document id when known, version read,
version committed, attempt count, and the last completed step sequence.

Unique constraints prevent duplicate ordinals and duplicate paths within one
run. States are `planned`, `running`, `committed`, `dry_run_complete`, and
`failed`. A resumed run copies the original accepted worklist, marks pages at
or before the parent commit boundary as already complete, and never calls the
planner again.

### `rag_steps`

Step types are `plan`, `retrieve`, `read`, `draft`, `validate`, `write`,
`lint`, and `conflict`. Statuses are `running`, `succeeded`, and `failed`.
Rows contain:

- run id, nullable run-page id, and a strictly increasing sequence;
- input digest and bounded output summary;
- bounded citation identities, never retrieved document bodies;
- prompt/model profile version hashes, never prompt text or credentials;
- input, output, and total token counts;
- latency, stable error code, bounded sanitized message, and timestamps.

Only one running step is allowed per run. Database checks cap every stored text
and JSON field. RLS and composite foreign keys enforce the same user and
knowledge-base owner throughout the run, page, step, job, and document graph.

## Configuration and hard bounds

`SERVER_RAG_ENABLED` defaults to `false`. Requests can reduce, but cannot
exceed, server limits. Initial defaults and hard caps are:

| Limit | Default | Hard cap |
|---|---:|---:|
| Pages per run | 8 | 32 |
| Persisted steps | 96 | 512 |
| Model tokens per run | 64,000 | 250,000 |
| Retrieved context characters per page | 120,000 | 240,000 |
| Page characters | 40,000 | 120,000 |
| Model call timeout | 60 seconds | 180 seconds |
| Attempts per page | 2 | 3 |
| Version-conflict retries per page | 1 | 3 |
| Goal characters | 4,000 | 4,000 |
| Work-item intent/query characters | 1,000 | 2,000 |
| Step summary bytes | 8 KiB | 16 KiB |
| Citation identities per step | 64 | 128 |

The operator defines named model profiles in server configuration. A profile
contains an OpenAI-compatible base URL, model id, timeout ceilings, and a
reference to a separately injected credential. Only the profile name and a
non-secret configuration version are persisted. Model endpoints and keys in a
request are rejected as unknown fields.

The retrieval profile is `lexical` by default. `hybrid` is accepted only when
the existing hybrid feature flag, model/profile compatibility, embeddings,
and promotion rules permit it. Typed vector availability failures follow the
existing lexical fallback contract and are recorded as a bounded retrieval
summary; lexical, isolation, or configuration failures remain visible.

## Request and response API

### Create

`POST /v1/rag/build-wiki` accepts:

- `knowledge_base_id`;
- `goal`;
- `target_path_prefix`;
- `model_profile`;
- optional `retrieval_profile`, defaulting to `lexical`;
- optional bounded budget overrides; and
- optional `dry_run`, defaulting to `false`.

`Idempotency-Key` is a required request header. The endpoint returns HTTP 202
with `run_id`, `job_id`, job state, and links to the run and generic job
resources. A disabled feature returns stable code `rag_disabled`; invalid or
unknown profiles fail before creating either row.

### Observe

`GET /v1/rag/runs/{run_id}` returns a tenant-scoped summary: immutable request
configuration, job state, current page ordinal and path, budget use, committed
page count, completion reason, last committed boundary, and sanitized error.

`GET /v1/rag/runs/{run_id}/steps` returns cursor-paginated bounded trace rows.
It never returns full prompts, raw model responses, retrieved content, keys,
connection strings, or internal exception strings.

### Cancel and resume

Cancellation uses the existing `POST /v1/jobs/{job_id}/cancel` endpoint. A
running orchestrator observes cancellation at every lease checkpoint and at
page boundaries. Cancellation never rolls back an already committed page.

`POST /v1/rag/runs/{run_id}/resume` accepts optional higher budgets within the
server caps and a new `Idempotency-Key`. It is valid only for a failed run. The
new attempt preserves user, knowledge base, goal, path scope, profiles,
`dry_run`, and worklist. It starts after the last committed ordinal. Cancelled
or succeeded runs cannot be resumed; callers create a new run instead.

### CLI

The CLI provides:

- `rag build-wiki`;
- `rag status`;
- `rag steps`; and
- `rag resume`.

It uses the same authenticated REST API, emits stable exit codes, supports JSON
output, and never accepts or prints an API key for the model provider.

## Worklist planning

Planning occurs once for a root run. The planner receives the bounded goal,
allowed wiki prefix, a bounded catalog of existing wiki paths/titles, and
bounded shared-search context. Retrieved text is delimited and labeled as
untrusted evidence, not instructions.

The model must return one structured object containing no more than
`max_pages` entries. Each entry has a relative page path, intent, and retrieval
query. The application:

1. parses the response with duplicate-key rejection and exact field schemas;
2. normalizes every path through the shared logical path contract;
3. requires a Markdown page below the requested `/wiki/` prefix;
4. rejects traversal, absolute host paths, duplicate normalized paths, and
   structural pages protected by existing write policy;
5. truncates nothing silently; any oversized field fails the plan; and
6. persists the complete accepted worklist before page execution begins.

An empty plan succeeds with completion reason `no_work`. Invalid structured
output may receive one model repair attempt if budget remains; another invalid
response fails with `rag_invalid_plan`.

## Per-page execution

For each incomplete work item in ordinal order, the worker performs:

1. **Checkpoint:** verify the live job lease, cancellation, feature flag, and
   remaining step/token budgets.
2. **Retrieve:** build an immutable shared `SearchQuery` from the work-item
   query and configured profile. Filters apply before the candidate limit.
3. **Read:** fetch only the selected current document/chunk identities and
   requested page ranges under the same tenant and knowledge base. Total
   context is bounded before prompt construction.
4. **Draft:** ask the configured model for a structured page draft and citation
   map. Existing page content and retrieved evidence are delimited as untrusted
   data. Neither can change system rules or request tools.
5. **Validate:** enforce page size, YAML frontmatter, title/tags, footnote
   syntax, citation resolution against selected evidence, wiki-link path
   validity, at least one supported visual element, no unsupported asset
   references, and budget accounting.
6. **Write:** for a new page use `expected_version=None`; for an update use the
   version read at step 3. Commit content, chunks, citation/link edges, facet
   rollups, page state, step success, and the commit boundary in one Postgres
   transaction guarded by the active job lease.
7. **Lint:** run deterministic wiki lint against the committed version and
   persist the bounded result.

One invalid draft may receive one structured repair call within the page
attempt and remaining budget. A second invalid draft is terminal for that page.
The orchestrator does not synthesize missing citations or silently drop invalid
content.

Entering an uncommitted page increments its page-attempt count. Worker/job
retries, a crash followed by reconstruction, and redrafting after a transient
failure all consume this same counter; the page fails closed when
`max_page_attempts` is exceeded. Conflict retries additionally consume the
separate, lower conflict counter and still count as page attempts.

In `dry_run`, validation and deterministic lint run against the draft, but the
write transaction is omitted. A bounded draft preview and citation list are
persisted, the page becomes `dry_run_complete`, and no commit boundary is
created. If the page exceeds the step-summary cap, the preview is explicitly
marked truncated and includes the complete draft digest and character count;
the full draft is not stored elsewhere.

## Atomicity, conflicts, and recovery

The wiki commit and RAG page/step boundary share one database transaction. It
locks or proves the live job lease before side effects and uses the existing
wiki compare-and-swap contract. This prevents a worker that lost its lease from
publishing or marking a page complete.

A `VersionConflict` records a conflict step, rereads the current page, and
redrafts at most `max_conflict_retries` times. Exceeding the limit fails with
`rag_version_conflict`; it never overwrites the concurrent revision.

Transient database, object-store, and allowlisted model availability failures
map to the existing bounded job retry mechanism. The same job is retried and
reconstructs state from the last committed page. Validation, isolation,
configuration, invalid model output, or exhausted retry failures are terminal.

Budget exhaustion is a terminal failed job with stable code
`rag_budget_exhausted`; already committed pages remain current. An explicit
resume may add budget within server caps and begins at the next page. A worker
crash during retrieve, read, draft, validate, or dry-run processing repeats the
uncommitted page. A crash after the atomic write sees the committed boundary
and advances without rewriting it.

Disabling `SERVER_RAG_ENABLED` rejects new create/resume requests and causes
running work to stop with `rag_disabled` at the next page boundary. A page
transaction already in progress is allowed to finish atomically.

## Model and prompt safety

The model is a structured generation dependency, not an agent. It receives no
tools, credentials, environment variables, network client, database handle, or
filesystem access.

Every planner and writer prompt:

- separates system policy, caller goal, existing wiki content, and retrieved
  evidence with explicit typed delimiters;
- states that wiki and retrieved content are untrusted and instructions inside
  them must not be followed;
- permits citations only from the supplied immutable evidence identities;
- requires exact JSON output parsed with duplicate-key and non-finite-number
  rejection; and
- uses a server-owned template and version hash.

The adapter supports OpenAI-compatible chat/completions through an injected
async HTTP client. It enforces connect/read/total timeouts, response-size caps,
strict status handling, strict token usage, and verified client cleanup.
Provider errors are sanitized before becoming typed application errors. A
provider response that omits or falsifies required token accounting fails
closed and cannot bypass the run budget.

## Budget accounting

Every persisted step attempt consumes one step before its external operation.
Before a model call, the orchestrator reserves the requested maximum output
tokens plus the locally counted input estimate; the reservation must fit the
remaining run budget. After a valid response, provider usage replaces the
reservation. Usage greater than the reservation or missing required fields is
an invalid provider response.

Retries, repair calls, planning, conflicts, and failed calls with reported
usage all count. Database-only cleanup and deterministic lint do not consume
model tokens but do consume steps. A run cannot start a page unless at least
the configured minimum page-step allowance remains.

## Errors and public completion reasons

Stable error families include:

- `rag_disabled`;
- `rag_invalid_request` and `rag_idempotency_conflict`;
- `rag_model_profile_unavailable` and `rag_retrieval_profile_unavailable`;
- `rag_invalid_plan` and `rag_invalid_draft`;
- `rag_citation_invalid`, `rag_frontmatter_invalid`, and `rag_link_invalid`;
- `rag_version_conflict`;
- `rag_budget_exhausted`;
- `rag_model_unavailable`, `rag_retrieval_failed`, and `rag_write_failed`;
- `rag_lease_lost` and existing job cancellation; and
- `rag_internal_error` for sanitized unknown failures.

RAG summaries distinguish `completed`, `no_work`, `dry_run`,
`budget_exhausted`, and `partial_failure` without adding states to the generic
job ledger. Public messages are bounded, stable, and contain no goal text,
page body, prompt, evidence, endpoint, key, DSN, or private exception text.

## Observability

Structured logs may contain run id, job id, request id, user-id hash,
knowledge-base id, page document id, page ordinal, step type, error code,
latency, token counts, and bounded counts. They must not contain goals, raw page
paths, prompts, raw model output, retrieved content, credentials, endpoints
with query strings, DSNs, or exception messages.

Metrics cover:

- queued/running/completed/failed/cancelled RAG jobs;
- page and step outcomes;
- plan size and committed-page count;
- token usage and model latency by non-secret profile name;
- retrieval profile/fallback outcomes;
- validation and lint error codes;
- version conflicts and retry counts; and
- time to first and final committed page.

Health remains process liveness. Worker readiness requires the existing job
dependencies plus a resolvable allowlisted model profile when RAG is enabled.

## Testing strategy

Implementation follows red-green-refactor and uses no real provider or public
network in CI.

### Pure core tests

Cover exact configuration validation, hard caps, path containment, worklist
deduplication, immutable transitions, budget reservation/accounting, resume
calculation, citation identities, completion reasons, and sanitized errors.

### Repository and schema tests

Against real Postgres, cover RLS, composite ownership, run/job one-to-one
ownership, idempotent create, conflicting idempotency reuse, page/step sequence
constraints, output caps, one running step, resume lineage, and concurrent step
append behavior.

### Orchestrator tests

With deterministic fake ports, cover empty plan, create, refresh, `dry_run`,
invalid planner/draft JSON, prompt-injection evidence, unsupported citations,
budget exhaustion, lease loss at every side-effect boundary, cancellation,
retry classification, model timeout, lint failure, and client cleanup.

### Real Postgres integration tests

Cover an atomic fake-model wiki build; content/chunk/reference/facet version
invariants; new-page races; stale update conflicts; bounded conflict retry;
worker crash before and after commit; explicit resume; tenant isolation for
create/read/steps/resume/cancel; and job reaper recovery.

### Multi-replica and adapter tests

The OpenAI-compatible adapter runs against a local fake HTTP service for
success, timeout, non-2xx, response cap, malformed JSON, duplicate keys,
invalid usage, cancellation, and cleanup. Required scaled Compose proves that
one run is advanced by only one of two workers while either API replica can
create, observe, cancel, and resume it.

## Rollout and rollback

Rollout order is:

1. deploy the additive migration and code with `SERVER_RAG_ENABLED=false`;
2. configure allowlisted model profiles through server secrets;
3. run fake-model schema, API, worker, and atomic-write smoke tests;
4. enable one internal profile and a bounded tenant cohort;
5. observe failure, validation, token, latency, conflict, and cancellation
   metrics; and
6. expand the cohort without changing default budgets or retrieval profile.

Rollback sets `SERVER_RAG_ENABLED=false`. New and resumed work is rejected;
running work stops at a page boundary; existing wiki commits and complete
audit traces remain. The additive tables can remain deployed while the feature
is disabled. Hybrid retrieval remains independently gated and lexical remains
the default.

## Acceptance criteria

The milestone is complete only when:

- the feature is default-off and rejects unknown/client-supplied model
  configuration;
- an authenticated fake-model run creates a linted, cited wiki page through a
  durable worker;
- a refresh uses expected-version compare-and-swap and cannot overwrite a
  concurrent edit;
- cancellation, lease loss, retry, budget exhaustion, and explicit resume all
  preserve the last committed page boundary;
- `dry_run` produces bounded validated previews without any wiki mutation;
- tenant isolation holds across jobs, runs, pages, steps, documents, cancel,
  and resume;
- two API and two worker replicas pass the RAG smoke workflow;
- focused and complete isolated CI partitions pass on the exact pushed SHA;
  and
- specification and code-quality reviews report no unresolved critical,
  important, or minor findings.
