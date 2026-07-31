# Server-side RAG Orchestration

## Status

Implemented on `feat/platform-architecture-evolution` as an optional,
default-off Hosted capability. It complements direct MCP tools with one bounded
server-owned `build_wiki` workflow.

The canonical operator contract, limits, verification counts, and rollback are
in [server-rag.md](../../architecture/server-rag.md). This specification keeps
the product and architectural decisions.

## Goals and non-goals

The workflow creates or refreshes a bounded set of wiki pages from one
authenticated knowledge base, preserves citations and concurrent edits,
persists enough state to inspect/cancel/recover/resume, and executes model calls
outside API request processes. It supports deterministic offline CI through an
OpenAI-compatible fake model.

It is not a general agent, workflow language, plugin system, arbitrary tool
runner, browser, shell, URL fetcher, or recursive planner. Clients cannot
provide endpoints, credentials, prompts, tools, or unbounded work. Vector-only
retrieval and automatic hybrid promotion remain out of scope. Local mode keeps
the MCP workflow and does not run server orchestration.

## Alternatives considered

### Selected: purpose-built workflow on the durable ledger

`build_wiki` is a supported durable job type with a pure state model,
RAG-specific repository, explicit model/retrieval/wiki ports, and page-boundary
commits. This reuses claim, retry, heartbeat, cancellation, and reaper behavior
while keeping recovery auditable.

### Rejected: monolithic handler result

One handler with a large result blob would make step bounds, conflict handling,
partial commits, and privacy implicit. It would also leave an ambiguous crash
window between a wiki write and a reported page boundary.

### Rejected: general DAG engine

A workflow DSL and generic node scheduler add dependency semantics and an
extension surface before a second server workflow exists. The dedicated state
machine is smaller, safer, and can later inform a general engine without
exposing one now.

## Architectural boundaries

`llmwiki_core.rag` defines immutable budgets, work items, page/step states,
completion reasons, public failures, transition rules, and resume lineage. It
imports no database, HTTP, or model implementation.

The application orchestrator advances one bounded transition at a time through
ports for run persistence, retrieval, model completion, and atomic wiki writes.
Adapters own asyncpg records, OpenAI-compatible HTTP, shared retrieval service,
and Postgres wiki commits. The durable worker constructs those dependencies and
executes the run; API routes only validate, persist, enqueue, and project safe
state.

The dependency relationship is documented in the
[platform overview](../../architecture/overview.md),
[shared-kernel contract](../../architecture/shared-kernel.md), and
[durable-job contract](../../architecture/durable-jobs.md).

## Persistent model

`rag_runs` records tenant/knowledge-base identity, root and parent lineage,
goal digest, target prefix, immutable budget, model profile name/version,
retrieval profile, dry-run flag, durable job identity, worklist digest, usage,
completion reason, committed boundary, and sanitized error.

`rag_run_pages` is the immutable ordered worklist. Each page records path,
intent, source identity/version for refresh, state, attempts, expected and
committed wiki versions, bounded preview metadata, digest, and final step
sequence. Ordinals are unique within a run.

`rag_steps` is a bounded append-only trace for plan, retrieval, draft,
validation, lint, write, and conflict attempts. It stores prompt identity,
bounded summaries, citation identities, token/latency counts, and stable error
codes—not prompt text, evidence bodies, raw responses, or private errors.

Migration `015_server_rag.sql` adds these tables, constraints, indexes, RLS,
and the `build_wiki` durable job type. It is additive and remains installed
during flag rollback.

## Configuration and hard bounds

`SERVER_RAG_ENABLED` defaults to false and requires Hosted durable jobs. API and
worker receive identical `RAG_MODEL_PROFILES_JSON` metadata and a separate
`RAG_MODEL_API_KEYS_JSON` secret mapping. Profiles are closed-schema,
size-bounded, matched by name, and restricted to approved HTTP(S) base URLs.
Only profile name and non-secret version persist.

Requests may lower but never exceed server caps for pages, steps, tokens,
context characters, output characters, per-call timeout, page attempts, and
conflict retries. The current exact defaults and hard caps are owned by
[server-rag.md](../../architecture/server-rag.md).

Configuration validation makes no provider call. Provider URL, model ID,
credential, prompt, raw response, evidence, DSN, goal, path, and private
exception text are excluded from public responses and telemetry.

## Request and response API

The Hosted REST surface is:

- `POST /v1/rag/build-wiki` with `Idempotency-Key`;
- `GET /v1/rag/runs/{run_id}`;
- `GET /v1/rag/runs/{run_id}/steps` with bounded cursor pagination;
- `POST /v1/rag/runs/{run_id}/resume` with a new idempotency key; and
- `POST /v1/jobs/{job_id}/cancel`.

Create accepts only knowledge base, bounded goal, `/wiki/.../` target prefix,
server-owned model profile, optional retrieval profile, lower budget overrides,
and dry-run. Unknown fields are rejected. Every resource is tenant scoped; a
missing, foreign, or cross-tenant resource has the same not-found projection.

`python -m scripts.rag` is a REST client providing `build-wiki`, `status`,
`steps`, and `resume`. Its LLMWiki bearer token is not a provider key and is
not echoed in errors or command output.

## Planning and per-page execution

Planning retrieves bounded context, sends a versioned system prompt, validates
one exact structured worklist, normalizes paths, rejects duplicates and prefix
escapes, enforces page/step budgets, resolves refresh targets, and persists the
entire immutable worklist before page execution. An empty plan succeeds as
`no_work`.

For each pending ordinal, the worker retrieves current evidence, calls the
model for a structured draft, validates frontmatter/path/content/citations,
runs deterministic lint, and passes one write bundle to the atomic writer. A
lease checkpoint fences every remote or database boundary.

Dry-run performs the same plan, retrieval, draft, validation, and lint path but
writes no wiki/chunk/reference/facet rows. It persists only a bounded preview,
full character count, digest, and truncation marker.

## Atomicity, conflicts, and recovery

A wiki revision and the matching `rag_run_pages` committed boundary update in
one Postgres transaction. A crash before that transaction repeats the page
after lease recovery; a crash after it observes the durable ordinal and cannot
rewrite the committed page.

Refresh uses shared compare-and-swap wiki versions. A conflict rereads the
current page and redrafts within the configured retry cap. It never silently
overwrites a concurrent edit. The transaction locks run and page rows in stable
order and rejects a stale worker or mismatched job/run relationship.

Cancellation and terminal failures retain already committed pages. Resume
creates a new run/job with root and parent lineage, copies the immutable
worklist, skips committed ordinals, and accepts only budget increases within
caps. Successful and cancelled runs are not resumable.

## Model, prompt, and retrieval safety

Prompts separate trusted instructions, untrusted goal text, and untrusted
retrieved evidence. Retrieved instructions are data and never tools or policy.
The model adapter accepts no tools, follows strict request/response size caps,
uses bounded timeout and retry policy, and sanitizes provider failures.

Lexical retrieval is the default. Hybrid is accepted only when its independent
Hosted configuration and current-version embedding coverage are valid. Only a
typed vector availability failure follows the shared lexical fallback contract;
authorization, lexical, configuration, and unexpected failures remain visible.

## Budget, errors, and observability

Planning and every page step reserve persisted step budget before execution.
Model usage is recorded from validated provider totals; missing or malformed
usage is a model failure. Budget exhaustion is terminal for the current run and
preserves the committed prefix for explicit resume.

Public failures use stable RAG codes and vetted messages. Unknown exceptions
become sanitized internal failures; process control and cancellation propagate
through their dedicated paths. Persisted trace summaries are capped and
validated before storage.

Structured events are `rag_run_started`, `rag_step_finished`,
`rag_page_committed`, `rag_run_finished`, and `rag_run_failed`. Safe dimensions
are identifiers, ordinal, step type, profile name/version, counts, tokens,
duration, and stable error code. Content and private configuration are
forbidden.

## Verification and operations

The substantive server-RAG candidate is
`f4c22afa39c249e000ac5becf10d8bfb212be75c`. All six jobs passed in
[GitHub Actions run 30251148719](https://github.com/ben0112/llmwiki-goglobal/actions/runs/30251148719),
including isolated Postgres RAG tests and required two-API/two-worker live
recovery.

The deterministic E2E dataset covers initial generation and prompt injection,
refresh, dry-run, budget exhaustion plus resume, timeout persistence, real
version conflict, and private configuration exclusion. The successful path has
the exact seven-step trace and 42 model tokens. Independent specification and
quality reviews concluded Critical `0`, Important `0`, Minor `0`, `Ready: Yes`.

Rollout deploys migration and matched API/worker code with the flag false,
validates allowlisted profiles and deterministic gates, then enables one
internal profile for a bounded tenant cohort. Rollback sets
`SERVER_RAG_ENABLED=false` on both roles without dropping migration `015` or
deleting committed/audit state.

Current operations and client usage are in:

- [Server-side RAG operations](../../architecture/server-rag.md)
- [Retrieval contract](../../architecture/retrieval.md)
- [Self-hosting guide](../../self-hosting.md)
- [Agent integration guide](../../agent-integration.md)
